"""Single-Prompt baseline agent for VeriStat (A.4 / D.3 Baseline).

Executes a 1-shot LLM reasoning pipeline without external tool loops:
1. Receives socioeconomic statistical QA question.
2. Builds single unified prompt containing database schema/table catalog context.
3. Invokes model (Qwen2.5-7B-Instruct / MockLLM) exactly once with temperature=0.0.
4. Parses output into standardized FinalAnswer & 1-step Trajectory.
5. Saves evaluation records and execution trace for eval harness and C.4 error taxonomy.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from veristat.agents.contracts import AgentAction
from veristat.agents.parser import parse_agent_action
from veristat.agents.types import (
    FinalAnswer,
    LLMClient,
    LLMResponse,
    ReActStep,
    StepObservation,
    ToolAction,
    Trajectory,
    _normalize_final_answer_value,
)
from veristat.data.schema import BenchmarkQuestion, open_readonly
from veristat.eval.metrics import EvalRecord, build_eval_record, summarize_records
from veristat.eval.runner import load_questions, load_table_domain_map
from veristat.llm.client_adapter import AdaptiveLLMClient
from veristat.llm.mock_client import MockLLMClient
from veristat.llm.vllm_client import VLLMClient, VLLMConfig

# Default concise statistical catalog used when veristat.db is offline
_FALLBACK_TABLE_CATALOG = """\
- NG2025_T001: Diện tích, dân số và mật độ dân số phân theo địa phương (Đơn vị: km2, nghìn người, người/km2; Kỳ: 2020-2024; Toàn quốc và 63 tỉnh/thành)
- NG2025_T017: Dân số trung bình phân theo giới tính và thành thị, nông thôn (Đơn vị: nghìn người; Kỳ: 2020-2024; Toàn quốc và 6 vùng)
- NG2025_T024: Tỷ suất sinh thô và tỷ suất chết thô phân theo địa phương (Đơn vị: ‰; Kỳ: 2020-2024; 63 tỉnh/thành)
- NG2025_T035: Lực lượng lao động từ 15 tuổi trở lên phân theo địa phương (Đơn vị: nghìn người; Kỳ: 2020-2024; 63 tỉnh/thành)
- NG2025_T042: Tỷ lệ thất nghiệp trong độ tuổi lao động phân theo địa phương (Đơn vị: %; Kỳ: 2020-2024; 63 tỉnh/thành)
- NG2025_T048: Thu nhập bình quân đầu người một tháng phân theo vùng (Đơn vị: nghìn đồng; Kỳ: 2020-2024; 6 vùng kinh tế)
- NG2025_T056: Số trường học mầm non và phổ thông phân theo địa phương (Đơn vị: trường; Kỳ: 2020-2024; 63 tỉnh/thành)
- NG2025_T065: Số bác sĩ và giường bệnh phân theo địa phương (Đơn vị: người, giường; Kỳ: 2020-2024; 63 tỉnh/thành)
- NG2025_T080: Số doanh nghiệp đang hoạt động có kết quả SXKD phân theo địa phương (Đơn vị: doanh nghiệp; Kỳ: 2020-2024; 63 tỉnh/thành)
- T10_2026_B01: Chỉ số sản xuất công nghiệp (IIP) tháng 10/2026 (Đơn vị: % so với cùng kỳ; Kỳ: 10/2026; Cả nước)
- T10_2026_B04: Tổng mức bán lẻ hàng hóa và doanh thu dịch vụ tiêu dùng tháng 10/2026 (Đơn vị: nghìn tỷ đồng; Kỳ: 10/2026; Cả nước)
"""


def build_catalog_context(db_path: Path | str | None = None) -> str:
    """Extract catalog context from `tables_catalog` table in `veristat.db`."""
    if db_path is None:
        db_path = Path("data/processed/veristat.db")
    path = Path(db_path)
    if not path.exists():
        return _FALLBACK_TABLE_CATALOG

    try:
        conn = open_readonly(path)
        rows = conn.execute(
            "SELECT table_id, title, unit, reporting_period, scope FROM tables_catalog ORDER BY table_id"
        ).fetchall()
        conn.close()
        if not rows:
            return _FALLBACK_TABLE_CATALOG

        lines = []
        for tid, title, unit, period, scope in rows:
            lines.append(f"- {tid}: {title} (Đơn vị: {unit}; Kỳ: {period}; Phạm vi: {scope})")
        return "\n".join(lines)
    except Exception:
        return _FALLBACK_TABLE_CATALOG


def build_single_prompt(
    question: BenchmarkQuestion | dict[str, Any] | str,
    catalog_context: str | None = None,
) -> list[dict[str, str]]:
    """Build a unified, instruction-rich prompt for single-shot baseline reasoning."""
    if catalog_context is None:
        catalog_context = build_catalog_context()

    if isinstance(question, str):
        q_text = question
        meta_info = ""
    elif isinstance(question, dict):
        q_text = question.get("question", "")
        meta_info = (
            f"[Siêu dữ liệu: ID={question.get('question_id', '')}, "
            f"Lĩnh vực={question.get('domain', question.get('category', ''))}, "
            f"Kiểu đáp số={question.get('answer_type', 'number')}, "
            f"Đơn vị kỳ vọng={question.get('unit', '')}]"
        )
    else:
        domain_val = getattr(question, "domain", question.category)
        q_text = question.question
        meta_info = (
            f"[Siêu dữ liệu: ID={question.question_id}, Lĩnh vực={domain_val}, "
            f"Kiểu đáp số={question.answer_type}, Đơn vị kỳ vọng={question.unit}]"
        )

    system_instruction = (
        "Bạn là Chuyên gia Phân tích Số liệu Thống kê Kinh tế – Xã hội Việt Nam (Dự án VeriStat).\n"
        "Nhiệm vụ của bạn là trả lời các câu hỏi thống kê từ Niên giám Thống kê 2025 và "
        "Báo cáo KTXH tháng 10/2026 với độ chính xác số học cao nhất.\n\n"
        "--- DANH MỤC CSDL BẢNG BIỂU THỐNG KÊ ---\n"
        f"{catalog_context}\n\n"
        "--- HƯỚNG DẪN ĐỊNH DẠNG ĐẦU RA ---\n"
        "Bắt buộc trả về duy nhất một khối JSON hợp lệ theo schema sau (không thêm văn bản ngoài JSON):\n"
        "{\n"
        '  "thought": "Các bước phân tích logic, công thức thống kê và suy luận đáp số",\n'
        '  "answer": <kết quả số chính xác, tên địa phương, hoặc danh sách top-k>,\n'
        '  "unit": "<đơn vị tính tương ứng>",\n'
        '  "cited_tables": ["<mã bảng nguồn tham chiếu, ví dụ: NG2025_T017>"],\n'
        '  "computation_code": "<đoạn mã Python tính toán độc lập nếu có>"\n'
        "}"
    )

    user_content = f"Câu hỏi thống kê:\n{q_text}\n{meta_info}\n\nHãy suy luận và trả về kết quả theo đúng cấu trúc JSON yêu cầu."

    return [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_content},
    ]


def parse_single_prompt_response(raw_text: str) -> FinalAnswer:
    """Parse raw LLM response text into structured FinalAnswer conforming to schema."""
    cleaned = raw_text.strip()

    # 1. Check if ReAct parser can extract finish action
    parsed_action: AgentAction = parse_agent_action(cleaned)
    if parsed_action.is_finish and parsed_action.final_answer:
        fa = parsed_action.final_answer
        raw_ans = fa.get("answer") if "answer" in fa else fa.get("value")
        unit = str(fa.get("unit", ""))
        cited = tuple(str(t) for t in fa.get("cited_tables", []))
        code = str(fa.get("computation_code", ""))
        return FinalAnswer(
            answer=_normalize_final_answer_value(raw_ans),
            unit=unit,
            cited_tables=cited,
            computation_code=code,
            parse_ok=raw_ans is not None,
        )

    # 2. Direct JSON extraction
    json_candidate = None
    json_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if json_block_match:
        json_candidate = json_block_match.group(1)
    else:
        obj_match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        if obj_match:
            json_candidate = obj_match.group(1)

    if json_candidate:
        try:
            data = json.loads(json_candidate)
            if isinstance(data, dict):
                raw_ans = data.get("answer") if "answer" in data else data.get("value")
                unit = str(data.get("unit", ""))
                raw_cited = data.get("cited_tables", [])
                cited = tuple(str(t) for t in raw_cited) if isinstance(raw_cited, list) else ()
                code = str(data.get("computation_code", ""))
                return FinalAnswer(
                    answer=_normalize_final_answer_value(raw_ans),
                    unit=unit,
                    cited_tables=cited,
                    computation_code=code,
                    parse_ok=raw_ans is not None,
                )
        except json.JSONDecodeError:
            pass

    # 3. Fallback: regex search for numeric value or entity
    num_match = re.search(r"\b[-+]?\d+(?:\.\d+)?\b", cleaned)
    if num_match:
        val = float(num_match.group(0))
        return FinalAnswer(
            answer=val,
            unit="",
            cited_tables=(),
            computation_code="",
            parse_ok=True,
        )

    return FinalAnswer(
        answer=None,
        unit="",
        cited_tables=(),
        computation_code="",
        parse_ok=False,
    )


def run_single_prompt(
    question: BenchmarkQuestion,
    llm: LLMClient,
    *,
    catalog_context: str | None = None,
    seed: int | None = 42,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> tuple[Trajectory, FinalAnswer]:
    """Execute single 1-shot prompt call on LLM and return standardized Trajectory."""
    messages = build_single_prompt(question, catalog_context=catalog_context)

    t0 = time.perf_counter()
    resp: LLMResponse = llm.generate(
        messages,
        temperature=temperature,
        seed=seed,
        max_tokens=max_tokens,
    )
    latency_s = time.perf_counter() - t0

    final_answer = parse_single_prompt_response(resp.text)

    # Build 1-step Trajectory representation
    action = ToolAction(
        thought="Suy luận trực tiếp bằng prompt duy nhất không qua công cụ (Single Prompt Baseline).",
        tool_name="final_answer",
        tool_args={
            "answer": final_answer.answer,
            "unit": final_answer.unit,
            "cited_tables": list(final_answer.cited_tables),
            "computation_code": final_answer.computation_code,
            "parse_ok": final_answer.parse_ok,
        },
        raw_text=resp.text,
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
        latency_s=latency_s,
    )

    observation = StepObservation(
        output="[SINGLE_PROMPT_FINISH] Đáp số trực tiếp từ mô hình.",
        ok=final_answer.parse_ok,
        error=None if final_answer.parse_ok else "PARSE_ERROR",
        safety_violation=False,
        queried_tables=final_answer.cited_tables,
        computed_values=(final_answer.answer,) if final_answer.answer is not None else (),
        latency_s=0.0,
    )

    step = ReActStep(step_idx=1, action=action, observation=observation)

    trajectory = Trajectory(
        question_id=question.question_id,
        steps=(step,),
        final=final_answer,
        terminated_reason="final_answer" if final_answer.parse_ok else "error",
        total_prompt_tokens=resp.prompt_tokens,
        total_completion_tokens=resp.completion_tokens,
        total_latency_s=latency_s,
        seed=seed,
    )

    return trajectory, final_answer


def run_batch_single_prompt(
    questions: Sequence[BenchmarkQuestion],
    llm: LLMClient,
    out_path: Path | str,
    *,
    db_path: Path | str = "data/processed/veristat.db",
    seed: int = 42,
    resume: bool = True,
) -> tuple[dict[str, Any], list[EvalRecord]]:
    """Run batch Single Prompt baseline evaluation over a question set with JSONL checkpointing."""
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    domain_map = load_table_domain_map(db_path) if Path(db_path).exists() else {}
    catalog = build_catalog_context(db_path)

    # Resume from existing checkpoint
    completed_ids: set[str] = set()
    records: list[EvalRecord] = []
    if resume and out_file.exists():
        for line in out_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    data = json.loads(line)
                    rec = EvalRecord.from_dict(data)
                    records.append(rec)
                    completed_ids.add(rec.question_id)
                except Exception:
                    pass

    with out_file.open("a", encoding="utf-8") as f:
        for idx, q in enumerate(questions):
            if q.question_id in completed_ids:
                continue

            traj, final_ans = run_single_prompt(
                q,
                llm,
                catalog_context=catalog,
                seed=seed,
            )

            record = build_eval_record(
                q,
                method="single_prompt",
                trajectory=traj,
                predicted_answer=final_ans.answer,
                table_domain_map=domain_map,
                prompt_tokens=traj.total_prompt_tokens,
                completion_tokens=traj.total_completion_tokens,
                verifier_tokens=0,
                latency_s=traj.total_latency_s,
                safety_violation=False,
            )

            records.append(record)
            completed_ids.add(q.question_id)

            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            f.flush()

    summary = summarize_records(records)
    return summary, records


def check_vllm_connection(base_url: str, timeout: float = 2.0) -> bool:
    """Quick pre-flight check if vLLM server is responding."""
    try:
        url = f"{base_url.rstrip('/')}/models"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status in (200, 404, 405)
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="VeriStat Single Prompt Baseline Runner (A.4 / D.3)")
    parser.add_argument("--questions", type=Path, default=Path("data/benchmark/dev_30.jsonl"))
    parser.add_argument("--db", type=Path, default=Path("data/processed/veristat.db"))
    parser.add_argument("--out", type=Path, default=Path("results/single_prompt_dev.jsonl"))
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--mock", action="store_true", help="Use MockLLMClient instead of live vLLM")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    questions = load_questions(args.questions)

    if args.mock:
        raw_llm = MockLLMClient()
    else:
        if not check_vllm_connection(args.vllm_url):
            print(
                "\n"
                + "=" * 70
                + "\n[LỖI KẾT NỐI SERVER vLLM]\n"
                f"Không thể kết nối tới server vLLM tại: {args.vllm_url}\n"
                "Nguyên nhân: Hiện tại không có server vLLM nào đang chạy trên cổng này.\n\n"
                "HƯỚNG DẪN XỬ LÝ:\n"
                "1. Nếu bạn đang chạy thử nghiệm cục bộ (CPU, không có GPU / vLLM):\n"
                "   Vui lòng thêm cờ `--mock`:\n"
                f"   uv run python scripts/run_single_prompt.py --mock --questions {args.questions} --out {args.out}\n\n"
                "2. Nếu bạn đang chạy trên Google Colab Pro (GPU L4/A100):\n"
                "   Hãy khởi động server vLLM trước trong một cell riêng:\n"
                f"   !python -m vllm.entrypoints.openai.api_server --model {args.model} --port 8000\n"
                + "=" * 70
                + "\n"
            )
            sys.exit(1)
        raw_llm = VLLMClient(config=VLLMConfig(base_url=args.vllm_url, model=args.model))

    adaptive_llm = AdaptiveLLMClient(backend_client=raw_llm, model_name=args.model)

    print(f"Running Single Prompt baseline on {len(questions)} questions -> {args.out}...")
    summary, _ = run_batch_single_prompt(
        questions=questions,
        llm=adaptive_llm,
        out_path=args.out,
        db_path=args.db,
        seed=args.seed,
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
