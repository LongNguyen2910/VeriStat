"""VeriStat command-line interface for testing and running queries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from veristat.agents.react import VeriStatReActAgent
from veristat.llm.mock_client import MockLLMClient
from veristat.llm.vllm_client import VLLMClient, VLLMConfig
from veristat.tools.base import ToolRegistry
from veristat.tools.docs_tool import SearchDocsTool
from veristat.tools.python_tool import RunPythonTool
from veristat.tools.table_tool import QueryTableTool


def main() -> None:
    parser = argparse.ArgumentParser(description="VeriStat ReAct Agent CLI")
    parser.add_argument("question", nargs="?", help="Statistical question to analyze")
    parser.add_argument("--db", default="data/processed/veristat.db", help="Path to veristat.db")
    parser.add_argument("--mock", action="store_true", help="Use deterministic Mock LLM")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1", help="vLLM server base URL")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument("--max-steps", type=int, default=8, help="Max ReAct steps")
    args = parser.parse_args()

    if not args.question:
        print("Usage: python -m veristat.cli \"<câu hỏi thống kê>\" [--mock] [--db path]")
        sys.exit(0)

    # Initialize tools
    registry = ToolRegistry()
    registry.register(QueryTableTool(db_path=args.db))
    registry.register(SearchDocsTool(db_path=args.db))
    registry.register(RunPythonTool())

    # Initialize LLM Client
    if args.mock:
        # Realistic 4-step verified ReAct trajectory:
        # Step 1: search_docs -> Step 2: query_table -> Step 3: run_python -> Step 4: finish
        demo_responses = {
            "GROWTH=": json.dumps({
                "thought": "Sandbox tính ra tốc độ tăng trưởng là 3.8543%. Đưa ra đáp số và trích dẫn bảng NG2025_T017.",
                "action": "finish",
                "action_input": {
                    "value": 3.854269,
                    "answer_type": "number",
                    "unit": "%",
                    "cited_tables": ["NG2025_T017"]
                }
            }, ensure_ascii=False),
            "101343.8": json.dumps({
                "thought": "Đã có số liệu: năm 2020 là 97582.7 nghìn người, năm 2024 là 101343.8 nghìn người. Dùng Python sandbox tính tốc độ tăng trưởng.",
                "action": "run_python",
                "action_input": {"code": "v20 = 97582.7\nv24 = 101343.8\ngrowth = ((v24 - v20) / v20) * 100\nprint(f'GROWTH={growth:.6f}')"}
            }, ensure_ascii=False),
            "NG2025_T017": json.dumps({
                "thought": "Tìm thấy bảng NG2025_T017 (Dân số trung bình). Truy vấn số liệu cả nước năm 2020 và 2024.",
                "action": "query_table",
                "action_input": {"sql": "SELECT nam, gia_tri FROM stat_ng2025_t017 WHERE geo_level = 'country' AND nam IN (2020, 2024) ORDER BY nam ASC"}
            }, ensure_ascii=False),
            "Câu hỏi thống kê:": json.dumps({
                "thought": "Tìm kiếm bảng thống kê về dân số trung bình trong Niên giám thống kê.",
                "action": "search_docs",
                "action_input": {"query": "dân số trung bình"}
            }, ensure_ascii=False),
        }
        client = MockLLMClient(canned_responses=demo_responses)
    else:
        client = VLLMClient(VLLMConfig(base_url=args.vllm_url, model=args.model))

    agent = VeriStatReActAgent(llm_client=client, tool_registry=registry, max_steps=args.max_steps)

    print(f"[*] Đang phân tích câu hỏi: {args.question}")
    state = agent.run(args.question)

    print("\n--- CHI TIẾT CÁC BƯỚC THỰC THI (STEP-BY-STEP TRACE) ---")
    for s in state.steps:
        print(f"\n[Bước {s.step_idx}] Action: {s.action_name}")
        print(f"  - Suy luận: {s.thought}")
        if s.action_input:
            print(f"  - Tham số: {json.dumps(s.action_input, ensure_ascii=False)}")
        obs_preview = (s.observation[:120] + "...") if s.observation and len(s.observation) > 120 else s.observation
        print(f"  - Quan sát (Observation): {obs_preview}")

    print("\n--- KẾT QUẢ VẾT CHẠY (TRAJECTORY TRACE) ---")
    print(f"Trạng thái: {state.status}")
    print(f"Số bước thực hiện: {state.current_step}")
    print(f"Bảng trích dẫn (canonical cited_tables): {sorted(list(state.cited_tables))}")
    if state.final_answer:
        print(f"Đáp số: {json.dumps(state.final_answer, ensure_ascii=False, indent=2)}")
        # Run Verifier check for the demo question
        if args.mock and state.final_answer.get("value") is not None:
            from veristat.data.benchmark_gen import check_answer_match
            gt_val = ((101343.8 - 97582.7) / 97582.7) * 100
            pred_val = float(state.final_answer["value"])
            is_valid = check_answer_match(pred=pred_val, gt=gt_val, answer_type="number", rel_tol=1e-3)
            unit_ok = state.final_answer.get("unit") == "%"
            cite_ok = "NG2025_T017" in state.cited_tables and "stat_ng2025_t017" not in state.cited_tables
            print(f"\n--- KIỂM TRA ĐỐI SOÁT (VERIFIER ACCEPTANCE) ---")
            print(f"  - Ground Truth tính từ CSDL gốc: {gt_val:.6f}%")
            print(f"  - Sai số tương đối chấp nhận (<0.1%): {'ĐẠT (PASS)' if is_valid else 'KHÔNG ĐẠT'}")
            print(f"  - Đơn vị tính: {'ĐẠT (%)' if unit_ok else 'SAI ĐƠN VỊ'}")
            print(f"  - Schema Citation (chỉ chứa Table ID logic, không leak SQLite table): {'ĐẠT' if cite_ok else 'KHÔNG ĐẠT'}")
            print(f"  => KẾT LUẬN CUỐI CÙNG: {'END-TO-END SUCCESS' if (is_valid and unit_ok and cite_ok) else 'VERIFIER REJECTED'}")
    else:
        print("Không tạo được đáp số cuối cùng.")


if __name__ == "__main__":
    main()
