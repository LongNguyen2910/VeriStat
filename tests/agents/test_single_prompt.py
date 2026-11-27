"""Unit tests for VeriStat Single Prompt Baseline Agent."""

from __future__ import annotations

import json
from pathlib import Path

from veristat.agents.single_prompt import (
    build_catalog_context,
    build_single_prompt,
    parse_single_prompt_response,
    run_batch_single_prompt,
    run_single_prompt,
)
from veristat.agents.types import Trajectory
from veristat.data.schema import BenchmarkQuestion
from veristat.eval.metrics import EvalRecord
from veristat.llm.client_adapter import AdaptiveLLMClient
from veristat.llm.mock_client import MockLLMClient


def test_build_catalog_context() -> None:
    catalog = build_catalog_context()
    assert isinstance(catalog, str)
    assert len(catalog) > 50
    assert "NG2025_T" in catalog


def test_build_single_prompt() -> None:
    q = BenchmarkQuestion(
        question_id="TEST_01",
        question="Tỷ lệ thất nghiệp năm 2024 là bao nhiêu?",
        split="test",
        category="growth",
        difficulty="easy",
        answer_type="number",
        ground_truth=2.28,
        unit="%",
        required_table_ids=["NG2025_T042"],
        reference_code="print(2.28)",
        uses_admin_mapping=False,
    )

    messages = build_single_prompt(q)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "DANH MỤC CSDL BẢNG BIỂU" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "Tỷ lệ thất nghiệp" in messages[1]["content"]
    assert "TEST_01" in messages[1]["content"]


def test_parse_single_prompt_response_formats() -> None:
    # 1. Action finish format
    text1 = json.dumps({
        "thought": "Tra cứu T042",
        "action": "finish",
        "action_input": {
            "value": 2.28,
            "unit": "%",
            "cited_tables": ["NG2025_T042"],
            "computation_code": "val = 2.28",
        },
    })
    fa1 = parse_single_prompt_response(text1)
    assert fa1.parse_ok
    assert fa1.answer == 2.28
    assert fa1.unit == "%"
    assert "NG2025_T042" in fa1.cited_tables

    # 2. Direct JSON format
    text2 = json.dumps({
        "thought": "Tính toán trực tiếp",
        "answer": 42.5,
        "unit": "nghìn người",
        "cited_tables": ["NG2025_T001"],
        "computation_code": "print(42.5)",
    })
    fa2 = parse_single_prompt_response(text2)
    assert fa2.parse_ok
    assert fa2.answer == 42.5
    assert fa2.unit == "nghìn người"

    # 3. Markdown codeblock
    text3 = "Kết quả phân tích:\n```json\n{\"answer\": \"Hà Nội\", \"unit\": \"tỉnh/thành\", \"cited_tables\": [\"NG2025_T001\"]}\n```\nChúc bạn thành công."
    fa3 = parse_single_prompt_response(text3)
    assert fa3.parse_ok
    assert fa3.answer == "Hà Nội"

    # 4. Fallback regex number
    text4 = "Theo số liệu thống kê, đáp án là 105.75% so với năm trước."
    fa4 = parse_single_prompt_response(text4)
    assert fa4.parse_ok
    assert fa4.answer == 105.75


def test_run_single_prompt_and_trajectory(tmp_path: Path) -> None:
    canned = {
        "Câu hỏi thống kê:": json.dumps({
            "thought": "Suy luận 1 lượt",
            "answer": 3.14,
            "unit": "%",
            "cited_tables": ["NG2025_T001"],
            "computation_code": "print(3.14)",
        }),
    }
    raw_llm = MockLLMClient(canned_responses=canned)
    llm = AdaptiveLLMClient(backend_client=raw_llm)

    q = BenchmarkQuestion(
        question_id="TEST_SP_01",
        question="Tăng trưởng GRDP?",
        split="test",
        category="growth",
        difficulty="easy",
        answer_type="number",
        ground_truth=3.14,
        unit="%",
        required_table_ids=["NG2025_T001"],
        reference_code="print(3.14)",
        uses_admin_mapping=False,
    )

    traj, fa = run_single_prompt(q, llm, seed=42)
    assert isinstance(traj, Trajectory)
    assert traj.question_id == "TEST_SP_01"
    assert len(traj.steps) == 1
    assert traj.steps[0].action.tool_name == "final_answer"
    assert traj.final == fa
    assert fa.answer == 3.14
    assert traj.total_tokens > 0


def test_run_batch_single_prompt_and_resume(tmp_path: Path) -> None:
    canned = {
        "Câu hỏi thống kê:": json.dumps({
            "thought": "Batch single prompt answer",
            "answer": 10.0,
            "unit": "%",
            "cited_tables": ["NG2025_T001"],
        }),
    }
    raw_llm = MockLLMClient(canned_responses=canned)
    llm = AdaptiveLLMClient(backend_client=raw_llm)

    q1 = BenchmarkQuestion(
        question_id="BATCH_01",
        question="Câu hỏi 1",
        split="test",
        category="growth",
        difficulty="easy",
        answer_type="number",
        ground_truth=10.0,
        unit="%",
        required_table_ids=["NG2025_T001"],
        reference_code="print(10.0)",
        uses_admin_mapping=False,
    )
    q2 = BenchmarkQuestion(
        question_id="BATCH_02",
        question="Câu hỏi 2",
        split="test",
        category="share",
        difficulty="medium",
        answer_type="number",
        ground_truth=20.0,
        unit="%",
        required_table_ids=["NG2025_T001"],
        reference_code="print(20.0)",
        uses_admin_mapping=False,
    )

    out_file = tmp_path / "single_results.jsonl"

    # Run batch
    summary1, records1 = run_batch_single_prompt([q1, q2], llm, out_file, seed=42, resume=True)
    assert len(records1) == 2
    assert summary1["count"] == 2
    assert out_file.exists()

    # Re-run batch with resume should skip and read existing records without making LLM calls
    initial_calls = raw_llm.call_count
    summary2, records2 = run_batch_single_prompt([q1, q2], llm, out_file, seed=42, resume=True)
    assert len(records2) == 2
    assert raw_llm.call_count == initial_calls
