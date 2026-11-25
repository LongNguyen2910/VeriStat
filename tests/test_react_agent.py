"""Unit tests for VeriStatReActAgent and action parser."""

import pytest
from veristat.agents.parser import parse_agent_action
from veristat.agents.react import VeriStatReActAgent
from veristat.agents.state import AgentState
from veristat.llm.mock_client import MockLLMClient
from veristat.tools.base import BaseTool, ToolRegistry, ToolResult


def test_parser_extracts_json_action():
    text = """
```json
{
  "thought": "Cần tra cứu bảng dân số",
  "action": "query_table",
  "action_input": {"sql": "SELECT * FROM NG2025_T015"}
}
```
"""
    action = parse_agent_action(text)
    assert action.is_finish is False
    assert action.tool_call is not None
    assert action.tool_call.tool_name == "query_table"
    assert action.tool_call.tool_input["sql"] == "SELECT * FROM NG2025_T015"


def test_parser_extracts_finish_action():
    text = """
{
  "thought": "Đã tính ra kết quả",
  "action": "finish",
  "action_input": {
    "value": 15.4,
    "answer_type": "number",
    "unit": "%",
    "cited_tables": ["NG2025_T015"]
  }
}
"""
    action = parse_agent_action(text)
    assert action.is_finish is True
    assert action.final_answer["value"] == 15.4
    assert action.final_answer["unit"] == "%"


def test_react_agent_full_trajectory_with_mock():
    # Setup dummy tool
    class DummyQueryTool(BaseTool):
        name = "query_table"
        description = "query db"
        input_schema = {}

        def execute(self, **kwargs):
            return ToolResult(tool_name=self.name, success=True, output="dân số: 100M", metadata={"tables": ["NG2025_T015"]})

    registry = ToolRegistry()
    registry.register(DummyQueryTool())

    # Setup 2-step mock LLM: step 1 calls query_table, step 2 finishes
    mock_responses = {
        "Câu hỏi thống kê: Tăng trưởng dân số": (
            '{"thought": "Tra cứu số liệu", "action": "query_table", "action_input": {"sql": "SELECT"}}'
        ),
        "Observation: dân số: 100M": (
            '{"thought": "Đã có số", "action": "finish", "action_input": {"value": 100.0, "unit": "triệu người", "cited_tables": ["NG2025_T015"]}}'
        ),
    }
    client = MockLLMClient(canned_responses=mock_responses)
    agent = VeriStatReActAgent(llm_client=client, tool_registry=registry, max_steps=5)

    final_state = agent.run("Tăng trưởng dân số", question_id="Q_TEST")

    assert final_state.is_terminal is True
    assert final_state.status == "SUCCESS"
    assert len(final_state.steps) == 2
    assert "NG2025_T015" in final_state.cited_tables
    assert final_state.final_answer["value"] == 100.0


def test_react_agent_e2e_four_steps_with_real_tools_and_verifier():
    """End-to-End verified 4-step trajectory: search_docs -> query_table -> run_python -> finish."""
    from veristat.data.benchmark_gen import check_answer_match
    from veristat.tools.docs_tool import SearchDocsTool
    from veristat.tools.python_tool import RunPythonTool
    from veristat.tools.table_tool import QueryTableTool

    registry = ToolRegistry()
    registry.register(SearchDocsTool(db_path="data/processed/veristat.db"))
    registry.register(QueryTableTool(db_path="data/processed/veristat.db"))
    registry.register(RunPythonTool())

    question = "Tốc độ tăng trưởng (%) dân số trung bình của Cả nước giai đoạn 2020–2024 là bao nhiêu?"

    # Define 4-step canned LLM responses
    canned_dialogue = {
        question: (
            '{"thought": "Tìm kiếm bảng thống kê về dân số trung bình.", '
            '"action": "search_docs", "action_input": {"query": "dân số trung bình"}}'
        ),
        "NG2025_T017": (
            '{"thought": "Tìm thấy bảng NG2025_T017. Truy vấn số liệu dân số trung bình cả nước năm 2020 và 2024.", '
            '"action": "query_table", "action_input": {"sql": "SELECT nam, gia_tri FROM stat_ng2025_t017 WHERE geo_level = \'country\' AND nam IN (2020, 2024) ORDER BY nam ASC"}}'
        ),
        "101343.8": (
            '{"thought": "Đã có số liệu: 2020=97582.7, 2024=101343.8 nghìn người. Dùng python sandbox tính tốc độ tăng trưởng.", '
            '"action": "run_python", "action_input": {"code": "v20 = 97582.7\\nv24 = 101343.8\\ngrowth = ((v24 - v20) / v20) * 100\\nprint(f\'GROWTH={growth:.6f}\')"}}'
        ),
        "GROWTH=3.854269": (
            '{"thought": "Kết quả tính toán tăng trưởng là 3.8543%. Kết thúc và đưa ra đáp số chuẩn.", '
            '"action": "finish", "action_input": {"value": 3.854269, "answer_type": "number", "unit": "%", "cited_tables": ["NG2025_T017"]}}'
        ),
    }

    client = MockLLMClient(canned_responses=canned_dialogue)
    agent = VeriStatReActAgent(llm_client=client, tool_registry=registry, max_steps=6)

    state = agent.run(question, question_id="Q_E2E_001")

    # 1. Trajectory assertions
    assert state.is_terminal is True
    assert state.status == "SUCCESS"
    assert len(state.steps) == 4
    assert [s.action_name for s in state.steps] == ["search_docs", "query_table", "run_python", "finish"]

    # 2. Strict citation assertion (no raw stat_ table leaks, only valid table_id)
    assert state.cited_tables == {"NG2025_T017"}

    # 3. Verifier assertion
    gt_value = ((101343.8 - 97582.7) / 97582.7) * 100
    pred_value = state.final_answer["value"]
    assert check_answer_match(pred=pred_value, gt=gt_value, answer_type="number", rel_tol=1e-3) is True
    assert state.final_answer["unit"] == "%"
