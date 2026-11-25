"""Unit tests for AgentState cloneability, serialization, and branch isolation."""

import pytest
from veristat.agents.contracts import AgentAction, ToolCall, VerificationResult
from veristat.agents.state import AgentState, StepRecord
from veristat.tools.base import BaseTool, ToolRegistry, ToolResult


def test_agent_state_deep_clone_isolation():
    state1 = AgentState(
        question_id="Q001",
        question="Tăng trưởng dân số năm 2024?",
        current_step=1,
        steps=[
            StepRecord(
                step_idx=1,
                thought="Cần tra cứu bảng dân số",
                action_name="query_table",
                action_input={"sql": "SELECT * FROM NG2025_T015 LIMIT 1"},
                observation="Dân số: 100300 nghìn người",
            )
        ],
        cited_tables={"NG2025_T015"},
    )

    # Clone state for MCTS branch
    state2 = state1.clone()
    assert state2.question_id == state1.question_id
    assert len(state2.steps) == 1
    assert "NG2025_T015" in state2.cited_tables

    # Mutate branch 2
    state2.current_step = 2
    state2.steps.append(
        StepRecord(
            step_idx=2,
            thought="Tính tỷ lệ",
            action_name="run_python",
            action_input={"code": "print(100300)"},
            observation="100300",
        )
    )
    state2.cited_tables.add("NG2025_T017")

    # Verify branch 1 is completely untouched
    assert len(state1.steps) == 1
    assert state1.current_step == 1
    assert state1.cited_tables == {"NG2025_T015"}
    assert len(state2.steps) == 2
    assert state2.cited_tables == {"NG2025_T015", "NG2025_T017"}


def test_agent_state_serialization_roundtrip():
    state = AgentState(
        question_id="Q002",
        question="Tỷ trọng doanh nghiệp?",
        current_step=1,
        steps=[
            StepRecord(
                step_idx=1,
                thought="Tra cứu",
                action_name="search_docs",
                action_input={"query": "doanh nghiệp"},
                observation="Tìm thấy bảng NG2025_T136",
                tool_error=None,
            )
        ],
        cited_tables={"NG2025_T136"},
        status="RUNNING",
    )

    data = state.to_dict()
    assert data["question_id"] == "Q002"
    assert data["cited_tables"] == ["NG2025_T136"]

    restored = AgentState.from_dict(data)
    assert restored.question_id == state.question_id
    assert restored.current_step == state.current_step
    assert restored.cited_tables == state.cited_tables
    assert len(restored.steps) == 1
    assert restored.steps[0].action_name == "search_docs"


def test_tool_registry_and_contracts():
    class DummyTool(BaseTool):
        name = "dummy"
        description = "A dummy test tool"
        input_schema = {
            "type": "object",
            "properties": {"val": {"type": "string"}},
            "required": ["val"],
        }

        def execute(self, **kwargs):
            return ToolResult(tool_name=self.name, success=True, output=f"result: {kwargs.get('val')}")

    registry = ToolRegistry()
    registry.register(DummyTool())

    assert registry.get("dummy") is not None
    assert registry.get("non_existent") is None
    schemas = registry.get_schemas()
    assert len(schemas) == 1
    assert schemas[0]["name"] == "dummy"

    res = registry.get("dummy").execute(val="test_123")
    assert res.success is True
    assert res.output == "result: test_123"
