"""Integration tests for VeriStatStepEnvironment and AdaptiveLLMClient.

Verifies end-to-end integration between Khoa's ToolRegistry/Sandbox and Sơn's
StepEnvironment/Trajectory contracts, as well as HybridVerifier compatibility.
"""

from __future__ import annotations

import json
from pathlib import Path

from veristat.agents.environment import VeriStatStepEnvironment
from veristat.agents.types import StepEnvironment, ToolAction, Trajectory
from veristat.data.schema import BenchmarkQuestion
from veristat.llm.cache import SQLiteLLMCache
from veristat.llm.client_adapter import AdaptiveLLMClient
from veristat.llm.mock_client import MockLLMClient
from veristat.tools.base import ToolRegistry
from veristat.tools.docs_tool import SearchDocsTool
from veristat.tools.python_tool import RunPythonTool
from veristat.tools.table_tool import QueryTableTool
from veristat.verifier.hybrid import HybridVerifier


def _build_test_env(tmp_path: Path, canned_responses: dict[str, str] | None = None) -> tuple[VeriStatStepEnvironment, SQLiteLLMCache]:
    cache_db = tmp_path / "test_cache.db"
    cache = SQLiteLLMCache(db_path=cache_db)
    mock_backend = MockLLMClient(canned_responses=canned_responses)
    adaptive_llm = AdaptiveLLMClient(backend_client=mock_backend, cache=cache)

    registry = ToolRegistry()
    db_path = Path("data/processed/veristat.db")
    if db_path.exists():
        registry.register(QueryTableTool(db_path=db_path))
        registry.register(SearchDocsTool(db_path=db_path))
    registry.register(RunPythonTool(timeout_seconds=5.0))

    env = VeriStatStepEnvironment(llm=adaptive_llm, tools=registry)
    return env, cache


def test_step_environment_protocol_conformance(tmp_path: Path) -> None:
    env, _ = _build_test_env(tmp_path)
    # Check that VeriStatStepEnvironment satisfies StepEnvironment protocol
    assert isinstance(env, StepEnvironment)


def test_adaptive_llm_cache_integration(tmp_path: Path) -> None:
    canned = {"dân số": '{"thought": "test", "action": "search_docs", "action_input": {"query": "dân số"}}'}
    env, cache = _build_test_env(tmp_path, canned_responses=canned)

    messages = [{"role": "user", "content": "Tìm kiếm dân số"}]
    resp1 = env.llm.generate(messages, temperature=0.0, seed=42)
    assert not resp1.cached
    assert "search_docs" in resp1.text

    # Second call with same parameters must hit cache
    resp2 = env.llm.generate(messages, temperature=0.0, seed=42)
    assert resp2.cached
    assert resp2.text == resp1.text


def test_execute_action_python_sandbox_and_computed_values(tmp_path: Path) -> None:
    env, _ = _build_test_env(tmp_path)

    # Valid Python code execution
    valid_action = ToolAction(
        thought="Tính toán tỷ lệ tăng trưởng",
        tool_name="run_python",
        tool_args={"code": "pop_2024 = 100\npop_2025 = 105.5\ngrowth = (pop_2025 - pop_2024) / pop_2024 * 100\nprint(growth)"},
        raw_text="",
        prompt_tokens=10,
        completion_tokens=20,
        latency_s=0.1,
    )

    obs = env.execute_action(valid_action)
    assert obs.ok
    assert not obs.safety_violation
    assert 5.5 in obs.computed_values


def test_execute_action_catches_ast_safety_violation(tmp_path: Path) -> None:
    env, _ = _build_test_env(tmp_path)

    # Forbidden socket import attempting network access
    bad_action = ToolAction(
        thought="Cố gắng kết nối mạng",
        tool_name="run_python",
        tool_args={"code": "import socket\ns = socket.socket()\ns.connect(('8.8.8.8', 53))"},
        raw_text="",
        prompt_tokens=10,
        completion_tokens=20,
        latency_s=0.1,
    )

    obs = env.execute_action(bad_action)
    assert not obs.ok
    assert obs.safety_violation
    assert "AST_FORBIDDEN" in str(obs.output)


def test_execute_action_table_tool_tracks_queried_tables(tmp_path: Path) -> None:
    env, _ = _build_test_env(tmp_path)
    db_path = Path("data/processed/veristat.db")
    if not db_path.exists():
        return

    action = ToolAction(
        thought="Truy vấn bảng dân số",
        tool_name="query_table",
        tool_args={"sql": "SELECT * FROM stat_ng2025_t017 LIMIT 5"},
        raw_text="",
        prompt_tokens=10,
        completion_tokens=20,
        latency_s=0.1,
    )

    obs = env.execute_action(action)
    assert obs.ok
    assert not obs.safety_violation
    assert any("T017" in t for t in obs.queried_tables)
    assert len(obs.computed_values) > 0


def test_run_react_and_hybrid_verifier_compatibility(tmp_path: Path) -> None:
    # Set up canned response for a 2-step ReAct trajectory: search_docs -> finish
    final_payload = {
        "thought": "Đã tính toán xong kết quả",
        "action": "finish",
        "action_input": {
            "answer": 42.5,
            "unit": "%",
            "cited_tables": ["NG2025_T017"],
            "computation_code": "print(42.5)",
        },
    }

    canned = {
        "Câu hỏi:": json.dumps(final_payload, ensure_ascii=False),
    }

    env, _ = _build_test_env(tmp_path, canned_responses=canned)

    q = BenchmarkQuestion(
        question_id="TEST_Q001",
        question="Tốc độ tăng dân số là bao nhiêu?",
        split="test",
        category="growth",
        difficulty="easy",
        answer_type="number",
        ground_truth=42.5,
        unit="%",
        required_table_ids=["NG2025_T017"],
        reference_code="print(42.5)",
        uses_admin_mapping=False,
    )

    traj = env.run_react(q, max_steps=4)
    assert isinstance(traj, Trajectory)
    assert traj.terminated_reason == "final_answer"
    assert traj.final is not None
    assert traj.final.parse_ok
    assert traj.final.answer == 42.5

    # Verify compatibility with HybridVerifier
    db_path = Path("data/processed/veristat.db")
    if db_path.exists():
        verifier = HybridVerifier.from_db(
            llm=env.llm,
            db_path=db_path,
        )
        res = verifier.verify(q, traj)
        # Trajectory has finish without python or sql step, Tier 1 rule 2 requires tool call
        assert isinstance(res.tier1_passed, bool)
