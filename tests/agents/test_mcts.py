import math

import pytest

from tests.agents.mocks import (
    MockLLMClient,
    MockStepEnvironment,
    make_question,
)
from veristat.agents.mcts import MCTSNode, canonical_action_key, run_mcts
from veristat.agents.types import LLMResponse, StepObservation, ToolAction
from veristat.verifier import HybridVerifier

VALID_TABLES = frozenset({"NG2025_T017", "NG2025_T020"})


def test_mcts_node_uct_score_exact_and_unvisited():
    root = MCTSNode(node_id=0, parent=None, step=None, depth=0, action_key="root")
    root.visits = 10
    root.value_sum = 6.0

    unvisited = MCTSNode(node_id=1, parent=root, step=None, depth=1, action_key="a1")
    assert math.isinf(unvisited.uct_score(c=math.sqrt(2.0)))

    visited = MCTSNode(node_id=2, parent=root, step=None, depth=1, action_key="a2")
    visited.visits = 4
    visited.value_sum = 3.0  # q = 0.75
    c = math.sqrt(2.0)
    expected = 0.75 + c * math.sqrt(math.log(10) / 4.0)
    assert visited.uct_score(c=c) == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_mcts_four_phases_dedup_zero_intermediate_llm_and_exhausted():
    q = make_question(ground_truth=10.0)

    def propose_fn(question, history, k, temperature, seed):
        depth = len(history)
        if depth == 0:
            # Return duplicate SQL + unique SQL to verify dedup
            a1 = ToolAction("s0", "query_table", {"sql": "SELECT 1"}, "r1", 20, 10, 0.05)
            a1_dup = ToolAction("s0_dup", "query_table", {"sql": "SELECT 1"}, "r1", 20, 10, 0.05)
            a2 = ToolAction("s0_b", "query_table", {"sql": "SELECT 2"}, "r2", 20, 10, 0.05)
            return [a1, a1_dup, a2]
        if depth == 1:
            if history[0].action.tool_args["sql"] == "SELECT 1":
                return [
                    ToolAction("py1", "run_python", {"code": "res=10"}, "r3", 20, 10, 0.05),
                ]
            return [
                ToolAction("py2", "run_python", {"code": "res=20"}, "r4", 20, 10, 0.05),
            ]
        # depth >= 2: final answers
        if history[0].action.tool_args["sql"] == "SELECT 1":
            ans = 10.0
        else:
            ans = 20.0
        return [
            ToolAction(
                "done",
                "final_answer",
                {
                    "answer": ans,
                    "unit": "%",
                    "cited_tables": ["NG2025_T017"],
                    "computation_code": "res=10",
                    "parse_ok": True,
                },
                f"final:{ans}",
                15,
                10,
                0.05,
            )
        ]

    def execute_fn(action, history):
        if action.tool_name == "query_table":
            return StepObservation(
                output=[{"gia_tri": 100.0}],
                ok=True,
                queried_tables=("NG2025_T017",),
                computed_values=(),
                latency_s=0.01,
            )
        if action.tool_name == "run_python":
            val = 10.0 if action.tool_args["code"] == "res=10" else 20.0
            return StepObservation(
                output=str(val),
                ok=True,
                computed_values=(val,),
                latency_s=0.01,
            )
        return StepObservation(output=None, ok=True)

    llm_calls_seen_on_terminal = []

    def llm_responder(messages, temperature, max_tokens, seed, logprobs):
        content = messages[1]["content"]
        llm_calls_seen_on_terminal.append(content)
        if "Đáp án cuối: 10.0" in content:
            txt = '{"table_and_geo": 1.0, "formula": 1.0, "unit_and_type": 1.0, "verdict": "yes", "reason": "đúng"}'
        else:
            txt = '{"table_and_geo": 1.0, "formula": 0.0, "unit_and_type": 0.5, "verdict": "no", "reason": "sai"}'
        return LLMResponse(text=txt, prompt_tokens=50, completion_tokens=20, latency_s=0.05)

    mock_llm = MockLLMClient(llm_responder)
    hv = HybridVerifier(mock_llm, VALID_TABLES)
    env = MockStepEnvironment(propose_fn=propose_fn, execute_fn=execute_fn)

    # First, run 1 iteration -> expands root (2 unique query_table nodes), 0 LLM verifier calls!
    res_iter1 = run_mcts(env, hv, q, max_iterations=1, max_depth=6, branching_factor=2, seed=42)
    assert res_iter1.nodes_created == 3  # root + 2 deduped query children
    assert res_iter1.terminal_nodes_evaluated == 0
    assert len(mock_llm.calls) == 0

    # Reset mock LLM and run full search -> explores both branches, finishes when exhausted
    mock_llm2 = MockLLMClient(llm_responder)
    hv2 = HybridVerifier(mock_llm2, VALID_TABLES)
    env2 = MockStepEnvironment(propose_fn=propose_fn, execute_fn=execute_fn)
    res_full = run_mcts(env2, hv2, q, max_iterations=20, max_depth=6, branching_factor=2, seed=42)

    assert res_full.terminated_reason == "final_answer"
    assert res_full.answer == 10.0
    assert res_full.best_score == pytest.approx(1.0)
    assert res_full.terminal_nodes_evaluated == 2
    assert len(mock_llm2.calls) == 2  # LLM called strictly for the 2 terminal nodes
    assert res_full.iterations_run < 20  # stops cleanly once tree exhausted
    assert res_full.root_visits >= 6


def test_mcts_prunes_safety_violation_and_enforces_token_budget():
    q = make_question(ground_truth=10.0)

    def propose_fn(question, history, k, temperature, seed):
        if not history:
            return [
                ToolAction("bad", "run_python", {"code": "import os"}, "bad", 40, 20, 0.05),
                ToolAction("good", "query_table", {"sql": "SELECT 1"}, "good", 40, 20, 0.05),
            ]
        return [
            ToolAction(
                "done",
                "final_answer",
                {
                    "answer": 10.0,
                    "unit": "%",
                    "cited_tables": ["NG2025_T017"],
                    "computation_code": "10",
                    "parse_ok": True,
                },
                "done",
                30,
                20,
                0.05,
            )
        ]

    def execute_fn(action, history):
        if action.tool_name == "run_python":
            return StepObservation(
                output=None,
                ok=False,
                error="Blocked import os",
                safety_violation=True,
                latency_s=0.01,
            )
        return StepObservation(
            output=[{"gia_tri": 10.0}],
            ok=True,
            queried_tables=("NG2025_T017",),
            computed_values=(10.0,),
            latency_s=0.01,
        )

    mock_llm = MockLLMClient(
        LLMResponse(
            text='{"table_and_geo": 1, "formula": 1, "unit_and_type": 1, "verdict": "yes", "reason": "ok"}',
            prompt_tokens=30,
            completion_tokens=10,
            latency_s=0.02,
        )
    )
    hv = HybridVerifier(mock_llm, VALID_TABLES)
    env = MockStepEnvironment(propose_fn=propose_fn, execute_fn=execute_fn)

    # Full run skips the safety-pruned branch and reaches valid final answer
    res = run_mcts(env, hv, q, max_iterations=6, max_depth=4, seed=42)
    assert res.terminated_reason == "final_answer"
    assert res.answer == 10.0
    assert res.best_trajectory.has_safety_violation is False

    # Very tight token_budget=100 stops after first iteration without valid final answer
    env_tight = MockStepEnvironment(propose_fn=propose_fn, execute_fn=execute_fn)
    res_tight = run_mcts(
        env_tight,
        hv,
        q,
        max_iterations=10,
        max_depth=4,
        token_budget=100,
        seed=42,
    )
    assert res_tight.terminated_reason == "token_budget"
    assert res_tight.iterations_run == 1


def test_canonical_action_key_deterministic():
    a1 = ToolAction("t1", "query_table", {"a": 1, "b": 2}, "r", 1, 1, 0.1)
    a2 = ToolAction("t2", "query_table", {"b": 2, "a": 1}, "r", 1, 1, 0.1)
    assert canonical_action_key(a1) == canonical_action_key(a2)
