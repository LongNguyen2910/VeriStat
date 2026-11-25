import json
from pathlib import Path

import pytest

from tests.agents.mocks import MockLLMClient, make_question, make_trajectory
from veristat.agents.types import LLMResponse, ReActStep, StepObservation, ToolAction
from veristat.verifier import (
    HybridVerifier,
    load_valid_table_ids,
    parse_tier2_response,
    score_intermediate_state,
    verify_tier1,
)

VALID_TABLES = frozenset({"NG2025_T017", "NG2025_T020", "NG2025_T136"})


def test_tier1_passes_valid_grounded_trajectory():
    traj = make_trajectory(
        answer=10.005,
        cited_tables=("NG2025_T017",),
        queried_tables=("NG2025_T017",),
        py_values=(10.0,),
    )
    res = verify_tier1(traj, "number", VALID_TABLES)
    assert res.passed is True
    assert res.score == 1.0
    assert res.reasons == ()


def test_tier1_rule1_safety_violation():
    traj = make_trajectory(safety_violation=True)
    res = verify_tier1(traj, "number", VALID_TABLES)
    assert res.passed is False
    assert res.score == 0.0
    assert "safety_violation" in res.reasons


def test_tier1_rule2_requires_successful_query_or_python():
    search_only_step = ReActStep(
        step_idx=0,
        action=ToolAction("t", "search_docs", {"query": "dân số"}, "raw", 10, 5, 0.1),
        observation=StepObservation(output=[{"table_id": "NG2025_T017"}], ok=True),
    )
    traj = make_trajectory(include_query_step=False, py_values=())
    from dataclasses import replace

    traj_search_only = replace(traj, steps=(search_only_step, traj.steps[-1]))
    res = verify_tier1(traj_search_only, "number", VALID_TABLES)
    assert res.passed is False
    assert "no_successful_tool_call" in res.reasons


def test_tier1_rule3_citations_checks():
    # Missing cited_tables
    t_empty = make_trajectory(cited_tables=(), queried_tables=("NG2025_T017",))
    r_empty = verify_tier1(t_empty, "number", VALID_TABLES)
    assert r_empty.passed is False
    assert "missing_cited_tables" in r_empty.reasons

    # Unknown table_id
    t_unknown = make_trajectory(
        cited_tables=("FAKE_TABLE",), queried_tables=("FAKE_TABLE",)
    )
    r_unknown = verify_tier1(t_unknown, "number", VALID_TABLES)
    assert r_unknown.passed is False
    assert "unknown_cited_table:FAKE_TABLE" in r_unknown.reasons

    # Valid table_id but never actually queried in trace
    t_unqueried = make_trajectory(
        cited_tables=("NG2025_T020",), queried_tables=("NG2025_T017",)
    )
    r_unqueried = verify_tier1(t_unqueried, "number", VALID_TABLES)
    assert r_unqueried.passed is False
    assert "unqueried_cited_table:NG2025_T020" in r_unqueried.reasons

    # Invalid final answer
    t_no_final = make_trajectory(include_final=False)
    r_no_final = verify_tier1(t_no_final, "number", VALID_TABLES)
    assert r_no_final.passed is False
    assert "invalid_final_answer" in r_no_final.reasons


def test_tier1_rule4_anti_hallucination_prefers_python_over_sql():
    # Hallucinated answer not matching run_python output (even though it matches SQL row!)
    traj = make_trajectory(
        answer=100.0,
        sql_values=(100.0, 110.0),
        py_values=(10.0,),
    )
    res = verify_tier1(traj, "number", VALID_TABLES)
    assert res.passed is False
    assert "ungrounded_final_answer" in res.reasons

    # Direct lookup with no run_python falls back to query_table computed_values
    traj_sql_only = make_trajectory(
        answer="TP.HCM",
        sql_values=("TP. Hồ Chí Minh",),
        py_values=(),
    )
    res_sql = verify_tier1(traj_sql_only, "entity", VALID_TABLES)
    assert res_sql.passed is True


def test_hybrid_verifier_short_circuits_tier2_on_tier1_fail():
    mock_llm = MockLLMClient(
        LLMResponse(
            text='{"table_and_geo": 1, "formula": 1, "unit_and_type": 1, "verdict": "yes", "reason": "ok"}',
            prompt_tokens=120,
            completion_tokens=30,
            latency_s=0.1,
        )
    )
    verifier = HybridVerifier(mock_llm, VALID_TABLES)
    question = make_question()

    bad_traj = make_trajectory(answer=999.0, py_values=(10.0,))
    bad_res = verifier.verify(question, bad_traj, seed=1)
    assert bad_res.tier1_passed is False
    assert bad_res.score == 0.0
    assert bad_res.tier2 is None
    assert bad_res.verifier_total_tokens == 0
    assert len(mock_llm.calls) == 0

    good_traj = make_trajectory(answer=10.0, py_values=(10.0,))
    good_res = verifier.verify(question, good_traj, seed=1)
    assert good_res.tier1_passed is True
    assert good_res.score == pytest.approx(1.0)
    assert good_res.tier2 is not None
    assert good_res.verifier_total_tokens == 150
    assert len(mock_llm.calls) == 1


def test_hybrid_verifier_loads_real_veristat_db_readonly():
    db_path = Path("data/processed/veristat.db")
    if not db_path.exists():
        pytest.skip("data/processed/veristat.db not built")
    table_ids = load_valid_table_ids(db_path)
    assert len(table_ids) == 32
    assert "NG2025_T017" in table_ids
    assert "B1026_S05" in table_ids

    mock_llm = MockLLMClient(
        LLMResponse(text="{}", prompt_tokens=10, completion_tokens=5, latency_s=0.05)
    )
    hv = HybridVerifier.from_db(mock_llm, db_path)
    assert len(hv.valid_table_ids) == 32


def test_tier2_rubric_parsing_fences_fallback_and_logprobs():
    fenced = 'Đánh giá:\n```json\n{"table_and_geo": 1.0, "formula": 0.5, "unit_and_type": 0.0, "verdict": "no", "reason": "sai công thức"}\n```'
    tg, fm, ut, verdict, reason, fb = parse_tier2_response(fenced)
    assert (tg, fm, ut, verdict, reason, fb) == (1.0, 0.5, 0.0, "no", "sai công thức", False)

    # Malformed JSON falls back conservatively to 0.5
    malformed = "Không thể chấm điểm lúc này"
    tg_f, fm_f, ut_f, v_f, r_f, fb_f = parse_tier2_response(malformed)
    assert (tg_f, fm_f, ut_f, v_f, fb_f) == (0.5, 0.5, 0.5, "unknown", True)
    assert r_f == "parse_fallback"

    # Weighted score 0.4*1 + 0.4*0.5 + 0.2*0 = 0.60 blended with logprobs
    # logprobs: "Yes" = 0.0, "No" = -10.0 -> p_yes ~ 1.0
    mock_llm = MockLLMClient(
        LLMResponse(
            text=fenced,
            prompt_tokens=80,
            completion_tokens=20,
            latency_s=0.1,
            logprobs={" Yes": 0.0, "No": -100.0},
        )
    )
    hv = HybridVerifier(mock_llm, VALID_TABLES, use_logprobs=True, logprob_weight=0.2)
    res = hv.verify(make_question(), make_trajectory())
    assert res.tier1_passed is True
    assert res.score == pytest.approx(0.8 * 0.6 + 0.2 * 1.0, abs=1e-6)


def test_score_intermediate_state_ordering():
    assert score_intermediate_state([], VALID_TABLES) == 0.5

    unsafe = ReActStep(
        0,
        ToolAction("t", "run_python", {"code": "import os"}, "raw", 10, 5, 0.1),
        StepObservation(output=None, ok=False, safety_violation=True),
    )
    assert score_intermediate_state([unsafe], VALID_TABLES) == 0.0

    search_ok = ReActStep(
        0,
        ToolAction("t", "search_docs", {"query": "dân số"}, "raw", 10, 5, 0.1),
        StepObservation(output=[{"table_id": "NG2025_T017"}], ok=True),
    )
    query_ok = ReActStep(
        1,
        ToolAction("t", "query_table", {"sql": "SELECT 1"}, "raw", 10, 5, 0.1),
        StepObservation(output=[{"gia_tri": 10}], ok=True, queried_tables=("NG2025_T017",)),
    )
    py_ok = ReActStep(
        2,
        ToolAction("t", "run_python", {"code": "result = 10"}, "raw", 10, 5, 0.1),
        StepObservation(output="10", ok=True, computed_values=(10.0,)),
    )
    query_empty = ReActStep(
        1,
        ToolAction("t", "query_table", {"sql": "SELECT 0"}, "raw", 10, 5, 0.1),
        StepObservation(output=[], ok=True, queried_tables=("NG2025_T017",)),
    )
    query_err = ReActStep(
        1,
        ToolAction("t", "query_table", {"sql": "BAD"}, "raw", 10, 5, 0.1),
        StepObservation(output=None, ok=False, error="syntax error"),
    )

    s_good = score_intermediate_state([search_ok, query_ok, py_ok], VALID_TABLES)
    s_empty = score_intermediate_state([search_ok, query_empty], VALID_TABLES)
    s_err = score_intermediate_state([search_ok, query_err], VALID_TABLES)
    s_dup = score_intermediate_state([search_ok, query_ok, query_ok], VALID_TABLES)

    assert s_good > s_empty > s_err
    assert s_dup < score_intermediate_state([search_ok, query_ok], VALID_TABLES)
