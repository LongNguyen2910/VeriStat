import itertools

import pytest

from tests.agents.mocks import (
    MockLLMClient,
    MockStepEnvironment,
    make_question,
    make_trajectory,
)
from veristat.agents.best_of_n import (
    calibrate_verifier_threshold,
    run_best_of_n,
    select_best_of_n,
)
from veristat.agents.sc import cluster_and_vote, run_self_consistency
from veristat.agents.types import LLMResponse
from veristat.verifier import HybridVerifier

VALID_TABLES = frozenset({"NG2025_T017", "NG2025_T020"})


def test_sc_numeric_clustering_median_and_zero_boundary():
    # Two numbers within < 0.1% cluster together; median selected
    t1 = make_trajectory(answer=100.00, py_values=(100.00,))
    t2 = make_trajectory(answer=999.00, py_values=(999.00,))
    t3 = make_trajectory(answer=100.04, py_values=(100.04,))
    t4 = make_trajectory(answer=100.08, py_values=(100.08,))

    res = cluster_and_vote([t1, t2, t3, t4], "number")
    assert res.answer == pytest.approx(100.04)
    assert res.selected_trajectory is t1
    assert len(res.clusters) == 2

    # Edge case: ground truth / clustered values at 0.0
    z1 = make_trajectory(answer=0.0, py_values=(0.0,))
    z2 = make_trajectory(answer=1e-7, py_values=(1e-7,))
    z3 = make_trajectory(answer=5.0, py_values=(5.0,))
    res_zero = cluster_and_vote([z1, z2, z3], "number")
    assert res_zero.answer == pytest.approx(5e-8)


def test_sc_entity_and_entity_list_admin_mapping_and_ties():
    # entity normalization: TP.HCM == TP. Hồ Chí Minh; Thừa Thiên Huế == Huế
    e1 = make_trajectory(answer="TP.HCM", py_values=("TP.HCM",))
    e2 = make_trajectory(answer="Thừa Thiên Huế", py_values=("Thừa Thiên Huế",))
    e3 = make_trajectory(answer="TP. Hồ Chí Minh", py_values=("TP. Hồ Chí Minh",))
    res_ent = cluster_and_vote([e1, e2, e3], "entity")
    assert res_ent.answer == "TP. Hồ Chí Minh"
    assert res_ent.selected_trajectory is e1

    # entity_list: order-sensitive, ties broken by first appearance
    l1 = make_trajectory(answer=["TP.HCM", "Hà Tây"], py_values=(["TP.HCM", "Hà Tây"],))
    l2 = make_trajectory(answer=["Hà Nội", "TP. Hồ Chí Minh"], py_values=(["Hà Nội", "TP. Hồ Chí Minh"],))
    l3 = make_trajectory(answer=["TP. Hồ Chí Minh", "Hà Nội"], py_values=(["TP. Hồ Chí Minh", "Hà Nội"],))
    res_list = cluster_and_vote([l1, l2, l3], "entity_list")
    assert res_list.answer == ["TP. Hồ Chí Minh", "Hà Nội"]
    assert res_list.selected_trajectory is l1

    # All invalid trajectories -> None
    bad1 = make_trajectory(include_final=False)
    bad2 = make_trajectory(safety_violation=True)
    res_bad = cluster_and_vote([bad1, bad2], "number")
    assert res_bad.answer is None
    assert res_bad.selected_trajectory is None


@pytest.mark.parametrize("n", [3, 5, 8])
def test_sc_theorem_b5_early_stop_equivalence(n: int):
    majority = n // 2 + 1
    correct = [
        make_trajectory(answer=10.0, py_values=(10.0,), prompt_tokens=100, completion_tokens=40)
        for _ in range(majority)
    ]
    wrong = [
        make_trajectory(answer=99.0, py_values=(99.0,), prompt_tokens=100, completion_tokens=40)
        for _ in range(n - majority)
    ]
    pool = correct + wrong

    for idx, perm in enumerate(itertools.permutations(pool)):
        if idx >= 30:
            break
        full_res = cluster_and_vote(perm, "number", n_budget=n, early_stop=False)
        early_res = cluster_and_vote(perm, "number", n_budget=n, early_stop=True)
        assert early_res.answer == pytest.approx(full_res.answer)
        assert early_res.selected_trajectory is full_res.selected_trajectory
        assert early_res.calls_used <= n
        if perm[-1] is wrong[-1]:
            assert early_res.early_stopped is True
            assert early_res.calls_saved > 0
            assert early_res.tokens_saved > 0


def test_run_self_consistency_live_early_stop_and_token_budget():
    q = make_question()
    trajs = [
        make_trajectory(answer=10.0, py_values=(10.0,), prompt_tokens=60, completion_tokens=40)
        for _ in range(5)
    ]
    env = MockStepEnvironment(react_trajectories=trajs)
    res = run_self_consistency(env, q, n=5, early_stop=True, seed=10)
    assert res.early_stopped is True
    assert res.calls_used == 3  # majority for n=5 is 3
    assert res.calls_saved == 2
    assert len(env.run_react_calls) == 3

    # Token budget halts before exceeding cap
    env2 = MockStepEnvironment(react_trajectories=trajs)
    res_budget = run_self_consistency(env2, q, n=5, early_stop=False, token_budget=200)
    assert len(env2.run_react_calls) == 2
    assert res_budget.total_tokens == 200


def test_best_of_n_argmax_tau_and_token_accounting():
    q = make_question(ground_truth=10.0)
    # Candidate 0: fails Tier 1 (hallucinated) -> 0.0, 0 verifier tokens
    t0 = make_trajectory(answer=99.0, py_values=(10.0,), prompt_tokens=100, completion_tokens=50)
    # Candidate 1: passes Tier 1, Tier 2 gives 0.6
    t1 = make_trajectory(answer=12.0, py_values=(12.0,), prompt_tokens=100, completion_tokens=50)
    # Candidate 2: passes Tier 1, Tier 2 gives 0.9
    t2 = make_trajectory(answer=10.0, py_values=(10.0,), prompt_tokens=100, completion_tokens=50)

    def responder(messages, temperature, max_tokens, seed, logprobs):
        content = messages[1]["content"]
        if "Đáp án cuối: 10.0" in content:
            txt = '{"table_and_geo": 1.0, "formula": 1.0, "unit_and_type": 0.5, "verdict": "yes", "reason": "tốt"}'
        else:
            txt = '{"table_and_geo": 1.0, "formula": 0.5, "unit_and_type": 0.0, "verdict": "no", "reason": "lệch"}'
        return LLMResponse(text=txt, prompt_tokens=70, completion_tokens=30, latency_s=0.1)

    hv = HybridVerifier(MockLLMClient(responder), VALID_TABLES)
    env = MockStepEnvironment(react_trajectories=[t0, t1, t2])
    res = run_best_of_n(env, hv, q, n=3, tau=0.8, seed=42)

    assert res.selected_index == 2
    assert res.answer == 10.0
    assert res.best_score == pytest.approx(0.9)
    # Generation: 3 * 150 = 450; Verifier called twice (t0 short-circuited): 2 * 100 = 200
    assert res.generation_prompt_tokens == 300
    assert res.generation_completion_tokens == 150
    assert res.verifier_prompt_tokens == 140
    assert res.verifier_completion_tokens == 60
    assert res.total_tokens == 650

    assert res.threshold_result is not None
    assert res.threshold_result.accepted_indices == (2,)
    assert res.threshold_result.abstained is False
    assert res.threshold_result.fallback_answer == 10.0
    assert res.threshold_result.abstain_answer == 10.0

    # When 0 candidates meet high tau=0.95 -> fallback picks one, abstain returns None
    res_high_tau = select_best_of_n(res.trajectories, res.verifications, tau=0.95, seed=42)
    assert res_high_tau.threshold_result is not None
    assert res_high_tau.threshold_result.accepted_indices == ()
    assert res_high_tau.threshold_result.abstained is True
    assert res_high_tau.threshold_result.abstain_answer is None
    assert res_high_tau.threshold_result.fallback_selected_index in (0, 1, 2)


def test_calibrate_verifier_threshold():
    q1 = make_question(question_id="Q1", ground_truth=10.0)
    q2 = make_question(question_id="Q2", ground_truth=20.0)

    def responder(messages, temperature, max_tokens, seed, logprobs):
        content = messages[1]["content"]
        if "Đáp án cuối: 10.0" in content or "Đáp án cuối: 20.0" in content:
            txt = '{"table_and_geo": 1, "formula": 1, "unit_and_type": 1, "verdict": "yes", "reason": "đúng"}'
        else:
            txt = '{"table_and_geo": 0.5, "formula": 0.0, "unit_and_type": 0.5, "verdict": "no", "reason": "sai"}'
        return LLMResponse(text=txt, prompt_tokens=50, completion_tokens=20, latency_s=0.05)

    hv = HybridVerifier(MockLLMClient(responder), VALID_TABLES)

    q1_trajs = [
        make_trajectory(question_id="Q1", answer=99.0, py_values=(99.0,)),  # score 0.3
        make_trajectory(question_id="Q1", answer=10.0, py_values=(10.0,)),  # score 1.0
    ]
    q2_trajs = [
        make_trajectory(question_id="Q2", answer=88.0, py_values=(88.0,)),  # score 0.3
        make_trajectory(question_id="Q2", answer=20.0, py_values=(20.0,)),  # score 1.0
    ]
    trajs_by_q = {"Q1": q1_trajs, "Q2": q2_trajs}
    vers_by_q = {
        "Q1": [hv.verify(q1, t) for t in q1_trajs],
        "Q2": [hv.verify(q2, t) for t in q2_trajs],
    }

    cal = calibrate_verifier_threshold(
        trajs_by_q,
        vers_by_q,
        [q1, q2],
        candidate_taus=[0.2, 0.5, 0.9],
        seed=42,
    )
    assert cal["best_tau"] in (0.5, 0.9)
    assert cal["best_accuracy"] == pytest.approx(1.0)
    assert cal["best_tpr"] == pytest.approx(1.0)
    assert cal["best_fpr"] == pytest.approx(0.0)
    assert len(cal["grid"]) == 3
