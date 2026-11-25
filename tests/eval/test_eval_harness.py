import importlib.util
import math
from pathlib import Path

import pytest

from tests.agents.mocks import (
    MockLLMClient,
    MockStepEnvironment,
    make_question,
    make_trajectory,
)
from veristat.agents.types import LLMResponse, StepObservation, ToolAction
from veristat.data.c4_taxonomy import sample_traces_for_c4
from veristat.eval import (
    bootstrap_ci,
    build_eval_record,
    build_stratified_subset,
    collect_mcts_runs,
    collect_trajectory_pool,
    compute_citation_metrics,
    evaluate_from_pool,
    evaluate_prediction,
    export_c4_traces,
    load_questions,
    load_table_domain_map,
    read_jsonl,
    summarize_records,
)
from veristat.verifier import HybridVerifier


def _load_script_module(script_rel: str, module_name: str):
    path = Path(script_rel).resolve()
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_eval_metrics_citations_bootstrap_and_summary():
    q = make_question(ground_truth=10.0, required_table_ids=["NG2025_T017"])
    assert evaluate_prediction(q, 10.005) is True
    assert evaluate_prediction(q, None) is False

    exact, recall = compute_citation_metrics(["NG2025_T017"], ["NG2025_T017"])
    assert exact is True and recall == 1.0
    exact2, recall2 = compute_citation_metrics(["NG2025_T017", "NG2025_T020"], ["NG2025_T017"])
    assert exact2 is False and recall2 == 1.0

    t_good = make_trajectory(answer=10.0, cited_tables=("NG2025_T017",))
    t_bad = make_trajectory(answer=99.0, cited_tables=("NG2025_T020",), safety_violation=True)
    dmap = {"NG2025_T017": "dan_so", "NG2025_T020": "dan_so"}

    r1 = build_eval_record(
        q,
        "react_n1",
        t_good,
        10.0,
        dmap,
        prompt_tokens=100,
        completion_tokens=50,
        verifier_tokens=0,
        latency_s=0.2,
    )
    r2 = build_eval_record(
        q,
        "react_n1",
        t_bad,
        99.0,
        dmap,
        prompt_tokens=100,
        completion_tokens=50,
        verifier_tokens=0,
        latency_s=0.2,
    )
    assert r1.is_correct is True
    assert r2.is_correct is False  # safety violation forces is_correct=False

    summary = summarize_records([r1, r2])
    assert summary["count"] == 2
    assert summary["accuracy"] == pytest.approx(0.5)
    assert summary["safety_violation_rate"] == pytest.approx(0.5)
    assert "dan_so" in summary["by_domain"]
    assert "growth" in summary["by_category"]
    assert "easy" in summary["by_difficulty"]

    ci1 = bootstrap_ci([1.0, 0.0, 1.0, 1.0, 0.0], n_resamples=500, seed=42)
    ci2 = bootstrap_ci([1.0, 0.0, 1.0, 1.0, 0.0], n_resamples=500, seed=42)
    assert ci1 == ci2
    assert ci1["ci_low"] <= ci1["mean"] <= ci1["ci_high"]


def test_build_stratified_subset_preserves_giao_duc_and_strata():
    bench_path = Path("data/benchmark/benchmark_400.jsonl")
    db_path = Path("data/processed/veristat.db")
    if not bench_path.exists() or not db_path.exists():
        pytest.skip("benchmark_400.jsonl or veristat.db missing")

    qs = load_questions(bench_path)
    dmap = load_table_domain_map(db_path)
    subset = build_stratified_subset(qs, dmap, n=80, seed=42)

    assert len(subset) == 80
    assert len({q.question_id for q in subset}) == 80

    total_edu = sum(
        1 for q in qs if any(dmap.get(t) == "giao_duc" for t in q.required_table_ids)
    )
    subset_edu = sum(
        1 for q in subset if any(dmap.get(t) == "giao_duc" for t in q.required_table_ids)
    )
    assert subset_edu == total_edu
    assert subset_edu >= 6
    assert {q.category for q in subset} == {"growth", "share", "ranking", "correlation"}
    assert {q.difficulty for q in subset} == {"easy", "medium", "hard"}


def test_checkpoint_resume_pool_mcts_offline_replay_and_c4_export(tmp_path: Path):
    questions = [
        make_question(question_id=f"Q_{i}", ground_truth=10.0 + i)
        for i in range(5)
    ]
    dmap = {"NG2025_T017": "dan_so"}
    valid_tables = frozenset({"NG2025_T017"})

    def react_fn(q, seed):
        gt = float(q.ground_truth)
        s_val = 0 if seed is None else int(seed)
        ans = gt if (s_val % 4 != 0) else gt + 99.0
        return make_trajectory(
            question_id=q.question_id,
            answer=ans,
            py_values=(ans,),
            cited_tables=("NG2025_T017",),
            queried_tables=("NG2025_T017",),
            prompt_tokens=80,
            completion_tokens=40,
            seed=seed,
        )

    def propose_fn(q, history, k, temperature, seed):
        if not history:
            return [
                ToolAction("q", "query_table", {"sql": "SELECT 1"}, "q", 30, 10, 0.01)
            ]
        return [
            ToolAction(
                "done",
                "final_answer",
                {
                    "answer": q.ground_truth,
                    "unit": "%",
                    "cited_tables": ["NG2025_T017"],
                    "computation_code": "gt",
                    "parse_ok": True,
                },
                "done",
                20,
                10,
                0.01,
            )
        ]

    def execute_fn(action, history):
        return StepObservation(
            output=[{"gia_tri": 10.0}],
            ok=True,
            queried_tables=("NG2025_T017",),
            computed_values=(10.0, 11.0, 12.0, 13.0, 14.0),
            latency_s=0.01,
        )

    mock_llm = MockLLMClient(
        LLMResponse(
            text='{"table_and_geo": 1.0, "formula": 1.0, "unit_and_type": 1.0, "verdict": "yes", "reason": "ok"}',
            prompt_tokens=40,
            completion_tokens=15,
            latency_s=0.02,
        )
    )
    hv = HybridVerifier(mock_llm, valid_tables)
    env = MockStepEnvironment(
        react_trajectories=react_fn,
        propose_fn=propose_fn,
        execute_fn=execute_fn,
    )

    # Simulate partial pool checkpoint of 2 questions, then resume for all 5
    pool_jsonl = tmp_path / "pool.jsonl"
    n1 = collect_trajectory_pool(env, hv, questions[:2], pool_jsonl, pool_size=6, seed=10)
    assert n1 == 2
    n2 = collect_trajectory_pool(env, hv, questions, pool_jsonl, pool_size=6, seed=10)
    assert n2 == 3

    pool_rows = read_jsonl(pool_jsonl)
    assert len(pool_rows) == 5
    assert len({r["question_id"] for r in pool_rows}) == 5

    # Offline replay across 3 seeds
    eval_res = evaluate_from_pool(
        pool_jsonl,
        questions,
        dmap,
        seeds=(42, 43, 44),
        n_values=(3, 5),
        tau=0.6,
    )
    assert "react_n1" in eval_res["methods"]
    assert "sc_n5" in eval_res["methods"]
    assert "sc_early_n5" in eval_res["methods"]
    assert "bon_argmax_n5" in eval_res["methods"]
    assert eval_res["methods"]["sc_early_n5"]["avg_calls_saved"]["mean"] >= 0.0
    assert "tpr" in eval_res["verifier_diagnostics"]

    # Simulate partial MCTS checkpoint of 2 questions, then resume for all 5
    mcts_jsonl = tmp_path / "mcts.jsonl"
    m1 = collect_mcts_runs(
        env, hv, questions[:2], dmap, mcts_jsonl, max_iterations=4, seed=42
    )
    assert m1 == 2
    m2 = collect_mcts_runs(
        env, hv, questions, dmap, mcts_jsonl, max_iterations=4, seed=42
    )
    assert m2 == 3
    assert len(read_jsonl(mcts_jsonl)) == 5

    # Export C.4 traces and verify compatibility with prepare_c4_traces / sample_traces_for_c4
    traces_jsonl = tmp_path / "eval_traces.jsonl"
    exported = export_c4_traces(eval_res["representative_records"], traces_jsonl)
    assert exported == len(eval_res["representative_records"])

    c4_csv = tmp_path / "c4_sample.csv"
    sampled = sample_traces_for_c4(traces_jsonl, c4_csv, n=10, seed=42)
    assert sampled == 10
    assert c4_csv.exists()


def test_plot_results_generates_png_and_pdf_figures(tmp_path: Path):
    plot_mod = _load_script_module("scripts/plot_results.py", "plot_results")
    theory_mod = _load_script_module("scripts/verify_theory_c.py", "verify_theory_c")

    theory_path = tmp_path / "theory.json"
    import json

    theory_path.write_text(
        json.dumps(theory_mod.generate_theory_report()), encoding="utf-8"
    )

    summary_payload = {
        "verifier_diagnostics": {"tpr": 0.9, "fpr": 0.1},
        "methods": {
            "single_prompt": {
                "accuracy": 0.4,
                "avg_total_tokens": 120.0,
                "by_difficulty": {"easy": {"accuracy": 0.6}, "medium": {"accuracy": 0.4}, "hard": {"accuracy": 0.2}},
                "by_category": {"growth": {"accuracy": 0.5}, "share": {"accuracy": 0.4}, "ranking": {"accuracy": 0.4}, "correlation": {"accuracy": 0.3}},
            },
            "react_n1": {
                "accuracy": {"mean": 0.6, "std": 0.03},
                "avg_total_tokens": {"mean": 220.0, "std": 10.0},
                "by_difficulty": {"easy": {"accuracy": {"mean": 0.8, "std": 0.02}}, "medium": {"accuracy": {"mean": 0.6, "std": 0.03}}, "hard": {"accuracy": {"mean": 0.4, "std": 0.04}}},
                "by_category": {"growth": {"accuracy": {"mean": 0.65, "std": 0.02}}, "share": {"accuracy": {"mean": 0.6, "std": 0.03}}, "ranking": {"accuracy": {"mean": 0.6, "std": 0.03}}, "correlation": {"accuracy": {"mean": 0.55, "std": 0.04}}},
            },
            "sc_n3": {"accuracy": {"mean": 0.68, "std": 0.02}, "avg_total_tokens": {"mean": 660.0, "std": 15.0}},
            "sc_n5": {
                "accuracy": {"mean": 0.74, "std": 0.02},
                "avg_total_tokens": {"mean": 1100.0, "std": 20.0},
                "by_difficulty": {"easy": {"accuracy": {"mean": 0.9, "std": 0.02}}, "medium": {"accuracy": {"mean": 0.75, "std": 0.02}}, "hard": {"accuracy": {"mean": 0.55, "std": 0.03}}},
                "by_category": {"growth": {"accuracy": {"mean": 0.8, "std": 0.02}}, "share": {"accuracy": {"mean": 0.75, "std": 0.02}}, "ranking": {"accuracy": {"mean": 0.72, "std": 0.03}}, "correlation": {"accuracy": {"mean": 0.69, "std": 0.03}}},
            },
            "sc_early_n5": {"accuracy": {"mean": 0.74, "std": 0.02}, "avg_total_tokens": {"mean": 820.0, "std": 25.0}, "avg_calls_saved": {"mean": 1.3, "std": 0.1}},
            "bon_argmax_n5": {
                "accuracy": {"mean": 0.78, "std": 0.02},
                "avg_total_tokens": {"mean": 1350.0, "std": 25.0},
                "by_difficulty": {"easy": {"accuracy": {"mean": 0.92, "std": 0.02}}, "medium": {"accuracy": {"mean": 0.8, "std": 0.02}}, "hard": {"accuracy": {"mean": 0.62, "std": 0.03}}},
                "by_category": {"growth": {"accuracy": {"mean": 0.82, "std": 0.02}}, "share": {"accuracy": {"mean": 0.8, "std": 0.02}}, "ranking": {"accuracy": {"mean": 0.78, "std": 0.02}}, "correlation": {"accuracy": {"mean": 0.72, "std": 0.03}}},
            },
            "bon_tau_fallback_n5": {"accuracy": {"mean": 0.75, "std": 0.02}, "avg_total_tokens": {"mean": 1350.0, "std": 25.0}},
            "mcts_high": {
                "accuracy": 0.82,
                "accuracy_ci": {"ci_low": 0.78, "ci_high": 0.86},
                "avg_total_tokens": 1300.0,
                "by_difficulty": {"easy": {"accuracy": 0.94}, "medium": {"accuracy": 0.84}, "hard": {"accuracy": 0.68}},
                "by_category": {"growth": {"accuracy": 0.85}, "share": {"accuracy": 0.83}, "ranking": {"accuracy": 0.82}, "correlation": {"accuracy": 0.78}},
            },
        },
    }
    summary_path = tmp_path / "eval_summary.json"
    summary_path.write_text(json.dumps(summary_payload), encoding="utf-8")

    figs = plot_mod.generate_all_figures(
        summary_path, tmp_path / "figures", theory_path=theory_path
    )
    assert len(figs) == 6  # 3 figures x (.png + .pdf)
    for fp in figs:
        assert fp.exists() and fp.stat().st_size > 0


def test_verify_theory_c_numerical_and_monte_carlo():
    theory_mod = _load_script_module("scripts/verify_theory_c.py", "verify_theory_c")

    c1 = theory_mod.verify_c1(p=0.6, n_values=(1, 5, 15))
    assert c1["rows"][0]["exact_prob"] == pytest.approx(0.6)
    assert c1["rows"][1]["exact_prob"] == pytest.approx(0.68256, abs=1e-6)
    for row in c1["rows"]:
        assert row["exact_prob"] >= row["hoeffding_lower_bound"]

    c2 = theory_mod.verify_c2(p=0.3, tpr=0.9, fpr=0.1, n_values=(1, 4, 16), mc_trials=80_000, seed=42)
    assert c2["q_acc"] == pytest.approx(0.34)
    assert c2["limit_n_inf"] == pytest.approx(0.27 / 0.34, abs=1e-9)
    assert c2["rows"][0]["random_fallback_prob"] == pytest.approx(0.3, abs=1e-9)
    for row in c2["rows"]:
        assert abs(row["abstain_prob"] - row["mc_abstain_prob"]) < 0.01
        assert abs(row["random_fallback_prob"] - row["mc_random_fallback_prob"]) < 0.01

    b5 = theory_mod.verify_b5_early_stop(p=0.6, n_values=(3, 5, 8, 15))
    assert b5["rows"][0]["expected_calls"] == pytest.approx(2.48, abs=1e-9)
    for row in b5["rows"]:
        assert row["expected_calls"] < row["n"]
        assert row["expected_calls_saved"] > 0.0

    c3 = theory_mod.trace_c3_mcts_two_rounds(c=math.sqrt(2.0))
    assert len(c3["rounds"]) == 2
    assert c3["rounds"][0]["selection_path"][0] == "root"
    assert c3["rounds"][1]["backprop_updates"][-1]["visits"] == 6
