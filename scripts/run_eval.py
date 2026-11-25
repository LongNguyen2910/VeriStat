"""Run VeriStat evaluation pipeline (CPU mock-demo or Phase 2 offline pool replay).

Usage:
    # CPU dry-run on dev_30 with deterministic mocks:
    python scripts/run_eval.py --mode mock-demo --db data/processed/veristat.db \
        --questions data/benchmark/dev_30.jsonl --out-dir results/mock_demo

    # Phase 2 offline evaluation from Colab JSONL artifacts:
    python scripts/run_eval.py --mode offline-pool --db data/processed/veristat.db \
        --questions data/benchmark/benchmark_400.jsonl --pool results/pool_12_test.jsonl \
        --out-dir results/final
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from veristat.agents.best_of_n import calibrate_verifier_threshold
from veristat.agents.types import (
    FinalAnswer,
    LLMResponse,
    ReActStep,
    StepObservation,
    ToolAction,
    Trajectory,
)
from veristat.data.schema import BenchmarkQuestion
from veristat.eval.metrics import (
    EvalRecord,
    bootstrap_ci,
    build_eval_record,
    summarize_records,
)
from veristat.eval.runner import (
    collect_mcts_runs,
    collect_trajectory_pool,
    evaluate_from_pool,
    export_c4_traces,
    load_pool_file,
    load_questions,
    load_table_domain_map,
    read_jsonl,
)
from veristat.verifier.hybrid import HybridVerifier
from veristat.verifier.tier1_rules import load_valid_table_ids


class _DemoMockLLM:
    """Deterministic local mock LLM used exclusively by `--mode mock-demo`."""

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
        logprobs: bool = False,
    ) -> LLMResponse:
        text = (
            '{"table_and_geo": 1.0, "formula": 1.0, "unit_and_type": 1.0, '
            '"verdict": "yes", "reason": "mock_demo"}'
        )
        return LLMResponse(
            text=text,
            prompt_tokens=60,
            completion_tokens=24,
            latency_s=0.01,
        )


class _DemoMockEnv:
    """Deterministic local mock `StepEnvironment` for `--mode mock-demo`."""

    def _wrong_answer(self, q: BenchmarkQuestion) -> float | str | list[str]:
        if q.answer_type == "number":
            return float(q.ground_truth) + 50.0  # type: ignore[arg-type]
        if q.answer_type == "entity":
            return "Không xác định"
        return ["Không xác định"]

    def propose_actions(
        self,
        question: BenchmarkQuestion,
        history: tuple[ReActStep, ...],
        *,
        k: int = 1,
        temperature: float = 0.7,
        seed: int | None = None,
    ) -> list[ToolAction]:
        tid = question.required_table_ids[0] if question.required_table_ids else "NG2025_T017"
        if not history:
            return [
                ToolAction(
                    thought="Truy vấn bảng nguồn",
                    tool_name="query_table",
                    tool_args={"sql": f"SELECT * FROM stat_{tid.lower()}", "table_id": tid},
                    raw_text="query_table",
                    prompt_tokens=50,
                    completion_tokens=20,
                    latency_s=0.01,
                )
                for _ in range(min(1, k))
            ]
        if len(history) == 1:
            return [
                ToolAction(
                    thought="Tính kết quả bằng Python",
                    tool_name="run_python",
                    tool_args={"code": question.reference_code, "branch": b},
                    raw_text="run_python",
                    prompt_tokens=45,
                    completion_tokens=25,
                    latency_s=0.01,
                )
                for b in range(k)
            ]
        # Depth >= 2: propose final_answer
        gt = question.ground_truth
        wrong = self._wrong_answer(question)
        actions: list[ToolAction] = []
        for b in range(k):
            is_good = b == 0 or ((seed or 0) + b) % 3 != 0
            ans = gt if is_good else wrong
            actions.append(
                ToolAction(
                    thought="Tổng hợp đáp án",
                    tool_name="final_answer",
                    tool_args={
                        "answer": ans,
                        "unit": question.unit,
                        "cited_tables": list(question.required_table_ids),
                        "computation_code": question.reference_code,
                        "parse_ok": True,
                    },
                    raw_text=f"final_answer:{ans}",
                    prompt_tokens=40,
                    completion_tokens=18,
                    latency_s=0.01,
                )
            )
        return actions

    def execute_action(
        self,
        action: ToolAction,
        history: tuple[ReActStep, ...] = (),
    ) -> StepObservation:
        if action.tool_name == "query_table":
            tid = str(action.tool_args.get("table_id", "NG2025_T017"))
            return StepObservation(
                output=[{"table_id": tid, "ok": True}],
                ok=True,
                queried_tables=(tid,),
                computed_values=(),
                latency_s=0.005,
            )
        if action.tool_name == "run_python":
            return StepObservation(
                output="computed",
                ok=True,
                computed_values=(),
                latency_s=0.005,
            )
        return StepObservation(output=None, ok=True, latency_s=0.001)

    def extract_final_answer(self, action: ToolAction) -> FinalAnswer:
        args = action.tool_args
        return FinalAnswer(
            answer=args.get("answer"),
            unit=str(args.get("unit", "")),
            cited_tables=tuple(args.get("cited_tables", ())),
            computation_code=str(args.get("computation_code", "")),
            parse_ok=bool(args.get("parse_ok", True)),
        )

    def run_react(
        self,
        question: BenchmarkQuestion,
        *,
        temperature: float = 0.0,
        seed: int | None = None,
        max_steps: int = 8,
        max_seconds: float = 60.0,
        max_tokens: int | None = None,
    ) -> Trajectory:
        # ~70% accurate when temperature > 0
        s_val = 0 if seed is None else int(seed)
        is_correct = (s_val % 10) < 7
        ans = question.ground_truth if is_correct else self._wrong_answer(question)
        tids = tuple(question.required_table_ids)

        s0 = ReActStep(
            step_idx=0,
            action=ToolAction(
                thought="Truy vấn dữ liệu",
                tool_name="query_table",
                tool_args={"sql": "SELECT ..."},
                raw_text="query_table",
                prompt_tokens=60,
                completion_tokens=20,
                latency_s=0.01,
            ),
            observation=StepObservation(
                output=[{"ok": True}],
                ok=True,
                queried_tables=tids,
                computed_values=(),
                latency_s=0.005,
            ),
        )
        s1 = ReActStep(
            step_idx=1,
            action=ToolAction(
                thought="Tính bằng Python",
                tool_name="run_python",
                tool_args={"code": question.reference_code},
                raw_text="run_python",
                prompt_tokens=50,
                completion_tokens=25,
                latency_s=0.01,
            ),
            observation=StepObservation(
                output=str(ans),
                ok=True,
                computed_values=(ans,),
                latency_s=0.005,
            ),
        )
        s2 = ReActStep(
            step_idx=2,
            action=ToolAction(
                thought="Trả lời",
                tool_name="final_answer",
                tool_args={"answer": ans},
                raw_text="final_answer",
                prompt_tokens=40,
                completion_tokens=15,
                latency_s=0.01,
            ),
            observation=None,
        )
        return Trajectory(
            question_id=question.question_id,
            steps=(s0, s1, s2),
            final=FinalAnswer(
                answer=ans,
                unit=question.unit,
                cited_tables=tids,
                computation_code=question.reference_code,
                parse_ok=True,
            ),
            terminated_reason="final_answer",
            total_prompt_tokens=150,
            total_completion_tokens=60,
            total_latency_s=0.04,
            seed=seed,
        )


def _summarize_single_file_method(
    jsonl_path: Path,
    *,
    seed: int = 42,
) -> tuple[dict[str, Any], list[EvalRecord]]:
    rows = read_jsonl(jsonl_path)
    records = [EvalRecord.from_dict(r) for r in rows]
    summary = summarize_records(records)
    acc_vals = [1.0 if r.is_correct else 0.0 for r in records]
    tok_vals = [float(r.total_tokens) for r in records]
    summary["accuracy_ci"] = bootstrap_ci(acc_vals, n_resamples=1000, seed=seed)
    summary["tokens_ci"] = bootstrap_ci(tok_vals, n_resamples=1000, seed=seed)
    return summary, records


def run_mock_demo(
    db_path: Path,
    questions_path: Path,
    out_dir: Path,
    *,
    pool_size: int = 12,
    n_values: Sequence[int] = (3, 5, 8),
    seed: int = 42,
) -> dict[str, Any]:
    """Run an end-to-end deterministic CPU dry-run on `dev_30.jsonl`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    questions = load_questions(questions_path)
    domain_map = load_table_domain_map(db_path)
    valid_tables = load_valid_table_ids(db_path)

    mock_llm = _DemoMockLLM()
    mock_env = _DemoMockEnv()
    verifier = HybridVerifier(mock_llm, valid_tables)

    # 1. Pool collection (checkpointed)
    pool_path = out_dir / "pool_12_dev.jsonl"
    collect_trajectory_pool(
        mock_env, verifier, questions, pool_path, pool_size=pool_size, seed=seed
    )
    trajs_by_q, vers_by_q = load_pool_file(pool_path)
    cal = calibrate_verifier_threshold(
        trajs_by_q, vers_by_q, questions, seed=seed
    )
    tau_star = float(cal["best_tau"])

    # 2. Offline 5-seed evaluation over pool
    pool_eval = evaluate_from_pool(
        pool_path,
        questions,
        domain_map,
        seeds=tuple(seed + i for i in range(5)),
        n_values=n_values,
        tau=tau_star,
    )

    # 3. Single prompt baseline (mocked)
    single_records: list[EvalRecord] = []
    for q_idx, q in enumerate(questions):
        pred = q.ground_truth if (q_idx % 2 == 0) else None
        single_records.append(
            build_eval_record(
                q,
                "single_prompt",
                None,
                pred,
                domain_map,
                prompt_tokens=90,
                completion_tokens=30,
                verifier_tokens=0,
                latency_s=0.02,
            )
        )
    single_summary = summarize_records(single_records)
    single_summary["accuracy_ci"] = bootstrap_ci(
        [1.0 if r.is_correct else 0.0 for r in single_records], seed=seed
    )

    # 4. MCTS mock run (note mock_env keeps run_python computed_values empty,
    #    so adapt run_python observation inside mock_env for clean MCTS demo)
    class _MCTSDemoEnv(_DemoMockEnv):
        def execute_action(self, action: ToolAction, history: tuple[ReActStep, ...] = ()) -> StepObservation:
            obs = super().execute_action(action, history)
            if action.tool_name == "run_python":
                # Expose gt computed value matching reference_code for Tier 1 grounding
                for q in questions:
                    if q.reference_code == action.tool_args.get("code"):
                        return StepObservation(
                            output=str(q.ground_truth),
                            ok=True,
                            computed_values=(q.ground_truth,),
                            latency_s=0.005,
                        )
            return obs

    mcts_env = _MCTSDemoEnv()
    mcts_methods: dict[str, Any] = {}
    mcts_records_for_c4: list[EvalRecord] = []
    for label, budget in (("mcts_low", 300), ("mcts_mid", 600), ("mcts_high", 1000)):
        mcts_path = out_dir / f"{label}_dev.jsonl"
        collect_mcts_runs(
            mcts_env,
            verifier,
            questions,
            domain_map,
            mcts_path,
            method_name=label,
            max_iterations=8,
            max_depth=4,
            branching_factor=2,
            token_budget=budget,
            seed=seed,
        )
        m_sum, m_recs = _summarize_single_file_method(mcts_path, seed=seed)
        mcts_methods[label] = m_sum
        if label == "mcts_high":
            mcts_records_for_c4.extend(m_recs)

    methods: dict[str, Any] = {"single_prompt": single_summary}
    methods.update(pool_eval["methods"])
    methods.update(mcts_methods)

    all_c4_records = list(pool_eval["representative_records"]) + mcts_records_for_c4
    traces_path = out_dir / "eval_traces.jsonl"
    export_c4_traces(all_c4_records, traces_path)

    summary_payload = {
        "mode": "mock-demo",
        "n_questions": len(questions),
        "dev_calibration": cal,
        "verifier_diagnostics": pool_eval["verifier_diagnostics"],
        "methods": methods,
    }
    summary_path = out_dir / "eval_summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary_payload


def run_offline_pool(
    db_path: Path,
    questions_path: Path,
    pool_path: Path,
    out_dir: Path,
    *,
    dev_questions_path: Path | None = None,
    dev_pool_path: Path | None = None,
    single_path: Path | None = None,
    mcts_specs: Sequence[str] = (),
    tau: float = 0.6,
    n_values: Sequence[int] = (3, 5, 8),
    seed: int = 42,
) -> dict[str, Any]:
    """Run Phase 2 offline evaluation over pre-collected Colab JSONL artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    questions = load_questions(questions_path)
    domain_map = load_table_domain_map(db_path)

    dev_cal: dict[str, Any] | None = None
    effective_tau = tau
    if dev_pool_path is not None and dev_questions_path is not None:
        dev_qs = load_questions(dev_questions_path)
        d_trajs, d_vers = load_pool_file(dev_pool_path)
        dev_cal = calibrate_verifier_threshold(d_trajs, d_vers, dev_qs, seed=seed)
        effective_tau = float(dev_cal["best_tau"])

    pool_eval = evaluate_from_pool(
        pool_path,
        questions,
        domain_map,
        seeds=tuple(seed + i for i in range(5)),
        n_values=n_values,
        tau=effective_tau,
    )

    methods: dict[str, Any] = {}
    all_c4_records = list(pool_eval["representative_records"])

    if single_path is not None and single_path.exists():
        s_sum, s_recs = _summarize_single_file_method(single_path, seed=seed)
        methods["single_prompt"] = s_sum
        all_c4_records.extend(s_recs)

    methods.update(pool_eval["methods"])

    for spec in mcts_specs:
        if "=" not in spec:
            raise ValueError(f"Expected --mcts name=path, got {spec!r}")
        name, raw_p = spec.split("=", 1)
        m_sum, m_recs = _summarize_single_file_method(Path(raw_p), seed=seed)
        methods[name.strip()] = m_sum
        all_c4_records.extend(m_recs)

    export_c4_traces(all_c4_records, out_dir / "eval_traces.jsonl")
    summary_payload = {
        "mode": "offline-pool",
        "n_questions": len(questions),
        "tau_used": effective_tau,
        "dev_calibration": dev_cal,
        "verifier_diagnostics": pool_eval["verifier_diagnostics"],
        "methods": methods,
    }
    (out_dir / "eval_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="VeriStat Evaluation Runner (A.4)")
    parser.add_argument(
        "--mode",
        choices=("mock-demo", "offline-pool"),
        required=True,
    )
    parser.add_argument("--db", type=Path, default=Path("data/processed/veristat.db"))
    parser.add_argument(
        "--questions", type=Path, default=Path("data/benchmark/dev_30.jsonl")
    )
    parser.add_argument("--pool", type=Path, default=None)
    parser.add_argument("--dev-questions", type=Path, default=None)
    parser.add_argument("--dev-pool", type=Path, default=None)
    parser.add_argument("--single", type=Path, default=None)
    parser.add_argument(
        "--mcts",
        action="append",
        default=[],
        help="MCTS checkpoint JSONL in name=path format (repeatable)",
    )
    parser.add_argument("--tau", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("results/eval"))
    args = parser.parse_args()

    if args.mode == "mock-demo":
        summary = run_mock_demo(
            args.db, args.questions, args.out_dir, seed=args.seed
        )
    else:
        if args.pool is None:
            parser.error("--pool is required in offline-pool mode")
        summary = run_offline_pool(
            args.db,
            args.questions,
            args.pool,
            args.out_dir,
            dev_questions_path=args.dev_questions,
            dev_pool_path=args.dev_pool,
            single_path=args.single,
            mcts_specs=args.mcts,
            tau=args.tau,
            seed=args.seed,
        )
    print(
        json.dumps(
            {
                "mode": summary["mode"],
                "n_questions": summary["n_questions"],
                "methods": sorted(summary["methods"].keys()),
                "out_dir": str(args.out_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
