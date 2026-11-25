"""Checkpointed execution runner, offline pool replay, and C.4 trace export."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from veristat.agents.best_of_n import select_best_of_n
from veristat.agents.mcts import run_mcts
from veristat.agents.sc import cluster_and_vote
from veristat.agents.types import StepEnvironment, Trajectory
from veristat.data.schema import BenchmarkQuestion, open_readonly
from veristat.eval.metrics import (
    EvalRecord,
    aggregate_seed_summaries,
    build_eval_record,
    evaluate_prediction,
    summarize_records,
)
from veristat.verifier.hybrid import HybridVerifier, VerificationResult


def load_questions(jsonl_path: Path | str) -> list[BenchmarkQuestion]:
    """Load `dev_30.jsonl` or `benchmark_400.jsonl` into `BenchmarkQuestion` objects."""
    path = Path(jsonl_path)
    items: list[BenchmarkQuestion] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(BenchmarkQuestion.from_dict(json.loads(line)))
    return items


def load_table_domain_map(db_path: Path | str) -> dict[str, str]:
    """Load `{table_id: domain}` mapping from `veristat.db` in read-only mode."""
    conn = open_readonly(db_path)
    try:
        rows = conn.execute("SELECT table_id, domain FROM tables_catalog").fetchall()
        return {str(r[0]): str(r[1]) for r in rows}
    finally:
        conn.close()


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """Read all non-empty JSON lines from `path` (returns [] if missing)."""
    p = Path(path)
    if not p.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def load_checkpoint_ids(path: Path | str, key: str = "question_id") -> set[str]:
    """Return the set of IDs already persisted in a checkpoint JSONL file."""
    return {
        str(rec[key])
        for rec in read_jsonl(path)
        if key in rec
    }


def append_jsonl_record(path: Path | str, payload: dict[str, Any]) -> None:
    """Append one JSON record immediately to disk for crash-safe checkpointing."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        f.flush()


def collect_trajectory_pool(
    env: StepEnvironment,
    verifier: HybridVerifier,
    questions: Sequence[BenchmarkQuestion],
    out_jsonl: Path | str,
    *,
    pool_size: int = 12,
    temperature: float = 0.7,
    seed: int = 42,
    max_steps: int = 8,
    max_seconds: float = 60.0,
) -> int:
    """Sample `pool_size` ReAct trajectories per question and verify, with resume."""
    done_ids = load_checkpoint_ids(out_jsonl)
    processed = 0
    for q_idx, q in enumerate(questions):
        if q.question_id in done_ids:
            continue
        trajs: list[dict[str, Any]] = []
        vers: list[dict[str, Any]] = []
        for k in range(pool_size):
            item_seed = seed + q_idx * 100 + k
            traj = env.run_react(
                q,
                temperature=temperature,
                seed=item_seed,
                max_steps=max_steps,
                max_seconds=max_seconds,
            )
            ver = verifier.verify(q, traj, seed=item_seed)
            trajs.append(traj.to_dict())
            vers.append(ver.to_dict())

        append_jsonl_record(
            out_jsonl,
            {
                "question_id": q.question_id,
                "trajectories": trajs,
                "verifications": vers,
            },
        )
        done_ids.add(q.question_id)
        processed += 1
    return processed


def collect_mcts_runs(
    env: StepEnvironment,
    verifier: HybridVerifier,
    questions: Sequence[BenchmarkQuestion],
    table_domain_map: Mapping[str, str],
    out_jsonl: Path | str,
    *,
    method_name: str = "mcts",
    max_iterations: int = 12,
    max_depth: int = 6,
    branching_factor: int = 2,
    token_budget: int | None = None,
    seed: int = 42,
    rel_tol: float = 1e-3,
) -> int:
    """Run MCTS-UCT across `questions` and checkpoint `EvalRecord`s per question."""
    done_ids = load_checkpoint_ids(out_jsonl)
    processed = 0
    for q_idx, q in enumerate(questions):
        if q.question_id in done_ids:
            continue
        res = run_mcts(
            env,
            verifier,
            q,
            max_iterations=max_iterations,
            max_depth=max_depth,
            branching_factor=branching_factor,
            token_budget=token_budget,
            seed=seed + q_idx,
        )
        rec = build_eval_record(
            q,
            method_name,
            res.best_trajectory,
            res.answer,
            table_domain_map,
            prompt_tokens=res.generation_prompt_tokens,
            completion_tokens=res.generation_completion_tokens,
            verifier_tokens=res.verifier_prompt_tokens + res.verifier_completion_tokens,
            latency_s=res.total_latency_s,
            calls_used=res.iterations_run,
            calls_saved=0,
            rel_tol=rel_tol,
        )
        append_jsonl_record(out_jsonl, rec.to_dict())
        done_ids.add(q.question_id)
        processed += 1
    return processed


def load_pool_file(
    pool_path: Path | str,
) -> tuple[dict[str, list[Trajectory]], dict[str, list[VerificationResult]]]:
    """Load pre-sampled trajectory + verification pools keyed by `question_id`."""
    trajs_by_q: dict[str, list[Trajectory]] = {}
    vers_by_q: dict[str, list[VerificationResult]] = {}
    for row in read_jsonl(pool_path):
        qid = str(row["question_id"])
        trajs_by_q[qid] = [Trajectory.from_dict(t) for t in row["trajectories"]]
        vers_by_q[qid] = [VerificationResult.from_dict(v) for v in row["verifications"]]
    return trajs_by_q, vers_by_q


def _compute_verifier_diagnostics(
    questions: Sequence[BenchmarkQuestion],
    trajs_by_q: Mapping[str, Sequence[Trajectory]],
    vers_by_q: Mapping[str, Sequence[VerificationResult]],
    *,
    tau: float,
    rel_tol: float,
) -> dict[str, float | int]:
    tp = fp = fn = tn = 0
    total_trajs = 0
    for q in questions:
        trajs = trajs_by_q.get(q.question_id, ())
        vers = vers_by_q.get(q.question_id, ())
        for traj, ver in zip(trajs, vers):
            total_trajs += 1
            ans = (
                traj.final.answer
                if (
                    not traj.has_safety_violation
                    and traj.final is not None
                    and traj.final.parse_ok
                )
                else None
            )
            is_corr = evaluate_prediction(q, ans, rel_tol=rel_tol)
            accepted = ver.score >= tau
            if is_corr and accepted:
                tp += 1
            elif is_corr and not accepted:
                fn += 1
            elif not is_corr and accepted:
                fp += 1
            else:
                tn += 1
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "tau": float(tau),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total_trajectories": total_trajs,
    }


def evaluate_from_pool(
    pool_path: Path | str,
    questions: Sequence[BenchmarkQuestion],
    table_domain_map: Mapping[str, str],
    *,
    seeds: Sequence[int] = (42, 43, 44, 45, 46),
    n_values: Sequence[int] = (3, 5, 8),
    tau: float = 0.6,
    rel_tol: float = 1e-3,
) -> dict[str, Any]:
    """Offline multi-seed subsampling evaluation over a pre-sampled trajectory pool."""
    trajs_by_q, vers_by_q = load_pool_file(pool_path)
    missing = [q.question_id for q in questions if q.question_id not in trajs_by_q]
    if missing:
        raise ValueError(
            f"Pool file missing {len(missing)} question(s), e.g. {missing[:5]}"
        )

    method_seed_summaries: dict[str, list[dict[str, Any]]] = {}
    representative_records: list[EvalRecord] = []
    max_n = max(n_values) if n_values else 1

    for seed_idx, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        records_by_method: dict[str, list[EvalRecord]] = {"react_n1": []}
        for k in n_values:
            records_by_method[f"sc_n{k}"] = []
            records_by_method[f"sc_early_n{k}"] = []
            records_by_method[f"bon_argmax_n{k}"] = []
            records_by_method[f"bon_tau_fallback_n{k}"] = []
            records_by_method[f"bon_tau_abstain_n{k}"] = []

        for q_idx, q in enumerate(questions):
            pool_trajs = trajs_by_q[q.question_id]
            pool_vers = vers_by_q[q.question_id]
            perm = rng.permutation(len(pool_trajs))

            # ReAct (n=1)
            idx0 = int(perm[0])
            t0 = pool_trajs[idx0]
            ans0 = (
                t0.final.answer
                if (not t0.has_safety_violation and t0.final is not None and t0.final.parse_ok)
                else None
            )
            records_by_method["react_n1"].append(
                build_eval_record(
                    q,
                    "react_n1",
                    t0,
                    ans0,
                    table_domain_map,
                    prompt_tokens=t0.total_prompt_tokens,
                    completion_tokens=t0.total_completion_tokens,
                    verifier_tokens=0,
                    latency_s=t0.total_latency_s,
                    rel_tol=rel_tol,
                )
            )

            for k in n_values:
                sub_indices = [int(i) for i in perm[: min(k, len(pool_trajs))]]
                sub_trajs = [pool_trajs[i] for i in sub_indices]
                sub_vers = [pool_vers[i] for i in sub_indices]

                # Self-Consistency (full)
                sc_full = cluster_and_vote(
                    sub_trajs, q.answer_type, n_budget=k, early_stop=False, rel_tol=rel_tol
                )
                records_by_method[f"sc_n{k}"].append(
                    build_eval_record(
                        q,
                        f"sc_n{k}",
                        sc_full.selected_trajectory,
                        sc_full.answer,
                        table_domain_map,
                        prompt_tokens=sc_full.total_prompt_tokens,
                        completion_tokens=sc_full.total_completion_tokens,
                        verifier_tokens=0,
                        latency_s=sc_full.total_latency_s,
                        calls_used=sc_full.calls_used,
                        calls_saved=sc_full.calls_saved,
                        rel_tol=rel_tol,
                    )
                )

                # Self-Consistency (early stop B.5)
                sc_early = cluster_and_vote(
                    sub_trajs, q.answer_type, n_budget=k, early_stop=True, rel_tol=rel_tol
                )
                records_by_method[f"sc_early_n{k}"].append(
                    build_eval_record(
                        q,
                        f"sc_early_n{k}",
                        sc_early.selected_trajectory,
                        sc_early.answer,
                        table_domain_map,
                        prompt_tokens=sc_early.total_prompt_tokens,
                        completion_tokens=sc_early.total_completion_tokens,
                        verifier_tokens=0,
                        latency_s=sc_early.total_latency_s,
                        calls_used=sc_early.calls_used,
                        calls_saved=sc_early.calls_saved,
                        rel_tol=rel_tol,
                    )
                )

                # Best-of-n (argmax & tau-threshold modes)
                bon = select_best_of_n(
                    sub_trajs, sub_vers, tau=tau, seed=seed + q_idx
                )
                ver_toks = bon.verifier_prompt_tokens + bon.verifier_completion_tokens
                records_by_method[f"bon_argmax_n{k}"].append(
                    build_eval_record(
                        q,
                        f"bon_argmax_n{k}",
                        bon.selected_trajectory,
                        bon.answer,
                        table_domain_map,
                        prompt_tokens=bon.generation_prompt_tokens,
                        completion_tokens=bon.generation_completion_tokens,
                        verifier_tokens=ver_toks,
                        latency_s=bon.total_latency_s,
                        calls_used=bon.n_candidates,
                        rel_tol=rel_tol,
                    )
                )

                assert bon.threshold_result is not None
                fb_idx = bon.threshold_result.fallback_selected_index
                fb_traj = sub_trajs[fb_idx] if fb_idx is not None else None
                records_by_method[f"bon_tau_fallback_n{k}"].append(
                    build_eval_record(
                        q,
                        f"bon_tau_fallback_n{k}",
                        fb_traj,
                        bon.threshold_result.fallback_answer,
                        table_domain_map,
                        prompt_tokens=bon.generation_prompt_tokens,
                        completion_tokens=bon.generation_completion_tokens,
                        verifier_tokens=ver_toks,
                        latency_s=bon.total_latency_s,
                        calls_used=bon.n_candidates,
                        rel_tol=rel_tol,
                    )
                )

                ab_idx = bon.threshold_result.abstain_selected_index
                ab_traj = sub_trajs[ab_idx] if ab_idx is not None else None
                records_by_method[f"bon_tau_abstain_n{k}"].append(
                    build_eval_record(
                        q,
                        f"bon_tau_abstain_n{k}",
                        ab_traj,
                        bon.threshold_result.abstain_answer,
                        table_domain_map,
                        prompt_tokens=bon.generation_prompt_tokens,
                        completion_tokens=bon.generation_completion_tokens,
                        verifier_tokens=ver_toks,
                        latency_s=bon.total_latency_s,
                        calls_used=bon.n_candidates,
                        safety_violation=False if ab_traj is None else None,
                        rel_tol=rel_tol,
                    )
                )

        if seed_idx == 0:
            representative_records.extend(records_by_method["react_n1"])
            if n_values:
                representative_records.extend(records_by_method[f"sc_n{max_n}"])
                representative_records.extend(records_by_method[f"bon_argmax_n{max_n}"])

        for m_name, recs in records_by_method.items():
            method_seed_summaries.setdefault(m_name, []).append(summarize_records(recs))

    aggregated_methods = {
        m_name: aggregate_seed_summaries(sums)
        for m_name, sums in method_seed_summaries.items()
    }
    diagnostics = _compute_verifier_diagnostics(
        questions, trajs_by_q, vers_by_q, tau=tau, rel_tol=rel_tol
    )
    return {
        "methods": aggregated_methods,
        "per_seed": method_seed_summaries,
        "verifier_diagnostics": diagnostics,
        "representative_records": representative_records,
    }


def export_c4_traces(
    records: Sequence[EvalRecord],
    out_path: Path | str,
) -> int:
    """Write `EvalRecord`s in the exact schema expected by `prepare_c4_traces.py`."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_c4_trace_dict(), ensure_ascii=False) + "\n")
    return len(records)
