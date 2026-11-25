"""Evaluation metrics, breakdown summaries, bootstrap CI, and stratification."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from veristat.agents.types import Trajectory
from veristat.data.benchmark_gen import check_answer_match
from veristat.data.schema import AnswerType, BenchmarkQuestion


@dataclass(frozen=True)
class EvalRecord:
    """Per-question evaluation record compatible with metrics and C.4 error taxonomy."""

    question_id: str
    method: str
    category: str
    difficulty: str
    domains: tuple[str, ...]
    uses_admin_mapping: bool
    answer_type: AnswerType
    ground_truth: float | str | list[str]
    predicted_answer: float | str | list[str] | None
    is_correct: bool
    required_table_ids: tuple[str, ...]
    cited_tables: tuple[str, ...]
    citation_exact_match: bool
    citation_recall: float
    tool_calls: int
    prompt_tokens: int
    completion_tokens: int
    verifier_tokens: int
    total_tokens: int
    latency_s: float
    safety_violation: bool
    calls_used: int = 1
    calls_saved: int = 0
    trace: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "method": self.method,
            "category": self.category,
            "difficulty": self.difficulty,
            "domains": list(self.domains),
            "uses_admin_mapping": bool(self.uses_admin_mapping),
            "answer_type": self.answer_type,
            "ground_truth": self.ground_truth,
            "predicted_answer": self.predicted_answer,
            "is_correct": bool(self.is_correct),
            "required_table_ids": list(self.required_table_ids),
            "cited_tables": list(self.cited_tables),
            "citation_exact_match": bool(self.citation_exact_match),
            "citation_recall": float(self.citation_recall),
            "tool_calls": int(self.tool_calls),
            "prompt_tokens": int(self.prompt_tokens),
            "completion_tokens": int(self.completion_tokens),
            "verifier_tokens": int(self.verifier_tokens),
            "total_tokens": int(self.total_tokens),
            "latency_s": float(self.latency_s),
            "safety_violation": bool(self.safety_violation),
            "calls_used": int(self.calls_used),
            "calls_saved": int(self.calls_saved),
            "trace": list(self.trace),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalRecord:
        return cls(
            question_id=str(data["question_id"]),
            method=str(data["method"]),
            category=str(data["category"]),
            difficulty=str(data["difficulty"]),
            domains=tuple(str(d) for d in (data.get("domains") or ())),
            uses_admin_mapping=bool(data.get("uses_admin_mapping", False)),
            answer_type=data["answer_type"],
            ground_truth=data["ground_truth"],
            predicted_answer=data.get("predicted_answer"),
            is_correct=bool(data["is_correct"]),
            required_table_ids=tuple(
                str(t) for t in (data.get("required_table_ids") or ())
            ),
            cited_tables=tuple(str(t) for t in (data.get("cited_tables") or ())),
            citation_exact_match=bool(data.get("citation_exact_match", False)),
            citation_recall=float(data.get("citation_recall", 0.0)),
            tool_calls=int(data.get("tool_calls", 0)),
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            verifier_tokens=int(data.get("verifier_tokens", 0)),
            total_tokens=int(data.get("total_tokens", 0)),
            latency_s=float(data.get("latency_s", 0.0)),
            safety_violation=bool(data.get("safety_violation", False)),
            calls_used=int(data.get("calls_used", 1)),
            calls_saved=int(data.get("calls_saved", 0)),
            trace=tuple(dict(s) for s in (data.get("trace") or ())),
        )

    def to_c4_trace_dict(self) -> dict[str, Any]:
        """Serialize into exact schema required by `sample_traces_for_c4`."""
        return {
            "question_id": self.question_id,
            "method": self.method,
            "category": self.category,
            "difficulty": self.difficulty,
            "is_correct": bool(self.is_correct),
            "predicted_answer": self.predicted_answer,
            "ground_truth": self.ground_truth,
            "trace": list(self.trace),
        }


def evaluate_prediction(
    question: BenchmarkQuestion,
    predicted_answer: float | str | list[str] | None,
    *,
    rel_tol: float = 1e-3,
) -> bool:
    """Compare predicted answer against ground truth using D.4 tolerance."""
    if predicted_answer is None:
        return False
    return check_answer_match(
        predicted_answer,
        question.ground_truth,
        question.answer_type,
        rel_tol=rel_tol,
    )


def compute_citation_metrics(
    cited_tables: Sequence[str],
    required_table_ids: Sequence[str],
) -> tuple[bool, float]:
    """Compute exact table-set match and citation recall."""
    req_set = set(required_table_ids)
    cited_set = set(cited_tables)
    if not req_set:
        return (len(cited_set) == 0, 1.0)
    exact = cited_set == req_set
    recall = len(cited_set & req_set) / len(req_set)
    return (exact, float(recall))


def build_eval_record(
    question: BenchmarkQuestion,
    method: str,
    trajectory: Trajectory | None,
    predicted_answer: float | str | list[str] | None,
    table_domain_map: Mapping[str, str],
    *,
    prompt_tokens: int,
    completion_tokens: int,
    verifier_tokens: int,
    latency_s: float,
    calls_used: int = 1,
    calls_saved: int = 0,
    safety_violation: bool | None = None,
    rel_tol: float = 1e-3,
) -> EvalRecord:
    """Construct a complete `EvalRecord` from a method output and question spec."""
    domains = tuple(
        dict.fromkeys(
            table_domain_map[tid]
            for tid in question.required_table_ids
            if tid in table_domain_map
        )
    )
    cited = (
        trajectory.final.cited_tables
        if (trajectory is not None and trajectory.final is not None)
        else ()
    )
    cit_exact, cit_recall = compute_citation_metrics(cited, question.required_table_ids)
    tool_calls = trajectory.tool_calls_count if trajectory is not None else 0
    is_unsafe = (
        bool(safety_violation)
        if safety_violation is not None
        else (trajectory.has_safety_violation if trajectory is not None else False)
    )
    is_correct = (
        False
        if is_unsafe
        else evaluate_prediction(question, predicted_answer, rel_tol=rel_tol)
    )
    trace = tuple(s.to_dict() for s in trajectory.steps) if trajectory is not None else ()
    total_tok = int(prompt_tokens) + int(completion_tokens) + int(verifier_tokens)

    return EvalRecord(
        question_id=question.question_id,
        method=method,
        category=question.category,
        difficulty=question.difficulty,
        domains=domains,
        uses_admin_mapping=bool(question.uses_admin_mapping),
        answer_type=question.answer_type,
        ground_truth=question.ground_truth,
        predicted_answer=predicted_answer,
        is_correct=is_correct,
        required_table_ids=tuple(question.required_table_ids),
        cited_tables=tuple(cited),
        citation_exact_match=cit_exact,
        citation_recall=cit_recall,
        tool_calls=int(tool_calls),
        prompt_tokens=int(prompt_tokens),
        completion_tokens=int(completion_tokens),
        verifier_tokens=int(verifier_tokens),
        total_tokens=total_tok,
        latency_s=float(latency_s),
        safety_violation=is_unsafe,
        calls_used=int(calls_used),
        calls_saved=int(calls_saved),
        trace=trace,
    )


def _compute_group_metrics(records: Sequence[EvalRecord]) -> dict[str, float | int]:
    n = len(records)
    if n == 0:
        return {
            "count": 0,
            "accuracy": 0.0,
            "answered_rate": 0.0,
            "citation_exact_rate": 0.0,
            "avg_citation_recall": 0.0,
            "avg_tool_calls": 0.0,
            "avg_prompt_tokens": 0.0,
            "avg_completion_tokens": 0.0,
            "avg_verifier_tokens": 0.0,
            "avg_total_tokens": 0.0,
            "avg_latency_s": 0.0,
            "safety_violation_rate": 0.0,
            "avg_calls_used": 0.0,
            "avg_calls_saved": 0.0,
        }
    return {
        "count": n,
        "accuracy": sum(1 for r in records if r.is_correct) / n,
        "answered_rate": sum(1 for r in records if r.predicted_answer is not None) / n,
        "citation_exact_rate": sum(1 for r in records if r.citation_exact_match) / n,
        "avg_citation_recall": sum(r.citation_recall for r in records) / n,
        "avg_tool_calls": sum(r.tool_calls for r in records) / n,
        "avg_prompt_tokens": sum(r.prompt_tokens for r in records) / n,
        "avg_completion_tokens": sum(r.completion_tokens for r in records) / n,
        "avg_verifier_tokens": sum(r.verifier_tokens for r in records) / n,
        "avg_total_tokens": sum(r.total_tokens for r in records) / n,
        "avg_latency_s": sum(r.latency_s for r in records) / n,
        "safety_violation_rate": sum(1 for r in records if r.safety_violation) / n,
        "avg_calls_used": sum(r.calls_used for r in records) / n,
        "avg_calls_saved": sum(r.calls_saved for r in records) / n,
    }


def summarize_records(records: Sequence[EvalRecord]) -> dict[str, Any]:
    """Compute overall metrics and breakdowns by category, difficulty, domain, and admin."""
    summary: dict[str, Any] = dict(_compute_group_metrics(records))

    by_cat: dict[str, list[EvalRecord]] = defaultdict(list)
    by_diff: dict[str, list[EvalRecord]] = defaultdict(list)
    by_dom: dict[str, list[EvalRecord]] = defaultdict(list)
    by_admin: dict[str, list[EvalRecord]] = defaultdict(list)

    for rec in records:
        by_cat[rec.category].append(rec)
        by_diff[rec.difficulty].append(rec)
        for dom in rec.domains:
            by_dom[dom].append(rec)
        admin_key = "uses_admin_mapping" if rec.uses_admin_mapping else "no_admin_mapping"
        by_admin[admin_key].append(rec)

    summary["by_category"] = {
        k: _compute_group_metrics(v) for k, v in sorted(by_cat.items())
    }
    summary["by_difficulty"] = {
        k: _compute_group_metrics(v) for k, v in sorted(by_diff.items())
    }
    summary["by_domain"] = {
        k: _compute_group_metrics(v) for k, v in sorted(by_dom.items())
    }
    summary["by_admin_mapping"] = {
        k: _compute_group_metrics(v) for k, v in sorted(by_admin.items())
    }
    return summary


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Compute deterministic percentile bootstrap confidence interval."""
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    if n == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "n": 0,
            "n_resamples": n_resamples,
        }
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = arr[idx].mean(axis=1)
    ci_low = float(np.percentile(means, 100.0 * (alpha / 2.0)))
    ci_high = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return {
        "mean": float(arr.mean()),
        "std": float(means.std(ddof=1)) if n_resamples > 1 else 0.0,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": n,
        "n_resamples": n_resamples,
    }


def _aggregate_dict_tree(dicts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not dicts:
        return {}
    out: dict[str, Any] = {}
    keys: list[str] = list(dicts[0].keys())
    for key in keys:
        vals = [d[key] for d in dicts if key in d]
        if isinstance(vals[0], Mapping):
            out[key] = _aggregate_dict_tree(vals)
        elif isinstance(vals[0], (int, float)) and not isinstance(vals[0], bool):
            arr = np.asarray(vals, dtype=float)
            std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            out[key] = {"mean": float(arr.mean()), "std": std}
        else:
            out[key] = vals[0]
    return out


def aggregate_seed_summaries(
    seed_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-seed summaries into mean ± std dictionaries."""
    if not seed_summaries:
        raise ValueError("seed_summaries must not be empty")
    agg = _aggregate_dict_tree(seed_summaries)
    agg["n_seeds"] = len(seed_summaries)
    return agg


def build_stratified_subset(
    questions: Sequence[BenchmarkQuestion],
    table_domain_map: Mapping[str, str],
    *,
    n: int = 80,
    seed: int = 42,
) -> list[BenchmarkQuestion]:
    """Select `n` questions preserving `(category, difficulty)` and all `giao_duc` items."""
    if n <= 0 or not questions:
        return []
    if n >= len(questions):
        return list(questions)

    rng = random.Random(seed)
    selected: list[BenchmarkQuestion] = []
    selected_ids: set[str] = set()
    selected_strata: Counter[tuple[str, str]] = Counter()

    # 1. Include all `giao_duc` questions first (scarce domain in benchmark)
    giao_duc_qs = sorted(
        (
            q
            for q in questions
            if any(
                table_domain_map.get(tid) == "giao_duc"
                for tid in q.required_table_ids
            )
        ),
        key=lambda q: q.question_id,
    )
    for q in giao_duc_qs:
        if len(selected) >= n:
            break
        selected.append(q)
        selected_ids.add(q.question_id)
        selected_strata[(q.category, q.difficulty)] += 1

    if len(selected) >= n:
        return selected

    # 2. Proportional quota per (category, difficulty) stratum
    strata_pools: dict[tuple[str, str], list[BenchmarkQuestion]] = defaultdict(list)
    total_strata: Counter[tuple[str, str]] = Counter()
    for q in questions:
        key = (q.category, q.difficulty)
        total_strata[key] += 1
        if q.question_id not in selected_ids:
            strata_pools[key].append(q)

    for key in strata_pools:
        strata_pools[key].sort(key=lambda x: x.question_id)
        rng.shuffle(strata_pools[key])

    total_len = len(questions)
    exact_quotas = {k: (cnt * n) / total_len for k, cnt in total_strata.items()}
    target_quotas = {k: int(val) for k, val in exact_quotas.items()}
    leftover = n - sum(target_quotas.values())
    by_rem = sorted(
        exact_quotas.keys(),
        key=lambda k: (exact_quotas[k] - target_quotas[k], total_strata[k]),
        reverse=True,
    )
    for k in by_rem[:leftover]:
        target_quotas[k] += 1

    # Fill each stratum up to its target quota
    for key in sorted(target_quotas.keys()):
        needed = max(0, target_quotas[key] - selected_strata[key])
        pool = strata_pools.get(key, [])
        take = min(needed, len(pool))
        for _ in range(take):
            q = pool.pop()
            selected.append(q)
            selected_ids.add(q.question_id)

    # Top up if any stratum had more pre-selected giao_duc items than its target
    if len(selected) < n:
        remaining = [q for q in questions if q.question_id not in selected_ids]
        remaining.sort(key=lambda x: x.question_id)
        rng.shuffle(remaining)
        selected.extend(remaining[: n - len(selected)])

    return selected[:n]
