"""Tier 1 deterministic verification rules and MCTS intermediate heuristic."""

from __future__ import annotations

import json
from collections.abc import Sequence, Set as AbstractSet
from collections.abc import Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from veristat.agents.types import ReActStep, Trajectory
from veristat.data.benchmark_gen import check_answer_match
from veristat.data.schema import AnswerType, open_readonly


@dataclass(frozen=True)
class Tier1Result:
    """Outcome of deterministic Tier-1 rule checks on a completed trajectory."""

    passed: bool
    score: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "score": float(self.score),
            "reasons": list(self.reasons),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Tier1Result:
        return cls(
            passed=bool(data["passed"]),
            score=float(data["score"]),
            reasons=tuple(str(r) for r in (data.get("reasons") or ())),
        )


def load_valid_table_ids(db_path: Path | str) -> frozenset[str]:
    """Load the set of valid `table_id`s from `veristat.db` in read-only mode."""
    conn = open_readonly(db_path)
    try:
        rows = conn.execute("SELECT table_id FROM tables_catalog").fetchall()
        return frozenset(str(row[0]) for row in rows)
    finally:
        conn.close()


def _is_non_empty_output(output: Any) -> bool:
    if output is None:
        return False
    if isinstance(output, str):
        return bool(output.strip())
    if isinstance(output, Sized):
        return len(output) > 0
    return True


def _collect_candidate_values(trajectory: Trajectory) -> tuple[list[Any], list[Any]]:
    py_vals: list[Any] = []
    sql_vals: list[Any] = []
    for step in trajectory.steps:
        obs = step.observation
        if obs is None or not obs.ok:
            continue
        if step.action.tool_name == "run_python":
            py_vals.extend(obs.computed_values)
        elif step.action.tool_name == "query_table":
            sql_vals.extend(obs.computed_values)
    return py_vals, sql_vals


def verify_tier1(
    trajectory: Trajectory,
    answer_type: AnswerType,
    valid_table_ids: AbstractSet[str],
    *,
    rel_tol: float = 1e-3,
) -> Tier1Result:
    """Check 4 hard deterministic rules before invoking LLM Tier 2."""
    reasons: list[str] = []

    # 1. Zero safety violations
    if trajectory.has_safety_violation:
        reasons.append("safety_violation")

    # 2. At least one successful query_table or run_python call
    has_ok_tool = any(
        step.action.tool_name in ("query_table", "run_python")
        and step.observation is not None
        and step.observation.ok
        for step in trajectory.steps
    )
    if not has_ok_tool:
        reasons.append("no_successful_tool_call")

    # 3. Grounded cited_tables & valid final answer structure
    final = trajectory.final
    if final is None or not final.parse_ok or final.answer is None:
        reasons.append("invalid_final_answer")
        return Tier1Result(passed=False, score=0.0, reasons=tuple(reasons))

    if len(final.cited_tables) == 0:
        reasons.append("missing_cited_tables")
    else:
        queried_set = set(trajectory.all_queried_tables)
        seen_cited: set[str] = set()
        for tid in final.cited_tables:
            if tid in seen_cited:
                continue
            seen_cited.add(tid)
            if tid not in valid_table_ids:
                reasons.append(f"unknown_cited_table:{tid}")
            if tid not in queried_set:
                reasons.append(f"unqueried_cited_table:{tid}")

    # 4. Output consistency against observed tool outputs (prefer run_python)
    py_vals, sql_vals = _collect_candidate_values(trajectory)
    candidate_pool = py_vals if py_vals else sql_vals
    grounded = any(
        check_answer_match(final.answer, val, answer_type, rel_tol=rel_tol)
        for val in candidate_pool
    )
    if not grounded:
        reasons.append("ungrounded_final_answer")

    passed = len(reasons) == 0
    return Tier1Result(
        passed=passed,
        score=1.0 if passed else 0.0,
        reasons=tuple(reasons),
    )


def score_intermediate_state(
    steps: Sequence[ReActStep],
    valid_table_ids: AbstractSet[str] | None = None,
) -> float:
    """Deterministic Tier-1 heuristic in [0, 1] for non-terminal MCTS nodes."""
    if not steps:
        return 0.5

    for step in steps:
        if step.observation is not None and step.observation.safety_violation:
            return 0.0

    score = 0.40
    saw_search_docs = False
    saw_run_python = False
    query_table_bonus = 0.0
    valid_tables_seen: set[str] = set()
    seen_actions: set[tuple[str, str]] = set()

    for step in steps:
        action = step.action
        obs = step.observation

        canonical_args = json.dumps(
            action.tool_args, sort_keys=True, ensure_ascii=False, default=str
        )
        action_sig = (action.tool_name, canonical_args)
        if action_sig in seen_actions:
            score -= 0.20
        else:
            seen_actions.add(action_sig)

        if obs is None:
            continue

        if not obs.ok:
            score -= 0.15
            continue

        if action.tool_name == "search_docs":
            if not saw_search_docs:
                score += 0.10
                saw_search_docs = True
        elif action.tool_name == "query_table":
            if not _is_non_empty_output(obs.output):
                score -= 0.10
                continue
            valid_here = [
                tid
                for tid in obs.queried_tables
                if valid_table_ids is None or tid in valid_table_ids
            ]
            for tid in valid_here:
                if tid in valid_tables_seen:
                    continue
                valid_tables_seen.add(tid)
                delta = 0.20 if len(valid_tables_seen) == 1 else 0.05
                allowed = max(0.0, 0.30 - query_table_bonus)
                gain = min(delta, allowed)
                query_table_bonus += gain
                score += gain
        elif action.tool_name == "run_python":
            has_data = len(obs.computed_values) > 0 or _is_non_empty_output(obs.output)
            if has_data and not saw_run_python:
                score += 0.25
                saw_run_python = True

    return float(np.clip(score, 0.0, 1.0))
