"""Self-Consistency majority voting with Theorem B.5 sequential early stopping."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from veristat.agents.types import StepEnvironment, Trajectory
from veristat.data.admin_mapping import normalize_entity, normalize_entity_list
from veristat.data.benchmark_gen import check_answer_match
from veristat.data.schema import AnswerType, BenchmarkQuestion


@dataclass(frozen=True)
class ClusterInfo:
    """Summary of one normalized answer cluster in Self-Consistency voting."""

    representative: float | str | list[str]
    votes: int
    first_idx: int
    member_indices: tuple[int, ...]


@dataclass(frozen=True)
class SCResult:
    """Outcome of Self-Consistency voting (full or sequential early stopping)."""

    answer: float | str | list[str] | None
    selected_trajectory: Trajectory | None
    n_budget: int
    calls_used: int
    calls_saved: int
    early_stopped: bool
    majority_threshold: int
    clusters: tuple[ClusterInfo, ...]
    total_prompt_tokens: int
    total_completion_tokens: int
    tokens_saved: int
    total_latency_s: float
    trajectories: tuple[Trajectory, ...]

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens


@dataclass
class _MutableCluster:
    anchor: float | str | tuple[str, ...]
    first_idx: int
    member_indices: list[int] = field(default_factory=list)
    numeric_values: list[float] = field(default_factory=list)


def _extract_valid_value(
    trajectory: Trajectory,
    answer_type: AnswerType,
) -> float | str | tuple[str, ...] | None:
    if trajectory.has_safety_violation:
        return None
    final = trajectory.final
    if final is None or not final.parse_ok or final.answer is None:
        return None

    raw = final.answer
    if answer_type == "number":
        if isinstance(raw, bool):
            return None
        try:
            num = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return num if math.isfinite(num) else None

    if answer_type == "entity":
        if not isinstance(raw, str) or not raw.strip():
            return None
        return normalize_entity(raw)

    if answer_type == "entity_list":
        if not isinstance(raw, list) or not raw:
            return None
        return tuple(normalize_entity_list([str(x) for x in raw]))

    raise ValueError(f"Unsupported answer_type: {answer_type!r}")


def _numbers_equivalent(a: float, b: float, rel_tol: float) -> bool:
    return check_answer_match(a, b, "number", rel_tol=rel_tol) or check_answer_match(
        b, a, "number", rel_tol=rel_tol
    )


class _VoteAccumulator:
    """Shared incremental voting engine for offline replay and live sampling."""

    def __init__(
        self,
        answer_type: AnswerType,
        *,
        n_budget: int,
        early_stop: bool,
        rel_tol: float,
    ) -> None:
        self.answer_type = answer_type
        self.n_budget = max(0, int(n_budget))
        self.early_stop = early_stop
        self.rel_tol = rel_tol
        self.majority_threshold = (self.n_budget // 2) + 1 if self.n_budget > 0 else 1
        self._clusters: list[_MutableCluster] = []
        self._trajectories: list[Trajectory] = []
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_latency_s = 0.0

    def add(self, trajectory: Trajectory) -> bool:
        """Add one trajectory; return True if sequential early-stopping triggers."""
        idx = len(self._trajectories)
        self._trajectories.append(trajectory)
        self._total_prompt_tokens += int(trajectory.total_prompt_tokens)
        self._total_completion_tokens += int(trajectory.total_completion_tokens)
        self._total_latency_s += float(trajectory.total_latency_s)

        val = _extract_valid_value(trajectory, self.answer_type)
        if val is None:
            return False

        matched_cluster: _MutableCluster | None = None
        for cluster in self._clusters:
            if self.answer_type == "number":
                if _numbers_equivalent(
                    float(val), float(cluster.anchor), self.rel_tol  # type: ignore[arg-type]
                ):
                    matched_cluster = cluster
                    break
            elif val == cluster.anchor:
                matched_cluster = cluster
                break

        if matched_cluster is None:
            matched_cluster = _MutableCluster(
                anchor=val,
                first_idx=idx,
                member_indices=[idx],
                numeric_values=[float(val)] if self.answer_type == "number" else [],  # type: ignore[arg-type]
            )
            self._clusters.append(matched_cluster)
        else:
            matched_cluster.member_indices.append(idx)
            if self.answer_type == "number":
                matched_cluster.numeric_values.append(float(val))  # type: ignore[arg-type]

        return self.early_stop and len(matched_cluster.member_indices) >= self.majority_threshold

    def finalize(self, *, early_stopped: bool, tokens_saved: int = 0) -> SCResult:
        cluster_infos: list[ClusterInfo] = []
        for c in self._clusters:
            rep: float | str | list[str]
            if self.answer_type == "number":
                rep = float(np.median(np.asarray(c.numeric_values, dtype=float)))
            elif isinstance(c.anchor, tuple):
                rep = list(c.anchor)
            else:
                rep = str(c.anchor)
            cluster_infos.append(
                ClusterInfo(
                    representative=rep,
                    votes=len(c.member_indices),
                    first_idx=c.first_idx,
                    member_indices=tuple(c.member_indices),
                )
            )

        calls_used = len(self._trajectories)
        calls_saved = (
            max(0, self.n_budget - calls_used) if early_stopped else 0
        )

        if not cluster_infos:
            return SCResult(
                answer=None,
                selected_trajectory=None,
                n_budget=self.n_budget,
                calls_used=calls_used,
                calls_saved=calls_saved,
                early_stopped=early_stopped,
                majority_threshold=self.majority_threshold,
                clusters=(),
                total_prompt_tokens=self._total_prompt_tokens,
                total_completion_tokens=self._total_completion_tokens,
                tokens_saved=int(tokens_saved),
                total_latency_s=self._total_latency_s,
                trajectories=tuple(self._trajectories),
            )

        winner = max(cluster_infos, key=lambda cl: (cl.votes, -cl.first_idx))
        return SCResult(
            answer=winner.representative,
            selected_trajectory=self._trajectories[winner.first_idx],
            n_budget=self.n_budget,
            calls_used=calls_used,
            calls_saved=calls_saved,
            early_stopped=early_stopped,
            majority_threshold=self.majority_threshold,
            clusters=tuple(cluster_infos),
            total_prompt_tokens=self._total_prompt_tokens,
            total_completion_tokens=self._total_completion_tokens,
            tokens_saved=int(tokens_saved),
            total_latency_s=self._total_latency_s,
            trajectories=tuple(self._trajectories),
        )


def cluster_and_vote(
    trajectories: Sequence[Trajectory],
    answer_type: AnswerType,
    *,
    n_budget: int | None = None,
    early_stop: bool = False,
    rel_tol: float = 1e-3,
) -> SCResult:
    """Run Self-Consistency majority voting over a pre-sampled trajectory list."""
    n_eff = len(trajectories) if n_budget is None else max(0, int(n_budget))
    window = list(trajectories[:n_eff])
    acc = _VoteAccumulator(
        answer_type,
        n_budget=n_eff,
        early_stop=early_stop,
        rel_tol=rel_tol,
    )
    for idx, traj in enumerate(window):
        if acc.add(traj):
            saved_tokens = sum(t.total_tokens for t in window[idx + 1 :])
            return acc.finalize(early_stopped=True, tokens_saved=saved_tokens)
    return acc.finalize(early_stopped=False, tokens_saved=0)


def run_self_consistency(
    env: StepEnvironment,
    question: BenchmarkQuestion,
    *,
    n: int = 5,
    temperature: float = 0.7,
    early_stop: bool = False,
    seed: int = 42,
    max_steps: int = 8,
    max_seconds: float = 60.0,
    token_budget: int | None = None,
    rel_tol: float = 1e-3,
) -> SCResult:
    """Sample up to `n` ReAct trajectories sequentially with optional early stopping."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    acc = _VoteAccumulator(
        question.answer_type,
        n_budget=n,
        early_stop=early_stop,
        rel_tol=rel_tol,
    )
    tokens_used = 0
    for idx in range(n):
        if token_budget is not None and tokens_used >= token_budget:
            break
        remaining = max(1, token_budget - tokens_used) if token_budget is not None else None
        traj = env.run_react(
            question,
            temperature=temperature,
            seed=seed + idx,
            max_steps=max_steps,
            max_seconds=max_seconds,
            max_tokens=remaining,
        )
        tokens_used += traj.total_tokens
        if acc.add(traj):
            return acc.finalize(early_stopped=True, tokens_saved=0)

    return acc.finalize(early_stopped=False, tokens_saved=0)
