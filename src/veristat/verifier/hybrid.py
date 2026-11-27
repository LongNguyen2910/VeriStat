"""Hybrid Verifier combining Tier-1 deterministic rules and Tier-2 LLM rubric."""

from __future__ import annotations

from collections.abc import Sequence, Set as AbstractSet
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from veristat.agents.types import LLMClient, ReActStep, Trajectory
from veristat.data.schema import BenchmarkQuestion
from veristat.verifier.tier1_rules import (
    load_valid_table_ids,
    score_intermediate_state,
    verify_tier1,
)
from veristat.verifier.tier2_llm import Tier2Result, verify_tier2


@dataclass(frozen=True)
class VerificationResult:
    """Combined Hybrid Verifier output with full token & latency accounting."""

    score: float
    tier1_passed: bool
    tier1_reasons: tuple[str, ...]
    tier2: Tier2Result | None
    verifier_prompt_tokens: int
    verifier_completion_tokens: int
    latency_s: float

    @property
    def verifier_total_tokens(self) -> int:
        return self.verifier_prompt_tokens + self.verifier_completion_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": float(self.score),
            "tier1_passed": bool(self.tier1_passed),
            "tier1_reasons": list(self.tier1_reasons),
            "tier2": self.tier2.to_dict() if self.tier2 is not None else None,
            "verifier_prompt_tokens": int(self.verifier_prompt_tokens),
            "verifier_completion_tokens": int(self.verifier_completion_tokens),
            "latency_s": float(self.latency_s),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerificationResult:
        t2_raw = data.get("tier2")
        return cls(
            score=float(data["score"]),
            tier1_passed=bool(data["tier1_passed"]),
            tier1_reasons=tuple(str(r) for r in (data.get("tier1_reasons") or ())),
            tier2=Tier2Result.from_dict(t2_raw) if t2_raw is not None else None,
            verifier_prompt_tokens=int(data.get("verifier_prompt_tokens", 0)),
            verifier_completion_tokens=int(data.get("verifier_completion_tokens", 0)),
            latency_s=float(data.get("latency_s", 0.0)),
        )


class HybridVerifier:
    """Two-tier verifier: Tier 1 short-circuits Tier 2 on technical failures."""

    def __init__(
        self,
        llm: LLMClient,
        valid_table_ids: AbstractSet[str],
        *,
        rel_tol: float = 1e-3,
        use_logprobs: bool = False,
        logprob_weight: float = 0.2,
    ) -> None:
        self.llm = llm
        self.valid_table_ids = frozenset(valid_table_ids)
        self.rel_tol = rel_tol
        self.use_logprobs = use_logprobs
        self.logprob_weight = logprob_weight

    @classmethod
    def from_db(
        cls,
        llm: LLMClient,
        db_path: Path | str,
        *,
        rel_tol: float = 1e-3,
        use_logprobs: bool = False,
        logprob_weight: float = 0.2,
    ) -> HybridVerifier:
        """Instantiate HybridVerifier by loading valid table IDs from `veristat.db`."""
        valid_ids = load_valid_table_ids(db_path)
        return cls(
            llm=llm,
            valid_table_ids=valid_ids,
            rel_tol=rel_tol,
            use_logprobs=use_logprobs,
            logprob_weight=logprob_weight,
        )

    def verify(
        self,
        question: BenchmarkQuestion,
        trajectory: Trajectory,
        *,
        seed: int | None = None,
    ) -> VerificationResult:
        """Run Tier 1 first; only invoke Tier 2 LLM judge when Tier 1 passes."""
        t1 = verify_tier1(
            trajectory,
            question.answer_type,
            self.valid_table_ids,
            rel_tol=self.rel_tol,
        )
        if not t1.passed:
            return VerificationResult(
                score=0.0,
                tier1_passed=False,
                tier1_reasons=t1.reasons,
                tier2=None,
                verifier_prompt_tokens=0,
                verifier_completion_tokens=0,
                latency_s=0.0,
            )

        t2 = verify_tier2(
            self.llm,
            question,
            trajectory,
            seed=seed,
            use_logprobs=self.use_logprobs,
            logprob_weight=self.logprob_weight,
        )
        return VerificationResult(
            score=t2.score,
            tier1_passed=True,
            tier1_reasons=(),
            tier2=t2,
            verifier_prompt_tokens=t2.prompt_tokens,
            verifier_completion_tokens=t2.completion_tokens,
            latency_s=t2.latency_s,
        )

    def score_intermediate(self, steps: Sequence[ReActStep]) -> float:
        """Score a non-terminal MCTS prefix using zero-token Tier-1 heuristics."""
        return score_intermediate_state(steps, valid_table_ids=self.valid_table_ids)
