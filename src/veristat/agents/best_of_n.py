"""Best-of-n selection with Hybrid Verifier (argmax and tau-threshold modes)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from veristat.agents.types import StepEnvironment, Trajectory
from veristat.data.benchmark_gen import check_answer_match
from veristat.data.schema import BenchmarkQuestion
from veristat.verifier.hybrid import HybridVerifier, VerificationResult


@dataclass(frozen=True)
class ThresholdSelection:
    """Dual tau-threshold outcome (random fallback & abstain) for Theorem B.3 / C.2."""

    tau: float
    accepted_indices: tuple[int, ...]
    fallback_selected_index: int | None
    fallback_answer: float | str | list[str] | None
    abstain_selected_index: int | None
    abstain_answer: float | str | list[str] | None
    abstained: bool


@dataclass(frozen=True)
class BestOfNResult:
    """Result of Best-of-n selection with combined generation + verifier accounting."""

    selected_index: int | None
    selected_trajectory: Trajectory | None
    answer: float | str | list[str] | None
    best_score: float
    scores: tuple[float, ...]
    threshold_result: ThresholdSelection | None
    n_candidates: int
    generation_prompt_tokens: int
    generation_completion_tokens: int
    verifier_prompt_tokens: int
    verifier_completion_tokens: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    total_latency_s: float
    trajectories: tuple[Trajectory, ...]
    verifications: tuple[VerificationResult, ...]


def _extract_valid_answer(
    trajectory: Trajectory,
) -> float | str | list[str] | None:
    if trajectory.has_safety_violation:
        return None
    final = trajectory.final
    if final is None or not final.parse_ok or final.answer is None:
        return None
    return final.answer


def select_best_of_n(
    trajectories: Sequence[Trajectory],
    verifications: Sequence[VerificationResult],
    *,
    tau: float | None = None,
    seed: int = 42,
) -> BestOfNResult:
    """Select the highest-scoring trajectory and optionally evaluate tau-thresholding."""
    if len(trajectories) != len(verifications):
        raise ValueError(
            f"Length mismatch: {len(trajectories)} trajectories vs {len(verifications)} verifications"
        )

    trajs = tuple(trajectories)
    vers = tuple(verifications)
    scores = tuple(float(v.score) for v in vers)

    gen_prompt = sum(int(t.total_prompt_tokens) for t in trajs)
    gen_comp = sum(int(t.total_completion_tokens) for t in trajs)
    ver_prompt = sum(int(v.verifier_prompt_tokens) for v in vers)
    ver_comp = sum(int(v.verifier_completion_tokens) for v in vers)
    total_prompt = gen_prompt + ver_prompt
    total_comp = gen_comp + ver_comp
    total_tokens = total_prompt + total_comp
    total_latency = sum(float(t.total_latency_s) for t in trajs) + sum(
        float(v.latency_s) for v in vers
    )

    if not trajs:
        thresh = (
            ThresholdSelection(
                tau=float(tau),
                accepted_indices=(),
                fallback_selected_index=None,
                fallback_answer=None,
                abstain_selected_index=None,
                abstain_answer=None,
                abstained=True,
            )
            if tau is not None
            else None
        )
        return BestOfNResult(
            selected_index=None,
            selected_trajectory=None,
            answer=None,
            best_score=0.0,
            scores=(),
            threshold_result=thresh,
            n_candidates=0,
            generation_prompt_tokens=0,
            generation_completion_tokens=0,
            verifier_prompt_tokens=0,
            verifier_completion_tokens=0,
            total_prompt_tokens=0,
            total_completion_tokens=0,
            total_tokens=0,
            total_latency_s=0.0,
            trajectories=(),
            verifications=(),
        )

    best_idx = max(
        range(len(trajs)),
        key=lambda i: (
            scores[i],
            1 if _extract_valid_answer(trajs[i]) is not None else 0,
            -i,
        ),
    )
    best_traj = trajs[best_idx]
    best_ans = _extract_valid_answer(best_traj)

    threshold_res: ThresholdSelection | None = None
    if tau is not None:
        accepted = tuple(i for i, sc in enumerate(scores) if sc >= tau)
        rng = np.random.default_rng(seed)
        if accepted:
            chosen = int(rng.choice(accepted))
            chosen_ans = _extract_valid_answer(trajs[chosen])
            threshold_res = ThresholdSelection(
                tau=float(tau),
                accepted_indices=accepted,
                fallback_selected_index=chosen,
                fallback_answer=chosen_ans,
                abstain_selected_index=chosen,
                abstain_answer=chosen_ans,
                abstained=False,
            )
        else:
            fallback_idx = int(rng.choice(len(trajs)))
            threshold_res = ThresholdSelection(
                tau=float(tau),
                accepted_indices=(),
                fallback_selected_index=fallback_idx,
                fallback_answer=_extract_valid_answer(trajs[fallback_idx]),
                abstain_selected_index=None,
                abstain_answer=None,
                abstained=True,
            )

    return BestOfNResult(
        selected_index=best_idx,
        selected_trajectory=best_traj,
        answer=best_ans,
        best_score=scores[best_idx],
        scores=scores,
        threshold_result=threshold_res,
        n_candidates=len(trajs),
        generation_prompt_tokens=gen_prompt,
        generation_completion_tokens=gen_comp,
        verifier_prompt_tokens=ver_prompt,
        verifier_completion_tokens=ver_comp,
        total_prompt_tokens=total_prompt,
        total_completion_tokens=total_comp,
        total_tokens=total_tokens,
        total_latency_s=total_latency,
        trajectories=trajs,
        verifications=vers,
    )


def run_best_of_n(
    env: StepEnvironment,
    verifier: HybridVerifier,
    question: BenchmarkQuestion,
    *,
    n: int = 5,
    temperature: float = 0.7,
    tau: float | None = None,
    seed: int = 42,
    max_steps: int = 8,
    max_seconds: float = 60.0,
    token_budget: int | None = None,
) -> BestOfNResult:
    """Sample up to `n` trajectories, verify each, and select the best candidate."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    trajs: list[Trajectory] = []
    vers: list[VerificationResult] = []
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
        ver = verifier.verify(question, traj, seed=seed + idx)
        trajs.append(traj)
        vers.append(ver)
        tokens_used += traj.total_tokens + ver.verifier_total_tokens

    return select_best_of_n(trajs, vers, tau=tau, seed=seed)


def calibrate_verifier_threshold(
    trajectories_by_q: Mapping[str, Sequence[Trajectory]],
    verifications_by_q: Mapping[str, Sequence[VerificationResult]],
    questions: Sequence[BenchmarkQuestion],
    *,
    candidate_taus: Sequence[float] | None = None,
    seed: int = 42,
    rel_tol: float = 1e-3,
) -> dict[str, Any]:
    """Grid-search Verifier threshold `tau` on dev_30 and measure empirical (TPR, FPR)."""
    if not questions:
        raise ValueError("questions must not be empty")

    taus = (
        [float(t) for t in candidate_taus]
        if candidate_taus is not None
        else [round(float(x), 2) for x in np.linspace(0.1, 0.95, 18)]
    )
    if not taus:
        raise ValueError("candidate_taus must not be empty")

    grid: list[dict[str, float]] = []
    for tau in taus:
        fallback_correct = 0
        abstain_correct = 0
        non_abstained = 0
        tp = fp = fn = tn = 0

        for q_idx, q in enumerate(questions):
            trajs = trajectories_by_q.get(q.question_id, ())
            vers = verifications_by_q.get(q.question_id, ())
            res = select_best_of_n(trajs, vers, tau=tau, seed=seed + q_idx)

            assert res.threshold_result is not None
            fb_ans = res.threshold_result.fallback_answer
            if fb_ans is not None and check_answer_match(
                fb_ans, q.ground_truth, q.answer_type, rel_tol=rel_tol
            ):
                fallback_correct += 1

            if not res.threshold_result.abstained:
                non_abstained += 1
                ab_ans = res.threshold_result.abstain_answer
                if ab_ans is not None and check_answer_match(
                    ab_ans, q.ground_truth, q.answer_type, rel_tol=rel_tol
                ):
                    abstain_correct += 1

            for traj, ver in zip(trajs, vers):
                cand_ans = _extract_valid_answer(traj)
                is_correct = cand_ans is not None and check_answer_match(
                    cand_ans, q.ground_truth, q.answer_type, rel_tol=rel_tol
                )
                accepted = ver.score >= tau
                if is_correct and accepted:
                    tp += 1
                elif is_correct and not accepted:
                    fn += 1
                elif not is_correct and accepted:
                    fp += 1
                else:
                    tn += 1

        n_q = len(questions)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        grid.append(
            {
                "tau": tau,
                "fallback_accuracy": fallback_correct / n_q,
                "abstain_accuracy": (
                    abstain_correct / non_abstained if non_abstained > 0 else 0.0
                ),
                "coverage": non_abstained / n_q,
                "tpr": tpr,
                "fpr": fpr,
            }
        )

    best_row = max(
        grid,
        key=lambda r: (
            r["fallback_accuracy"],
            r["tpr"] - r["fpr"],
            -abs(r["tau"] - 0.5),
        ),
    )
    return {
        "best_tau": best_row["tau"],
        "best_accuracy": best_row["fallback_accuracy"],
        "best_tpr": best_row["tpr"],
        "best_fpr": best_row["fpr"],
        "grid": grid,
    }
