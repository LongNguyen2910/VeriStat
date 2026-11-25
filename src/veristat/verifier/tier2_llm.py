"""Tier 2 LLM-as-a-Verifier rubric scoring (with optional logprobs blending)."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from veristat.agents.types import LLMClient, Trajectory
from veristat.data.schema import BenchmarkQuestion

VerdictType = Literal["yes", "no", "unknown"]

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

_TIER2_SYSTEM_PROMPT = (
    "Bạn là giám khảo kiểm chứng chất lượng suy luận thống kê kinh tế - xã hội Việt Nam. "
    "Nhiệm vụ của bạn là đánh giá vết truy vấn SQL và mã Python đã qua kiểm tra cú pháp "
    "xem có đúng bảng chỉ tiêu, đúng địa giới hành chính/geo_level, đúng công thức toán học "
    "và đúng đơn vị đo hay không. Chỉ trả về duy nhất một JSON hợp lệ."
)


@dataclass(frozen=True)
class Tier2Result:
    """Outcome of Tier-2 LLM rubric evaluation."""

    score: float
    table_and_geo: float
    formula: float
    unit_and_type: float
    verdict: VerdictType
    reason: str
    parse_fallback: bool
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    raw_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": float(self.score),
            "table_and_geo": float(self.table_and_geo),
            "formula": float(self.formula),
            "unit_and_type": float(self.unit_and_type),
            "verdict": self.verdict,
            "reason": self.reason,
            "parse_fallback": bool(self.parse_fallback),
            "prompt_tokens": int(self.prompt_tokens),
            "completion_tokens": int(self.completion_tokens),
            "latency_s": float(self.latency_s),
            "raw_text": self.raw_text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Tier2Result:
        return cls(
            score=float(data["score"]),
            table_and_geo=float(data["table_and_geo"]),
            formula=float(data["formula"]),
            unit_and_type=float(data["unit_and_type"]),
            verdict=data.get("verdict", "unknown"),
            reason=str(data.get("reason", "")),
            parse_fallback=bool(data.get("parse_fallback", False)),
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            latency_s=float(data.get("latency_s", 0.0)),
            raw_text=str(data.get("raw_text", "")),
        )


def _truncate_text(value: Any, max_chars: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def build_tier2_messages(
    question: BenchmarkQuestion,
    trajectory: Trajectory,
    *,
    max_obs_chars: int = 1200,
) -> list[dict[str, str]]:
    """Build rubric prompt messages for Tier-2 LLM verification."""
    step_lines: list[str] = []
    for step in trajectory.steps:
        act = step.action
        args_str = _truncate_text(act.tool_args, 400)
        obs_str = "None"
        if step.observation is not None:
            obs = step.observation
            if obs.ok:
                obs_str = _truncate_text(obs.output, max_obs_chars)
            else:
                obs_str = f"ERROR: {obs.error}"
        step_lines.append(
            f"- Bước {step.step_idx} [{act.tool_name}] args={args_str} -> obs={obs_str}"
        )

    final = trajectory.final
    final_ans = final.answer if final is not None else None
    final_unit = final.unit if final is not None else ""
    cited = list(final.cited_tables) if final is not None else []
    comp_code = final.computation_code if final is not None else ""

    user_prompt = (
        f"Câu hỏi: {question.question}\n"
        f"Kiểu đáp án kỳ vọng: {question.answer_type} | Đơn vị kỳ vọng: {question.unit}\n"
        f"Các bảng đã trích dẫn: {cited}\n"
        f"Đáp án cuối: {final_ans!r} | Đơn vị trả về: {final_unit!r}\n"
        f"Mã tính toán:\n{comp_code or '(trống)'}\n\n"
        f"Tóm tắt các bước thực thi:\n" + "\n".join(step_lines) + "\n\n"
        "Chấm 3 tiêu chí theo thang {0, 0.5, 1}:\n"
        "1. table_and_geo (0.4): đúng bảng chỉ tiêu, đúng năm/kỳ, lọc đúng geo_level và xử lý đúng ánh xạ 63<->34 tỉnh nếu cần.\n"
        "2. formula (0.4): đúng công thức thống kê (tăng trưởng, CAGR, tỷ trọng, xếp hạng, tương quan).\n"
        "3. unit_and_type (0.2): đúng đơn vị đo và đúng kiểu dữ liệu đầu ra.\n\n"
        'Trả về đúng định dạng JSON:\n'
        '{"table_and_geo": <0|0.5|1>, "formula": <0|0.5|1>, '
        '"unit_and_type": <0|0.5|1>, "verdict": "yes"|"no", "reason": "<ngắn gọn>"}'
    )
    return [
        {"role": "system", "content": _TIER2_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _extract_json_dict(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None

    candidates: list[str] = []
    fence_match = _CODE_FENCE_RE.search(stripped)
    if fence_match:
        candidates.append(fence_match.group(1).strip())
    candidates.append(stripped)

    decoder = json.JSONDecoder()
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        idx = cand.find("{")
        while idx != -1:
            try:
                obj, _ = decoder.raw_decode(cand, idx)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
            idx = cand.find("{", idx + 1)
    return None


def _parse_rubric_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    return float(np.clip(num, 0.0, 1.0))


def _normalize_verdict(raw: Any) -> VerdictType:
    val = str(raw).strip().lower()
    if val in ("yes", "true", "pass", "1"):
        return "yes"
    if val in ("no", "false", "fail", "0"):
        return "no"
    return "unknown"


def parse_tier2_response(
    text: str,
) -> tuple[float, float, float, VerdictType, str, bool]:
    """Parse Tier-2 JSON rubric output, falling back conservatively to 0.5 on error."""
    data = _extract_json_dict(text)
    if data is None:
        return (0.5, 0.5, 0.5, "unknown", "parse_fallback", True)

    t_geo = _parse_rubric_float(data.get("table_and_geo"))
    formula = _parse_rubric_float(data.get("formula"))
    u_type = _parse_rubric_float(data.get("unit_and_type"))
    if t_geo is None or formula is None or u_type is None:
        return (0.5, 0.5, 0.5, "unknown", "parse_fallback", True)

    verdict = _normalize_verdict(data.get("verdict", "unknown"))
    reason = str(data.get("reason", "")).strip() or "ok"
    return (t_geo, formula, u_type, verdict, reason, False)


def _extract_yes_probability(logprobs: dict[str, float] | None) -> float | None:
    if not logprobs:
        return None
    yes_lp: float | None = None
    no_lp: float | None = None
    for tok, lp in logprobs.items():
        key = tok.strip().lower()
        if not math.isfinite(float(lp)):
            continue
        if key == "yes":
            yes_lp = float(lp) if yes_lp is None else max(yes_lp, float(lp))
        elif key == "no":
            no_lp = float(lp) if no_lp is None else max(no_lp, float(lp))
    if yes_lp is None or no_lp is None:
        return None
    m = max(yes_lp, no_lp)
    exp_y = math.exp(yes_lp - m)
    exp_n = math.exp(no_lp - m)
    return exp_y / (exp_y + exp_n)


def verify_tier2(
    llm: LLMClient,
    question: BenchmarkQuestion,
    trajectory: Trajectory,
    *,
    seed: int | None = None,
    max_tokens: int = 256,
    use_logprobs: bool = False,
    logprob_weight: float = 0.2,
) -> Tier2Result:
    """Run Tier-2 LLM-as-a-Verifier rubric check on a Tier-1-passing trajectory."""
    messages = build_tier2_messages(question, trajectory)
    response = llm.generate(
        messages,
        temperature=0.0,
        max_tokens=max_tokens,
        seed=seed,
        logprobs=use_logprobs,
    )
    t_geo, formula, u_type, verdict, reason, parse_fallback = parse_tier2_response(
        response.text
    )
    rubric_score = 0.4 * t_geo + 0.4 * formula + 0.2 * u_type

    final_score = rubric_score
    if use_logprobs:
        p_yes = _extract_yes_probability(response.logprobs)
        if p_yes is not None:
            w = float(np.clip(logprob_weight, 0.0, 1.0))
            final_score = (1.0 - w) * rubric_score + w * p_yes

    final_score = float(np.clip(final_score, 0.0, 1.0))
    return Tier2Result(
        score=final_score,
        table_and_geo=t_geo,
        formula=formula,
        unit_and_type=u_type,
        verdict=verdict,
        reason=reason,
        parse_fallback=parse_fallback,
        prompt_tokens=int(response.prompt_tokens),
        completion_tokens=int(response.completion_tokens),
        latency_s=float(response.latency_s),
        raw_text=response.text,
    )
