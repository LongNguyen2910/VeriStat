"""Shared immutable data contracts and protocols between Khoa (Tools/LLM/ReAct)
and Sơn (SC/Verifier/Best-of-n/MCTS/Eval).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from veristat.data.schema import BenchmarkQuestion

ToolName = Literal["search_docs", "query_table", "run_python", "final_answer"]
TerminatedReason = Literal[
    "final_answer",
    "max_steps",
    "timeout",
    "token_budget",
    "safety_violation",
    "error",
]


def _normalize_final_answer_value(raw: Any) -> float | str | list[str] | None:
    if raw is None:
        return None
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, bool):
        return str(raw)
    if isinstance(raw, (int, float)):
        return float(raw)
    return str(raw)


@dataclass(frozen=True)
class LLMResponse:
    """Normalized response returned by `LLMClient.generate`."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    logprobs: dict[str, float] | None = None
    cached: bool = False


@dataclass(frozen=True)
class ToolAction:
    """Single Thought + Action parsed from one LLM step."""

    thought: str
    tool_name: ToolName
    tool_args: dict[str, Any]
    raw_text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "thought": self.thought,
            "tool_name": self.tool_name,
            "tool_args": dict(self.tool_args),
            "raw_text": self.raw_text,
            "prompt_tokens": int(self.prompt_tokens),
            "completion_tokens": int(self.completion_tokens),
            "latency_s": float(self.latency_s),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolAction:
        return cls(
            thought=str(data.get("thought", "")),
            tool_name=data["tool_name"],
            tool_args=dict(data.get("tool_args") or {}),
            raw_text=str(data.get("raw_text", "")),
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            latency_s=float(data.get("latency_s", 0.0)),
        )


@dataclass(frozen=True)
class StepObservation:
    """Structured observation produced by executing one `ToolAction`."""

    output: Any
    ok: bool
    error: str | None = None
    safety_violation: bool = False
    queried_tables: tuple[str, ...] = ()
    computed_values: tuple[Any, ...] = ()
    latency_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "output": self.output,
            "ok": bool(self.ok),
            "error": self.error,
            "safety_violation": bool(self.safety_violation),
            "queried_tables": list(self.queried_tables),
            "computed_values": list(self.computed_values),
            "latency_s": float(self.latency_s),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StepObservation:
        err = data.get("error")
        return cls(
            output=data.get("output"),
            ok=bool(data["ok"]),
            error=str(err) if err is not None else None,
            safety_violation=bool(data.get("safety_violation", False)),
            queried_tables=tuple(str(t) for t in (data.get("queried_tables") or ())),
            computed_values=tuple(data.get("computed_values") or ()),
            latency_s=float(data.get("latency_s", 0.0)),
        )


@dataclass(frozen=True)
class ReActStep:
    """One Thought -> Action -> Observation transition in a trajectory."""

    step_idx: int
    action: ToolAction
    observation: StepObservation | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_idx": int(self.step_idx),
            "action": self.action.to_dict(),
            "observation": (
                self.observation.to_dict() if self.observation is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReActStep:
        obs_raw = data.get("observation")
        return cls(
            step_idx=int(data["step_idx"]),
            action=ToolAction.from_dict(data["action"]),
            observation=(
                StepObservation.from_dict(obs_raw) if obs_raw is not None else None
            ),
        )


@dataclass(frozen=True)
class FinalAnswer:
    """Structured final output extracted from a `final_answer` action."""

    answer: float | str | list[str] | None
    unit: str
    cited_tables: tuple[str, ...]
    computation_code: str
    parse_ok: bool

    def to_dict(self) -> dict[str, Any]:
        ans = (
            list(self.answer)
            if isinstance(self.answer, list)
            else self.answer
        )
        return {
            "answer": ans,
            "unit": self.unit,
            "cited_tables": list(self.cited_tables),
            "computation_code": self.computation_code,
            "parse_ok": bool(self.parse_ok),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FinalAnswer:
        return cls(
            answer=_normalize_final_answer_value(data.get("answer")),
            unit=str(data.get("unit", "")),
            cited_tables=tuple(str(t) for t in (data.get("cited_tables") or ())),
            computation_code=str(data.get("computation_code", "")),
            parse_ok=bool(data.get("parse_ok", False)),
        )


@dataclass(frozen=True)
class Trajectory:
    """Complete execution trace for a single question."""

    question_id: str
    steps: tuple[ReActStep, ...]
    final: FinalAnswer | None
    terminated_reason: TerminatedReason
    total_prompt_tokens: int
    total_completion_tokens: int
    total_latency_s: float
    seed: int | None = None

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def tool_calls_count(self) -> int:
        return sum(1 for s in self.steps if s.action.tool_name != "final_answer")

    @property
    def has_safety_violation(self) -> bool:
        return self.terminated_reason == "safety_violation" or any(
            s.observation is not None and s.observation.safety_violation
            for s in self.steps
        )

    @property
    def all_queried_tables(self) -> tuple[str, ...]:
        seen: list[str] = []
        for step in self.steps:
            if step.observation is None:
                continue
            for tid in step.observation.queried_tables:
                if tid not in seen:
                    seen.append(tid)
        return tuple(seen)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "steps": [s.to_dict() for s in self.steps],
            "final": self.final.to_dict() if self.final is not None else None,
            "terminated_reason": self.terminated_reason,
            "total_prompt_tokens": int(self.total_prompt_tokens),
            "total_completion_tokens": int(self.total_completion_tokens),
            "total_latency_s": float(self.total_latency_s),
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trajectory:
        final_raw = data.get("final")
        seed_raw = data.get("seed")
        return cls(
            question_id=str(data["question_id"]),
            steps=tuple(ReActStep.from_dict(s) for s in (data.get("steps") or ())),
            final=FinalAnswer.from_dict(final_raw) if final_raw is not None else None,
            terminated_reason=data["terminated_reason"],
            total_prompt_tokens=int(data.get("total_prompt_tokens", 0)),
            total_completion_tokens=int(data.get("total_completion_tokens", 0)),
            total_latency_s=float(data.get("total_latency_s", 0.0)),
            seed=int(seed_raw) if seed_raw is not None else None,
        )


class LLMClient(Protocol):
    """OpenAI-compatible / vLLM client (with SQLite cache) implemented by Khoa."""

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
        logprobs: bool = False,
    ) -> LLMResponse:
        ...


class StepEnvironment(Protocol):
    """Step-level ReAct + 3-tool environment shared by ReAct, SC, BoN, and MCTS."""

    def propose_actions(
        self,
        question: BenchmarkQuestion,
        history: tuple[ReActStep, ...],
        *,
        k: int = 1,
        temperature: float = 0.7,
        seed: int | None = None,
    ) -> list[ToolAction]:
        ...

    def execute_action(
        self,
        action: ToolAction,
        history: tuple[ReActStep, ...] = (),
    ) -> StepObservation:
        ...

    def extract_final_answer(self, action: ToolAction) -> FinalAnswer:
        ...

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
        ...
