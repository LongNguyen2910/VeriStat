"""Deterministic CPU mocks and trajectory builders for unit tests."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from veristat.agents.types import (
    FinalAnswer,
    LLMResponse,
    ReActStep,
    StepObservation,
    TerminatedReason,
    ToolAction,
    Trajectory,
)
from veristat.data.schema import AnswerType, BenchmarkQuestion


class MockLLMClient:
    """Scripted or callback-driven mock `LLMClient` recording all calls."""

    def __init__(
        self,
        responder: (
            LLMResponse
            | Sequence[LLMResponse]
            | Callable[[list[dict[str, str]], float, int, int | None, bool], LLMResponse]
        ),
    ) -> None:
        self._responder = responder
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
        logprobs: bool = False,
    ) -> LLMResponse:
        idx = len(self.calls)
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "seed": seed,
                "logprobs": logprobs,
            }
        )
        if callable(self._responder):
            return self._responder(messages, temperature, max_tokens, seed, logprobs)
        if isinstance(self._responder, LLMResponse):
            return self._responder
        return self._responder[idx % len(self._responder)]


class MockStepEnvironment:
    """Deterministic mock `StepEnvironment` for CPU unit testing."""

    def __init__(
        self,
        *,
        react_trajectories: Sequence[Trajectory] | Callable[[BenchmarkQuestion, int | None], Trajectory] = (),
        propose_fn: Callable[[BenchmarkQuestion, tuple[ReActStep, ...], int, float, int | None], list[ToolAction]] | None = None,
        execute_fn: Callable[[ToolAction, tuple[ReActStep, ...]], StepObservation] | None = None,
    ) -> None:
        self._react_trajectories = react_trajectories
        self._propose_fn = propose_fn
        self._execute_fn = execute_fn
        self.run_react_calls: list[dict[str, Any]] = []
        self.propose_calls: list[dict[str, Any]] = []
        self.execute_calls: list[ToolAction] = []

    def propose_actions(
        self,
        question: BenchmarkQuestion,
        history: tuple[ReActStep, ...],
        *,
        k: int = 1,
        temperature: float = 0.7,
        seed: int | None = None,
    ) -> list[ToolAction]:
        self.propose_calls.append(
            {
                "question_id": question.question_id,
                "depth": len(history),
                "k": k,
                "temperature": temperature,
                "seed": seed,
            }
        )
        if self._propose_fn is not None:
            return self._propose_fn(question, history, k, temperature, seed)
        return []

    def execute_action(
        self,
        action: ToolAction,
        history: tuple[ReActStep, ...] = (),
    ) -> StepObservation:
        self.execute_calls.append(action)
        if self._execute_fn is not None:
            return self._execute_fn(action, history)
        return StepObservation(output="ok", ok=True)

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
        idx = len(self.run_react_calls)
        self.run_react_calls.append(
            {
                "question_id": question.question_id,
                "temperature": temperature,
                "seed": seed,
                "max_steps": max_steps,
                "max_seconds": max_seconds,
                "max_tokens": max_tokens,
            }
        )
        if callable(self._react_trajectories):
            return self._react_trajectories(question, seed)
        return self._react_trajectories[idx % len(self._react_trajectories)]


def make_question(
    *,
    question_id: str = "Q_TEST_001",
    question: str = "Tốc độ tăng trưởng dân số trung bình của Hà Nội giai đoạn 2020-2025 là bao nhiêu %?",
    answer_type: AnswerType = "number",
    ground_truth: float | str | list[str] = 10.0,
    unit: str = "%",
    required_table_ids: list[str] | None = None,
) -> BenchmarkQuestion:
    return BenchmarkQuestion(
        question_id=question_id,
        question=question,
        split="dev",
        category="growth",
        difficulty="easy",
        answer_type=answer_type,
        ground_truth=ground_truth,
        unit=unit,
        required_table_ids=required_table_ids or ["NG2025_T017"],
        reference_code="result = 10.0",
        uses_admin_mapping=False,
    )


def make_trajectory(
    *,
    question_id: str = "Q_TEST_001",
    answer: float | str | list[str] | None = 10.0,
    unit: str = "%",
    cited_tables: tuple[str, ...] = ("NG2025_T017",),
    queried_tables: tuple[str, ...] = ("NG2025_T017",),
    py_values: tuple[Any, ...] | None = None,
    sql_values: tuple[Any, ...] = (100.0, 110.0),
    parse_ok: bool = True,
    include_final: bool = True,
    safety_violation: bool = False,
    include_query_step: bool = True,
    terminated_reason: TerminatedReason = "final_answer",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    latency_s: float = 0.25,
    seed: int | None = 42,
) -> Trajectory:
    steps: list[ReActStep] = []
    idx = 0

    if include_query_step:
        steps.append(
            ReActStep(
                step_idx=idx,
                action=ToolAction(
                    thought="Truy vấn bảng thống kê",
                    tool_name="query_table",
                    tool_args={"sql": "SELECT gia_tri FROM stat_ng2025_t017"},
                    raw_text="Action: query_table",
                    prompt_tokens=40,
                    completion_tokens=20,
                    latency_s=0.1,
                ),
                observation=StepObservation(
                    output=[{"gia_tri": v} for v in sql_values],
                    ok=True,
                    queried_tables=queried_tables,
                    computed_values=sql_values,
                    latency_s=0.01,
                ),
            )
        )
        idx += 1

    resolved_py = (answer,) if (py_values is None and answer is not None) else py_values
    if resolved_py is not None or safety_violation:
        steps.append(
            ReActStep(
                step_idx=idx,
                action=ToolAction(
                    thought="Tính toán kiểm chứng bằng Python",
                    tool_name="run_python",
                    tool_args={"code": "result = 10.0"},
                    raw_text="Action: run_python",
                    prompt_tokens=30,
                    completion_tokens=15,
                    latency_s=0.1,
                ),
                observation=StepObservation(
                    output=str(resolved_py[0]) if resolved_py else "",
                    ok=not safety_violation,
                    error="Blocked import" if safety_violation else None,
                    safety_violation=safety_violation,
                    computed_values=resolved_py or (),
                    latency_s=0.02,
                ),
            )
        )
        idx += 1

    final_obj: FinalAnswer | None = None
    if include_final:
        final_obj = FinalAnswer(
            answer=answer,
            unit=unit,
            cited_tables=cited_tables,
            computation_code="result = 10.0",
            parse_ok=parse_ok,
        )
        steps.append(
            ReActStep(
                step_idx=idx,
                action=ToolAction(
                    thought="Kết luận đáp số",
                    tool_name="final_answer",
                    tool_args={
                        "answer": answer,
                        "unit": unit,
                        "cited_tables": list(cited_tables),
                    },
                    raw_text="Final Answer",
                    prompt_tokens=30,
                    completion_tokens=15,
                    latency_s=0.05,
                ),
                observation=None,
            )
        )

    return Trajectory(
        question_id=question_id,
        steps=tuple(steps),
        final=final_obj,
        terminated_reason="safety_violation" if safety_violation else terminated_reason,
        total_prompt_tokens=prompt_tokens,
        total_completion_tokens=completion_tokens,
        total_latency_s=latency_s,
        seed=seed,
    )
