"""Pure-Python + NumPy agent contracts, search algorithms, and evaluation utilities."""

from typing import Any

from veristat.agents.environment import VeriStatStepEnvironment
from veristat.agents.mcts import MCTSNode, MCTSResult, canonical_action_key, run_mcts
from veristat.agents.types import (
    FinalAnswer,
    LLMClient,
    LLMResponse,
    ReActStep,
    StepEnvironment,
    StepObservation,
    TerminatedReason,
    ToolAction,
    ToolName,
    Trajectory,
)

__all__ = [
    "FinalAnswer",
    "LLMClient",
    "LLMResponse",
    "MCTSNode",
    "MCTSResult",
    "ReActStep",
    "StepEnvironment",
    "StepObservation",
    "TerminatedReason",
    "ToolAction",
    "ToolName",
    "Trajectory",
    "VeriStatStepEnvironment",
    "build_catalog_context",
    "build_single_prompt",
    "canonical_action_key",
    "parse_single_prompt_response",
    "run_batch_single_prompt",
    "run_mcts",
    "run_single_prompt",
]


def __getattr__(name: str) -> Any:
    if name in (
        "build_catalog_context",
        "build_single_prompt",
        "parse_single_prompt_response",
        "run_batch_single_prompt",
        "run_single_prompt",
    ):
        from veristat.agents import single_prompt

        return getattr(single_prompt, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

