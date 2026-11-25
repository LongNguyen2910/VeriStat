"""Contracts and shared types for VeriStat Agent system.

Provides data structures for actions, tool results, and verification records
shared between ReAct (Member 2 - Khoa) and Verifier/MCTS (Member 3 - Sơn).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    """Represents a structured tool execution request from the Agent."""

    tool_name: str
    tool_input: dict[str, Any] = field(default_factory=dict)
    raw_thought: str = ""


@dataclass(frozen=True)
class AgentAction:
    """Action produced by the agent parser (either a tool call or terminal finish)."""

    is_finish: bool
    tool_call: ToolCall | None = None
    final_answer: dict[str, Any] | None = None
    raw_text: str = ""


@dataclass(frozen=True)
class VerificationResult:
    """Evaluation score and diagnostic feedback from the Verifier."""

    score: float  # 0.0 to 1.0
    passed_tier1: bool
    passed_tier2: bool = True
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
