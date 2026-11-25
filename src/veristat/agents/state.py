"""Agent state representation with deep cloneability and serialization.

Supports atomic state transitions for ReAct single-trajectory execution and
isolated branch expansion for MCTS-UCT tree search.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class StepRecord:
    """Record of a single reasoning step in an agent trajectory."""

    step_idx: int
    thought: str
    action_name: str | None = None
    action_input: dict[str, Any] | None = None
    observation: str = ""
    tool_error: str | None = None
    error_type: str | None = None
    execution_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    """State-centric representation of an agent's reasoning trajectory.

    Designed for functional state transitions and tree branch isolation.
    """

    question_id: str
    question: str
    current_step: int = 0
    steps: list[StepRecord] = field(default_factory=list)
    cited_tables: set[str] = field(default_factory=set)
    is_terminal: bool = False
    final_answer: dict[str, Any] | None = None
    status: str = "RUNNING"  # RUNNING, SUCCESS, MAX_STEPS_REACHED, SAFETY_HALT, ERROR
    metadata: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> AgentState:
        """Create an isolated, deep copy of this state for branch exploration."""
        return copy.deepcopy(self)

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to standard JSON-compatible dictionary."""
        data = asdict(self)
        data["cited_tables"] = sorted(list(self.cited_tables))
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentState:
        """Deserialize dictionary back to an AgentState object."""
        steps_raw = data.get("steps", [])
        steps = [
            StepRecord(
                step_idx=s["step_idx"],
                thought=s.get("thought", ""),
                action_name=s.get("action_name"),
                action_input=s.get("action_input"),
                observation=s.get("observation", ""),
                tool_error=s.get("tool_error"),
                error_type=s.get("error_type"),
                execution_time=s.get("execution_time", 0.0),
                metadata=s.get("metadata", {}),
            )
            for s in steps_raw
        ]
        return cls(
            question_id=data.get("question_id", ""),
            question=data.get("question", ""),
            current_step=data.get("current_step", 0),
            steps=steps,
            cited_tables=set(data.get("cited_tables", [])),
            is_terminal=data.get("is_terminal", False),
            final_answer=data.get("final_answer"),
            status=data.get("status", "RUNNING"),
            metadata=data.get("metadata", {}),
        )
