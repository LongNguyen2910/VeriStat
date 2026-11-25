"""Base interfaces and response contracts for LLM communication."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LLMResponse:
    """Standardized response from any LLM client."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    cached: bool = False
    model: str = ""
    request_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMClient(ABC):
    """Abstract base class for all LLM client implementations."""

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int = 1024,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response from the model given chat messages and parameters."""
        ...
