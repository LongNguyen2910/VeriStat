"""Deterministic Mock LLM client for CPU-only testing and continuous integration."""

from __future__ import annotations

import re
from typing import Any

from veristat.llm.base import LLMClient, LLMResponse


class MockLLMClient(LLMClient):
    """Deterministic Mock LLM returning rule-based responses matching questions."""

    def __init__(self, canned_responses: dict[str, str] | None = None) -> None:
        self.canned_responses = canned_responses or {}
        self.call_count = 0
        self.history: list[dict[str, Any]] = []

    def generate(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int = 1024,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.call_count += 1
        self.history.append({
            "messages": messages,
            "temperature": temperature,
            "seed": seed,
            "max_tokens": max_tokens,
        })

        last_user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user_msg = str(m.get("content", ""))
                break

        # Check explicit canned matches
        for pattern, response_text in self.canned_responses.items():
            if pattern in last_user_msg or re.search(pattern, last_user_msg, re.IGNORECASE):
                return LLMResponse(
                    text=response_text,
                    prompt_tokens=50,
                    completion_tokens=len(response_text) // 4,
                    finish_reason="stop",
                    model="mock-qwen-7b",
                    request_id=f"mock-{self.call_count}",
                )

        # Default fallback response for ReAct loop tests
        default_resp = (
            '{\n'
            '  "thought": "Tra cứu bảng số liệu liên quan",\n'
            '  "action": "search_docs",\n'
            '  "action_input": {"query": "dân số"}\n'
            '}'
        )
        return LLMResponse(
            text=default_resp,
            prompt_tokens=40,
            completion_tokens=25,
            finish_reason="stop",
            model="mock-qwen-7b",
            request_id=f"mock-{self.call_count}",
        )
