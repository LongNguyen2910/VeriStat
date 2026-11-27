"""Adapter bridging Khoa's LLM clients (VLLMClient, MockLLMClient) and SQLite cache
with Sơn's typed Protocol `veristat.agents.types.LLMClient`.
"""

from __future__ import annotations

import json
import time
from typing import Any

from veristat.agents.types import LLMResponse as TypedLLMResponse
from veristat.llm.base import LLMClient as BaseLLMClient
from veristat.llm.cache import SQLiteLLMCache, compute_canonical_cache_key


class AdaptiveLLMClient:
    """Wraps any VeriStat LLMClient (vLLM or Mock) with request caching and typed responses."""

    def __init__(
        self,
        backend_client: BaseLLMClient,
        cache: SQLiteLLMCache | None = None,
        *,
        model_name: str = "qwen2.5-7b",
    ) -> None:
        self.backend = backend_client
        self.cache = cache
        self.model_name = model_name

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
        logprobs: bool = False,
    ) -> TypedLLMResponse:
        """Generate model response conforming to `veristat.agents.types.LLMClient`."""
        cache_key = None
        cache_payload: dict[str, Any] | None = None

        if self.cache is not None:
            cache_payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "seed": seed,
            }
            cache_key = compute_canonical_cache_key(cache_payload)
            cached_resp = self.cache.get(cache_key)
            if cached_resp is not None:
                latency = float(cached_resp.metadata.get("latency", 0.001))
                return TypedLLMResponse(
                    text=cached_resp.text,
                    prompt_tokens=cached_resp.prompt_tokens,
                    completion_tokens=cached_resp.completion_tokens,
                    latency_s=latency,
                    logprobs=None,
                    cached=True,
                )

        start_time = time.perf_counter()
        raw_resp = self.backend.generate(
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
        )
        elapsed = time.perf_counter() - start_time

        if self.cache is not None and cache_key is not None and cache_payload is not None:
            self.cache.put(cache_key, raw_resp, cache_payload)

        # Extract logprobs if returned in metadata or raw choice
        extracted_logprobs: dict[str, float] | None = None
        if logprobs and "logprobs" in raw_resp.metadata:
            raw_lp = raw_resp.metadata["logprobs"]
            if isinstance(raw_lp, dict):
                extracted_logprobs = {str(k): float(v) for k, v in raw_lp.items()}

        return TypedLLMResponse(
            text=raw_resp.text,
            prompt_tokens=raw_resp.prompt_tokens,
            completion_tokens=raw_resp.completion_tokens,
            latency_s=elapsed,
            logprobs=extracted_logprobs,
            cached=False,
        )
