"""OpenAI-compatible vLLM client with conditional retries and latency tracking."""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from veristat.llm.base import LLMClient, LLMResponse


@dataclass
class VLLMConfig:
    base_url: str = "http://localhost:8000/v1"
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    timeout_seconds: float = 60.0
    max_retries: int = 3
    base_backoff: float = 1.0


class VLLMClient(LLMClient):
    """Client for querying OpenAI-compatible vLLM server with robust retry policies."""

    def __init__(self, config: VLLMConfig | None = None) -> None:
        self.config = config or VLLMConfig()

    def generate(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int = 1024,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        endpoint = f"{self.config.base_url.rstrip('/')}/chat/completions"

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed
        if tools:
            payload["tools"] = tools
        for k, v in kwargs.items():
            if v is not None:
                payload[k] = v

        data_bytes = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}

        retries = 0
        last_error = None
        start_time = time.perf_counter()

        while retries <= self.config.max_retries:
            try:
                req = urllib.request.Request(endpoint, data=data_bytes, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    elapsed = time.perf_counter() - start_time

                    choice = resp_data["choices"][0]
                    usage = resp_data.get("usage", {})
                    return LLMResponse(
                        text=choice["message"]["content"] or "",
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                        finish_reason=choice.get("finish_reason", "stop"),
                        model=resp_data.get("model", self.config.model),
                        request_id=resp_data.get("id"),
                        metadata={"retries": retries, "latency": elapsed},
                    )
            except urllib.error.HTTPError as e:
                last_error = e
                # Do NOT retry client errors (400, 401, 403, 404, 422)
                if e.code in (400, 401, 403, 404, 422):
                    raise RuntimeError(f"VLLMClient fatal HTTP error {e.code}: {e.read().decode('utf-8', errors='ignore')}") from e
                # Retry on 429 and 5xx
                if e.code == 429 or e.code >= 500:
                    retries += 1
                    if retries > self.config.max_retries:
                        break
                    backoff = self.config.base_backoff * (2 ** (retries - 1)) + random.uniform(0, 0.5)
                    time.sleep(backoff)
                else:
                    raise
            except (urllib.error.URLError, TimeoutError) as e:
                last_error = e
                retries += 1
                if retries > self.config.max_retries:
                    break
                backoff = self.config.base_backoff * (2 ** (retries - 1)) + random.uniform(0, 0.5)
                time.sleep(backoff)

        raise RuntimeError(f"VLLMClient failed after {self.config.max_retries} retries. Last error: {last_error}")
