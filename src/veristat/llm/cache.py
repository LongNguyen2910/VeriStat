"""SQLite-backed persistent LLM request cache using canonical JSON hashing."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

from veristat.llm.base import LLMResponse

_CACHE_DDL = """
CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    prompt_json TEXT NOT NULL,
    response_text TEXT NOT NULL,
    finish_reason TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    request_metadata TEXT
);
CREATE INDEX IF NOT EXISTS idx_llm_cache_model ON llm_cache (model);
"""


def compute_canonical_cache_key(payload: dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash from canonical JSON payload."""
    canonical = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class SQLiteLLMCache:
    """Persistent SQLite cache for LLM requests across sessions."""

    def __init__(self, db_path: Path | str = "data/processed/llm_cache.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        conn.execute("PRAGMA busy_timeout = 5000")
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connection() as conn:
            conn.executescript(_CACHE_DDL)
            conn.commit()

    def get(self, cache_key: str) -> LLMResponse | None:
        """Lookup cached response by canonical cache key."""
        with self._connection() as conn:
            cursor = conn.cursor()
            row = cursor.execute(
                """
                SELECT response_text, prompt_tokens, completion_tokens, finish_reason, model
                FROM llm_cache WHERE cache_key = ?
                """,
                (cache_key,),
            ).fetchone()

        if row is None:
            return None

        text, p_tokens, c_tokens, f_reason, model = row
        return LLMResponse(
            text=text,
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            finish_reason=f_reason,
            cached=True,
            model=model,
            request_id=f"cache-{cache_key[:8]}",
        )

    def put(
        self,
        cache_key: str,
        response: LLMResponse,
        request_payload: dict[str, Any],
    ) -> None:
        """Store a successful, valid LLM response in cache."""
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO llm_cache (
                    cache_key, model, prompt_json, response_text, finish_reason,
                    prompt_tokens, completion_tokens, created_at, request_metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cache_key,
                    response.model or request_payload.get("model", "unknown"),
                    json.dumps(request_payload, sort_keys=True, ensure_ascii=False),
                    response.text,
                    response.finish_reason,
                    response.prompt_tokens,
                    response.completion_tokens,
                    now_iso,
                    json.dumps(response.metadata, ensure_ascii=False),
                ),
            )
            conn.commit()
