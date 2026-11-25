"""TaskCheckpoint: JSONL progress tracking and session resume for benchmarks."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class TaskCheckpoint:
    """Manages benchmark progress via append-only JSONL files with corrupt-tail tolerance."""

    def __init__(self, file_path: Path | str) -> None:
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def load_completed_keys(self) -> set[tuple[str, str, str, str]]:
        """Load set of completed (dataset_version, method, config_hash, question_id) tuples."""
        completed: set[tuple[str, str, str, str]] = set()
        if not self.file_path.exists():
            return completed

        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                    # Only mark SUCCESS records as completed
                    if record.get("status") == "SUCCESS":
                        key = (
                            str(record.get("dataset_version", "")),
                            str(record.get("method", "")),
                            str(record.get("config_hash", "")),
                            str(record.get("question_id", "")),
                        )
                        completed.add(key)
                except json.JSONDecodeError:
                    # Ignore incomplete or corrupted trailing lines (e.g. from Colab abrupt shutdown)
                    continue
        return completed

    def is_completed(
        self,
        question_id: str,
        dataset_version: str = "benchmark_v1",
        method: str = "react",
        config_hash: str = "default",
    ) -> bool:
        """Check if a specific task compound key is already completed."""
        completed = self.load_completed_keys()
        return (dataset_version, method, config_hash, question_id) in completed

    def append_result(self, record: dict[str, Any]) -> None:
        """Safely append a benchmark result line with flush and fsync."""
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
