"""Database schema, core data types, and read-only SQLite helpers for VeriStat."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

AnswerType = Literal["number", "entity", "entity_list"]
GeoLevel = Literal["country", "region", "province", "other"]
DomainType = Literal["dan_so", "lao_dong", "giao_duc", "y_te", "doanh_nghiep"]
DifficultyType = Literal["easy", "medium", "hard"]
TaskCategory = Literal["growth", "share", "ranking", "correlation"]


@dataclass
class TableMeta:
    """Metadata describing a digitized statistical table stored in veristat.db."""

    table_id: str
    sql_table: str
    title: str
    domain: DomainType
    unit: str
    reporting_period: str
    scope: str
    columns_json: str
    source_ref: str
    retrieved_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TableMeta:
        return cls(
            table_id=str(data["table_id"]),
            sql_table=str(data["sql_table"]),
            title=str(data["title"]),
            domain=data["domain"],
            unit=str(data["unit"]),
            reporting_period=str(data["reporting_period"]),
            scope=str(data["scope"]),
            columns_json=str(data["columns_json"]),
            source_ref=str(data["source_ref"]),
            retrieved_at=str(data["retrieved_at"]),
        )


@dataclass
class BenchmarkQuestion:
    """A single verifiable socioeconomic QA item in dev_30 or benchmark_400."""

    question_id: str
    question: str
    split: Literal["dev", "test"]
    category: TaskCategory
    difficulty: DifficultyType
    answer_type: AnswerType
    ground_truth: float | str | list[str]
    unit: str
    required_table_ids: list[str]
    reference_code: str
    uses_admin_mapping: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkQuestion:
        return cls(
            question_id=str(data["question_id"]),
            question=str(data["question"]),
            split=data["split"],
            category=data["category"],
            difficulty=data["difficulty"],
            answer_type=data["answer_type"],
            ground_truth=data["ground_truth"],
            unit=str(data["unit"]),
            required_table_ids=list(data["required_table_ids"]),
            reference_code=str(data["reference_code"]),
            uses_admin_mapping=bool(data["uses_admin_mapping"]),
        )


_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS tables_catalog (
    table_id TEXT PRIMARY KEY,
    sql_table TEXT NOT NULL,
    title TEXT NOT NULL,
    domain TEXT NOT NULL,
    unit TEXT NOT NULL,
    reporting_period TEXT NOT NULL,
    scope TEXT NOT NULL,
    columns_json TEXT NOT NULL,
    source_ref TEXT NOT NULL,
    retrieved_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS admin_mapping (
    raw_name TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    geo_level TEXT NOT NULL CHECK (geo_level IN ('country', 'region', 'province', 'other')),
    region_name TEXT,
    canonical_2026 TEXT,
    effective_from TEXT,
    change_type TEXT NOT NULL,
    note TEXT
);

CREATE INDEX IF NOT EXISTS idx_admin_mapping_canonical
    ON admin_mapping (canonical_name);

CREATE INDEX IF NOT EXISTS idx_admin_mapping_geo_level
    ON admin_mapping (geo_level);

CREATE VIRTUAL TABLE IF NOT EXISTS docs_glossary USING fts5(
    table_id,
    title,
    domain,
    content,
    tokenize='unicode61'
);
"""


def init_database(db_path: Path | str) -> sqlite3.Connection:
    """Initialize SQLite database with catalog, admin_mapping, and FTS5 glossary."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.executescript(_SCHEMA_DDL)
    conn.commit()
    return conn


def open_readonly(db_path: Path | str) -> sqlite3.Connection:
    """Open an existing SQLite database in strict read-only mode (URI mode=ro)."""
    resolved = Path(db_path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Database file not found: {resolved}")
    uri = f"file:{resolved.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.execute("PRAGMA query_only = ON")
    return conn
