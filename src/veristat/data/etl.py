"""ETL pipeline and FTS5 search helpers for building and querying `veristat.db`."""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any

import pandas as pd

from veristat.data.admin_mapping import classify_and_normalize_geo
from veristat.data.pdf_parser import parse_gso_number
from veristat.data.schema import TableMeta

_SAFE_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FTS_RESERVED = {"and", "or", "not", "near"}
_TEXT_COLUMNS = {
    "dia_phuong",
    "geo_level",
    "chi_tieu",
    "phan_loai",
    "cap_hoc",
    "loai_truong",
    "nam_hoc",
}


def _validate_identifier(name: str) -> str:
    if not _SAFE_IDENT_RE.match(name):
        raise ValueError(f"Unsafe SQL identifier: {name!r}")
    return name


def sanitize_fts_query(query: str) -> str:
    """Sanitize a free-form Vietnamese/English query string for SQLite FTS5 MATCH.

    Strips FTS5 syntax operators (*, ", (), -, :, ^, +) and reserved boolean
    keywords, quotes each remaining token, and joins with OR so multi-word
    queries match robustly without raising sqlite3.OperationalError.
    """
    if not query or not str(query).strip():
        return ""
    # Extract alphanumeric/Unicode word tokens (keeps Vietnamese diacritics)
    raw_tokens = re.findall(r"[^\W_]+", str(query), flags=re.UNICODE)
    tokens: list[str] = []
    seen: set[str] = set()
    for tok in raw_tokens:
        low = tok.lower()
        if low in _FTS_RESERVED or low in seen:
            continue
        seen.add(low)
        tokens.append(f'"{tok}"')
    return " OR ".join(tokens)


def search_docs_fts(
    conn: sqlite3.Connection,
    query: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Search `docs_glossary` FTS5 table safely and return top-k ranked hits."""
    safe_q = sanitize_fts_query(query)
    if not safe_q:
        return []
    rows = conn.execute(
        """
        SELECT table_id, title, domain, content, bm25(docs_glossary) AS score
        FROM docs_glossary
        WHERE docs_glossary MATCH ?
        ORDER BY score ASC
        LIMIT ?
        """,
        (safe_q, int(top_k)),
    ).fetchall()
    return [
        {
            "table_id": r[0],
            "title": r[1],
            "domain": r[2],
            "content": r[3],
            "score": float(r[4]),
        }
        for r in rows
    ]


def _should_parse_as_numeric(col_name: str, series: pd.Series) -> bool:
    """Return True if an object/string column holds GSO numeric strings."""
    if col_name in _TEXT_COLUMNS:
        return False
    if pd.api.types.is_numeric_dtype(series):
        return False
    if col_name == "nam":
        # Keep school-year strings like '2020-2021' as TEXT
        return False
    # Check if majority of non-null values parse cleanly as GSO numbers or missing markers
    non_null = series.dropna().tolist()
    if not non_null:
        return False
    parsed = [parse_gso_number(v) for v in non_null]
    return any(v is not None for v in parsed)


def prepare_clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize geographic labels (+ add `geo_level`) and parse GSO numeric strings."""
    clean = df.copy()

    if "dia_phuong" in clean.columns:
        normalized = [classify_and_normalize_geo(str(v)) for v in clean["dia_phuong"]]
        clean["dia_phuong"] = [c for c, _ in normalized]
        geo_levels = [lvl for _, lvl in normalized]
        if "geo_level" in clean.columns:
            clean["geo_level"] = geo_levels
        else:
            pos = list(clean.columns).index("dia_phuong") + 1
            clean.insert(pos, "geo_level", geo_levels)

    for col in list(clean.columns):
        if _should_parse_as_numeric(col, clean[col]):
            clean[col] = [parse_gso_number(v) for v in clean[col]]

    return clean


def _sqlite_col_type(col_name: str, series: pd.Series) -> str:
    if col_name in _TEXT_COLUMNS:
        return "TEXT"
    if pd.api.types.is_integer_dtype(series):
        return "INTEGER"
    if pd.api.types.is_float_dtype(series):
        return "REAL"
    return "TEXT"


def _to_sql_value(val: Any) -> Any:
    if val is None or (isinstance(val, float) and pd.isna(val)) or pd.isna(val):
        return None
    if hasattr(val, "item"):
        return val.item()
    return val


def ingest_table(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    meta: TableMeta,
    glossary_note: str = "",
) -> int:
    """Idempotently normalize and ingest a DataFrame + metadata + FTS glossary into SQLite."""
    sql_table = _validate_identifier(meta.sql_table)
    clean = prepare_clean_dataframe(df)

    col_defs: list[str] = []
    col_names: list[str] = []
    for col in clean.columns:
        safe_col = _validate_identifier(str(col))
        col_names.append(safe_col)
        col_defs.append(f"{safe_col} {_sqlite_col_type(safe_col, clean[col])}")

    conn.execute(f"DROP TABLE IF EXISTS {sql_table}")
    conn.execute(f"CREATE TABLE {sql_table} ({', '.join(col_defs)})")

    if len(clean) > 0:
        placeholders = ", ".join(["?"] * len(col_names))
        col_list = ", ".join(col_names)
        records = [
            tuple(_to_sql_value(v) for v in row)
            for row in clean.itertuples(index=False, name=None)
        ]
        conn.executemany(
            f"INSERT INTO {sql_table} ({col_list}) VALUES ({placeholders})",
            records,
        )

    if "geo_level" in col_names:
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{sql_table}_geo_level ON {sql_table} (geo_level)"
        )
    if "dia_phuong" in col_names:
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{sql_table}_dia_phuong ON {sql_table} (dia_phuong)"
        )

    actual_cols_json = json.dumps(col_names, ensure_ascii=False)
    conn.execute(
        """
        INSERT OR REPLACE INTO tables_catalog
            (table_id, sql_table, title, domain, unit, reporting_period,
             scope, columns_json, source_ref, retrieved_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            meta.table_id,
            sql_table,
            meta.title,
            meta.domain,
            meta.unit,
            meta.reporting_period,
            meta.scope,
            actual_cols_json,
            meta.source_ref,
            meta.retrieved_at,
        ),
    )

    glossary_content_parts = [
        f"Bảng {meta.table_id} ({sql_table}): {meta.title}.",
        f"Đơn vị: {meta.unit}. Kỳ báo cáo: {meta.reporting_period}. Phạm vi: {meta.scope}.",
        f"Nguồn: {meta.source_ref}. Các cột: {', '.join(col_names)}.",
    ]
    if glossary_note:
        glossary_content_parts.append(glossary_note.strip())
    glossary_content = " ".join(glossary_content_parts)

    conn.execute("DELETE FROM docs_glossary WHERE table_id = ?", (meta.table_id,))
    conn.execute(
        """
        INSERT INTO docs_glossary (table_id, title, domain, content)
        VALUES (?, ?, ?, ?)
        """,
        (meta.table_id, meta.title, meta.domain, glossary_content),
    )

    conn.commit()
    return len(clean)


def ingest_glossary_section(
    conn: sqlite3.Connection,
    section_id: str,
    title: str,
    domain: str,
    content: str,
) -> None:
    """Idempotently store a domain methodology/glossary document in `docs_glossary` FTS5."""
    conn.execute("DELETE FROM docs_glossary WHERE table_id = ?", (section_id,))
    conn.execute(
        """
        INSERT INTO docs_glossary (table_id, title, domain, content)
        VALUES (?, ?, ?, ?)
        """,
        (section_id, title, domain, content),
    )
    conn.commit()
