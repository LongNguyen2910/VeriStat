"""QueryTable tool: read-only SQL execution on veristat.db with metadata enrichment."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from veristat.data.schema import open_readonly
from veristat.tools.base import BaseTool, ToolResult

_FORBIDDEN_SQL_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|PRAGMA|REPLACE|TRUNCATE)\b",
    re.IGNORECASE,
)


class QueryTableTool(BaseTool):
    """Executes SELECT queries on the verified SQLite database in read-only mode."""

    name = "query_table"
    description = (
        "Thực thi câu truy vấn SQL (SELECT only) trên CSDL thống kê SQLite veristat.db. "
        "Tự động giới hạn tối đa 200 dòng và đính kèm siêu dữ liệu đơn vị tính."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "Câu lệnh SQL chỉ đọc (SELECT). Nghiêm cấm mọi lệnh DDL/DML.",
            }
        },
        "required": ["sql"],
        "additionalProperties": False,
    }

    def __init__(self, db_path: Path | str = "data/processed/veristat.db") -> None:
        self.db_path = Path(db_path)

    def execute(self, **kwargs: Any) -> ToolResult:
        sql = kwargs.get("sql", "").strip()
        if not sql:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error="SQL query string cannot be empty.",
                error_type="SQL_SYNTAX",
            )

        # 1. Enforce read-only constraint
        if _FORBIDDEN_SQL_PATTERN.search(sql):
            match = _FORBIDDEN_SQL_PATTERN.search(sql)
            forbidden_word = match.group(0).upper() if match else "MODIFICATION"
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error=f"SecurityViolation: Prohibited DDL/DML statement keyword '{forbidden_word}'. Only SELECT queries are permitted.",
                error_type="SECURITY_VIOLATION",
                metadata={"prohibited_keyword": forbidden_word},
            )

        cleaned_sql = sql.rstrip("; \t\n")
        # Ensure query begins with SELECT or WITH
        first_token = cleaned_sql.split(None, 1)[0].upper() if cleaned_sql else ""
        if first_token not in ("SELECT", "WITH"):
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error=f"SQL_SYNTAX: Query must start with SELECT or WITH, got '{first_token}'.",
                error_type="SQL_SYNTAX",
            )

        # 2. Limit row count (maximum 200)
        has_limit = re.search(r"\bLIMIT\s+(\d+)", cleaned_sql, re.IGNORECASE)
        if not has_limit:
            executed_sql = f"{cleaned_sql} LIMIT 200"
        else:
            limit_val = int(has_limit.group(1))
            if limit_val > 200:
                executed_sql = re.sub(
                    r"\bLIMIT\s+\d+", "LIMIT 200", cleaned_sql, flags=re.IGNORECASE
                )
            else:
                executed_sql = cleaned_sql

        # 3. Execute query in read-only mode
        try:
            conn = open_readonly(self.db_path)
            try:
                df = pd.read_sql_query(executed_sql, conn)
            finally:
                conn.close()
        except FileNotFoundError:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error=f"Database file not found at {self.db_path}",
                error_type="RUNTIME_ERROR",
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error=f"SQL Execution Error: {e}",
                error_type="SQL_SYNTAX",
            )

        # 4. Extract referenced tables and lookup metadata
        referenced_tables = self._extract_table_names(cleaned_sql)
        metadata = self._lookup_table_metadata(referenced_tables)
        metadata["row_count"] = len(df)
        metadata["truncated"] = len(df) >= 200

        # Format output as readable CSV format
        output_csv = df.to_csv(index=False)
        return ToolResult(
            tool_name=self.name,
            success=True,
            output=output_csv,
            metadata=metadata,
        )

    def _extract_table_names(self, sql: str) -> list[str]:
        """Extract candidate table identifiers from FROM and JOIN clauses."""
        pattern = re.compile(r"\b(?:FROM|JOIN)\s+([a-zA-Z0-9_]+)", re.IGNORECASE)
        matches = pattern.findall(sql)
        # Deduplicate while preserving order
        seen = set()
        tables = []
        for m in matches:
            m_clean = m.strip()
            if m_clean.lower() not in seen and m_clean.lower() not in ("select", "where", "group", "order"):
                seen.add(m_clean.lower())
                tables.append(m_clean)
        return tables

    def _lookup_table_metadata(self, table_names: list[str]) -> dict[str, Any]:
        """Lookup units and table metadata from tables_catalog."""
        meta: dict[str, Any] = {"tables": [], "units": {}}
        if not table_names or not self.db_path.exists():
            return meta

        try:
            conn = open_readonly(self.db_path)
            cursor = conn.cursor()
            placeholders = ",".join("?" for _ in table_names)
            query = f"SELECT table_id, sql_table, unit, scope, title FROM tables_catalog WHERE table_id IN ({placeholders}) OR sql_table IN ({placeholders})"
            rows = cursor.execute(query, table_names + table_names).fetchall()
            conn.close()

            found_tables = []
            units = {}
            for t_id, s_table, unit, scope, title in rows:
                if t_id not in found_tables:
                    found_tables.append(t_id)
                units[t_id] = {"unit": unit, "scope": scope, "title": title}
            if found_tables:
                meta["tables"] = found_tables
                meta["verified_tables"] = found_tables
                meta["units"] = units
        except Exception:
            pass
        return meta
