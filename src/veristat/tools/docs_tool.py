"""SearchDocs tool: FTS5 BM25 search over statistical glossary and tables."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

from veristat.data.schema import open_readonly
from veristat.tools.base import BaseTool, ToolResult


class SearchDocsTool(BaseTool):
    """Searches statistical table catalog and glossary documentation using SQLite FTS5."""

    name = "search_docs"
    description = (
        "Tìm kiếm định nghĩa chỉ tiêu, ngữ cảnh bảng số liệu và ghi chú địa giới hành chính "
        "bằng chỉ mục toàn văn FTS5 (BM25)."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Từ khóa hoặc câu hỏi cần tra cứu tài liệu.",
            },
            "top_k": {
                "type": "integer",
                "default": 5,
                "description": "Số lượng kết quả cần trả về (1 đến 10).",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, db_path: Path | str = "data/processed/veristat.db") -> None:
        self.db_path = Path(db_path)

    def execute(self, **kwargs: Any) -> ToolResult:
        query_str = str(kwargs.get("query", "")).strip()
        top_k = int(kwargs.get("top_k", 5))
        top_k = max(1, min(top_k, 10))

        if not query_str:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error="Search query cannot be empty.",
                error_type="RUNTIME_ERROR",
            )

        if not self.db_path.exists():
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error=f"Database file not found at {self.db_path}",
                error_type="RUNTIME_ERROR",
            )

        # Sanitize query for FTS5 (remove special FTS5 operators to prevent syntax errors)
        sanitized = re.sub(r'[^\w\s\u00C0-\u1EF9]', ' ', query_str)
        tokens = [t.strip() for t in sanitized.split() if len(t.strip()) > 1]
        if not tokens:
            tokens = [t.strip() for t in sanitized.split() if t.strip()]

        # Construct safe match expression: "token1" OR "token2"
        fts_match_expr = " OR ".join(f'"{t}"' for t in tokens) if tokens else f'"{query_str}"'

        try:
            conn = open_readonly(self.db_path)
            cursor = conn.cursor()
            sql = """
                SELECT table_id, title, domain, content, rank
                FROM docs_glossary
                WHERE docs_glossary MATCH ?
                ORDER BY rank
                LIMIT ?
            """
            rows = cursor.execute(sql, (fts_match_expr, top_k)).fetchall()

            # If no direct FTS match, fallback to LIKE on title and content
            if not rows and tokens:
                like_clauses = []
                params = []
                for t in tokens[:3]:
                    like_clauses.append("(title LIKE ? OR content LIKE ?)")
                    params.extend([f"%{t}%", f"%{t}%"])
                where_like = " OR ".join(like_clauses)
                fallback_sql = f"""
                    SELECT table_id, title, domain, content, 0.0 as rank
                    FROM docs_glossary
                    WHERE {where_like}
                    LIMIT ?
                """
                params.append(top_k)
                rows = cursor.execute(fallback_sql, params).fetchall()

            conn.close()
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error=f"FTS Search Error: {e}",
                error_type="RUNTIME_ERROR",
            )

        if not rows:
            return ToolResult(
                tool_name=self.name,
                success=True,
                output=f"Không tìm thấy tài liệu phù hợp cho từ khóa: '{query_str}'.",
                metadata={"results_count": 0},
            )

        formatted_items = []
        candidate_tables = []
        for idx, (t_id, title, domain, content, rank) in enumerate(rows, 1):
            candidate_tables.append(t_id)
            snippet = content[:300] + "..." if len(content) > 300 else content
            formatted_items.append(
                f"### {idx}. Bảng [{t_id}] - {title} (Lĩnh vực: {domain})\n"
                f"- **Nội dung:** {snippet}\n"
            )

        output_md = "\n".join(formatted_items)
        return ToolResult(
            tool_name=self.name,
            success=True,
            output=output_md,
            metadata={"results_count": len(rows), "candidate_tables": candidate_tables},
        )
