"""Unit tests for QueryTableTool and SearchDocsTool."""

import pytest
from veristat.tools.docs_tool import SearchDocsTool
from veristat.tools.table_tool import QueryTableTool


def test_query_table_success():
    tool = QueryTableTool(db_path="data/processed/veristat.db")
    res = tool.execute(sql="SELECT * FROM tables_catalog LIMIT 5")
    assert res.success is True
    assert "table_id" in res.output
    assert res.metadata.get("row_count") == 5


def test_query_table_blocks_ddl_and_dml():
    tool = QueryTableTool(db_path="data/processed/veristat.db")
    # Test DROP
    res_drop = tool.execute(sql="DROP TABLE tables_catalog")
    assert res_drop.success is False
    assert res_drop.error_type == "SECURITY_VIOLATION"

    # Test DELETE
    res_del = tool.execute(sql="DELETE FROM tables_catalog WHERE 1=1")
    assert res_del.success is False
    assert res_del.error_type == "SECURITY_VIOLATION"

    # Test UPDATE
    res_update = tool.execute(sql="UPDATE tables_catalog SET title = 'hacked'")
    assert res_update.success is False
    assert res_update.error_type == "SECURITY_VIOLATION"


def test_query_table_enforces_limit_200():
    tool = QueryTableTool(db_path="data/processed/veristat.db")
    res = tool.execute(sql="SELECT * FROM admin_mapping LIMIT 500")
    assert res.success is True
    assert res.metadata.get("row_count") <= 200


def test_search_docs_finds_relevant_tables():
    tool = SearchDocsTool(db_path="data/processed/veristat.db")
    res = tool.execute(query="dân số", top_k=3)
    assert res.success is True
    assert res.metadata["results_count"] > 0
    assert "Bảng [" in res.output


def test_search_docs_handles_special_characters():
    tool = SearchDocsTool(db_path="data/processed/veristat.db")
    # Punctuation or operator-heavy queries shouldn't crash SQLite FTS
    res = tool.execute(query="dân số! (2024)? AND * 'test'", top_k=2)
    assert res.success is True
