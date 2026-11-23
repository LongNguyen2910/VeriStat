from pathlib import Path
import pandas as pd
from veristat.data.schema import TableMeta, init_database
from veristat.data.pdf_parser import parse_gso_number
from veristat.data.etl import ingest_table, search_docs_fts


def test_parse_gso_number_handles_vietnamese_formats_and_footnotes():
    assert parse_gso_number("1 234,5") == 1234.5
    assert parse_gso_number("9.389,7") == 9389.7
    assert parse_gso_number("45.67") == 45.67
    assert parse_gso_number("120,5 (*)") == 120.5
    assert parse_gso_number("..") is None
    assert parse_gso_number("-") is None
    assert parse_gso_number("Sơ bộ") is None


def test_ingest_adds_geo_level_is_idempotent_and_fts_safe(tmp_path: Path):
    conn = init_database(tmp_path / "veristat.db")
    df = pd.DataFrame({
        "dia_phuong": ["Cả nước", "Đồng bằng sông Hồng", "TP.HCM", "Bà Rịa - Vũng Tàu"],
        "nam": [2024, 2024, 2024, 2024],
        "gia_tri": ["100 300,0", "23 500,0", "9 456,7", ".."],
    })
    meta = TableMeta(
        table_id="NG2025_02_01",
        sql_table="stat_ng2025_02_01",
        title="Dân số trung bình theo địa phương",
        domain="dan_so",
        unit="nghìn người",
        reporting_period="2020-2024",
        scope="Cả nước, 6 vùng, 63 tỉnh/thành",
        columns_json='["dia_phuong", "geo_level", "nam", "gia_tri"]',
        source_ref="NG-TOAN-QUOC-2025-1.pdf",
        retrieved_at="2026-11-22",
    )
    n1 = ingest_table(conn, df, meta, glossary_note="Đơn vị: nghìn người")
    n2 = ingest_table(conn, df, meta, glossary_note="Đơn vị: nghìn người")
    assert n1 == 4 and n2 == 4
    rows = conn.execute(
        "SELECT dia_phuong, geo_level, gia_tri FROM stat_ng2025_02_01 ORDER BY dia_phuong"
    ).fetchall()
    assert ("Cả nước", "country", 100300.0) in rows
    assert ("TP. Hồ Chí Minh", "province", 9456.7) in rows
    assert ("Bà Rịa - Vũng Tàu", "province", None) in rows

    hits = search_docs_fts(conn, 'Bà Rịa - Vũng Tàu (NG2025) * "dân số"', top_k=3)
    assert len(hits) >= 1 and hits[0]["table_id"] == "NG2025_02_01"
    conn.close()
