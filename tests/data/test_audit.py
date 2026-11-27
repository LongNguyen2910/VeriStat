import pytest
from pathlib import Path
from veristat.data.audit import (
    extract_pdf_toc,
    audit_excel_workbook,
    scan_chapter_tables,
)


@pytest.mark.skipif(
    not Path("NG-TOAN-QUOC-2025-1.pdf").exists() or not Path("2.-Bieu-thang-10.2026.xlsx").exists(),
    reason="Raw GSO 2025 PDF or Oct 2026 Excel not found in workspace root",
)
def test_audit_real_source_files():
    pdf_path = Path("NG-TOAN-QUOC-2025-1.pdf")
    xlsx_path = Path("2.-Bieu-thang-10.2026.xlsx")
    toc = extract_pdf_toc(pdf_path, cover_offset=1)
    sheets = audit_excel_workbook(xlsx_path)
    assert len(toc) > 0
    assert toc[0]["pdf_page"] == 2
    assert toc[0]["page"] == 1
    assert all(item["page"] == item["pdf_page"] - 1 for item in toc)
    assert len(sheets) == 20
    assert all("name" in s and "nrows" in s and "preview_rows" in s for s in sheets)


@pytest.mark.skipif(
    not Path("NG-TOAN-QUOC-2025-1.pdf").exists(),
    reason="Raw GSO 2025 PDF not found in workspace root",
)
def test_scan_chapter_tables_extracts_catalog():
    pdf_path = Path("NG-TOAN-QUOC-2025-1.pdf")
    edu_tables = scan_chapter_tables(pdf_path, toc_pdf_pages=[830, 831], domain="giao_duc", cover_offset=1)
    assert len(edu_tables) >= 10
    first = edu_tables[0]
    assert first["table_num"] == 328
    assert first["book_page"] == 843
    assert first["pdf_page"] == 844
    assert first["domain"] == "giao_duc"
    assert "Số người hoạt động" in first["title"]
    assert "Number of" not in first["title"]
    assert isinstance(first["has_geo"], bool)
