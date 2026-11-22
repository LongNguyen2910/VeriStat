"""Phase 1 EDA & Data Audit utilities for GSO PDF Yearbooks and Excel reports."""

from __future__ import annotations

from collections import Counter
from datetime import date
import json
import re
from pathlib import Path
from typing import Any

import openpyxl
import pymupdf
import yaml


_TOC_LINE_RE = re.compile(
    r"^\s*(?:PHẦN|CHƯƠNG|BIỂU|\d+(?:\.\d+)*\.?)\s+.+?\s+(\d{1,4})\s*$",
    re.IGNORECASE,
)


def extract_pdf_toc(
    pdf_path: Path,
    cover_offset: int = 1,
    fallback_scan_pages: int = 45,
) -> list[dict[str, Any]]:
    """Extract table of contents from a GSO Statistical Yearbook PDF.

    Uses embedded PDF bookmarks first; falls back to scanning early TOC pages
    when bookmarks are absent. Returns both printed book ``page`` (after
    subtracting ``cover_offset``) and 1-based physical ``pdf_page``.
    """
    pdf_path = Path(pdf_path)
    doc = pymupdf.open(pdf_path)
    try:
        raw_toc = doc.get_toc(simple=True)
        if raw_toc:
            return [
                {
                    "level": int(level),
                    "title": str(title).strip(),
                    "page": int(page) - cover_offset,
                    "pdf_page": int(page),
                }
                for level, title, page in raw_toc
            ]

        entries: list[dict[str, Any]] = []
        limit = min(fallback_scan_pages, len(doc))
        for page_idx in range(limit):
            text = doc[page_idx].get_text("text") or ""
            for line in text.splitlines():
                cleaned = " ".join(line.split())
                if not cleaned or len(cleaned) < 6:
                    continue
                m = _TOC_LINE_RE.match(cleaned)
                if m:
                    book_page = int(m.group(1))
                    pdf_page = book_page + cover_offset
                    if 1 <= pdf_page <= len(doc):
                        level = 1 if cleaned.upper().startswith(("PHẦN", "CHƯƠNG")) else 2
                        entries.append(
                            {
                                "level": level,
                                "title": cleaned,
                                "page": book_page,
                                "pdf_page": pdf_page,
                                "toc_page": page_idx + 1,
                            }
                        )
        return entries
    finally:
        doc.close()


def audit_excel_workbook(xlsx_path: Path, max_preview_rows: int = 5) -> list[dict[str, Any]]:
    """Audit all sheets in a GSO Excel report: dimensions and top non-empty rows."""
    xlsx_path = Path(xlsx_path)
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    results: list[dict[str, Any]] = []
    try:
        for name in wb.sheetnames:
            ws = wb[name]
            preview_rows: list[list[Any]] = []
            nrows = 0
            ncols = 0
            for row in ws.iter_rows(values_only=True):
                nrows += 1
                if row:
                    ncols = max(ncols, len(row))
                if len(preview_rows) < max_preview_rows and any(
                    cell is not None and str(cell).strip() != "" for cell in row
                ):
                    preview_rows.append(
                        [
                            str(cell).strip() if cell is not None else ""
                            for cell in row[:10]
                        ]
                    )
            results.append(
                {
                    "name": name,
                    "nrows": nrows,
                    "ncols": ncols,
                    "preview_rows": preview_rows,
                }
            )
    finally:
        wb.close()
    return results


_INLINE_ENTRY_RE = re.compile(r"^(\d{1,3})\s+([^\d].+)$")
_INT_LINE_RE = re.compile(r"^\d{1,4}$")
_SKIP_EXACT = {"biểu", "table", "trang", "page"}


_VI_DIACRITIC_RE = re.compile(
    r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
    r"ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]"
)


def _extract_vietnamese_title(lines: list[str]) -> str:
    """Keep only the Vietnamese title lines and strip inline ' - English' suffixes."""
    vi_lines: list[str] = []
    for line in lines:
        has_vi = bool(_VI_DIACRITIC_RE.search(line))
        if vi_lines and not has_vi:
            break
        if " - " in line:
            left, right = line.split(" - ", 1)
            if _VI_DIACRITIC_RE.search(left) and not _VI_DIACRITIC_RE.search(right):
                vi_lines.append(left.strip())
                break
        vi_lines.append(line.strip())
    return " ".join(vi_lines).strip()


def _is_running_header(line: str) -> bool:
    low = line.lower()
    if low in _SKIP_EXACT:
        return True
    # Running chapter headers have Vietnamese - English with a book page number at start or end
    if " - " in line and (re.match(r"^\d{1,4}\s+", line) or re.search(r"\s+\d{1,4}$", line)):
        return True
    return False


def scan_chapter_tables(
    pdf_path: Path,
    toc_pdf_pages: list[int],
    domain: str,
    cover_offset: int = 1,
) -> list[dict[str, Any]]:
    """Auto-scan chapter TOC pages in a GSO Statistical Yearbook PDF into a structured catalog."""
    pdf_path = Path(pdf_path)
    doc = pymupdf.open(pdf_path)
    entries: list[dict[str, Any]] = []
    try:
        total_pages = len(doc)
        lines: list[str] = []
        for pdf_p in toc_pdf_pages:
            if not (1 <= pdf_p <= total_pages):
                continue
            text = doc[pdf_p - 1].get_text("text") or ""
            for raw in text.splitlines():
                cleaned = " ".join(raw.split())
                if not cleaned or _is_running_header(cleaned):
                    continue
                lines.append(cleaned)

        cur_num: int | None = None
        cur_lines: list[str] = []

        for line in lines:
            if cur_num is None:
                m_inline = _INLINE_ENTRY_RE.match(line)
                if m_inline:
                    cur_num = int(m_inline.group(1))
                    cur_lines = [m_inline.group(2).strip()]
                elif _INT_LINE_RE.match(line):
                    val = int(line)
                    if 1 <= val <= 500:
                        cur_num = val
                        cur_lines = []
            else:
                if _INT_LINE_RE.match(line) and cur_lines:
                    book_page = int(line)
                    pdf_page = book_page + cover_offset
                    if 1 <= pdf_page <= total_pages:
                        vi_title = _extract_vietnamese_title(cur_lines)
                        low = vi_title.lower()
                        entries.append(
                            {
                                "table_num": cur_num,
                                "title": vi_title,
                                "book_page": book_page,
                                "pdf_page": pdf_page,
                                "domain": domain,
                                "has_geo": any(
                                    k in low
                                    for k in (
                                        "phân theo địa phương",
                                        "phân theo vùng",
                                    )
                                ),
                            }
                        )
                        cur_num = None
                        cur_lines = []
                        continue
                # Also handle inline page number at the end of a single-line bilingual entry
                m_end_page = re.match(r"^(.+?)\s+(\d{2,4})$", line)
                if m_end_page and not _INLINE_ENTRY_RE.match(line) and cur_lines:
                    book_page = int(m_end_page.group(2))
                    pdf_page = book_page + cover_offset
                    if book_page >= 40 and 1 <= pdf_page <= total_pages:
                        cur_lines.append(m_end_page.group(1).strip())
                        vi_title = _extract_vietnamese_title(cur_lines)
                        low = vi_title.lower()
                        entries.append(
                            {
                                "table_num": cur_num,
                                "title": vi_title,
                                "book_page": book_page,
                                "pdf_page": pdf_page,
                                "domain": domain,
                                "has_geo": any(
                                    k in low
                                    for k in (
                                        "phân theo địa phương",
                                        "phân theo vùng",
                                    )
                                ),
                            }
                        )
                        cur_num = None
                        cur_lines = []
                        continue
                cur_lines.append(line)
    finally:
        doc.close()
    return entries


DEFAULT_CHAPTER_TOC_SPECS: list[tuple[list[int], str]] = [
    ([62, 63, 64, 65, 66], "dan_so_lao_dong"),
    ([346, 347, 348, 349, 350], "doanh_nghiep"),
    ([830, 831], "giao_duc"),
    ([858, 859, 860, 861, 862], "y_te_msdc"),
]


def generate_data_audit(
    pdf_path: Path,
    xlsx_path: Path,
    selection_yaml: Path,
    out_json: Path,
    cover_offset: int = 1,
) -> dict[str, Any]:
    """Run full Phase 1 EDA audit on the PDF Yearbook and Excel report and write JSON."""
    pdf_path = Path(pdf_path)
    xlsx_path = Path(xlsx_path)
    selection_yaml = Path(selection_yaml)
    out_json = Path(out_json)

    doc = pymupdf.open(pdf_path)
    pdf_pages = len(doc)
    doc.close()

    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb["5. LĐCN_DP"]
    excel_34_raw = [
        str(r[0]).strip()
        for r in list(ws.iter_rows(values_only=True))[7:]
        if r and r[0] is not None and str(r[0]).strip()
    ]
    wb.close()

    sel = yaml.safe_load(selection_yaml.read_text(encoding="utf-8"))
    scanned: list[dict[str, Any]] = []
    chapter_counts: dict[str, dict[str, int]] = {}
    for pages, dom in DEFAULT_CHAPTER_TOC_SPECS:
        rows = scan_chapter_tables(pdf_path, toc_pdf_pages=pages, domain=dom, cover_offset=cover_offset)
        chapter_counts[dom] = {
            "total": len(rows),
            "with_geo": sum(1 for r in rows if r["has_geo"]),
        }
        scanned.extend(rows)

    audit: dict[str, Any] = {
        "audit_date": str(date.today()),
        "sources": {
            "pdf_2025": {
                "file": pdf_path.name,
                "pages": pdf_pages,
                "cover_offset": cover_offset,
                "toc": extract_pdf_toc(pdf_path, cover_offset=cover_offset),
            },
            "excel_2026": {
                "file": xlsx_path.name,
                "sheets_count": len(audit_excel_workbook(xlsx_path)),
                "sheets": [s["name"] for s in audit_excel_workbook(xlsx_path)],
            },
        },
        "curated_summary": {
            "total_tables": len(sel["tables"]),
            "by_domain": dict(Counter(t["domain"] for t in sel["tables"])),
            "by_source": dict(Counter(t["source"] for t in sel["tables"])),
        },
        "findings": {
            "pdf_number_format": "Vietnamese decimal comma without thousands separator (e.g. 102345,3), footnotes like (*)",
            "missing_markers": ["..", "...", "-", "–", "x", "Sơ bộ"],
            "geo_structure_pdf_2025": "CẢ NƯỚC - WHOLE COUNTRY + 6 socio-economic regions + 63 provinces (Thừa Thiên Huế already named Huế)",
            "geo_structure_excel_2026": f"CẢ NƯỚC + {len(excel_34_raw) - 1} provinces after 2025 Vietnam provincial merger",
            "excel_2026_geo_entities": excel_34_raw,
            "glossary_sections_available": len(sel["glossary_sections"]),
        },
        "auto_scanned_pdf_catalog": {
            "total_tables_scanned": len(scanned),
            "by_chapter": chapter_counts,
            "tables": scanned,
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit
