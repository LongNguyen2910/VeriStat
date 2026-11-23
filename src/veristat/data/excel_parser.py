"""GSO Oct 2026 Excel (`2.-Bieu-thang-10.2026.xlsx`) sheet parser.

Extracts clean, SQL-ready DataFrames from the 5 selected sheets in
`docs/table_selection.yaml`, ignoring merged title banners, empty separator
columns, and right-hand scratch/formula columns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import openpyxl
import pandas as pd

from veristat.data.admin_mapping import classify_and_normalize_geo, normalize_unicode_vi
from veristat.data.pdf_parser import parse_gso_number


def _clean_label(val: object) -> str:
    """Normalize a Vietnamese Excel row label (collapse newlines and spaces)."""
    if val is None:
        return ""
    text = str(val).replace("\n", " ").replace("\r", " ")
    return normalize_unicode_vi(text)


def _is_footnote_row(label: str) -> bool:
    """Return True if a row label starts a footer/note block at the bottom of a sheet."""
    low = label.lower()
    return low.startswith(("(*)", "(**)", "ghi chú", "nguồn:", "nguồn :", "chú thích"))


def _extract_s05_ldcn_dp(ws: Any) -> pd.DataFrame:
    """Sheet '5. LĐCN_DP': industrial enterprise employment index by province (Oct 2026)."""
    records: list[dict[str, Any]] = []
    for r in range(7, ws.max_row + 1):
        raw_label = ws.cell(r, 1).value
        if raw_label is None:
            continue
        label = _clean_label(raw_label)
        if not label or _is_footnote_row(label):
            break
        v_mom = parse_gso_number(ws.cell(r, 2).value)
        v_yoy = parse_gso_number(ws.cell(r, 3).value)
        if v_mom is None and v_yoy is None:
            continue
        canonical, _ = classify_and_normalize_geo(label)
        records.append({
            "dia_phuong": canonical,
            "chi_so_so_voi_thang_truoc": v_mom,
            "chi_so_so_voi_nam_truoc": v_yoy,
        })
    return pd.DataFrame(
        records,
        columns=["dia_phuong", "chi_so_so_voi_thang_truoc", "chi_so_so_voi_nam_truoc"],
    )


def _extract_s06_chi_tieu_dn(ws: Any) -> pd.DataFrame:
    """Sheet '6.Chi tieu DN': national enterprise indicators for 10M 2026."""
    records: list[dict[str, Any]] = []
    for r in range(9, ws.max_row + 1):
        raw_label = ws.cell(r, 1).value
        if raw_label is None:
            continue
        label = _clean_label(raw_label)
        if not label or _is_footnote_row(label):
            break
        vals = [
            parse_gso_number(ws.cell(r, c).value)
            for c in (2, 3, 4, 5, 6, 7, 9, 10)
        ]
        if all(v is None for v in vals):
            continue
        records.append({
            "chi_tieu": label,
            "thang_9_2026": vals[0],
            "thang_10_2026": vals[1],
            "muoi_thang_2026": vals[2],
            "t10_2026_so_voi_t9_2026_pct": vals[3],
            "t10_2026_so_voi_t10_2025_pct": vals[4],
            "muoi_thang_2026_so_voi_2025_pct": vals[5],
            "thang_10_2025": vals[6],
            "muoi_thang_2025": vals[7],
        })
    return pd.DataFrame(
        records,
        columns=[
            "chi_tieu",
            "thang_9_2026",
            "thang_10_2026",
            "muoi_thang_2026",
            "t10_2026_so_voi_t9_2026_pct",
            "t10_2026_so_voi_t10_2025_pct",
            "muoi_thang_2026_so_voi_2025_pct",
            "thang_10_2025",
            "muoi_thang_2025",
        ],
    )


def _extract_s07_dn_thanh_lap(ws: Any) -> pd.DataFrame:
    """Sheet '7.DN DK thanh lap': newly registered enterprises 10M 2026 by sector/region."""
    records: list[dict[str, Any]] = []
    current_group = "Tổng số"
    for r in range(8, ws.max_row + 1):
        c1 = _clean_label(ws.cell(r, 1).value)
        c2 = _clean_label(ws.cell(r, 2).value)
        label = c1 or c2
        if not label:
            continue
        if _is_footnote_row(label):
            break
        vals = [parse_gso_number(ws.cell(r, c).value) for c in (3, 4, 5, 7, 8, 9)]
        if all(v is None for v in vals):
            if label.lower().startswith("phân theo "):
                current_group = label
            continue
        records.append({
            "phan_loai": "Tổng số" if label.upper() == "TỔNG SỐ" else current_group,
            "chi_tieu": label,
            "so_dn_10t_2026": vals[0],
            "von_dang_ky_ty_dong_10t_2026": vals[1],
            "so_lao_dong_10t_2026": vals[2],
            "so_dn_so_voi_2025_pct": vals[3],
            "von_dang_ky_so_voi_2025_pct": vals[4],
            "so_lao_dong_so_voi_2025_pct": vals[5],
        })
    return pd.DataFrame(
        records,
        columns=[
            "phan_loai",
            "chi_tieu",
            "so_dn_10t_2026",
            "von_dang_ky_ty_dong_10t_2026",
            "so_lao_dong_10t_2026",
            "so_dn_so_voi_2025_pct",
            "von_dang_ky_so_voi_2025_pct",
            "so_lao_dong_so_voi_2025_pct",
        ],
    )


def _extract_s08_s10_dn_count(ws: Any) -> pd.DataFrame:
    """Sheets '8.DN quay lai hoat dong' & '10.DN giải thể': 10M 2025 vs 10M 2026 counts."""
    records: list[dict[str, Any]] = []
    current_group = "Phân theo ngành kinh tế"
    for r in range(6, ws.max_row + 1):
        label = _clean_label(ws.cell(r, 1).value)
        if not label:
            continue
        if _is_footnote_row(label):
            break
        v_2025 = parse_gso_number(ws.cell(r, 2).value)
        v_2026 = parse_gso_number(ws.cell(r, 3).value)
        v_pct = parse_gso_number(ws.cell(r, 4).value)
        if v_2025 is None and v_2026 is None and v_pct is None:
            if label.lower().startswith("phân theo "):
                current_group = label
            continue
        records.append({
            "phan_loai": "Tổng số" if label.upper() == "TỔNG SỐ" else current_group,
            "chi_tieu": label,
            "muoi_thang_2025": v_2025,
            "muoi_thang_2026": v_2026,
            "so_voi_cung_ky_2025_pct": v_pct,
        })
    return pd.DataFrame(
        records,
        columns=[
            "phan_loai",
            "chi_tieu",
            "muoi_thang_2025",
            "muoi_thang_2026",
            "so_voi_cung_ky_2025_pct",
        ],
    )


def extract_excel_sheet(
    xlsx_path: Path,
    sheet_name: str,
    spec: dict[str, Any],
) -> pd.DataFrame:
    """Extract a clean DataFrame from a selected sheet in `2.-Bieu-thang-10.2026.xlsx`."""
    wb = openpyxl.load_workbook(str(xlsx_path), data_only=True)
    try:
        if sheet_name not in wb.sheetnames:
            raise KeyError(f"Sheet '{sheet_name}' not found in {xlsx_path}")
        ws = wb[sheet_name]
        table_id = str(spec.get("table_id", ""))
        if table_id == "B1026_S05" or sheet_name == "5. LĐCN_DP":
            return _extract_s05_ldcn_dp(ws)
        if table_id == "B1026_S06" or sheet_name == "6.Chi tieu DN":
            return _extract_s06_chi_tieu_dn(ws)
        if table_id == "B1026_S07" or sheet_name == "7.DN DK thanh lap":
            return _extract_s07_dn_thanh_lap(ws)
        if table_id in ("B1026_S08", "B1026_S10"):
            return _extract_s08_s10_dn_count(ws)
        # Generic fallback: use 3-column count layout
        return _extract_s08_s10_dn_count(ws)
    finally:
        wb.close()
