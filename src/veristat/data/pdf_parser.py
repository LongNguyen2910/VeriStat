"""GSO 2025 Yearbook PDF table & number parser.

Handles Vietnamese statistical number formatting and coordinate-based extraction
of borderless / multi-page tables from `NG-TOAN-QUOC-2025-1.pdf`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
import pymupdf

from veristat.data.admin_mapping import classify_and_normalize_geo, normalize_unicode_vi

_FOOTNOTE_SUFFIX_RE = re.compile(r"\s*\([*0-9a-zA-Z]+\)\s*$")
_MISSING_TOKENS: set[str] = {
    "",
    "-",
    "–",
    "—",
    "..",
    "...",
    "x",
    "X",
    "n/a",
    "N/A",
    "nan",
    "NaN",
    "Sơ bộ",
    "sơ bộ",
    "Prel.",
    "prel.",
}
_NUMERIC_TOKEN_RE = re.compile(
    r"^(?:-?\d+(?:[.,]\d+)?|\.{2,3}|[-–—])$"
)
_YEAR_TOKEN_RE = re.compile(r"^(?:19|20)\d{2}(?:\*)?$")
_SCHOOL_YEAR_RE = re.compile(r"^(?:20\d{2})-(?:20\d{2})$")


def parse_gso_number(val: object) -> float | None:
    """Parse a GSO Vietnamese/English statistical number into float or None.

    Handles:
    - Python int/float (returns None for NaN)
    - Footnote markers like "120,5 (*)" or "120,5 (1)"
    - Space thousands separator + comma decimal: "1 234,5" -> 1234.5
    - Dot thousands separator + comma decimal: "9.389,7" -> 9389.7
    - Comma decimal without thousands separator: "102345,3" -> 102345.3
    - Dot decimal (English/Excel): "45.67" -> 45.67
    - Missing / qualitative markers: "..", "...", "-", "Sơ bộ", "x", "" -> None
    """
    if val is None:
        return None
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        if pd.isna(val):
            return None
        return float(val)

    s = str(val).strip()
    if not s:
        return None

    # Strip trailing footnote markers like (*), (1), (a)
    s = _FOOTNOTE_SUFFIX_RE.sub("", s).strip()
    if s in _MISSING_TOKENS or s.lower() in _MISSING_TOKENS:
        return None

    # Normalize unicode minus/dashes and non-breaking spaces
    s = s.replace("\u00a0", " ").replace("−", "-").strip()
    # Remove spaces between digits (e.g. "1 234,5" -> "1234,5")
    s_nospace = re.sub(r"(?<=\d)\s+(?=\d)", "", s)

    # Case 1: both '.' and ',' present
    if "." in s_nospace and "," in s_nospace:
        # Vietnamese format: 9.389,7 -> dot is thousands, comma is decimal
        if s_nospace.rfind(",") > s_nospace.rfind("."):
            normalized = s_nospace.replace(".", "").replace(",", ".")
        else:
            # English format: 9,389.7 -> comma is thousands, dot is decimal
            normalized = s_nospace.replace(",", "")
    elif "," in s_nospace:
        # Comma decimal (GSO standard): 821,8 or 102345,3
        # Check if multiple commas (English thousands like 1,234,567)
        if s_nospace.count(",") > 1:
            normalized = s_nospace.replace(",", "")
        else:
            normalized = s_nospace.replace(",", ".")
    else:
        # Only '.' or digits
        if s_nospace.count(".") > 1:
            # Multiple dots = Vietnamese thousands separator (e.g. 1.234.567)
            normalized = s_nospace.replace(".", "")
        else:
            normalized = s_nospace

    try:
        return float(normalized)
    except ValueError:
        return None


def _group_words_into_lines(
    words: list[tuple[Any, ...]],
    y_tol: float = 3.5,
) -> list[dict[str, Any]]:
    """Group PyMuPDF word tuples into visual lines sorted top-to-bottom."""
    if not words:
        return []
    # Sort by vertical midpoint then x0
    sorted_words = sorted(words, key=lambda w: ((w[1] + w[3]) / 2.0, w[0]))
    lines: list[dict[str, Any]] = []
    for w in sorted_words:
        x0, y0, x1, y1, text = float(w[0]), float(w[1]), float(w[2]), float(w[3]), str(w[4])
        ymid = (y0 + y1) / 2.0
        if lines and abs(ymid - lines[-1]["y"]) <= y_tol:
            lines[-1]["words"].append((x0, x1, text))
            # Running average y
            n = len(lines[-1]["words"])
            lines[-1]["y"] = ((lines[-1]["y"] * (n - 1)) + ymid) / n
        else:
            lines.append({"y": ymid, "words": [(x0, x1, text)]})
    for ln in lines:
        ln["words"].sort(key=lambda t: t[0])
        ln["text"] = " ".join(t[2] for t in ln["words"])
    return lines


def _is_footnote_or_page_chrome(text: str, y: float, page_height: float) -> bool:
    """Filter running headers, page numbers, and bottom footnotes."""
    stripped = text.strip()
    if not stripped:
        return True
    if y < 32 or y > page_height - 55:
        return True
    if stripped.startswith(("(*)", "(**)", "Ghi chú", "Note:", "Nguồn:", "Source:")):
        return True
    if "(Tiếp theo)" in stripped or "(Cont.)" in stripped:
        return True
    if any(
        marker in stripped
        for marker in (
            "Population and Employment",
            "Enterprise, Cooperative and Non-farm",
            "Health, Culture, Sport",
            "Education",
        )
    ):
        return True
    return False


def _is_ascii_english(text: str) -> bool:
    """Return True if text contains only ASCII characters (e.g. English translation line)."""
    return bool(text) and bool(re.match(r"^[A-Za-z0-9\s,.;:()/%\-&']+$", text))


def _detect_year_columns(
    lines: list[dict[str, Any]],
) -> tuple[list[tuple[str, float]], float]:
    """Detect year column headers and their x-centers, plus the header bottom y."""
    # Pattern 1 (check first): Wrapped school years ("2020-" / "2021" + "2025-2026")
    for idx, ln in enumerate(lines):
        sy_hits: list[tuple[str, float]] = []
        for x0, x1, tok in ln["words"]:
            if _SCHOOL_YEAR_RE.match(tok) or re.match(r"^20\d{2}-$", tok):
                sy_hits.append((tok, (x0 + x1) / 2.0))
        if len(sy_hits) >= 3:
            header_y = ln["y"]
            if idx + 1 < len(lines):
                nxt = lines[idx + 1]
                if nxt["y"] - ln["y"] <= 20:
                    nxt_words = list(nxt["words"])
                    resolved: list[tuple[str, float]] = []
                    for label, xc in sy_hits:
                        if label.endswith("-"):
                            best = min(
                                nxt_words,
                                key=lambda w: abs(((w[0] + w[1]) / 2.0) - xc),
                            )
                            resolved.append((f"{label}{best[2]}", xc))
                        else:
                            resolved.append((label, xc))
                    for x0, x1, tok in nxt_words:
                        if _SCHOOL_YEAR_RE.match(tok) and tok not in {y for y, _ in resolved}:
                            resolved.append((tok, (x0 + x1) / 2.0))
                    header_y = max(header_y, nxt["y"])
                    resolved.sort(key=lambda t: t[1])
                    return resolved, header_y
            sy_hits.sort(key=lambda t: t[1])
            return sy_hits, header_y

    # Pattern 2: Standard 4-digit years (e.g. 2020 2022 2023 2024 [Prel.] 2025)
    for idx, ln in enumerate(lines):
        year_hits: list[tuple[str, float]] = []
        for x0, x1, tok in ln["words"]:
            if _YEAR_TOKEN_RE.match(tok):
                year_hits.append((tok.rstrip("*"), (x0 + x1) / 2.0))
        if len(year_hits) >= 4:
            header_y = ln["y"]
            if idx + 1 < len(lines):
                nxt = lines[idx + 1]
                if nxt["y"] - ln["y"] <= 20:
                    for x0, x1, tok in nxt["words"]:
                        if _YEAR_TOKEN_RE.match(tok) and tok.rstrip("*") not in {y for y, _ in year_hits}:
                            year_hits.append((tok.rstrip("*"), (x0 + x1) / 2.0))
                            header_y = max(header_y, nxt["y"])
            year_hits.sort(key=lambda t: t[1])
            return year_hits, header_y

    return [], 0.0


def _assign_tokens_to_columns(
    num_words: list[tuple[float, float, str]],
    col_centers: list[float],
    max_dist: float = 42.0,
) -> list[str | None]:
    """Assign numeric tokens to nearest column centers."""
    values: list[str | None] = [None] * len(col_centers)
    for x0, x1, tok in num_words:
        xc = (x0 + x1) / 2.0
        # Use right-edge alignment fallback since GSO numbers are right-aligned
        best_idx = min(range(len(col_centers)), key=lambda i: abs(xc - col_centers[i]))
        if abs(xc - col_centers[best_idx]) <= max_dist:
            values[best_idx] = tok
    return values


def _extract_geo_year_table(
    doc: pymupdf.Document,
    pages: list[int],
) -> pd.DataFrame:
    """Extract standard provincial multi-year tables into long format (dia_phuong, nam, gia_tri)."""
    records: list[dict[str, Any]] = []
    global_cols: list[tuple[str, float]] = []

    for pno in pages:
        page = doc[pno - 1]
        lines = _group_words_into_lines(page.get_text("words"))
        cols, header_y = _detect_year_columns(lines)
        if cols:
            global_cols = cols
        elif not global_cols:
            continue

        col_names = [c[0] for c in global_cols]
        col_centers = [c[1] for c in global_cols]
        split_x = col_centers[0] - 28.0

        # Collect candidate lines below header
        body_lines = [
            ln for ln in lines
            if ln["y"] > header_y + 4 and not _is_footnote_or_page_chrome(ln["text"], ln["y"], page.rect.height)
        ]

        pending_label: str | None = None
        pending_y: float = -100.0

        for ln in body_lines:
            left_words = [(x0, x1, w) for x0, x1, w in ln["words"] if x0 < split_x]
            right_words = [
                (x0, x1, w) for x0, x1, w in ln["words"]
                if x0 >= split_x and _NUMERIC_TOKEN_RE.match(w)
            ]

            left_text = " ".join(w for _, _, w in left_words).strip()

            if left_text and not right_words:
                # Label-only line (e.g., Phú Yên on y=197.5 with values on y=200.0)
                if pending_label and (ln["y"] - pending_y) <= 14.0:
                    pending_label = f"{pending_label} {left_text}"
                else:
                    pending_label = left_text
                pending_y = ln["y"]
                continue

            if right_words and not left_text:
                if pending_label and (ln["y"] - pending_y) <= 18.0:
                    label = pending_label
                    pending_label = None
                else:
                    continue
            elif left_text and right_words:
                if pending_label and (ln["y"] - pending_y) <= 16.0:
                    _, lvl_pending = classify_and_normalize_geo(pending_label)
                    _, lvl_cur = classify_and_normalize_geo(left_text)
                    if _is_ascii_english(left_text) or (lvl_pending != "other" and lvl_cur == "other"):
                        # Keep Vietnamese label from previous line; drop English line
                        label = pending_label
                    elif lvl_cur == "other":
                        label = f"{pending_label} {left_text}"
                    else:
                        label = left_text
                else:
                    label = left_text
                pending_label = None
            else:
                continue

            assigned = _assign_tokens_to_columns(right_words, col_centers)
            canonical, _ = classify_and_normalize_geo(label)
            if not canonical:
                continue
            for yr, raw_val in zip(col_names, assigned):
                records.append({
                    "dia_phuong": canonical,
                    "nam": int(yr) if yr.isdigit() else yr,
                    "gia_tri": parse_gso_number(raw_val),
                })

    return pd.DataFrame(records, columns=["dia_phuong", "nam", "gia_tri"])


def _extract_t015_pop_density(
    doc: pymupdf.Document,
    pages: list[int],
) -> pd.DataFrame:
    """Extract NG2025_T015: 2025 population and population density by province."""
    records: list[dict[str, Any]] = []
    for pno in pages:
        page = doc[pno - 1]
        lines = _group_words_into_lines(page.get_text("words"))
        pending_label: str | None = None
        pending_y: float = -100.0
        for ln in lines:
            if _is_footnote_or_page_chrome(ln["text"], ln["y"], page.rect.height):
                continue
            words = ln["words"]
            num_tokens: list[tuple[float, float, str]] = []
            label_words: list[str] = []
            for x0, x1, w in words:
                if x0 >= 230.0 and _NUMERIC_TOKEN_RE.match(w):
                    num_tokens.append((x0, x1, w))
                elif x0 < 230.0:
                    label_words.append(w)
            left_text = " ".join(label_words).strip()

            if left_text and not num_tokens:
                pending_label = left_text
                pending_y = ln["y"]
                continue

            if len(num_tokens) == 2:
                raw_label = left_text
                if pending_label and (ln["y"] - pending_y) <= 16.0:
                    if not raw_label or _is_ascii_english(raw_label):
                        raw_label = pending_label
                pending_label = None
                if not raw_label:
                    continue
                canonical, lvl = classify_and_normalize_geo(raw_label)
                if lvl in ("country", "region", "province"):
                    records.append({
                        "dia_phuong": canonical,
                        "nam": 2025,
                        "dan_so_trung_binh": parse_gso_number(num_tokens[0][2]),
                        "mat_do_dan_so": parse_gso_number(num_tokens[1][2]),
                    })
    return pd.DataFrame(
        records,
        columns=["dia_phuong", "nam", "dan_so_trung_binh", "mat_do_dan_so"],
    )


def _extract_t332_schools_by_type(
    doc: pymupdf.Document,
    pages: list[int],
) -> pd.DataFrame:
    """Extract NG2025_T332: schools & classes by school year and type (Công lập / Ngoài công lập).

    Includes both absolute counts and development index
    """
    school_years = ["2020-2021", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]
    col_meta: list[tuple[str, str]] = []
    for sy in school_years:
        col_meta.append((sy, "Công lập"))
        col_meta.append((sy, "Ngoài công lập"))

    records: list[dict[str, Any]] = []
    col_centers: list[float] = []
    index_mode = False
    section_prefix = ""

    for pno in pages:
        page = doc[pno - 1]
        lines = _group_words_into_lines(page.get_text("words"))
        pending_vi: list[str] = []
        for ln in lines:
            if _is_footnote_or_page_chrome(ln["text"], ln["y"], page.rect.height):
                continue
            if ln["y"] < 210:
                continue

            full_line = normalize_unicode_vi(ln["text"])
            if "Chỉ số phát triển" in full_line:
                index_mode = True
                section_prefix = ""
                pending_vi = []
                continue
            if full_line.startswith("Index (Previous year"):
                pending_vi = []
                continue

            left_words = [w for x0, _, w in ln["words"] if x0 < 170.0]
            right_nums = [
                (x0, x1, w)
                for x0, x1, w in ln["words"]
                if x0 >= 170.0 and _NUMERIC_TOKEN_RE.match(w)
            ]
            left_text = " ".join(left_words).strip()

            if left_text and not right_nums:
                if not _is_ascii_english(left_text):
                    pending_vi.append(left_text)
                continue

            if len(right_nums) >= 8:
                if len(right_nums) == 10 and not col_centers:
                    col_centers = [(x0 + x1) / 2.0 for x0, x1, _ in right_nums]

                if pending_vi:
                    vi_label = normalize_unicode_vi(" ".join(pending_vi))
                    pending_vi = []
                else:
                    vi_label = normalize_unicode_vi(left_text.split(" - ")[0])
                if not vi_label:
                    continue

                if vi_label.startswith(("Số trường học", "Số lớp học", "Trường học", "Lớp học")):
                    section_prefix = (
                        f"Chỉ số phát triển (%) - {vi_label}" if index_mode else vi_label
                    )
                    full_label = section_prefix
                elif section_prefix:
                    full_label = f"{section_prefix} - {vi_label}"
                else:
                    full_label = (
                        f"Chỉ số phát triển (%) - {vi_label}" if index_mode else vi_label
                    )

                if col_centers:
                    assigned = _assign_tokens_to_columns(right_nums, col_centers, max_dist=18.0)
                else:
                    assigned = [w for _, _, w in right_nums] + [None] * (10 - len(right_nums))

                for (sy, ltype), raw_v in zip(col_meta, assigned):
                    records.append({
                        "chi_tieu": full_label,
                        "nam_hoc": sy,
                        "loai_truong": ltype,
                        "gia_tri": parse_gso_number(raw_v),
                    })
    return pd.DataFrame(records, columns=["chi_tieu", "nam_hoc", "loai_truong", "gia_tri"])


def _extract_t338_permanent_classrooms(
    doc: pymupdf.Document,
    pages: list[int],
) -> pd.DataFrame:
    """Extract NG2025_T338: permanent classroom ratio by year, education level, and school type."""
    col_meta = [
        ("Tiểu học", "Chung"),
        ("Tiểu học", "Công lập"),
        ("Tiểu học", "Ngoài công lập"),
        ("Trung học cơ sở", "Chung"),
        ("Trung học cơ sở", "Công lập"),
        ("Trung học cơ sở", "Ngoài công lập"),
        ("Trung học phổ thông", "Chung"),
        ("Trung học phổ thông", "Công lập"),
        ("Trung học phổ thông", "Ngoài công lập"),
    ]
    records: list[dict[str, Any]] = []
    for pno in pages:
        page = doc[pno - 1]
        lines = _group_words_into_lines(page.get_text("words"))
        for ln in lines:
            if _is_footnote_or_page_chrome(ln["text"], ln["y"], page.rect.height):
                continue
            words = ln["words"]
            if not words:
                continue
            # Row label on the left may be "2016" or "Sơ bộ - Prel. 2025"
            left_yr: int | None = None
            yr_idx = -1
            for i, (x0, _, tok) in enumerate(words):
                if x0 < 125.0 and _YEAR_TOKEN_RE.match(tok):
                    left_yr = int(tok.rstrip("*"))
                    yr_idx = i
                    break
            if left_yr is None:
                continue
            num_toks = [w for x0, _, w in words[yr_idx + 1 :] if x0 >= 110.0 and _NUMERIC_TOKEN_RE.match(w)]
            if len(num_toks) == 9:
                for (level, ltype), raw_v in zip(col_meta, num_toks):
                    records.append({
                        "nam": left_yr,
                        "cap_hoc": level,
                        "loai_truong": ltype,
                        "gia_tri": parse_gso_number(raw_v),
                    })
    return pd.DataFrame(records, columns=["nam", "cap_hoc", "loai_truong", "gia_tri"])


def _extract_national_indicator_table(
    doc: pymupdf.Document,
    pages: list[int],
) -> pd.DataFrame:
    """Extract national multi-year indicator tables (T331, T333, T335, T336, T339, T340, T341).

    Preserves hierarchical prefixes (parent indicator & 'Chỉ số phát triển (%)')
    so sub-items like 'Tiểu học' or 'Nam' are unambiguous across sections.
    """
    records: list[dict[str, Any]] = []
    global_cols: list[tuple[str, float]] = []
    index_mode = False
    parent_group = ""
    section_ctx = ""

    for pno in pages:
        page = doc[pno - 1]
        lines = _group_words_into_lines(page.get_text("words"))
        cols, header_y = _detect_year_columns(lines)
        if cols:
            global_cols = cols
        elif not global_cols:
            continue

        col_names = [c[0] for c in global_cols]
        col_centers = [c[1] for c in global_cols]
        split_x = col_centers[0] - 25.0

        body_lines = [
            ln for ln in lines
            if ln["y"] > header_y + 4 and not _is_footnote_or_page_chrome(ln["text"], ln["y"], page.rect.height)
        ]

        pending_vi: list[tuple[float, str]] = []
        for ln in body_lines:
            full_line = normalize_unicode_vi(ln["text"])
            if "Chỉ số phát triển" in full_line and "con người" not in full_line.lower():
                index_mode = True
                parent_group = ""
                section_ctx = ""
                pending_vi = []
                continue
            if full_line.startswith("Index (Previous year"):
                pending_vi = []
                continue

            left_words = [(x0, x1, w) for x0, x1, w in ln["words"] if x0 < split_x]
            right_nums = [
                (x0, x1, w) for x0, x1, w in ln["words"]
                if x0 >= split_x and _NUMERIC_TOKEN_RE.match(w)
            ]
            left_text = " ".join(w for _, _, w in left_words).strip()
            line_x0 = left_words[0][0] if left_words else 999.0

            # Skip standalone footnote markers like "(*)" or "(**)"
            if left_text in ("(*)", "(**)", "(***)") and not right_nums:
                continue

            if left_text and not right_nums:
                # Section headers like "Phân theo giới tính - By sex"
                if "Phân theo " in left_text:
                    section_ctx = normalize_unicode_vi(left_text.split(" - ")[0])
                    pending_vi = []
                elif not _is_ascii_english(left_text):
                    pending_vi.append((line_x0, left_text))
                continue

            if right_nums:
                if pending_vi:
                    label_x0 = pending_vi[0][0]
                    label = normalize_unicode_vi(" ".join(txt for _, txt in pending_vi))
                    pending_vi = []
                elif left_text:
                    label_x0 = line_x0
                    label = normalize_unicode_vi(left_text.split(" - ")[0])
                else:
                    continue

                if not label:
                    continue

                # Determine whether this row is a top-level indicator (x0 < 50) or indented sub-item
                if label_x0 < 50.0:
                    parent_group = f"Chỉ số phát triển (%) - {label}" if index_mode else label
                    section_ctx = ""
                    full_label = parent_group
                elif section_ctx and parent_group:
                    full_label = f"{parent_group} - {section_ctx}: {label}"
                elif parent_group:
                    full_label = f"{parent_group} - {label}"
                else:
                    full_label = f"Chỉ số phát triển (%) - {label}" if index_mode else label

                assigned = _assign_tokens_to_columns(right_nums, col_centers, max_dist=45.0)
                for yr, raw_v in zip(col_names, assigned):
                    records.append({
                        "chi_tieu": full_label,
                        "nam": int(yr) if yr.isdigit() else yr,
                        "gia_tri": parse_gso_number(raw_v),
                    })

    return pd.DataFrame(records, columns=["chi_tieu", "nam", "gia_tri"])


def extract_pdf_table(
    pdf_path: Path,
    pages: list[int],
    spec: dict[str, Any],
) -> pd.DataFrame:
    """Extract a structured DataFrame from `NG-TOAN-QUOC-2025-1.pdf` for a table spec.

    Supports multi-page merging, header stripping, "(Tiếp theo)" removal,
    and Vietnamese GSO number parsing.
    """
    table_id = str(spec.get("table_id", ""))
    has_geo = bool(spec.get("has_geo", False))

    with pymupdf.open(str(pdf_path)) as doc:
        if table_id == "NG2025_T015":
            return _extract_t015_pop_density(doc, pages)
        if table_id == "NG2025_T332":
            return _extract_t332_schools_by_type(doc, pages)
        if table_id == "NG2025_T338":
            return _extract_t338_permanent_classrooms(doc, pages)
        if has_geo:
            return _extract_geo_year_table(doc, pages)
        return _extract_national_indicator_table(doc, pages)
