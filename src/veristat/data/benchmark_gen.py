"""Independent programmatic benchmark generator & answer verification for VeriStat.

Produces non-overlapping stratified QA splits (`dev_30` + `benchmark_400`) across:
- 4 analytical categories: `growth`, `share`, `ranking`, `correlation`
- 3 difficulty levels: `easy` (1 step), `medium` (2-3 steps), `hard` (>=3 steps / multi-table / `admin_mapping`)
- 5 socioeconomic domains: `dan_so`, `lao_dong`, `giao_duc`, `y_te`, `doanh_nghiep`

Every ground-truth answer is computed independently by executing `reference_code`
against `veristat.db` opened in strict read-only mode (`mode=ro`).
"""

from __future__ import annotations

import math
import random
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from veristat.data.admin_mapping import PROVINCES_63, normalize_entity, normalize_entity_list
from veristat.data.schema import (
    AnswerType,
    BenchmarkQuestion,
    DifficultyType,
    TaskCategory,
    open_readonly,
)

# Curated socioeconomic indicator pairs with genuine domain meaning for Pearson correlation.
# Format: (table_id_x, table_id_y, Vietnamese description of the pair)
VALID_CORRELATION_PAIRS: list[tuple[str, str, str]] = [
    ("NG2025_T017", "NG2025_T136", "dân số trung bình và số doanh nghiệp đang hoạt động"),
    ("NG2025_T017", "NG2025_T139", "dân số trung bình và tổng số lao động trong doanh nghiệp"),
    ("NG2025_T017", "NG2025_T151", "dân số trung bình và doanh thu thuần SXKD của doanh nghiệp"),
    ("NG2025_T020", "NG2025_T136", "dân số thành thị trung bình và số doanh nghiệp đang hoạt động"),
    ("NG2025_T020", "NG2025_T151", "dân số thành thị trung bình và doanh thu thuần SXKD"),
    ("NG2025_T061", "NG2025_T068", "tỷ lệ lao động đã qua đào tạo và tỷ lệ thất nghiệp trong độ tuổi lao động"),
    ("NG2025_T026", "NG2025_T356", "tỷ suất sinh thô và thu nhập bình quân một lao động"),
    ("NG2025_T046", "NG2025_T136", "lực lượng lao động và số doanh nghiệp đang hoạt động"),
    ("NG2025_T046", "NG2025_T139", "lực lượng lao động và số lao động trong doanh nghiệp"),
    ("NG2025_T061", "NG2025_T064", "tỷ lệ lao động đã qua đào tạo và năng suất lao động"),
    ("NG2025_T061", "NG2025_T356", "tỷ lệ lao động đã qua đào tạo và thu nhập bình quân một lao động"),
    ("NG2025_T061", "NG2025_T163", "tỷ lệ lao động đã qua đào tạo và thu nhập bình quân trong doanh nghiệp"),
    ("NG2025_T061", "NG2025_T366", "tỷ lệ lao động đã qua đào tạo và tỷ lệ hộ nghèo"),
    ("NG2025_T064", "NG2025_T356", "năng suất lao động và thu nhập bình quân một lao động"),
    ("NG2025_T064", "NG2025_T163", "năng suất lao động và thu nhập bình quân trong doanh nghiệp"),
    ("NG2025_T064", "NG2025_T366", "năng suất lao động và tỷ lệ hộ nghèo"),
    ("NG2025_T356", "NG2025_T366", "thu nhập bình quân một lao động và tỷ lệ hộ nghèo"),
    ("NG2025_T356", "NG2025_T039", "thu nhập bình quân một lao động và tuổi thọ trung bình"),
    ("NG2025_T356", "NG2025_T371", "thu nhập bình quân một lao động và tỷ lệ dùng nước hợp vệ sinh"),
    ("NG2025_T356", "NG2025_T163", "thu nhập bình quân lao động toàn nền kinh tế và trong doanh nghiệp"),
    ("NG2025_T039", "NG2025_T366", "tuổi thọ trung bình và tỷ lệ hộ nghèo"),
    ("NG2025_T039", "NG2025_T371", "tuổi thọ trung bình và tỷ lệ dân số dùng nước hợp vệ sinh"),
    ("NG2025_T136", "NG2025_T151", "số doanh nghiệp đang hoạt động và doanh thu thuần SXKD"),
    ("NG2025_T139", "NG2025_T151", "tổng số lao động trong doanh nghiệp và doanh thu thuần SXKD"),
]

# Metadata for provincial multi-year tables used in question generation.
# (table_id, short Vietnamese indicator name, unit, domain, years, is_additive)
_PROVINCIAL_TABLES: list[tuple[str, str, str, str, list[int], bool]] = [
    ("NG2025_T017", "dân số trung bình", "nghìn người", "dan_so", [2020, 2022, 2023, 2024, 2025], True),
    ("NG2025_T020", "dân số thành thị trung bình", "nghìn người", "dan_so", [2020, 2022, 2023, 2024, 2025], True),
    ("NG2025_T026", "tỷ suất sinh thô", "‰", "dan_so", [2020, 2022, 2023, 2024, 2025], False),
    ("NG2025_T034", "tỷ lệ tăng dân số", "%", "dan_so", [2020, 2022, 2023, 2024, 2025], False),
    ("NG2025_T039", "tuổi thọ trung bình", "năm", "dan_so", [2020, 2022, 2023, 2024, 2025], False),
    ("NG2025_T046", "lực lượng lao động từ 15 tuổi trở lên", "nghìn người", "lao_dong", [2020, 2022, 2023, 2024, 2025], True),
    ("NG2025_T053", "số lao động có việc làm", "nghìn người", "lao_dong", [2020, 2022, 2023, 2024, 2025], True),
    ("NG2025_T061", "tỷ lệ lao động từ 15 tuổi trở lên đã qua đào tạo", "%", "lao_dong", [2020, 2022, 2023, 2024, 2025], False),
    ("NG2025_T064", "năng suất lao động", "triệu đồng/lao động", "lao_dong", [2020, 2022, 2023, 2024, 2025], False),
    ("NG2025_T068", "tỷ lệ thất nghiệp trong độ tuổi lao động", "%", "lao_dong", [2020, 2022, 2023, 2024, 2025], False),
    ("NG2025_T356", "thu nhập bình quân một lao động đang làm việc", "nghìn đồng/tháng", "y_te", [2020, 2022, 2023, 2024, 2025], False),
    ("NG2025_T366", "tỷ lệ hộ nghèo", "%", "y_te", [2020, 2022, 2023, 2024, 2025], False),
    ("NG2025_T371", "tỷ lệ dân số sử dụng nguồn nước hợp vệ sinh", "%", "y_te", [2020, 2022, 2023, 2024, 2025], False),
    ("NG2025_T136", "số doanh nghiệp đang hoạt động có kết quả SXKD", "doanh nghiệp", "doanh_nghiep", [2020, 2021, 2022, 2023, 2024], True),
    ("NG2025_T139", "tổng số lao động trong các doanh nghiệp đang hoạt động", "người", "doanh_nghiep", [2020, 2021, 2022, 2023, 2024], True),
    ("NG2025_T151", "doanh thu thuần SXKD của các doanh nghiệp đang hoạt động", "tỷ đồng", "doanh_nghiep", [2020, 2021, 2022, 2023, 2024], True),
    ("NG2025_T163", "thu nhập bình quân một tháng của lao động trong doanh nghiệp", "nghìn đồng", "doanh_nghiep", [2020, 2021, 2022, 2023, 2024], False),
]

_PROV_TABLE_MAP = {t[0]: t for t in _PROVINCIAL_TABLES}

# Regions with >= 10 provinces (safe for region-level Pearson correlation n >= 10)
_LARGE_REGIONS = [
    "Đồng bằng sông Hồng",
    "Trung du và miền núi phía Bắc",
    "Bắc Trung Bộ và Duyên hải miền Trung",
    "Đồng bằng sông Cửu Long",
]

# Merged 2025 provinces (canonical_2026 -> constituent 63-province names)
_MERGED_GROUPS_2025: dict[str, list[str]] = {}
for _p63, (_reg, _t26) in PROVINCES_63.items():
    _MERGED_GROUPS_2025.setdefault(_t26, []).append(_p63)
_MERGED_MULTI_2025: dict[str, list[str]] = {
    k: v for k, v in _MERGED_GROUPS_2025.items() if len(v) > 1
}


def execute_reference_code(db_path: Path, reference_code: str) -> float | str | list[str]:
    """Execute independent `reference_code` against `db_path` in strict read-only mode."""
    conn = open_readonly(db_path)
    try:
        safe_globals: dict[str, Any] = {
            "__builtins__": __builtins__,
            "conn": conn,
            "math": math,
            "statistics": statistics,
            "pd": pd,
            "np": np,
        }
        local_ns: dict[str, Any] = {}
        exec(reference_code, safe_globals, local_ns)  # noqa: S102
        if "result" not in local_ns:
            raise ValueError("reference_code did not assign `result`")
        res = local_ns["result"]
        if isinstance(res, (np.floating, np.integer)):
            return float(res)
        if isinstance(res, list):
            return [str(x) for x in res]
        if isinstance(res, (int, float)):
            return float(res)
        return str(res)
    finally:
        conn.close()


def has_ambiguous_topk_tie(
    values: list[float],
    k: int,
    rel_tol: float = 1e-9,
) -> bool:
    """Return True if the k-th and (k+1)-th highest values are tied within `rel_tol`."""
    if k <= 0 or k >= len(values):
        return False
    sorted_vals = sorted((float(v) for v in values), reverse=True)
    return math.isclose(sorted_vals[k - 1], sorted_vals[k], rel_tol=rel_tol, abs_tol=1e-12)


def check_answer_match(
    pred: float | str | list[str],
    gt: float | str | list[str],
    answer_type: AnswerType,
    rel_tol: float = 1e-3,
) -> bool:
    """Standardized answer matcher shared with the agent evaluation harness (A.4).

    - `number`: relative error strictly `< rel_tol` (default `1e-3` = 0.1% per D.4);
      if `gt == 0.0`, uses `abs(pred) <= 1e-6`.
    - `entity`: canonical Vietnamese entity equality via `normalize_entity`.
    - `entity_list`: ordered canonical entity list equality via `normalize_entity_list`.
    """
    if answer_type == "number":
        try:
            p_val = float(pred)  # type: ignore[arg-type]
            g_val = float(gt)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False
        if not (math.isfinite(p_val) and math.isfinite(g_val)):
            return False
        if g_val == 0.0:
            return abs(p_val) <= 1e-6
        return (abs(p_val - g_val) / abs(g_val)) < rel_tol

    if answer_type == "entity":
        if not isinstance(pred, str) or not isinstance(gt, str):
            return False
        return normalize_entity(pred) == normalize_entity(gt)

    if answer_type == "entity_list":
        if not isinstance(pred, list) or not isinstance(gt, list):
            return False
        return normalize_entity_list([str(x) for x in pred]) == normalize_entity_list(
            [str(x) for x in gt]
        )

    raise ValueError(f"Unsupported answer_type: {answer_type!r}")


def _is_valid_number(val: Any) -> bool:
    return isinstance(val, (int, float)) and math.isfinite(float(val))


def _make_candidate(
    db_path: Path,
    question: str,
    category: TaskCategory,
    difficulty: DifficultyType,
    answer_type: AnswerType,
    unit: str,
    required_table_ids: list[str],
    reference_code: str,
    uses_admin_mapping: bool,
) -> dict[str, Any] | None:
    """Execute reference_code, filter invalid/NaN/zero-denom results, return candidate dict."""
    try:
        gt = execute_reference_code(db_path, reference_code)
    except Exception:
        return None
    if answer_type == "number":
        if not _is_valid_number(gt):
            return None
        gt = float(gt)
    elif answer_type == "entity":
        if not isinstance(gt, str) or not gt.strip():
            return None
        gt = normalize_entity(gt)
    elif answer_type == "entity_list":
        if not isinstance(gt, list) or not gt:
            return None
        gt = normalize_entity_list(gt)
    return {
        "question": question,
        "category": category,
        "difficulty": difficulty,
        "answer_type": answer_type,
        "ground_truth": gt,
        "unit": unit,
        "required_table_ids": required_table_ids,
        "reference_code": reference_code,
        "uses_admin_mapping": uses_admin_mapping,
    }


def _gen_growth_candidates(db_path: Path, rng: random.Random) -> list[dict[str, Any]]:
    """Generate diverse Vietnamese growth questions across easy, medium, and hard levels."""
    cands: list[dict[str, Any]] = []
    provinces = sorted(PROVINCES_63.keys())
    rng.shuffle(provinces)

    easy_templates = [
        "Tốc độ tăng trưởng (%) {ind} của {place} từ năm {y0} đến năm {y1} là bao nhiêu?",
        "Giai đoạn {y0}–{y1}, {ind} tại {place} tăng (hoặc giảm) bao nhiêu phần trăm?",
        "So với năm {y0}, {ind} của {place} năm {y1} thay đổi bao nhiêu %?",
        "Tính phần trăm tăng trưởng {ind} của {place} giữa hai mốc năm {y0} và {y1}.",
    ]

    # 1. EASY: single-province / national growth (1 step)
    for idx, (tid, ind, _unit, _dom, years, _add) in enumerate(_PROVINCIAL_TABLES):
        if tid in ("NG2025_T034", "NG2025_T068", "NG2025_T366"):
            # Avoid rate-of-rate growth where small denominators/zeros occur
            continue
        sql_tbl = f"stat_{tid.lower()}"
        y0, y1 = years[0], years[-1]
        y_mid = years[2]
        picked_provs = provinces[(idx * 4) % len(provinces) : (idx * 4) % len(provinces) + 4]
        if len(picked_provs) < 4:
            picked_provs = provinces[:4]
        for j, prov in enumerate(picked_provs):
            a_yr, b_yr = (y0, y1) if j % 2 == 0 else (y_mid, y1)
            q = easy_templates[(idx + j) % len(easy_templates)].format(
                ind=ind, place=prov, y0=a_yr, y1=b_yr
            )
            code = (
                f"v0 = conn.execute(\"SELECT gia_tri FROM {sql_tbl} WHERE dia_phuong={prov!r} AND geo_level='province' AND nam={a_yr}\").fetchone()[0]\n"
                f"v1 = conn.execute(\"SELECT gia_tri FROM {sql_tbl} WHERE dia_phuong={prov!r} AND geo_level='province' AND nam={b_yr}\").fetchone()[0]\n"
                f"assert v0 and v0 != 0\n"
                f"result = (v1 - v0) / v0 * 100.0"
            )
            c = _make_candidate(db_path, q, "growth", "easy", "number", "%", [tid], code, False)
            if c and abs(float(c["ground_truth"])) > 1e-4:
                cands.append(c)

    # National education & health & Oct 2026 enterprise growth (EASY)
    nat_growth_specs = [
        ("NG2025_T331", "Số trường học (Trường)", "tổng số trường học phổ thông cả nước", "2020-2021", "2025-2026"),
        ("NG2025_T331", "Số lớp học (Nghìn lớp)", "tổng số lớp học phổ thông cả nước", "2020-2021", "2025-2026"),
        ("NG2025_T333", "Số giáo viên trực tiếp giảng dạy (Nghìn người)", "số giáo viên phổ thông trực tiếp giảng dạy cả nước", "2020-2021", "2025-2026"),
        ("NG2025_T333", "Số học sinh (Nghìn học sinh)", "tổng số học sinh phổ thông cả nước", "2020-2021", "2025-2026"),
        ("NG2025_T335", "Nữ giáo viên (Nghìn người)", "số nữ giáo viên phổ thông cả nước", "2020-2021", "2025-2026"),
        ("NG2025_T335", "Nữ học sinh (Nghìn học sinh)", "số nữ học sinh phổ thông cả nước", "2020-2021", "2025-2026"),
        ("NG2025_T340", "Số giường bệnh (Nghìn giường)", "tổng số giường bệnh cả nước", 2019, 2023),
        ("NG2025_T340", "Giường bệnh bình quân 1 vạn dân (Giường)", "số giường bệnh bình quân 1 vạn dân cả nước", 2019, 2023),
        ("NG2025_T340", "Số bác sĩ (Nghìn người)", "tổng số bác sĩ cả nước", 2019, 2022),
    ]
    for tid, chi_tieu, desc, y0, y1 in nat_growth_specs:
        sql_tbl = f"stat_{tid.lower()}"
        q = f"Tốc độ tăng trưởng (%) {desc} từ kỳ {y0} đến kỳ {y1} là bao nhiêu?"
        code = (
            f"v0 = conn.execute(\"SELECT gia_tri FROM {sql_tbl} WHERE chi_tieu={chi_tieu!r} AND nam={y0!r}\").fetchone()[0]\n"
            f"v1 = conn.execute(\"SELECT gia_tri FROM {sql_tbl} WHERE chi_tieu={chi_tieu!r} AND nam={y1!r}\").fetchone()[0]\n"
            f"assert v0 and v0 != 0\n"
            f"result = (v1 - v0) / v0 * 100.0"
        )
        c = _make_candidate(db_path, q, "growth", "easy", "number", "%", [tid], code, False)
        if c:
            cands.append(c)

    # 2. MEDIUM: CAGR (%) or growth-rate gap vs Cả nước (2-3 steps)
    cagr_templates = [
        "Tính tốc độ tăng trưởng bình quân hàng năm (CAGR, %) của {ind} tại {place} giai đoạn {y0}–{y1}.",
        "Trong giai đoạn {y0}–{y1}, {ind} của {place} tăng trưởng bình quân mỗi năm (CAGR) bao nhiêu phần trăm?",
    ]
    gap_templates = [
        "Tốc độ tăng trưởng (%) {ind} giai đoạn {y0}–{y1} của {place} cao hơn (hoặc thấp hơn) mức bình quân Cả nước bao nhiêu điểm phần trăm?",
        "Chênh lệch tốc độ tăng trưởng {ind} ({y0}–{y1}) giữa {place} và Cả nước là bao nhiêu điểm phần trăm?",
    ]
    for idx, (tid, ind, _unit, _dom, years, _add) in enumerate(_PROVINCIAL_TABLES):
        if tid in ("NG2025_T034", "NG2025_T068", "NG2025_T366"):
            continue
        sql_tbl = f"stat_{tid.lower()}"
        y0, y1 = years[0], years[-1]
        span = y1 - y0
        p_cagr = provinces[(idx * 5 + 1) % len(provinces)]
        p_gap = provinces[(idx * 5 + 2) % len(provinces)]
        p_cagr2 = provinces[(idx * 5 + 3) % len(provinces)]

        for k_i, prov in enumerate((p_cagr, p_cagr2)):
            q = cagr_templates[(idx + k_i) % len(cagr_templates)].format(
                ind=ind, place=prov, y0=y0, y1=y1
            )
            code = (
                f"v0 = conn.execute(\"SELECT gia_tri FROM {sql_tbl} WHERE dia_phuong={prov!r} AND geo_level='province' AND nam={y0}\").fetchone()[0]\n"
                f"v1 = conn.execute(\"SELECT gia_tri FROM {sql_tbl} WHERE dia_phuong={prov!r} AND geo_level='province' AND nam={y1}\").fetchone()[0]\n"
                f"assert v0 > 0 and v1 > 0\n"
                f"result = ((v1 / v0) ** (1.0 / {span}) - 1.0) * 100.0"
            )
            c = _make_candidate(db_path, q, "growth", "medium", "number", "%", [tid], code, False)
            if c:
                cands.append(c)

        q_gap = gap_templates[idx % len(gap_templates)].format(
            ind=ind, place=p_gap, y0=y0, y1=y1
        )
        code_gap = (
            f"p0 = conn.execute(\"SELECT gia_tri FROM {sql_tbl} WHERE dia_phuong={p_gap!r} AND geo_level='province' AND nam={y0}\").fetchone()[0]\n"
            f"p1 = conn.execute(\"SELECT gia_tri FROM {sql_tbl} WHERE dia_phuong={p_gap!r} AND geo_level='province' AND nam={y1}\").fetchone()[0]\n"
            f"c0 = conn.execute(\"SELECT gia_tri FROM {sql_tbl} WHERE geo_level='country' AND nam={y0}\").fetchone()[0]\n"
            f"c1 = conn.execute(\"SELECT gia_tri FROM {sql_tbl} WHERE geo_level='country' AND nam={y1}\").fetchone()[0]\n"
            f"result = ((p1 - p0) / p0 - (c1 - c0) / c0) * 100.0"
        )
        c = _make_candidate(
            db_path, q_gap, "growth", "medium", "number", "điểm %", [tid], code_gap, False
        )
        if c:
            cands.append(c)

    # 3. HARD: Post-2025 merged province growth via admin_mapping + historical aliases
    merged_keys = sorted(_MERGED_MULTI_2025.keys())
    additive_tables = [t for t in _PROVINCIAL_TABLES if t[5]]
    for idx, m_prov in enumerate(merged_keys):
        tid, ind, _unit, _dom, years, _add = additive_tables[idx % len(additive_tables)]
        sql_tbl = f"stat_{tid.lower()}"
        y0, y1 = years[0], years[-1]
        parts = ", ".join(_MERGED_MULTI_2025[m_prov])
        q = (
            f"Theo cơ cấu hành chính sau sáp nhập năm 2025 (NQ 202/2025/QH15), tổng {ind} "
            f"của tỉnh/thành phố {m_prov} mới (gộp từ {parts}) tăng bao nhiêu % từ năm {y0} đến năm {y1}?"
        )
        code = (
            f"s0 = conn.execute(\"SELECT SUM(t.gia_tri) FROM {sql_tbl} t "
            f"JOIN (SELECT DISTINCT canonical_name, canonical_2026 FROM admin_mapping WHERE geo_level='province') m "
            f"ON t.dia_phuong = m.canonical_name WHERE t.geo_level='province' AND m.canonical_2026={m_prov!r} AND t.nam={y0}\").fetchone()[0]\n"
            f"s1 = conn.execute(\"SELECT SUM(t.gia_tri) FROM {sql_tbl} t "
            f"JOIN (SELECT DISTINCT canonical_name, canonical_2026 FROM admin_mapping WHERE geo_level='province') m "
            f"ON t.dia_phuong = m.canonical_name WHERE t.geo_level='province' AND m.canonical_2026={m_prov!r} AND t.nam={y1}\").fetchone()[0]\n"
            f"assert s0 and s0 != 0\n"
            f"result = (s1 - s0) / s0 * 100.0"
        )
        c = _make_candidate(db_path, q, "growth", "hard", "number", "%", [tid], code, True)
        if c:
            cands.append(c)

    # Historical/alias growth questions read directly from `admin_mapping` (HARD, uses_admin_mapping=True)
    conn = open_readonly(db_path)
    try:
        alias_rows = conn.execute(
            """
            SELECT raw_name, canonical_name, change_type
            FROM admin_mapping
            WHERE geo_level = 'province' AND raw_name != canonical_name
            ORDER BY raw_name
            """
        ).fetchall()
    finally:
        conn.close()

    for idx, (raw_name, canon, ctype) in enumerate(alias_rows):
        alias_txt = f"{raw_name} (nay thuộc {canon})" if ctype == "merged_historical" else raw_name
        for offset in (0, 5):
            tid, ind, _unit, _dom, years, _add = additive_tables[(idx + offset) % len(additive_tables)]
            sql_tbl = f"stat_{tid.lower()}"
            y0, y1 = years[0], years[-1]
            q = f"Tốc độ tăng trưởng (%) {ind} của {alias_txt} giai đoạn {y0}–{y1} là bao nhiêu?"
            code = (
                f"v0 = conn.execute(\"SELECT gia_tri FROM {sql_tbl} WHERE dia_phuong={canon!r} AND geo_level='province' AND nam={y0}\").fetchone()[0]\n"
                f"v1 = conn.execute(\"SELECT gia_tri FROM {sql_tbl} WHERE dia_phuong={canon!r} AND geo_level='province' AND nam={y1}\").fetchone()[0]\n"
                f"assert v0 and v0 != 0\n"
                f"result = (v1 - v0) / v0 * 100.0"
            )
            c = _make_candidate(db_path, q, "growth", "hard", "number", "%", [tid], code, True)
            if c:
                cands.append(c)

    return cands


def _gen_share_candidates(db_path: Path, rng: random.Random) -> list[dict[str, Any]]:
    """Generate diverse Vietnamese share/proportion questions across easy, medium, and hard."""
    cands: list[dict[str, Any]] = []
    provinces = sorted(PROVINCES_63.keys())
    rng.shuffle(provinces)
    additive_tables = [t for t in _PROVINCIAL_TABLES if t[5]]

    easy_templates = [
        "Năm {yr}, {ind} của {place} chiếm tỷ trọng bao nhiêu % so với Cả nước?",
        "Tỷ trọng (%) {ind} của {place} trong tổng số Cả nước năm {yr} là bao nhiêu?",
        "Đóng góp của {place} vào tổng {ind} toàn quốc năm {yr} đạt bao nhiêu phần trăm?",
    ]

    # 1. EASY: single province share of Cả nước
    for idx, (tid, ind, _unit, _dom, years, _add) in enumerate(additive_tables):
        sql_tbl = f"stat_{tid.lower()}"
        for j in range(6):
            prov = provinces[(idx * 7 + j) % len(provinces)]
            yr = years[-1] if j % 2 == 0 else years[0]
            q = easy_templates[(idx + j) % len(easy_templates)].format(
                ind=ind, place=prov, yr=yr
            )
            code = (
                f"p = conn.execute(\"SELECT gia_tri FROM {sql_tbl} WHERE dia_phuong={prov!r} AND geo_level='province' AND nam={yr}\").fetchone()[0]\n"
                f"c = conn.execute(\"SELECT gia_tri FROM {sql_tbl} WHERE geo_level='country' AND nam={yr}\").fetchone()[0]\n"
                f"assert c and c != 0\n"
                f"result = p / c * 100.0"
            )
            c = _make_candidate(db_path, q, "share", "easy", "number", "%", [tid], code, False)
            if c:
                cands.append(c)

    # Education public-school & female-teacher shares (EASY)
    for sy in ("2020-2021", "2023-2024", "2024-2025", "2025-2026"):
        q1 = f"Năm học {sy}, số trường phổ thông công lập chiếm bao nhiêu % tổng số trường phổ thông cả nước?"
        code1 = (
            f"pub = conn.execute(\"SELECT gia_tri FROM stat_ng2025_t332 WHERE chi_tieu='Số trường học (Trường)' AND nam_hoc={sy!r} AND loai_truong='Công lập'\").fetchone()[0]\n"
            f"prv = conn.execute(\"SELECT gia_tri FROM stat_ng2025_t332 WHERE chi_tieu='Số trường học (Trường)' AND nam_hoc={sy!r} AND loai_truong='Ngoài công lập'\").fetchone()[0]\n"
            f"result = pub / (pub + prv) * 100.0"
        )
        c1 = _make_candidate(db_path, q1, "share", "easy", "number", "%", ["NG2025_T332"], code1, False)
        if c1:
            cands.append(c1)

        q2 = f"Năm học {sy}, nữ giáo viên chiếm tỷ lệ bao nhiêu % tổng số giáo viên phổ thông trực tiếp giảng dạy cả nước?"
        code2 = (
            f"f = conn.execute(\"SELECT gia_tri FROM stat_ng2025_t335 WHERE chi_tieu='Nữ giáo viên (Nghìn người)' AND nam={sy!r}\").fetchone()[0]\n"
            f"t = conn.execute(\"SELECT gia_tri FROM stat_ng2025_t333 WHERE chi_tieu='Số giáo viên trực tiếp giảng dạy (Nghìn người)' AND nam={sy!r}\").fetchone()[0]\n"
            f"result = f / t * 100.0"
        )
        c2 = _make_candidate(
            db_path, q2, "share", "medium", "number", "%", ["NG2025_T335", "NG2025_T333"], code2, False
        )
        if c2:
            cands.append(c2)

    # 2. MEDIUM: urbanization rate (T020 / T017) & regional share of Cả nước & top-3 share
    for j in range(18):
        prov = provinces[(j * 3) % len(provinces)]
        yr = 2025 if j % 2 == 0 else 2020
        q = f"Tỷ lệ dân số thành thị (%) của {prov} năm {yr} là bao nhiêu (tính từ dân số thành thị trên tổng dân số trung bình)?"
        code = (
            f"u = conn.execute(\"SELECT gia_tri FROM stat_ng2025_t020 WHERE dia_phuong={prov!r} AND geo_level='province' AND nam={yr}\").fetchone()[0]\n"
            f"t = conn.execute(\"SELECT gia_tri FROM stat_ng2025_t017 WHERE dia_phuong={prov!r} AND geo_level='province' AND nam={yr}\").fetchone()[0]\n"
            f"assert t and t != 0\n"
            f"result = u / t * 100.0"
        )
        c = _make_candidate(
            db_path, q, "share", "medium", "number", "%", ["NG2025_T020", "NG2025_T017"], code, False
        )
        if c:
            cands.append(c)

    # Top-3 provinces combined share of Cả nước (MEDIUM)
    for idx, (tid, ind, _unit, _dom, years, _add) in enumerate(additive_tables):
        sql_tbl = f"stat_{tid.lower()}"
        for yr in (years[0], years[2], years[-1]):
            q = f"Năm {yr}, tổng {ind} của 3 tỉnh/thành phố dẫn đầu cả nước chiếm bao nhiêu % so với Cả nước?"
            code = (
                f"top3 = [r[0] for r in conn.execute(\"SELECT gia_tri FROM {sql_tbl} WHERE geo_level='province' AND nam={yr} ORDER BY gia_tri DESC LIMIT 3\").fetchall()]\n"
                f"c = conn.execute(\"SELECT gia_tri FROM {sql_tbl} WHERE geo_level='country' AND nam={yr}\").fetchone()[0]\n"
                f"result = sum(top3) / c * 100.0"
            )
            c = _make_candidate(db_path, q, "share", "medium", "number", "%", [tid], code, False)
            if c:
                cands.append(c)

    # Enterprise share of total employment (T139 / (T053 * 1000)) — cross-table + unit conversion (HARD)
    for j in range(10):
        prov = provinces[(j * 5 + 2) % len(provinces)]
        yr = 2024 if j % 2 == 0 else 2022
        q = (
            f"Năm {yr}, lao động trong các doanh nghiệp đang hoạt động của {prov} "
            f"chiếm bao nhiêu % tổng số lao động có việc làm của tỉnh (lưu ý quy đổi đơn vị nghìn người và người)?"
        )
        code = (
            f"ent = conn.execute(\"SELECT gia_tri FROM stat_ng2025_t139 WHERE dia_phuong={prov!r} AND geo_level='province' AND nam={yr}\").fetchone()[0]\n"
            f"tot = conn.execute(\"SELECT gia_tri FROM stat_ng2025_t053 WHERE dia_phuong={prov!r} AND geo_level='province' AND nam={yr}\").fetchone()[0]\n"
            f"assert tot and tot != 0\n"
            f"result = ent / (tot * 1000.0) * 100.0"
        )
        c = _make_candidate(
            db_path, q, "share", "hard", "number", "%", ["NG2025_T139", "NG2025_T053"], code, False
        )
        if c:
            cands.append(c)

    # 3. HARD: Post-2025 merged province share of Cả nước via admin_mapping
    merged_keys = sorted(_MERGED_MULTI_2025.keys())
    for idx, m_prov in enumerate(merged_keys):
        for offset in (0, 3):
            tid, ind, _unit, _dom, years, _add = additive_tables[(idx + offset) % len(additive_tables)]
            sql_tbl = f"stat_{tid.lower()}"
            yr = years[-1]
            parts = ", ".join(_MERGED_MULTI_2025[m_prov])
            q = (
                f"Nếu gộp theo địa giới 34 tỉnh sau sáp nhập năm 2025, tỉnh/thành phố {m_prov} mới "
                f"(gồm {parts}) chiếm bao nhiêu % tổng {ind} của Cả nước năm {yr}?"
            )
            code = (
                f"s = conn.execute(\"SELECT SUM(t.gia_tri) FROM {sql_tbl} t "
                f"JOIN (SELECT DISTINCT canonical_name, canonical_2026 FROM admin_mapping WHERE geo_level='province') m "
                f"ON t.dia_phuong = m.canonical_name WHERE t.geo_level='province' AND m.canonical_2026={m_prov!r} AND t.nam={yr}\").fetchone()[0]\n"
                f"c = conn.execute(\"SELECT gia_tri FROM {sql_tbl} WHERE geo_level='country' AND nam={yr}\").fetchone()[0]\n"
                f"assert c and c != 0\n"
                f"result = s / c * 100.0"
            )
            c = _make_candidate(db_path, q, "share", "hard", "number", "%", [tid], code, True)
            if c:
                cands.append(c)

    return cands


def _gen_ranking_candidates(db_path: Path, rng: random.Random) -> list[dict[str, Any]]:
    """Generate tie-free ranking questions (entity & ordered entity_list) across easy/medium/hard."""
    cands: list[dict[str, Any]] = []
    conn = open_readonly(db_path)

    try:
        # 1. EASY: Top-1 highest or lowest province (entity)
        max_templates = [
            "Năm {yr}, tỉnh/thành phố nào có {ind} cao nhất cả nước?",
            "Địa phương cấp tỉnh nào đứng đầu cả nước về {ind} trong năm {yr}?",
        ]
        min_templates = [
            "Năm {yr}, tỉnh/thành phố nào có {ind} thấp nhất cả nước?",
            "Địa phương cấp tỉnh nào ghi nhận mức {ind} thấp nhất trong năm {yr}?",
        ]
        for idx, (tid, ind, _unit, _dom, years, _add) in enumerate(_PROVINCIAL_TABLES):
            sql_tbl = f"stat_{tid.lower()}"
            for j, yr in enumerate((years[0], years[2], years[-1])):
                rows = conn.execute(
                    f"SELECT dia_phuong, gia_tri FROM {sql_tbl} WHERE geo_level='province' AND nam=? AND gia_tri IS NOT NULL",
                    (yr,),
                ).fetchall()
                vals = [float(r[1]) for r in rows]
                if len(vals) < 10:
                    continue

                if not has_ambiguous_topk_tie(vals, k=1):
                    q_max = max_templates[(idx + j) % len(max_templates)].format(yr=yr, ind=ind)
                    code_max = (
                        f"result = conn.execute(\"SELECT dia_phuong FROM {sql_tbl} "
                        f"WHERE geo_level='province' AND nam={yr} AND gia_tri IS NOT NULL "
                        f"ORDER BY gia_tri DESC LIMIT 1\").fetchone()[0]"
                    )
                    c = _make_candidate(db_path, q_max, "ranking", "easy", "entity", "địa phương", [tid], code_max, False)
                    if c:
                        cands.append(c)

                neg_vals = [-v for v in vals]
                if not has_ambiguous_topk_tie(neg_vals, k=1):
                    q_min = min_templates[(idx + j) % len(min_templates)].format(yr=yr, ind=ind)
                    code_min = (
                        f"result = conn.execute(\"SELECT dia_phuong FROM {sql_tbl} "
                        f"WHERE geo_level='province' AND nam={yr} AND gia_tri IS NOT NULL "
                        f"ORDER BY gia_tri ASC LIMIT 1\").fetchone()[0]"
                    )
                    c = _make_candidate(db_path, q_min, "ranking", "easy", "entity", "địa phương", [tid], code_min, False)
                    if c:
                        cands.append(c)

        # Oct 2026 Excel (34 provinces) ranking (EASY & MEDIUM)
        for col, col_desc in (
            ("chi_so_so_voi_thang_truoc", "chỉ số sử dụng lao động công nghiệp so với tháng trước"),
            ("chi_so_so_voi_nam_truoc", "chỉ số sử dụng lao động công nghiệp so với cùng kỳ năm trước"),
        ):
            rows = conn.execute(
                f"SELECT dia_phuong, {col} FROM stat_b1026_s05 WHERE geo_level='province' AND {col} IS NOT NULL"
            ).fetchall()
            vals = [float(r[1]) for r in rows]
            if not has_ambiguous_topk_tie(vals, k=3):
                q = f"Tháng 10/2026 (theo cơ cấu 34 tỉnh sau sáp nhập), liệt kê theo thứ tự giảm dần 3 tỉnh/thành phố có {col_desc} cao nhất."
                code = (
                    f"result = [r[0] for r in conn.execute(\"SELECT dia_phuong FROM stat_b1026_s05 "
                    f"WHERE geo_level='province' AND {col} IS NOT NULL ORDER BY {col} DESC LIMIT 3\").fetchall()]"
                )
                c = _make_candidate(db_path, q, "ranking", "medium", "entity_list", "danh sách địa phương", ["B1026_S05"], code, False)
                if c:
                    cands.append(c)

        # 2. MEDIUM: Top-3 / Top-5 provinces nationwide or within a region (entity_list)
        for idx, (tid, ind, _unit, _dom, years, _add) in enumerate(_PROVINCIAL_TABLES):
            sql_tbl = f"stat_{tid.lower()}"
            for k_val, yr in ((3, years[-1]), (5, years[-1]), (3, years[0])):
                rows = conn.execute(
                    f"SELECT dia_phuong, gia_tri FROM {sql_tbl} WHERE geo_level='province' AND nam=? AND gia_tri IS NOT NULL",
                    (yr,),
                ).fetchall()
                vals = [float(r[1]) for r in rows]
                if not has_ambiguous_topk_tie(vals, k=k_val):
                    q = f"Liệt kê theo thứ tự giảm dần top {k_val} tỉnh/thành phố có {ind} cao nhất cả nước năm {yr}."
                    code = (
                        f"result = [r[0] for r in conn.execute(\"SELECT dia_phuong FROM {sql_tbl} "
                        f"WHERE geo_level='province' AND nam={yr} AND gia_tri IS NOT NULL "
                        f"ORDER BY gia_tri DESC LIMIT {k_val}\").fetchall()]"
                    )
                    c = _make_candidate(db_path, q, "ranking", "medium", "entity_list", "danh sách địa phương", [tid], code, False)
                    if c:
                        cands.append(c)

            # Region-scoped Top-3 (uses admin_mapping.region_name)
            reg = _LARGE_REGIONS[idx % len(_LARGE_REGIONS)]
            yr = years[-1]
            rows_reg = conn.execute(
                f"SELECT t.dia_phuong, t.gia_tri FROM {sql_tbl} t "
                f"JOIN (SELECT DISTINCT canonical_name, region_name FROM admin_mapping WHERE geo_level='province') m "
                f"ON t.dia_phuong = m.canonical_name "
                f"WHERE t.geo_level='province' AND m.region_name=? AND t.nam=? AND t.gia_tri IS NOT NULL",
                (reg, yr),
            ).fetchall()
            vals_reg = [float(r[1]) for r in rows_reg]
            if len(vals_reg) >= 5 and not has_ambiguous_topk_tie(vals_reg, k=3):
                q_reg = f"Trong vùng {reg} năm {yr}, nêu theo thứ tự giảm dần 3 tỉnh/thành phố dẫn đầu về {ind}."
                code_reg = (
                    f"result = [r[0] for r in conn.execute(\"SELECT t.dia_phuong FROM {sql_tbl} t "
                    f"JOIN (SELECT DISTINCT canonical_name, region_name FROM admin_mapping WHERE geo_level='province') m "
                    f"ON t.dia_phuong = m.canonical_name "
                    f"WHERE t.geo_level='province' AND m.region_name={reg!r} AND t.nam={yr} AND t.gia_tri IS NOT NULL "
                    f"ORDER BY t.gia_tri DESC LIMIT 3\").fetchall()]"
                )
                c = _make_candidate(db_path, q_reg, "ranking", "medium", "entity_list", "danh sách địa phương", [tid], code_reg, True)
                if c:
                    cands.append(c)

        # 3. HARD: Top-3 by growth rate, Top-3 by post-2025 merged 34-province totals, Top-3 by urbanization
        for tid, ind, _unit, _dom, years, is_add in _PROVINCIAL_TABLES:
            sql_tbl = f"stat_{tid.lower()}"
            y0, y1 = years[0], years[-1]
            if tid not in ("NG2025_T034", "NG2025_T068", "NG2025_T366"):
                g_rows = conn.execute(
                    f"SELECT a.dia_phuong, (b.gia_tri - a.gia_tri) / a.gia_tri * 100.0 AS g "
                    f"FROM {sql_tbl} a JOIN {sql_tbl} b ON a.dia_phuong = b.dia_phuong "
                    f"WHERE a.geo_level='province' AND b.geo_level='province' AND a.nam=? AND b.nam=? "
                    f"AND a.gia_tri > 0 AND b.gia_tri IS NOT NULL",
                    (y0, y1),
                ).fetchall()
                g_vals = [float(r[1]) for r in g_rows]
                if len(g_vals) >= 10 and not has_ambiguous_topk_tie(g_vals, k=3):
                    q_g = f"Liệt kê theo thứ tự giảm dần 3 tỉnh/thành phố có tốc độ tăng trưởng (%) {ind} cao nhất giai đoạn {y0}–{y1}."
                    code_g = (
                        f"result = [r[0] for r in conn.execute("
                        f"\"SELECT a.dia_phuong FROM {sql_tbl} a JOIN {sql_tbl} b ON a.dia_phuong = b.dia_phuong "
                        f"WHERE a.geo_level='province' AND b.geo_level='province' AND a.nam={y0} AND b.nam={y1} "
                        f"AND a.gia_tri > 0 AND b.gia_tri IS NOT NULL "
                        f"ORDER BY (b.gia_tri - a.gia_tri) / a.gia_tri DESC LIMIT 3\").fetchall()]"
                    )
                    c = _make_candidate(db_path, q_g, "ranking", "hard", "entity_list", "danh sách địa phương", [tid], code_g, False)
                    if c:
                        cands.append(c)

            if is_add:
                for yr in (years[0], years[-1]):
                    m_rows = conn.execute(
                        f"SELECT m.canonical_2026, SUM(t.gia_tri) AS s "
                        f"FROM {sql_tbl} t "
                        f"JOIN (SELECT DISTINCT canonical_name, canonical_2026 FROM admin_mapping WHERE geo_level='province') m "
                        f"ON t.dia_phuong = m.canonical_name "
                        f"WHERE t.geo_level='province' AND t.nam=? GROUP BY m.canonical_2026",
                        (yr,),
                    ).fetchall()
                    m_vals = [float(r[1]) for r in m_rows]
                    if len(m_vals) == 34 and not has_ambiguous_topk_tie(m_vals, k=3):
                        q_m = (
                            f"Nếu tổng hợp số liệu năm {yr} theo cơ cấu 34 tỉnh/thành phố mới sau sáp nhập năm 2025, "
                            f"nêu theo thứ tự giảm dần 3 địa phương có tổng {ind} lớn nhất."
                        )
                        code_m = (
                            f"result = [r[0] for r in conn.execute("
                            f"\"SELECT m.canonical_2026 FROM {sql_tbl} t "
                            f"JOIN (SELECT DISTINCT canonical_name, canonical_2026 FROM admin_mapping WHERE geo_level='province') m "
                            f"ON t.dia_phuong = m.canonical_name "
                            f"WHERE t.geo_level='province' AND t.nam={yr} GROUP BY m.canonical_2026 "
                            f"ORDER BY SUM(t.gia_tri) DESC LIMIT 3\").fetchall()]"
                        )
                        c = _make_candidate(db_path, q_m, "ranking", "hard", "entity_list", "danh sách địa phương", [tid], code_m, True)
                        if c:
                            cands.append(c)

        # Multi-table derived ranking: Top-3 urbanization rate (T020 / T017)
        for yr in (2020, 2023, 2024, 2025):
            u_rows = conn.execute(
                "SELECT a.dia_phuong, a.gia_tri / b.gia_tri * 100.0 "
                "FROM stat_ng2025_t020 a JOIN stat_ng2025_t017 b ON a.dia_phuong = b.dia_phuong AND a.nam = b.nam "
                "WHERE a.geo_level='province' AND b.geo_level='province' AND a.nam=? AND b.gia_tri > 0",
                (yr,),
            ).fetchall()
            u_vals = [float(r[1]) for r in u_rows]
            if len(u_vals) >= 10 and not has_ambiguous_topk_tie(u_vals, k=3):
                q_u = f"Năm {yr}, liệt kê theo thứ tự giảm dần 3 tỉnh/thành phố có tỷ lệ dân số thành thị (dân số thành thị / dân số trung bình) cao nhất cả nước."
                code_u = (
                    f"result = [r[0] for r in conn.execute("
                    f"\"SELECT a.dia_phuong FROM stat_ng2025_t020 a JOIN stat_ng2025_t017 b "
                    f"ON a.dia_phuong = b.dia_phuong AND a.nam = b.nam "
                    f"WHERE a.geo_level='province' AND b.geo_level='province' AND a.nam={yr} AND b.gia_tri > 0 "
                    f"ORDER BY a.gia_tri / b.gia_tri DESC LIMIT 3\").fetchall()]"
                )
                c = _make_candidate(
                    db_path, q_u, "ranking", "hard", "entity_list", "danh sách địa phương",
                    ["NG2025_T020", "NG2025_T017"], code_u, False,
                )
                if c:
                    cands.append(c)
    finally:
        conn.close()

    return cands


def _gen_correlation_candidates(db_path: Path, rng: random.Random) -> list[dict[str, Any]]:
    """Generate meaningful Pearson correlation questions from `VALID_CORRELATION_PAIRS` (n >= 10)."""
    cands: list[dict[str, Any]] = []
    easy_templates = [
        "Tính hệ số tương quan Pearson giữa {desc} trên 63 tỉnh/thành phố năm {yr}.",
        "Năm {yr}, hệ số tương quan tuyến tính Pearson giữa {desc} (tính trên 63 tỉnh/thành phố) bằng bao nhiêu?",
        "Xác định hệ số tương quan Pearson r giữa {desc} cấp tỉnh trong năm {yr}.",
    ]

    for idx, (tid_x, tid_y, desc) in enumerate(VALID_CORRELATION_PAIRS):
        meta_x = _PROV_TABLE_MAP[tid_x]
        meta_y = _PROV_TABLE_MAP[tid_y]
        sql_x = f"stat_{tid_x.lower()}"
        sql_y = f"stat_{tid_y.lower()}"
        common_years = sorted(set(meta_x[4]) & set(meta_y[4]))
        if not common_years:
            continue

        # 1. EASY: single-year Pearson r across all 63 provinces
        for j, yr in enumerate((common_years[0], common_years[-1])):
            q = easy_templates[(idx + j) % len(easy_templates)].format(desc=desc, yr=yr)
            code = (
                f"rows = conn.execute(\"SELECT x.gia_tri, y.gia_tri FROM {sql_x} x JOIN {sql_y} y "
                f"ON x.dia_phuong = y.dia_phuong AND x.nam = y.nam "
                f"WHERE x.geo_level='province' AND y.geo_level='province' AND x.nam={yr} "
                f"AND x.gia_tri IS NOT NULL AND y.gia_tri IS NOT NULL\").fetchall()\n"
                f"assert len(rows) >= 10\n"
                f"xs = [r[0] for r in rows]; ys = [r[1] for r in rows]\n"
                f"result = float(np.corrcoef(xs, ys)[0, 1])"
            )
            c = _make_candidate(db_path, q, "correlation", "easy", "number", "hệ số tương quan", [tid_x, tid_y], code, False)
            if c and abs(float(c["ground_truth"])) > 1e-4:
                cands.append(c)

        # 2. MEDIUM: region-scoped Pearson r (n >= 10 provinces) & multi-year mean Pearson r
        reg = _LARGE_REGIONS[idx % len(_LARGE_REGIONS)]
        yr_last = common_years[-1]
        q_reg = (
            f"Riêng tại vùng {reg} năm {yr_last}, hệ số tương quan Pearson giữa {desc} là bao nhiêu?"
        )
        code_reg = (
            f"rows = conn.execute(\"SELECT x.gia_tri, y.gia_tri FROM {sql_x} x JOIN {sql_y} y "
            f"ON x.dia_phuong = y.dia_phuong AND x.nam = y.nam "
            f"JOIN (SELECT DISTINCT canonical_name, region_name FROM admin_mapping WHERE geo_level='province') m "
            f"ON x.dia_phuong = m.canonical_name "
            f"WHERE x.geo_level='province' AND y.geo_level='province' AND m.region_name={reg!r} "
            f"AND x.nam={yr_last} AND x.gia_tri IS NOT NULL AND y.gia_tri IS NOT NULL\").fetchall()\n"
            f"assert len(rows) >= 10\n"
            f"xs = [r[0] for r in rows]; ys = [r[1] for r in rows]\n"
            f"result = float(np.corrcoef(xs, ys)[0, 1])"
        )
        c_reg = _make_candidate(db_path, q_reg, "correlation", "medium", "number", "hệ số tương quan", [tid_x, tid_y], code_reg, True)
        if c_reg and abs(float(c_reg["ground_truth"])) > 1e-4:
            cands.append(c_reg)

        y0, y1 = common_years[0], common_years[-1]
        q_avg = (
            f"Tính hệ số tương quan Pearson giữa giá trị trung bình giai đoạn {y0}–{y1} "
            f"của {desc} trên 63 tỉnh/thành phố."
        )
        code_avg = (
            f"rows = conn.execute(\"SELECT ax.mx, ay.my FROM "
            f"(SELECT dia_phuong, AVG(gia_tri) AS mx FROM {sql_x} WHERE geo_level='province' AND nam IN ({','.join(map(str, common_years))}) GROUP BY dia_phuong) ax "
            f"JOIN (SELECT dia_phuong, AVG(gia_tri) AS my FROM {sql_y} WHERE geo_level='province' AND nam IN ({','.join(map(str, common_years))}) GROUP BY dia_phuong) ay "
            f"ON ax.dia_phuong = ay.dia_phuong WHERE ax.mx IS NOT NULL AND ay.my IS NOT NULL\").fetchall()\n"
            f"assert len(rows) >= 10\n"
            f"xs = [r[0] for r in rows]; ys = [r[1] for r in rows]\n"
            f"result = float(np.corrcoef(xs, ys)[0, 1])"
        )
        c_avg = _make_candidate(db_path, q_avg, "correlation", "medium", "number", "hệ số tương quan", [tid_x, tid_y], code_avg, False)
        if c_avg and abs(float(c_avg["ground_truth"])) > 1e-4:
            cands.append(c_avg)

        # 3. HARD: growth-rate correlation (2020->last) & 34-province merged correlation (for additive pairs)
        if tid_x != "NG2025_T366" and tid_y != "NG2025_T366":
            q_gr = (
                f"Tính hệ số tương quan Pearson trên 63 tỉnh/thành phố giữa tốc độ tăng trưởng (%) "
                f"giai đoạn {y0}–{y1} của {desc}."
            )
            code_gr = (
                f"rows = conn.execute("
                f"\"SELECT (x1.gia_tri - x0.gia_tri)/x0.gia_tri, (y1.gia_tri - y0.gia_tri)/y0.gia_tri "
                f"FROM {sql_x} x0 JOIN {sql_x} x1 ON x0.dia_phuong=x1.dia_phuong "
                f"JOIN {sql_y} y0 ON x0.dia_phuong=y0.dia_phuong "
                f"JOIN {sql_y} y1 ON x0.dia_phuong=y1.dia_phuong "
                f"WHERE x0.geo_level='province' AND x1.geo_level='province' AND y0.geo_level='province' AND y1.geo_level='province' "
                f"AND x0.nam={y0} AND x1.nam={y1} AND y0.nam={y0} AND y1.nam={y1} "
                f"AND x0.gia_tri > 0 AND y0.gia_tri > 0\").fetchall()\n"
                f"assert len(rows) >= 10\n"
                f"xs = [r[0] for r in rows]; ys = [r[1] for r in rows]\n"
                f"result = float(np.corrcoef(xs, ys)[0, 1])"
            )
            c_gr = _make_candidate(db_path, q_gr, "correlation", "hard", "number", "hệ số tương quan", [tid_x, tid_y], code_gr, False)
            if c_gr and abs(float(c_gr["ground_truth"])) > 1e-4:
                cands.append(c_gr)

        if meta_x[5] and meta_y[5]:
            q_34 = (
                f"Sau khi gộp số liệu năm {yr_last} theo cơ cấu 34 tỉnh/thành phố mới (NQ 202/2025/QH15), "
                f"hệ số tương quan Pearson giữa tổng {desc} bằng bao nhiêu?"
            )
            code_34 = (
                f"rows = conn.execute("
                f"\"SELECT ax.sx, ay.sy FROM "
                f"(SELECT m.canonical_2026 AS p, SUM(t.gia_tri) AS sx FROM {sql_x} t "
                f"JOIN (SELECT DISTINCT canonical_name, canonical_2026 FROM admin_mapping WHERE geo_level='province') m ON t.dia_phuong=m.canonical_name "
                f"WHERE t.geo_level='province' AND t.nam={yr_last} GROUP BY m.canonical_2026) ax "
                f"JOIN (SELECT m.canonical_2026 AS p, SUM(t.gia_tri) AS sy FROM {sql_y} t "
                f"JOIN (SELECT DISTINCT canonical_name, canonical_2026 FROM admin_mapping WHERE geo_level='province') m ON t.dia_phuong=m.canonical_name "
                f"WHERE t.geo_level='province' AND t.nam={yr_last} GROUP BY m.canonical_2026) ay "
                f"ON ax.p = ay.p\").fetchall()\n"
                f"assert len(rows) == 34\n"
                f"xs = [r[0] for r in rows]; ys = [r[1] for r in rows]\n"
                f"result = float(np.corrcoef(xs, ys)[0, 1])"
            )
            c_34 = _make_candidate(db_path, q_34, "correlation", "hard", "number", "hệ số tương quan", [tid_x, tid_y], code_34, True)
            if c_34 and abs(float(c_34["ground_truth"])) > 1e-4:
                cands.append(c_34)

    return cands


def _dedup_and_stratified_sample(
    candidates: list[dict[str, Any]],
    total_needed: int,
    difficulty_ratios: dict[DifficultyType, float],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Deduplicate by question text and sample with target difficulty proportions."""
    seen: set[str] = set()
    by_diff: dict[DifficultyType, list[dict[str, Any]]] = {"easy": [], "medium": [], "hard": []}
    for c in candidates:
        q = c["question"].strip()
        if q in seen:
            continue
        seen.add(q)
        by_diff[c["difficulty"]].append(c)

    for lst in by_diff.values():
        rng.shuffle(lst)

    targets: dict[DifficultyType, int] = {
        "easy": round(total_needed * difficulty_ratios["easy"]),
        "medium": round(total_needed * difficulty_ratios["medium"]),
        "hard": total_needed
        - round(total_needed * difficulty_ratios["easy"])
        - round(total_needed * difficulty_ratios["medium"]),
    }

    selected: list[dict[str, Any]] = []
    leftover: list[dict[str, Any]] = []
    for diff in ("easy", "medium", "hard"):
        take = min(targets[diff], len(by_diff[diff]))
        selected.extend(by_diff[diff][:take])
        leftover.extend(by_diff[diff][take:])

    if len(selected) < total_needed:
        rng.shuffle(leftover)
        selected.extend(leftover[: total_needed - len(selected)])

    if len(selected) < total_needed:
        raise RuntimeError(
            f"Not enough valid candidates: needed {total_needed}, got {len(selected)}"
        )
    return selected[:total_needed]


def generate_benchmarks(
    db_path: Path,
    dev_size: int = 30,
    test_per_category: int = 100,
    seed: int = 42,
) -> tuple[list[BenchmarkQuestion], list[BenchmarkQuestion]]:
    """Generate non-overlapping `dev` and `test` BenchmarkQuestion lists deterministically."""
    rng = random.Random(seed)
    diff_ratios: dict[DifficultyType, float] = {"easy": 0.30, "medium": 0.40, "hard": 0.30}

    # Distribute dev_size across 4 categories (e.g. 30 -> 8, 8, 7, 7)
    base_dev, rem_dev = divmod(dev_size, 4)
    categories: list[TaskCategory] = ["growth", "share", "ranking", "correlation"]
    dev_per_cat = {cat: base_dev + (1 if i < rem_dev else 0) for i, cat in enumerate(categories)}

    generators = {
        "growth": _gen_growth_candidates,
        "share": _gen_share_candidates,
        "ranking": _gen_ranking_candidates,
        "correlation": _gen_correlation_candidates,
    }

    dev_questions: list[BenchmarkQuestion] = []
    test_questions: list[BenchmarkQuestion] = []
    dev_idx = 1
    test_idx = 1

    for cat in categories:
        raw_cands = generators[cat](db_path, rng)
        n_dev = dev_per_cat[cat]
        n_total = n_dev + test_per_category
        picked = _dedup_and_stratified_sample(raw_cands, n_total, diff_ratios, rng)
        rng.shuffle(picked)

        # Stratified split into dev and test by difficulty
        dev_slice = picked[:n_dev]
        test_slice = picked[n_dev:]

        for item in dev_slice:
            dev_questions.append(
                BenchmarkQuestion(
                    question_id=f"DEV_{cat.upper()[:3]}_{dev_idx:03d}",
                    question=item["question"],
                    split="dev",
                    category=cat,
                    difficulty=item["difficulty"],
                    answer_type=item["answer_type"],
                    ground_truth=item["ground_truth"],
                    unit=item["unit"],
                    required_table_ids=item["required_table_ids"],
                    reference_code=item["reference_code"],
                    uses_admin_mapping=item["uses_admin_mapping"],
                )
            )
            dev_idx += 1

        for item in test_slice:
            test_questions.append(
                BenchmarkQuestion(
                    question_id=f"TEST_{cat.upper()[:3]}_{test_idx:03d}",
                    question=item["question"],
                    split="test",
                    category=cat,
                    difficulty=item["difficulty"],
                    answer_type=item["answer_type"],
                    ground_truth=item["ground_truth"],
                    unit=item["unit"],
                    required_table_ids=item["required_table_ids"],
                    reference_code=item["reference_code"],
                    uses_admin_mapping=item["uses_admin_mapping"],
                )
            )
            test_idx += 1

    return dev_questions, test_questions
