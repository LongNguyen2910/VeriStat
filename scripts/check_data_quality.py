"""Data Quality Checks and Audit Report Generator for `data/processed/veristat.db`.

Runs automated assertions across all digitized tables:
1. All 5 required socioeconomic domains covered (`dan_so`, `lao_dong`, `giao_duc`, `y_te`, `doanh_nghiep`).
2. Per-table row counts, column lists, year coverage, and null rates.
3. Zero unexpected `geo_level = 'other'` entities in provincial tables
   (only GSO's explicit 'Không xác định - Nec.' row in enterprise tables is allowed).
4. Additive reconciliation: sum of provinces (+ 'Không xác định - Nec.' where applicable)
   vs published 'Cả nước' row must differ by < 1.0% across all years.
5. Read-only mode (`open_readonly`) and FTS5 search sanity check (`search_docs_fts`).
6. Deterministic random spot-check sample (10 cells) for manual cross-checking against
   the original PDF/Excel sources.

Usage:
    python scripts/check_data_quality.py \
        --db data/processed/veristat.db \
        --report docs/data_quality_report.json
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from pathlib import Path
from typing import Any

from veristat.data.etl import search_docs_fts
from veristat.data.schema import open_readonly

REQUIRED_DOMAINS = {"dan_so", "lao_dong", "giao_duc", "y_te", "doanh_nghiep"}
ALLOWED_OTHER_ENTITIES = {"Không xác định - Nec."}

# Additive provincial tables -> (value_col, include_nec_other)
ADDITIVE_TABLES: dict[str, tuple[str, bool]] = {
    "NG2025_T015": ("dan_so_trung_binh", False),
    "NG2025_T017": ("gia_tri", False),
    "NG2025_T020": ("gia_tri", False),
    "NG2025_T046": ("gia_tri", False),
    "NG2025_T053": ("gia_tri", False),
    "NG2025_T136": ("gia_tri", True),
    "NG2025_T139": ("gia_tri", True),
    "NG2025_T151": ("gia_tri", True),
}


def run_quality_checks(db_path: Path, max_rel_diff: float = 0.01) -> dict[str, Any]:
    """Run all Data Quality Checks on `db_path` in read-only mode."""
    conn = open_readonly(db_path)
    failures: list[str] = []

    try:
        catalog_rows = conn.execute(
            """
            SELECT table_id, sql_table, title, domain, unit, reporting_period,
                   scope, columns_json, source_ref
            FROM tables_catalog
            ORDER BY table_id
            """
        ).fetchall()

        # 1. Domain coverage
        domain_counts: dict[str, int] = {}
        for r in catalog_rows:
            domain_counts[r[3]] = domain_counts.get(r[3], 0) + 1
        missing_domains = sorted(REQUIRED_DOMAINS - set(domain_counts.keys()))
        if missing_domains:
            failures.append(f"Missing required domains: {missing_domains}")

        # 2. Per-table stats + geo_level check
        table_reports: list[dict[str, Any]] = []
        unexpected_others: list[dict[str, Any]] = []
        spot_pool: list[dict[str, Any]] = []

        for (
            table_id,
            sql_table,
            title,
            domain,
            unit,
            reporting_period,
            scope,
            columns_json,
            source_ref,
        ) in catalog_rows:
            cols = json.loads(columns_json)
            row_count = int(conn.execute(f"SELECT COUNT(*) FROM {sql_table}").fetchone()[0])
            if row_count == 0:
                failures.append(f"Table {table_id} ({sql_table}) has 0 rows")

            # Compute null counts across all columns
            total_cells = row_count * max(1, len(cols))
            null_cells = 0
            col_nulls: dict[str, int] = {}
            for col in cols:
                n_null = int(
                    conn.execute(f"SELECT COUNT(*) FROM {sql_table} WHERE {col} IS NULL").fetchone()[0]
                )
                null_cells += n_null
                if n_null > 0:
                    col_nulls[col] = n_null
            null_rate_pct = round(100.0 * null_cells / max(1, total_cells), 2)

            # Year coverage if applicable
            years: list[Any] = []
            if "nam" in cols:
                years = [
                    r[0]
                    for r in conn.execute(
                        f"SELECT DISTINCT nam FROM {sql_table} ORDER BY nam"
                    ).fetchall()
                ]
            elif "nam_hoc" in cols:
                years = [
                    r[0]
                    for r in conn.execute(
                        f"SELECT DISTINCT nam_hoc FROM {sql_table} ORDER BY nam_hoc"
                    ).fetchall()
                ]

            # Geo-level breakdown & unexpected 'other' assertion
            geo_breakdown: dict[str, int] | None = None
            if "geo_level" in cols and "dia_phuong" in cols:
                geo_rows = conn.execute(
                    f"SELECT geo_level, COUNT(DISTINCT dia_phuong) FROM {sql_table} GROUP BY geo_level"
                ).fetchall()
                geo_breakdown = {str(g): int(cnt) for g, cnt in geo_rows}
                other_names = [
                    r[0]
                    for r in conn.execute(
                        f"SELECT DISTINCT dia_phuong FROM {sql_table} WHERE geo_level = 'other'"
                    ).fetchall()
                ]
                bad_others = [n for n in other_names if n not in ALLOWED_OTHER_ENTITIES]
                if bad_others:
                    unexpected_others.append({"table_id": table_id, "entities": bad_others})
                    failures.append(f"Unexpected geo_level='other' in {table_id}: {bad_others}")

            table_reports.append({
                "table_id": table_id,
                "sql_table": sql_table,
                "domain": domain,
                "row_count": row_count,
                "null_rate_pct": null_rate_pct,
                "col_nulls": col_nulls,
                "years": years,
                "geo_unique_counts": geo_breakdown,
            })

            # Collect candidate cells for deterministic manual spot-check
            if "dia_phuong" in cols and "gia_tri" in cols and "nam" in cols:
                sample_rows = conn.execute(
                    f"""
                    SELECT dia_phuong, geo_level, nam, gia_tri
                    FROM {sql_table}
                    WHERE gia_tri IS NOT NULL AND geo_level IN ('country', 'province')
                    ORDER BY dia_phuong, nam
                    """
                ).fetchall()
                for dp, gl, yr, val in sample_rows:
                    spot_pool.append({
                        "table_id": table_id,
                        "title": title,
                        "source_ref": source_ref,
                        "entity": dp,
                        "geo_level": gl,
                        "period": yr,
                        "value": val,
                        "unit": unit,
                    })

        # 3. Additive reconciliation: sum(provinces [+ Nec.]) vs 'Cả nước'
        additive_checks: list[dict[str, Any]] = []
        for table_id, (val_col, include_nec) in ADDITIVE_TABLES.items():
            sql_table = f"stat_{table_id.lower()}"
            years = [
                r[0]
                for r in conn.execute(
                    f"SELECT DISTINCT nam FROM {sql_table} ORDER BY nam"
                ).fetchall()
            ]
            for yr in years:
                country_row = conn.execute(
                    f"SELECT {val_col} FROM {sql_table} WHERE geo_level = 'country' AND nam = ?",
                    (yr,),
                ).fetchone()
                if include_nec:
                    sum_row = conn.execute(
                        f"""
                        SELECT SUM({val_col}) FROM {sql_table}
                        WHERE nam = ? AND (geo_level = 'province' OR dia_phuong = 'Không xác định - Nec.')
                        """,
                        (yr,),
                    ).fetchone()
                else:
                    sum_row = conn.execute(
                        f"SELECT SUM({val_col}) FROM {sql_table} WHERE geo_level = 'province' AND nam = ?",
                        (yr,),
                    ).fetchone()
                c_val = float(country_row[0]) if country_row and country_row[0] is not None else None
                s_val = float(sum_row[0]) if sum_row and sum_row[0] is not None else None
                if c_val is None or s_val is None or c_val == 0:
                    failures.append(f"Additive check missing value for {table_id} year={yr}")
                    continue
                rel_diff = abs(s_val - c_val) / abs(c_val)
                passed = rel_diff < max_rel_diff
                if not passed:
                    failures.append(
                        f"Additive mismatch {table_id} ({yr}): sum={s_val:.2f} vs country={c_val:.2f} "
                        f"(rel_diff={rel_diff * 100:.3f}% >= {max_rel_diff * 100:.1f}%)"
                    )
                additive_checks.append({
                    "table_id": table_id,
                    "year": yr,
                    "country_value": round(c_val, 3),
                    "sum_provinces": round(s_val, 3),
                    "rel_diff_pct": round(rel_diff * 100.0, 4),
                    "passed": passed,
                })

        # 4. FTS5 & admin_mapping sanity check
        admin_rows = int(conn.execute("SELECT COUNT(*) FROM admin_mapping").fetchone()[0])
        if admin_rows < 63:
            failures.append(f"admin_mapping has too few rows: {admin_rows}")

        fts_queries = [
            "dân số trung bình",
            'Bà Rịa - Vũng Tàu (NG2025) * "doanh nghiệp"',
            "sáp nhập 34 tỉnh Nghị quyết 202",
        ]
        fts_results: dict[str, int] = {}
        for q in fts_queries:
            hits = search_docs_fts(conn, q, top_k=5)
            fts_results[q] = len(hits)
            if len(hits) == 0:
                failures.append(f"FTS5 query returned 0 hits: {q!r}")

        # 5. Deterministic 10-cell spot-check sample across distinct tables
        rng = random.Random(42)
        rng.shuffle(spot_pool)
        spot_samples: list[dict[str, Any]] = []
        used_tables: set[str] = set()
        for item in spot_pool:
            if item["table_id"] in used_tables:
                continue
            spot_samples.append(item)
            used_tables.add(item["table_id"])
            if len(spot_samples) == 10:
                break

        return {
            "db_path": str(db_path),
            "status": "PASS" if not failures else "FAIL",
            "failures": failures,
            "summary": {
                "total_tables": len(table_reports),
                "domain_counts": domain_counts,
                "admin_mapping_rows": admin_rows,
                "unexpected_other_count": len(unexpected_others),
                "additive_checks_total": len(additive_checks),
                "additive_checks_passed": sum(1 for c in additive_checks if c["passed"]),
                "max_additive_rel_diff_pct": max(
                    (c["rel_diff_pct"] for c in additive_checks), default=0.0
                ),
            },
            "fts_sanity": fts_results,
            "additive_reconciliation": additive_checks,
            "table_reports": table_reports,
            "manual_spot_check_samples": spot_samples,
        }
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VeriStat Data Quality Checks.")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/processed/veristat.db"),
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/data_quality_report.json"),
        help="Output JSON report path",
    )
    args = parser.parse_args()

    report = run_quality_checks(args.db)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"\nStatus: {report['status']} | Report written to: {args.report}")
    if report["failures"]:
        print("FAILURES:")
        for f in report["failures"]:
            print("  -", f)
        sys.exit(1)


if __name__ == "__main__":
    main()
