"""Build `data/processed/veristat.db` from GSO 2025 PDF & Oct 2026 Excel (or clean CSVs).

Usage:
    # Full digitization from raw PDF/Excel + export 31 clean CSVs for inspection/reproducibility:
    python scripts/build_db.py \
        --selection docs/table_selection.yaml \
        --out data/processed/veristat.db \
        --csv-dir data/interim/tables

    # Fast deterministic rebuild directly from the 31 exported clean CSVs (no PDF/Excel needed):
    python scripts/build_db.py \
        --selection docs/table_selection.yaml \
        --out data/processed/veristat.db \
        --from-csv data/interim/tables
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from veristat.data.admin_mapping import populate_admin_mapping
from veristat.data.etl import ingest_glossary_section, ingest_table, prepare_clean_dataframe
from veristat.data.excel_parser import extract_excel_sheet
from veristat.data.pdf_parser import extract_pdf_table
from veristat.data.schema import TableMeta, init_database

_ADMIN_MERGER_NOTE = (
    "Lưu ý thay đổi địa giới hành chính Việt Nam 2025–2026: "
    "(1) Từ 01/01/2025, tỉnh Thừa Thiên Huế trở thành thành phố trực thuộc Trung ương mang tên Huế "
    "theo Nghị quyết 175/2024/QH15 (Niên giám 2025 đã ghi tên 'Huế'). "
    "(2) Từ 01/07/2025, theo Nghị quyết 202/2025/QH15, cả nước sắp xếp từ 63 xuống 34 tỉnh/thành "
    "(11 tỉnh giữ nguyên, 23 tỉnh/thành hình thành sau hợp nhất). "
    "Các bảng từ Niên giám Thống kê 2025 dùng cơ cấu 63 tỉnh + 6 vùng + Cả nước; "
    "bảng B1026_S05 từ Biểu tháng 10/2026 dùng cơ cấu 34 tỉnh + Cả nước. "
    "Tra cứu bảng admin_mapping (cột canonical_name, canonical_2026, geo_level) để ánh xạ và tránh cộng trùng."
)


def build_database(
    selection_path: Path,
    out_db: Path,
    csv_dir: Path | None = None,
    from_csv: Path | None = None,
) -> dict[str, Any]:
    """Digitize all selected tables, optionally export/read clean CSVs, and populate SQLite."""
    cfg = yaml.safe_load(selection_path.read_text(encoding="utf-8"))
    repo_root = selection_path.resolve().parents[1]
    pdf_path = repo_root / cfg["sources"]["pdf_2025"]
    xlsx_path = repo_root / cfg["sources"]["excel_2026"]

    if csv_dir is not None:
        csv_dir.mkdir(parents=True, exist_ok=True)

    conn = init_database(out_db)
    admin_count = populate_admin_mapping(conn)
    today = date.today().isoformat()

    ingested_summary: list[dict[str, Any]] = []
    for spec in cfg["tables"]:
        table_id = str(spec["table_id"])
        sql_table = f"stat_{table_id.lower()}"
        source_key = str(spec["source"])
        source_file = cfg["sources"][source_key]

        if from_csv is not None:
            csv_file = from_csv / f"{table_id}.csv"
            if not csv_file.exists():
                raise FileNotFoundError(f"Clean CSV not found for {table_id}: {csv_file}")
            raw_df = pd.read_csv(csv_file, encoding="utf-8-sig")
        elif source_key == "pdf_2025":
            raw_df = extract_pdf_table(pdf_path, list(spec["pdf_pages"]), spec)
        elif source_key == "excel_2026":
            raw_df = extract_excel_sheet(xlsx_path, str(spec["sheet_name"]), spec)
        else:
            raise ValueError(f"Unknown source type: {source_key}")

        clean_df = prepare_clean_dataframe(raw_df)

        if csv_dir is not None and from_csv is None:
            clean_df.to_csv(
                csv_dir / f"{table_id}.csv",
                index=False,
                encoding="utf-8-sig",
            )

        meta = TableMeta(
            table_id=table_id,
            sql_table=sql_table,
            title=str(spec["title"]),
            domain=spec["domain"],
            unit=str(spec["unit"]),
            reporting_period=str(spec["reporting_period"]),
            scope=str(spec["scope"]),
            columns_json=json.dumps(list(clean_df.columns), ensure_ascii=False),
            source_ref=source_file,
            retrieved_at=today,
        )
        glossary_note = (
            f"Cơ cấu hành chính: {spec.get('admin_scheme', 'national')}. "
            + ("Có cột geo_level (country/region/province/other)." if spec.get("has_geo") else "Bảng cấp quốc gia.")
        )
        n_rows = ingest_table(conn, clean_df, meta, glossary_note=glossary_note)
        ingested_summary.append({
            "table_id": table_id,
            "sql_table": sql_table,
            "domain": spec["domain"],
            "rows": n_rows,
            "columns": list(clean_df.columns),
        })

    # Always ingest administrative boundary merger note (keeps --from-csv 100% equivalent)
    ingest_glossary_section(
        conn,
        section_id="GLOSSARY_ADMIN_MERGER_2025",
        title="Quy tắc ánh xạ địa giới hành chính 63 tỉnh (2025) sang 34 tỉnh (2026)",
        domain="dan_so",
        content=_ADMIN_MERGER_NOTE,
    )

    conn.close()
    return {
        "db_path": str(out_db),
        "admin_mapping_rows": admin_count,
        "tables_ingested": len(ingested_summary),
        "total_data_rows": sum(x["rows"] for x in ingested_summary),
        "csv_dir": str(csv_dir) if csv_dir is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build VeriStat SQLite DB from GSO sources.")
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("docs/table_selection.yaml"),
        help="Path to table_selection.yaml",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/veristat.db"),
        help="Output SQLite database path",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("data/interim/tables"),
        help="Directory to export cleaned CSV files (default: data/interim/tables)",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Disable exporting intermediate clean CSV files",
    )
    parser.add_argument(
        "--from-csv",
        type=Path,
        default=None,
        help="Rebuild DB from previously exported clean CSV directory instead of raw PDF/Excel",
    )
    args = parser.parse_args()

    summary = build_database(
        selection_path=args.selection,
        out_db=args.out,
        csv_dir=None if args.no_csv else args.csv_dir,
        from_csv=args.from_csv,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
