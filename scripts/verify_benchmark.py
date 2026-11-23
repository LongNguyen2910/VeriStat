"""Independently re-verify 100% of benchmark reference_code and compute Sanity Report.

Usage:
    python scripts/verify_benchmark.py \
        --db data/processed/veristat.db \
        --dev data/benchmark/dev_30.jsonl \
        --test data/benchmark/benchmark_400.jsonl \
        --sanity-out docs/benchmark_sanity_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from veristat.data.benchmark_gen import check_answer_match, execute_reference_code
from veristat.data.schema import BenchmarkQuestion, open_readonly


def _load_jsonl(path: Path) -> list[BenchmarkQuestion]:
    items: list[BenchmarkQuestion] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(BenchmarkQuestion.from_dict(json.loads(line)))
    return items


def verify_benchmarks(
    db_path: Path,
    dev_path: Path,
    test_path: Path,
) -> dict[str, Any]:
    """Verify 100% of dev & test questions against `veristat.db` in read-only mode."""
    dev_qs = _load_jsonl(dev_path)
    test_qs = _load_jsonl(test_path)
    all_qs = dev_qs + test_qs

    failures: list[str] = []

    # 1. Size & category requirements
    if len(dev_qs) != 30:
        failures.append(f"Expected dev_30 to have 30 questions, got {len(dev_qs)}")
    if len(test_qs) < 400:
        failures.append(f"Expected benchmark_400 to have >= 400 questions, got {len(test_qs)}")

    test_cat_counts = Counter(q.category for q in test_qs)
    for cat in ("growth", "share", "ranking", "correlation"):
        if test_cat_counts[cat] < 100:
            failures.append(f"Test category {cat!r} has {test_cat_counts[cat]} < 100 questions")

    # 2. Zero duplicates & zero dev/test overlap
    dev_ids = {q.question_id for q in dev_qs}
    test_ids = {q.question_id for q in test_qs}
    if len(dev_ids) != len(dev_qs):
        failures.append("Duplicate question_id detected inside dev_30")
    if len(test_ids) != len(test_qs):
        failures.append("Duplicate question_id detected inside benchmark_400")
    if dev_ids & test_ids:
        failures.append(f"Overlap in question_id between dev and test: {sorted(dev_ids & test_ids)[:5]}")

    dev_texts = {q.question.strip() for q in dev_qs}
    test_texts = {q.question.strip() for q in test_qs}
    if len(dev_texts) != len(dev_qs):
        failures.append("Duplicate question text detected inside dev_30")
    if len(test_texts) != len(test_qs):
        failures.append("Duplicate question text detected inside benchmark_400")
    if dev_texts & test_texts:
        failures.append(f"Overlap in question text between dev and test ({len(dev_texts & test_texts)} items)")

    # 3. Load catalog table_ids -> domain map
    conn = open_readonly(db_path)
    try:
        table_domain_map = {
            r[0]: r[1]
            for r in conn.execute("SELECT table_id, domain FROM tables_catalog").fetchall()
        }
    finally:
        conn.close()

    # 4. Re-execute 100% of reference_code in mode=ro & check required_table_ids
    verified_count = 0
    for q in all_qs:
        for tid in q.required_table_ids:
            if tid not in table_domain_map:
                failures.append(f"{q.question_id}: unknown required_table_id {tid!r}")
        try:
            recomputed = execute_reference_code(db_path, q.reference_code)
            matched = check_answer_match(recomputed, q.ground_truth, q.answer_type, rel_tol=1e-6)
            if not matched:
                failures.append(
                    f"{q.question_id}: ground_truth mismatch (stored={q.ground_truth!r}, recomputed={recomputed!r})"
                )
            else:
                verified_count += 1
        except Exception as exc:
            failures.append(f"{q.question_id}: reference_code execution failed: {exc}")

    # 5. Statistical Sanity Report
    def _split_stats(qs: list[BenchmarkQuestion]) -> dict[str, Any]:
        cat_diff: dict[str, dict[str, int]] = {}
        domain_counter: Counter[str] = Counter()
        num_by_cat: dict[str, list[float]] = {}

        for q in qs:
            cat_diff.setdefault(q.category, {"easy": 0, "medium": 0, "hard": 0})
            cat_diff[q.category][q.difficulty] += 1
            for tid in q.required_table_ids:
                if tid in table_domain_map:
                    domain_counter[table_domain_map[tid]] += 1
            if q.answer_type == "number":
                num_by_cat.setdefault(q.category, []).append(float(q.ground_truth))

        numeric_dist: dict[str, dict[str, float | int]] = {}
        for cat, vals in sorted(num_by_cat.items()):
            arr = np.array(vals, dtype=float)
            numeric_dist[cat] = {
                "count": int(len(arr)),
                "min": round(float(np.min(arr)), 4),
                "p05": round(float(np.percentile(arr, 5)), 4),
                "median": round(float(np.median(arr)), 4),
                "p95": round(float(np.percentile(arr, 95)), 4),
                "max": round(float(np.max(arr)), 4),
            }

        admin_cnt = sum(1 for q in qs if q.uses_admin_mapping)
        return {
            "total": len(qs),
            "by_category": dict(Counter(q.category for q in qs)),
            "by_difficulty": dict(Counter(q.difficulty for q in qs)),
            "by_answer_type": dict(Counter(q.answer_type for q in qs)),
            "category_x_difficulty": cat_diff,
            "table_reference_by_domain": dict(domain_counter),
            "uses_admin_mapping_count": admin_cnt,
            "uses_admin_mapping_pct": round(100.0 * admin_cnt / max(1, len(qs)), 2),
            "numeric_ground_truth_distribution": numeric_dist,
        }

    return {
        "status": "VERIFIED 100%" if not failures else "FAIL",
        "total_questions": len(all_qs),
        "verified_reference_code_count": verified_count,
        "dev_test_overlap_count": len(dev_texts & test_texts),
        "failures": failures,
        "dev_30": _split_stats(dev_qs),
        "benchmark_400": _split_stats(test_qs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify VeriStat benchmarks 100%.")
    parser.add_argument("--db", type=Path, default=Path("data/processed/veristat.db"))
    parser.add_argument("--dev", type=Path, default=Path("data/benchmark/dev_30.jsonl"))
    parser.add_argument("--test", type=Path, default=Path("data/benchmark/benchmark_400.jsonl"))
    parser.add_argument(
        "--sanity-out",
        type=Path,
        default=Path("docs/benchmark_sanity_report.json"),
    )
    args = parser.parse_args()

    report = verify_benchmarks(args.db, args.dev, args.test)
    args.sanity_out.parent.mkdir(parents=True, exist_ok=True)
    args.sanity_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "status": report["status"],
        "total_questions": report["total_questions"],
        "verified_reference_code_count": report["verified_reference_code_count"],
        "dev_test_overlap_count": report["dev_test_overlap_count"],
        "benchmark_400_summary": {
            "by_category": report["benchmark_400"]["by_category"],
            "by_difficulty": report["benchmark_400"]["by_difficulty"],
            "uses_admin_mapping_pct": report["benchmark_400"]["uses_admin_mapping_pct"],
            "numeric_dist": report["benchmark_400"]["numeric_ground_truth_distribution"],
        },
    }, ensure_ascii=False, indent=2))
    print(f"\n{report['status']} | Sanity report written to: {args.sanity_out}")

    if report["failures"]:
        print("FAILURES:")
        for f in report["failures"][:20]:
            print("  -", f)
        sys.exit(1)


if __name__ == "__main__":
    main()
