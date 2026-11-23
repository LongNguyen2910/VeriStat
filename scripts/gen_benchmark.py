"""Generate `dev_30.jsonl`, `benchmark_400.jsonl`, and `crosscheck_sample.csv`.

Usage:
    python scripts/gen_benchmark.py \
        --db data/processed/veristat.db \
        --out-dir data/benchmark \
        --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from veristat.data.benchmark_gen import generate_benchmarks
from veristat.data.schema import BenchmarkQuestion


def _write_jsonl(path: Path, items: list[BenchmarkQuestion]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for q in items:
            f.write(json.dumps(q.to_dict(), ensure_ascii=False) + "\n")


def _write_crosscheck_csv(
    path: Path,
    test_questions: list[BenchmarkQuestion],
    sample_per_category: int = 20,
    seed: int = 42,
) -> int:
    """Write stratified 20% sample (80 rows) for 3-member manual review."""
    rng = random.Random(seed)
    by_cat: dict[str, list[BenchmarkQuestion]] = {}
    for q in test_questions:
        by_cat.setdefault(q.category, []).append(q)

    selected: list[BenchmarkQuestion] = []
    for cat in ("growth", "share", "ranking", "correlation"):
        pool = list(by_cat.get(cat, []))
        rng.shuffle(pool)
        selected.extend(pool[:sample_per_category])

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question_id",
        "category",
        "difficulty",
        "answer_type",
        "uses_admin_mapping",
        "question",
        "ground_truth",
        "unit",
        "required_table_ids",
        "reviewer_long_ok",
        "reviewer_khoa_ok",
        "reviewer_son_ok",
        "review_note",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for q in selected:
            writer.writerow({
                "question_id": q.question_id,
                "category": q.category,
                "difficulty": q.difficulty,
                "answer_type": q.answer_type,
                "uses_admin_mapping": q.uses_admin_mapping,
                "question": q.question,
                "ground_truth": json.dumps(q.ground_truth, ensure_ascii=False)
                if isinstance(q.ground_truth, list)
                else q.ground_truth,
                "unit": q.unit,
                "required_table_ids": ", ".join(q.required_table_ids),
                "reviewer_long_ok": "",
                "reviewer_khoa_ok": "",
                "reviewer_son_ok": "",
                "review_note": "",
            })
    return len(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate VeriStat dev_30 and benchmark_400.")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/processed/veristat.db"),
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/benchmark"),
        help="Output directory for JSONL and crosscheck CSV",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dev-size", type=int, default=30, help="Number of dev questions")
    parser.add_argument(
        "--test-per-category",
        type=int,
        default=100,
        help="Number of test questions per category (4 categories)",
    )
    args = parser.parse_args()

    dev_qs, test_qs = generate_benchmarks(
        db_path=args.db,
        dev_size=args.dev_size,
        test_per_category=args.test_per_category,
        seed=args.seed,
    )

    dev_path = args.out_dir / "dev_30.jsonl"
    test_path = args.out_dir / "benchmark_400.jsonl"
    csv_path = args.out_dir / "crosscheck_sample.csv"

    _write_jsonl(dev_path, dev_qs)
    _write_jsonl(test_path, test_qs)
    n_sample = _write_crosscheck_csv(csv_path, test_qs, sample_per_category=20, seed=args.seed)

    summary = {
        "dev_path": str(dev_path),
        "dev_count": len(dev_qs),
        "test_path": str(test_path),
        "test_count": len(test_qs),
        "crosscheck_csv": str(csv_path),
        "crosscheck_count": n_sample,
        "seed": args.seed,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
