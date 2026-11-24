"""CLI for C.4 Error Taxonomy trace sampling and Cohen's kappa evaluation.

Usage:
    # 1. Sample 50 traces for 2-annotator labeling (Day 9):
    python scripts/prepare_c4_traces.py \
        --traces results/eval_traces.jsonl \
        --out docs/error_taxonomy_50.csv \
        --n 50 --seed 42

    # 2. Compute Cohen's kappa after both annotators finish:
    python scripts/prepare_c4_traces.py --evaluate docs/error_taxonomy_50.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from veristat.data.c4_taxonomy import evaluate_annotation_csv, sample_traces_for_c4


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare C.4 error taxonomy sample CSV or evaluate Cohen's kappa."
    )
    parser.add_argument("--traces", type=Path, help="Input JSONL file of execution traces")
    parser.add_argument("--out", type=Path, help="Output CSV path for manual annotation")
    parser.add_argument("--n", type=int, default=50, help="Number of traces to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--evaluate",
        type=Path,
        help="Path to completed annotation CSV to compute Cohen's kappa",
    )
    args = parser.parse_args()

    if args.evaluate:
        report = evaluate_annotation_csv(args.evaluate)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    if not args.traces or not args.out:
        parser.error("Provide either --evaluate <csv> or both --traces <jsonl> and --out <csv>")

    count = sample_traces_for_c4(args.traces, args.out, n=args.n, seed=args.seed)
    print(
        json.dumps(
            {
                "traces": str(args.traces),
                "out_csv": str(args.out),
                "sampled_count": count,
                "seed": args.seed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
