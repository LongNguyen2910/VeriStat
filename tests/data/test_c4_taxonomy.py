"""Unit tests for C.4 Error Taxonomy sampling and Cohen's kappa."""

import json
import math
from pathlib import Path

import pandas as pd

from veristat.data.c4_taxonomy import compute_cohens_kappa, sample_traces_for_c4


def test_sample_traces_for_c4_and_cohens_kappa(tmp_path: Path):
    traces_file = tmp_path / "traces.jsonl"
    with traces_file.open("w", encoding="utf-8") as f:
        for i in range(80):
            f.write(
                json.dumps(
                    {
                        "question_id": f"Q{i:03d}",
                        "method": "react" if i % 2 == 0 else "mcts",
                        "is_correct": i % 4 == 0,
                        "predicted_answer": 1.0,
                        "ground_truth": 2.0,
                        "trace": "Thought: ...",
                    }
                )
                + "\n"
            )
    out_csv = tmp_path / "error_taxonomy_50.csv"
    assert sample_traces_for_c4(traces_file, out_csv, n=50, seed=42) == 50
    df = pd.read_csv(out_csv)
    assert {
        "question_id",
        "method",
        "is_correct",
        "annotator_1",
        "annotator_2",
        "final_label",
        "notes",
    }.issubset(df.columns)
    assert math.isclose(compute_cohens_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]), 1.0)
    assert compute_cohens_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == 0.0
