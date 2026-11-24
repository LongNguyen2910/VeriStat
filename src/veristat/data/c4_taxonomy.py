"""C.4 Error Taxonomy sampling and Inter-Annotator Agreement (Cohen's kappa)."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ERROR_TAXONOMY_LABELS: tuple[str, ...] = (
    "wrong_tool",
    "bad_args_or_sql",
    "infinite_loop",
    "number_hallucination",
    "unit_or_admin_mismatch",
    "wrong_formula_or_logic",
)


def _stratified_pick(
    records: list[dict[str, Any]],
    k: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Pick up to `k` records stratified by (method, difficulty)."""
    if k <= 0 or not records:
        return []
    if len(records) <= k:
        chosen = list(records)
        rng.shuffle(chosen)
        return chosen

    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        key = (str(rec.get("method", "unknown")), str(rec.get("difficulty", "unknown")))
        buckets[key].append(rec)

    for key in buckets:
        rng.shuffle(buckets[key])

    keys = sorted(buckets.keys())
    selected: list[dict[str, Any]] = []
    idx = 0
    while len(selected) < k:
        progressed = False
        for key in keys:
            if len(selected) >= k:
                break
            if buckets[key]:
                selected.append(buckets[key].pop())
                progressed = True
        idx += 1
        if not progressed:
            break
    return selected


def sample_traces_for_c4(
    traces_path: Path,
    out_csv: Path,
    n: int = 50,
    seed: int = 42,
) -> int:
    """Sample `n` execution traces for C.4 error taxonomy annotation.

    Prioritizes incorrect traces (`is_correct == False`) stratified across
    `(method, difficulty)`, topping up with correct traces only if fewer
    than `n` incorrect traces are available.
    """
    records: list[dict[str, Any]] = []
    for line in traces_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))

    rng = random.Random(seed)
    incorrect = [r for r in records if not bool(r.get("is_correct", False))]
    correct = [r for r in records if bool(r.get("is_correct", False))]

    selected = _stratified_pick(incorrect, n, rng)
    if len(selected) < n:
        selected.extend(_stratified_pick(correct, n - len(selected), rng))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question_id",
        "method",
        "category",
        "difficulty",
        "is_correct",
        "predicted_answer",
        "ground_truth",
        "trace_preview",
        "annotator_1",
        "annotator_2",
        "final_label",
        "notes",
    ]
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in selected:
            raw_trace = rec.get("trace", "")
            trace_str = (
                json.dumps(raw_trace, ensure_ascii=False)
                if not isinstance(raw_trace, str)
                else raw_trace
            )
            writer.writerow({
                "question_id": rec.get("question_id", ""),
                "method": rec.get("method", ""),
                "category": rec.get("category", ""),
                "difficulty": rec.get("difficulty", ""),
                "is_correct": rec.get("is_correct", ""),
                "predicted_answer": rec.get("predicted_answer", ""),
                "ground_truth": rec.get("ground_truth", ""),
                "trace_preview": trace_str[:500].replace("\n", " "),
                "annotator_1": "",
                "annotator_2": "",
                "final_label": "",
                "notes": "",
            })

    return len(selected)


def compute_cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Compute unweighted Cohen's kappa between two equal-length label lists."""
    if len(labels_a) != len(labels_b):
        raise ValueError(
            f"Label lists must have equal length, got {len(labels_a)} and {len(labels_b)}"
        )
    n = len(labels_a)
    if n == 0:
        raise ValueError("Label lists must not be empty")

    observed = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    all_labels = set(counts_a) | set(counts_b)
    expected = sum((counts_a[lbl] / n) * (counts_b[lbl] / n) for lbl in all_labels)

    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def evaluate_annotation_csv(csv_path: Path) -> dict[str, Any]:
    """Compute Cohen's kappa and label distribution from a filled C.4 CSV."""
    labels_a: list[str] = []
    labels_b: list[str] = []
    final_labels: list[str] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            a = (row.get("annotator_1") or "").strip()
            b = (row.get("annotator_2") or "").strip()
            fin = (row.get("final_label") or "").strip()
            if a and b:
                labels_a.append(a)
                labels_b.append(b)
            if fin:
                final_labels.append(fin)

    if not labels_a:
        raise ValueError("No rows with both annotator_1 and annotator_2 filled")

    kappa = compute_cohens_kappa(labels_a, labels_b)
    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    return {
        "n_paired": len(labels_a),
        "raw_agreement_pct": round(100.0 * agree / len(labels_a), 2),
        "cohens_kappa": round(kappa, 4),
        "final_label_counts": dict(Counter(final_labels)),
    }
