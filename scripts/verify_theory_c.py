"""Independent numerical verification for Part B & Exercises C.1, C.2, C.3.

Usage:
    python scripts/verify_theory_c.py --out results/theory_c_report.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def verify_c1(
    p: float = 0.6,
    n_values: Sequence[int] = (1, 5, 15),
) -> dict[str, Any]:
    """Exact binomial majority accuracy vs Hoeffding lower bound (B.2 / C.1)."""
    if not (0.5 < p < 1.0):
        raise ValueError(f"Expected 0.5 < p < 1.0, got {p}")
    rows: list[dict[str, float | int]] = []
    for n in n_values:
        m = (n // 2) + 1
        exact = sum(
            math.comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))
            for k in range(m, n + 1)
        )
        hoeffding = 1.0 - math.exp(-2.0 * n * ((p - 0.5) ** 2))
        rows.append(
            {
                "n": int(n),
                "majority_threshold": int(m),
                "exact_prob": float(exact),
                "hoeffding_lower_bound": float(hoeffding),
                "bound_gap": float(exact - hoeffding),
            }
        )
    return {"p": float(p), "rows": rows}


def verify_c2(
    p: float = 0.3,
    tpr: float = 0.9,
    fpr: float = 0.1,
    n_values: Sequence[int] = (1, 4, 16),
    *,
    mc_trials: int = 100_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Best-of-n accuracy with imperfect verifier (TPR, FPR) and limit n -> inf (B.3 / C.2)."""
    q_acc = p * tpr + (1.0 - p) * fpr
    p_cond = (p * tpr) / q_acc if q_acc > 0 else 0.0
    p_rej = (p * (1.0 - tpr)) / (1.0 - q_acc) if q_acc < 1.0 else 0.0

    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    for n in n_values:
        prob_any_acc = 1.0 - ((1.0 - q_acc) ** n)
        abstain_prob = prob_any_acc * p_cond
        fallback_prob = prob_any_acc * p_cond + ((1.0 - q_acc) ** n) * p_rej

        # Deterministic Monte Carlo verification
        is_corr = rng.random((mc_trials, n)) < p
        acc_thresholds = np.where(is_corr, tpr, fpr)
        is_acc = rng.random((mc_trials, n)) < acc_thresholds
        has_acc = is_acc.any(axis=1)
        first_acc_idx = np.argmax(is_acc, axis=1)
        acc_selected_corr = is_corr[np.arange(mc_trials), first_acc_idx]

        mc_abstain = float(np.where(has_acc, acc_selected_corr, 0.0).mean())
        mc_fallback = float(np.where(has_acc, acc_selected_corr, is_corr[:, 0]).mean())

        rows.append(
            {
                "n": int(n),
                "prob_at_least_one_accepted": float(prob_any_acc),
                "abstain_prob": float(abstain_prob),
                "random_fallback_prob": float(fallback_prob),
                "mc_abstain_prob": mc_abstain,
                "mc_random_fallback_prob": mc_fallback,
            }
        )

    return {
        "p": float(p),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "q_acc": float(q_acc),
        "p_cond": float(p_cond),
        "p_rej": float(p_rej),
        "limit_n_inf": float(p_cond),
        "mc_trials": int(mc_trials),
        "rows": rows,
    }


def verify_b5_early_stop(
    p: float = 0.6,
    n_values: Sequence[int] = (3, 5, 8, 15),
) -> dict[str, Any]:
    """Exact expected calls E[T] for sequential majority early stopping (B.5)."""
    rows: list[dict[str, float | int]] = []
    for n in n_values:
        m = (n // 2) + 1
        # E[T] = sum_{t=0}^{n-1} P(T > t), where T > t iff max(c1, c0) < m after t draws
        expected_calls = 0.0
        for t in range(n):
            c1_low = max(0, t - m + 1)
            c1_high = min(t, m - 1)
            prob_not_stopped = sum(
                math.comb(t, c1) * (p ** c1) * ((1.0 - p) ** (t - c1))
                for c1 in range(c1_low, c1_high + 1)
            )
            expected_calls += prob_not_stopped

        expected_saved = float(n) - expected_calls
        rows.append(
            {
                "n": int(n),
                "majority_threshold": int(m),
                "expected_calls": float(expected_calls),
                "expected_calls_saved": float(expected_saved),
                "expected_saved_pct": float(100.0 * expected_saved / n),
            }
        )
    return {"p": float(p), "rows": rows}


def default_c3_tree_spec() -> dict[str, Any]:
    """Illustrative MCTS decision tree ready to be swapped with lecturer's tree."""
    return {
        "name": "Illustrative_VeriStat_2Round_Tree",
        "nodes": {
            "root": {
                "visits": 4,
                "value_sum": 2.2,
                "children": ["A1", "A2"],
            },
            "A1": {
                "visits": 2,
                "value_sum": 1.3,
                "children": ["B1", "B2"],
            },
            "A2": {
                "visits": 2,
                "value_sum": 0.9,
                "children": ["C1"],
            },
            "B1": {"visits": 1, "value_sum": 0.8, "children": [], "expand_children": [("D1", 0.9), ("D2", 0.4)]},
            "B2": {"visits": 1, "value_sum": 0.5, "children": [], "expand_children": [("E1", 0.6)]},
            "C1": {"visits": 2, "value_sum": 0.9, "children": [], "expand_children": [("F1", 0.7)]},
        },
    }


def trace_c3_mcts_two_rounds(
    tree_spec: dict[str, Any] | None = None,
    c: float = math.sqrt(2.0),
) -> dict[str, Any]:
    """Trace 2 full MCTS rounds (Selection -> Expansion -> Evaluation -> Backprop)."""
    spec = tree_spec or default_c3_tree_spec()
    nodes: dict[str, dict[str, Any]] = {
        nid: {
            "visits": int(info["visits"]),
            "value_sum": float(info["value_sum"]),
            "children": list(info.get("children", ())),
            "expand_children": list(info.get("expand_children", ())),
        }
        for nid, info in spec["nodes"].items()
    }
    parents: dict[str, str | None] = {"root": None}
    for nid, info in nodes.items():
        for ch in info["children"]:
            parents[ch] = nid

    def _uct(nid: str, parent_id: str) -> float:
        n_sa = nodes[nid]["visits"]
        if n_sa == 0:
            return float("inf")
        n_s = nodes[parent_id]["visits"]
        q = nodes[nid]["value_sum"] / n_sa
        return q + c * math.sqrt(math.log(max(1, n_s)) / n_sa)

    rounds: list[dict[str, Any]] = []
    for round_idx in (1, 2):
        path = ["root"]
        selection_steps: list[dict[str, Any]] = []
        cur = "root"
        while nodes[cur]["children"]:
            candidates = nodes[cur]["children"]
            scored = []
            for ch in candidates:
                score = _uct(ch, cur)
                scored.append(
                    {
                        "child": ch,
                        "visits": nodes[ch]["visits"],
                        "q_value": (
                            nodes[ch]["value_sum"] / nodes[ch]["visits"]
                            if nodes[ch]["visits"] > 0
                            else 0.0
                        ),
                        "uct": None if math.isinf(score) else float(score),
                        "is_unvisited": math.isinf(score),
                    }
                )
            chosen = max(
                candidates,
                key=lambda ch: _uct(ch, cur),
            )
            selection_steps.append(
                {"parent": cur, "candidates": scored, "selected": chosen}
            )
            cur = chosen
            path.append(cur)

        # Expand first available unexpanded action at selected leaf (or evaluate leaf directly)
        expand_queue = nodes[cur]["expand_children"]
        if expand_queue:
            new_id, eval_val = expand_queue.pop(0)
            nodes[new_id] = {
                "visits": 0,
                "value_sum": 0.0,
                "children": [],
                "expand_children": [],
            }
            nodes[cur]["children"].append(new_id)
            parents[new_id] = cur
            eval_node = new_id
            backprop_path = path + [new_id]
        else:
            eval_node = cur
            eval_val = nodes[cur]["value_sum"] / max(1, nodes[cur]["visits"])
            backprop_path = list(path)

        # Backpropagate
        updated_nodes: list[dict[str, Any]] = []
        for nid in reversed(backprop_path):
            nodes[nid]["visits"] += 1
            nodes[nid]["value_sum"] += float(eval_val)
            updated_nodes.append(
                {
                    "node": nid,
                    "visits": nodes[nid]["visits"],
                    "value_sum": float(nodes[nid]["value_sum"]),
                    "q_value": float(nodes[nid]["value_sum"] / nodes[nid]["visits"]),
                }
            )

        rounds.append(
            {
                "round": round_idx,
                "selection_path": path,
                "selection_steps": selection_steps,
                "expanded_or_evaluated_node": eval_node,
                "evaluation_value": float(eval_val),
                "backprop_updates": updated_nodes,
            }
        )

    return {
        "tree_name": spec.get("name", "C3_Tree"),
        "c_uct": float(c),
        "rounds": rounds,
    }


def generate_theory_report() -> dict[str, Any]:
    return {
        "c1": verify_c1(),
        "c2": verify_c2(),
        "b5": verify_b5_early_stop(),
        "c3": trace_c3_mcts_two_rounds(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify numerical solutions for Part B.5 & Exercises C.1-C.3."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON output path for the verification report",
    )
    args = parser.parse_args()

    report = generate_theory_report()
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
