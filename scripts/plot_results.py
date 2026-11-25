"""Generate report-ready compute-curve, breakdown, and theory-vs-empirical figures.

Usage:
    python scripts/plot_results.py \
        --summary results/eval_summary.json \
        --theory results/theory_c_report.json \
        --out-dir results/figures
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _extract_mean_err(
    entry: dict[str, Any],
    metric: str,
) -> tuple[float, float]:
    val = entry.get(metric)
    if isinstance(val, dict):
        return (float(val.get("mean", 0.0)), float(val.get("std", 0.0)))
    mean_val = float(val) if val is not None else 0.0
    ci = entry.get(f"{metric}_ci")
    if isinstance(ci, dict):
        half = 0.5 * (
            float(ci.get("ci_high", mean_val)) - float(ci.get("ci_low", mean_val))
        )
        return (mean_val, max(0.0, half))
    return (mean_val, 0.0)


def _save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def _plot_compute_curve(methods: dict[str, Any], out_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    series_specs: list[tuple[str, str, str, str]] = [
        ("sc_n", "Self-Consistency (Đủ)", "o", "#1f77b4"),
        ("sc_early_n", "Self-Consistency (Dừng sớm B.5)", "s", "#17becf"),
        ("bon_argmax_n", "Best-of-n + Hybrid Verifier", "^", "#2ca02c"),
        ("mcts_", "MCTS-UCT", "D", "#d62728"),
    ]

    for prefix, label, marker, color in series_specs:
        matched = [
            (name, entry)
            for name, entry in methods.items()
            if name.startswith(prefix)
        ]
        if not matched:
            continue
        pts = []
        for name, entry in matched:
            tok_m, _ = _extract_mean_err(entry, "avg_total_tokens")
            acc_m, acc_e = _extract_mean_err(entry, "accuracy")
            pts.append((tok_m, acc_m * 100.0, acc_e * 100.0, name))
        pts.sort(key=lambda x: x[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        es = [p[2] for p in pts]
        ax.errorbar(
            xs,
            ys,
            yerr=es,
            marker=marker,
            color=color,
            label=label,
            linewidth=2.0,
            capsize=4,
        )

    for single_name, label, marker, color in (
        ("single_prompt", "Single Prompt (Baseline)", "X", "#7f7f7f"),
        ("react_n1", "ReAct (n=1)", "P", "#ff7f0e"),
    ):
        if single_name in methods:
            entry = methods[single_name]
            tok_m, _ = _extract_mean_err(entry, "avg_total_tokens")
            acc_m, acc_e = _extract_mean_err(entry, "accuracy")
            ax.errorbar(
                [tok_m],
                [acc_m * 100.0],
                yerr=[acc_e * 100.0],
                marker=marker,
                markersize=9,
                color=color,
                label=label,
                linestyle="none",
                capsize=4,
            )

    ax.set_xlabel("Tổng token trung bình mỗi câu (Prompt + Completion + Verifier)")
    ax.set_ylabel("Độ chính xác (%)")
    ax.set_title("Đường cong Độ chính xác theo Ngân sách Tính toán (VeriStat)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=9)
    return _save_fig(fig, out_dir, "fig1_compute_curve")


def _select_representative_methods(methods: dict[str, Any]) -> list[tuple[str, str]]:
    preferred = [
        ("single_prompt", "Single Prompt"),
        ("react_n1", "ReAct (n=1)"),
        ("sc_n5", "SC (n=5)"),
        ("bon_argmax_n5", "BoN (n=5)"),
    ]
    chosen = [(k, lbl) for k, lbl in preferred if k in methods]
    mcts_keys = sorted(k for k in methods if k.startswith("mcts_"))
    if mcts_keys:
        chosen.append((mcts_keys[-1], mcts_keys[-1].upper()))
    if not chosen:
        for k in sorted(methods.keys())[:5]:
            chosen.append((k, k))
    return chosen


def _plot_breakdown(methods: dict[str, Any], out_dir: Path) -> list[Path]:
    rep = _select_representative_methods(methods)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)

    panels = [
        ("by_difficulty", ["easy", "medium", "hard"], "Theo Độ khó", axes[0]),
        (
            "by_category",
            ["growth", "share", "ranking", "correlation"],
            "Theo Nhóm Câu hỏi",
            axes[1],
        ),
    ]

    for group_key, buckets, title, ax in panels:
        x = np.arange(len(buckets))
        width = 0.8 / max(1, len(rep))
        for m_idx, (m_key, m_label) in enumerate(rep):
            group_dict = methods[m_key].get(group_key, {})
            heights = []
            errs = []
            for b in buckets:
                b_entry = group_dict.get(b, {})
                acc_m, acc_e = _extract_mean_err(b_entry, "accuracy")
                heights.append(acc_m * 100.0)
                errs.append(acc_e * 100.0)
            offset = (m_idx - (len(rep) - 1) / 2.0) * width
            ax.bar(
                x + offset,
                heights,
                width=width * 0.92,
                yerr=errs,
                capsize=2,
                label=m_label,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(buckets)
        ax.set_title(title)
        ax.set_ylim(0, 105)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    axes[0].set_ylabel("Độ chính xác (%)")
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle("Phân rã Độ chính xác theo Độ khó và Nhóm Câu hỏi")
    return _save_fig(fig, out_dir, "fig2_breakdown")


def _plot_theory_vs_empirical(
    summary: dict[str, Any],
    theory: dict[str, Any] | None,
    out_dir: Path,
) -> list[Path]:
    methods = summary.get("methods", {})
    diag = summary.get("verifier_diagnostics", {})

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))

    # Panel A: B.2 / C.1 SC exact vs Hoeffding vs empirical
    ax = axes[0]
    p_emp = _extract_mean_err(methods.get("react_n1", {}), "accuracy")[0]
    p_sc = p_emp if 0.5 < p_emp < 1.0 else (
        float(theory["c1"]["p"]) if theory and "c1" in theory else 0.6
    )
    ns_theory = np.arange(1, 16)
    exact_curve = []
    hoeff_curve = []
    for n in ns_theory:
        m = int(n // 2) + 1
        ex = sum(
            math.comb(int(n), k) * (p_sc ** k) * ((1.0 - p_sc) ** (int(n) - k))
            for k in range(m, int(n) + 1)
        )
        hb = 1.0 - math.exp(-2.0 * int(n) * ((p_sc - 0.5) ** 2))
        exact_curve.append(ex * 100.0)
        hoeff_curve.append(hb * 100.0)
    ax.plot(ns_theory, exact_curve, label=f"Nhị thức lý thuyết (p={p_sc:.2f})", color="#1f77b4")
    ax.plot(ns_theory, hoeff_curve, "--", label="Cận dưới Hoeffding (B.2)", color="#7f7f7f")

    emp_sc_x, emp_sc_y = [], []
    for n_val, key in ((1, "react_n1"), (3, "sc_n3"), (5, "sc_n5"), (8, "sc_n8")):
        if key in methods:
            emp_sc_x.append(n_val)
            emp_sc_y.append(_extract_mean_err(methods[key], "accuracy")[0] * 100.0)
    if emp_sc_x:
        ax.scatter(emp_sc_x, emp_sc_y, color="#d62728", zorder=5, label="Thực nghiệm SC")
    ax.set_title("B.2 / C.1: Bỏ phiếu Đa số")
    ax.set_xlabel("Số mẫu n")
    ax.set_ylabel("Xác suất / Độ chính xác (%)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)

    # Panel B: B.3 / C.2 Best-of-n TPR/FPR & ceiling vs empirical
    ax = axes[1]
    p_bon = p_emp if 0.0 < p_emp < 1.0 else 0.3
    tpr = float(diag.get("tpr", theory["c2"]["tpr"] if theory and "c2" in theory else 0.9))
    fpr = float(diag.get("fpr", theory["c2"]["fpr"] if theory and "c2" in theory else 0.1))
    q_acc = p_bon * tpr + (1.0 - p_bon) * fpr
    p_cond = (p_bon * tpr) / q_acc if q_acc > 0 else 0.0
    p_rej = (p_bon * (1.0 - tpr)) / (1.0 - q_acc) if q_acc < 1.0 else 0.0
    ns_bon = np.arange(1, 17)
    abstain_curve = [(1.0 - (1.0 - q_acc) ** int(n)) * p_cond * 100.0 for n in ns_bon]
    fallback_curve = [
        ((1.0 - (1.0 - q_acc) ** int(n)) * p_cond + ((1.0 - q_acc) ** int(n)) * p_rej)
        * 100.0
        for n in ns_bon
    ]
    ax.plot(ns_bon, fallback_curve, label="B.3 Fallback ngẫu nhiên", color="#2ca02c")
    ax.plot(ns_bon, abstain_curve, "--", label="B.3 Từ chối khi 0 đạt", color="#9467bd")
    ax.axhline(p_cond * 100.0, linestyle=":", color="black", label=f"Giới hạn n→∞ ({p_cond*100:.1f}%)")

    emp_bon_x, emp_bon_y = [], []
    for n_val in (3, 5, 8):
        key = f"bon_tau_fallback_n{n_val}"
        if key in methods:
            emp_bon_x.append(n_val)
            emp_bon_y.append(_extract_mean_err(methods[key], "accuracy")[0] * 100.0)
    if emp_bon_x:
        ax.scatter(emp_bon_x, emp_bon_y, color="#d62728", zorder=5, label="Thực nghiệm BoN (τ)")
    ax.set_title("B.3 / C.2: Best-of-n (TPR/FPR)")
    ax.set_xlabel("Số ứng viên n")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)

    # Panel C: B.5 Early-stopping calls saved
    ax = axes[2]
    ns_es = [3, 5, 8]
    th_saved = []
    for n in ns_es:
        m = (n // 2) + 1
        e_calls = sum(
            sum(
                math.comb(t, c1) * (p_sc ** c1) * ((1.0 - p_sc) ** (t - c1))
                for c1 in range(max(0, t - m + 1), min(t, m - 1) + 1)
            )
            for t in range(n)
        )
        th_saved.append(float(n) - e_calls)
    x_pos = np.arange(len(ns_es))
    ax.bar(x_pos - 0.18, th_saved, width=0.35, label="Lý thuyết E[n - T] (B.5)", color="#17becf")
    emp_saved = [
        _extract_mean_err(methods.get(f"sc_early_n{n}", {}), "avg_calls_saved")[0]
        for n in ns_es
    ]
    ax.bar(x_pos + 0.18, emp_saved, width=0.35, label="Thực nghiệm tiết kiệm", color="#ff7f0e")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"n={n}" for n in ns_es])
    ax.set_title("B.5: Tiết kiệm Lời gọi Dừng sớm")
    ax.set_ylabel("Số lời gọi tiết kiệm TB")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)

    fig.suptitle("Đối chiếu Lý thuyết (Phần B & C) với Thực nghiệm (Phần D)")
    return _save_fig(fig, out_dir, "fig3_theory_vs_empirical")


def generate_all_figures(
    summary_path: Path,
    out_dir: Path,
    *,
    theory_path: Path | None = None,
) -> list[Path]:
    """Generate all report figures (.png and .pdf) from `eval_summary.json`."""
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    theory = (
        json.loads(Path(theory_path).read_text(encoding="utf-8"))
        if theory_path is not None and Path(theory_path).exists()
        else None
    )
    methods = summary.get("methods", {})
    outputs: list[Path] = []
    outputs.extend(_plot_compute_curve(methods, out_dir))
    outputs.extend(_plot_breakdown(methods, out_dir))
    outputs.extend(_plot_theory_vs_empirical(summary, theory, out_dir))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate VeriStat evaluation figures.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--theory", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()

    files = generate_all_figures(args.summary, args.out_dir, theory_path=args.theory)
    print(json.dumps({"generated": [str(f) for f in files]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
