#!/usr/bin/env python3
"""
Generate paper-quality Table and Figures for Real VLM + H-MDP results.

Outputs:
  - Console: Formatted Table 3 (Real VLM) + LaTeX source
  - PDF/PNG: Fig 3a — Point Accuracy comparison (Base vs H-MDP)
  - PDF/PNG: Fig 3b — Avg Distance comparison (Base vs H-MDP)
  - PDF/PNG: Fig 3  — Combined (a)+(b) figure
  - PDF/PNG: Fig 4  — Detailed metric breakdown (4-panel)

Usage:
    python -m hmdp.plot_real_vlm_results
    python -m hmdp.plot_real_vlm_results --results results/real_vlm_hmdp_results_v2.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ════════════════════════════════════════════════════════════════
#  Shared plot style (consistent with plot_fig2b.py)
# ════════════════════════════════════════════════════════════════

PAPER_RCPARAMS = {
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
}

# Color palette: professional, colorblind-friendly
COLORS = {
    "base":      "#5B7FA5",   # steel blue
    "hmdp":      "#E8734A",   # coral orange
    "gain_pos":  "#4CAF50",   # green
    "gain_neg":  "#E53935",   # red
    "bg_light":  "#F7F9FC",
}

MODEL_ORDER = ["aguvis-7b", "qwen2.5-vl-7b", "uground-7b", "gui-actor-7b"]


def _apply_style(ax, grid_alpha=0.25):
    """Apply consistent grid and spine style."""
    ax.grid(True, axis="y", alpha=grid_alpha, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ════════════════════════════════════════════════════════════════
#  Table: Console + LaTeX
# ════════════════════════════════════════════════════════════════

def print_table(results, title="Table 3 (Real VLM)"):
    """Print formatted console table."""
    print(f"\n{'='*100}")
    print(f"  {title}: Cross-Backbone Real VLM Evaluation with H-MDP Pipeline")
    print(f"{'='*100}")
    print(f"  {'VLM Backbone':<22}  {'Base':^30}  {'H-MDP (Ours)':^34}  {'Improvement':^16}")
    delta_pt = '\u0394Pt'
    delta_dist = '\u0394Dist'
    print(f"  {'':<22}  {'Act%':>8} {'Pt%':>8} {'Dist':>9}   {'Act%':>8} {'Pt%':>8} {'Dist':>9}   {delta_pt:>6} {delta_dist:>8}")
    print(f"  {'-'*94}")

    for key in MODEL_ORDER:
        if key not in results:
            continue
        r = results[key]
        b, h = r["base"], r["hmdp"]
        d_pt = (h["point_accuracy"] - b["point_accuracy"]) * 100
        d_dist = h["avg_distance"] - b["avg_distance"]

        pt_marker = "\u2191" if d_pt > 0 else ("\u2193" if d_pt < 0 else " ")
        dist_marker = "\u2191" if d_dist > 0 else ("\u2193" if d_dist < 0 else " ")

        print(f"  {r['display_name']:<22}"
              f"  {b['action_accuracy']*100:>7.1f}% {b['point_accuracy']*100:>7.1f}% {b['avg_distance']:>9.4f}"
              f"   {h['action_accuracy']*100:>7.1f}% {h['point_accuracy']*100:>7.1f}% {h['avg_distance']:>9.4f}"
              f"   {d_pt:>+5.1f}%{pt_marker} {d_dist:>+7.4f}{dist_marker}")

    print(f"  {'-'*94}")

    # Average row
    gains_pt = []
    gains_dist = []
    for key in MODEL_ORDER:
        if key not in results:
            continue
        b, h = results[key]["base"], results[key]["hmdp"]
        gains_pt.append((h["point_accuracy"] - b["point_accuracy"]) * 100)
        gains_dist.append(h["avg_distance"] - b["avg_distance"])

    print(f"  {'Average':>22}  {'':<30}  {'':<34}  {np.mean(gains_pt):>+5.1f}%  {np.mean(gains_dist):>+7.4f}")
    print(f"{'='*100}")


def generate_latex(results, label="tab:real_vlm"):
    """Generate LaTeX table source."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Cross-backbone evaluation with real VLM inference. We compare")
    lines.append(r"    base single-call performance against the full H-MDP pipeline")
    lines.append(r"    ($\varepsilon{=}5.0$, $k{=}5$) with DINOv2 proxy encoder, LDP,")
    lines.append(r"    GoT $k$-path reasoning, and LTM priors.}")
    lines.append(r"  \label{" + label + r"}")
    lines.append(r"  \small")
    lines.append(r"  \setlength{\tabcolsep}{4pt}")
    lines.append(r"  \begin{tabular}{l cc c cc c rr}")
    lines.append(r"    \toprule")
    lines.append(r"    & \multicolumn{2}{c}{\textbf{Base}} & & \multicolumn{2}{c}{\textbf{H-MDP (Ours)}} & & \multicolumn{2}{c}{\textbf{Improvement}} \\")
    lines.append(r"    \cmidrule{2-3} \cmidrule{5-6} \cmidrule{8-9}")
    lines.append(r"    \textbf{VLM Backbone} & Pt.\% $\uparrow$ & Dist $\downarrow$ & & Pt.\% $\uparrow$ & Dist $\downarrow$ & & $\Delta$Pt. & $\Delta$Dist \\")
    lines.append(r"    \midrule")

    for key in MODEL_ORDER:
        if key not in results:
            continue
        r = results[key]
        b, h = r["base"], r["hmdp"]
        d_pt = (h["point_accuracy"] - b["point_accuracy"]) * 100
        d_dist = h["avg_distance"] - b["avg_distance"]

        name_tex = r["display_name"].replace("-", "{-}")
        pt_str = f"{d_pt:+.1f}\\%"
        dist_str = f"{d_dist:+.3f}"

        lines.append(
            f"    {name_tex} & {b['point_accuracy']*100:.1f} & {b['avg_distance']:.3f}"
            f" & & {h['point_accuracy']*100:.1f} & {h['avg_distance']:.3f}"
            f" & & {pt_str} & {dist_str} \\\\"
        )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
#  Fig 3: Base vs H-MDP bar charts
# ════════════════════════════════════════════════════════════════

def plot_fig3_combined(results, save_dir="results"):
    """
    Fig 3: Two-panel bar chart.
      (a) Point Accuracy: Base vs H-MDP
      (b) Avg Distance: Base vs H-MDP
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(PAPER_RCPARAMS)

    models = [results[k]["display_name"] for k in MODEL_ORDER if k in results]
    base_pt = [results[k]["base"]["point_accuracy"] * 100 for k in MODEL_ORDER if k in results]
    hmdp_pt = [results[k]["hmdp"]["point_accuracy"] * 100 for k in MODEL_ORDER if k in results]
    base_dist = [results[k]["base"]["avg_distance"] for k in MODEL_ORDER if k in results]
    hmdp_dist = [results[k]["hmdp"]["avg_distance"] for k in MODEL_ORDER if k in results]

    n = len(models)
    x = np.arange(n)
    w = 0.32

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── (a) Point Accuracy ──
    bars1 = ax1.bar(x - w/2, base_pt, w, label="Base (single call)",
                    color=COLORS["base"], edgecolor="white", linewidth=0.8, zorder=3)
    bars2 = ax1.bar(x + w/2, hmdp_pt, w, label="H-MDP (Ours)",
                    color=COLORS["hmdp"], edgecolor="white", linewidth=0.8, zorder=3)

    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                 f"{h:.1f}", ha="center", va="bottom", fontsize=8.5, color="#555")
    for bar in bars2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                 f"{h:.1f}", ha="center", va="bottom", fontsize=8.5, color="#555")

    # Add gain annotations
    for i in range(n):
        gain = hmdp_pt[i] - base_pt[i]
        if abs(gain) > 0.01:
            color = COLORS["gain_pos"] if gain > 0 else COLORS["gain_neg"]
            y_pos = max(base_pt[i], hmdp_pt[i]) + 2.5
            ax1.annotate(f"{gain:+.1f}%", xy=(x[i], y_pos),
                         ha="center", va="bottom", fontsize=9, fontweight="bold",
                         color=color)

    ax1.set_xlabel("")
    ax1.set_ylabel("Point Accuracy (%)")
    ax1.set_title("(a) Grounding Accuracy")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=12, ha="right")
    ax1.set_ylim(0, max(max(base_pt), max(hmdp_pt)) * 1.35)
    ax1.legend(loc="upper right", framealpha=0.9)
    _apply_style(ax1)

    # ── (b) Avg Distance ──
    bars3 = ax2.bar(x - w/2, base_dist, w, label="Base (single call)",
                    color=COLORS["base"], edgecolor="white", linewidth=0.8, zorder=3)
    bars4 = ax2.bar(x + w/2, hmdp_dist, w, label="H-MDP (Ours)",
                    color=COLORS["hmdp"], edgecolor="white", linewidth=0.8, zorder=3)

    for bar in bars3:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                 f"{h:.3f}", ha="center", va="bottom", fontsize=8.5, color="#555")
    for bar in bars4:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                 f"{h:.3f}", ha="center", va="bottom", fontsize=8.5, color="#555")

    # Add gain annotations
    for i in range(n):
        gain = hmdp_dist[i] - base_dist[i]
        color = COLORS["gain_pos"] if gain < 0 else COLORS["gain_neg"]
        y_pos = max(base_dist[i], hmdp_dist[i]) + 0.025
        ax2.annotate(f"{gain:+.3f}", xy=(x[i], y_pos),
                     ha="center", va="bottom", fontsize=9, fontweight="bold",
                     color=color)

    ax2.set_xlabel("")
    ax2.set_ylabel("Avg Distance (lower is better)")
    ax2.set_title("(b) Average Localization Distance")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=12, ha="right")
    ax2.set_ylim(0, max(max(base_dist), max(hmdp_dist)) * 1.20)
    ax2.legend(loc="upper right", framealpha=0.9)
    _apply_style(ax2)

    fig.suptitle("Real VLM + H-MDP Pipeline: Base vs Full Pipeline",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(save_dir, f"fig3_real_vlm.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=300 if ext == "png" else 150)
    plt.close(fig)
    print(f"  Saved: {save_dir}/fig3_real_vlm.pdf / .png")


# ════════════════════════════════════════════════════════════════
#  Fig 4: Detailed 4-panel breakdown
# ════════════════════════════════════════════════════════════════

def plot_fig4_detailed(results, save_dir="results"):
    """
    Fig 4: 4-panel detailed breakdown.
      (a) Point Accuracy per model
      (b) Avg Distance per model
      (c) Action Accuracy per model
      (d) H-MDP pipeline characteristics (LTM episodes, time overhead)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(PAPER_RCPARAMS)

    models = [results[k]["display_name"] for k in MODEL_ORDER if k in results]
    keys = [k for k in MODEL_ORDER if k in results]
    n = len(models)
    x = np.arange(n)
    w = 0.32

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # ── (a) Point Accuracy with gain arrows ──
    ax = axes[0, 0]
    base_pt = [results[k]["base"]["point_accuracy"] * 100 for k in keys]
    hmdp_pt = [results[k]["hmdp"]["point_accuracy"] * 100 for k in keys]

    ax.bar(x - w/2, base_pt, w, label="Base", color=COLORS["base"],
           edgecolor="white", linewidth=0.8, zorder=3)
    ax.bar(x + w/2, hmdp_pt, w, label="H-MDP", color=COLORS["hmdp"],
           edgecolor="white", linewidth=0.8, zorder=3)

    for i in range(n):
        gain = hmdp_pt[i] - base_pt[i]
        if abs(gain) > 0.01:
            color = COLORS["gain_pos"] if gain > 0 else "#888"
            ax.annotate(f"{gain:+.1f}%", xy=(x[i] + w/2, hmdp_pt[i]),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=9, fontweight="bold", color=color)

    ax.set_ylabel("Point Accuracy (%)")
    ax.set_title("(a) Grounding Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, max(max(base_pt), max(hmdp_pt)) * 1.3)
    _apply_style(ax)

    # ── (b) Avg Distance ──
    ax = axes[0, 1]
    base_dist = [results[k]["base"]["avg_distance"] for k in keys]
    hmdp_dist = [results[k]["hmdp"]["avg_distance"] for k in keys]

    ax.bar(x - w/2, base_dist, w, label="Base", color=COLORS["base"],
           edgecolor="white", linewidth=0.8, zorder=3)
    ax.bar(x + w/2, hmdp_dist, w, label="H-MDP", color=COLORS["hmdp"],
           edgecolor="white", linewidth=0.8, zorder=3)

    for i in range(n):
        gain = hmdp_dist[i] - base_dist[i]
        color = COLORS["gain_pos"] if gain < 0 else "#888"
        ax.annotate(f"{gain:+.3f}", xy=(x[i] + w/2, hmdp_dist[i]),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=9, fontweight="bold", color=color)

    ax.set_ylabel("Avg Distance")
    ax.set_title("(b) Localization Distance (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.legend(loc="upper right", fontsize=9)
    _apply_style(ax)

    # ── (c) Action Accuracy ──
    ax = axes[1, 0]
    base_act = [results[k]["base"]["action_accuracy"] * 100 for k in keys]
    hmdp_act = [results[k]["hmdp"]["action_accuracy"] * 100 for k in keys]

    ax.bar(x - w/2, base_act, w, label="Base", color=COLORS["base"],
           edgecolor="white", linewidth=0.8, zorder=3)
    ax.bar(x + w/2, hmdp_act, w, label="H-MDP", color=COLORS["hmdp"],
           edgecolor="white", linewidth=0.8, zorder=3)

    ax.set_ylabel("Action Accuracy (%)")
    ax.set_title("(c) Action Type Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylim(80, 90)
    ax.legend(loc="lower right", fontsize=9)
    _apply_style(ax)

    # ── (d) LTM episodes & time overhead ──
    ax = axes[1, 1]
    ltm_eps = [results[k]["hmdp"].get("ltm_episodes", 0) for k in keys]
    base_time = [results[k]["base"]["time_sec"] for k in keys]
    hmdp_time = [results[k]["hmdp"]["time_sec"] for k in keys]
    overhead = [h / max(b, 1) for h, b in zip(hmdp_time, base_time)]

    ax2_twin = ax.twinx()

    bars_ltm = ax.bar(x - w/2, ltm_eps, w, label="LTM Episodes",
                      color="#7E57C2", edgecolor="white", linewidth=0.8, zorder=3)
    bars_overhead = ax2_twin.bar(x + w/2, overhead, w, label="Time Overhead (x)",
                                 color="#FFB74D", edgecolor="white", linewidth=0.8, zorder=3)

    for i, bar in enumerate(bars_ltm):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(ltm_eps[i]), ha="center", va="bottom", fontsize=9)
    for i, bar in enumerate(bars_overhead):
        ax2_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                      f"{overhead[i]:.1f}x", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("LTM Episodes Stored")
    ax2_twin.set_ylabel("Time Overhead (H-MDP / Base)")
    ax.set_title("(d) Pipeline Characteristics")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.legend(loc="upper left", fontsize=9)
    ax2_twin.legend(loc="upper right", fontsize=9)
    _apply_style(ax)

    fig.suptitle("Fig 4: Detailed Real VLM + H-MDP Analysis ($\\varepsilon$=5.0, k=5)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(save_dir, f"fig4_detailed.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=300 if ext == "png" else 150)
    plt.close(fig)
    print(f"  Saved: {save_dir}/fig4_detailed.pdf / .png")


# ════════════════════════════════════════════════════════════════
#  Fig 5: v1 vs v2 improvement comparison
# ════════════════════════════════════════════════════════════════

def plot_fig5_v1_vs_v2(v1_results, v2_results, save_dir="results"):
    """
    Fig 5: Horizontal bar chart showing Point Accuracy gain (v1 vs v2).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(PAPER_RCPARAMS)

    models = []
    v1_gains = []
    v2_gains = []

    for key in MODEL_ORDER:
        if key in v1_results and key in v2_results:
            models.append(v2_results[key]["display_name"])
            v1_gains.append(v1_results[key]["gain"] * 100)
            v2_gains.append(v2_results[key]["gain"] * 100)

    n = len(models)
    y = np.arange(n)
    h = 0.30

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.barh(y - h/2, v1_gains, h, label="v1 ($\\varepsilon$=1.0, weighted avg)",
            color="#B0BEC5", edgecolor="white", linewidth=0.8, zorder=3)
    ax.barh(y + h/2, v2_gains, h, label="v2 ($\\varepsilon$=5.0, geo-median + greedy anchor)",
            color=COLORS["hmdp"], edgecolor="white", linewidth=0.8, zorder=3)

    # Value labels
    for i in range(n):
        ax.text(v1_gains[i] + (0.08 if v1_gains[i] >= 0 else -0.08),
                y[i] - h/2, f"{v1_gains[i]:+.1f}%",
                ha="left" if v1_gains[i] >= 0 else "right",
                va="center", fontsize=9, color="#666")
        ax.text(v2_gains[i] + (0.08 if v2_gains[i] >= 0 else -0.08),
                y[i] + h/2, f"{v2_gains[i]:+.1f}%",
                ha="left" if v2_gains[i] >= 0 else "right",
                va="center", fontsize=10, fontweight="bold",
                color=COLORS["gain_pos"] if v2_gains[i] > 0 else "#888")

    ax.axvline(x=0, color="#333", linewidth=0.8, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.set_xlabel("Point Accuracy Gain (%p)")
    ax.set_title("H-MDP Pipeline Improvement: v1 vs v2")
    ax.legend(loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.25, linestyle="--", linewidth=0.6)

    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(save_dir, f"fig5_v1_vs_v2.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=300 if ext == "png" else 150)
    plt.close(fig)
    print(f"  Saved: {save_dir}/fig5_v1_vs_v2.pdf / .png")


# ════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Plot Real VLM H-MDP Results")
    parser.add_argument("--results", default="results/json/real_vlm_hmdp_results.json",
                        help="Path to v2 results JSON")
    parser.add_argument("--results-v1", default="results/real_vlm_hmdp_results.json",
                        help="Path to v1 results JSON (for comparison)")
    parser.add_argument("--save-dir", default="results",
                        help="Directory to save figures")
    args = parser.parse_args()

    print("=" * 70)
    print("  Real VLM + H-MDP: Paper Tables & Figures")
    print("=" * 70)

    # Load results
    with open(args.results) as f:
        results = json.load(f)
    print(f"  Loaded v2: {args.results}")

    v1_results = None
    if os.path.exists(args.results_v1):
        with open(args.results_v1) as f:
            v1_results = json.load(f)
        print(f"  Loaded v1: {args.results_v1}")

    # ── Console Table ──
    print_table(results, title="Table 3 (Real VLM)")

    # ── LaTeX Table ──
    latex = generate_latex(results)
    print(f"\n{'─'*70}")
    print("  LaTeX Source:")
    print(f"{'─'*70}")
    print(latex)

    latex_path = os.path.join(args.save_dir, "table3_real_vlm.tex")
    os.makedirs(args.save_dir, exist_ok=True)
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"\n  Saved: {latex_path}")

    # ── Figures ──
    print(f"\n{'─'*70}")
    print("  Generating Figures...")
    print(f"{'─'*70}")

    plot_fig3_combined(results, save_dir=args.save_dir)
    plot_fig4_detailed(results, save_dir=args.save_dir)

    if v1_results:
        plot_fig5_v1_vs_v2(v1_results, results, save_dir=args.save_dir)

    print(f"\n{'='*70}")
    print("  All tables and figures generated!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
