#!/usr/bin/env python3
"""
Generate Paper-Formatted Results Table (Table 3 style).

This script collects experiment results and formats them in the paper's table format.
"""

import json
import os
import sys
import argparse
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def format_table_3(results: Dict[str, Dict], title: str = "Table 3 (Real VLM)") -> str:
    """Format results in paper Table 3 style."""
    lines = []
    lines.append("=" * 110)
    lines.append(f"  {title}: Cross-Backbone Real VLM Evaluation with H-MDP Pipeline")
    lines.append("=" * 110)
    lines.append(f"  {'VLM Backbone':<22}  {'Base':^30}  {'H-MDP (Ours)':^34}  {'Improvement':^16}")
    lines.append(f"  {'':<22}  {'Act%':>8} {'Pt%':>8} {'Dist':>9}   {'Act%':>8} {'Pt%':>8} {'Dist':>9}   {'ΔPt':>6} {'ΔDist':>8}")
    lines.append("  " + "-" * 100)

    model_order = ["aguvis-7b", "qwen2.5-vl-7b", "uground-7b", "gui-actor-7b", "ui-tars-1.5-7b"]

    for key in model_order:
        if key not in results:
            continue
        r = results[key]
        b, h = r["base"], r["hmdp"]
        d_pt = (h["point_accuracy"] - b["point_accuracy"]) * 100
        d_dist = h["avg_distance"] - b["avg_distance"]

        pt_marker = "↑" if d_pt > 0 else ("↓" if d_pt < 0 else " ")
        dist_marker = "↑" if d_dist > 0 else ("↓" if d_dist < 0 else " ")

        lines.append(
            f"  {r['display_name']:<22}"
            f"  {b['action_accuracy']*100:>7.1f}% {b['point_accuracy']*100:>7.1f}% {b['avg_distance']:>9.4f}"
            f"   {h['action_accuracy']*100:>7.1f}% {h['point_accuracy']*100:>7.1f}% {h['avg_distance']:>9.4f}"
            f"   {d_pt:>+5.1f}%{pt_marker} {d_dist:>+7.4f}{dist_marker}"
        )

    lines.append("  " + "-" * 100)

    # Average row
    gains_pt = []
    gains_dist = []
    for key in model_order:
        if key not in results:
            continue
        b, h = results[key]["base"], results[key]["hmdp"]
        gains_pt.append((h["point_accuracy"] - b["point_accuracy"]) * 100)
        gains_dist.append(h["avg_distance"] - b["avg_distance"])

    if gains_pt:
        avg_pt = sum(gains_pt) / len(gains_pt)
        avg_dist = sum(gains_dist) / len(gains_dist)
        lines.append(
            f"  {'Average':<22}"
            f"  {'':30}  {'':34}  {avg_pt:>+5.1f}%   {avg_dist:>+7.4f}"
        )

    lines.append("=" * 110)
    return "\n".join(lines)


def generate_latex_table(results: Dict[str, Dict]) -> str:
    """Generate LaTeX table code."""
    lines = []
    lines.append("% LaTeX Table 3: Cross-Backbone Real VLM Evaluation")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Cross-Backbone Real VLM Evaluation with H-MDP Pipeline}")
    lines.append("\\label{tab:real_vlm_results}")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\toprule")
    lines.append("\\multirow{2}{*}{\\textbf{VLM Backbone}} & \\multicolumn{3}{c}{\\textbf{Base}} & \\multicolumn{3}{c}{\\textbf{H-MDP (Ours)}} \\\\")
    lines.append("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}")
    lines.append("& Act Acc & Pt Acc & Avg Dist & Act Acc & Pt Acc & Avg Dist \\\\")
    lines.append("\\midrule")

    model_order = ["aguvis-7b", "qwen2.5-vl-7b", "uground-7b", "gui-actor-7b", "ui-tars-1.5-7b"]

    for key in model_order:
        if key not in results:
            continue
        r = results[key]
        b, h = r["base"], r["hmdp"]
        lines.append(
            f"{r['display_name']} & "
            f"{b['action_accuracy']*100:.1f}\\% & {b['point_accuracy']*100:.1f}\\% & {b['avg_distance']:.4f} & "
            f"{h['action_accuracy']*100:.1f}\\% & {h['point_accuracy']*100:.1f}\\% & {h['avg_distance']:.4f} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate paper Table 3 text from a result JSON")
    parser.add_argument("--results", default="results/json/real_vlm_hmdp_results.json",
                        help="Path to hmdp.run_real_vlm result JSON")
    parser.add_argument("--out", default="results/paper_table_3.txt",
                        help="Path to write formatted table text")
    args = parser.parse_args()
    results_file = args.results

    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
        print(f"Loaded results from: {results_file}")
    else:
        print(f"Results file not found: {results_file}")
        return

    # Print formatted table
    print("\n" + format_table_3(results))

    # Print LaTeX table
    print("\n" + "=" * 110)
    print("  LaTeX Table Source")
    print("=" * 110)
    print(generate_latex_table(results))

    # Save formatted results
    output_file = args.out
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(format_table_3(results))
        f.write("\n\n")
        f.write("=" * 110 + "\n")
        f.write("  LaTeX Table Source\n")
        f.write("=" * 110 + "\n")
        f.write(generate_latex_table(results))

    print(f"\nFormatted results saved to: {output_file}")


if __name__ == "__main__":
    main()
