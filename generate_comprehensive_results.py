#!/usr/bin/env python3
"""
Generate Comprehensive Paper Results (Tables + Figures).

This script generates paper-quality tables and figures from experiment results.
"""

import json
import os
import sys
import argparse
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def format_table_with_epsilons(results: Dict, title: str = "Table: Epsilon Ablation") -> str:
    """Format epsilon ablation study results."""
    lines = []
    lines.append("=" * 100)
    lines.append(f"  {title}: Privacy-Utility Trade-off Analysis")
    lines.append("=" * 100)
    lines.append(f"  {'VLM Backbone':<20}  {'ε':>6}  {'k':>4}  {'Base Pt%':>10}  {'H-MDP Pt%':>12}  {'Gain':>10}  {'Dist':>10}")
    lines.append("  " + "-" * 90)

    # Group by model and epsilon
    for model_key, data in sorted(results.items()):
        if "ablation" not in data:
            continue
        for entry in data["ablation"]:
            eps = entry.get("epsilon", 5.0)
            k = entry.get("k", 3)
            base_pt = entry["base"]["point_accuracy"] * 100
            hmdp_pt = entry["hmdp"]["point_accuracy"] * 100
            gain = hmdp_pt - base_pt
            dist = entry["hmdp"]["avg_distance"]

            lines.append(
                f"  {data['display_name']:<20}  {eps:>6.1f}  {k:>4}  "
                f"{base_pt:>9.1f}%  {hmdp_pt:>11.1f}%  {gain:>+9.1f}%  {dist:>10.4f}"
            )

    lines.append("=" * 100)
    return "\n".join(lines)


def format_summary_statistics(results: Dict) -> str:
    """Generate summary statistics."""
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("  Summary Statistics")
    lines.append("=" * 100)

    total_models = len(results)
    lines.append(f"  Total Models Evaluated: {total_models}")

    for model_key, data in results.items():
        b = data["base"]
        h = data["hmdp"]
        lines.append(f"\n  {data['display_name']}:")
        lines.append(f"    Base:  Action Acc = {b['action_accuracy']*100:.1f}%, Point Acc = {b['point_accuracy']*100:.1f}%, Dist = {b['avg_distance']:.4f}")
        lines.append(f"    H-MDP: Action Acc = {h['action_accuracy']*100:.1f}%, Point Acc = {h['point_accuracy']*100:.1f}%, Dist = {h['avg_distance']:.4f}")
        lines.append(f"    Samples: {b.get('num_evaluated', 'N/A')}")

    lines.append("=" * 100)
    return "\n".join(lines)


def generate_markdown_report(results: Dict, output_path: str):
    """Generate a markdown report."""
    lines = []
    lines.append("# H-MDP Real VLM Evaluation Results\n")
    lines.append("Generated: Auto-generated from experiment results\n")

    # Main results table
    lines.append("## Table 3: Cross-Backbone Real VLM Evaluation\n")
    lines.append("| VLM Backbone | Base Act% | Base Pt% | Base Dist | H-MDP Act% | H-MDP Pt% | H-MDP Dist | ΔPt | ΔDist |")
    lines.append("|-------------|-----------|----------|-----------|------------|-----------|------------|-----|-------|")

    model_order = ["aguvis-7b", "qwen2.5-vl-7b", "uground-7b", "gui-actor-7b", "ui-tars-1.5-7b"]

    for key in model_order:
        if key not in results:
            continue
        r = results[key]
        b, h = r["base"], r["hmdp"]
        d_pt = (h["point_accuracy"] - b["point_accuracy"]) * 100
        d_dist = h["avg_distance"] - b["avg_distance"]

        lines.append(
            f"| {r['display_name']} | "
            f"{b['action_accuracy']*100:.1f}% | {b['point_accuracy']*100:.1f}% | {b['avg_distance']:.4f} | "
            f"{h['action_accuracy']*100:.1f}% | {h['point_accuracy']*100:.1f}% | {h['avg_distance']:.4f} | "
            f"{d_pt:+.1f}% | {d_dist:+.4f} |"
        )

    # Configuration
    lines.append("\n## Configuration\n")
    lines.append("- Privacy budget (ε): see source result JSON / command line")
    lines.append("- GoT paths (k): see source result JSON / command line")
    lines.append("- Temperature (τ): see source result JSON / command line")
    lines.append("- Dataset: GUI-360 Action Prediction")

    # Observations
    lines.append("\n## Key Observations\n")
    lines.append("1. **Action Accuracy**: H-MDP maintains high action accuracy while adding privacy protection")
    lines.append("2. **Point Accuracy**: Trade-off between privacy and grounding precision")
    lines.append("3. **Note**: Interpret grounding accuracy together with the exact W_proj/Eq.8 checkpoint used.")
    lines.append("   Report the checkpoint path and whether the task/LTM heads were loaded.")

    with open(output_path, 'w') as f:
        f.write("\n".join(lines))

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive report from a result JSON")
    parser.add_argument("--results", default="results/json/real_vlm_hmdp_results.json",
                        help="Path to hmdp.run_real_vlm result JSON")
    parser.add_argument("--report-out", default="results/evaluation_report.md",
                        help="Path to write markdown report")
    parser.add_argument("--out", default="results/comprehensive_results.txt",
                        help="Path to write comprehensive text report")
    args = parser.parse_args()
    results_file = args.results

    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
        print(f"Loaded results from: {results_file}")
    else:
        print(f"Results file not found: {results_file}")
        return

    # Print all formats
    print("\n" + "=" * 100)
    print("  PAPER-FORMATTED RESULTS")
    print("=" * 100)

    # Import and use the table formatting from generate_paper_results
    from generate_paper_results import format_table_3, generate_latex_table

    print(format_table_3(results))
    print("\n" + generate_latex_table(results))

    # Summary statistics
    print(format_summary_statistics(results))

    # Generate markdown report
    md_path = args.report_out
    md_dir = os.path.dirname(md_path)
    if md_dir:
        os.makedirs(md_dir, exist_ok=True)
    generate_markdown_report(results, md_path)
    print(f"\nMarkdown report saved to: {md_path}")

    # Save comprehensive results
    output_file = args.out
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("  H-MDP REAL VLM EVALUATION - COMPREHENSIVE RESULTS\n")
        f.write("=" * 100 + "\n\n")
        f.write(format_table_3(results) + "\n\n")
        f.write(generate_latex_table(results) + "\n\n")
        f.write(format_summary_statistics(results) + "\n")

    print(f"Comprehensive results saved to: {output_file}")


if __name__ == "__main__":
    main()
