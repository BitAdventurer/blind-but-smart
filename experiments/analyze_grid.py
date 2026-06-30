#!/usr/bin/env python3
"""
Analyze epsilon/k grid search results and recommend the best configuration.

Usage:
    python3 experiments/analyze_grid.py results/epsilon_k_grid/summary.json
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


def main(summary_path: str):
    with open(summary_path, "r") as f:
        data = json.load(f)

    runs = data.get("runs", [])
    if not runs:
        print("No runs found in summary.")
        return

    # Print table
    print("\n" + "=" * 90)
    print("Epsilon / k Grid Search Results")
    print("=" * 90)
    print(f"{'Run':<35} {'Epsilon':>8} {'k':>4} {'Temp':>6} {'Act%':>8} {'Pt%':>8} {'Dist':>8} {'LTM':>5} {'Time':>6}")
    print("-" * 90)

    best_point = None
    best_action = None
    best_dist = None

    for r in runs:
        name = r.get("run_name", "")
        eps = r.get("epsilon", "")
        k = r.get("k", "")
        temp = r.get("temperature", "")
        act = r.get("action_acc")
        pt = r.get("point_acc")
        dist = r.get("avg_dist")
        ltm = r.get("ltm_used")
        t = r.get("total_time_s")

        act_str = f"{act:.1f}" if act is not None else "N/A"
        pt_str = f"{pt:.1f}" if pt is not None else "N/A"
        dist_str = f"{dist:.4f}" if dist is not None else "N/A"
        ltm_str = str(ltm) if ltm is not None else "N/A"
        t_str = f"{t}" if t is not None else "N/A"

        print(f"{name:<35} {eps:>8} {k:>4} {temp:>6} {act_str:>8} {pt_str:>8} {dist_str:>8} {ltm_str:>5} {t_str:>6}")

        if pt is not None:
            if best_point is None or pt > best_point.get("point_acc", -1):
                best_point = r
        if act is not None:
            if best_action is None or act > best_action.get("action_acc", -1):
                best_action = r
        if dist is not None:
            if best_dist is None or dist < best_dist.get("avg_dist", float("inf")):
                best_dist = r

    print("=" * 90)
    print("\nBest configurations:")
    if best_point:
        print(f"  Highest Point Acc: {best_point['point_acc']:.1f}%  "
              f"(eps={best_point['epsilon']}, k={best_point['k']}, temp={best_point['temperature']}, "
              f"ckpt={best_point['ckpt_name']})")
    if best_action:
        print(f"  Highest Action Acc: {best_action['action_acc']:.1f}%  "
              f"(eps={best_action['epsilon']}, k={best_action['k']}, temp={best_action['temperature']}, "
              f"ckpt={best_action['ckpt_name']})")
    if best_dist:
        print(f"  Lowest Avg Dist: {best_dist['avg_dist']:.4f}  "
              f"(eps={best_dist['epsilon']}, k={best_dist['k']}, temp={best_dist['temperature']}, "
              f"ckpt={best_dist['ckpt_name']})")

    # Aggregate by checkpoint
    print("\nPer-checkpoint averages:")
    by_ckpt = defaultdict(lambda: {"count": 0, "point_sum": 0.0, "action_sum": 0.0, "dist_sum": 0.0})
    for r in runs:
        if r.get("point_acc") is None:
            continue
        c = by_ckpt[r["ckpt_name"]]
        c["count"] += 1
        c["point_sum"] += r["point_acc"]
        c["action_sum"] += r["action_acc"]
        c["dist_sum"] += r["avg_dist"]

    for ckpt, agg in by_ckpt.items():
        n = agg["count"]
        print(f"  {ckpt}: n={n}, avg_point={agg['point_sum']/n:.1f}%, "
              f"avg_action={agg['action_sum']/n:.1f}%, avg_dist={agg['dist_sum']/n:.4f}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "results/epsilon_k_grid/summary.json"
    main(path)
