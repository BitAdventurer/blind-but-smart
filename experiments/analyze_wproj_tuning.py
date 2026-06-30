#!/usr/bin/env python3
"""
Analyze W_proj tuning results and recommend the best training configuration.

Usage:
    python3 experiments/analyze_wproj_tuning.py results/wproj_tuning/summary.json
"""

import json
import sys
from pathlib import Path


def main(summary_path: str):
    with open(summary_path, "r") as f:
        data = json.load(f)

    runs = data.get("runs", [])
    eval_cfg = data.get("eval_config", {})

    print("\n" + "=" * 90)
    print("W_proj Training Tuning Results")
    print("=" * 90)
    print(f"Evaluation config: {eval_cfg}")
    print("-" * 90)
    print(f"{'Name':<20} {'Samples':>8} {'Align':>6} {'Task':>6} {'LR':>8} "
          f"{'Act%':>8} {'Pt%':>8} {'Dist':>8} {'LTM':>5} {'Time':>7}")
    print("-" * 90)

    best_point = None
    best_action = None
    best_dist = None

    for r in runs:
        name = r.get("name", "")
        samples = r.get("num_samples", "")
        align = r.get("align_epochs", "")
        task = r.get("task_epochs", "")
        lr = r.get("lr", "")
        act = r.get("action_acc")
        pt = r.get("point_acc")
        dist = r.get("avg_dist")
        ltm = r.get("ltm_used")
        t = r.get("total_time_s")

        act_str = f"{act*100:.1f}" if act is not None else "N/A"
        pt_str = f"{pt*100:.1f}" if pt is not None else "N/A"
        dist_str = f"{dist:.4f}" if dist is not None else "N/A"
        ltm_str = str(ltm) if ltm is not None else "N/A"
        t_str = f"{t}" if t is not None else "N/A"

        print(f"{name:<20} {samples:>8} {align:>6} {task:>6} {lr:>8} "
              f"{act_str:>8} {pt_str:>8} {dist_str:>8} {ltm_str:>5} {t_str:>7}")

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
        print(f"  Highest Point Acc: {best_point['point_acc']*100:.1f}%  "
              f"(samples={best_point['num_samples']}, align={best_point['align_epochs']}, "
              f"task={best_point['task_epochs']}, lr={best_point['lr']})")
    if best_action:
        print(f"  Highest Action Acc: {best_action['action_acc']*100:.1f}%  "
              f"(samples={best_action['num_samples']}, align={best_action['align_epochs']}, "
              f"task={best_action['task_epochs']}, lr={best_action['lr']})")
    if best_dist:
        print(f"  Lowest Avg Dist: {best_dist['avg_dist']:.4f}  "
              f"(samples={best_dist['num_samples']}, align={best_dist['align_epochs']}, "
              f"task={best_dist['task_epochs']}, lr={best_dist['lr']})")

    if best_point:
        print(f"\nRecommended checkpoint: {best_point['ckpt']}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "results/wproj_tuning/summary.json"
    main(path)
