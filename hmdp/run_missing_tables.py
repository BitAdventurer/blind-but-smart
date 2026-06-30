#!/usr/bin/env python3
"""
Run the three missing paper experiments using BlindVLMPipeline + wproj_eq8.pt

Missing experiments:
  1. Table 3  : Random-ε row  (GoT+LTM, ε̄≈0.82, k=3, random per-region ε)
  2. Table B.9: wpes sensitivity sweep  (wpes ∈ {0.0, 1.0, 5.0}, 2.0=default already done)
  3. Table B.12: Grid resolution        (3×3, 7×7; 5×5 = main config, already done)

Usage:
    conda activate py358
    cd /path/to/ESWA2/code
    python -m hmdp.run_missing_tables --experiment all
    python -m hmdp.run_missing_tables --experiment random_eps
    python -m hmdp.run_missing_tables --experiment wpes
    python -m hmdp.run_missing_tables --experiment grid
"""

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import numpy as np
import torch
from PIL import Image
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hmdp.run_real_vlm import (
    _parse_sample, _normalize_bbox, _compute_metrics, _print_progress,
    got_aggregate, ALPHA_W, T_AGG, TAU_LTM,
)
from hmdp.vlm_inference import ACTION_TYPES, point_in_bbox, point_distance_to_center
from hmdp.ltm import LongTermMemory
from hmdp_sim.sac_policy import SACMetaPolicy


# ═══════════════════════════════════════════════════════════════════════
#  Shared: load BlindVLMPipeline once and re-use across conditions
# ═══════════════════════════════════════════════════════════════════════

def _load_blind_pipe(model_key: str, device: str, wproj_ckpt: str,
                     grid: int = 5):
    from hmdp.blind_vlm import BlindVLMPipeline
    pipe = BlindVLMPipeline(model_key, device=device, grid=grid)
    pipe.load_models()
    if wproj_ckpt and os.path.exists(wproj_ckpt):
        pipe.load_projection(wproj_ckpt)
        print(f"  [pipe] loaded W_proj from {wproj_ckpt}")
    else:
        raise FileNotFoundError(
            f"wproj_ckpt not found: {wproj_ckpt}. Missing-table experiments "
            "must use a trained W_proj checkpoint."
        )
    return pipe


def _fresh_ltm(device: str, capacity: int = 10000, top_k: int = 8):
    return LongTermMemory(embedding_dim=256, capacity=capacity, top_k=top_k).to(device)


# ═══════════════════════════════════════════════════════════════════════
#  Core evaluation loop (blind, configurable)
# ═══════════════════════════════════════════════════════════════════════

def _eval_blind(pipe, ltm, data: List[Dict], indices: np.ndarray,
                image_base: str,
                eps_fn,          # callable(i, phi_clean) -> (eps_vec, k_t)
                label: str = "",
                log_every: int = 10) -> Dict:
    """
    Generic blind-pipeline evaluation loop.

    eps_fn(step_i, phi_clean) must return (eps_vec: List[float], k_t: int).
    """
    correct_actions = point_hits = joint_successes = evaluated = skipped = ltm_used = 0
    distances = []
    epsilons_used = []

    for i, idx in enumerate(indices):
        sample = data[idx]
        instruction, gt_action, gt_bbox_raw, img_path = _parse_sample(sample, image_base)
        if img_path is None or gt_bbox_raw is None:
            skipped += 1
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            img_w, img_h = img.size
        except Exception:
            skipped += 1
            continue

        gt_bbox_norm = _normalize_bbox(gt_bbox_raw, img_w, img_h)

        try:
            with torch.no_grad():
                e_regions = pipe.dino.extract_regions(img)       # (M, 1024)
                phi_clean = pipe.proxy_encoder(e_regions)        # (M, d)

            eps_vec, k_t = eps_fn(i, phi_clean)
            step_eps_mean = float(np.mean(eps_vec))
            epsilons_used.append(step_eps_mean)

            phi_private = pipe.ldp.privatize_regions(
                phi_clean.unsqueeze(0), eps_vec).squeeze(0)      # (M, d)

            theta_hat = pipe.task_direction(instruction)
            paths = pipe.vlm_got_k_paths_blind(
                phi_private, instruction, img_w, img_h,
                k=k_t, temperature=0.5)
        except Exception as e:
            print(f"    [ERROR] idx={idx}: {e}")
            skipped += 1
            continue

        pred_action, pred_point, u_t = got_aggregate(
            paths, phi_private, theta_hat.detach().cpu(),
            alpha_w=ALPHA_W, T_agg=T_AGG, grid=pipe.grid)

        if u_t > TAU_LTM and len(ltm.episodes) > 0:
            _ = ltm.generate_prior(phi_private.mean(dim=0))
            ltm_used += 1

        evaluated += 1
        action_ok = (pred_action == gt_action)
        correct_actions += int(action_ok)

        if pred_point is not None:
            hit = point_in_bbox(pred_point, gt_bbox_norm)
            dist = point_distance_to_center(pred_point, gt_bbox_norm)
        else:
            hit, dist = False, 1.0

        point_hits += int(hit)
        joint_successes += int(action_ok and hit)
        distances.append(dist)

        if pred_point is not None and (hit and action_ok):
            ltm.store_episode(
                state_embedding=phi_private.mean(dim=0).detach().cpu(),
                action_taken=ACTION_TYPES.index(pred_action) if pred_action in ACTION_TYPES else 0,
                bbox_target=torch.tensor(pred_point + pred_point),
                epsilon_used=step_eps_mean, k_used=k_t, reward=1.0,
                sensitivity=0.5, uncertainty=u_t, success=True)

        if (i + 1) % log_every == 0:
            _print_progress(i + 1, len(indices), evaluated, correct_actions,
                            point_hits, distances, ltm_count=ltm_used)

    metrics = _compute_metrics(evaluated, correct_actions, point_hits, distances,
                               skipped, joint_successes)
    avg_eps = float(np.mean(epsilons_used)) if epsilons_used else 1.0
    metrics["avg_epsilon"] = avg_eps
    metrics["ltm_retrievals"] = ltm_used

    M = pipe.grid * pipe.grid
    metrics["pes"] = (metrics["success_rate"] * M / avg_eps) if avg_eps > 0 else 0.0
    metrics["M"] = M
    metrics["label"] = label
    return metrics


# ═══════════════════════════════════════════════════════════════════════
#  Experiment 1: Table 3 – Random-ε row
# ═══════════════════════════════════════════════════════════════════════

def run_random_eps(model_key: str, data: List[Dict], indices: np.ndarray,
                   image_base: str, device: str, wproj_ckpt: str,
                   epsilon_levels: List[float] = None,
                   target_eps_mean: float = 0.82,
                   k_fixed: int = 3,
                   seed: int = 42) -> Dict:
    """Table 3 Random-ε row: GoT+LTM with budget-matched random per-region ε."""
    if epsilon_levels is None:
        epsilon_levels = [0.1, 0.5, 1.0, 2.5, 5.0]

    print("\n" + "=" * 70)
    print("  Table 3 — Random-ε (GoT+LTM, ε̄≈0.82, k=3)")
    print("=" * 70)

    pipe = _load_blind_pipe(model_key, device, wproj_ckpt)
    ltm  = _fresh_ltm(device)
    rng  = np.random.RandomState(seed)
    M    = pipe.grid * pipe.grid

    def eps_fn(i, phi_clean):
        for _ in range(20):
            eps_vec = rng.choice(epsilon_levels, size=M).tolist()
            if abs(np.mean(eps_vec) - target_eps_mean) / target_eps_mean < 0.05:
                break
        return eps_vec, k_fixed

    t0 = time.time()
    r = _eval_blind(pipe, ltm, data, indices, image_base, eps_fn,
                    label="Random-ε (GoT+LTM)")
    r["time_sec"] = time.time() - t0

    print(f"\n  → SR={r['success_rate']*100:.1f}%  Grnd={r['grounding_accuracy']*100:.1f}%"
          f"  ε̄={r['avg_epsilon']:.3f}  PES={r['pes']:.2f}  ({r['time_sec']:.0f}s)")

    pipe.vlm.unload()
    pipe.dino.unload()
    return r


# ═══════════════════════════════════════════════════════════════════════
#  Experiment 2: Table B.9 – wpes sensitivity sweep
# ═══════════════════════════════════════════════════════════════════════

def run_wpes_sweep(model_key: str, data: List[Dict], indices: np.ndarray,
                   image_base: str, device: str, wproj_ckpt: str,
                   wpes_values: List[float] = None,
                   epsilon: float = 1.0,
                   k: int = 5) -> Dict:
    """Table B.9: wpes sensitivity — H-MDP at ε=1.0 for each wpes."""
    if wpes_values is None:
        wpes_values = [0.0, 1.0, 5.0]   # 2.0 = default, already in remaining_tables_results

    print("\n" + "=" * 70)
    print("  Table B.9 — wpes sensitivity sweep")
    print("=" * 70)

    pipe = _load_blind_pipe(model_key, device, wproj_ckpt)
    results = {}

    def make_eps_fn(eps, k_):
        def eps_fn(i, phi_clean):
            M = phi_clean.shape[0]
            return [eps] * M, k_
        return eps_fn

    for wpes in wpes_values:
        label = f"wpes={wpes:.1f}" + (" [default]" if wpes == 2.0 else "")
        print(f"\n  [{wpes_values.index(wpes)+1}/{len(wpes_values)}] {label}  (ε={epsilon}, k={k})")

        ltm = _fresh_ltm(device)
        t0  = time.time()
        r = _eval_blind(pipe, ltm, data, indices, image_base,
                        make_eps_fn(epsilon, k), label=label)
        r["time_sec"] = time.time() - t0
        r["wpes"] = wpes

        # Terminal bonus (informational — wpes affects training, not inference SR)
        r["r_pes_bonus"] = SACMetaPolicy.compute_terminal_pes_bonus(
            sr_t=r["success_rate"],
            eps_means=[r["avg_epsilon"]],
            w_pes=wpes,
            num_regions=r["M"],
        )
        results[label] = r
        print(f"    → SR={r['success_rate']*100:.1f}%  ε̄={r['avg_epsilon']:.2f}"
              f"  PES={r['pes']:.2f}  r_pes_bonus={r['r_pes_bonus']:.4f}  ({r['time_sec']:.0f}s)")

    pipe.vlm.unload()
    pipe.dino.unload()
    return results


# ═══════════════════════════════════════════════════════════════════════
#  Experiment 3: Table B.12 – grid resolution
# ═══════════════════════════════════════════════════════════════════════

def run_grid_resolution(model_key: str, data: List[Dict], indices: np.ndarray,
                        image_base: str, device: str, wproj_ckpt: str,
                        grids: List[Tuple[int, int]] = None,
                        epsilon: float = 1.0,
                        k: int = 5) -> Dict:
    """Table B.12: grid resolution sensitivity (3×3, 5×5, 7×7)."""
    if grids is None:
        grids = [(3, 3), (5, 5), (7, 7)]

    print("\n" + "=" * 70)
    print("  Table B.12 — Grid resolution sensitivity")
    print("=" * 70)

    results = {}

    for (rows, cols) in grids:
        M = rows * cols
        label = f"{rows}×{cols}"
        print(f"\n  Grid {label} (M={M})")
        print(f"  {'-'*50}")

        assert rows == cols, f"BlindVLMPipeline only supports square grids; got {rows}x{cols}"
        pipe = _load_blind_pipe(model_key, device, wproj_ckpt, grid=rows)
        ltm = _fresh_ltm(device)

        def make_eps_fn(eps, k_):
            def eps_fn(i, phi_clean):
                return [eps] * phi_clean.shape[0], k_
            return eps_fn

        t0 = time.time()
        r = _eval_blind(pipe, ltm, data, indices, image_base,
                        make_eps_fn(epsilon, k), label=label)
        elapsed = time.time() - t0
        r["time_sec"] = elapsed
        r["latency_per_step"] = elapsed / max(r.get("num_evaluated", 1), 1)
        r["grid"] = f"{rows}x{cols}"
        r["action_dim"] = M + 1

        results[label] = r
        print(f"    → SR={r['success_rate']*100:.1f}%  PES={r['pes']:.2f}"
              f"  ε̄={r['avg_epsilon']:.2f}  lat={r['latency_per_step']:.2f}s/step  ({elapsed:.0f}s)")

        pipe.vlm.unload()
        pipe.dino.unload()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


# ═══════════════════════════════════════════════════════════════════════
#  Summary printers
# ═══════════════════════════════════════════════════════════════════════

def print_random_eps_summary(r: Dict):
    print(f"\n  Random-ε row:")
    print(f"    SR={r['success_rate']*100:.1f}%  Grnd={r['grounding_accuracy']*100:.1f}%"
          f"  ε̄={r['avg_epsilon']:.3f}  PES={r['pes']:.2f}")


def print_wpes_summary(results: Dict):
    print(f"\n  {'wpes':<14} {'SR↑':>7} {'ε̄↓':>6} {'PES↑':>7} {'r_pes_bonus':>13}")
    print(f"  {'-'*50}")
    for name, r in results.items():
        print(f"  {name:<14} {r['success_rate']*100:>6.1f}%"
              f" {r['avg_epsilon']:>6.2f} {r['pes']:>7.2f}"
              f" {r.get('r_pes_bonus',0):>13.4f}")


def print_grid_summary(results: Dict):
    print(f"\n  {'Grid':<10} {'M':>4} {'ActionDim':>10} {'SR↑':>7} {'PES↑':>7} {'lat/step':>10}")
    print(f"  {'-'*52}")
    for name, r in results.items():
        print(f"  {name:<10} {r['M']:>4} {r['action_dim']:>10}"
              f" {r['success_rate']*100:>6.1f}% {r['pes']:>7.2f}"
              f" {r.get('latency_per_step',0):>9.2f}s")


# ═══════════════════════════════════════════════════════════════════════
#  Save
# ═══════════════════════════════════════════════════════════════════════

def _save(data: Dict, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def _ser(obj):
        if isinstance(obj, (np.integer,)):    return int(obj)
        if isinstance(obj, (np.floating,)):   return float(obj)
        if isinstance(obj, np.ndarray):       return obj.tolist()
        if isinstance(obj, torch.Tensor):     return obj.tolist()
        return obj

    # Merge with existing file instead of overwriting
    existing = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                existing = json.load(f)
        except Exception:
            pass
    existing.update(data)

    with open(path, "w") as f:
        json.dump(existing, f, indent=2, default=_ser)
    print(f"  Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Run missing paper experiments (Blind pipeline)")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "random_eps", "wpes", "grid"],
                        help="Which experiment to run")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="qwen2.5-vl-7b")
    parser.add_argument("--wproj-ckpt", default="checkpoints/wproj_eq8.pt")
    parser.add_argument("--num-samples", type=int, default=150)
    parser.add_argument("--test-split", type=str,
                        default="results/json/test_split_1000.json",
                        help="Fixed test split JSON (preferred for reproducibility)")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--wpes-values", type=float, nargs="+", default=[0.0, 1.0, 5.0])
    parser.add_argument("--grids", type=str, nargs="+", default=["3x3", "5x5", "7x7"])
    parser.add_argument("--data-path", type=str,
                        default="gui360_full/processed_data/action_prediction_train_resize/training_data.json")
    parser.add_argument("--image-base", type=str,
                        default="gui360_full/processed_data/action_prediction_train_resize/")
    parser.add_argument("--out", type=str, default="results/missing_tables_results.json")
    args = parser.parse_args()

    print("=" * 70)
    print("  Missing Table Experiments — BlindVLMPipeline")
    print(f"  Experiment: {args.experiment}")
    print(f"  Model: {args.model}  |  wproj: {args.wproj_ckpt}")
    print("=" * 70)

    # Load dataset
    with open(args.data_path) as f:
        data = json.load(f)
    print(f"  Dataset: {len(data)} samples")

    if args.test_split and os.path.exists(args.test_split):
        with open(args.test_split) as f:
            indices = np.array(json.load(f))
        print(f"  Using fixed split: {len(indices)} samples from {args.test_split}")
    else:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(data), min(args.num_samples, len(data)), replace=False)
        print(f"  Random split: {len(indices)} samples (seed=42)")

    # Parse grids
    parsed_grids = []
    for g in args.grids:
        r, c = g.lower().split("x")
        parsed_grids.append((int(r), int(c)))

    all_results = {}
    t_total = time.time()

    if args.experiment in ("all", "random_eps"):
        r = run_random_eps(
            args.model, data, indices, args.image_base,
            args.device, args.wproj_ckpt,
        )
        all_results["random_eps"] = r
        print_random_eps_summary(r)
        _save(all_results, args.out)

    if args.experiment in ("all", "wpes"):
        r = run_wpes_sweep(
            args.model, data, indices, args.image_base,
            args.device, args.wproj_ckpt,
            wpes_values=args.wpes_values,
            epsilon=args.epsilon, k=args.k,
        )
        all_results["wpes_sweep"] = r
        print_wpes_summary(r)
        _save(all_results, args.out)

    if args.experiment in ("all", "grid"):
        r = run_grid_resolution(
            args.model, data, indices, args.image_base,
            args.device, args.wproj_ckpt,
            grids=parsed_grids,
            epsilon=args.epsilon, k=args.k,
        )
        all_results["grid_resolution"] = r
        print_grid_summary(r)
        _save(all_results, args.out)

    elapsed = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"  DONE — {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Results saved to {args.out}")

    if "random_eps" in all_results:
        print("\n── Random-ε (Table 3) ──")
        print_random_eps_summary(all_results["random_eps"])
    if "wpes_sweep" in all_results:
        print("\n── wpes sweep (Table B.9) ──")
        print_wpes_summary(all_results["wpes_sweep"])
    if "grid_resolution" in all_results:
        print("\n── Grid resolution (Table B.12) ──")
        print_grid_summary(all_results["grid_resolution"])


if __name__ == "__main__":
    main()
