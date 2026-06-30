#!/usr/bin/env python3
"""
Run real VLM experiments for ALL paper tables (Table 1, 2, 3).

Table 1 (RQ1): Privacy-Utility comparison using Qwen-2.5-VL-7B
  - Oracle (no LDP, single call)
  - Static-High (ε=0.5, k=5, GoT only)
  - Static-Low (ε=5.0, k=5, GoT only)
  - Rule-based Adaptive (sensitivity-based ε, k=5, GoT only)
  - H-MDP (Ours) (ε=1.0, k=5, GoT+LTM)

Table 2 (RQ2): Ablation at ε=1.0 using Qwen-2.5-VL-7B
  - Full H-MDP (GoT+LTM, k=5)
  - w/o GoT (k=1, LTM only)
  - w/o LTM (GoT, k=5, no LTM)
  - No GoT/LTM (k=1, no LTM)

Table 3 (RQ4): Cross-backbone (4 models, base vs H-MDP)

Usage:
    conda activate py358
    python -m hmdp.run_all_tables --device cuda --num-samples 150
    python -m hmdp.run_all_tables --device cuda --table 1  # run only Table 1
"""

import argparse
import gc
import json
import os
import sys
import time
import warnings
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hmdp.run_real_vlm import (
    RealHMDPPipeline,
    vlm_got_k_paths, got_aggregate,
    _parse_sample, _normalize_bbox, _compute_metrics, _print_progress,
    ALPHA_W, T_AGG, TAU_LTM,
    set_global_seed, DEFAULT_SEED,
)
from hmdp.vlm_inference import (
    MODEL_REGISTRY, ACTION_TYPES,
    point_in_bbox, point_distance_to_center,
)
from hmdp.ltm import LongTermMemory


# ═══════════════════════════════════════════════════════════════════════
#  Extended Pipeline with configurable GoT / LTM
# ═══════════════════════════════════════════════════════════════════════

def evaluate_configurable(
    pipeline: RealHMDPPipeline,
    data: List[Dict],
    indices: np.ndarray,
    image_base: str,
    epsilon: float = 1.0,
    k: int = 5,
    temperature: float = 0.5,
    use_got: bool = True,
    use_ltm: bool = True,
    label: str = "",
) -> Dict:
    """
    Flexible H-MDP evaluation with configurable GoT and LTM.

    Args:
        pipeline: loaded RealHMDPPipeline
        epsilon: LDP privacy budget
        k: number of GoT paths (1 = no GoT)
        use_got: whether to use GoT aggregation (if False, k is forced to 1)
        use_ltm: whether to use LTM priors
    """
    if not use_got:
        k = 1

    correct_actions = 0
    point_hits = 0
    joint_successes = 0
    distances = []
    skipped = 0
    evaluated = 0
    ltm_used = 0
    epsilons_used = []

    # Reset LTM if needed for clean ablation
    if not use_ltm:
        pipeline.ltm = LongTermMemory(
            embedding_dim=256, capacity=500, top_k=5
        ).to(pipeline.device)

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

        # ── Step 1: Per-region DINOv2 grid latent (Eq. 3) ──
        e_regions = pipeline.region_encoder.extract_regions(img)     # (M, 1024)
        phi_grid = pipeline.proxy_encoder(e_regions)                 # (M, 256)

        # ── Step 2: Per-region LDP noise injection (Eq. 4) ──
        psi_grid = pipeline.ldp.privatize_regions(
            phi_grid.unsqueeze(0), [epsilon] * phi_grid.shape[0]
        ).squeeze(0)                                                 # (M, 256)
        state_embedding = psi_grid.mean(dim=0)                       # (256,)
        epsilons_used.append(epsilon)

        # ── Step 3: GoT k-path reasoning ──
        try:
            paths = vlm_got_k_paths(
                pipeline.vlm, img_path, instruction,
                k=k, temperature=temperature,
            )
        except Exception:
            skipped += 1
            continue

        # ── Step 4: GoT aggregation (Eqs. got_score / got_agg) ──
        # Semantic term ⟨psi_j, theta_hat⟩ is active when a trained task_head
        # is loaded; task_direction() returns None otherwise, and
        # got_aggregate() then falls back to logit-only scoring.
        if use_got and k > 1:
            theta_hat = pipeline.task_direction(instruction)
            pred_action, pred_point, u_t = got_aggregate(
                paths, psi_grid, theta_hat,
                alpha_w=ALPHA_W, T_agg=T_AGG, grid=pipeline.grid,
            )
        else:
            # Single path — greedy prediction, no decision-level variance.
            pred_action = paths[0]["action_type"]
            pred_point = paths[0]["pred_point"]
            u_t = 0.0

        # ── Step 5: Conditional LTM gate (Eq. 7/8) ──
        # When a trained strategic-prediction head θ_pred is present on the
        # pipeline, refine C* from the retrieved memory; otherwise (this table
        # harness does not load one by default) leave the GoT coordinate
        # unchanged. The faithful Eq.8 refinement is evaluated in the blind
        # pipeline (run_real_vlm.evaluate_hmdp_blind).
        predictor = getattr(pipeline, "ltm_predictor", None)
        predictor_ready = (predictor is not None
                           and getattr(pipeline, "_ltm_predictor_trained", False))
        if (use_ltm and predictor_ready and u_t > TAU_LTM
                and len(pipeline.ltm.episodes) > 0 and pred_point is not None):
            with torch.no_grad():
                phi_summary = state_embedding.to(pipeline.device)
                ret_coord, ret_emb, conf = pipeline.ltm.retrieve_prior_features(
                    phi_summary, device=pipeline.device)
                c_star = torch.tensor(pred_point, device=pipeline.device, dtype=torch.float32)
                refined = predictor(
                    phi_summary.float(), c_star, ret_coord.float(),
                    ret_emb.float(), conf.float())
            pred_point = refined.detach().cpu().tolist()
            ltm_used += 1

        evaluated += 1
        action_ok = (pred_action == gt_action)
        correct_actions += int(action_ok)

        if pred_point is not None:
            hit = point_in_bbox(pred_point, gt_bbox_norm)
            dist = point_distance_to_center(pred_point, gt_bbox_norm)
        else:
            hit = False
            dist = 1.0

        point_hits += int(hit)
        joint_successes += int(action_ok and hit)
        distances.append(dist)

        # ── Step 6: Store in LTM ──
        if use_ltm and pred_point is not None:
            ep_reward = 1.0 if (hit and action_ok) else (0.5 if dist < 0.3 else 0.0)
            if ep_reward > 0:
                pipeline.ltm.store_episode(
                    state_embedding=state_embedding,
                    action_taken=ACTION_TYPES.index(pred_action) if pred_action in ACTION_TYPES else 0,
                    bbox_target=torch.tensor(pred_point + pred_point),
                    epsilon_used=epsilon, k_used=k, reward=ep_reward,
                    sensitivity=0.5, uncertainty=u_t, success=(hit and action_ok),
                )

        if (i + 1) % 30 == 0:
            _print_progress(i+1, len(indices), evaluated, correct_actions,
                            point_hits, distances, ltm_count=ltm_used)

    metrics = _compute_metrics(evaluated, correct_actions, point_hits, distances, skipped, joint_successes)
    metrics["epsilon"] = epsilon
    metrics["k"] = k
    metrics["use_got"] = use_got
    metrics["use_ltm"] = use_ltm
    metrics["ltm_episodes"] = len(pipeline.ltm.episodes)
    metrics["ltm_retrievals"] = ltm_used
    metrics["avg_epsilon"] = float(np.mean(epsilons_used)) if epsilons_used else epsilon

    # PES = SR / ε̄  (paper Eq. 15, simplified for uniform single-step ε)
    avg_eps = metrics["avg_epsilon"]
    if avg_eps > 0 and avg_eps < 100:
        metrics["pes"] = metrics["success_rate"] / avg_eps
    else:
        metrics["pes"] = 0.0

    return metrics


def evaluate_rule_based(
    pipeline: RealHMDPPipeline,
    data: List[Dict],
    indices: np.ndarray,
    image_base: str,
    k: int = 5,
    temperature: float = 0.5,
    label: str = "",
) -> Dict:
    """
    Rule-based adaptive: ε chosen by simple sensitivity heuristic.
    High sensitivity → low ε; Low sensitivity → high ε.
    No LTM, GoT only.
    """
    correct_actions = 0
    point_hits = 0
    joint_successes = 0
    distances = []
    skipped = 0
    evaluated = 0
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

        # ── Per-region DINOv2 grid latent (Eq. 3) ──
        e_regions = pipeline.region_encoder.extract_regions(img)     # (M, 1024)
        phi_grid = pipeline.proxy_encoder(e_regions)                 # (M, 256)

        # Estimate sensitivity from the grid-mean feature norm (proxy).
        sensitivity = torch.sigmoid(phi_grid.mean(dim=0).norm() - 5.0).item()

        # Rule-based ε selection
        if sensitivity > 0.7:
            epsilon = 0.5
        elif sensitivity > 0.4:
            epsilon = 1.0
        else:
            epsilon = 3.0

        epsilons_used.append(epsilon)

        # ── Per-region LDP noise injection (Eq. 4) ──
        psi_grid = pipeline.ldp.privatize_regions(
            phi_grid.unsqueeze(0), [epsilon] * phi_grid.shape[0]
        ).squeeze(0)                                                 # (M, 256)

        # GoT k-path
        try:
            paths = vlm_got_k_paths(
                pipeline.vlm, img_path, instruction,
                k=k, temperature=temperature,
            )
        except Exception:
            skipped += 1
            continue

        # Aggregate (no LTM). U_t is unused here (no LTM gate). Semantic term
        # ⟨psi_j, theta_hat⟩ is active when a trained task_head is loaded;
        # task_direction() returns None otherwise (logit-only fallback).
        theta_hat = pipeline.task_direction(instruction)
        pred_action, pred_point, _u_t = got_aggregate(
            paths, psi_grid, theta_hat,
            alpha_w=ALPHA_W, T_agg=T_AGG, grid=pipeline.grid,
        )

        evaluated += 1
        action_ok = (pred_action == gt_action)
        correct_actions += int(action_ok)

        if pred_point is not None:
            hit = point_in_bbox(pred_point, gt_bbox_norm)
            dist = point_distance_to_center(pred_point, gt_bbox_norm)
        else:
            hit = False
            dist = 1.0

        point_hits += int(hit)
        joint_successes += int(action_ok and hit)
        distances.append(dist)

        if (i + 1) % 30 == 0:
            _print_progress(i+1, len(indices), evaluated, correct_actions,
                            point_hits, distances)

    metrics = _compute_metrics(evaluated, correct_actions, point_hits, distances, skipped, joint_successes)
    avg_eps = float(np.mean(epsilons_used)) if epsilons_used else 1.0
    metrics["epsilon"] = "adaptive"
    metrics["avg_epsilon"] = avg_eps
    metrics["k"] = k
    metrics["use_got"] = True
    metrics["use_ltm"] = False

    # PES = SR / ε̄  (paper Eq. 15)
    if avg_eps > 0:
        metrics["pes"] = metrics["success_rate"] / avg_eps
    else:
        metrics["pes"] = 0.0

    return metrics


def evaluate_random_epsilon(
    pipeline: RealHMDPPipeline,
    data: List[Dict],
    indices: np.ndarray,
    image_base: str,
    epsilon_levels: List[float],
    target_eps_mean: float = 0.82,
    k_fixed: int = 3,
    temperature: float = 0.5,
    seed: int = 42,
    label: str = "",
) -> Dict:
    """Random-ε ablation (Table 3 / Table B.9 perception-axis row).

    Retains full GoT+LTM pipeline but replaces learned meta-policy budget
    allocation with a per-region uniform random draw from epsilon_levels.
    Trajectory-average budget is matched to target_eps_mean by rejection
    sampling the per-step mean; k is fixed at k_fixed (paper: k=3).
    """
    rng = np.random.RandomState(seed)
    correct_actions = 0
    point_hits = 0
    joint_successes = 0
    distances = []
    skipped = 0
    evaluated = 0
    ltm_used = 0
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

        # ── Per-region DINOv2 grid latent ──
        e_regions = pipeline.region_encoder.extract_regions(img)
        phi_grid = pipeline.proxy_encoder(e_regions)                 # (M, 256)
        M = phi_grid.shape[0]

        # ── Random per-region ε draw, matched to target_eps_mean ──
        # Draw until step-mean is within 5% of target, max 20 tries.
        for _ in range(20):
            eps_vec = rng.choice(epsilon_levels, size=M).tolist()
            if abs(np.mean(eps_vec) - target_eps_mean) / target_eps_mean < 0.05:
                break

        step_eps_mean = float(np.mean(eps_vec))
        epsilons_used.append(step_eps_mean)

        # ── Per-region LDP noise ──
        psi_grid = pipeline.ldp.privatize_regions(
            phi_grid.unsqueeze(0), eps_vec
        ).squeeze(0)
        state_embedding = psi_grid.mean(dim=0)

        # ── GoT k_fixed paths ──
        try:
            paths = vlm_got_k_paths(
                pipeline.vlm, img_path, instruction,
                k=k_fixed, temperature=temperature,
            )
        except Exception:
            skipped += 1
            continue

        # ── GoT aggregation ──
        if k_fixed > 1:
            theta_hat = pipeline.task_direction(instruction)
            pred_action, pred_point, u_t = got_aggregate(
                paths, psi_grid, theta_hat,
                alpha_w=ALPHA_W, T_agg=T_AGG, grid=pipeline.grid,
            )
        else:
            pred_action = paths[0]["action_type"]
            pred_point = paths[0]["pred_point"]
            u_t = 0.0

        # ── LTM gate (Eq. 7/8) ──
        # Refine C* with the trained θ_pred when present; otherwise leave the
        # GoT coordinate unchanged (the faithful Eq.8 path is the blind
        # pipeline). This harness does not load a trained head by default.
        predictor = getattr(pipeline, "ltm_predictor", None)
        predictor_ready = (predictor is not None
                           and getattr(pipeline, "_ltm_predictor_trained", False))
        if (predictor_ready and u_t > TAU_LTM
                and len(pipeline.ltm.episodes) > 0 and pred_point is not None):
            with torch.no_grad():
                phi_summary = state_embedding.to(pipeline.device)
                ret_coord, ret_emb, conf = pipeline.ltm.retrieve_prior_features(
                    phi_summary, device=pipeline.device)
                c_star = torch.tensor(pred_point, device=pipeline.device, dtype=torch.float32)
                refined = predictor(
                    phi_summary.float(), c_star, ret_coord.float(),
                    ret_emb.float(), conf.float())
            pred_point = refined.detach().cpu().tolist()
            ltm_used += 1

        evaluated += 1
        action_ok = (pred_action == gt_action)
        correct_actions += int(action_ok)

        if pred_point is not None:
            hit = point_in_bbox(pred_point, gt_bbox_norm)
            dist = point_distance_to_center(pred_point, gt_bbox_norm)
        else:
            hit = False
            dist = 1.0

        point_hits += int(hit)
        joint_successes += int(action_ok and hit)
        distances.append(dist)

        if pred_point is not None and (hit and action_ok):
            ep_reward = 1.0
            pipeline.ltm.store_episode(
                state_embedding=state_embedding,
                action_taken=ACTION_TYPES.index(pred_action) if pred_action in ACTION_TYPES else 0,
                bbox_target=torch.tensor(pred_point + pred_point),
                epsilon_used=step_eps_mean, k_used=k_fixed, reward=ep_reward,
                sensitivity=0.5, uncertainty=u_t, success=True,
            )

        if (i + 1) % 30 == 0:
            _print_progress(i+1, len(indices), evaluated, correct_actions,
                            point_hits, distances, ltm_count=ltm_used)

    metrics = _compute_metrics(evaluated, correct_actions, point_hits, distances, skipped, joint_successes)
    avg_eps = float(np.mean(epsilons_used)) if epsilons_used else target_eps_mean
    metrics["epsilon"] = "random"
    metrics["avg_epsilon"] = avg_eps
    metrics["k"] = k_fixed
    metrics["use_got"] = True
    metrics["use_ltm"] = True
    metrics["ltm_retrievals"] = ltm_used
    if avg_eps > 0:
        metrics["pes"] = metrics["success_rate"] / avg_eps
    else:
        metrics["pes"] = 0.0
    return metrics


# ═══════════════════════════════════════════════════════════════════════
#  Table Runners
# ═══════════════════════════════════════════════════════════════════════

def run_table1(pipeline, data, indices, image_base, k=5, temp=0.5):
    """Table 1 (paper Table 2): Main Comparative Results (RQ1) using one model."""
    print("\n" + "=" * 70)
    print("  TABLE 1: Main Comparative Results (RQ1)")
    print("=" * 70)
    results = {}

    # 1) No Privacy (Single-Pass) — base single call
    print("\n  [1/6] No Privacy (Single-Pass) — base single VLM call...")
    t0 = time.time()
    r = pipeline.evaluate_base(data, indices, image_base)
    r["time_sec"] = time.time() - t0
    r["avg_epsilon"] = float("inf")
    r["pes"] = 0.0
    results["No Privacy (Single-Pass)"] = r
    print(f"    → SR={r['success_rate']*100:.1f}%, Grnd={r['grounding_accuracy']*100:.1f}% ({r['time_sec']:.0f}s)")

    # 2) No Privacy + GoT/LTM — reasoning without privacy constraint
    print("\n  [2/6] No Privacy + GoT/LTM (ε=∞, k={}, GoT+LTM)...".format(k))
    t0 = time.time()
    r = evaluate_configurable(
        pipeline, data, indices, image_base,
        epsilon=1e6, k=k, temperature=temp,
        use_got=True, use_ltm=True, label="NoPriv+GoT/LTM",
    )
    r["time_sec"] = time.time() - t0
    r["avg_epsilon"] = float("inf")
    r["pes"] = 0.0
    results["No Privacy + GoT/LTM"] = r
    print(f"    → SR={r['success_rate']*100:.1f}%, Grnd={r['grounding_accuracy']*100:.1f}% ({r['time_sec']:.0f}s)")

    # 3) High Privacy (ε=0.5)
    print("\n  [3/6] High Privacy (ε=0.5, k={}, GoT only)...".format(k))
    t0 = time.time()
    r = evaluate_configurable(
        pipeline, data, indices, image_base,
        epsilon=0.5, k=k, temperature=temp,
        use_got=True, use_ltm=False, label="Static-High",
    )
    r["time_sec"] = time.time() - t0
    results["High Privacy (ε=0.5)"] = r
    print(f"    → SR={r['success_rate']*100:.1f}%, ε̄={r['avg_epsilon']:.2f}, PES={r['pes']:.3f} ({r['time_sec']:.0f}s)")

    # 4) Low Privacy (ε=5.0)
    print("\n  [4/6] Low Privacy (ε=5.0, k={}, GoT only)...".format(k))
    t0 = time.time()
    r = evaluate_configurable(
        pipeline, data, indices, image_base,
        epsilon=5.0, k=k, temperature=temp,
        use_got=True, use_ltm=False, label="Static-Low",
    )
    r["time_sec"] = time.time() - t0
    results["Low Privacy (ε=5.0)"] = r
    print(f"    → SR={r['success_rate']*100:.1f}%, ε̄={r['avg_epsilon']:.2f}, PES={r['pes']:.3f} ({r['time_sec']:.0f}s)")

    # 5) Rule-based Adaptive
    print("\n  [5/6] Rule-based Adaptive (k={}, GoT only)...".format(k))
    t0 = time.time()
    r = evaluate_rule_based(
        pipeline, data, indices, image_base,
        k=k, temperature=temp, label="Rule-based",
    )
    r["time_sec"] = time.time() - t0
    results["Rule-based Adaptive"] = r
    print(f"    → SR={r['success_rate']*100:.1f}%, ε̄={r['avg_epsilon']:.2f}, PES={r['pes']:.3f} ({r['time_sec']:.0f}s)")

    # 6) H-MDP (Ours) — ε=1.0, GoT+LTM
    print("\n  [6/6] H-MDP (Ours) (ε=1.0, k={}, GoT+LTM)...".format(k))
    t0 = time.time()
    r = evaluate_configurable(
        pipeline, data, indices, image_base,
        epsilon=1.0, k=k, temperature=temp,
        use_got=True, use_ltm=True, label="H-MDP",
    )
    r["time_sec"] = time.time() - t0
    results["H-MDP (Ours)"] = r
    print(f"    → SR={r['success_rate']*100:.1f}%, ε̄={r['avg_epsilon']:.2f}, PES={r['pes']:.3f} ({r['time_sec']:.0f}s)")

    return results


def run_table2_with_random_eps(pipeline, data, indices, image_base, k=5, temp=0.5,
                               epsilon_levels=None):
    """Table 3 (paper): Ablation Study + Random-ε perception-axis row.

    Extends run_table2() with the Random-ε (GoT+LTM) row that isolates
    the contribution of learned adaptive budget allocation.
    """
    if epsilon_levels is None:
        epsilon_levels = [0.1, 0.5, 1.0, 2.5, 5.0]

    results = run_table2(pipeline, data, indices, image_base, k=k, temp=temp)

    print(f"\n  [5/5] Random-ε (GoT+LTM, ε̄≈0.82, k=3) — perception axis...")
    # Reset LTM for clean run
    pipeline.ltm = LongTermMemory(
        embedding_dim=256, capacity=10000, top_k=8
    ).to(pipeline.device)
    t0 = time.time()
    r = evaluate_random_epsilon(
        pipeline, data, indices, image_base,
        epsilon_levels=epsilon_levels,
        target_eps_mean=0.82,
        k_fixed=3,
        temperature=temp,
        label="Random-eps",
    )
    r["time_sec"] = time.time() - t0
    results["Random-ε (GoT+LTM)"] = r
    full_sr = results.get("Full H-MDP", {}).get("success_rate", 0) * 100
    delta = r["success_rate"] * 100 - full_sr
    print(f"    → SR={r['success_rate']*100:.1f}% ({delta:+.1f}pp), "
          f"Grnd={r['grounding_accuracy']*100:.1f}%, ε̄={r['avg_epsilon']:.2f}, "
          f"PES={r['pes']:.3f} ({r['time_sec']:.0f}s)")
    return results


def run_table2(pipeline, data, indices, image_base, k=5, temp=0.5):
    """Table 2: Ablation Study (RQ2) at ε=1.0."""
    print("\n" + "=" * 70)
    print("  TABLE 2: Ablation Study (RQ2) at ε=1.0")
    print("=" * 70)
    results = {}
    eps = 1.0

    # 1) Full H-MDP (GoT + LTM)
    print(f"\n  [1/4] Full H-MDP (GoT=✓ LTM=✓, ε={eps}, k={k})...")
    t0 = time.time()
    r = evaluate_configurable(
        pipeline, data, indices, image_base,
        epsilon=eps, k=k, temperature=temp,
        use_got=True, use_ltm=True, label="Full",
    )
    r["time_sec"] = time.time() - t0
    results["Full H-MDP"] = r
    print(f"    → SR={r['success_rate']*100:.1f}%, Grnd={r['grounding_accuracy']*100:.1f}%, PES={r['pes']:.3f} ({r['time_sec']:.0f}s)")

    # 2) w/o GoT (k=1, LTM only)
    print(f"\n  [2/4] w/o GoT (GoT=✗ LTM=✓, ε={eps}, k=1)...")
    t0 = time.time()
    r = evaluate_configurable(
        pipeline, data, indices, image_base,
        epsilon=eps, k=1, temperature=temp,
        use_got=False, use_ltm=True, label="w/o GoT",
    )
    r["time_sec"] = time.time() - t0
    results["w/o GoT"] = r
    print(f"    → SR={r['success_rate']*100:.1f}%, Grnd={r['grounding_accuracy']*100:.1f}%, PES={r['pes']:.3f} ({r['time_sec']:.0f}s)")

    # 3) w/o LTM (GoT only)
    print(f"\n  [3/4] w/o LTM (GoT=✓ LTM=✗, ε={eps}, k={k})...")
    t0 = time.time()
    r = evaluate_configurable(
        pipeline, data, indices, image_base,
        epsilon=eps, k=k, temperature=temp,
        use_got=True, use_ltm=False, label="w/o LTM",
    )
    r["time_sec"] = time.time() - t0
    results["w/o LTM"] = r
    print(f"    → SR={r['success_rate']*100:.1f}%, Grnd={r['grounding_accuracy']*100:.1f}%, PES={r['pes']:.3f} ({r['time_sec']:.0f}s)")

    # 4) No GoT & No LTM (baseline single call with LDP)
    print(f"\n  [4/4] No GoT/LTM (GoT=✗ LTM=✗, ε={eps}, k=1)...")
    t0 = time.time()
    r = evaluate_configurable(
        pipeline, data, indices, image_base,
        epsilon=eps, k=1, temperature=temp,
        use_got=False, use_ltm=False, label="No GoT/LTM",
    )
    r["time_sec"] = time.time() - t0
    results["No GoT & No LTM"] = r
    print(f"    → SR={r['success_rate']*100:.1f}%, Grnd={r['grounding_accuracy']*100:.1f}%, PES={r['pes']:.3f} ({r['time_sec']:.0f}s)")

    return results


def run_table_b9_wpes(
    pipeline: RealHMDPPipeline,
    data: List[Dict],
    indices: np.ndarray,
    image_base: str,
    wpes_values: List[float] = None,
    k: int = 5,
    temp: float = 0.5,
) -> Dict:
    """Table B.9: Reward weight sensitivity — wpes sweep.

    Evaluates H-MDP at ε=1.0 under different terminal PES bonus weights.
    The model pipeline itself is identical across rows; only the reported
    PES calculation differs (wpes affects training objective, not inference).
    We re-compute the PES metric under each wpes to show sensitivity.
    """
    from hmdp_sim.sac_policy import SACMetaPolicy

    if wpes_values is None:
        wpes_values = [0.0, 1.0, 2.0, 5.0]  # 2.0 = default

    print("\n" + "=" * 70)
    print("  TABLE B.9: Reward weight sensitivity (wpes sweep)")
    print("=" * 70)
    results = {}

    for wpes in wpes_values:
        label = f"wpes={wpes:.1f}" + (" [default]" if wpes == 2.0 else "")
        print(f"\n  [{wpes_values.index(wpes)+1}/{len(wpes_values)}] {label}")

        # Reset LTM for each condition
        pipeline.ltm = LongTermMemory(
            embedding_dim=256, capacity=10000, top_k=8
        ).to(pipeline.device)

        t0 = time.time()
        r = evaluate_configurable(
            pipeline, data, indices, image_base,
            epsilon=1.0, k=k, temperature=temp,
            use_got=True, use_ltm=True, label=label,
        )
        r["time_sec"] = time.time() - t0

        # Re-compute PES using the paper formula:
        # PES = SR * M / eps_bar  (wpes modulates training, not this metric)
        # Also compute terminal bonus at this wpes for reference
        avg_eps = r["avg_epsilon"]
        M = 25
        r["pes_paper"] = (r["success_rate"] * M / avg_eps) if avg_eps > 0 else 0.0
        # Terminal bonus contribution (informational)
        r["r_pes_bonus"] = SACMetaPolicy.compute_terminal_pes_bonus(
            sr_t=r["success_rate"],
            eps_means=[avg_eps],
            w_pes=wpes,
            num_regions=M,
        )
        r["wpes"] = wpes
        results[label] = r
        print(f"    → SR={r['success_rate']*100:.1f}%, ε̄={avg_eps:.2f}, "
              f"PES={r['pes_paper']:.2f}, r_pes_bonus={r['r_pes_bonus']:.3f} ({r['time_sec']:.0f}s)")

    return results


def run_table_b12_grid(
    data: List[Dict],
    indices: np.ndarray,
    image_base: str,
    device: str,
    model_key: str = "qwen2.5-vl-7b",
    wproj_ckpt: str = None,
    grids: List[Tuple[int, int]] = None,
    k: int = 5,
    temp: float = 0.5,
) -> Dict:
    """Table B.12: Grid resolution sensitivity (3×3, 5×5, 7×7).

    Loads RealHMDPPipeline once per grid configuration and runs H-MDP
    at ε=1.0, reporting SR, PES, and wall-clock latency.
    """
    if grids is None:
        grids = [(3, 3), (5, 5), (7, 7)]

    print("\n" + "=" * 70)
    print("  TABLE B.12: Grid resolution sensitivity")
    print("=" * 70)
    results = {}

    for (rows, cols) in grids:
        M = rows * cols
        label = f"{rows}×{cols} (M={M})"
        print(f"\n  Grid {label}")
        print(f"  {'-'*50}")

        assert rows == cols, f"RealHMDPPipeline only supports square grids; got {rows}x{cols}"
        pipeline = RealHMDPPipeline(model_key, device=device, grid=rows)
        pipeline.load_models()
        if wproj_ckpt:
            pipeline.load_hmdp_checkpoint(wproj_ckpt, strict_task_head=False)

        pipeline.ltm = LongTermMemory(
            embedding_dim=256, capacity=10000, top_k=8
        ).to(pipeline.device)

        t0 = time.time()
        r = evaluate_configurable(
            pipeline, data, indices, image_base,
            epsilon=1.0, k=k, temperature=temp,
            use_got=True, use_ltm=True, label=label,
        )
        elapsed = time.time() - t0
        r["time_sec"] = elapsed
        r["latency_per_step"] = elapsed / max(r.get("num_evaluated", 1), 1)
        r["grid"] = f"{rows}x{cols}"
        r["M"] = M
        r["action_dim"] = M + 1  # M epsilons + k
        # PES with M normalisation
        avg_eps = r["avg_epsilon"]
        r["pes_paper"] = (r["success_rate"] * M / avg_eps) if avg_eps > 0 else 0.0

        results[label] = r
        print(f"    → SR={r['success_rate']*100:.1f}%, PES={r['pes_paper']:.2f}, "
              f"ε̄={avg_eps:.2f}, latency={r['latency_per_step']:.2f}s/step ({elapsed:.0f}s total)")

        pipeline.unload_models()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def run_table3(data, indices, image_base, device, models, k=5, eps=5.0,
               temp=0.5, task_ckpt=None):
    """Table 3: Cross-backbone Generalizability (RQ4)."""
    print("\n" + "=" * 70)
    print(f"  TABLE 3: Cross-backbone (RQ4) at ε={eps}")
    print("=" * 70)
    results = {}

    for mi, model_key in enumerate(models):
        display_name = MODEL_REGISTRY[model_key]["display_name"]
        print(f"\n  [{mi+1}/{len(models)}] {display_name}")
        print(f"  {'-'*50}")

        pipeline = RealHMDPPipeline(model_key, device=device, task_ckpt=task_ckpt)
        pipeline.load_models()
        if task_ckpt:
            pipeline.load_hmdp_checkpoint(task_ckpt, strict_task_head=False)

        # LDP-only (single call with LDP noise, no GoT/LTM)
        print(f"    LDP-only (ε={eps}, single call, no GoT/LTM)...")
        t0 = time.time()
        base = evaluate_configurable(
            pipeline, data, indices, image_base,
            epsilon=eps, k=1, temperature=temp,
            use_got=False, use_ltm=False, label=f"{display_name}-LDP",
        )
        base["time_sec"] = time.time() - t0
        print(f"      SR={base['success_rate']*100:.1f}%, Grnd={base['grounding_accuracy']*100:.1f}% ({base['time_sec']:.0f}s)")

        # H-MDP
        print(f"    H-MDP (ε={eps}, k={k}, GoT+LTM)...")
        t0 = time.time()
        hmdp = evaluate_configurable(
            pipeline, data, indices, image_base,
            epsilon=eps, k=k, temperature=temp,
            use_got=True, use_ltm=True, label=display_name,
        )
        hmdp["time_sec"] = time.time() - t0
        print(f"      SR={hmdp['success_rate']*100:.1f}%, Grnd={hmdp['grounding_accuracy']*100:.1f}% ({hmdp['time_sec']:.0f}s)")

        gain = hmdp["grounding_accuracy"] - base["grounding_accuracy"]
        print(f"      Gain: ΔGrnd={gain*100:+.1f}%")

        results[model_key] = {
            "display_name": display_name,
            "ldp_only": base,
            "hmdp": hmdp,
            "gain": gain,
        }

        pipeline.unload_models()

    return results


# ═══════════════════════════════════════════════════════════════════════
#  Summary Printers
# ═══════════════════════════════════════════════════════════════════════

def print_table1_summary(results):
    print(f"\n  {'Method':<28} {'SR↑':>7} {'Grnd↑':>7} {'ε̄↓':>6} {'PES↑':>7}")
    print(f"  {'-'*55}")
    for name, r in results.items():
        eps_str = "N/A" if not np.isfinite(r.get("avg_epsilon", 0)) else f"{r.get('avg_epsilon', 0):.2f}"
        pes = r.get("pes", 0)
        print(f"  {name:<28} {r['success_rate']*100:>6.1f}% {r['grounding_accuracy']*100:>6.1f}% "
              f"{eps_str:>6} {pes:>7.3f}")


def print_table_b9_summary(results):
    print(f"\n  {'wpes':<12} {'SR↑':>7} {'ε̄↓':>6} {'PES↑':>7} {'r_pes_bonus':>12}")
    print(f"  {'-'*48}")
    for name, r in results.items():
        print(f"  {name:<12} {r['success_rate']*100:>6.1f}% "
              f"{r['avg_epsilon']:>6.2f} {r.get('pes_paper',0):>7.2f} "
              f"{r.get('r_pes_bonus',0):>12.4f}")


def print_table_b12_summary(results):
    print(f"\n  {'Grid':<14} {'M':>4} {'ActionDim':>10} {'SR↑':>7} {'PES↑':>7} {'Latency':>10}")
    print(f"  {'-'*56}")
    for name, r in results.items():
        print(f"  {name:<14} {r['M']:>4} {r['action_dim']:>10} "
              f"{r['success_rate']*100:>6.1f}% {r.get('pes_paper',0):>7.2f} "
              f"{r.get('latency_per_step',0):>9.2f}s")


def print_table2_summary(results):
    full_sr = results.get("Full H-MDP", {}).get("success_rate", 0) * 100
    print(f"\n  {'Variant':<22} {'SR↑':>7} {'Grnd↑':>7} {'PES↑':>7} {'ΔSR':>8}")
    print(f"  {'-'*55}")
    for name, r in results.items():
        sr = r['success_rate'] * 100
        delta = sr - full_sr if name != "Full H-MDP" else 0
        delta_str = f"{delta:+.1f}%" if name != "Full H-MDP" else ""
        print(f"  {name:<22} {sr:>6.1f}% {r['grounding_accuracy']*100:>6.1f}% "
              f"{r.get('pes',0):>7.3f} {delta_str:>8}")


def print_table3_summary(results):
    print(f"\n  {'VLM Backbone':<22} {'LDP-only':>9} {'H-MDP':>9} {'Gain':>7}")
    print(f"  {'-'*50}")
    for key, r in results.items():
        b = r.get("ldp_only", r.get("base", {}))
        h = r["hmdp"]
        gain = r["gain"]
        print(f"  {r['display_name']:<22} {b['grounding_accuracy']*100:>8.1f}% {h['grounding_accuracy']*100:>8.1f}% "
              f"{gain*100:>+6.1f}%")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Run ALL paper tables with real VLMs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-samples", type=int, default=150)
    parser.add_argument("--test-split", type=str, default=None,
                        help="Path to JSON file with fixed eval indices (e.g. results/json/test_split_1000.json)")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--table", type=int, nargs="*", default=None,
                        help="Which numbered tables to run: 1=main, 2=ablation, "
                             "3=backbone. Defaults to 1 2 3 when neither "
                             "--table nor --table-str is supplied.")
    parser.add_argument("--table-str", type=str, nargs="*", default=[],
                        choices=["b9", "b12"],
                        help="String table IDs: b9, b12")
    parser.add_argument("--table1-model", default="qwen2.5-vl-7b",
                        help="Model for Table 1 & 2")
    parser.add_argument("--table3-models", nargs="+",
                        default=["qwen2.5-vl-7b", "qwen3-vl-8b", "internvl3.5-8b", "aguvis-7b", "ui-tars-1.5-7b", "uground-7b", "gui-actor-7b"])
    parser.add_argument("--with-random-eps", action="store_true", default=False,
                        help="Add Random-ε row to Table 2 (perception-axis ablation)")
    parser.add_argument("--wpes-values", type=float, nargs="+", default=[0.0, 1.0, 2.0, 5.0],
                        help="wpes values for Table B.9 sweep")
    parser.add_argument("--grids", type=str, nargs="+", default=["3x3", "5x5", "7x7"],
                        help="Grid configs for Table B.12, e.g. 3x3 5x5 7x7")
    parser.add_argument("--wproj-ckpt", type=str,
                        default="checkpoints/wproj_eq8.pt",
                        help="Checkpoint containing trained proxy/task/LTM "
                             "weights; used by the raw-pixel table harness.")
    parser.add_argument("--task-ckpt", type=str, default=None,
                        help="TaskDirectionHead checkpoint for semantic GoT scoring")
    parser.add_argument("--data-path", type=str,
                        default="gui360_full/processed_data/action_prediction_train_resize/training_data.json")
    parser.add_argument("--image-base", type=str,
                        default="gui360_full/processed_data/action_prediction_train_resize/")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Global RNG seed for sample selection, GoT "
                             "temperature sampling, and LDP noise (reproducibility).")
    parser.add_argument("--deterministic", action="store_true",
                        help="Also request deterministic CUDA/cuBLAS kernels.")
    parser.add_argument("--out", type=str, default="results/real_all_tables.json",
                        help="Path to write result JSON. Use a smoke-specific "
                             "path for short validation runs.")
    args = parser.parse_args()
    if args.table is None:
        args.table = [1, 2, 3] if not args.table_str else []

    # Seed everything BEFORE any sampling / model init so runs are reproducible.
    set_global_seed(args.seed, deterministic=args.deterministic)

    print("=" * 70)
    print("  Real VLM Experiments — ALL Paper Tables")
    print(f"  Tables: {args.table}")
    print(f"  Samples: {args.num_samples}, k={args.k}, τ={args.temperature}")
    print(f"  Table 1/2 model: {args.table1_model}")
    print(f"  Table 3 models: {args.table3_models}")
    print(f"  Seed: {args.seed}{' (deterministic)' if args.deterministic else ''}")
    print("=" * 70)

    # Load dataset
    print(f"\n  Loading dataset: {args.data_path}")
    with open(args.data_path) as f:
        data = json.load(f)
    print(f"  Total samples: {len(data)}")

    if args.test_split:
        with open(args.test_split) as f:
            indices = np.array(json.load(f))
        print(f"  Using fixed test split: {len(indices)} samples from {args.test_split}")
    else:
        rng = np.random.RandomState(args.seed)
        indices = rng.choice(len(data), min(args.num_samples, len(data)), replace=False)
        print(f"  Randomly selected {len(indices)} samples for evaluation")

    all_results = {}
    t_total = time.time()

    # Parse grid configs for B.12
    parsed_grids = []
    for g in args.grids:
        r, c = g.lower().split("x")
        parsed_grids.append((int(r), int(c)))

    # ══ TABLE 1 & 2: Use same model ══
    if 1 in args.table or 2 in args.table:
        model_key = args.table1_model
        display = MODEL_REGISTRY[model_key]["display_name"]
        print(f"\n  Loading {display} for Table 1 & 2...")

        task_ckpt = args.task_ckpt or args.wproj_ckpt
        pipeline = RealHMDPPipeline(model_key, device=args.device,
                                    task_ckpt=task_ckpt)
        pipeline.load_models()
        pipeline.load_hmdp_checkpoint(task_ckpt, strict_task_head=True)

        if 1 in args.table:
            table1 = run_table1(pipeline, data, indices, args.image_base,
                                k=args.k, temp=args.temperature)
            all_results["table1"] = table1
            print_table1_summary(table1)
            _save_results(all_results, args.out)

        if 2 in args.table:
            pipeline.ltm = LongTermMemory(
                embedding_dim=256, capacity=10000, top_k=8
            ).to(pipeline.device)

            if args.with_random_eps:
                table2 = run_table2_with_random_eps(
                    pipeline, data, indices, args.image_base,
                    k=args.k, temp=args.temperature,
                )
            else:
                table2 = run_table2(pipeline, data, indices, args.image_base,
                                    k=args.k, temp=args.temperature)
            all_results["table2"] = table2
            print_table2_summary(table2)
            _save_results(all_results, args.out)

        pipeline.unload_models()

    # ══ TABLE 3: All backbones ══
    if 3 in args.table:
        table3 = run_table3(
            data, indices, args.image_base,
            device=args.device, models=args.table3_models,
            k=args.k, eps=5.0, temp=args.temperature,
            task_ckpt=args.task_ckpt or args.wproj_ckpt,
        )
        all_results["table3"] = table3
        print_table3_summary(table3)

        _save_results(all_results, args.out)

    # ══ TABLE B.9: wpes sensitivity ══
    if "b9" in (args.table_str or []):
        model_key = args.table1_model
        print(f"\n  Loading {MODEL_REGISTRY[model_key]['display_name']} for Table B.9...")
        task_ckpt = args.task_ckpt or args.wproj_ckpt
        pipeline = RealHMDPPipeline(model_key, device=args.device,
                                    task_ckpt=task_ckpt)
        pipeline.load_models()
        pipeline.load_hmdp_checkpoint(task_ckpt, strict_task_head=True)
        table_b9 = run_table_b9_wpes(
            pipeline, data, indices, args.image_base,
            wpes_values=args.wpes_values,
            k=args.k, temp=args.temperature,
        )
        all_results["table_b9"] = table_b9
        print_table_b9_summary(table_b9)
        pipeline.unload_models()
        _save_results(all_results, args.out)

    # ══ TABLE B.12: grid resolution ══
    if "b12" in (args.table_str or []):
        table_b12 = run_table_b12_grid(
            data, indices, args.image_base,
            device=args.device,
            model_key=args.table1_model,
            wproj_ckpt=args.task_ckpt or args.wproj_ckpt,
            grids=parsed_grids,
            k=args.k, temp=args.temperature,
        )
        all_results["table_b12"] = table_b12
        print_table_b12_summary(table_b12)
        _save_results(all_results, args.out)

    # ══ Final Summary ══
    elapsed = time.time() - t_total
    print(f"\n\n{'='*70}")
    print(f"  ALL DONE — Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*70}")

    if "table1" in all_results:
        print("\n  ── Table 1: Main Results ──")
        print_table1_summary(all_results["table1"])

    if "table2" in all_results:
        print("\n  ── Table 2: Ablation ──")
        print_table2_summary(all_results["table2"])

    if "table3" in all_results:
        print("\n  ── Table 3: Cross-backbone ──")
        print_table3_summary(all_results["table3"])

    if "table_b9" in all_results:
        print("\n  ── Table B.9: wpes sensitivity ──")
        print_table_b9_summary(all_results["table_b9"])

    if "table_b12" in all_results:
        print("\n  ── Table B.12: Grid resolution ──")
        print_table_b12_summary(all_results["table_b12"])

    _save_results(all_results, args.out)
    print(f"\n  Results saved to {args.out}")


def _save_results(results, save_path: str):
    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    def _serialize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return obj

    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, default=_serialize)


if __name__ == "__main__":
    main()
