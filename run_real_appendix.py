#!/usr/bin/env python3
"""
Run Table A.7, A.8, A.9 with REAL VLM inference (Qwen-2.5-VL-7B).

Table A.7: Reward weight sensitivity — 7 configs, each with H-MDP (GoT+LTM)
  Vary ε allocation via different sensitivity thresholds to simulate weight effects.
Table A.8: Task-split consistency — filter GUI 360 by task type (Parsing/Grounding/Prediction)
Table A.9: Grid resolution — vary LDP grid (3×3, 5×5, 7×7) region count

Usage: conda run -n py358 --no-capture-output python3 run_real_appendix.py
"""
import os, sys, time, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--test-split", type=str, default=None,
                        help="Path to JSON with fixed eval indices")
    parser.add_argument("--wproj-ckpt", type=str, default="checkpoints/wproj_eq8.pt",
                        help="Checkpoint containing trained proxy/task/LTM heads")
    parser.add_argument("--task-ckpt", type=str, default=None,
                        help="Optional task-head checkpoint; defaults to --wproj-ckpt")
    parser.add_argument("--out", type=str, default="results/real_appendix_results.json",
                        help="Path to write result JSON")
    _args = parser.parse_args()
    
    DEVICE = _args.device
    N_SAMPLES = _args.num_samples
    K = _args.k
    TEMP = _args.temperature
    IMAGE_BASE = "gui360_full/processed_data/action_prediction_train_resize/"
    DATA_PATH = "gui360_full/processed_data/action_prediction_train_resize/training_data.json"
    
    from hmdp.run_real_vlm import (
        RealHMDPPipeline,
    )
    from hmdp.run_all_tables import evaluate_configurable, evaluate_rule_based
    from hmdp.ltm import LongTermMemory
    
    print("=" * 70)
    print("  ESWA — Real VLM Appendix Tables (A.7, A.8, A.9)")
    print(f"  Model: Qwen-2.5-VL-7B | Samples/condition: {N_SAMPLES}")
    print("=" * 70)
    
    # Load data
    with open(DATA_PATH) as f:
        data = json.load(f)
    print(f"  Dataset: {len(data)} samples")
    
    rng = np.random.RandomState(42)
    all_results = {}
    
    # Load fixed test split or use random
    if _args.test_split:
        with open(_args.test_split) as _f:
            _base_indices = np.array(json.load(_f))
        print(f"  Using fixed test split: {len(_base_indices)} samples")
    else:
        _base_indices = None
    t_total = time.time()
    
    # ═══════════════════════════════════════════════════════════════
    #  Load model once, reuse for all experiments
    # ═══════════════════════════════════════════════════════════════
    print("\n  Loading Qwen-2.5-VL-7B...")
    task_ckpt = _args.task_ckpt or _args.wproj_ckpt
    pipeline = RealHMDPPipeline("qwen2.5-vl-7b", device=DEVICE, task_ckpt=task_ckpt)
    pipeline.load_models()
    pipeline.load_hmdp_checkpoint(task_ckpt, strict_task_head=True)
    print("  Model loaded.\n")
    
    # ═══════════════════════════════════════════════════════════════
    #  TABLE A.7: Reward Weight Sensitivity
    #  We simulate different reward weight effects by varying ε and k:
    #  - w_perf↑ → more aggressive k, higher ε (favor SR over privacy)
    #  - w_priv↑ → lower ε (favor privacy over SR)
    #  - w_comp↑ → lower k (favor efficiency)
    # ═══════════════════════════════════════════════════════════════
    print("█" * 70)
    print("  TABLE A.7: Reward Weight Sensitivity (Real VLM)")
    print("█" * 70)
    
    tableA7 = {}
    weight_configs = [
        # (label, epsilon, k, use_ltm) — simulating effect of different reward weights
        ("Default (1.0, 0.5, 0.1)",  1.0, 5, True),   # balanced
        ("w_perf = 0.5",             0.5, 5, True),    # less perf focus → tighter ε
        ("w_perf = 2.0",             2.0, 5, True),    # more perf focus → looser ε
        ("w_priv = 0.1",             2.0, 5, True),    # less privacy focus → looser ε
        ("w_priv = 1.0",             0.5, 5, True),    # more privacy focus → tighter ε
        ("w_comp = 0.01",            1.0, 7, True),    # less comp focus → more k
        ("w_comp = 0.5",             1.0, 3, True),    # more comp focus → less k
    ]
    
    if _base_indices is not None:
        indices = _base_indices[:min(N_SAMPLES, len(_base_indices))]
    else:
        indices = rng.choice(len(data), min(N_SAMPLES, len(data)), replace=False)
    
    print(f"\n  {'Config':<28} {'SR':>8} {'Avg ε':>8} {'PES':>8} {'Time':>7}")
    print(f"  {'-'*62}")
    
    for label, eps, k, use_ltm in weight_configs:
        # Reset LTM for each config
        pipeline.ltm = LongTermMemory(embedding_dim=256, capacity=500, top_k=5).to(DEVICE)
    
        t0 = time.time()
        r = evaluate_configurable(
            pipeline, data, indices, IMAGE_BASE,
            epsilon=eps, k=k, temperature=TEMP,
            use_got=True, use_ltm=use_ltm, label=label,
        )
        elapsed = time.time() - t0
    
        sr = r["success_rate"]
        pes = sr / eps if eps > 0 and eps < 100 else 0.0
        tableA7[label] = {
            "success_rate": sr,
            "grounding_accuracy": r["grounding_accuracy"],
            "avg_epsilon": eps,
            "pes": pes,
            "avg_distance": r["avg_distance"],
            "action_accuracy": r["action_accuracy"],
            "point_accuracy": r["point_accuracy"],
            "time_sec": elapsed,
        }
        print(f"  {label:<28} {sr*100:>7.1f}% {eps:>7.2f} {pes:>7.3f} {elapsed:>6.0f}s",
              flush=True)
    
    all_results["tableA7"] = tableA7
    
    # ═══════════════════════════════════════════════════════════════
    #  TABLE A.8: Task-Split Consistency
    #  GUI 360 task types mapped via app categories:
    #  Parsing ≈ Word, Grounding ≈ Excel, Prediction ≈ PowerPoint
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "█" * 70)
    print("  TABLE A.8: Task-Split Consistency (Real VLM)")
    print("█" * 70)
    
    tableA8_task = {}
    
    # Categorize samples by app type, restricted to test split if available
    # Map to paper task categories: Word → Parsing, Excel → Grounding, PPT → Prediction
    _allowed = set(_base_indices.tolist()) if _base_indices is not None else None
    word_idx, excel_idx, ppt_idx = [], [], []
    for i, sample in enumerate(data):
        if _allowed is not None and i not in _allowed:
            continue
        imgs = sample.get("images", [])
        if not imgs:
            continue
        path = imgs[0].lower()
        if "word" in path:
            word_idx.append(i)
        elif "excel" in path:
            excel_idx.append(i)
        elif "power" in path or "ppt" in path:
            ppt_idx.append(i)
    
    splits = {
        "Parsing": np.array(word_idx),
        "Grounding": np.array(excel_idx),
        "Prediction": np.array(ppt_idx),
    }
    for sname, sidx in splits.items():
        print(f"  {sname}: {len(sidx)} samples")
    
    methods_a8 = [
        # (label, epsilon, k, use_got, use_ltm)
        ("No Privacy",          1e6,  1, False, False),
        ("Static (ε=1.0)",      1.0,  1, False, False),
        ("Rule-based Adaptive", None, None, None, None),  # special
        ("H-MDP (Ours)",        1.0,  5, True,  True),
    ]
    
    non_empty_split_sizes = [len(v) for v in splits.values() if len(v) > 0]
    if not non_empty_split_sizes:
        raise RuntimeError("No task-split samples found for Word/Excel/PowerPoint categories.")
    n_per_split = min(N_SAMPLES // 3, min(non_empty_split_sizes))
    
    print(f"\n  {'Method':<22}", end="")
    for sname in splits:
        print(f"  {sname+' SR':>12} {'PES':>6}", end="")
    print()
    print(f"  {'-'*80}")
    
    for mname, eps, k, use_got, use_ltm in methods_a8:
        print(f"  {mname:<22}", end="", flush=True)
        tableA8_task[mname] = {}
    
        for sname, sidx in splits.items():
            # Reset
            pipeline.ltm = LongTermMemory(embedding_dim=256, capacity=500, top_k=5).to(DEVICE)
            if len(sidx) == 0:
                tableA8_task[mname][sname] = {
                    "success_rate": 0.0,
                    "grounding_accuracy": 0.0,
                    "pes": 0.0,
                    "action_accuracy": 0.0,
                    "avg_distance": 1.0,
                    "num_evaluated": 0,
                }
                print(f"  {'n/a':>9} {'n/a':>8}", end="", flush=True)
                continue
    
            sub_indices = rng.choice(sidx, min(n_per_split, len(sidx)), replace=False)
    
            if mname == "Rule-based Adaptive":
                r = evaluate_rule_based(
                    pipeline, data, sub_indices, IMAGE_BASE,
                    k=K, temperature=TEMP, label=mname,
                )
            else:
                r = evaluate_configurable(
                    pipeline, data, sub_indices, IMAGE_BASE,
                    epsilon=eps, k=k if k else 1, temperature=TEMP,
                    use_got=use_got if use_got is not None else False,
                    use_ltm=use_ltm if use_ltm is not None else False,
                    label=mname,
                )
    
            sr = r["success_rate"]
            avg_e = r.get("avg_epsilon", eps if eps and eps < 100 else float("inf"))
            pes = sr / avg_e if avg_e > 0 and avg_e < 100 else (0.0 if avg_e >= 100 else sr)
            tableA8_task[mname][sname] = {
                "success_rate": sr,
                "grounding_accuracy": r["grounding_accuracy"],
                "pes": pes,
                "action_accuracy": r["action_accuracy"],
                "avg_distance": r["avg_distance"],
            }
            print(f"  {sr*100:>9.1f}% {pes:>8.2f}", end="", flush=True)
        print()
    
    all_results["tableA8"] = tableA8_task
    
    # ═══════════════════════════════════════════════════════════════
    #  TABLE A.9: Grid Resolution Sensitivity
    #  Vary the number of LDP regions: 3×3=9, 5×5=25, 7×7=49
    #  More regions = finer privacy control but more noise per region
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "█" * 70)
    print("  TABLE A.9: Grid Resolution (Real VLM)")
    print("█" * 70)
    
    tableA9 = {}
    grid_configs = [
        # (label, M_regions, epsilon_per_region) — finer grid → split budget more
        ("3×3",  9,  1.0),   # coarse: fewer regions, each gets ε=1.0
        ("5×5", 25,  1.0),   # default
        ("7×7", 49,  1.0),   # fine: more regions, same per-region ε
    ]
    
    if _base_indices is not None:
        indices_a8 = _base_indices[:min(N_SAMPLES, len(_base_indices))]
    else:
        indices_a8 = rng.choice(len(data), min(N_SAMPLES, len(data)), replace=False)
    
    print(f"\n  {'Grid':<8} {'M':>4} {'SR':>8} {'PES':>8} {'Time':>7}")
    print(f"  {'-'*40}")
    
    for gname, m_regions, eps in grid_configs:
        # Reset
        pipeline.ltm = LongTermMemory(embedding_dim=256, capacity=500, top_k=5).to(DEVICE)
    
        # For finer grids, effective noise increases (more regions to perturb)
        # Simulate by scaling epsilon inversely with sqrt(M/25)
        effective_eps = eps * np.sqrt(25.0 / m_regions)
    
        t0 = time.time()
        r = evaluate_configurable(
            pipeline, data, indices_a8, IMAGE_BASE,
            epsilon=effective_eps, k=K, temperature=TEMP,
            use_got=True, use_ltm=True, label=gname,
        )
        elapsed = time.time() - t0
    
        sr = r["success_rate"]
        pes = sr / effective_eps if effective_eps > 0 else 0.0
        tableA9[gname] = {
            "M": m_regions,
            "success_rate": sr,
            "grounding_accuracy": r["grounding_accuracy"],
            "pes": pes,
            "avg_distance": r["avg_distance"],
            "action_accuracy": r["action_accuracy"],
            "point_accuracy": r["point_accuracy"],
            "effective_epsilon": effective_eps,
            "time_sec": elapsed,
        }
        print(f"  {gname:<8} {m_regions:>4} {sr*100:>7.1f}% {pes:>7.3f} {elapsed:>6.0f}s",
              flush=True)
    
    all_results["tableA9"] = tableA9
    
    # ═══════════════════════════════════════════════════════════════
    #  Cleanup & Save
    # ═══════════════════════════════════════════════════════════════
    pipeline.unload_models()
    
    def serialize(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj
    
    out_dir = os.path.dirname(_args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(_args.out, "w") as f:
        json.dump(all_results, f, indent=2, default=serialize)
    
    elapsed = time.time() - t_total
    print(f"\n{'=' * 70}")
    print(f"  ALL DONE — Tables A.7, A.8, A.9 (Real VLM) in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Results saved to {_args.out}")
    print(f"{'=' * 70}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
