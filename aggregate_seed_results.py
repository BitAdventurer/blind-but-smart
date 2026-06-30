#!/usr/bin/env python3
"""
Aggregate 3-seed results → mean ± std for all tables.

Input:
  results/real_all_tables_seed{0,1,2}.json
  results/real_appendix_seed{0,1,2}.json

Output:
  results/aggregated_results.json   — all metrics with mean, std, ci95

Metrics (paper-centric):
  SR = success_rate (joint: action AND grounding correct)
  Grounding Acc = grounding_accuracy (point in bbox)
  PES = SR / ε̄  (Privacy-Efficiency Score, paper Eq. 15)
"""
import json, os
import numpy as np

os.makedirs("results", exist_ok=True)


def ci95(values):
    """95% confidence interval half-width."""
    n = len(values)
    if n < 2:
        return 0.0
    return 1.96 * np.std(values, ddof=1) / np.sqrt(n)


def agg(values):
    """Return dict with mean, std, ci95."""
    arr = [v for v in values if v is not None and np.isfinite(v)]
    if not arr:
        return {"mean": None, "std": None, "ci95": None}
    return {
        "mean": float(np.mean(arr)),
        "std":  float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "ci95": float(ci95(arr)),
    }


def fmt(a, pct=False, dec=1):
    """Format mean±std string for display."""
    if a["mean"] is None:
        return "—"
    scale = 100 if pct else 1
    m = a["mean"] * scale
    s = a["std"] * scale if a["std"] is not None else 0
    return f"{m:.{dec}f}±{s:.{dec}f}"


def _safe_get(d, key, default=None):
    """Get value from dict, returning default for missing or non-finite."""
    v = d.get(key, default)
    if v is None:
        return default
    return v


# ── Load seed files ──────────────────────────────────────────────────────────
N_SEEDS = 3
main_seeds = []
appendix_seeds = []

def load_json_safe(path, desc=""):
    """Load JSON with error handling."""
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"  [WARN] {desc} JSON parse error in {path}: {e}")
        return None
    except Exception as e:
        print(f"  [WARN] {desc} Failed to load {path}: {e}")
        return None


for s in range(N_SEEDS):
    mp = f"results/real_all_tables_seed{s}.json"
    ap = f"results/real_appendix_seed{s}.json"
    if os.path.exists(mp):
        data = load_json_safe(mp, f"seed {s} main")
        if data is not None:
            main_seeds.append(data)
    if os.path.exists(ap):
        data = load_json_safe(ap, f"seed {s} appendix")
        if data is not None:
            appendix_seeds.append(data)

print(f"  Main seeds loaded: {len(main_seeds)} / {N_SEEDS}")
print(f"  Appendix seeds loaded: {len(appendix_seeds)} / {N_SEEDS}")

if not main_seeds and not appendix_seeds:
    print("  No seed results found. Run run_full_eval.sh first.")
    raise SystemExit(1)

aggregated = {}

# ── TABLE 1 (paper Table 2): Main Results ──────────────────────────────────
if main_seeds and "table1" in main_seeds[0]:
    print("\n  TABLE 1 (paper Table 2): Main Comparative Results")
    t1_agg = {}
    methods = list(main_seeds[0]["table1"].keys())
    for method in methods:
        sr_vals   = [s["table1"][method]["success_rate"]       for s in main_seeds if method in s["table1"]]
        grnd_vals = [s["table1"][method]["grounding_accuracy"] for s in main_seeds if method in s["table1"]]
        pes_vals  = [s["table1"][method].get("pes", 0)         for s in main_seeds if method in s["table1"]]
        eps_vals  = [s["table1"][method].get("avg_epsilon", None) for s in main_seeds if method in s["table1"]]
        t1_agg[method] = {
            "success_rate":       agg(sr_vals),
            "grounding_accuracy": agg(grnd_vals),
            "pes":                agg(pes_vals),
            "avg_epsilon":        agg([e for e in eps_vals if e is not None and np.isfinite(e)]),
        }
        print(f"    {method:<28} SR={fmt(t1_agg[method]['success_rate'],pct=True)}%  "
              f"Grnd={fmt(t1_agg[method]['grounding_accuracy'],pct=True)}%  "
              f"PES={fmt(t1_agg[method]['pes'],dec=3)}")
    aggregated["table1"] = t1_agg

# ── TABLE 2 (paper Table 3): Ablation ──────────────────────────────────────
if main_seeds and "table2" in main_seeds[0]:
    print("\n  TABLE 2 (paper Table 3): Ablation Study")
    t2_agg = {}
    methods = list(main_seeds[0]["table2"].keys())
    for method in methods:
        sr_vals   = [s["table2"][method]["success_rate"]       for s in main_seeds if method in s["table2"]]
        grnd_vals = [s["table2"][method]["grounding_accuracy"] for s in main_seeds if method in s["table2"]]
        pes_vals  = [s["table2"][method].get("pes", 0)         for s in main_seeds if method in s["table2"]]
        t2_agg[method] = {
            "success_rate":       agg(sr_vals),
            "grounding_accuracy": agg(grnd_vals),
            "pes":                agg(pes_vals),
        }
        print(f"    {method:<22} SR={fmt(t2_agg[method]['success_rate'],pct=True)}%  "
              f"Grnd={fmt(t2_agg[method]['grounding_accuracy'],pct=True)}%  "
              f"PES={fmt(t2_agg[method]['pes'],dec=3)}")
    aggregated["table2"] = t2_agg

# ── TABLE 3 (paper Table 4): Cross-backbone ────────────────────────────────
if main_seeds and "table3" in main_seeds[0]:
    print("\n  TABLE 3 (paper Table 4): Cross-backbone Generalizability")
    t3_agg = {}
    backbones = list(main_seeds[0]["table3"].keys())
    for bb in backbones:
        # Support both old key "base" and new key "ldp_only"
        ldp_key = "ldp_only" if "ldp_only" in main_seeds[0]["table3"][bb] else "base"
        def _agg_cond(cond, metric):
            return agg([s["table3"][bb][cond][metric] for s in main_seeds
                        if bb in s["table3"] and cond in s["table3"][bb]])
        t3_agg[bb] = {
            "display_name": main_seeds[0]["table3"][bb].get("display_name", bb),
            "ldp_only": {
                "grounding_accuracy": _agg_cond(ldp_key, "grounding_accuracy"),
                "success_rate":       _agg_cond(ldp_key, "success_rate"),
            },
            "hmdp": {
                "grounding_accuracy": _agg_cond("hmdp", "grounding_accuracy"),
                "success_rate":       _agg_cond("hmdp", "success_rate"),
            },
            "gain": agg([s["table3"][bb]["gain"] for s in main_seeds if bb in s["table3"]]),
        }
        ldp = t3_agg[bb]["ldp_only"]["grounding_accuracy"]
        hmdp = t3_agg[bb]["hmdp"]["grounding_accuracy"]
        print(f"    {bb:<20} LDP-only={fmt(ldp,pct=True)}%  H-MDP={fmt(hmdp,pct=True)}%  "
              f"Gain={fmt(t3_agg[bb]['gain'],pct=True)}%")
    aggregated["table3"] = t3_agg

# ── TABLE A.7 (paper): Reward Weight Sensitivity ──────────────────────────
if appendix_seeds and "tableA7" in appendix_seeds[0]:
    print("\n  TABLE A.7: Reward Weight Sensitivity")
    ta7_agg = {}
    configs = list(appendix_seeds[0]["tableA7"].keys())
    for cfg in configs:
        sr_vals  = [s["tableA7"][cfg]["success_rate"]  for s in appendix_seeds if cfg in s["tableA7"] and "success_rate" in s["tableA7"][cfg]]
        pes_vals = [s["tableA7"][cfg]["pes"]            for s in appendix_seeds if cfg in s["tableA7"] and "pes" in s["tableA7"][cfg]]
        ta7_agg[cfg] = {
            "success_rate":  agg(sr_vals),
            "pes":           agg(pes_vals),
            "avg_epsilon":   appendix_seeds[0]["tableA7"][cfg].get("avg_epsilon", None),
        }
        print(f"    {cfg:<28} SR={fmt(ta7_agg[cfg]['success_rate'],pct=True)}%  "
              f"PES={fmt(ta7_agg[cfg]['pes'],dec=3)}")
    aggregated["tableA7"] = ta7_agg

# ── TABLE A.8 (paper): Task-Split Consistency ─────────────────────────────
if appendix_seeds and "tableA8" in appendix_seeds[0]:
    print("\n  TABLE A.8: Task-Split Consistency")
    ta8_agg = {}
    methods = list(appendix_seeds[0]["tableA8"].keys())
    for method in methods:
        ta8_agg[method] = {}
        splits = list(appendix_seeds[0]["tableA8"][method].keys())
        for split in splits:
            sr_vals = [s["tableA8"][method][split]["success_rate"]
                       for s in appendix_seeds
                       if method in s["tableA8"] and split in s["tableA8"][method]]
            pes_vals = [s["tableA8"][method][split]["pes"]
                        for s in appendix_seeds
                        if method in s["tableA8"] and split in s["tableA8"][method]]
            ta8_agg[method][split] = {
                "success_rate": agg(sr_vals),
                "pes":          agg(pes_vals),
            }
        row = "  ".join(f"{sp}: SR={fmt(ta8_agg[method][sp]['success_rate'],pct=True)}% PES={fmt(ta8_agg[method][sp]['pes'],dec=2)}"
                        for sp in splits)
        print(f"    {method:<22} {row}")
    aggregated["tableA8"] = ta8_agg

# ── TABLE A.9 (paper): Grid Resolution ────────────────────────────────────
if appendix_seeds and "tableA9" in appendix_seeds[0]:
    print("\n  TABLE A.9: Grid Resolution")
    ta9_agg = {}
    grids = list(appendix_seeds[0]["tableA9"].keys())
    for grid in grids:
        sr_vals  = [s["tableA9"][grid]["success_rate"]  for s in appendix_seeds if grid in s["tableA9"] and "success_rate" in s["tableA9"][grid]]
        pes_vals = [s["tableA9"][grid]["pes"]            for s in appendix_seeds if grid in s["tableA9"] and "pes" in s["tableA9"][grid]]
        ta9_agg[grid] = {
            "M": appendix_seeds[0]["tableA9"][grid]["M"],
            "success_rate": agg(sr_vals),
            "pes":          agg(pes_vals),
        }
        print(f"    {grid:<6} SR={fmt(ta9_agg[grid]['success_rate'],pct=True)}%  "
              f"PES={fmt(ta9_agg[grid]['pes'],dec=3)}")
    aggregated["tableA9"] = ta9_agg

# ── Save ─────────────────────────────────────────────────────────────────────
with open("results/aggregated_results.json", "w") as f:
    json.dump(aggregated, f, indent=2)

print(f"\n  Saved → results/aggregated_results.json")
print(f"  Seeds used: main={len(main_seeds)}, appendix={len(appendix_seeds)}")
