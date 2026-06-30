#!/usr/bin/env python3
"""
Create 3 non-overlapping 1,000-sample stratified evaluation splits
from the held-out test set (results/json/test_split_indices.json).

Outputs:
  results/json/eval_seed0.json   — 1,000 samples (seed 0)
  results/json/eval_seed1.json   — 1,000 samples (seed 1)
  results/json/eval_seed2.json   — 1,000 samples (seed 2)
  (non-overlapping, together = 3,000 of 20,358 test samples)
"""
import json, os, random
import numpy as np
from collections import defaultdict

DATA_PATH = "gui360_full/processed_data/action_prediction_train_resize/training_data.json"
OUT_DIR = "results/json"
TEST_SPLIT_PATH = os.path.join(OUT_DIR, "test_split_indices.json")
TARGET_PER_SEED = 1000
N_SEEDS = 3

print("Loading dataset...")
with open(DATA_PATH) as f:
    data = json.load(f)

with open(TEST_SPLIT_PATH) as f:
    splits = json.load(f)
test_idx = splits["test"]
print(f"  Test pool: {len(test_idx):,} samples")

# ── Categorize test samples by stratum ──────────────────────────
def get_stratum(sample):
    imgs = sample.get("images", [])
    path = imgs[0].lower() if imgs else ""
    if "word" in path: return "word"
    if "excel" in path: return "excel"
    if "ppt" in path or "power" in path: return "ppt"
    return "other"

strata = defaultdict(list)
for i in test_idx:
    strata[get_stratum(data[i])].append(i)

total_test = len(test_idx)
print(f"  Strata: { {k: len(v) for k,v in strata.items()} }")

# ── Allocate proportionally across 3 seeds (non-overlapping) ────
# Total needed: 3 × 1,000 = 3,000 samples
TOTAL_NEEDED = TARGET_PER_SEED * N_SEEDS

rng = random.Random(2025)

# Shuffle each stratum
for key in strata:
    rng.shuffle(strata[key])

# Proportional allocation of TOTAL_NEEDED across strata
allocated = {}
for key, idxs in strata.items():
    n = max(N_SEEDS, round(len(idxs) / total_test * TOTAL_NEEDED))
    allocated[key] = idxs[:n]

# Pool all allocated, shuffle, split into 3 × 1,000
pool = []
for idxs in allocated.values():
    pool.extend(idxs)
rng.shuffle(pool)

# Trim to exactly 3,000
pool = pool[:TOTAL_NEEDED]
assert len(pool) == TOTAL_NEEDED, f"Expected {TOTAL_NEEDED}, got {len(pool)}"

# Split into 3 non-overlapping groups of 1,000
seed_splits = [
    pool[i * TARGET_PER_SEED : (i + 1) * TARGET_PER_SEED]
    for i in range(N_SEEDS)
]

# Verify no overlap
for i in range(N_SEEDS):
    for j in range(i + 1, N_SEEDS):
        overlap = set(seed_splits[i]) & set(seed_splits[j])
        assert len(overlap) == 0, f"Overlap between seed {i} and {j}: {len(overlap)}"

# ── Save & report ────────────────────────────────────────────────
for seed_id, indices in enumerate(seed_splits):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"eval_seed{seed_id}.json")
    with open(path, "w") as f:
        json.dump(indices, f)

    # Composition of this seed
    comp = defaultdict(int)
    for i in indices:
        comp[get_stratum(data[i])] += 1
    print(f"  Seed {seed_id}: {len(indices)} samples — {dict(comp)}  → {path}")

print(f"\n  3 non-overlapping splits created. Total used: {TOTAL_NEEDED} / {len(test_idx)} test samples.")
