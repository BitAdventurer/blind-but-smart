#!/usr/bin/env python3
"""
Create a proper 80/20 stratified test split from GUI 360 training data.
Stratified by:  app type  (inferred from image path)
                task type (action_type from conversation)

Saves:
  results/json/test_split_indices.json   — 20% held-out indices (20,360 samples)
  results/json/test_split_1000.json      — 1,000 stratified sample from test split
  results/logs/split_stats.txt           — split statistics
"""
import json, os, re, random
import numpy as np
from collections import defaultdict

DATA_PATH = "gui360_full/processed_data/action_prediction_train_resize/training_data.json"
OUT_DIR = "results/json"
LOG_DIR = "results/logs"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

print("Loading dataset...")
with open(DATA_PATH) as f:
    data = json.load(f)
N = len(data)
print(f"  Total samples: {N}")

# ─── Categorize each sample ──────────────────────────────────
def get_app(sample):
    imgs = sample.get("images", [])
    if not imgs:
        return "other"
    p = imgs[0].lower()
    for app in ["word", "excel", "powerpoint", "chrome", "vlc", "firefox",
                "acrobat", "calculator", "notepad", "paint", "explorer"]:
        if app in p:
            return app
    # extract first directory segment
    parts = p.replace("\\", "/").split("/")
    for part in parts:
        if part not in ("images", "") and not part.endswith(".png"):
            return part.split("_")[0]
    return "other"

def get_action_type(sample):
    conv = sample.get("conversation", [])
    for turn in conv:
        val = turn.get("value", "")
        for atype in ["click", "type", "scroll", "drag", "press", "select", "open"]:
            if atype in val.lower():
                return atype
    return "other"

print("Categorizing samples by app and action type...")
strata = defaultdict(list)  # (app, action_type) → [indices]
for i, sample in enumerate(data):
    app = get_app(sample)
    action = get_action_type(sample)
    strata[(app, action)].append(i)
    if (i + 1) % 20000 == 0:
        print(f"  {i+1}/{N}", flush=True)

print(f"\n  Strata found: {len(strata)}")
top_strata = sorted(strata.items(), key=lambda x: -len(x[1]))[:10]
for (app, act), idxs in top_strata:
    print(f"    {app:15s} / {act:10s}: {len(idxs):6d}")

# ─── 80/20 stratified split ──────────────────────────────────
rng = random.Random(2025)
train_idx, test_idx = [], []

for (app, act), idxs in strata.items():
    rng.shuffle(idxs)
    n_test = max(1, int(len(idxs) * 0.2))
    test_idx.extend(idxs[:n_test])
    train_idx.extend(idxs[n_test:])

rng.shuffle(test_idx)
rng.shuffle(train_idx)

print(f"\n  Train: {len(train_idx):,}  Test: {len(test_idx):,}")
print(f"  Test ratio: {len(test_idx)/N*100:.1f}%")

with open(os.path.join(OUT_DIR, "test_split_indices.json"), "w") as f:
    json.dump({"train": train_idx, "test": test_idx}, f)

# ─── Stratified 1,000-sample subset from test split ──────────
TARGET = 1000
test_strata = defaultdict(list)
test_set = set(test_idx)
for i, sample in enumerate(data):
    if i not in test_set:
        continue
    app = get_app(sample)
    action = get_action_type(sample)
    test_strata[(app, action)].append(i)

# Proportional allocation
total_test = len(test_idx)
sample_1000 = []
for (app, act), idxs in test_strata.items():
    n_alloc = max(1, round(len(idxs) / total_test * TARGET))
    rng.shuffle(idxs)
    sample_1000.extend(idxs[:n_alloc])

# Trim/pad to exactly TARGET
rng.shuffle(sample_1000)
if len(sample_1000) > TARGET:
    sample_1000 = sample_1000[:TARGET]
elif len(sample_1000) < TARGET:
    # fill from remaining test set
    remaining = list(test_set - set(sample_1000))
    rng.shuffle(remaining)
    sample_1000.extend(remaining[:TARGET - len(sample_1000)])

assert len(sample_1000) == TARGET, f"Expected {TARGET}, got {len(sample_1000)}"
print(f"\n  1,000-sample stratified subset created.")

with open(os.path.join(OUT_DIR, "test_split_1000.json"), "w") as f:
    json.dump(sample_1000, f)

# ─── Stats ───────────────────────────────────────────────────
stats_lines = [
    f"GUI 360 Dataset Split Statistics",
    f"{'='*50}",
    f"Total samples:     {N:,}",
    f"Train:             {len(train_idx):,} ({len(train_idx)/N*100:.1f}%)",
    f"Test:              {len(test_idx):,} ({len(test_idx)/N*100:.1f}%)",
    f"Eval subset:       {TARGET:,} (stratified from test split)",
    f"",
    f"Top strata in test split:",
]
test_strata_top = sorted(test_strata.items(), key=lambda x: -len(x[1]))[:15]
for (app, act), idxs in test_strata_top:
    stats_lines.append(f"  {app:15s} / {act:10s}: {len(idxs):5d}")

stats_txt = "\n".join(stats_lines)
print("\n" + stats_txt)
with open(os.path.join(LOG_DIR, "split_stats.txt"), "w") as f:
    f.write(stats_txt)

print(f"\nSaved:")
print(f"  {OUT_DIR}/test_split_indices.json  ({len(test_idx):,} test indices)")
print(f"  {OUT_DIR}/test_split_1000.json     (1,000 eval indices)")
print(f"  {LOG_DIR}/split_stats.txt")
