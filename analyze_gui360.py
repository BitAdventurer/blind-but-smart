"""
GUI-360 Dataset Statistical Analysis & Visualization
=====================================================
Generates plots and statistics for the three task datasets:
  - Screen Parsing  (97,351 train / 1,969 bench)
  - Grounding       (79,487 train / 1,585 bench)
  - Action Prediction (101,800 train / 1,972 bench)

Outputs:
  results/analysis/*.png  — visualizations
  results/analysis/stats.md — markdown summary
"""

import json
import os
import re
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
import io

mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False

BASE_DIR = Path(os.environ.get("GUI360_ROOT", Path.cwd()))
OUT_DIR = Path(os.environ.get("HMDP_ANALYSIS_DIR", "results/analysis"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATHS = {
    "parsing":    str(BASE_DIR / "gui360_full/processed_data/screen_parsing_train_resize/training_data.json"),
    "grounding":  str(BASE_DIR / "gui360_full/processed_data/grounding_resize/training_data.json"),
    "prediction": str(BASE_DIR / "gui360_full/processed_data/action_prediction_train_resize/training_data.json"),
}
BENCH_PATHS = {
    "parsing":    str(BASE_DIR / "gui360_bench/desktop/understanding/eval/screen_parsing.parquet"),
    "grounding":  str(BASE_DIR / "gui360_bench/desktop/grounding/point/eval/point.parquet"),
    "prediction": str(BASE_DIR / "gui360_bench/desktop/grounding/action/eval/action.parquet"),
}
FAIL_PATH = str(BASE_DIR / "gui360_full/converted_fail_data.json")

# Consistent color palette
APP_COLORS = {"excel": "#1F7244", "word": "#2A579A", "ppt": "#B7472A", "other": "#888"}
TASK_COLORS = {"parsing": "#5D3FD3", "grounding": "#FF6B35", "prediction": "#0EAD69"}
SPLIT_COLORS = {"train": "#3498db", "bench": "#9b59b6", "fail": "#e74c3c"}

SAMPLE_LIMIT = 5000  # sample size for stats from train data
FAIL_SAMPLE_LIMIT = 5000  # sample size for fail data


# ───────────────────────────────────────────────────────────────────────
#  Loader
# ───────────────────────────────────────────────────────────────────────

def classify_app(image_path: str) -> str:
    p = image_path.lower().replace("\\", "/")
    if "excel" in p: return "excel"
    if "word" in p:  return "word"
    if "ppt" in p or "powerpoint" in p: return "ppt"
    return "other"


def load_train_samples(task: str, n: int = SAMPLE_LIMIT) -> list:
    with open(TRAIN_PATHS[task]) as f:
        data = json.load(f)
    random.seed(42)
    return random.sample(data, min(n, len(data)))


def load_bench_samples(task: str) -> pd.DataFrame:
    return pd.read_parquet(BENCH_PATHS[task])


def load_fail_samples(n: int = FAIL_SAMPLE_LIMIT) -> list:
    """Load fail data samples."""
    with open(FAIL_PATH) as f:
        data = json.load(f)
    random.seed(42)
    return random.sample(data, min(n, len(data)))


# ───────────────────────────────────────────────────────────────────────
#  Plot 1: App distribution per task
# ───────────────────────────────────────────────────────────────────────

def plot_app_distribution():
    print("[1/7] App distribution …")
    rows = []
    for task in ["parsing", "grounding", "prediction"]:
        for split, loader in [("train", lambda t: [s["images"][0] for s in load_train_samples(t)]),
                              ("bench", lambda t: [r["images"][0]["bytes"][:0] or r["metadata"]["others"].get("id", "")
                                                    for _, r in load_bench_samples(t).iterrows()]),
                              ("fail", lambda t: [s.get("images", [""])[0] for s in load_fail_samples()])]:
            paths = loader(task)
            apps = Counter(classify_app(p) for p in paths)
            total = sum(apps.values())
            for app in ["excel", "word", "ppt", "other"]:
                pct = 100 * apps.get(app, 0) / max(total, 1)
                rows.append({"task": task, "split": split, "app": app, "pct": pct})
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, split in zip(axes, ["train", "bench", "fail"]):
        sub = df[df["split"] == split].pivot(index="task", columns="app", values="pct").fillna(0)
        sub = sub[["excel", "word", "ppt", "other"]] if "other" in sub.columns else sub[["excel", "word", "ppt"]]
        sub.plot(kind="bar", stacked=True, ax=ax,
                 color=[APP_COLORS[c] for c in sub.columns],
                 edgecolor="white", width=0.6)
        ax.set_title(f"Application Distribution — {split.upper()}", fontsize=13, weight="bold")
        ax.set_ylabel("Percentage (%)")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title="App", loc="upper right", bbox_to_anchor=(1.18, 1))
        ax.set_ylim(0, 100)
        for c in ax.containers:
            ax.bar_label(c, fmt="%.0f%%", label_type="center", color="white", fontsize=8, weight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_app_distribution.png", dpi=130, bbox_inches="tight")
    plt.close()
    return df


# ───────────────────────────────────────────────────────────────────────
#  Plot 2: Element type distribution (parsing)
# ───────────────────────────────────────────────────────────────────────

def plot_element_types():
    print("[2/6] Element type distribution …")
    samples = load_train_samples("parsing", n=500)
    counter = Counter()
    n_elements_per_sample = []
    for s in samples:
        try:
            elems = json.loads(s["conversation"][1]["value"])
            n_elements_per_sample.append(len(elems))
            for e in elems:
                counter[e.get("control_type", "Unknown")] += 1
        except Exception:
            pass
    top = counter.most_common(15)
    types, counts = zip(*top)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart of types
    bars = ax1.barh(range(len(types)), counts, color=TASK_COLORS["parsing"], alpha=0.85)
    ax1.set_yticks(range(len(types)))
    ax1.set_yticklabels(types)
    ax1.invert_yaxis()
    ax1.set_xlabel("Total count (across 500 samples)")
    ax1.set_title("Top 15 UI Control Types", fontsize=13, weight="bold")
    ax1.set_xscale("log")
    for bar, c in zip(bars, counts):
        ax1.text(c, bar.get_y() + bar.get_height()/2, f" {c:,}", va="center", fontsize=9)

    # Histogram of element counts
    ax2.hist(n_elements_per_sample, bins=40, color=TASK_COLORS["parsing"], alpha=0.8, edgecolor="white")
    ax2.set_xlabel("# Elements per Screenshot")
    ax2.set_ylabel("# Samples")
    median_v = np.median(n_elements_per_sample)
    mean_v = np.mean(n_elements_per_sample)
    ax2.axvline(median_v, color="red", linestyle="--", linewidth=2, label=f"Median = {median_v:.0f}")
    ax2.axvline(mean_v, color="orange", linestyle="--", linewidth=2, label=f"Mean = {mean_v:.0f}")
    ax2.set_title(f"Elements/Screenshot — Parsing (n={len(samples)})", fontsize=13, weight="bold")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "02_element_types.png", dpi=130, bbox_inches="tight")
    plt.close()
    return counter, n_elements_per_sample


# ───────────────────────────────────────────────────────────────────────
#  Plot 3: Action type distribution (prediction)
# ───────────────────────────────────────────────────────────────────────

def plot_action_distribution():
    print("[3/6] Action type distribution …")
    # Train: parse from GT text
    train_samples = load_train_samples("prediction", n=3000)
    train_acts = Counter()
    for s in train_samples:
        gt = s["conversation"][1]["value"]
        m = re.search(r'"function"\s*:\s*"([^"]+)"', gt)
        if m:
            train_acts[m.group(1)] += 1
        else:
            train_acts["unknown"] += 1

    # Bench: parse from tool_calls
    bench_df = load_bench_samples("prediction")
    bench_acts = Counter()
    for _, row in bench_df.iterrows():
        for m in row["messages"]:
            if m.get("role") == "assistant":
                tcs = m.get("tool_calls")
                if tcs is not None:
                    for tc in tcs:
                        fn = tc.get("function", {})
                        bench_acts[fn.get("name", "unknown")] += 1
    
    # Parse fail data actions (same format as train)
    fail_samples = load_fail_samples(3000)
    fail_acts = Counter()
    for s in fail_samples:
        gt = s["conversation"][1]["value"]
        m = re.search(r'"function"\s*:\s*"([^"]+)"', gt)
        if m:
            fail_acts[m.group(1)] += 1
        else:
            fail_acts["unknown"] += 1

    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    for ax, acts, split in zip(axes, [train_acts, bench_acts, fail_acts], 
                                ["TRAIN (n=3,000)", f"BENCH (n={len(bench_df)})", f"FAIL (n={len(fail_samples)})"]):
        # Sort by count desc, take top 10
        items = acts.most_common(10)
        labels, vals = zip(*items)
        colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
        
        # Use horizontal bar chart instead of pie to avoid label overlap
        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, vals, color=colors, edgecolor="white", height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Count", fontsize=11)
        ax.set_title(f"Action Distribution — {split}", fontsize=13, weight="bold")
        
        # Add percentage labels on bars
        total = sum(vals)
        for bar, val in zip(bars, vals):
            pct = 100 * val / total
            ax.text(val + max(vals)*0.01, bar.get_y() + bar.get_height()/2,
                   f" {val:,} ({pct:.1f}%)", va='center', fontsize=9, color='black')
        
        ax.set_xlim(0, max(vals) * 1.3)
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / "03_action_distribution.png", dpi=130, bbox_inches="tight")
    plt.close()
    return train_acts, bench_acts, fail_acts


# ───────────────────────────────────────────────────────────────────────
#  Plot 4: GT coordinate heatmap (grounding + prediction)
# ───────────────────────────────────────────────────────────────────────

def plot_coordinate_heatmap():
    print("[4/7] Coordinate heatmaps …")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Grounding train
    samples = load_train_samples("grounding", n=5000)
    xs, ys = [], []
    for s in samples:
        gt = s["conversation"][1]["value"]
        m = re.search(r"\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]", gt)
        if m:
            x, y = float(m.group(1)), float(m.group(2))
            # Filter outliers (multi-monitor)
            if 0 < x < 1200 and 0 < y < 800:
                xs.append(x); ys.append(y)

    ax = axes[0]
    h = ax.hist2d(xs, ys, bins=60, cmap="YlOrRd", cmin=1)
    ax.invert_yaxis()
    ax.set_xlim(0, 1040); ax.set_ylim(740, 0)
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_title(f"Grounding GT Click Density (Train, n={len(xs):,})", fontsize=13, weight="bold")
    plt.colorbar(h[3], ax=ax, label="Frequency")

    # Prediction train (from bbox centers)
    samples = load_train_samples("prediction", n=5000)
    xs, ys = [], []
    for s in samples:
        bbox = s.get("bbox")
        if bbox and len(bbox) == 4:
            cx, cy = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
            if 0 < cx < 1200 and 0 < cy < 800:
                xs.append(cx); ys.append(cy)

    ax = axes[1]
    h = ax.hist2d(xs, ys, bins=60, cmap="YlGnBu", cmin=1)
    ax.invert_yaxis()
    ax.set_xlim(0, 1040); ax.set_ylim(740, 0)
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_title(f"Prediction Target BBox Centers (Train, n={len(xs):,})", fontsize=13, weight="bold")
    plt.colorbar(h[3], ax=ax, label="Frequency")
    
    # Fail data bbox centers
    fail_samples = load_fail_samples(5000)
    xs_fail, ys_fail = [], []
    for s in fail_samples:
        bbox = s.get("bbox")
        if bbox and len(bbox) == 4:
            cx, cy = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
            if 0 < cx < 1200 and 0 < cy < 800:
                xs_fail.append(cx); ys_fail.append(cy)
    
    ax = axes[2]
    h = ax.hist2d(xs_fail, ys_fail, bins=60, cmap="OrRd", cmin=1)
    ax.invert_yaxis()
    ax.set_xlim(0, 1040); ax.set_ylim(740, 0)
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_title(f"Fail Data Target BBox Centers (n={len(xs_fail):,})", fontsize=13, weight="bold")
    plt.colorbar(h[3], ax=ax, label="Frequency")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "04_coordinate_heatmap.png", dpi=130, bbox_inches="tight")
    plt.close()


# ───────────────────────────────────────────────────────────────────────
#  Plot 5: BBox size distribution
# ───────────────────────────────────────────────────────────────────────

def plot_bbox_sizes():
    print("[5/7] BBox size distribution …")
    
    # Train data
    samples = load_train_samples("prediction", n=5000)
    widths_train, heights_train, area_ratios_train = [], [], []
    img_area = 1036 * 728
    for s in samples:
        b = s.get("bbox")
        if b and len(b) == 4:
            w, h = b[2]-b[0], b[3]-b[1]
            if w > 0 and h > 0 and w < 1100 and h < 800:
                widths_train.append(w); heights_train.append(h)
                area_ratios_train.append(100*(w*h)/img_area)
    
    # Fail data
    fail_samples = load_fail_samples(5000)
    widths_fail, heights_fail, area_ratios_fail = [], [], []
    for s in fail_samples:
        b = s.get("bbox")
        if b and len(b) == 4:
            w, h = b[2]-b[0], b[3]-b[1]
            if w > 0 and h > 0 and w < 1100 and h < 800:
                widths_fail.append(w); heights_fail.append(h)
                area_ratios_fail.append(100*(w*h)/img_area)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Train: Width vs Height scatter
    ax = axes[0, 0]
    ax.scatter(widths_train, heights_train, s=4, alpha=0.3, color=SPLIT_COLORS["train"], label="Train")
    ax.set_xlabel("BBox Width (pixels)")
    ax.set_ylabel("BBox Height (pixels)")
    ax.set_title(f"Train: BBox Dimensions (n={len(widths_train):,})", fontsize=12, weight="bold")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Train: Area % histogram
    ax = axes[0, 1]
    ax.hist(area_ratios_train, bins=50, range=(0, 30), color=SPLIT_COLORS["train"], alpha=0.85, edgecolor="white")
    ax.set_xlabel("BBox Area / Screen Area (%)")
    ax.set_ylabel("# Samples")
    med_train = np.median(area_ratios_train)
    ax.axvline(med_train, color="red", linestyle="--", linewidth=2, label=f"Median = {med_train:.2f}%")
    ax.set_title("Train: Size Distribution", fontsize=12, weight="bold")
    ax.legend()
    
    # Fail: Width vs Height scatter
    ax = axes[1, 0]
    ax.scatter(widths_fail, heights_fail, s=4, alpha=0.3, color=SPLIT_COLORS["fail"], label="Fail")
    ax.set_xlabel("BBox Width (pixels)")
    ax.set_ylabel("BBox Height (pixels)")
    ax.set_title(f"Fail: BBox Dimensions (n={len(widths_fail):,})", fontsize=12, weight="bold")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Fail: Area % histogram
    ax = axes[1, 1]
    ax.hist(area_ratios_fail, bins=50, range=(0, 30), color=SPLIT_COLORS["fail"], alpha=0.85, edgecolor="white")
    ax.set_xlabel("BBox Area / Screen Area (%)")
    ax.set_ylabel("# Samples")
    med_fail = np.median(area_ratios_fail)
    ax.axvline(med_fail, color="red", linestyle="--", linewidth=2, label=f"Median = {med_fail:.2f}%")
    ax.set_title("Fail: Size Distribution", fontsize=12, weight="bold")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / "05_bbox_sizes.png", dpi=130, bbox_inches="tight")
    plt.close()
    return area_ratios_train, area_ratios_fail


# ───────────────────────────────────────────────────────────────────────
#  Plot 6: Instruction length distribution
# ───────────────────────────────────────────────────────────────────────

def plot_instruction_lengths():
    print("[6/7] Instruction length distribution …")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    data = {}
    for i, task in enumerate(["grounding", "prediction"]):
        # Train data
        samples = load_train_samples(task, n=5000)
        lengths_train = []
        for s in samples:
            human = s["conversation"][0]["value"]
            m = re.search(r"[Tt]he instruction is:\s*(.+?)(?:\n\n|The history|The actions|Output)",
                          human, re.DOTALL)
            if not m:
                m = re.search(r"[Ii]nstruction(?:s)?:\s*(.+?)(?:\n\n|Output)", human, re.DOTALL)
            if m:
                lengths_train.append(len(m.group(1).split()))
        
        # Fail data
        fail_samples = load_fail_samples(5000)
        lengths_fail = []
        for s in fail_samples:
            human = s["conversation"][0]["value"]
            m = re.search(r"[Tt]he instruction is:\s*(.+?)(?:\n\n|The history|The actions|Output)",
                          human, re.DOTALL)
            if not m:
                m = re.search(r"[Ii]nstruction(?:s)?:\s*(.+?)(?:\n\n|Output)", human, re.DOTALL)
            if m:
                lengths_fail.append(len(m.group(1).split()))
        
        data[task] = {"train": lengths_train, "fail": lengths_fail}
        
        # Plot train
        ax = axes[0, i]
        ax.hist(lengths_train, bins=40, color=SPLIT_COLORS["train"], alpha=0.85, edgecolor="white")
        med = np.median(lengths_train)
        ax.axvline(med, color="red", linestyle="--", linewidth=2, label=f"Median = {med:.0f} words")
        ax.set_xlabel("# Words in Instruction")
        ax.set_ylabel("# Samples")
        ax.set_title(f"Train: {task.title()} (n={len(lengths_train):,})", fontsize=12, weight="bold")
        ax.legend()
        if lengths_train:
            ax.set_xlim(0, np.percentile(lengths_train, 99) * 1.1)
        
        # Plot fail
        ax = axes[1, i]
        ax.hist(lengths_fail, bins=40, color=SPLIT_COLORS["fail"], alpha=0.85, edgecolor="white")
        med = np.median(lengths_fail)
        ax.axvline(med, color="red", linestyle="--", linewidth=2, label=f"Median = {med:.0f} words")
        ax.set_xlabel("# Words in Instruction")
        ax.set_ylabel("# Samples")
        ax.set_title(f"Fail: {task.title()} (n={len(lengths_fail):,})", fontsize=12, weight="bold")
        ax.legend()
        if lengths_fail:
            ax.set_xlim(0, np.percentile(lengths_fail, 99) * 1.1)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / "06_instruction_lengths.png", dpi=130, bbox_inches="tight")
    plt.close()
    return data


# ───────────────────────────────────────────────────────────────────────
#  Summary Markdown
# ───────────────────────────────────────────────────────────────────────

def write_summary(app_df, type_counter, n_elem, train_acts, bench_acts, fail_acts, area_ratios_train, area_ratios_fail, instr_data):
    print("Writing markdown summary …")
    lines = [
        "# GUI-360 Dataset — Statistical Analysis (with Fail Data)\n",
        f"_Generated: analyze_gui360.py_\n",
        "## 1. Sample Counts\n",
        "| Task | Train | Bench | Fail | Total |",
        "|------|------:|------:|-----:|------:|",
        "| Screen Parsing  | 97,351 | 1,969 | 1,093,525 | 1,192,845 |",
        "| Grounding       | 79,487 | 1,585 | 1,093,525 | 1,174,597 |",
        "| Action Prediction | 101,800 | 1,972 | 1,093,525 | 1,197,297 |\n",

        "## 2. Application Distribution\n",
        "See `01_app_distribution.png`. Train/Bench/Fail all roughly balanced across Excel/Word/PPT.\n",

        "## 3. UI Element Statistics (Parsing)\n",
        f"- Mean elements/screenshot: **{np.mean(n_elem):.1f}**",
        f"- Median elements/screenshot: **{np.median(n_elem):.0f}**",
        f"- 95th percentile: {np.percentile(n_elem, 95):.0f}",
        f"- Max: {max(n_elem)}",
        "\n**Top control types**:\n",
    ]
    for typ, cnt in type_counter.most_common(10):
        lines.append(f"  - `{typ}`: {cnt:,}")
    lines.append("\n> ⚠️ `DataItem` (Excel cells) dominates → use `salient_only=True` for fair eval.")

    lines += [
        "\n## 4. Action Distribution (Prediction)\n",
        "**Train (n=3,000):**\n",
    ]
    for act, cnt in train_acts.most_common(8):
        pct = 100 * cnt / sum(train_acts.values())
        lines.append(f"  - `{act}`: {cnt:,} ({pct:.1f}%)")
    lines.append("\n**Bench (full):**\n")
    for act, cnt in bench_acts.most_common(8):
        pct = 100 * cnt / sum(bench_acts.values())
        lines.append(f"  - `{act}`: {cnt:,} ({pct:.1f}%)")
    lines.append("\n**Fail (n=3,000 sample):**\n")
    for act, cnt in fail_acts.most_common(8):
        pct = 100 * cnt / sum(fail_acts.values())
        lines.append(f"  - `{act}`: {cnt:,} ({pct:.1f}%)")

    lines += [
        "\n## 5. Target BBox Sizes (Prediction)\n",
        "**Train:**\n",
        f"- Median area / screen: **{np.median(area_ratios_train):.2f}%**",
        f"- 10th percentile: {np.percentile(area_ratios_train, 10):.2f}%",
        f"- 90th percentile: {np.percentile(area_ratios_train, 90):.2f}%",
        "\n**Fail:**\n",
        f"- Median area / screen: **{np.median(area_ratios_fail):.2f}%**",
        f"- 10th percentile: {np.percentile(area_ratios_fail, 10):.2f}%",
        f"- 90th percentile: {np.percentile(area_ratios_fail, 90):.2f}%",
        "\n> Many targets are tiny (< 1% of screen) → precision is critical for grounding/prediction.",
    ]

    lines += [
        "\n## 6. Instruction Length (words)\n",
        "| Task | Split | Median | 90th-%ile |",
        "|------|-------|-------:|----------:|",
    ]
    for task, splits in instr_data.items():
        for split, lengths in splits.items():
            lines.append(f"| {task.title()} | {split} | {np.median(lengths):.0f} | {np.percentile(lengths, 90):.0f} |")

    lines += [
        "\n## 7. Key Pitfalls (Handled)\n",
        "| Issue | Resolution |",
        "|-------|-----------|",
        "| Windows backslash paths (Grounding train) | `os.path.normpath` after `\\\\` → `/` |",
        "| Train data 100% success (reward=1) | Use **GUI-360-Bench** parquet eval set |",
        "| Fail data for negative RL samples | See `analyze_fail_data.py` for trajectory analysis |",
        "| Parsing 400+ elements/screen | `salient_only=True` filter (Button/Menu/Edit/…) |",
        "| Out-of-bound GT coords (multi-monitor) | Skip samples where GT > 1.05×img_size |",
        "| Image actual size ≠ metadata resolution | Use VLM's image PIL size for normalization |",
    ]

    (OUT_DIR / "stats.md").write_text("\n".join(lines))


# ───────────────────────────────────────────────────────────────────────
#  Main
# ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Output dir: {OUT_DIR}")
    app_df = plot_app_distribution()
    type_counter, n_elem = plot_element_types()
    train_acts, bench_acts, fail_acts = plot_action_distribution()
    plot_coordinate_heatmap()
    area_ratios_train, area_ratios_fail = plot_bbox_sizes()
    instr_data = plot_instruction_lengths()
    write_summary(app_df, type_counter, n_elem, train_acts, bench_acts, fail_acts, area_ratios_train, area_ratios_fail, instr_data)
    print(f"\n✓ Done. See {OUT_DIR}/")
    print("\nGenerated files:")
    for f in sorted(OUT_DIR.iterdir()):
        if f.suffix == ".png":
            print(f"  - {f.name}")
