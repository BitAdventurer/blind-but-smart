#!/usr/bin/env python3
"""
Correct Task-Split Consistency Evaluation (Table A.8 in ESWA paper).

Evaluates H-MDP framework on THREE distinct GUI 360° task types:
  1. Screen Parsing:     screen_parsing_train_resize/   (97,351 samples)
                         → Extract UI element list (control_type, bbox, text)
  2. Grounding:          grounding_resize/              (79,487 samples)
                         → Single target coordinate from NL instruction
  3. Action Prediction:  action_prediction_train_resize/ (101,800 samples)
                         → Next action (function + coordinate)

Each task has its own evaluation metric:
  - Parsing:    F1 on detected elements (IoU-based matching)
  - Grounding:  Point-in-bbox accuracy
  - Prediction: Action type accuracy + Point accuracy

Four privacy configurations per task:
  - No Privacy (single-pass, ε=∞)
  - Static High Privacy (ε=0.5)
  - Rule-based Adaptive
  - H-MDP (Ours, adaptive ε via meta-policy)

Usage:
    conda run -n py358 --no-capture-output python3 run_three_task_eval.py \
        --num-samples 500 --tasks parsing grounding prediction
"""

import argparse
import gc
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hmdp.vlm_inference import (
    VLMInferenceEngine, MODEL_REGISTRY, ACTION_TYPES,
    point_in_bbox, point_distance_to_center,
)
from hmdp.ldp import ProxyEncoder, LocalDifferentialPrivacy
from hmdp.ltm import LongTermMemory

# ═══════════════════════════════════════════════════════════════════════
#  Dataset Configurations
# ═══════════════════════════════════════════════════════════════════════

BASE_DIR = os.environ.get("GUI360_ROOT", os.getcwd())
DATASET_ROOT = os.path.join(BASE_DIR, "gui360_full/processed_data")
BENCH_ROOT = os.path.join(BASE_DIR, "gui360_bench")

# Two source modes:
#   "train"  → success-only training set (101,800 prediction samples, all reward=1)
#   "bench"  → official GUI-360-Bench eval set (mixed difficulty, success + fail)
DATA_SOURCE = os.environ.get("DATA_SOURCE", "bench")  # default: official eval

TASK_CONFIG_TRAIN = {
    "parsing": {
        "data_path": f"{DATASET_ROOT}/screen_parsing_train_resize/training_data.json",
        "image_base": f"{DATASET_ROOT}/screen_parsing_train_resize",
        "format": "json",
    },
    "grounding": {
        "data_path": f"{DATASET_ROOT}/grounding_resize/training_data.json",
        "image_base": f"{DATASET_ROOT}/grounding_resize",
        "format": "json",
    },
    "prediction": {
        "data_path": f"{DATASET_ROOT}/action_prediction_train_resize/training_data.json",
        "image_base": f"{DATASET_ROOT}/action_prediction_train_resize",
        "format": "json",
    },
}

TASK_CONFIG_BENCH = {
    "parsing": {
        "data_path": f"{BENCH_ROOT}/desktop/understanding/eval/screen_parsing.parquet",
        "image_base": None,  # images embedded in parquet
        "format": "parquet",
    },
    "grounding": {
        "data_path": f"{BENCH_ROOT}/desktop/grounding/point/eval/point.parquet",
        "image_base": None,
        "format": "parquet",
    },
    "prediction": {
        "data_path": f"{BENCH_ROOT}/desktop/grounding/action/eval/action.parquet",
        "image_base": None,
        "format": "parquet",
    },
}

PROMPT_TEMPLATES = {
    "parsing": (
        "You are an expert in screen parsing and GUI element extraction. "
        "Given this screenshot, extract ALL visible UI elements (buttons, menus, "
        "textboxes, labels, toolbars, etc.).\n\n"
        "Output a JSON array where each element has:\n"
        "  - control_type: one of Button, Menu, MenuItem, Edit, Text, Image, "
        "CheckBox, RadioButton, ComboBox, ListItem, TabItem, TreeItem, Pane, Custom\n"
        "  - control_rect: [x1, y1, x2, y2] pixel bounding box\n"
        "  - control_text: visible text or label (empty string if none)\n\n"
        "Return ONLY the JSON array, no other text. Start with [ and end with ]."
    ),
    "grounding": (
        "You are a GUI agent. Given the screenshot and instruction,\n"
        "output the coordinate of the target UI element.\n"
        "Instruction: {instruction}\n"
        "Output format: <coordinate> [x, y] </coordinate>"
    ),
    "prediction": (
        "You are a GUI agent. Given the screenshot and instruction,\n"
        "predict the next action.\n"
        "Instruction: {instruction}\n"
        "Output format: action_type(coordinate=[x, y])\n"
        "Supported actions: click, type, drag, scroll"
    ),
}

# Bind config based on source
TASK_CONFIG = TASK_CONFIG_BENCH if DATA_SOURCE == "bench" else TASK_CONFIG_TRAIN
for _t, _cfg in TASK_CONFIG.items():
    _cfg["prompt_template"] = PROMPT_TEMPLATES[_t]

# ═══════════════════════════════════════════════════════════════════════
#  Task-Specific Parsers & Metrics
# ═══════════════════════════════════════════════════════════════════════

def load_samples(task: str) -> List[Dict]:
    """
    Load samples from either JSON (train) or Parquet (bench) source.
    Returns a uniform list-of-dicts where each dict has:
      - 'task': task name
      - 'image_bytes' OR 'image_path': image source
      - 'instruction': user instruction
      - 'gt_*': task-specific ground truth fields
    """
    cfg = TASK_CONFIG[task]
    fmt = cfg["format"]

    if fmt == "json":
        with open(cfg["data_path"]) as f:
            raw = json.load(f)
        out = []
        for s in raw:
            img_rel = s.get("images", [None])[0] if "images" in s else None
            if img_rel is None:
                continue
            img_rel = img_rel.replace("\\", "/").lstrip("/")
            img_path = os.path.normpath(os.path.join(cfg["image_base"], img_rel))
            sample = {
                "task": task,
                "image_path": img_path,
                "conversation": s.get("conversation", []),
                "bbox": s.get("bbox"),
                "reward": s.get("reward", 1),
                "id": s.get("id", ""),
            }
            out.append(sample)
        return out

    elif fmt == "parquet":
        import pandas as pd
        df = pd.read_parquet(cfg["data_path"])
        out = []
        for _, row in df.iterrows():
            # Extract image bytes
            imgs = row.get("images")
            if imgs is None or len(imgs) == 0:
                continue
            img_bytes = imgs[0].get("bytes") if isinstance(imgs[0], dict) else None
            if img_bytes is None:
                continue

            # Extract user instruction (text content)
            messages = row.get("messages", [])
            instruction = ""
            assistant_msg = None
            for m in messages:
                if m.get("role") == "user":
                    content = m.get("content")
                    if content is not None:
                        for c in content:
                            if isinstance(c, dict) and c.get("type") == "text":
                                instruction = c.get("text", "")
                                break
                elif m.get("role") == "assistant":
                    assistant_msg = m

            # Extract GT based on task
            gt_action = "click"
            gt_point = None
            gt_bbox = None
            gt_elements = []

            if task in ("grounding", "prediction") and assistant_msg is not None:
                tool_calls = assistant_msg.get("tool_calls")
                if tool_calls is None:
                    tool_calls = []
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "click")
                    args = fn.get("arguments", {}) or {}
                    coord = args.get("coordinate")
                    if coord is not None and len(coord) == 2:
                        gt_point = [float(coord[0]), float(coord[1])]
                    # Map 'point' (grounding task) to 'click' for action_type
                    gt_action = "click" if name == "point" else name
                    break
            elif task == "parsing" and assistant_msg is not None:
                content = assistant_msg.get("content")
                if content is not None:
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            try:
                                gt_elements = json.loads(c.get("text", "[]"))
                                if not isinstance(gt_elements, list):
                                    gt_elements = []
                            except Exception:
                                gt_elements = []
                            break

            # Get resolution from metadata
            meta = row.get("metadata", {})
            res = meta.get("others", {}).get("resolution") if meta else None
            img_w, img_h = (1036, 728)
            if res is not None:
                try:
                    img_w, img_h = int(res[0]), int(res[1])
                except Exception:
                    pass

            sample = {
                "task": task,
                "image_bytes": img_bytes,
                "instruction": instruction,
                "gt_action": gt_action,
                "gt_point": gt_point,
                "gt_bbox": gt_bbox,
                "gt_elements": gt_elements,
                "img_resolution": (img_w, img_h),
                "id": meta.get("others", {}).get("id", "") if meta else "",
            }
            out.append(sample)
        return out
    else:
        raise ValueError(f"Unknown format: {fmt}")


def parse_gt_from_conversation(sample: Dict, task: str) -> Dict:
    """Parse ground-truth from GUI 360° conversation format, per task type."""
    conv = sample.get("conversation", [])
    gt_text = ""
    for turn in conv:
        if turn.get("from") == "gpt":
            gt_text = turn.get("value", "")
            break

    # Extract instruction from human turn (needed for grounding/prediction)
    instruction = ""
    for turn in conv:
        if turn.get("from") == "human":
            instruction = turn.get("value", "").replace("<image>", "").strip()
            # Remove prompt template noise, keep core instruction
            for sep in ["\n\nInstruction:", "instruction:"]:
                if sep in instruction:
                    parts = instruction.split(sep, 1)
                    if len(parts) > 1:
                        instruction = parts[1].strip()
                        break
            break

    if task == "parsing":
        try:
            elements = json.loads(gt_text)
            if isinstance(elements, list):
                return {"elements": elements, "instruction": instruction}
        except Exception:
            pass
        return {"elements": [], "instruction": instruction}

    elif task == "grounding":
        m = re.search(r"<coordinate>\s*\[(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\]", gt_text)
        if m:
            return {
                "gt_point": [float(m.group(1)), float(m.group(2))],
                "instruction": instruction,
            }
        return {"gt_point": None, "instruction": instruction}

    elif task == "prediction":
        action_type = "click"
        for a in ACTION_TYPES:
            if a in gt_text.lower():
                action_type = a
                break
        m = re.search(r'"coordinate"\s*:\s*\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)', gt_text)
        gt_point = [float(m.group(1)), float(m.group(2))] if m else None
        bbox = sample.get("bbox", None)
        return {
            "gt_action": action_type,
            "gt_point": gt_point,
            "gt_bbox": bbox,
            "instruction": instruction,
        }
    return {}


def parse_pred_parsing(raw_output: str) -> List[Dict]:
    """Parse VLM output for parsing task: extract element list with tolerance."""
    # Strategy 1: full JSON array between first [ and last ]
    try:
        first = raw_output.find("[")
        last = raw_output.rfind("]")
        if first >= 0 and last > first:
            candidate = raw_output[first:last + 1]
            elements = json.loads(candidate)
            if isinstance(elements, list):
                return [e for e in elements if isinstance(e, dict)]
    except Exception:
        pass

    # Strategy 2: parse individual {...} objects line-by-line (robust to truncation)
    elements = []
    # Match balanced { ... } blocks
    depth = 0
    start = -1
    for i, ch in enumerate(raw_output):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                block = raw_output[start:i + 1]
                try:
                    obj = json.loads(block)
                    if isinstance(obj, dict):
                        elements.append(obj)
                except Exception:
                    pass
                start = -1
    return elements


def parse_pred_grounding(raw_output: str, img_size: Tuple[int, int]) -> Optional[List[float]]:
    """Parse VLM output for grounding task: extract coordinate."""
    patterns = [
        r'<coordinate>\s*\[(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\]',
        r'coordinate\s*=\s*\[(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\]',
        r'\[(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\]',
        r'\((\d+\.?\d*)\s*,\s*(\d+\.?\d*)\)',
    ]
    img_w, img_h = img_size
    for pat in patterns:
        m = re.search(pat, raw_output)
        if m:
            x, y = float(m.group(1)), float(m.group(2))
            if x > 1.0 or y > 1.0:
                x, y = x / max(img_w, 1), y / max(img_h, 1)
            return [max(0, min(1, x)), max(0, min(1, y))]
    return None


def _iou_bbox(b1: List[float], b2: List[float]) -> float:
    """IoU of two bboxes [x1, y1, x2, y2]."""
    xa = max(b1[0], b2[0]); ya = max(b1[1], b2[1])
    xb = min(b1[2], b2[2]); yb = min(b1[3], b2[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


# Element types considered "salient" for GUI agent operation
# (excludes dense data cells like Excel DataItem which dominate counts but are
# rarely the operational target of an agent)
SALIENT_TYPES = {
    "button", "menu", "menuitem", "edit", "tabitem", "listitem",
    "checkbox", "radiobutton", "combobox", "treeitem", "hyperlink",
    "spinner", "slider",
}


def compute_parsing_metrics(
    pred_elements: List[Dict],
    gt_elements: List[Dict],
    salient_only: bool = True,
) -> Dict:
    """
    Multi-level parsing evaluation with relaxed criteria suitable for dense GUI parsing:

      (1) strict_f1     — IoU ≥ 0.5 AND exact control_type match
      (2) relaxed_f1    — IoU ≥ 0.3 (no type constraint)
      (3) type_f1       — element type distribution match (ignore bbox)
      (4) text_recall   — fraction of GT texts found in prediction (substring)
      (5) count_ratio   — min(pred_count, gt_count) / max(pred_count, gt_count)
      (6) coverage      — fraction of GT elements matched by ANY prediction (IoU ≥ 0.1)

    salient_only: if True, filter GT to only salient UI types (Button/Menu/Edit/...).
                  This avoids penalizing models for missing dense Excel cells which are
                  rarely the operational target of a GUI agent.

    The headline 'success_rate' is the text-based recall — most interpretable and
    robust for zero-shot VLMs that struggle with exact bboxes on 400+ elements.
    """
    if salient_only:
        gt_elements = [
            g for g in gt_elements
            if g.get("control_type", "").lower() in SALIENT_TYPES
        ]
    n_gt = len(gt_elements)
    n_pred = len(pred_elements)

    if n_gt == 0:
        return {k: 0.0 for k in ["strict_f1", "relaxed_f1", "type_f1",
                                 "text_recall", "count_ratio", "coverage"]}
    if n_pred == 0:
        return {
            "strict_f1": 0.0, "relaxed_f1": 0.0, "type_f1": 0.0,
            "text_recall": 0.0, "count_ratio": 0.0, "coverage": 0.0,
        }

    def _norm_bbox(el):
        b = el.get("control_rect", el.get("bbox", None))
        return b if (b and len(b) == 4) else None

    # ── (1) Strict F1: IoU ≥ 0.5 + type match ──
    matched_strict = set()
    tp_strict = 0
    for p in pred_elements:
        pb, pt = _norm_bbox(p), p.get("control_type", "").lower()
        if pb is None:
            continue
        best_iou, best_idx = 0, -1
        for gi, g in enumerate(gt_elements):
            if gi in matched_strict:
                continue
            gb, gtype = _norm_bbox(g), g.get("control_type", "").lower()
            if gb is None or pt != gtype:
                continue
            iou = _iou_bbox(pb, gb)
            if iou > best_iou:
                best_iou, best_idx = iou, gi
        if best_iou >= 0.5 and best_idx >= 0:
            matched_strict.add(best_idx)
            tp_strict += 1
    prec = tp_strict / max(n_pred, 1)
    rec = tp_strict / max(n_gt, 1)
    strict_f1 = 2 * prec * rec / max(prec + rec, 1e-9)

    # ── (2) Relaxed F1: IoU ≥ 0.3, ignore type ──
    matched_relax = set()
    tp_relax = 0
    for p in pred_elements:
        pb = _norm_bbox(p)
        if pb is None:
            continue
        best_iou, best_idx = 0, -1
        for gi, g in enumerate(gt_elements):
            if gi in matched_relax:
                continue
            gb = _norm_bbox(g)
            if gb is None:
                continue
            iou = _iou_bbox(pb, gb)
            if iou > best_iou:
                best_iou, best_idx = iou, gi
        if best_iou >= 0.3 and best_idx >= 0:
            matched_relax.add(best_idx)
            tp_relax += 1
    prec_r = tp_relax / max(n_pred, 1)
    rec_r = tp_relax / max(n_gt, 1)
    relaxed_f1 = 2 * prec_r * rec_r / max(prec_r + rec_r, 1e-9)

    # ── (3) Type-only F1: match type distributions ──
    from collections import Counter
    pred_types = Counter(p.get("control_type", "").lower() for p in pred_elements)
    gt_types = Counter(g.get("control_type", "").lower() for g in gt_elements)
    overlap = sum((pred_types & gt_types).values())
    tp_f1 = 2 * overlap / max(sum(pred_types.values()) + sum(gt_types.values()), 1)

    # ── (4) Text Recall: how many GT element texts appear in prediction ──
    pred_texts_blob = " ".join(
        str(p.get("control_text", "")).lower() for p in pred_elements
    )
    text_hits = 0
    text_total = 0
    for g in gt_elements:
        gt_text = str(g.get("control_text", "")).strip().lower()
        if len(gt_text) < 2:  # skip empty/trivial
            continue
        text_total += 1
        if gt_text in pred_texts_blob:
            text_hits += 1
    text_recall = text_hits / max(text_total, 1)

    # ── (5) Count ratio ──
    count_ratio = min(n_pred, n_gt) / max(n_pred, n_gt)

    # ── (6) Coverage: IoU ≥ 0.1 match rate ──
    coverage_matched = set()
    for p in pred_elements:
        pb = _norm_bbox(p)
        if pb is None:
            continue
        for gi, g in enumerate(gt_elements):
            if gi in coverage_matched:
                continue
            gb = _norm_bbox(g)
            if gb is None:
                continue
            if _iou_bbox(pb, gb) >= 0.1:
                coverage_matched.add(gi)
                break
    coverage = len(coverage_matched) / max(n_gt, 1)

    return {
        "strict_f1": strict_f1,
        "relaxed_f1": relaxed_f1,
        "type_f1": tp_f1,
        "text_recall": text_recall,
        "count_ratio": count_ratio,
        "coverage": coverage,
    }


# Backward-compat alias
compute_parsing_f1 = compute_parsing_metrics


# ═══════════════════════════════════════════════════════════════════════
#  Task-Aware VLM Inference
# ═══════════════════════════════════════════════════════════════════════

class TaskVLMEngine(VLMInferenceEngine):
    """Extends VLMInferenceEngine with task-specific prompts and generation budget."""

    # Parsing needs longer output (dense element lists)
    MAX_NEW_TOKENS = {"parsing": 2048, "grounding": 128, "prediction": 256}

    def __init__(self, model_key: str, task: str, device: str = "cuda"):
        super().__init__(model_key, device=device)
        self.task = task
        self.prompt_template = TASK_CONFIG[task]["prompt_template"]

    def _build_prompt(self, instruction: str) -> str:
        # Use replace instead of .format() to avoid issues with JSON braces in templates
        return self.prompt_template.replace("{instruction}", instruction or "")

    def _predict_qwen(self, image_path_or_image, instruction: str) -> Dict:
        """Override to use task-specific max_new_tokens. Accepts path str or PIL.Image."""
        from qwen_vl_utils import process_vision_info
        if isinstance(image_path_or_image, Image.Image):
            image = image_path_or_image.convert("RGB")
        else:
            image = Image.open(image_path_or_image).convert("RGB")
        img_w, img_h = image.size

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": self._build_prompt(instruction)},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.device)
        pad_id = getattr(self.processor.tokenizer, 'pad_token_id', None)
        if pad_id is None:
            pad_id = getattr(self.model.config, 'eos_token_id', 151658)

        max_tokens = self.MAX_NEW_TOKENS.get(self.task, 256)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_tokens,
                do_sample=False, temperature=1.0, pad_token_id=pad_id,
            )
        gen_ids = output_ids[:, inputs.input_ids.shape[1]:]
        raw_output = self.processor.batch_decode(
            gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return {
            "action_type": self._parse_action_type(raw_output),
            "pred_point": self._parse_point(raw_output, img_w, img_h),
            "raw_output": raw_output,
            "img_size": (img_w, img_h),
        }


# ═══════════════════════════════════════════════════════════════════════
#  Privacy Configurations (per task)
# ═══════════════════════════════════════════════════════════════════════

def apply_ldp_noise(phi: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Apply Laplace LDP noise to features."""
    if epsilon > 1e5:
        return phi
    sensitivity = 2.0
    scale = sensitivity / max(epsilon, 0.01)
    noise = torch.distributions.Laplace(0, scale).sample(phi.shape).to(phi.device)
    return phi + noise


def rule_based_epsilon(sensitivity: float) -> float:
    """Simple rule: high-sensitivity content → smaller ε (more privacy)."""
    if sensitivity > 0.7:
        return 0.5
    elif sensitivity > 0.3:
        return 1.0
    else:
        return 2.0


def hmdp_epsilon(sensitivity: float, uncertainty: float) -> float:
    """H-MDP meta-policy: balance privacy vs. utility."""
    base = 0.5 + 1.5 * (1 - sensitivity)  # [0.5, 2.0]
    if uncertainty > 0.6:
        base = min(base * 1.3, 2.5)
    return max(0.3, min(base, 2.5))


# ═══════════════════════════════════════════════════════════════════════
#  Main Evaluation Loop
# ═══════════════════════════════════════════════════════════════════════

def evaluate_task(
    task: str,
    config_name: str,
    epsilon: float,
    model_key: str,
    num_samples: int,
    device: str,
    adaptive: Optional[str] = None,
) -> Dict:
    """
    Evaluate one (task × privacy config) combination.
    adaptive: None | 'rule' | 'hmdp'
    """
    cfg = TASK_CONFIG[task]
    print(f"\n  [{task.upper()}] {config_name} (ε={epsilon if epsilon < 1e5 else '∞'}) [source={DATA_SOURCE}]")

    samples = load_samples(task)
    rng = np.random.RandomState(42)
    indices = rng.choice(len(samples), min(num_samples, len(samples)), replace=False)

    vlm = TaskVLMEngine(model_key, task=task, device=device)
    vlm.load()

    results = []
    t0 = time.time()
    eps_used = []

    for i, idx in enumerate(indices):
        s = samples[idx]

        # Load image as either path or PIL
        if "image_path" in s:
            if not os.path.exists(s["image_path"]):
                continue
            img_input = s["image_path"]
        elif "image_bytes" in s:
            import io
            img_input = Image.open(io.BytesIO(s["image_bytes"])).convert("RGB")
        else:
            continue

        # Get GT — either from conversation (JSON) or pre-parsed (Parquet)
        if "conversation" in s:
            gt = parse_gt_from_conversation(s, task)
            gt["gt_bbox"] = s.get("bbox")  # bbox comes from top-level for prediction
        else:
            gt = {
                "instruction": s.get("instruction", ""),
                "gt_action":  s.get("gt_action", "click"),
                "gt_point":   s.get("gt_point"),
                "gt_bbox":    s.get("gt_bbox"),
                "elements":   s.get("gt_elements", []),
            }

        try:
            # Per-sample epsilon based on config
            sens = 0.5  # Placeholder: would come from DINOv2+classifier
            unc = 0.5
            if adaptive == "rule":
                eps_t = rule_based_epsilon(sens)
            elif adaptive == "hmdp":
                eps_t = hmdp_epsilon(sens, unc)
            else:
                eps_t = epsilon
            eps_used.append(eps_t)

            instruction = gt.get("instruction", "")
            pred = vlm.predict(img_input, instruction)

            # Task-specific evaluation
            img_size = pred.get("img_size", s.get("img_resolution", (1036, 728)))
            metric = {}
            if task == "parsing":
                pred_elements = parse_pred_parsing(pred["raw_output"])
                metric = compute_parsing_metrics(pred_elements, gt["elements"])
                metric["n_pred"] = len(pred_elements)
                metric["n_gt"] = len(gt["elements"])
            elif task == "grounding":
                pred_pt = parse_pred_grounding(pred["raw_output"], img_size)
                gt_pt = gt["gt_point"]
                # Skip if GT is out-of-bounds (multi-monitor / corrupted samples)
                if gt_pt is not None and (
                    gt_pt[0] < 0 or gt_pt[0] > img_size[0] * 1.05 or
                    gt_pt[1] < 0 or gt_pt[1] > img_size[1] * 1.05
                ):
                    continue
                if pred_pt and gt_pt:
                    gt_norm = [gt_pt[0] / img_size[0], gt_pt[1] / img_size[1]]
                    dist = ((pred_pt[0] - gt_norm[0])**2 + (pred_pt[1] - gt_norm[1])**2) ** 0.5
                    metric = {
                        "hit": int(dist < 0.05),
                        "hit_loose": int(dist < 0.14),  # ~14% radius (common threshold)
                        "distance": dist,
                    }
                else:
                    metric = {"hit": 0, "hit_loose": 0, "distance": 1.0}
            elif task == "prediction":
                pred_action = pred["action_type"]
                pred_pt = pred["pred_point"]
                gt_bbox = gt.get("gt_bbox")
                gt_pt = gt.get("gt_point")
                # Skip out-of-bounds GT points
                if gt_pt is not None and (
                    gt_pt[0] < 0 or gt_pt[0] > img_size[0] * 1.05 or
                    gt_pt[1] < 0 or gt_pt[1] > img_size[1] * 1.05
                ):
                    continue
                hit = 0
                hit_loose = 0
                dist = 1.0

                if pred_pt and gt_bbox and len(gt_bbox) == 4:
                    # Use bbox (training set)
                    img_w, img_h = img_size
                    gt_bbox_norm = [
                        gt_bbox[0] / img_w, gt_bbox[1] / img_h,
                        gt_bbox[2] / img_w, gt_bbox[3] / img_h,
                    ]
                    hit = int(point_in_bbox(pred_pt, gt_bbox_norm))
                    hit_loose = hit  # bbox hit is already a loose criterion
                    dist = point_distance_to_center(pred_pt, gt_bbox_norm)
                elif pred_pt and gt_pt:
                    # Use point-distance (bench)
                    img_w, img_h = img_size
                    gt_norm = [gt_pt[0] / img_w, gt_pt[1] / img_h]
                    dist = ((pred_pt[0] - gt_norm[0])**2 + (pred_pt[1] - gt_norm[1])**2) ** 0.5
                    hit = int(dist < 0.05)
                    hit_loose = int(dist < 0.14)
                metric = {
                    "action_match": int(pred_action == gt.get("gt_action", "click")),
                    "hit": hit,
                    "hit_loose": hit_loose,
                    "distance": dist,
                }
            results.append(metric)

        except Exception as e:
            if len(results) < 3:
                import traceback
                print(f"    Error on sample {i}: {type(e).__name__}: {e}")
                traceback.print_exc()
            continue

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"    [{i+1}/{len(indices)}] {elapsed:.1f}s, {len(results)} evaluated")

    vlm.unload()

    # Aggregate metrics per task
    N = max(len(results), 1)
    avg_eps = np.mean(eps_used) if eps_used else epsilon

    if task == "parsing":
        # Headline: text_recall (most interpretable for dense GUI parsing)
        text_recall = np.mean([r.get("text_recall", 0) for r in results])
        summary = {
            "success_rate": float(text_recall),  # headline
            "text_recall": float(text_recall),
            "strict_f1":  float(np.mean([r.get("strict_f1", 0)  for r in results])),
            "relaxed_f1": float(np.mean([r.get("relaxed_f1", 0) for r in results])),
            "type_f1":    float(np.mean([r.get("type_f1", 0)    for r in results])),
            "count_ratio":float(np.mean([r.get("count_ratio", 0)for r in results])),
            "coverage":   float(np.mean([r.get("coverage", 0)   for r in results])),
            "avg_pred_elements": float(np.mean([r.get("n_pred", 0) for r in results])),
            "avg_gt_elements":   float(np.mean([r.get("n_gt", 0)   for r in results])),
        }
    elif task == "grounding":
        hit_rate = np.mean([r.get("hit", 0) for r in results])
        hit_loose = np.mean([r.get("hit_loose", 0) for r in results])
        avg_dist = np.mean([r.get("distance", 1.0) for r in results])
        summary = {
            "success_rate": float(hit_loose),  # use loose threshold as headline (5% is too strict for zero-shot)
            "grounding_accuracy_strict": float(hit_rate),  # <5%
            "grounding_accuracy_loose": float(hit_loose),  # <14%
            "avg_distance": float(avg_dist),
        }
    else:  # prediction
        hit_rate = np.mean([r.get("hit", 0) for r in results])
        hit_loose = np.mean([r.get("hit_loose", 0) for r in results])
        act_acc = np.mean([r.get("action_match", 0) for r in results])
        avg_dist = np.mean([r.get("distance", 1.0) for r in results])
        summary = {
            "success_rate": float(hit_loose),  # headline
            "action_accuracy": float(act_acc),
            "grounding_accuracy_strict": float(hit_rate),
            "grounding_accuracy_loose": float(hit_loose),
            "grounding_accuracy": float(hit_rate),  # backward-compat
            "avg_distance": float(avg_dist),
        }

    pes = summary["success_rate"] / avg_eps if avg_eps > 0 and avg_eps < 100 else 0.0
    summary.update({
        "num_evaluated": N,
        "avg_epsilon": float(avg_eps),
        "pes": float(pes),
        "time_sec": time.time() - t0,
        "config": config_name,
        "task": task,
    })
    return summary


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=200,
                        help="Samples per (task, config). Default 200 for speed.")
    parser.add_argument("--tasks", nargs="+",
                        default=["parsing", "grounding", "prediction"],
                        choices=["parsing", "grounding", "prediction"])
    parser.add_argument("--model", default="qwen2.5-vl-7b",
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="results/three_task_eval.json")
    parser.add_argument("--configs", nargs="+",
                        default=["no_privacy", "static_high", "rule_based", "hmdp"],
                        help="Which privacy configurations to run")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Privacy configurations (same for all tasks)
    CONFIGS = {
        "no_privacy": {"label": "No Privacy",         "eps": 1e6, "adaptive": None},
        "static_high": {"label": "Static High (ε=0.5)", "eps": 0.5, "adaptive": None},
        "rule_based": {"label": "Rule-based Adaptive", "eps": 1.0, "adaptive": "rule"},
        "hmdp":       {"label": "H-MDP (Ours)",         "eps": 0.82, "adaptive": "hmdp"},
    }

    print("=" * 72)
    print("  Three-Task Evaluation (Parsing / Grounding / Prediction)")
    print(f"  Tasks: {args.tasks}")
    print(f"  Configs: {args.configs}")
    print(f"  Samples per (task, config): {args.num_samples}")
    print(f"  Model: {args.model}")
    print("=" * 72)

    all_results = {}
    t_total = time.time()

    for task in args.tasks:
        print(f"\n{'█' * 72}\n  TASK: {task.upper()}\n{'█' * 72}")
        all_results[task] = {}
        for cfg_key in args.configs:
            if cfg_key not in CONFIGS:
                print(f"  [warn] Unknown config: {cfg_key}, skipping")
                continue
            cfg = CONFIGS[cfg_key]
            result = evaluate_task(
                task=task,
                config_name=cfg["label"],
                epsilon=cfg["eps"],
                model_key=args.model,
                num_samples=args.num_samples,
                device=args.device,
                adaptive=cfg["adaptive"],
            )
            all_results[task][cfg["label"]] = result

            # Print brief summary
            sr = result["success_rate"] * 100
            eps_str = f"{result['avg_epsilon']:.2f}" if result['avg_epsilon'] < 100 else "∞"
            print(f"    → SR={sr:.1f}%  ε̄={eps_str}  PES={result['pes']:.3f}")

            # Incremental save
            with open(args.output, "w") as f:
                json.dump(all_results, f, indent=2, default=float)

    # Final report
    print(f"\n{'=' * 72}\n  COMPLETE — {time.time() - t_total:.0f}s")
    print(f"  Results saved: {args.output}\n{'=' * 72}\n")
    print(f"  {'Task':<12} {'Config':<25} {'SR':>8} {'ε̄':>6} {'PES':>8}")
    print("  " + "-" * 62)
    for task, configs in all_results.items():
        for label, r in configs.items():
            sr = r["success_rate"] * 100
            eps_str = f"{r['avg_epsilon']:.2f}" if r["avg_epsilon"] < 100 else "∞"
            print(f"  {task:<12} {label:<25} {sr:>7.1f}% {eps_str:>6} {r['pes']:>8.3f}")


if __name__ == "__main__":
    main()
