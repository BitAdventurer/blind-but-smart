#!/usr/bin/env python3
"""
Real VLM inference pipeline for GUI grounding evaluation.

Loads actual VLM backbones (Qwen2-VL / Qwen2.5-VL based) and runs
inference on GUI 360 screenshots to measure base action accuracy
and bbox grounding accuracy per backbone.

Usage:
    conda activate py358
    python -m hmdp.vlm_inference --model qwen2.5-vl-7b --num-samples 500
"""

import argparse
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ═══════════════════════════════════════════════════════════════════════
#  Model Registry
# ═══════════════════════════════════════════════════════════════════════

MODEL_REGISTRY = {
    "qwen2.5-vl-7b": {
        "hf_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "display_name": "Qwen-2.5-VL-7B",
        "arch": "qwen2_5_vl",
    },
    "uground-7b": {
        "hf_id": "osunlp/UGround-V1-7B",
        "display_name": "UGround-7B",
        "arch": "qwen2_vl",
    },
    "aguvis-7b": {
        "hf_id": "xlangai/Aguvis-7B-720P",
        "display_name": "Aguvis-7B",
        "arch": "qwen2_vl",
    },
    "gui-actor-7b": {
        "hf_id": "microsoft/GUI-Actor-7B-Qwen2.5-VL",
        "display_name": "GUI-Actor-7B",
        "arch": "qwen2_5_vl",
    },
    "ui-tars-1.5-7b": {
        "hf_id": "ByteDance-Seed/UI-TARS-1.5-7B",
        "display_name": "UI-TARS-1.5-7B",
        "arch": "qwen2_5_vl",
    },
    "internvl2.5-8b": {
        "hf_id": "OpenGVLab/InternVL2_5-8B",
        "display_name": "InternVL2.5-8B",
        "arch": "internvl2_5",
    },
    "qwen3-vl-8b": {
        "hf_id": "Qwen/Qwen3-VL-8B-Instruct",
        "display_name": "Qwen3-VL-8B",
        "arch": "qwen2_5_vl",
    },
    "internvl3.5-8b": {
        "hf_id": "OpenGVLab/InternVL3_5-8B",
        "display_name": "InternVL3.5-8B",
        "arch": "internvl2_5",
    },
}

ACTION_TYPES = ["click", "type", "drag", "scroll"]


# ═══════════════════════════════════════════════════════════════════════
#  VLM Inference Engine
# ═══════════════════════════════════════════════════════════════════════

# Note: InternVL2.5 preprocessing helpers have been moved to
# hmdp.vlm_adapters.internvl_adapter for better modularity.

class VLMInferenceEngine:
    """Unified inference engine for Qwen2-VL family models."""

    def __init__(self, model_key: str, device: str = "cuda", dtype=torch.bfloat16):
        assert model_key in MODEL_REGISTRY, f"Unknown model: {model_key}"
        self.model_key = model_key
        self.info = MODEL_REGISTRY[model_key]
        self.device = device
        self.dtype = dtype
        self.model = None
        self.processor = None

    def load(self):
        """Load model and processor from HuggingFace."""
        hf_id = self.info["hf_id"]
        arch = self.info["arch"]
        print(f"  Loading {hf_id} (arch={arch})...")
        t0 = time.time()

        if arch == "internvl2_5":
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True, use_fast=False)
            self.model = AutoModel.from_pretrained(
                hf_id,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map="auto",
            ).eval()
            self.processor = None  # not used for InternVL
        else:
            from transformers import AutoProcessor, AutoModelForImageTextToText
            self.processor = AutoProcessor.from_pretrained(
                hf_id, trust_remote_code=True,
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                hf_id,
                torch_dtype=self.dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            self.model.eval()
            self.tokenizer = None  # not used for Qwen family

        elapsed = time.time() - t0
        param_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        print(f"  Loaded in {elapsed:.1f}s  ({param_bytes/1e9:.1f} GB)")

    def unload(self):
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        torch.cuda.empty_cache()
        import gc; gc.collect()

    def predict(self, image_path: str, instruction: str) -> Dict:
        """
        Run VLM inference on a single screenshot.

        Returns:
            dict with keys: action_type (str), pred_point ([x_norm, y_norm]),
                            raw_output (str)
        """
        arch = self.info["arch"]
        if arch == "internvl2_5":
            return self._predict_internvl(image_path, instruction)
        else:
            return self._predict_qwen(image_path, instruction)

    def _predict_qwen(self, image_path: str, instruction: str) -> Dict:
        """Qwen2-VL / Qwen2.5-VL family inference."""
        from qwen_vl_utils import process_vision_info

        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self._build_prompt(instruction)},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        pad_id = getattr(self.processor.tokenizer, 'pad_token_id', None)
        if pad_id is None:
            pad_id = getattr(self.model.config, 'eos_token_id', 151658)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=1.0,
                pad_token_id=pad_id,
            )

        gen_ids = output_ids[:, inputs.input_ids.shape[1]:]
        raw_output = self.processor.batch_decode(
            gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        action_type = self._parse_action_type(raw_output)
        pred_point = self._parse_point(raw_output, img_w, img_h)

        return {
            "action_type": action_type,
            "pred_point": pred_point,
            "raw_output": raw_output,
            "img_size": (img_w, img_h),
        }

    def _predict_internvl(self, image_path: str, instruction: str) -> Dict:
        """InternVL2.5 family inference via manual greedy decoding."""
        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size

        from hmdp.vlm_adapters import _load_image_internvl
        pixel_values = _load_image_internvl(image_path, max_num=12).to(
            dtype=self.dtype, device=self.device
        )
        num_patches = pixel_values.shape[0]

        # ── Build prompt (replicate InternVLChatModel.chat logic) ──
        import importlib
        conv_mod = importlib.import_module(
            type(self.model).__module__.replace('modeling_internvl_chat', 'conversation'))
        get_conv_template = conv_mod.get_conv_template

        question = '<image>\n' + self._build_prompt(instruction)
        IMG_START = '<img>'
        IMG_END = '</img>'
        IMG_CTX = '<IMG_CONTEXT>'

        template = get_conv_template(self.model.template)
        template.system_message = self.model.system_message
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        eos_token_id = self.tokenizer.convert_tokens_to_ids(template.sep.strip())

        image_tokens = IMG_START + IMG_CTX * self.model.num_image_token * num_patches + IMG_END
        query = query.replace('<image>', image_tokens, 1)

        model_inputs = self.tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)

        # ── Build input embeddings with vision features ──
        img_ctx_id = self.tokenizer.convert_tokens_to_ids(IMG_CTX)
        self.model.img_context_token_id = img_ctx_id

        with torch.no_grad():
            vit_embeds = self.model.extract_feature(pixel_values)
            input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            flat_embeds = input_embeds.reshape(B * N, C)
            flat_ids = input_ids.reshape(B * N)
            selected = (flat_ids == img_ctx_id)
            flat_embeds[selected] = vit_embeds.reshape(-1, C).to(flat_embeds.device)
            input_embeds = flat_embeds.reshape(B, N, C)

        # ── Greedy decoding loop ──
        max_new_tokens = 256
        generated_ids = []
        past_key_values = None
        cur_embeds = input_embeds
        cur_mask = attention_mask

        with torch.no_grad():
            for _ in range(max_new_tokens):
                if past_key_values is None:
                    out = self.model.language_model(
                        inputs_embeds=cur_embeds,
                        attention_mask=cur_mask,
                        use_cache=True,
                    )
                else:
                    next_emb = self.model.language_model.get_input_embeddings()(
                        next_id.unsqueeze(0)
                    )
                    out = self.model.language_model(
                        inputs_embeds=next_emb,
                        attention_mask=cur_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

                next_id = out.logits[:, -1, :].argmax(dim=-1)
                token_id = next_id.item()
                generated_ids.append(token_id)

                if token_id == eos_token_id or token_id == self.tokenizer.eos_token_id:
                    break

                past_key_values = out.past_key_values
                cur_mask = torch.cat([
                    cur_mask,
                    torch.ones((1, 1), device=self.device, dtype=cur_mask.dtype),
                ], dim=1)

        raw_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        raw_output = raw_output.split(template.sep.strip())[0].strip()

        action_type = self._parse_action_type(raw_output)
        pred_point = self._parse_point(raw_output, img_w, img_h)

        return {
            "action_type": action_type,
            "pred_point": pred_point,
            "raw_output": raw_output,
            "img_size": (img_w, img_h),
        }

    def _build_prompt(self, instruction: str) -> str:
        """Build the grounding prompt for the VLM."""
        return (
            f"You are a GUI agent. Given the screenshot, predict the next action.\n\n"
            f"Instruction: {instruction}\n\n"
            f"Output format: action_type(coordinate=[x, y])\n"
            f"Supported actions: click, type, drag, scroll\n"
            f"Coordinates should be absolute pixel positions.\n"
            f"Example: click(coordinate=[500, 300])"
        )

    @staticmethod
    def _parse_action_type(output: str) -> str:
        """Extract action type from VLM output."""
        output_lower = output.lower()
        for action in ACTION_TYPES:
            if action in output_lower:
                return action
        return "click"  # default

    @staticmethod
    def _parse_point(output: str, img_w: int, img_h: int) -> Optional[List[float]]:
        """
        Extract predicted point from VLM output, normalized to [0, 1].
        Returns [x_norm, y_norm] or None if parsing fails.
        """
        # Try to find coordinate patterns (ordered by specificity)
        patterns = [
            r'coordinate\s*=\s*\[(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\]',
            r'\"coordinate\"\s*:\s*\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]',
            r'pyautogui\.\w+\(x\s*=\s*(\d+\.?\d*)\s*,\s*y\s*=\s*(\d+\.?\d*)',
            r'click\(x\s*=\s*(\d+\.?\d*)\s*,\s*y\s*=\s*(\d+\.?\d*)',
            r'\[(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\]',
            r'\((\d+\.?\d*)\s*,\s*(\d+\.?\d*)\)',
        ]
        for pat in patterns:
            match = re.search(pat, output)
            if match:
                x = float(match.group(1))
                y = float(match.group(2))

                # Normalize to [0, 1]
                if x > 1.0 or y > 1.0:
                    x_norm = x / max(img_w, 1)
                    y_norm = y / max(img_h, 1)
                else:
                    x_norm = x
                    y_norm = y

                return [np.clip(x_norm, 0, 1), np.clip(y_norm, 0, 1)]

        return None  # parsing failed


# ═══════════════════════════════════════════════════════════════════════
#  Evaluation on GUI 360
# ═══════════════════════════════════════════════════════════════════════

def point_in_bbox(pred_point, gt_bbox_norm):
    """Check if predicted point [x, y] falls within GT bbox (all normalized [0,1])."""
    px, py = pred_point
    return (gt_bbox_norm[0] <= px <= gt_bbox_norm[2] and
            gt_bbox_norm[1] <= py <= gt_bbox_norm[3])


def point_distance_to_center(pred_point, gt_bbox_norm):
    """Euclidean distance from predicted point to GT bbox center (normalized)."""
    cx = (gt_bbox_norm[0] + gt_bbox_norm[2]) / 2
    cy = (gt_bbox_norm[1] + gt_bbox_norm[3]) / 2
    return ((pred_point[0] - cx)**2 + (pred_point[1] - cy)**2) ** 0.5


def evaluate_model_on_gui360(
    model_key: str,
    data_path: str,
    image_base: str,
    num_samples: int = 500,
    device: str = "cuda",
    save_dir: str = "results",
    test_split: str = None,
) -> Dict:
    """
    Evaluate a VLM model on GUI 360 dataset.

    Returns dict with:
        action_accuracy, bbox_iou_mean, bbox_iou_50, point_accuracy,
        base_action_acc, base_bbox_sigma (calibrated params for simulation)
    """
    # Load data
    with open(data_path) as f:
        data = json.load(f)

    # Sample subset
    if test_split is not None:
        with open(test_split) as f:
            indices = np.array(json.load(f))
        print(f"  Using fixed test split: {len(indices)} samples from {test_split}")
    else:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(data), min(num_samples, len(data)), replace=False)

    # Load VLM
    engine = VLMInferenceEngine(model_key, device=device)
    engine.load()

    results = []
    correct_actions = 0
    point_hits = 0
    distances = []
    skipped = 0

    print(f"\n  Evaluating {engine.info['display_name']} on {len(indices)} samples...")

    for i, idx in enumerate(indices):
        sample = data[idx]

        # Parse ground truth
        instruction = ""
        gt_action = "click"
        if "conversation" in sample:
            for msg in sample["conversation"]:
                if msg["from"] == "human":
                    if "\nThe instruction is:\n" in msg["value"]:
                        instruction = msg["value"].split("\nThe instruction is:\n")[1].split("\n\n")[0]
                    else:
                        instruction = msg["value"].replace("<image>\n", "").split("\n")[0]
                    break
            for msg in sample["conversation"]:
                if msg["from"] == "gpt":
                    val_lower = msg["value"].lower()
                    for act in ACTION_TYPES:
                        if act in val_lower:
                            gt_action = act
                            break
                    break

        gt_bbox_raw = sample.get("bbox", None)
        if gt_bbox_raw is None:
            skipped += 1
            continue

        # Get image path and actual dimensions
        images = sample.get("images", [])
        if not images:
            skipped += 1
            continue
        img_path = os.path.join(image_base, images[0])
        if not os.path.exists(img_path):
            skipped += 1
            continue

        # Get image dimensions for normalization
        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
        except Exception:
            skipped += 1
            continue

        # Normalize GT bbox using actual image dimensions
        gt_bbox_norm = [
            gt_bbox_raw[0] / max(img_w, 1),
            gt_bbox_raw[1] / max(img_h, 1),
            gt_bbox_raw[2] / max(img_w, 1),
            gt_bbox_raw[3] / max(img_h, 1),
        ]

        # Run inference
        try:
            pred = engine.predict(img_path, instruction)
        except Exception as e:
            print(f"    [ERROR] Sample {idx}: {e}")
            skipped += 1
            continue

        # Evaluate
        action_correct = (pred["action_type"] == gt_action)
        correct_actions += int(action_correct)

        pred_point = pred["pred_point"]
        if pred_point is not None:
            hit = point_in_bbox(pred_point, gt_bbox_norm)
            dist = point_distance_to_center(pred_point, gt_bbox_norm)
        else:
            hit = False
            dist = 1.0  # max penalty

        point_hits += int(hit)
        distances.append(dist)

        results.append({
            "idx": int(idx),
            "gt_action": gt_action,
            "pred_action": pred["action_type"],
            "gt_bbox_norm": gt_bbox_norm,
            "pred_point": pred_point,
            "action_correct": action_correct,
            "point_hit": hit,
            "distance": dist,
            "raw_output": pred["raw_output"][:200],
        })

        if (i + 1) % 50 == 0:
            n = len(results)
            print(f"    [{i+1}/{len(indices)}] act_acc={correct_actions/n:.3f}  "
                  f"point_acc={point_hits/n:.3f}  avg_dist={np.mean(distances):.4f}  "
                  f"skipped={skipped}")

    # Free GPU memory
    engine.unload()

    n = len(results)
    if n == 0:
        print("  WARNING: No valid samples evaluated!")
        return {}

    action_acc = correct_actions / n
    point_acc = point_hits / n
    avg_dist = np.mean(distances)

    # Calibrate simulation parameters:
    # base_bbox_sigma ≈ average distance (indicates prediction spread)
    base_bbox_sigma = float(avg_dist)

    metrics = {
        "model": model_key,
        "display_name": engine.info["display_name"],
        "num_evaluated": n,
        "num_skipped": skipped,
        "action_accuracy": float(action_acc),
        "point_accuracy": float(point_acc),
        "avg_distance": float(avg_dist),
        "base_bbox_sigma": float(base_bbox_sigma),
    }

    print(f"\n  === {engine.info['display_name']} Results ===")
    print(f"  Action Accuracy:  {action_acc*100:.1f}%")
    print(f"  Point Accuracy:   {point_acc*100:.1f}%")
    print(f"  Avg Distance:     {avg_dist:.4f}")
    print(f"  BBox σ (calib):   {base_bbox_sigma:.4f}")

    # Save detailed results (convert numpy types for JSON)
    def _jsonify(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, list):
            return [_jsonify(v) for v in obj]
        if isinstance(obj, dict):
            return {k: _jsonify(v) for k, v in obj.items()}
        return obj

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"vlm_eval_{model_key}.json")
    with open(save_path, "w") as f:
        json.dump(_jsonify({"metrics": metrics, "results": results[:100]}), f, indent=2)
    print(f"  Saved to {save_path}")

    return metrics


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="VLM Inference on GUI 360")
    parser.add_argument("--model", type=str, default="qwen2.5-vl-7b",
                        choices=list(MODEL_REGISTRY.keys()) + ["all"],
                        help="Model to evaluate")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--test-split", type=str, default=None,
                        help="Path to JSON file with fixed eval indices (e.g. results/json/eval_seed0.json)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-suffix", type=str, default="",
                        help="Suffix for output file (e.g. _seed0)")
    parser.add_argument("--data-path", type=str,
                        default="gui360_full/processed_data/action_prediction_train_resize/training_data.json")
    parser.add_argument("--image-base", type=str,
                        default="gui360_full/processed_data/action_prediction_train_resize/")
    args = parser.parse_args()

    models = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    all_metrics = {}

    for model_key in models:
        print(f"\n{'='*60}")
        print(f"  Evaluating: {MODEL_REGISTRY[model_key]['display_name']}")
        print(f"{'='*60}")

        metrics = evaluate_model_on_gui360(
            model_key=model_key,
            data_path=args.data_path,
            image_base=args.image_base,
            num_samples=args.num_samples,
            device=args.device,
            test_split=args.test_split,
        )
        if metrics:
            all_metrics[model_key] = metrics

    # Print summary
    if len(all_metrics) > 1:
        print(f"\n{'='*70}")
        print(f"  Summary: VLM Base Accuracy on GUI 360")
        print(f"{'='*70}")
        print(f"  {'Model':<24} {'Act.Acc':>8} {'Pt.Acc':>8} {'Avg.Dist':>9}")
        print(f"  {'-'*55}")
        for k, m in all_metrics.items():
            print(f"  {m['display_name']:<24} "
                  f"{m['action_accuracy']*100:>7.1f}% "
                  f"{m['point_accuracy']*100:>7.1f}% "
                  f"{m['avg_distance']:>8.4f}")
        print(f"{'='*70}")

    # Save combined results
    os.makedirs("results", exist_ok=True)
    suffix = getattr(args, 'save_suffix', '')
    save_name = f"results/vlm_base_accuracy{suffix}.json"
    with open(save_name, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nAll metrics saved to {save_name}")


if __name__ == "__main__":
    main()
