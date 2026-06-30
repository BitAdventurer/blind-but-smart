#!/usr/bin/env python3
"""Fast runtime sanity checks before launching GPU/VLM experiments.

This script intentionally avoids loading VLM or DINOv2 model weights. It checks
the parts that commonly fail after a long launch: dataset paths, split indices,
sample image paths, W_proj checkpoint structure, and optional SAC checkpoint
loadability.
"""

import argparse
import json
import os
from typing import Any, Dict, List

import torch


DEFAULT_DATA = "gui360_full/processed_data/action_prediction_train_resize/training_data.json"
DEFAULT_IMAGE_BASE = "gui360_full/processed_data/action_prediction_train_resize"
DEFAULT_SPLIT = "results/json/test_split_1000.json"
DEFAULT_WPROJ = "checkpoints/wproj_eq8.pt"


def _load_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def _check_file(path: str, label: str) -> bool:
    ok = os.path.isfile(path)
    print(f"[{'OK' if ok else 'FAIL'}] {label}: {path}")
    return ok


def _check_dataset(data_path: str, image_base: str, max_images: int) -> List[Dict]:
    """Validate the GUI-360 JSON and a small prefix of referenced images.

    The full image tree can be large, so this checks only the first
    ``max_images`` samples. That is enough to catch the common clone/setup
    mistakes: missing data, wrong image root, or stale relative paths.
    """
    if not _check_file(data_path, "dataset"):
        raise FileNotFoundError(data_path)
    data = _load_json(data_path)
    if not isinstance(data, list) or not data:
        raise ValueError(f"dataset must be a non-empty list: {data_path}")
    print(f"[OK] dataset samples: {len(data)}")

    missing = []
    for i, sample in enumerate(data[:max_images]):
        images = sample.get("images", [])
        if not images:
            missing.append((i, "<no images>"))
            continue
        img_path = os.path.join(image_base, images[0])
        if not os.path.isfile(img_path):
            missing.append((i, img_path))
    if missing:
        preview = ", ".join(f"{i}:{p}" for i, p in missing[:5])
        raise FileNotFoundError(f"missing image paths in first {max_images} samples: {preview}")
    print(f"[OK] first {min(max_images, len(data))} image paths exist")
    return data


def _check_split(split_path: str, data_len: int) -> None:
    """Validate a fixed evaluation split against the loaded dataset length."""
    if not split_path:
        print("[SKIP] split: not supplied")
        return
    if not _check_file(split_path, "split"):
        raise FileNotFoundError(split_path)
    split = _load_json(split_path)
    if isinstance(split, dict):
        indices = split.get("test") or split.get("indices")
    else:
        indices = split
    if not isinstance(indices, list) or not indices:
        raise ValueError(f"split must be a non-empty index list or dict: {split_path}")
    bad = [x for x in indices if not isinstance(x, int) or x < 0 or x >= data_len]
    if bad:
        raise ValueError(f"split has out-of-range/non-int indices, first bad={bad[:5]}")
    print(f"[OK] split indices: {len(indices)}")


def _check_wproj(path: str, expected_grid: int) -> None:
    """Validate the offline W_proj/Eq.8 checkpoint without loading a VLM."""
    if not _check_file(path, "W_proj checkpoint"):
        raise FileNotFoundError(path)
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise RuntimeError(f"W_proj checkpoint failed torch.load: {path}") from e
    if not isinstance(ckpt, dict):
        raise ValueError(f"W_proj checkpoint must be a dict: {path}")
    required = {"proj_layer", "proxy_encoder", "task_head"}
    missing = sorted(required - set(ckpt))
    if missing:
        raise KeyError(f"W_proj checkpoint missing keys: {missing}")
    grid = ckpt.get("grid")
    has_ltm = "ltm_predictor" in ckpt
    if grid is not None and int(grid) != int(expected_grid):
        raise ValueError(f"W_proj grid mismatch: ckpt grid={grid}, expected={expected_grid}")
    if grid is None and expected_grid != 5:
        print(f"[WARN] W_proj has no grid metadata; grid={expected_grid} cannot be verified")
    print(f"[OK] W_proj keys present; grid={grid if grid is not None else 'unknown'}; ltm_predictor={has_ltm}")


def _check_sac(path: str) -> None:
    """Validate the optional SAC governor checkpoint used by --adaptive."""
    if not path:
        print("[SKIP] SAC checkpoint: not supplied")
        return
    if not _check_file(path, "SAC checkpoint"):
        raise FileNotFoundError(path)
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise RuntimeError(f"SAC checkpoint failed torch.load: {path}") from e
    if not isinstance(ckpt, dict):
        raise ValueError(f"SAC checkpoint must be a dict: {path}")
    actor = ckpt.get("actor")
    if actor is None and isinstance(ckpt.get("meta_policy"), dict):
        actor = ckpt["meta_policy"].get("actor")
    if actor is None:
        raise KeyError("SAC checkpoint has no actor or meta_policy.actor state")
    print("[OK] SAC actor state present")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate local H-MDP runtime inputs")
    parser.add_argument("--data-path", default=DEFAULT_DATA,
                        help="Path to GUI-360 action prediction training_data.json")
    parser.add_argument("--image-base", default=DEFAULT_IMAGE_BASE,
                        help="Directory used to resolve sample image paths")
    parser.add_argument("--test-split", default=DEFAULT_SPLIT,
                        help="Optional JSON list/dict of evaluation indices")
    parser.add_argument("--wproj-ckpt", default=DEFAULT_WPROJ,
                        help="Offline W_proj/Eq.8 checkpoint to validate")
    parser.add_argument("--sac-ckpt", default=None,
                        help="Optional SAC governor checkpoint to validate")
    parser.add_argument("--grid", type=int, default=5,
                        help="Expected G for a GxG region grid")
    parser.add_argument("--max-images", type=int, default=20,
                        help="Number of leading samples whose image paths are checked")
    args = parser.parse_args()

    data = _check_dataset(args.data_path, args.image_base, args.max_images)
    _check_split(args.test_split, len(data))
    _check_wproj(args.wproj_ckpt, args.grid)
    _check_sac(args.sac_ckpt)
    print("\nRuntime inputs look usable for a VLM launch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
