#!/usr/bin/env python3
"""
Qwen2-VL / Qwen2.5-VL Family Adapter.

Implements loading and inference for Qwen-based VLMs:
  - Qwen2.5-VL-7B-Instruct
  - UGround-V1-7B
  - Aguvis-7B-720P
  - GUI-Actor-7B-Qwen2.5-VL
  - UI-TARS-1.5-7B
"""

import time
from typing import Dict, List, Optional

import torch
from PIL import Image


def load_qwen_vlm(hf_id: str, device: str = "cuda", dtype=torch.bfloat16):
    """
    Load a Qwen2-VL family model and processor.

    Args:
        hf_id: HuggingFace model ID
        device: Target device
        dtype: Model dtype

    Returns:
        (model, processor) tuple
    """
    from transformers import AutoProcessor, AutoModelForImageTextToText

    processor = AutoProcessor.from_pretrained(
        hf_id, trust_remote_code=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        hf_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    return model, processor


def _predict_qwen(model, processor, image_path: str, instruction: str, build_prompt_fn) -> Dict:
    """
    Run inference with a Qwen2-VL family model.

    Args:
        model: Loaded VLM model
        processor: Corresponding processor
        image_path: Path to input image
        instruction: Text instruction
        build_prompt_fn: Function to format the instruction

    Returns:
        Dict with action_type, pred_point, raw_output
    """
    from qwen_vl_utils import process_vision_info
    from hmdp.vlm_inference import _parse_action_type, _parse_point

    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": build_prompt_fn(instruction)},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
        )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    raw_output = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    action_type = _parse_action_type(raw_output)
    pred_point = _parse_point(raw_output, img_w, img_h)

    return {
        "action_type": action_type,
        "pred_point": pred_point,
        "raw_output": raw_output,
    }


def _build_qwen_prompt(instruction: str) -> str:
    """
    Build a standardized prompt for GUI tasks.

    Format follows the Aguvis/UGround convention with explicit coordinate
    output format.
    """
    return (
        "You are a GUI automation assistant. Given a screenshot and an instruction, "
        "predict the action to take.\n\n"
        f"Instruction: {instruction}\n\n"
        "Respond ONLY with a JSON object in this exact format:\n"
        '{"action": "<click|type|drag|scroll>", "coordinate": [<x>, <y>]}\n'
        "If typing is needed, also include \"text\": \"<text>\".\n"
        "Use normalized coordinates in [0.000, 1.000] range."
    )
