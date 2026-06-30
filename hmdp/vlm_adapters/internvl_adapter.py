#!/usr/bin/env python3
"""
InternVL2.5 Family Adapter.

Implements loading and inference for InternVL2.5-based VLMs:
  - OpenGVLab/InternVL2_5-8B

Uses custom image preprocessing compatible with the InternVL architecture.
"""

import torch
from PIL import Image


# ═══════════════════════════════════════════════════════════════════════
#  InternVL2.5 Image Preprocessing Helpers
# ═══════════════════════════════════════════════════════════════════════

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def _build_internvl_transform(input_size: int):
    """Build torchvision transform for InternVL image preprocessing."""
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find the closest aspect ratio from target_ratios."""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_ar)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """
    Dynamic preprocessing for InternVL: split image into patches based on
    aspect ratio, similar to the original implementation.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    n = int(max_num ** 0.5)
    target_ratios = set(
        (i, j) for n in range(min_num, n + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_ar = _find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_ar[0]
    target_height = image_size * target_ar[1]
    blocks = target_ar[0] * target_ar[1]
    resized_img = image.resize((target_width, target_height))
    processed = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed.append(resized_img.crop(box))
    if use_thumbnail and len(processed) != 1:
        processed.append(image.resize((image_size, image_size)))
    return processed


def _load_image_internvl(image_path: str, input_size: int = 448, max_num: int = 12):
    """
    Load and preprocess an image for InternVL models.

    Args:
        image_path: Path to input image
        input_size: Target image size (default: 448)
        max_num: Maximum number of patches (default: 12)

    Returns:
        Tensor of shape (num_patches, 3, input_size, input_size)
    """
    image = Image.open(image_path).convert('RGB')
    transform = _build_internvl_transform(input_size)
    images = _dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(img) for img in images])
    return pixel_values


def load_internvl_vlm(hf_id: str, device: str = "cuda", dtype=torch.bfloat16):
    """
    Load an InternVL2.5 family model and tokenizer.

    Args:
        hf_id: HuggingFace model ID
        device: Target device
        dtype: Model dtype

    Returns:
        (model, tokenizer) tuple
    """
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        hf_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto",
    ).eval()

    return model, tokenizer


def _predict_internvl(model, tokenizer, image_path: str, instruction: str) -> "Dict":
    """
    Run inference with an InternVL2.5 model.

    Args:
        model: Loaded InternVL model
        tokenizer: Corresponding tokenizer
        image_path: Path to input image
        instruction: Text instruction

    Returns:
        Dict with action_type, pred_point, raw_output
    """
    import importlib
    from hmdp.vlm_inference import _parse_action_type, _parse_point

    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size

    pixel_values = _load_image_internvl(image_path, max_num=12).to(
        dtype=model.dtype, device=model.device
    )

    # Build prompt using InternVL conversation template
    conv_mod = importlib.import_module(
        type(model).__module__.replace('modeling_internvl_chat', 'conversation'))
    get_conv_template = conv_mod.get_conv_template

    question = f'<image>\n{instruction}'
    template = get_conv_template(model.template)
    template.system_message = model.system_message
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()

    # Generate
    with torch.no_grad():
        response = model.chat(tokenizer, pixel_values, query, {
            'max_new_tokens': 128,
            'temperature': 0.2,
        })

    raw_output = response
    action_type = _parse_action_type(raw_output)
    pred_point = _parse_point(raw_output, img_w, img_h)

    return {
        "action_type": action_type,
        "pred_point": pred_point,
        "raw_output": raw_output,
    }
