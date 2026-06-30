#!/usr/bin/env python3
"""
VLM Adapters Module for H-MDP.

Architecture-specific adapters for different VLM families:
  - Qwen2-VL / Qwen2.5-VL family
  - InternVL2.5 family

Usage:
    from hmdp.vlm_adapters import load_qwen_vlm, load_internvl_vlm
    from hmdp.vlm_adapters.base import VLMModelInfo
"""

from .qwen_adapter import load_qwen_vlm
from .internvl_adapter import load_internvl_vlm, _load_image_internvl

__all__ = [
    "load_qwen_vlm",
    "load_internvl_vlm",
    "_load_image_internvl",
]
