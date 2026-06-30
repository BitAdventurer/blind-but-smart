#!/usr/bin/env python3
"""
Projection Module for H-MDP Blind Pipeline.

Implements the learned projection layers (Sec. 3.4-3.5):
  - BlindProjector: W_proj : R^256 -> R^{d_llm}
  - TaskDirectionHead: theta_hat = Embed(instruction)
  - Offline training protocol (Sec. 4.1.4)

Usage:
    from hmdp.projection import BlindProjector, TaskDirectionHead
    from hmdp.projection.training import train_projection_offline
"""

from .projector import BlindProjector
from .task_head import TaskDirectionHead
from .ltm_predictor import LTMPredictor
from .training import train_projection_offline, train_ltm_predictor_offline

__all__ = [
    "BlindProjector",
    "TaskDirectionHead",
    "LTMPredictor",
    "train_projection_offline",
    "train_ltm_predictor_offline",
]
