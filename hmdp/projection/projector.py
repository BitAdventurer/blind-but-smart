#!/usr/bin/env python3
"""
Blind Projector: W_proj layer for latent-to-VLM projection.

Implements the projection from privatized latent space to VLM embedding space:
    W_proj : R^{d_latent} -> R^{d_llm}

A deeper MLP with LayerNorm bridges the privatized DINOv2 latent space to
the VLM token space, giving enough capacity to preserve the spatial cues
needed for coordinate grounding.
"""

import torch.nn as nn

from hmdp.constants import ModelDims


class BlindProjector(nn.Module):
    """
    Higher-capacity W_proj for the blind boundary.

    Projects privatized latent features phi~(R_t) into the VLM's token embedding
    space, enabling the VLM to reason about screen content without seeing
    raw pixels (Sec. 3.5, Stage-1).
    """

    def __init__(
        self,
        latent_dim: int = ModelDims.LATENT_DEFAULT,  # 256
        llm_dim: int = ModelDims.QWEN2_5_VL_7B,  # 3584
        hidden: int = ModelDims.PROJ_HIDDEN,  # 2048
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, llm_dim),
        )

    def forward(self, phi_private) -> "torch.Tensor":
        """
        Project privatized latents to VLM embedding space.

        Args:
            phi_private: (..., latent_dim) privatized latent features

        Returns:
            (..., llm_dim) projected embeddings for VLM injection
        """
        return self.net(phi_private)
