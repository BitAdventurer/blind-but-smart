#!/usr/bin/env python3
"""
Task Direction Head: theta_hat = Embed(task instruction).

Implements the server-side grounding head producing the task direction vector
in R^256 (Sec. 3.4 / Alg. 1).

It mean-pools the VLM token embeddings of the instruction and projects
them to the d=256 latent space so that <phi~(c_i), theta_hat> is defined.
Trained jointly with W_proj in the offline protocol.
"""

import torch.nn as nn
import torch.nn.functional as F

from hmdp.constants import ModelDims


class TaskDirectionHead(nn.Module):
    """
    Server-side grounding head for task-conditional reasoning.

    Produces theta_hat = Embed(instruction) for the GoT semantic term
    <psi_j, theta_hat> in Eq. got_score. This enables task-aware coordinate
    aggregation even when only privatized latents are available.
    """

    def __init__(
        self,
        llm_dim: int,
        latent_dim: int = ModelDims.LATENT_DEFAULT,  # 256
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(llm_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, pooled_instruction_emb) -> "torch.Tensor":
        """
        Project instruction embedding to latent space.

        Args:
            pooled_instruction_emb: (..., llm_dim) mean-pooled VLM embeddings

        Returns:
            (..., latent_dim) L2-normalized task direction vector
        """
        v = self.proj(pooled_instruction_emb)
        return F.normalize(v, dim=-1)
