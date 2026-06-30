"""
Projection layer for the Blind-but-Smart latent-injection boundary.

Defines ProjectionLayer W_proj: R^{d_latent} -> R^{d_llm}, which bridges the
frozen DINOv2 latent space (d=256) and the VLM's internal representation space.
Privatized latent features are projected and injected at the visual-token
positions, replacing standard image-patch embeddings.

This is the only execution-engine component used by the real-VLM pipeline
(hmdp.blind_vlm). The former simulation-only ExecutionEngine (SAC governance
state, GoT/LTM orchestration, server-side theta-hat updates) has been removed
along with the rest of the simulation path.
"""

import torch
import torch.nn as nn


class ProjectionLayer(nn.Module):
    """
    Learned projection W_proj: R^{d_latent} → R^{d_llm}.

    Bridges the DINOv2 latent space (d=256) and the VLM's internal
    representation space.  Privatized latent features are projected and
    injected at the visual token positions, replacing standard image patch
    embeddings.  This layer is trained offline in two stages (Section 4.1.4):
      (1) Feature alignment on screenshot–latent pairs (ε ∈ [3.0, 5.0])
      (2) Task-supervised fine-tuning with LoRA adapter (ε ∈ [0.1, 5.0])
    Both stages freeze the DINOv2 encoder and VLM backbone weights.
    """

    def __init__(self, latent_dim: int = 256, llm_dim: int = 4096):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, phi_private: torch.Tensor) -> torch.Tensor:
        """
        Args:
            phi_private: privatized latent features (batch, M, d) or (batch, d)
        Returns:
            VLM-space tokens (batch, M, d_llm) or (batch, d_llm)
        """
        return self.proj(phi_private)
