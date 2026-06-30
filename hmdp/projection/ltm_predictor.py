#!/usr/bin/env python3
"""
LTM Prediction Head θ_pred (Eq. 8).

Implements the learned strategic-prediction head of Sec. 3.5 / Eq. 8:

    u_τ = argmax_u P(u | φ̃(s_τ), C*, K_ret ; θ_pred)

where the optimal action u_τ is realised here as a refined grounding
coordinate. The head fuses three faithful inputs:

  - φ̃(s_τ)  : the privatized latent observation (region-mean summary, R^d),
  - C*       : the GoT-aggregated coordinate from Stage-3 (R^2),
  - K_ret    : the retrieved long-term knowledge (Eq. 7), summarised as a
               similarity-weighted target point (R^2), state embedding (R^d),
               and a scalar retrieval confidence (R^1).

The head predicts a *residual* on C* (bounded by ``max_shift``) so that an
uninformative / empty memory leaves the GoT coordinate essentially unchanged,
and only confident, relevant retrievals move the prediction.
"""

import torch
import torch.nn as nn

from hmdp.constants import ModelDims


class LTMPredictor(nn.Module):
    """
    θ_pred: P(u | φ̃, C*, K_ret) realised as a bounded residual coordinate head.

    forward(phi_summary, c_star, ret_coord, ret_emb, conf) -> refined_coord (2,)
    """

    def __init__(
        self,
        latent_dim: int = ModelDims.LATENT_DEFAULT,  # 256
        hidden: int = 256,
        max_shift: float = 0.25,
    ):
        super().__init__()
        # Input: φ̃ (d) + C* (2) + K_ret coord (2) + K_ret emb (d) + conf (1)
        in_dim = latent_dim + 2 + 2 + latent_dim + 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        self.max_shift = max_shift

    def forward(
        self,
        phi_summary: torch.Tensor,   # (..., d)
        c_star: torch.Tensor,        # (..., 2)
        ret_coord: torch.Tensor,     # (..., 2)
        ret_emb: torch.Tensor,       # (..., d)
        conf: torch.Tensor,          # (..., 1)
    ) -> torch.Tensor:
        """
        Returns the refined coordinate C_final = clamp(C* + Δ, 0, 1), where the
        residual Δ = max_shift · tanh(MLP(features)) is bounded so an
        uninformative memory yields Δ ≈ 0 (prediction stays at C*).
        """
        feats = torch.cat([phi_summary, c_star, ret_coord, ret_emb, conf], dim=-1)
        delta = torch.tanh(self.net(feats)) * self.max_shift
        return (c_star + delta).clamp(0.0, 1.0)
