"""
Local Differential Privacy (LDP) Module.

Simulation LDP implementation used by the H-MDP training pipeline. It supports
Laplace noise for pure ε-LDP, Gaussian noise for (ε, δ)-LDP, and privatized
statistics required by ExecutionEngine.

Implements:
  - Latent Feature Mapping φ via Universal Proxy Encoder E (Eq. 3)
  - Per-region ε-LDP Laplace privatization φ̃(R) = φ(R) + Lap(Δ₂/ε) (Eq. 4)
  - Analytic Gaussian mechanism for (ε, δ)-LDP
  - Privatized server statistics (ΔΛ_τ, Δu_τ) for global value-weight θ̂
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple


class ProxyEncoder(nn.Module):
    """
    Universal Proxy Encoder E with projection φ.

    Eq. 1/3: φ(s_τ) = Proj( E(I_τ) ) / max(1, ||Proj(E(I_τ))||_2) ∈ R^d

    The clamped unit-norm projection is applied to the *output* of Proj so that
    ||φ(s_τ)||_2 ≤ 1 holds for the quantity that LDP actually protects. Two such
    outputs differ by at most 2 in L2 distance, so the L2 sensitivity is
    Δ₂ ≤ 2 — exactly the value used to calibrate the Laplace/Gaussian noise in
    ``LocalDifferentialPrivacy`` (scale = Δ₂/ε). Normalising the *input* instead
    would bound ||E(I)||₂ but not ||φ||₂, since the unconstrained linear map Proj
    can amplify the norm by its spectral norm ||W||₂; the Δ₂ ≤ 2 claim would then
    not hold for the privatised quantity.
    """

    def __init__(self, input_dim: int = 1024, output_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='linear')
        # Larger init gain so the pre-normalisation projection output has
        # magnitude > 1 in general; otherwise the clamp(min=1.0) below would be
        # inactive and φ would never be normalised.
        self.proj.weight.data *= (input_dim ** 0.5)
        self.output_dim = output_dim

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        if encoder_output.dim() == 3:
            pooled = encoder_output.mean(dim=1)
        else:
            pooled = encoder_output
        # Project first, then clamp the OUTPUT to the unit ball so the quantity
        # that LDP protects satisfies ||φ||_2 ≤ 1  =>  Δ₂ ≤ 2.
        projected = self.proj(pooled)
        norm = torch.clamp(projected.norm(dim=-1, keepdim=True), min=1.0)
        return projected / norm


# Backward-compatible alias
LatentFeatureMapper = ProxyEncoder


class LocalDifferentialPrivacy(nn.Module):
    """
    ε-Local Differential Privacy module.

    Applies calibrated noise to latent features so raw data never leaves the
    local device. The post-processing property guarantees downstream reasoning
    cannot increase privacy loss beyond ε.
    """

    def __init__(self, feature_dim: int = 256, sensitivity: float = 2.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.sensitivity = sensitivity

    # ── Noise samplers ──────────────────────────────────────────────────

    def _laplace_noise(self, shape: Tuple, epsilon: float,
                       device: torch.device) -> torch.Tensor:
        """Sample Laplace noise calibrated to ε-LDP (scale = Δ₂/ε)."""
        scale = self.sensitivity / epsilon
        return torch.distributions.Laplace(
            torch.zeros(shape, device=device),
            torch.full(shape, scale, device=device),
        ).sample()

    def _gaussian_noise(self, shape: Tuple, epsilon: float, delta: float,
                        device: torch.device) -> torch.Tensor:
        """Sample Gaussian noise for (ε, δ)-LDP (Analytic Gaussian Mechanism)."""
        sigma = self.sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        return torch.randn(shape, device=device) * sigma

    # ── Core privatization ──────────────────────────────────────────────

    def privatize_features(self, phi: torch.Tensor, epsilon: float,
                           mechanism: str = 'laplace', delta: float = 1e-05) -> torch.Tensor:
        """
        Add ε-LDP noise to the latent feature φ(s_τ) (Eq. 4):
            φ̃(s_τ) = φ(s_τ) + Lap(Δ₂ / ε)   [or Gaussian for (ε,δ)-LDP]
        """
        if mechanism == 'laplace':
            noise = self._laplace_noise(phi.shape, epsilon, phi.device)
            return phi + noise
        if mechanism == 'gaussian':
            noise = self._gaussian_noise(phi.shape, epsilon, delta, phi.device)
            return phi + noise
        raise ValueError(f'Unknown mechanism: {mechanism}')

    def generate_privatized_statistics(self, phi: torch.Tensor, value_estimate: torch.Tensor,
                                       epsilon: float, mechanism: str = 'laplace',
                                       delta: float = 1e-05) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Privatized server-side statistics:
            ΔΛ_τ = φ(s_τ) φ(s_τ)^T + W        (W symmetric Laplace noise)
            Δu_τ = φ(s_τ) · V_τ      + ζ        (ζ Laplace noise)
        Returns (ΔΛ_τ, Δu_τ); consumed by ExecutionEngine.update_global_weight
        to solve θ̂ = (ΔΛ + λI)^{-1} Δu.
        """
        batch = phi.shape[0]
        d = phi.shape[1]
        phi_unsqueeze = phi.unsqueeze(2)
        outer = torch.bmm(phi_unsqueeze, phi_unsqueeze.transpose(1, 2))
        W = self._laplace_noise((batch, d, d), epsilon, phi.device)
        W = (W + W.transpose(1, 2)) / 2
        delta_lambda = outer + W
        if value_estimate.dim() == 1:
            value_estimate = value_estimate.unsqueeze(1)
        phi_v = phi * value_estimate
        zeta = self._laplace_noise((batch, d), epsilon, phi.device)
        delta_u = phi_v + zeta
        return delta_lambda, delta_u

    # ── Per-region privatization ────────────────────────────────────────

    def privatize_regions(self, phi_regions: torch.Tensor, epsilons: List[float],
                          mechanism: str = 'laplace', delta: float = 1e-05) -> torch.Tensor:
        """
        Apply per-region ε-LDP noise to M region latent features (Eq. 4):
            φ̃(R^(i)) = φ(R^(i)) + Lap(Δ₂ / ε_t^(i))
        """
        batch, M, d = phi_regions.shape
        assert len(epsilons) == M, f"Expected {M} epsilons, got {len(epsilons)}"
        out = torch.empty_like(phi_regions)
        for i, eps in enumerate(epsilons):
            out[:, i, :] = self.privatize_features(phi_regions[:, i, :], eps, mechanism, delta)
        return out

    # ── Utility helpers ─────────────────────────────────────────────────

    @staticmethod
    def estimate_noise_magnitude(epsilon: float, sensitivity: float = 2.0) -> float:
        """Expected L1 noise magnitude for Laplace mechanism (scale = Δ₂/ε)."""
        return sensitivity / epsilon

    def forward(self, phi: torch.Tensor, epsilon: float, mechanism: str = 'laplace') -> torch.Tensor:
        """Convenience forward: privatize features."""
        return self.privatize_features(phi, epsilon, mechanism)
