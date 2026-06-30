"""
Local Differential Privacy (LDP) Module.

Implements:
  - Latent Feature Mapping φ via Universal Proxy Encoder E (Eq. 3)
  - Per-region (ε,δ)-LDP Gaussian privatization φ̃(R) = φ(R) + N(0, σ²) (Eq. 4)
    where σ is calibrated by the Analytic Gaussian Mechanism (Balle & Wang 2018),
    valid for all ε > 0 (the classical σ = Δ₂√(2ln(1.25/δ))/ε holds only for ε ≤ 1)
  - Analytic Gaussian mechanism for (ε, δ)-LDP (default)
  - Laplace mechanism for pure ε-LDP (optional)
  - Privatized server statistics (ΔΛ_τ, Δu_τ) for global value-weight θ̂
"""

import math

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple

from hmdp.constants import ModelDims, LDPConfig


def analytic_gaussian_sigma(epsilon: float, delta: float, sensitivity: float,
                            tol: float = 1e-12) -> float:
    """
    Noise std σ for the Analytic Gaussian Mechanism (Balle & Wang, 2018).

    Calibrates the tightest σ such that adding N(0, σ²) to a query with L2
    sensitivity ``sensitivity`` is (ε, δ)-DP. Unlike the classical closed form
    σ = Δ₂√(2ln(1.25/δ))/ε — which is only valid for ε ≤ 1 and is loose — this
    is valid for all ε > 0 and yields strictly less noise (better utility) at
    the same privacy guarantee. Uses only ``math`` (no SciPy dependency).
    """
    def Phi(t: float) -> float:
        return 0.5 * (1.0 + math.erf(t / math.sqrt(2.0)))

    def _log_Phi_neg(x: float) -> float:
        """log Phi(-x) = log(0.5 * erfc(x/√2)) for x >= 0, stable for large x.

        For large x, erfc(x/√2) underflows to 0.0, so the naive log(0) = -inf.
        We switch to the asymptotic log erfc(z) ≈ -z² - 0.5 ln(π) - ln(z) to keep
        the value finite, which is what makes the exp(ε)·Phi(-…) term computable
        at large ε (see ``_exp_term``) instead of overflowing to inf·0 = nan.
        """
        z = x / math.sqrt(2.0)
        if z < 25.0:
            return math.log(0.5 * math.erfc(z))
        return math.log(0.5) - z * z - 0.5 * math.log(math.pi) - math.log(z)

    def _exp_term(eps: float, s: float) -> float:
        """exp(ε)·Phi(-√(ε(s+2))) computed in log-space to avoid inf·0 = nan.

        log of the term is ε + log Phi(-√(ε(s+2))); since the second summand is
        ≈ -ε(s+2)/2, the total is ≈ -εs/2 ≤ 0 for s ≥ 0, so exp() never overflows
        (the naive math.exp(eps) factor would overflow for ε ≳ 709).
        """
        log_t = eps + _log_Phi_neg(math.sqrt(eps * (s + 2.0)))
        return math.exp(log_t) if log_t < 700.0 else math.inf

    def caseA(eps: float, s: float) -> float:
        return Phi(math.sqrt(eps * s)) - _exp_term(eps, s)

    def caseB(eps: float, s: float) -> float:
        return Phi(-math.sqrt(eps * s)) - _exp_term(eps, s)

    delta_thr = caseA(epsilon, 0.0)
    if delta == delta_thr:
        alpha = 1.0
    else:
        if delta > delta_thr:
            stop = lambda s: caseA(epsilon, s) >= delta
            f = lambda s: caseA(epsilon, s)
            left = lambda s: f(s) > delta
            to_alpha = lambda s: math.sqrt(1.0 + s / 2.0) - math.sqrt(s / 2.0)
        else:
            stop = lambda s: caseB(epsilon, s) <= delta
            f = lambda s: caseB(epsilon, s)
            left = lambda s: f(s) < delta
            to_alpha = lambda s: math.sqrt(1.0 + s / 2.0) + math.sqrt(s / 2.0)

        # Doubling trick to bracket the root, then binary search.
        s_inf, s_sup = 0.0, 1.0
        while not stop(s_sup):
            s_inf = s_sup
            s_sup = 2.0 * s_inf
        s_mid = s_inf + (s_sup - s_inf) / 2.0
        while abs(f(s_mid) - delta) > tol:
            if left(s_mid):
                s_sup = s_mid
            else:
                s_inf = s_mid
            s_mid = s_inf + (s_sup - s_inf) / 2.0
        alpha = to_alpha(s_mid)

    return alpha * sensitivity / math.sqrt(2.0 * epsilon)


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

    def __init__(
        self,
        input_dim: int = ModelDims.DINOV2_OUTPUT,  # 1024
        output_dim: int = ModelDims.LATENT_DEFAULT,  # 256
    ):
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

    def __init__(
        self,
        feature_dim: int = ModelDims.LATENT_DEFAULT,  # 256
        sensitivity: float = LDPConfig.SENSITIVITY,  # 2.0
    ):
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
        """Sample Gaussian noise for (ε, δ)-LDP (Analytic Gaussian Mechanism).

        σ is calibrated by the Analytic Gaussian Mechanism (Balle & Wang, 2018),
        which is valid for all ε > 0 and tighter than the classical
        σ = Δ₂√(2ln(1.25/δ))/ε closed form (the latter is only valid for ε ≤ 1).
        """
        sigma = analytic_gaussian_sigma(epsilon, delta, self.sensitivity)
        return torch.randn(shape, device=device) * sigma

    # ── Core privatization ──────────────────────────────────────────────

    def privatize_features(self, phi: torch.Tensor, epsilon: float,
                           mechanism: str = 'gaussian', delta: float = 1e-05) -> torch.Tensor:
        """
        Add (ε,δ)-LDP noise to the latent feature φ(s_τ) (Eq. 4):
            φ̃(s_τ) = φ(s_τ) + N(0, σ²)   where σ = Δ₂√(2ln(1.25/δ))/ε
        Uses Gaussian mechanism by default for (ε,δ)-differential privacy.
        """
        if mechanism == 'laplace':
            noise = self._laplace_noise(phi.shape, epsilon, phi.device)
            return phi + noise
        if mechanism == 'gaussian':
            noise = self._gaussian_noise(phi.shape, epsilon, delta, phi.device)
            return phi + noise
        raise ValueError(f'Unknown mechanism: {mechanism}')

    def generate_privatized_statistics(self, phi: torch.Tensor, value_estimate: torch.Tensor,
                                       epsilon: float, mechanism: str = 'gaussian',
                                       delta: float = 1e-05) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Privatized server-side statistics:
            ΔΛ_τ = φ(s_τ) φ(s_τ)^T + W        (W symmetric noise)
            Δu_τ = φ(s_τ) · V_τ      + ζ        (ζ noise)
        Noise is sampled with the selected ``mechanism`` (Gaussian (ε,δ)-LDP by
        default, matching the per-region privatization). Returns (ΔΛ_τ, Δu_τ);
        consumed by ExecutionEngine.update_global_weight to solve
        θ̂ = (ΔΛ + λI)^{-1} Δu.
        """
        def _noise(shape):
            if mechanism == 'laplace':
                return self._laplace_noise(shape, epsilon, phi.device)
            if mechanism == 'gaussian':
                return self._gaussian_noise(shape, epsilon, delta, phi.device)
            raise ValueError(f'Unknown mechanism: {mechanism}')

        batch = phi.shape[0]
        d = phi.shape[1]
        phi_unsqueeze = phi.unsqueeze(2)
        outer = torch.bmm(phi_unsqueeze, phi_unsqueeze.transpose(1, 2))
        W = _noise((batch, d, d))
        W = (W + W.transpose(1, 2)) / 2
        delta_lambda = outer + W
        if value_estimate.dim() == 1:
            value_estimate = value_estimate.unsqueeze(1)
        phi_v = phi * value_estimate
        zeta = _noise((batch, d))
        delta_u = phi_v + zeta
        return delta_lambda, delta_u

    # ── Per-region privatization ────────────────────────────────────────

    def privatize_regions(self, phi_regions: torch.Tensor, epsilons: List[float],
                          mechanism: str = 'gaussian', delta: float = 1e-05) -> torch.Tensor:
        """
        Apply per-region (ε,δ)-LDP noise to M region latent features (Eq. 4):
            φ̃(R^(i)) = φ(R^(i)) + N(0, σ²)   where σ = Δ₂√(2ln(1.25/δ))/ε_t^(i)
        Uses Gaussian mechanism by default for (ε,δ)-differential privacy.
        """
        batch, M, d = phi_regions.shape
        assert len(epsilons) == M, f"Expected {M} epsilons, got {len(epsilons)}"
        out = torch.empty_like(phi_regions)
        for i, eps in enumerate(epsilons):
            out[:, i, :] = self.privatize_features(phi_regions[:, i, :], eps, mechanism, delta)
        return out

    # ── Utility helpers ─────────────────────────────────────────────────

    @staticmethod
    def estimate_noise_magnitude(
        epsilon: float,
        sensitivity: float = LDPConfig.SENSITIVITY,  # 2.0
    ) -> float:
        """Expected L1 noise magnitude for Laplace mechanism (scale = Δ₂/ε)."""
        return sensitivity / epsilon

    def forward(self, phi: torch.Tensor, epsilon: float, mechanism: str = 'gaussian') -> torch.Tensor:
        """Convenience forward: privatize features (Gaussian (ε,δ)-LDP by default)."""
        return self.privatize_features(phi, epsilon, mechanism)
