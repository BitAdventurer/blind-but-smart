"""
Lower-level Functional Execution Engine (M_low).

Simulation ExecutionEngine for the full Parse → Project → (LTM) → Ground →
Predict pipeline.

Orchestrates:
  - ProxyEncoder E (frozen DINOv2-ViT) for per-region φ extraction     (Eq. 3)
  - LocalDifferentialPrivacy for per-region ε-LDP Laplace noise         (Eq. 4)
  - ProjectionLayer W_proj: R^d → R^{d_llm} for VLM token injection
  - GoTEngine for k-path semantic reasoning with θ̂ scoring              (Eq. 5, 7)
  - LongTermMemory conditionally activated when U_t > τ                 (Eq. 10)
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from .ldp import ProxyEncoder, LocalDifferentialPrivacy
from .got_engine import GoTEngine
from .ltm import LongTermMemory


class ProjectionLayer(nn.Module):
    """Learned projection W_proj: R^{d_latent} → R^{d_llm}."""

    def __init__(self, latent_dim: int, llm_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, llm_dim), nn.GELU(),
            nn.Linear(llm_dim, llm_dim))

    def forward(self, phi_private: torch.Tensor) -> torch.Tensor:
        return self.proj(phi_private)


class ExecutionEngine(nn.Module):
    """Lower-level MDP (M_low): proxy-based latent execution loop."""

    def __init__(self, proxy_encoder_dim: int = 1024, feature_dim: int = 256,
                 llm_dim: int = 4096, max_k: int = 20, num_actions: int = 4,
                 num_regions: int = 25, ltm_capacity: int = 1000, ltm_top_k: int = 5,
                 ltm_uncertainty_threshold: float = 0.5):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_regions = num_regions
        self.ltm_uncertainty_threshold = ltm_uncertainty_threshold
        self.proxy_encoder = ProxyEncoder(proxy_encoder_dim, feature_dim)
        self.ldp = LocalDifferentialPrivacy(feature_dim)
        self.proj_layer = ProjectionLayer(feature_dim, llm_dim)
        self.got_engine = GoTEngine(feature_dim, max_k, num_actions)
        self.ltm = LongTermMemory(feature_dim, ltm_capacity, ltm_top_k)
        self.global_weight = nn.Parameter(torch.zeros(feature_dim), requires_grad=False)

    @staticmethod
    def compute_latent_entropy(phi_regions: torch.Tensor) -> torch.Tensor:
        """
        Per-region latent (Shannon) entropy λ_t^(i) (Eq. 1):
            λ_t^(i) = -Σ_j p_j log p_j,  p_j = |φ_j| / Σ_k |φ_k|
        """
        abs_phi = phi_regions.abs()
        denom = abs_phi.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        p = abs_phi / denom
        entropy = -(p * (p + 1e-12).log()).sum(dim=-1)
        return entropy

    def update_global_weight(self, delta_lambda: torch.Tensor, delta_u: torch.Tensor,
                             ridge_lambda: float = 1.0):
        """
        Update server-side global value weight θ̂ via regularized least-squares:
            θ̂ = (ΔΛ + λI)^{-1} Δu
        """
        d = delta_u.shape[-1]
        avg_lambda = delta_lambda.mean(dim=0)
        avg_u = delta_u.mean(dim=0)
        reg = avg_lambda + ridge_lambda * torch.eye(d, device=avg_lambda.device)
        self.global_weight.data = torch.linalg.solve(reg, avg_u)

    def extract_region_features(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """Extract per-region latent features from a flat encoder output."""
        if encoder_output.dim() == 2:
            encoder_output = encoder_output.unsqueeze(1).expand(-1, self.num_regions, -1)
        batch, M, hidden = encoder_output.shape
        flat = encoder_output.reshape(batch * M, hidden)
        phi_flat = self.proxy_encoder(flat)
        phi_regions = phi_flat.reshape(batch, M, -1)
        return phi_regions

    def forward(self, encoder_output: torch.Tensor, epsilons: List[float], k: int,
                current_uncertainty: float = 0, value_estimate: Optional[torch.Tensor] = None,
                update_server: bool = True) -> Dict:
        """Full execution pipeline: Parse → Project → (LTM) → Ground → Predict."""
        phi_regions = self.extract_region_features(encoder_output)
        phi_private_regions = self.ldp.privatize_regions(phi_regions, epsilons)
        vlm_tokens = self.proj_layer(phi_private_regions)
        lambda_t = self.compute_latent_entropy(phi_regions)
        phi_private_agg = phi_private_regions.mean(dim=1)
        phi_agg = phi_regions.mean(dim=1)
        if value_estimate is not None and update_server:
            delta_lambda, delta_u = self.ldp.generate_privatized_statistics(
                phi_agg, value_estimate, float(sum(epsilons) / len(epsilons)))
            self.update_global_weight(delta_lambda, delta_u)
        ltm_prior = None
        if current_uncertainty > self.ltm_uncertainty_threshold:
            ltm_prior = self.ltm.generate_prior(phi_private_agg)
        got_output = self.got_engine(phi_private_agg, k, ltm_prior,
                                     global_weight=self.global_weight.data)
        got_output['phi_regions'] = phi_regions
        got_output['phi_private_regions'] = phi_private_regions
        got_output['phi_private'] = phi_private_agg
        got_output['phi'] = phi_agg
        got_output['vlm_tokens'] = vlm_tokens
        got_output['lambda_t'] = lambda_t
        return got_output

    def build_governance_state(self, uncertainty: float, lambda_t: torch.Tensor) -> torch.Tensor:
        """Build the 26-dim SAC state s_t = [U_t, λ_t^(1..M)]."""
        ut = torch.tensor([uncertainty], dtype=torch.float32)
        lam = lambda_t.detach().cpu().float()
        if lam.dim() == 2:
            lam = lam.mean(dim=0)
        return torch.cat([ut, lam], dim=0)
