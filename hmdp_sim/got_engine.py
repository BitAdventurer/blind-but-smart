"""
Graph of Thoughts (GoT) Reasoning Engine.

Implements k parallel reasoning paths for functional grounding.
  - k parallel thought paths (controlled by SAC meta-policy)
  - Eq.5: s_i = α·Logit(ĉ_i) + (1-α)·⟨φ̃(ĉ_i), θ̂⟩
  - Eq.7: w_j = softmax(s_j/Tagg),  C* = Σ w_j · ĉ_j
  - Eq.8: U_t = (1/k) Σ ||ĉ_i - C*||²
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class ThoughtNode:
    """A single node in the Graph of Thoughts.

    NOTE: defined for completeness but NOT used in the active reasoning path
    (GoTEngine.forward operates on stacked tensors / returns a dict).
    """
    node_id: str
    content: torch.Tensor
    action_logits: Optional[torch.Tensor] = None
    bbox_pred: Optional[torch.Tensor] = None
    confidence: float = 0.0
    parent_ids: List[str] = field(default_factory=list)
    children_ids: List[str] = field(default_factory=list)


class ThoughtPathGenerator(nn.Module):
    """Generates a single reasoning path from privatized latent features."""

    def __init__(self, feature_dim: int, num_actions: int):
        super().__init__()
        self.reasoning_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2), nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim), nn.GELU())
        self.action_head = nn.Linear(feature_dim, num_actions)
        self.bbox_head = nn.Sequential(
            nn.Linear(feature_dim, 128), nn.GELU(),
            nn.Linear(128, 4), nn.Sigmoid())
        self.confidence_head = nn.Sequential(
            nn.Linear(feature_dim, 64), nn.GELU(),
            nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, phi_private: torch.Tensor):
        reasoning = self.reasoning_mlp(phi_private)
        action_logits = self.action_head(reasoning)
        bbox = self.bbox_head(reasoning)
        confidence = self.confidence_head(reasoning).squeeze(-1)
        return reasoning, action_logits, bbox, confidence


class GoTAggregator(nn.Module):
    """
    Aggregates outputs from k parallel thought paths.
      Eq.5: s_i = α · Logit(ĉ_i) + (1-α) · ⟨φ̃(ĉ_i), θ̂⟩
      Eq.7: w_j = softmax(s_j/Tagg),  C* = Σ w_j · ĉ_j
      Eq.8: U_t = (1/k) Σ ||ĉ_i - C*||²
    """

    def __init__(self, feature_dim: int, alpha: float, temperature: float):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature

    def forward(self, reasoning_features: torch.Tensor, action_logits: torch.Tensor,
                bbox_preds: torch.Tensor, confidences: torch.Tensor,
                global_weight: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        logit_score = confidences
        if global_weight is not None:
            theta_score = torch.einsum('bkd,d->bk', reasoning_features, global_weight)
        else:
            theta_score = torch.zeros_like(logit_score)
        scores = self.alpha * logit_score + (1 - self.alpha) * theta_score
        weights = F.softmax(scores / self.temperature, dim=-1)
        agg_bbox = torch.einsum('bk,bkc->bc', weights, bbox_preds)
        agg_action = torch.einsum('bk,bka->ba', weights, action_logits)
        agg_features = torch.einsum('bk,bkd->bd', weights, reasoning_features)
        diff = bbox_preds - agg_bbox.unsqueeze(1)
        sq_dist = diff.pow(2).sum(dim=-1)
        uncertainty = sq_dist.mean().item()
        return agg_action, agg_bbox, agg_features, uncertainty


class GoTEngine(nn.Module):
    """Full Graph of Thoughts engine with k parallel paths (k set by SAC)."""

    def __init__(self, feature_dim: int, max_k: int, num_actions: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_k = max_k
        self.num_actions = num_actions
        self.path_generator = ThoughtPathGenerator(feature_dim, num_actions)
        self.diversity_projs = nn.ModuleList(
            [nn.Linear(feature_dim, feature_dim, bias=False) for _ in range(max_k)])
        for proj in self.diversity_projs:
            nn.init.eye_(proj.weight)
            proj.weight.data += torch.randn_like(proj.weight) * 0.05
        self.aggregator = GoTAggregator(feature_dim, alpha=0.5, temperature=1.0)

    def forward(self, phi_private: torch.Tensor, k: int,
                ltm_prior: Optional[torch.Tensor] = None,
                global_weight: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        k = min(k, self.max_k)
        batch_size = phi_private.shape[0]
        all_reasoning, all_actions, all_bboxes, all_confs = [], [], [], []
        for i in range(k):
            diversified = self.diversity_projs[i](phi_private)
            if ltm_prior is not None:
                diversified = diversified + 0.1 * ltm_prior
            reasoning, action_logits, bbox, conf = self.path_generator(diversified)
            all_reasoning.append(reasoning)
            all_actions.append(action_logits)
            all_bboxes.append(bbox)
            all_confs.append(conf)
        reasoning_stack = torch.stack(all_reasoning, dim=1)
        action_stack = torch.stack(all_actions, dim=1)
        bbox_stack = torch.stack(all_bboxes, dim=1)
        conf_stack = torch.stack(all_confs, dim=1)
        agg_action, agg_bbox, agg_features, uncertainty = self.aggregator(
            reasoning_stack, action_stack, bbox_stack, conf_stack, global_weight=global_weight)
        return {
            'action_logits': agg_action,
            'bbox': agg_bbox,
            'features': agg_features,
            'uncertainty': uncertainty,
            'all_action_logits': action_stack,
            'all_bboxes': bbox_stack,
        }
