#!/usr/bin/env python3
"""
GoT Aggregation: Confidence-weighted coordinate fusion.

Implements Stage-3 of the GoT pipeline (Sec. 3.5):
  - Score-based path weighting (Eq. got_score)
  - Softmax aggregation (Eq. got_agg)
  - Uncertainty estimation (U_t for LTM gating)
  - Geometric median and clustering utilities
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from hmdp.constants import GridConfig, GoTHyperparams


def _geometric_median(points: torch.Tensor, max_iter: int = 20) -> torch.Tensor:
    """Compute geometric median via Weiszfeld's algorithm. More robust than mean."""
    y = points.mean(dim=0)
    for _ in range(max_iter):
        dists = torch.norm(points - y, dim=1, keepdim=True).clamp(min=1e-8)
        weights = 1.0 / dists
        y_new = (weights * points).sum(dim=0) / weights.sum()
        if torch.norm(y_new - y) < 1e-7:
            break
        y = y_new
    return y


def _cluster_paths(
    points: torch.Tensor,
    radius: float = GoTHyperparams.CLUSTER_RADIUS,
) -> List[List[int]]:
    """
    Simple proximity-based clustering of 2D points.

    Two points belong to the same cluster if they are within `radius`.
    Uses single-linkage: if any member of a cluster is within radius
    of a new point, that point joins the cluster.
    """
    n = points.shape[0]
    visited = [False] * n
    clusters = []
    for i in range(n):
        if visited[i]:
            continue
        cluster = [i]
        visited[i] = True
        queue = [i]
        while queue:
            cur = queue.pop(0)
            for j in range(n):
                if not visited[j]:
                    if torch.norm(points[cur] - points[j]).item() < radius:
                        visited[j] = True
                        cluster.append(j)
                        queue.append(j)
        clusters.append(cluster)
    return clusters


def _coord_to_grid_index(point: List[float], grid: int = GridConfig.SIZE) -> int:
    """
    Map a normalized coordinate [x, y] in [0,1]^2 to the row-major index of
    the grid x grid region that encloses it.

    Must match DINOv2RegionEncoder.partition (hmdp.dinov2_encoder), which iterates
    rows then columns: idx = row * grid + col, with box columns scaled by x
    and rows by y.
    """
    x = min(max(point[0], 0.0), 1.0 - 1e-9)
    y = min(max(point[1], 0.0), 1.0 - 1e-9)
    col = min(int(x * grid), grid - 1)
    row = min(int(y * grid), grid - 1)
    return row * grid + col


def got_aggregate(
    paths: List[Dict],
    psi_grid: Optional[torch.Tensor],
    theta_hat: Optional[torch.Tensor],
    alpha_w: float = GoTHyperparams.ALPHA_W,  # 0.5
    T_agg: float = GoTHyperparams.T_AGG,  # 1.0
    grid: int = GridConfig.SIZE,  # 5
) -> Tuple[str, Optional[List[float]], float]:
    """
    GoT confidence-weighted coordinate aggregation (Eqs. got_score, got_agg).

        s_j = alpha_w * Logit_VLM(c_j) + (1 - alpha_w) * <psi_j, theta_hat>
        w_j = softmax(s_j / T_agg)
        C*  = sum_j w_j * c_j
        U_t = (1/k) * sum_j || c_j - C* ||^2          (decision-level variance)

    where:
      Logit_VLM(c_j) = path['logit'] (mean logprob of the coordinate tokens),
      psi_j          = psi_grid[ region(c_j) ]  (privatized latent at the grid
                       region enclosing c_j); when psi_grid or theta_hat is
                       None the semantic term is dropped (logit-only scoring),
      theta_hat      = task embedding Embed(instruction).

    Returns (action_type, C*, U_t). U_t is consumed by the caller for the
    conditional LTM gate (query LTM only when U_t > tau_LTM).
    """
    greedy_action = paths[0]["action_type"] if paths else "click"
    greedy_point = paths[0]["pred_point"] if paths else None

    valid_paths = [p for p in paths if p.get("pred_point") is not None]
    if not valid_paths:
        return greedy_action, greedy_point, 0.0
    if len(valid_paths) == 1:
        return valid_paths[0]["action_type"], valid_paths[0]["pred_point"], 0.0

    coords = torch.tensor(
        [p["pred_point"] for p in valid_paths], dtype=torch.float32
    )  # (k, 2)

    # Semantic alignment term <psi_j, theta_hat> per path.
    use_semantic = psi_grid is not None and theta_hat is not None
    if use_semantic:
        th = theta_hat.detach().to(torch.float32).flatten()  # (d,)
        psi = psi_grid.detach().to(torch.float32)            # (M, d)

    scores = []
    for j, p in enumerate(valid_paths):
        logit = p.get("logit")
        logit_term = float(logit) if logit is not None else 0.0
        if use_semantic:
            gidx = _coord_to_grid_index(p["pred_point"], grid=grid)
            gidx = min(gidx, psi.shape[0] - 1)
            sem_term = float(torch.dot(psi[gidx], th.to(psi.device)).item())
        else:
            sem_term = 0.0
        scores.append(alpha_w * logit_term + (1.0 - alpha_w) * sem_term)

    s = torch.tensor(scores, dtype=torch.float32)
    w = torch.softmax(s / max(T_agg, 1e-6), dim=0)          # (k,)

    c_star = (w.unsqueeze(1) * coords).sum(dim=0)           # (2,)
    u_t = float(((coords - c_star.unsqueeze(0)) ** 2).sum(dim=1).mean().item())

    # Action type: confidence-weighted vote (same weights as coordinates),
    # greedy as tiebreaker.
    action_weight: Dict[str, float] = {}
    for j, p in enumerate(valid_paths):
        a = p["action_type"]
        action_weight[a] = action_weight.get(a, 0.0) + float(w[j].item())
    max_w = max(action_weight.values())
    top_actions = [a for a, wv in action_weight.items() if abs(wv - max_w) < 1e-9]
    best_action = greedy_action if greedy_action in top_actions else top_actions[0]

    return best_action, c_star.tolist(), u_t
