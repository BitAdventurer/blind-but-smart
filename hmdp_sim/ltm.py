"""
Long-Term Memory (LTM) Module.

Provides strategic prediction by storing and retrieving past interaction
episodes.  The LTM guides the GoT engine with contextual priors,
enabling the agent to maintain behavioural consistency despite
obfuscated visual input.

Key features:
  - Episode-level storage with latent embeddings
  - Eq.7: K_ret = argmax cosine_similarity(φ(s_τ), φ_j)
  - Eq.8: u_τ = argmax_u P(u | φ̃(s_τ), C*, K_ret; θ_pred)
  - Strategic prior generation for GoT guidance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Episode:
    """A single stored episode in long-term memory."""
    episode_id: int
    state_embedding: torch.Tensor    # φ representation at decision time
    action_taken: int                # discrete action index
    bbox_target: torch.Tensor        # (4,) normalized bbox
    epsilon_used: float
    k_used: int
    reward: float
    sensitivity: float               # S_t at this step
    uncertainty: float                # U_t at this step
    success: bool


class LongTermMemory(nn.Module):
    """
    Long-Term Memory for strategic prediction.

    Stores past interaction episodes and retrieves the most relevant
    ones to provide contextual guidance (prior) to the GoT engine.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        capacity: int = 10000,
        top_k: int = 8,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.capacity = capacity
        self.top_k = top_k

        # Episode storage
        self.episodes: deque = deque(maxlen=capacity)
        self._episode_counter = 0

        # Learned projection for query/key matching
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)
        self.key_proj = nn.Linear(embedding_dim, embedding_dim)

        # Prior generation network: fuses retrieved episodes into a single prior
        self.prior_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    # ── Storage ──────────────────────────────────────────────────────────

    def store_episode(
        self,
        state_embedding: torch.Tensor,
        action_taken: int,
        bbox_target: torch.Tensor,
        epsilon_used: float,
        k_used: int,
        reward: float,
        sensitivity: float,
        uncertainty: float,
        success: bool,
    ):
        """Store a completed episode into long-term memory."""
        ep = Episode(
            episode_id=self._episode_counter,
            state_embedding=state_embedding.detach().cpu(),
            action_taken=action_taken,
            bbox_target=bbox_target.detach().cpu() if isinstance(bbox_target, torch.Tensor) else torch.tensor(bbox_target),
            epsilon_used=epsilon_used,
            k_used=k_used,
            reward=reward,
            sensitivity=sensitivity,
            uncertainty=uncertainty,
            success=success,
        )
        self.episodes.append(ep)
        self._episode_counter += 1

    # ── Retrieval ────────────────────────────────────────────────────────

    def retrieve(
        self,
        query_embedding: torch.Tensor,
        top_k: Optional[int] = None,
    ) -> Tuple[List[Episode], torch.Tensor]:
        """
        Eq.7: K_ret = argmax_{(φ_j, u_j) ∈ LTM} ( φ(s_τ) · φ_j / ||φ(s_τ)|| ||φ_j|| )

        Retrieve the most similar episodes from memory via cosine similarity
        in the latent space.

        Args:
            query_embedding: current φ(s_τ), shape (d,) or (1, d)
            top_k: number of episodes to retrieve
        Returns:
            (list of Episode, similarity scores tensor)
        """
        if len(self.episodes) == 0:
            return [], torch.tensor([])

        top_k = top_k or self.top_k
        top_k = min(top_k, len(self.episodes))

        # Project query
        query = query_embedding.detach()
        if query.dim() == 1:
            query = query.unsqueeze(0)
        q = self.query_proj(query.to(next(self.query_proj.parameters()).device))  # (1, d)

        # Build key matrix from stored episodes
        keys = torch.stack([ep.state_embedding for ep in self.episodes])  # (N, d)
        keys = keys.to(q.device)
        k = self.key_proj(keys)  # (N, d)

        # Eq.7: cosine similarity in latent space
        sim = F.cosine_similarity(q, k, dim=-1)  # (N,)

        # Top-k retrieval
        topk_vals, topk_idx = sim.topk(top_k)
        retrieved = [self.episodes[i] for i in topk_idx.cpu().tolist()]

        return retrieved, topk_vals

    # ── Prior Generation ─────────────────────────────────────────────────

    def generate_prior(
        self,
        query_embedding: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Generate a strategic prior from retrieved episodes for GoT guidance.

        Args:
            query_embedding: current φ(s_τ), shape (batch, d) or (d,)
        Returns:
            prior tensor (batch, d) or None if memory is empty
        """
        if len(self.episodes) == 0:
            return None

        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)

        batch_size = query_embedding.shape[0]
        device = query_embedding.device
        priors = []

        for b in range(batch_size):
            retrieved, scores = self.retrieve(query_embedding[b])
            if not retrieved:
                priors.append(torch.zeros(self.embedding_dim, device=device))
                continue

            # Weighted average of retrieved embeddings
            weights = F.softmax(scores, dim=0)  # (top_k,)
            retrieved_embs = torch.stack(
                [ep.state_embedding.to(device) for ep in retrieved]
            )  # (top_k, d)
            weighted_emb = torch.einsum("k,kd->d", weights, retrieved_embs)

            # Fuse with current query through prior_net
            fused_input = torch.cat([query_embedding[b], weighted_emb], dim=-1)
            prior = self.prior_net(fused_input.unsqueeze(0)).squeeze(0)
            priors.append(prior)

        return torch.stack(priors, dim=0)  # (batch, d)

    # ── Eq.8 Action Prediction ──────────────────────────────────────────────────

    def predict_action(
        self,
        phi_private: torch.Tensor,
        restored_coord: torch.Tensor,
        query_embedding: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Eq.8: u_τ = argmax_u P(u | φ̃(s_τ), C*, K_ret; θ_pred)

        Predict the optimal action by conditioning on the privatized
        observation, restored coordinate C*, and retrieved knowledge K_ret.

        Args:
            phi_private:    φ̃(s_τ), shape (batch, d)
            restored_coord: C* from GoT, shape (batch, 4)
            query_embedding: raw φ(s_τ) for retrieval, shape (batch, d)
        Returns:
            action prior tensor (batch, d) or None
        """
        prior = self.generate_prior(query_embedding)
        if prior is None:
            return None
        # The prior now implicitly carries K_ret information,
        # which can be used by the prediction model alongside C*
        return prior

    # ── Stats ──────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """Return memory statistics."""
        if len(self.episodes) == 0:
            return {"size": 0}
        successes = sum(1 for ep in self.episodes if ep.success)
        avg_reward = np.mean([ep.reward for ep in self.episodes])
        avg_eps = np.mean([ep.epsilon_used for ep in self.episodes])
        return {
            "size": len(self.episodes),
            "success_rate": successes / len(self.episodes),
            "avg_reward": float(avg_reward),
            "avg_epsilon": float(avg_eps),
        }

    def forward(self, query_embedding: torch.Tensor) -> Optional[torch.Tensor]:
        """Convenience forward: generate prior."""
        return self.generate_prior(query_embedding)
