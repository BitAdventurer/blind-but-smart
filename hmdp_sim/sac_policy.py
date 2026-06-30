"""
Soft Actor-Critic (SAC) Meta-Policy (M_high).

Continuous SAC (Haarnoja et al., 2018) faithful to Sec. 3.6:
  - State:  s_t = [U_t, λ_t^(1..M)] ∈ ℝ^26
  - Action: a_t = [ε_t^(1..M), k̂_t] ∈ ℝ^26  (tanh-squashed in [-1,1]^26)
  - Dense reward (Eq. 2): r_τ = w_perf·q_τ + w_priv·log(M/ε̄_τ) − w_comp
  - Terminal PES bonus (Eq. 4): r_pes = w_pes·SR_t·exp(trajectory-avg log(M/ε̄))
  - Total return (Eq. 5): G_t = Σr_τ + r_pes
  - Objective (Eq. 11): J(π) = Σ_t E[R + α H(π(·|s_t))]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from collections import deque
import random


class ReplayBuffer:
    """Experience replay buffer for SAC training."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.stack(actions).float(),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


LOG_STD_MIN = -20
LOG_STD_MAX = 2


class SACQNetwork(nn.Module):
    """Continuous critic Q(s, a) → scalar value."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x).squeeze(-1)


class SACPolicyNetwork(nn.Module):
    """Continuous Gaussian actor π(a|s) with tanh squashing."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor):
        h = self.backbone(state)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor, deterministic: bool = False):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        if deterministic:
            x_t = mean
        else:
            x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log((1 - action.pow(2)) + 1e-06)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        with torch.no_grad():
            action, _ = self.sample(state.unsqueeze(0), deterministic)
        return action.squeeze(0).cpu().numpy()


class SACMetaPolicy(nn.Module):
    """Continuous Soft Actor-Critic for hierarchical meta-governance."""

    def __init__(self, state_dim: int = 26, action_dim: int = 26, hidden_dim: int = 256,
                 lr_actor: float = 3e-4, lr_critic: float = 3e-4, lr_alpha: float = 3e-4,
                 gamma: float = 0.99, tau: float = 0.005, alpha_init: float = 0.2,
                 auto_alpha: bool = True, buffer_size: int = 100000, batch_size: int = 256,
                 device: str = 'cpu'):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.actor = SACPolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q1 = SACQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q2 = SACQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q1_target = SACQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q2_target = SACQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy = -float(action_dim)
            self.log_alpha = torch.tensor(np.log(alpha_init), dtype=torch.float32,
                                          requires_grad=True, device=self.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr_alpha)
        else:
            self.log_alpha = torch.tensor(np.log(alpha_init), device=self.device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.q1_optim = torch.optim.Adam(self.q1.parameters(), lr=lr_critic)
        self.q2_optim = torch.optim.Adam(self.q2.parameters(), lr=lr_critic)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.train_step = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state: torch.Tensor, deterministic: bool = False):
        state = state.to(self.device)
        return self.actor.get_action(state, deterministic)

    @staticmethod
    def straight_through_round(x):
        """Round in the forward pass, identity gradient in the backward pass."""
        return x + (torch.round(x) - x).detach()

    def store_transition(self, state, action, reward, next_state, done):
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32)
        self.replay_buffer.push(state.detach().cpu(), action.detach().cpu().float(),
                                reward, next_state.detach().cpu(), done)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return {}
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next = self.q1_target(next_states, next_actions)
            q2_next = self.q2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next)
            v_next = q_next - self.alpha * next_log_probs.squeeze(-1)
            target_q = rewards + (1 - dones) * self.gamma * v_next
        q1_vals = self.q1(states, actions)
        q2_vals = self.q2(states, actions)
        q1_loss = F.mse_loss(q1_vals, target_q)
        q2_loss = F.mse_loss(q2_vals, target_q)
        self.q1_optim.zero_grad(); q1_loss.backward(); self.q1_optim.step()
        self.q2_optim.zero_grad(); q2_loss.backward(); self.q2_optim.step()
        new_actions, log_probs = self.actor.sample(states)
        q1_pi = self.q1(states, new_actions)
        q2_pi = self.q2(states, new_actions)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * log_probs.squeeze(-1) - q_pi).mean()
        self.actor_optim.zero_grad(); actor_loss.backward(); self.actor_optim.step()
        alpha_loss = torch.tensor(0)
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs.detach().squeeze(-1) + self.target_entropy)).mean()
            self.alpha_optim.zero_grad(); alpha_loss.backward(); self.alpha_optim.step()
            with torch.no_grad():
                self.log_alpha.clamp_(-5, 2)
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        self.train_step += 1
        return {
            'q1_loss': q1_loss.item(), 'q2_loss': q2_loss.item(),
            'actor_loss': actor_loss.item(), 'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha.item(), 'q_mean': q_pi.mean().item(),
        }

    @staticmethod
    def compute_step_reward(q_tau: float, epsilons: List[float],
                            w_perf: float = 1.0, w_priv: float = 0.5,
                            w_comp: float = 0.1, num_regions: int = 25) -> float:
        """Dense per-step reward r_τ (Eq. 2).

        r_τ = w_perf·q_τ + w_priv·log(M/ε̄_τ) − w_comp
        where ε̄_τ = mean(epsilons) and q_τ ∈ [0,1] is the task-quality signal.
        """
        eps_mean = max(sum(epsilons) / max(len(epsilons), 1), 1e-6)
        r_perf = w_perf * float(q_tau)
        r_priv = w_priv * np.log(num_regions / eps_mean)
        r_comp = w_comp
        return r_perf + r_priv - r_comp

    @staticmethod
    def compute_terminal_pes_bonus(sr_t: float, eps_means: List[float],
                                   w_pes: float = 2.0,
                                   num_regions: int = 25) -> float:
        """Terminal PES-aligned bonus r_pes (Eq. 4).

        r_pes = w_pes · SR_t · exp( (1/T) Σ_τ log(M/ε̄_τ) )
        eps_means: list of per-step mean epsilons [ε̄_1, ..., ε̄_T].
        """
        if not eps_means:
            return 0.0
        avg_log_budget = np.mean([np.log(num_regions / max(e, 1e-6)) for e in eps_means])
        return w_pes * float(sr_t) * np.exp(avg_log_budget)

    @staticmethod
    def compute_reward(success: bool, epsilons: List[float], k: int,
                       w_perf: float = 1.0, w_priv: float = 0.5, w_comp: float = 0.1,
                       num_regions: int = 25) -> float:
        """Single-step reward (legacy compatibility shim — special case of Eq.2 with q_τ=I(success)).

        For multi-step trajectories use compute_step_reward + compute_terminal_pes_bonus.
        """
        return SACMetaPolicy.compute_step_reward(
            q_tau=float(success),
            epsilons=epsilons,
            w_perf=w_perf,
            w_priv=w_priv,
            w_comp=w_comp,
            num_regions=num_regions,
        )

    def save(self, path: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'q1': self.q1.state_dict(), 'q2': self.q2.state_dict(),
            'q1_target': self.q1_target.state_dict(),
            'q2_target': self.q2_target.state_dict(),
            'log_alpha': self.log_alpha, 'train_step': self.train_step,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.q1.load_state_dict(ckpt['q1'])
        self.q2.load_state_dict(ckpt['q2'])
        self.q1_target.load_state_dict(ckpt['q1_target'])
        self.q2_target.load_state_dict(ckpt['q2_target'])
        self.log_alpha = ckpt['log_alpha']
        self.train_step = ckpt['train_step']
