"""
H-MDP Framework: Main Orchestrator.

Integrates the upper-level SAC meta-policy (M_high) with the
lower-level execution engine (M_low) into a unified hierarchical reinforcement
learning framework.

Closed-loop flow:
  1. Observe GUI screen -> compute governance state s_t = [U_t, Lambda_t] (26-dim)
  2. SAC selects (eps_t^(1..M), k_t) based on s_t  (26-dim action)
  3. Execution engine: per-region LDP -> W_proj -> (LTM if U_t>tau) -> GoT(k paths) -> action
  4. Environment evaluates action -> reward R_t (Eq. 2)
  5. U_{t+1} and Lambda_{t+1} fed back; SAC updates
"""
import torch
from torch import nn
import numpy as np
import os
import json
from typing import Dict, List, Optional, Tuple

from .config import HMDPConfig
from .sac_policy import SACMetaPolicy
from .execution_engine import ExecutionEngine
from .gui_env import GUIEnvironment


class HMDPFramework:
    """
    Main H-MDP framework that ties everything together.

    Usage:
        config = HMDPConfig()
        framework = HMDPFramework(config)
        framework.train(num_episodes=5000)
        framework.evaluate(num_episodes=100)
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        self.meta_policy = SACMetaPolicy(
            state_dim=config.sac_state_dim,
            action_dim=config.sac_action_dim,
            hidden_dim=config.sac_hidden_dim,
            lr_actor=config.sac_lr_actor,
            lr_critic=config.sac_lr_critic,
            lr_alpha=config.sac_lr_alpha,
            gamma=config.sac_gamma,
            tau=config.sac_tau,
            alpha_init=config.sac_alpha_init,
            auto_alpha=config.sac_auto_alpha,
            buffer_size=config.replay_buffer_size,
            batch_size=config.batch_size,
            device=config.device)
        self.exec_engine = ExecutionEngine(
            proxy_encoder_dim=config.proxy_encoder_dim,
            feature_dim=config.latent_dim,
            max_k=max(config.k_levels),
            num_actions=4,
            num_regions=config.num_regions,
            ltm_capacity=config.ltm_capacity,
            ltm_top_k=config.ltm_top_k,
            ltm_uncertainty_threshold=config.ltm_uncertainty_threshold).to(self.device)
        self.env = GUIEnvironment(
            data_path=config.gui360_data_path,
            image_base_path=config.gui360_image_base,
            proxy_encoder_dim=config.proxy_encoder_dim,
            max_steps=config.max_steps_per_episode,
            device=config.device)
        self.train_log = []
        self.eval_log = []

    def _random_action_vec(self):
        """Sample a random 26-dim continuous action in [-1, 1] during warmup."""
        return np.random.uniform(-1, 1, size=self.config.sac_action_dim).astype(np.float32)

    def _policy_action_vec(self, gov_state, deterministic):
        """
        Continuous SAC: the actor emits a tanh-squashed 26-dim action in
        [-1, 1]^26 (25 per-region eps slots + 1 reasoning-depth slot k_hat), which
        HMDPConfig.action_to_params affinely rescales to (eps_t^(1..M), k_t).
        """
        return self.meta_policy.select_action(gov_state, deterministic)

    def run_episode(self, deterministic=False, training=True):
        """
        Run a single episode of the H-MDP closed loop.

        Returns:
            Episode statistics dict
        """
        state = self.env.reset()
        current_uncertainty = 0.5
        lambda_t_init = torch.zeros(self.config.num_regions, dtype=torch.float32)
        gov_state = torch.cat([torch.tensor([current_uncertainty]), lambda_t_init]).to(self.device)
        episode_reward = 0.0
        episode_steps = 0
        episode_success = False
        all_epsilons = []
        ks_used = []
        for t in range(self.config.max_steps_per_episode):
            if training and self.meta_policy.train_step < self.config.warmup_steps:
                action_vec = self._random_action_vec()
            else:
                action_vec = self._policy_action_vec(gov_state, deterministic)
            epsilons_t, k_t = self.config.action_to_params(action_vec)
            all_epsilons.append(epsilons_t)
            ks_used.append(k_t)
            with (torch.no_grad() if not training else torch.enable_grad()):
                exec_output = self.exec_engine(
                    encoder_output=state.encoder_output,
                    epsilons=epsilons_t,
                    k=k_t,
                    current_uncertainty=current_uncertainty,
                    update_server=training)
            uncertainty_t = exec_output['uncertainty']
            lambda_t = exec_output['lambda_t']
            next_state, success_signal, done, info = self.env.step(
                pred_action_logits=exec_output['action_logits'].squeeze(0),
                pred_bbox=exec_output['bbox'].squeeze(0),
                epsilon_used=float(np.mean(epsilons_t)),
                k_used=k_t,
                uncertainty=uncertainty_t)
            reward = SACMetaPolicy.compute_step_reward(
                q_tau=float(info.get('q_tau', info['success'])),
                epsilons=epsilons_t,
                w_perf=self.config.w_perf,
                w_priv=self.config.w_priv,
                w_comp=self.config.w_comp,
                num_regions=self.config.num_regions)
            episode_reward += reward
            next_gov_state = self.exec_engine.build_governance_state(
                uncertainty=uncertainty_t,
                lambda_t=lambda_t.squeeze(0) if lambda_t.dim() == 2 else lambda_t).to(self.device)
            action_tensor = torch.as_tensor(action_vec, dtype=torch.float32)
            if training:
                self.meta_policy.store_transition(gov_state, action_tensor, reward, next_gov_state, done)
                self.meta_policy.update()
                if exec_output['phi_private'] is not None:
                    self.exec_engine.ltm.store_episode(
                        state_embedding=exec_output['phi_private'].squeeze(0),
                        action_taken=exec_output['action_logits'].argmax(dim=-1).item(),
                        bbox_target=exec_output['bbox'].squeeze(0),
                        epsilon_used=float(np.mean(epsilons_t)),
                        k_used=k_t,
                        reward=reward,
                        sensitivity=state.sensitivity,
                        uncertainty=uncertainty_t,
                        success=info['success'])
            episode_steps += 1
            current_uncertainty = uncertainty_t
            if info['success']:
                episode_success = True
            if done:
                break
            else:
                state = next_state
                gov_state = next_gov_state
        avg_eps_per_step = [np.mean(eps) for eps in all_epsilons]
        r_pes = SACMetaPolicy.compute_terminal_pes_bonus(
            sr_t=float(episode_success),
            eps_means=avg_eps_per_step,
            w_pes=getattr(self.config, 'w_pes', 2.0),
            num_regions=self.config.num_regions)
        total_return = episode_reward + r_pes
        return {
            'reward': total_return,
            'steps': episode_steps,
            'success': episode_success,
            'avg_epsilon': float(np.mean(avg_eps_per_step)),
            'all_epsilons': all_epsilons,
            'avg_k': float(np.mean(ks_used)),
            'final_iou': info.get('bbox_iou', 0.0),
            'r_pes': r_pes}

    def train(self, num_episodes=None, verbose=True):
        """
        Train the H-MDP framework.

        Args:
            num_episodes: override config.num_episodes
            verbose: print progress
        """
        if not num_episodes:
            num_episodes = self.config.num_episodes
        os.makedirs(self.config.log_dir, exist_ok=True)
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        self.exec_engine.train()
        running_reward = 0.0
        running_sr = 0.0
        for ep in range(1, num_episodes + 1):
            stats = self.run_episode(training=True)
            self.train_log.append(stats)
            running_reward = 0.95 * running_reward + 0.05 * stats['reward']
            running_sr = 0.95 * running_sr + 0.05 * float(stats['success'])
            if verbose and ep % self.config.eval_interval == 0:
                ltm_stats = self.exec_engine.ltm.get_stats()
                print(f"[Train EP {ep:5d}] R={running_reward:.3f}  SR={running_sr:.3f}  "
                      f"\u03b5\u0304={stats['avg_epsilon']:.2f}  k\u0304={stats['avg_k']:.1f}  "
                      f"Steps={stats['steps']}  LTM={ltm_stats['size']}  "
                      f"\u03b1={self.meta_policy.alpha.item():.4f}")
            if ep % self.config.eval_interval == 0:
                eval_stats = self.evaluate(num_episodes=20, verbose=False)
                if verbose:
                    print(f"  [Eval]  SR={eval_stats['success_rate']:.3f}  "
                          f"\u03b5\u0304={eval_stats['avg_epsilon']:.2f}  PES={eval_stats['pes']:.4f}")
            if ep % self.config.save_interval == 0:
                self.save_checkpoint(os.path.join(self.config.checkpoint_dir, f'hmdp_ep{ep}.pt'))
        self.save_checkpoint(os.path.join(self.config.checkpoint_dir, 'hmdp_final.pt'))
        self._save_log(os.path.join(self.config.log_dir, 'train_log.json'), self.train_log)
        print(f'Training complete. {num_episodes} episodes.')

    def evaluate(self, num_episodes=100, verbose=True):
        """
        Evaluate the trained framework.

        Returns:
            dict with success_rate, avg_epsilon, pes, etc.
        """
        self.exec_engine.eval()
        results = []
        for ep in range(num_episodes):
            stats = self.run_episode(deterministic=True, training=False)
            results.append(stats)
        sr = np.mean([r['success'] for r in results])
        avg_eps = np.mean([r['avg_epsilon'] for r in results])
        avg_k = np.mean([r['avg_k'] for r in results])
        avg_steps = np.mean([r['steps'] for r in results])
        avg_reward = np.mean([r['reward'] for r in results])
        M = self.config.num_regions
        pes_terms = []
        for r in results:
            for eps_step in r['all_epsilons']:
                eps_sum = max(sum(eps_step), 1e-06)
                pes_terms.append(np.log(M / eps_sum))
        pes = sr * np.exp(np.mean(pes_terms)) if pes_terms else 0.0
        eval_result = {
            'success_rate': sr,
            'avg_epsilon': avg_eps,
            'avg_k': avg_k,
            'avg_steps': avg_steps,
            'avg_reward': avg_reward,
            'pes': pes}
        self.eval_log.append(eval_result)
        if verbose:
            sep = '=' * 60
            print(f'\n{sep}')
            print(f'  Evaluation Results ({num_episodes} episodes)')
            print(f'{sep}')
            print(f'  Success Rate (SR):         {sr:.4f}')
            print(f'  Avg Privacy (\u03b5\u0304):           {avg_eps:.4f}')
            print(f'  Avg Reasoning Paths (k\u0304):   {avg_k:.1f}')
            print(f'  Avg Steps:                 {avg_steps:.1f}')
            print(f'  Avg Reward:                {avg_reward:.4f}')
            print(f'  Privacy-Efficiency (PES):  {pes:.4f}')
            print(f'{sep}\n')
        self.exec_engine.train()
        return eval_result

    def save_checkpoint(self, path):
        """Save full framework checkpoint."""
        torch.save({
            'meta_policy': {
                'actor': self.meta_policy.actor.state_dict(),
                'q1': self.meta_policy.q1.state_dict(),
                'q2': self.meta_policy.q2.state_dict(),
                'q1_target': self.meta_policy.q1_target.state_dict(),
                'q2_target': self.meta_policy.q2_target.state_dict(),
                'log_alpha': self.meta_policy.log_alpha,
                'train_step': self.meta_policy.train_step},
            'exec_engine': self.exec_engine.state_dict(),
            'config': self.config.__dict__}, path)

    def load_checkpoint(self, path):
        """Load framework checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        mp = ckpt['meta_policy']
        self.meta_policy.actor.load_state_dict(mp['actor'])
        self.meta_policy.q1.load_state_dict(mp['q1'])
        self.meta_policy.q2.load_state_dict(mp['q2'])
        self.meta_policy.q1_target.load_state_dict(mp['q1_target'])
        self.meta_policy.q2_target.load_state_dict(mp['q2_target'])
        self.meta_policy.log_alpha = mp['log_alpha']
        self.meta_policy.train_step = mp['train_step']
        self.exec_engine.load_state_dict(ckpt['exec_engine'])
        print(f'Loaded checkpoint from {path}')

    @staticmethod
    def _save_log(path, log):
        with open(path, 'w') as f:
            json.dump(log, f, indent=2, default=str)
