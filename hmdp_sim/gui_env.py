"""
GUI Environment Wrapper.

Gym-like interface for training the H-MDP agent on the GUI-360 benchmark under
black-box constraints (raw pixels only; no DOM/accessibility APIs).
"""
import torch
import numpy as np
import json
import os
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class GUIState:
    """Represents the current state of the GUI environment."""
    image: object
    encoder_output: torch.Tensor
    instruction: str
    target_bbox: torch.Tensor
    target_action: int
    step: int
    sensitivity: float
    uncertainty: float = 0.5
    is_fail_sample: bool = False
    true_reward: float = None


class GUIEnvironment:
    """
    Simulated black-box GUI environment using GUI 360 data.

    The environment:
      - Loads screenshots and action labels from the dataset
      - Provides proxy encoder features E(I_t) (simulated or from DINOv2)
      - Evaluates agent actions against ground-truth
      - Computes success/failure signals
    """

    ACTION_TYPES = ['click', 'type', 'drag', 'scroll']

    def __init__(self, data_path: str, image_base_path: str, proxy_encoder_dim: int = 1024,
                 max_steps: int = 15, device: str = 'cpu',
                 fail_data_path: Optional[str] = None, fail_ratio: float = 0.0):
        self.data_path = data_path
        self.image_base_path = image_base_path
        self.proxy_encoder_dim = proxy_encoder_dim
        self.max_steps = max_steps
        self.device = device
        self.fail_data_path = fail_data_path
        self.fail_ratio = fail_ratio
        self.data = []
        self.fail_data = []
        self._load_data()
        self.current_episode_idx = 0
        self.current_step = 0
        self.current_state = None
        self.episode_history = []

    def _load_data(self):
        """Load GUI-360 dataset (success data) and optionally fail data."""
        if os.path.exists(self.data_path):
            with open(self.data_path, 'r') as f:
                self.data = json.load(f)
            print(f'[GUIEnv] Loaded {len(self.data)} SUCCESS samples from {self.data_path}')
        else:
            print(f'[GUIEnv] Dataset not found at {self.data_path}, using synthetic data')
            self._generate_synthetic_data(500)
        if self.fail_data_path and os.path.exists(self.fail_data_path):
            with open(self.fail_data_path, 'r') as f:
                self.fail_data = json.load(f)
            print(f'[GUIEnv] Loaded {len(self.fail_data)} FAIL samples from {self.fail_data_path}')
            self.fail_rewards = [s.get('reward', 0) for s in self.fail_data]
            print(f'[GUIEnv] Fail reward range: [{min(self.fail_rewards):.2f}, {max(self.fail_rewards):.2f}]')
            return
        if self.fail_ratio > 0:
            print('[GUIEnv] Warning: fail_data_path not found or not specified')
        self.fail_data = []
        self.fail_rewards = []

    def _generate_synthetic_data(self, n: int):
        """Generate synthetic data for testing without real dataset."""
        for i in range(n):
            action_type = np.random.randint(0, len(self.ACTION_TYPES))
            bbox = sorted(np.random.uniform(0.1, 0.9, 2).tolist()) + \
                   sorted(np.random.uniform(0.1, 0.9, 2).tolist())
            self.data.append({
                'id': f'synthetic_{i}',
                'instruction': f'Perform {self.ACTION_TYPES[action_type]} on element',
                'action_type': action_type,
                'bbox': bbox,
                'sensitivity': np.random.uniform(0.1, 1),
            })

    def _parse_sample(self, sample):
        """Parse a dataset sample into (instruction, action_type, bbox, sensitivity)."""
        if 'conversation' in sample:
            instruction = ''
            for msg in sample['conversation']:
                if msg['from'] == 'human':
                    if '\nThe instruction is:\n' in msg['value']:
                        instruction = msg['value'].split('\nThe instruction is:\n')[1].split('\n\n')[0]
                    else:
                        instruction = msg['value'].replace('<image>\n', '').split('\n')[0]
            action_type = 0
            for msg in sample['conversation']:
                if msg['from'] == 'gpt':
                    val = msg['value'].lower()
                    for i, at in enumerate(self.ACTION_TYPES):
                        if at in val:
                            action_type = i
            bbox = sample.get('bbox', None) or [0.3, 0.3, 0.7, 0.7]
            if any(v > 1 for v in bbox):
                bbox = [b / 1920 if i % 2 == 0 else b / 1080 for i, b in enumerate(bbox)]
            sensitivity = np.random.uniform(0.2, 0.8)
        else:
            instruction = sample.get('instruction', 'click on element')
            action_type = sample.get('action_type', 0)
            bbox = sample.get('bbox', [0.3, 0.3, 0.7, 0.7])
            sensitivity = sample.get('sensitivity', 0.5)
        return instruction, action_type, bbox, sensitivity

    def _simulate_encoder_output(self, sample):
        """Simulate proxy encoder output E(I_t) from a frozen DINOv2-ViT."""
        instruction, action_type, bbox, sensitivity = self._parse_sample(sample)
        seed = hash(str(sample.get('id', 0))) % 0x100000000
        rng = np.random.RandomState(seed)
        d = self.proxy_encoder_dim
        latent = np.zeros(d, dtype=np.float32)
        amp = 8.0
        for i, val in enumerate(bbox):
            blk = d // 8
            start = i * blk
            latent[start:start + blk] = val * amp
            latent[start + blk // 3:start + 2 * blk // 3] += val ** 2 * amp
        act_start = d // 2
        act_blk = d // 8
        latent[act_start + action_type * act_blk:act_start + (action_type + 1) * act_blk] = amp
        tail = d * 3 // 4
        latent[tail:] += sensitivity * 2.0
        latent += rng.randn(d).astype(np.float32) * 0.3
        norm = np.linalg.norm(latent) + 1e-08
        latent = latent / norm
        return torch.from_numpy(latent).reshape(1, d).to(self.device)

    def _sample_mixed_episode(self):
        """Sample episode from mixed success/fail data. Returns (sample, is_fail)."""
        if self.fail_data and self.fail_ratio > 0:
            is_fail = np.random.random() < self.fail_ratio
            if is_fail:
                weights = np.array(self.fail_rewards) + 0.1
                probs = weights / weights.sum()
                idx = np.random.choice(len(self.fail_data), p=probs)
                return self.fail_data[idx], True
            idx = np.random.randint(len(self.data))
            return self.data[idx], False
        idx = np.random.randint(len(self.data))
        return self.data[idx], False

    def reset(self, episode_idx: Optional[int] = None) -> GUIState:
        """Reset to a new episode (supports mixed success/fail sampling)."""
        if episode_idx is not None and self.fail_ratio == 0:
            self.current_episode_idx = episode_idx % len(self.data)
            sample = self.data[self.current_episode_idx]
            is_fail_sample = False
        else:
            sample, is_fail_sample = self._sample_mixed_episode()
            self.current_episode_idx = -1
        instruction, action_type, bbox, sensitivity = self._parse_sample(sample)
        encoder_output = self._simulate_encoder_output(sample)
        self.current_step = 0
        self.episode_history = []
        true_reward = sample.get('reward', 1 if not is_fail_sample else 0)
        self.current_state = GUIState(
            image=None, encoder_output=encoder_output, instruction=instruction,
            target_bbox=torch.tensor(bbox, dtype=torch.float32, device=self.device),
            target_action=action_type, step=0, sensitivity=sensitivity,
            uncertainty=0.5, is_fail_sample=is_fail_sample, true_reward=true_reward)
        return self.current_state

    def step(self, pred_action_logits: torch.Tensor, pred_bbox: torch.Tensor,
             epsilon_used: float, k_used: int, uncertainty: float
             ) -> Tuple[GUIState, float, bool, Dict]:
        """Execute one step in the environment."""
        assert self.current_state is not None, 'Call reset() first'
        pred_action = pred_action_logits.argmax(dim=-1).item()
        action_correct = (pred_action == self.current_state.target_action)
        iou = self._compute_iou(pred_bbox, self.current_state.target_bbox)
        bbox_correct = iou > 0.5
        if self.current_state.is_fail_sample:
            task_reward = self.current_state.true_reward * (0.5 + 0.5 * iou)
            success = False
        else:
            success = action_correct and bbox_correct
            task_reward = 1.0 if success else 0.0
        self.episode_history.append({
            'step': self.current_step,
            'action_correct': action_correct,
            'bbox_iou': iou,
            'success': success,
            'is_fail_sample': self.current_state.is_fail_sample,
            'true_reward': self.current_state.true_reward if self.current_state.is_fail_sample else 1.0,
            'epsilon': epsilon_used,
            'k': k_used,
        })
        self.current_step += 1
        done = success or self.current_step >= self.max_steps or self.current_state.is_fail_sample
        if not done:
            next_sample_idx = (self.current_episode_idx + self.current_step) % len(self.data)
            next_sample = self.data[next_sample_idx]
            instruction, action_type, bbox, sensitivity = self._parse_sample(next_sample)
            encoder_output = self._simulate_encoder_output(next_sample)
            self.current_state = GUIState(
                image=None, encoder_output=encoder_output, instruction=instruction,
                target_bbox=torch.tensor(bbox, dtype=torch.float32, device=self.device),
                target_action=action_type, step=self.current_step,
                sensitivity=sensitivity, uncertainty=uncertainty)
        info = {
            'action_correct': action_correct,
            'bbox_iou': iou,
            'success': success,
            'is_fail_sample': self.current_state.is_fail_sample,
            'true_reward': self.current_state.true_reward if self.current_state.is_fail_sample else 1.0,
            'steps_taken': self.current_step,
        }
        return self.current_state, float(task_reward), done, info

    @staticmethod
    def _compute_iou(pred, target) -> float:
        """Compute Intersection over Union between two bboxes."""
        pred = pred.detach().cpu()
        target = target.detach().cpu()
        x1 = max(pred[0].item(), target[0].item())
        y1 = max(pred[1].item(), target[1].item())
        x2 = min(pred[2].item(), target[2].item())
        y2 = min(pred[3].item(), target[3].item())
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        pred_area = max(0, (pred[2] - pred[0]).item()) * max(0, (pred[3] - pred[1]).item())
        target_area = max(0, (target[2] - target[0]).item()) * max(0, (target[3] - target[1]).item())
        union = pred_area + target_area - intersection
        return intersection / max(union, 1e-08)

    def get_governance_state(self, num_regions: int) -> torch.Tensor:
        """Get initial governance state s_t = [U_t, Λ_t placeholder] (26-dim)."""
        if self.current_state is None:
            return torch.zeros(1 + num_regions)
        ut = torch.tensor([self.current_state.uncertainty], dtype=torch.float32)
        lambda_placeholder = torch.zeros(num_regions, dtype=torch.float32)
        return torch.cat([ut, lambda_placeholder], dim=0)

    @property
    def num_episodes(self) -> int:
        """Total number of episodes (success + fail data if mixed)."""
        total = len(self.data)
        if self.fail_data and self.fail_ratio > 0:
            total = int(total * (1 + self.fail_ratio))
        return total

    def get_stats(self) -> Dict:
        """Return environment statistics."""
        stats = {
            'success_samples': len(self.data),
            'fail_samples': len(self.fail_data) if self.fail_data else 0,
            'fail_ratio': self.fail_ratio,
            'mixed_episodes': self.num_episodes,
        }
        if self.fail_data:
            stats['fail_reward_mean'] = np.mean(self.fail_rewards)
            stats['fail_reward_std'] = np.std(self.fail_rewards)
        return stats
