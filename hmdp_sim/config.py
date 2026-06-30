"""
Master configuration for the H-MDP framework.
"""
import torch
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
from pathlib import Path


@dataclass
class HMDPConfig:
    """Master configuration for the H-MDP framework."""

    # ---- Device / proxy encoder ----
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    proxy_encoder: str = 'facebook/dinov2-large'
    proxy_encoder_dim: int = 1024
    latent_dim: int = 256
    freeze_proxy: bool = True

    # ---- Screen partitioning ----
    num_regions: int = 25  # 5x5 grid (M)

    # ---- LDP / privacy budget ----
    epsilon_min: float = 0.1
    epsilon_max: float = 5.0
    epsilon_levels: List[float] = field(default_factory=lambda: [0.1, 0.5, 1.0, 2.5, 5.0])
    epsilon_default: float = 1.0
    ldp_sensitivity: float = 2.0

    # ---- GoT reasoning depth k ----
    k_min: int = 1
    k_max: int = 20
    k_levels: List[int] = field(default_factory=lambda: [1, 5, 10, 15, 20])
    k_default: int = 5

    # ---- GoT aggregation ----
    got_aggregation: str = 'softmax_weighted'
    got_score_alpha: float = 0.5
    got_temperature: float = 1.0

    # ---- Long-Term Memory ----
    ltm_capacity: int = 10000
    ltm_embedding_dim: int = 256
    ltm_top_k: int = 8
    ltm_uncertainty_threshold: float = 0.15

    # ---- SAC meta-policy ----
    sac_state_dim: int = 26
    sac_action_dim: int = 26
    sac_hidden_dim: int = 256
    sac_lr_actor: float = 3e-4
    sac_lr_critic: float = 3e-4
    sac_lr_alpha: float = 3e-4
    sac_gamma: float = 0.99
    sac_tau: float = 0.005
    sac_alpha_init: float = 0.2
    sac_auto_alpha: bool = True
    replay_buffer_size: int = 100000
    batch_size: int = 256
    warmup_steps: int = 1000

    # ---- Reward weights (Eq. 2 + terminal PES bonus Eq. 4) ----
    w_perf: float = 1.0
    w_priv: float = 0.5
    w_comp: float = 0.1
    w_pes: float = 2.0

    # ---- Training schedule ----
    num_episodes: int = 5000
    max_steps_per_episode: int = 15
    eval_interval: int = 100
    save_interval: int = 500
    log_dir: str = 'logs/hmdp'
    checkpoint_dir: str = 'checkpoints/hmdp'

    # ---- VLM backbone ----
    vlm_backbone: str = 'Qwen/Qwen2.5-VL-7B-Instruct'
    vlm_dtype: str = 'bfloat16'

    # ---- GUI-360 dataset paths ----
    # Default paths assume project structure with data at project root
    _project_root: str = field(default='', repr=False)
    gui360_data_path: str = field(default='', repr=True)
    gui360_image_base: str = field(default='', repr=True)
    gui360_fail_data_path: str = field(default='', repr=True)

    def __post_init__(self):
        """Initialize paths with proper defaults if not set."""
        import os
        # Use environment variable if set; otherwise infer the code root from
        # this file location (hmdp_sim/config.py -> project root).
        project_root = os.environ.get('HMDP_PROJECT_ROOT') or self._project_root
        if not project_root:
            project_root = str(Path(__file__).resolve().parents[1])

        if not self.gui360_data_path:
            self.gui360_data_path = os.path.join(
                project_root,
                'gui360_full/processed_data/action_prediction_train_resize/training_data.json'
            )
        if not self.gui360_image_base:
            self.gui360_image_base = os.path.join(
                project_root,
                'gui360_full/processed_data/action_prediction_train_resize'
            )
        if not self.gui360_fail_data_path:
            self.gui360_fail_data_path = os.path.join(
                project_root,
                'gui360_full/converted_fail_data.json'
            )
    fail_data_ratio: float = 0.3
    image_size: Tuple[int, int] = (1920, 1080)

    # ------------------------------------------------------------------ #
    # Discrete-level helpers
    # ------------------------------------------------------------------ #
    def get_epsilon(self, index):
        return self.epsilon_levels[min(index, len(self.epsilon_levels) - 1)]

    def get_k(self, index):
        return self.k_levels[min(index, len(self.k_levels) - 1)]

    # ------------------------------------------------------------------ #
    # Continuous action <-> physical params
    # ------------------------------------------------------------------ #
    def action_to_params(self, action_vec):
        """
        Map a continuous SAC action a in [-1, 1]^26 to physical params.

          a               = clip(action_vec, -1, 1)
          eps_raw         = a[:num_regions]                      (first 25 slots)
          epsilons^(i)    = epsilon_min + (eps_raw+1)/2 * (epsilon_max - epsilon_min)
          k_hat           = k_min + (a[num_regions]+1)/2 * (k_max - k_min)   (26th slot)
          k               = int(clip(round(k_hat), k_min, k_max))

        Returns:
            (epsilons: List[float] (len M), k: int)
        """
        import numpy as np
        a = np.asarray(action_vec, dtype=np.float64).reshape(-1)
        a = np.clip(a, -1.0, 1.0)
        eps_raw = a[:self.num_regions]
        epsilons = (self.epsilon_min + (eps_raw + 1.0) / 2.0 * (self.epsilon_max - self.epsilon_min)).tolist()
        k_hat = self.k_min + (a[self.num_regions] + 1.0) / 2.0 * (self.k_max - self.k_min)
        k = int(np.clip(round(k_hat), self.k_min, self.k_max))
        return epsilons, k

    def params_to_action(self, epsilons, k):
        """
        Inverse map: physical params -> continuous action a in [-1, 1]^26.
        """
        import numpy as np
        eps = np.asarray(epsilons, dtype=np.float64)
        a_eps = 2.0 * (eps - self.epsilon_min) / (self.epsilon_max - self.epsilon_min) - 1.0
        a_k = 2.0 * (float(k) - self.k_min) / (self.k_max - self.k_min) - 1.0
        a = np.concatenate([a_eps, [a_k]])
        return np.clip(a, -1.0, 1.0).astype(np.float32)
