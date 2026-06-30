#!/usr/bin/env python3
"""
Centralized constants for the H-MDP framework.

This module provides frozen dataclasses for all magic numbers used across
the codebase, ensuring consistency and maintainability.

Usage:
    from hmdp.constants import GridConfig, ModelDims, GoTHyperparams

    grid_size = GridConfig.SIZE  # 5
    latent_dim = ModelDims.LATENT_DEFAULT  # 256
"""

from dataclasses import dataclass
from typing import Tuple


# ═══════════════════════════════════════════════════════════════════════
#  Model Architecture Dimensions
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ModelDims:
    """Neural network architecture dimensions."""

    # Proxy encoder (DINOv2)
    DINOV2_OUTPUT: int = 1024  # ViT-L/14 CLS token dimension

    # Latent feature space (Eq. 3: phi(s_t))
    LATENT_DEFAULT: int = 256  # Default latent dimension d

    # Projection layer
    PROJ_HIDDEN: int = 2048  # Hidden dim for W_proj MLP

    # VLM backbone hidden dimensions
    QWEN2_5_VL_7B: int = 3584  # Qwen2.5-VL-7B-Instruct
    QWEN3_VL_8B: int = 3584  # Qwen3-VL-8B-Instruct (Qwen3 architecture)
    INTERNVL2_5_8B: int = 4096  # InternVL2.5-8B
    INTERNVL3_5_8B: int = 4096  # InternVL3.5-8B
    UGROUND_7B: int = 3584  # UGround-V1-7B (Qwen2-based)
    AGUVIS_7B: int = 3584  # Aguvis-7B (Qwen2-based)
    GUI_ACTOR_7B: int = 3584  # GUI-Actor-7B (Qwen2.5-based)
    UI_TARS_1_5_7B: int = 3584  # UI-TARS-1.5-7B (Qwen2-based)

    @classmethod
    def get_vlm_dim(cls, model_key: str) -> int:
        """Get hidden dimension for a VLM backbone."""
        mapping = {
            "qwen2.5-vl-7b": cls.QWEN2_5_VL_7B,
            "qwen2.5-vl-3b": 3072,  # Smaller variant
            "qwen3-vl-8b": cls.QWEN3_VL_8B,
            "internvl3.5-8b": cls.INTERNVL3_5_8B,
            "uground-7b": cls.UGROUND_7B,
            "aguvis-7b": cls.AGUVIS_7B,
            "gui-actor-7b": cls.GUI_ACTOR_7B,
            "ui-tars-1.5-7b": cls.UI_TARS_1_5_7B,
            "internvl2.5-8b": cls.INTERNVL2_5_8B,
        }
        return mapping.get(model_key, cls.QWEN2_5_VL_7B)  # Default to Qwen


# ═══════════════════════════════════════════════════════════════════════
#  Screen Partitioning (Grid)
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class GridConfig:
    """Screen partitioning configuration (Sec. 3.3)."""

    SIZE: int = 5  # Grid size (5x5)
    NUM_REGIONS: int = 25  # SIZE * SIZE

    @classmethod
    def get_num_regions(cls, grid_size: int) -> int:
        """Calculate number of regions for a given grid size."""
        return grid_size * grid_size


# ═══════════════════════════════════════════════════════════════════════
#  Local Differential Privacy (LDP)
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class LDPConfig:
    """LDP noise and privacy budget configuration (Sec. 3.3, Eq. 4)."""

    # Sensitivity (Delta_2 in Eq. 4)
    # L2-clipped latents give Delta_2 <= 2 regardless of dimension
    SENSITIVITY: float = 2.0

    # Default epsilon (privacy budget per region)
    EPSILON_DEFAULT: float = 1.0
    EPSILON_MIN: float = 0.1
    EPSILON_MAX: float = 5.0

    # Epsilon levels for discrete experiments
    EPSILON_LEVELS: Tuple[float, ...] = (0.1, 0.5, 1.0, 2.5, 5.0)

    # Delta for (eps, delta)-LDP Gaussian mechanism
    DELTA_DEFAULT: float = 1e-5


# ═══════════════════════════════════════════════════════════════════════
#  Graph of Thought (GoT) Aggregation
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class GoTHyperparams:
    """GoT reasoning and aggregation hyperparameters (Sec. 3.5)."""

    # Aggregation weights (Eq. got_score)
    ALPHA_W: float = 0.5  # Weight for value alignment term (logit vs value)

    # Temperature parameters
    T_AGG: float = 1.0  # Softmax temperature for path aggregation
    T_SAMPLE: float = 0.7  # Temperature for coordinate sampling
    T_SAMPLE_BLIND: float = 0.5  # Temperature for blind VLM (more conservative)

    # LTM gating
    TAU_LTM: float = 0.15  # Uncertainty threshold for LTM query (Eq. LTM gate)

    # Path clustering
    CLUSTER_RADIUS: float = 0.08  # Normalized coordinate distance for clustering

    # Reasoning depth
    K_DEFAULT: int = 5  # Default number of reasoning paths
    K_MIN: int = 1
    K_MAX: int = 20
    K_LEVELS: Tuple[int, ...] = (1, 5, 10, 15, 20)

    # Decoding
    TOP_P: float = 0.9  # Nucleus sampling parameter
    TOP_P_GREEDY: float = 1.0  # Greedy decoding (no nucleus truncation)


# ═══════════════════════════════════════════════════════════════════════
#  Long-Term Memory (LTM)
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class LTMConfig:
    """LTM storage and retrieval configuration (Sec. 3.5)."""

    CAPACITY: int = 10000  # Maximum stored episodes
    CAPACITY_EVAL: int = 500  # Smaller capacity for evaluation
    TOP_K: int = 8  # Number of episodes to retrieve (kLTM)
    EMBEDDING_DIM: int = 256  # Must match ModelDims.LATENT_DEFAULT


# ═══════════════════════════════════════════════════════════════════════
#  Reward Computation (Eq. 2)
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RewardConfig:
    """Reward function weights and thresholds (Sec. 3.6, Eq. 2)."""

    # Success reward components
    FULL_SUCCESS: float = 1.0  # Action correct AND grounding correct
    PARTIAL_SUCCESS: float = 0.5  # Action correct but grounding close
    FAILURE: float = 0.0

    # Distance threshold for partial reward (normalized coordinates)
    DISTANCE_THRESHOLD: float = 0.3

    # Reward weights (Eq. 2 dense + Eq. 4 terminal PES bonus)
    W_PERF: float = 1.0  # Performance (task-quality) weight
    W_PRIV: float = 0.5  # Privacy log-budget weight
    W_COMP: float = 0.1  # Per-step computational cost
    W_PES: float = 2.0   # Terminal PES-aligned bonus weight

    # Action parsing bonus
    NON_DEFAULT_ACTION_BONUS: float = 0.1  # Explicit action type parsing


# ═══════════════════════════════════════════════════════════════════════
#  SAC Meta-Policy (Sec. 3.6)
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SACConfig:
    """SAC training hyperparameters."""

    # Network architecture
    STATE_DIM: int = 26  # 1 uncertainty + 25 region entropies
    ACTION_DIM: int = 26  # 25 epsilon values + 1 k value
    HIDDEN_DIM: int = 256

    # Learning rates
    LR_ACTOR: float = 3e-4
    LR_CRITIC: float = 3e-4
    LR_ALPHA: float = 3e-4

    # RL hyperparameters
    GAMMA: float = 0.99  # Discount factor
    TAU: float = 0.005  # Target network soft update rate
    ALPHA_INIT: float = 0.2  # Initial entropy temperature
    AUTO_ALPHA: bool = True  # Automatic entropy tuning

    # Training schedule
    WARMUP_STEPS: int = 1000
    BATCH_SIZE: int = 256
    REPLAY_BUFFER_SIZE: int = 100000


# ═══════════════════════════════════════════════════════════════════════
#  Training & Evaluation
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TrainConfig:
    """Training and evaluation settings."""

    # Projection training (Sec. 4.1.4)
    ALIGN_EPOCHS: int = 2  # Stage 1: Feature alignment
    TASK_EPOCHS: int = 3  # Stage 2: Task-supervised fine-tuning
    PROJ_LR: float = 1e-4

    # Default sample sizes
    NUM_TRAIN_SAMPLES: int = 5000
    NUM_EVAL_SAMPLES: int = 100

    # Image dimensions (normalized)
    IMAGE_WIDTH: int = 1920
    IMAGE_HEIGHT: int = 1080


# ═══════════════════════════════════════════════════════════════════════
#  Convenience Exports
# ═══════════════════════════════════════════════════════════════════════

# Single import for most common constants
DEFAULT_GRID_SIZE = GridConfig.SIZE
DEFAULT_NUM_REGIONS = GridConfig.NUM_REGIONS
DEFAULT_LATENT_DIM = ModelDims.LATENT_DEFAULT
DEFAULT_EPSILON = LDPConfig.EPSILON_DEFAULT
DEFAULT_K = GoTHyperparams.K_DEFAULT
