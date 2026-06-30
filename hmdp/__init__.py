"""
H-MDP: Hierarchical Joint Control of Perception Budget and Reasoning Depth
for Agentic GUI Systems — real-VLM (Blind-but-Smart) implementation.

The simulation path (SAC meta-policy, GoT/LTM orchestration engine, synthetic
GUI environments, and their training/plotting entrypoints) has been removed.
The package now contains only the paper-faithful real-VLM pipeline.

Modules:
    - ldp:            Local Differential Privacy (ε-LDP) + frozen ProxyEncoder (DINOv2)
    - ltm:            Long-Term Memory module
    - execution_engine: ProjectionLayer W_proj (latent → VLM token space)
    - vlm_inference:  Real VLM backbones, action parsing, GoT k-path sampling
    - blind_vlm:      Blind-but-Smart pipeline (privatized latent injection via W_proj)
    - run_real_vlm:   Real-VLM evaluation entrypoint (default = blind; --raw-pixel ablation)
    - run_all_tables: Multi-configuration real-VLM table generation
"""

from .ldp import ProxyEncoder, LocalDifferentialPrivacy
from .ltm import LongTermMemory
from .execution_engine import ProjectionLayer

__version__ = "2.0.0"
