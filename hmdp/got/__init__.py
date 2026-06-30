#!/usr/bin/env python3
"""
Graph of Thought (GoT) Module for H-MDP.

This package implements the k-path reasoning and aggregation components
of the H-MDP framework (Sec. 3.5 of the paper):

- path_generation: Generate k diverse VLM reasoning paths
- aggregation: Score-based coordinate aggregation with clustering
- scoring: Path quality evaluation utilities

Usage:
    from hmdp.got import vlm_got_k_paths, got_aggregate
    from hmdp.got.path_generation import _find_coord_span
"""

# Main entry points
from .path_generation import vlm_got_k_paths
from .aggregation import got_aggregate
from .scoring import path_quality

__all__ = [
    "vlm_got_k_paths",
    "got_aggregate",
    "path_quality",
]
