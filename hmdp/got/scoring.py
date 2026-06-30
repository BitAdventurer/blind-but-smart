#!/usr/bin/env python3
"""
GoT Path Scoring: Quality evaluation for reasoning paths.

Utility functions for assessing the reliability of generated paths,
used in filtering and weighted aggregation.
"""

from typing import Dict


def path_quality(path: Dict) -> float:
    """
    Score a path's quality based on parse reliability.

    Scoring criteria:
      - 1.0: Valid coordinate range [0, 1]
      - 0.3: Not near screen edge (avoids boundary parse errors)
      - 0.1: Non-default action type (explicit parsing)

    Returns a score in [0, 1.4] where higher is better.
    """
    score = 0.0
    pt = path.get("pred_point")
    if pt is None:
        return 0.0

    # Valid coordinate range [0, 1]
    if 0.0 <= pt[0] <= 1.0 and 0.0 <= pt[1] <= 1.0:
        score += 1.0
    else:
        score += 0.2

    # Penalize extreme edges (often parse errors)
    if 0.02 < pt[0] < 0.98 and 0.02 < pt[1] < 0.98:
        score += 0.3

    # Action type parsed successfully
    if path.get("action_type", "click") != "click":
        score += 0.1  # non-default action = explicit parse

    return score


def path_confidence(path: Dict) -> float:
    """
    Extract confidence metric from a path.

    Returns the logit (mean token logprob) if available, otherwise 0.
    """
    logit = path.get("logit")
    return float(logit) if logit is not None else 0.0


def filter_low_quality_paths(paths: list, threshold: float = 0.5) -> list:
    """
    Filter paths below a quality threshold.

    Useful for discarding obvious parsing failures before aggregation.
    """
    return [p for p in paths if path_quality(p) >= threshold]
