"""Release-only relevance and deterministic sampled-output aggregation.

These routines select a weighted medoid from sampled outputs. They never receive
clean image features, a probe, or a reference target. Scoring uses float64.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Mapping, Sequence

import numpy as np


N_REGIONS = 25
LATENT_DIM = 256
NORM_FLOOR = 1e-12
LOSS_TOLERANCE = 1e-12


def _finite_array(value, shape=None) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if shape is not None and value.shape != shape:
        raise ValueError(f"Expected shape {shape}, received {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("Scoring inputs must be finite")
    return value


def unit(value) -> np.ndarray:
    value = _finite_array(value)
    return value / max(float(np.linalg.norm(value)), NORM_FLOOR)


def cell_index(coordinate: Sequence[float]) -> int:
    """Zero-based row-major cell, including closed right/bottom boundaries."""
    x, y = _finite_array(coordinate, (2,))
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise ValueError("Cell lookup requires validated normalized coordinates")
    return 5 * min(int(5 * y), 4) + min(int(5 * x), 4)


def score_relevance(release, projection, instruction_embeddings,
                    spatial_keys: Sequence[Sequence[float]] = ()) -> float:
    """Compute signed raw r, without clipping, shifting, or a sigmoid.

    ``instruction_embeddings`` is the frozen input-table embedding of each
    non-special token in the post-truncation scoring span, in original order.
    Repeated spatial keys count separately; a nonspatial output averages 25
    cells. A zero vector paired with a unit vector scores .5; two zeros score 1.
    """
    release = _finite_array(release, (N_REGIONS, LATENT_DIM))
    projection = _finite_array(projection)
    if projection.ndim != 2 or projection.shape[1] != LATENT_DIM:
        raise ValueError("Projection must have shape (hidden_size, 256)")
    embeddings = _finite_array(instruction_embeddings)
    if embeddings.ndim != 2 or embeddings.shape[0] == 0 or embeddings.shape[1] != projection.shape[0]:
        raise ValueError("A nonempty (tokens, hidden_size) scoring span is required")
    indices = [cell_index(c) for c in spatial_keys]
    latent = release[indices].mean(axis=0, dtype=np.float64) if indices else release.mean(axis=0, dtype=np.float64)
    visual = unit(projection @ latent)
    language = unit(embeddings.mean(axis=0, dtype=np.float64))
    return 1.0 - float(np.square(visual - language).sum(dtype=np.float64)) / 2.0


@dataclass(frozen=True)
class GroundingCandidate:
    index: int
    coordinate: tuple[float, float] | None
    valid: bool
    logprob_mean: float
    relevance: float
    text: str = ""
    token_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class GroundingResult:
    selected_index: int
    coordinate: tuple[float, float] | None
    feedback: float
    weights: tuple[tuple[int, float], ...]


def _validate_candidates(candidates):
    indices = [c.index for c in candidates]
    if len(set(indices)) != len(indices) or any(i < 0 for i in indices):
        raise ValueError("Candidate indices must be unique and nonnegative")
    for candidate in candidates:
        if candidate.valid:
            if not math.isfinite(candidate.logprob_mean) or candidate.logprob_mean > 1e-12:
                raise ValueError("Valid candidate log probability must be finite and nonpositive")
            if not math.isfinite(candidate.relevance):
                raise ValueError("Valid relevance must be finite")


def aggregate_grounding(candidates: Sequence[GroundingCandidate], rule: str = "full") -> GroundingResult:
    if rule not in {"full", "no_relevance", "uniform"}:
        raise ValueError(f"Unknown selection rule: {rule}")
    if not 1 <= len(candidates) <= 20:
        raise ValueError("An invocation must contain 1 through 20 candidates, including invalid ones")
    _validate_candidates(candidates)
    valid = sorted((c for c in candidates if c.valid), key=lambda c: c.index)
    for c in valid:
        cell_index(c.coordinate)
    if not valid:
        return GroundingResult(-1, None, 2.0, ())
    weights = [math.exp(c.logprob_mean) * c.relevance ** 2 if rule == "full"
               else math.exp(c.logprob_mean) if rule == "no_relevance"
               else 1.0 / len(valid) for c in valid]
    best, best_cost, best_weight = None, None, None
    for i, ci in enumerate(valid):
        # Ordered Python sum and absolute tolerance deliberately match replay.
        cost = sum(w * ((ci.coordinate[0] - cj.coordinate[0]) ** 2
                        + (ci.coordinate[1] - cj.coordinate[1]) ** 2)
                   for w, cj in zip(weights, valid))
        if (best is None or cost < best_cost - LOSS_TOLERANCE
                or (abs(cost - best_cost) <= LOSS_TOLERANCE
                    and (weights[i], -ci.index) > (best_weight, -best.index))):
            best, best_cost, best_weight = ci, cost, weights[i]
    dispersion = sum((c.coordinate[0] - best.coordinate[0]) ** 2
                     + (c.coordinate[1] - best.coordinate[1]) ** 2 for c in valid)
    feedback = (dispersion + 2 * (len(candidates) - len(valid))) / len(candidates)
    return GroundingResult(best.index, best.coordinate, feedback,
                           tuple((c.index, w) for c, w in zip(valid, weights)))


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def strict_json(text: str):
    def reject(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=reject)


def parse_grounding(text: str, *, terminated: bool = True) -> tuple[float, float] | None:
    """Reference decoder format: exactly {\"x\": number, \"y\": number}.

    This strict normalized-JSON output format is a new implementation choice;
    it is not a reconstruction of an unavailable original checkpoint parser.
    Finite coordinates are clipped to the closed screen rectangle.
    """
    if not terminated:
        return None
    try:
        obj = strict_json(text)
        if not isinstance(obj, dict) or set(obj) != {"x", "y"}:
            return None
        if any(isinstance(obj[k], bool) or not isinstance(obj[k], (float, int)) for k in ("x", "y")):
            return None
        coords = _finite_array([obj["x"], obj["y"]], (2,))
        return tuple(float(v) for v in np.clip(coords, 0.0, 1.0))
    except (ValueError, TypeError, OverflowError):
        return None


@dataclass(frozen=True)
class ActionFunctionSchema:
    required: frozenset[str]
    optional: frozenset[str]
    spatial: frozenset[str]
    statuses: frozenset[str]
    argument_types: Mapping[str, str] | None = None
    canonical_function: str | None = None
    status_aliases: Mapping[str, str] | None = None


@dataclass(frozen=True)
class ActionCandidate:
    index: int
    function: str
    arguments: Mapping
    status: str
    valid: bool
    logprob_mean: float
    relevance: float
    text: str = ""
    token_ids: tuple[int, ...] = ()


def parse_action(text: str, schemas: Mapping[str, ActionFunctionSchema], *, terminated=True):
    """Validate a caller-pinned action vocabulary; never infer function aliases."""
    if not terminated:
        return None
    try:
        value = strict_json(text)
        if not isinstance(value, dict) or set(value) != {"function", "arguments", "status"}:
            return None
        if not isinstance(value["function"], str) or not isinstance(value["status"], str):
            return None
        spec = schemas.get(value["function"])
        args = value["arguments"]
        if spec is None or not isinstance(args, dict):
            return None
        value["function"] = spec.canonical_function or value["function"]
        value["status"] = (spec.status_aliases or {}).get(value["status"], value["status"])
        if value["status"] not in spec.statuses:
            return None
        if not spec.required <= set(args) or not set(args) <= spec.required | spec.optional:
            return None
        for key in set(args) & spec.spatial:
            if (not isinstance(args[key], list) or len(args[key]) != 2
                    or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in args[key])):
                return None
            coord = _finite_array(args[key], (2,))
            if (coord < -1e-6).any() or (coord > 1 + 1e-6).any():
                return None
            args[key] = np.clip(coord, 0.0, 1.0).tolist()
        types = {"string": lambda x: isinstance(x, str),
                 "integer": lambda x: isinstance(x, int) and not isinstance(x, bool),
                 "number": lambda x: isinstance(x, (int, float)) and not isinstance(x, bool),
                 "boolean": lambda x: isinstance(x, bool),
                 "array": lambda x: isinstance(x, list), "object": lambda x: isinstance(x, dict),
                 "null": lambda x: x is None}
        for key, kind in (spec.argument_types or {}).items():
            if kind not in types or (key in args and not types[kind](args[key])):
                return None
        # No semantic repair or Unicode normalization of candidate arguments.
        json.dumps(args, ensure_ascii=False, allow_nan=False).encode("utf-8", errors="strict")
        return value
    except (ValueError, TypeError, KeyError, OverflowError, UnicodeError):
        return None


def aggregate_action(candidates: Sequence[ActionCandidate], schemas: Mapping[str, ActionFunctionSchema]):
    """Subset plurality, sampled Euclidean medoids, and missing-support feedback."""
    if not 1 <= len(candidates) <= 20:
        raise ValueError("An invocation must contain 1 through 20 candidates")
    _validate_candidates(candidates)
    valid = sorted((c for c in candidates if c.valid), key=lambda c: c.index)
    invalid = ({"function": "INVALID", "arguments": {}, "status": "INVALID"}, 2.0)
    if not valid:
        return invalid
    for candidate in valid:
        payload = {"function": candidate.function, "arguments": dict(candidate.arguments), "status": candidate.status}
        if parse_action(json.dumps(payload, allow_nan=False), schemas) is None:
            raise ValueError("A candidate marked valid violates its action schema")
    omega = {c.index: (math.exp(c.logprob_mean) + c.relevance) / 2.0 for c in valid}

    def plural(population, key):
        groups = {}
        for c in population:
            groups.setdefault(key(c), []).append(c)
        def rank(group):
            return (sum(math.exp(omega[c.index]) for c in group),
                    sum(omega[c.index] for c in group), -min(c.index for c in group))
        return max(groups.items(), key=lambda item: rank(item[1]))

    function, population = plural(valid, lambda c: c.function)
    status, population = plural(population, lambda c: c.status)
    spec = schemas[function]
    # The complete nonspatial object and optional-key presence are one vote.
    def symbolic_key(c):
        return json.dumps({"symbolic": {k: v for k, v in c.arguments.items() if k not in spec.spatial},
                           "present": sorted(set(c.arguments) & spec.optional)},
                          ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    encoded, _ = plural(population, symbolic_key)
    symbolic = json.loads(encoded)
    arguments = symbolic["symbolic"]
    spatial_keys = (spec.required | set(symbolic["present"])) & spec.spatial
    feedback_sum = 0.0
    for key in sorted(spatial_keys):
        support = [c for c in population if key in c.arguments]
        if not support:
            return invalid
        norm = sum(math.exp(omega[c.index]) for c in support)
        best = min(support, key=lambda c: (
            sum(math.exp(omega[j.index]) / norm * math.dist(c.arguments[key], j.arguments[key]) for j in support),
            -omega[c.index], c.index))
        arguments[key] = list(best.arguments[key])
        feedback_sum += sum(math.dist(c.arguments[key], arguments[key]) ** 2 for c in support) + 2 * (len(candidates) - len(support))
    output = {"function": function, "arguments": arguments, "status": status}
    if parse_action(json.dumps(output), schemas) is None:
        return invalid
    feedback = (feedback_sum / (len(candidates) * len(spatial_keys)) if spatial_keys
                else 2 * (len(candidates) - len(population)) / len(candidates))
    return output, feedback
