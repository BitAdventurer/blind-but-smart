"""Trusted-side, two-stage feature-neighborhood release for a new implementation.

This module implements the manuscript's mathematical mechanism, not recovered
original experiment code.  Sensitivity 0.01 describes a feature neighborhood;
unit clipping does not make 0.01 a bound for arbitrary pairs of screens.

The default NumPy streams have independent OS-entropy seeds, but NumPy's normal
sampler is not a cryptographic or finite-precision DP implementation.  Treat this
as research software, not an audited production privacy boundary.  Never publish
private RNG state or reuse private streams across exposed runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
import secrets
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import log_ndtr


REGIONS = 25
FEATURE_DIM = 256
MAX_SLOTS = 56
DELTA_STAGE = 1e-5 / (2 * REGIONS * MAX_SLOTS)
NEIGHBORHOOD_RADIUS = 0.01
PROBE_EPSILON = 1.0
MIN_TOTAL_EPSILON = 1.5
MAX_TOTAL_EPSILON = 5.0
MIN_INVOCATION_COST = REGIONS * MIN_TOTAL_EPSILON

EXEC_PROP = "EXEC_PROP"
EXEC_FALLBACK = "EXEC_FALLBACK"
STRUCTURAL_PAD = "STRUCTURAL_PAD"
TASK_PAD = "TASK_PAD"
FILTER_EXHAUSTED = "FILTER_EXHAUSTED"

FloatArray = NDArray[np.float64]


def _finite_scalar(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite scalar")
    return result


def _log_gaussian_delta(epsilon: float, log_ratio: float) -> float:
    """Log hockey-stick divergence with ratio=sigma/sensitivity.

    Evaluating exp(epsilon) * Phi(...) directly overflows or loses the small
    positive difference.  log_ndtr and expm1 retain that difference in the tail.
    """
    ratio = math.exp(log_ratio)
    a = 0.5 / ratio
    b = epsilon * ratio
    log_first = float(log_ndtr(a - b))
    log_second = epsilon + float(log_ndtr(-a - b))
    difference = log_second - log_first
    if difference >= 0.0:
        # The mathematical difference is positive.  Equality here means it is
        # below the floating-point resolution of these logarithms.
        return -math.inf
    return log_first + math.log(-math.expm1(difference))


@lru_cache(maxsize=8192)
def analytic_gaussian_sigma(
    epsilon: float,
    delta: float = DELTA_STAGE,
    sensitivity: float = NEIGHBORHOOD_RADIUS,
) -> float:
    """Smallest analytic-Gaussian coordinate scale, up to float64 tolerance.

    Solves Phi(s/(2 sigma)-eps*sigma/s) - exp(eps) *
    Phi(-s/(2 sigma)-eps*sigma/s) = delta in log(sigma/s) space.
    Epsilon may be zero; delta and sensitivity must be strictly positive.
    """
    epsilon = _finite_scalar(epsilon, "epsilon")
    delta = _finite_scalar(delta, "delta")
    sensitivity = _finite_scalar(sensitivity, "sensitivity")
    if epsilon < 0 or not 0 < delta < 1 or sensitivity <= 0:
        raise ValueError("require epsilon >= 0, 0 < delta < 1, sensitivity > 0")
    log_target = math.log(delta)

    def residual(log_ratio: float) -> float:
        return _log_gaussian_delta(epsilon, log_ratio) - log_target

    lower, upper = -8.0, 8.0
    while residual(lower) <= 0:
        lower -= 8.0
        if lower < -300:
            raise ValueError("calibration is outside supported float64 range")
    while residual(upper) > 0:
        upper += 8.0
        if upper > 300:
            raise ValueError("calibration is outside supported float64 range")
    root = brentq(residual, lower, upper, xtol=1e-13, rtol=1e-14)
    # A tiny conservative step avoids under-calibration due to solver rounding.
    sigma = sensitivity * math.exp(root + 2e-13)
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("calibrated scale is outside supported float64 range")
    return sigma


def _features(value: object, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (REGIONS, FEATURE_DIM):
        raise ValueError(f"{name} must have shape ({REGIONS}, {FEATURE_DIM})")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def clip_features(features: object) -> FloatArray:
    """Return an independent float64 array with every regional norm <= 1.

    Scaling before the norm also handles very large finite feature magnitudes.
    Zero and subunit rows are preserved; this is clipping, not unit normalization.
    """
    values = _features(features, "features")
    max_abs = np.max(np.abs(values), axis=1, keepdims=True)
    scale = np.maximum(max_abs, 1.0)
    scaled = values / scale
    norms = np.linalg.norm(scaled, axis=1, keepdims=True)
    return scaled / np.maximum(norms, 1.0 / scale)


def probe_entropy(probe: object) -> FloatArray:
    """Normalized magnitude entropy; an all-zero regional vector gives zero."""
    values = _features(probe, "probe")
    magnitude = np.abs(values)
    row_max = magnitude.max(axis=1, keepdims=True)
    scaled = np.divide(magnitude, row_max, out=np.zeros_like(magnitude), where=row_max > 0)
    total = scaled.sum(axis=1, keepdims=True)
    probabilities = np.divide(scaled, total, out=np.zeros_like(scaled), where=total > 0)
    logs = np.zeros_like(probabilities)
    np.log(probabilities, out=logs, where=probabilities > 0)
    entropy = -(probabilities * logs).sum(axis=1) / math.log(FEATURE_DIM)
    return np.clip(entropy, 0.0, 1.0)


def _budget_vector(proposed: object) -> FloatArray:
    values = np.asarray(proposed, dtype=np.float64)
    if values.shape != (REGIONS,) or not np.isfinite(values).all():
        raise ValueError(f"proposed budgets must be a finite ({REGIONS},) vector")
    if np.any(values < MIN_TOTAL_EPSILON) or np.any(values > MAX_TOTAL_EPSILON):
        raise ValueError("proposed regional total budgets must lie in [1.5, 5.0]")
    return values


def filter_budgets(proposed: object, remaining: float) -> tuple[FloatArray | None, str]:
    """Execute a fitting proposal, otherwise a fixed minimum vector, or pad.

    Invalid proposals are rejected, never silently clipped.  The minimum-vector
    fallback does not rescale the rejected proposal.  Budget comparisons use
    float64 totals without a positive overspend tolerance.
    """
    values = _budget_vector(proposed)
    remaining = _finite_scalar(remaining, "remaining")
    if remaining < 0:
        raise ValueError("remaining budget must be nonnegative")
    if remaining < MIN_INVOCATION_COST:
        return None, FILTER_EXHAUSTED
    if math.fsum(values.tolist()) <= remaining:
        return values.copy(), EXEC_PROP
    return np.full(REGIONS, MIN_TOTAL_EPSILON, dtype=np.float64), EXEC_FALLBACK


def protected_observation(
    probe: object,
    prior_feedback: float,
    budget_fraction: float,
    slot_index: int,
) -> FloatArray:
    """Build the allocator's 28 inputs inside the trusted runtime.

    slot_index is zero-based; the last coordinate is t/56 for t=1,...,56.
    Despite the function name, this probe-derived vector is not an exported
    transcript field.  Prior feedback must come only from protected outputs.
    """
    prior_feedback = _finite_scalar(prior_feedback, "prior_feedback")
    budget_fraction = _finite_scalar(budget_fraction, "budget_fraction")
    if not 0 <= prior_feedback <= 2 or not 0 <= budget_fraction <= 1:
        raise ValueError("require prior_feedback in [0,2] and budget_fraction in [0,1]")
    if isinstance(slot_index, (bool, np.bool_)) or not isinstance(slot_index, (int, np.integer)):
        raise ValueError("slot_index must be an integer in [0,55]")
    if not 0 <= slot_index < MAX_SLOTS:
        raise ValueError("slot_index must be an integer in [0,55]")
    return np.concatenate((probe_entropy(probe), [prior_feedback, budget_fraction, (slot_index + 1) / MAX_SLOTS]))


def _readonly(array: FloatArray | None) -> FloatArray | None:
    if array is not None:
        array.setflags(write=False)
    return array


@dataclass(frozen=True)
class StepRelease:
    """Public record: no probe, clean features, rejected proposal, or RNG state."""

    slot_index: int
    status: str
    invoked: bool
    executed_budgets: FloatArray | None
    k: int
    release: FloatArray | None
    refinement_scales: FloatArray | None
    used_budget: float
    remaining_budget: float


def _mask(value: Sequence[bool], name: str) -> NDArray[np.bool_]:
    array = np.asarray(value)
    if array.ndim != 1 or len(array) > MAX_SLOTS:
        raise ValueError(f"{name} must be a one-dimensional mask of at most 56 slots")
    if not np.isin(array, [False, True]).all():
        raise ValueError(f"{name} must contain only boolean or binary values")
    return array.astype(np.bool_, copy=True)


class PrivacyLedger:
    """Fixed-mask 56-slot release mechanism with admission before screen access.

    eligible_mask describes recorded task-eligible slots.  recorded_mask defaults
    to presence for every supplied slot.  Both are padded to 56 structural slots
    and frozen.  cap = 75 * eligible_count.  Call release_step exactly 56 times.

    The allocation callback is trusted-side and sees only the 28-dimensional
    observation.  It returns (regional_total_budgets, candidate_count).  Decoder,
    controller, probe and refinement random streams must be independent.  Test
    callers may inject two *distinct* private Generators; do not publish seeds.

    Exceptions after admission invalidate this ledger: retrying after a partial
    private computation is prohibited.  An invalid-input exception is not a
    supported published transcript; validation belongs before a production run.
    """

    def __init__(
        self,
        eligible_mask: Sequence[bool],
        recorded_mask: Sequence[bool] | None = None,
        *,
        probe_rng: np.random.Generator | None = None,
        refinement_rng: np.random.Generator | None = None,
    ) -> None:
        eligible = _mask(eligible_mask, "eligible_mask")
        recorded = np.ones(len(eligible), dtype=bool) if recorded_mask is None else _mask(recorded_mask, "recorded_mask")
        if len(eligible) != len(recorded) or np.any(eligible & ~recorded):
            raise ValueError("eligibility must be a subset of the same-length recorded mask")
        self._eligible = np.pad(eligible, (0, MAX_SLOTS - len(eligible)))
        self._recorded = np.pad(recorded, (0, MAX_SLOTS - len(recorded)))
        self._eligible.setflags(write=False)
        self._recorded.setflags(write=False)
        self._probe_rng = probe_rng if probe_rng is not None else np.random.default_rng(secrets.randbits(256))
        self._refinement_rng = refinement_rng if refinement_rng is not None else np.random.default_rng(secrets.randbits(256))
        if not isinstance(self._probe_rng, np.random.Generator) or not isinstance(self._refinement_rng, np.random.Generator):
            raise ValueError("private RNGs must be numpy.random.Generator instances")
        probe_state = json.dumps(self._probe_rng.bit_generator.state, sort_keys=True, default=lambda value: np.asarray(value).tolist())
        refinement_state = json.dumps(self._refinement_rng.bit_generator.state, sort_keys=True, default=lambda value: np.asarray(value).tolist())
        if self._probe_rng.bit_generator is self._refinement_rng.bit_generator or probe_state == refinement_state:
            raise ValueError("probe and refinement must use distinct random streams")
        self._cap = 75.0 * int(eligible.sum())
        self._used = 0.0
        self._slot = 0
        self._failed = False
        self._records: list[StepRelease] = []

    @property
    def eligible_count(self) -> int:
        return int(self._eligible.sum())

    @property
    def cap(self) -> float:
        return self._cap

    @property
    def used_budget(self) -> float:
        return self._used

    @property
    def remaining_budget(self) -> float:
        return self._cap - self._used

    @property
    def slot_index(self) -> int:
        return self._slot

    @property
    def records(self) -> tuple[StepRelease, ...]:
        return tuple(self._records)

    @property
    def eligible_mask(self) -> tuple[bool, ...]:
        return tuple(bool(value) for value in self._eligible)

    @property
    def recorded_mask(self) -> tuple[bool, ...]:
        return tuple(bool(value) for value in self._recorded)

    def release_step(
        self,
        feature_loader: Callable[[], object],
        allocation_callback: Callable[[FloatArray], tuple[object, int]],
        *,
        prior_feedback: float = 0.0,
    ) -> StepRelease:
        if self._failed:
            raise RuntimeError("ledger invalidated by a failed admitted computation; retries are prohibited")
        if self._slot >= MAX_SLOTS:
            raise RuntimeError("the fixed 56-slot transcript is already complete")
        t = self._slot
        if not self._recorded[t]:
            return self._pad(STRUCTURAL_PAD)
        if not self._eligible[t]:
            return self._pad(TASK_PAD)
        if self.remaining_budget < MIN_INVOCATION_COST:
            return self._pad(FILTER_EXHAUSTED)

        try:
            # Do not move screen access or Gaussian draws above admission.
            clean = clip_features(feature_loader())
            probe_scale = analytic_gaussian_sigma(PROBE_EPSILON)
            probe = clean + self._probe_rng.normal(0.0, probe_scale, clean.shape)
            observation = protected_observation(probe, prior_feedback, self.remaining_budget / self.cap, t)
            observation.setflags(write=False)
            proposed, k = allocation_callback(observation)
            if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or not 1 <= k <= 20:
                raise ValueError("candidate count must be an integer in [1,20]")
            executed, status = filter_budgets(proposed, self.remaining_budget)
            assert executed is not None  # The admitted reserve funds the fallback.
            refinement_scales = np.array([analytic_gaussian_sigma(float(e - PROBE_EPSILON)) for e in executed])
            release = clean + self._refinement_rng.normal(size=clean.shape) * refinement_scales[:, None]
            charge = math.fsum(executed.tolist())
            next_used = self._used + charge
            if next_used > self.cap:
                raise RuntimeError("budget ledger rounding would exceed its public cap")
            self._used = next_used
            record = StepRelease(t, status, True, _readonly(executed), int(k), _readonly(release), _readonly(refinement_scales), self._used, self.remaining_budget)
        except Exception:
            self._failed = True
            raise
        self._records.append(record)
        self._slot += 1
        return record

    def _pad(self, status: str) -> StepRelease:
        record = StepRelease(self._slot, status, False, None, 0, None, None, self._used, self.remaining_budget)
        self._records.append(record)
        self._slot += 1
        return record
