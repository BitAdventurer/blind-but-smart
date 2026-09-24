"""Validated, immutable offline replay for the new reference implementation.

No model results are synthesized here. An NPZ must contain actual transition
records (or clearly identified software fixtures); loading never uses pickle.
The sampler first draws a success/failure stratum with probabilities .7/.3,
then draws uniformly with replacement within that stratum.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np


FIELDS = (
    "observation", "executed_budgets", "candidate_count", "reward",
    "next_observation", "terminal", "remaining_budget",
    "next_remaining_budget", "success",
)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _immutable(array: np.ndarray) -> np.ndarray:
    """A bytes-backed copy cannot have WRITEABLE enabled again by a caller."""
    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(contiguous.shape)


class ReplayBuffer:
    """A fixed, validated task/family transition population.

    Observations are 28-dimensional; critic actions comprise 25 executed total
    regional budgets and a count in 1..20. Both strata must be present. Terminal
    successors may have less than 37.5 budget; nonterminal successors must be
    invocable. Optional NPZ arrays are retained as immutable metadata and enter
    the content hash, but never enter the actor/critic input.
    """

    def __init__(self, arrays: Mapping[str, np.ndarray], *, source_sha256: str | None = None):
        missing = sorted(set(FIELDS) - set(arrays))
        if missing:
            raise ValueError(f"Replay is missing required fields: {', '.join(missing)}")
        raw = {key: np.asarray(value) for key, value in arrays.items()}
        if any(value.dtype.hasobject for value in raw.values()):
            raise ValueError("Replay object arrays are not allowed")
        observation = raw["observation"]
        if observation.ndim != 2 or observation.shape[1] != 28:
            raise ValueError("observation must have shape [n, 28]")
        count = len(observation)
        if not 0 < count <= 1_000_000:
            raise ValueError("Replay must contain 1..1,000,000 transitions; no eviction is performed")
        expected_shapes = {
            "observation": (count, 28), "next_observation": (count, 28),
            "executed_budgets": (count, 25),
        }
        for field in FIELDS:
            shape = expected_shapes.get(field, (count,))
            if raw[field].shape != shape:
                raise ValueError(f"{field} must have shape {shape}")
            if field in ("terminal", "success"):
                if raw[field].dtype.kind != "b":
                    raise ValueError(f"{field} must be a boolean array")
            elif field == "candidate_count":
                if raw[field].dtype.kind not in "iu":
                    raise ValueError("candidate_count must be an integer array")
            elif raw[field].dtype.kind not in "fiu":
                raise ValueError(f"{field} must be a real numeric array")
            if not np.isfinite(raw[field]).all():
                raise ValueError(f"{field} must contain only finite values")
        budgets = raw["executed_budgets"]
        if np.any((budgets < 1.5) | (budgets > 5.0)):
            raise ValueError("executed_budgets must lie in [1.5, 5.0]")
        if np.any((raw["candidate_count"] < 1) | (raw["candidate_count"] > 20)):
            raise ValueError("candidate_count must lie in 1..20")
        remaining = raw["remaining_budget"]
        next_remaining = raw["next_remaining_budget"]
        if np.any(remaining < 37.5):
            raise ValueError("Every replay transition must start with at least 37.5 budget")
        if np.any(next_remaining < 0) or np.any(next_remaining > remaining + 1e-5):
            raise ValueError("next_remaining_budget must be nonnegative and cannot increase")
        if np.any(budgets.sum(axis=1, dtype=np.float64) > remaining + 1e-5):
            raise ValueError("Executed regional costs exceed remaining_budget")
        if np.any(next_remaining[~raw["terminal"]] < 37.5):
            raise ValueError("Nonterminal transitions require an invocable successor (budget >= 37.5)")
        if not raw["success"].any() or raw["success"].all():
            raise ValueError("Both success and failure replay strata are required; no fallback sampling")
        values = {}
        for field in FIELDS:
            dtype = (np.bool_ if field in ("terminal", "success") else
                     np.int64 if field == "candidate_count" else
                     np.float64 if field in ("remaining_budget", "next_remaining_budget") else np.float32)
            with np.errstate(over="ignore", invalid="ignore"):
                converted = raw[field].astype(dtype, copy=True)
            if not np.isfinite(converted).all():
                raise ValueError(f"{field} cannot be represented by its required {np.dtype(dtype)} dtype")
            values[field] = _immutable(converted)
        self.arrays = MappingProxyType(values)
        self.metadata = MappingProxyType({key: _immutable(value) for key, value in raw.items() if key not in FIELDS})
        if ('slot_id' in self.metadata) != ('next_slot_id' in self.metadata):
            raise ValueError('Replay slot_id and next_slot_id must be supplied together')
        for key in ('slot_id', 'next_slot_id'):
            if key in self.metadata:
                ids = self.metadata[key]
                if ids.shape != (count,) or ids.dtype.kind != 'U':
                    raise ValueError(f'{key} must be a Unicode string array with one public ID per transition')
                required = np.ones(count, dtype=bool) if key == 'slot_id' else ~values['terminal']
                if np.any(ids[required] == ''):
                    raise ValueError(f'{key} is missing an invocable public slot ID')
        if 'next_slot_id' in self.metadata and np.any(self.metadata['next_slot_id'][values['terminal']] != ''):
            raise ValueError('Terminal next_slot_id must be empty')
        self.success_indices = _immutable(np.flatnonzero(values["success"]))
        self.failure_indices = _immutable(np.flatnonzero(~values["success"]))
        self.source_sha256 = source_sha256
        digest = sha256()
        for key, value in sorted({**self.arrays, **self.metadata}.items()):
            digest.update(key.encode("utf-8") + b"\0")
            digest.update(str(value.dtype).encode("ascii") + b"\0")
            digest.update(str(value.shape).encode("ascii") + b"\0")
            digest.update(value.tobytes())
        self.content_sha256 = digest.hexdigest()

    @classmethod
    def from_npz(cls, path: str | Path) -> "ReplayBuffer":
        path = Path(path)
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        return cls(arrays, source_sha256=_file_sha256(path))

    def __len__(self) -> int:
        return len(self.arrays["observation"])

    def sample_indices(self, batch_size: int, rng: np.random.Generator) -> np.ndarray:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        choose_success = rng.random(batch_size) < 0.7
        indices = np.empty(batch_size, dtype=np.int64)
        indices[choose_success] = rng.choice(self.success_indices, int(choose_success.sum()), replace=True)
        indices[~choose_success] = rng.choice(self.failure_indices, int((~choose_success).sum()), replace=True)
        return indices

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        indices = self.sample_indices(batch_size, rng)
        return {**{field: values[indices].copy() for field, values in self.arrays.items()},
                **{field: self.metadata[field][indices].copy() for field in ('slot_id', 'next_slot_id') if field in self.metadata},
                "indices": indices}

    def describe(self) -> dict:
        return {
            "transitions": len(self), "successes": len(self.success_indices),
            "failures": len(self.failure_indices), "source_sha256": self.source_sha256,
            "content_sha256": self.content_sha256, "immutable": True,
            "sampling": "Bernoulli(.7) success stratum, then uniform with replacement",
        }
