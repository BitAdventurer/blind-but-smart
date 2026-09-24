"""Bound development-mean TMS schedules for new runs.

The caller supplies real development means and the complete eligible population.
This module never estimates them from replay rewards or manuscript result tables.
Count assignments follow public hash ordering, stable-ID tie breaking, and
half-up rounding over the entire declared population (not each mini-batch).
"""
from __future__ import annotations

from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_UP
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from types import MappingProxyType

import numpy as np


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _digest(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


class TMSSchedule:
    """Immutable task/family and population-bound TMS artifact.

    Schema version 1 requires ``task`` (G/A), ``family``, ``population``
    (split name), ``population_manifest_sha256``, ``public_hash_seed`` (256-bit
    hex), and ``slots`` containing every eligible ``slot_id`` and 1-based
    ``original_step``. ``development`` binds the selected H checkpoint and
    development manifest digests plus ``mean_regional_budget`` and
    ``mean_candidate_count``. Original steps are checked against observation
    coordinate 27, t/56; they do not determine hash-ordered count assignments.
    """

    def __init__(self, artifact: dict, *, source_sha256: str | None = None):
        if not isinstance(artifact, dict) or artifact.get("schema_version") != 1:
            raise ValueError("TMS artifact schema_version must be 1")
        artifact = json.loads(_canonical(artifact))
        if artifact.get("task") not in ("G", "A"):
            raise ValueError("TMS task must be G or A")
        for key in ("family", "population"):
            if not isinstance(artifact.get(key), str) or not artifact[key].strip():
                raise ValueError(f"TMS {key} must be a nonempty string")
        for key in ("population_manifest_sha256", "public_hash_seed"):
            _digest(artifact.get(key), key)
        development = artifact.get("development")
        if not isinstance(development, dict):
            raise ValueError("TMS development provenance and means are required")
        for key in ("selected_h_checkpoint_sha256", "development_manifest_sha256"):
            _digest(development.get(key), f"development.{key}")
        for key, lower, upper in (("mean_regional_budget", 1.5, 5.0), ("mean_candidate_count", 1.0, 20.0)):
            value = development.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"TMS development.{key} must lie in [{lower},{upper}]")
        slots = artifact.get("slots")
        if not isinstance(slots, list) or not slots:
            raise ValueError("TMS slots must declare a nonempty complete eligible population")
        steps = {}
        for slot in slots:
            if not isinstance(slot, dict):
                raise ValueError("TMS slots must be objects")
            identity, step = slot.get("slot_id"), slot.get("original_step")
            if not isinstance(identity, str) or not identity or identity in steps:
                raise ValueError("TMS slot IDs must be unique nonempty strings")
            if isinstance(step, bool) or not isinstance(step, int) or not 1 <= step <= 56:
                raise ValueError("TMS original_step must be an integer in 1..56")
            steps[identity] = step
        # Canonical population order does not depend on JSON serialization order.
        artifact["slots"] = sorted(slots, key=lambda entry: entry["slot_id"])
        self.task, self.family = artifact["task"], artifact["family"]
        self.population = artifact["population"]
        self.population_manifest_sha256 = artifact["population_manifest_sha256"]
        self.source_sha256 = source_sha256
        self.content_sha256 = sha256(_canonical(artifact).encode("utf-8")).hexdigest()
        mean = Decimal(str(development["mean_candidate_count"]))
        floor = int(mean.to_integral_value(rounding=ROUND_FLOOR))
        ceiling_slots = int((Decimal(len(slots)) * (mean - floor)).to_integral_value(rounding=ROUND_HALF_UP))
        ordered = sorted(steps, key=lambda identity: (
            sha256(_canonical(["TMS-count-v1", artifact["public_hash_seed"], self.task, self.family, identity]).encode("utf-8")).hexdigest(), identity))
        self._counts = MappingProxyType({identity: floor + int(index < ceiling_slots) for index, identity in enumerate(ordered)})
        self._steps = MappingProxyType(steps)
        self._budget = float(development["mean_regional_budget"])
        self._development = MappingProxyType(development)
        self._artifact_json = _canonical(artifact)

    @classmethod
    def from_json(cls, path: str | Path):
        data = Path(path).read_bytes()
        return cls(json.loads(data), source_sha256=sha256(data).hexdigest())

    def proposals(self, slot_ids, observation_time=None):
        identities = np.asarray(slot_ids)
        if identities.ndim != 1 or identities.dtype.kind not in "US":
            raise ValueError("TMS slot_ids must be a one-dimensional string array")
        identities = identities.astype(str).tolist()
        missing = [identity for identity in identities if identity not in self._counts]
        if missing:
            raise ValueError("TMS schedule does not contain every requested slot ID")
        if observation_time is not None:
            times = np.asarray(observation_time, dtype=np.float64)
            expected = np.array([self._steps[identity] / 56.0 for identity in identities])
            if times.shape != expected.shape or not np.isfinite(times).all() or not np.allclose(times, expected, rtol=0, atol=1e-7):
                raise ValueError("TMS slot original_step disagrees with observation time t/56")
        return (np.full((len(identities), 25), self._budget, dtype=np.float64),
                np.array([self._counts[identity] for identity in identities], dtype=np.int64))

    def describe(self):
        return {"content_sha256": self.content_sha256, "source_sha256": self.source_sha256,
                "task": self.task, "family": self.family, "population": self.population,
                "population_manifest_sha256": self.population_manifest_sha256,
                "slots": len(self._counts), "development": dict(self._development),
                "scheduled_candidate_mean": sum(self._counts.values()) / len(self._counts),
                "count_rule": "public SHA-256 order, stable-ID ties, half-up number of ceilings"}

    def to_dict(self):
        """Return an independently owned, JSON-serializable artifact."""
        return json.loads(self._artifact_json)


def build_tms_schedule(*, task: str, family: str, population: str,
                       population_manifest_sha256: str, public_hash_seed: str,
                       slots: list[dict], mean_regional_budget: float,
                       mean_candidate_count: float, selected_h_checkpoint_sha256: str,
                       development_manifest_sha256: str) -> TMSSchedule:
    """Bind supplied development measurements; do not invent missing means."""
    return TMSSchedule({
        "schema_version": 1, "task": task, "family": family, "population": population,
        "population_manifest_sha256": population_manifest_sha256,
        "public_hash_seed": public_hash_seed, "slots": slots,
        "development": {"mean_regional_budget": mean_regional_budget,
                        "mean_candidate_count": mean_candidate_count,
                        "selected_h_checkpoint_sha256": selected_h_checkpoint_sha256,
                        "development_manifest_sha256": development_manifest_sha256},
    })
