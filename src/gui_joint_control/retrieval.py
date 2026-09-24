"""Immutable Train-only demonstration banks and release-only queries.

Membership is fixed against the complete public evaluation manifest before any
query. Source declarations and hashes are recorded, not treated as proof that
the supplied data was collected correctly. No evaluation-time insertion exists.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata

import numpy as np

from .prompt_policy import Retrieval
from .scoring import parse_action, strict_json, unit


def canonical_json(value):
    """NFC, sorted keys, compact UTF-8, Python 3 shortest-round-trip floats."""
    def normalize(item):
        if isinstance(item, str):
            return unicodedata.normalize("NFC", item)
        if isinstance(item, dict):
            result = {}
            for key, val in item.items():
                if not isinstance(key, str):
                    raise ValueError("Demonstration object keys must be strings")
                key = unicodedata.normalize("NFC", key)
                if key in result:
                    raise ValueError("NFC normalization creates duplicate object keys")
                result[key] = normalize(val)
            return result
        if isinstance(item, list):
            return [normalize(x) for x in item]
        if item is None or isinstance(item, (bool, int, float)):
            return item
        raise ValueError("Demonstrations must contain typed JSON values")
    encoded = json.dumps(normalize(value), ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
    encoded.encode("utf-8", errors="strict")
    return encoded


def serialize_demonstrations(entries):
    """Preserve retrieval ranking and emit exactly one ordered JSON array."""
    return "[" + ",".join(canonical_json(strict_json(entry.text)) for entry in entries) + "]"


def _sha256(value, field):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError(f"{field} must be an explicit SHA256")
    return value.lower()


def _metadata(raw, *, strict):
    keys = ["source_task_id", "template_id", "document_id", "normalized_instruction"]
    if strict:
        keys += ["template_family", "document_family"]
    if not isinstance(raw, dict) or any(not isinstance(raw.get(k), str) or not raw[k] for k in keys):
        raise ValueError("Exclusion views require complete public identity and normalized-instruction metadata")
    return raw


def _excluded(candidate, evaluation, *, strict):
    for target in evaluation:
        if any(candidate[k] == target[k] for k in
               ("source_task_id", "template_id", "document_id", "normalized_instruction")):
            return True
        if strict:
            if any(candidate[k] == target[k] for k in ("template_family", "document_family")):
                return True
            left = set(candidate["normalized_instruction"].split())
            right = set(target["normalized_instruction"].split())
            if left and right and len(left & right) / len(left | right) >= .80:
                return True
    return False


@dataclass(frozen=True)
class BankEntry:
    trajectory_id: str
    original_slot: int
    task: str
    text: str


class FrozenDemonstrationBank:
    """At most 10,000 FIFO entries; top eight using preceding protected U."""

    def __init__(self, entries, normalized_keys, *, task, provenance, evaluation_public_records=()):
        if task not in {"G", "A"} or len(entries) > 10000:
            raise ValueError("Bank requires G/A and at most 10,000 entries")
        keys = np.asarray(normalized_keys, dtype=np.float64).copy()
        if keys.shape != (len(entries), 256) or not np.isfinite(keys).all():
            raise ValueError("Bank keys must be finite [entries,256] vectors")
        if any(entry.task != task for entry in entries):
            raise ValueError("Cross-task retrieval is prohibited")
        if np.any(np.linalg.norm(keys, axis=1) > 1 + 1e-10):
            raise ValueError("Bank keys must be normalized with the declared norm floor")
        self.entries = tuple(entries)
        # Immutable bytes backing prevents WRITEABLE from being re-enabled.
        self._keys = np.frombuffer(keys.tobytes(order="C"), dtype=np.float64).reshape(keys.shape)
        self.task = task
        from types import MappingProxyType
        self.provenance = MappingProxyType(dict(provenance))
        self._evaluation_public_records = tuple(canonical_json(row) for row in evaluation_public_records)

    def validate_evaluation_manifest(self, path):
        """Bind the complete fixed exclusion view to the actual public manifest.

        For a combined development/test manifest, the driver must additionally
        verify that every current run identity belongs to that complete manifest.
        No image path, pixels, feature vector, target, or prediction affects this
        public-only membership check.
        """
        from collections import Counter
        payload = Path(path).read_bytes()
        if hashlib.sha256(payload).hexdigest() != self.provenance.get("evaluation_manifest_sha256"):
            raise ValueError("Retrieval exclusion evaluation-manifest digest mismatch")
        strict = self.provenance.get("view") == "strict"
        fields = ["source_task_id", "template_id", "document_id", "normalized_instruction"]
        if strict:
            fields += ["template_family", "document_family"]

        def signature(raw):
            raw = _metadata(raw, strict=strict)
            return canonical_json({key: raw[key] for key in fields})

        actual = []
        for line in payload.decode("utf-8").splitlines():
            if not line.strip():
                continue
            row = strict_json(line)
            if row.get("task", "G") == self.task:
                actual.append(signature(row.get("public_metadata")))
        declared = [signature(strict_json(row)) for row in self._evaluation_public_records]
        if not actual or Counter(actual) != Counter(declared):
            raise ValueError("Declared bank exclusion metadata does not equal the complete public evaluation manifest")
        return self.provenance["evaluation_manifest_sha256"]

    @classmethod
    def from_files(cls, manifest_json, keys_npy, *, task, view="primary", schemas=None):
        """Load a declared fit-train bank and construct a global exclusion view.

        ``keys_npy`` stores mean clean fit-train features in manifest row order,
        never current/evaluation features. The JSON carries source/evaluation
        manifest SHA256s, success_binding, entries and evaluation_public_records.
        Every source entry binds split, eligibility, schema validity and upstream
        trajectory success. False values are excluded; missing labels abort.
        """
        if view not in {"primary", "strict"}:
            raise ValueError("Bank view must be primary or strict")
        if task == "A" and not schemas:
            raise ValueError("Action banks require the explicitly pinned action schemas")
        raw_bytes = Path(manifest_json).read_bytes()
        manifest = strict_json(raw_bytes.decode("utf-8"))
        if manifest.get("source_split") != "fit-train" or manifest.get("task") != task:
            raise ValueError("Bank must bind the same task and fit-train split")
        source_hash = _sha256(manifest.get("source_manifest_sha256"), "source_manifest_sha256")
        evaluation_hash = _sha256(manifest.get("evaluation_manifest_sha256"), "evaluation_manifest_sha256")
        keys_hash = _sha256(manifest.get("keys_sha256"), "keys_sha256")
        if hashlib.sha256(Path(keys_npy).read_bytes()).hexdigest() != keys_hash:
            raise ValueError("Bank key artifact digest mismatch")
        success = manifest.get("success_binding")
        if (not isinstance(success, dict) or not isinstance(success.get("source_field"), str)
                or not success["source_field"] or success.get("granularity") != "trajectory"
                or success.get("missing_policy") not in {"error", "exclude"}):
            raise ValueError("Bank requires explicit upstream trajectory-success and missing-value binding")
        evaluation = manifest.get("evaluation_public_records")
        if not isinstance(evaluation, list) or not evaluation:
            raise ValueError("Complete public evaluation metadata must be supplied for a frozen exclusion view")
        evaluation = [_metadata(item, strict=view == "strict") for item in evaluation]
        rows = manifest.get("entries")
        if not isinstance(rows, list):
            raise ValueError("Bank entries must be an array")
        keys = np.load(keys_npy, allow_pickle=False)
        if keys.shape != (len(rows), 256) or not np.isfinite(keys).all():
            raise ValueError("Bank key rows must correspond exactly to source entries")
        eligible = []
        seen = set()
        for index, row in enumerate(rows):
            if row.get("source_split") != "fit-train" or row.get("task") != task:
                raise ValueError("Every bank source row must be task-matched fit-train")
            slot, trajectory = row.get("original_slot"), row.get("trajectory_id")
            if (not isinstance(trajectory, str) or not trajectory or type(slot) is not int or slot < 0
                    or (trajectory, slot) in seen):
                raise ValueError("Invalid or duplicate bank trajectory/slot identity")
            seen.add((trajectory, slot))
            flags = [row.get("eligible"), row.get("schema_valid"), row.get("trajectory_success")]
            if any(type(flag) is not bool for flag in flags):
                if row.get("trajectory_success") is None and success["missing_policy"] == "exclude" and all(type(flag) is bool for flag in flags[:2]):
                    continue
                raise ValueError("Bank admission requires explicit eligibility, schema validity, and success labels")
            if not all(flags):
                continue
            metadata = _metadata(row.get("public_metadata"), strict=view == "strict")
            value = row.get("value")
            required = ({"instruction", "coordinate", "element_type"} if task == "G"
                        else {"instruction", "reference_action", "schema_id"})
            if not isinstance(value, dict) or set(value) != required or not isinstance(value["instruction"], str) or not value["instruction"]:
                raise ValueError("Bank values must be typed task-specific demonstration objects")
            if task == "G":
                coord = value["coordinate"]
                if (not isinstance(coord, list) or len(coord) != 2 or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in coord)
                        or not isinstance(value["element_type"], str) or not value["element_type"]):
                    raise ValueError("Grounding bank requires canonical normalized coordinate and element type")
            elif (not isinstance(value["schema_id"], str) or not value["schema_id"]
                  or not isinstance(value["reference_action"], dict)
                  or set(value["reference_action"]) != {"function", "arguments", "status"}):
                raise ValueError("Action bank requires canonical reference JSON and schema identifier")
            elif task == "A":
                parsed = parse_action(canonical_json(value["reference_action"]), schemas)
                if parsed is None or parsed != value["reference_action"]:
                    raise ValueError("Action bank reference must already satisfy the canonical frozen schema")
            entry = BankEntry(trajectory, slot, task, canonical_json(value))
            eligible.append((entry, unit(keys[index]), metadata))
        # Source ordering establishes FIFO before fixed evaluation exclusions.
        eligible.sort(key=lambda x: (x[0].trajectory_id, x[0].original_slot, x[0].task))
        retained = [item for item in eligible[-10000:] if not _excluded(item[2], evaluation, strict=view == "strict")]
        return cls([item[0] for item in retained], np.asarray([item[1] for item in retained]).reshape(-1, 256),
                   task=task, provenance={"manifest_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                                          "source_manifest_sha256": source_hash, "evaluation_manifest_sha256": evaluation_hash,
                                          "keys_sha256": keys_hash, "view": view, "success_binding": canonical_json(success),
                                          "serialization": "NFC-Python3-shortest-roundtrip-sorted-compact-json-v1"},
                   evaluation_public_records=evaluation)

    def retrieve(self, release, prior_feedback, threshold):
        """One completed release is reused for all retrieval and decoder paths."""
        release = np.asarray(release, dtype=np.float64)
        if release.shape != (25, 256) or not np.isfinite(release).all():
            raise ValueError("Queries must be the finite completed 25x256 refinement release")
        if not math.isfinite(prior_feedback) or not 0 <= prior_feedback <= 2:
            raise ValueError("Retrieval gate requires protected preceding feedback in [0,2]")
        if not math.isfinite(threshold) or threshold not in {i / 4 for i in range(9)}:
            raise ValueError("Threshold must be a development-selected member of {0,.25,...,2}")
        if prior_feedback <= threshold or not self.entries:
            return ()
        query = unit(release.mean(axis=0, dtype=np.float64))
        scores = self._keys @ query
        order = sorted((i for i, score in enumerate(scores) if score > .15),
                       key=lambda i: (-float(scores[i]), self.entries[i].trajectory_id,
                                      self.entries[i].original_slot, self.entries[i].task))[:8]
        return tuple(Retrieval(self.entries[i].text, float(scores[i])) for i in order)
