"""Explicit Action schema and offline-evaluator boundary.

No benchmark function vocabulary, alias map, or official evaluator is guessed.
The caller binds a frozen Train-only schema artifact and a pinned evaluator.
"""
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .scoring import ActionFunctionSchema, strict_json


@dataclass(frozen=True)
class ActionSlot:
    request: str
    reference_action: Mapping | None
    history: tuple[str, ...] = ()
    eligible: bool = True
    recorded: bool = True
    # Trusted offline metadata for the externally supplied official evaluator.
    # Neither field crosses the executor or controller boundary.
    screen_size: tuple[int, int] | None = None
    reference_boxes: Mapping | None = None


@dataclass(frozen=True)
class ActionPrediction:
    action: Mapping
    feedback: float


@dataclass(frozen=True)
class ActionScores:
    function: bool
    arguments: bool
    status: bool

    def __post_init__(self):
        if any(type(x) is not bool for x in (self.function, self.arguments, self.status)):
            raise TypeError("Offline Action evaluator must return three boolean component scores")

    @property
    def step(self):
        return self.function and self.arguments and self.status


def load_action_schemas(path):
    """Read a caller-authored schema, including only explicit Train-only aliases.

    File keys: schema_id, source_split='fit-train', source_manifest_sha256,
    functions, optional function_aliases. Each function supplies required,
    optional, spatial, statuses; optional argument_types and status_aliases.
    This loader validates the declaration; it cannot authenticate its provenance.
    Save the file digest with the run, together with the official scorer digest.
    """
    import re
    payload = strict_json(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(payload, dict) or not payload.get("schema_id")
            or payload.get("source_split") != "fit-train"
            or not re.fullmatch(r"[0-9a-fA-F]{64}", str(payload.get("source_manifest_sha256", "")))):
        raise ValueError("Action schema requires ID and explicit fit-train manifest SHA256")
    functions = payload.get("functions")
    if not isinstance(functions, dict) or not functions:
        raise ValueError("Action schema functions must be a nonempty object")
    schemas = {}
    for name, raw in functions.items():
        if not isinstance(name, str) or not name or name == "INVALID" or not isinstance(raw, dict):
            raise ValueError("Invalid action function definition")
        fields = {}
        for field in ("required", "optional", "spatial", "statuses"):
            values = raw.get(field)
            if not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values) or len(set(values)) != len(values):
                raise ValueError(f"{name}.{field} must be a unique string array")
            fields[field] = frozenset(values)
        if fields["required"] & fields["optional"] or not fields["spatial"] <= fields["required"] | fields["optional"] or not fields["statuses"]:
            raise ValueError("Action schema keys/statuses are inconsistent")
        types = raw.get("argument_types", {})
        kinds = {"string", "integer", "number", "boolean", "array", "object", "null"}
        if (not isinstance(types, dict) or not set(types) <= fields["required"] | fields["optional"]
                or any(v not in kinds for v in types.values())
                or any(types.get(k, "array") != "array" for k in fields["spatial"])):
            raise ValueError("Invalid explicit argument type contract")
        aliases = raw.get("status_aliases", {})
        if (not isinstance(aliases, dict) or any(not isinstance(k, str) or not k or v not in fields["statuses"] for k, v in aliases.items())
                or any(k in fields["statuses"] and k != v for k, v in aliases.items())):
            raise ValueError("Invalid or conflicting status alias")
        schemas[name] = ActionFunctionSchema(**fields, argument_types=MappingProxyType(dict(types)),
                                             canonical_function=name, status_aliases=MappingProxyType(dict(aliases)))
    aliases = payload.get("function_aliases", {})
    if not isinstance(aliases, dict):
        raise ValueError("Function aliases must be an object")
    for alias, name in aliases.items():
        if not isinstance(alias, str) or not alias or name not in functions or (alias in functions and alias != name):
            raise ValueError("Invalid or conflicting function alias")
        schemas[alias] = schemas[name]
    return MappingProxyType(schemas)
