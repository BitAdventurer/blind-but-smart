"""Aggregate an existing recorded run without copying its trusted/raw records.

This exporter verifies recorded bindings and arithmetic; it does not authenticate
author measurements, run models, or recover missing historical settings.
"""
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import re


RUN_KINDS = {"new_recorded_controller_evaluation", "new_behavior_collection"}
METHODS = {"H", "CB", "Independent", "Independent-1M", "Disclosure-only",
           "Count-only", "TMS", "behavior"}
STATUSES = {"EXEC_PROP", "EXEC_FALLBACK", "FILTER_EXHAUSTED", "TASK_PAD", "STRUCTURAL_PAD"}
BINDINGS = ("manifest_sha256", "controller_checkpoint_sha256", "controller_config_sha256",
            "training_replay_sha256", "tms_schedule_sha256", "evaluation_tms_schedule_sha256",
            "projection_sha256", "public_projection_sha256", "action_schema_sha256",
            "retrieval_bank_sha256", "retrieval_keys_sha256", "retrieval_exclusion_manifest_sha256")
PROMPT_FIELDS = ("id", "version", "token_limit", "max_previous_steps", "history_policy",
                 "retrieval_policy", "overflow_policy", "renderer_sha256", "template_sha256",
                 "policy_sha256", "executor_sha256", "scoring_sha256")
RUNTIME_FIELDS = ("python", "torch", "numpy", "scipy", "transformers", "peft", "Pillow",
                  "accelerate", "safetensors", "os", "os_release", "machine", "unicode",
                  "device", "dtype", "cuda", "cudnn", "gpu_name")


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return value


def _hash(value, name):
    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError(f"{name} must be a SHA256 binding")
    return value.lower()


def _label(value, name):
    if (not isinstance(value, str) or not value or len(value) > 160
            or any(char in value for char in "\\/\n\r\t") or ":" in value):
        raise ValueError(f"{name} must be a public label, not a path")
    return value


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _device(value):
    if not isinstance(value, str) or not re.fullmatch(r"(?:cpu|cuda|mps|xpu)(?::[0-9]+)?", value):
        raise ValueError("device must be a CPU/CUDA/MPS/XPU device with optional integer index")
    return value


def _provenance(run):
    """An explicit allowlist: unknown nested fields never reach the report."""
    executor = run.get("executor") or {}
    runtime = run.get("runtime") or {}
    if not isinstance(executor, dict) or not isinstance(runtime, dict):
        raise ValueError("Executor/runtime provenance must be objects")
    result = {"artifact_bindings": {key: _hash(run.get(key), key) for key in BINDINGS},
              "executor": {key: _label(executor[key], key) if executor.get(key) else None
                           for key in ("model_revision", "tokenizer_revision")},
              "snapshots": {}, "prompt_policy": None, "manifest_split": None,
              "model_dtype": run.get("model_dtype"),
              "model_device": _device(run["model_device"]) if run.get("model_device") is not None else None,
              "retrieval": {"enabled": run.get("retrieval_enabled"),
                            "view": run.get("retrieval_view"),
                            "threshold": run.get("retrieval_threshold")},
              "runtime": {key: runtime.get(key) for key in RUNTIME_FIELDS}}
    for key in ("model_snapshot", "tokenizer_snapshot", "dino_snapshot"):
        snapshot = run.get(key) or {}
        if not isinstance(snapshot, dict):
            raise ValueError("Snapshot bindings must be objects")
        result["snapshots"][key] = {
            "revision": _label(snapshot["revision"], key) if snapshot.get("revision") else None,
            "tree_sha256": _hash(snapshot.get("tree_sha256"), key)}
    evaluator = run.get("action_evaluator") or {}
    if not isinstance(evaluator, dict):
        raise ValueError("Action evaluator binding must be an object")
    result["artifact_bindings"]["action_evaluator_source_sha256"] = _hash(
        evaluator.get("source_sha256"), "action_evaluator_source_sha256")
    policy = run.get("prompt_policy")
    if isinstance(policy, dict):
        result["prompt_policy"] = {}
        for key in PROMPT_FIELDS:
            value = policy.get(key)
            if value is not None:
                result["prompt_policy"][key] = (_hash(value, key) if key.endswith("_sha256")
                    else _number(value, key) if key in ("token_limit", "max_previous_steps")
                    else _label(value, key))
    elif policy is not None:
        result["prompt_policy"] = {"id": _label(policy, "prompt_policy")}
    split = run.get("manifest_split")
    if split is not None:
        if not isinstance(split, dict) or split.get("requested_split") != run.get("split"):
            raise ValueError("Manifest split binding differs from run.json")
        roles = split.get("declared_roles")
        missing = split.get("undeclared_records")
        if (not isinstance(roles, list) or len(roles) > 1
                or any(role not in ("fit-train", "development", "test") for role in roles)
                or type(missing) is not int or missing < 0):
            raise ValueError("Invalid manifest split binding")
        requested = split["requested_split"]
        aliases = {"fit-train": "fit-train", "development": "development", "test": "test", "evaluation": "test"}
        if requested is not None and (requested not in aliases or roles and roles[0] != aliases[requested]):
            raise ValueError("Declared manifest role differs from run split")
        result["manifest_split"] = {"requested_split": requested,
                                    "declared_roles": roles, "undeclared_records": missing}
    # Scalar settings also pass validation instead of allowing paths/raw objects.
    for key in ("model_dtype",):
        if result[key] is not None:
            result[key] = _label(result[key], key)
    retrieval = result["retrieval"]
    if retrieval["enabled"] is not None and type(retrieval["enabled"]) is not bool:
        raise ValueError("retrieval_enabled must be boolean")
    if retrieval["view"] is not None:
        retrieval["view"] = _label(retrieval["view"], "retrieval_view")
    if retrieval["threshold"] is not None:
        _number(retrieval["threshold"], "retrieval_threshold")
    for key, value in result["runtime"].items():
        if value is not None:
            result["runtime"][key] = (_device(value) if key == "device" else
                _number(value, key) if key == "cudnn" else _label(value, key))
    return result


def build_summary(run_dir):
    """Read only completed run.json/transcript.jsonl pairs; return safe aggregates.

Old transcripts lack row-level run identities. Their absence is recorded, while
any identity that is present must match run.json. Every trajectory must retain
all 56 slots, including exhausted eligible misses and structural/task padding.
"""
    directory = Path(run_dir)
    if (directory / "aborted.json").exists():
        raise ValueError("Cannot summarize an aborted run as completed evaluation")
    run_bytes = (directory / "run.json").read_bytes()
    run = json.loads(run_bytes)
    if not isinstance(run, dict):
        raise ValueError("run.json must contain an object")
    if run.get("kind") not in RUN_KINDS:
        raise ValueError("Expected a completed new recorded evaluation or behavior collection")
    if run.get("reproduces_historical_results") is not False:
        raise ValueError("Recorded reference runs must explicitly disclaim historical reproduction")
    identity = {key: _label(run.get(key), key)
                for key in ("task", "family_id", "replicate_id", "controller_method")}
    if identity["task"] not in ("G", "A") or identity["controller_method"] not in METHODS:
        raise ValueError("Unsupported task or controller method")
    raw = (directory / "transcript.jsonl").read_bytes()
    digest = sha256(raw).hexdigest()
    if _hash(run.get("transcript_sha256"), "transcript_sha256") != digest:
        raise ValueError("Transcript SHA256 differs from run.json")
    groups = {}
    absent = Counter()
    statuses, eligible_statuses = Counter(), Counter()
    action_fields = ("function_correct", "arguments_correct", "status_correct") if identity["task"] == "A" else ()
    totals = Counter(eligible=0, invoked=0, correct=0, candidates=0)
    components = Counter({key: 0 for key in action_fields})
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("Transcript rows must be objects")
        for key, value in identity.items():
            if key not in row:
                absent[key] += 1
            elif row[key] != value:
                raise ValueError(f"Transcript {key} differs from run.json")
        trajectory, slot = row.get("trajectory_id"), row.get("slot")
        if not isinstance(trajectory, str) or not trajectory or type(slot) is not int or not 0 <= slot < 56:
            raise ValueError("Transcript requires trajectory ID and integer slot 0..55")
        group = groups.setdefault(trajectory, {})
        if slot in group:
            raise ValueError("Duplicate transcript trajectory/slot")
        group[slot] = row
        if any(type(row.get(key)) is not bool for key in ("eligible", "invoked", "correct", *action_fields)):
            raise ValueError("Eligibility, invocation and correctness must be boolean")
        eligible, invoked, correct = (row[key] for key in ("eligible", "invoked", "correct"))
        if invoked and not eligible or correct and not invoked:
            raise ValueError("Correctness requires invocation, and invocation requires eligibility")
        status = row.get("status")
        if status not in STATUSES:
            raise ValueError("Unsupported transcript status; incomplete/failed runs cannot be summarized")
        if (invoked != (status in ("EXEC_PROP", "EXEC_FALLBACK"))
                or eligible != (status not in ("TASK_PAD", "STRUCTURAL_PAD"))):
            raise ValueError("Transcript status conflicts with eligibility/invocation")
        candidates = row.get("candidate_count")
        if type(candidates) is not int or not (1 <= candidates <= 20 if invoked else candidates == 0):
            raise ValueError("Candidate count must be 1..20 per invocation and zero for padding")
        if action_fields and (correct != all(row[key] for key in action_fields)
                              or not invoked and any(row[key] for key in action_fields)):
            raise ValueError("Action component scores conflict with step correctness/invocation")
        totals.update(eligible=int(eligible), invoked=int(invoked), correct=int(correct), candidates=candidates)
        components.update({key: int(row[key]) for key in action_fields})
        statuses[status] += 1
        if eligible:
            eligible_statuses[status] += 1
    if not groups or any(set(group) != set(range(56)) for group in groups.values()):
        raise ValueError("Transcript must preserve the complete 56-slot grid per trajectory")
    debits = []
    for group in groups.values():
        cap, used = 75 * sum(row["eligible"] for row in group.values()), 0.0
        for slot in range(56):
            row = group[slot]
            budgets = row.get("executed_budgets")
            if row["invoked"]:
                if used > cap - 37.5:
                    raise ValueError("Invocation after budget exhaustion")
                if (not isinstance(budgets, list) or len(budgets) != 25
                        or any(not 1.5 <= _number(value, "executed budget") <= 5 for value in budgets)):
                    raise ValueError("Invocation requires 25 executed regional budgets in [1.5,5]")
                if row["status"] == "EXEC_FALLBACK" and budgets != [1.5] * 25:
                    raise ValueError("Fallback must use the fixed 1.5 regional budget")
                used += math.fsum(budgets)
            elif budgets is not None:
                raise ValueError("Padding cannot contain executed budgets")
            elif row["status"] == "FILTER_EXHAUSTED" and used <= cap - 37.5:
                raise ValueError("Exhaustion padding before the remaining reserve is exhausted")
            if (used > cap + 1e-8
                    or not math.isclose(_number(row.get("used_budget"), "used_budget"), used, abs_tol=1e-8)
                    or not math.isclose(_number(row.get("remaining_budget"), "remaining_budget"), cap - used, abs_tol=1e-8)):
                raise ValueError("Transcript cumulative budget debit/remaining budget is inconsistent")
        debits.append(used)
    for key in ("eligible", "invoked", "correct", *action_fields):
        actual = totals[key] if key in totals else components[key]
        if type(run.get(key)) is not int or run[key] != actual:
            raise ValueError(f"Recorded {key} differs from transcript")
    eligible, invoked = totals["eligible"], totals["invoked"]
    accuracy = _ratio(totals["correct"], eligible)
    if run.get("accuracy") != accuracy:
        raise ValueError("Recorded accuracy differs from correct/eligible transcript arithmetic")
    debit = math.fsum(debits)
    counts = {"trajectories": len(groups), "fixed_slots": len(groups) * 56, **totals}
    metrics = {"accuracy": accuracy, "accuracy_per_invocation": _ratio(totals["correct"], invoked),
               "invocation_rate": _ratio(invoked, eligible), "regional_debit": debit,
               "mean_regional_budget_per_eligible": _ratio(debit, 25 * eligible),
               "mean_regional_budget_per_invocation": _ratio(debit, 25 * invoked),
               "candidates_per_invocation": _ratio(totals["candidates"], invoked),
               "candidates_per_eligible": _ratio(totals["candidates"], eligible),
               "budget_cap_fraction": _ratio(debit, 75 * eligible)}
    for key in action_fields:
        metrics[key.replace("_correct", "_accuracy")] = _ratio(components[key], eligible)
    provenance = _provenance(run)
    missing = [f"transcript.{key}: absent in {count} rows" for key, count in sorted(absent.items())]
    missing.extend(f"executor.{key}" for key, value in provenance["executor"].items() if value is None)
    if not provenance["prompt_policy"]:
        missing.append("prompt_policy")
    if provenance["manifest_split"] is None:
        missing.append("manifest_split")
    elif provenance["manifest_split"]["undeclared_records"]:
        missing.append(f"manifest.split: absent in {provenance['manifest_split']['undeclared_records']} records")
    return {"schema_version": 1, "scope": "aggregates of one supplied completed reference run",
            "reproduces_historical_results": False,
            "identity": {"kind": run["kind"], **identity,
                         "split": _label(run["split"], "split") if run.get("split") else None},
            "source_bindings": {"run_json_sha256": sha256(run_bytes).hexdigest(),
                                "transcript_sha256": digest, "transcript_hash_verified": True},
            "provenance": provenance, "missing_information": missing,
            "counts": counts, "metrics": metrics,
            "action_correct_counts": dict(components) if action_fields else None,
            "status_counts": {key: statuses[key] for key in sorted(STATUSES)},
            "status_proportions": {
                "per_fixed_slot": {key: _ratio(statuses[key], counts["fixed_slots"]) for key in sorted(STATUSES)},
                "per_eligible_slot": {key: _ratio(eligible_statuses[key], eligible) for key in sorted(STATUSES)}}}


def summarize_run(run_dir, output_path):
    """Create a new JSON summary; never replace run data or an existing output."""
    target = Path(output_path)
    if target.exists():
        raise FileExistsError(target)
    summary = build_summary(run_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return summary
