"""Explicit ten-family controller fitting and three-post-fit-evaluation plans.

Input is a registry of existing frozen family/task executor artifacts, replay,
manifests, public roots, seeds, and exact argv arrays. No data, fitted executor,
historical root, checkpoint or measurement is invented. The plan starts AFTER
executor fitting and therefore does not by itself establish ten complete refits.

Registry schema (all paths should be absolute):
  {"schema_version": 1, "families": [{"family_id": "...", "public_root":
  "<64 lowercase hex>", "tasks": [{"task": "G" or "A", "model": "...",
  "revision": "<immutable 40-hex revision>", "config": {"path": ..., "sha256": ...},
  "replay": artifact, "dev_manifest": artifact, "manifest": artifact,
  "projection": artifact, "methods": [{"method": "H", "seed": integer,
  "fit_output": "...", "fit_argv": [...], "evaluations": [{"replicate_id":
  "...", "seed": integer, "output": "...", "argv": [...]} x 3]}]}]} x 10]}.

Each argv starts with bbs or python -m gui_joint_control.cli. Binding-related
flags are checked against the registry, so a command cannot silently substitute
another replay, executor, split, family, method, seed, or selected checkpoint.
build_plan never executes commands; execute_plan is the explicit execution API.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
from typing import Callable, Mapping

from .selection import _atomic_json, _canonical, file_sha256


UPDATES = {"H": 1_000_000, "CB": 1_000_000, "Disclosure-only": 1_000_000,
           "Count-only": 1_000_000, "Independent": 500_000, "Independent-1M": 1_000_000}
EXECUTOR_FLAGS = {"--dtype", "--device", "--tokenizer", "--tokenizer-revision", "--dino-model", "--dino-revision",
                  "--retrieval-threshold", "--retrieval-view", "--action-evaluator", "--disable-retrieval"}
EXECUTOR_ARTIFACTS = {"--public-projection", "--action-schema", "--retrieval-bank", "--retrieval-keys", "--retrieval-exclusion-manifest"}
SCHEDULE_FLAGS = {"--tms-schedule", "--dev-tms-schedule", "--evaluation-tms-schedule"}


def artifact_digest(path: str | Path) -> str:
    """File SHA-256, or SHA-256 of sorted [relative POSIX path, file SHA] pairs.

Local model/tokenizer directories are pinned by their actual contents, because
a remote revision flag does not freeze local weights. No files are excluded.
"""
    path = Path(path)
    if path.is_file():
        return file_sha256(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    entries = [[item.relative_to(path).as_posix(), file_sha256(item)]
               for item in sorted(path.rglob("*")) if item.is_file()]
    if not entries:
        raise ValueError("An empty model/tokenizer directory is not a bound artifact")
    return sha256(_canonical(sorted(entries)).encode("utf-8")).hexdigest()


def _identifier(value, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, str)) or not str(value):
        raise ValueError(f"{name} must be an explicit nonempty ID")
    return str(value)


def _seed(value) -> int:
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError("Seeds must be explicitly bound integers in [0,2**63)")
    return value


def _path(value) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("Artifact and output paths must be nonempty strings")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("Protocol paths must be absolute to avoid working-directory drift")
    return str(path.resolve())


def _artifact(value, verify: bool) -> dict:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError("Every artifact requires exactly path and sha256")
    path, digest = _path(value["path"]), value["sha256"]
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Artifact SHA-256 must be explicitly bound lowercase hex")
    if verify and artifact_digest(path) != digest:
        raise ValueError(f"Artifact content differs from registry: {path}")
    return {"path": path, "sha256": digest}


def _command(argv, expected_command: str) -> tuple[list[str], dict[str, str | bool]]:
    if not isinstance(argv, list) or not argv or any(not isinstance(word, str) or not word or "\0" in word for word in argv):
        raise ValueError("Commands must be nonempty string argv arrays, never shell command strings")
    executable = Path(argv[0]).name.lower()
    if executable in ("bbs", "bbs.exe"):
        command_index = 1
    elif re.fullmatch(r"python(?:\d+(?:\.\d+)*)?(?:\.exe)?", executable) and argv[1:3] == ["-m", "gui_joint_control.cli"]:
        command_index = 3
    else:
        raise ValueError("Protocol commands must use bbs or python -m gui_joint_control.cli")
    if len(argv) <= command_index or argv[command_index] != expected_command:
        raise ValueError(f"Expected protocol command {expected_command}")
    flags, index = {}, command_index + 1
    while index < len(argv):
        flag = argv[index]
        if not flag.startswith("--") or flag in flags or "=" in flag:
            raise ValueError("Use unique explicit --flag value pairs in protocol commands")
        if flag in ("--allow-download", "--disable-retrieval", "--software-only"):
            flags[flag] = True
            index += 1
        else:
            if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                raise ValueError(f"Missing explicit value for {flag}")
            flags[flag] = argv[index + 1]
            index += 2
    if any(flag in flags for flag in ("--resume", "--software-only")):
        raise ValueError("A new manuscript protocol cannot resume unrelated runs or use software fixtures")
    return list(argv), flags


def _check_flags(flags: Mapping, expected: Mapping, *, paths: set[str]) -> None:
    for flag, value in expected.items():
        actual = flags.get(flag)
        if flag in paths:
            if actual is None or _path(actual) != _path(value):
                raise ValueError(f"Command {flag} does not match the registered artifact/output")
        elif actual != (value if isinstance(value, bool) else str(value)):
            raise ValueError(f"Command {flag} does not match the registered binding")


def build_plan(registry: Mapping, output_root: str | Path, *, verify_artifacts: bool = True) -> dict:
    """Validate all ten families and produce a dependency-ordered immutable plan.

Trainable methods are selected explicitly, with the same method/task population
across families. Methods share the family/task frozen executor and replay.
Decoder seeds/replicate IDs are paired across methods. Secret Gaussian roots
are deliberately absent: runtime supplies fresh independent secret streams.
"""
    if not isinstance(registry, Mapping) or registry.get("schema_version") != 1:
        raise ValueError("Protocol registry schema_version must be 1")
    families = registry.get("families")
    if not isinstance(families, list) or len(families) != 10:
        raise ValueError("The manuscript controller protocol requires exactly ten explicitly bound families")
    root = _path(str(output_root))
    family_ids, roots, output_paths, controller_seeds = set(), set(), set(), set()
    jobs, artifacts = [], {}
    def register_artifact(artifact):
        previous = artifacts.setdefault(artifact["path"], artifact["sha256"])
        if previous != artifact["sha256"]:
            raise ValueError("One artifact path has conflicting content hashes")
    population = None
    task_frames = {}
    task_banks = {}
    for family in families:
        family_id = _identifier(family.get("family_id"), "family_id")
        public_root = family.get("public_root")
        if family_id in family_ids or not isinstance(public_root, str) or not re.fullmatch(r"[0-9a-f]{64}", public_root) or public_root in roots:
            raise ValueError("Families require unique IDs and unique explicitly committed 256-bit public roots")
        family_ids.add(family_id)
        roots.add(public_root)
        tasks = family.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("Each family requires explicit task bindings")
        task_ids, this_population = set(), {}
        for task in tasks:
            task_id = task.get("task")
            if task_id not in ("G", "A") or task_id in task_ids:
                raise ValueError("Task IDs must be distinct G or A within each family")
            task_ids.add(task_id)
            bound = {field: _artifact(task.get(field), verify_artifacts)
                     for field in ("config", "replay", "dev_manifest", "manifest", "projection")}
            for artifact in bound.values():
                register_artifact(artifact)
            if bound["dev_manifest"]["sha256"] == bound["manifest"]["sha256"]:
                raise ValueError("Development and final evaluation manifests cannot be the same artifact")
            frame = (bound["dev_manifest"]["sha256"], bound["manifest"]["sha256"])
            if task_frames.setdefault(task_id, frame) != frame:
                raise ValueError("All fitting families must share the task's development and final evaluation manifests")
            model, revision = task.get("model"), task.get("revision")
            if not isinstance(model, str) or not model or not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise ValueError("Each frozen executor needs a model ID/path and immutable 40-hex revision")
            if Path(model).is_absolute() or Path(model).exists():
                local_model = _artifact(task.get("model_artifact"), verify_artifacts)
                if local_model["path"] != _path(model):
                    raise ValueError("Local model content binding must cover the --model directory")
                register_artifact(local_model)
                bound["model_artifact"] = local_model
            executor_flags = task.get("executor_flags", {})
            executor_artifacts = task.get("executor_artifacts", {})
            if not isinstance(executor_flags, dict) or set(executor_flags) - EXECUTOR_FLAGS:
                raise ValueError("Unsupported executor_flags; bind documented flags explicitly")
            if not isinstance(executor_artifacts, dict) or set(executor_artifacts) - EXECUTOR_ARTIFACTS:
                raise ValueError("Unsupported executor_artifacts")
            if any(not isinstance(value, (str, bool, int, float)) or value is None for value in executor_flags.values()):
                raise ValueError("Executor flags require explicit scalar values")
            if "--disable-retrieval" in executor_flags and executor_flags["--disable-retrieval"] is not True:
                raise ValueError("Omit --disable-retrieval unless its explicit value is true")
            extra_artifacts = {flag: _artifact(value, verify_artifacts) for flag, value in executor_artifacts.items()}
            for artifact in extra_artifacts.values():
                register_artifact(artifact)
            if executor_flags.get("--disable-retrieval"):
                if any(flag in extra_artifacts for flag in ("--retrieval-bank", "--retrieval-keys")):
                    raise ValueError("A retrieval-disabled condition cannot also supply a retrieval bank")
            elif not {"--retrieval-bank", "--retrieval-keys"}.issubset(extra_artifacts) or "--retrieval-threshold" not in executor_flags:
                raise ValueError("Bind retrieval bank, keys and selected threshold, or explicitly disable retrieval")
            bank_binding = (bool(executor_flags.get("--disable-retrieval")),
                            tuple(extra_artifacts.get(flag, {}).get("sha256") for flag in
                                  ("--retrieval-bank", "--retrieval-keys", "--retrieval-exclusion-manifest")),
                            executor_flags.get("--retrieval-view", "primary"))
            if task_banks.setdefault(task_id, bank_binding) != bank_binding:
                raise ValueError("Fitting families must share each task's frozen retrieval bank, keys and view; gates may differ")
            for name in ("tokenizer", "dino-model"):
                value = executor_flags.get("--" + name)
                revision_flag = "--tokenizer-revision" if name == "tokenizer" else "--dino-revision"
                if value is None:
                    continue
                if not isinstance(value, str) or not value:
                    raise ValueError("Model/tokenizer IDs must be nonempty strings")
                if Path(value).is_absolute() or Path(value).exists():
                    artifact = _artifact(task.get(name.replace("-", "_") + "_artifact"), verify_artifacts)
                    if artifact["path"] != _path(value):
                        raise ValueError("Local tokenizer/DINO content binding must cover its directory")
                    register_artifact(artifact)
                    bound[name.replace("-", "_") + "_artifact"] = artifact
                elif not re.fullmatch(r"[0-9a-f]{40}", str(executor_flags.get(revision_flag, ""))):
                    raise ValueError("Remote tokenizer/DINO IDs require immutable 40-hex revisions")
            if task_id == "A":
                if "--action-schema" not in extra_artifacts or "--action-evaluator" not in executor_flags:
                    raise ValueError("Action protocols require a bound typed schema and exact-step evaluator")
                scorer_source = _artifact(task.get("action_evaluator_source"), verify_artifacts)
                register_artifact(scorer_source)
                bound["action_evaluator_source"] = scorer_source
            bound["executor_flags"] = dict(executor_flags)
            bound["executor_artifacts"] = extra_artifacts
            dev_replicates = task.get("dev_replicates", 3)
            if type(dev_replicates) is not int or dev_replicates < 1:
                raise ValueError("Registered development replicate count must be a positive integer")
            bound["dev_replicates"] = dev_replicates
            methods = task.get("methods")
            if not isinstance(methods, list) or not methods:
                raise ValueError("Each task needs an explicit nonempty controller method list")
            role_methods = {"Disclosure-only", "Count-only", "Independent", "Independent-1M"}
            requested = {method.get("method") for method in methods}
            if "H" in requested and requested & role_methods:
                raise ValueError("Fit H first, then construct its TMS artifacts and a separate baseline plan; pre-existing schedules cannot bind a newly fitted H")
            paired_h = None
            if requested & role_methods:
                paired_h = _artifact(task.get("paired_h_checkpoint"), verify_artifacts)
                register_artifact(paired_h)
                bound["paired_h_checkpoint"] = paired_h
            method_ids, evaluation_pairing = set(), None
            for method in methods:
                method_id = method.get("method")
                if method_id not in UPDATES or method_id in method_ids:
                    raise ValueError("Unknown or duplicate trainable controller method")
                method_ids.add(method_id)
                seed = _seed(method.get("seed"))
                if seed in controller_seeds:
                    raise ValueError("Controller seeds must be distinct across family/task/method streams")
                controller_seeds.add(seed)
                schedules = method.get("schedule_artifacts", {})
                if not isinstance(schedules, dict) or set(schedules) - SCHEDULE_FLAGS:
                    raise ValueError("Unsupported schedule_artifacts flags")
                schedules = {flag: _artifact(value, verify_artifacts) for flag, value in schedules.items()}
                if method_id in ("Disclosure-only", "Count-only") and set(schedules) != SCHEDULE_FLAGS:
                    raise ValueError("Standalone single-head methods require train, development and evaluation TMS schedule bindings")
                if method_id in ("Independent", "Independent-1M") and "--tms-schedule" not in schedules:
                    raise ValueError("Independent fitting requires its training TMS schedule binding")
                for artifact in schedules.values():
                    register_artifact(artifact)
                    if paired_h is not None:
                        from .tms import TMSSchedule
                        declared = TMSSchedule.from_json(artifact["path"]).to_dict()
                        if declared["task"] != task_id or declared["family"] != family_id:
                            raise ValueError("TMS task/family differs from its paired baseline plan")
                        development = declared["development"]
                        if development["selected_h_checkpoint_sha256"] != paired_h["sha256"] or development["development_manifest_sha256"] != bound["dev_manifest"]["sha256"]:
                            raise ValueError("TMS schedule is not derived from the bound selected H and development manifest")
                fit_output = _path(method.get("fit_output"))
                selected = str(Path(fit_output) / "selected.pt")
                argv, flags = _command(method.get("fit_argv"), "train")
                common = {"--model": model, "--revision": revision, "--projection": bound["projection"]["path"],
                          "--family-id": family_id, "--task": task_id, **executor_flags,
                          **{flag: artifact["path"] for flag, artifact in extra_artifacts.items()}}
                expected = {**common, "--config": bound["config"]["path"], "--replay": bound["replay"]["path"],
                            "--dev-manifest": bound["dev_manifest"]["path"], "--method": method_id,
                            "--seed": seed, "--updates": UPDATES[method_id], "--output": fit_output,
                            **{flag: artifact["path"] for flag, artifact in schedules.items() if flag != "--evaluation-tms-schedule"}}
                _check_flags(flags, expected, paths={"--config", "--replay", "--dev-manifest", "--projection", "--output"} | set(extra_artifacts) | set(schedules))
                if (set(flags) & (EXECUTOR_FLAGS | EXECUTOR_ARTIFACTS | SCHEDULE_FLAGS)) - set(expected):
                    raise ValueError("Command supplies executor/schedule flags absent from its registry binding")
                if flags.get("--batch-size", "256") != "256":
                    raise ValueError("Manuscript fitting uses a replay batch size of 256")
                if flags.get("--dev-replicates", "3") != str(dev_replicates):
                    raise ValueError("Development replicate count differs from its registered binding")
                if fit_output in output_paths:
                    raise ValueError("Every protocol fit/evaluation needs a unique output directory")
                output_paths.add(fit_output)
                prefix = f"{family_id}/{task_id}/{method_id}"
                binding = {"family_id": family_id, "public_root": public_root, "task": task_id,
                           "method": method_id, "model": model, "revision": revision,
                           "schedule_artifacts": schedules, **bound}
                jobs.append({"job_id": prefix + "/fit", "kind": "controller_fit", "depends_on": [],
                             "argv": argv, "output": fit_output, "seed": seed, "binding": binding})
                evaluations = method.get("evaluations")
                if not isinstance(evaluations, list) or len(evaluations) != 3:
                    raise ValueError("Each selected controller requires exactly three post-fit evaluations")
                replicate_ids, seeds, pairing = set(), set(), []
                for evaluation in evaluations:
                    replicate = _identifier(evaluation.get("replicate_id"), "replicate_id")
                    eval_seed = _seed(evaluation.get("seed"))
                    if replicate in replicate_ids or eval_seed in seeds:
                        raise ValueError("Replicate IDs and public decoder seeds must be distinct within a method")
                    replicate_ids.add(replicate)
                    seeds.add(eval_seed)
                    pairing.append((replicate, eval_seed))
                    output = _path(evaluation.get("output"))
                    argv, flags = _command(evaluation.get("argv"), "collect-grounding" if task_id == "G" else "collect-action")
                    expected = {**common, "--manifest": bound["manifest"]["path"], "--training-replay": bound["replay"]["path"],
                                "--controller-config": bound["config"]["path"], "--controller-method": method_id,
                                "--controller-checkpoint": selected, "--seed": eval_seed, "--replicate-id": replicate, "--output": output,
                                **{flag: artifact["path"] for flag, artifact in schedules.items() if flag != "--dev-tms-schedule"}}
                    _check_flags(flags, expected, paths={"--manifest", "--training-replay", "--controller-config", "--controller-checkpoint", "--projection", "--output"} | set(extra_artifacts) | set(schedules))
                    if (set(flags) & (EXECUTOR_FLAGS | EXECUTOR_ARTIFACTS | SCHEDULE_FLAGS)) - set(expected):
                        raise ValueError("Command supplies executor/schedule flags absent from its registry binding")
                    if output in output_paths:
                        raise ValueError("Every protocol fit/evaluation needs a unique output directory")
                    output_paths.add(output)
                    jobs.append({"job_id": prefix + "/evaluate/" + replicate, "kind": "post_fit_evaluation",
                                 "depends_on": [prefix + "/fit"], "argv": argv, "output": output,
                                 "seed": eval_seed, "replicate_id": replicate, "selected_checkpoint": selected, "binding": binding})
                if evaluation_pairing is not None and sorted(pairing) != evaluation_pairing:
                    raise ValueError("Methods within a family/task must pair public decoder seeds and replicate IDs")
                evaluation_pairing = sorted(pairing)
            this_population[task_id] = sorted(method_ids)
        if population is not None and this_population != population:
            raise ValueError("Every fitting family must register the same task/method population")
        population = this_population
    plan = {"format_version": 1, "kind": "controller_fitting_and_post_fit_evaluation_plan",
            "scope": "starts from supplied frozen executor artifacts; does not itself perform or prove complete executor refits",
            "output_root": root, "family_count": 10, "evaluation_replicates": 3,
            "registry_sha256": sha256(_canonical(registry).encode("utf-8")).hexdigest(),
            "artifacts": [{"path": path, "sha256": digest} for path, digest in sorted(artifacts.items())],
            "jobs": jobs, "reproduces_historical_results": False}
    plan["plan_sha256"] = sha256(_canonical(plan).encode("utf-8")).hexdigest()
    return plan


def execute_plan(plan: Mapping, *, runner: Callable | None = None) -> dict:
    """Run only an explicitly supplied validated plan, in order, without a shell.

The caller must obtain authorization before invoking this for real experiments.
build_plan is the safe preparation step. Failed/incomplete journals are not
silently retried; users must diagnose the partial run and create a new plan.
A custom runner(argv) is useful for software tests and orchestration backends.
"""
    payload = dict(plan)
    digest = payload.pop("plan_sha256", None)
    if digest != sha256(_canonical(payload).encode("utf-8")).hexdigest():
        raise ValueError("Protocol plan content changed after validation")
    root = Path(plan["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    journal_path = root / "protocol-execution.json"
    if journal_path.exists():
        raise FileExistsError("Execution journal already exists; inspect prior/partial results before a new run")
    journal = {"format_version": 1, "plan_sha256": digest, "status": "running", "jobs": [],
               "reproduces_historical_results": False}
    _atomic_json(journal_path, journal)
    completed = set()
    selected_hashes = {}
    run = runner or (lambda argv: subprocess.run(argv, check=True, shell=False))
    try:
        for job in plan["jobs"]:
            if not set(job["depends_on"]).issubset(completed):
                raise ValueError("Protocol dependency has not completed")
            for artifact in plan["artifacts"]:
                if artifact_digest(artifact["path"]) != artifact["sha256"]:
                    raise ValueError("An input artifact changed after plan validation")
            if job["kind"] == "post_fit_evaluation" and not Path(job["selected_checkpoint"]).is_file():
                raise FileNotFoundError("Training did not produce a development-selected checkpoint")
            entry = {"job_id": job["job_id"], "status": "running"}
            if job["kind"] == "post_fit_evaluation":
                checkpoint = job["selected_checkpoint"]
                actual = file_sha256(checkpoint)
                if selected_hashes.setdefault(checkpoint, actual) != actual:
                    raise ValueError("The selected controller changed between post-fit replicates")
                entry["selected_checkpoint_sha256"] = actual
            journal["jobs"].append(entry)
            _atomic_json(journal_path, journal)
            result = run(list(job["argv"]))
            if getattr(result, "returncode", 0) != 0:
                raise RuntimeError(f"Protocol command failed: {job['job_id']}")
            entry["status"] = "completed"
            completed.add(job["job_id"])
            _atomic_json(journal_path, journal)
        journal["status"] = "completed"
        _atomic_json(journal_path, journal)
    except BaseException as exc:
        journal["status"] = "failed"
        journal["error_type"] = type(exc).__name__
        if journal["jobs"] and journal["jobs"][-1]["status"] == "running":
            journal["jobs"][-1]["status"] = "failed"
        _atomic_json(journal_path, journal)
        raise
    return journal
