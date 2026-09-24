"""Bound plan software fixtures: no model, dataset, fitting or evaluation runs."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from gui_joint_control.protocol import artifact_digest, build_plan, execute_plan
from gui_joint_control.selection import file_sha256


def registry_fixture(root):
    artifacts = {}
    for field in ("config", "replay", "dev_manifest", "manifest", "projection", "train_tms", "dev_tms", "eval_tms"):
        path = root / (field + ".fixture")
        path.write_text("synthetic software fixture " + field, encoding="utf-8")
        artifacts[field] = {"path": str(path), "sha256": file_sha256(path)}
    families = []
    for family_index in range(10):
        family_id = "fixture-family-" + str(family_index)
        common = ["--model", "fixture-not-a-real-model", "--revision", "a" * 40,
                  "--projection", artifacts["projection"]["path"], "--family-id", family_id, "--task", "G", "--disable-retrieval"]
        methods = []
        for method_index, method in enumerate(("H", "CB")):
            output = str(root / family_id / method / "fit")
            seed = 100 + 2 * family_index + method_index
            schedules = {"--tms-schedule": artifacts["train_tms"]} if method == "Independent" else {}
            fit = ["bbs", "train", *common, "--config", artifacts["config"]["path"],
                   "--replay", artifacts["replay"]["path"], "--dev-manifest", artifacts["dev_manifest"]["path"],
                   "--method", method, "--seed", str(seed), "--updates", "500000" if method == "Independent" else "1000000",
                   "--output", output]
            for flag, artifact in schedules.items():
                if flag != "--evaluation-tms-schedule":
                    fit.extend([flag, artifact["path"]])
            evaluations = []
            for replicate in range(3):
                eval_output = str(root / family_id / method / ("eval-" + str(replicate)))
                argv = ["bbs", "collect-grounding", *common, "--manifest", artifacts["manifest"]["path"],
                        "--training-replay", artifacts["replay"]["path"], "--controller-config", artifacts["config"]["path"],
                        "--controller-method", method, "--controller-checkpoint", str(Path(output) / "selected.pt"),
                        "--seed", str(200 + replicate), "--replicate-id", str(replicate), "--output", eval_output]
                for flag, artifact in schedules.items():
                    if flag != "--dev-tms-schedule":
                        argv.extend([flag, artifact["path"]])
                evaluations.append({"replicate_id": str(replicate), "seed": 200 + replicate, "output": eval_output, "argv": argv})
            methods.append({"method": method, "seed": seed, "fit_output": output, "fit_argv": fit,
                            "schedule_artifacts": schedules, "evaluations": evaluations})
        families.append({"family_id": family_id, "public_root": format(family_index + 1, "064x"),
                         "tasks": [{"task": "G", "model": "fixture-not-a-real-model", "revision": "a" * 40,
                                    **artifacts, "executor_flags": {"--disable-retrieval": True}, "methods": methods}]})
    return {"schema_version": 1, "families": families}


def independent_registry_fixture(root):
    """A second-stage plan paired to explicit artificial selected-H bytes."""
    registry = registry_fixture(root)
    for family in registry["families"]:
        task = family["tasks"][0]
        method = task["methods"][1]
        task["methods"] = [method]
        method["method"] = "Independent"
        fit = method["fit_argv"]
        fit[fit.index("--method") + 1] = "Independent"
        fit[fit.index("--updates") + 1] = "500000"
        h_checkpoint = root / (family["family_id"] + "-selected-H.fixture")
        h_checkpoint.write_text("artificial prior-stage H checkpoint " + family["family_id"])
        task["paired_h_checkpoint"] = {"path": str(h_checkpoint), "sha256": file_sha256(h_checkpoint)}
        schedule = root / (family["family_id"] + "-train-tms.json")
        schedule.write_text(json.dumps({
            "schema_version": 1, "task": "G", "family": family["family_id"], "population": "fit-train",
            "population_manifest_sha256": "e" * 64, "public_hash_seed": "f" * 64,
            "slots": [{"slot_id": "synthetic-slot", "original_step": 1}],
            "development": {"mean_regional_budget": 3.0, "mean_candidate_count": 5.0,
                            "selected_h_checkpoint_sha256": task["paired_h_checkpoint"]["sha256"],
                            "development_manifest_sha256": task["dev_manifest"]["sha256"]},
        }), encoding="utf-8")
        artifact = {"path": str(schedule), "sha256": file_sha256(schedule)}
        method["schedule_artifacts"] = {"--tms-schedule": artifact}
        fit.extend(["--tms-schedule", str(schedule)])
        for evaluation in method["evaluations"]:
            argv = evaluation["argv"]
            argv[argv.index("--controller-method") + 1] = "Independent"
            argv.extend(["--tms-schedule", str(schedule)])
    return registry


class ProtocolTests(unittest.TestCase):
    def test_ten_families_three_replicates_plan_has_frozen_checkpoint_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan(registry_fixture(root), root / "plan")
            self.assertEqual(len(plan["jobs"]), 80)
            self.assertEqual(sum(job["kind"] == "controller_fit" for job in plan["jobs"]), 20)
            self.assertEqual(sum(job["kind"] == "post_fit_evaluation" for job in plan["jobs"]), 60)
            for index in range(0, 80, 4):
                fit, *evaluations = plan["jobs"][index:index + 4]
                self.assertTrue(all(row["depends_on"] == [fit["job_id"]] for row in evaluations))
                self.assertEqual(len({row["selected_checkpoint"] for row in evaluations}), 1)

    def test_commands_cannot_substitute_replay_seed_or_last_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = registry_fixture(root)
            for flag, wrong, key in (("--replay", str(root / "other"), "fit_argv"),
                                     ("--seed", "999", "fit_argv"),
                                     ("--controller-checkpoint", str(root / "last.pt"), "evaluation")):
                changed = copy.deepcopy(registry)
                method = changed["families"][0]["tasks"][0]["methods"][0]
                argv = method["fit_argv"] if key == "fit_argv" else method["evaluations"][0]["argv"]
                argv[argv.index(flag) + 1] = wrong
                with self.assertRaisesRegex(ValueError, "does not match"):
                    build_plan(changed, root / "plan")

    def test_family_roots_replicate_count_and_pairing_are_not_inferred(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = registry_fixture(root)
            for change in (
                lambda obj: obj["families"].pop(),
                lambda obj: obj["families"][0].update(public_root=""),
                lambda obj: obj["families"][0]["tasks"][0]["methods"][0]["evaluations"].pop(),
            ):
                altered = copy.deepcopy(registry)
                change(altered)
                with self.assertRaises(ValueError):
                    build_plan(altered, root / "plan")
            altered = copy.deepcopy(registry)
            evaluation = altered["families"][0]["tasks"][0]["methods"][1]["evaluations"][0]
            evaluation["seed"] = 999
            argv = evaluation["argv"]
            argv[argv.index("--seed") + 1] = "999"
            with self.assertRaisesRegex(ValueError, "pair public"):
                build_plan(altered, root / "plan")

    def test_explicit_runner_executes_only_registered_jobs_and_journals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan(registry_fixture(root), root / "plan")
            seen = []
            def fixture_runner(argv):
                seen.append(argv)
                output = Path(argv[argv.index("--output") + 1])
                output.mkdir(parents=True)
                if argv[1] == "train":
                    (output / "selected.pt").write_text("software fixture only")
                return SimpleNamespace(returncode=0)
            journal = execute_plan(plan, runner=fixture_runner)
            self.assertEqual(journal["status"], "completed")
            self.assertEqual(len(seen), 80)
            with self.assertRaises(FileExistsError):
                execute_plan(plan, runner=fixture_runner)

    def test_plan_or_artifact_drift_stops_before_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan(registry_fixture(root), root / "plan")
            altered = copy.deepcopy(plan)
            altered["jobs"][0]["argv"].append("--software-only")
            with self.assertRaisesRegex(ValueError, "content changed"):
                execute_plan(altered, runner=lambda _: self.fail("No command should run"))
            Path(plan["artifacts"][0]["path"]).write_text("modified")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                execute_plan(plan, runner=lambda _: self.fail("No command should run"))

    def test_local_model_revision_does_not_replace_content_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = registry_fixture(root)
            model = root / "fitted-model"
            model.mkdir()
            (model / "weights.fixture").write_text("software-only model bytes")
            task = registry["families"][0]["tasks"][0]
            task["model"] = str(model)
            for method in task["methods"]:
                commands = [method["fit_argv"], *[item["argv"] for item in method["evaluations"]]]
                for argv in commands:
                    argv[argv.index("--model") + 1] = str(model)
            with self.assertRaisesRegex(ValueError, "path and sha256"):
                build_plan(registry, root / "plan")
            task["model_artifact"] = {"path": str(model), "sha256": artifact_digest(model)}
            plan = build_plan(registry, root / "plan")
            (model / "weights.fixture").write_text("changed bytes despite unchanged revision flag")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                execute_plan(plan, runner=lambda _: self.fail("No command should run"))

    def test_selected_controller_cannot_change_between_three_evaluations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan(registry_fixture(root), root / "plan")
            def fixture_runner(argv):
                output = Path(argv[argv.index("--output") + 1])
                output.mkdir(parents=True)
                if argv[1] == "train":
                    (output / "selected.pt").write_text("software fixture checkpoint")
                else:
                    Path(argv[argv.index("--controller-checkpoint") + 1]).write_text("unexpected mutation")
                return SimpleNamespace(returncode=0)
            with self.assertRaisesRegex(ValueError, "changed between"):
                execute_plan(plan, runner=fixture_runner)

    def test_duplicate_controller_seed_is_not_an_independent_method_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = registry_fixture(root)
            methods = registry["families"][0]["tasks"][0]["methods"]
            methods[1]["seed"] = methods[0]["seed"]
            argv = methods[1]["fit_argv"]
            argv[argv.index("--seed") + 1] = str(methods[1]["seed"])
            with self.assertRaisesRegex(ValueError, "Controller seeds"):
                build_plan(registry, root / "plan")

    def test_independent_needs_training_tms_but_standalone_needs_split_schedules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = independent_registry_fixture(root)
            # Independent composes its learned heads during development and Bench.
            build_plan(registry, root / "plan")
            changed = copy.deepcopy(registry)
            for family in changed["families"]:
                method = family["tasks"][0]["methods"][0]
                method["method"] = "Count-only"
                fit = method["fit_argv"]
                fit[fit.index("--method") + 1] = "Count-only"
                fit[fit.index("--updates") + 1] = "1000000"
                for evaluation in method["evaluations"]:
                    argv = evaluation["argv"]
                    argv[argv.index("--controller-method") + 1] = "Count-only"
            with self.assertRaisesRegex(ValueError, "Standalone single-head"):
                build_plan(changed, root / "plan")

    def test_tms_cannot_silently_refer_to_a_different_h_or_a_future_refit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = independent_registry_fixture(root)
            changed = copy.deepcopy(registry)
            task = changed["families"][0]["tasks"][0]
            artifact = task["methods"][0]["schedule_artifacts"]["--tms-schedule"]
            path = Path(artifact["path"])
            contents = json.loads(path.read_text())
            contents["development"]["selected_h_checkpoint_sha256"] = "0" * 64
            path.write_text(json.dumps(contents))
            artifact["sha256"] = file_sha256(path)
            with self.assertRaisesRegex(ValueError, "not derived from the bound selected H"):
                build_plan(changed, root / "plan")
            task["methods"].append({"method": "H"})
            with self.assertRaisesRegex(ValueError, "Fit H first"):
                build_plan(changed, root / "plan")

    def test_all_families_share_the_fixed_task_evaluation_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = registry_fixture(root)
            path = root / "different-evaluation.fixture"
            path.write_text("different software fixture frame")
            task = registry["families"][1]["tasks"][0]
            task["manifest"] = {"path": str(path), "sha256": file_sha256(path)}
            for method in task["methods"]:
                for evaluation in method["evaluations"]:
                    argv = evaluation["argv"]
                    argv[argv.index("--manifest") + 1] = str(path)
            with self.assertRaisesRegex(ValueError, "share the task's development"):
                build_plan(registry, root / "plan")

    def test_task_retrieval_bank_shared_across_families_but_gate_can_differ(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = registry_fixture(root)
            artifacts = {}
            for name in ("bank", "keys", "changed-bank"):
                path = root / (name + ".fixture")
                path.write_text("software-only " + name)
                artifacts[name] = {"path": str(path), "sha256": file_sha256(path)}
            for index, family in enumerate(registry["families"]):
                task = family["tasks"][0]
                gate = str((index % 9) / 4)
                task["executor_flags"] = {"--retrieval-threshold": gate}
                task["executor_artifacts"] = {"--retrieval-bank": artifacts["bank"], "--retrieval-keys": artifacts["keys"]}
                for method in task["methods"]:
                    for argv in [method["fit_argv"], *[item["argv"] for item in method["evaluations"]]]:
                        argv.remove("--disable-retrieval")
                        argv.extend(["--retrieval-threshold", gate, "--retrieval-bank", artifacts["bank"]["path"],
                                     "--retrieval-keys", artifacts["keys"]["path"]])
            build_plan(registry, root / "plan")
            task = registry["families"][1]["tasks"][0]
            task["executor_artifacts"]["--retrieval-bank"] = artifacts["changed-bank"]
            for method in task["methods"]:
                for argv in [method["fit_argv"], *[item["argv"] for item in method["evaluations"]]]:
                    argv[argv.index("--retrieval-bank") + 1] = artifacts["changed-bank"]["path"]
            with self.assertRaisesRegex(ValueError, "share each task's frozen retrieval"):
                build_plan(registry, root / "plan")


if __name__ == "__main__":
    unittest.main()
