"""Synthetic checks of the new F.5 control, never measurements for Table F.6."""
import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from gui_joint_control.controller import parameter_counts
from gui_joint_control.protocol import build_plan
from gui_joint_control.replay import ReplayBuffer
from gui_joint_control.reporting import build_summary
from gui_joint_control.selection import CheckpointManager, file_sha256
from gui_joint_control.tms import TMSSchedule
from gui_joint_control.trainer import Trainer, filter_budgets
from test_protocol import independent_registry_fixture
from test_reporting import write_run, save_run
from test_selection import MANIFEST, result
from test_training import config, fixture_arrays, fixture_schedule


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def trainer(method="H-ActorTMS", *, seed=123, schedule=None):
    return Trainer(config(), method, ReplayBuffer(fixture_arrays()), seed=seed,
                   tms_schedule=fixture_schedule() if schedule is None else schedule,
                   task="G", family="fixture-family")


def tensor_batch():
    return {key: value if key in ("slot_id", "next_slot_id") else torch.from_numpy(value)
            for key, value in fixture_arrays().items()}


def test_actor_tms_has_exact_h_modules_counts_and_initialization():
    reference, control = trainer("H"), trainer()
    assert parameter_counts("H-ActorTMS") == {**parameter_counts("H"), "method": "H-ActorTMS"}
    assert control.reference_parameter_counts["trainable"] == 261192
    for name in ("actor", "critic1", "critic2", "target1", "target2"):
        for key, value in reference.bundles["joint"][name].state_dict().items():
            torch.testing.assert_close(value, control.bundles["joint"][name].state_dict()[key], rtol=0, atol=0)


def test_actor_tms_requires_current_tms_but_never_uses_it_for_targets_or_evaluation():
    with pytest.raises(ValueError, match="TMS schedule"):
        Trainer(config(), "H-ActorTMS", ReplayBuffer(fixture_arrays()))
    artifact = fixture_schedule().to_dict()
    artifact["slots"] = [slot for slot in artifact["slots"] if "next" not in slot["slot_id"]]
    control = trainer(schedule=TMSSchedule(artifact))
    reference = trainer("H")
    batch = tensor_batch()
    with patch.object(control.tms_schedule, "proposals", side_effect=AssertionError("TMS is actor-only")):
        with control._random_context():
            actual = control._target(control.bundles["joint"], batch, 0)
        with reference._random_context():
            expected = reference._target(reference.bundles["joint"], batch, 0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual = control.evaluation_action(batch["observation"], batch["remaining_budget"])
        expected = reference.evaluation_action(batch["observation"], batch["remaining_budget"])
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_actor_tms_uses_filtered_fixed_counterparts_and_once_only_entropy_gradients():
    control = trainer()
    batch = tensor_batch()
    batch = {key: value[:2] for key, value in batch.items()}
    batch["remaining_budget"] = torch.tensor([125., 50.], dtype=torch.float64)
    proposed = torch.full((2, 25), 4., requires_grad=True)
    score_log_prob = torch.tensor([.3, .4], requires_grad=True)
    continuous_entropy = torch.tensor([.1, .2], requires_grad=True)
    logits = torch.linspace(-1., 1., 40).reshape(2, 20).requires_grad_()
    probabilities = logits.softmax(-1)
    one_hot = torch.nn.functional.one_hot(torch.tensor([2, 7]), 20).float()
    one_hot = one_hot + probabilities - probabilities.detach()
    action = {"executed_budgets": filter_budgets(proposed, batch["remaining_budget"]),
              "count_one_hot": one_hot, "continuous_score_log_prob": score_log_prob,
              "continuous_log_prob": continuous_entropy, "probs": probabilities,
              "log_probs": logits.log_softmax(-1)}
    inputs = []
    counts_axis = torch.arange(1., 21.)
    def value(x):
        inputs.append(x.detach().clone())
        return x[:, 28:53].mean(-1) * (x[:, 53:] * counts_axis).sum(-1)
    loss = control._actor_tms_loss({"critic1": value, "critic2": lambda x: value(x) + 1}, batch, action)
    loss.backward()
    fixed_counts = control.tms_schedule.proposals(batch["slot_id"])[1]
    torch.testing.assert_close(inputs[0][:, 28:53], torch.tensor([[4.] * 25, [1.5] * 25]))
    torch.testing.assert_close(inputs[0][:, 53:].argmax(-1) + 1, torch.tensor(fixed_counts))
    torch.testing.assert_close(inputs[2][:, 28:53], torch.tensor([[3.2] * 25, [1.5] * 25]))
    torch.testing.assert_close(inputs[2][:, 53:].argmax(-1) + 1, torch.tensor([3, 8]))
    budget_values = torch.tensor([4., 1.5]) * torch.tensor(fixed_counts)
    torch.testing.assert_close(score_log_prob.grad, -budget_values / 2)
    torch.testing.assert_close(continuous_entropy.grad, torch.full((2,), .2 / 2))
    assert proposed.grad is None
    # Analytic softmax Jacobian for -Q(fixed budgets, sampled count) plus one
    # exact categorical entropy term. The budget branch cannot reach logits.
    p = probabilities.detach()
    mean_count = (p * counts_axis).sum(-1, keepdim=True)
    count_gradient = -torch.tensor([3.2, 1.5])[:, None] * p * (counts_axis - mean_count)
    log_p = p.log()
    entropy_gradient = .2 * p * (log_p - (p * log_p).sum(-1, keepdim=True))
    torch.testing.assert_close(logits.grad, (count_gradient + entropy_gradient) / 2)


def test_matched_h_and_actor_tms_keep_replay_draws_and_sampling_streams_paired():
    reference, control = trainer("H"), trainer()
    seen_h, seen_control = [], []
    h_sample, control_sample = reference.buffer.sample, control.buffer.sample
    def capture(original, seen):
        def sample(*args, **kwargs):
            batch = original(*args, **kwargs)
            seen.append(batch["indices"].copy())
            return batch
        return sample
    with patch.object(reference.buffer, "sample", side_effect=capture(h_sample, seen_h)), \
         patch.object(control.buffer, "sample", side_effect=capture(control_sample, seen_control)):
        for _ in range(3):
            reference.step(8)
            metrics = control.step(8)
            torch.testing.assert_close(reference._torch_rng, control._torch_rng, rtol=0, atol=0)
    for left, right in zip(seen_h, seen_control):
        np.testing.assert_array_equal(left, right)
    assert metrics["component_updates"] == {"joint": 3}
    assert metrics["aggregate_optimizer_calls"] == 6
    assert any(not torch.equal(a, b) for a, b in zip(reference.bundles["joint"]["actor"].parameters(),
                                                   control.bundles["joint"]["actor"].parameters()))


def test_actor_tms_single_optimizer_sequence_and_checkpoint_resume(tmp_path):
    control = trainer()
    bundle = control.bundles["joint"]
    from gui_joint_control.trainer import polyak_update
    events = []
    def record(name, original):
        def call(*args, **kwargs):
            events.append(name)
            return original(*args, **kwargs)
        return call
    with patch.object(bundle["critic_optimizer"], "step", side_effect=record("critic", bundle["critic_optimizer"].step)), \
         patch.object(bundle["actor_optimizer"], "step", side_effect=record("actor", bundle["actor_optimizer"].step)), \
         patch("gui_joint_control.trainer.polyak_update", side_effect=record("target", polyak_update)):
        control.step(8)
    assert events == ["critic", "actor", "target"]
    manager = CheckpointManager(tmp_path / "selection", binding={"dev_manifest_sha256": MANIFEST}, interval=1)
    assert manager.consider(control, lambda _: result())["selected"]
    restored = trainer(seed=999)
    restored.load_checkpoint(manager.selected_path)
    assert control.step(8) == restored.step(8)
    for module in ("actor", "critic1", "critic2", "target1", "target2"):
        for key, value in control.bundles["joint"][module].state_dict().items():
            torch.testing.assert_close(value, restored.bundles["joint"][module].state_dict()[key], rtol=0, atol=0)
    changed_schedule = trainer(schedule=fixture_schedule(budget=3.4))
    with pytest.raises(ValueError, match="tms_schedule_sha256"):
        changed_schedule.load_checkpoint(manager.selected_path)
    with pytest.raises(ValueError, match="method"):
        trainer("H").load_checkpoint(manager.selected_path)


def actor_pair_registry(root):
    registry = independent_registry_fixture(root)
    for family in registry["families"]:
        task = family["tasks"][0]
        control = task["methods"][0]
        old_fit = control["fit_output"]
        control["method"] = "H-ActorTMS"
        control["fit_output"] = str(root / family["family_id"] / "H-ActorTMS" / "fit")
        fit = control["fit_argv"]
        fit[fit.index("--method") + 1] = "H-ActorTMS"
        fit[fit.index("--updates") + 1] = "1000000"
        fit[fit.index("--output") + 1] = control["fit_output"]
        for evaluation in control["evaluations"]:
            argv = evaluation["argv"]
            argv[argv.index("--controller-method") + 1] = "H-ActorTMS"
            argv[argv.index("--controller-checkpoint") + 1] = str(Path(control["fit_output"]) / "selected.pt")
        reference = copy.deepcopy(control)
        reference["method"] = "H"
        reference["fit_output"] = old_fit
        reference["schedule_artifacts"] = {}
        fit = reference["fit_argv"]
        fit[fit.index("--method") + 1] = "H"
        fit[fit.index("--output") + 1] = old_fit
        del fit[fit.index("--tms-schedule"):fit.index("--tms-schedule") + 2]
        for evaluation in reference["evaluations"]:
            argv = evaluation["argv"]
            argv[argv.index("--controller-method") + 1] = "H"
            argv[argv.index("--controller-checkpoint") + 1] = str(Path(old_fit) / "selected.pt")
            evaluation["output"] += "-matched-H"
            argv[argv.index("--output") + 1] = evaluation["output"]
            del argv[argv.index("--tms-schedule"):argv.index("--tms-schedule") + 2]
        task["methods"] = [reference, control]
        task["actor_tms_matched_pair"] = True
    return registry


def test_explicit_actor_pair_protocol_binds_fixed_schedule_and_same_seed(tmp_path):
    registry = actor_pair_registry(tmp_path)
    plan = build_plan(registry, tmp_path / "plan")
    assert len(plan["jobs"]) == 80
    fits = [job for job in plan["jobs"] if job["kind"] == "controller_fit"]
    for left, right in zip(fits[::2], fits[1::2]):
        assert left["seed"] == right["seed"]
        assert left["binding"]["replay"] == right["binding"]["replay"]
        assert left["binding"]["actor_tms_matched_pair"] is True
    assert plan["reproduces_historical_results"] is False
    changed = copy.deepcopy(registry)
    changed["families"][0]["tasks"][0]["actor_tms_matched_pair"] = False
    with pytest.raises(ValueError, match="actor_tms_matched_pair"):
        build_plan(changed, tmp_path / "plan")
    changed = copy.deepcopy(registry)
    changed["families"][0]["tasks"][0]["methods"][0]["seed"] += 1000
    with pytest.raises(ValueError, match="share one"):
        build_plan(changed, tmp_path / "plan")
    changed = copy.deepcopy(registry)
    first_seed = changed["families"][0]["tasks"][0]["methods"][0]["seed"]
    for method in changed["families"][1]["tasks"][0]["methods"]:
        method["seed"] = first_seed
    with pytest.raises(ValueError, match="distinct across families"):
        build_plan(changed, tmp_path / "plan")
    artifact = registry["families"][0]["tasks"][0]["methods"][1]["schedule_artifacts"]["--tms-schedule"]
    path = Path(artifact["path"])
    schedule = json.loads(path.read_text())
    schedule["development"]["selected_h_checkpoint_sha256"] = "0" * 64
    path.write_text(json.dumps(schedule))
    artifact["sha256"] = file_sha256(path)
    with pytest.raises(ValueError, match="not derived from the bound selected H"):
        build_plan(registry, tmp_path / "plan")


def test_actor_tms_results_can_be_summarized_without_becoming_historical(tmp_path):
    source = tmp_path / "run"
    run, rows = write_run(source)
    run["controller_method"] = "H-ActorTMS"
    for row in rows:
        row["controller_method"] = "H-ActorTMS"
    save_run(source, run, rows)
    summary = build_summary(source)
    assert summary["identity"]["controller_method"] == "H-ActorTMS"
    assert summary["reproduces_historical_results"] is False
