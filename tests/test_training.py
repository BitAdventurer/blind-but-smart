"""Software fixtures test the new trainer; no fixture is an experiment result."""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from gui_joint_control.replay import ReplayBuffer
from gui_joint_control.trainer import Trainer, filter_budgets
from gui_joint_control.tms import TMSSchedule, build_tms_schedule


def fixture_arrays(n=20):
    """Artificial transition data used only for deterministic software tests."""
    rng = np.random.default_rng(14)
    terminal = np.arange(n) % 3 == 0
    arrays = {
        "observation": rng.normal(size=(n, 28)).astype(np.float32),
        "executed_budgets": rng.uniform(1.5, 3.0, (n, 25)).astype(np.float32),
        "candidate_count": rng.integers(1, 21, n, dtype=np.int64),
        "reward": rng.uniform(-0.1, 1.5, n).astype(np.float32),
        "next_observation": rng.normal(size=(n, 28)).astype(np.float32),
        "terminal": terminal,
        "remaining_budget": np.full(n, 125.0, dtype=np.float32),
        "next_remaining_budget": np.where(terminal, 0.0, 50.0).astype(np.float32),
        "success": np.arange(n) % 2 == 0,
        "slot_id": np.array([f"fixture-{i}" for i in range(n)]),
        "next_slot_id": np.array(["" if terminal[i] else f"fixture-next-{i}" for i in range(n)]),
    }
    arrays["observation"][:, 27] = 1 / 56
    arrays["next_observation"][:, 27] = 2 / 56
    return arrays


def fixture_schedule(n=20, mean_count=5.5, budget=3.2):
    """Artificial TMS means and provenance used solely for software tests."""
    return build_tms_schedule(
        task="G", family="fixture-family", population="software-test",
        population_manifest_sha256="1" * 64, public_hash_seed="2" * 64,
        slots=[{"slot_id": f"fixture-{i}", "original_step": 1} for i in range(n)]
        + [{"slot_id": f"fixture-next-{i}", "original_step": 2} for i in range(n)],
        mean_regional_budget=budget, mean_candidate_count=mean_count,
        selected_h_checkpoint_sha256="3" * 64, development_manifest_sha256="4" * 64)


def config():
    return {
        "profile": "software_test_fixture_not_an_experiment",
        "training": {"replay_batch_size": 8},
    }


class ReplayTests(unittest.TestCase):
    def test_npz_is_immutable_and_hashed(self):
        arrays = fixture_arrays()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.npz"
            np.savez(path, **arrays)
            replay = ReplayBuffer.from_npz(path)
            self.assertEqual(len(replay.source_sha256), 64)
            self.assertEqual(len(replay.content_sha256), 64)
            self.assertEqual(replay.content_sha256, ReplayBuffer(arrays).content_sha256)
            with self.assertRaises(ValueError):
                replay.arrays["observation"][0, 0] = 1
            with self.assertRaises(ValueError):
                replay.arrays["observation"].setflags(write=True)
            with self.assertRaises(TypeError):
                replay.arrays["reward"] = np.ones(20)
            before = replay.arrays["observation"][0, 0]
            arrays["observation"][0, 0] = 99
            self.assertEqual(replay.arrays["observation"][0, 0], before)

    def test_sampling_stratum_then_uniform_and_state_restoration(self):
        replay = ReplayBuffer(fixture_arrays())
        rng = np.random.default_rng(52)
        state = copy.deepcopy(rng.bit_generator.state)
        draw = replay.sample_indices(30_000, rng)
        proportion = replay.arrays["success"][draw].mean()
        self.assertLess(abs(proportion - 0.7), 0.015)
        rng.bit_generator.state = state
        np.testing.assert_array_equal(draw, replay.sample_indices(30_000, rng))
        self.assertEqual(len(np.unique(draw)), len(replay))

    def test_malformed_replay_is_rejected(self):
        for field, change, message in (
            ("observation", lambda a: np.zeros((20, 27)), "shape"),
            ("reward", lambda a: np.full(20, np.nan), "finite"),
            ("success", lambda a: np.ones(20, dtype=bool), "Both success"),
            ("terminal", lambda a: a.astype(int), "boolean"),
            ("candidate_count", lambda a: np.ones(20, dtype=float), "integer"),
            ("candidate_count", lambda a: np.full(20, 21), "1..20"),
            ("executed_budgets", lambda a: np.ones_like(a), "1.5"),
            ("remaining_budget", lambda a: np.full(20, 37.4), "37.5"),
            ("next_remaining_budget", lambda a: np.full(20, 36.0), "invocable"),
        ):
            with self.subTest(field=field, message=message):
                arrays = fixture_arrays()
                arrays[field] = change(arrays[field])
                with self.assertRaisesRegex(ValueError, message):
                    ReplayBuffer(arrays)
        arrays = fixture_arrays()
        del arrays["terminal"]
        with self.assertRaisesRegex(ValueError, "missing"):
            ReplayBuffer(arrays)

    def test_pickle_object_replay_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "object.npz"
            np.savez(path, **fixture_arrays(), unsafe=np.array([{}], dtype=object))
            with self.assertRaises(ValueError):
                ReplayBuffer.from_npz(path)

    def test_executed_cost_must_be_affordable(self):
        arrays = fixture_arrays()
        arrays["remaining_budget"][:] = 40
        arrays["next_remaining_budget"][:] = 0
        arrays["terminal"][:] = True
        with self.assertRaisesRegex(ValueError, "exceed"):
            ReplayBuffer(arrays)

    def test_absolute_ledger_metadata_retains_float64_precision(self):
        arrays = fixture_arrays()
        remaining = np.nextafter(100.0, -np.inf)
        next_remaining = np.nextafter(50.0, -np.inf)
        arrays["remaining_budget"] = np.full(20, remaining, dtype=np.float64)
        arrays["next_remaining_budget"] = np.full(20, next_remaining, dtype=np.float64)
        replay = ReplayBuffer(arrays)
        for key, expected in (("remaining_budget", remaining), ("next_remaining_budget", next_remaining)):
            self.assertEqual(replay.arrays[key].dtype, np.dtype(np.float64))
            self.assertEqual(float(replay.arrays[key][0]), expected)
        self.assertEqual(replay.arrays["observation"].dtype, np.dtype(np.float32))
        self.assertEqual(replay.arrays["executed_budgets"].dtype, np.dtype(np.float32))


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Small CPU fixtures are faster and bitwise stable with one worker.
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def make_trainer(self, method="H", arrays=None, seed=123):
        return Trainer(config(), method, ReplayBuffer(fixture_arrays() if arrays is None else arrays), seed=seed,
                       tms_schedule=fixture_schedule(), task="G", family="fixture-family")

    def tensor_batch(self, arrays=None):
        return {key: value if key in ("slot_id", "next_slot_id") else torch.from_numpy(value)
                for key, value in (fixture_arrays() if arrays is None else arrays).items()}

    def test_filter_affordability_fallback_and_gradient_boundary(self):
        proposals = torch.full((2, 25), 4.0, requires_grad=True)
        filtered = filter_budgets(proposals, torch.tensor([100.0, 50.0]))
        torch.testing.assert_close(filtered[0], torch.full((25,), 4.0))
        torch.testing.assert_close(filtered[1], torch.full((25,), 1.5))
        self.assertFalse(filtered.requires_grad)
        with self.assertRaisesRegex(ValueError, "37.5"):
            filter_budgets(proposals, torch.tensor([100.0, 37.4]))

    def test_boundary_filter_matches_deployed_float64_ledger(self):
        from gui_joint_control.privacy import filter_budgets as ledger_filter
        proposals = torch.full((3, 25), 4.0, dtype=torch.float32)
        remaining = np.array([np.nextafter(100.0, -np.inf), 100.0,
                              np.nextafter(100.0, np.inf)], dtype=np.float64)
        result = filter_budgets(proposals, torch.from_numpy(remaining))
        self.assertEqual(result.dtype, torch.float32)
        for index in range(3):
            expected, _ = ledger_filter(proposals[index].numpy(), remaining[index])
            np.testing.assert_array_equal(result[index].numpy(), expected)
        self.assertTrue(torch.all(result[0] == 1.5))
        self.assertTrue(torch.all(result[1:] == 4.0))
        trainer = self.make_trainer()
        fixed = {"budgets": proposals[:1], "candidate_count": torch.ones(1, dtype=torch.long)}
        with patch.object(trainer.bundles["joint"]["actor"], "evaluation_action", return_value=fixed):
            evaluated = trainer.evaluation_action(np.zeros((1, 28)), remaining[:1])
        self.assertTrue(torch.all(evaluated["executed_budgets"] == 1.5))

    def test_real_h_update_changes_actor_critics_and_polyak_targets(self):
        trainer = self.make_trainer()
        bundle = trainer.bundles["joint"]
        before = {key: [p.detach().clone() for p in bundle[key].parameters()] for key in ("actor", "critic1", "critic2", "target1", "target2")}
        metrics = trainer.step(batch_size=8)
        self.assertEqual(metrics["component_updates"], {"joint": 1})
        for key in ("actor", "critic1", "critic2"):
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before[key], bundle[key].parameters())))
        for source, target in (("critic1", "target1"), ("critic2", "target2")):
            for previous, online, actual in zip(before[target], bundle[source].parameters(), bundle[target].parameters()):
                torch.testing.assert_close(actual, previous * 0.995 + online.detach() * 0.005)
                self.assertFalse(actual.requires_grad)
        self.assertTrue(all(p.grad is None for p in bundle["critic1"].parameters()))
        self.assertTrue(all(np.isfinite(value) for value in metrics["components"]["joint"].values()))
        self.assertEqual(metrics["components"]["joint"]["gumbel_temperature"], 1.0)

    def test_update_order_is_critic_actor_target(self):
        from gui_joint_control.trainer import polyak_update
        trainer = self.make_trainer()
        bundle = trainer.bundles["joint"]
        events = []
        critic_step, actor_step = bundle["critic_optimizer"].step, bundle["actor_optimizer"].step
        def critic(*args, **kwargs):
            events.append("critic")
            return critic_step(*args, **kwargs)
        def actor(*args, **kwargs):
            events.append("actor")
            self.assertFalse(any(p.requires_grad for p in bundle["critic1"].parameters()))
            return actor_step(*args, **kwargs)
        def target(*args, **kwargs):
            events.append("target")
            return polyak_update(*args, **kwargs)
        with patch.object(bundle["critic_optimizer"], "step", side_effect=critic), patch.object(bundle["actor_optimizer"], "step", side_effect=actor), patch("gui_joint_control.trainer.polyak_update", side_effect=target):
            trainer.step(4)
        self.assertEqual(events, ["critic", "actor", "target"])

    def test_terminal_and_cb_targets_never_sample_successor(self):
        for method, terminal in (("H", True), ("CB", False)):
            with self.subTest(method=method):
                arrays = fixture_arrays()
                arrays["terminal"][:] = terminal
                arrays["next_remaining_budget"][:] = 0 if terminal else 50
                arrays["next_slot_id"] = np.array(["" if terminal else f"fixture-next-{i}" for i in range(len(arrays["terminal"]))])
                trainer = self.make_trainer(method, arrays)
                batch = self.tensor_batch(arrays)
                with patch.object(trainer, "_action", side_effect=AssertionError("Successor must not be sampled")):
                    target = trainer._target(trainer.bundles["joint"], batch, 1)
                torch.testing.assert_close(target, batch["reward"], rtol=0, atol=0)
                trainer.step(8)

    def test_only_nonterminal_rows_enter_bellman_bootstrap(self):
        trainer = self.make_trainer()
        batch = self.tensor_batch()
        bundle = trainer.bundles["joint"]
        with torch.no_grad():
            for name, bias in (("target1", 3), ("target2", 4)):
                for parameter in bundle[name].parameters():
                    parameter.zero_()
                bundle[name].network[-1].bias.fill_(bias)
        calls = []

        def fake_action(bundle, observation, remaining, **kwargs):
            calls.append(len(observation))
            self.assertTrue(torch.all(remaining >= 37.5))
            return {"executed_budgets": torch.full((len(observation), 25), 1.5),
                    "count_one_hot": torch.nn.functional.one_hot(torch.zeros(len(observation), dtype=torch.long), 20).float()}

        with patch.object(trainer, "_action", side_effect=fake_action):
            target = trainer._target(bundle, batch, 1)
        self.assertEqual(calls, [int((~batch["terminal"]).sum())])
        expected = batch["reward"] + (~batch["terminal"]).float() * (0.99 * 3)
        torch.testing.assert_close(target, expected)

    def test_independent_heads_are_separately_fitted_then_composed(self):
        trainer = self.make_trainer("Independent")
        count_before = [p.detach().clone() for p in trainer.bundles["count"]["actor"].parameters()]
        with patch.object(trainer.bundles["count"]["actor"], "sample", side_effect=AssertionError("Other actor cannot enter disclosure training")):
            trainer._update_component("disclosure", self.tensor_batch())
        for before, after in zip(count_before, trainer.bundles["count"]["actor"].parameters()):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        # A fresh trainer is synchronized and checkpointable after full steps.
        trainer = self.make_trainer("Independent")
        metrics = trainer.step(8)
        self.assertEqual(metrics["aggregate_replay_iterations"], 2)
        self.assertEqual(metrics["aggregate_optimizer_calls"], 4)
        self.assertEqual(metrics["component_updates"], {"disclosure": 1, "count": 1})
        observations = torch.zeros((3, 28))
        result = trainer.evaluation_action(observations, torch.full((3,), 125.0))
        with torch.no_grad():
            expected = trainer.bundles["disclosure"]["actor"].evaluation_action(observations)["budgets"]
            expected_count = trainer.bundles["count"]["actor"].evaluation_action(observations)["candidate_count"]
        torch.testing.assert_close(result["executed_budgets"], expected)
        torch.testing.assert_close(result["candidate_count"], expected_count)

    def test_missing_fixed_schedule_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "TMS schedule"):
            Trainer({}, "Independent", ReplayBuffer(fixture_arrays()))

    def test_tms_current_and_successor_use_bound_ids_and_actual_filter(self):
        trainer = self.make_trainer("Independent")
        batch = self.tensor_batch()
        original_rewards = trainer.buffer.arrays["reward"].copy()
        count_bundle = trainer.bundles["count"]
        current = trainer._action(count_bundle, batch["observation"][:2], torch.tensor([125., 50.], dtype=torch.float64), update_index=0, slot_ids=batch["slot_id"][:2])
        self.assertTrue(torch.all(current["executed_budgets"][0] == 3.2))
        self.assertTrue(torch.all(current["executed_budgets"][1] == 1.5))
        calls = []
        original_action = trainer._action
        def record(*args, **kwargs):
            result = original_action(*args, **kwargs)
            calls.append((kwargs.get("slot_ids"), result))
            return result
        with patch.object(trainer, "_action", side_effect=record):
            trainer._target(count_bundle, batch, 0)
        expected_ids = batch["next_slot_id"][~batch["terminal"].numpy()]
        np.testing.assert_array_equal(calls[0][0], expected_ids)
        self.assertTrue(torch.all(calls[0][1]["executed_budgets"] == 1.5))
        np.testing.assert_array_equal(trainer.buffer.arrays["reward"], original_rewards)
        disclosure = trainer._action(trainer.bundles["disclosure"], batch["observation"][:2], batch["remaining_budget"][:2], update_index=0, slot_ids=batch["slot_id"][:2])
        expected_counts = trainer.tms_schedule.proposals(batch["slot_id"][:2])[1]
        np.testing.assert_array_equal(disclosure["candidate_count"].numpy(), expected_counts)

    def test_single_head_evaluation_can_bind_different_population(self):
        trainer = self.make_trainer("Count-only")
        artifact = fixture_schedule(budget=4.0).to_dict()
        artifact["population"] = "evaluation-software-test"
        schedule = TMSSchedule(artifact)
        obs = self.tensor_batch()["observation"][:1]
        action = trainer.proposal_action(obs, slot_ids=np.array(["fixture-0"]), tms_schedule=schedule)
        self.assertTrue(torch.all(action["budgets"] == 4.0))
        artifact["family"] = "other-family"
        with self.assertRaisesRegex(ValueError, "task/family"):
            trainer.proposal_action(obs, slot_ids=np.array(["fixture-0"]), tms_schedule=TMSSchedule(artifact))

    def test_tms_hash_is_bound_to_checkpoint(self):
        trainer = self.make_trainer("Independent")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            trainer.save_checkpoint(path)
            changed = Trainer(config(), "Independent", trainer.buffer, tms_schedule=fixture_schedule(budget=3.4), task="G", family="fixture-family")
            with self.assertRaisesRegex(ValueError, "tms_schedule_sha256"):
                changed.load_checkpoint(path)

    def test_single_head_variants_update(self):
        for method in ("Disclosure-only", "Count-only", "Independent-1M"):
            with self.subTest(method=method):
                trainer = self.make_trainer(method)
                self.assertEqual(trainer.step(8)["iteration"], 1)
                obs = np.zeros((2, 28), np.float32)
                obs[:, 27] = 1 / 56
                result = trainer.evaluation_action(obs, np.full(2, 100.0), slot_ids=np.array(["fixture-0", "fixture-1"]))
                self.assertEqual(result["candidate_count"].shape, (2,))

    def test_checkpoint_resume_equals_uninterrupted_cpu_training(self):
        for method in ("H", "CB", "Independent"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                uninterrupted = self.make_trainer(method)
                for _ in range(4):
                    full_metrics = uninterrupted.step(8)
                interrupted = self.make_trainer(method)
                for _ in range(2):
                    interrupted.step(8)
                path = Path(directory) / "resume.pt"
                interrupted.save_checkpoint(path)
                restored = self.make_trainer(method, seed=999)
                restored.load_checkpoint(path)
                for _ in range(2):
                    resumed_metrics = restored.step(8)
                self.assertEqual(full_metrics, resumed_metrics)
                self.assertEqual(restored.sampler.bit_generator.state, uninterrupted.sampler.bit_generator.state)
                for component in uninterrupted.bundles:
                    for module in ("actor", "critic1", "critic2", "target1", "target2"):
                        for key, tensor in uninterrupted.bundles[component][module].state_dict().items():
                            torch.testing.assert_close(tensor, restored.bundles[component][module].state_dict()[key], rtol=0, atol=0)

    def test_checkpoint_rejects_different_data_method_or_configuration(self):
        trainer = self.make_trainer()
        trainer.step(8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            trainer.save_checkpoint(path)
            with self.assertRaisesRegex(ValueError, "method"):
                self.make_trainer("CB").load_checkpoint(path)
            changed = fixture_arrays()
            changed["reward"][0] += 0.1
            with self.assertRaisesRegex(ValueError, "replay_content"):
                self.make_trainer(arrays=changed).load_checkpoint(path)
            altered_config = config()
            altered_config["run_id"] = "different-config-binding"
            with self.assertRaisesRegex(ValueError, "config_sha256"):
                Trainer(altered_config, "H", trainer.buffer, tms_schedule=fixture_schedule(), task="G", family="fixture-family").load_checkpoint(path)

    def test_trainer_owned_rng_does_not_consume_global_stream(self):
        global_before = torch.get_rng_state().clone()
        trainer = self.make_trainer()
        torch.testing.assert_close(global_before, torch.get_rng_state(), rtol=0, atol=0)
        trainer.step(8)
        torch.testing.assert_close(global_before, torch.get_rng_state(), rtol=0, atol=0)


class TMSScheduleTests(unittest.TestCase):
    def test_global_floor_ceiling_half_up_and_stable_assignments(self):
        schedule = fixture_schedule(n=5, mean_count=5.25)
        ids = np.array([f"fixture-{i}" for i in range(5)] + [f"fixture-next-{i}" for i in range(5)])
        budgets, counts = schedule.proposals(ids)
        self.assertEqual(int((counts == 6).sum()), 3)  # round-half-up(10 * .25)
        self.assertTrue(np.all(np.isin(counts, [5, 6])))
        self.assertTrue(np.all(budgets == 3.2))
        reverse_artifact = schedule.to_dict()
        reverse_artifact["slots"].reverse()
        reordered = TMSSchedule(reverse_artifact)
        self.assertEqual(schedule.content_sha256, reordered.content_sha256)
        np.testing.assert_array_equal(counts[::-1], reordered.proposals(ids[::-1])[1])
        self.assertLessEqual(abs(float(counts.mean()) - 5.25), 1 / len(ids))

    def test_missing_ids_time_and_provenance_fail_instead_of_fallback(self):
        schedule = fixture_schedule()
        with self.assertRaisesRegex(ValueError, "every requested"):
            schedule.proposals(np.array(["missing-id"]))
        with self.assertRaisesRegex(ValueError, "observation time"):
            schedule.proposals(np.array(["fixture-0"]), np.array([2 / 56]))
        artifact = schedule.to_dict()
        del artifact["development"]["selected_h_checkpoint_sha256"]
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            TMSSchedule(artifact)


if __name__ == "__main__":
    unittest.main()
