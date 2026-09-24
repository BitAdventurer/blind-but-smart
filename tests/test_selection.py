"""Synthetic callbacks test selection arithmetic; these are not paper results."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from gui_joint_control.selection import (
    CheckpointManager, DevelopmentScore, score_development, select_retrieval_threshold,
)


MANIFEST = "a" * 64


def episode(trajectory="one", replicate="r1", invoked=(0,), exhausted=(), correct=True):
    records = [{"slot": slot, "eligible": slot in invoked or slot in exhausted,
                "invoked": slot in invoked, "correct": correct,
                "executed_budgets": [1.5] * 25 if slot in invoked else None,
                "candidate_count": 1 if slot in invoked else 0} for slot in range(56)]
    return {"trajectory_id": trajectory, "replicate": replicate, "records": records}


def result(episodes=None):
    return {"split": "development", "manifest_sha256": MANIFEST,
            "episodes": [episode()] if episodes is None else episodes}


class FakeTrainer:
    """Only writes a tiny fixture checkpoint; never loads torch or trains a model."""
    method = "Independent"
    seed = 27
    config_sha256 = "b" * 64

    def __init__(self, iteration):
        self.iteration = iteration

    def save_checkpoint(self, path):
        Path(path).write_text(json.dumps({"fixture_only": True, "iteration": self.iteration}), encoding="utf-8")


class DevelopmentScoringTests(unittest.TestCase):
    def test_eligible_clock_skips_ineligible_and_penalizes_exhausted(self):
        score = score_development([episode(invoked=(0,), exhausted=(3,))])
        self.assertAlmostEqual(score.mean_return, 1.5 - .99)
        self.assertEqual(score.mean_regional_budget, .75)
        self.assertEqual(score.mean_candidate_count, .5)
        self.assertEqual(score.eligible_slots, 2)

    def test_equal_trajectory_and_replicate_weights_are_not_step_weights(self):
        score = score_development([
            episode("short", "r1"), episode("long", "r1", invoked=(0, 1)),
            episode("short", "r2", correct=False), episode("long", "r2", invoked=(0, 1), correct=False),
        ])
        expected = ((1.5 + 1.5 * 1.99) / 2 + (.5 + .5 * 1.99) / 2) / 2
        self.assertAlmostEqual(score.mean_return, expected)
        self.assertEqual((score.trajectories, score.replicates, score.eligible_slots), (2, 2, 6))

    def test_replay_tail_adjustment_is_not_used_twice(self):
        item = episode(invoked=(0,), exhausted=(1,))
        item["records"][0]["reward"] = -99  # not the checkpoint-return source
        self.assertAlmostEqual(score_development([item]).mean_return, .51)

    def test_invalid_grids_scores_and_missing_slots_fail(self):
        bad = [episode(), episode("different", "r2")]
        with self.assertRaisesRegex(ValueError, "share trajectory"):
            score_development(bad)
        for transform in (
            lambda row: row["records"].pop(),
            lambda row: row["records"][0].update(executed_budgets=[math.nan] * 25),
            lambda row: row["records"][0].update(candidate_count=True),
        ):
            row = episode()
            transform(row)
            with self.assertRaises(ValueError):
                score_development([row])
        with self.assertRaisesRegex(ValueError, "nonempty"):
            score_development([episode(invoked=())])

    def test_exact_tie_order(self):
        base = dict(mean_return=1., mean_regional_budget=2., mean_candidate_count=3.,
                    eligible_slots=1, trajectories=1, replicates=1, evaluation_grid_sha256=MANIFEST)
        current = DevelopmentScore(**base)
        self.assertLess(DevelopmentScore(**{**base, "mean_return": 2}).ranking_key(100), current.ranking_key(1))
        self.assertLess(DevelopmentScore(**{**base, "mean_regional_budget": 1}).ranking_key(100), current.ranking_key(1))
        self.assertLess(DevelopmentScore(**{**base, "mean_candidate_count": 2}).ranking_key(100), current.ranking_key(1))
        self.assertLess(current.ranking_key(1), current.ranking_key(100))


class SelectionManagerTests(unittest.TestCase):
    def test_selected_and_last_differ_and_resume_preserves_best(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = CheckpointManager(directory, binding={"dev_manifest_sha256": MANIFEST}, interval=2)
            trainer = FakeTrainer(1)
            self.assertIsNone(manager.consider(trainer, lambda _: self.fail("Must not evaluate before interval")))
            trainer.iteration = 2
            self.assertTrue(manager.consider(trainer, lambda _: result())["selected"])
            trainer.iteration = 4
            self.assertFalse(manager.consider(trainer, lambda _: result([episode(correct=False)]))["selected"])
            self.assertEqual(json.loads(manager.selected_path.read_text())["iteration"], 2)
            self.assertEqual(json.loads(manager.last_path.read_text())["iteration"], 4)
            restored = CheckpointManager(directory, binding={"dev_manifest_sha256": MANIFEST}, interval=2, resume=True)
            restored.validate_resume(trainer)
            trainer.iteration = 6
            self.assertFalse(restored.consider(trainer, lambda _: result())["selected"])
            self.assertEqual(restored.best["iteration"], 2)

    def test_bench_wrong_manifest_skipped_interval_and_changed_grid_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = CheckpointManager(directory, binding={"dev_manifest_sha256": MANIFEST}, interval=2)
            for wrong in ({**result(), "split": "test"}, {**result(), "manifest_sha256": "b" * 64}):
                with self.assertRaises(ValueError):
                    manager.consider(FakeTrainer(2), lambda _: wrong)
            with self.assertRaisesRegex(ValueError, "skipped"):
                manager.consider(FakeTrainer(4), lambda _: result())
            manager.consider(FakeTrainer(2), lambda _: result())
            with self.assertRaisesRegex(ValueError, "grid changed"):
                manager.consider(FakeTrainer(4), lambda _: result([episode("other")]))

    def test_resume_rejects_tampering_wrong_seed_and_selected_instead_of_last(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = CheckpointManager(directory, binding={"dev_manifest_sha256": MANIFEST}, interval=2)
            manager.consider(FakeTrainer(2), lambda _: result())
            trainer = FakeTrainer(3)
            manager.save_last(trainer)
            with self.assertRaises(ValueError):
                manager.validate_resume(FakeTrainer(2))
            trainer.seed = 999
            with self.assertRaises(ValueError):
                manager.validate_resume(trainer)
            manager.selected_path.write_text("changed")
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                CheckpointManager(directory, binding={"dev_manifest_sha256": MANIFEST}, interval=2, resume=True)

    def test_no_dev_check_means_no_selected_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = CheckpointManager(directory, binding={"dev_manifest_sha256": MANIFEST})
            manager.save_last(FakeTrainer(12))
            self.assertIsNone(manager.best)
            self.assertFalse(manager.selected_path.exists())

    def test_retrieval_grid_and_larger_threshold_tie(self):
        seen = []
        selected = select_retrieval_threshold(lambda threshold: seen.append(threshold) or result(), manifest_sha256=MANIFEST)
        self.assertEqual(seen, [index / 4 for index in range(9)])
        self.assertEqual(selected["selected_threshold"], 2.)
        with self.assertRaises(ValueError):
            select_retrieval_threshold(lambda _: result([]), manifest_sha256=MANIFEST)


if __name__ == "__main__":
    unittest.main()
