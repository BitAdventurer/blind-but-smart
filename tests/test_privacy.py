"""Mathematical and access-boundary tests for the new privacy runtime."""

import dataclasses
import math
import unittest

import numpy as np
from scipy.special import ndtr

from gui_joint_control.privacy import (
    DELTA_STAGE, EXEC_FALLBACK, EXEC_PROP, FILTER_EXHAUSTED, STRUCTURAL_PAD,
    TASK_PAD, PrivacyLedger, analytic_gaussian_sigma, clip_features,
    filter_budgets, probe_entropy, protected_observation,
)


def zero_features():
    return np.zeros((25, 256))


def ledger(mask, recorded=None):
    return PrivacyLedger(mask, recorded, probe_rng=np.random.default_rng(31), refinement_rng=np.random.default_rng(97))


class CalibrationTests(unittest.TestCase):
    def test_manuscript_scales_and_calibration_equation(self):
        for eps, reference, places in [(0.5, 0.10232, 5), (1, 0.0528, 4), (4, 0.0144, 4)]:
            sigma = analytic_gaussian_sigma(eps)
            self.assertAlmostEqual(sigma, reference, places=places)
            a, b = .01 / (2 * sigma), eps * sigma / .01
            attained = ndtr(a - b) - math.exp(eps) * ndtr(-a - b)
            self.assertLessEqual(attained, DELTA_STAGE * (1 + 1e-9))
            self.assertAlmostEqual(attained / DELTA_STAGE, 1, places=8)

    def test_sensitivity_scaling_and_budget_monotonicity(self):
        scales = [analytic_gaussian_sigma(e) for e in [.5, 1, 2, 4]]
        self.assertTrue(all(a > b for a, b in zip(scales, scales[1:])))
        self.assertAlmostEqual(analytic_gaussian_sigma(1, sensitivity=.02), 2 * scales[1])
        self.assertGreater(analytic_gaussian_sigma(1, delta=1e-12), scales[1])

    def test_zero_epsilon_and_large_epsilon_are_supported(self):
        sigma = analytic_gaussian_sigma(0, delta=0.01)
        self.assertAlmostEqual(2 * ndtr(.01 / (2 * sigma)) - 1, .01, places=12)
        self.assertTrue(np.isfinite(analytic_gaussian_sigma(1000, delta=1e-20)))

    def test_bad_calibration_inputs(self):
        for args in [(-1, .1, .01), (1, 0, .01), (1, 1, .01), (1, .1, 0), (np.nan, .1, .01), (1, np.inf, .01)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                analytic_gaussian_sigma(*args)


class FeatureAndFilterTests(unittest.TestCase):
    def test_clipping_preserves_small_vectors_and_clips_large(self):
        original = zero_features()
        original[0, 0] = .4
        original[1, :2] = [3, 4]
        original[2] = 1e308
        clipped = clip_features(original)
        np.testing.assert_array_equal(clipped[0], original[0])
        np.testing.assert_allclose(clipped[1, :2], [.6, .8])
        self.assertTrue(np.isfinite(clipped).all())
        self.assertTrue(np.all(np.linalg.norm(clipped, axis=1) <= 1 + 1e-15))
        self.assertFalse(np.shares_memory(clipped, original))

    def test_entropy_known_cases_and_scale_invariance(self):
        probe = zero_features()
        probe[1, 0] = 2
        probe[2] = 1
        entropy = probe_entropy(probe)
        self.assertEqual(entropy[0], 0)
        self.assertEqual(entropy[1], 0)
        self.assertAlmostEqual(entropy[2], 1)
        np.testing.assert_allclose(entropy, probe_entropy(-probe * 1e200))

    def test_observation_shape_and_normalized_time(self):
        obs = protected_observation(zero_features(), .7, .8, 0)
        self.assertEqual(obs.shape, (28,))
        np.testing.assert_allclose(obs[-3:], [.7, .8, 1 / 56])
        self.assertEqual(protected_observation(zero_features(), 2, 0, 55)[-1], 1)
        for values in [(3, .8, 0), (.7, -.1, 0), (.7, .8, 56), (.7, .8, True)]:
            with self.assertRaises(ValueError):
                protected_observation(zero_features(), *values)

    def test_filter_fitting_fallback_exhausted_and_invalid(self):
        proposal = np.full(25, 3.0)
        executed, status = filter_budgets(proposal, 75)
        self.assertEqual(status, EXEC_PROP)
        np.testing.assert_array_equal(executed, proposal)
        self.assertFalse(np.shares_memory(executed, proposal))
        for proposed in [np.full(25, 3), np.full(25, 5), np.linspace(1.5, 5, 25)]:
            executed, status = filter_budgets(proposed, 40)
            self.assertEqual(status, EXEC_FALLBACK)
            np.testing.assert_array_equal(executed, np.full(25, 1.5))
        self.assertEqual(filter_budgets(proposal, np.nextafter(37.5, 0)), (None, FILTER_EXHAUSTED))
        self.assertEqual(filter_budgets(proposal, 37.5)[1], EXEC_FALLBACK)
        for bad in [np.ones(25), np.full(25, 6), np.full(25, np.nan), np.ones(24)]:
            with self.assertRaises(ValueError):
                filter_budgets(bad, 75)

    def test_features_reject_wrong_shape_and_nonfinite(self):
        for bad in [np.zeros((24, 256)), np.zeros((25, 255)), np.full((25, 256), np.inf)]:
            with self.assertRaises(ValueError):
                clip_features(bad)
            with self.assertRaises(ValueError):
                probe_entropy(bad)


class LedgerTests(unittest.TestCase):
    def test_admission_precedes_every_private_callback(self):
        calls = []
        mechanism = ledger([True, True, True])
        def load():
            calls.append("features")
            return zero_features()
        def allocate(obs):
            calls.append("allocator")
            self.assertEqual(obs.shape, (28,))
            self.assertFalse(obs.flags.writeable)
            return np.full(25, 5), 4
        first, second, third = [mechanism.release_step(load, allocate) for _ in range(3)]
        self.assertEqual([first.status, second.status, third.status], [EXEC_PROP, EXEC_FALLBACK, EXEC_FALLBACK])
        self.assertEqual(calls, ["features", "allocator"] * 3)
        self.assertEqual(mechanism.used_budget, 200)
        self.assertEqual(mechanism.remaining_budget, 25)

        exhausted = ledger([True] * 4)
        rows = [exhausted.release_step(load, allocate) for _ in range(4)]
        self.assertEqual([x.status for x in rows], [EXEC_PROP, EXEC_PROP, EXEC_FALLBACK, FILTER_EXHAUSTED])
        self.assertEqual(exhausted.used_budget, 287.5)
        self.assertEqual(exhausted.remaining_budget, 12.5)
        self.assertEqual(calls.count("features"), 6)
        self.assertFalse(rows[-1].invoked)
        self.assertIsNone(rows[-1].release)

    def test_padding_reads_nothing_carries_budget_and_always_56_slots(self):
        mechanism = ledger([False, False], [True, False])
        def forbidden(*args):
            self.fail("padding accessed a private callback")
        rows = [mechanism.release_step(forbidden, forbidden) for _ in range(56)]
        self.assertEqual(rows[0].status, TASK_PAD)
        self.assertTrue(all(x.status == STRUCTURAL_PAD for x in rows[1:]))
        self.assertTrue(all(x.used_budget == 0 and not x.invoked and x.k == 0 for x in rows))
        self.assertEqual(len(mechanism.records), 56)
        with self.assertRaises(RuntimeError):
            mechanism.release_step(forbidden, forbidden)

    def test_full_trajectory_does_not_overspend(self):
        mechanism = ledger([True] * 56)
        rows = [mechanism.release_step(zero_features, lambda obs: (np.full(25, 5), 20)) for _ in range(56)]
        self.assertEqual(mechanism.cap, 4200)
        self.assertTrue(all(r.used_budget <= 4200 and r.remaining_budget >= 0 for r in rows))
        self.assertTrue(any(r.status == FILTER_EXHAUSTED for r in rows))
        self.assertAlmostEqual(sum(0 if r.executed_budgets is None else r.executed_budgets.sum() for r in rows), mechanism.used_budget)

    def test_independent_innovations_and_no_private_fields(self):
        mechanism = ledger([True])
        row = mechanism.release_step(zero_features, lambda obs: (np.full(25, 2), 3))
        expected = np.random.default_rng(97).normal(size=(25, 256)) * analytic_gaussian_sigma(1)
        np.testing.assert_array_equal(row.release, expected)
        self.assertFalse(np.array_equal(row.release, np.random.default_rng(31).normal(size=(25, 256)) * analytic_gaussian_sigma(1)))
        forbidden = {"probe", "clean", "seed", "rng", "observation", "proposed"}
        self.assertTrue(forbidden.isdisjoint({f.name for f in dataclasses.fields(row)}))
        self.assertFalse(row.release.flags.writeable)
        self.assertEqual(row.used_budget, 50)
        self.assertEqual(row.status, EXEC_PROP)

    def test_frozen_masks_and_no_shared_rng(self):
        mask = [True]
        mechanism = ledger(mask)
        mask[0] = False
        self.assertEqual(mechanism.eligible_count, 1)
        self.assertEqual(mechanism.cap, 75)
        rng = np.random.default_rng(4)
        with self.assertRaises(ValueError):
            PrivacyLedger([True], probe_rng=rng, refinement_rng=rng)
        with self.assertRaises(ValueError):
            PrivacyLedger([True], probe_rng=np.random.default_rng(4), refinement_rng=np.random.default_rng(4))
        for eligible, recorded in [([True], [False]), ([True], [True, True]), ([True] * 57, None), ([2], None)]:
            with self.assertRaises(ValueError):
                ledger(eligible, recorded)

    def test_invalid_callback_closes_ledger_without_retry(self):
        mechanism = ledger([True])
        with self.assertRaises(ValueError):
            mechanism.release_step(zero_features, lambda obs: (np.full(25, 3), 0))
        with self.assertRaises(RuntimeError):
            mechanism.release_step(zero_features, lambda obs: (np.full(25, 3), 1))


if __name__ == "__main__":
    unittest.main()
