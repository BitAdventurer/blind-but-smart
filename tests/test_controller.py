"""Software tests for new reference components, not synthetic paper results."""
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gui_joint_control.controller import (
    build_controllers, describe_controllers, gumbel_temperature, model_spec, parameter_counts, polyak_update,
)


class ReferenceContractTests(unittest.TestCase):
    def test_reference_is_explicitly_distinct_from_reported_runtime(self):
        spec = model_spec()
        self.assertEqual(spec["status"], "new_reference_not_original_runtime")
        self.assertEqual(spec["settings"]["critic_input_dim"], 28 + 25 + 20)
        self.assertNotEqual(parameter_counts()["total_including_targets"], 261192)
        self.assertEqual(model_spec("CB")["gamma"], 0.0)
        self.assertEqual(model_spec("H")["gamma"], 0.99)

    def test_parameter_accounting_includes_distinct_targets_and_heads(self):
        joint = parameter_counts()
        self.assertEqual(joint["trainable"], 81480)
        self.assertEqual(joint["frozen"], 52226)
        self.assertEqual(joint["total_including_targets"], 133706)
        independent = parameter_counts("Independent")
        self.assertEqual(independent["total_including_targets"], 258382)
        self.assertEqual(independent["frozen"], 2 * joint["frozen"])
        self.assertEqual(independent["components"]["disclosure"]["actor"], 26674)
        self.assertEqual(independent["components"]["count"]["actor"], 22804)
        self.assertEqual(parameter_counts("Independent-1M"),
                         {**independent, "method": "Independent-1M"})

    def test_tiny_architecture_can_be_checked_by_hand(self):
        # Actor: 29*2 + 3*3 + 4*70 = 347. Each critic: 74*2 + 3*3 + 4 = 161.
        count = parameter_counts(config={"reference_controller": {"hidden_sizes": [2, 3]}})
        self.assertEqual(count["components"]["joint"]["actor"], 347)
        self.assertEqual(count["total_including_targets"], 347 + 4 * 161)

    def test_temperature_has_explicit_endpoints_and_is_monotone(self):
        self.assertEqual(gumbel_temperature(1), 1.0)
        self.assertAlmostEqual(gumbel_temperature(500000), 0.1)
        self.assertAlmostEqual(gumbel_temperature(500001), 0.1)
        self.assertAlmostEqual(gumbel_temperature(1000000), 0.1)
        temperatures = [gumbel_temperature(n) for n in (1, 2, 1000, 250000, 499999, 500000)]
        self.assertEqual(temperatures, sorted(temperatures, reverse=True))
        for invalid in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                gumbel_temperature(invalid)

    def test_invalid_model_contract_cannot_silently_change_inputs(self):
        for key, value in (("observation_dim", 29), ("critic_input_dim", 72),
                           ("regions", 24), ("activation", "tanh"),
                           ("hidden_sizes", [128]), ("log_std_bounds", [2, -5]),
                           ("optimizer", {"lr": float("nan")})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                model_spec(config={"reference_controller": {key: value}})
        with self.assertRaises(ValueError):
            model_spec("Made-up")

    def test_fixed_configuration_drift_is_rejected(self):
        fields = {
            "gumbel": {"schedule": "constant", "start": 1.0, "end": 1.0, "first_update": 1, "last_anneal_update": 500000},
            "budget_range": [0.0, 1.0], "gamma_H": 0.95, "encoder": "linear",
            "dtype": "float64", "polyak_tau": 0.01, "gradient_clip_l2": 0.5,
            "entropy_coefficients": [0.1, 0.2], "target_initialization": "random",
            "unexpected_option": True,
        }
        for name, value in fields.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                model_spec(config={"reference_controller": {name: value}})
        with self.assertRaises(ValueError):
            model_spec(config={"reference_controller": {"optimizer": {"momentum": 0.9}}})


    def setUp(self):
        import torch
        self.torch = torch
        self.config = {"reference_controller": {"hidden_sizes": [4, 4]}}

    def test_deterministic_action_and_tie_rule(self):
        torch = self.torch
        bundle = build_controllers(self.config, seed=7)["joint"]
        with torch.no_grad():
            for parameter in bundle["actor"].parameters():
                parameter.zero_()
        result = bundle["actor"].evaluation_action(torch.zeros(2, 28))
        self.assertTrue(torch.all(result["budgets"] == 3.25))
        self.assertEqual(result["candidate_count"].tolist(), [1, 1])

    def test_continuous_log_density_matches_change_of_variables(self):
        torch = self.torch
        bundle = build_controllers(self.config, seed=9)["joint"]
        sample = bundle["actor"].sample(torch.zeros(3, 28), update_index=1)
        raw = sample["raw_budget_sample"]
        normal = torch.distributions.Normal(sample["mean"], sample["log_std"].exp())
        expected = (normal.log_prob(raw) - (1 - raw.tanh().square()).log() - math.log(1.75)).sum(-1)
        self.assertTrue(torch.allclose(sample["continuous_log_prob"], expected, atol=1e-4))
        self.assertTrue(torch.all((sample["budgets"] >= 1.5) & (sample["budgets"] <= 5)))
        self.assertEqual(sample["count_one_hot"].sum(-1).tolist(), [1.0] * 3)
        self.assertTrue(sample["continuous_log_prob"].isfinite().all())

    def test_score_function_density_has_fixed_draw_derivative(self):
        torch = self.torch
        bundle = build_controllers(self.config, seed=9)["joint"]
        sample = bundle["actor"].sample(torch.zeros(3, 28), update_index=1)
        derivative = torch.autograd.grad(sample["continuous_score_log_prob"].sum(), sample["mean"], retain_graph=True)[0]
        expected = (sample["raw_budget_sample_detached"] - sample["mean"]) / sample["log_std"].exp().square()
        self.assertTrue(torch.allclose(derivative, expected))
        self.assertTrue(torch.allclose(sample["continuous_score_log_prob"], sample["continuous_log_prob"]))
        self.assertFalse(sample["budgets_detached"].requires_grad)

    def test_independent_components_do_not_share_parameters_or_optimizer_state(self):
        bundles = build_controllers(self.config, "Independent", seed=21)
        left, right = bundles["disclosure"], bundles["count"]
        ids_left = {id(p) for name in ("actor", "critic1", "critic2", "target1", "target2") for p in left[name].parameters()}
        ids_right = {id(p) for name in ("actor", "critic1", "critic2", "target1", "target2") for p in right[name].parameters()}
        self.assertFalse(ids_left & ids_right)
        self.assertIsNot(left["actor_optimizer"], right["actor_optimizer"])
        self.assertFalse(any(p.requires_grad for p in left["target1"].parameters()))

    def test_polyak_updates_targets_without_gradients(self):
        torch = self.torch
        bundle = build_controllers(self.config, seed=4)["joint"]
        with torch.no_grad():
            for name in ("critic1", "critic2"):
                for p in bundle[name].parameters():
                    p.fill_(2)
            for name in ("target1", "target2"):
                for p in bundle[name].parameters():
                    p.fill_(0)
        polyak_update(bundle, 0.25)
        self.assertTrue(all(torch.all(p == 0.5) for p in bundle["target1"].parameters()))

    def test_inventory_is_json_safe_and_excludes_tensor_values(self):
        bundle = build_controllers(self.config, seed=2)
        metadata = describe_controllers(bundle)
        encoded = json.dumps(metadata)
        self.assertNotIn("tensor(", encoded)
        joint = metadata["components"]["joint"]
        self.assertEqual(joint["modules"]["target1"]["trainable"], 0)
        self.assertEqual(joint["optimizers"]["actor_optimizer"]["groups"][0]["lr"], 0.0003)


if __name__ == "__main__":
    unittest.main()
