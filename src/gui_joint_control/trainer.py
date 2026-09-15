"""Offline hybrid SAC training for a newly specified reference implementation.

This code does not reconstruct historical checkpoints or establish the paper's
reported accuracy. Each step uses immutable offline transitions. Independent
heads are fitted separately against fixed complementary schedules, and are
composed only for evaluation; they never update one another's parameters.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .controller import build_controllers, gumbel_temperature, parameter_counts, polyak_update
from .replay import ReplayBuffer


def filter_budgets(proposals: torch.Tensor, remaining_budget: torch.Tensor) -> torch.Tensor:
    """Public all-or-fallback filter; never rescale or differentiate its branch.

    All 25 proposals execute when affordable. Otherwise the whole minimum
    vector executes. An unaffordable minimum denotes a terminal/padding event,
    not an executable transition, and must be handled outside this function.
    """
    if proposals.ndim != 2 or proposals.shape[1] != 25:
        raise ValueError("proposals must have shape [batch, 25]")
    if remaining_budget.shape != proposals.shape[:1]:
        raise ValueError("remaining_budget must have one value per proposal vector")
    if not torch.isfinite(proposals).all() or not torch.isfinite(remaining_budget).all():
        raise ValueError("Filter inputs must be finite")
    if torch.any((proposals < 1.5) | (proposals > 5.0)):
        raise ValueError("Proposal budgets must lie in [1.5, 5.0]")
    if torch.any(remaining_budget < 37.5):
        raise ValueError("A nonterminal filter invocation requires at least 37.5 budget")
    detached = proposals.detach()
    # The public ledger is float64 metadata, never an actor observation. For
    # these 25 bounded float32 proposals their promoted sum is exactly
    # representable in float64, matching the deployed ledger's math.fsum.
    affordable = detached.to(torch.float64).sum(dim=-1) <= remaining_budget.to(torch.float64)
    return torch.where(affordable[:, None], detached, torch.full_like(detached, 1.5))


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _cpu_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return deepcopy(value)


class Trainer:
    """An actual optimizer loop with a resumable, owned sampling/RNG stream.

    ``step`` advances each component by one critic, one actor and one target
    update. Thus one Independent step consumes two component iterations; an H
    step consumes one. Set iteration limits accordingly. Rewards must already
    have been computed from the behavior transition; they are never recomputed
    with a newly proposed action while training from offline replay.
    """

    def __init__(self, config: dict, method: str, buffer: ReplayBuffer,
                 seed: int = 20260916, device: str = "cpu"):
        if not isinstance(buffer, ReplayBuffer):
            raise TypeError("buffer must be a validated ReplayBuffer")
        if not isinstance(config, dict):
            raise ValueError("config must be a JSON object")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.config = json.loads(_canonical(config))
        self.config_sha256 = sha256(_canonical(self.config).encode("utf-8")).hexdigest()
        self.method, self.buffer, self.seed = method, buffer, seed
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("This reference trainer supports CPU or CUDA devices")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        self.bundles = build_controllers(self.config, method, seed=seed, device=str(self.device))
        self.controllers = self.bundles
        self.reference_parameter_counts = parameter_counts(method, self.config)
        self.component_updates = {name: 0 for name in self.bundles}
        self.iteration = 0
        self.sampler = np.random.default_rng(seed)
        self._torch_rng = torch.get_rng_state().clone()
        self._cuda_rng = torch.cuda.get_rng_state(self.device).clone() if self.device.type == "cuda" else None
        self.training = self.config.get("training", {})
        if not isinstance(self.training, dict):
            raise ValueError("training must be a JSON object")
        self._validate_training()
        self.fixed_complement = self.training.get("fixed_complement")
        if any(bundle["actor"].role != "joint" for bundle in self.bundles.values()):
            if not isinstance(self.fixed_complement, dict) or set(self.fixed_complement) != {"regional_budget", "candidate_count"}:
                raise ValueError("Single-head fitting requires explicit training.fixed_complement regional_budget and candidate_count")
            budget, count = self.fixed_complement["regional_budget"], self.fixed_complement["candidate_count"]
            if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or not 1.5 <= budget <= 5:
                raise ValueError("fixed_complement.regional_budget must lie in [1.5,5]")
            if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 20:
                raise ValueError("fixed_complement.candidate_count must be an integer in 1..20")
        self.code_sha256 = {
            name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("controller.py", "replay.py", "trainer.py")
        }

    def _validate_training(self) -> None:
        sampling = self.training.get("sampling", {})
        expected = {"success_probability": 0.7, "failure_probability": 0.3,
                    "within_stratum": "uniform with replacement", "empty_stratum": "abort run",
                    "capacity_overflow": "abort run"}
        if not isinstance(sampling, dict) or any(key not in expected or value != expected[key] for key, value in sampling.items()):
            raise ValueError("training.sampling differs from the implemented immutable 70/30 contract")
        capacity = self.training.get("replay_capacity", 1_000_000)
        if isinstance(capacity, bool) or capacity != 1_000_000:
            raise ValueError("The reference replay capacity is 1,000,000 with abort on overflow")
        batch_size = self.training.get("replay_batch_size", 256)
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("training.replay_batch_size must be a positive integer")

    @contextmanager
    def _random_context(self):
        cuda_devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.set_rng_state(self._torch_rng)
            if self._cuda_rng is not None:
                torch.cuda.set_rng_state(self._cuda_rng, self.device)
            try:
                yield
            finally:
                self._torch_rng = torch.get_rng_state().clone()
                if self._cuda_rng is not None:
                    self._cuda_rng = torch.cuda.get_rng_state(self.device).clone()

    def _action(self, bundle: dict, observation: torch.Tensor, remaining: torch.Tensor,
                *, update_index: int, deterministic: bool = False) -> dict:
        actor = bundle["actor"]
        sample = actor.evaluation_action(observation) if deterministic else actor.sample(observation, update_index=update_index)
        if "budgets" not in sample:
            sample["budgets"] = torch.full((len(observation), 25), self.fixed_complement["regional_budget"], device=self.device)
        sample["executed_budgets"] = filter_budgets(sample["budgets"], remaining)
        if "candidate_count" not in sample:
            sample["candidate_count"] = torch.full((len(observation),), self.fixed_complement["candidate_count"], dtype=torch.long, device=self.device)
        if "count_one_hot" not in sample:
            sample["count_one_hot"] = F.one_hot(sample["candidate_count"] - 1, 20).to(torch.float32)
        return sample

    @staticmethod
    def _critic_input(observation: torch.Tensor, budgets: torch.Tensor, count_one_hot: torch.Tensor) -> torch.Tensor:
        return torch.cat((observation, budgets, count_one_hot), dim=-1)

    def _target(self, bundle: dict, batch: dict, update_index: int) -> torch.Tensor:
        """Do not even sample/filter successor actions for terminal rows or CB."""
        target = batch["reward"].clone()
        nonterminal = ~batch["terminal"]
        if bundle["gamma"] == 0.0 or not bool(nonterminal.any()):
            return target
        with torch.no_grad():
            next_observation = batch["next_observation"][nonterminal]
            action = self._action(bundle, next_observation, batch["next_remaining_budget"][nonterminal], update_index=update_index)
            critic_input = self._critic_input(next_observation, action["executed_budgets"], action["count_one_hot"])
            value = torch.minimum(bundle["target1"](critic_input), bundle["target2"](critic_input))
            if "continuous_log_prob" in action:
                value = value - 0.2 * action["continuous_log_prob"]
            if "probs" in action:
                # Exact categorical entropy, while Q uses the sampled hard count.
                value = value - 0.2 * (action["probs"] * action["log_probs"]).sum(dim=-1)
            target[nonterminal] += bundle["gamma"] * value
        return target

    def _update_component(self, name: str, batch: dict) -> dict:
        bundle = self.bundles[name]
        update_index = self.component_updates[name] + 1
        target = self._target(bundle, batch, update_index)
        observed_input = self._critic_input(batch["observation"], batch["executed_budgets"], F.one_hot(batch["candidate_count"] - 1, 20).to(torch.float32))
        critic_loss = F.mse_loss(bundle["critic1"](observed_input), target) + F.mse_loss(bundle["critic2"](observed_input), target)
        if not torch.isfinite(critic_loss):
            raise FloatingPointError("Nonfinite critic loss; no optimizer step performed")
        critic_parameters = list(bundle["critic1"].parameters()) + list(bundle["critic2"].parameters())
        bundle["critic_optimizer"].zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad = torch.nn.utils.clip_grad_norm_(critic_parameters, 1.0, error_if_nonfinite=True)
        bundle["critic_optimizer"].step()
        bundle["critic_optimizer"].zero_grad(set_to_none=True)
        for parameter in critic_parameters:
            parameter.requires_grad_(False)
        try:
            action = self._action(bundle, batch["observation"], batch["remaining_budget"], update_index=update_index)
            critic_input = self._critic_input(batch["observation"], action["executed_budgets"], action["count_one_hot"])
            value = torch.minimum(bundle["critic1"](critic_input), bundle["critic2"](critic_input))
            # Budget/filter/Q paths are detached. Continuous control uses the
            # score-function Q gradient, and its entropy uses the rsample path.
            # The direct -Q term supplies only the categorical ST gradient.
            losses = -value
            if "continuous_score_log_prob" in action:
                losses = losses - value.detach() * action["continuous_score_log_prob"]
                losses = losses + 0.2 * action["continuous_log_prob"]
            if "probs" in action:
                losses = losses + 0.2 * (action["probs"] * action["log_probs"]).sum(dim=-1)
            actor_loss = losses.mean()
            if not torch.isfinite(actor_loss):
                raise FloatingPointError("Nonfinite actor loss; critic step completed but actor step aborted")
            bundle["actor_optimizer"].zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad = torch.nn.utils.clip_grad_norm_(bundle["actor"].parameters(), 1.0, error_if_nonfinite=True)
            bundle["actor_optimizer"].step()
        finally:
            for parameter in critic_parameters:
                parameter.requires_grad_(True)
        with torch.no_grad():
            polyak_update(bundle, tau=0.005)
        self.component_updates[name] = update_index
        return {
            "critic_loss": float(critic_loss.detach()), "actor_loss": float(actor_loss.detach()),
            "critic_gradient_norm_before_clip": float(critic_grad),
            "actor_gradient_norm_before_clip": float(actor_grad),
            "target_mean": float(target.mean()), "gumbel_temperature": gumbel_temperature(update_index),
            "success_fraction": float(batch["success"].to(torch.float32).mean()),
        }

    def step(self, batch_size: int | None = None) -> dict:
        batch_size = self.training.get("replay_batch_size", 256) if batch_size is None else batch_size
        metrics = {}
        with self._random_context():
            for name in self.bundles:
                sampled = self.buffer.sample(batch_size, self.sampler)
                batch = {key: torch.from_numpy(value).to(self.device) for key, value in sampled.items() if key != "indices"}
                metrics[name] = self._update_component(name, batch)
        self.iteration += 1
        return {
            "iteration": self.iteration, "method": self.method,
            "component_updates": dict(self.component_updates),
            "aggregate_replay_iterations": sum(self.component_updates.values()),
            "aggregate_optimizer_calls": 2 * sum(self.component_updates.values()),
            "components": metrics,
        }

    def evaluation_action(self, observation: np.ndarray | torch.Tensor,
                          remaining_budget: np.ndarray | torch.Tensor) -> dict[str, torch.Tensor]:
        """Compose separately fitted heads only here, using deterministic means."""
        with torch.no_grad():
            observation = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
            remaining = torch.as_tensor(remaining_budget, dtype=torch.float64, device=self.device)
            if observation.ndim != 2 or observation.shape[1] != 28 or not torch.isfinite(observation).all():
                raise ValueError("Evaluation observation must be finite with shape [batch,28]")
            if self.method in ("Independent", "Independent-1M"):
                budget = self.bundles["disclosure"]["actor"].evaluation_action(observation)["budgets"]
                count = self.bundles["count"]["actor"].evaluation_action(observation)["candidate_count"]
                return {"executed_budgets": filter_budgets(budget, remaining), "candidate_count": count}
            bundle = next(iter(self.bundles.values()))
            action = self._action(bundle, observation, remaining, update_index=1, deterministic=True)
            return {"executed_budgets": action["executed_budgets"], "candidate_count": action["candidate_count"]}

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1, "status": "new_reference_not_original_runtime",
            "method": self.method, "config": self.config, "config_sha256": self.config_sha256,
            "replay_content_sha256": self.buffer.content_sha256,
            "replay_source_sha256": self.buffer.source_sha256, "code_sha256": self.code_sha256,
            "torch_version": str(torch.__version__), "numpy_version": str(np.__version__),
            "device_type": self.device.type, "seed": self.seed, "iteration": self.iteration,
            "component_updates": self.component_updates, "parameter_counts": self.reference_parameter_counts,
            "sampler_state": deepcopy(self.sampler.bit_generator.state),
            "torch_cpu_rng": self._torch_rng, "torch_device_rng": self._cuda_rng,
            "components": {
                name: {key: value.state_dict() for key, value in bundle.items()
                       if key in ("actor", "critic1", "critic2", "target1", "target2", "actor_optimizer", "critic_optimizer")}
                for name, bundle in self.bundles.items()
            },
        }
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt.tmp", delete=False) as handle:
                temporary = Path(handle.name)
            torch.save(_cpu_tree(payload), temporary)
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def load_checkpoint(self, path: str | Path) -> None:
        """Load only a matching reference checkpoint using PyTorch's safe loader."""
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("format_version") != 1:
            raise ValueError("Unsupported reference checkpoint")
        matches = {
            "method": self.method, "config_sha256": self.config_sha256,
            "replay_content_sha256": self.buffer.content_sha256, "code_sha256": self.code_sha256,
            "torch_version": str(torch.__version__), "numpy_version": str(np.__version__),
            "device_type": self.device.type,
        }
        for field, expected in matches.items():
            if payload.get(field) != expected:
                raise ValueError(f"Checkpoint {field} does not match this run; start a new run instead")
        if _canonical(payload.get("config")) != _canonical(self.config):
            raise ValueError("Checkpoint embedded config does not match")
        if set(payload.get("components", {})) != set(self.bundles):
            raise ValueError("Checkpoint components do not match")
        updates = payload.get("component_updates", {})
        iteration = payload.get("iteration")
        if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
            raise ValueError("Invalid checkpoint iteration")
        if set(updates) != set(self.bundles) or any(value != iteration for value in updates.values()):
            raise ValueError("Checkpoint component counters are inconsistent")
        # Validate tensors before changing live modules.
        for name, bundle in self.bundles.items():
            for key in ("actor", "critic1", "critic2", "target1", "target2"):
                state = payload["components"][name].get(key, {})
                current = bundle[key].state_dict()
                if state.keys() != current.keys() or any(not torch.is_tensor(state[k]) or state[k].shape != v.shape or not torch.isfinite(state[k]).all() for k, v in current.items()):
                    raise ValueError(f"Invalid checkpoint tensors for {name}.{key}")
        for name, bundle in self.bundles.items():
            for key in ("actor", "critic1", "critic2", "target1", "target2", "actor_optimizer", "critic_optimizer"):
                # Optimizer.load_state_dict places moments with parameters and
                # preserves Adam's CPU step counters for non-capturable mode.
                bundle[key].load_state_dict(payload["components"][name][key])
        self.iteration, self.component_updates = iteration, dict(updates)
        self.seed = payload["seed"]
        self.sampler.bit_generator.state = payload["sampler_state"]
        self._torch_rng = payload["torch_cpu_rng"].cpu().clone()
        self._cuda_rng = payload["torch_device_rng"].cpu().clone() if payload["torch_device_rng"] is not None else None
