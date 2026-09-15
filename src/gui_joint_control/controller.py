"""New reference controller components; not the historical training implementation.

The reference makes architecture and sampling choices explicit.  It does not
load the original checkpoints, reproduce the paper's parameter totals, collect
replay, construct Bellman targets, or run experiments.  The statistical-result
files are intentionally never read or modified here.  PyTorch is imported only
when ``build_controllers`` is called; the specification and count helpers use
the Python standard library alone.
"""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any


REFERENCE_GAPS = (
    "This architecture is a new reference choice, not recovered original code.",
    "Reference parameter counts must not replace reported experimental counts.",
    "Frozen vision/language models and projection are external to the controller.",
    "No replay collector, privacy filter, executor, or full SAC trainer is supplied by this module.",
    "Evaluation of this implementation requires new runs and new result identifiers.",
)

DEFAULTS = {
    "observation_dim": 28,
    "regions": 25,
    "count_classes": 20,
    "critic_input_dim": 73,
    "hidden_sizes": [128, 128],
    "activation": "relu",
    "log_std_bounds": [-5.0, 2.0],
    "optimizer": {
        "name": "Adam", "lr": 3e-4, "betas": [0.9, 0.999],
        "eps": 1e-8, "weight_decay": 0.0,
    },
}

# These declared fields describe fixed behavior or the required downstream
# training contract. Reject alternatives instead of accepting a misleading
# configuration that the component implementation would silently ignore.
FIXED_SEMANTICS = {
    "encoder": "identity",
    "initialization": "PyTorch Linear default; record PyTorch version and initialization seed",
    "dtype": "float32",
    "gamma_H": 0.99,
    "gamma_CB": 0.0,
    "polyak_tau": 0.005,
    "gradient_clip_l2": 1.0,
    "entropy_coefficients": [0.2, 0.2],
    "gumbel": {"schedule": "linear_then_constant", "start": 1.0, "end": 0.1,
               "first_update": 1, "last_anneal_update": 500000},
    "update_order": ["one joint twin-critic optimizer step", "one actor optimizer step with critics frozen", "one Polyak update of both targets"],
    "target_initialization": "exact frozen copy of online critics",
    "target_update_formula": "target = (1-tau)*target + tau*online",
    "actor_gradients": "continuous score-function critic gradient and reparameterized entropy; categorical straight-through Gumbel; see trainer.py and docs/NAACL_RUNTIME.md",
    "budget_range": [1.5, 5.0],
    "evaluation": "tanh(mean) affine mapped to budget range; smallest categorical argmax + 1",
}


def _same_semantics(actual: Any, expected: Any) -> bool:
    """JSON-like equality without accepting true as 1 or false as 0."""
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and actual.keys() == expected.keys()
                and all(_same_semantics(actual[key], value) for key, value in expected.items()))
    if isinstance(expected, list):
        return (isinstance(actual, (list, tuple)) and len(actual) == len(expected)
                and all(_same_semantics(left, right) for left, right in zip(actual, expected)))
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return isinstance(actual, (int, float)) and not isinstance(actual, bool) and actual == expected
    return type(actual) is type(expected) and actual == expected


def _configuration(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Read the reference block; never infer original settings from counts."""
    block = (config or {}).get("reference_controller", {})
    if not isinstance(block, dict):
        raise ValueError("reference_controller must be an object")
    extra = set(block) - set(DEFAULTS) - set(FIXED_SEMANTICS)
    if extra:
        raise ValueError(f"Unsupported reference_controller fields: {', '.join(sorted(extra))}")
    for key, expected in FIXED_SEMANTICS.items():
        if key in block and not _same_semantics(block[key], expected):
            raise ValueError(f"reference_controller.{key} differs from the implemented fixed reference contract")
    out = deepcopy(DEFAULTS)
    for key in DEFAULTS:
        if key in block:
            out[key] = deepcopy(block[key])
    optimizer = deepcopy(DEFAULTS["optimizer"])
    if isinstance(out["optimizer"], dict):
        extra_optimizer = set(out["optimizer"]) - set(optimizer)
        if extra_optimizer:
            raise ValueError(f"Unsupported optimizer fields: {', '.join(sorted(extra_optimizer))}")
        optimizer.update(out["optimizer"])
    else:
        raise ValueError("optimizer must be an object")
    out["optimizer"] = optimizer
    for name, value in (("observation_dim", 28), ("regions", 25),
                        ("count_classes", 20), ("critic_input_dim", 73)):
        if not isinstance(out[name], int) or isinstance(out[name], bool) or out[name] != value:
            raise ValueError(f"{name} must be {value} for this reference contract")
    if out["activation"] != "relu":
        raise ValueError("This reference implements activation='relu' only")
    widths = out["hidden_sizes"]
    if (not isinstance(widths, (list, tuple)) or len(widths) != 2
            or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in widths)):
        raise ValueError("hidden_sizes must contain two positive integer widths")
    bounds = out["log_std_bounds"]
    if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in bounds)
            or bounds[0] >= bounds[1]):
        raise ValueError("log_std_bounds must contain two finite increasing values")
    if optimizer["name"] != "Adam":
        raise ValueError("This reference implements the Adam optimizer only")
    for field in ("lr", "eps"):
        value = optimizer[field]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"optimizer.{field} must be finite and positive")
    betas = optimizer["betas"]
    if (not isinstance(betas, (list, tuple)) or len(betas) != 2
            or any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v)
                   or not 0 <= v < 1 for v in betas)):
        raise ValueError("optimizer.betas must contain two values in [0,1)")
    decay = optimizer["weight_decay"]
    if not isinstance(decay, (int, float)) or isinstance(decay, bool) or not math.isfinite(decay) or decay < 0:
        raise ValueError("optimizer.weight_decay must be finite and nonnegative")
    return out


def _roles(method: str) -> dict[str, str]:
    methods = {
        "H": {"joint": "joint"}, "CB": {"joint": "joint"},
        "Disclosure-only": {"disclosure": "disclosure"},
        "Count-only": {"count": "count"},
        "Independent": {"disclosure": "disclosure", "count": "count"},
        "Independent-1M": {"disclosure": "disclosure", "count": "count"},
    }
    if method not in methods:
        raise ValueError(f"Unsupported reference method: {method}")
    return methods[method]


def model_spec(method: str = "H", config: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = _configuration(config)
    return {
        "status": "new_reference_not_original_runtime",
        "method": method,
        "components": _roles(method),
        "settings": settings,
        "fixed_contract": deepcopy(FIXED_SEMANTICS),
        "controller_encoder": "Identity (zero parameters)",
        "controller_dtype": "float32",
        "linear_initialization": "PyTorch Linear.reset_parameters: Kaiming-uniform with a=sqrt(5); bias uniform +/-1/sqrt(fan_in)",
        "critic_input": "28 observation + 25 executed total regional budgets + 20 count one-hot",
        "critic_count_per_component": 2,
        "frozen_target_count_per_component": 2,
        "gamma": 0.0 if method == "CB" else 0.99,
        "entropy_temperatures": {"continuous": 0.2, "categorical": 0.2},
        "polyak_tau": 0.005,
        "gradient_l2_clip": 1.0,
        "continuous_action_bounds": [1.5, 5.0],
        "evaluation_count_tie": "smallest candidate count (first argmax)",
        "gumbel_schedule": {
            "start": 1.0, "end": 0.1, "last_decay_update": 500000,
            "index": "1-based; update 1 has temperature 1.0; update 500000 and later have 0.1",
        },
        "gaps": list(REFERENCE_GAPS),
    }


def parameter_counts(method: str = "H", config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Count bias-inclusive Linear layers without loading a tensor library."""
    settings = _configuration(config)
    h1, h2 = settings["hidden_sizes"]
    actor_trunk = (settings["observation_dim"] + 1) * h1 + (h1 + 1) * h2
    critic = (settings["critic_input_dim"] + 1) * h1 + (h1 + 1) * h2 + h2 + 1
    components = {}
    for name, role in _roles(method).items():
        head_width = (50 if role in ("joint", "disclosure") else 0)
        head_width += 20 if role in ("joint", "count") else 0
        actor = actor_trunk + (h2 + 1) * head_width
        components[name] = {
            "actor": actor, "critic1": critic, "critic2": critic,
            "target1": critic, "target2": critic, "encoder": 0,
            "trainable": actor + 2 * critic,
            "frozen": 2 * critic,
            "total_including_targets": actor + 4 * critic,
        }
    return {
        "status": "new_reference_not_original_runtime",
        "method": method, "components": components,
        "trainable": sum(v["trainable"] for v in components.values()),
        "frozen": sum(v["frozen"] for v in components.values()),
        "total_including_targets": sum(v["total_including_targets"] for v in components.values()),
        "external_frozen_models_included": False,
    }


def gumbel_temperature(update_index: int) -> float:
    """Explicit new reference schedule, indexed by component optimizer iteration."""
    if not isinstance(update_index, int) or isinstance(update_index, bool) or update_index < 1:
        raise ValueError("update_index must be a positive 1-based integer")
    fraction = min(update_index - 1, 499999) / 499999
    return max(0.1, 1.0 - 0.9 * fraction)


def build_controllers(config: dict[str, Any], method: str = "H", *,
                      seed: int | None = None, device: str = "cpu") -> dict[str, dict[str, Any]]:
    """Construct actor/twin-critic/target bundles; caller supplies the SAC loop.

    ``seed`` explicitly seeds PyTorch before construction.  Nothing is seeded on
    import. Independent components have separate modules and optimizers, with
    no shared learned parameters. For a one-head actor, the other executed
    action must be supplied by the experiment's fixed schedule/other controller
    before forming the full 73-dimensional critic input.
    """
    settings = _configuration(config)
    roles = _roles(method)
    try:
        import torch
        from torch import nn
        from torch.nn import functional as functional
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to instantiate reference controllers; metadata/count helpers do not require it") from exc
    if seed is not None:
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        torch.manual_seed(seed)

    h1, h2 = settings["hidden_sizes"]

    class Actor(nn.Module):
        def __init__(self, role: str):
            super().__init__()
            self.role = role
            self.encoder = nn.Identity()
            self.trunk = nn.Sequential(nn.Linear(28, h1, dtype=torch.float32), nn.ReLU(), nn.Linear(h1, h2, dtype=torch.float32), nn.ReLU())
            self.mean = nn.Linear(h2, 25, dtype=torch.float32) if role in ("joint", "disclosure") else None
            self.log_std = nn.Linear(h2, 25, dtype=torch.float32) if role in ("joint", "disclosure") else None
            self.logits = nn.Linear(h2, 20, dtype=torch.float32) if role in ("joint", "count") else None

        def forward(self, observation):
            if observation.shape[-1] != 28:
                raise ValueError("Actor observation must have last dimension 28")
            features = self.trunk(self.encoder(observation))
            output = {}
            if self.mean is not None:
                output["mean"] = self.mean(features)
                output["log_std"] = self.log_std(features).clamp(*settings["log_std_bounds"])
            if self.logits is not None:
                output["logits"] = self.logits(features)
                output["log_probs"] = functional.log_softmax(output["logits"], dim=-1)
                output["probs"] = output["log_probs"].exp()
            return output

        def evaluation_action(self, observation):
            parameters = self(observation)
            output = {}
            if "mean" in parameters:
                output["budgets"] = 1.5 + 1.75 * (parameters["mean"].tanh() + 1.0)
            if "logits" in parameters:
                output["candidate_count"] = parameters["logits"].argmax(dim=-1) + 1
            return output

        def sample(self, observation, *, update_index: int):
            """Reparameterized budget plus straight-through Gumbel count.

            Continuous density includes both tanh and affine Jacobians. The
            ``continuous_log_prob`` path retains the rsample derivative for the
            entropy term. ``continuous_score_log_prob`` evaluates the SAME draw
            with its value detached, retaining parameter derivatives in the
            Normal density; use that quantity for a score-function critic term
            with a detached advantage. Do not use ``continuous_log_prob`` for
            that score-function derivative. Raw/normalized sampled actions are
            exposed so a downstream filter/critic can use the required path.
            Count
            log probability is the ordinary categorical log probability of the
            sampled class, not a Concrete density. The caller defines the
            discrete SAC expectation/Bellman loss; this method is not that loss.
            """
            temperature = gumbel_temperature(update_index)
            output = self(observation)
            if "mean" in output:
                distribution = torch.distributions.Normal(output["mean"], output["log_std"].exp())
                raw = distribution.rsample()
                output["budgets"] = 1.5 + 1.75 * (raw.tanh() + 1.0)
                log_tanh_jacobian = 2.0 * (math.log(2.0) - raw - functional.softplus(-2.0 * raw))
                output["continuous_log_prob"] = (
                    distribution.log_prob(raw) - log_tanh_jacobian - math.log(1.75)
                ).sum(dim=-1)
                output["continuous_score_log_prob"] = (
                    distribution.log_prob(raw.detach()) - log_tanh_jacobian.detach() - math.log(1.75)
                ).sum(dim=-1)
                output["raw_budget_sample"] = raw
                output["raw_budget_sample_detached"] = raw.detach()
                output["budgets_detached"] = output["budgets"].detach()
            if "logits" in output:
                one_hot = functional.gumbel_softmax(output["logits"], tau=temperature, hard=True, dim=-1)
                output["count_one_hot"] = one_hot
                output["candidate_count"] = one_hot.argmax(dim=-1) + 1
                output["categorical_log_prob"] = (one_hot * output["log_probs"]).sum(dim=-1)
            output["gumbel_temperature"] = temperature
            return output

    class Critic(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(nn.Linear(73, h1, dtype=torch.float32), nn.ReLU(), nn.Linear(h1, h2, dtype=torch.float32), nn.ReLU(), nn.Linear(h2, 1, dtype=torch.float32))

        def forward(self, executed_state_action):
            if executed_state_action.shape[-1] != 73:
                raise ValueError("Critic input must have last dimension 73 (28+25+20)")
            return self.network(executed_state_action).squeeze(-1)

    optimizer = settings["optimizer"]
    optimizer_args = {
        "lr": optimizer["lr"], "betas": tuple(optimizer["betas"]),
        "eps": optimizer["eps"], "weight_decay": optimizer["weight_decay"],
    }
    result = {}
    expected = parameter_counts(method, config)["components"]
    for name, role in roles.items():
        actor = Actor(role).to(device)
        critic1, critic2 = Critic().to(device), Critic().to(device)
        target1, target2 = deepcopy(critic1), deepcopy(critic2)
        for target in (target1, target2):
            target.requires_grad_(False)
            target.eval()
        modules = {"actor": actor, "critic1": critic1, "critic2": critic2,
                   "target1": target1, "target2": target2}
        actual = {key: sum(p.numel() for p in module.parameters()) for key, module in modules.items()}
        if any(actual[key] != expected[name][key] for key in actual):
            raise RuntimeError("Reference implementation and parameter-accounting specification disagree")
        result[name] = {
            **modules,
            "actor_optimizer": torch.optim.Adam(actor.parameters(), **optimizer_args),
            "critic_optimizer": torch.optim.Adam(list(critic1.parameters()) + list(critic2.parameters()), **optimizer_args),
            "reference_parameter_counts": expected[name],
            "gamma": 0.0 if method == "CB" else 0.99,
            "gaps": list(REFERENCE_GAPS),
        }
    return result


def polyak_update(bundle: dict[str, Any], tau: float = 0.005) -> None:
    """Call once after each component's critic optimizer iteration."""
    if not isinstance(tau, (int, float)) or not math.isfinite(tau) or not 0 <= tau <= 1:
        raise ValueError("tau must be finite and lie in [0,1]")
    for source_key, target_key in (("critic1", "target1"), ("critic2", "target2")):
        source_parameters = list(bundle[source_key].parameters())
        target_parameters = list(bundle[target_key].parameters())
        if len(source_parameters) != len(target_parameters):
            raise ValueError("Critic and target parameter structures disagree")
        for source, target in zip(source_parameters, target_parameters):
            if source.shape != target.shape:
                raise ValueError("Critic and target parameter shapes disagree")
            if target.requires_grad:
                raise ValueError("Target parameters must remain frozen")
            target.mul_(1.0 - tau).add_(source.detach(), alpha=tau)


def describe_controllers(bundles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Return JSON-safe layer/optimizer metadata, never weights or optimizer state.

    The return value describes newly instantiated modules; it is not evidence
    about the original experimental checkpoint. Only selected non-tensor
    optimizer hyperparameters are exported, so momentum buffers, parameter
    values and unrelated custom configuration are excluded.
    """
    safe_optimizer_keys = ("lr", "betas", "eps", "weight_decay", "amsgrad", "maximize",
                           "capturable", "differentiable", "fused", "foreach")
    description = {"status": "new_reference_not_original_runtime", "components": {}}
    for name, bundle in bundles.items():
        modules = {}
        for key in ("actor", "critic1", "critic2", "target1", "target2"):
            module = bundle[key]
            layers = []
            for layer_name, layer in module.named_modules():
                own_parameters = list(layer.named_parameters(recurse=False))
                if not own_parameters:
                    continue
                layers.append({
                    "path": layer_name,
                    "class": type(layer).__name__,
                    "parameters": [{"name": parameter_name, "shape": list(parameter.shape),
                                    "count": parameter.numel(), "trainable": parameter.requires_grad,
                                    "dtype": str(parameter.dtype)}
                                   for parameter_name, parameter in own_parameters],
                })
            parameters = list(module.parameters())
            modules[key] = {"layers": layers, "total": sum(p.numel() for p in parameters),
                            "trainable": sum(p.numel() for p in parameters if p.requires_grad)}
        optimizers = {}
        for key in ("actor_optimizer", "critic_optimizer"):
            optimizer = bundle[key]
            groups = []
            for group in optimizer.param_groups:
                values = {}
                for setting in safe_optimizer_keys:
                    if setting not in group:
                        continue
                    value = group[setting]
                    if value is None or isinstance(value, (str, bool, int, float)):
                        values[setting] = value
                    elif isinstance(value, (list, tuple)) and all(isinstance(v, (int, float)) for v in value):
                        values[setting] = list(value)
                    else:
                        raise ValueError(f"Optimizer {setting} is not a supported scalar/list hyperparameter")
                values["parameter_count"] = sum(p.numel() for p in group["params"])
                groups.append(values)
            optimizers[key] = {"class": type(optimizer).__name__, "groups": groups}
        description["components"][name] = {"modules": modules, "optimizers": optimizers}
    return description
