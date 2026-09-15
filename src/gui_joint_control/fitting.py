"""New Stage 1/2 projection and language-adapter fitting implementation.

Training-only inputs may include clean clipped features and stopped native
vision tokens. These never enter the deployed released-input executor. The
functions below fit new artifacts; they do not reconstruct missing historical
checkpoints or imply reproduction of the manuscript's reported accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .privacy import analytic_gaussian_sigma, clip_features


@dataclass(frozen=True)
class FitSettings:
    learning_rate: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    optimizer_epsilon: float = 1e-8
    weight_decay: float = 0.01
    alignment_temperature: float = 0.07
    stage1_epochs: int = 2
    stage2_epochs: int = 3
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0


@dataclass(frozen=True)
class AlignmentExample:
    record_id: str
    clean_features: object  # 25 x 256; trusted training-only features
    native_target_tokens: object  # 25 x model-hidden; frozen native vision output


@dataclass(frozen=True)
class TeacherForcedExample:
    record_id: str
    clean_features: object
    prompt_token_ids: Sequence[int]  # Original IDs including the 25 visual pads
    target_token_ids: Sequence[int]  # Complete canonical target, including EOS


def initialize_projection(hidden_size: int = 3584, *, seed: int, device="cpu"):
    import torch
    # Xavier initialization has its own stream and does not consume global RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        layer = torch.nn.Linear(256, hidden_size, bias=False, dtype=torch.float32)
        torch.nn.init.xavier_uniform_(layer.weight)
    return layer.to(device)


def symmetric_alignment_loss(projected, stopped_targets, *, temperature=0.07):
    import torch
    import torch.nn.functional as F
    if projected.ndim != 2 or projected.shape != stopped_targets.shape or projected.shape[0] != 25:
        raise ValueError("Alignment requires one screen with 25 paired tokens")
    if not temperature > 0 or not torch.isfinite(projected).all() or not torch.isfinite(stopped_targets).all():
        raise ValueError("Alignment inputs and positive temperature must be finite")
    z = F.normalize(projected.float(), dim=-1, eps=1e-12)
    target = F.normalize(stopped_targets.detach().float(), dim=-1, eps=1e-12)
    similarity = z @ target.T / temperature
    labels = torch.arange(25, device=projected.device)
    return (F.cross_entropy(similarity, labels) + F.cross_entropy(similarity.T, labels)) / 2


def _streams(seed):
    children = np.random.SeedSequence(seed).spawn(3)
    return tuple(np.random.default_rng(child) for child in children)


def _augment(clean_features, interval, budget_rng, noise_rng):
    clipped = clip_features(clean_features)
    epsilon = budget_rng.uniform(interval[0], interval[1], size=25)
    scale = np.array([analytic_gaussian_sigma(float(e)) for e in epsilon])
    return clipped + noise_rng.normal(size=(25, 256)) * scale[:, None], epsilon


def _records(records):
    records = tuple(records)
    if not records:
        raise ValueError("Cannot fit an empty training set")
    if len({record.record_id for record in records}) != len(records):
        raise ValueError("Training record IDs must be unique")
    return records


def _optimizer(parameters, settings):
    import torch
    return torch.optim.AdamW(parameters, lr=settings.learning_rate, betas=settings.betas,
                             eps=settings.optimizer_epsilon, weight_decay=settings.weight_decay)


def _projection_array(projection):
    if getattr(projection, "bias", None) is not None or projection.weight.ndim != 2 or projection.weight.shape[1] != 256:
        raise ValueError("Expected a bias-free Linear(256, hidden_size) projection")
    return projection.weight.detach().cpu().contiguous()


def save_projection(projection, directory, *, metadata: dict):
    """Write explicit new weights in non-pickle formats; refuse overwrite."""
    from safetensors.torch import save_file
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / "projection.safetensors", directory / "projection.npy", directory / "fit_metadata.json"]
    if any(path.exists() for path in paths):
        raise FileExistsError("A fitting artifact already exists in the output directory")
    tensor = _projection_array(projection)
    save_file({"W_proj": tensor}, str(paths[0]))
    array = tensor.numpy()
    np.save(paths[1], array, allow_pickle=False)
    metadata = {**metadata, "artifact_kind": "new-reference-fit",
                "projection_shape": list(array.shape),
                "projection_dtype": str(array.dtype),
                "projection_array_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}
    paths[2].write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


class Stage1Trainer:
    def __init__(self, projection, *, seed: int, settings: FitSettings = FitSettings()):
        _projection_array(projection)
        self.projection = projection.train().requires_grad_(True)
        self.seed = seed
        self.settings = settings
        self.optimizer = _optimizer(self.projection.parameters(), settings)
        self.order_rng, self.budget_rng, self.noise_rng = _streams(seed)

    def fit(self, records: Sequence[AlignmentExample], *, epochs: int | None = None):
        import torch
        records = _records(records)
        epochs = self.settings.stage1_epochs if epochs is None else epochs
        if not isinstance(epochs, int) or epochs < 1:
            raise ValueError("epochs must be a positive integer")
        device = self.projection.weight.device
        history = []
        for epoch in range(epochs):
            for index in self.order_rng.permutation(len(records)):
                record = records[index]
                release, epsilon = _augment(record.clean_features, (2.0, 4.0), self.budget_rng, self.noise_rng)
                targets = torch.as_tensor(record.native_target_tokens, dtype=torch.float32, device=device).detach()
                released = torch.tensor(release, dtype=self.projection.weight.dtype, device=device)
                self.optimizer.zero_grad(set_to_none=True)
                loss = symmetric_alignment_loss(self.projection(released), targets,
                                                temperature=self.settings.alignment_temperature)
                loss.backward()
                self.optimizer.step()
                history.append({"epoch": epoch + 1, "record_id": record.record_id,
                                "loss": float(loss.detach()), "mean_refinement_epsilon": float(epsilon.mean())})
        return history

    def save(self, directory, *, provenance: dict):
        save_projection(self.projection, directory, metadata={
            "stage": 1, "seed": self.seed, "settings": asdict(self.settings), "provenance": provenance})


def attach_language_lora(model, *, seed: int, settings: FitSettings = FitSettings()):
    """Adapt only language self-attention query/value matrices, never vision."""
    import torch
    import peft
    from peft import LoraConfig, get_peft_model
    if peft.__version__ != "0.18.1":
        raise RuntimeError("This fitting implementation requires peft==0.18.1")
    if hasattr(model, "peft_config"):
        raise ValueError("Provide a base checkpoint without an attached PEFT adapter")
    model.requires_grad_(False)
    targets = [name for name, module in model.named_modules()
               if ".language_model." in name and ".self_attn." in name
               and name.rsplit(".", 1)[-1] in {"q_proj", "v_proj"}
               and isinstance(module, torch.nn.Linear)]
    if not targets:
        raise ValueError("No Qwen language query/value modules found")
    config = LoraConfig(r=settings.lora_rank, lora_alpha=settings.lora_alpha,
                        lora_dropout=settings.lora_dropout, bias="none",
                        target_modules=targets, task_type="CAUSAL_LM")
    # torch.manual_seed seeds all visible CUDA streams; restore all of them.
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        adapted = get_peft_model(model, config)
    for name, parameter in adapted.named_parameters():
        if parameter.requires_grad and ("lora_" not in name or ".language_model." not in name):
            raise RuntimeError(f"Unexpected trainable base/vision parameter: {name}")
    return adapted


def teacher_forced_inputs(model, projection, release, prompt_token_ids, target_token_ids):
    """Differentiable released-image embeddings; loss only on complete target.

    No processor or native vision forward pass is used. Native M-RoPE is
    computed from original prompt+target IDs before replacing visual pads.
    """
    import torch
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    device = base.get_input_embeddings().weight.device
    prompt = list(prompt_token_ids)
    target = list(target_token_ids)
    if not prompt or not target:
        raise ValueError("A nonempty prompt and complete target are required")
    eos = base.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    if target[-1] not in eos:
        raise ValueError("Target must end in a pinned EOS token")
    forbidden = {base.config.image_token_id, base.config.video_token_id,
                 base.config.vision_start_token_id, base.config.vision_end_token_id}
    if any(token in forbidden for token in target):
        raise ValueError("Canonical target must not contain visual placeholders")
    pad = base.generation_config.pad_token_id
    if pad not in eos and pad in target:
        raise ValueError("Targets must be unpadded complete canonical outputs")
    if any(token in eos for token in target[:-1]):
        raise ValueError("Target contains an interior EOS token")
    ids = torch.tensor([prompt + target], device=device, dtype=torch.long)
    mask = torch.ones_like(ids)
    visual_mask = ids[0] == base.config.image_token_id
    visual_positions = visual_mask.nonzero().flatten()
    if (len(visual_positions) != 25 or int(visual_positions[0]) == 0
            or not torch.equal(visual_positions, torch.arange(int(visual_positions[0]),
                                                             int(visual_positions[0]) + 25, device=device))
            or int(ids[0, visual_positions[0] - 1]) != base.config.vision_start_token_id
            or int(ids[0, visual_positions[-1] + 1]) != base.config.vision_end_token_id):
        raise ValueError("Prompt must contain one contiguous 25-pad visual wrapper")
    positions, _ = base.model.get_rope_index(ids, image_grid_thw=torch.tensor([[1, 10, 10]], device=device),
                                            attention_mask=mask)
    embeddings = base.get_input_embeddings()(ids)
    released = torch.as_tensor(release, dtype=projection.weight.dtype, device=projection.weight.device)
    if released.shape != (25, 256) or not torch.isfinite(released).all():
        raise ValueError("Expected finite 25x256 released training features")
    embeddings = embeddings.clone()
    embeddings[0, visual_mask] = projection(released).to(device=device, dtype=embeddings.dtype)
    labels = ids.clone()
    labels[:, :len(prompt)] = -100
    text_positions = torch.arange(ids.shape[1], device=device).reshape(1, 1, -1)
    return {"inputs_embeds": embeddings, "attention_mask": mask,
            "position_ids": torch.cat((text_positions, positions), dim=0),
            "labels": labels, "use_cache": False}


class Stage2Trainer:
    def __init__(self, model, projection, *, seed: int, settings: FitSettings = FitSettings()):
        _projection_array(projection)
        self.model = attach_language_lora(model, seed=seed, settings=settings).train()
        self.projection = projection.train().requires_grad_(True)
        self.seed = seed
        self.settings = settings
        parameters = list(projection.parameters()) + [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = _optimizer(parameters, settings)
        self.order_rng, self.budget_rng, self.noise_rng = _streams(seed)

    def fit(self, records: Sequence[TeacherForcedExample], *, epochs: int | None = None):
        records = _records(records)
        epochs = self.settings.stage2_epochs if epochs is None else epochs
        if not isinstance(epochs, int) or epochs < 1:
            raise ValueError("epochs must be a positive integer")
        history = []
        for epoch in range(epochs):
            for index in self.order_rng.permutation(len(records)):
                record = records[index]
                release, epsilon = _augment(record.clean_features, (0.5, 4.0), self.budget_rng, self.noise_rng)
                inputs = teacher_forced_inputs(self.model, self.projection, release,
                                               record.prompt_token_ids, record.target_token_ids)
                self.optimizer.zero_grad(set_to_none=True)
                loss = self.model(**inputs).loss
                if not loss.isfinite():
                    raise RuntimeError("Nonfinite Stage 2 loss")
                loss.backward()
                self.optimizer.step()
                history.append({"epoch": epoch + 1, "record_id": record.record_id,
                                "loss": float(loss.detach()), "mean_refinement_epsilon": float(epsilon.mean())})
        return history

    def save(self, directory, *, provenance: dict):
        directory = Path(directory)
        if (directory / "adapter").exists():
            raise FileExistsError("Adapter output directory already exists")
        save_projection(self.projection, directory, metadata={
            "stage": 2, "seed": self.seed, "settings": asdict(self.settings), "provenance": provenance})
        self.model.save_pretrained(directory / "adapter", safe_serialization=True)

    def merge_for_inference(self):
        """Finalize this trainer; returned model/projection are frozen."""
        merged = self.model.merge_and_unload(safe_merge=True).eval().requires_grad_(False)
        self.projection.eval().requires_grad_(False)
        return merged, _projection_array(self.projection).double().numpy()
