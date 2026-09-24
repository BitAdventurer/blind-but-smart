"""Frozen, released-latent-only Qwen2.5-VL execution boundary.

Supported upstream API: transformers 4.57.6. The model receives projected
release embeddings, text embeddings, and native M-RoPE positions. It never
receives pixel_values, probe tensors, or clean current-screen representations.
No model weights are downloaded by importing this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
import threading
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .scoring import (GroundingCandidate, GroundingResult, aggregate_grounding,
                      ActionCandidate, ActionFunctionSchema, aggregate_action,
                      parse_action, parse_grounding, score_relevance)


SUPPORTED_TRANSFORMERS = "4.57.6"
_QWEN_CLASS = None


def released_qwen_class():
    """Return the native Qwen class with explicit released-input generation.

    Upstream's default generation preparation infers M-RoPE from input IDs or
    embeddings. Image-pad embeddings have already been replaced here, so we
    preserve the positions computed *before* replacement, including cached
    continuation. This override does not modify attention or model weights.
    """
    global _QWEN_CLASS
    if _QWEN_CLASS is not None:
        return _QWEN_CLASS
    import transformers
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration
    if transformers.__version__ != SUPPORTED_TRANSFORMERS:
        raise RuntimeError(f"Released Qwen adapter requires transformers=={SUPPORTED_TRANSFORMERS}; "
                           f"received {transformers.__version__}")

    class ReleasedQwenForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
        def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None,
            inputs_embeds=None, cache_position=None, use_cache=True,
            latent_position_ids=None, latent_rope_deltas=None, **kwargs,
        ):
            if any(kwargs.get(key) is not None for key in
                   ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw")):
                raise ValueError("Pixels and vision inputs are forbidden at the released-input boundary")
            if latent_position_ids is None or latent_rope_deltas is None or cache_position is None:
                raise ValueError("Precomputed native M-RoPE positions and deltas are required")
            if int(cache_position[0]) == 0:
                if inputs_embeds is None:
                    raise ValueError("Released prefill requires projected inputs_embeds")
                embeddings = inputs_embeds
                positions = latent_position_ids
            else:
                ids = input_ids[:, -cache_position.numel():]
                embeddings = self.get_input_embeddings()(ids)
                positions = (cache_position.reshape(1, 1, -1)
                             + latent_rope_deltas.reshape(1, -1, 1)).expand(3, -1, -1)
            # Three multimodal axes and an independent monotonically increasing
            # text/cache axis are required by the 4.57.6 causal-mask interface.
            text_positions = cache_position.reshape(1, 1, -1).expand(1, embeddings.shape[0], -1)
            return {"inputs_embeds": embeddings,
                    "position_ids": torch.cat((text_positions, positions), dim=0),
                    "attention_mask": attention_mask, "past_key_values": past_key_values,
                    "cache_position": cache_position, "use_cache": use_cache,
                    "logits_to_keep": 1}

        def _expand_inputs_for_generation(self, expand_size=1, is_encoder_decoder=False,
                                          input_ids=None, **model_kwargs):
            if expand_size != 1 or is_encoder_decoder:
                raise ValueError("Batch isolated candidates explicitly; beam expansion is unsupported")
            return input_ids, model_kwargs

    _QWEN_CLASS = ReleasedQwenForConditionalGeneration
    return _QWEN_CLASS


def _as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().double().numpy()
    return np.asarray(value, dtype=np.float64)


def _immutable_model_ref(model_id: str, revision: str | None, *, kind: str = "model"):
    if kind not in {"model", "tokenizer"}:
        raise ValueError("Artifact kind must be model or tokenizer")
    if Path(model_id).is_dir():
        required = "config.json" if kind == "model" else "tokenizer_config.json"
        if not (Path(model_id) / required).is_file():
            raise ValueError(f"Local {kind} path must contain {required}")
        return
    if revision is None or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("Remote models/tokenizers require an immutable 40-character commit revision")


@dataclass(frozen=True)
class GeneratedCandidate:
    index: int
    text: str
    token_ids: tuple[int, ...]
    logprob_mean: float
    terminated: bool


def _per_candidate_sampler(seeds: Sequence[int], device, temperature: float, top_p: float):
    """A deterministic processor draws each candidate from its own RNG stream.

    Generation then takes argmax of the forced one-hot scores. This avoids the
    batch-size-dependent global torch.multinomial stream used by default HF
    sampling. Returned output_logits remain the untouched model logits.
    """
    import torch
    from transformers import LogitsProcessor

    class IndependentPathSampler(LogitsProcessor):
        def __init__(self):
            self.generators = [torch.Generator(device=device).manual_seed(int(seed)) for seed in seeds]

        def __call__(self, input_ids, scores):
            forced = torch.full_like(scores, -float("inf"))
            for row, generator in enumerate(self.generators):
                scaled = scores[row].double() / temperature
                ordered, indices = torch.sort(scaled, descending=True, stable=True)
                probabilities = torch.softmax(ordered, dim=-1)
                # Retain the first token whose cumulative mass crosses top_p.
                remove = probabilities.cumsum(-1) - probabilities >= top_p
                ordered = ordered.masked_fill(remove, -float("inf"))
                chosen = torch.multinomial(torch.softmax(ordered, dim=-1), 1, generator=generator)
                forced[row, indices[chosen]] = 0.0
            return forced

    return IndependentPathSampler()


class ReleasedQwenExecutor:
    """Accept only an already released 25x256 matrix plus public prompt text.

    ``projection`` must be the explicitly supplied frozen bias-free matrix,
    shape (model hidden size, 256). Tiny random models are allowed for software
    tests only; they are never an inference fallback or a paper result.
    Instances serialize requests: model caches must never be shared concurrently.
    """

    def __init__(self, model, tokenizer, projection, *, provenance: Mapping[str, str]):
        import torch
        if not isinstance(model, released_qwen_class()):
            raise TypeError("Load with released_qwen_class() or ReleasedQwenExecutor.from_pretrained()")
        if not provenance:
            raise ValueError("Explicit model/tokenizer/projection provenance is required")
        self.model = model.eval().requires_grad_(False)
        self.tokenizer = tokenizer
        self.projection = _as_numpy(projection).copy()
        hidden = model.get_input_embeddings().weight.shape[1]
        if self.projection.shape != (hidden, 256) or not np.isfinite(self.projection).all():
            raise ValueError(f"Projection must be a finite ({hidden}, 256) matrix")
        self.projection.setflags(write=False)
        self.provenance = dict(provenance)
        self.provenance["projection_array_sha256"] = hashlib.sha256(self.projection.tobytes(order="C")).hexdigest()
        if model.config.vision_config.spatial_merge_size != 2:
            raise ValueError("The 25-slot interface requires native spatial_merge_size=2")
        self.device = model.get_input_embeddings().weight.device
        self.dtype = model.get_input_embeddings().weight.dtype
        self._request_lock = threading.RLock()

    @classmethod
    def from_pretrained(cls, model_id: str, *, revision: str | None,
                        tokenizer_id: str, tokenizer_revision: str | None,
                        projection, device="cpu", dtype="float32", local_files_only=True):
        """Load explicit fitted artifacts; no random projection or model fallback."""
        import torch
        from transformers import AutoTokenizer
        _immutable_model_ref(model_id, revision)
        _immutable_model_ref(tokenizer_id, tokenizer_revision, kind="tokenizer")
        if dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("Unsupported model dtype")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, revision=tokenizer_revision,
                                                  local_files_only=local_files_only, trust_remote_code=False)
        model = released_qwen_class().from_pretrained(
            model_id, revision=revision, local_files_only=local_files_only,
            trust_remote_code=False, torch_dtype=getattr(torch, dtype),
            attn_implementation="eager").to(device)
        return cls(model, tokenizer, projection, provenance={
            "model": model_id, "model_revision": revision or "local-explicit-checkpoint",
            "tokenizer": tokenizer_id, "tokenizer_revision": tokenizer_revision or "local-explicit-checkpoint",
        })

    def prompt_token_ids(self, prompt_text: str):
        """Pinned tokenizer template around 25 contiguous native image pads."""
        import torch
        if not isinstance(prompt_text, str):
            raise TypeError("Prompt must be text")
        marker = "<|vision_start|>" + "<|image_pad|>" * 25 + "<|vision_end|>"
        messages = [{"role": "user", "content": marker + "\n" + prompt_text}]
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        ids = torch.tensor([ids], device=self.device, dtype=torch.long)
        image_id = self.model.config.image_token_id
        vision_start = self.model.config.vision_start_token_id
        positions = (ids[0] == image_id).nonzero().flatten()
        if (len(positions) != 25 or int((ids == vision_start).sum()) != 1
                or not torch.equal(positions, torch.arange(int(positions[0]), int(positions[0]) + 25, device=self.device))
                or int(ids[0, positions[0] - 1]) != vision_start):
            raise ValueError("Tokenizer/chat template does not preserve the declared 25-slot visual wrapper")
        return ids

    def instruction_embeddings(self, scoring_text: str):
        """No wrapper, retrieved text, generated text, or special tokens."""
        import torch
        ids = self.tokenizer.encode(scoring_text, add_special_tokens=False)
        if not ids or any(i in set(self.tokenizer.all_special_ids) for i in ids):
            raise ValueError("The relevance span must contain nonempty nonspecial instruction tokens")
        with torch.inference_mode():
            result = self.model.get_input_embeddings()(torch.tensor(ids, device=self.device))
        return result.cpu().double().numpy()

    def prepare_release_inputs(self, release, input_ids, *, count: int):
        """Compute positions from original IDs before replacing pad embeddings."""
        import torch
        release = _as_numpy(release)
        if release.shape != (25, 256) or not np.isfinite(release).all():
            raise ValueError("Executor accepts exactly a finite released (25, 256) array")
        if not 1 <= count <= 20 or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Provide one unpadded prompt and 1 through 20 candidates")
        image_mask = input_ids[0] == self.model.config.image_token_id
        image_positions = image_mask.nonzero().flatten()
        if (len(image_positions) != 25 or int(image_positions[0]) == 0
                or int(image_positions[-1]) + 1 >= input_ids.shape[1]
                or int((input_ids == self.model.config.vision_start_token_id).sum()) != 1
                or not torch.equal(image_positions, torch.arange(int(image_positions[0]),
                                   int(image_positions[0]) + 25, device=input_ids.device))
                or int(input_ids[0, image_positions[0] - 1]) != self.model.config.vision_start_token_id
                or int(input_ids[0, image_positions[-1] + 1]) != self.model.config.vision_end_token_id):
            raise ValueError("Expected one contiguous 25-pad native visual wrapper")
        mask = torch.ones_like(input_ids)
        positions, deltas = self.model.model.get_rope_index(
            input_ids, image_grid_thw=torch.tensor([[1, 10, 10]], device=self.device), attention_mask=mask)
        with torch.inference_mode():
            embeddings = self.model.get_input_embeddings()(input_ids).clone()
            projected = torch.tensor(release @ self.projection.T, dtype=self.dtype, device=self.device)
            embeddings[0, image_mask] = projected
        return {"inputs_embeds": embeddings.repeat(count, 1, 1),
                "attention_mask": mask.repeat(count, 1),
                "latent_position_ids": positions.repeat(1, count, 1),
                "latent_rope_deltas": deltas.repeat(count, 1)}

    def generate_tokens(self, release, input_ids, *, seeds: Sequence[int], max_new_tokens=32,
                        temperature=0.7, top_p=0.9) -> tuple[GeneratedCandidate, ...]:
        """One batched request, isolated per-candidate caches and RNG streams."""
        import torch
        from transformers import GenerationConfig, LogitsProcessorList
        if not 0 < temperature or not 0 < top_p <= 1 or not 1 <= max_new_tokens <= 128:
            raise ValueError("Invalid decoding configuration")
        prepared = self.prepare_release_inputs(release, input_ids, count=len(seeds))
        eos = self.model.generation_config.eos_token_id
        eos_ids = [eos] if isinstance(eos, int) else list(eos or [])
        if not eos_ids:
            raise ValueError("Pinned EOS token IDs are required")
        pad = self.model.generation_config.pad_token_id
        if pad is None:
            raise ValueError("Pinned pad token ID is required")
        generation = GenerationConfig(
            do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
            eos_token_id=eos_ids, pad_token_id=pad, bos_token_id=self.model.generation_config.bos_token_id,
            use_cache=True, return_dict_in_generate=True, output_logits=True,
            repetition_penalty=1.0, length_penalty=1.0)
        sampler = _per_candidate_sampler(seeds, self.device, temperature, top_p)
        # No request reads another request's mutable rope state or KV cache.
        with self._request_lock:
            self.model.model.rope_deltas = None
            try:
                with torch.inference_mode():
                    output = self.model.generate(**prepared, generation_config=generation,
                                                 logits_processor=LogitsProcessorList([sampler]))
            finally:
                self.model.model.rope_deltas = None
        generated = output.sequences[:, -len(output.logits):]
        candidates = []
        for m in range(len(seeds)):
            tokens = generated[m].tolist()
            end = next((i + 1 for i, token in enumerate(tokens) if token in eos_ids), len(tokens))
            terminated = end > 0 and tokens[end - 1] in eos_ids
            tokens = tokens[:end]
            # Complete conditional output including EOS, excluding trailing pad.
            values = [torch.log_softmax(output.logits[t][m].double(), dim=-1)[token].item()
                      for t, token in enumerate(tokens)]
            text = self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            candidates.append(GeneratedCandidate(m, text, tuple(tokens), float(np.mean(values)), terminated))
        return tuple(candidates)

    def predict_grounding(self, release, *, prompt_text: str, scoring_text: str,
                          seeds: Sequence[int], rule="full") -> tuple[GroundingResult, tuple[GroundingCandidate, ...]]:
        input_ids = self.prompt_token_ids(prompt_text)
        embeddings = self.instruction_embeddings(scoring_text)
        generated = self.generate_tokens(release, input_ids, seeds=seeds, max_new_tokens=32)
        candidates = []
        for candidate in generated:
            coordinate = parse_grounding(candidate.text, terminated=candidate.terminated)
            relevance = score_relevance(release, self.projection, embeddings, [coordinate]) if coordinate is not None else 0.0
            candidates.append(GroundingCandidate(candidate.index, coordinate, coordinate is not None,
                                                 candidate.logprob_mean, relevance, candidate.text, candidate.token_ids))
        return aggregate_grounding(candidates, rule), tuple(candidates)

    def predict_action(self, release, *, prompt_text: str, scoring_text: str,
                       seeds: Sequence[int], schemas: Mapping[str, ActionFunctionSchema]):
        """Generate and aggregate Action candidates from the same completed release.

        The externally pinned schema/alias map contains no evaluation targets.
        Each spatial key contributes separately to relevance, including drag
        endpoints in the same cell. Reference actions are never accepted here.
        """
        if not schemas:
            raise ValueError("Action execution requires an explicit frozen schema")
        input_ids = self.prompt_token_ids(prompt_text)
        embeddings = self.instruction_embeddings(scoring_text)
        generated = self.generate_tokens(release, input_ids, seeds=seeds, max_new_tokens=128)
        candidates = []
        for candidate in generated:
            action = parse_action(candidate.text, schemas, terminated=candidate.terminated)
            if action is None:
                function, arguments, status, relevance = "INVALID", {}, "INVALID", 0.0
            else:
                function, arguments, status = action["function"], action["arguments"], action["status"]
                spatial = [arguments[key] for key in sorted(schemas[function].spatial) if key in arguments]
                relevance = score_relevance(release, self.projection, embeddings, spatial)
            candidates.append(ActionCandidate(candidate.index, function, arguments, status, action is not None,
                                              candidate.logprob_mean, relevance, candidate.text, candidate.token_ids))
        output, feedback = aggregate_action(candidates, schemas)
        return output, feedback, tuple(candidates)
