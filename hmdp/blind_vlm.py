#!/usr/bin/env python3
"""
Blind-but-Smart VLM execution boundary (faithful to Sec. 3.4-3.5 of the paper).

This module implements the central contribution that the original simulated /
raw-pixel pipelines omitted: the VLM **never observes raw screen pixels**.
Instead, each screenshot is

  1. partitioned into a uniform 5x5 = M=25 grid (Sec. 3.3),
  2. independently encoded by a frozen DINOv2 proxy encoder E and L2-clipped
     to a d=256 latent  phi(R_t^(i))                                   (Eq. 3),
  3. perturbed in latent space with per-region (ε,δ)-LDP Gaussian noise
     phi~(R_t^(i)) = phi(R_t^(i)) + N(0, σ²), σ = Δ₂√(2ln(1.25/δ))/ε_t^(i)   (Eq. 4),
  4. projected into the VLM embedding space by a learned layer
     W_proj : R^256 -> R^{d_llm} and injected at the visual-token positions
     (Stage-1 of Sec. 3.5),

after which the VLM produces k_t temperature-sampled coordinate candidates
(Stage-2) that are aggregated by the GoT scorer (Stage-3).  Because only the
privatized latents reach the model, the post-processing immunity of LDP gives
a formally grounded privacy boundary.

W_proj (and the task-direction head producing theta_hat) are *learned offline*
under (ε,δ)-LDP Gaussian noise augmentation; `train_projection_offline` implements
the two-stage protocol of Sec. 4.1.4.

Usage (inference, requires GPU + model weights):
    from hmdp.blind_vlm import BlindVLMPipeline
    pipe = BlindVLMPipeline("qwen2.5-vl-7b", device="cuda")
    pipe.load_models()
    pipe.load_projection("checkpoints/wproj_eq8.pt")  # trained offline
    out = pipe.predict_blind(image, "Click the Submit button", epsilons=[1.0]*25, k=5)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from hmdp.ldp import ProxyEncoder, LocalDifferentialPrivacy
from hmdp.execution_engine import ProjectionLayer
from hmdp.vlm_inference import VLMInferenceEngine, MODEL_REGISTRY, ACTION_TYPES
from hmdp.dinov2_encoder import DINOv2RegionEncoder
from hmdp.constants import GridConfig, ModelDims, GoTHyperparams, LDPConfig
from hmdp.projection import (
    BlindProjector, TaskDirectionHead, LTMPredictor, train_projection_offline,
)


# ═══════════════════════════════════════════════════════════════════════
#  Blind VLM pipeline
# ═══════════════════════════════════════════════════════════════════════

# Note: Projection components (BlindProjector, TaskDirectionHead, training)
# are imported from hmdp.projection module for better modularity.

class BlindVLMPipeline(nn.Module):
    """
    End-to-end Blind-but-Smart pipeline around a decoder VLM.

    The VLM is queried via `inputs_embeds` only: M projected privatized latent
    tokens are prepended as a visual prefix, followed by the (text-only)
    instruction tokens.  No `pixel_values` are ever passed, so the model is
    structurally blind to raw screen content.
    """

    def __init__(
        self,
        model_key: str,
        device: str = "cuda",
        latent_dim: int = ModelDims.LATENT_DEFAULT,  # 256
        grid: int = GridConfig.SIZE,  # 5 -> M = 25 regions (5x5)
        ldp_sensitivity: float = LDPConfig.SENSITIVITY,  # 2.0
    ):
        super().__init__()
        assert model_key in MODEL_REGISTRY, f"Unknown model: {model_key}"
        self.model_key = model_key
        self.device = device
        self.latent_dim = latent_dim
        # Grid resolution is the single source of truth; the region count and
        # the per-region DINOv2 encoder are derived from it so they can never
        # disagree. Higher grids give finer spatial localisation at the cost of
        # more visual tokens; privacy is unaffected (parallel composition over
        # disjoint regions keeps the per-query budget at epsilon).
        self.grid = grid
        self.num_regions = grid * grid

        self.vlm = VLMInferenceEngine(model_key, device=device)
        self.dino = DINOv2RegionEncoder(device=device, grid=self.grid)

        # Client-side: project E(R) -> phi in R^256, then LDP perturb.
        self.proxy_encoder = ProxyEncoder(input_dim=1024, output_dim=latent_dim).to(device)
        self.ldp = LocalDifferentialPrivacy(feature_dim=latent_dim, sensitivity=ldp_sensitivity)

        # Server-side learned components (trained offline).
        self.proj_layer: Optional[ProjectionLayer] = None     # W_proj : R^256 -> R^{d_llm}
        self.task_head: Optional[TaskDirectionHead] = None
        self.ltm_predictor: Optional[LTMPredictor] = None     # theta_pred (Eq. 8)
        # Only use θ_pred at inference once it has been trained (loaded from a
        # checkpoint that contains it); a randomly-initialised head would add
        # meaningless noise to the grounding coordinate.
        self._ltm_predictor_trained: bool = False
        self.llm_dim: Optional[int] = None

    # ── Model loading ────────────────────────────────────────────────────

    def load_models(self):
        self.dino.load()
        self.vlm.load()
        self.llm_dim = self._infer_llm_dim()
        self.proj_layer = BlindProjector(self.latent_dim, self.llm_dim).to(self.device)
        self.task_head = TaskDirectionHead(self.llm_dim, self.latent_dim).to(self.device)
        self.ltm_predictor = LTMPredictor(self.latent_dim).to(self.device)

    def _infer_llm_dim(self) -> int:
        emb = self._token_embedding()
        return emb.weight.shape[1]

    def _token_embedding(self) -> nn.Embedding:
        """Locate the input-token embedding module across VLM architectures."""
        model = self.vlm.model
        # InternVL exposes a language_model submodule.
        if hasattr(model, "language_model") and hasattr(model.language_model, "get_input_embeddings"):
            return model.language_model.get_input_embeddings()
        return model.get_input_embeddings()

    def load_projection(self, path: str):
        """Load offline-trained W_proj + task head weights."""
        if self.proj_layer is None:
            raise RuntimeError("Call load_models() before load_projection().")
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load W_proj checkpoint '{path}'. The file may be "
                "missing, corrupted, or incompatible with torch.load."
            ) from e
        # Guard against a grid-resolution mismatch: the learned weights are all
        # M-independent (per-token), so they would load without a shape error,
        # but the alignment-target pooling and token count differ, so a silent
        # mismatch would corrupt results. Fail loudly instead.
        ckpt_grid = ckpt.get("grid")
        if ckpt_grid is not None and ckpt_grid != self.grid:
            raise ValueError(
                f"Checkpoint was trained with grid={ckpt_grid} (M={ckpt_grid**2}) "
                f"but this pipeline uses grid={self.grid} (M={self.num_regions}). "
                f"Re-run with --grid {ckpt_grid}.")
        if ckpt_grid is None and self.grid != GridConfig.SIZE:
            print(
                f"  [W_proj][WARN] checkpoint has no grid metadata; using it "
                f"with grid={self.grid}. Re-train W_proj with --grid {self.grid} "
                f"for strict grid-resolution experiments."
            )
        self.proj_layer.load_state_dict(ckpt["proj_layer"])
        self.task_head.load_state_dict(ckpt["task_head"])
        self.proxy_encoder.load_state_dict(ckpt["proxy_encoder"])
        # θ_pred (Eq. 8) is optional for backward compatibility with older
        # checkpoints that predate the LTM predictor. When absent, the LTM
        # refinement is skipped at inference (GoT coordinate used unchanged).
        if "ltm_predictor" in ckpt and self.ltm_predictor is not None:
            self.ltm_predictor.load_state_dict(ckpt["ltm_predictor"])
            self._ltm_predictor_trained = True
        else:
            self._ltm_predictor_trained = False
            if "ltm_predictor" not in ckpt:
                print("  [W_proj] checkpoint has no 'ltm_predictor' (Eq. 8 head); "
                      "LTM refinement disabled at inference.")

    def save_projection(self, path: str):
        ckpt = {
            "proj_layer": self.proj_layer.state_dict(),
            "task_head": self.task_head.state_dict(),
            "proxy_encoder": self.proxy_encoder.state_dict(),
            # Persist the grid resolution so eval can validate it matches.
            "grid": self.grid,
            "num_regions": self.num_regions,
            "latent_dim": self.latent_dim,
        }
        if self.ltm_predictor is not None:
            ckpt["ltm_predictor"] = self.ltm_predictor.state_dict()
        torch.save(ckpt, path)

    # ── Client-side privatization (Eq. 3-4) ──────────────────────────────

    def privatize(
        self, image: Image.Image, epsilons: List[float]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run DINOv2 -> proxy projection -> per-region LDP.

        Returns:
            phi:         (M, d) clean latents (never leaves device; returned
                         here only for offline training / analysis)
            phi_private: (M, d) privatized latents phi~  (transmitted)
        """
        assert len(epsilons) == self.num_regions
        with torch.no_grad():
            e_regions = self.dino.extract_regions(image)          # (M, 1024)
            phi = self.proxy_encoder(e_regions)                    # (M, d)
        phi_private = self.ldp.privatize_regions(
            phi.unsqueeze(0), epsilons
        ).squeeze(0)                                               # (M, d)
        return phi, phi_private

    # ── Instruction embedding for theta_hat and the text prefix ──────────

    def _instruction_inputs_embeds(self, instruction: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build text-only token embeddings for the grounding prompt and a mean
        pooled instruction embedding (for theta_hat).

        Returns:
            text_embeds: (1, L, d_llm)
            pooled:      (1, d_llm)
        """
        prompt = self.vlm._build_prompt(instruction)
        tok = self._tokenizer()
        ids = tok(prompt, return_tensors="pt").input_ids.to(self.device)
        emb = self._token_embedding()
        text_embeds = emb(ids)                                     # (1, L, d_llm)
        pooled = text_embeds.mean(dim=1)                           # (1, d_llm)
        return text_embeds, pooled

    def _tokenizer(self):
        if self.vlm.processor is not None:
            return self.vlm.processor.tokenizer
        return self.vlm.tokenizer

    def task_direction(self, instruction: str) -> torch.Tensor:
        """theta_hat = Embed(task instruction) in R^256 (unit-normalised)."""
        _, pooled = self._instruction_inputs_embeds(instruction)
        # VLM embeddings are bf16; task_head runs in float32.
        pooled = pooled.to(next(self.task_head.parameters()).dtype)
        return self.task_head(pooled).squeeze(0)                   # (d,)

    # ── Blind generation: inject projected latents, no raw pixels ────────

    def _build_blind_inputs_embeds(
        self, phi_private: torch.Tensor, instruction: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compose [W_proj(phi~)_1..M ; text_tokens] as inputs_embeds.

        Returns:
            inputs_embeds:  (1, M+L, d_llm)
            attention_mask: (1, M+L)
        """
        visual_tokens = self.proj_layer(phi_private.to(self.device))   # (M, d_llm)
        visual_tokens = visual_tokens.unsqueeze(0)                     # (1, M, d_llm)
        text_embeds, _ = self._instruction_inputs_embeds(instruction)  # (1, L, d_llm)
        # Match the VLM backbone dtype (e.g. bfloat16); W_proj runs in float32.
        visual_tokens = visual_tokens.to(text_embeds.dtype)
        inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)
        attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=self.device)
        return inputs_embeds, attn

    @torch.no_grad()
    def vlm_got_k_paths_blind(
        self,
        phi_private: torch.Tensor,
        instruction: str,
        img_w: int,
        img_h: int,
        k: int,
        temperature: float = GoTHyperparams.T_SAMPLE_BLIND,  # 0.5
    ) -> List[Dict]:
        """
        Stage-2: generate k temperature-sampled coordinate candidates from the
        SAME privatized latents.  Path 0 is greedy; paths 1..k-1 are sampled.
        """
        from hmdp.got.path_generation import (
            _generate_step_logits, _selected_token_logprobs,
            _incremental_token_texts, _coord_token_logprob,
        )

        inputs_embeds, attn = self._build_blind_inputs_embeds(phi_private, instruction)
        gen_model = self._generation_model()

        tok = self._tokenizer()
        pad_id = getattr(tok, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.vlm.model.config, "eos_token_id", 0)

        paths = []
        for i in range(k):
            gen_out = gen_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attn,
                max_new_tokens=256,
                do_sample=(i > 0),
                temperature=temperature if i > 0 else 1.0,
                top_p=0.9 if i > 0 else 1.0,
                pad_token_id=pad_id,
                return_dict_in_generate=True,
                output_logits=True,
            )
            # With inputs_embeds, generate returns only newly generated ids.
            gen_id_list = gen_out.sequences[0].tolist()
            raw = tok.decode(gen_id_list, skip_special_tokens=True)

            step_logits = _generate_step_logits(gen_out)
            token_logprobs = _selected_token_logprobs(step_logits, gen_id_list)
            token_texts = _incremental_token_texts(tok, gen_id_list)
            logit = _coord_token_logprob(token_texts, token_logprobs, raw)

            paths.append({
                "action_type": self.vlm._parse_action_type(raw),
                "pred_point": self.vlm._parse_point(raw, img_w, img_h),
                "raw_output": raw,
                "logit": logit,
            })
        return paths

    def _generation_model(self):
        """Return the module exposing .generate() that accepts inputs_embeds."""
        model = self.vlm.model
        # For Qwen2.5-VL the top-level conditional-generation model accepts
        # inputs_embeds and skips the vision tower when pixel_values is None.
        return model

    # ── One-shot convenience predictor ───────────────────────────────────

    def predict_blind(
        self,
        image: Image.Image,
        instruction: str,
        epsilons: List[float],
        k: int = GoTHyperparams.K_DEFAULT,  # 5
        temperature: float = GoTHyperparams.T_SAMPLE_BLIND,  # 0.5
    ) -> Dict:
        """Full blind step: privatize -> inject -> k-path GoT (returns raw paths)."""
        if self.proj_layer is None:
            raise RuntimeError("Call load_models() (and ideally load_projection()) first.")
        img_w, img_h = image.size
        _, phi_private = self.privatize(image, epsilons)
        paths = self.vlm_got_k_paths_blind(
            phi_private, instruction, img_w, img_h, k=k, temperature=temperature
        )
        return {"paths": paths, "phi_private": phi_private}


# ═══════════════════════════════════════════════════════════════════════
#  Offline W_proj training (Sec. 4.1.4)
# ═══════════════════════════════════════════════════════════════════════

# Note: Training functions (train_projection_offline, _vlm_visual_token_summary,
# _task_supervised_loss) are imported from hmdp.projection.training for
# better modularity. See hmdp/projection/training.py for the full implementation.


# ═══════════════════════════════════════════════════════════════════════
#  CLI: offline W_proj training (two-stage protocol, Sec. 4.1.4)
# ═══════════════════════════════════════════════════════════════════════

def _main():
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(description="Offline W_proj training (Blind-but-Smart)")
    parser.add_argument("--model", default="qwen2.5-vl-7b", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-path", required=True, help="GUI 360 training_data.json")
    parser.add_argument("--image-base", required=True)
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--grid", type=int, default=GridConfig.SIZE,
                        help="Grid resolution G (M=G*G regions). Higher G gives "
                             "finer spatial localisation; privacy is unaffected "
                             "(parallel composition over disjoint regions).")
    parser.add_argument("--align-epochs", type=int, default=1)
    parser.add_argument("--task-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ltm-epochs", type=int, default=2,
                        help="Epochs for the Eq.8 strategic-predictor θ_pred "
                             "(0 disables Stage 3; checkpoint then omits it).")
    parser.add_argument("--ltm-lr", type=float, default=1e-3,
                        help="Learning rate for the θ_pred head.")
    parser.add_argument("--c-star-sigma", type=float, default=0.2,
                        help="Std of the simulated GoT coordinate C*_sim = "
                             "GT + N(0,σ²) used to train θ_pred (models the "
                             "VLM grounding-error magnitude).")
    parser.add_argument("--out", default="checkpoints/wproj_eq8_run.pt")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global RNG seed for sample selection and the LDP "
                             "noise augmentation during training (reproducibility).")
    parser.add_argument("--deterministic", action="store_true",
                        help="Also request deterministic CUDA/cuBLAS kernels.")
    args = parser.parse_args()

    # Seed everything BEFORE sampling / model init so training is reproducible.
    from hmdp.run_real_vlm import set_global_seed
    set_global_seed(args.seed, deterministic=args.deterministic)

    with open(args.data_path) as f:
        data = json.load(f)
    rng = np.random.RandomState(args.seed)
    idx = rng.choice(len(data), min(args.num_samples, len(data)), replace=False)
    samples = [data[i] for i in idx]

    pipe = BlindVLMPipeline(args.model, device=args.device, grid=args.grid)
    pipe.load_models()
    print(f"Grid: {args.grid}x{args.grid} (M={pipe.num_regions} regions)")

    print(f"Stage 1: feature alignment ({args.align_epochs} epoch(s))")
    train_projection_offline(pipe, samples, args.image_base, stage="align",
                             epochs=args.align_epochs, lr=args.lr,
                             eps_range=(3.0, 5.0), device=args.device)
    print(f"Stage 2: task-supervised fine-tuning ({args.task_epochs} epoch(s))")
    train_projection_offline(pipe, samples, args.image_base, stage="task",
                             epochs=args.task_epochs, lr=args.lr,
                             eps_range=(0.1, 5.0), device=args.device)

    if args.ltm_epochs > 0:
        from hmdp.projection import train_ltm_predictor_offline
        print(f"Stage 3: LTM strategic-predictor θ_pred ({args.ltm_epochs} epoch(s))")
        train_ltm_predictor_offline(
            pipe, samples, args.image_base,
            epochs=args.ltm_epochs, lr=args.ltm_lr,
            eps_range=(0.1, 5.0), c_star_sigma=args.c_star_sigma,
            device=args.device, seed=args.seed)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pipe.save_projection(args.out)
    print(f"Saved W_proj + task head + θ_pred to {args.out}")


if __name__ == "__main__":
    _main()
