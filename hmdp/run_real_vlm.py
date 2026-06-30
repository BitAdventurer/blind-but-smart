#!/usr/bin/env python3
"""
Real VLM + H-MDP Pipeline Evaluation.

Runs actual VLM backbones (Aguvis-7B, Qwen-2.5-VL-7B, UGround-7B, GUI-Actor-7B)
through the full H-MDP framework:
  DINOv2 (proxy encoder) → LDP → GoT (k real VLM calls) → LTM → aggregation

Compares:
  - Base: single VLM call, no privacy, no GoT/LTM
  - H-MDP: full pipeline with ε-LDP + k GoT paths + LTM priors

Usage:
    conda activate py358
    python -m hmdp.run_real_vlm --device cuda
    python -m hmdp.run_real_vlm --device cuda --models qwen2.5-vl-7b aguvis-7b
    python -m hmdp.run_real_vlm --device cuda --num-samples 100 --k 3
"""

import argparse
import gc
import json
import os
import sys
import time
import warnings
from typing import Dict, List, Optional

warnings.filterwarnings("ignore", message=".*pad_token_id.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*")
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hmdp.vlm_inference import (
    VLMInferenceEngine, MODEL_REGISTRY, ACTION_TYPES,
    point_in_bbox, point_distance_to_center,
)
from hmdp.ldp import ProxyEncoder, LocalDifferentialPrivacy
from hmdp.ltm import LongTermMemory
from hmdp.projection import TaskDirectionHead, LTMPredictor
from hmdp.dinov2_encoder import DINOv2Encoder, DINOv2RegionEncoder
from hmdp.constants import GridConfig, GoTHyperparams, LDPConfig, ModelDims, RewardConfig

# GoT module imports (refactored from this file)
from hmdp.got import vlm_got_k_paths, got_aggregate

# Adaptive per-region epsilon governor (recovered SAC meta-policy). Optional:
# import failures must NOT break the legacy uniform-epsilon evaluation, so this
# is guarded and the symbol is set to None on any error.
try:
    from sac_governor import AdaptiveEpsilonGovernor
except Exception:  # pragma: no cover
    try:
        from hmdp.sac_governor import AdaptiveEpsilonGovernor
    except Exception:
        AdaptiveEpsilonGovernor = None


# ═══════════════════════════════════════════════════════════════════════
#  Reproducibility
# ═══════════════════════════════════════════════════════════════════════

# Default seed used for sample selection and all stochastic ops (GoT
# temperature sampling, LDP Gaussian noise) unless overridden via --seed.
DEFAULT_SEED = 42


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """
    Seed every RNG that affects an evaluation/training run so results are
    reproducible: Python ``random``, NumPy's global RNG, and torch (CPU + all
    CUDA devices).

    Without this, GoT temperature sampling (``generate(do_sample=True)``) and
    the LDP Gaussian noise (``torch.randn``) draw from unseeded global RNGs, so
    the same configuration yields different point accuracy across runs.

    Args:
        seed:          integer seed.
        deterministic: if True, additionally request deterministic CUDA/cuBLAS
                       kernels (``torch.use_deterministic_algorithms``). This
                       removes residual nondeterminism from some GPU ops at a
                       possible speed cost and may raise for unsupported ops.
    """
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as e:  # pragma: no cover - best effort
            print(f"  [seed] deterministic algorithms unavailable: {e}")
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# GoT aggregation hyperparameters (paper Sec. 4; L571 / L1076).
# Using centralized constants for consistency across codebase.
ALPHA_W = GoTHyperparams.ALPHA_W      # alpha_w: equal weight to VLM logit and value alignment
T_AGG = GoTHyperparams.T_AGG          # T_agg: aggregation softmax temperature
TAU_LTM = GoTHyperparams.TAU_LTM      # tau_LTM: U_t threshold above which the LTM is queried


# ═══════════════════════════════════════════════════════════════════════
#  Real H-MDP Pipeline
# ═══════════════════════════════════════════════════════════════════════

class RealHMDPPipeline:
    """Full H-MDP pipeline with real VLM + DINOv2 + LDP + GoT + LTM."""

    def __init__(self, model_key: str, device: str = "cuda",
                 task_ckpt: Optional[str] = None,
                 grid: int = GridConfig.SIZE):
        self.model_key = model_key
        self.device = device

        # VLM backbone
        self.vlm = VLMInferenceEngine(model_key, device=device)

        # DINOv2 proxy encoder (full-image, used by the analysis paths)
        self.dino = DINOv2Encoder(device=device)
        # Per-region DINOv2 encoder for the M=5x5 grid latent psi (Eq. 3).
        self.grid = grid
        self.region_encoder = DINOv2RegionEncoder(device=device, grid=self.grid)

        # H-MDP components. sensitivity = Delta_2 = 2.0 (paper: L2-clipped
        # latents give Delta_2 <= 2 regardless of d).
        self.proxy_encoder = ProxyEncoder(
            input_dim=ModelDims.DINOV2_OUTPUT,
            output_dim=ModelDims.LATENT_DEFAULT,
        ).to(device)
        self.ldp = LocalDifferentialPrivacy(
            feature_dim=ModelDims.LATENT_DEFAULT,
            sensitivity=LDPConfig.SENSITIVITY,
        )
        self.ltm = LongTermMemory(
            embedding_dim=ModelDims.LATENT_DEFAULT,  # 256
            capacity=500,
            top_k=5,
        ).to(device)

        # Server-side grounding head theta_hat = Embed(instruction) (Eq. got_score
        # semantic term <psi_j, theta_hat>). Built in load_models() once llm_dim
        # is known; weights are loaded from task_ckpt when available. When no
        # valid head is loaded, task_direction() returns None and the GoT
        # aggregation degrades to logit-only scoring (the legacy behaviour).
        self.task_ckpt = task_ckpt
        self.task_head: Optional[TaskDirectionHead] = None
        self.llm_dim: Optional[int] = None
        # Eq.8 strategic-prediction head θ_pred. This raw-pixel/table harness
        # does not load one by default (its proxy_encoder is not the trained
        # blind checkpoint), so LTM refinement stays disabled here and the
        # faithful Eq.8 path is the blind pipeline (evaluate_hmdp_blind).
        self.ltm_predictor = None
        self._ltm_predictor_trained: bool = False

    def load_models(self):
        """Load DINOv2 (full + per-region) and the VLM, then the grounding head."""
        self.dino.load()
        self.region_encoder.load()
        self.vlm.load()
        self._init_task_head()

    # ── Server-side grounding head (theta_hat) ───────────────────────────

    def _token_embedding(self):
        """Locate the input-token embedding module across VLM architectures."""
        model = self.vlm.model
        if hasattr(model, "language_model") and hasattr(
            model.language_model, "get_input_embeddings"
        ):
            return model.language_model.get_input_embeddings()
        return model.get_input_embeddings()

    def _init_task_head(self) -> None:
        """Build the TaskDirectionHead and load its weights if a ckpt exists.

        Mirrors BlindVLMPipeline.task_head construction so the raw-pixel /
        table paths can reproduce the same semantic term as the blind path.
        On any failure (no ckpt, missing key, shape/arch mismatch) the head is
        left as None so GoT scoring falls back to logit-only — an UNTRAINED
        head would inject noise into <psi_j, theta_hat> and is never used.
        """
        try:
            self.llm_dim = self._token_embedding().weight.shape[1]
        except Exception as e:
            print(f"  [task-head][WARN] could not infer llm_dim ({e!r}); "
                  f"semantic term disabled (logit-only GoT).")
            self.task_head = None
            return

        head = TaskDirectionHead(self.llm_dim, 256).to(self.device)
        if not self.task_ckpt:
            print("  [task-head][WARN] no task_ckpt supplied; semantic term "
                  "disabled (logit-only GoT).")
            self.task_head = None
            return
        if not os.path.exists(self.task_ckpt):
            print(f"  [task-head][WARN] task_ckpt not found: {self.task_ckpt}; "
                  f"semantic term disabled (logit-only GoT).")
            self.task_head = None
            return
        try:
            ckpt = torch.load(self.task_ckpt, map_location=self.device,
                              weights_only=False)
            sd = ckpt.get("task_head") if isinstance(ckpt, dict) else None
            if sd is None:
                raise KeyError("'task_head' sub-state absent")
            head.load_state_dict(sd)  # strict: untrained/partial head is unusable
            head.eval()
            self.task_head = head
            print(f"  [task-head] loaded (semantic term active; "
                  f"ckpt={self.task_ckpt})")
        except Exception as e:
            print(f"  [task-head][WARN] failed to load task_head from "
                  f"{self.task_ckpt}: {e!r}; semantic term disabled (logit-only GoT).")
            self.task_head = None

    def load_hmdp_checkpoint(
        self,
        path: str,
        *,
        strict_task_head: bool = True,
    ) -> None:
        """Load trained H-MDP components from a W_proj/Eq.8 checkpoint.

        RealHMDPPipeline is the raw-pixel/table harness, so it does not use the
        W_proj layer itself. It still needs the checkpoint's trained proxy
        encoder, task-direction head, and optional Eq.8 LTM predictor; otherwise
        the LDP/GoT/LTM rows silently run with random local modules.
        """
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"H-MDP checkpoint not found: {path}")
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load H-MDP checkpoint: {path}") from e
        if not isinstance(ckpt, dict):
            raise ValueError(f"H-MDP checkpoint must be a dict: {path}")

        ckpt_grid = ckpt.get("grid")
        if ckpt_grid is not None and int(ckpt_grid) != int(self.grid):
            raise ValueError(
                f"Checkpoint was trained with grid={ckpt_grid}, but this "
                f"pipeline uses grid={self.grid}.")
        if ckpt_grid is None and self.grid != GridConfig.SIZE:
            print(f"  [H-MDP][WARN] checkpoint has no grid metadata; "
                  f"using it with grid={self.grid}.")

        if "proxy_encoder" not in ckpt:
            raise KeyError("H-MDP checkpoint missing 'proxy_encoder'")
        self.proxy_encoder.load_state_dict(ckpt["proxy_encoder"])
        self.proxy_encoder.eval()

        task_sd = ckpt.get("task_head")
        if task_sd is None:
            msg = "H-MDP checkpoint missing 'task_head'; semantic GoT disabled."
            if strict_task_head:
                raise KeyError(msg)
            print(f"  [task-head][WARN] {msg}")
            self.task_head = None
        else:
            try:
                if self.llm_dim is None:
                    self.llm_dim = self._token_embedding().weight.shape[1]
                head = TaskDirectionHead(self.llm_dim, ModelDims.LATENT_DEFAULT).to(self.device)
                head.load_state_dict(task_sd)
                head.eval()
                self.task_head = head
                print(f"  [H-MDP] loaded proxy_encoder + task_head from {path}")
            except Exception as e:
                msg = (f"failed to load task_head from {path}: {e!r}; "
                       "semantic GoT disabled.")
                if strict_task_head:
                    raise RuntimeError(msg) from e
                print(f"  [task-head][WARN] {msg}")
                self.task_head = None

        if "ltm_predictor" in ckpt:
            predictor = LTMPredictor(ModelDims.LATENT_DEFAULT).to(self.device)
            predictor.load_state_dict(ckpt["ltm_predictor"])
            predictor.eval()
            self.ltm_predictor = predictor
            self._ltm_predictor_trained = True
            print("  [H-MDP] loaded Eq.8 LTM predictor")
        else:
            self.ltm_predictor = None
            self._ltm_predictor_trained = False
            print("  [H-MDP] checkpoint has no Eq.8 LTM predictor; "
                  "LTM refinement disabled.")

    @torch.no_grad()
    def task_direction(self, instruction: str) -> Optional[torch.Tensor]:
        """theta_hat = Embed(instruction) in R^256, or None if no head is loaded.

        Returns None when the grounding head is unavailable; callers must treat
        None as 'no semantic term' (logit-only GoT).
        """
        if self.task_head is None:
            return None
        prompt = self.vlm._build_prompt(instruction)
        tok = self.vlm.processor.tokenizer if self.vlm.processor is not None \
            else self.vlm.tokenizer
        ids = tok(prompt, return_tensors="pt").input_ids.to(self.device)
        pooled = self._token_embedding()(ids).mean(dim=1)          # (1, d_llm)
        pooled = pooled.to(next(self.task_head.parameters()).dtype)
        return self.task_head(pooled).squeeze(0)                   # (256,)

    def unload_models(self):
        """Free GPU memory (idempotent — safe to call more than once)."""
        if getattr(self.vlm, "model", None) is not None:
            self.vlm.unload()
        if getattr(self.dino, "model", None) is not None:
            self.dino.unload()
        if getattr(self.region_encoder, "model", None) is not None:
            self.region_encoder.unload()
        torch.cuda.empty_cache()
        gc.collect()

    def evaluate_base(
        self,
        data: List[Dict],
        indices: np.ndarray,
        image_base: str,
    ) -> Dict:
        """
        Base evaluation: single VLM call per sample, no LDP/GoT/LTM.
        """
        correct_actions = 0
        point_hits = 0
        joint_successes = 0
        distances = []
        skipped = 0
        evaluated = 0

        for i, idx in enumerate(indices):
            sample = data[idx]
            instruction, gt_action, gt_bbox_raw, img_path = _parse_sample(sample, image_base)
            if img_path is None or gt_bbox_raw is None:
                skipped += 1
                continue

            try:
                img = Image.open(img_path)
                img_w, img_h = img.size
            except Exception:
                skipped += 1
                continue

            gt_bbox_norm = _normalize_bbox(gt_bbox_raw, img_w, img_h)

            try:
                pred = self.vlm.predict(img_path, instruction)
            except Exception as e:
                skipped += 1
                continue

            evaluated += 1
            action_ok = (pred["action_type"] == gt_action)
            correct_actions += int(action_ok)

            if pred["pred_point"] is not None:
                hit = point_in_bbox(pred["pred_point"], gt_bbox_norm)
                dist = point_distance_to_center(pred["pred_point"], gt_bbox_norm)
            else:
                hit = False
                dist = 1.0

            point_hits += int(hit)
            joint_successes += int(action_ok and hit)
            distances.append(dist)

            if (i + 1) % 50 == 0:
                _print_progress(i+1, len(indices), evaluated, correct_actions, point_hits, distances)

        return _compute_metrics(evaluated, correct_actions, point_hits, distances, skipped, joint_successes)

    def evaluate_hmdp(
        self,
        data: List[Dict],
        indices: np.ndarray,
        image_base: str,
        epsilon: float = LDPConfig.EPSILON_DEFAULT,  # 1.0
        k: int = GoTHyperparams.K_DEFAULT,  # 5
        temperature: float = GoTHyperparams.T_SAMPLE,  # 0.7
        governor=None,
    ) -> Dict:
        """
        H-MDP evaluation: DINOv2 → LDP → GoT (k VLM calls) → LTM → aggregation.

        If ``governor`` (AdaptiveEpsilonGovernor) is supplied and enabled, the
        per-region privacy budget is chosen adaptively from s_t = [U_t, λ_t]
        instead of the uniform scalar ``epsilon``. The carried U_t is updated
        from each sample's post-GoT uncertainty u_t.
        """
        correct_actions = 0
        point_hits = 0
        joint_successes = 0
        distances = []
        skipped = 0
        evaluated = 0
        ltm_used = 0
        if governor is not None:
            governor.reset_episode()

        for i, idx in enumerate(indices):
            sample = data[idx]
            instruction, gt_action, gt_bbox_raw, img_path = _parse_sample(sample, image_base)
            if img_path is None or gt_bbox_raw is None:
                skipped += 1
                continue

            try:
                img = Image.open(img_path).convert("RGB")
                img_w, img_h = img.size
            except Exception:
                skipped += 1
                continue

            gt_bbox_norm = _normalize_bbox(gt_bbox_raw, img_w, img_h)

            # ── Step 1: Per-region DINOv2 grid latent (Eq. 3) ──
            # Encode the M=5x5 grid so psi (privatized per-region latent) is
            # available for U_t-gated LTM. The GoT semantic term is enabled
            # when a trained task_head checkpoint is loaded (see task_direction).
            e_regions = self.region_encoder.extract_regions(img)     # (M, 1024)
            phi_grid = self.proxy_encoder(e_regions)                 # (M, 256)

            # ── Step 2: Per-region LDP noise injection (Eq. 4) ──
            # Adaptive per-region epsilon from the SAC meta-policy when a
            # governor is active; otherwise the legacy uniform broadcast.
            if governor is not None and governor.enabled:
                epsilons_t, k_t = governor.compute_epsilons(phi_grid)
            else:
                epsilons_t = [epsilon] * phi_grid.shape[0]
                k_t = k
            psi_grid = self.ldp.privatize_regions(
                phi_grid.unsqueeze(0), epsilons_t
            ).squeeze(0)                                             # (M, 256)
            state_embedding = psi_grid.mean(dim=0)                   # (256,)

            # ── Step 3: GoT k-path reasoning (real VLM) ──
            # When adaptive, the SAC policy also chooses k (k_t); else use k.
            try:
                paths = vlm_got_k_paths(
                    self.vlm, img_path, instruction,
                    k=k_t, temperature=temperature,
                )
            except Exception as e:
                skipped += 1
                continue

            # ── Step 4: GoT aggregation (Eqs. got_score / got_agg) ──
            # Semantic term ⟨psi_j, theta_hat⟩ is active when a trained
            # task_head is loaded; task_direction() returns None otherwise,
            # in which case got_aggregate() falls back to logit-only scoring.
            theta_hat = self.task_direction(instruction)
            pred_action, pred_point, u_t = got_aggregate(
                paths, psi_grid, theta_hat,
                alpha_w=ALPHA_W, T_agg=T_AGG, grid=self.grid,
            )
            # Carry this sample's uncertainty into the NEXT state's U_t.
            if governor is not None:
                governor.update_uncertainty(u_t)

            # ── Step 5: Conditional LTM gate (Eq. 7/8) ──
            # The trained strategic-prediction head θ_pred refines C* from the
            # retrieved memory. It is only applied when a trained head is present
            # (loaded into this pipeline); this raw-pixel/table harness does not
            # carry one by default, so the GoT coordinate is used unchanged and
            # the faithful Eq.8 refinement is evaluated in the blind pipeline.
            predictor = getattr(self, "ltm_predictor", None)
            predictor_ready = (predictor is not None
                               and getattr(self, "_ltm_predictor_trained", False))
            if (predictor_ready and u_t > TAU_LTM
                    and len(self.ltm.episodes) > 0 and pred_point is not None):
                with torch.no_grad():
                    phi_summary = state_embedding.to(self.device)
                    ret_coord, ret_emb, conf = self.ltm.retrieve_prior_features(
                        phi_summary, device=self.device)
                    c_star = torch.tensor(pred_point, device=self.device, dtype=torch.float32)
                    refined = predictor(
                        phi_summary.float(), c_star, ret_coord.float(),
                        ret_emb.float(), conf.float())
                pred_point = refined.detach().cpu().tolist()
                ltm_used += 1

            evaluated += 1
            action_ok = (pred_action == gt_action)
            correct_actions += int(action_ok)

            if pred_point is not None:
                hit = point_in_bbox(pred_point, gt_bbox_norm)
                dist = point_distance_to_center(pred_point, gt_bbox_norm)
            else:
                hit = False
                dist = 1.0

            point_hits += int(hit)
            joint_successes += int(action_ok and hit)
            distances.append(dist)

            # ── Step 6: Store episodes in LTM (FIFO buffer of past steps) ──
            if pred_point is not None:
                ep_reward = 1.0 if (hit and action_ok) else (0.5 if dist < 0.3 else 0.0)
                if ep_reward > 0:
                    self.ltm.store_episode(
                        state_embedding=state_embedding,
                        action_taken=ACTION_TYPES.index(pred_action) if pred_action in ACTION_TYPES else 0,
                        bbox_target=torch.tensor(pred_point + pred_point),
                        epsilon_used=float(np.mean(epsilons_t)),
                        k_used=k_t,
                        reward=ep_reward,
                        sensitivity=0.5, uncertainty=u_t, success=(hit and action_ok),
                    )

            if (i + 1) % 20 == 0:
                _print_progress(i+1, len(indices), evaluated, correct_actions,
                                point_hits, distances, ltm_count=ltm_used)

        metrics = _compute_metrics(evaluated, correct_actions, point_hits, distances, skipped, joint_successes)
        metrics["ltm_episodes"] = len(self.ltm.episodes)
        metrics["ltm_retrievals"] = ltm_used
        return metrics


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def _parse_sample(sample: Dict, image_base: str):
    """Parse instruction, GT action, GT bbox, image path from a GUI 360 sample."""
    instruction = ""
    gt_action = "click"

    if "conversation" in sample:
        for msg in sample["conversation"]:
            if msg["from"] == "human":
                if "\nThe instruction is:\n" in msg["value"]:
                    instruction = msg["value"].split("\nThe instruction is:\n")[1].split("\n\n")[0]
                else:
                    instruction = msg["value"].replace("<image>\n", "").split("\n")[0]
                break
        for msg in sample["conversation"]:
            if msg["from"] == "gpt":
                val_lower = msg["value"].lower()
                for act in ACTION_TYPES:
                    if act in val_lower:
                        gt_action = act
                        break
                break

    gt_bbox_raw = sample.get("bbox", None)
    images = sample.get("images", [])
    if not images or gt_bbox_raw is None:
        return instruction, gt_action, None, None

    img_path = os.path.join(image_base, images[0])
    if not os.path.exists(img_path):
        return instruction, gt_action, gt_bbox_raw, None

    return instruction, gt_action, gt_bbox_raw, img_path


def _normalize_bbox(bbox_raw, img_w, img_h):
    return [
        bbox_raw[0] / max(img_w, 1),
        bbox_raw[1] / max(img_h, 1),
        bbox_raw[2] / max(img_w, 1),
        bbox_raw[3] / max(img_h, 1),
    ]


def _print_progress(step, total, evaluated, correct_actions, point_hits, distances,
                    ltm_count=None):
    n = max(evaluated, 1)
    msg = (f"    [{step}/{total}] act={correct_actions/n:.3f}  "
           f"point={point_hits/n:.3f}  dist={np.mean(distances):.4f}")
    if ltm_count is not None:
        msg += f"  ltm_used={ltm_count}"
    print(msg)


def _compute_metrics(evaluated, correct_actions, point_hits, distances, skipped, joint_successes=None):
    n = max(evaluated, 1)
    if joint_successes is None:
        joint_successes = point_hits  # fallback: treat grounding hit as success
    return {
        "num_evaluated": evaluated,
        "num_skipped": skipped,
        "action_accuracy": correct_actions / n,
        "point_accuracy": point_hits / n,
        "grounding_accuracy": point_hits / n,
        "avg_distance": float(np.mean(distances)) if distances else 1.0,
        "success_rate": joint_successes / n,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Blind-but-Smart evaluation (faithful boundary, no raw pixels)
# ═══════════════════════════════════════════════════════════════════════

def evaluate_hmdp_blind(
    model_key: str,
    data: List[Dict],
    indices: np.ndarray,
    image_base: str,
    device: str,
    epsilon: float,
    k: int,
    temperature: float,
    wproj_ckpt: Optional[str] = None,
    governor=None,
    grid: int = GridConfig.SIZE,
    allow_random_wproj: bool = False,
) -> Dict:
    """
    Faithful H-MDP evaluation where the VLM NEVER sees raw pixels: privatized
    DINOv2 latents are projected by W_proj and injected at the visual-token
    positions (hmdp.blind_vlm.BlindVLMPipeline). theta_hat = Embed(instruction).

    ``grid`` selects the G×G region partition (M=G²); it must match the grid the
    supplied W_proj checkpoint was trained at (load_projection validates this).

    If ``governor`` (AdaptiveEpsilonGovernor) is supplied and enabled, the
    per-region epsilon is chosen adaptively from s_t = [U_t, λ_t] computed on the
    CLEAN proxy latents φ (obtained before privatisation), instead of the
    uniform ``epsilon`` broadcast.
    """
    from hmdp.blind_vlm import BlindVLMPipeline

    pipe = BlindVLMPipeline(model_key, device=device, grid=grid)
    pipe.load_models()
    if wproj_ckpt and os.path.exists(wproj_ckpt):
        pipe.load_projection(wproj_ckpt)
        print(f"  Loaded offline-trained W_proj from {wproj_ckpt}")
    else:
        msg = (
            "No trained W_proj checkpoint was supplied/found for the Blind-but-Smart "
            "path. Pass --wproj-ckpt for paper-faithful evaluation, or explicitly "
            "use --allow-random-wproj for a smoke test."
        )
        if not allow_random_wproj:
            raise FileNotFoundError(msg)
        print(f"  [WARN] {msg} Using random projection.")

    ltm = LongTermMemory(embedding_dim=256, capacity=500, top_k=5).to(device)
    correct_actions = point_hits = joint_successes = evaluated = skipped = ltm_used = 0
    distances = []
    epsilons = [epsilon] * pipe.num_regions   # legacy uniform default
    if governor is not None:
        governor.reset_episode()

    for i, idx in enumerate(indices):
        sample = data[idx]
        instruction, gt_action, gt_bbox_raw, img_path = _parse_sample(sample, image_base)
        if img_path is None or gt_bbox_raw is None:
            skipped += 1
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            img_w, img_h = img.size
        except Exception:
            skipped += 1
            continue
        gt_bbox_norm = _normalize_bbox(gt_bbox_raw, img_w, img_h)

        try:
            k_t = k
            if governor is not None and governor.enabled:
                # Obtain CLEAN proxy latents φ (M, d) BEFORE privatisation so the
                # governor can compute λ_t = entropy(φ). Mirrors the first half
                # of BlindVLMPipeline.privatize (DINOv2 → proxy projection).
                with torch.no_grad():
                    e_regions = pipe.dino.extract_regions(img)       # (M, 1024)
                    phi_clean = pipe.proxy_encoder(e_regions)        # (M, d)
                epsilons_t, k_t = governor.compute_epsilons(phi_clean)
                phi_private = pipe.ldp.privatize_regions(
                    phi_clean.unsqueeze(0), epsilons_t).squeeze(0)   # (M, d)
            else:
                epsilons_t = epsilons
                _, phi_private = pipe.privatize(img, epsilons)       # (M, d) psi grid
            theta_hat = pipe.task_direction(instruction)            # Embed(instruction)
            paths = pipe.vlm_got_k_paths_blind(
                phi_private, instruction, img_w, img_h, k=k_t, temperature=temperature)
        except Exception as e:
            print(f"    [ERROR] sample {idx}: {e}")
            skipped += 1
            continue

        # GoT aggregation (Eqs. got_score / got_agg): per-path psi_j is taken
        # from the enclosing grid region of c_j inside phi_private (M, d).
        pred_action, pred_point, u_t = got_aggregate(
            paths, phi_private, theta_hat.detach().cpu(),
            alpha_w=ALPHA_W, T_agg=T_AGG, grid=pipe.grid)
        # Carry this sample's uncertainty into the NEXT state's U_t.
        if governor is not None:
            governor.update_uncertainty(u_t)

        # Conditional LTM gate (Eq. 7/8): when the reasoning variance is high,
        # retrieve K_ret = argmax cos(φ̃, φ_j) and refine the GoT coordinate C*
        # with the trained strategic-prediction head θ_pred:
        #     C_final = θ_pred(φ̃(s_τ), C*, K_ret).
        # θ_pred is a bounded residual head, so an uninformative memory leaves
        # C* essentially unchanged. Requires a checkpoint containing the trained
        # head; otherwise the GoT coordinate is used as-is.
        predictor = getattr(pipe, "ltm_predictor", None)
        predictor_ready = predictor is not None and getattr(pipe, "_ltm_predictor_trained", False)
        if (predictor_ready and u_t > TAU_LTM
                and len(ltm.episodes) > 0 and pred_point is not None):
            with torch.no_grad():
                phi_summary = phi_private.mean(dim=0).to(pipe.device)
                ret_coord, ret_emb, conf = ltm.retrieve_prior_features(
                    phi_summary, device=pipe.device)
                c_star = torch.tensor(pred_point, device=pipe.device, dtype=torch.float32)
                refined = predictor(
                    phi_summary.float(), c_star, ret_coord.float(),
                    ret_emb.float(), conf.float())
            pred_point = refined.detach().cpu().tolist()
            ltm_used += 1

        evaluated += 1
        action_ok = (pred_action == gt_action)
        correct_actions += int(action_ok)
        if pred_point is not None:
            hit = point_in_bbox(pred_point, gt_bbox_norm)
            dist = point_distance_to_center(pred_point, gt_bbox_norm)
        else:
            hit, dist = False, 1.0
        point_hits += int(hit)
        joint_successes += int(action_ok and hit)
        distances.append(dist)

        # Store both joint successes and near-misses (point close to the target)
        # so the LTM accumulates usable spatial priors instead of cold-starting
        # almost empty. The reward distinguishes the two for downstream use.
        if pred_point is not None:
            joint_ok = bool(hit and action_ok)
            near_miss = dist < RewardConfig.DISTANCE_THRESHOLD
            if joint_ok or near_miss:
                ltm.store_episode(
                    state_embedding=phi_private.mean(dim=0).detach().cpu(),
                    action_taken=ACTION_TYPES.index(pred_action) if pred_action in ACTION_TYPES else 0,
                    bbox_target=torch.tensor(pred_point + pred_point),
                    epsilon_used=float(np.mean(epsilons_t)),
                    k_used=k_t,
                    reward=1.0 if joint_ok else 0.5,
                    sensitivity=0.5, uncertainty=u_t, success=joint_ok)

        if (i + 1) % 20 == 0:
            _print_progress(i + 1, len(indices), evaluated, correct_actions,
                            point_hits, distances, ltm_count=ltm_used)

    metrics = _compute_metrics(evaluated, correct_actions, point_hits, distances,
                               skipped, joint_successes)
    metrics["ltm_episodes"] = len(ltm.episodes)
    metrics["ltm_retrievals"] = ltm_used
    metrics["blind"] = True
    pipe.vlm.unload()
    pipe.dino.unload()
    return metrics


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Real VLM H-MDP Pipeline")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--models", nargs="+",
                        default=list(MODEL_REGISTRY.keys()),
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Models to evaluate")
    parser.add_argument("--num-samples", type=int, default=200,
                        help="Number of GUI 360 samples per model")
    parser.add_argument("--epsilon", type=float, default=5.0,
                        help="LDP privacy budget ε")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of GoT reasoning paths")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Temperature for VLM sampling (GoT paths)")
    # Blind-but-Smart latent injection is the DEFAULT (paper-faithful boundary:
    # the VLM never sees raw pixels). Use --raw-pixel to run the ablation where
    # the VLM is given the original screenshot directly.
    parser.add_argument("--raw-pixel", action="store_true",
                        help="ABLATION ONLY: feed raw screenshots to the VLM "
                             "instead of injecting privatized DINOv2 latents via "
                             "W_proj. This breaks the Blind-but-Smart privacy "
                             "boundary and is NOT the paper's main configuration.")
    parser.add_argument("--wproj-ckpt", type=str, default="checkpoints/wproj_eq8.pt",
                        help="Path to an offline-trained W_proj checkpoint "
                             "(see hmdp.blind_vlm.train_projection_offline).")
    parser.add_argument("--allow-random-wproj", action="store_true",
                        help="Smoke-test only: allow the Blind-but-Smart path "
                             "to run with a randomly initialized W_proj when "
                             "--wproj-ckpt is missing. Paper-faithful runs "
                             "should not use this.")
    parser.add_argument("--task-ckpt", type=str, default=None,
                        help="Path to a checkpoint containing a trained "
                             "'task_head' state dict. When supplied, the GoT "
                             "semantic term <psi_j, theta_hat> is enabled; "
                             "otherwise GoT scoring degrades to logit-only.")
    parser.add_argument("--adaptive", action="store_true",
                        help="Use the recovered SAC meta-policy to choose a "
                             "per-region epsilon vector (and k) from the "
                             "governance state s_t=[U_t, lambda_t] instead of "
                             "the uniform --epsilon broadcast. Requires a "
                             "loadable --sac-ckpt; otherwise degrades to "
                             "uniform epsilon with a warning.")
    parser.add_argument("--sac-ckpt", type=str, default=None,
                        help="Path to a trained SAC meta-policy checkpoint "
                             "(SACMetaPolicy.save format). Only the 'actor' "
                             "sub-state is required for inference.")
    parser.add_argument("--sac-carry-uncertainty", action="store_true",
                        help="Carry u_t across samples into the next state's "
                             "U_t (simulation multi-step semantics). Default is "
                             "per-sample reset (order-independent, reproducible).")
    parser.add_argument("--data-path", type=str,
                        default="gui360_full/processed_data/action_prediction_train_resize/training_data.json")
    parser.add_argument("--image-base", type=str,
                        default="gui360_full/processed_data/action_prediction_train_resize/")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Global RNG seed for sample selection, GoT "
                             "temperature sampling, and LDP noise (reproducibility).")
    parser.add_argument("--deterministic", action="store_true",
                        help="Also request deterministic CUDA/cuBLAS kernels "
                             "(slower; removes residual GPU nondeterminism).")
    parser.add_argument("--grid", type=int, default=GridConfig.SIZE,
                        help="Blind-path grid resolution G (M=G*G regions). Must "
                             "match the --grid the W_proj checkpoint was trained "
                             "at (validated on load).")
    parser.add_argument("--out", type=str,
                        default="results/json/real_vlm_hmdp_results.json",
                        help="Path to write result JSON. Use a smoke-specific "
                             "path for short validation runs.")
    args = parser.parse_args()

    # Seed everything BEFORE any sampling / model init so runs are reproducible.
    set_global_seed(args.seed, deterministic=args.deterministic)

    print("=" * 70)
    print("  Real VLM + H-MDP Pipeline Evaluation")
    print(f"  ε={args.epsilon}, k={args.k}, τ={args.temperature}")
    print(f"  Samples per model: {args.num_samples}")
    print(f"  Models: {', '.join(args.models)}")
    print(f"  Seed: {args.seed}{' (deterministic)' if args.deterministic else ''}")
    print("=" * 70)

    # ── Adaptive per-region epsilon governor (recovered SAC meta-policy) ──
    # Built once and reused across models. When --adaptive is off, or the
    # checkpoint cannot be loaded, governor stays None / disabled and both eval
    # paths fall back to the uniform --epsilon broadcast.
    governor = None
    if args.adaptive:
        if AdaptiveEpsilonGovernor is None:
            print("  [WARN] --adaptive requested but sac_governor could not be "
                  "imported; using uniform epsilon.")
        else:
            governor = AdaptiveEpsilonGovernor(
                ckpt_path=args.sac_ckpt,
                device=args.device,
                fallback_epsilon=args.epsilon,
                fallback_k=args.k,
                deterministic=True,
                reset_per_sample=not args.sac_carry_uncertainty,
                verbose=True,
            )
            mode_eps = "ADAPTIVE per-region" if governor.enabled else "uniform (fallback)"
            print(f"  Epsilon policy: {mode_eps}")

    # Load dataset
    print(f"\n  Loading dataset: {args.data_path}")
    with open(args.data_path) as f:
        data = json.load(f)
    print(f"  Total samples: {len(data)}")

    rng = np.random.RandomState(args.seed)
    indices = rng.choice(len(data), min(args.num_samples, len(data)), replace=False)

    all_results = {}
    t_total = time.time()

    for model_key in args.models:
        display_name = MODEL_REGISTRY[model_key]["display_name"]
        print(f"\n{'='*70}")
        print(f"  Model: {display_name}")
        print(f"{'='*70}")

        task_ckpt = args.task_ckpt or args.wproj_ckpt
        pipeline = RealHMDPPipeline(model_key, device=args.device,
                                    task_ckpt=task_ckpt)
        pipeline.load_models()
        if args.raw_pixel:
            pipeline.load_hmdp_checkpoint(task_ckpt, strict_task_head=False)

        # ── Base evaluation (single VLM call) ──
        print(f"\n  >>> Base Evaluation (no LDP, no GoT, no LTM)...")
        t0 = time.time()
        base_metrics = pipeline.evaluate_base(data, indices, args.image_base)
        t_base = time.time() - t0
        base_metrics["time_sec"] = t_base

        print(f"\n  Base Results ({t_base:.0f}s):")
        print(f"    Action Acc: {base_metrics['action_accuracy']*100:.1f}%")
        print(f"    Point Acc:  {base_metrics['point_accuracy']*100:.1f}%")
        print(f"    Avg Dist:   {base_metrics['avg_distance']:.4f}")

        # ── H-MDP evaluation (full pipeline) ──
        # Default = Blind-but-Smart latent injection (paper-faithful). The
        # raw-pixel path is selected only via the explicit --raw-pixel ablation.
        mode = "raw-pixel (ablation)" if args.raw_pixel else "Blind-but-Smart (W_proj latent injection)"
        print(f"\n  >>> H-MDP Evaluation [{mode}] (ε={args.epsilon}, k={args.k})...")
        t0 = time.time()
        if args.raw_pixel:
            hmdp_metrics = pipeline.evaluate_hmdp(
                data, indices, args.image_base,
                epsilon=args.epsilon, k=args.k, temperature=args.temperature,
                governor=governor,
            )
        else:
            # The blind pipeline loads its own VLM; free the base copy first so
            # two 7B models don't exceed GPU memory (which forces CPU offload).
            pipeline.unload_models()
            hmdp_metrics = evaluate_hmdp_blind(
                model_key, data, indices, args.image_base,
                device=args.device, epsilon=args.epsilon, k=args.k,
                temperature=args.temperature, wproj_ckpt=args.wproj_ckpt,
                governor=governor, grid=args.grid,
                allow_random_wproj=args.allow_random_wproj,
            )
        t_hmdp = time.time() - t0
        hmdp_metrics["time_sec"] = t_hmdp

        print(f"\n  H-MDP Results ({t_hmdp:.0f}s):")
        print(f"    Action Acc: {hmdp_metrics['action_accuracy']*100:.1f}%")
        print(f"    Point Acc:  {hmdp_metrics['point_accuracy']*100:.1f}%")
        print(f"    Avg Dist:   {hmdp_metrics['avg_distance']:.4f}")
        print(f"    LTM Episodes: {hmdp_metrics['ltm_episodes']}")

        gain = hmdp_metrics["point_accuracy"] - base_metrics["point_accuracy"]
        print(f"\n  Gain (Point Acc): {gain*100:+.1f}%")

        all_results[model_key] = {
            "display_name": display_name,
            "base": base_metrics,
            "hmdp": hmdp_metrics,
            "gain": gain,
        }

        pipeline.unload_models()

    # ── Summary Table ──
    elapsed = time.time() - t_total
    print(f"\n\n{'='*96}")
    print(f"  Table 3 (Real): Cross-backbone H-MDP at ε={args.epsilon}, k={args.k}, τ={args.temperature}")
    print(f"{'='*96}")
    print(f"  {'VLM Backbone':<22} {'Base Act%':>9} {'Base Pt%':>9} {'Base Dist':>9}"
          f"  {'H-MDP Act%':>10} {'H-MDP Pt%':>10} {'H-MDP Dist':>10} {'ΔPt':>6} {'ΔDist':>7}")
    print(f"  {'-'*88}")
    for key, r in all_results.items():
        b, h = r["base"], r["hmdp"]
        d_pt = r["gain"]
        d_dist = h['avg_distance'] - b['avg_distance']
        print(f"  {r['display_name']:<22} "
              f"{b['action_accuracy']*100:>8.1f}% {b['point_accuracy']*100:>8.1f}% {b['avg_distance']:>9.4f}"
              f"  {h['action_accuracy']*100:>9.1f}% {h['point_accuracy']*100:>9.1f}% {h['avg_distance']:>10.4f}"
              f" {d_pt*100:>+5.1f}% {d_dist:>+7.4f}")
    print(f"{'='*96}")
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # ── Save results ──
    save_path = args.out
    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"  Results saved to {save_path}")


if __name__ == "__main__":
    main()
