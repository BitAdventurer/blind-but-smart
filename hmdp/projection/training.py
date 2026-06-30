#!/usr/bin/env python3
"""
Offline Projection Training (Sec. 4.1.4).

Two-stage training protocol for W_proj, proxy_encoder, and task_head:
  1. Stage "align": Feature alignment (MSE) against native VLM visual tokens
  2. Stage "task": Task-supervised teacher forcing (cross-entropy)

Both stages train under (ε,δ)-LDP Gaussian noise augmentation while DINOv2 
and VLM backbone remain frozen.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from hmdp.constants import ModelDims


def train_projection_offline(
    pipe,
    samples: List[Dict],
    image_base: str,
    stage: str = "align",
    epochs: int = 1,
    lr: float = 1e-4,
    eps_range: Tuple[float, float] = (3.0, 5.0),
    batch_log: int = 20,
    device: str = "cuda",
) -> None:
    """
    Train W_proj (+ proxy_encoder, task_head) offline under LDP-noise
    augmentation. Both DINOv2 and the VLM backbone stay frozen.

    Two stages (Sec. 4.1.4):
      stage="align": feature alignment. Match the M projected privatized
                     latent tokens to the VLM's *native* visual-token
                     embedding of the same screenshot (MSE). eps in [3, 5].
      stage="task":  task-supervised teacher forcing. Cross-entropy on the
                     ground-truth answer string conditioned on the injected
                     blind prefix. eps in [0.1, 5].

    `samples` follows the GUI 360 schema used elsewhere (conversation/images/bbox).
    """
    import os
    from hmdp.run_real_vlm import _parse_sample  # reuse the shared parser

    assert pipe.proj_layer is not None, "Call pipe.load_models() first."
    params = (
        list(pipe.proj_layer.parameters())
        + list(pipe.proxy_encoder.parameters())
        + list(pipe.task_head.parameters())
    )
    optim = torch.optim.AdamW(params, lr=lr)
    tok = pipe._tokenizer()

    # Stage-2 backprops through the (frozen) 7B backbone, which is memory
    # heavy. Activation checkpointing + disabled KV cache keeps it on a
    # single GPU while only W_proj / proxy / task-head receive gradients.
    if stage == "task":
        try:
            pipe.vlm.model.config.use_cache = False
            pipe.vlm.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            # Checkpointing is only active when the module is in train() mode.
            # Backbone weights stay frozen (requires_grad=False); only W_proj /
            # proxy / task-head are optimised. Qwen2.5 LM dropout is 0.0, so
            # train mode does not introduce stochasticity.
            for p in pipe.vlm.model.parameters():
                p.requires_grad_(False)
            pipe.vlm.model.train()
        except Exception as e:
            print(f"  [offline:task] gradient checkpointing unavailable: {e}")

    for ep in range(epochs):
        running = 0.0
        n = 0
        for i, sample in enumerate(samples):
            instruction, gt_action, gt_bbox_raw, img_path = _parse_sample(sample, image_base)
            if img_path is None:
                continue
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            eps = float(np.random.uniform(*eps_range))
            epsilons = [eps] * pipe.num_regions

            # Client-side privatization with gradient through proxy + W_proj.
            # Uses Gaussian mechanism (ε,δ)-LDP by default for better composition
            with torch.no_grad():
                e_regions = pipe.dino.extract_regions(image)            # (M, 1024)
            phi = pipe.proxy_encoder(e_regions)                         # (M, d)
            phi_private = pipe.ldp.privatize_features(phi, eps, mechanism='gaussian', delta=1e-05)
            visual_tokens = pipe.proj_layer(phi_private)                # (M, d_llm)

            if stage == "align":
                target = _vlm_visual_token_summary(pipe, image, pipe.num_regions)
                if target is None:
                    continue
                loss = F.mse_loss(visual_tokens, target.detach())
            elif stage == "task":
                loss = _task_supervised_loss(pipe, visual_tokens, instruction, sample, tok)
                if loss is None:
                    continue
            else:
                raise ValueError(f"Unknown stage: {stage}")

            optim.zero_grad()
            loss.backward()
            optim.step()

            running += float(loss.item())
            n += 1
            if n % batch_log == 0:
                print(f"  [offline:{stage}] ep{ep} step{n}  loss={running/n:.4f}")

        if n:
            print(f"  [offline:{stage}] epoch {ep} done  mean_loss={running/n:.4f}")


def train_ltm_predictor_offline(
    pipe,
    samples: List[Dict],
    image_base: str,
    epochs: int = 2,
    lr: float = 1e-3,
    eps_range: Tuple[float, float] = (0.1, 5.0),
    c_star_sigma: float = 0.2,
    batch_log: int = 50,
    device: str = "cuda",
    seed: int = 42,
) -> "object":
    """
    Train the Eq.8 strategic-prediction head θ_pred (``pipe.ltm_predictor``).

    Faithful to Eq.7/8: an episodic memory is populated with the privatized
    region-mean latent φ̃ of each training screenshot keyed to its ground-truth
    target point. For each sample we then retrieve K_ret = argmax cos(φ̃, φ_j)
    (Eq.7, excluding the sample's own episode), and train θ_pred to recover the
    GT coordinate from (φ̃, C*, K_ret) (Eq.8).

    Because the GoT coordinate C* is produced by k expensive VLM generations,
    running it inside the training loop is impractical. We therefore *simulate*
    it as C*_sim = GT + N(0, c_star_sigma²) clipped to [0,1], modelling the
    VLM's grounding-error distribution (≈0.2 normalized, calibrated from the
    base point-distance). θ_pred thus learns to fuse a noisy estimate, the
    privatized observation, and memory into a corrected coordinate.

    DINOv2/proxy run frozen (no grad); only θ_pred is optimised. Returns the
    populated LTM (useful for inspection/tests).
    """
    import os
    import numpy as np
    from hmdp.ltm import LongTermMemory
    from hmdp.run_real_vlm import _parse_sample, _normalize_bbox

    assert pipe.ltm_predictor is not None, "Call pipe.load_models() first."

    rng = np.random.RandomState(seed)
    ltm = LongTermMemory(
        embedding_dim=pipe.latent_dim, capacity=len(samples) + 16, top_k=8,
    ).to(device)

    # ── Phase A: cache clean latents + GT, populate the episodic memory ──
    records = []  # (episode_id, phi_clean (M,d) cpu, gt_coord [x,y])
    for sample in samples:
        instruction, gt_action, gt_bbox_raw, img_path = _parse_sample(sample, image_base)
        if img_path is None or gt_bbox_raw is None:
            continue
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        img_w, img_h = image.size
        gt = _normalize_bbox(gt_bbox_raw, img_w, img_h)
        gt_coord = [(gt[0] + gt[2]) / 2.0, (gt[1] + gt[3]) / 2.0]
        with torch.no_grad():
            e_regions = pipe.dino.extract_regions(image)            # (M, 1024)
            phi = pipe.proxy_encoder(e_regions)                     # (M, d)
            eps = float(rng.uniform(*eps_range))
            phi_priv = pipe.ldp.privatize_features(
                phi, eps, mechanism='gaussian', delta=1e-05)
            summary = phi_priv.mean(dim=0).detach().cpu()           # (d,)
        ep_id = ltm._episode_counter
        ltm.store_episode(
            state_embedding=summary,
            action_taken=0,
            bbox_target=torch.tensor(gt_coord + gt_coord),
            epsilon_used=eps, k_used=1, reward=1.0,
            sensitivity=0.5, uncertainty=0.0, success=True)
        records.append((ep_id, phi.detach().cpu(), gt_coord))

    if not records:
        print("  [offline:ltm_predict] no usable samples; skipping θ_pred training.")
        return ltm

    # ── Phase B: train θ_pred on (φ̃, C*_sim, K_ret) -> GT ──
    pred = pipe.ltm_predictor
    pred.train()
    optim = torch.optim.AdamW(pred.parameters(), lr=lr)

    for ep in range(epochs):
        order = rng.permutation(len(records))
        running, n = 0.0, 0
        for j in order:
            ep_id, phi_cpu, gt_coord = records[j]
            phi = phi_cpu.to(device)
            # Fresh privatization draw for the query (independent of the noise
            # that was memorised), matching the inference-time regime.
            eps = float(rng.uniform(*eps_range))
            with torch.no_grad():
                phi_priv = pipe.ldp.privatize_features(
                    phi, eps, mechanism='gaussian', delta=1e-05)
                phi_summary = phi_priv.mean(dim=0).detach()         # (d,)
            ret_coord, ret_emb, conf = ltm.retrieve_prior_features(
                phi_summary, exclude_id=ep_id, device=device)

            gt = torch.tensor(gt_coord, device=device, dtype=torch.float32)
            c_star_sim = (gt + torch.randn(2, device=device) * c_star_sigma).clamp(0.0, 1.0)

            refined = pred(
                phi_summary.float(), c_star_sim, ret_coord.float(),
                ret_emb.float(), conf.float())
            loss = F.mse_loss(refined, gt)

            optim.zero_grad()
            loss.backward()
            optim.step()
            running += float(loss.item())
            n += 1
            if n % batch_log == 0:
                print(f"  [offline:ltm_predict] ep{ep} step{n}  mse={running/n:.5f}")
        if n:
            print(f"  [offline:ltm_predict] epoch {ep} done  mean_mse={running/n:.5f}")

    pred.eval()
    pipe._ltm_predictor_trained = True
    return ltm


@torch.no_grad()
def _vlm_visual_token_summary(
    pipe, image: Image.Image, num_regions: int
) -> Optional[torch.Tensor]:
    """
    Stage-1 alignment target: the VLM's own visual-token embeddings for the
    screenshot, average-pooled into M=num_regions region summaries (5x5 grid
    over the patch sequence). Returns (M, d_llm) or None if unsupported.
    """
    try:
        from qwen_vl_utils import process_vision_info
    except Exception:
        return None
    if pipe.vlm.processor is None:
        return None

    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": image},
                    {"type": "text", "text": "describe"}],
    }]
    text = pipe.vlm.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = pipe.vlm.processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt").to(pipe.device)

    model = pipe.vlm.model
    if not hasattr(model, "visual"):
        return None
    pixel_values = inputs["pixel_values"].type(model.visual.dtype)
    grid_thw = inputs["image_grid_thw"]                                 # (1, 3) = [t, h, w]
    image_embeds = model.visual(pixel_values, grid_thw=grid_thw)        # (P, d_llm)

    # Recover the 2D token grid so the alignment target preserves spatial
    # layout (matching DINOv2's row-major 5x5 region partition). Qwen2.5-VL
    # merges `spatial_merge_size`^2 patches per visual token.
    t, h, w = [int(x) for x in grid_thw[0].tolist()]
    merge = int(getattr(model.config.vision_config, "spatial_merge_size", 2))
    H, W = h // merge, w // merge
    d = image_embeds.shape[-1]
    grid = int(round(num_regions ** 0.5))                              # 5
    if image_embeds.shape[0] != t * H * W:
        # Fallback: cannot reshape reliably -> sequential pooling.
        P = image_embeds.shape[0]
        idx = torch.linspace(0, P, steps=num_regions + 1).long()
        summaries = [image_embeds[idx[r]: max(idx[r + 1], idx[r] + 1)].mean(0)
                     for r in range(num_regions)]
        return torch.stack(summaries, dim=0).float()
    # (P, d) -> (1, d, H, W) -> adaptive pool to (grid, grid) -> (grid*grid, d)
    fmap = image_embeds.reshape(t, H, W, d).mean(0)                    # (H, W, d)
    fmap = fmap.permute(2, 0, 1).unsqueeze(0).float()                 # (1, d, H, W)
    pooled = F.adaptive_avg_pool2d(fmap, output_size=(grid, grid))     # (1, d, g, g)
    summaries = pooled.squeeze(0).permute(1, 2, 0).reshape(grid * grid, d)  # row-major (M, d)
    return summaries


def _task_supervised_loss(
    pipe,
    visual_tokens: torch.Tensor,
    instruction: str,
    sample: Dict,
    tok,
) -> Optional[torch.Tensor]:
    """
    Stage-2: teacher-forced cross-entropy on the ground-truth answer string,
    conditioned on the blind visual prefix + instruction prompt.
    """
    answer = None
    for msg in sample.get("conversation", []):
        if msg.get("from") == "gpt":
            answer = msg["value"]
            break
    if not answer:
        return None

    emb = pipe._token_embedding()
    from hmdp.vlm_inference import VLMInferenceEngine
    prompt = pipe.vlm._build_prompt(instruction)
    prompt_ids = tok(prompt, return_tensors="pt").input_ids.to(pipe.device)
    answer_ids = tok(answer, return_tensors="pt").input_ids.to(pipe.device)
    # Bound activation memory: the coordinate/action answer is short, so cap
    # the supervised answer span.
    max_ans, max_prompt = 64, 256
    answer_ids = answer_ids[:, :max_ans]
    prompt_ids = prompt_ids[:, :max_prompt]

    prompt_emb = emb(prompt_ids)
    answer_emb = emb(answer_ids)
    prefix = visual_tokens.unsqueeze(0).to(prompt_emb.dtype)            # (1, M, d_llm)
    inputs_embeds = torch.cat([prefix, prompt_emb, answer_emb], dim=1)
    attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=pipe.device)

    model = pipe.vlm.model
    out = model(inputs_embeds=inputs_embeds, attention_mask=attn)
    logits = out.logits                                                 # (1, T, V)

    # Only the answer positions are supervised.
    n_prefix = prefix.shape[1] + prompt_emb.shape[1]
    ans_logits = logits[:, n_prefix - 1: n_prefix - 1 + answer_ids.shape[1], :]
    loss = F.cross_entropy(
        ans_logits.reshape(-1, ans_logits.shape[-1]),
        answer_ids.reshape(-1),
    )
    return loss
