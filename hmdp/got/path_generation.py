#!/usr/bin/env python3
"""
GoT Path Generation: k diverse VLM reasoning paths.

Implements Stage-2 of the GoT pipeline (Sec. 3.5):
  - Generate k temperature-sampled coordinate candidates
  - Extract per-token logprobs for confidence scoring
  - Support multiple VLM architectures (Qwen, InternVL)

This module is decoupled from aggregation so either can evolve independently.
"""

import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from hmdp.constants import GoTHyperparams


# Coordinate-string regex patterns, kept in sync with
# VLMInferenceEngine._parse_point. Used to locate the char span of the
# predicted coordinate so we can average the logprob of only the tokens
# that produced it (Logit_VLM(c_j) in Eq. got_score).
_COORD_PATTERNS = [
    r'coordinate\s*=\s*\[(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\]',
    r'\"coordinate\"\s*:\s*\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]',
    r'pyautogui\.\w+\(x\s*=\s*(\d+\.?\d*)\s*,\s*y\s*=\s*(\d+\.?\d*)',
    r'click\(x\s*=\s*(\d+\.?\d*)\s*,\s*y\s*=\s*(\d+\.?\d*)',
    r'\[(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\]',
    r'\((\d+\.?\d*)\s*,\s*(\d+\.?\d*)\)',
]


def _find_coord_span(text: str) -> Optional[Tuple[int, int]]:
    """Char span [start, end) of the first parsed coordinate's numeric part."""
    for pat in _COORD_PATTERNS:
        m = re.search(pat, text)
        if m:
            return (m.start(1), m.end(2))
    return None


def _coord_token_logprob(
    token_texts: List[str],
    token_logprobs: List[float],
    full_text: str,
) -> Optional[float]:
    """
    Logit_VLM(c_j): mean logprob of the tokens that produced the predicted
    coordinate string.

    token_texts[i]    = incremental decoded text contributed by token i
                        (sum(token_texts) reconstructs full_text)
    token_logprobs[i] = log P(token_i | prefix) under the sampling distribution

    Falls back to the whole-sequence mean logprob when the coordinate span
    cannot be located or no token overlaps it. Returns None only when no
    tokens were generated.
    """
    if not token_logprobs:
        return None
    span = _find_coord_span(full_text)
    if span is None:
        return sum(token_logprobs) / len(token_logprobs)
    cs, ce = span
    overlap = []
    pos = 0
    for txt, lp in zip(token_texts, token_logprobs):
        t_start = pos
        t_end = pos + len(txt)
        pos = t_end
        if t_start < ce and cs < t_end:   # [t_start,t_end) overlaps [cs,ce)
            overlap.append(lp)
    if not overlap:
        return sum(token_logprobs) / len(token_logprobs)
    return sum(overlap) / len(overlap)


def _incremental_token_texts(tokenizer, gen_ids: List[int]) -> List[str]:
    """
    Decode a token-id sequence incrementally, returning the text piece each
    token contributes. sum(pieces) == tokenizer.decode(gen_ids, skip_special).

    Uses prefix-decode differencing so multi-byte / merged tokens are handled
    consistently with the final decoded string.
    """
    pieces = []
    prev = ""
    for i in range(len(gen_ids)):
        cur = tokenizer.decode(gen_ids[: i + 1], skip_special_tokens=True)
        pieces.append(cur[len(prev):])
        prev = cur
    return pieces


def _generate_step_logits(gen_out):
    """
    Return a list of per-step logit tensors (vocab,) from a generate() output.

    Prefers `.logits` (raw, transformers>=4.38); falls back to `.scores`
    (post-warping). Each entry corresponds to one generated token.
    """
    seq = getattr(gen_out, "logits", None)
    if seq is None:
        seq = getattr(gen_out, "scores", None)
    if seq is None:
        return []
    return [s[0] for s in seq]  # batch index 0


def _selected_token_logprobs(step_logits, gen_id_list):
    """
    logprob of each emitted token under its step's softmax.

    step_logits[t] is aligned with gen_id_list[t]. If lengths differ (e.g. an
    appended EOS without a recorded score) the overlap is used.
    """
    out = []
    n = min(len(step_logits), len(gen_id_list))
    for t in range(n):
        lp = F.log_softmax(step_logits[t].float(), dim=-1)
        out.append(lp[gen_id_list[t]].item())
    return out


def vlm_got_k_paths(
    vlm,
    image_path: str,
    instruction: str,
    k: int,
    temperature: float = GoTHyperparams.T_SAMPLE,  # 0.7
) -> List[Dict]:
    """
    Generate k diverse reasoning paths by calling the VLM k times
    with temperature sampling.

    Returns list of k dicts: {action_type, pred_point, raw_output, logit}
    """
    from hmdp.vlm_inference import VLMInferenceEngine
    arch = vlm.info.get("arch", "qwen2_5_vl")
    if arch == "internvl2_5":
        return _got_k_paths_internvl(vlm, image_path, instruction, k, temperature)
    else:
        return _got_k_paths_qwen(vlm, image_path, instruction, k, temperature)


def _got_k_paths_qwen(vlm, image_path, instruction, k, temperature):
    """GoT k-path for Qwen2-VL / Qwen2.5-VL family."""
    from qwen_vl_utils import process_vision_info

    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": vlm._build_prompt(instruction)},
            ],
        }
    ]
    text = vlm.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = vlm.processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(vlm.device)

    pad_id = getattr(vlm.processor.tokenizer, 'pad_token_id', None)
    if pad_id is None:
        pad_id = getattr(vlm.model.config, 'eos_token_id', 151658)

    tok = vlm.processor.tokenizer
    paths = []
    for i in range(k):
        with torch.no_grad():
            gen_out = vlm.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=(i > 0),  # first path is greedy, rest are sampled
                temperature=temperature if i > 0 else 1.0,
                top_p=GoTHyperparams.TOP_P if i > 0 else GoTHyperparams.TOP_P_GREEDY,
                pad_token_id=pad_id,
                return_dict_in_generate=True,
                output_logits=True,
            )
        output_ids = gen_out.sequences
        gen_ids = output_ids[:, inputs.input_ids.shape[1]:]
        gen_id_list = gen_ids[0].tolist()
        raw_output = vlm.processor.batch_decode(
            gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Per-token logprob of the actually-emitted tokens from raw logits.
        step_logits = _generate_step_logits(gen_out)
        token_logprobs = _selected_token_logprobs(step_logits, gen_id_list)
        token_texts = _incremental_token_texts(tok, gen_id_list)
        logit = _coord_token_logprob(token_texts, token_logprobs, raw_output)

        action_type = vlm._parse_action_type(raw_output)
        pred_point = vlm._parse_point(raw_output, img_w, img_h)

        paths.append({
            "action_type": action_type,
            "pred_point": pred_point,
            "raw_output": raw_output,
            "logit": logit,
        })

    return paths


def _got_k_paths_internvl(vlm, image_path, instruction, k, temperature):
    """GoT k-path for InternVL2.5 family via manual decoding (no .generate())."""
    import importlib
    from hmdp.vlm_adapters import _load_image_internvl

    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size

    pixel_values = _load_image_internvl(image_path, max_num=12).to(
        dtype=vlm.dtype, device=vlm.device
    )
    num_patches = pixel_values.shape[0]

    # ── Build prompt (replicate InternVLChatModel.chat logic) ──
    conv_mod = importlib.import_module(
        type(vlm.model).__module__.replace('modeling_internvl_chat', 'conversation'))
    get_conv_template = conv_mod.get_conv_template

    question = '<image>\n' + vlm._build_prompt(instruction)
    IMG_START, IMG_END, IMG_CTX = '<img>', '</img>', '<IMG_CONTEXT>'

    template = get_conv_template(vlm.model.template)
    template.system_message = vlm.model.system_message
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()
    eos_token_id = vlm.tokenizer.convert_tokens_to_ids(template.sep.strip())

    image_tokens = IMG_START + IMG_CTX * vlm.model.num_image_token * num_patches + IMG_END
    query = query.replace('<image>', image_tokens, 1)

    model_inputs = vlm.tokenizer(query, return_tensors='pt')
    input_ids = model_inputs['input_ids'].to(vlm.device)
    attention_mask = model_inputs['attention_mask'].to(vlm.device)

    # ── Build input embeddings with vision features ──
    img_ctx_id = vlm.tokenizer.convert_tokens_to_ids(IMG_CTX)
    vlm.model.img_context_token_id = img_ctx_id

    with torch.no_grad():
        vit_embeds = vlm.model.extract_feature(pixel_values)
        base_embeds = vlm.model.language_model.get_input_embeddings()(input_ids)
        B, N, C = base_embeds.shape
        flat_embeds = base_embeds.reshape(B * N, C)
        flat_ids = input_ids.reshape(B * N)
        selected = (flat_ids == img_ctx_id)
        flat_embeds[selected] = vit_embeds.reshape(-1, C).to(flat_embeds.device)
        input_embeds = flat_embeds.reshape(B, N, C)

    # ── Generate k paths via manual decoding ──
    max_new_tokens = 256
    paths = []

    for path_i in range(k):
        do_sample = (path_i > 0)
        cur_temp = temperature if do_sample else 1.0
        top_p = GoTHyperparams.TOP_P if do_sample else GoTHyperparams.TOP_P_GREEDY

        generated_ids = []
        token_logprobs = []
        past_key_values = None
        cur_embeds = input_embeds
        cur_mask = attention_mask.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                if past_key_values is None:
                    out = vlm.model.language_model(
                        inputs_embeds=cur_embeds,
                        attention_mask=cur_mask,
                        use_cache=True,
                    )
                else:
                    next_emb = vlm.model.language_model.get_input_embeddings()(
                        next_id.unsqueeze(0)
                    )
                    out = vlm.model.language_model(
                        inputs_embeds=next_emb,
                        attention_mask=cur_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

                raw_logits = out.logits[:, -1, :]
                # logprob under the RAW model distribution (pre-warping),
                # so Logit_VLM reflects model confidence, not sampling warps.
                raw_logprob = F.log_softmax(raw_logits.float(), dim=-1)

                if do_sample:
                    # top-p (nucleus) filtering
                    sorted_logits, sorted_idx = torch.sort(raw_logits, descending=True, dim=-1)
                    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_idx_to_remove = cum_probs > top_p
                    sorted_idx_to_remove[..., 0] = False
                    indices_to_remove = sorted_idx_to_remove.scatter(
                        1, sorted_idx, sorted_idx_to_remove
                    )
                    warped_logits = raw_logits.clone()
                    warped_logits[indices_to_remove] = float('-inf')
                    warped_logits = warped_logits / cur_temp
                    probs = F.softmax(warped_logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1).item()
                else:
                    next_id = int(torch.argmax(raw_logits, dim=-1).item())

                # Record the logprob under the raw distribution for this chosen token.
                token_logprobs.append(raw_logprob[0, next_id].item())
                generated_ids.append(next_id)

                if next_id == eos_token_id:
                    break

                past_key_values = out.past_key_values
                cur_mask = torch.cat([cur_mask, torch.ones((B, 1), device=vlm.device)], dim=1)

        # Decode and parse
        raw_output = vlm.tokenizer.decode(generated_ids, skip_special_tokens=True)
        logit = sum(token_logprobs) / len(token_logprobs) if token_logprobs else None

        action_type = vlm._parse_action_type(raw_output)
        pred_point = vlm._parse_point(raw_output, img_w, img_h)

        paths.append({
            "action_type": action_type,
            "pred_point": pred_point,
            "raw_output": raw_output,
            "logit": logit,
        })

    return paths
