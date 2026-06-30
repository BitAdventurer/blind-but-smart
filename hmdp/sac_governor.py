"""Adaptive per-region epsilon governor for the real-VLM H-MDP pipeline.

This module wires the H-MDP SAC meta-policy into ``run_real_vlm.py`` so that the
uniform privacy-budget broadcast ``[epsilon] * M`` can be replaced by an
adaptive, per-region budget vector chosen from the governance state

    s_t = [U_t, lambda_t^(1), ..., lambda_t^(M)]      (1 + M = 26 dims, M = 25)

where

* ``U_t``       is a scalar reasoning-uncertainty feature. The per-step
                uncertainty ``u_t`` is only known *after* GoT aggregation, while
                the privacy budget must be chosen *before* privatisation. The
                governor therefore uses a carried value: ``U_0`` on the first
                state and (optionally) the previous step's ``u_t`` afterwards.
                See ``reset_per_sample`` for the two carry regimes.
* ``lambda_t``  is the per-region latent Shannon entropy of the **clean** proxy
                latents ``phi`` (computed *before* privatisation), via
                ``ExecutionEngine.compute_latent_entropy`` (Eq. 1).

The continuous SAC action ``a in [-1, 1]^26`` is mapped to
``(epsilons: List[float] of length M, k: int)`` by ``HMDPConfig.action_to_params``.

Graceful degradation
---------------------
If the simulation package cannot be imported, or no checkpoint is supplied, or
the checkpoint is missing / unreadable / structurally invalid (e.g. a truncated
``.pt`` whose zip central directory is absent), the governor disables itself and
``compute_epsilons`` returns a uniform vector built from ``fallback_epsilon`` and
``fallback_k`` — i.e. exactly the legacy behaviour — after printing an explicit,
non-silent warning. Only the actor sub-state is required for inference.

Package layout
--------------
The full H-MDP modules (``config``, ``sac_policy``, ``execution_engine``) live in
a package importable as ``hmdp_sim`` by default. Override the package name with
the ``HMDP_SIM_PKG`` environment variable if it is installed elsewhere.
"""

from __future__ import annotations

import os
import importlib
from typing import List, Optional, Tuple

import torch


# ──────────────────────────────────────────────────────────────────────────
#  Locate the H-MDP simulation package (config / sac_policy / execution_engine)
# ──────────────────────────────────────────────────────────────────────────

def _import_sim_symbols():
    """Import ``HMDPConfig``, ``SACMetaPolicy`` and ``ExecutionEngine``.

    Package-name resolution order:

      1. ``$HMDP_SIM_PKG``  (explicit override)
      2. ``"hmdp_sim"``     (default install location)

    The package's ``ExecutionEngine`` must expose ``compute_latent_entropy``;
    a build lacking it (e.g. a trimmed projection-only variant) is rejected so
    the wrong symbol is never picked up silently.
    """
    candidates: List[str] = []
    env = os.environ.get("HMDP_SIM_PKG")
    if env:
        candidates.append(env)
    if "hmdp_sim" not in candidates:
        candidates.append("hmdp_sim")

    last_err: Optional[Exception] = None
    for pkg in candidates:
        try:
            cfg_mod = importlib.import_module(f"{pkg}.config")
            sac_mod = importlib.import_module(f"{pkg}.sac_policy")
            eng_mod = importlib.import_module(f"{pkg}.execution_engine")
            HMDPConfig = getattr(cfg_mod, "HMDPConfig")
            SACMetaPolicy = getattr(sac_mod, "SACMetaPolicy")
            ExecutionEngine = getattr(eng_mod, "ExecutionEngine")
            if not hasattr(ExecutionEngine, "compute_latent_entropy"):
                raise ImportError(
                    f"'{pkg}.execution_engine.ExecutionEngine' lacks "
                    f"compute_latent_entropy"
                )
            return pkg, HMDPConfig, SACMetaPolicy, ExecutionEngine
        except Exception as e:  # pragma: no cover - import-time diagnostics
            last_err = e
            continue
    raise ImportError(
        "Could not import the H-MDP simulation package (tried: %s). Install the "
        "modules as 'hmdp_sim' or set $HMDP_SIM_PKG. Last error: %r"
        % (", ".join(candidates), last_err)
    )


def _resolve_device(device: str) -> torch.device:
    """Resolve a device string, falling back to CPU when unavailable.

    Honours ``cuda``/``cuda:N`` when CUDA is present and ``mps`` when the Apple
    Metal backend is available; otherwise falls back to CPU. The SAC actor is
    tiny, so CPU is a perfectly adequate default.
    """
    d = str(device).lower()
    if d.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device)
    if d == "mps" and getattr(torch.backends, "mps", None) is not None \
            and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ──────────────────────────────────────────────────────────────────────────
#  Adaptive epsilon governor
# ──────────────────────────────────────────────────────────────────────────

class AdaptiveEpsilonGovernor:
    """Produce adaptive per-region epsilon vectors from the governance state.

    Typical use inside an evaluation loop::

        gov = AdaptiveEpsilonGovernor(ckpt_path=..., device="cpu")
        gov.reset_episode()                              # start of a sample
        epsilons, k = gov.compute_epsilons(phi_clean)    # phi_clean: (M, d)
        psi = ldp.privatize_regions(
            phi_clean.unsqueeze(0), epsilons).squeeze(0)
        # ... run GoT, obtain u_t ...
        gov.update_uncertainty(u_t)                      # carried to next step

    When ``enabled`` is ``False`` (no/invalid checkpoint), ``compute_epsilons``
    returns a uniform ``fallback_epsilon`` vector and ``fallback_k`` — the legacy
    behaviour.

    Parameters
    ----------
    reset_per_sample :
        Controls the ``U_t`` carry regime. ``True`` (default) resets ``U_t`` to
        ``u_init`` at the start of every sample, so the chosen budget depends
        only on the current sample's ``lambda_t`` (and the fixed ``U_0``); this
        makes the per-sample evaluation order-independent and reproducible.
        ``False`` keeps the previous step's ``u_t`` across consecutive calls
        (matching the simulation's multi-step carry); under this regime the
        result is order-dependent, so a fixed evaluation order is required for
        reproducibility.
    """

    def __init__(
        self,
        ckpt_path: Optional[str],
        device: str = "cpu",
        num_regions: int = 25,
        u_init: float = 0.5,
        fallback_epsilon: float = 5.0,
        fallback_k: int = 5,
        deterministic: bool = True,
        reset_per_sample: bool = True,
        verbose: bool = True,
    ) -> None:
        self.device = _resolve_device(device)
        self.num_regions = int(num_regions)
        self.u_init = float(u_init)
        self.prev_u = float(u_init)
        self.fallback_epsilon = float(fallback_epsilon)
        self.fallback_k = int(fallback_k)
        self.deterministic = bool(deterministic)
        self.reset_per_sample = bool(reset_per_sample)
        self.verbose = bool(verbose)

        self.enabled = False
        self.cfg = None
        self.meta_policy = None
        self._compute_latent_entropy = None

        self._init_policy(ckpt_path)

    # ── logging helpers ─────────────────────────────────────────────────────

    def _warn(self, msg: str) -> None:
        if self.verbose:
            print(f"  [SAC-governor][WARN] {msg}")

    def _info(self, msg: str) -> None:
        if self.verbose:
            print(f"  [SAC-governor] {msg}")

    # ── construction helpers ────────────────────────────────────────────────

    def _init_policy(self, ckpt_path: Optional[str]) -> None:
        try:
            _pkg, HMDPConfig, SACMetaPolicy, ExecutionEngine = _import_sim_symbols()
        except Exception as e:
            self._warn(f"adaptive disabled — sim package import failed: {e}")
            return

        self.cfg = HMDPConfig()
        if self.cfg.num_regions != self.num_regions:
            self._warn(
                f"config num_regions={self.cfg.num_regions} != pipeline "
                f"num_regions={self.num_regions}; using config value"
            )
            self.num_regions = self.cfg.num_regions
        self._compute_latent_entropy = ExecutionEngine.compute_latent_entropy

        self.meta_policy = SACMetaPolicy(
            state_dim=self.cfg.sac_state_dim,
            action_dim=self.cfg.sac_action_dim,
            hidden_dim=self.cfg.sac_hidden_dim,
            device=str(self.device),
        )
        self.meta_policy.eval()

        if not ckpt_path:
            self._warn("no checkpoint supplied; adaptive disabled (uniform epsilon)")
            return
        if not os.path.exists(ckpt_path):
            self._warn(f"checkpoint not found: {ckpt_path}; adaptive disabled")
            return

        if self._load_actor(ckpt_path):
            self.enabled = True
            self._info(f"adaptive SAC governor active (ckpt={ckpt_path})")

    def _load_actor(self, ckpt_path: str) -> bool:
        """Load only the actor sub-state for inference.

        Tolerant of partial checkpoints (critics absent). Returns ``False`` on
        any unrecoverable error, in which case the governor falls back to
        uniform epsilon.
        """
        try:
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        except Exception as e:
            self._warn(
                f"torch.load failed for {ckpt_path}: {e!r}. The file may be "
                f"truncated or corrupted. Falling back to uniform epsilon."
            )
            return False

        actor_sd = None
        if isinstance(ckpt, dict):
            if "actor" in ckpt and isinstance(ckpt["actor"], dict):
                actor_sd = ckpt["actor"]
            elif "actor_state_dict" in ckpt and isinstance(ckpt["actor_state_dict"], dict):
                actor_sd = ckpt["actor_state_dict"]
            elif ckpt and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                actor_sd = ckpt  # flat actor state_dict saved directly
        if actor_sd is None:
            keys = list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt).__name__
            self._warn(
                f"no actor sub-state found in {ckpt_path} (keys={keys}); "
                f"falling back to uniform epsilon."
            )
            return False

        try:
            missing, unexpected = self.meta_policy.actor.load_state_dict(actor_sd, strict=False)
        except Exception as e:
            self._warn(f"actor.load_state_dict failed: {e!r}; uniform epsilon.")
            return False

        if missing:
            self._warn(f"actor missing keys (left at init): {list(missing)}")
        if unexpected:
            self._warn(f"actor unexpected keys (ignored): {list(unexpected)}")
        # Essential layers absent -> policy is effectively random; treat as a
        # failed load so we never advertise a meaningless 'adaptive' run.
        essential = {"backbone.0.weight", "mean_head.weight"}
        if essential & set(missing):
            self._warn("essential actor layers missing; uniform epsilon.")
            return False
        return True

    # ── per-step API ─────────────────────────────────────────────────────────

    def reset_episode(self) -> None:
        """Reset the carried uncertainty to ``u_init``.

        Call at the start of every sample. Under ``reset_per_sample=True`` this
        is the only thing that sets ``U_t``; under ``reset_per_sample=False`` it
        re-initialises the carry at the start of a new episode.
        """
        self.prev_u = self.u_init

    def update_uncertainty(self, u_t: float) -> None:
        """Carry the post-GoT uncertainty into the next state's ``U_t``.

        A no-op (beyond bookkeeping) when ``reset_per_sample=True``, since
        ``reset_episode`` overwrites ``U_t`` before the next ``compute_epsilons``.
        """
        if self.reset_per_sample:
            return
        try:
            self.prev_u = float(u_t)
        except (TypeError, ValueError):
            pass  # keep the previous value if u_t is malformed

    @torch.no_grad()
    def build_state(self, phi_clean: torch.Tensor) -> torch.Tensor:
        """Build ``s_t = [U_t ; lambda_t^(1..M)]`` of shape ``(1 + M,)``.

        ``phi_clean`` are the clean proxy latents ``(M, d)`` taken *before*
        privatisation. A batched ``(B, M, d)`` tensor is averaged over the batch.
        """
        lam = self._compute_latent_entropy(phi_clean.detach())
        if lam.dim() == 2:  # (B, M) -> (M,)
            lam = lam.mean(dim=0)
        lam = lam.reshape(-1).float().to(self.device)
        if lam.numel() != self.num_regions:
            # Defensive: pad/truncate to M so a shape mismatch never aborts eval.
            fill = float(lam.mean().item()) if lam.numel() else 0.0
            fixed = torch.full((self.num_regions,), fill, device=self.device)
            n = min(self.num_regions, lam.numel())
            fixed[:n] = lam[:n]
            lam = fixed
        u = torch.tensor([self.prev_u], dtype=torch.float32, device=self.device)
        return torch.cat([u, lam], dim=0)  # (1 + M,)

    @torch.no_grad()
    def compute_epsilons(self, phi_clean: torch.Tensor) -> Tuple[List[float], int]:
        """Return ``(epsilons: List[float] of length M, k: int)``.

        Adaptive when ``enabled``; otherwise a uniform fallback vector.
        ``action_to_params`` already guarantees the length and clips epsilons to
        ``[epsilon_min, epsilon_max]`` and ``k`` to ``[k_min, k_max]``.
        """
        if not self.enabled:
            return [self.fallback_epsilon] * self.num_regions, self.fallback_k
        state = self.build_state(phi_clean)
        action = self.meta_policy.select_action(state, deterministic=self.deterministic)
        epsilons, k = self.cfg.action_to_params(action)
        return epsilons, int(k)
