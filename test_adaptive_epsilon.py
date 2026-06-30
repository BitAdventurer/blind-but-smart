#!/usr/bin/env python3
"""Smoke-test for AdaptiveEpsilonGovernor (no trained checkpoint required).

Verifies the integration contract of ``hmdp.sac_governor.AdaptiveEpsilonGovernor``
without needing a trained SAC checkpoint or any VLM:

  T1  No checkpoint            -> disabled, uniform fallback
  T2  Corrupted checkpoint     -> graceful fallback (no crash)
  T3  Valid (untrained) ckpt   -> adaptive, in-range, non-uniform epsilons
  T4  build_state shape (26,)  + U_t carry semantics
  T5  Determinism              -> identical epsilons for identical inputs
  T6  reset_per_sample regimes -> True: update no-op; False: carries
  T7  device fallback          -> cuda:0/mps -> cpu when unavailable

Requirements
------------
* The H-MDP simulation package must be importable as ``hmdp_sim`` (or set
  ``$HMDP_SIM_PKG`` to its package name). It must expose
  ``config.HMDPConfig``, ``sac_policy.SACMetaPolicy`` and
  ``execution_engine.ExecutionEngine.compute_latent_entropy``.
* ``sac_governor.py`` must be importable (run this from the repo root, or add
  the ``hmdp/`` directory to ``PYTHONPATH``).

Usage
-----
    export HMDP_SIM_PKG=hmdp_sim          # if not the default
    python3 test_adaptive_epsilon.py
"""

import os
import sys
import tempfile

import numpy as np
import torch

# Allow running from the repo root: make both the package root and the hmdp/
# directory importable so `import hmdp_sim` and `from sac_governor import ...`
# both resolve regardless of invocation directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "hmdp")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("HMDP_SIM_PKG", "hmdp_sim")

try:
    from sac_governor import AdaptiveEpsilonGovernor
except ImportError:
    from hmdp.sac_governor import AdaptiveEpsilonGovernor


def main() -> int:
    torch.manual_seed(0)
    np.random.seed(0)
    M, d = 25, 256
    phi = torch.randn(M, d)  # clean proxy latents (M, d)

    print("=" * 60)
    print("[T1] No checkpoint -> disabled, uniform fallback")
    g = AdaptiveEpsilonGovernor(ckpt_path=None, device="cpu",
                                fallback_epsilon=5.0, fallback_k=5)
    eps, k = g.compute_epsilons(phi)
    uniform = len({round(e, 6) for e in eps}) == 1
    print(f"  enabled={g.enabled} len(eps)={len(eps)} uniform={uniform} "
          f"eps[0]={eps[0]} k={k}")
    assert g.enabled is False and len(eps) == 25
    assert abs(eps[0] - 5.0) < 1e-9 and k == 5

    print("=" * 60)
    print("[T2] Corrupted checkpoint -> graceful fallback")
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tf:
        tf.write(b"not a real torch checkpoint")  # unreadable by torch.load
        bad_ckpt = tf.name
    try:
        g2 = AdaptiveEpsilonGovernor(ckpt_path=bad_ckpt, device="cpu",
                                     fallback_epsilon=3.0, fallback_k=10)
        eps2, k2 = g2.compute_epsilons(phi)
        print(f"  enabled={g2.enabled} eps[0]={eps2[0]} k={k2}")
        assert g2.enabled is False and abs(eps2[0] - 3.0) < 1e-9 and k2 == 10
    finally:
        os.unlink(bad_ckpt)

    print("=" * 60)
    print("[T3] Valid (untrained) checkpoint -> adaptive, in-range, non-uniform")
    from hmdp_sim.config import HMDPConfig
    from hmdp_sim.sac_policy import SACMetaPolicy
    cfg = HMDPConfig()
    mp = SACMetaPolicy(state_dim=cfg.sac_state_dim, action_dim=cfg.sac_action_dim,
                       hidden_dim=cfg.sac_hidden_dim, device="cpu")
    with tempfile.TemporaryDirectory() as td:
        ckpt = os.path.join(td, "valid_sac_test.pt")
        mp.save(ckpt)

        g3 = AdaptiveEpsilonGovernor(ckpt_path=ckpt, device="cpu")
        eps3, k3 = g3.compute_epsilons(phi)
        arr = np.array(eps3)
        print(f"  enabled={g3.enabled} len={len(eps3)} "
              f"min/max/std={arr.min():.4f}/{arr.max():.4f}/{arr.std():.4f} k={k3}")
        assert g3.enabled is True and len(eps3) == 25
        assert (arr >= 0.1).all() and (arr <= 5.0).all()
        assert 1 <= k3 <= 20
        assert arr.std() > 1e-6, "adaptive epsilons should be non-uniform"

        print("=" * 60)
        print("[T4] build_state shape (26,) + U_t carry semantics")
        g4 = AdaptiveEpsilonGovernor(ckpt_path=ckpt, device="cpu",
                                     reset_per_sample=False)
        st = g4.build_state(phi)
        print(f"  s_t shape={tuple(st.shape)} U_t(init)={float(st[0])}")
        assert tuple(st.shape) == (26,) and abs(float(st[0]) - 0.5) < 1e-9
        g4.update_uncertainty(0.83)
        st2 = g4.build_state(phi)
        print(f"  after update_uncertainty(0.83): U_t={float(st2[0])}")
        assert abs(float(st2[0]) - 0.83) < 1e-5  # float32 storage tolerance
        g4.reset_episode()
        st3 = g4.build_state(phi)
        print(f"  after reset_episode: U_t={float(st3[0])}")
        assert abs(float(st3[0]) - 0.5) < 1e-9

        print("=" * 60)
        print("[T5] Determinism: identical phi+U_t -> identical epsilons")
        g4.reset_episode()
        e_a, _ = g4.compute_epsilons(phi)
        g4.reset_episode()
        e_b, _ = g4.compute_epsilons(phi)
        print(f"  identical={np.allclose(e_a, e_b)}")
        assert np.allclose(e_a, e_b)

        print("=" * 60)
        print("[T6] reset_per_sample regimes")
        g_rp = AdaptiveEpsilonGovernor(ckpt_path=ckpt, device="cpu",
                                       reset_per_sample=True)
        g_rp.reset_episode()
        g_rp.update_uncertainty(0.9)
        print(f"  reset_per_sample=True  after update(0.9): U_t={g_rp.prev_u}")
        assert abs(g_rp.prev_u - 0.5) < 1e-9, "update must be no-op when reset_per_sample"
        g_cr = AdaptiveEpsilonGovernor(ckpt_path=ckpt, device="cpu",
                                       reset_per_sample=False)
        g_cr.reset_episode()
        g_cr.update_uncertainty(0.9)
        print(f"  reset_per_sample=False after update(0.9): U_t={g_cr.prev_u}")
        assert abs(g_cr.prev_u - 0.9) < 1e-5, "update must carry when not reset_per_sample"

    print("=" * 60)
    print("[T7] device fallback when cuda/mps unavailable")
    g_dev = AdaptiveEpsilonGovernor(ckpt_path=None, device="cuda:0", verbose=False)
    print(f"  requested cuda:0 -> resolved {g_dev.device}")
    if not torch.cuda.is_available():
        assert g_dev.device.type == "cpu"

    print("\nALL GOVERNOR TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
