"""Backward-pass switches of the patched MuJoCo Warp (patches/mujoco_warp, applied by scripts/setup_mujoco_warp.sh).

All are read by MuJoCo Warp at import, so configure() must run before mujoco_warp is imported (sim.make_warp calls it
before building the physics; standalone tools can set the environment variables instead). configure() always exports
all six, so nothing is inherited from the shell:
  MJW_CLAMP_KV_FIX=1   0001: no actuator damping in the qvel VJP of dofs whose actuator force is clamped by
                       actuatorfrcrange (the forward does not depend on qvel there);
  MJW_ADJ_NORMAL=k     0001: derivative of the sphere / capsule self-contact normal n = v / |v|: 0 exact (MuJoCo Warp),
                       1 n constant, 2 (I - n n') / max(|v|, eps), 3 n constant where |v| < eps (NablaEngine's
                       gradient convention);
  MJW_ADJ_NORMAL_EPS   eps in m (default 1e-3);
  MJW_ADJ_PARALLEL=1   0002: near-parallel capsule pairs (sin^2 of the axis angle < MJW_ADJ_PARALLEL_EPS, default 1.2e-3)
                       keep their segment parameters constant in the backward (as NablaEngine);
  MJW_CLAMP_SOFT=d     0003: soft actuator clamp with width d > 0 in the backward (overrides MJW_CLAMP_KV_FIX);
                       g1_fast.adjoint_patch follows it on the training path;
  MJW_IFT_SOLVE=k      0004: 1 scales the IFT solve's right-hand side by a power of two (the 1e30-border corner pivot
                       then does not become NaN for large right-hand sides), 2 solves without the augmented border;
                       finite gradients unchanged beyond float32 rounding.
The forward is the same with every setting.
"""

import os
import sys


def configure(clamp_kv_fix=False, adj_normal=0, adj_normal_eps=1e-3, adj_parallel=0, clamp_soft=0.0, ift_solve=0):
    want = {"MJW_CLAMP_KV_FIX": "1" if clamp_kv_fix else "0", "MJW_ADJ_NORMAL": str(int(adj_normal)),
            "MJW_ADJ_NORMAL_EPS": repr(float(adj_normal_eps)), "MJW_ADJ_PARALLEL": str(int(adj_parallel)),
            "MJW_CLAMP_SOFT": repr(float(clamp_soft)), "MJW_IFT_SOLVE": str(int(ift_solve))}
    if "mujoco_warp" in sys.modules:  # too late to change: only accept what the loaded modules already use
        have = active()
        asked = {"clamp_kv_fix": bool(clamp_kv_fix), "adj_normal": int(adj_normal), "adj_normal_eps": float(adj_normal_eps),
                 "adj_parallel": int(adj_parallel), "clamp_soft": float(clamp_soft), "ift_solve": int(ift_solve)}
        assert have == asked, f"mujoco_warp was imported with backward options {have}; configure() must run before the import"
    os.environ.update(want)
    return want


def active():
    """The options of the loaded MuJoCo Warp modules (for logs / config)."""
    from mujoco_warp._src import adjoint as adj
    from mujoco_warp._src import collision_primitive_core as cpc
    from mujoco_warp._src import forward_adjoint as fa
    return {"clamp_kv_fix": bool(getattr(fa, "CLAMP_KV_FIX", False)), "adj_normal": int(getattr(cpc, "ADJ_NORMAL", 0)),
            "adj_normal_eps": float(getattr(cpc, "ADJ_NORMAL_EPS", 1e-3)), "adj_parallel": int(getattr(cpc, "ADJ_PARALLEL", 0)),
            "clamp_soft": float(getattr(fa, "CLAMP_SOFT", 0.0)), "ift_solve": int(getattr(adj, "IFT_SOLVE", 0))}
