"""An adjoint path for the G1 that computes lam^T d(qfrc_bias)/d(qvel) as one reverse-mode VJP instead of forming the
nv x nv Jacobian; the same gradients up to float rounding.

mujoco_warp's smooth_qvel_backward builds d(qfrc_bias)/d(qvel) as an nv x nv matrix (deriv_rne_vel: nv forward-mode
directions through the tree, level by level) and multiplies it by one cotangent lam. This patch keeps the actuator /
passive velocity derivative (deriv_smooth_vel) and subtracts lam^T d(qfrc_bias)/d(qvel) computed by the adjoint of a
fused, tree-unrolled RNE kernel (smooth_gen.generate_rne_bias), with cdof and cinert held at the linearization point
as deriv_rne_vel does.

  from nablamimic.g1_fast import adjoint_patch
  adjoint_patch.install(mjm)      # before the first backward; affects every MJWarp adjoint in the process

The replacement follows mujoco_warp/_src/forward_adjoint.py smooth_qvel_backward (MuJoCo Warp, Apache-2.0, The Newton
Developers) and calls its kernels; see THIRD_PARTY_NOTICES.md.
"""

import dataclasses

import warp as wp

from mujoco_warp._src import adjoint as _adjoint
from mujoco_warp._src import derivative as _derivative
from mujoco_warp._src import forward as _forward
from mujoco_warp._src import forward_adjoint as _fa

from .smooth_gen import build_rne_bias

_STATE = {}


@wp.kernel
def _sub(a: wp.array2d(dtype=float), out: wp.array2d(dtype=float)):
    w, i = wp.tid()
    out[w, i] -= a[w, i]


def clamp_kv_fix():
    """Whether this MuJoCo Warp drops the actuator damping of clamped dofs from the qvel VJP: patch 0001 calls
    forward_adjoint.clamped_actuator_qvel_backward when forward_adjoint.CLAMP_KV_FIX (MJW_CLAMP_KV_FIX=1 at import);
    without the function the damping stays. The fused path here (_clamped_kv) follows the same test."""
    return hasattr(_fa, "clamped_actuator_qvel_backward") and bool(getattr(_fa, "CLAMP_KV_FIX", True))


def clamp_soft():
    """Patch 0003's soft clamp: forward_adjoint.CLAMP_SOFT, delta > 0 from MJW_CLAMP_SOFT at import, 0.0 when off or
    without the patch. With it forward_adjoint.actuator_lam scales the clamped dofs' multiplier by w instead of zeroing
    it (the qpos and ctrl paths here call it too) and smooth_qvel_backward applies the qvel correction with it whatever
    MJW_CLAMP_KV_FIX says (_clamped_kv here as well)."""
    return float(getattr(_fa, "CLAMP_SOFT", 0.0))


def _clamped_kv(m, d_out, lam, adj_qvel, bc):
    """As the patched smooth_qvel_backward: drop the actuator damping of clamped dofs from adj_qvel when
    MJW_CLAMP_KV_FIX=1 (patch 0001), or keep kv w lam with the soft clamp (MJW_CLAMP_SOFT > 0, patch 0003);
    otherwise nothing changes."""
    if clamp_kv_fix() or clamp_soft() > 0.0:
        _fa.clamped_actuator_qvel_backward(m, d_out, lam, adj_qvel, bc)


def _smooth_qvel_backward(m, d, d_out, lam, adj_qvel, bc):
    """smooth_qvel_backward without deriv_rne_vel; the RNE term comes from a reverse-mode VJP."""
    nworld = d.nworld
    linearization = dataclasses.replace(d_out, qvel=d.qvel)
    _derivative.deriv_smooth_vel(m, linearization, bc.qDeriv)
    wp.launch(_forward._map_m2d, dim=(nworld, m.nD), inputs=[m.mapM2D, bc.qDeriv], outputs=[bc.qLU])
    wp.launch(_forward._map_m2d, dim=(nworld, m.nD), inputs=[m.mapM2D, d_out.M], outputs=[bc.M_D])
    wp.launch(_fa._smooth_qvel_vjp, dim=(nworld, m.nD),
              inputs=[m.opt.timestep, m.qD_fullm_i, m.qD_fullm_j, bc.qLU, bc.M_D, lam], outputs=[adj_qvel])
    _clamped_kv(m, d_out, lam, adj_qvel, bc)
    # adj_qvel -= lam^T d(qfrc_bias)/d(qvel)
    key = nworld
    if key not in _STATE["buf"]:
        nv = d.qvel.shape[1]
        _STATE["buf"][key] = dict(bias=wp.zeros((nworld, nv), dtype=float), vjp=wp.zeros((nworld, nv), dtype=float),
                                  lam=wp.zeros((nworld, nv), dtype=float),
                                  adj_cdof=wp.zeros(d_out.cdof.shape, dtype=wp.spatial_vector),
                                  adj_cinert=wp.zeros(d_out.cinert.shape, dtype=d_out.cinert.dtype))
    b = _STATE["buf"][key]
    b["vjp"].zero_()
    wp.copy(b["lam"], lam)  # the adjoint launch consumes (clears) its output cotangent; lam is used again below
    wp.launch(_STATE["mod"].g1_rne_bias, dim=nworld, inputs=[d.qvel, d_out.cdof, d_out.cinert], outputs=[b["bias"]],
              adj_inputs=[b["vjp"], b["adj_cdof"], b["adj_cinert"]], adj_outputs=[b["lam"]], adjoint=True)
    wp.launch(_sub, dim=(nworld, d.qvel.shape[1]), inputs=[b["vjp"]], outputs=[adj_qvel])
    if m.nu and d.ctrl.requires_grad:
        wp.launch(_fa._smooth_ctrl_vjp, dim=(nworld, m.nu),
                  inputs=[m.actuator_gainprm, d_out.moment_rownnz, d_out.moment_rowadr, d_out.moment_colind,
                          d_out.actuator_moment, _fa.actuator_lam(m, d_out, lam, bc)],
                  outputs=[d.ctrl.grad])
    if d.xfrc_applied.requires_grad:
        raise NotImplementedError("xfrc_applied gradients are not supported by the patched adjoint")


def install(mjm):
    """Patch mujoco_warp's adjoint for this model (idempotent)."""
    if _STATE.get("installed"):
        return
    _STATE.update(mod=build_rne_bias(mjm), buf={}, orig=(_adjoint.smooth_qvel_backward, _fa.smooth_qvel_backward))
    _adjoint.smooth_qvel_backward = _fa.smooth_qvel_backward = _smooth_qvel_backward
    _STATE["installed"] = True


def uninstall():
    if _STATE.get("installed"):
        _adjoint.smooth_qvel_backward, _fa.smooth_qvel_backward = _STATE["orig"]
        _STATE["installed"] = False
