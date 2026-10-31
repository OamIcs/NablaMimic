"""MuJoCo Warp (with its analytic adjoint) physics for TrackingEnv as one batched, differentiable JAX op.

  step(qpos, qvel, warm, ctrl, friction, com) -> (qpos', qvel', warm')      all (B, .)

runs SUBSTEPS x mjw.step on the whole batch in one Warp call. Its custom VJP recomputes the
substeps under a wp.Tape and runs MuJoCo Warp's analytic adjoint, returning d/dqpos, d/dqvel and
d/dctrl (the warm start and the DR parameters get no gradient). DR is the same as
TrackingEnv.model_for: per-world tangential friction of the foot-floor pairs and a torso
centre-of-mass offset, written into the model arrays of each world before every step.

Both FFI callbacks are captured as CUDA graphs (JaxCallableGraphMode.WARP_STAGED). Construct
WarpPhysics before tracing anything that calls it: that loads every Warp module and runs the
callbacks once outside any capture. One instance per batch size; one process per GPU.

The patched MuJoCo Warp of scripts/setup_mujoco_warp.sh is required (the base adjoint plus patches/mujoco_warp); its
backward switches are read when mujoco_warp is imported, so import this module only after mjw_options.configure().
Contact / constraint capacity defaults to MJX's static padding (x2), sized so that it did not overflow
in any reported run; `stats()` reports the high-water marks of the last call.
"""

import dataclasses

import mujoco
import numpy as np
import warp as wp
from mujoco import mjx

import mujoco_warp as mjw
from mujoco_warp._src.types import vec5

SUBSTEPS = 4


@wp.kernel
def _set_dr(friction: wp.array(dtype=float), com: wp.array2d(dtype=float), foot_pairs: wp.array(dtype=int),
            torso: int, base_ipos: wp.vec3, pair_friction: wp.array2d(dtype=vec5),
            body_ipos: wp.array2d(dtype=wp.vec3)):
    w = wp.tid()
    f = friction[w]
    for i in range(foot_pairs.shape[0]):
        p = foot_pairs[i]
        v = pair_friction[w, p]
        v[0] = f
        v[1] = f
        pair_friction[w, p] = v
    body_ipos[w, torso] = base_ipos + wp.vec3(com[w, 0], com[w, 1], com[w, 2])


@wp.kernel(enable_backward=False)
def _store_warm(src: wp.array2d(dtype=float), k: int, nv: int, dst: wp.array2d(dtype=float)):
    w, i = wp.tid()
    dst[w, k * nv + i] = src[w, i]


@wp.kernel(enable_backward=False)
def _load_warm(src: wp.array2d(dtype=float), k: int, nv: int, dst: wp.array2d(dtype=float)):
    w, i = wp.tid()
    dst[w, i] = src[w, k * nv + i]


@wp.kernel
def _accum(a: wp.array2d(dtype=float), out: wp.array2d(dtype=float)):
    w, i = wp.tid()
    out[w, i] += a[w, i]


class _Sim:
    """MJWarp model and forward/backward buffers for one device and batch size."""

    def __init__(self, mjm, nworld, foot_pairs, torso_id, nconmax, njmax):
        mjd = mujoco.MjData(mjm)
        mujoco.mj_forward(mjm, mjd)
        self.m = mjw.put_model(mjm, batch_sizes={"pair_friction": nworld, "body_ipos": nworld})
        self.m.opt.warn_overflow = 0
        kw = dict(nworld=nworld, nconmax=nconmax, njmax=njmax)
        self.df = mjw.put_data(mjm, mjd, **kw)
        self.D = [mjw.put_data(mjm, mjd, **kw) for _ in range(SUBSTEPS + 1)]
        for a in (self.D[0].qpos, self.D[0].qvel, self.D[0].ctrl):
            a.requires_grad = True
        self.bc = mjw.create_backward_context(self.m, self.D[0])
        self.foot_pairs = wp.array(np.asarray(foot_pairs, np.int32), dtype=int)
        self.torso = int(torso_id)
        self.base_ipos = wp.vec3(*mjm.body_ipos[torso_id].astype(np.float32))
        self.nworld = nworld
        self.nv = int(mjm.nv)
        # the warm-started recompute runs one solver iteration (same arrays, so DR applies to both views)
        self.m_ws = dataclasses.replace(self.m, opt=dataclasses.replace(self.m.opt, iterations=1))

    def _dr(self, friction, com):
        wp.launch(_set_dr, dim=self.nworld,
                  inputs=[friction, com, self.foot_pairs, self.torso, self.base_ipos],
                  outputs=[self.m.pair_friction, self.m.body_ipos])

    def forward(self, qpos, qvel, warm, ctrl, friction, com, qpos_out, qvel_out, warm_out):
        d = self.df
        self._dr(friction, com)
        wp.copy(d.qpos, qpos)
        wp.copy(d.qvel, qvel)
        wp.copy(d.qacc_warmstart, warm)
        wp.copy(d.ctrl, ctrl)
        for _ in range(SUBSTEPS):
            mjw.step(self.m, d)
        wp.copy(qpos_out, d.qpos)
        wp.copy(qvel_out, d.qvel)
        wp.copy(warm_out, d.qacc_warmstart)

    def forward_res(self, qpos, qvel, warm, ctrl, friction, com, qpos_out, qvel_out, warm_out, warm_seq):
        """forward(), also writing each substep's converged acceleration into warm_seq (nworld, SUBSTEPS * nv)."""
        d = self.df
        self._dr(friction, com)
        wp.copy(d.qpos, qpos)
        wp.copy(d.qvel, qvel)
        wp.copy(d.qacc_warmstart, warm)
        wp.copy(d.ctrl, ctrl)
        for k in range(SUBSTEPS):
            mjw.step(self.m, d)
            wp.launch(_store_warm, dim=(self.nworld, self.nv), inputs=[d.qacc_warmstart, k, self.nv], outputs=[warm_seq])
        wp.copy(qpos_out, d.qpos)
        wp.copy(qvel_out, d.qvel)
        wp.copy(warm_out, d.qacc_warmstart)

    def backward_ws(self, qpos, qvel, warm, ctrl, friction, com, warm_seq, g_qpos, g_qvel, gq_out, gv_out, gu_out):
        """backward() whose recompute starts every substep's solver at the forward's converged acceleration and runs
        one iteration (the adjoint is taken at the converged solution, so the gradients are the same up to float
        rounding)."""
        D = self.D
        self._dr(friction, com)
        wp.copy(D[0].qpos, qpos)
        wp.copy(D[0].qvel, qvel)
        wp.copy(D[0].qacc_warmstart, warm)
        wp.copy(D[0].ctrl, ctrl)
        tape = wp.Tape()
        with mjw.backward_context(self.bc):
            for k in range(SUBSTEPS):
                wp.launch(_load_warm, dim=(self.nworld, self.nv), inputs=[warm_seq, k, self.nv],
                          outputs=[D[k].qacc_warmstart])
                with tape:
                    mjw.step(self.m_ws, D[k], D[k + 1])
            tape.zero()
            wp.copy(D[-1].qpos.grad, g_qpos)
            wp.copy(D[-1].qvel.grad, g_qvel)
            tape.backward()
        wp.copy(gq_out, D[0].qpos.grad)
        wp.copy(gv_out, D[0].qvel.grad)
        gu_out.zero_()
        for k in range(SUBSTEPS):
            wp.launch(_accum, dim=gu_out.shape, inputs=[D[k].ctrl.grad], outputs=[gu_out])

    def backward(self, qpos, qvel, warm, ctrl, friction, com, g_qpos, g_qvel, gq_out, gv_out, gu_out):
        D = self.D
        self._dr(friction, com)
        wp.copy(D[0].qpos, qpos)
        wp.copy(D[0].qvel, qvel)
        wp.copy(D[0].qacc_warmstart, warm)
        wp.copy(D[0].ctrl, ctrl)
        tape = wp.Tape()
        with mjw.backward_context(self.bc):
            with tape:
                for k in range(SUBSTEPS):
                    mjw.step(self.m, D[k], D[k + 1])
            tape.zero()
            wp.copy(D[-1].qpos.grad, g_qpos)
            wp.copy(D[-1].qvel.grad, g_qvel)
            tape.backward()
        wp.copy(gq_out, D[0].qpos.grad)
        wp.copy(gv_out, D[0].qvel.grad)
        gu_out.zero_()
        for k in range(SUBSTEPS):
            wp.launch(_accum, dim=gu_out.shape, inputs=[D[k].ctrl.grad], outputs=[gu_out])


def mjx_capacity(mjm):
    """(contacts, constraint rows) per world that MJX pads this model to."""
    d = mjx.make_data(mjm)
    return int(d._impl.contact.dist.shape[0]), int(d._impl.efc_J.shape[0])


class WarpPhysics:
    def __init__(self, mjm, nworld, foot_pairs, torso_id, capacity_factor=2.0, nconmax=None, njmax=None,
                 rne_vjp=False, warm_recompute=False):
        """nconmax / njmax: contacts and constraint rows per world (default: capacity_factor x MJX's padding);
        rne_vjp: install g1_fast.adjoint_patch (reverse-mode RNE VJP in the adjoint, same gradients);
        warm_recompute: the forward keeps each substep's converged acceleration as a residual and the backward's
        recompute starts its solver there with one iteration (_Sim.backward_ws)."""
        import jax
        import jax.numpy as jnp

        ncon, nefc = mjx_capacity(mjm)
        self.nconmax, self.njmax = int(np.ceil(capacity_factor * ncon)), int(np.ceil(capacity_factor * nefc))
        self.nconmax, self.njmax = int(nconmax or self.nconmax), int(njmax or self.njmax)
        if rne_vjp:
            from .g1_fast import adjoint_patch
            adjoint_patch.install(mjm)  # before the warm-up, so its buffers are allocated outside any capture
        wp.config.quiet = True
        mjw.enable_grad()
        dev = wp.get_device("cuda:0")
        with wp.ScopedDevice(dev):
            self.sim = _Sim(mjm, nworld, foot_pairs, torso_id, self.nconmax, self.njmax)
            self._warmup(warm_recompute)
            wp.synchronize_device(dev)
        sim = self.sim

        def fwd_cb(qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float), warm: wp.array2d(dtype=float),
                   ctrl: wp.array2d(dtype=float), friction: wp.array(dtype=float), com: wp.array2d(dtype=float),
                   qpos_out: wp.array2d(dtype=float), qvel_out: wp.array2d(dtype=float),
                   warm_out: wp.array2d(dtype=float)):
            sim.forward(qpos, qvel, warm, ctrl, friction, com, qpos_out, qvel_out, warm_out)

        def bwd_cb(qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float), warm: wp.array2d(dtype=float),
                   ctrl: wp.array2d(dtype=float), friction: wp.array(dtype=float), com: wp.array2d(dtype=float),
                   g_qpos: wp.array2d(dtype=float), g_qvel: wp.array2d(dtype=float),
                   gq_out: wp.array2d(dtype=float), gv_out: wp.array2d(dtype=float),
                   gu_out: wp.array2d(dtype=float)):
            sim.backward(qpos, qvel, warm, ctrl, friction, com, g_qpos, g_qvel, gq_out, gv_out, gu_out)

        gm = wp.JaxCallableGraphMode.WARP_STAGED
        fwd_call = wp.jax_callable(fwd_cb, num_outputs=3, graph_mode=gm)
        bwd_call = wp.jax_callable(bwd_cb, num_outputs=3, graph_mode=gm)

        def fwd_res_cb(qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float), warm: wp.array2d(dtype=float),
                       ctrl: wp.array2d(dtype=float), friction: wp.array(dtype=float), com: wp.array2d(dtype=float),
                       qpos_out: wp.array2d(dtype=float), qvel_out: wp.array2d(dtype=float),
                       warm_out: wp.array2d(dtype=float), warm_seq: wp.array2d(dtype=float)):
            sim.forward_res(qpos, qvel, warm, ctrl, friction, com, qpos_out, qvel_out, warm_out, warm_seq)

        def bwd_ws_cb(qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float), warm: wp.array2d(dtype=float),
                      ctrl: wp.array2d(dtype=float), friction: wp.array(dtype=float), com: wp.array2d(dtype=float),
                      warm_seq: wp.array2d(dtype=float), g_qpos: wp.array2d(dtype=float),
                      g_qvel: wp.array2d(dtype=float), gq_out: wp.array2d(dtype=float),
                      gv_out: wp.array2d(dtype=float), gu_out: wp.array2d(dtype=float)):
            sim.backward_ws(qpos, qvel, warm, ctrl, friction, com, warm_seq, g_qpos, g_qvel, gq_out, gv_out, gu_out)

        fwd_res_call = wp.jax_callable(fwd_res_cb, num_outputs=4, graph_mode=gm)
        bwd_ws_call = wp.jax_callable(bwd_ws_cb, num_outputs=3, graph_mode=gm)
        nseq = SUBSTEPS * int(mjm.nv)

        @jax.custom_vjp
        def step(qpos, qvel, warm, ctrl, friction, com):
            dims = {"qpos_out": qpos.shape, "qvel_out": qvel.shape, "warm_out": warm.shape}
            return tuple(fwd_call(qpos, qvel, warm, ctrl, friction, com, output_dims=dims))

        def step_fwd(qpos, qvel, warm, ctrl, friction, com):
            if warm_recompute:
                dims = {"qpos_out": qpos.shape, "qvel_out": qvel.shape, "warm_out": warm.shape,
                        "warm_seq": (qpos.shape[0], nseq)}
                qo, vo, wo, ws = fwd_res_call(qpos, qvel, warm, ctrl, friction, com, output_dims=dims)
                return (qo, vo, wo), (qpos, qvel, warm, ctrl, friction, com, ws)
            return step(qpos, qvel, warm, ctrl, friction, com), (qpos, qvel, warm, ctrl, friction, com, None)

        def step_bwd(res, cot):
            qpos, qvel, warm, ctrl, friction, com, ws = res
            gq, gv, _ = cot
            dims = {"gq_out": qpos.shape, "gv_out": qvel.shape, "gu_out": ctrl.shape}
            if warm_recompute:
                dq, dv, du = bwd_ws_call(qpos, qvel, warm, ctrl, friction, com, ws, gq, gv, output_dims=dims)
            else:
                dq, dv, du = bwd_call(qpos, qvel, warm, ctrl, friction, com, gq, gv, output_dims=dims)
            return dq, dv, jnp.zeros_like(warm), du, jnp.zeros_like(friction), jnp.zeros_like(com)

        step.defvjp(step_fwd, step_bwd)
        self.step = step

    def _warmup(self, warm_recompute=False):
        s, nw = self.sim, self.sim.nworld
        z = lambda a: wp.zeros_like(a, requires_grad=False)  # noqa: E731
        fr = wp.full(nw, 0.75, dtype=float)
        com = wp.zeros((nw, 3), dtype=float)
        s.forward(s.df.qpos, s.df.qvel, s.df.qacc_warmstart, s.df.ctrl, fr, com,
                  z(s.df.qpos), z(s.df.qvel), z(s.df.qvel))
        s.backward(s.df.qpos, s.df.qvel, s.df.qacc_warmstart, s.df.ctrl, fr, com,
                   z(s.df.qpos), z(s.df.qvel), z(s.df.qpos), z(s.df.qvel), z(s.df.ctrl))
        if warm_recompute:
            ws = wp.zeros((nw, SUBSTEPS * s.nv), dtype=float)
            s.forward_res(s.df.qpos, s.df.qvel, s.df.qacc_warmstart, s.df.ctrl, fr, com,
                          z(s.df.qpos), z(s.df.qvel), z(s.df.qvel), ws)
            s.backward_ws(s.df.qpos, s.df.qvel, s.df.qacc_warmstart, s.df.ctrl, fr, com, ws,
                          z(s.df.qpos), z(s.df.qvel), z(s.df.qpos), z(s.df.qvel), z(s.df.ctrl))

    def stats(self):
        """High-water marks of the last forward call (host sync; call outside jit)."""
        d = self.sim.df
        return {"nacon": int(d.nacon.numpy()[0]), "naconmax": self.nconmax * self.sim.nworld,
                "nefc_max": int(d.nefc.numpy().max()), "njmax": self.njmax}
