"""Batched physics for TrackingEnv: NablaEngine (the fused differentiable engine) or patched MuJoCo Warp, between
vmapped per-env pre- and post-step functions, and the per-step reset select."""

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from .nets import tree_where

BIN = 50  # reset-sampling bins of 1 s

# Per State leaf, True where the reset select may keep the stepped value: leaves the physics step never writes and every
# reset state shares (MJX's constraint and contact buffers, which the engine and MuJoCo Warp paths never touch). Found at
# start-up from probe states (skip_mask); None selects every leaf.
STATE_SKIP = None


def select_reset(done, reset, s2):
    """tree_where(done, reset, s2), skipping the leaves STATE_SKIP marks (their two values are identical)."""
    if STATE_SKIP is None:
        return tree_where(done, reset, s2)
    return jax.tree_util.tree_map(
        lambda skip, x, y: y if skip else jnp.where(done.reshape(done.shape + (1,) * (x.ndim - done.ndim)), x, y),
        STATE_SKIP, reset, s2)


def skip_mask(x0, *others):
    """Per leaf of the batched state x0: True where it equals the same leaf of every state in others and is the same in
    every env (never for PRNG keys)."""
    def skippable(x, *ys):
        if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
            return False
        x, ys = np.asarray(x), [np.asarray(y) for y in ys]
        nan = x.dtype.kind == "f"
        return bool(all(np.array_equal(x, y, equal_nan=nan) for y in ys)
                    and np.array_equal(x, np.broadcast_to(x[:1], x.shape), equal_nan=nan))
    return jax.tree_util.tree_map(skippable, x0, *others)


def vmapped(env, tables):
    """The env functions vmapped over a batch; "step" runs env.warp ({batch size: physics} or one physics object) on the
    whole batch and finishes each env's state."""
    e = env.bind(tables)
    f = {k: jax.vmap(getattr(e, k)) for k in
         ("reset", "policy_obs", "critic_obs", "reward", "terminated", "clip_ended", "hazard_features")}

    def step_batched(state, action):
        wp_ = env.warp[state.frame.shape[0]] if isinstance(env.warp, dict) else env.warp
        out = wp_.step(state.dx.qpos, state.dx.qvel, state.dx.qacc_warmstart,
                       jax.vmap(e.ctrl_of)(action, state.frame), state.xi.friction, state.xi.com)
        if len(out) == 8:  # engine with kinematics: (qpos', qvel', warm', qpos_n, xpos, xquat, cvel, subtree_com)
            _, v, w, qn, xp, xq, cv, sc = out
            return jax.vmap(e.finish_physics_kin)(state, action, qn, v, w, xp, xq, cv, sc)
        q, v, w = out
        return jax.vmap(e.finish_physics)(state, action, q, v, w)

    f["step"] = step_batched
    return f


def make_engine(env, sizes, **options):
    """{batch size: nablaengine.jax_ffi.EnginePhysics} with the body kinematics as step outputs, its library built for
    env.m (use_model=True; built once per GPU architecture into $DK_CACHE_DIR). options: the engine's gradient options
    (clamp_kv, normal, clamp_soft)."""
    from nablaengine.jax_ffi import EnginePhysics
    env.m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)  # sensors are never read
    return {b: EnginePhysics(env.m, b, np.asarray(env.foot_pairs), env.torso_id, kinematics=True, use_model=True,
                             **options) for b in sizes}


def engine_info(phys):
    """The NablaEngine library's identity for logs and config.json."""
    i = phys.info
    return {k: i.get(k) for k in ("version", "commit", "dirty", "key", "arch", "built", "consts_sha256")}


def make_warp(env, sizes, train=True, adj_normal=0, adj_parallel=0, ift_solve=0):
    """{batch size: warp_physics.WarpPhysics}. train: the training setting (64 contacts and 192 constraint rows per
    world, reverse-mode RNE VJP in the adjoint, warm-started recompute, sensors disabled) with the backward switches of
    patches/mujoco_warp (mjw_options); otherwise the forward-only evaluation setting (capacity from MJX's padding)."""
    from . import mjw_options
    mjw_options.configure(False, adj_normal, 1e-3, adj_parallel, 0.0, ift_solve)
    from .warp_physics import WarpPhysics
    fast = {}
    if train:
        print(f"MuJoCo Warp backward options: {mjw_options.active()}", flush=True)
        fast = dict(nconmax=64, njmax=192, rne_vjp=True, warm_recompute=True)
        env.m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)  # sensors are never read
    return {b: WarpPhysics(env.m, b, np.asarray(env.foot_pairs), env.torso_id, **fast) for b in sizes}
