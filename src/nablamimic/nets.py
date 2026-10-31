"""Actor and critic networks, observation normalization and the per-sample action-gradient clip."""

import jax
import jax.numpy as jnp
import numpy as np

# Matmul precision of the actor / critic MLPs. None follows jax_default_matmul_precision; "tensorfloat32" under a
# "highest" default (set_matmul_precision("nets_tf32"), the training setting) keeps the physics in fp32.
NET_PRECISION = None


def net_precision(f):
    def g(*args):
        if NET_PRECISION is None:
            return f(*args)
        with jax.default_matmul_precision(NET_PRECISION):
            return f(*args)
    return g


def set_matmul_precision(mode):
    """nets_tf32: actor and critic in TF32 (functions wrapped by net_precision), everything else fp32;
    highest: fp32 everywhere (evaluation)."""
    global NET_PRECISION
    assert mode in ("nets_tf32", "highest"), mode
    jax.config.update("jax_default_matmul_precision", "highest")
    NET_PRECISION = "tensorfloat32" if mode == "nets_tf32" else None


def mlp_init(key, sizes, out_scale=1.0):
    params = []
    keys = jax.random.split(key, len(sizes) - 1)
    for i, (k, n_in, n_out) in enumerate(zip(keys, sizes[:-1], sizes[1:])):
        w = jax.nn.initializers.orthogonal(np.sqrt(2.0))(k, (n_in, n_out))
        if i == len(sizes) - 2:
            w = w * out_scale
        params.append((w, jnp.zeros(n_out)))
    return params


def mlp_apply(params, x):
    for w, b in params[:-1]:
        x = jax.nn.elu(x @ w + b)
    w, b = params[-1]
    return x @ w + b


def tree_where(mask, a, b):
    return jax.tree_util.tree_map(
        lambda x, y: jnp.where(mask.reshape(mask.shape + (1,) * (x.ndim - mask.ndim)), x, y), a, b)


def td_lambda(r, term, done, v2, gamma, lam):
    """r, term, done, v2: (H, N). v2 is the value of the pre-reset next state."""
    def back(g_next, x):
        r_t, term_t, done_t, v2_t = x
        boot = (1 - lam) * v2_t + lam * jnp.where(done_t, v2_t, g_next)
        g = r_t + gamma * (1.0 - term_t) * boot
        return g, g
    _, g = jax.lax.scan(back, v2[-1], (r, term.astype(r.dtype), done, v2), reverse=True)
    return g


def normalize(norm, x):
    mean, var = norm
    return jnp.clip((x - mean) / jnp.sqrt(var + 1e-5), -10.0, 10.0)


critic_apply = net_precision(mlp_apply)
ACTOR_HIDDEN, ACTOR_LATENT, CRITIC_HIDDEN = (2048, 1024, 512, 256), 32, (256, 128)


def actor_init(key, env, hidden=ACTOR_HIDDEN, latent=ACTOR_LATENT, anchor_to_decoder=0):
    """Encoder-decoder actor (SuperMimic's): the goal encoder maps the reference frames to z = tanh(.) of width latent,
    the decoder maps [z, proprioception history] to the action. anchor_to_decoder = k > 0: the goal's last k entries
    (the anchor-relative orientation) go to the decoder beside the history instead of through the encoder, which then
    sees only the future reference frames (marked by the parameter "anchor_dec" of shape (k,), never trained)."""
    ke, kd = jax.random.split(key)
    k = anchor_to_decoder
    p = dict(enc=mlp_init(ke, [env.goal_dim - k, *hidden, latent]),
             dec=mlp_init(kd, [latent + env.hist_dim + k, *hidden, env.nu], out_scale=0.01))
    if k:
        p["anchor_dec"] = jnp.zeros((k,), jnp.float32)
    return p


@net_precision
def actor_apply(p, hist, goal):
    if "net" in p:  # PPO MLP policy (ppo.py checkpoints): mean action from the normalized [history, goal]
        return mlp_apply(p["net"], normalize(jax.lax.stop_gradient(p["obs_norm"]), jnp.concatenate([hist, goal], -1)))
    if "obs_norm" in p:  # --actor-obs-norm: running mean / variance of [history, goal], not trained
        x = normalize(jax.lax.stop_gradient(p["obs_norm"]), jnp.concatenate([hist, goal], -1))
        hist, goal = x[..., :hist.shape[-1]], x[..., hist.shape[-1]:]
    if "anchor_dec" in p:  # --anchor-to-decoder: the goal's anchor part bypasses the encoder
        k = p["anchor_dec"].shape[-1]
        z = jnp.tanh(mlp_apply(p["enc"], goal[..., :-k]))
        return mlp_apply(p["dec"], jnp.concatenate([z, hist, goal[..., -k:]], -1))
    z = jnp.tanh(mlp_apply(p["enc"], goal))
    return mlp_apply(p["dec"], jnp.concatenate([z, hist], -1))


@net_precision
def actor_encode(p, goal_ref):
    """--anchor-to-decoder actor: the motion encoder on the goal's reference part alone."""
    return jnp.tanh(mlp_apply(p["enc"], goal_ref))


@net_precision
def actor_decode(p, z, hist, anchor):
    """--anchor-to-decoder actor: the decoder on (encoder output, history, anchor part of the goal)."""
    return mlp_apply(p["dec"], jnp.concatenate([z, hist, anchor], -1))


def clip_rows_to_median(x, kappa, probe=None):
    """Identity on x (n, d); the backward clips each row's cotangent norm to kappa x the
    median row norm of the batch (per-sample clipping that adapts to the gradient scale).

    probe: optional zeros of shape (2,); the backward writes [fraction of live rows clipped,
    fraction of the summed row norm removed] into its cotangent, so the gradient with
    respect to probe accumulates these statistics over all calls (e.g. the steps of a scan)."""

    @jax.custom_vjp
    def ident(x, p):
        return x

    def fwd(x, p):
        return x, None

    def bwd(_, g):
        nrm = jnp.sqrt(jnp.sum(g ** 2, -1))
        # median over rows that carry gradient: rows of envs masked out of the objective (e.g. after an
        # episode end with mask_after_done) are exactly zero and would otherwise pull the threshold down
        live = nrm > 0
        thr = kappa * jnp.nanmedian(jnp.where(live, nrm, jnp.nan))
        thr = jnp.where(jnp.isfinite(thr), thr, jnp.inf)
        s = jnp.minimum(1.0, thr / (nrm + 1e-30))
        frac = jnp.sum(live & (s < 1.0)) / jnp.maximum(jnp.sum(live), 1)
        stats = jnp.stack([frac.astype(g.dtype), 1.0 - jnp.sum(nrm * s) / (jnp.sum(nrm) + 1e-30)])
        return g * s[:, None], stats

    ident.defvjp(fwd, bwd)
    return ident(x, jnp.zeros(2, x.dtype) if probe is None else probe)
