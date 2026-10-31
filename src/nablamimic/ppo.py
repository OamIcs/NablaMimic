"""PPO baselines in the same environment as train.py.

Everything outside the algorithm is shared with train.py: env.TrackingEnv (proprioceptive actor observation with sensor
noise and the anchor orientation, privileged critic observation, Table A.2 reward, DR, terminations, 10 s episodes),
the reset bins of 1 s over the concatenated library and the forward of NablaEngine (PPO needs no
gradients). Checkpoints are evaluated by evaluate.py like train.py's (deterministic mean action).

An independent JAX implementation that follows rsl_rl's PPO (https://github.com/leggedrobotics/rsl_rl, BSD-3-Clause,
see THIRD_PARTY_NOTICES.md) with the BeyondMimic / mjlab motion-tracking settings as defaults: 4096 envs x 24 steps per
iteration, 5 epochs x 4 minibatches, one Adam over actor and critic with lr 1e-3 and the adaptive-KL schedule (target
0.01, per minibatch: lr / 1.5 above twice the target, x 1.5 below half of it, within [1e-5, 1e-2]), clip 0.2, clipped
value loss with coefficient 1, entropy 0.005, gamma 0.99, lambda 0.95, global grad-norm clip 1, Gaussian policy with a
state-independent std (init 1), ELU MLPs (512, 256, 128), empirical observation normalization, advantages normalized
over the batch. Where it differs from rsl_rl: orthogonal initialization (actor output scale 0.01), a per-joint log-std
parameter, the normalizers use sqrt(var + 1e-5), clip at +-10 and are updated once per iteration after the update;
time-outs and clip ends bootstrap the value of the pre-reset next state through the reward (r += gamma V(s'));
terminations do not; the KL has no epsilon; the advantage std is the population std.

--reset-sampling beyondmimic: BeyondMimic's adaptive sampling (Liao et al. 2025, arXiv:2508.08241;
HybridRobotics/whole_body_tracking MotionCommand._adaptive_sampling, MIT): per 1 s bin an EMA (alpha 0.001 per step,
applied once per iteration for its T steps) of the failures that ended there, sampling probability proportional to it
plus 0.1 / bins, uniform within a bin. --reset-sampling supermimic: SuperMimic's return-based sampling (as train.py's,
with the GAE return). --arch supermimic: SuperMimic's encoder-decoder actor (2048-1024-512-256, latent 32; the whole
goal into the encoder, train.py's actor without --anchor-to-decoder).
Matmul precision: the MLP actor and the critic run in fp32, the encoder-decoder actor in TF32 (nets.actor_apply).

  python -m nablamimic.ppo --config configs/ppo_beyondmimic.args --out runs/ppo_bm_s0
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import jax_cache, nets, sim
from .env import BM_BODIES, EnvConfig, TrackingEnv
from .motion import all_clip_names, load_library
from .nets import actor_apply, mlp_apply, mlp_init, normalize
from .params import git_state, latest_checkpoint, resolve_params
from .sim import BIN, select_reset, vmapped

FIXED = dict(robot="mjlab", obs_mode="anchor_ori", contacts="all", reward_frame="world", hist_len=10,
             state_select="live", physics="engine", matmul_precision="nets_tf32")


def gauss_logp(a, mu, log_std):
    return jnp.sum(-0.5 * ((a - mu) / jnp.exp(log_std)) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi), -1)


def norm_update(norm, x):
    """Running mean / variance (Chan et al.) over the rows of x; norm = (mean, var, count)."""
    mean, var, count = norm
    bm, bv, bc = x.mean(0), x.var(0), jnp.float32(x.shape[0])
    delta, tot = bm - mean, count + bc
    return (mean + delta * bc / tot, (var * count + bv * bc + delta ** 2 * count * bc / tot) / tot, tot)


def build_parser():
    ap = argparse.ArgumentParser(prog="python -m nablamimic.ppo",
                                 description="PPO baselines (see configs/ppo_*.args).")
    ap.add_argument("--config", default=None, metavar="FILE",
                    help="take the arguments from this configuration file (overrides may follow on the first start); "
                         "they are locked into OUT/config.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", default=None, help="checkpoint to continue from, or 'latest'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clips", nargs="+", default=["all"])
    ap.add_argument("--solver-iterations", type=int, default=10)
    ap.add_argument("--arch", default="mlp", choices=["mlp", "supermimic"])
    ap.add_argument("--hidden", type=int, nargs="+", default=[512, 256, 128])
    ap.add_argument("--critic-hidden", type=int, nargs="+", default=None, help="default: --hidden")
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=24, help="rollout steps per env per iteration")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--desired-kl", type=float, default=0.01, help="target of the adaptive learning rate")
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--value-coef", type=float, default=1.0)
    ap.add_argument("--entropy", type=float, default=0.005)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--init-std", type=float, default=1.0)
    ap.add_argument("--max-episode-steps", type=int, default=500)
    ap.add_argument("--reset-sampling", default="supermimic", choices=["supermimic", "beyondmimic"])
    ap.add_argument("--hours", type=float, default=12.0, help="wall-clock budget (training time)")
    ap.add_argument("--iters", type=int, default=10 ** 9)
    ap.add_argument("--ckpt-minutes", type=float, default=15.0, help="numbered checkpoints for the evaluation")
    ap.add_argument("--ckpt-dense", type=float, nargs=2, default=[1.0, 2.0], metavar=("HOURS", "MINUTES"),
                    help="a checkpoint every MINUTES during the first HOURS (dense early evaluation)")
    return ap


def main(argv=None):
    ap = build_parser()
    argv, params_info = resolve_params(sys.argv[1:] if argv is None else argv)
    args = ap.parse_args(argv)
    out = Path(args.out)
    if args.resume == "latest":
        args.resume = latest_checkpoint(out)
    assert args.desired_kl > 0, "--desired-kl: > 0"
    jax_cache.enable()
    nets.set_matmul_precision("nets_tf32")

    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    commit, dirty, _ = git_state()
    seg = {"start": time.strftime("%Y-%m-%dT%H:%M:%S"), "commit": commit, "dirty": dirty, "resume": args.resume,
           "gpu": jax.devices()[0].device_kind}
    if params_info is not None and params_info["locked"]:
        cfg = json.loads((out / "config.json").read_text())
        cfg["segments"] = cfg.get("segments", []) + [seg]
    else:
        cfg = dict(vars(args), **FIXED, algo="ppo", devices=1, git_commit=commit, git_dirty=dirty, gpu=seg["gpu"])
        if params_info is not None:
            cfg.update(argv=params_info["argv"], config_args=params_info["config_args"], config_file=str(args.config),
                       segments=[seg])
    log = open(out / "log.jsonl", "a")

    clips = all_clip_names() if args.clips == ["all"] else args.clips
    assert clips, "no LAFAN1 clips found (scripts/download_lafan1.sh, LAFAN1_G1_DIR)"
    lib = load_library(clips, bodies=BM_BODIES)
    env = TrackingEnv(lib, EnvConfig(solver_iterations=args.solver_iterations))
    n, T = args.num_envs, args.steps
    env.warp = sim.make_engine(env, [n])
    kinfo = sim.engine_info(env.warp[n])
    print(f"engine physics (NablaEngine {kinfo['version']} @ {kinfo['commit']}, library {kinfo['key']} for "
          f"{kinfo['arch']}): options {env.warp[n].options}", flush=True)
    (cfg["segments"][-1] if params_info is not None else cfg)["engine"] = kinfo
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    tables = env.tables()
    obs_dim = env.hist_dim + env.goal_dim
    print(f"ppo: clips={len(clips)} envs={n} steps={T} arch={args.arch} obs={obs_dim} critic={env.critic_dim}",
          flush=True)

    key = jax.random.PRNGKey(args.seed)
    key, ka, kc = jax.random.split(key, 3)
    if args.arch == "mlp":
        actor = {"net": mlp_init(ka, [obs_dim, *args.hidden, env.nu], out_scale=0.01)}
    else:
        actor = nets.actor_init(ka, env)
    actor["log_std"] = jnp.full(env.nu, np.log(args.init_std), jnp.float32)
    critic = mlp_init(kc, [env.critic_dim, *(args.critic_hidden or args.hidden), 1])
    params = {"actor": actor, "critic": critic}
    opt = optax.chain(optax.clip_by_global_norm(args.max_grad_norm),
                      optax.inject_hyperparams(optax.adam)(learning_rate=args.lr))
    opt_state = opt.init(params)
    zero_norm = lambda d: (jnp.zeros(d), jnp.ones(d), jnp.float32(1e-4))  # noqa: E731
    onorm, cnorm = zero_norm(obs_dim), zero_norm(env.critic_dim)

    # Precision as in the reported runs: the MLP policy and the critic call mlp_apply directly (fp32 under the "highest"
    # default), the encoder-decoder actor goes through nets.actor_apply (TF32).
    def policy_mean(actor, onorm, hist, goal):
        if args.arch == "mlp":
            return mlp_apply(actor["net"], normalize(onorm[:2], jnp.concatenate([hist, goal], -1)))
        return actor_apply({"enc": actor["enc"], "dec": actor["dec"]}, hist, goal)

    def value(critic, cnorm, cob):
        return mlp_apply(critic, normalize(cnorm[:2], cob))[..., 0]

    # Reset sampling over 1 s bins: SuperMimic's (50% uniform over bins, 40% softmax(-25 G_b/G_max), 10% the bins'
    # slow-moving past softmax weight; G_b an EMA of the GAE return from visited states) or BeyondMimic's (bin_q row 2).
    valid = np.asarray(env.valid)
    bins_np = np.arange(env.n_frames) // BIN
    n_bins = int(bins_np.max()) + 1
    bins, valid_j = jnp.asarray(bins_np), jnp.asarray(valid, jnp.float32)
    bin_count = jnp.zeros(n_bins).at[bins].add(valid_j)
    has = bin_count > 0
    bin_q = jnp.stack([jnp.ones(n_bins), has / has.sum(), jnp.zeros(n_bins)])  # row 2: BeyondMimic's failure EMA
    g_max = env.max_reward * (1 - args.gamma ** args.max_episode_steps) / (1 - args.gamma)

    def bin_softmax(stat):
        w = jnp.where(has, jax.nn.softmax(jnp.where(has, -25.0 * stat, -1e9)), 0.0)
        return w / w.sum()

    @jax.jit
    def frame_probs(bin_q):
        per_frame = lambda w: jnp.where(valid_j > 0, w[bins] / jnp.maximum(bin_count[bins], 1), 0.0)  # noqa: E731
        uni = per_frame(has / has.sum())
        if args.reset_sampling == "beyondmimic":  # failure EMA + 0.1 / bins (row 2 of bin_q), uniform within a bin
            w = jnp.where(has, bin_q[2] + 0.1 / has.sum(), 0.0)
            return per_frame(w / w.sum())
        return 0.5 * uni + 0.4 * per_frame(bin_softmax(bin_q[0])) + 0.1 * per_frame(bin_q[1] / bin_q[1].sum())

    def sample_resets(key, probs, tables):
        k1, k2 = jax.random.split(key)
        frames = jax.random.choice(k1, env.n_frames, (n,), p=probs)
        return vmapped(env, tables)["reset"](jax.random.split(k2, n), frames)

    @jax.jit
    def rollout(params, onorm, cnorm, state, probs, key, tables):
        f = vmapped(env, tables)
        actor, critic = params["actor"], params["critic"]
        kr, ks = jax.random.split(key)
        reset_state = sample_resets(kr, probs, tables)  # one batch of reset states per iteration

        def body(s, k):
            k1, k2 = jax.random.split(k)
            hist, goal = f["policy_obs"](s, jax.random.split(k1, n))
            mu = policy_mean(actor, onorm, hist, goal)
            a = mu + jnp.exp(actor["log_std"]) * jax.random.normal(k2, mu.shape)
            cob = f["critic_obs"](s)
            s2 = f["step"](s, a)
            r, terms = f["reward"](s2, a, s.prev_a)
            term = f["terminated"](s2)
            trunc = (f["clip_ended"](s2) | (s2.ep >= args.max_episode_steps)) & ~term
            v_next = value(critic, cnorm, f["critic_obs"](s2))
            r = jnp.where(jnp.isfinite(r), r, 0.0)
            v_next = jnp.where(jnp.isfinite(v_next), v_next, 0.0)
            done = term | trunc
            o = dict(hist=hist, goal=goal, cob=cob, a=a, mu=mu, logp=gauss_logp(a, mu, actor["log_std"]),
                     v=value(critic, cnorm, cob), r=r + args.gamma * v_next * trunc, raw_r=r, done=done,
                     term=term, frame=s2.frame, terms=terms)
            return select_reset(done, reset_state, s2), o

        state, o = jax.lax.scan(body, state, jax.random.split(ks, T))
        v_last = value(critic, cnorm, f["critic_obs"](state))

        def back(c, x):
            adv_next, v_next = c
            r, v, d = x
            delta = r + args.gamma * v_next * (1.0 - d) - v
            adv = delta + args.gamma * args.lam * (1.0 - d) * adv_next
            return (adv, v), adv

        _, adv = jax.lax.scan(back, (jnp.zeros(n), v_last), (o["r"], o["v"], o["done"].astype(jnp.float32)),
                              reverse=True)
        o["adv"], o["ret"] = adv, adv + o["v"]
        return state, o

    @jax.jit
    def update(params, opt_state, onorm, cnorm, o, key):
        m = n * T
        flat = {k: o[k].reshape((m,) + o[k].shape[2:]) for k in ("hist", "goal", "cob", "a", "mu", "logp", "v",
                                                                  "adv", "ret")}
        flat["adv"] = (flat["adv"] - flat["adv"].mean()) / (flat["adv"].std() + 1e-8)
        old_log_std = params["actor"]["log_std"]
        mb = m // args.minibatches

        def loss_fn(p, b):
            mu = policy_mean(p["actor"], onorm, b["hist"], b["goal"])
            log_std = p["actor"]["log_std"]
            ratio = jnp.exp(gauss_logp(b["a"], mu, log_std) - b["logp"])
            surr = -jnp.mean(jnp.minimum(ratio * b["adv"], jnp.clip(ratio, 1 - args.clip, 1 + args.clip) * b["adv"]))
            v = value(p["critic"], cnorm, b["cob"])
            v_clip = b["v"] + jnp.clip(v - b["v"], -args.clip, args.clip)
            vloss = jnp.mean(jnp.maximum((v - b["ret"]) ** 2, (v_clip - b["ret"]) ** 2))
            ent = jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e))
            std, std_old = jnp.exp(log_std), jnp.exp(old_log_std)
            kl = jnp.mean(jnp.sum(log_std - old_log_std + (std_old ** 2 + (b["mu"] - mu) ** 2) / (2 * std ** 2) - 0.5,
                                  -1))
            return surr + args.value_coef * vloss - args.entropy * ent, (surr, vloss, kl)

        def mini(c, idx):
            p, s = c
            b = {k: v[idx] for k, v in flat.items()}
            (loss, (surr, vloss, kl)), g = jax.value_and_grad(loss_fn, has_aux=True)(p, b)
            # rsl_rl's adaptive schedule, from the KL to the rollout policy, before this minibatch's step
            lr = s[1].hyperparams["learning_rate"]
            lr = jnp.where(kl > 2 * args.desired_kl, jnp.maximum(1e-5, lr / 1.5),
                           jnp.where(kl < args.desired_kl / 2, jnp.minimum(1e-2, lr * 1.5), lr))
            s[1].hyperparams["learning_rate"] = lr
            u, s = opt.update(g, s, p)
            return (optax.apply_updates(p, u), s), jnp.stack([loss, surr, vloss, kl])

        perms = jnp.concatenate([jax.random.permutation(k, m)[None] for k in jax.random.split(key, args.epochs)])
        idx = perms.reshape(args.epochs * args.minibatches, mb)
        (params, opt_state), stats = jax.lax.scan(mini, (params, opt_state), idx)
        onorm = norm_update(onorm, jnp.concatenate([flat["hist"], flat["goal"]], -1))
        cnorm = norm_update(cnorm, flat["cob"])
        return params, opt_state, onorm, cnorm, stats.mean(0)

    @jax.jit
    def bin_stats(bin_q, o):
        q = (o["ret"] / g_max).reshape(-1)
        b = bins[o["frame"].reshape(-1)]
        s = jnp.zeros(n_bins).at[b].add(q)
        c = jnp.zeros(n_bins).at[b].add(1.0)
        stat = jnp.where(c > 0, 0.9 * bin_q[0] + 0.1 * s / jnp.maximum(c, 1.0), bin_q[0])
        # BeyondMimic: per step, fail_b <- 0.001 * (failures in bin b this step) + 0.999 * fail_b (here the T steps of
        # the iteration at once; the order within it shifts weights by under 2.4%)
        fails = jnp.zeros(n_bins).at[bins[o["frame"].reshape(-1)]].add(o["term"].reshape(-1).astype(jnp.float32))
        fail = 0.999 ** T * bin_q[2] + 0.001 * fails
        return jnp.stack([stat, 0.995 * bin_q[1] + 0.005 * bin_softmax(stat), fail])

    start_it, env_steps, wall0 = 0, 0, 0.0
    if args.resume:
        with open(args.resume, "rb") as f:
            ck = pickle.load(f)
        as_j = lambda t: jax.tree_util.tree_map(jnp.asarray, t)  # noqa: E731
        params, opt_state = as_j(ck["params"]), as_j(ck["opt_state"])
        onorm, cnorm, bin_q = as_j(ck["onorm"]), as_j(ck["cnorm_full"]), as_j(ck["bin_q"])
        start_it, env_steps, wall0 = int(ck["iter"]), int(ck["env_steps"]), float(ck["wall_s"])
        print(f"resumed from {args.resume} at iteration {start_it}", flush=True)

    key, kr = jax.random.split(key)
    probs = frame_probs(bin_q)
    state = jax.jit(sample_resets)(kr, probs, tables)
    # the live reset select (as train.py's): leaves no step writes and every reset state shares are skipped
    e_b = env.bind(tables)

    def probe(s_r, k):
        k1, k2 = jax.random.split(k)
        q = s_r.dx.qpos + 0.01 * jax.random.normal(k1, s_r.dx.qpos.shape)
        v = s_r.dx.qvel + 0.1 * jax.random.normal(k2, s_r.dx.qvel.shape)
        return jax.vmap(e_b.finish_physics)(s_r, jnp.zeros((n, env.nu)), q, v, s_r.dx.qacc_warmstart)

    s_a = jax.jit(probe)(state, jax.random.fold_in(kr, 21))
    sim.STATE_SKIP = sim.skip_mask(state, s_a)
    print(f"state select: {sum(jax.tree_util.tree_leaves(sim.STATE_SKIP))} of "
          f"{len(jax.tree_util.tree_leaves(state))} leaves skipped", flush=True)
    del s_a
    state = state._replace(ep=jax.random.randint(jax.random.fold_in(kr, 1), (n,), 0,
                                                 args.max_episode_steps).astype(state.ep.dtype))

    def save(path, it, wall_s):
        # actor in the format nets.actor_apply / evaluate.py understand (mean action only)
        a = params["actor"]
        ev_actor = ({"net": a["net"], "obs_norm": tuple(onorm[:2])} if args.arch == "mlp"
                    else {"enc": a["enc"], "dec": a["dec"]})
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(jax.device_get(dict(actor=ev_actor, critic=params["critic"], cnorm=tuple(cnorm[:2]),
                                            params=params, opt_state=opt_state, onorm=onorm, cnorm_full=cnorm,
                                            bin_q=bin_q, iter=it, wall_s=wall_s, env_steps=env_steps)), f)
        tmp.rename(path)

    t_start = time.time() - wall0
    next_ckpt = wall0 if wall0 > 0 else 0.0
    t0 = time.time()
    it = start_it
    for it in range(start_it, args.iters):
        wall = time.time() - t_start
        if wall >= next_ckpt:
            save(out / "ckpt" / f"it_{it:06d}.pkl", it, wall)
            dense = next_ckpt < 3600 * args.ckpt_dense[0]
            next_ckpt += 60 * (args.ckpt_dense[1] if dense else args.ckpt_minutes)
        if wall >= 3600 * args.hours:
            break
        key, k1, k2 = jax.random.split(key, 3)
        t1 = time.time()
        state, o = rollout(params, onorm, cnorm, state, probs, k1, tables)
        params, opt_state, onorm, cnorm, st = update(params, opt_state, onorm, cnorm, o, k2)
        bin_q = bin_stats(bin_q, o)
        probs = frame_probs(bin_q)
        jax.block_until_ready(params)
        env_steps += n * T
        if it == start_it:
            print(f"first iteration (compile) {time.time() - t0:.0f}s", flush=True)
        if it % 10 == 0:
            rec = {"iter": it, "wall_s": time.time() - t_start, "iter_s": time.time() - t1, "env_steps": env_steps,
                   "loss": float(st[0]), "surr": float(st[1]), "value_loss": float(st[2]), "kl": float(st[3]),
                   "lr": float(opt_state[1].hyperparams["learning_rate"]),
                   "std": float(jnp.exp(params["actor"]["log_std"]).mean()),
                   "rew_mean": float(o["raw_r"].mean() / env.max_reward), "term_rate": float(o["term"].mean()),
                   **{f"rew/{k}": float(v.mean() / env.max_reward) for k, v in o["terms"].items()}}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
    # the final checkpoint: after the last iteration of the loop it carries that iteration's label (when the loop ends
    # on --iters N, it_{N-1} holds N updates)
    save(out / "ckpt" / f"it_{it:06d}.pkl", it, time.time() - t_start)
    (out / "DONE").write_text(str(it))


if __name__ == "__main__":
    main()
