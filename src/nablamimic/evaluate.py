"""Nominal LAFAN1 evaluation of train.py / ppo.py checkpoints.

Every clip is played once from its first frame with the deterministic mean action, no domain randomization, no
observation or reset noise, MuJoCo Warp physics (forward only) with --solver-iterations Newton iterations (10, as
training) and fp32 matmuls, in the run's action space with the action-rate term on the commanded setpoints (the same
reward for absolute and residual actions). Per checkpoint: full_norm_return (the reward summed over each whole clip, a
fall forfeiting the rest, divided by its maximum, averaged over clips), alive_frac, failed_frac, return_per_alive_step,
fall causes (height / tilt test of TrackingEnv.terminated), every reward term per alive step and per 10 s of clip time,
joint errors and per-clip values. Results are merged into the --out JSON under it<N>/nominal; existing keys are skipped.

  python -m nablamimic.eval_plan RUN                       # the checkpoints of the protocol
  python -m nablamimic.evaluate --run RUN --iters 250 500 ... --out RUN/eval_nominal.json
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import jax_cache, sim
from .env import BM_BODIES, W, EnvConfig, TrackingEnv, _gravity_b
from .motion import all_clip_names, load_library
from .nets import actor_apply
from .sim import vmapped

BIN = 500  # steps per time bin (10 s)
TERMS = tuple(W)


def make_rollout(env, steps):
    nb = steps // BIN + 1

    @jax.jit
    def run(actor, frames, key, tables):
        f = vmapped(env, tables)
        e = env.bind(tables)
        n = frames.shape[0]
        k0, k1 = jax.random.split(key)
        state = f["reset"](jax.random.split(k0, n), frames)

        def body(c, xs):
            t, k = xs
            state, alive, fall_t, cause, tsum, tbin, cbin, jerr, jabs = c
            hist, goal = f["policy_obs"](state, jax.random.split(k, n))
            a = actor_apply(actor, hist, goal)
            s2 = f["step"](state, a)
            _, terms = f["reward"](s2, a, state.prev_a, prev=state)
            tv = jnp.stack([terms[nm] for nm in TERMS], -1)  # (n, T)
            rq = e.ref_qpos[s2.frame]
            q = s2.dx.qpos
            dz = jnp.abs(q[:, 2] - rq[:, 2])
            dg = jnp.abs(jax.vmap(_gravity_b)(q[:, 3:7])[:, 2] - jax.vmap(_gravity_b)(rq[:, 3:7])[:, 2])
            term = f["terminated"](s2)
            am = alive[:, None]
            tsum = tsum + jnp.where(am, tv, 0.0)
            b = t // BIN
            tbin = tbin.at[:, b].add(jnp.where(am, tv, 0.0))
            cbin = cbin.at[:, b].add(alive)
            dj = q[:, 7:] - rq[:, 7:]
            jerr = jerr + jnp.where(am, dj, 0.0)
            jabs = jabs + jnp.where(am, jnp.abs(dj), 0.0)
            newfall = alive & term
            fall_t = jnp.where(newfall, t + 1, fall_t)
            cause = jnp.where(newfall, (dz > env.cfg.term_height) * 1 + (dg > env.cfg.term_tilt) * 2, cause)
            alive = alive & ~term & ~f["clip_ended"](s2)
            return (s2, alive, fall_t, cause, tsum, tbin, cbin, jerr, jabs), None

        z = jnp.zeros
        init = (state, jnp.ones(n, bool), jnp.full(n, -1), jnp.zeros(n, jnp.int32), z((n, len(TERMS))),
                z((n, nb, len(TERMS))), z((n, nb)), z((n, env.nu)), z((n, env.nu)))
        (_, _, fall_t, cause, tsum, tbin, cbin, jerr, jabs), _ = jax.lax.scan(
            body, init, (jnp.arange(steps), jax.random.split(k1, steps)))
        return dict(fall_t=fall_t, cause=cause, tsum=tsum, tbin=tbin, cbin=cbin, jerr=jerr, jabs=jabs)

    return run


def summarize(res, env, names):
    full_len = env.lengths - 1
    fall_t = np.asarray(res["fall_t"])
    nal = np.where(fall_t > 0, fall_t, full_len)
    tsum = np.asarray(res["tsum"])
    tmax = np.array([abs(W[t]) * 0.02 for t in TERMS])
    out = {
        "full_norm_return": float((tsum.sum(1) / (full_len * env.max_reward)).mean()),
        "alive_frac": float((nal / full_len).mean()),
        "failed_frac": float((fall_t > 0).mean()),
        "fall_cause_counts": {c: int(((fall_t > 0) & (np.asarray(res["cause"]) == v)).sum())
                              for c, v in (("height", 1), ("tilt", 2), ("both", 3), ("nonfinite", 0))},
        # per alive step, each tracking term as a fraction of its own maximum; penalties per step in reward units
        "term_frac_alive": {t: float(tsum[:, i].sum() / nal.sum() / tmax[i]) for i, t in enumerate(TERMS)},
        "return_per_alive_step": float(tsum.sum() / nal.sum() / env.max_reward),
    }
    cb = np.asarray(res["cbin"]).sum(0)
    tb = np.asarray(res["tbin"]).sum(0)
    keep = cb > 0
    out["bins_s"] = (np.nonzero(keep)[0] * BIN * 0.02).tolist()
    out["bins_alive"] = cb[keep].tolist()
    out["term_frac_by_bin"] = {t: (tb[keep, i] / cb[keep] / tmax[i]).round(4).tolist() for i, t in enumerate(TERMS)}
    nal_s = np.maximum(nal, 1)[:, None]
    out["joint_signed_err"] = (np.asarray(res["jerr"]) / nal_s).mean(0).round(4).tolist()
    out["joint_abs_err"] = (np.asarray(res["jabs"]) / nal_s).mean(0).round(4).tolist()
    track = [i for i, t in enumerate(TERMS) if W[t] > 0]
    pct = {TERMS[i]: tsum[:, i] / np.maximum(nal, 1) / tmax[i] for i in track}
    out["per_clip"] = [{"clip": names[i], "alive_frac": round(float(nal[i] / full_len[i]), 4),
                        "terms": {t: round(float(v[i]), 4) for t, v in pct.items()},
                        "fall_s": round(float(fall_t[i]) * 0.02, 2) if fall_t[i] > 0 else None,
                        "cause": int(res["cause"][i]) if fall_t[i] > 0 else None,
                        "return": round(float(tsum[i].sum() / (full_len[i] * env.max_reward)), 4)}
                       for i in range(len(names))]
    return out


def build_parser():
    ap = argparse.ArgumentParser(prog="python -m nablamimic.evaluate",
                                 description="Nominal LAFAN1 evaluation of a run's checkpoints.")
    ap.add_argument("--run", required=True, help="run directory (config.json, ckpt/)")
    ap.add_argument("--iters", type=int, nargs="+", required=True, help="checkpoint iterations")
    ap.add_argument("--solver-iterations", type=int, default=10, help="Newton iterations of the evaluation physics")
    ap.add_argument("--out", required=True, help="JSON file the results are merged into")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    jax_cache.enable()
    jax.config.update("jax_default_matmul_precision", "highest")
    run = Path(args.run)
    cfg = json.loads((run / "config.json").read_text())
    names = all_clip_names() if cfg["clips"] == ["all"] else cfg["clips"]
    lib = load_library(names, bodies=BM_BODIES)
    env = TrackingEnv(lib, EnvConfig(solver_iterations=args.solver_iterations, dr=False, obs_noise=False,
                                     reset_noise=False, residual_action=bool(cfg.get("residual_action", False)),
                                     action_rate_space="setpoint"))
    nclip = len(env.starts)
    env.warp = sim.make_warp(env, [nclip], train=False)
    steps = int(env.lengths.max() - 1)
    starts = env.starts.astype(np.int32)
    fn = make_rollout(env, steps)
    out_path = Path(args.out)
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    out.setdefault("clips", names)
    out.setdefault("terms", list(TERMS))
    out.setdefault("run", str(run))
    for it in sorted(set(args.iters)):
        key = f"it{it}/nominal"
        if key in out:
            continue
        with open(run / "ckpt" / f"it_{it:06d}.pkl", "rb") as f:
            ck = pickle.load(f)
        actor = jax.tree_util.tree_map(jnp.asarray, ck["actor"])
        t0 = time.time()
        res = jax.device_get(fn(actor, jnp.asarray(starts), jax.random.PRNGKey(1), env.tables()))
        rec = summarize(res, env, names)
        rec.update(iter=int(ck["iter"]), wall_s=float(ck["wall_s"]), seconds=round(time.time() - t0, 1),
                   gpu=jax.devices()[0].device_kind, solver_iterations=args.solver_iterations,
                   action_rate_space="setpoint")
        out[key] = rec
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out))
        print(key, json.dumps({k: rec[k] for k in ("full_norm_return", "alive_frac", "failed_frac",
                                                   "return_per_alive_step", "fall_cause_counts", "seconds")}),
              flush=True)


if __name__ == "__main__":
    main()
