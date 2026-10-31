"""SuperMimic's filtered BONES-SEED training-set benchmark (its Appendix F) for train.py checkpoints.

Each motion plays once from its first frame with deterministic physics, no domain randomization and no observation
noise (MuJoCo Warp, --solver-iterations); a motion fails when the rollout triggers the training termination (height or
tilt, TrackingEnv.terminated). Per motion: the normalized return (reward summed over the motion, a fall forfeiting
the rest, divided by its maximum; the action-rate term on the commanded setpoints, as in evaluate.py) and whether it
failed; the benchmark reports their mean and the failed count.

The ~129k motions are sorted by length and run in batches of --batch worlds, each batch as one fixed-capacity
MotionLibrary built when it runs (same table shapes, so nothing recompiles) stepped in chunks of --chunk steps until
every motion in it has ended. --subset N scores a fixed random sample (seed 0) instead of every motion. --lafan1 runs
the 40 LAFAN1 clips through the same path (to check it against evaluate.py's nominal score).

Summaries are merged into the --out JSON (under a lock, so evaluation parts can run concurrently) and the per-motion
results go to <out>_it<N>_<tag>.npz beside it. The dataset is licensed for non-commercial use: keep both out of any
public place.

  python -m nablamimic.bones.benchmark --run RUN --lib DATA/lib50 --iters 8000 16000 --subset 10000 --out RUN/bones.json
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .. import jax_cache, sim
from ..env import BM_BODIES, EnvConfig, TrackingEnv
from ..motion import MotionLibrary, all_clip_names, load_library
from ..motion_lib import KEYS, ShardedLibrary
from ..nets import actor_apply
from ..sim import vmapped


def batch_plan(clips, batch):
    """clips: [(name, frames dict)] -> (groups of clips, longest motions first, the last one padded with repeats of its
    shortest motion; the common table capacity in frames). The libraries are built one at a time by batch_library:
    holding all of them pads every batch to the first one's capacity (~250 GB host memory for the full benchmark)."""
    clips = sorted(clips, key=lambda c: -len(c[1]["qpos"]))
    groups = [clips[i:i + batch] for i in range(0, len(clips), batch)]
    cap = max(sum(len(c[1]["qpos"]) for c in g) + (batch - len(g)) * len(g[-1][1]["qpos"]) for g in groups)
    return groups, cap


def batch_library(g, batch, cap):
    """One group -> (motions scored, their lengths, MotionLibrary of capacity cap)."""
    n_real = len(g)
    g = g + [g[-1]] * (batch - len(g))  # the last batch: repeats of its shortest motion fill the worlds (not scored)
    starts = np.cumsum([0] + [len(c[1]["qpos"]) for c in g])[:-1]
    off = int(starts[-1] + len(g[-1][1]["qpos"]))
    arr = {k: np.concatenate([c[1][k] for c in g] + [np.repeat(g[-1][1][k][-1:], cap - off, 0)]).astype(np.float32)
           for k in KEYS}
    last = np.empty(cap, np.int64)
    for s0, c in zip(starts, g):
        last[s0:s0 + len(c[1]["qpos"])] = s0 + len(c[1]["qpos"]) - 1
    last[off:] = np.arange(off, cap)
    lib = MotionLibrary(qpos=arr["qpos"], qvel=arr["qvel"], body_pos=arr["body_pos"], last=last,
                        names=[c[0] for c in g], starts=starts, fps=50.0, bodies=BM_BODIES,
                        body_quat=arr["body_quat"], body_lin_vel=arr["body_lin_vel"], body_ang_vel=arr["body_ang_vel"])
    return n_real, np.array([len(c[1]["qpos"]) for c in g[:n_real]]), lib


def save_merged(out_path, key, rec):
    """Add one summary to the JSON under an exclusive lock, merged with what is on disk now: the evaluation parts of a
    run may run concurrently and share the file."""
    import fcntl
    import os
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path.with_suffix(".lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        out = json.loads(out_path.read_text()) if out_path.exists() else {}
        out[key] = rec
        tmp = out_path.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(out, indent=1))
        os.replace(tmp, out_path)


def bones_clips(lib_dir, subset):
    """[(name, frames)] of the filtered library (or a fixed random sample of it), read shard by shard."""
    slib = ShardedLibrary(lib_dir, filtered=True)
    names = sorted(slib.clips)
    if subset:
        names = sorted(np.random.RandomState(0).choice(names, subset, replace=False).tolist())
    want = set(names)
    clips = []
    for s in slib.shards:
        z = np.load(Path(lib_dir) / f"{s}.npz")
        arr = {k: z[k] for k in KEYS}
        for i, nm in enumerate(z["names"]):
            nm = str(nm)
            if nm in want:
                a, n = int(z["starts"][i]), int(z["lengths"][i])
                clips.append((nm, {k: arr[k][a:a + n] for k in KEYS}))
    return clips


def lafan1_clips():
    lib = load_library(all_clip_names(), bodies=BM_BODIES)
    ends = np.append(lib.starts[1:], len(lib.qpos))
    return [(nm, {k: getattr(lib, k)[s:e] for k in KEYS}) for nm, s, e in zip(lib.names, lib.starts, ends)]


def make_chunk(env, chunk):
    @jax.jit
    def run(actor, carry, key, tables):
        f = vmapped(env, tables)
        n = carry[0].frame.shape[0]

        def body(c, k):
            state, alive, fall_t, ret, t = c
            hist, goal = f["policy_obs"](state, jax.random.split(k, n))
            a = actor_apply(actor, hist, goal)
            s2 = f["step"](state, a)
            r, _ = f["reward"](s2, a, state.prev_a, prev=state)
            term = f["terminated"](s2)
            ret = ret + jnp.where(alive, r, 0.0)
            fall_t = jnp.where(alive & term, t + 1, fall_t)
            alive = alive & ~term & ~f["clip_ended"](s2)
            return (s2, alive, fall_t, ret, t + 1), None

        carry, _ = jax.lax.scan(body, carry, jax.random.split(key, chunk))
        return carry

    return run


def build_parser():
    ap = argparse.ArgumentParser(prog="python -m nablamimic.bones.benchmark",
                                 description="BONES-SEED training-set benchmark of a run's checkpoints.")
    ap.add_argument("--run", required=True, help="run directory (ckpt/)")
    ap.add_argument("--iters", type=int, nargs="+", required=True)
    ap.add_argument("--lib", default=None, help="the converted library (nablamimic.bones.convert --out)")
    ap.add_argument("--subset", type=int, default=0, help="score a fixed random sample of this many motions (0: all)")
    ap.add_argument("--lafan1", action="store_true", help="the 40 LAFAN1 clips instead (to check the path)")
    ap.add_argument("--batch", type=int, default=4096, help="worlds per batch (2048 fit a 48 GB GPU)")
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--solver-iterations", type=int, default=10)
    ap.add_argument("--out", required=True)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    assert args.lafan1 or args.lib, "--lib DIR (or --lafan1)"
    jax_cache.enable()
    jax.config.update("jax_default_matmul_precision", "highest")
    run = Path(args.run)
    t0 = time.time()
    clips = lafan1_clips() if args.lafan1 else bones_clips(args.lib, args.subset)
    batch = min(args.batch, len(clips))
    groups, cap = batch_plan(clips, batch)
    print(f"{len(clips)} motions, {sum(len(c[1]['qpos']) for c in clips)} frames, {len(groups)} batches of {batch} "
          f"(capacity {cap} frames), loaded in {time.time() - t0:.0f} s", flush=True)
    del clips
    cfgp = run / "config.json"
    resid = bool(json.loads(cfgp.read_text()).get("residual_action", False)) if cfgp.exists() else False
    env = TrackingEnv(batch_library(groups[0], batch, cap)[2],
                      EnvConfig(solver_iterations=args.solver_iterations, dr=False, obs_noise=False, reset_noise=False,
                                residual_action=resid, action_rate_space="setpoint"))
    env.warp = sim.make_warp(env, [batch], train=False)
    chunk_fn = make_chunk(env, args.chunk)
    reset = jax.jit(lambda keys, frames, tables: vmapped(env, tables)["reset"](keys, frames))
    out_path = Path(args.out)
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    tag = "lafan1" if args.lafan1 else (f"subset{args.subset}" if args.subset else "full")
    for it in args.iters:
        key = f"it{it}/{tag}"
        if key in out:
            continue
        with open(run / "ckpt" / f"it_{it:06d}.pkl", "rb") as fh:
            ck = pickle.load(fh)
        actor = jax.tree_util.tree_map(jnp.asarray, ck["actor"])
        t1, names, rets, falls, lens = time.time(), [], [], [], []
        for g in groups:
            n_real, lengths, lib = batch_library(g, batch, cap)
            env.set_library(lib)
            tables = env.tables()
            state = reset(jax.random.split(jax.random.PRNGKey(0), batch), jnp.asarray(lib.starts, jnp.int32), tables)
            carry = (state, jnp.ones(batch, bool), jnp.full(batch, -1), jnp.zeros(batch), jnp.zeros((), jnp.int32))
            steps = int(lengths.max() - 1)
            for c in range((steps + args.chunk - 1) // args.chunk):
                carry = chunk_fn(actor, carry, jax.random.PRNGKey(c), tables)
                if not bool(jnp.any(carry[1][:n_real])):
                    break
            _, _, fall_t, ret, _ = jax.device_get(carry)
            names += lib.names[:n_real]
            rets.append(ret[:n_real] / ((lengths - 1) * env.max_reward))
            falls.append(fall_t[:n_real] > 0)
            lens.append(lengths)
        rets, falls, lens = np.concatenate(rets), np.concatenate(falls), np.concatenate(lens)
        rec = dict(iter=int(ck["iter"]), wall_s=float(ck["wall_s"]), motions=int(len(rets)),
                   norm_return=float(rets.mean()), failed=int(falls.sum()), failed_frac=float(falls.mean()),
                   frame_weighted_return=float((rets * lens).sum() / lens.sum()), seconds=round(time.time() - t1, 1),
                   gpu=jax.devices()[0].device_kind, solver_iterations=args.solver_iterations,
                   action_rate_space="setpoint")
        save_merged(out_path, key, rec)
        np.savez_compressed(out_path.with_suffix("").as_posix() + f"_it{it}_{tag}.npz", names=np.asarray(names),
                            norm_return=rets.astype(np.float32), failed=falls, length=lens.astype(np.int32))
        print(key, json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
