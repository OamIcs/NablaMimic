"""Training of NablaMimic and of the SHAC baseline: backpropagation through the simulator over short windows.

NablaMimic (configs/ours.args, ours_mjw.args, bones.args with residual actions; configs/reported/ours_absolute*.args
with absolute actions), every update:
  - a data collector (--collect-envs envs running the policy with N(0, --collect-noise^2) action noise and SuperMimic-style
    reset sampling) takes 1/--utd steps on average and writes its states into a FIFO replay of --replay-steps rows;
  - --num-envs windows of --horizon steps start from states drawn uniformly from the filled replay and run through the
    differentiable simulator (NablaEngine or MuJoCo Warp) with N(0, --act-noise^2) action noise; the objective is
    the discounted return over the window, bootstrapped with the critic at the window end and at time-outs / clip ends
    (not at terminations), with a smooth termination penalty in the reward (--term-penalty) and nothing counted after an
    episode end inside the window (--mask-after-done); each step's action cotangent dJ/da_t is clipped per sample to
    --act-cot-clip x the batch median in the backward;
  - PODS-style action targets (Zamora Mora et al. 2021): a_t + eta dJ/da_t, each sample's gradient clipped again to
    the median, eta set so the RMS joint-setpoint step is DELTA rad (--action-target M K DELTA, annealed to
    --action-target-delta-floor x DELTA with the LR schedule); M regression minibatch steps of the actor's own Adam
    (--action-target-lr) on the last K windows;
  - critic: TD(lambda) targets on the same windows, fit on the state each transition starts from (--critic-pre-state),
    --critic-epochs x --critic-minibatches Adam steps.
SHAC (Xu et al. 2022, arXiv:2204.07137; configs/shac*.args): windows continued across updates (no replay), one Adam
step on the actor's window gradient, the critic on the post-transition state, actor-input normalization; with its
published update rule and gradient guard (--shac-strict): global-norm clipping only and a stop at SHAC's own
exploding-gradient guard.
Actions: joint setpoints around the default pose (absolute), or with --residual-action around the next reference
frame's joint angles (configs/ours.args); the reward's action-rate term is on the commanded setpoints
(--action-rate-space setpoint, so both action spaces have the same reward), or on the raw policy outputs (raw).
Ablations: without --action-target the actor takes one Adam step per update on its own window gradient, with
--act-cot-clip still clipping each step's action cotangent in the backward (configs/reported/ablations/no_pods.args);
without --act-cot-clip the action targets come from unclipped gradients (configs/reported/ablations/no_clip.args).

Actor: SuperMimic's goal encoder (2048, 1024, 512, 256) -> z (32, tanh) and decoder (2048, 1024, 512, 256) on
[z, history]; Adam(0.9, 0.95) behind global-norm 1.0 and element-wise 0.1 gradient clipping (global-norm only with
--shac-strict); critic MLP (256, 128). One GPU per run. Outputs in --out: config.json, log.jsonl (every 10 updates),
ckpt/it_XXXXXX.pkl (every --ckpt-every updates), params.pkl (every 500 updates), DONE, and DIVERGED (non-finite loss)
or HALTED (--shac-strict guard).

  python -m nablamimic.train --config configs/ours.args --out runs/ours_s0 [--seed 1] [--hours 3]
  python -m nablamimic.train --config configs/ours.args --out runs/ours_s0 --resume latest
"""

import argparse
import gc
import json
import math
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import jax_cache, nets, sim
from .env import BM_BODIES, GOAL_ANCHOR, EnvConfig, TrackingEnv
from .motion import all_clip_names, load_library
from .nets import actor_apply, actor_decode, actor_encode, actor_init, clip_rows_to_median, critic_apply, normalize
from .params import git_state, latest_checkpoint, resolve_params
from .sim import BIN, select_reset, vmapped

# Fixed settings of this release (recorded in config.json; the evaluators of the same names read them).
FIXED = dict(robot="mjlab", obs_mode="anchor_ori", contacts="all", reward_frame="world", hist_len=10,
             state_select="live", matmul_precision="nets_tf32", actor_hidden=list(nets.ACTOR_HIDDEN),
             actor_latent=nets.ACTOR_LATENT, critic_hidden=list(nets.CRITIC_HIDDEN))
AT_BATCH, AT_EPS = 16384, 1e-8  # samples per action-target regression minibatch, epsilon of its Adam
SHAC_GUARD = 1e6  # --shac-strict: SHAC's exploding-gradient guard on the pre-clip actor gradient norm


# ------------------------------------------------------------------ rollout
def termination_penalty(env, f, s2, term_penalty):
    """-W*dt*[(relu(dz-DZ0)/(h-DZ0))^2 + (relu(dtilt-DT0)/(t-DT0))^2] per env (see --term-penalty)."""
    w, dz0, dg0 = term_penalty
    hf = f["hazard_features"](s2)  # [h - dz, rate, t - dtilt, rate]
    h, tt = env.cfg.term_height, env.cfg.term_tilt
    return -w * 0.02 * ((jnp.maximum(h - hf[:, 0] - dz0, 0.0) / (h - dz0)) ** 2
                        + (jnp.maximum(tt - hf[:, 2] - dg0, 0.0) / (tt - dg0)) ** 2)


def make_rollout(env, horizon, gamma, max_ep, remat, act_clip=None, act_noise=0.0, mask_after_done=False,
                 term_penalty=None, record_pre=False, record_obs=False, enc_hoist=False, remat_policy=None,
                 v_bound=None, clip_end_pre=False):
    """The window objective over `horizon` steps; returns -J (mean over envs) and (final state, per-step records).

    act_noise=sigma executes a = pi(o) + sigma * N(0, I) (action units), so the objective is the return of the noisy
    policy; the pathwise gradient still flows through pi. act_clip=kappa clips, at every step, each env's action
    cotangent dJ/da to kappa x the batch median (per-sample clipping; the forward is unchanged); differentiating with
    respect to probe=zeros(2) yields the clipping statistics summed over the steps (nets.clip_rows_to_median), and
    with respect to delta=zeros((H, n, nu)) the per-step action gradients dJ/da_t. mask_after_done: after an episode end
    (termination, clip end or time limit, bootstrapped as usual) the rest of the window adds nothing to J; the env still
    restarts from reset_state, so the critic keeps training on those steps. enc_hoist (actors with the anchor in the
    decoder): the motion encoder's outputs for the whole window in two batched calls before the scan (the frames along
    the clip from the start state and from the reset state) instead of one call per step; the same policy. remat:
    jax.checkpoint of each step with remat_policy (the engine's saves only its step tapes). v_bound: the critic's
    values clipped to [-v_bound, v_bound] where they bootstrap (and so in the critic's targets). clip_end_pre: an
    episode end at a clip end bootstraps with V of the transition's start state (a fitted critic input) instead of V of
    the clip's last frame (goal window frozen at it, never a critic training input)."""
    def rollout(actor, critic, cnorm, state, reset_state, key, tables, probe=None, delta=None):
        f = vmapped(env, tables)
        v_pobs, v_cobs, v_step, v_rew = f["policy_obs"], f["critic_obs"], f["step"], f["reward"]
        v_term, v_end = f["terminated"], f["clip_ended"]
        n = state.frame.shape[0]
        hoist = enc_hoist and "anchor_dec" in actor
        if hoist:
            assert "obs_norm" not in actor, "--enc-hoist: not with --actor-obs-norm"
            k_anc = actor["anchor_dec"].shape[-1]
            e_ref = env.bind(tables)
            v_goal_ref = jax.vmap(e_ref.goal_ref)

            def enc_along(f0):  # (n,) start frames -> (horizon, n, latent): the encoder along the clip, one batched call
                fr = jnp.minimum(f0[None, :] + jnp.arange(horizon)[:, None], e_ref.ref_last[f0][None, :])
                return actor_encode(actor, v_goal_ref(fr.reshape(-1))).reshape(horizon, n, -1)

            z_cont, z_rst = enc_along(state.frame), enc_along(reset_state.frame)

        def body(c, xs):
            if hoist:  # z_t from the continuation table until the env's first reset, then from the reset table
                xs, zc = xs
                c, jr = c[:-1], c[-1]
            k, d = xs if delta is not None else (xs, None)
            state, acc, gam, total, surv = c
            hist, goal = v_pobs(state, jax.random.split(k, n))
            if hoist:
                zt = jnp.where((jr >= 0)[:, None], z_rst[jnp.maximum(jr, 0), jnp.arange(n)], zc)
                a = actor_decode(actor, zt, hist, goal[..., -k_anc:])
            else:
                a = actor_apply(actor, hist, goal)
            a_pi = a
            if act_noise > 0:
                a = a + act_noise * jax.random.normal(jax.random.fold_in(k, 7), a.shape)
            if act_clip is not None:
                a = clip_rows_to_median(a, act_clip, probe)
            if d is not None:  # zero offset whose gradient is dJ/da_t (unclipped)
                a = a + d
            s2 = v_step(state, a)
            r, terms = v_rew(s2, a, state.prev_a, prev=state)
            if term_penalty is not None:  # smooth penalty on the termination quantities (training objective only)
                pen = termination_penalty(env, f, s2, term_penalty)
                r = r + pen
                terms = {**terms, "term_pen": pen}
            term = v_term(s2)
            cend = v_end(s2)
            done = term | cend | (s2.ep >= max_ep)
            cob = v_cobs(s2)
            v2 = critic_apply(critic, normalize(cnorm, cob))[:, 0]
            if clip_end_pre:
                cob_pre = v_cobs(state)
                v_pre = critic_apply(critic, normalize(cnorm, cob_pre))[:, 0]
                v_gap = jax.lax.stop_gradient(v2 - v_pre)  # logged: the extrapolation step V(s_last) - V(s_t)
                v2 = jnp.where(cend, v_pre, v2)
            r = jnp.where(jnp.isfinite(r), r, 0.0)
            v2 = jnp.where(jnp.isfinite(v2), v2, 0.0)
            if v_bound is not None:
                v2 = jnp.clip(v2, -v_bound, v_bound)
            acc = acc + gam * surv * r  # surv: 1, or the alive mask with mask_after_done
            gam = gam * gamma
            total = total + jnp.where(done, acc + gam * surv * v2 * (1.0 - term), 0.0)
            if mask_after_done:
                surv = jnp.where(done, 0.0, surv)
            acc = jnp.where(done, 0.0, acc)
            gam = jnp.where(done, 1.0, gam)
            s_next = select_reset(done, reset_state, s2)
            out = dict(cobs=jax.lax.stop_gradient(cob), r=jax.lax.stop_gradient(r), term=term, done=done,
                       v2=jax.lax.stop_gradient(v2), frame=s2.frame, terms=jax.lax.stop_gradient(terms))
            if clip_end_pre:
                out["cend"], out["v_gap"] = cend & ~term, v_gap
            if record_pre:  # critic input of the state the transition starts from (the target's state)
                out["cobs_pre"] = jax.lax.stop_gradient(cob_pre if clip_end_pre else v_cobs(state))
            if record_obs:  # policy inputs and output, for --action-target
                out["hist"] = jax.lax.stop_gradient(hist).astype(jnp.bfloat16)
                out["goal"] = jax.lax.stop_gradient(goal).astype(jnp.bfloat16)
                out["a_pi"] = jax.lax.stop_gradient(a_pi)
            if hoist:  # steps since the env's last reset (-1: not reset in this window); a reset restarts at reset_state
                jr = jnp.where(done, 0, jnp.where(jr >= 0, jr + 1, -1))
                return (s_next, acc, gam, total, surv, jr), out
            return (s_next, acc, gam, total, surv), out

        step_fn = jax.checkpoint(body, policy=remat_policy) if remat else body
        init = (state, jnp.zeros(n), jnp.ones(n), jnp.zeros(n), jnp.ones(n))
        keys = jax.random.split(key, horizon)
        xs = (keys, delta) if delta is not None else keys
        if hoist:
            init, xs = init + (jnp.full(n, -1, jnp.int32),), (xs, z_cont)
        (state, acc, gam, total, surv, *_), outs = jax.lax.scan(step_fn, init, xs)
        v_h = critic_apply(critic, normalize(cnorm, v_cobs(state)))[:, 0]
        if v_bound is not None:
            v_h = jnp.clip(v_h, -v_bound, v_bound)
        total = total + acc + gam * surv * jnp.where(jnp.isfinite(v_h), v_h, 0.0)
        return -jnp.mean(total), (jax.lax.stop_gradient(state), outs)

    return rollout


# ------------------------------------------------------------------ main
def build_parser():
    ap = argparse.ArgumentParser(prog="python -m nablamimic.train",
                                 description="Train NablaMimic or the SHAC baseline (see configs/).")
    ap.add_argument("--config", default=None, metavar="FILE",
                    help="take the training arguments from this configuration file (overrides may follow on the first "
                         "start); they are locked into OUT/config.json with the source revision")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", default=None,
                    help="checkpoint to continue from, or 'latest' (the newest OUT/ckpt/it_*.pkl): actor, critic, "
                         "normalizer, optimizer and reset-bin state")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clips", nargs="+", default=["all"], help="LAFAN1 clip names, or all")
    ap.add_argument("--motion-lib", default=None, metavar="DIR",
                    help="train on a sharded reference library (nablamimic.bones.convert output) instead of --clips: "
                         "a working set of --ws-shards shards (each a random sample of the library) is on the GPU and is "
                         "replaced by the next shards, in a fixed random order, every --ws-swap updates; the replay and "
                         "the reset-bin statistics restart with each working set")
    ap.add_argument("--ws-shards", type=int, default=4)
    ap.add_argument("--ws-swap", type=int, default=1000)
    ap.add_argument("--num-envs", type=int, default=4096, help="BPTT windows per update")
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--iters", type=int, default=400000)
    ap.add_argument("--hours", type=float, default=None,
                    help="wall-clock budget (training time, as log.jsonl's wall_s): stop at the first update after it "
                         "with a last checkpoint")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95, help="TD(lambda) of the critic targets")
    ap.add_argument("--lr", type=float, default=2e-4, help="actor (without --action-target-lr) and critic learning rate")
    ap.add_argument("--critic-epochs", type=int, default=5)
    ap.add_argument("--critic-minibatches", type=int, default=4)
    ap.add_argument("--v-bound", action="store_true",
                    help="clip the critic's values where they bootstrap (window ends, episode ends inside a window, "
                         "hence also the critic's lambda-return targets) to +-g_max, the largest possible discounted "
                         "return: stops a value runaway that the window objective can feed (BONES-SEED)")
    ap.add_argument("--clip-end-boot", default="next", choices=["next", "pre"],
                    help="value an episode end at a clip end bootstraps with: next = V of the clip's last frame (goal "
                         "window frozen at it, never a critic training input); pre = V of the transition's start state")
    ap.add_argument("--shac-strict", action="store_true",
                    help="SHAC with its published update rule and gradient guard: the actor update clips only the "
                         "global norm (1.0), no element-wise clip; at the first update whose pre-clip actor gradient "
                         "norm is non-finite or > 1e6 (SHAC's own guard, which raises) the update is not applied, a "
                         "checkpoint and OUT/HALTED are written and training stops; with the direct actor gradient "
                         "only (no --action-target)")
    ap.add_argument("--max-episode-steps", type=int, default=500)
    ap.add_argument("--physics", default="engine", choices=["engine", "warp"],
                    help="engine: NablaEngine, the fused differentiable engine (installed separately); warp: MuJoCo "
                         "Warp with its analytic adjoint (patched, scripts/setup_mujoco_warp.sh)")
    ap.add_argument("--solver-iterations", type=int, default=10, help="Newton iterations of the training physics")
    ap.add_argument("--no-remat", action="store_true",
                    help="keep every step's residuals instead of rematerializing the window in the backward")
    ap.add_argument("--mjw-adj-normal", type=int, default=0, choices=[0, 1, 2, 3],
                    help="warp: derivative of the sphere / capsule self-contact normal v/|v| in the backward: 0 exact, "
                         "1 constant, 2 regularized 1/max(|v|, 1 mm), 3 constant where |v| < 1 mm (mjw_options)")
    ap.add_argument("--mjw-adj-parallel", type=int, default=0, choices=[0, 1],
                    help="warp: near-parallel capsule pairs keep their segment parameters in the backward")
    ap.add_argument("--mjw-ift-solve", type=int, default=0, choices=[0, 1, 2],
                    help="warp: IFT solve of patch 0004 (1: scaled right-hand side, 2: no augmented border)")
    ap.add_argument("--actor-obs-norm", action="store_true",
                    help="normalize the actor input [history, goal] by running statistics (EMA 0.01 per update)")
    ap.add_argument("--anchor-to-decoder", action="store_true",
                    help="the anchor-relative orientation goes to the decoder beside the history; the motion encoder "
                         "sees only the future reference frames")
    ap.add_argument("--enc-hoist", default="off", choices=["off", "on"],
                    help="with --anchor-to-decoder: the encoder outputs of a window in two batched calls before the "
                         "scan instead of once per step (the same policy, faster)")
    ap.add_argument("--action-target", type=float, nargs=3, default=None, metavar=("M", "K", "DELTA"),
                    help="instead of one gradient step per update: from each window take the per-sample action "
                         "gradient dJ/da_t (clipped as --act-cot-clip), form target actions a + eta*g with eta set so "
                         "the RMS setpoint change is DELTA rad, keep the last K windows' (observation, target) in a GPU "
                         "ring buffer and take M regression minibatch steps per update (PODS-style)")
    ap.add_argument("--action-target-lr", type=float, default=None,
                    help="learning rate of the regression steps' Adam (same clipping, betas and schedule as the actor's)")
    ap.add_argument("--action-target-delta-floor", type=float, default=1.0,
                    help="<1 with --lr-decay-steps: the step DELTA follows the same cosine, down to this fraction")
    ap.add_argument("--critic-pre-state", action="store_true",
                    help="fit the critic on the state each transition starts from (whose lambda-return the target is); "
                         "default: the state after the transition")
    ap.add_argument("--term-penalty", type=float, nargs=3, default=None, metavar=("W", "DZ0", "DTILT0"),
                    help="add -W*dt*[(relu(dz-DZ0)/(0.25-DZ0))^2 + (relu(dtilt-DTILT0)/(0.8-DTILT0))^2] to the training "
                         "reward (dz, dtilt: the termination test's height and tilt errors); evaluation unchanged")
    ap.add_argument("--mask-after-done", action="store_true",
                    help="an episode end inside a window ends that env's contribution to the actor objective")
    ap.add_argument("--act-noise", type=float, default=0.0,
                    help="Gaussian action noise (action units) executed in the training windows")
    ap.add_argument("--act-cot-clip", type=float, default=None,
                    help="per-sample clipping of the action cotangent to this many times the batch median, in the "
                         "window's backward at every step; with --action-target also on the targets")
    ap.add_argument("--residual-action", action="store_true",
                    help="joint setpoints = the next reference frame's joint angles + scale * action (instead of the "
                         "default pose q0 + scale * action)")
    ap.add_argument("--action-rate-space", default="setpoint", choices=["setpoint", "raw"],
                    help="the reward's action-rate term: on the commanded unclipped setpoints over the action "
                         "scale (setpoint: with --residual-action the absolute-action term of the same setpoints, the "
                         "first step after a reset against the default pose), or on the raw policy outputs (raw); the "
                         "same for absolute actions. The evaluators use setpoint")
    ap.add_argument("--reset-sampling", default="uniform", choices=["uniform", "supermimic"],
                    help="supermimic: 1 s bins weighted by the discounted (lambda-)return from them, mixing 50%% "
                         "uniform, 40%% softmax(-25 G_b/G_max), 10%% the bins' slow-moving past softmax weight")
    ap.add_argument("--uniform-over", default="frames", choices=["frames", "bins"],
                    help="the uniform part of the reset distribution: over valid frames, or over 1 s bins")
    ap.add_argument("--replay-steps", type=int, default=0,
                    help=">0: SuperMimic-style data collection: --collect-envs envs run the policy with "
                         "N(0, collect-noise^2) action noise and the reset sampling, 1/utd steps per update on average, "
                         "into a FIFO replay of replay-steps x collect-envs states; every update's windows start from "
                         "states drawn uniformly from the filled part of the replay")
    ap.add_argument("--collect-envs", type=int, default=0, help="0: same as --num-envs")
    ap.add_argument("--collect-noise", type=float, default=0.1)
    ap.add_argument("--utd", type=float, default=0.4, help="policy updates per collector step")
    ap.add_argument("--lr-decay-steps", type=int, default=0,
                    help=">0: cosine decay of the learning rates to --lr-decay-to x lr over this many updates, then "
                         "constant")
    ap.add_argument("--lr-decay-to", type=float, default=0.03)
    ap.add_argument("--lr-warmup-steps", type=int, default=0,
                    help=">0 with --lr-decay-steps: linear warmup from 0 over this many updates, then the cosine decay "
                         "(warmup included in --lr-decay-steps)")
    ap.add_argument("--ckpt-every", type=int, default=250, help="numbered checkpoints for the evaluation")
    return ap


def main(argv=None):
    ap = build_parser()
    argv, params_info = resolve_params(sys.argv[1:] if argv is None else argv)
    args = ap.parse_args(argv)
    out = Path(args.out)
    if args.resume == "latest":
        args.resume = latest_checkpoint(out)
    use_at = bool(args.action_target)
    assert not use_at or args.action_target_lr is not None, "--action-target: with --action-target-lr"
    assert not (args.shac_strict and use_at), "--shac-strict: the direct actor gradient (no --action-target)"
    assert args.enc_hoist == "off" or (args.anchor_to_decoder and not args.actor_obs_norm), \
        "--enc-hoist: with --anchor-to-decoder (the encoder input must not depend on the state), not --actor-obs-norm"
    assert not args.motion_lib or args.replay_steps > 0, "--motion-lib restarts the collector's replay with each working set"
    assert args.motion_lib or (args.ws_shards, args.ws_swap) == (4, 1000), "--ws-shards / --ws-swap: with --motion-lib DIR"
    jax_cache.enable()
    nets.set_matmul_precision("nets_tf32")

    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    devices = jax.local_devices()
    nd = len(devices)
    assert nd == 1, "one GPU per run (set CUDA_VISIBLE_DEVICES)"
    commit, dirty, diff = git_state()
    gpu = jax.devices()[0].device_kind
    seg = {"start": time.strftime("%Y-%m-%dT%H:%M:%S"), "commit": commit, "dirty": dirty, "resume": args.resume,
           "gpu": gpu}
    if params_info is not None and params_info["locked"]:  # keep the locked record, add this segment
        cfg = json.loads((out / "config.json").read_text())
        cfg["segments"] = cfg.get("segments", []) + [seg]
    else:
        cfg = dict(vars(args), **FIXED, devices=nd, processes=1, git_commit=commit, git_dirty=dirty, gpu=gpu)
        if params_info is not None:
            cfg.update(argv=params_info["argv"], config_args=params_info["config_args"], config_file=str(args.config),
                       segments=[seg])
    if dirty:
        (out / f"git_diff_{commit[:10]}.patch").write_text(diff)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    log = open(out / "log.jsonl", "a")

    if args.motion_lib:  # working sets of a sharded library (motion_lib), all of the same size
        from .motion_lib import ShardedLibrary
        slib = ShardedLibrary(args.motion_lib)
        ws_order = [slib.shards[i] for i in np.random.RandomState(args.seed).permutation(len(slib.shards))]
        ws_cap = slib.capacity(args.ws_shards)

        def ws_lib(it_):  # the working set in use at update it_
            w = it_ // args.ws_swap
            return slib.working_set([ws_order[(w * args.ws_shards + j) % len(ws_order)] for j in range(args.ws_shards)],
                                    ws_cap)

        lib = ws_lib(0)
        clips = lib.names
        print(f"motion library {args.motion_lib}: {len(slib.clips)} clips in {len(slib.shards)} shards, working set "
              f"{args.ws_shards} shards ({ws_cap} frames), next one every {args.ws_swap} updates", flush=True)
    else:
        clips = all_clip_names() if args.clips == ["all"] else args.clips
        assert clips, "no LAFAN1 clips found (scripts/download_lafan1.sh, LAFAN1_G1_DIR)"
        lib = load_library(clips, bodies=BM_BODIES)
    env = TrackingEnv(lib, EnvConfig(solver_iterations=args.solver_iterations, residual_action=args.residual_action,
                                     action_rate_space=args.action_rate_space))
    n = args.num_envs
    nc = args.collect_envs or n  # replay columns (envs per row)
    sizes = sorted({n, nc} if args.replay_steps > 0 else {n})
    if args.physics == "engine":
        from nablaengine import jax_ffi
        env.warp = sim.make_engine(env, sizes, clamp_kv=0, normal=3, clamp_soft=None)
        w0 = env.warp[n]
        kinfo = sim.engine_info(w0)
        print(f"engine physics (NablaEngine {kinfo['version']} @ {kinfo['commit']}, library {kinfo['key']} for "
              f"{kinfo['arch']}{', built now' if kinfo['built'] else ''}): batch sizes {sizes}, options {w0.options}, "
              f"nconmax {w0.nconmax}/world, njmax {w0.njmax}", flush=True)
        (cfg["segments"][-1] if params_info is not None else cfg)["engine"] = kinfo
        (out / "config.json").write_text(json.dumps(cfg, indent=2))
    else:
        env.warp = sim.make_warp(env, sizes, train=True, adj_normal=args.mjw_adj_normal,
                                 adj_parallel=args.mjw_adj_parallel, ift_solve=args.mjw_ift_solve)
        w0 = env.warp[n]
        print(f"warp physics: batch sizes {sizes}, nconmax {w0.nconmax}/world, njmax {w0.njmax}", flush=True)
    print(f"clips={len(clips)} frames={env.n_frames} envs/update={n} "
          f"goal={env.goal_dim} hist={env.hist_dim} critic={env.critic_dim}", flush=True)

    key = jax.random.PRNGKey(args.seed)
    key, ka, kc = jax.random.split(key, 3)
    k_anchor = GOAL_ANCHOR.stop - GOAL_ANCHOR.start if args.anchor_to_decoder else 0
    actor = actor_init(ka, env, anchor_to_decoder=k_anchor)
    if args.actor_obs_norm:  # placeholder; set from the first reset batch below
        actor["obs_norm"] = (jnp.zeros(env.hist_dim + env.goal_dim), jnp.ones(env.hist_dim + env.goal_dim))
    critic = nets.mlp_init(kc, [env.critic_dim, *nets.CRITIC_HIDDEN, 1])
    key = jax.random.fold_in(key, 0)  # the environment stream (process index 0)

    def sched(per, lr0=args.lr):  # schedules count optimizer steps: `per` of them per update
        if args.lr_decay_steps <= 0:
            return lr0
        if args.lr_warmup_steps > 0:  # standard warmup + cosine decay
            return optax.warmup_cosine_decay_schedule(0.0, lr0, args.lr_warmup_steps * per, args.lr_decay_steps * per,
                                                      end_value=lr0 * args.lr_decay_to)
        return optax.cosine_decay_schedule(lr0, args.lr_decay_steps * per, alpha=args.lr_decay_to)
    opt_a = optax.chain(optax.clip_by_global_norm(1.0), optax.identity() if args.shac_strict else optax.clip(0.1),
                        optax.adam(sched(1), b1=0.9, b2=0.95))
    if use_at:  # the regression's own Adam, M steps per update (st_a is then kept but unused)
        m_reg = int(args.action_target[0])
        # same warmup + decay as the critic, otherwise the policy moves at full rate while the critic warms up
        opt_r = optax.chain(optax.clip_by_global_norm(1.0), optax.clip(0.1),
                            optax.adam(sched(m_reg, args.action_target_lr), b1=0.9, b2=0.95, eps=AT_EPS))
    opt_c = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched(args.critic_epochs * args.critic_minibatches)))
    st_a, st_c = opt_a.init(actor), opt_c.init(critic)
    cnorm = (jnp.zeros(env.critic_dim), jnp.ones(env.critic_dim))

    # Reset sampling over 1 s bins (SuperMimic App. D.1): uniform over valid frames or bins, or with supermimic a mix
    # of uniform, softmax(-25 G_b / G_max) of an EMA of the discounted return G_b from bin b, and its slow-moving past.
    bins = np.arange(env.n_frames) // BIN
    n_bins = int(bins.max()) + 1

    def make_aux():  # sampling tables of the current reference library, and its initial bin statistics
        valid = np.asarray(env.valid)
        aux_ = dict(bins=jnp.asarray(bins), valid=jnp.asarray(valid, jnp.float32))
        aux_["bin_count"] = jnp.zeros(n_bins).at[aux_["bins"]].add(aux_["valid"])
        bin_count0 = np.bincount(bins[valid], minlength=n_bins).astype(np.float32)
        # row 0: per-bin statistic (normalized return, or reward with uniform sampling); row 1: slow-moving past weight
        bq = jnp.stack([jnp.ones(n_bins), jnp.asarray(bin_count0 > 0, jnp.float32) / max((bin_count0 > 0).sum(), 1)])
        return aux_, bq

    tables = env.tables()
    aux, bin_q = make_aux()
    g_max = env.max_reward * (1 - args.gamma ** args.max_episode_steps) / (1 - args.gamma)
    if args.v_bound:
        print(f"value bound: +-{float(g_max):.3f} (g_max)", flush=True)

    def bin_weights(stat, bin_count):
        scale = 1.0 if args.reset_sampling == "supermimic" else jnp.maximum(stat.max(), 1e-6)
        w = jnp.where(bin_count > 0, jax.nn.softmax(jnp.where(bin_count > 0, -25.0 * stat / scale, -1e9)), 0.0)
        return w / w.sum()

    def frame_probs(bin_q, aux):
        bins_j, valid_j, bin_count = aux["bins"], aux["valid"], aux["bin_count"]
        per_frame = lambda w: jnp.where(valid_j > 0, w[bins_j] / jnp.maximum(bin_count[bins_j], 1), 0.0)  # noqa: E731
        if args.uniform_over == "bins":
            uni = per_frame(jnp.where(bin_count > 0, 1.0, 0.0) / jnp.sum(bin_count > 0))
        else:
            uni = valid_j / valid_j.sum()
        if args.reset_sampling == "uniform":
            return uni
        hard = per_frame(bin_weights(bin_q[0], bin_count))
        hist = per_frame(bin_q[1] / bin_q[1].sum())
        return 0.5 * uni + 0.4 * hard + 0.1 * hist

    use_replay = args.replay_steps > 0
    rollout = make_rollout(env, args.horizon, args.gamma, args.max_episode_steps, not args.no_remat,
                           act_clip=args.act_cot_clip, act_noise=args.act_noise, mask_after_done=args.mask_after_done,
                           term_penalty=tuple(args.term_penalty) if args.term_penalty else None,
                           record_pre=args.critic_pre_state, record_obs=use_at, enc_hoist=args.enc_hoist == "on",
                           remat_policy=jax_ffi.remat_policy() if args.physics == "engine" else None,
                           v_bound=float(g_max) if args.v_bound else None, clip_end_pre=args.clip_end_boot == "pre")

    def sample_resets(key, probs, tables, size=None):
        size = size or n
        k1, k2 = jax.random.split(key)
        frames = jax.random.choice(k1, env.n_frames, (size,), p=probs)
        return vmapped(env, tables)["reset"](jax.random.split(k2, size), frames)

    def collect_step(actor, cstate, pool, ptr, probs, key, tables):
        """One step of the data collector: noisy policy, reset sampling, states into the replay."""
        f = vmapped(env, tables)
        k1, k2, k3 = jax.random.split(key, 3)
        hist, goal = f["policy_obs"](cstate, jax.random.split(k1, nc))
        a = actor_apply(actor, hist, goal) + args.collect_noise * jax.random.normal(k2, (nc, env.nu))
        s2 = f["step"](cstate, a)
        term = f["terminated"](s2)
        done = term | f["clip_ended"](s2) | (s2.ep >= args.max_episode_steps)
        s_next = select_reset(done, sample_resets(k3, probs, tables, nc), s2)
        psize = pool["frame"].shape[0]
        pos = (ptr[0] + jnp.arange(nc)) % psize
        row = env.compact(s_next)
        pool = jax.tree_util.tree_map(lambda p, v: p.at[pos].set(v), pool, row)
        ptr = jnp.stack([(ptr[0] + nc) % psize, jnp.minimum(ptr[1] + nc, psize)])
        return s_next, pool, ptr, jax.lax.pmean(jnp.mean(term), "d")

    def action_target_update(actor, buf, gd, outs, key):
        """--action-target: targets from this window's action gradients into the ring buffer, then M regression steps."""
        m_steps, k_win, delta = int(args.action_target[0]), int(args.action_target[1]), args.action_target[2]
        if args.action_target_delta_floor < 1.0 and args.lr_decay_steps > 0:  # cosine-annealed step
            frac = jnp.minimum(buf["it"].astype(jnp.float32) / args.lr_decay_steps, 1.0)
            fl = args.action_target_delta_floor
            delta = delta * (fl + (1.0 - fl) * 0.5 * (1.0 + jnp.cos(jnp.pi * frac)))
        g = -gd  # ascent direction per step and env (H, n, nu); exactly-zero rows are masked out of the objective
        nonfin = jnp.any(~jnp.isfinite(g), -1)
        g = jnp.where(jnp.isfinite(g), g, 0.0)
        nrm = jnp.sqrt(jnp.sum(g ** 2, -1))  # |dJ/da_t|: row norms of the window clip and the per-step profile
        live = nrm > 0
        sc = jnp.ones_like(nrm)
        if args.act_cot_clip is not None:
            nl = jnp.where(live, nrm, jnp.nan)
            thr = args.act_cot_clip * jnp.nanmedian(nl)
            thr = jnp.where(jnp.isfinite(thr), thr, jnp.inf)
            sc = jnp.minimum(1.0, thr / (nrm + 1e-30))
            g = g * sc[..., None]
        cl = live & (sc < 1.0)
        nlive = jnp.maximum(jnp.sum(live), 1)
        med_raw = jnp.nanmedian(jnp.where(live, nrm, jnp.nan))
        # robust to single huge rows (norm sums are not): [share of live rows clipped, mean shrink 1 - scale over live
        # rows, share of live rows above 100 x the median |dJ/da_t|, rows with non-finite entries (zeroed)]
        outs["at_clip"] = jnp.stack([jnp.sum(cl) / nlive, jnp.sum(jnp.where(live, 1.0 - sc, 0.0)) / nlive,
                                     jnp.sum(live & (nrm > 100.0 * med_raw)) / nlive,
                                     jnp.sum(nonfin).astype(g.dtype)])
        if args.horizon % 8 == 0:  # per eighth of the window: median |dJ/da_t| / overall median, share clipped
            lb = live.reshape(8, -1)
            mb = jnp.nanmedian(jnp.where(live, nrm, jnp.nan).reshape(8, -1), 1)
            outs["at_prof"] = jnp.stack([jnp.where(jnp.isfinite(mb), mb, 0.0) / (med_raw + 1e-30),
                                         jnp.sum(cl.reshape(8, -1), 1) / jnp.maximum(jnp.sum(lb, 1), 1)])
        s2 = env.action_scale ** 2  # the step is measured in joint-setpoint radians
        ss = jax.lax.pmean(jnp.sum(jnp.where(live[..., None], g ** 2 * s2, 0.0)), "d")
        cnt = jax.lax.pmean(jnp.maximum(jnp.sum(live), 1).astype(g.dtype) * env.nu, "d")
        eta = delta / jnp.sqrt(ss / cnt + 1e-30)
        tgt = outs.pop("a_pi") + eta * g
        j = buf["count"] % k_win
        buf = {**buf, "hist": buf["hist"].at[j].set(outs.pop("hist")), "goal": buf["goal"].at[j].set(outs.pop("goal")),
               "tgt": buf["tgt"].at[j].set(tgt), "w": buf["w"].at[j].set(live.astype(g.dtype)),
               "count": buf["count"] + 1, "it": buf["it"] + 1}
        nfill = jnp.minimum(buf["count"], k_win)
        w_err = env.action_scale  # regression error in setpoint radians

        def reg(carry, kk):
            p, st = carry
            ka, kb, kc = jax.random.split(kk, 3)
            si = jax.random.randint(ka, (AT_BATCH,), 0, nfill)
            ti = jax.random.randint(kb, (AT_BATCH,), 0, args.horizon)
            ei = jax.random.randint(kc, (AT_BATCH,), 0, n)
            hb, gb = buf["hist"][si, ti, ei].astype(jnp.float32), buf["goal"][si, ti, ei].astype(jnp.float32)
            yb, wb = buf["tgt"][si, ti, ei], buf["w"][si, ti, ei]

            def lf(q):
                err = (actor_apply(q, hb, gb) - yb) * w_err
                return jnp.sum(wb * jnp.mean(err ** 2, -1)) / jnp.maximum(jnp.sum(wb), 1.0)

            lb, gr = jax.value_and_grad(lf)(p)
            gr = jax.lax.pmean(gr, "d")
            u, st = opt_r.update(gr, st, p)
            p = optax.apply_updates(p, u)
            return (p, st), lb

        (actor, st_r), ls = jax.lax.scan(reg, (actor, buf["opt"]), jax.random.split(key, m_steps))
        return actor, {**buf, "opt": st_r}, jax.lax.pmean(jnp.mean(ls), "d")

    def actor_step(actor, st_a, critic, cnorm, state, pool, ptr, probs, key, tables, buf):
        k1, k2 = jax.random.split(key)
        k3, k4 = jax.random.split(jax.random.fold_in(key, 1))
        reset_state = jax.lax.stop_gradient(sample_resets(k1, probs, tables))
        if use_replay:  # ptr = [write position, number of filled entries]
            pick = jax.random.randint(k3, (n,), 0, ptr[1])
            start = jax.vmap(env.bind(tables).restore)(jax.tree_util.tree_map(lambda x: x[pick], pool))
            # fresh keys, so a state drawn twice does not replay the same sensor noise
            start = start._replace(key=jax.random.split(jax.random.fold_in(k4, 1), n))
        else:  # continue the previous windows
            start = state
        if use_at:  # gradient with respect to the actions only
            (loss, (state, outs)), (gd, g_probe) = jax.value_and_grad(rollout, argnums=(8, 7), has_aux=True)(
                actor, critic, cnorm, start, reset_state, k2, tables, jnp.zeros(2),
                jnp.zeros((args.horizon, n, env.nu)))
            clip_stats = jax.lax.pmean(g_probe / args.horizon, "d")  # in-graph clipping: [rows clipped, norm removed]
        elif args.act_cot_clip is not None:  # the actor's window gradient with the in-graph clipping (no action targets)
            (loss, (state, outs)), (g, g_probe) = jax.value_and_grad(rollout, argnums=(0, 7), has_aux=True)(
                actor, critic, cnorm, start, reset_state, k2, tables, jnp.zeros(2))
            clip_stats = jax.lax.pmean(g_probe / args.horizon, "d")  # [rows clipped, norm removed]
        else:
            (loss, (state, outs)), g = jax.value_and_grad(rollout, has_aux=True)(
                actor, critic, cnorm, start, reset_state, k2, tables)
            clip_stats = jnp.zeros(2)
        if use_replay:
            pool = jnp.zeros(())  # the replay is read-only here and stays with the caller
        # RMS change of the joint setpoint [rad] an update causes at the batch's current states (logged)
        hist, goal = vmapped(env, tables)["policy_obs"](state, jax.random.split(k1, state.frame.shape[0]))
        a_old = actor_apply(actor, hist, goal)

        def da_of(params):
            return (actor_apply(params, hist, goal) - a_old) * env.action_scale

        def dq_of(params):
            return jnp.sqrt(jax.lax.pmean(jnp.mean(da_of(params) ** 2), "d"))

        def dq_env(params):  # per-env RMS over joints
            return jax.lax.all_gather(jnp.sqrt(jnp.mean(da_of(params) ** 2, -1)), "d").reshape(-1)

        if use_at:  # gnorm reports the mean regression loss (rad^2)
            actor, buf, gnorm = action_target_update(actor, buf, gd, outs, jax.random.fold_in(key, 2))
        else:
            g = jax.lax.pmean(g, "d")
            gnorm = optax.global_norm(g)
            finite = jnp.isfinite(gnorm) & (gnorm <= SHAC_GUARD) if args.shac_strict else jnp.isfinite(gnorm)
            g = jax.tree_util.tree_map(lambda x: jnp.where(finite, x, 0.0), g)
            upd, st_a_new = opt_a.update(g, st_a, actor)
            actor_new = optax.apply_updates(actor, upd)
            actor = jax.tree_util.tree_map(lambda a, b: jnp.where(finite, a, b), actor_new, actor)
            st_a = jax.tree_util.tree_map(lambda a, b: jnp.where(finite, a, b), st_a_new, st_a)
        outs["dq_rms"] = dq_of(actor)
        outs["dq_env_q"] = jnp.quantile(dq_env(actor), jnp.array([0.5, 0.95, 1.0]))
        outs["clip_stats"] = jnp.concatenate([clip_stats, outs.pop("at_clip")]) if use_at else clip_stats
        if "obs_norm" in actor:  # EMA of the actor-input statistics at the batch's current states
            x = jnp.concatenate([hist, goal], -1)
            mean, sq = jax.lax.pmean(x.mean(0), "d"), jax.lax.pmean((x ** 2).mean(0), "d")
            m0, v0 = actor["obs_norm"]
            actor = {**actor, "obs_norm": (0.99 * m0 + 0.01 * mean, 0.99 * v0 + 0.01 * jnp.maximum(sq - mean ** 2, 1e-6))}
        return actor, st_a, state, pool, ptr, outs, jax.lax.pmean(loss, "d"), gnorm, buf

    def critic_step(critic, st_c, cnorm, outs, key):
        tgt = nets.td_lambda(outs["r"], outs["term"], outs["done"], outs["v2"], args.gamma, args.lam)
        x = normalize(cnorm, outs["cobs_pre" if args.critic_pre_state else "cobs"].reshape(-1, env.critic_dim))
        y = tgt.reshape(-1)
        wt = jnp.ones_like(y)  # unit sample weights (the loss keeps its weighted-mean form)
        m = x.shape[0]
        mb = m // args.critic_minibatches

        def fit(c):  # MSE and explained variance of the window's targets
            e = critic_apply(c, x)[:, 0] - y
            mse, ym = jax.lax.pmean(jnp.mean(e ** 2), "d"), jax.lax.pmean(jnp.mean(y), "d")
            vy = jax.lax.pmean(jnp.mean((y - ym) ** 2), "d")
            return jnp.stack([mse, 1.0 - mse / (vy + 1e-12)])

        fit_pre = fit(critic)

        def loss_fn(c, xb, yb, wb):
            return jnp.sum(wb * (critic_apply(c, xb)[:, 0] - yb) ** 2) / jnp.maximum(jnp.sum(wb), 1.0)

        def epoch(carry, k):
            c, s = carry
            perm = jax.random.permutation(k, m)

            def mini(carry, i):
                c, s = carry
                idx = jax.lax.dynamic_slice(perm, (i * mb,), (mb,))
                lb, gc = jax.value_and_grad(loss_fn)(c, x[idx], y[idx], wt[idx])
                gc = jax.lax.pmean(gc, "d")
                u, s = opt_c.update(gc, s, c)
                return (optax.apply_updates(c, u), s), lb

            (c, s), ls = jax.lax.scan(mini, (c, s), jnp.arange(args.critic_minibatches))
            return (c, s), ls.mean()

        (critic, st_c), ls = jax.lax.scan(epoch, (critic, st_c), jax.random.split(key, args.critic_epochs))
        # [MSE before, after, explained variance before, after]: "before" is the fit to fresh data
        cfit = jnp.concatenate([fit_pre, fit(critic)])[jnp.array([0, 2, 1, 3])]
        return critic, st_c, jax.lax.pmean(ls.mean(), "d"), tgt, cfit

    def stats_step(cnorm, bin_q, outs, aux, tgt):
        """Update the critic-input normalizer and the reset-bin statistics."""
        x = outs["cobs"].reshape(-1, env.critic_dim)
        mean = jax.lax.pmean(x.mean(0), "d")
        sq = jax.lax.pmean((x ** 2).mean(0), "d")
        m0, v0 = cnorm
        cnorm = (0.99 * m0 + 0.01 * mean, 0.99 * v0 + 0.01 * jnp.maximum(sq - mean ** 2, 1e-6))
        if args.reset_sampling == "supermimic":
            q = tgt / g_max  # discounted (lambda-)return from each visited state, normalized
        else:
            q = (outs["r"] / env.max_reward) * (1.0 - outs["term"])
        b = aux["bins"][outs["frame"].reshape(-1)]
        s = jax.lax.psum(jnp.zeros(n_bins).at[b].add(q.reshape(-1)), "d")
        c = jax.lax.psum(jnp.zeros(n_bins).at[b].add(1.0), "d")
        stat = jnp.where(c > 0, 0.9 * bin_q[0] + 0.1 * s / jnp.maximum(c, 1.0), bin_q[0])
        past = 0.995 * bin_q[1] + 0.005 * bin_weights(stat, aux["bin_count"])
        return cnorm, jnp.stack([stat, past])

    start_it, env_steps0, wall0 = 0, 0, 0.0
    if args.resume:
        with open(args.resume, "rb") as f:
            ck = pickle.load(f)
        as_j = lambda t: jax.tree_util.tree_map(jnp.asarray, t)  # noqa: E731
        actor, critic, cnorm = as_j(ck["actor"]), as_j(ck["critic"]), as_j(tuple(ck["cnorm"]))
        st_a, st_c, bin_q = as_j(ck["st_a"]), as_j(ck["st_c"]), as_j(ck["bin_q"])
        start_it, env_steps0, wall0 = int(ck["iter"]), int(ck["env_steps"]), float(ck["wall_s"])
        print(f"resumed from {args.resume} at update {start_it}", flush=True)

    if args.motion_lib and start_it // args.ws_swap > 0:  # back on the working set the run was using
        env.set_library(ws_lib(start_it))
        tables, aux = env.tables(), make_aux()[0]

    # Parameters replicated over the (single) device axis "d" of the pmapped steps.
    rep = lambda tree: jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (nd,) + jnp.shape(x)), tree)  # noqa: E731
    first = lambda tree: jax.tree_util.tree_map(lambda x: x[0], tree)  # noqa: E731
    actor, critic, st_a, st_c, cnorm, bin_q = map(rep, (actor, critic, st_a, st_c, cnorm, bin_q))
    key, kr = jax.random.split(key)
    probs0 = jax.jit(frame_probs)(first(bin_q), aux)

    # The live reset select: probe reset states, the same states after the post-physics update at perturbed qpos / qvel,
    # and as replay starts are rebuilt; leaves equal in all three and across envs are skipped by select_reset.
    e_b = env.bind(tables)

    def probe(k):
        k1, k2, k3 = jax.random.split(k, 3)
        s_r = sample_resets(k1, probs0, tables, 256)
        q = s_r.dx.qpos + 0.01 * jax.random.normal(k2, s_r.dx.qpos.shape)
        v = s_r.dx.qvel + 0.1 * jax.random.normal(k3, s_r.dx.qvel.shape)
        s_a = jax.vmap(e_b.finish_physics)(s_r, jnp.zeros((256, env.nu)), q, v, s_r.dx.qacc_warmstart)
        return s_r, s_a, jax.vmap(e_b.restore)(env.compact(s_r))

    s_r, s_a, s_c = jax.jit(probe)(jax.random.fold_in(kr, 21))
    sim.STATE_SKIP = sim.skip_mask(s_r, s_a, s_c)
    sk = [(x.size // x.shape[0]) * x.dtype.itemsize for x, m in zip(jax.tree_util.tree_leaves(s_r),
                                                                      jax.tree_util.tree_leaves(sim.STATE_SKIP)) if m]
    tot = sum((x.size // x.shape[0]) * x.dtype.itemsize for x in jax.tree_util.tree_leaves(s_r))
    print(f"state select: {len(sk)} of {len(jax.tree_util.tree_leaves(s_r))} leaves skipped, "
          f"{sum(sk) / 1e3:.0f} of {tot / 1e3:.0f} KB per env", flush=True)
    del s_r, s_a, s_c

    def reset_states(k, probs_):  # n reset states, with random episode ages so time-outs do not all land in one window
        st = jax.pmap(sample_resets, in_axes=(0, None, None))(jax.random.split(k, nd), probs_, tables)
        ages = jax.random.randint(jax.random.fold_in(k, 1), (nd, n), 0, args.max_episode_steps)
        return st._replace(ep=ages.astype(st.ep.dtype))

    state = reset_states(kr, probs0)
    if args.actor_obs_norm and not args.resume:
        hist0, goal0 = jax.jit(lambda s, k: vmapped(env, tables)["policy_obs"](s, jax.random.split(k, n)))(
            first(state), jax.random.fold_in(kr, 4))
        x0 = jnp.concatenate([hist0, goal0], -1)
        actor["obs_norm"] = rep((x0.mean(0), jnp.maximum(x0.var(0), 1e-6)))

    if use_replay:
        psize = args.replay_steps * nc
        p_creset = jax.pmap(lambda k, p_, t_: sample_resets(k, p_, t_, nc), in_axes=(0, None, None))
        p_ctile = jax.pmap(lambda c: jax.tree_util.tree_map(
            lambda x: jnp.tile(x, (args.replay_steps,) + (1,) * (x.ndim - 1)), env.compact(c)))

        def init_collector(k, probs_):
            # collector envs with random episode ages; the replay starts as copies of their states with only
            # those nc entries counted as filled, and grows by nc states per collector step
            kc1, kc2 = jax.random.split(k)
            cs = p_creset(jax.random.split(kc1, nd), probs_, tables)
            cs = cs._replace(ep=jax.random.randint(kc2, (nd, nc), 0, args.max_episode_steps).astype(cs.ep.dtype))
            return cs, p_ctile(cs), jnp.tile(jnp.array([nc % psize, nc], jnp.int32), (nd, 1))

        cstate, pool, ptr = init_collector(jax.random.fold_in(kr, 3), probs0)
        print(f"replay: {psize} states ({sum(x[0].nbytes for x in jax.tree_util.tree_leaves(pool)) / 2**30:.1f} GiB), "
              f"collector {nc} envs, {1 / args.utd:.2f} steps/update, noise {args.collect_noise}", flush=True)
    else:
        pool = jnp.zeros((nd, 1))
        ptr = jnp.zeros((nd, 2), jnp.int32)
    if use_at:
        k_win = int(args.action_target[1])
        buf = rep(dict(hist=jnp.zeros((k_win, args.horizon, n, env.hist_dim), jnp.bfloat16),
                       goal=jnp.zeros((k_win, args.horizon, n, env.goal_dim), jnp.bfloat16),
                       tgt=jnp.zeros((k_win, args.horizon, n, env.nu)), w=jnp.zeros((k_win, args.horizon, n)),
                       count=jnp.int32(0), it=jnp.int32(start_it)))  # it: update counter for the DELTA schedule
        buf["opt"] = opt_r.init(first(actor))  # the regression optimizer state lives in the buffer
        if args.resume and "st_r" in ck:  # keeps its schedule position on resume
            buf["opt"] = as_j(ck["st_r"])
        buf["opt"] = rep(buf["opt"])
        print(f"action-target buffer: {k_win} windows x {args.horizon} x {n} "
              f"({sum(x[0].nbytes for x in jax.tree_util.tree_leaves(buf)) / 2**30:.1f} GiB)", flush=True)
    else:
        buf = rep(dict(count=jnp.int32(0)))  # unused placeholder
    p_actor = jax.pmap(actor_step, axis_name="d", in_axes=(0,) * 9 + (None, 0), donate_argnums=(10,))
    p_critic = jax.pmap(critic_step, axis_name="d")
    p_stats = jax.pmap(stats_step, axis_name="d", in_axes=(0, 0, 0, None, 0))
    p_probs = jax.pmap(frame_probs, in_axes=(0, None))
    probs = p_probs(bin_q, aux)
    keys = jax.random.split(key, nd)
    t0 = time.time()
    p_actor = p_actor.lower(actor, st_a, critic, cnorm, state, pool, ptr, probs, keys, tables, buf).compile()
    if use_replay:
        p_collect = jax.pmap(collect_step, axis_name="d", in_axes=(0,) * 6 + (None,), donate_argnums=(2, 3))
        p_collect = p_collect.lower(actor, cstate, pool, ptr, probs, keys, tables).compile()
    print(f"compiled actor step in {time.time() - t0:.0f}s", flush=True)
    col_acc, col_terms = 0.0, []

    def save(path, it, wall_s, env_steps):
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(dict(actor=jax.device_get(first(actor)), critic=jax.device_get(first(critic)),
                             cnorm=jax.device_get(first(cnorm)), st_a=jax.device_get(first(st_a)),
                             st_c=jax.device_get(first(st_c)), bin_q=jax.device_get(first(bin_q)),
                             **({"st_r": jax.device_get(first(buf["opt"]))} if "opt" in buf else {}),
                             iter=it, wall_s=wall_s, env_steps=env_steps), f)
        tmp.rename(path)  # atomic: the evaluator never sees a partial file

    t_start, env_steps = time.time() - wall0, env_steps0
    # per update: raw grad norm (or regression loss), RMS setpoint change, per-env setpoint change quantiles (p50, p95,
    # max) and action-cotangent clipping stats
    gnorms, dqs, dq_qs, clips = [], [], [], []
    it = start_it
    for it in range(start_it, args.iters + 1):
        if args.motion_lib and it > start_it and it % args.ws_swap == 0:  # the next working set (same table shapes)
            t_ws = time.time()
            # drop the old working set's device arrays first: a second replay pool next to the old one may not fit
            tables = cstate = pool = None
            gc.collect()
            env.set_library(ws_lib(it))
            tables = env.tables()
            aux, bq = make_aux()
            bin_q = rep(bq)
            probs = p_probs(bin_q, aux)
            k_ws = jax.random.fold_in(key, it)
            state = reset_states(k_ws, first(probs))
            cstate, pool, ptr = init_collector(jax.random.fold_in(k_ws, 3), first(probs))
            print(f"update {it}: working set {it // args.ws_swap} ({len(env.starts)} clips) in "
                  f"{time.time() - t_ws:.1f} s", flush=True)
        out_of_time = args.hours is not None and time.time() - t_start >= 3600 * args.hours
        if it % args.ckpt_every == 0 or it == args.iters or out_of_time:
            save(out / "ckpt" / f"it_{it:06d}.pkl", it, time.time() - t_start, env_steps)
        if it == args.iters or out_of_time:
            break
        key, k1, k2 = jax.random.split(key, 3)
        t1 = time.time()
        keys_now = jax.random.split(k1, nd)
        if use_replay:  # collector steps for this update (1/utd on average), then the update from the replay
            col_acc += 1.0 / args.utd
            for j in range(int(col_acc)):
                cstate, pool, ptr, ct = p_collect(actor, cstate, pool, ptr, probs,
                                                  jax.random.split(jax.random.fold_in(k1, 100 + j), nd), tables)
                col_terms.append(ct)
            col_acc -= int(col_acc)
            actor, st_a, state, _, ptr, outs, loss, gnorm, buf = p_actor(actor, st_a, critic, cnorm, state, pool, ptr,
                                                                         probs, keys_now, tables, buf)
        else:
            actor, st_a, state, pool, ptr, outs, loss, gnorm, buf = p_actor(actor, st_a, critic, cnorm, state, pool,
                                                                            ptr, probs, keys_now, tables, buf)
        gn = float(gnorm[0])
        if args.shac_strict and not (math.isfinite(gn) and gn <= SHAC_GUARD):  # SHAC raises here; not applied
            save(out / "ckpt" / f"it_{it:06d}.pkl", it, time.time() - t_start, env_steps)
            halt = dict(iter=it, grad_norm=gn if math.isfinite(gn) else str(gn), wall_s=time.time() - t_start,
                        env_steps=env_steps, rule="SHAC guard: pre-clip actor gradient norm non-finite or > 1e6")
            (out / "HALTED").write_text(json.dumps(halt) + "\n")
            print(f"halted (--shac-strict): {json.dumps(halt)}", flush=True)
            break
        gnorms.append(gn)
        dqs.append(float(outs["dq_rms"][0]))
        dq_qs.append([round(float(x), 5) for x in outs["dq_env_q"][0]])
        clips.append([round(float(x), 4) for x in outs["clip_stats"][0]])
        critic, st_c, closs, tgt, cfit = p_critic(critic, st_c, cnorm, outs, jax.random.split(k2, nd))
        cnorm, bin_q = p_stats(cnorm, bin_q, outs, aux, tgt)
        probs = p_probs(bin_q, aux)
        jax.block_until_ready(critic)
        env_steps += n * nd * args.horizon
        if it % 10 == 0:
            o = first(outs)
            rec = {"iter": it, "wall_s": time.time() - t_start, "iter_s": time.time() - t1,
                   "env_steps": env_steps, "actor_loss": float(loss[0]), "grad_norm": float(gnorm[0]),
                   "grad_norms": [round(x, 4) for x in gnorms],
                   "dq_rms": [round(x, 5) for x in dqs],
                   "dq_env_q": dq_qs,
                   **({"clip_stats": clips} if args.act_cot_clip is not None else {}),
                   **({"at_prof": [[round(float(x), 4) for x in r] for r in o["at_prof"]]} if "at_prof" in o else {}),
                   **({"collector_term_rate": float(np.mean([float(c[0]) for c in col_terms])) if col_terms else None,
                       "replay_fill": int(ptr[0, 1])} if use_replay else {}),
                   "critic_loss": float(closs[0]), "critic_fit": [round(float(x), 5) for x in cfit[0][:4]],
                   "rew_mean": float(o["r"].mean() / env.max_reward),
                   "term_rate": float(o["term"].mean()),
                   **({"clip_end_rate": float(o["cend"].mean()),
                       "v_end_gap": float((o["v_gap"] * o["cend"]).sum() / max(float(o["cend"].sum()), 1.0))}
                      if args.clip_end_boot == "pre" else {}),
                   **({"warp_peak": env.warp[n].stats()} if it % 100 == 0 else {}),
                   # the allocator's own numbers: in use / peak in use (live buffers) vs reserved from the GPU and the
                   # device limit, GB
                   **({"gpu_mem_gb": {k: round(v / 1e9, 2) for k, v in (jax.local_devices()[0].memory_stats() or {}).items()
                                      if k in ("bytes_in_use", "peak_bytes_in_use", "bytes_reserved", "pool_bytes",
                                               "largest_alloc_size", "bytes_limit")}}
                      if it % 100 == 0 or it in (10, 30) else {}),
                   **({"engine_diag": {k: v for k, v in jax_ffi.read_diagnostics(reset=True).items()
                                       if k in ("bwd_rows", "nonfinite_rows", "nonfinite_cotangent_rows",
                                                "fwd_iteration_limit_worlds", "replay_mismatch_worlds")}}
                      if args.physics == "engine" else {}),
                   **{f"rew/{k}": float(v.mean() / env.max_reward) for k, v in o["terms"].items()}}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if not (np.isfinite(rec["actor_loss"]) and np.isfinite(rec["critic_loss"])):  # no way back from NaN
                print(f"non-finite loss at update {it}: stopping, run diverged", flush=True)
                (out / "DIVERGED").write_text(str(it))
                return
            gnorms, dqs, dq_qs, clips, col_terms = [], [], [], [], []
        if it % 500 == 0 or it == args.iters - 1:
            save(out / "params.pkl", it, time.time() - t_start, env_steps)
    (out / "DONE").write_text(str(it))


if __name__ == "__main__":
    main()
