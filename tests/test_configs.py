"""The configuration files resolve (includes, last value wins) and parse with the trainers' argument parsers (CPU, no
data needed)."""

from pathlib import Path

import pytest

from nablamimic import ppo, train
from nablamimic.params import read_params

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def parse(module, name, *extra):
    return module.build_parser().parse_args(read_params(CONFIGS / name) + ["--out", "unused", *extra])


def test_ours():
    """The recipe on both simulators; configs/ours*.args use residual actions, configs/reported/ours_absolute*.args
    (the reported comparisons) absolute ones, and otherwise equal them."""
    for name, physics, residual in (("ours.args", "engine", True), ("ours_mjw.args", "warp", True),
                                    ("reported/ours_absolute.args", "engine", False),
                                    ("reported/ours_absolute_mjw.args", "warp", False)):
        a = parse(train, name)
        assert a.physics == physics and a.horizon == 64 and a.num_envs == 4096 and a.no_remat
        assert a.action_target == [8, 1, 0.03] and a.action_target_lr == 6e-4 and a.action_target_delta_floor == 0.3
        assert a.act_cot_clip == 3 and a.act_noise == 0.05 and a.anchor_to_decoder and a.enc_hoist == "on"
        assert a.critic_pre_state and a.mask_after_done and a.term_penalty == [2, 0.08, 0.10]
        assert (a.replay_steps, a.collect_envs, a.collect_noise, a.utd) == (1000, 4096, 0.1, 0.4)
        assert (a.reset_sampling, a.uniform_over) == ("supermimic", "bins")
        assert (a.lr_warmup_steps, a.lr_decay_steps, a.lr_decay_to) == (500, 16000, 0.1)
        assert a.solver_iterations == 10 and a.hours == 3 and not a.actor_obs_norm
        assert not a.v_bound and a.clip_end_boot == "next" and not a.shac_strict and a.residual_action == residual
        assert a.action_rate_space == "setpoint"
    for name in ("ours_mjw.args", "reported/ours_absolute_mjw.args"):
        a = parse(train, name)
        assert (a.mjw_adj_normal, a.mjw_adj_parallel, a.mjw_ift_solve) == (3, 1, 1)
    for res, ab in (("ours.args", "reported/ours_absolute.args"), ("ours_mjw.args", "reported/ours_absolute_mjw.args")):
        a, b = vars(parse(train, res)), vars(parse(train, ab))
        assert {k for k in a if a[k] != b[k]} == {"residual_action"}, res


def test_shac():
    for strict, (eng, mjw) in ((True, ("shac.args", "shac_mjw.args")),
                               (False, ("shac_guard_changes.args", "shac_guard_changes_mjw.args"))):
        a = parse(train, eng)
        assert a.physics == "engine" and not a.no_remat  # remat with the engine's policy
        b = parse(train, mjw)
        assert b.physics == "warp" and b.no_remat and (b.mjw_adj_normal, b.mjw_adj_parallel, b.mjw_ift_solve) == (3, 1, 1)
        for x in (a, b):
            assert x.shac_strict == strict and x.hours == 3 and not x.v_bound and x.clip_end_boot == "next"
            assert not x.residual_action  # SHAC keeps the absolute action space
            assert x.action_rate_space == "setpoint"
            assert x.horizon == 32 and x.num_envs == 4096 and x.actor_obs_norm and x.action_target is None
            assert (x.reset_sampling, x.uniform_over, x.replay_steps, x.act_cot_clip) == ("uniform", "frames", 0, None)
            assert (x.lr, x.lr_warmup_steps, x.lr_decay_steps, x.lr_decay_to, x.act_noise) == (2e-4, 500, 16000, 0.1, 0.05)


def test_ablations():
    """Each ablation is the absolute-action recipe (configs/reported/ours_absolute.args, configs/reported/ablations/) or
    the residual-action recipe (configs/ours.args, configs/reported/ablations_residual/) with exactly the intended
    options changed, and a 1.5 h budget; the residual-action ablation's arm is configs/ours.args with
    --hours 1.5 and the action-rate term on the raw policy outputs, as in the reported runs."""
    ours = vars(parse(train, "reported/ours_absolute.args"))
    res = vars(parse(train, "ours.args", "--hours", "1.5", "--action-rate-space", "raw"))
    assert {k for k in res if res[k] != ours[k]} == {"residual_action", "action_rate_space", "hours"}
    assert res["residual_action"] and res["action_rate_space"] == "raw" and res["hours"] == 1.5
    changed = {
        "no_pods.args": {"action_target": None, "action_target_lr": None, "action_target_delta_floor": 1.0,
                         "no_remat": False},
        "no_clip.args": {"act_cot_clip": None},
        "h32.args": {"horizon": 32},
        "no_replay.args": {"replay_steps": 0, "collect_envs": 0},
        "uniform_resets.args": {"reset_sampling": "uniform"},
    }
    # the same ablations of the residual-action recipe (configs/ours.args)
    ours_res = vars(parse(train, "ours.args"))
    for d, base in (("reported/ablations", ours), ("reported/ablations_residual", ours_res)):
        assert sorted(p.name for p in (CONFIGS / d).glob("*.args")) == sorted(changed), d
        for name, diff in changed.items():
            a = vars(parse(train, f"{d}/{name}"))
            assert a["hours"] == 1.5
            assert {k for k in a if a[k] != base[k]} == set(diff) | {"hours"}, (d, name)
            assert all(a[k] == v for k, v in diff.items()), (d, name)


def test_bones():
    a = parse(train, "bones.args", "--motion-lib", "DATA/lib50")
    assert a.physics == "engine" and (a.ws_shards, a.ws_swap, a.lr_decay_steps, a.hours) == (8, 2000, 52000, 6)
    assert a.motion_lib == "DATA/lib50" and a.horizon == 64 and a.enc_hoist == "on"
    assert a.v_bound and a.clip_end_boot == "pre" and not a.shac_strict and a.residual_action
    assert a.action_rate_space == "setpoint"


def test_ppo():
    a = parse(ppo, "ppo_beyondmimic.args")
    assert (a.arch, a.hidden, a.critic_hidden, a.num_envs, a.reset_sampling) == ("mlp", [512, 256, 128], None, 4096,
                                                                               "beyondmimic")
    b = parse(ppo, "ppo_supermimic_style.args")
    assert (b.arch, b.critic_hidden, b.num_envs, b.reset_sampling) == ("supermimic", [256, 128], 8192, "supermimic")
    for x in (a, b):
        assert (x.steps, x.epochs, x.minibatches, x.lr, x.desired_kl, x.clip) == (24, 5, 4, 1e-3, 0.01, 0.2)
        assert (x.value_coef, x.entropy, x.gamma, x.lam, x.max_grad_norm, x.init_std) == (1, 0.005, 0.99, 0.95, 1, 1)
        assert (x.hours, x.ckpt_minutes, x.ckpt_dense, x.solver_iterations) == (6, 15, [1, 2], 10)


@pytest.mark.parametrize("name", sorted(str(p.relative_to(CONFIGS)) for p in CONFIGS.rglob("*.args")))
def test_no_site_paths(name):
    args = read_params(CONFIGS / name)
    assert not any(x.startswith("/") for x in args), args
