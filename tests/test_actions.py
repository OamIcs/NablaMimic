"""The joint setpoints of an action (around the default pose, or with residual actions around the next reference
frame) and the reward's action-rate term on the commanded setpoints (CPU, no data needed)."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from nablamimic.env import EnvConfig, TrackingEnv

NU = 3
Q0 = np.array([0.1, -0.2, 0.3])
SCALE = np.array([0.5, 0.5, 0.25])
REF_QPOS = np.arange(4 * (7 + NU), dtype=np.float32).reshape(4, 7 + NU) / 100.0
REF_LAST = np.array([2, 2, 2, 3])  # frames 0-2 one clip (last frame 2), frame 3 another


def fake_env(residual, space="setpoint"):
    e = object.__new__(TrackingEnv)  # only the attributes the action functions read
    e.__dict__.update(cfg=EnvConfig(residual_action=residual, action_rate_space=space), q0=jnp.asarray(Q0, jnp.float32),
                      action_scale=jnp.asarray(SCALE, jnp.float32), ctrl_lo=jnp.full(NU, -1.0),
                      ctrl_hi=jnp.full(NU, 1.0), ref_qpos=jnp.asarray(REF_QPOS), ref_last=jnp.asarray(REF_LAST))
    return e


def prev_state(frame, ep, hist_same=False):
    """The fields of the pre-step State that action_change reads (a reset's history has identical rows)."""
    hist = np.ones((4, 5)) if hist_same else np.arange(20.0).reshape(4, 5)
    return SimpleNamespace(frame=jnp.int32(frame), ep=jnp.int32(ep), hist=jnp.asarray(hist, jnp.float32))


def setpoint(frame, a):
    """The commanded (unclipped) residual setpoint at reference frame `frame`, float64."""
    return REF_QPOS[min(frame + 1, REF_LAST[frame]), 7:].astype(np.float64) + SCALE * np.asarray(a, np.float64)


def test_default_pose():
    e, a = fake_env(False), jnp.array([1.0, -1.0, 8.0])
    np.testing.assert_allclose(TrackingEnv.ctrl_of(e, a), [0.6, -0.7, 1.0])
    np.testing.assert_allclose(TrackingEnv.ctrl_of(e, a, jnp.int32(1)), [0.6, -0.7, 1.0])  # the frame is not used


def test_residual_action():
    e, a = fake_env(True), jnp.array([0.2, 0.0, -0.4])
    for frame, ref in ((0, 1), (1, 2), (2, 2), (3, 3)):  # the next frame, held at the clip's last frame
        want = np.clip(np.asarray(e.ref_qpos[ref, 7:]) + np.asarray(e.action_scale) * np.asarray(a), -1.0, 1.0)
        np.testing.assert_allclose(TrackingEnv.ctrl_of(e, a, jnp.int32(frame)), want, rtol=1e-6)


@pytest.mark.parametrize("residual,space", [(False, "setpoint"), (False, "raw"), (True, "raw")])
def test_action_rate_raw_outputs(residual, space):
    """Absolute actions (either space) and residual actions with "raw": a_t - a_{t-1} of the policy outputs."""
    e = fake_env(residual, space)
    a, pa = jnp.array([0.3, -2.0, 5.0]), jnp.array([0.1, 0.4, -1.0])
    for prev in (prev_state(1, 5), prev_state(1, 0), None):
        np.testing.assert_array_equal(TrackingEnv.action_change(e, a, pa, prev), a - pa)


def test_action_rate_setpoint():
    """Residual actions with "setpoint": (u_t - u_{t-1}) / scale of the commanded unclipped setpoints, u_{-1} = q0 after
    a reset, so the term is the absolute-action one of the same setpoints."""
    e = fake_env(True)
    a = np.array([0.3, -2.0, 9.0])   # the last joint's setpoint is beyond the setpoint range: the term ignores the clip
    pa = np.array([0.1, 0.4, -1.0])
    cases = [  # (pre-step frame, previous action's frame, episode age, previous action, reset history)
        (1, 0, 5, pa, False),            # continuing: the previous action was taken one frame before
        (2, 2, 5, pa, False),            # at the clip's last frame: both bases are its joint angles
        (1, None, 0, pa, False),         # ep == 0: a reset whatever prev_a holds (a stale action)
        (1, None, 7, 0 * pa, True),      # a reset state with a random episode age: prev_a = 0, identical history rows
        (1, 0, 7, 0 * pa, False),        # an exactly zero previous action mid-episode is not a reset
    ]
    for frame, pframe, ep, p, hist_same in cases:
        u = setpoint(frame, a)
        u_prev = Q0 if pframe is None else setpoint(pframe, p)
        want = (u - u_prev) / SCALE
        got = TrackingEnv.action_change(e, jnp.asarray(a, jnp.float32), jnp.asarray(p, jnp.float32),
                                        prev_state(frame, ep, hist_same))
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5, err_msg=str((frame, pframe, ep, hist_same)))
    with pytest.raises(AssertionError):
        TrackingEnv.action_change(e, jnp.asarray(a), jnp.asarray(pa))  # needs the pre-step state
