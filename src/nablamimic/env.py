"""G1 motion-tracking environment.

Follows SuperMimic (Liao et al. 2026, arXiv:2610.28480) App. A and, where it defers to them, BeyondMimic / mjlab:

- robot: the mjlab G1 (robot.py); actuators: mjlab G1 PD gains, kp = armature * (2*pi*10 Hz)^2, kd = 2*2*armature*omega;
  actions q_sp = q0 + 0.25 * effort / kp * a around mjlab's knees-bent pose (residual_action: around the next reference
  frame's joint angles instead)
- 4 x 5 ms physics substeps, 50 Hz control
- reward: BeyondMimic tracking terms of Table A.2 over 14 bodies, anchor torso_link,
  every term multiplied by dt = 0.02 (max 0.1 per step); the action-rate term is on the commanded joint setpoints
  (action_rate_space="setpoint"): with residual actions the reward is the absolute-action one of the same setpoints
- policy input: 10-frame history of 93-d proprioception with Table A.1 noise, and a goal of K = 10 reference frames
  (68-d each, reference-intrinsic) plus the orientation part of BeyondMimic's anchor-relative reference pose (6-d: needs
  the robot's own orientation, i.e. the IMU, not its global position), as in BeyondMimic's
  G1FlatWoStateEstimationEnvCfg
- domain randomization per episode: foot friction U[0.3, 1.2], encoder bias U[-0.01, 0.01],
  torso COM offset (Table A.3); none at evaluation (friction 1.0)
- termination: |z_root - z_ref| > 0.25 m or gravity-z difference > 0.8 (pelvis)
- privileged critic input: clean proprioception, goal, root errors, anchor pose and the DR context

The physics step itself is batched outside (sim.vmapped: NablaEngine or MuJoCo Warp, then finish_physics[_kin]);
MJX provides the model / data containers and the kinematics of reset and restored states. All functions act on one
environment; callers vmap them.
"""

import copy
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx
from mujoco.mjx._src import math as mjm
from mujoco.mjx._src import smooth

from . import robot
from .motion import SCENE, MotionLibrary

BM_BODIES = (
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
)
ANCHOR = "torso_link"
SUBSTEPS, TIMESTEP, DT = 4, 0.005, 0.02
HIST, FUTURE = 10, 10
PROPRIO_DIM, GOAL_FRAME_DIM, ANCHOR_DIM = 93, 68, 9
GOAL_ANCHOR = slice(3, 9)  # the orientation part of _anchor_rel at the end of the goal
OMEGA = 10 * 2.0 * np.pi
# mjlab knees-bent keyframe (joint name substring -> angle); other joints 0.
Q0 = {"hip_pitch": -0.312, "knee": 0.669, "ankle_pitch": -0.363, "elbow": 0.6,
      "left_shoulder_roll": 0.2, "right_shoulder_roll": -0.2, "shoulder_pitch": 0.2}
# Reward std and weights (Table A.2).
STD = dict(body_pos=0.3, body_ori=0.4, body_lin_vel=1.0, body_ang_vel=3.14, root_pos=0.3, root_ori=0.4)
W = dict(body_pos=1.0, body_ori=1.0, body_lin_vel=1.0, body_ang_vel=1.0, root_pos=0.5, root_ori=0.5,
         action_rate=-0.1, joint_limit=-2.0)
# Proprioception noise half-widths (Table A.1) and fixed input scales.
NOISE = dict(joint_pos=0.01, joint_vel=0.5, ang_vel=0.2, gravity=0.05)
SCALE = dict(joint_vel=1 / 5.0, ang_vel=1 / 2.0)


@dataclass(frozen=True)
class EnvConfig:
    solver_iterations: int = 10
    dr: bool = True                # domain randomization (training); off for evaluation
    obs_noise: bool = True
    reset_noise: bool = True
    term_height: float = 0.25
    term_tilt: float = 0.8
    residual_action: bool = False  # joint setpoints around the next reference frame's joint angles instead of q0
    action_rate_space: str = "setpoint"  # the reward's action-rate term on the commanded setpoints ("setpoint") or
                                         # on the raw policy outputs ("raw"); the same for absolute actions


class Xi(NamedTuple):
    """Per-episode domain-randomization context."""
    friction: jnp.ndarray   # ()
    enc_bias: jnp.ndarray   # (29,)
    com: jnp.ndarray        # (3,) torso COM offset


class State(NamedTuple):
    dx: mjx.Data
    frame: jnp.ndarray      # reference frame the robot is at
    ep: jnp.ndarray         # steps since reset
    hist: jnp.ndarray       # (HIST, PROPRIO_DIM) noisy proprioception, newest last
    prev_a: jnp.ndarray     # (29,)
    xi: Xi
    key: jnp.ndarray


def _yaw_quat(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    yaw = jnp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return jnp.array([jnp.cos(yaw / 2), 0.0, 0.0, jnp.sin(yaw / 2)])


def _quat_angle_sq(q1, q2):
    """Squared rotation angle between unit quaternions, with a finite gradient at zero."""
    e = mjm.quat_mul(q1, mjm.quat_inv(q2))
    vn = jnp.sqrt(jnp.sum(e[1:] ** 2) + 1e-12)
    return (2.0 * jnp.arctan2(vn, jnp.abs(e[0]))) ** 2


def _gravity_b(q):
    return mjm.rotate(jnp.array([0.0, 0.0, -1.0]), mjm.quat_inv(q))


def _np_rotate_inv(q, v):
    """Rotate v by the inverse of quaternion q (numpy, batched)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r = np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)
    return np.einsum("...ji,...j->...i", r, v)


def build_model(cfg: EnvConfig):
    m = robot.build(SCENE)
    m.opt.iterations = cfg.solver_iterations
    m.opt.timestep = TIMESTEP
    kp = np.empty(m.nu)
    for i in range(m.nu):
        dof = m.jnt_dofadr[m.actuator_trnid[i, 0]]
        arm = m.dof_armature[dof]
        kp[i] = arm * OMEGA ** 2
        m.actuator_gainprm[i, 0] = kp[i]
        m.actuator_biasprm[i, 1] = -kp[i]
        m.actuator_biasprm[i, 2] = -2.0 * 2.0 * arm * OMEGA
    robot.widen_ctrlrange(m, kp)
    return m, kp


class TrackingEnv:
    def __init__(self, lib: MotionLibrary, cfg: EnvConfig):
        assert lib.bodies == BM_BODIES, "load the library with bodies=BM_BODIES"
        assert cfg.action_rate_space in ("setpoint", "raw"), cfg.action_rate_space
        self.cfg = cfg
        self.m, kp = build_model(cfg)
        m = self.m
        dev = jax.local_devices()[0]
        self.mx = mjx.put_model(m, device=dev)
        d = mujoco.MjData(m)
        d.qpos[2] = 1.5  # template only; resets overwrite qpos/qvel
        mujoco.mj_forward(m, d)
        self.d0 = mjx.put_data(m, d, device=dev)
        self.nu = m.nu
        jnames = [m.joint(m.actuator_trnid[i, 0]).name for i in range(m.nu)]
        q0 = np.zeros(m.nu)
        for i, n in enumerate(jnames):
            for k, v in Q0.items():
                if k in n:
                    q0[i] = v
        self.q0, self._q0_np = jnp.asarray(q0, jnp.float32), q0
        effort = np.array([m.jnt_actfrcrange[m.actuator_trnid[i, 0], 1] for i in range(m.nu)])
        self.action_scale = jnp.asarray(0.25 * effort / kp, jnp.float32)
        self.ctrl_lo = jnp.asarray(m.actuator_ctrlrange[:, 0], jnp.float32)
        self.ctrl_hi = jnp.asarray(m.actuator_ctrlrange[:, 1], jnp.float32)
        jr = np.array([m.jnt_range[m.actuator_trnid[i, 0]] for i in range(m.nu)])
        mid, half = jr.mean(1), 0.5 * (jr[:, 1] - jr[:, 0]) * 0.9
        self.soft_lo, self.soft_hi = jnp.asarray(mid - half, jnp.float32), jnp.asarray(mid + half, jnp.float32)
        self.body_ids = jnp.asarray([m.body(n).id for n in BM_BODIES])
        self.anchor_k = BM_BODIES.index(ANCHOR)
        self.torso_id = m.body(ANCHOR).id
        self.root_com_id = int(m.body_rootid[m.body("pelvis").id])
        self.foot_pairs = jnp.asarray([i for i in range(m.npair) if "_foot" in m.pair(i).name and m.pair(i).name.endswith("_floor")])

        self.set_library(lib)
        self.goal_dim = FUTURE * GOAL_FRAME_DIM + (GOAL_ANCHOR.stop - GOAL_ANCHOR.start)
        self.hist_dim = HIST * PROPRIO_DIM
        self.xi_dim = 1 + m.nu + 3
        self.critic_dim = PROPRIO_DIM + ANCHOR_DIM + self.goal_dim + 1 + 3 + self.xi_dim

    TABLES = ("ref_qpos", "ref_qvel", "ref_body_pos", "ref_body_quat", "ref_body_lin", "ref_body_ang",
              "ref_last", "goal_feat")

    def set_library(self, lib):
        """(Re)load the reference arrays from a MotionLibrary. A library with the same number of frames gives tables of
        the same shapes, so jitted code that takes tables() as arguments runs on it without recompiling (working sets
        of a large library, motion_lib)."""
        self.ref_qpos = jnp.asarray(lib.qpos)
        self.ref_qvel = jnp.asarray(lib.qvel)
        self.ref_body_pos = jnp.asarray(lib.body_pos)
        self.ref_body_quat = jnp.asarray(lib.body_quat)
        self.ref_body_lin = jnp.asarray(lib.body_lin_vel)
        self.ref_body_ang = jnp.asarray(lib.body_ang_vel)
        self.ref_last = jnp.asarray(lib.last)
        self.starts = np.asarray(lib.starts)
        self.lengths = np.asarray([lib.last[s] - s + 1 for s in lib.starts])
        self.n_frames = len(lib.last)
        self.valid = jnp.asarray(np.arange(self.n_frames) < lib.last)  # at least one step left
        self.goal_feat = jnp.asarray(self._goal_features(lib, self._q0_np))

    def tables(self):
        """Reference arrays; pass them into jitted code and `bind` them there, so they are
        traced arguments rather than (very large) compile-time constants."""
        return {k: getattr(self, k) for k in self.TABLES}

    def bind(self, tables):
        e = copy.copy(self)
        e.__dict__.update(tables)
        return e

    # ---------------------------------------------------------------- reference features
    @staticmethod
    def _goal_features(lib, q0):
        q = lib.qpos[:, 3:7]
        grav = _np_rotate_inv(q, np.array([0.0, 0.0, -1.0]))
        w, z = q[:, 0], q[:, 3]
        yaw = np.arctan2(2 * (w * z + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + z ** 2))
        qy = np.stack([np.cos(yaw / 2), 0 * yaw, 0 * yaw, np.sin(yaw / 2)], -1)
        lin_h = _np_rotate_inv(qy, lib.qvel[:, :3])
        return np.concatenate([
            lib.qpos[:, 7:] - q0, lib.qvel[:, 6:] * SCALE["joint_vel"], grav,
            lib.qvel[:, 3:6] * SCALE["ang_vel"], lin_h, lib.qpos[:, 2:3],
        ], axis=1).astype(np.float32)

    # ---------------------------------------------------------------- DR / model
    def sample_xi(self, key):
        if not self.cfg.dr:
            return Xi(jnp.float32(1.0), jnp.zeros(self.nu), jnp.zeros(3))
        k1, k2, k3 = jax.random.split(key, 3)
        lim = jnp.array([0.025, 0.05, 0.05])
        return Xi(jax.random.uniform(k1, (), minval=0.3, maxval=1.2),
                  jax.random.uniform(k2, (self.nu,), minval=-0.01, maxval=0.01),
                  jax.random.uniform(k3, (3,), minval=-lim, maxval=lim))

    def model_for(self, xi):
        pf = self.mx.pair_friction.at[self.foot_pairs, :2].set(xi.friction)
        ipos = self.mx.body_ipos.at[self.torso_id].add(xi.com)
        return self.mx.replace(pair_friction=pf, body_ipos=ipos)

    # ---------------------------------------------------------------- observations
    def proprio(self, dx, xi, prev_a):
        q = dx.qpos[3:7]
        return jnp.concatenate([
            dx.qpos[7:] - self.q0 + xi.enc_bias, dx.qvel[6:] * SCALE["joint_vel"],
            dx.qvel[3:6] * SCALE["ang_vel"], _gravity_b(q), prev_a,
        ])

    def _noise(self, key):
        lim = jnp.concatenate([
            jnp.full(self.nu, NOISE["joint_pos"]), jnp.full(self.nu, NOISE["joint_vel"] * SCALE["joint_vel"]),
            jnp.full(3, NOISE["ang_vel"] * SCALE["ang_vel"]), jnp.full(3, NOISE["gravity"]), jnp.zeros(self.nu)])
        return jax.random.uniform(key, (PROPRIO_DIM,), minval=-1.0, maxval=1.0) * lim

    def _anchor_rel(self, dx, frame):
        """Reference anchor pose in the robot anchor frame: position (3) + first two rotation columns (6)."""
        return self._anchor_rel_pq(dx.xpos[self.torso_id], dx.xquat[self.torso_id], frame)

    def _anchor_rel_pq(self, ra_p, ra_q, frame):
        ma_p, ma_q = self.ref_body_pos[frame, self.anchor_k], self.ref_body_quat[frame, self.anchor_k]
        pos = mjm.rotate(ma_p - ra_p, mjm.quat_inv(ra_q))
        rot = mjm.quat_to_mat(mjm.quat_mul(mjm.quat_inv(ra_q), ma_q))
        return jnp.concatenate([pos, rot[:, 0], rot[:, 1]])

    def goal_ref(self, frame):
        """The goal's reference part (the next FUTURE frames' features): a function of the frame index alone."""
        idx = jnp.minimum(frame + jnp.arange(FUTURE), self.ref_last[frame])
        return self.goal_feat[idx].reshape(-1)

    def goal(self, state, key=None):
        g = self.goal_ref(state.frame)
        sl = GOAL_ANCHOR
        a = self._anchor_rel(state.dx, state.frame)[sl]
        if key is not None and self.cfg.obs_noise:
            lim = jnp.concatenate([jnp.full(3, 0.25), jnp.full(6, 0.05)])[sl]
            a = a + jax.random.uniform(key, a.shape, minval=-1.0, maxval=1.0) * lim
        return jnp.concatenate([g, a])

    def policy_obs(self, state, key):
        """(history (HIST*93,), goal) seen by the actor."""
        return state.hist.reshape(-1), self.goal(state, key)

    def critic_obs(self, state):
        dx = state.dx
        rq = self.ref_qpos[state.frame]
        qy_inv = mjm.quat_inv(_yaw_quat(dx.qpos[3:7]))
        root_err = jnp.concatenate([dx.qpos[2:3] - rq[2:3], mjm.rotate(dx.qvel[:3], qy_inv)])
        xi = state.xi
        xi_n = jnp.concatenate([((xi.friction - 0.75) / 0.45)[None], xi.enc_bias / 0.01, xi.com / 0.05])
        g = self.goal_feat[jnp.minimum(state.frame + jnp.arange(FUTURE), self.ref_last[state.frame])].reshape(-1)
        g = jnp.concatenate([g, self._anchor_rel(dx, state.frame)[GOAL_ANCHOR]])
        obs = [self.proprio(dx, xi, state.prev_a), g, root_err, self._anchor_rel(dx, state.frame), xi_n]
        return jnp.concatenate(obs)

    # ---------------------------------------------------------------- reset / step
    def reset(self, key, frame):
        """Start an episode at reference `frame` (single env)."""
        kq, kv, kj, kx, kn, ks = jax.random.split(key, 6)
        qpos, qvel = self.ref_qpos[frame], self.ref_qvel[frame]
        if self.cfg.reset_noise:
            u = lambda k, lim: jax.random.uniform(k, (len(lim),), minval=-jnp.asarray(lim), maxval=jnp.asarray(lim))  # noqa: E731
            k1, k2 = jax.random.split(kq)
            qpos = qpos.at[:3].add(u(k1, [0.05, 0.05, 0.01]))
            rpy = u(k2, [0.1, 0.1, 0.2])
            dq = mjm.axis_angle_to_quat(rpy / (jnp.linalg.norm(rpy) + 1e-9), jnp.linalg.norm(rpy))
            qpos = qpos.at[3:7].set(mjm.quat_mul(dq, qpos[3:7]))
            qvel = qvel.at[:6].add(u(kv, [0.5, 0.5, 0.2, 0.52, 0.52, 0.78]))
            qpos = qpos.at[7:].set(jnp.clip(qpos[7:] + u(kj, [0.1] * self.nu), self.soft_lo, self.soft_hi))
        xi = self.sample_xi(kx)
        dx = self.d0.replace(qpos=qpos, qvel=qvel, qacc_warmstart=jnp.zeros_like(self.d0.qacc))
        dx = self._post(self.model_for(xi), dx)
        p0 = self.proprio(dx, xi, jnp.zeros(self.nu))
        if self.cfg.obs_noise:
            p0 = p0 + self._noise(kn)
        hist = jnp.tile(p0, (HIST, 1))
        return State(dx, jnp.int32(frame), jnp.int32(0), hist, jnp.zeros(self.nu), xi, ks)

    def _post(self, mx, dx):
        dx = mjx.kinematics(mx, dx)
        dx = smooth.com_pos(mx, dx)
        return smooth.com_vel(mx, dx)

    def action_base(self, frame=None):
        """The setpoint an action of 0 commands: q0, or with residual_action the joint angles of the reference frame
        after `frame` (the state's reference frame), min(frame + 1, last[frame]), the frame the step moves it to."""
        if self.cfg.residual_action:
            return self.ref_qpos[jnp.minimum(frame + 1, self.ref_last[frame]), 7:]
        return self.q0

    def ctrl_of(self, action, frame=None):
        """Joint setpoints: q0 + scale * action, or with residual_action the next reference frame's joint angles
        + scale * action (frame: the state's reference frame); clipped to the setpoint range."""
        return jnp.clip(self.action_base(frame) + self.action_scale * action, self.ctrl_lo, self.ctrl_hi)

    def action_change(self, action, prev_action, prev=None):
        """The action-rate term's argument (per joint, action units). Absolute actions: a_t - a_{t-1} (both spaces).
        Residual actions with action_rate_space="setpoint": the change of the commanded (unclipped) setpoint
        u_t = base_t + scale * a_t over scale, (u_t - u_{t-1}) / scale, with u_{-1} = q0 after a reset (as absolute
        actions have, prev_a = 0 there); the reward is then the absolute-action one of the same setpoints. With "raw":
        a_t - a_{t-1} of the residual outputs.

        prev: the State the action was taken in (whose prev_a is prev_action), at reference frame f_t. The action's
        base is action_base(f_t) = ref[min(f_t + 1, last[f_t])]; the previous action's, taken at f_{t-1}, is
        ref[min(f_{t-1} + 1, last[f_{t-1}])], which is f_t itself (the step advanced f_{t-1} to it, held at a clip
        end), so ref[f_t]. prev is a reset state when prev.ep == 0, or when prev_action is exactly 0 and the history is
        a reset's (every row the same: reset() tiles one proprioception), since the training's initial and collector
        states are reset states with random episode ages. A policy output of exactly 0 mid-episode (replaying the
        reference) is not a reset: its history rows differ."""
        if not (self.cfg.residual_action and self.cfg.action_rate_space == "setpoint"):
            return action - prev_action
        assert prev is not None, "action_rate_space='setpoint' with residual actions: pass prev (the pre-step State)"
        f = prev.frame
        base = self.action_base(f)
        fresh = (prev.ep == 0) | (jnp.all(prev_action == 0) & jnp.all(prev.hist == prev.hist[:1]))
        return jnp.where(fresh, action + (base - self.q0) / self.action_scale,
                         action - prev_action + (base - self.ref_qpos[f, 7:]) / self.action_scale)

    def finish_physics(self, state, action, qpos, qvel, warm):
        """The next state (reference frame advanced) given the physics result (qpos, qvel, warm start) of a batched
        backend such as warp_physics.WarpPhysics; call it per env under vmap."""
        mx = self.model_for(state.xi)
        key, kn, _ = jax.random.split(state.key, 3)
        dx = state.dx.replace(qpos=qpos, qvel=qvel, qacc_warmstart=warm)
        return self._finish(state, action, mx, dx, key, kn)

    def finish_physics_kin(self, state, action, qpos_n, qvel, warm, xpos, xquat, cvel, subtree_com):
        """finish_physics given the body kinematics too (the engine's step with kinematics=True, MJX's semantics: qpos_n
        carries the renormalised free-joint quaternion): no MJX kinematics here. Only the Data fields the environment
        reads are set (qpos, qvel, xpos, xquat, cvel, subtree_com); the others keep their values."""
        key, kn, _ = jax.random.split(state.key, 3)
        dx = state.dx.replace(qpos=qpos_n, qvel=qvel, qacc_warmstart=warm, xpos=xpos, xquat=xquat, cvel=cvel,
                              subtree_com=subtree_com)
        return self._finish(state, action, None, dx, key, kn, post=False)

    def _finish(self, state, action, mx, dx, key, kn, post=True):
        if post:
            dx = self._post(mx, dx)
        frame = jnp.minimum(state.frame + 1, self.ref_last[state.frame])
        p = self.proprio(dx, state.xi, action)
        if self.cfg.obs_noise:
            p = p + self._noise(kn)
        hist = jnp.concatenate([state.hist[1:], p[None]], 0)
        return State(dx, frame, state.ep + 1, hist, action, state.xi, key)

    # ---------------------------------------------------------------- reward / termination
    def _body_vel(self, dx):
        ids = self.body_ids
        ang = dx.cvel[ids, :3]
        lin = dx.cvel[ids, 3:] - jnp.cross(dx.xpos[ids] - dx.subtree_com[self.root_com_id], ang)
        return lin, ang

    def _track(self, dx, f):
        """The tracked bodies' poses and velocities, the reference re-anchored to the robot anchor in xy and yaw
        (BeyondMimic) and the reference velocities."""
        ids, k = self.body_ids, self.anchor_k
        pos, quat = dx.xpos[ids], dx.xquat[ids]
        lin, ang = self._body_vel(dx)
        rp, rq = self.ref_body_pos[f], self.ref_body_quat[f]
        ra_p, ra_q, ma_p, ma_q = pos[k], quat[k], rp[k], rq[k]
        d_pos = jnp.concatenate([ra_p[:2], ma_p[2:]])
        d_ori = _yaw_quat(mjm.quat_mul(ra_q, mjm.quat_inv(ma_q)))
        rel_p = d_pos + jax.vmap(lambda p: mjm.rotate(p - ma_p, d_ori))(rp)
        rel_q = jax.vmap(lambda q: mjm.quat_mul(d_ori, q))(rq)
        ref_lin, ref_ang = self.ref_body_lin[f], self.ref_body_ang[f]
        return pos, quat, lin, ang, rel_p, rel_q, ref_lin, ref_ang, ra_p, ra_q, ma_p

    def reward(self, state, action, prev_action, prev=None):
        """Table A.2 reward (x dt) for being at state.dx at reference frame state.frame. prev: the State the action was
        taken in (needed with residual actions and action_rate_space="setpoint", see action_change)."""
        dx, f = state.dx, state.frame
        pos, quat, lin, ang, rel_p, rel_q, ref_lin, ref_ang, ra_p, ra_q, ma_p = self._track(dx, f)
        ma_q = self.ref_body_quat[f, self.anchor_k]
        err = dict(
            root_pos=jnp.sum((ma_p - ra_p) ** 2),
            root_ori=_quat_angle_sq(ma_q, ra_q),
            body_pos=jnp.mean(jnp.sum((rel_p - pos) ** 2, -1)),
            body_ori=jnp.mean(jax.vmap(_quat_angle_sq)(rel_q, quat)),
            body_lin_vel=jnp.mean(jnp.sum((ref_lin - lin) ** 2, -1)),
            body_ang_vel=jnp.mean(jnp.sum((ref_ang - ang) ** 2, -1)),
        )
        terms = {n: DT * W[n] * jnp.exp(-e / STD[n] ** 2) for n, e in err.items()}
        q = dx.qpos[7:]
        terms["action_rate"] = DT * W["action_rate"] * jnp.sum(self.action_change(action, prev_action, prev) ** 2)
        terms["joint_limit"] = DT * W["joint_limit"] * jnp.sum(
            jnp.maximum(q - self.soft_hi, 0.0) + jnp.maximum(self.soft_lo - q, 0.0))
        return sum(terms.values()), terms

    max_reward = DT * (W["body_pos"] + W["body_ori"] + W["body_lin_vel"] + W["body_ang_vel"]
                       + W["root_pos"] + W["root_ori"])

    def terminated(self, state):
        dx, rq = state.dx, self.ref_qpos[state.frame]
        dz = jnp.abs(dx.qpos[2] - rq[2])
        dg = jnp.abs(_gravity_b(dx.qpos[3:7])[2] - _gravity_b(rq[3:7])[2])
        bad = ~jnp.all(jnp.isfinite(dx.qpos)) | ~jnp.all(jnp.isfinite(dx.qvel))
        return (dz > self.cfg.term_height) | (dg > self.cfg.term_tilt) | bad

    def hazard_features(self, state):
        """Margins to the termination thresholds and their rates (smooth in the state):
        [0.25 - dz, d dz/dt, 0.8 - dtilt, d dtilt/dt], dz = |z - z_ref|, dtilt = |g_z - g_z,ref|."""
        dx, f = state.dx, state.frame
        rq, rv = self.ref_qpos[f], self.ref_qvel[f]
        ez = dx.qpos[2] - rq[2]
        rate_z = jnp.sign(ez) * (dx.qvel[2] - rv[2])
        g, gr = _gravity_b(dx.qpos[3:7]), _gravity_b(rq[3:7])
        # body-frame image of a fixed world vector: d/dt (R^T v) = (R^T v) x omega_body
        gdot, grdot = jnp.cross(g, dx.qvel[3:6])[2], jnp.cross(gr, rv[3:6])[2]
        et = g[2] - gr[2]
        rate_t = jnp.sign(et) * (gdot - grdot)
        return jnp.stack([self.cfg.term_height - jnp.abs(ez), rate_z, self.cfg.term_tilt - jnp.abs(et), rate_t])

    @staticmethod
    def compact(state):
        """The few arrays that determine a State (works batched): enough to restart from it."""
        dx = state.dx
        return dict(qpos=dx.qpos, qvel=dx.qvel, qacc_warmstart=dx.qacc_warmstart, frame=state.frame,
                    ep=state.ep, hist=state.hist, prev_a=state.prev_a, xi=state.xi, key=state.key)

    def restore(self, c):
        """Rebuild a full single-env State from compact(); derived Data fields are recomputed."""
        dx = self.d0.replace(qpos=c["qpos"], qvel=c["qvel"], qacc_warmstart=c["qacc_warmstart"])
        dx = self._post(self.model_for(c["xi"]), dx)
        return State(dx, c["frame"], c["ep"], c["hist"], c["prev_a"], c["xi"], c["key"])

    def clip_ended(self, state):
        return state.frame >= self.ref_last[state.frame]
