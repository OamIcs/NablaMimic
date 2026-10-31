"""LAFAN1 G1 retargeted clips -> MuJoCo-convention reference trajectories at the 50 Hz control rate.

Source: the G1 retargets of LAFAN1 (Ubisoft La Forge, CC BY-NC-ND 4.0) in the Hugging Face dataset
lvhaidong/LAFAN1_Retargeting_Dataset (g1/*.csv, 40 clips), fetched by scripts/download_lafan1.sh; never redistributed.
CSV rows (30 FPS): root xyz, root quaternion (x, y, z, w), then the 29 G1 joints in the same order as the Menagerie
model's actuators. Processed clips are cached as npz files on first use.

Paths (environment variables, defaults relative to the repository):
  LAFAN1_G1_DIR     the 40 CSVs (data/lafan1_g1)
  MOTION_CACHE_DIR  the processed clips (data/cache)
  MENAGERIE_DIR     the MuJoCo Menagerie checkout of scripts/setup_menagerie.sh (third_party/mujoco_menagerie)

  python -m nablamimic.motion          # build the cache of all clips (optional; training builds it too)
"""

import argparse
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MENAGERIE_DIR = Path(os.environ.get("MENAGERIE_DIR") or ROOT / "third_party/mujoco_menagerie")
SCENE = MENAGERIE_DIR / "unitree_g1/scene_mjx.xml"
LAFAN1_DIR = Path(os.environ.get("LAFAN1_G1_DIR") or ROOT / "data/lafan1_g1")

# Bodies whose root-relative positions enter the tracking reward.
TRACKED_BODIES = (
    "pelvis", "torso_link",
    "left_knee_link", "right_knee_link",
    "left_ankle_roll_link", "right_ankle_roll_link",
    "left_shoulder_roll_link", "right_shoulder_roll_link",
    "left_elbow_link", "right_elbow_link",
    "left_wrist_yaw_link", "right_wrist_yaw_link",
)
FOOT_GEOMS = tuple(f"{side}_foot{i}_collision" for side in ("left", "right") for i in (1, 2, 3))
CACHE_DIR = Path(os.environ.get("MOTION_CACHE_DIR") or ROOT / "data/cache")


@dataclass
class MotionLibrary:
    """Concatenated clips at the control rate.

    qpos/qvel/body_pos are indexed by a global frame id; `last[f]` is the last
    frame id of the clip containing f, so f + 1 > last[f] means the clip ended.
    """

    qpos: np.ndarray      # (F, nq)
    qvel: np.ndarray      # (F, nv)
    body_pos: np.ndarray  # (F, nb, 3) world positions of `bodies`
    last: np.ndarray      # (F,) int
    names: list
    starts: np.ndarray    # (nclip,) first frame id of each clip
    fps: float
    bodies: tuple = TRACKED_BODIES
    body_quat: np.ndarray = None     # (F, nb, 4) world orientations (wxyz)
    body_lin_vel: np.ndarray = None  # (F, nb, 3) world linear velocity of each body origin
    body_ang_vel: np.ndarray = None  # (F, nb, 3) world angular velocity


def _slerp(q0, q1, t):
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    small = theta < 1e-6
    s = np.where(small, 1.0, np.sin(theta))
    w0 = np.where(small, 1 - t, np.sin((1 - t) * theta) / s)
    w1 = np.where(small, t, np.sin(t * theta) / s)
    q = w0 * q0 + w1 * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _resample(raw, fps_in, fps_out):
    n = len(raw)
    t_out = np.arange(0.0, (n - 1) / fps_in, 1.0 / fps_out)
    x = t_out * fps_in
    i0 = np.minimum(np.floor(x).astype(int), n - 2)
    w = (x - i0)[:, None]
    pos = (1 - w) * raw[i0, :3] + w * raw[i0 + 1, :3]
    joints = (1 - w) * raw[i0, 7:] + w * raw[i0 + 1, 7:]
    quat_wxyz = raw[:, [6, 3, 4, 5]]
    quat = _slerp(quat_wxyz[i0], quat_wxyz[i0 + 1], w)
    return np.concatenate([pos, quat, joints], axis=1)


def _foot_min_z(m, d):
    zs = []
    for name in FOOT_GEOMS:
        g = m.geom(name).id
        half = m.geom_size[g, 1]
        axis = d.geom_xmat[g].reshape(3, 3)[:, 2]
        ends = d.geom_xpos[g][None] + np.array([[1.0], [-1.0]]) * half * axis[None]
        zs.append(ends[:, 2].min() - m.geom_size[g, 0])
    return min(zs)


def load_clip(m, path, fps_out=50.0, fps_in=30.0, bodies=TRACKED_BODIES):
    return process_clip(m, np.loadtxt(path, delimiter=","), fps_out, fps_in, bodies)


def process_clip(m, raw, fps_out=50.0, fps_in=30.0, bodies=TRACKED_BODIES):
    """raw rows (fps_in): root xyz, root quaternion (x, y, z, w), the 29 joints -> the reference at fps_out: qpos
    (floor-aligned), qvel, and the world poses and velocities of `bodies`."""
    qpos = _resample(raw, fps_in, fps_out)
    d = mujoco.MjData(m)

    # Align the floor: shift z so the 5th percentile of the lowest foot point is 0.
    low = np.empty(len(qpos))
    for i, q in enumerate(qpos):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        low[i] = _foot_min_z(m, d)
    qpos[:, 2] -= np.percentile(low, 5)

    dt = 1.0 / fps_out
    qvel = np.zeros((len(qpos), m.nv))
    for i in range(len(qpos)):
        a, b = max(i - 1, 0), min(i + 1, len(qpos) - 1)
        mujoco.mj_differentiatePos(m, qvel[i], (b - a) * dt, qpos[a], qpos[b])

    body_ids = [m.body(n).id for n in bodies]
    nb = len(body_ids)
    body_pos = np.empty((len(qpos), nb, 3))
    body_quat = np.empty((len(qpos), nb, 4))
    body_vel = np.empty((len(qpos), nb, 6))
    res = np.empty(6)
    for i in range(len(qpos)):
        d.qpos[:], d.qvel[:] = qpos[i], qvel[i]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_comVel(m, d)
        body_pos[i] = d.xpos[body_ids]
        body_quat[i] = d.xquat[body_ids]
        for k, b in enumerate(body_ids):
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, res, 0)
            body_vel[i, k] = res  # [angular, linear], world frame, at the body origin
    return qpos, qvel, body_pos, body_quat, body_vel[..., 3:], body_vel[..., :3]


def load_library(names, fps_out=50.0, model=None, bodies=TRACKED_BODIES):
    """Load and concatenate clips; per-clip results are cached under CACHE_DIR."""
    m = model or mujoco.MjModel.from_xml_path(str(SCENE))
    tag = f"{fps_out:g}hz_" + hashlib.md5(",".join(bodies).encode()).hexdigest()[:8]
    parts, starts, last, offset = [], [], [], 0
    for name in names:
        cache = CACHE_DIR / f"{name}_{tag}.npz" if model is None else None
        if cache is not None and cache.exists():
            z = np.load(cache)
            part = tuple(z[k] for k in ("qpos", "qvel", "body_pos", "body_quat", "body_lin_vel", "body_ang_vel"))
        else:
            part = load_clip(m, LAFAN1_DIR / f"{name}.csv", fps_out, bodies=bodies)
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez(cache, **dict(zip(("qpos", "qvel", "body_pos", "body_quat", "body_lin_vel", "body_ang_vel"), part)))
        parts.append(part)
        starts.append(offset)
        last.append(np.full(len(part[0]), offset + len(part[0]) - 1))
        offset += len(part[0])
    cat = lambda i: np.concatenate([p[i] for p in parts]).astype(np.float32)  # noqa: E731
    return MotionLibrary(
        qpos=cat(0), qvel=cat(1), body_pos=cat(2),
        last=np.concatenate(last),
        names=list(names),
        starts=np.asarray(starts),
        fps=fps_out,
        bodies=tuple(bodies),
        body_quat=cat(3), body_lin_vel=cat(4), body_ang_vel=cat(5),
    )


def all_clip_names():
    """Every CSV in LAFAN1_DIR, sorted (clip order sets the frame ids and the reset bins)."""
    return sorted(p.stem for p in LAFAN1_DIR.glob("*.csv"))


def main(argv=None):
    argparse.ArgumentParser(prog="python -m nablamimic.motion",
                            description="Build the 50 Hz reference cache of the LAFAN1 G1 clips in LAFAN1_G1_DIR "
                                        "(optional; training builds it on first use).").parse_args(argv)
    from .env import BM_BODIES
    names = all_clip_names()
    assert names, f"no CSVs in {LAFAN1_DIR} (scripts/download_lafan1.sh)"
    lib = load_library(names, bodies=BM_BODIES)
    print(f"{len(names)} clips, {len(lib.last)} frames at {lib.fps:g} Hz, cached in {CACHE_DIR}")


if __name__ == "__main__":
    main()
