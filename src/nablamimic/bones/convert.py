"""BONES-SEED G1 retargets -> the sharded 50 Hz reference library of --motion-lib (motion_lib.ShardedLibrary).

BONES-SEED (bones-studio/seed on Hugging Face) is gated and licensed for academic, non-commercial use; it is never
redistributed (scripts/download_bones_seed.sh fetches and converts it). The G1 CSVs (g1/csv/{date}/{motion}.csv, 120 fps)
have a header row: Frame, root_translateX/Y/Z [cm, Z up], root_rotateX/Y/Z [deg, Euler angles], then <joint>_dof [deg]
for the 29 joints, named as in the Menagerie model. Each clip is converted to the LAFAN1 row layout (root xyz [m],
quaternion xyzw, joints [rad] in actuator order) and processed exactly like a LAFAN1 clip by motion.process_clip (50 Hz,
floor alignment, finite-difference velocities, poses and velocities of the tracked bodies).

Euler convention (the dataset does not state it): zyx composes R = Rz Ry Rx (three.js order 'ZYX', what the dataset's
viewer renders), xyz composes R = Rx Ry Rz. The reported library uses zyx (the default). With --euler auto the
conversion first converts --check-clips random clips with both and keeps the one whose planted feet slide less (median
over clips of the mean horizontal speed of the lowest foot point while it is within 2 cm of the floor); the comparison
is printed and stored in index.json. --check-only stops after that comparison.

Output: <out>/shard_XXXXX.npz, --shard-clips clips each, drawn in a fixed random order so that every shard is a random
sample of the library (concatenated float32 per-frame arrays qpos, qvel, body_pos, body_quat, body_lin_vel,
body_ang_vel; clip starts and lengths; names) and <out>/index.json (per clip: shard, offset, length, source frames,
joint-limit excess, the name-filter flag (NAME_FILTER), the feasibility flags and the floor shift (feasibility()), and the
metadata columns package, category, is_mirror when --meta is given). Every clip is converted; the filters only flag
(training and the benchmark leave the flagged clips out). Clips that fail to parse or have fewer than 2 frames are
listed under "skipped".

  python -m nablamimic.bones.convert --src DATA/g1/csv --meta DATA/metadata/seed_metadata_v004.csv --out DATA/lib50
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import random
from pathlib import Path

import mujoco
import numpy as np

from ..env import BM_BODIES
from ..motion import FOOT_GEOMS, SCENE, _resample, process_clip

FPS_IN = 120.0
KEYS = ("qpos", "qvel", "body_pos", "body_quat", "body_lin_vel", "body_ang_vel")
# The BONES-SEED name filter of SONIC (https://github.com/NVlabs/GR00T-WholeBodyControl,
# gear_sonic/data_process/filter_and_copy_bones_data.py, Apache-2.0; see THIRD_PARTY_NOTICES.md) plus "box": 129,673 of
# the 142,220 names pass, against 129,667 after SuperMimic's name filter (its Table A.7). The entries are motion names
# of the dataset (objects and scene interactions the training set leaves out).
NAME_FILTER = ("bed", "bike", "chair", "climb", "com_up_50cm", "sitting", "step_on", "seat", "table", "_sit_", "sit_",
               "ladder", "crutch", "_bed_", "_ride_", "scooter", "stepdown", "acrobatics_", "box_HSPU", "cartwheel",
               "50cm_box_", "on_box", "fall_from", "handstand_ff_", "on_1m", "form_box", "off_1m", "230m",
               "jump_over_obstacle_", "lift_crate_come_up_", "jump_to_shoulder_roll", "kozak_dance", "stair",
               "handstand", "box_jump", "monkey_jump", "safety_roll", "box_dips", "walking_on_edge", "push_obstacle", "box")
_M = None


def model():
    global _M
    if _M is None:
        _M = mujoco.MjModel.from_xml_path(str(SCENE))
    return _M


def _axis_quat(axis, ang):  # (n,) angles [rad] -> (n, 4) wxyz
    q = np.zeros((len(ang), 4))
    q[:, 0] = np.cos(ang / 2)
    q[:, 1 + axis] = np.sin(ang / 2)
    return q


def _qmul(a, b):
    w1, x1, y1, z1 = a.T
    w2, x2, y2, z2 = b.T
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], 1)


def read_raw(path, euler="zyx"):
    """One G1 CSV -> LAFAN1-layout rows (root xyz [m], quaternion xyzw, joints [rad]) at 120 fps."""
    m = model()
    with open(path) as f:
        cols = [c.strip() for c in f.readline().split(",")]
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    ix = {c: i for i, c in enumerate(cols)}
    pos = data[:, [ix["root_translateX"], ix["root_translateY"], ix["root_translateZ"]]] * 0.01
    rx, ry, rz = (np.deg2rad(data[:, ix[f"root_rotate{a}"]]) for a in "XYZ")
    qx, qy, qz = _axis_quat(0, rx), _axis_quat(1, ry), _axis_quat(2, rz)
    q = _qmul(_qmul(qz, qy), qx) if euler == "zyx" else _qmul(_qmul(qx, qy), qz)
    names = [m.joint(m.actuator_trnid[i, 0]).name for i in range(m.nu)]
    joints = np.deg2rad(data[:, [ix[n + "_dof"] for n in names]])
    return np.concatenate([pos, q[:, [1, 2, 3, 0]], joints], 1)


def feasibility(qpos):
    """SuperMimic's feasibility filter (its Appendix E) on the 50 Hz trajectory before the floor alignment, floor at
    z = 0. z_min(t) is the lowest body-link origin. Flags: elevated support (z_min > 0.20 m for >= 0.6 s), ground
    penetration (z_min < -0.12 m, or < -0.08 m for > 80% of the clip), floating feet (both feet' collision geoms more
    than 0.02 m above the floor, mj_geomDistance, for >= 0.5 s while the pelvis is upright (gravity z > 0.9), at
    standing height (> 0.6 m) and nearly still (horizontal speed < 0.2 m/s); SuperMimic does not give these last three
    gates, so they are this converter's choice). Also the floor shift the alignment applies, for reference."""
    m = model()
    d = mujoco.MjData(m)
    floor = next(i for i in range(m.ngeom) if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE)
    feet = [[m.geom(f"{s}_foot{i}_collision").id for i in (1, 2, 3)] for s in ("left", "right")]
    n = len(qpos)
    zmin, gap, low = np.empty(n), np.empty((n, 2)), np.empty(n)
    ft = np.zeros(6)
    for t, qp in enumerate(qpos):
        d.qpos[:] = qp
        mujoco.mj_kinematics(m, d)
        zmin[t] = d.xpos[1:, 2].min()
        for k, gs in enumerate(feet):
            gap[t, k] = min(mujoco.mj_geomDistance(m, d, g, floor, 1.0, ft) for g in gs)
        low[t] = gap[t].min()
    w, x, y, z = qpos[:, 3:7].T
    up = 1 - 2 * (x * x + y * y)  # world z of the pelvis z axis
    speed = np.r_[0.0, np.linalg.norm(np.diff(qpos[:, :2], axis=0), axis=1) * 50.0]

    def run(mask, frames):  # a run of at least `frames` consecutive true entries
        c = 0
        for v in mask:
            c = c + 1 if v else 0
            if c >= frames:
                return True
        return False

    stand = (up > 0.9) & (qpos[:, 2] > 0.6) & (speed < 0.2)
    return dict(elevated=run(zmin > 0.20, 30), penetration=bool(zmin.min() < -0.12 or (zmin < -0.08).mean() > 0.8),
                floating=run((gap.min(1) > 0.02) & stand, 25), zmin=round(float(zmin.min()), 4),
                floor_shift=round(float(np.percentile(low, 5)), 4))


def convert(path, euler="zyx"):
    m = model()
    raw = read_raw(path, euler)
    if len(raw) < 2:
        raise ValueError("fewer than 2 frames")
    feas = feasibility(_resample(raw, FPS_IN, 50.0))
    part = process_clip(m, raw, 50.0, FPS_IN, BM_BODIES)
    q = part[0][:, 7:]
    lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
    excess = float(np.max(np.maximum(q - hi, 0) + np.maximum(lo - q, 0)))
    return [x.astype(np.float32) for x in part], len(raw), excess, feas


def foot_stats(path, euler):
    """Mean horizontal speed [m/s] and depth below the floor [m] of the lowest foot point while it is within 2 cm of
    the floor, for one clip converted with this Euler convention."""
    m = model()
    qpos = convert(path, euler)[0][0].astype(np.float64)  # the full conversion, floor-aligned
    d = mujoco.MjData(m)
    low = np.empty((len(qpos), 3))
    for i, qp in enumerate(qpos):
        d.qpos[:] = qp
        mujoco.mj_kinematics(m, d)
        best = None
        for name in FOOT_GEOMS:
            g = m.geom(name).id
            half, r = m.geom_size[g, 1], m.geom_size[g, 0]
            axis = d.geom_xmat[g].reshape(3, 3)[:, 2]
            for e in (d.geom_xpos[g] + half * axis, d.geom_xpos[g] - half * axis):
                p = e - np.array([0.0, 0.0, r])
                if best is None or p[2] < best[2]:
                    best = p
        low[i] = best
    contact = low[1:, 2] < 0.02
    speed = np.linalg.norm(np.diff(low[:, :2], axis=0), axis=1) * 50.0
    return float(speed[contact].mean()) if contact.any() else np.nan, float(np.maximum(-low[:, 2], 0).mean())


def _work(job):
    i, path, euler = job
    try:
        part, n_src, excess, feas = convert(path, euler)
        return i, part, n_src, excess, feas, None
    except Exception as e:  # noqa: BLE001 - a bad file is skipped and reported, not fatal
        return i, None, 0, 0.0, None, repr(e)


def _check(job):
    path, = job
    return {e: foot_stats(path, e) for e in ("zyx", "xyz")}


def load_meta(path):
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out[r["filename"]] = dict(package=r.get("package"), category=r.get("category"),
                                      is_mirror=r.get("is_mirror") in ("True", "true", "1"))
    return out


def main():
    ap = argparse.ArgumentParser(prog="python -m nablamimic.bones.convert",
                                 description="Convert the BONES-SEED G1 CSVs into the sharded 50 Hz reference library "
                                             "of --motion-lib.")
    ap.add_argument("--src", required=True, help="the extracted g1/csv directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", default=None, help="seed_metadata_*.csv (package, category, is_mirror into the index)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--shard-clips", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0, help=">0: only the first N clips (in the shuffled order)")
    ap.add_argument("--euler", choices=("auto", "zyx", "xyz"), default="zyx",
                    help="Euler convention of the root rotation (zyx: the reported library; auto: pick by foot sliding)")
    ap.add_argument("--check-clips", type=int, default=200)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    paths = sorted(Path(args.src).rglob("*.csv"))
    print(f"{len(paths)} csv files under {args.src}", flush=True)
    euler, check = args.euler, {}
    if euler == "auto" or args.check_only:
        sample = random.Random(0).sample(paths, min(args.check_clips, len(paths)))
        with mp.Pool(args.workers) as pool:
            res = pool.map(_check, [(str(p),) for p in sample])
        for e in ("zyx", "xyz"):
            check[e] = dict(contact_speed=float(np.nanmedian([r[e][0] for r in res])),
                            depth_cm=float(np.median([r[e][1] for r in res])) * 100, clips=len(res))
            print(f"euler {e}: contact-foot speed median {check[e]['contact_speed']:.3f} m/s, "
                  f"depth below floor median {check[e]['depth_cm']:.2f} cm", flush=True)
        if args.check_only:
            return
        if euler == "auto":
            euler = min(check, key=lambda e: check[e]["contact_speed"])
            print(f"using euler {euler}", flush=True)
    random.Random(1).shuffle(paths)  # every shard a random sample of the library (training loads a few shards at a time)
    if args.limit:
        paths = paths[:args.limit]
    meta = load_meta(args.meta) if args.meta else {}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index, skipped = {}, {}
    for s0 in range(0, len(paths), args.shard_clips):
        shard = f"shard_{s0 // args.shard_clips:05d}"
        if (out / f"{shard}.npz").exists() and (out / f"{shard}.json").exists():  # resume
            index.update(json.loads((out / f"{shard}.json").read_text()))
            continue
        chunk = paths[s0:s0 + args.shard_clips]
        with mp.Pool(args.workers) as pool:
            res = sorted(pool.map(_work, [(i, str(p), euler) for i, p in enumerate(chunk)]), key=lambda r: r[0])
        parts, names, lens, sidx, off = [], [], [], {}, 0
        for i, part, n_src, excess, feas, err in res:
            name = chunk[i].stem
            if err is not None:
                skipped[name] = err
                continue
            parts.append(part)
            names.append(name)
            lens.append(len(part[0]))
            sidx[name] = dict(shard=shard, offset=off, length=len(part[0]), source_frames=n_src,
                              joint_limit_excess=round(excess, 4), name_filtered=any(
                                  k.lower() in name.lower() for k in NAME_FILTER), **feas, **meta.get(name, {}))
            off += len(part[0])
        arrays = {k: np.concatenate([p[j] for p in parts]) for j, k in enumerate(KEYS)}
        np.savez(out / f"{shard}.npz", names=np.array(names), lengths=np.array(lens),
                 starts=np.cumsum([0] + lens[:-1]), **arrays)
        (out / f"{shard}.json").write_text(json.dumps(sidx))
        index.update(sidx)
        print(f"{shard}: {len(names)} clips, {off} frames, {len(skipped)} skipped so far", flush=True)
    frames = sum(v["length"] for v in index.values())
    (out / "index.json").write_text(json.dumps(dict(fps=50.0, bodies=list(BM_BODIES), euler=euler, euler_check=check,
                                                    clips=index, skipped=skipped,
                                                    frames=frames, hours=frames / 50.0 / 3600)))
    print(f"done: {len(index)} clips, {frames} frames ({frames / 50 / 3600:.1f} h), {len(skipped)} skipped", flush=True)


if __name__ == "__main__":
    main()
