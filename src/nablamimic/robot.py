"""The G1 as mjlab models it (mjlab.asset_zoo.robots.unitree_g1), built from MuJoCo Menagerie's MJX G1.

The foot capsules, contact pairs and exclusions, solref / solimp, joint parameters and the setpoint range below are
transcribed from mjlab (https://github.com/mujocolab/mjlab, Apache-2.0, Copyright 2025 The mjlab Developers); the
robot itself is Menagerie's unitree_g1 (https://github.com/google-deepmind/mujoco_menagerie, BSD-3-Clause, Unitree
Robotics), fetched by scripts/setup_menagerie.sh. See THIRD_PARTY_NOTICES.md.

The two share bodies, masses, inertias, joints, armature, PD gains and torque limits; they differ in:
  - feet: mjlab's seven thin capsules per sole instead of two toe capsules, one sole capsule and a sole box;
  - contacts: mjlab lets every pair of collision geoms on different bodies collide (contype / conaffinity 1), except
    parent-child bodies and four excluded body pairs, with solref 0.02 everywhere (Menagerie's foot-floor pairs use
    0.008) and MuJoCo's mixing rules (condim and friction: the larger of the two geoms'); here the same set is written
    as explicit pairs, so friction randomization of the foot-floor pairs and the explicit-pair code paths still apply;
  - joints: no frictionloss (Menagerie 0.1 N m) and default limit solref / solimp (Menagerie 0.008 and (0, 0.99, 0.01));
  - actuators: setpoints may go past the joint range by tau_max / kp on each side (mjlab's ctrlrange);
    Menagerie's ctrlrange is the joint range.
"""

import itertools
import re

import mujoco

# mjlab's sole capsules (radius, half length), position and orientation in the ankle-roll link frame, k = 1..7
FOOT = {
    "left": [((0.01, 0.025005), (0.075, -0.0265, -0.025), (0.707107, -0.014139, 0.706965, 0.0)),
             ((0.01, 0.0835), (0.0395, -0.018, -0.025), (0.707107, 0.0, -0.707107, 0.0)),
             ((0.01, 0.091), (0.039, -0.01, -0.025), (0.707107, 0.0, -0.707107, 0.0)),
             ((0.01, 0.093), (0.039, 0.0, -0.025), (0.707107, 0.0, -0.707107, 0.0)),
             ((0.01, 0.091), (0.039, 0.01, -0.025), (0.707107, 0.0, -0.707107, 0.0)),
             ((0.01, 0.0835), (0.0395, 0.018, -0.025), (0.707107, 0.0, -0.707107, 0.0)),
             ((0.01, 0.025), (0.075, 0.026, -0.025), (0.707107, 0.0, 0.707107, 0.0))],
    "right": [((0.01, 0.025), (0.075, -0.026, -0.025), (0.707107, 0.0, 0.707107, 0.0)),
              ((0.01, 0.0835), (0.0395, -0.018, -0.025), (0.707107, 0.0, -0.707107, 0.0)),
              ((0.01, 0.091), (0.039, -0.01, -0.025), (0.707107, 0.0, -0.707107, 0.0)),
              ((0.01, 0.093), (0.039, 0.0, -0.025), (0.707107, 0.0, -0.707107, 0.0)),
              ((0.01, 0.091), (0.039, 0.01, -0.025), (0.707107, 0.0, -0.707107, 0.0)),
              ((0.01, 0.0835), (0.0395, 0.018, -0.025), (0.707107, 0.0, -0.707107, 0.0)),
              ((0.01, 0.025), (0.075, 0.026, -0.025), (0.707107, 0.0, 0.707107, 0.0))],
}
FOOT_FRICTION = 0.6  # mjlab's sole geoms; the floor (1.0) wins foot-floor pairs, foot-foot pairs keep 0.6
EXCLUDE = {("pelvis", "left_hip_roll_link"), ("pelvis", "right_hip_roll_link"),
           ("left_elbow_link", "left_wrist_pitch_link"), ("right_elbow_link", "right_wrist_pitch_link")}
SOLREF, SOLIMP = (0.02, 1.0), (0.9, 0.95, 0.001, 0.5, 2.0)


def build(scene_xml):
    """Compiled model of the mjlab-matched G1 scene (pairs named <geom1>__<geom2>, foot-floor ones <side>_foot<k>_floor)."""
    return finish(build_spec(scene_xml).compile())


def finish(m):
    """The joint parameters mjlab leaves at MuJoCo's defaults (they live in the model arrays, not the spec)."""
    m.dof_frictionloss[:] = 0.0
    m.jnt_solref[:] = SOLREF
    m.jnt_solimp[:] = SOLIMP
    return m


def build_spec(scene_xml):
    """MjSpec of the mjlab-matched scene: mjlab's feet and contact set; joint parameters are set by finish()."""
    spec = mujoco.MjSpec.from_file(str(scene_xml))
    for g in list(spec.geoms):
        if re.fullmatch(r"(left|right)_foot(\d|_box)_collision", g.name or ""):
            spec.delete(g)
    for side, geoms in FOOT.items():
        body = spec.body(f"{side}_ankle_roll_link")
        for k, (size, pos, quat) in enumerate(geoms, 1):
            body.add_geom(name=f"{side}_foot{k}_collision", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[*size, 0.0],
                          pos=pos, quat=quat, contype=0, conaffinity=0, group=3, condim=3,
                          friction=[FOOT_FRICTION, 0.005, 0.0001])
    for p in list(spec.pairs):
        spec.delete(p)
    m0 = spec.compile()
    geoms = [g for g in range(m0.ngeom) if (m0.geom(g).name.endswith("_collision") or m0.geom(g).name == "floor")]
    body = lambda g: int(m0.geom_bodyid[g])  # noqa: E731
    bname = lambda b: m0.body(b).name  # noqa: E731
    foot = lambda g: "_foot" in m0.geom(g).name  # noqa: E731

    def add(g1, g2, name):
        f = max(1.0 if not foot(g1) else FOOT_FRICTION, 1.0 if not foot(g2) else FOOT_FRICTION)
        if m0.geom(g1).name == "floor" or m0.geom(g2).name == "floor":
            f = 1.0
        condim = 3 if (foot(g1) or foot(g2) or "floor" in (m0.geom(g1).name, m0.geom(g2).name)) else 1
        spec.add_pair(name=name, geomname1=m0.geom(g1).name, geomname2=m0.geom(g2).name, condim=condim,
                      friction=[f, f, 0.005, 0.0001, 0.0001], solref=list(SOLREF), solimp=list(SOLIMP))

    floor = next(g for g in geoms if m0.geom(g).name == "floor")
    robot = [g for g in geoms if g != floor]
    for g in robot:  # floor pairs first, named like Menagerie's so the foot-floor pairs are found the same way
        n = m0.geom(g).name
        add(g, floor, n.replace("_collision", "_floor") if foot(g) else f"{n}__floor")
    for g1, g2 in itertools.combinations(robot, 2):
        b1, b2 = body(g1), body(g2)
        if b1 == b2 or m0.body_parentid[b1] == b2 or m0.body_parentid[b2] == b1:
            continue
        if (bname(b1), bname(b2)) in EXCLUDE or (bname(b2), bname(b1)) in EXCLUDE:
            continue
        add(g1, g2, f"{m0.geom(g1).name}__{m0.geom(g2).name}")
    return spec


def widen_ctrlrange(m, kp):
    """mjlab's setpoint range: the joint range extended by tau_max / kp on each side."""
    for i in range(m.nu):
        j = m.actuator_trnid[i, 0]
        ext = m.jnt_actfrcrange[j, 1] / kp[i]
        m.actuator_ctrlrange[i] = (m.jnt_range[j, 0] - ext, m.jnt_range[j, 1] + ext)
    return m
