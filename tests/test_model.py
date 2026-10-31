"""The G1 model built from the pinned Menagerie checkout (skipped without it; CPU)."""

import mujoco
import numpy as np
import pytest

from nablamimic.env import EnvConfig, build_model
from nablamimic.motion import SCENE

pytestmark = pytest.mark.skipif(not SCENE.exists(), reason=f"no Menagerie checkout at {SCENE} (scripts/setup_menagerie.sh)")


def test_mjlab_g1():
    m, kp = build_model(EnvConfig())
    assert (m.nq, m.nv, m.nu, m.ngeom, m.npair) == (36, 35, 29, 69, 502)
    assert not np.any(m.geom_type == mujoco.mjtGeom.mjGEOM_BOX)
    feet = [i for i in range(m.npair) if "_foot" in m.pair(i).name and m.pair(i).name.endswith("_floor")]
    assert len(feet) == 14
    assert m.opt.iterations == 10 and m.opt.timestep == 0.005 and np.all(kp > 0)
