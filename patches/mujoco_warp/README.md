# Patched MuJoCo Warp

Training on MuJoCo Warp (`--physics warp`) and every evaluation use MuJoCo Warp with its analytic adjoint, plus four
small patches that add backward-pass switches. `scripts/setup_mujoco_warp.sh` builds the tree:

1. Base: [`etaoxing/mujoco_warp`](https://github.com/etaoxing/mujoco_warp) at
   `357a75d60a56d67d476942a1b6e54b3045ee8e87` (branch `adjoint`; MuJoCo Warp 3.12 with an analytic adjoint, the code of
   [google-deepmind/mujoco_warp#1535](https://github.com/google-deepmind/mujoco_warp/pull/1535)).
2. [etaoxing/mujoco_warp#1](https://github.com/etaoxing/mujoco_warp/pull/1), two commits on top of the base, fetched
   from GitHub and not shipped here (the patch comments cite them by commit): `c8c8890` (constraint adjoint: skip a
   pyramidal contact only when every edge row is satisfied) and `4af093f` (forward adjoint: zero the multiplier on dofs
   whose actuator force is clamped). The script checks out `4af093f` by its full hash.
3. The patches `0001`-`0004` in this directory, applied in order with `git apply`. Each switch is read from an
   environment variable when `mujoco_warp` is imported and is off by default (stock behaviour). **The forward pass is
   unchanged in every setting**; only gradients are affected.

| Patch | Switch | Effect when on |
|---|---|---|
| `0001` | `MJW_CLAMP_KV_FIX=1` | drops the actuator damping of dofs whose actuator force is clamped from the qvel VJP (the clamped force does not depend on qvel) |
| `0001` | `MJW_ADJ_NORMAL=k`, `MJW_ADJ_NORMAL_EPS` (1e-3 m) | derivative of the sphere / capsule self-contact normal v/\|v\|: 0 exact, 1 held constant, 2 regularized 1/max(\|v\|, eps), 3 held constant where \|v\| < eps |
| `0002` | `MJW_ADJ_PARALLEL=1`, `MJW_ADJ_PARALLEL_EPS` (1.2e-3) | capsule-capsule contacts with nearly parallel axes (sin^2 of the angle below eps) keep their segment parameters constant in the backward |
| `0003` | `MJW_CLAMP_SOFT=delta` | soft surrogate derivative of the actuator force clamp (not used by the provided configurations) |
| `0004` | `MJW_IFT_SOLVE=k` | 1: the implicit-function solve of the adjoint scales its right-hand side by a power of two, so that the corner pivot of the augmented Cholesky factorization does not become NaN for large right-hand sides (bitwise the stock result wherever that is finite); 2: solve without the augmented border |

`nablamimic.mjw_options.configure()` writes all six variables before `mujoco_warp` is imported, so nothing is inherited
from the shell. The MuJoCo Warp training configurations (`configs/*_mjw.args`, `configs/reported/*_mjw.args`) use
`MJW_ADJ_NORMAL=3`, `MJW_ADJ_PARALLEL=1`, `MJW_IFT_SOLVE=1` (and `MJW_CLAMP_KV_FIX=0`, `MJW_CLAMP_SOFT=0`): the same
gradient conventions as NablaEngine (the sphere / capsule self-contact normal held constant where |v| < 1 mm, the
segment parameters of near-parallel capsules held constant, the actuator damping of clamped dofs kept in the qvel VJP),
plus the scaled right-hand side of patch `0004` in the adjoint's IFT solve. The two simulators' forward passes are
separate implementations, so the results of runs on them can differ (see "Expected results" in the repository's
README). Evaluation runs the forward only.

`tree.sha256` holds the hash of the patched package sources (`mujoco_warp/**/*.py`); the setup script checks it.
MuJoCo Warp is not bitwise reproducible from run to run on the GPU (float atomics), with or without the patches.

The patched files carry a modification notice. MuJoCo Warp is licensed under the Apache License 2.0
(Copyright 2025 The Newton Developers); the patches are distributed under the same licence. See `THIRD_PARTY_NOTICES.md`
in the repository root.
