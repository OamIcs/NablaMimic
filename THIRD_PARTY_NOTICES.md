# Third-party notices

This repository is licensed under the Apache License 2.0 (`LICENSE`, `NOTICE`). It contains, adapts or re-implements
the third-party material below, and its setup scripts fetch further third-party code and data that are not part of the
repository. Each item names where it is used.

## Code and models

### MuJoCo Warp (Apache License 2.0, Copyright 2025 The Newton Developers)

<https://github.com/google-deepmind/mujoco_warp>; the analytic-adjoint fork <https://github.com/etaoxing/mujoco_warp>
(commit `357a75d`, pull request #1 at `4af093f`) is fetched by `scripts/setup_mujoco_warp.sh`, not shipped.

- `patches/mujoco_warp/0001-0004`: modifications of `mujoco_warp/_src/collision_primitive_core.py`,
  `mujoco_warp/_src/forward_adjoint.py` and `mujoco_warp/_src/adjoint.py` (backward-pass switches); each patched file
  carries a modification notice. Distributed under the Apache License 2.0.
- `src/nablamimic/g1_fast/smooth_gen.py` generates a Warp kernel whose formulas follow `mujoco_warp/_src/smooth.py`.
- `src/nablamimic/g1_fast/adjoint_patch.py` replaces `smooth_qvel_backward` of `mujoco_warp/_src/forward_adjoint.py` at
  run time with a version that follows it and calls its kernels.
- `src/nablamimic/warp_physics.py` drives MuJoCo Warp as a library.

### MuJoCo and MJX (Apache License 2.0, Google DeepMind)

<https://github.com/google-deepmind/mujoco>; used as libraries (pinned `mujoco==3.17.0`, `mujoco-mjx==3.17.0`), including
the internal modules `mujoco.mjx._src.math` and `mujoco.mjx._src.smooth`.

### mjlab (Apache License 2.0, Copyright 2025 The mjlab Developers)

<https://github.com/mujocolab/mjlab>. `src/nablamimic/robot.py` transcribes the G1 model settings of
`mjlab.asset_zoo.robots.unitree_g1`: the seven sole capsules per foot, the contact pairs and excluded body pairs, solref
/ solimp, the joint parameters and the widened setpoint range. `src/nablamimic/env.py` uses mjlab's PD-gain rule
(kp = armature x (2 pi 10 Hz)^2, kd = 4 x armature x 2 pi 10 Hz) and its knees-bent keyframe.

### MuJoCo Menagerie, Unitree G1 (BSD 3-Clause License, Unitree Robotics)

<https://github.com/google-deepmind/mujoco_menagerie>, `unitree_g1/` at commit `f3b6013`; fetched by
`scripts/setup_menagerie.sh`, not shipped. Licence of `unitree_g1/`:

```
Copyright (c) 2016-2023 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics")
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### rsl_rl (BSD 3-Clause License, ETH Zurich and NVIDIA)

<https://github.com/leggedrobotics/rsl_rl>. `src/nablamimic/ppo.py` is an independent JAX implementation that follows
rsl_rl's PPO update, its adaptive-KL learning-rate schedule and the motion-tracking hyperparameters used with it; no
rsl_rl source is copied (the differences are listed in the module docstring). rsl_rl's licence:

```
Copyright (c) 2026, ETH Zurich
Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### BeyondMimic, whole_body_tracking (MIT License, The Isaac Lab Project Developers)

<https://github.com/HybridRobotics/whole_body_tracking>. `src/nablamimic/ppo.py` re-implements its adaptive sampling
(`MotionCommand._adaptive_sampling`); `src/nablamimic/env.py` follows its tracking reward terms and anchor-relative
observation. No source is copied. Its licence:

```
Copyright (c) 2024, The Isaac Lab Project Developers.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### GR00T-WholeBodyControl / SONIC (source code: Apache License 2.0, Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES)

<https://github.com/NVlabs/GR00T-WholeBodyControl>. `src/nablamimic/bones/convert.py` copies the BONES-SEED motion-name
filter list of `gear_sonic/data_process/filter_and_copy_bones_data.py` (plus one entry), marked in the code.

### NablaEngine

A separate package (<https://github.com/OamIcs/NablaEngine>, distribution `nablaengine`, import package
`nablaengine`, pinned to its release v0.1.0), licensed under the Apache License 2.0 with its
own notices; installed by `scripts/setup_env.sh`, not shipped here.

## Data (never redistributed)

- **LAFAN1** (Ubisoft La Forge), licensed CC BY-NC-ND 4.0, and its Unitree G1 retargets in the Hugging Face dataset
  `lvhaidong/LAFAN1_Retargeting_Dataset`: downloaded by `scripts/download_lafan1.sh`. The repository contains only
  their SHA-256 checksums (`scripts/lafan1_g1.sha256`). The dataset's repository (at the revision the script pins)
  declares the BSD 3-Clause License in its `LICENSE` file ("Copyright (c) 2016-2024 HangZhou YuShu TECHNOLOGY CO.,LTD.
  ("Unitree Robotics")"), and its dataset card states that LAFAN1 is licensed under the Creative Commons
  Attribution-NonCommercial-NoDerivatives 4.0 International Public License; the card's metadata names no licence.
- **BONES-SEED** (`bones-studio/seed` on Hugging Face, gated), licensed for academic, non-commercial use: downloaded and
  converted by `scripts/download_bones_seed.sh` with the user's own access.

## Methods followed

The code follows, and cites in its docstrings, these published works: SuperMimic (Liao et al. 2026, arXiv:2610.28480;
environment, actor, reset sampling, BONES-SEED filters and benchmark), SHAC (Xu et al. 2022, arXiv:2204.07137;
baseline), PODS (Zamora Mora et al. 2021, "PODS: Policy Optimization via Differentiable Simulation"; action targets),
BeyondMimic (Liao et al. 2025, arXiv:2508.08241; reward, observations, adaptive sampling) and PPO (Schulman et al. 2017,
arXiv:1707.06347).
