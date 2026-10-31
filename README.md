# ∇Mimic

∇Mimic (NablaMimic) learns whole-body motion tracking for the Unitree G1 by backpropagating through an exact
differentiable simulator: the policy gradient comes from the simulator itself, with no learned world model.

- **Fast.** On LAFAN1 (40 clips, 2.45 h of motion), ∇Mimic reaches a normalized return of 0.82 after about 20 minutes on
  one RTX PRO 6000 and 0.875 after 3 hours, with no clip failed.
- **Stable.** Trains reliably from exact simulator gradients, where standard first-order methods such as SHAC break down.
- **One GPU, one command.** Training runs on [NablaEngine](https://github.com/OamIcs/NablaEngine), a fused
  differentiable engine for the G1, or on MuJoCo Warp.

Paper: *coming soon*. Project page: https://sailab.ai/nablamimic/

## Installation

Linux x86_64, an NVIDIA GPU with a CUDA 12 driver, Python 3.12, and the `nvcc` of a CUDA 12.x toolkit (12.8 or newer
for Blackwell GPUs) for the engine's one-time build.

```bash
git clone https://github.com/OamIcs/NablaMimic && cd NablaMimic
bash scripts/setup_env.sh    # .venv with pinned versions, the patched MuJoCo Warp, Menagerie's G1, NablaEngine, nablamimic
source .venv/bin/activate
```

### GPU memory

The configurations in `configs/` use about 50 GB of GPU memory (tested on an RTX PRO 6000 and an H100). On a smaller
GPU, lower `--replay-steps` / `--collect-envs` (the replay buffer) and `--num-envs` (BPTT windows per update) as needed;
the reported results use the defaults.

## Data

LAFAN1 retargeted to the G1 (non-commercial, CC BY-NC-ND 4.0), pinned and checked by SHA-256:

```bash
pip install huggingface_hub
bash scripts/download_lafan1.sh
```

BONES-SEED (gated, academic non-commercial licence) is optional: `bash scripts/download_bones_seed.sh DATA_DIR`.
Neither dataset may be redistributed.

## Train

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python -m nablamimic.train --config configs/ours.args --out runs/ours_s0
```

| Config | What |
|---|---|
| `configs/ours.args` | ∇Mimic on NablaEngine (recommended) |
| `configs/ours_mjw.args` | ∇Mimic on MuJoCo Warp |
| `configs/bones.args` | ∇Mimic on BONES-SEED (add `--motion-lib DATA_DIR/lib50`) |
| `configs/shac*.args`, `configs/ppo*.args` | SHAC and PPO baselines (PPO: `python -m nablamimic.ppo`) |
| `configs/reported/` | the configurations of the paper's comparisons and ablations |

Options after `--config` override the file (for example `--seed 1` or `--hours 1`). A stopped run continues with
`--resume latest` and the same `--config` and `--out`.

## Evaluate

```bash
bash scripts/evaluate_run.sh runs/ours_s0    # every LAFAN1 clip once, on MuJoCo Warp -> runs/ours_s0/eval_nominal.json
```

## License

Apache License 2.0 (`LICENSE`, `NOTICE`); third-party code, models and data: `THIRD_PARTY_NOTICES.md`.
