#!/usr/bin/env bash
# The Python environment: a venv with the tested versions (requirements-lock.txt), the CUDA 12 build of warp-lang 1.18.0,
# the patched MuJoCo Warp (scripts/setup_mujoco_warp.sh), MuJoCo Menagerie's G1 (scripts/setup_menagerie.sh),
# NablaEngine at the tested revision and this package (editable). Needs Python 3.12 ($PYTHON, default python3.12),
# git and network access. NablaEngine builds its CUDA library on first use and then needs the nvcc of a CUDA 12.x
# toolkit (12.8 or newer for sm_120) and a host C++ compiler, which this script does not install (the
# nvidia-cuda-nvcc-cu12 wheel of the lock file has ptxas, not nvcc). The engine takes the first that exists of $NVCC,
# nvcc on PATH, $CUDA_HOME/bin/nvcc, the nvcc of a pip package nvidia.cuda_nvcc and /usr/local/cuda/bin/nvcc, and
# builds for the first visible GPU's architecture unless $DK_ARCH is set (the engine's docs/configuration.md).
#
#   bash scripts/setup_env.sh [VENV]          # VENV default $VENV, else .venv
#   source VENV/bin/activate
#
# ENGINE_SPEC: the pip requirement NablaEngine is installed from (default: the tested revision, below; for example
# the path of a local checkout instead); set it empty to skip the engine (MuJoCo Warp only).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${1:-${VENV:-$ROOT/.venv}}"
PY="${PYTHON:-python3.12}"
WARP_WHL="https://github.com/NVIDIA/warp/releases/download/v1.18.0/warp_lang-1.18.0+cu12-py3-none-manylinux_2_28_x86_64.whl"
ENGINE_SPEC="${ENGINE_SPEC-nablaengine[jax] @ git+https://github.com/OamIcs/NablaEngine.git@c61a6fe7f5ec47c7f4f9b0a4c055d60cbe40807c}"
[[ -x "$VENV/bin/python" ]] || "$PY" -m venv "$VENV"
VPY="$VENV/bin/python"
"$VPY" -m pip install -q --upgrade pip
"$VPY" -m pip install -q -r "$ROOT/requirements-lock.txt"
"$VPY" -m pip install -q --no-deps "warp-lang @ $WARP_WHL"
PYTHON="$VPY" bash "$ROOT/scripts/setup_mujoco_warp.sh"
bash "$ROOT/scripts/setup_menagerie.sh"
"$VPY" -m pip install -q --no-deps -e "$ROOT"
if [[ -n "$ENGINE_SPEC" ]]; then  # its dependencies (NumPy, MuJoCo, JAX) are pinned in requirements-lock.txt
  "$VPY" -m pip install -q --no-deps "$ENGINE_SPEC"
fi
echo "environment ready: source $VENV/bin/activate"
