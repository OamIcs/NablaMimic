#!/usr/bin/env bash
# Patched MuJoCo Warp (patches/mujoco_warp/README.md): etaoxing/mujoco_warp at the head of its pull request #1
# (4af093f, on top of 357a75d), then patches 0001-0004 of this repository, the source-tree check, and an editable install
# into the active Python environment.
#
#   bash scripts/setup_mujoco_warp.sh [DEST] [--no-install]      # DEST default third_party/mujoco_warp
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/third_party/mujoco_warp"
INSTALL=1
for a in "$@"; do
  case "$a" in
    --no-install) INSTALL=0 ;;
    *) DEST="$a" ;;
  esac
done
PY="${PYTHON:-python}"
URL=https://github.com/etaoxing/mujoco_warp.git
BASE=357a75d60a56d67d476942a1b6e54b3045ee8e87
HEAD=4af093f02a3f48f0009ca7c916674b928136f310
PATCHES=("$ROOT"/patches/mujoco_warp/000[1-4]-*.diff)

tree_hash() { (cd "$1" && find mujoco_warp -type f -name '*.py' | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -d' ' -f1); }
want="$(grep -v '^#' "$ROOT/patches/mujoco_warp/tree.sha256" | head -n 1)"

if [[ ! -d "$DEST/.git" ]]; then
  mkdir -p "$(dirname "$DEST")"
  git clone -q "$URL" "$DEST"
fi
if [[ "$(tree_hash "$DEST")" != "$want" ]]; then
  git -C "$DEST" cat-file -e "$HEAD^{commit}" 2>/dev/null || git -C "$DEST" fetch -q origin pull/1/head
  git -C "$DEST" cat-file -e "$HEAD^{commit}"
  [[ -z "$(git -C "$DEST" status --porcelain --untracked-files=no)" ]] || {
    echo "$DEST has local changes; use a fresh directory" >&2; exit 1; }
  git -C "$DEST" -c advice.detachedHead=false checkout -q --detach "$HEAD"
  git -C "$DEST" merge-base --is-ancestor "$BASE" HEAD
  for p in "${PATCHES[@]}"; do
    git -C "$DEST" apply "$p"
    echo "applied $(basename "$p")"
  done
fi
got="$(tree_hash "$DEST")"
if [[ "$got" != "$want" ]]; then
  echo "patched MuJoCo Warp tree hash $got != $want (patches/mujoco_warp/tree.sha256)" >&2
  exit 1
fi
echo "MuJoCo Warp tree OK: $DEST"
if [[ "$INSTALL" == 1 ]]; then
  "$PY" -m pip install --no-deps -e "$DEST"
fi
