#!/usr/bin/env bash
# MuJoCo Menagerie's Unitree G1 (unitree_g1/, BSD-3-Clause, Unitree Robotics) at a pinned commit, checked against
# scripts/menagerie_unitree_g1.sha256: the robot model (robot.py builds the mjlab G1 from scene_mjx.xml) and the model
# the LAFAN1 references are processed with. The files must stay unmodified (meshes and geom order set the contact pairs).
#
#   bash scripts/setup_menagerie.sh [DEST]      # DEST default $MENAGERIE_DIR, else third_party/mujoco_menagerie
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-${MENAGERIE_DIR:-$ROOT/third_party/mujoco_menagerie}}"
REV=f3b6013629c7b94f8ee93fd977b161fa8350f9bc
if [[ ! -d "$DEST/.git" ]]; then
  mkdir -p "$(dirname "$DEST")"
  git clone -q --filter=blob:none --no-checkout https://github.com/google-deepmind/mujoco_menagerie.git "$DEST"
  git -C "$DEST" sparse-checkout set unitree_g1
fi
git -C "$DEST" -c advice.detachedHead=false checkout -q --detach "$REV"
(cd "$DEST" && grep -v '^#' "$ROOT/scripts/menagerie_unitree_g1.sha256" | sha256sum -c --quiet -)
echo "MuJoCo Menagerie unitree_g1 OK: $DEST (export MENAGERIE_DIR=$DEST if not the default)"
