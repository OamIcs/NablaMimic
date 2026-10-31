#!/usr/bin/env bash
# The 40 LAFAN1 G1 retarget CSVs from the Hugging Face dataset lvhaidong/LAFAN1_Retargeting_Dataset (g1/*.csv) at a
# pinned revision, checked against scripts/lafan1_g1.sha256; then (with --cache) their 50 Hz references cached by
# nablamimic.motion (training builds the cache on first use otherwise). Needs the Hugging Face CLI (huggingface_hub).
#
# LAFAN1 (Ubisoft La Forge) is licensed CC BY-NC-ND 4.0, non-commercial; the data is never part of this repository.
# Training uses every CSV in the directory, so it must hold exactly these 40 files.
#
#   bash scripts/download_lafan1.sh [DIR] [--cache]     # DIR default $LAFAN1_G1_DIR, else data/lafan1_g1
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIR="${LAFAN1_G1_DIR:-$ROOT/data/lafan1_g1}"
CACHE=0
for a in "$@"; do
  case "$a" in
    --cache) CACHE=1 ;;
    *) DIR="$a" ;;
  esac
done
REPO=lvhaidong/LAFAN1_Retargeting_Dataset
REV=07ff8845a94d59d4195a8ac04fe918d4e6338239
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
if command -v hf >/dev/null; then
  hf download "$REPO" --repo-type dataset --revision "$REV" --include 'g1/*.csv' --local-dir "$tmp" >/dev/null
else
  huggingface-cli download "$REPO" --repo-type dataset --revision "$REV" --include 'g1/*.csv' --local-dir "$tmp" >/dev/null
fi
mkdir -p "$DIR"
mv "$tmp"/g1/*.csv "$DIR"/
n="$(find "$DIR" -maxdepth 1 -name '*.csv' | wc -l)"
[[ "$n" == 40 ]] || { echo "$DIR holds $n CSVs, expected exactly 40" >&2; exit 1; }
(cd "$DIR" && grep -v '^#' "$ROOT/scripts/lafan1_g1.sha256" | sha256sum -c --quiet -)
echo "LAFAN1 G1: 40 CSVs in $DIR match scripts/lafan1_g1.sha256 (licence: CC BY-NC-ND 4.0, non-commercial)"
[[ "$DIR" -ef "${LAFAN1_G1_DIR:-$ROOT/data/lafan1_g1}" ]] ||
  echo "training and evaluation read \$LAFAN1_G1_DIR: export LAFAN1_G1_DIR=$DIR"
if [[ "$CACHE" == 1 ]]; then
  LAFAN1_G1_DIR="$DIR" "${PYTHON:-python}" -m nablamimic.motion
fi
