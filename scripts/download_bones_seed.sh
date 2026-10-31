#!/usr/bin/env bash
# BONES-SEED for configs/bones.args: the G1 retargets (g1.tar.gz, 23.5 GB) and the metadata from the gated Hugging Face
# dataset bones-studio/seed at a pinned revision, checked against scripts/bones_seed.sha256, extracted and converted to
# the sharded 50 Hz library DATA_DIR/lib50 (about 50 GB) by nablamimic.bones.convert (Euler convention zyx, as the
# reported library).
#
# Access must be granted to your Hugging Face account (log in with `hf auth login`); the dataset is licensed for
# academic, non-commercial use and must not be redistributed: keep DATA_DIR, the library and per-motion results out of
# any public place. The conversion uses WORKERS processes (default: all cores) and needs roughly 200 GB of memory with
# 48 workers. Disk: about 125 GB during the conversion; afterwards only DATA_DIR/lib50 is needed (the script says what
# can be deleted). An interrupted extraction is redone on the next call.
#
#   bash scripts/download_bones_seed.sh DATA_DIR [WORKERS]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
D="${1:?usage: download_bones_seed.sh DATA_DIR [WORKERS]}"
WORKERS="${2:-$(nproc)}"
PY="${PYTHON:-python}"
REV=2f59b2077b9da34dd4e43618e705c7cb962c9a66
mkdir -p "$D"
files=(g1.tar.gz README.md LICENSE.md metadata/seed_metadata_v004.csv)
if command -v hf >/dev/null; then
  hf download bones-studio/seed "${files[@]}" --repo-type dataset --revision "$REV" --local-dir "$D"
else
  huggingface-cli download bones-studio/seed "${files[@]}" --repo-type dataset --revision "$REV" --local-dir "$D"
fi
(cd "$D" && grep -v '^#' "$ROOT/scripts/bones_seed.sha256" | sha256sum -c --quiet -)
if [[ ! -f "$D/g1/.extracted" ]]; then  # extract beside, move into place only when tar has finished
  rm -rf -- "$D/g1.partial" "$D/g1"
  mkdir -p "$D/g1.partial"
  tar -xzf "$D/g1.tar.gz" -C "$D/g1.partial"
  mv "$D/g1.partial/g1" "$D/g1"
  rm -rf -- "$D/g1.partial"
  touch "$D/g1/.extracted"
fi
n="$(find "$D/g1/csv" -name '*.csv' | wc -l)"
[[ "$n" == 142220 ]] || {  # the dataset version configs/bones.args was converted from has 142220
  echo "$D/g1/csv holds $n CSVs, expected 142220 (incomplete files or another dataset version;" \
    "delete $D/g1 to extract again)" >&2; exit 1; }
"$PY" -m nablamimic.bones.convert --src "$D/g1/csv" --meta "$D/metadata/seed_metadata_v004.csv" --out "$D/lib50" \
  --euler zyx --workers "$WORKERS"
echo "BONES-SEED library: $D/lib50 (train with --motion-lib $D/lib50)"
echo "$D/g1.tar.gz and $D/g1/ (about 75 GB) are no longer needed and can be deleted"
