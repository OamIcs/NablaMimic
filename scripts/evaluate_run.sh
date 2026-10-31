#!/usr/bin/env bash
# The nominal LAFAN1 protocol for one run: the checkpoints eval_plan chooses, evaluated by nablamimic.evaluate (MuJoCo
# Warp, 10 solver iterations) into RUN/eval_nominal.json (already evaluated checkpoints are skipped).
#
#   bash scripts/evaluate_run.sh RUN [OUT_JSON]
set -euo pipefail
RUN="${1:?usage: evaluate_run.sh RUN [OUT_JSON]}"
OUT="${2:-$RUN/eval_nominal.json}"
PY="${PYTHON:-python}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
its="$("$PY" -m nablamimic.eval_plan "$RUN")"
[[ -n "$its" ]] || { echo "$RUN: no checkpoints to evaluate"; exit 0; }
# shellcheck disable=SC2086
"$PY" -m nablamimic.evaluate --run "$RUN" --iters $its --out "$OUT"
