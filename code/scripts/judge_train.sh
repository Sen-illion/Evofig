#!/bin/bash
# Run one DPO training pass over a built Judge dataset, from a config file.
#
#   scripts/judge_train.sh <dataset> <run-dir> [config] [python]
#
# The config supplies every hyper-parameter, so a run is reproduced from the file rather
# than from the person typing it; the dataset and the run directory stay on the command
# line because they are properties of this attempt, not of the recipe.
#
# Formal 9B runs pass configs/judge/train-9b-round2.json as the third argument.
# train-pilot.json remains the 8-step smoke default; do not mix the two.
# Peak VRAM is recorded next to the run so the next attempt knows if it has room.
# 9B 4-bit is comfortable on one 80 GB card; 35B-A3B is a deferred MoE recipe.
set -euo pipefail

DATASET="${1:?usage: judge_train.sh <dataset> <run-dir> [config] [python]}"
RUN_DIR="${2:?usage: judge_train.sh <dataset> <run-dir> [config] [python]}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${3:-$ROOT/configs/judge/train-pilot.json}"
PY="${4:-python}"

mkdir -p "$RUN_DIR"
echo "config:  $CONFIG"
echo "dataset: $DATASET"
echo "run dir: $RUN_DIR"

"$PY" - "$CONFIG" "$DATASET" "$RUN_DIR" <<'PYEOF'
import json, sys
from rfs.judge_training.training import TrainingConfig
parsed = TrainingConfig.from_file(sys.argv[1], dataset=sys.argv[2], out_dir=sys.argv[3])
print("training config:", json.dumps(parsed.to_dict(), ensure_ascii=False))
print("config fingerprint:", parsed.fingerprint())
PYEOF

PEAK_FILE="$RUN_DIR/vram_peak_mib.txt"
: > "$PEAK_FILE"
if command -v nvidia-smi >/dev/null 2>&1; then
  (
    peak=0
    while true; do
      used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)"
      used="${used:-0}"
      if [ "$used" -gt "$peak" ] 2>/dev/null; then
        peak="$used"
        printf '%s\n' "$peak" > "$PEAK_FILE"
      fi
      sleep 5
    done
  ) &
  MONITOR_PID="$!"
  trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
fi

STATUS=0
"$PY" -m rfs.cli judge train --config "$CONFIG" --dataset "$DATASET" --out "$RUN_DIR" || STATUS="$?"

if [ -n "${MONITOR_PID:-}" ]; then
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
  trap - EXIT
fi
PEAK="$(cat "$PEAK_FILE" 2>/dev/null || true)"
echo "peak VRAM: ${PEAK:-unknown} MiB"

REPORT="$RUN_DIR/training_report.json"
if [ -f "$REPORT" ]; then
  "$PY" - "$REPORT" <<'PYEOF'
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
steps = report.get("steps") or []
backend = report.get("backend")
print("summary:", report.get("summary"))
print("backend:", backend.get("name") if isinstance(backend, dict) else backend)
print("records:", report.get("records"), "steps run:", len(steps))
print("checkpoints:", report.get("checkpoints"))
print("version lock:", report.get("version_lock"))
PYEOF
else
  echo "no training_report.json was written; the run did not finish" >&2
fi

exit "$STATUS"
