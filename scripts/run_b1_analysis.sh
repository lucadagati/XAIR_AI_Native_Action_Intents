#!/usr/bin/env bash
# B1: blind grounding analysis on cached perception (no GPU).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
XAIR="$ROOT/XAIR_Runtime"
PY="${XAIR}/.venv/bin/python"
[ -x "$PY" ] || PY=python3
export PYTHONPATH="$XAIR:${PYTHONPATH:-}"

$PY "$XAIR/experiments/run_b1_grounding.py" --tag phase_p "$@"
