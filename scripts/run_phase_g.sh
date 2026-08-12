#!/usr/bin/env bash
# Phase G: offline validity-frontier replay (B2) on cached perception.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
XAIR="$ROOT/XAIR_Runtime"
PY="${XAIR}/.venv/bin/python"
[ -x "$PY" ] || PY=python3
export PYTHONPATH="$XAIR:${PYTHONPATH:-}"

$PY "$XAIR/experiments/run_b2_validity_frontier.py" --tag phase_p "$@"
