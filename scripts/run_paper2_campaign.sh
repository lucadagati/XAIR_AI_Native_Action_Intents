#!/usr/bin/env bash
# Paper 2 full campaign: A1–A4 (VisA + MVTec AD). Requires live Ollama on L40.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
XAIR="$ROOT/XAIR_Runtime"
DATASET="$XAIR/experiments/datasets/manufacturing-a1"

: "${OLLAMA_HOST:=http://100.86.223.16:11434}"
export OLLAMA_HOST

echo "== Paper 2 campaign (live Ollama: $OLLAMA_HOST) =="

if [ ! -f "$DATASET/manifest.jsonl" ]; then
  echo "Building dataset manifest (VisA + MVTec AD)..."
  bash "$DATASET/scripts/download_visa.sh"
  bash "$DATASET/scripts/download_mvtec.sh"
  python3 "$DATASET/scripts/build_manifest.py" --total 100 --seed 42
fi

PY="${XAIR}/.venv/bin/python"
[ -x "$PY" ] || PY=python3
export PYTHONPATH="$XAIR:${PYTHONPATH:-}"

echo "Checking Ollama..."
if ! $PY -c "from xair.ai.ollama_client import OllamaClient; c=OllamaClient(); raise SystemExit(0 if c.health() else 1)"; then
  echo "ERROR: Ollama unreachable at $OLLAMA_HOST" >&2
  exit 1
fi

echo "Starting stack..."
"$ROOT/scripts/start_full_stack.sh"
sleep 3
"$ROOT/scripts/verify_e2e.sh" || true

for arm in A1a A1b A1c A1d; do
  echo "A1 arm $arm..."
  $PY "$XAIR/experiments/run_a1_vlm_ais.py" --arm "$arm" --runs 100 --seed 42
done

echo "A2 latency sweep..."
$PY "$XAIR/experiments/run_a2_latency_sweep.py" --runs 20

echo "A3 agent loop..."
$PY "$XAIR/experiments/run_a3_agent_loop.py" --runs 30

echo "A4 evidence audit..."
$PY "$XAIR/experiments/run_a4_evidence_audit.py"

$PY "$XAIR/experiments/aggregate_experiment_results.py"
$PY "$XAIR/experiments/plot_results.py"

echo "Paper 2 campaign complete."
