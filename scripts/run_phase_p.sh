#!/usr/bin/env bash
# Phase P: cache VLM decisions for the AI-native evaluation (B1-B6).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
XAIR="$ROOT/XAIR_Runtime"
DATASET="$XAIR/experiments/datasets/manufacturing-a1"

: "${OLLAMA_HOST:=http://127.0.0.1:11434}"
export OLLAMA_HOST

echo "== Phase P perception campaign (Ollama: $OLLAMA_HOST) =="

if [ ! -f "$DATASET/manifest.jsonl" ]; then
  echo "Building dataset manifest..."
  bash "$DATASET/scripts/download_visa.sh"
  bash "$DATASET/scripts/download_mvtec.sh"
  python3 "$DATASET/scripts/build_manifest.py" --total 2000 --seed 42
fi

PY="${XAIR}/.venv/bin/python"
[ -x "$PY" ] || PY=python3
export PYTHONPATH="$XAIR:${PYTHONPATH:-}"

$PY "$XAIR/experiments/perception_cache.py" \
  --models qwen2.5vl:3b qwen2.5vl:7b llama3.2-vision:11b gemma3:12b qwen2.5vl:32b \
  --variants blind blind_cot blind_noctx blind_ref leaky \
  --primary-model qwen2.5vl:7b \
  --tag phase_p "$@"

echo "Phase P complete."
