#!/usr/bin/env bash
# Offline Paper-2 reproduce path (CPU): B1 baselines, B2-stats, B3, B5 from Phase-P cache.
# Does NOT re-run Phase P GPU inference.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
TAG="${PAPER2_TAG:-phase_p}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "[reproduce] split"
python3 experiments/paper2_splits.py

echo "[reproduce] B1 baselines"
python3 experiments/run_b1_baselines.py --tag "$TAG" --no-plot || python3 experiments/run_b1_baselines.py --tag "$TAG"

echo "[reproduce] B1 grounding (if present)"
if [[ -f experiments/run_b1_grounding.py ]]; then
  python3 experiments/run_b1_grounding.py --tag "$TAG" || true
fi

echo "[reproduce] B3 validity budget"
python3 experiments/run_b3_validity_budget.py --tag "$TAG" --no-notify

echo "[reproduce] B4 routing (test frames)"
python3 experiments/run_b4_model_routing.py --tag "$TAG" --no-notify

echo "[reproduce] B5 agent policy"
python3 experiments/run_b5_agent_policy.py --tag "$TAG"

echo "[reproduce] paper2 stats + utility sensitivity"
python3 experiments/run_paper2_stats.py --tag "$TAG" --n-boot "${N_BOOT:-1000}"
python3 experiments/run_utility_sensitivity.py

echo "[reproduce] release manifest"
python3 - <<'PY'
import hashlib, json, time
from pathlib import Path
root = Path("experiments/results")
files = [
    "paper2_frame_split.json",
    "b1_baselines.json",
    "b2_validity_frontier.json",
    "b2_fidelity.json",
    "b3_validity_budget.json",
    "b3_headline_table.csv",
    "b4_routing_table.csv",
    "b5_agent_policy.json",
    "b5_headline_table.csv",
    "paper2_stats.json",
    "utility_sensitivity.json",
]
entries = []
for name in files:
    p = root / name
    if not p.is_file():
        entries.append({"path": str(p), "missing": True})
        continue
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    entries.append({"path": str(p), "sha256": h, "bytes": p.stat().st_size})
out = {
    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "note": "Offline Paper-2 freeze; Phase P cache assumed present; no GPU re-infer.",
    "artifacts": entries,
}
path = root / "paper2_release_manifest.json"
path.write_text(json.dumps(out, indent=2))
print(path)
PY

echo "[reproduce] done"
