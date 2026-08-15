# Paper 2 offline artifact (B1–B5)

Public code repository: https://github.com/lucadagati/XAIR_AI_Native_Action_Intents

This document describes how to reproduce the **CPU/offline** Paper‑2 analyses from a Phase‑P perception cache. It does **not** re-run GPU Phase P.

## What is included

| Suite | Script | Notes |
|-------|--------|-------|
| Split | `experiments/paper2_splits.py` | 70/30 by `frame_id`, seed 42 |
| B1 | `experiments/run_b1_grounding.py`, `run_b1_baselines.py` | Blind grounding + majority/confusion |
| B2 | `experiments/run_b2_validity_frontier.py` | Exploratory full-cache grid |
| Stats | `experiments/run_paper2_stats.py` | Frame-cluster bootstrap; **frame-majority McNemar** |
| B3 | `experiments/run_b3_validity_budget.py` | Train on train frames; headline on test |
| B4 | `experiments/run_b4_model_routing.py` | Routers evaluated on test frames |
| B5 | `experiments/run_b5_agent_policy.py` | Cache reuse; no VLM re-infer |
| Gate latency | `experiments/run_gate_latency_bench.py` | Requires live XAIR/adapter |
| Bundle | `scripts/reproduce_paper2_offline.sh` | End-to-end offline path |
| Manifest | `experiments/results/paper2_release_manifest.json` | SHA-256 of aggregates |

## Quick start (offline)

```bash
cd XAIR_Runtime
export PYTHONPATH=$PWD
# Place phase_p perception cache at experiments/results/perception_cache/
./scripts/reproduce_paper2_offline.sh
```

Headline statistical claims use the **test split** (`n_frames=612`). The B2 frontier figure uses the full blind cache and is marked exploratory.

## Scope / non-goals

- No Unity Play Mode / OPC UA HIL in this artifact freeze
- Phase P GPU inference is not re-run here
- Zenodo DOI upload may lag the git tag; check Releases on GitHub

## Citation

See the companion manuscript *AI-Native Action Intents: Offline Evaluation of Execution-Time Validation for Vision-Language Proposals*.
