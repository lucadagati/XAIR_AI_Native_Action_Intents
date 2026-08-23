# Paper 2 offline artifact (B1–B5)

Public code repository: https://github.com/lucadagati/XAIR_AI_Native_Action_Intents

Companion manuscript: *AI-Native Action Intents: Evidence-Anchored Validation of Vision-Language Proposals for Industrial CPS*.

This document describes how to reproduce the **CPU/offline** Paper‑2 analyses from a Phase‑P perception cache. It does **not** re-run GPU Phase P and does not call Ollama.

**Environment:** Python 3.10+ with `numpy` (LinUCB). Optional: `scipy` (exact McNemar uses SciPy when present, otherwise a stdlib log-sum-exp fallback). No GPU.

## Release assets (`paper2-b1b5-v0.3`)

Download from GitHub Releases:

https://github.com/lucadagati/XAIR_AI_Native_Action_Intents/releases/tag/paper2-b1b5-v0.3

Place the cache at `XAIR_Runtime/experiments/results/perception_cache/phase_p.jsonl` (SHA-256 in `paper2_release_manifest.json`).

| File | Role |
|------|------|
| `phase_p.jsonl` | Phase P perception cache (~22 MB) |
| `paper2_frame_split.json` | 70/30 `frame_id` split, seed 42 |
| `b1_*.json`, `b2_validity_frontier.json`, `b3_*.json/csv`, `b4_*.json/csv`, `b5_*.json/csv` | Suite aggregates |
| `paper2_stats.json` | Exact McNemar, frame-cluster CIs, B1 paired test |
| `utility_sensitivity.json` | \(\lambda,\mu\) ranking on B2/B3/B5 |
| `gt_threshold_sensitivity.json` | Mask-area cut 40/60 percentile action flips |
| `paper2_release_manifest.json` | SHA-256 of the bundle |

## What is included in the code

| Suite | Script | Notes |
|-------|--------|-------|
| Split | `experiments/paper2_splits.py` | 70/30 by `frame_id`, seed 42 |
| B1 | `experiments/run_b1_grounding.py`, `run_b1_baselines.py` | Blind grounding + majority/confusion |
| B2 | `experiments/run_b2_validity_frontier.py` | Exploratory full-cache grid; XAIR blocks schema failures |
| Stats | `experiments/run_paper2_stats.py` | Frame-cluster bootstrap; exact McNemar (Yates secondary) |
| B3 | `experiments/run_b3_validity_budget.py` | No GT severity in headline features; 10 fixed + train-selected best; `--privileged-severity` ablation |
| B4 | `experiments/run_b4_model_routing.py` | Post-hoc selection among cached outputs, seeds `{1..5}` |
| B5 | `experiments/run_b5_agent_policy.py` | Cache-reuse policy simulation; no GT revoke features; no VLM re-infer |
| GT cut | `experiments/run_gt_threshold_sensitivity.py` | CPU mask-area threshold flips |
| Bundle | `scripts/reproduce_paper2_offline.sh` | Fail-fast if the cache is missing |

## Quick start (offline)

```bash
cd XAIR_Runtime
export PYTHONPATH=$PWD
# Download phase_p.jsonl from the v0.3 release into:
#   experiments/results/perception_cache/phase_p.jsonl
./scripts/reproduce_paper2_offline.sh
```

Exact CPU commands (same order as the bundle script):

```bash
python3 experiments/paper2_splits.py
python3 experiments/run_b1_baselines.py --tag phase_p --no-plot
python3 experiments/run_b1_grounding.py --tag phase_p
python3 experiments/run_gt_threshold_sensitivity.py
python3 experiments/run_b3_validity_budget.py --tag phase_p --no-notify
python3 experiments/run_b4_model_routing.py --tag phase_p --no-notify
python3 experiments/run_b5_agent_policy.py --tag phase_p
python3 experiments/run_paper2_stats.py --tag phase_p --n-boot 1000
python3 experiments/run_utility_sensitivity.py
```

Headline statistical claims use the **test split** (`n_frames=612`). The B2 frontier figure uses the full blind cache and is marked exploratory. B3 headline features are latency, confidence, precondition count, schema, VLM defect judgement, use-case, and model — not mask-derived severity.

## Scope / non-goals

- No Unity Play Mode / OPC UA HIL in this artifact freeze
- Phase P GPU inference is not re-run here
- Zenodo DOI is out of scope for this tag

## Citation

See the companion manuscript *AI-Native Action Intents: Evidence-Anchored Validation of Vision-Language Proposals for Industrial CPS*.
