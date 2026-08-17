# XAIR AI-Native Action Intents

Development, deployment, and evaluation harness for **vision-language models (VLMs) and agents** that propose [Action Intents (AIS)](XAIR_Runtime/schemas/action-intent-v1.json) validated at execution time by [XAIR](https://github.com/lucadagati/XAIR_eXecution-time_Action_Intent_Runtime).

This repository contains the **runtime extensions, experiment drivers, dataset tooling, and operational scripts** for the AI-native intents research track. The LaTeX manuscript lives in a separate private research repository and is **not** included here.

## What is here

| Path | Purpose |
|------|---------|
| `XAIR_Runtime/` | XAIR runtime + AI producer (`xair/ai/`), Paper 2 suites B1–B6 |
| `scripts/` | Stack startup, GPU access, campaign runners, Telegram notifications |
| `config/` | Example env files (copy to `*.env`, never commit secrets) |
| `docs/` | Operational notes: GPU node, dataset, research vision (markdown) |

## Quick start

### 1. Install

```bash
cd XAIR_Runtime
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. Start the validation stack

```bash
./scripts/start_full_stack.sh
./scripts/verify_e2e.sh
```

XAIR listens on `:8080`, the HTTP adapter on `:9092`.

### 3. Configure GPU / Ollama (optional)

```bash
cp config/compute-gpu.env.example config/compute-gpu.env   # edit with your node
source config/compute-gpu.env
export OLLAMA_HOST
```

See [docs/compute-gpu.md](docs/compute-gpu.md) for L40 node access via Tailscale.

### 4. Build the vision dataset

```bash
cd XAIR_Runtime/experiments/datasets/manufacturing-a1/scripts
./download_visa.sh && ./download_mvtec.sh
python3 build_manifest.py --total 2000 --seed 42
```

Manifest and stats are tracked; `frames/` and `context/` are regenerated locally (~1.2 GB).

### 5. Run the evaluation campaign

Three phases, designed so gate replay does not re-run GPU inference:

| Phase | Script | Description |
|-------|--------|-------------|
| **P** Perception | `./scripts/run_phase_p.sh` | Cache ~18k VLM decisions (resumable) |
| **G** Gating | `./scripts/run_phase_g.sh` | Offline validity-frontier replay (B2) |
| **B1** Analysis | `./scripts/run_b1_analysis.sh` | Blind grounding + leakage ablation |

Legacy saturated ablation (A1–A4): `./scripts/run_paper2_campaign.sh`

### 6. Telegram progress (optional)

```bash
cp config/notify.env.example config/notify.env   # add bot token
# message your bot once, then:
python3 scripts/notify.py --discover
./scripts/run_phase_p.sh   # sends progress every 5%
```

## Architecture

```
  Camera / dataset frame
         │
         ▼
  StructuredIntentProducer (Ollama VLM)
         │  PerceptionResult (cached)
         ▼
  build_submission(anchor=capture|emission)
         │
         ▼
  XAIR gate (direct | freshness_only | xair)
         │
         ▼
  ROS / adapter publication boundary
```

Key design choices documented in `docs/VISION.md`:

- **Evidence anchoring**: freshness measured from capture time, not model emission
- **Stochastic drift**: plant volatility as a controlled experimental factor
- **Blind prompts**: no label leakage; leaky control for ablation only
- **Learned validity budget**: RL/bandit policies over freshness windows (Phase RL; B3–B5 offline on the Phase-P cache)

## Tests

```bash
cd XAIR_Runtime && source .venv/bin/activate
pytest tests/ -q
```

## Relationship to XAIR Runtime

This repo extends the canonical [XAIR Runtime](https://github.com/lucadagati/XAIR_eXecution-time_Action_Intent_Runtime) with:

- `xair/ai/structured_intent.py` — blind/leaky VLM producer, `PerceptionResult`, anchoring
- `experiments/paper2_common.py` — stochastic drift, SAR/WRR/utility metrics
- `experiments/perception_cache.py`, `run_b1_grounding.py`, `run_b2_validity_frontier.py`
- `experiments/datasets/manufacturing-a1/` — 2k-frame VisA + MVTec AD manifest

## License

See [LICENSE](LICENSE). MVTec AD data is CC BY-NC-SA 4.0 (academic use only); VisA is CC BY 4.0.

## Citation

If you use this harness, please cite the XAIR runtime and the AI-native intents paper (when published). See `XAIR_Runtime/CITATION.cff`.

## Paper 2 offline reproduction (B1–B5)

See [`XAIR_Runtime/PAPER2_ARTIFACT.md`](XAIR_Runtime/PAPER2_ARTIFACT.md) and:

```bash
cd XAIR_Runtime && ./scripts/reproduce_paper2_offline.sh
```

Requires a Phase-P perception cache under `experiments/results/perception_cache/`. CPU-only; no GPU Phase P re-run.

