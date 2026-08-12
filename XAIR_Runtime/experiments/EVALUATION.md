# Evaluation methodology (IEEE TII)

## Client path: HTTP protocol replay (Path B)

Primary evaluation uses **HTTP protocol replay** via `run_e1_baselines.py` — identical AIS JSON to the Unity `DefectEventEmitter` client. Unity Editor Play Mode is supported as optional replication (see `AdaptiX-Quest/README.md`).

**Do not claim Unity Editor evaluation unless `ManufacturingTestOrchestrator` batch was executed on Mac and JSON exported with `_unity` suffix.**

## Symmetric baselines

All baselines use the same HTTP path (`POST :9092/intent?mode=`):

| Mode | Behavior |
|------|----------|
| `xair` | Full temporal + contextual validation via XAIR |
| `naive` | Freshness window only; ignores context |
| `direct` | No validation; always publishes to ROS |

## Scenario E1b

Stale **RESUME** when line is **PAUSED** and gripper **CLOSED** — precondition `line.state == 'RUN'` fails at $t_e$.

Protocol order (aligned Unity + Python): RUN → delay → context change → submit intent.

## Metrics

- **SER** = stale_executed / attempted (ros_published when context invalid)
- **SER$_{\mathrm{known}}$** (E8) = stale_executed / completed runs (excludes UNKNOWN transport failures)
- **POA** = correct_revokes / obsolete_intents (excludes UNKNOWN)
- **FPR** = wrongful_revokes / valid_intents (E1c)
- **CV** = conflict violations (E3)
- **UR** = unknown_rate

## Reproduce

```bash
./scripts/start_full_stack.sh
./scripts/verify_e2e.sh
cd XAIR_Runtime
.venv/bin/python experiments/run_e1_baselines.py --runs 30 --seed 42
.venv/bin/python experiments/run_e1_fpr.py --runs 30
.venv/bin/python experiments/run_e3_http_stack.py --runs 30
.venv/bin/python experiments/run_e4_http_load.py --intents 1000
.venv/bin/python experiments/aggregate_experiment_results.py
.venv/bin/python experiments/plot_results.py
```

## Paper 2 — AI-native suites (A1–A4)

**Vision dataset:** VisA (CC BY 4.0) + MVTec AD (CC BY-NC-SA, academic). Build manifest:

```bash
cd XAIR_Runtime/experiments/datasets/manufacturing-a1/scripts
./download_visa.sh && ./download_mvtec.sh
python3 build_manifest.py --total 100 --seed 42
```

| Suite | Driver | RQ |
|-------|--------|-----|
| A1 | `run_a1_vlm_ais.py` | Grounding (arms A1a–d) |
| A2 | `run_a2_latency_sweep.py` | Latency vs SER |
| A3 | `run_a3_agent_loop.py` | Agent multi-step |
| A4 | `run_a4_evidence_audit.py` | Audit traceability |

```bash
export OLLAMA_HOST=http://100.86.223.16:11434
./scripts/run_paper2_campaign.sh
```

Ground truth for visual defect/normal is declared in `manifest.jsonl` before scoring. Line PAUSED drift is injected via adapter context (same E1b protocol). All AI arms call live Ollama — there is no offline/mock producer.
