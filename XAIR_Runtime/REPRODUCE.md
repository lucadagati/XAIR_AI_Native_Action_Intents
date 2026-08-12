# Reproducing XAIR TII Evaluation

## Requirements

- Ubuntu 24.04 (or compatible)
- Python 3.12+
- Docker (Redis)
- Optional: ROS 2 Jazzy + Gazebo Harmonic for E8 cell simulation

## Gazebo Harmonic (E8 full sim)

```bash
cd adaptix/scripts && ./setup_gazebo.sh
./start_full_stack.sh
./start_gazebo_cell.sh
cd ../XAIR_Runtime
.venv/bin/python experiments/run_e8_gazebo_cell.py --runs 30 --use-gazebo
```

## One-command stack

```bash
cd adaptix/scripts && ./start_full_stack.sh && ./verify_e2e.sh
```

## Regenerate all paper metrics

```bash
cd adaptix/XAIR_Runtime
python -m venv .venv && .venv/bin/pip install -e ".[dev]" -q
.venv/bin/python experiments/run_e0_lifecycle.py
.venv/bin/python experiments/run_e1_baselines.py --runs 100 --seed 42
.venv/bin/python experiments/run_e1_fpr.py --runs 100
.venv/bin/python experiments/run_e1_variants.py --runs 30
.venv/bin/python experiments/run_e2_temporal.py --runs 5
.venv/bin/python experiments/run_e3_http_stack.py --runs 100
.venv/bin/python experiments/run_e4_http_load.py --intents 10000
.venv/bin/python experiments/run_e6_network.py --runs 5
.venv/bin/python experiments/run_e7_faults.py
source /opt/ros/jazzy/setup.bash
# Independent ROS witness (verifies actual middleware publication;
# started automatically by start_full_stack.sh, or manually:)
python3 ../scripts/ros_audit_subscriber.py &
.venv/bin/python experiments/run_e8_gazebo_cell.py --runs 30
.venv/bin/python experiments/run_e9_shared_context.py --runs 30
.venv/bin/python experiments/aggregate_experiment_results.py
.venv/bin/python experiments/plot_results.py
```

## Measurement integrity

E8/E9 publication rates are double-checked: the adapter's self-reported
`ros_published` flag is compared against message counts from an independent
ROS 2 subscriber (`scripts/ros_audit_subscriber.py`) on the actuator topics.
The aggregated summary reports `stale_observed_rate` and
`witness_agreement_rate` per baseline.

## Versions (tested)

| Component | Version |
|-----------|---------|
| OS | Ubuntu 24.04 |
| Python | 3.12 |
| ROS 2 | Jazzy |
| Gazebo | Harmonic (gz sim) via ros-jazzy-ros-gz |
| Redis | 7-alpine (Docker) |

## Artifact

Release tag: `v0.1.0-alpha-tii`  
Zenodo DOI: mint at publication (see GitHub release assets).

Raw results: `XAIR_Runtime/experiments/results/*.csv`
