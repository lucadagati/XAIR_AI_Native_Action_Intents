#!/usr/bin/env python3
"""
B5 p_recover sweep: utility/hazard vs plant-recovery probability.

Usage:
    python3 experiments/run_b5_precover_sweep.py --tag phase_p
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper2_common import RESULTS_DIR  # noqa: E402
from experiments.rl.agent_env import set_p_recover  # noqa: E402
from experiments.run_b2_validity_frontier import load_replay_records  # noqa: E402
from experiments.run_b4_model_routing import build_frame_index  # noqa: E402
from experiments.run_b5_agent_policy import (  # noqa: E402
    PRIMARY,
    aggregate_episodes,
    fixed_chooser,
    oracle_chooser,
    run_policy_on_frames,
    single_shot_chooser,
)

P_GRID = (0.0, 0.2, 0.35, 0.5, 0.7)
SEEDS = (1, 2, 3)
SETTING = {"label": "headline", "p_drift": 0.5, "drift_offset_ms": 250}
POLICIES = (
    ("single_shot", single_shot_chooser),
    ("always_reobserve", lambda: fixed_chooser("reobserve")),
    ("oracle", oracle_chooser),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="phase_p")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    records = load_replay_records(args.tag, models=None, variants={"blind"}, use_repaired=False)
    frames = list(build_frame_index(records).values())
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    print(f"[b5-sweep] frames={len(frames)}", flush=True)

    rows = []
    for p in P_GRID:
        set_p_recover(p)
        for name, factory in POLICIES:
            seed_aggs = []
            for seed in SEEDS:
                ep = run_policy_on_frames(
                    frames,
                    policy_name=name,
                    choose_action=factory(),
                    setting=SETTING,
                    seed=seed,
                )
                seed_aggs.append(aggregate_episodes(ep))
            rows.append(
                {
                    "p_recover": p,
                    "policy": name,
                    "utility": sum(a["utility"] for a in seed_aggs) / len(seed_aggs),
                    "SAR": sum(a["SAR"] for a in seed_aggs) / len(seed_aggs),
                    "hazard": sum(a["hazard"] for a in seed_aggs) / len(seed_aggs),
                    "WRR": sum(a["WRR"] for a in seed_aggs) / len(seed_aggs),
                    "mean_steps": sum(a["mean_steps"] for a in seed_aggs) / len(seed_aggs),
                    "n_seeds": len(seed_aggs),
                    "n_frames": len(frames),
                }
            )
            print(
                f"[b5-sweep] p={p:.2f} {name}: U={rows[-1]['utility']:.3f} haz={rows[-1]['hazard']:.3f}",
                flush=True,
            )

    set_p_recover(0.35)  # restore default
    csv_path = RESULTS_DIR / "b5_precover_sweep.csv"
    json_path = RESULTS_DIR / "b5_precover_sweep.json"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json_path.write_text(json.dumps({"rows": rows, "grid": list(P_GRID)}, indent=2))
    print(json.dumps({"out": str(json_path), "n": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
