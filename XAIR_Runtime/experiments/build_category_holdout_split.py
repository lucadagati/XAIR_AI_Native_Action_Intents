#!/usr/bin/env python3
"""
Category-held-out frame split for the generalization robustness check.

The headline frame split (paper2_splits.make_frame_split) is stratified by
category x use_case and shuffles *within* each stratum, so every category
appears in both train and test: it measures generalization across frames,
not across defect categories. This script instead assigns whole categories
to test, so a category-held-out learner never sees a single training frame
from any category it is evaluated on.

Output has the same {seed, train_frac, n_train_frames, n_test_frames,
assignment} shape as paper2_frame_split.json, so it can be passed straight
to run_b3_validity_budget.py --split-path.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from experiments.paper2_common import RESULTS_DIR, load_manifest

SEED = 42
TARGET_TEST_FRAC = 0.30


def build_category_holdout_split(
    *, seed: int = SEED, target_test_frac: float = TARGET_TEST_FRAC
) -> dict:
    episodes = load_manifest()
    by_category: dict[str, list[str]] = defaultdict(list)
    for ep in episodes:
        by_category[str(ep.get("category") or "unk")].append(ep["frame_id"])

    categories = sorted(by_category)
    rng = random.Random(seed)
    rng.shuffle(categories)

    total = len(episodes)
    target_test_n = round(total * target_test_frac)

    test_categories: list[str] = []
    n_test = 0
    for cat in categories:
        if n_test >= target_test_n:
            break
        test_categories.append(cat)
        n_test += len(by_category[cat])

    test_set = set(test_categories)
    assignment: dict[str, str] = {}
    for cat, fids in by_category.items():
        label = "test" if cat in test_set else "train"
        for fid in fids:
            assignment[fid] = label

    n_train = sum(1 for v in assignment.values() if v == "train")
    n_test = sum(1 for v in assignment.values() if v == "test")
    return {
        "seed": seed,
        "train_frac": 1.0 - target_test_frac,
        "split_kind": "category_holdout",
        "held_out_categories": sorted(test_categories),
        "train_categories": sorted(set(by_category) - test_set),
        "n_train_frames": n_train,
        "n_test_frames": n_test,
        "assignment": assignment,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--target-test-frac", type=float, default=TARGET_TEST_FRAC)
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS_DIR / "paper2_frame_split_category_holdout.json",
    )
    args = parser.parse_args()

    payload = build_category_holdout_split(
        seed=args.seed, target_test_frac=args.target_test_frac
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(
        json.dumps(
            {k: payload[k] for k in ("seed", "n_train_frames", "n_test_frames", "held_out_categories")},
            indent=2,
        )
    )
    print(f"[category-holdout] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
