"""Frame-level train/test splits for Paper-2 suites B3--B5."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from experiments.paper2_common import RESULTS_DIR, load_manifest

SPLIT_SEED = 42
TRAIN_FRAC = 0.70


def make_frame_split(
    *,
    train_frac: float = TRAIN_FRAC,
    seed: int = SPLIT_SEED,
) -> dict[str, str]:
    """
    Stratified 70/30 split by frame_id (stratum = category × use_case).

    All VLM decisions for a frame stay in the same split.
    """
    episodes = load_manifest()
    by_stratum: dict[tuple, list[str]] = defaultdict(list)
    for ep in episodes:
        key = (str(ep.get("category") or "unk"), str(ep.get("use_case") or "uc1_triage"))
        by_stratum[key].append(ep["frame_id"])

    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    for _stratum, fids in sorted(by_stratum.items()):
        rng.shuffle(fids)
        n_train = max(1, int(round(len(fids) * train_frac)))
        if n_train >= len(fids) and len(fids) > 1:
            n_train = len(fids) - 1
        for i, fid in enumerate(fids):
            assignment[fid] = "train" if i < n_train else "test"
    return assignment


def save_split(path: Path | None = None) -> dict:
    path = path or (RESULTS_DIR / "paper2_frame_split.json")
    assignment = make_frame_split()
    n_train = sum(1 for v in assignment.values() if v == "train")
    n_test = sum(1 for v in assignment.values() if v == "test")
    payload = {
        "seed": SPLIT_SEED,
        "train_frac": TRAIN_FRAC,
        "n_train_frames": n_train,
        "n_test_frames": n_test,
        "assignment": assignment,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return payload


def load_split(path: Path | None = None) -> dict[str, str]:
    path = path or (RESULTS_DIR / "paper2_frame_split.json")
    if not path.is_file():
        return save_split(path)["assignment"]
    return json.loads(path.read_text())["assignment"]


def filter_by_split(items, *, split: str, frame_id_fn, assignment: dict[str, str] | None = None):
    assignment = assignment or load_split()
    return [x for x in items if assignment.get(frame_id_fn(x)) == split]


if __name__ == "__main__":
    out = save_split()
    print(json.dumps({k: out[k] for k in ("seed", "train_frac", "n_train_frames", "n_test_frames")}, indent=2))
