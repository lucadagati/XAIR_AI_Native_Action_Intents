#!/usr/bin/env python3
"""
B1 baselines: majority-class accuracy, confusion matrices, severity stratification.

Computed offline from the Phase P perception cache (same path as ``run_b1_grounding.py``).
Headline model: qwen2.5vl:7b blind.

Usage:
    python3 experiments/run_b1_baselines.py --tag phase_p
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper2_common import DATASET_ROOT, RESULTS_DIR, load_manifest, wilson_ci  # noqa: E402
from experiments.paper2_splits import load_split  # noqa: E402
from experiments.perception_cache import cache_path, load_cache  # noqa: E402
from experiments.run_b1_grounding import confusion, evaluate_record  # noqa: E402

PAPER_FIG = ROOT.parent / "ResearchTrack" / "ai-native-intents-paper" / "figures"
DEFAULT_MODEL = "qwen2.5vl:7b"
DEFAULT_VARIANT = "blind"


def _majority_label(rows: list[dict]) -> tuple[str, Counter]:
    counts = Counter(r["gt_action"] for r in rows)
    if not counts:
        return "", counts
    label, _ = counts.most_common(1)[0]
    return label, counts


def majority_baseline(rows: list[dict], *, predict: str) -> dict:
    """Accuracy if every frame were predicted as ``predict``."""
    n = len(rows)
    if n == 0:
        return {"predict": predict, "n": 0, "k": 0, "accuracy": None, "ci95": None}
    k = sum(1 for r in rows if r["gt_action"] == predict)
    lo, hi = wilson_ci(k, n)
    return {
        "predict": predict,
        "n": n,
        "k": k,
        "accuracy": k / n,
        "ci95": [lo, hi],
    }


def model_accuracy(rows: list[dict]) -> dict:
    n = len(rows)
    k = sum(1 for r in rows if r["grounding_correct"])
    if n == 0:
        return {"n": 0, "k": 0, "accuracy": None, "ci95": None}
    lo, hi = wilson_ci(k, n)
    return {"n": n, "k": k, "accuracy": k / n, "ci95": [lo, hi]}


def macro_by_category(rows: list[dict]) -> dict:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"] or "unknown"].append(r)
    per_cat = {}
    accs = []
    for cat, grp in sorted(by_cat.items()):
        acc = model_accuracy(grp)
        per_cat[cat] = acc
        if acc["accuracy"] is not None:
            accs.append(acc["accuracy"])
    return {
        "macro_accuracy": sum(accs) / len(accs) if accs else None,
        "n_categories": len(per_cat),
        "by_category": per_cat,
    }


def _table_rows(summary: dict) -> list[dict]:
    """Flatten summary into CSV-friendly rows."""
    fields = (
        "scope",
        "key",
        "n",
        "k",
        "accuracy",
        "ci95_lo",
        "ci95_hi",
        "predict",
        "baseline_predict",
    )
    out: list[dict] = []

    def add(scope: str, key: str, blob: dict, *, extra: dict | None = None) -> None:
        row = {f: None for f in fields}
        row.update(
            {
                "scope": scope,
                "key": key,
                "n": blob.get("n", 0),
                "k": blob.get("k", 0),
                "accuracy": blob.get("accuracy"),
                "ci95_lo": (blob.get("ci95") or [None, None])[0],
                "ci95_hi": (blob.get("ci95") or [None, None])[1],
                "predict": blob.get("predict"),
            }
        )
        if extra:
            row.update(extra)
        out.append(row)

    add("overall", "model", summary["model_accuracy"])
    add("overall", "majority_eval", summary["majority_eval"])
    add("overall", "majority_train_on_eval", summary["majority_train_on_eval"])

    for uc, blob in sorted(summary.get("by_use_case", {}).items()):
        add("use_case", uc, blob["model_accuracy"], extra={"baseline_predict": blob["majority_eval"]["predict"]})
        add("use_case_baseline", uc, blob["majority_eval"])

    for sev, blob in sorted(summary.get("by_severity", {}).items()):
        add("severity", sev, blob["model_accuracy"])
        add("severity_baseline", sev, blob["majority_eval"])

    macro = summary.get("macro_by_category", {})
    out.append(
        {
            "scope": "macro",
            "key": "category",
            "n": macro.get("n_categories", 0),
            "k": None,
            "accuracy": macro.get("macro_accuracy"),
            "ci95_lo": None,
            "ci95_hi": None,
            "predict": None,
        }
    )
    return out


def plot_baselines(summary: dict, dest: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[b1-baselines] skip plot: matplotlib missing", flush=True)
        return None

    dest.mkdir(parents=True, exist_ok=True)

    # Bar chart: model vs baselines
    labels = ["Model", "Majority\n(eval)", "Majority\n(train→eval)"]
    vals = [
        summary["model_accuracy"]["accuracy"] or 0.0,
        summary["majority_eval"]["accuracy"] or 0.0,
        summary["majority_train_on_eval"]["accuracy"] or 0.0,
    ]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    colors = ["#2980b9", "#95a5a6", "#bdc3c7"]
    xs = np.arange(len(labels))
    ax.bar(xs, vals, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xticks(xs, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Grounding accuracy")
    ax.set_title(f"B1 baselines ({summary['model']} blind, n={summary['n_eval']})")
    fig.tight_layout()
    for ext in (".pdf", ".png"):
        fig.savefig(dest / f"b1_baselines{ext}", bbox_inches="tight", dpi=300 if ext == ".png" else None)
    plt.close(fig)

    # Confusion heatmap (normalized by row)
    cm = summary.get("confusion_overall", {})
    if cm:
        actions = sorted({a for gt in cm for a in cm[gt]})
        gt_labels = sorted(cm.keys())
        mat = np.zeros((len(gt_labels), len(actions)))
        for i, gt in enumerate(gt_labels):
            row_sum = sum(cm[gt].values()) or 1
            for j, act in enumerate(actions):
                mat[i, j] = cm[gt].get(act, 0) / row_sum
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(actions)), actions, rotation=45, ha="right")
        ax.set_yticks(range(len(gt_labels)), gt_labels)
        ax.set_xlabel("Predicted action")
        ax.set_ylabel("Ground truth")
        ax.set_title("Confusion matrix (row-normalized)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        for ext in (".pdf", ".png"):
            fig.savefig(dest / f"b1_confusion{ext}", bbox_inches="tight", dpi=300 if ext == ".png" else None)
        plt.close(fig)

    return dest


def summarize_rows(
    eval_rows: list[dict],
    *,
    train_rows: list[dict],
    model: str,
) -> dict:
    majority_train_label, train_counts = _majority_label(train_rows)
    majority_eval_label, eval_counts = _majority_label(eval_rows)

    by_uc: dict[str, list[dict]] = defaultdict(list)
    by_sev: dict[str, list[dict]] = defaultdict(list)
    for r in eval_rows:
        by_uc[r["use_case"]].append(r)
        by_sev[r["severity"] or "unknown"].append(r)

    return {
        "model": model,
        "prompt_variant": DEFAULT_VARIANT,
        "n_eval": len(eval_rows),
        "n_train": len(train_rows),
        "gt_action_counts_eval": dict(eval_counts),
        "gt_action_counts_train": dict(train_counts),
        "model_accuracy": model_accuracy(eval_rows),
        "majority_eval": majority_baseline(eval_rows, predict=majority_eval_label),
        "majority_train_on_eval": majority_baseline(eval_rows, predict=majority_train_label),
        "confusion_overall": confusion(eval_rows),
        "confusion_by_use_case": {uc: confusion(v) for uc, v in sorted(by_uc.items())},
        "by_use_case": {
            uc: {
                "model_accuracy": model_accuracy(v),
                "majority_eval": majority_baseline(v, predict=_majority_label(v)[0]),
            }
            for uc, v in sorted(by_uc.items())
        },
        "by_severity": {
            sev: {
                "model_accuracy": model_accuracy(v),
                "majority_eval": majority_baseline(v, predict=_majority_label(v)[0]),
            }
            for sev, v in sorted(by_sev.items())
        },
        "macro_by_category": macro_by_category(eval_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="B1 majority-class baselines and confusion")
    parser.add_argument("--tag", default="phase_p")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--out-json", type=Path, default=RESULTS_DIR / "b1_baselines.json")
    parser.add_argument("--out-csv", type=Path, default=RESULTS_DIR / "b1_baselines_table.csv")
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=PAPER_FIG,
        help="Directory for b1_baselines.pdf/.png (also writes b1_confusion)",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    episodes = {e["frame_id"]: e for e in load_manifest()}
    if not episodes:
        print(f"No manifest at {DATASET_ROOT / 'manifest.jsonl'}", file=sys.stderr)
        return 1

    records, _ = load_cache(cache_path(args.tag))
    if not records:
        print(f"Empty perception cache: {cache_path(args.tag)}", file=sys.stderr)
        return 1

    rows = [
        evaluate_record(r, episodes[r["frame_id"]])
        for r in records
        if r.get("frame_id") in episodes
        and r.get("model") == args.model
        and r.get("prompt_variant") == args.variant
    ]
    if not rows:
        print(f"No rows for {args.model}|{args.variant}", file=sys.stderr)
        return 1

    split = load_split()
    train_rows = [r for r in rows if split.get(r["frame_id"]) == "train"]
    test_rows = [r for r in rows if split.get(r["frame_id"]) == "test"]

    summary = {
        "cache": str(cache_path(args.tag)),
        "tag": args.tag,
        "headline": summarize_rows(rows, train_rows=train_rows, model=args.model),
        "test_split": summarize_rows(test_rows, train_rows=train_rows, model=args.model)
        if test_rows
        else None,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2))

    table = _table_rows(summary["headline"])
    fields = (
        "scope",
        "key",
        "n",
        "k",
        "accuracy",
        "ci95_lo",
        "ci95_hi",
        "predict",
        "baseline_predict",
    )
    with args.out_csv.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        w.writerows(table)

    fig_path = None
    if not args.no_plot:
        fig_path = plot_baselines(summary["headline"], args.fig_dir)

    headline = summary["headline"]
    print(
        json.dumps(
            {
                "n_eval": headline["n_eval"],
                "model_accuracy": headline["model_accuracy"]["accuracy"],
                "majority_eval": headline["majority_eval"]["accuracy"],
                "majority_train_on_eval": headline["majority_train_on_eval"]["accuracy"],
                "macro_by_category": headline["macro_by_category"]["macro_accuracy"],
                "out_json": str(args.out_json),
                "out_csv": str(args.out_csv),
                "figures": str(fig_path) if fig_path else None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
