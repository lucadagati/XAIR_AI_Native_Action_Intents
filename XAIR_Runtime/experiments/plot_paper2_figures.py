#!/usr/bin/env python3
"""Generate Paper 2 figures (B1–B2) for the ai-native-intents manuscript."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
CACHE = RESULTS / "perception_cache" / "phase_p.jsonl"
PAPER_FIG = ROOT.parent / "ResearchTrack" / "ai-native-intents-paper" / "figures"

GATE_ORDER = ("direct", "freshness_only", "xair")
GATE_LABEL = {
    "direct": "Direct",
    "freshness_only": "Freshness-only",
    "xair": "XAIR",
}
GATE_COLOR = {
    "direct": "#c0392b",
    "freshness_only": "#e67e22",
    "xair": "#27ae60",
}
MARKER = {"capture": "o", "emission": "s"}
HEADLINE = dict(freshness_ms=500, p_drift=0.5, drift_offset_ms=250, anchor="capture")


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
        }
    )


def save(fig, dest: Path, stem: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(dest / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def parse_ci(raw: str) -> tuple[float, float]:
    try:
        lo, hi = ast.literal_eval(raw)
        return float(lo), float(hi)
    except (ValueError, SyntaxError, TypeError):
        return 0.0, 1.0


def load_b2_grid(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            row["freshness_ms"] = int(row["freshness_ms"])
            row["p_drift"] = float(row["p_drift"])
            row["drift_offset_ms"] = int(float(row["drift_offset_ms"]))
            row["SAR"] = float(row["SAR"])
            row["SER"] = float(row["SER"])
            row["WRR"] = float(row["WRR"])
            row["utility"] = float(row["utility"])
            row["hazardous_publish_rate"] = float(row["hazardous_publish_rate"])
            row["SAR_ci95"] = parse_ci(row["SAR_ci95"])
            row["SER_ci95"] = parse_ci(row["SER_ci95"])
            rows.append(row)
    return rows


def filter_rows(rows: list[dict], **kwargs) -> list[dict]:
    out = rows
    for key, val in kwargs.items():
        out = [r for r in out if r.get(key) == val]
    return out


def plot_validity_frontier(rows: list[dict], dest: Path) -> None:
    """SAR, SER, and utility vs freshness window at the headline volatility setting."""
    subset = filter_rows(
        rows,
        anchor="capture",
        p_drift=HEADLINE["p_drift"],
        drift_offset_ms=HEADLINE["drift_offset_ms"],
    )
    windows = sorted({r["freshness_ms"] for r in subset})

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    metrics = [
        ("SAR", "Successful actuation rate", "SAR_ci95"),
        ("SER", "Stale execution rate", "SER_ci95"),
        ("utility", "Scalar utility $U$", None),
    ]
    for ax, (field, ylabel, ci_field) in zip(axes, metrics):
        for gate in GATE_ORDER:
            pts = sorted(
                [r for r in subset if r["gate"] == gate],
                key=lambda r: r["freshness_ms"],
            )
            if not pts:
                continue
            xs = [r["freshness_ms"] for r in pts]
            ys = [r[field] for r in pts]
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2,
                label=GATE_LABEL[gate],
                color=GATE_COLOR[gate],
            )
            if ci_field:
                lo = [r[ci_field][0] for r in pts]
                hi = [r[ci_field][1] for r in pts]
                ax.fill_between(xs, lo, hi, color=GATE_COLOR[gate], alpha=0.15)
        ax.set_xscale("log")
        ax.set_xlabel("Freshness window (ms)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(windows, [str(w) for w in windows])
        ax.legend(loc="best", frameon=True)
    fig.suptitle(
        (
            "Validity frontier (capture anchor, "
            f"$p_{{\\mathrm{{drift}}}}={HEADLINE['p_drift']}$, "
            f"$\\tau={HEADLINE['drift_offset_ms']}\\,\\mathrm{{ms}}$)"
        ),
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    save(fig, dest, "b2_validity_frontier")


def plot_anchor_blindspot(rows: list[dict], dest: Path) -> None:
    """SER under capture vs emission anchoring — the inference-latency blind spot."""
    subset = filter_rows(
        rows,
        gate="freshness_only",
        p_drift=0.75,
        drift_offset_ms=250,
    )
    windows = sorted({r["freshness_ms"] for r in subset})
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for anchor in ("capture", "emission"):
        pts = sorted(
            [r for r in subset if r["anchor"] == anchor],
            key=lambda r: r["freshness_ms"],
        )
        xs = [r["freshness_ms"] for r in pts]
        ys = [r["SER"] for r in pts]
        lo = [r["SER_ci95"][0] for r in pts]
        hi = [r["SER_ci95"][1] for r in pts]
        ax.plot(
            xs,
            ys,
            marker=MARKER[anchor],
            linewidth=2,
            label=f"{anchor} anchor",
            linestyle="-" if anchor == "capture" else "--",
        )
        ax.fill_between(xs, lo, hi, alpha=0.12)
    ax.set_xscale("log")
    ax.set_xlabel("Freshness window (ms)")
    ax.set_ylabel("Stale execution rate (SER)")
    ax.set_title("Emission anchoring hides staleness (freshness-only gate)")
    ax.set_xticks(windows, [str(w) for w in windows])
    ax.legend()
    fig.tight_layout()
    save(fig, dest, "b2_anchor_blindspot")


def plot_hazard_curve(rows: list[dict], dest: Path) -> None:
    """SER grows as drift lands earlier relative to submission (hazard curve)."""
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    offsets = sorted({r["drift_offset_ms"] for r in rows})
    for gate in GATE_ORDER:
        pts = sorted(
            filter_rows(
                rows,
                gate=gate,
                anchor="capture",
                p_drift=0.5,
                freshness_ms=500,
            ),
            key=lambda r: r["drift_offset_ms"],
        )
        if not pts:
            continue
        xs = [r["drift_offset_ms"] for r in pts]
        ys = [r["SER"] for r in pts]
        ax.plot(xs, ys, marker="o", linewidth=2, label=GATE_LABEL[gate], color=GATE_COLOR[gate])
    ax.set_xlabel("Drift offset $\\tau$ after evidence (ms)")
    ax.set_ylabel("Stale execution rate (SER)")
    ax.set_title("Hazard curve ($w{=}500$\\,ms, $p_{\\mathrm{drift}}{=}0.5$, capture anchor)")
    ax.set_xticks(offsets, [str(o) for o in offsets])
    ax.legend()
    fig.tight_layout()
    save(fig, dest, "b2_hazard_curve")


def plot_sar_hazard_tradeoff(rows: list[dict], dest: Path) -> None:
    """Pareto-style view: SAR vs hazardous publish rate across freshness windows."""
    subset = filter_rows(
        rows,
        anchor="capture",
        p_drift=HEADLINE["p_drift"],
        drift_offset_ms=HEADLINE["drift_offset_ms"],
    )
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    for gate in GATE_ORDER:
        pts = sorted(
            [r for r in subset if r["gate"] == gate],
            key=lambda r: r["freshness_ms"],
        )
        ax.plot(
            [r["hazardous_publish_rate"] for r in pts],
            [r["SAR"] for r in pts],
            marker="o",
            linewidth=2,
            color=GATE_COLOR[gate],
            label=GATE_LABEL[gate],
        )
        for r in pts:
            ax.annotate(
                f"{r['freshness_ms']}",
                (r["hazardous_publish_rate"], r["SAR"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
                color=GATE_COLOR[gate],
            )
    ax.set_xlabel("Hazardous publish rate")
    ax.set_ylabel("Successful actuation rate (SAR)")
    ax.set_title("Safety–throughput trade-off (labels = freshness ms)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    save(fig, dest, "b2_sar_hazard_tradeoff")


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def plot_leakage_effect(cache: Path, dest: Path, primary_model: str = "qwen2.5vl:7b") -> None:
    """Blind vs leaky grounding on the primary model (B1 leakage ablation)."""
    if not cache.is_file():
        return
    stats: dict[str, list[bool]] = defaultdict(list)
    for line in cache.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("model") != primary_model:
            continue
        v = r.get("prompt_variant")
        if v not in ("blind", "leaky"):
            continue
        stats[v].append(r.get("action") == r.get("gt_action"))

    if not stats.get("blind") or not stats.get("leaky"):
        return

    labels = ["Blind", "Leaky control"]
    keys = ["blind", "leaky"]
    ns = [len(stats[k]) for k in keys]
    ks = [sum(stats[k]) for k in keys]
    rates = [k / n if n else 0 for k, n in zip(ks, ns)]
    yerr_lo, yerr_hi = [], []
    for i, (k, n) in enumerate(zip(ks, ns)):
        lo, hi = wilson_ci(k, n)
        yerr_lo.append(max(0.0, rates[i] - lo))
        yerr_hi.append(max(0.0, hi - rates[i]))

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    xs = np.arange(len(labels))
    bars = ax.bar(xs, rates, color=["#2980b9", "#e74c3c"], edgecolor="black", width=0.55)
    ax.errorbar(
        xs,
        rates,
        yerr=[yerr_lo, yerr_hi],
        fmt="none",
        ecolor="black",
        capsize=4,
        linewidth=1.2,
    )
    for i, (bar, k, n) in enumerate(zip(bars, ks, ns)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + yerr_hi[i] + 0.03,
            f"{k}/{n}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xticks(xs, labels)
    ax.set_ylabel("Grounding accuracy")
    ax.set_ylim(0, 1.12)
    ax.set_title("Blind vs leaky grounding (primary VLM)")
    fig.tight_layout()
    save(fig, dest, "b1_leakage_effect")


def plot_anchoring_timeline(dest: Path) -> None:
    """Schematic: capture vs emission anchoring on a latency timeline."""
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 4.4), sharex=True)
    t_capture, t_emit, t_submit = 0.0, 3.2, 3.5
    t_drift = 1.0
    freshness = 2.0

    for ax, anchor, title in zip(
        axes,
        ("capture", "emission"),
        (
            "(a) Capture anchor: inference consumes the freshness budget",
            "(b) Emission anchor: inference latency is invisible to the validator",
        ),
    ):
        ax.set_ylim(0, 1.55)
        ax.axvline(t_capture, color="#2c3e50", linewidth=2.2, label="Evidence $t_c$")
        ax.axvline(t_emit, color="#8e44ad", linewidth=2.0, linestyle="--", label="Emission $t_d$")
        ax.axvline(t_drift, color="#c0392b", linewidth=1.8, linestyle=":", label="Drift $\\tau$")
        start = t_capture if anchor == "capture" else t_emit
        ax.axvspan(
            start,
            start + freshness,
            color="#27ae60",
            alpha=0.28,
            label="Freshness window $w$",
            zorder=0,
        )
        ax.axvline(t_submit, color="#2980b9", linewidth=2.2, label="Validation/submit")
        ax.annotate(
            "",
            xy=(t_emit, 0.35),
            xytext=(t_capture, 0.35),
            arrowprops=dict(arrowstyle="<->", color="#566573", lw=1.2),
        )
        ax.text((t_capture + t_emit) / 2, 0.42, r"$\Delta_{\mathrm{inf}}$", ha="center", fontsize=9, color="#566573")
        elapsed = (t_submit - start) if anchor == "capture" else max(0.0, t_submit - t_emit)
        stale = elapsed > freshness
        ax.text(
            min(t_submit + 0.15, 4.6),
            1.15,
            f"elapsed $= {elapsed:.1f}\\,\\mathrm{{s}}$"
            + (" (stale)" if stale else " (fresh)"),
            fontsize=10,
            color="#1a5276",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#2980b9", linewidth=1.0),
        )
        ax.set_yticks([])
        ax.set_title(title, fontsize=11, pad=6)
        ax.legend(loc="upper left", fontsize=8, ncol=3, framealpha=0.95)
        ax.set_xlim(-0.2, 6.2)
    axes[1].set_xlabel("Time since evidence acquisition (s)")
    fig.tight_layout()
    save(fig, dest, "fig_anchoring_timeline")


def _box(ax, x, y, w, h, text, *, facecolor, edgecolor, fontsize=9, fontweight="normal"):
    from matplotlib.patches import FancyBboxPatch

    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.8,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        zorder=3,
        multialignment="center",
        color="#1c2833",
    )
    return patch


def _arrow(ax, start, end, *, color="#2c3e50", style="-|>", lw=1.5, text="", text_offset=(0, 0.12)):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=4, shrinkB=4),
        zorder=1,
    )
    if text:
        mx = (start[0] + end[0]) / 2 + text_offset[0]
        my = (start[1] + end[1]) / 2 + text_offset[1]
        ax.text(
            mx,
            my,
            text,
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=color,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.9),
        )


def plot_architecture_diagram(dest: Path) -> None:
    """End-to-end AI-native stack for the manuscript."""
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    _box(
        ax, 0.3, 7.7, 2.1, 1.0,
        "Camera /\ninspection frame",
        facecolor="#eef2f5", edgecolor="#566573", fontsize=9,
    )
    _box(
        ax, 0.3, 6.4, 2.1, 1.0,
        "MES / OPC-UA\nplant state",
        facecolor="#eef2f5", edgecolor="#566573", fontsize=9,
    )

    _box(
        ax, 3.0, 6.85, 2.6, 1.45,
        "Proposer\n(VLM / agent)\nblind prompt",
        facecolor="#d6eaf8", edgecolor="#1a5276", fontsize=10, fontweight="bold",
    )
    _arrow(ax, (2.4, 8.2), (3.0, 7.7), text="$t_c$")
    _arrow(ax, (2.4, 6.9), (3.0, 7.25))

    _box(
        ax, 6.15, 6.85, 3.0, 1.45,
        "AIS contract\naction, preconditions,\nevidence, $\\Delta_{\\mathrm{inf}}$, anchor",
        facecolor="#fdebd0", edgecolor="#b9770e", fontsize=9.5, fontweight="bold",
    )
    _arrow(ax, (5.6, 7.55), (6.15, 7.55), text="structured JSON")

    _box(
        ax, 6.15, 5.35, 3.0, 0.9,
        "Perception cache\n(amortised Phase P)",
        facecolor="#f5eef8", edgecolor="#6c3483", fontsize=9,
    )
    _arrow(ax, (7.65, 6.85), (7.65, 6.25), color="#6c3483", lw=1.2)

    _box(
        ax, 3.0, 4.35, 2.6, 1.15,
        "Validity budget\n$f(\\mathrm{ctx})\\rightarrow w$, strictness",
        facecolor="#e8f8f5", edgecolor="#0e6655", fontsize=9.5, fontweight="bold",
    )
    _arrow(ax, (6.15, 7.0), (5.6, 5.1), color="#0e6655", text="features", text_offset=(-0.15, 0.05))

    _box(
        ax, 6.15, 3.85, 3.0, 1.35,
        "XAIR gate\nfreshness + preconditions\nrecheck at $t_v$, $t_p$",
        facecolor="#d5f5e3", edgecolor="#196f3d", fontsize=10, fontweight="bold",
    )
    _arrow(ax, (7.65, 6.85), (7.65, 5.2))
    _arrow(ax, (5.6, 4.9), (6.15, 4.7), text="policy")

    _box(
        ax, 6.15, 2.35, 1.35, 0.95,
        "Publish\n(SAR)",
        facecolor="#abebc6", edgecolor="#145a32", fontsize=9.5, fontweight="bold",
    )
    _box(
        ax, 7.8, 2.35, 1.35, 0.95,
        "Revoke\n(SER/WRR)",
        facecolor="#fadbd8", edgecolor="#922b21", fontsize=9.5, fontweight="bold",
    )
    _arrow(ax, (7.65, 3.85), (6.85, 3.3), color="#145a32")
    _arrow(ax, (7.65, 3.85), (8.45, 3.3), color="#922b21")

    _box(
        ax, 5.85, 0.85, 3.3, 1.0,
        "Plant / adapter\n(publication boundary)",
        facecolor="#eaecee", edgecolor="#1c2833", fontsize=10, fontweight="bold",
    )
    _arrow(ax, (6.85, 2.35), (6.85, 1.85), color="#145a32")
    _arrow(ax, (8.45, 2.35), (8.45, 1.85), color="#922b21", text="audit", text_offset=(0.35, 0.02))

    # Drift feeds the gate along a lower lane, clear of the budget box.
    _box(
        ax, 0.3, 2.55, 2.1, 1.15,
        "Stochastic drift\n$p_{\\mathrm{drift}}$, $\\tau$",
        facecolor="#fdedec", edgecolor="#922b21", fontsize=9,
    )
    _arrow(ax, (2.4, 3.15), (6.15, 4.2), color="#922b21", text="context patch", text_offset=(0.4, -0.05))

    ax.text(
        0.3,
        0.25,
        "The model proposes; XAIR authorises. No direct motor or topic publish from the VLM.",
        fontsize=9.5,
        color="#1c2833",
        style="italic",
    )
    ax.set_title("AI-native action-intent architecture", fontsize=13, fontweight="bold", pad=8)
    fig.tight_layout()
    save(fig, dest, "fig_architecture")


def plot_campaign_phases(dest: Path) -> None:
    """Three-phase campaign schematic (left-to-right flow)."""
    fig, ax = plt.subplots(figsize=(7.6, 2.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.1)
    ax.axis("off")
    phases = [
        ("P", "Perception", "Cached VLM decisions\n(amortised GPU cost)", "#3498db"),
        ("G", "Gating", "Offline replay over\nfreshness--volatility grid", "#9b59b6"),
        ("RL", "Learning", "Validity-budget policy\n(LinUCB / Q-learning)", "#27ae60"),
    ]
    w = 2.7
    for i, (tag, name, detail, color) in enumerate(phases):
        x = 0.4 + i * 3.25
        rect = plt.Rectangle(
            (x, 0.22), w, 0.62, facecolor=color, alpha=0.18, edgecolor=color, linewidth=2.2
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, 0.72, f"Phase {tag}", ha="center", va="center", fontsize=12, fontweight="bold")
        ax.text(x + w / 2, 0.52, name, ha="center", va="center", fontsize=11)
        ax.text(x + w / 2, 0.34, detail, ha="center", va="center", fontsize=8.5)
        if i < len(phases) - 1:
            # Tip points toward the next phase (left → right).
            ax.annotate(
                "",
                xy=(x + w + 0.45, 0.52),
                xytext=(x + w + 0.08, 0.52),
                arrowprops=dict(arrowstyle="-|>", lw=2.0, color="#2c3e50"),
            )
    ax.set_title("Three-phase evaluation campaign", fontsize=12, pad=10)
    fig.tight_layout()
    save(fig, dest, "fig_campaign_phases")


def plot_b3_learning_curves(curve_csv: Path, dest: Path) -> None:
    if not curve_csv.is_file():
        return
    by_pol: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with curve_csv.open() as fh:
        for row in csv.DictReader(fh):
            pol = row["policy"]
            if pol.startswith("fixed:"):
                continue
            by_pol[pol].append((int(row["step"]), float(row["mean_reward"])))
    if not by_pol:
        return
    # average across seeds at shared steps
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    colors = {"linucb:a1.0": "#2980b9", "qlearn:e0.15": "#8e44ad"}
    for pol, pts in sorted(by_pol.items()):
        buckets: dict[int, list[float]] = defaultdict(list)
        for step, val in pts:
            buckets[step].append(val)
        xs = sorted(buckets)
        ys = [sum(buckets[x]) / len(buckets[x]) for x in xs]
        ax.plot(xs, ys, linewidth=2.0, label=pol, color=colors.get(pol))
    ax.axhline(0.0, color="#7f8c8d", linewidth=1.0, linestyle=":")
    ax.set_xlabel("Online steps")
    ax.set_ylabel("Cumulative mean reward")
    ax.set_title("B3 learning curves (capture anchor, mixed volatility)")
    ax.legend(frameon=True)
    fig.tight_layout()
    save(fig, dest, "b3_learning_curves")


def plot_b3_policy_comparison(table_csv: Path, dest: Path) -> None:
    if not table_csv.is_file():
        return
    rows = list(csv.DictReader(table_csv.open()))
    if not rows:
        return
    # Keep oracle, learners, and train-selected best fixed.
    keep = []
    for r in rows:
        p = r["policy"]
        if p in ("oracle", "linucb:a1.0", "qlearn:e0.15", "best_fixed:train_selected"):
            keep.append(r)
    keep.sort(key=lambda r: float(r["utility"]), reverse=True)
    labels = [
        r["policy"]
        .replace("best_fixed:train_selected", "Best-fixed")
        .replace("linucb:a1.0", "LinUCB")
        .replace("qlearn:e0.15", "Q-learning")
        for r in keep
    ]
    utils = [float(r["utility"]) for r in keep]
    colors = []
    for r in keep:
        if r["policy"] == "oracle":
            colors.append("#196f3d")
        elif "linucb" in r["policy"]:
            colors.append("#2980b9")
        elif "qlearn" in r["policy"]:
            colors.append("#8e44ad")
        else:
            colors.append("#95a5a6")
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    xs = np.arange(len(labels))
    ax.bar(xs, utils, color=colors, edgecolor="black", width=0.7)
    ax.axhline(0.0, color="#2c3e50", linewidth=1.0)
    ax.set_xticks(xs, labels, rotation=20, ha="right")
    ax.set_ylabel("Aggregate utility $U$")
    ax.set_title("B3 headline: learned budget vs fixed vs oracle")
    for i, u in enumerate(utils):
        ax.text(i, u + (0.03 if u >= 0 else -0.08), f"{u:+.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    save(fig, dest, "b3_policy_comparison")


def plot_b4_static_pareto(pareto_csv: Path, dest: Path) -> None:
    if not pareto_csv.is_file():
        return
    rows = list(csv.DictReader(pareto_csv.open()))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for r in rows:
        x = float(r["latency_p50_ms"]) / 1000.0
        y = float(r["grounding"])
        label = r["model"].replace("qwen2.5vl:", "qwen:").replace("llama3.2-vision:", "llama:")
        ax.scatter([x], [y], s=80, zorder=3)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel(r"Inference latency $p_{50}$ (s)")
    ax.set_ylabel("Blind grounding accuracy")
    ax.set_title("B4 static model Pareto (grounding vs latency)")
    ax.set_ylim(0.3, 0.6)
    fig.tight_layout()
    save(fig, dest, "b4_static_pareto")


def plot_b4_utility_vs_volatility(table_csv: Path, dest: Path) -> None:
    if not table_csv.is_file():
        return
    keep = {
        "oracle": ("Oracle", "#196f3d", "-"),
        "static:qwen2.5vl:7b": ("Always 7b", "#2980b9", "-"),
        "static:qwen2.5vl:3b": ("Always 3b", "#27ae60", "--"),
        "static:qwen2.5vl:32b": ("Always 32b", "#e67e22", "-."),
        "cascade:vol": ("Cascade", "#8e44ad", "-"),
        "cap:8000ms": ("Cap 8s", "#c0392b", ":"),
    }
    order = ["p0", "p25", "p50", "p75"]
    xs = [0.0, 0.25, 0.5, 0.75]
    series: dict[str, list[float]] = {k: [] for k in keep}
    with table_csv.open() as fh:
        rows = list(csv.DictReader(fh))
    for setting, x in zip(order, xs):
        for router in keep:
            match = [r for r in rows if r["router"] == router and r["setting"] == setting]
            series[router].append(float(match[0]["utility"]) if match else float("nan"))
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    for router, (label, color, ls) in keep.items():
        ax.plot(xs, series[router], marker="o", linewidth=2.0, label=label, color=color, linestyle=ls)
    ax.set_xlabel(r"Plant volatility $p_{\mathrm{drift}}$")
    ax.set_ylabel("Aggregate utility $U$")
    ax.set_title("B4 routing under increasing volatility (capture, adaptive $w$)")
    ax.legend(fontsize=8, ncol=2, frameon=True)
    fig.tight_layout()
    save(fig, dest, "b4_utility_vs_volatility")


def plot_b5_policy_comparison(headline_csv: Path, dest: Path) -> None:
    if not headline_csv.is_file():
        return
    rows = list(csv.DictReader(headline_csv.open()))
    if not rows:
        return
    # Prefer a stable display order.
    prefer = [
        "oracle",
        "qlearn:e0.1",
        "single_shot",
        "always_reobserve",
    ]
    by = {r["policy"]: r for r in rows}
    order = [p for p in prefer if p in by]
    labels = {
        "oracle": "Oracle",
        "single_shot": "Single-shot",
        "retry_stale": "Retry stale",
        "always_reobserve": "Always re-obs.",
        "always_escalate": "Always escalate",
        "linucb:a0.75": "LinUCB",
        "qlearn:e0.1": "Q-learning",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    xs = list(range(len(order)))
    u = [float(by[p]["utility"]) for p in order]
    haz = [float(by[p]["hazard"]) for p in order]
    sar = [float(by[p]["SAR"]) for p in order]
    names = [labels.get(p, p) for p in order]
    colors = [
        "#196f3d" if p == "oracle" else "#c0392b" if p == "always_reobserve" else "#2980b9"
        for p in order
    ]
    axes[0].bar(xs, u, color=colors, edgecolor="black", linewidth=0.6)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    axes[0].set_ylabel("Utility $U$")
    axes[0].set_title("B5 post-revocation policy simulation")
    # Legend markers instead of overlapping point labels.
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for i, p in enumerate(order):
        axes[1].scatter(
            [sar[i]],
            [haz[i]],
            s=70,
            c=[colors[i]],
            marker=markers[i % len(markers)],
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
            label=names[i],
        )
    axes[1].set_xlabel("SAR")
    axes[1].set_ylabel("Hazard")
    axes[1].set_title("SAR–hazard trade-off")
    axes[1].legend(fontsize=6, loc="best", frameon=True, borderpad=0.3, labelspacing=0.25)
    fig.tight_layout()
    save(fig, dest, "b5_policy_comparison")


def plot_b5_learning_curves(curve_csv: Path, dest: Path) -> None:
    if not curve_csv.is_file():
        return
    rows = [r for r in csv.DictReader(curve_csv.open()) if r["setting"] == "headline"]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for policy, color in (("linucb:a0.75", "#8e44ad"), ("qlearn:e0.1", "#e67e22")):
        pts = [r for r in rows if r["policy"] == policy]
        if not pts:
            continue
        # Average across seeds at each episode index.
        by_ep: dict[int, list[float]] = {}
        for r in pts:
            by_ep.setdefault(int(r["episode"]), []).append(float(r["mean_total_reward"]))
        xs = sorted(by_ep)
        ys = [sum(by_ep[e]) / len(by_ep[e]) for e in xs]
        ax.plot(xs, ys, color=color, linewidth=2.0, label=policy.split(":")[0])
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("Rolling mean episode reward")
    ax.set_title("B5 online learning after revocation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, dest, "b5_learning_curves")


def plot_b5_precover_sweep(csv_path: Path, dest: Path) -> None:
    if not csv_path.is_file():
        return
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        return
    keep = {
        "single_shot": ("Single-shot", "#2980b9", "-"),
        "always_reobserve": ("Always re-obs.", "#c0392b", "-"),
        "oracle": ("Oracle", "#196f3d", "--"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for policy, (label, color, ls) in keep.items():
        pts = sorted((float(r["p_recover"]), float(r["utility"]), float(r["hazard"])) for r in rows if r["policy"] == policy)
        if not pts:
            continue
        xs = [p[0] for p in pts]
        axes[0].plot(xs, [p[1] for p in pts], marker="o", color=color, linestyle=ls, label=label, linewidth=2.0)
        axes[1].plot(xs, [p[2] for p in pts], marker="o", color=color, linestyle=ls, label=label, linewidth=2.0)
    axes[0].set_xlabel(r"$p_{\mathrm{recover}}$")
    axes[0].set_ylabel("Utility $U$")
    axes[0].set_title("B5 utility vs plant recovery")
    axes[1].set_xlabel(r"$p_{\mathrm{recover}}$")
    axes[1].set_ylabel("Hazard")
    axes[1].set_title("B5 hazard vs plant recovery")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    save(fig, dest, "b5_precover_sweep")


def plot_b1_model_table(json_path: Path, dest: Path) -> None:
    if not json_path.is_file():
        return
    data = json.loads(json_path.read_text())
    models = []
    for key, blob in sorted(data.get("by_model_variant", {}).items()):
        if not key.endswith("|blind"):
            continue
        model = key.split("|")[0]
        models.append(
            (
                model.replace("qwen2.5vl:", "qwen:").replace("llama3.2-vision:", "llama:"),
                float(blob["grounding_accuracy"]["rate"]),
                float(blob["protective_rate"]["rate"]),
                float(blob["self_catch_rate"]["rate"]),
                float(blob["false_block_rate"]["rate"]),
            )
        )
    if not models:
        return
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    xs = np.arange(len(models))
    w = 0.2
    for i, (name, color) in enumerate(
        (("Grounding", "#2980b9"), ("Protective", "#27ae60"), ("Self-catch", "#8e44ad"), ("False-block", "#e67e22"))
    ):
        vals = [m[i + 1] for m in models]
        ax.bar(xs + (i - 1.5) * w, vals, width=w, label=name, color=color, edgecolor="black", linewidth=0.4)
    ax.set_xticks(xs, [m[0] for m in models], rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("B1 blind grounding and precondition quality by model")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    save(fig, dest, "b1_model_preconditions")


def plot_downscale_ablation(json_path: Path, dest: Path) -> None:
    if not json_path.is_file():
        return
    data = json.loads(json_path.read_text())
    by = data.get("summary", {}).get("by_side", {})
    if not by:
        return
    sides = sorted(by.keys(), key=lambda s: int(s))
    g = [by[s]["grounding"] for s in sides]
    lat = [by[s]["latency_p50_ms"] / 1000.0 for s in sides]
    fig, ax1 = plt.subplots(figsize=(5.6, 3.4))
    ax2 = ax1.twinx()
    ax1.plot([int(s) for s in sides], g, "o-", color="#2980b9", linewidth=2.0, label="Grounding")
    ax2.plot([int(s) for s in sides], lat, "s--", color="#e67e22", linewidth=2.0, label="Latency $p_{50}$")
    ax1.set_xlabel("Encode max side (px)")
    ax1.set_ylabel("Blind grounding")
    ax2.set_ylabel("Latency $p_{50}$ (s)")
    ax1.set_title("Downscale ablation (7B blind subsample)")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=8)
    fig.tight_layout()
    save(fig, dest, "b1_downscale_ablation")


def plot_dataset_collage(dest: Path) -> None:
    """Nominal vs defective examples from VisA + MVTec AD."""
    try:
        from PIL import Image
    except ImportError:
        print("[paper2-fig] skip dataset collage: Pillow missing")
        return
    root = ROOT / "experiments" / "datasets" / "manufacturing-a1"
    manifest = root / "manifest.jsonl"
    if not manifest.is_file():
        return
    rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    by: dict[tuple, dict[str, list]] = defaultdict(lambda: {"ok": [], "def": []})
    for r in rows:
        key = (r.get("source_dataset"), r.get("category"))
        bucket = "def" if r.get("defect_present") else "ok"
        by[key][bucket].append(r)
    prefer = [
        ("mvtec_ad", "bottle"),
        ("visa", "pcb1"),
        ("mvtec_ad", "cable"),
        ("visa", "cashew"),
    ]
    picked = [k for k in prefer if by[k]["ok"] and by[k]["def"]][:4]
    if len(picked) < 4:
        return
    fig, axes = plt.subplots(2, 4, figsize=(9.4, 4.7))
    for col, key in enumerate(picked):
        src, cat = key
        for row_i, kind in enumerate(("ok", "def")):
            ax = axes[row_i][col]
            cand = sorted(by[key][kind], key=lambda r: r["frame_id"])
            rec = cand[len(cand) // 3]
            img = Image.open(root / rec["path"]).convert("RGB")
            w, h = img.size
            side = min(w, h)
            left, top = (w - side) // 2, (h - side) // 2
            img = img.crop((left, top, left + side, top + side)).resize(
                (384, 384), Image.Resampling.LANCZOS
            )
            ax.imshow(np.asarray(img))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#c0392b" if kind == "def" else "#1a5276")
                spine.set_linewidth(2.0)
            src_lab = "MVTec AD" if src == "mvtec_ad" else "VisA"
            if row_i == 0:
                ax.set_title(f"{src_lab}\n{cat.replace('_', ' ')}", fontsize=9, pad=3)
            if col == 0:
                ax.set_ylabel(
                    "Nominal" if kind == "ok" else "Defective",
                    fontsize=10,
                    fontweight="bold",
                )
            if kind == "def":
                sev = rec.get("severity") or "minor"
                ax.text(
                    0.03,
                    0.97,
                    sev,
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=8,
                    color="white",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="#c0392b",
                        edgecolor="none",
                        alpha=0.85,
                    ),
                )
    fig.suptitle(
        "Stratified inspection corpus: VisA and MVTec AD (nominal vs defective)",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0.01, 1, 0.96])
    save(fig, dest, "dataset_collage")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=PAPER_FIG)
    parser.add_argument("--b2-csv", type=Path, default=RESULTS / "b2_validity_frontier.csv")
    parser.add_argument("--cache", type=Path, default=CACHE)
    args = parser.parse_args()
    apply_style()

    if args.b2_csv.is_file():
        rows = load_b2_grid(args.b2_csv)
        plot_validity_frontier(rows, args.out)
        plot_anchor_blindspot(rows, args.out)
        plot_hazard_curve(rows, args.out)
        plot_sar_hazard_tradeoff(rows, args.out)
        print(f"[paper2-fig] B2 plots from {len(rows)} grid cells -> {args.out}")
    else:
        print(f"[paper2-fig] skip B2: {args.b2_csv} missing")

    plot_leakage_effect(args.cache, args.out)
    plot_dataset_collage(args.out)
    plot_anchoring_timeline(args.out)
    plot_architecture_diagram(args.out)
    plot_campaign_phases(args.out)
    plot_b3_learning_curves(RESULTS / "b3_learning_curves.csv", args.out)
    plot_b3_policy_comparison(RESULTS / "b3_headline_table.csv", args.out)
    plot_b4_static_pareto(RESULTS / "b4_static_pareto.csv", args.out)
    plot_b4_utility_vs_volatility(RESULTS / "b4_routing_table.csv", args.out)
    plot_b5_policy_comparison(RESULTS / "b5_headline_table.csv", args.out)
    plot_b5_learning_curves(RESULTS / "b5_learning_curves.csv", args.out)
    plot_b5_precover_sweep(RESULTS / "b5_precover_sweep.csv", args.out)
    plot_b1_model_table(RESULTS / "b1_grounding.json", args.out)
    plot_downscale_ablation(RESULTS / "b1_downscale_ablation.json", args.out)
    print(f"[paper2-fig] schematic + B1--B5 extras -> {args.out}")


if __name__ == "__main__":
    main()
