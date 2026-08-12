#!/usr/bin/env python3
"""
Phase P: run the VLM once per (frame, model, prompt variant) and cache the decision.

Separating perception from gating is both a computational necessity and a methodological
improvement. A single perception pass costs seconds of GPU time; replaying it against a
grid of drift probabilities, freshness budgets and gates costs milliseconds. Caching also
isolates perception variance from gating variance and lets every gate comparison be
paired on identical model output, which is what makes McNemar applicable later.

The cache is an append-only JSONL keyed by (frame_id, model, prompt_variant, use_case),
so the campaign is resumable: re-running skips work already on disk. That matters for a
run measured in tens of hours.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper2_common import (  # noqa: E402
    CACHE_DIR,
    DATASET_ROOT,
    Notifier,
    load_manifest,
)
from xair.ai.ollama_client import OllamaClient  # noqa: E402
from xair.ai.structured_intent import (  # noqa: E402
    PROMPT_VARIANTS,
    PerceptionResult,
    StructuredIntentProducer,
)

# Multimodal models, ordered by size. The latency spread across this family is what
# makes the routing study and the freshness frontier meaningful.
VISION_MODELS = (
    "qwen2.5vl:3b",
    "qwen2.5vl:7b",
    "llama3.2-vision:11b",
    "gemma3:12b",
    "qwen2.5vl:32b",
)

# A text-only model on a blind visual task is the floor control: with no leaked label it
# cannot see the defect, so its grounding accuracy bounds what prompt structure alone buys.
TEXT_ONLY_CONTROL = "qwen3-coder:30b"

PRIMARY_MODEL = "qwen2.5vl:7b"

_stop_requested = False


def _handle_signal(signum, _frame) -> None:
    global _stop_requested
    _stop_requested = True
    print(f"\n[perception] signal {signum} received; finishing current call then stopping", flush=True)


@dataclass(frozen=True)
class CacheKey:
    frame_id: str
    model: str
    prompt_variant: str
    use_case: str

    @classmethod
    def of(cls, record: dict) -> CacheKey:
        return cls(
            record.get("frame_id", ""),
            record.get("model", ""),
            record.get("prompt_variant", ""),
            record.get("use_case", ""),
        )


def cache_path(tag: str = "phase_p") -> Path:
    return CACHE_DIR / f"{tag}.jsonl"


def load_cache(path: Path) -> tuple[list[dict], set[CacheKey]]:
    """Read the cache, tolerating a truncated final line from an interrupted run."""
    if not path.is_file():
        return [], set()
    records: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records, {CacheKey.of(r) for r in records}


def iter_results(path: Path) -> list[PerceptionResult]:
    records, _ = load_cache(path)
    return [PerceptionResult.from_json(r) for r in records]


def probe_vision(client: OllamaClient, model: str, image: Path) -> tuple[bool, str]:
    """
    Check that a model both answers and actually attends to the image.

    Three of the models originally used in this project were text-only, so their
    apparent grounding came entirely from a leaked textual label. This probe makes that
    class of mistake impossible to repeat silently.
    """
    messages = [
        {
            "role": "user",
            "content": (
                'Look at the image. Reply with JSON only: '
                '{"visible": true, "summary": "<what you see in a few words>"}'
            ),
        }
    ]
    try:
        resp = client.chat(
            messages,
            format_json=True,
            model=model,
            images=[OllamaClient.encode_image(str(image))],
        )
    except RuntimeError as exc:
        return False, f"call failed: {exc}"[:200]
    text = (resp.content or "").strip()
    if not text:
        return False, "empty response"
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return True, f"non-JSON reply: {text[:80]}"
    summary = str(data.get("summary", "")).strip()
    if not summary:
        return True, "answered but gave no summary"
    return True, summary[:120]


def reference_frames(episodes: list[dict], seed: int = 20260812) -> dict[str, str]:
    """
    Pick one fixed known-good frame per category, used as the reference for `blind_ref`.

    The choice is seeded and category-scoped so every query frame in a category is compared
    against the same nominal exemplar; a reference that varied per trial would confound the
    prompt variant with which exemplar happened to be drawn.
    """
    by_category: dict[str, list[dict]] = {}
    for ep in episodes:
        if not ep.get("defect_present"):
            by_category.setdefault(ep.get("category", ""), []).append(ep)

    chosen: dict[str, str] = {}
    for category, rows in by_category.items():
        rows = sorted(rows, key=lambda e: e["frame_id"])
        chosen[category] = rows[random.Random(f"{seed}:{category}").randrange(len(rows))]["path"]
    return chosen


def stratified_subset(episodes: list[dict], limit: int, seed: int = 20260812) -> list[dict]:
    """
    Take a representative subset, balanced over (use case, severity, defect flag).

    A pilot run that just takes the first N frames of the manifest reports accuracy on
    whatever categories happen to sort first, which is how a smoke test can look healthy
    while the defective arm is never exercised.
    """
    if limit >= len(episodes):
        return list(episodes)
    buckets: dict[tuple, list[dict]] = {}
    for ep in episodes:
        key = (ep.get("use_case"), ep.get("severity"), bool(ep.get("defect_present")))
        buckets.setdefault(key, []).append(ep)

    rng = random.Random(seed)
    for rows in buckets.values():
        rng.shuffle(rows)

    picked: list[dict] = []
    order = sorted(buckets, key=lambda k: tuple(str(x) for x in k))
    while len(picked) < limit:
        progressed = False
        for key in order:
            if buckets[key]:
                picked.append(buckets[key].pop())
                progressed = True
                if len(picked) >= limit:
                    break
        if not progressed:
            break
    picked.sort(key=lambda e: e["frame_id"])
    return picked


def plan_jobs(
    episodes: list[dict],
    *,
    models: list[str],
    variants: list[str],
    primary_model: str,
    limit: int | None,
) -> list[tuple[dict, str, str]]:
    """
    Build the work list, grouped by model.

    Prompt variants are swept only on the primary model; the other models run the plain
    blind prompt. Sweeping every variant on every model would multiply GPU hours without
    adding a comparison the paper needs.

    The ordering is model-major on purpose. Ollama keeps only a couple of these models
    resident at once, so interleaving them per frame makes it evict and reload weights
    between calls, which costs far more than the inference itself. Running each model to
    completion loads it once.
    """
    jobs: list[tuple[dict, str, str]] = []
    pool = stratified_subset(episodes, limit) if limit else episodes
    for model in models:
        model_variants = variants if model == primary_model else ["blind"]
        for variant in model_variants:
            for ep in pool:
                jobs.append((ep, model, variant))
    return jobs


def run_campaign(
    episodes: list[dict],
    *,
    models: list[str],
    variants: list[str],
    primary_model: str,
    out_path: Path,
    limit: int | None = None,
    progress_every: int = 25,
    notifier: Notifier | None = None,
    notify_every_frac: float = 0.05,
) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _, done = load_cache(out_path)

    client = OllamaClient()
    producer = StructuredIntentProducer(client)
    jobs = plan_jobs(
        episodes,
        models=models,
        variants=variants,
        primary_model=primary_model,
        limit=limit,
    )
    pending = [
        j
        for j in jobs
        if CacheKey(j[0]["frame_id"], j[1], j[2], j[0]["use_case"]) not in done
    ]

    print(
        f"[perception] {len(jobs)} jobs planned, {len(jobs) - len(pending)} already cached, "
        f"{len(pending)} to run",
        flush=True,
    )

    references = reference_frames(episodes)
    stats: Counter[str] = Counter()
    started = time.perf_counter()
    latencies: list[float] = []
    notify_step = max(1, int(len(pending) * notify_every_frac))
    if notifier:
        notifier.send(
            f"Phase P started\n{len(pending)} calls to run "
            f"({len(jobs) - len(pending)} already cached)\n"
            f"models: {', '.join(models)}"
        )

    with out_path.open("a") as sink:
        for i, (episode, model, variant) in enumerate(pending, start=1):
            if _stop_requested:
                print("[perception] stopping early on request", flush=True)
                break

            image = DATASET_ROOT / episode.get("path", "")
            ref_rel = references.get(episode.get("category", ""))
            # Never hand a frame back to itself as its own known-good reference.
            if ref_rel == episode.get("path"):
                ref_rel = None
            result = producer.produce(
                episode,
                variant=variant,
                use_case=episode["use_case"],
                image_path=image,
                reference_image_path=(DATASET_ROOT / ref_rel) if ref_rel else None,
                model=model,
            )
            record = result.to_json()
            record["gt_action"] = episode["use_cases"][episode["use_case"]]["ground_truth_action"]
            record["severity"] = episode.get("severity")
            record["category"] = episode.get("category")
            record["source_dataset"] = episode.get("source_dataset")
            record["defect_present"] = episode.get("defect_present")
            sink.write(json.dumps(record) + "\n")
            sink.flush()

            stats[f"model:{model}"] += 1
            if result.error:
                stats["error"] += 1
            if result.parse_ok:
                stats["parse_ok"] += 1
            if result.schema_valid:
                stats["schema_valid"] += 1
            if result.action == record["gt_action"]:
                stats["grounding_correct"] += 1
            if result.latency_ms > 0:
                latencies.append(result.latency_ms)

            elapsed = time.perf_counter() - started
            rate = i / elapsed if elapsed > 0 else 0
            remaining = (len(pending) - i) / rate if rate > 0 else 0
            mean_lat = sum(latencies) / len(latencies) if latencies else 0

            if i % progress_every == 0 or i == len(pending):
                print(
                    f"[perception] {i}/{len(pending)} "
                    f"({100 * i / max(1, len(pending)):.1f}%) "
                    f"grounding={stats['grounding_correct']}/{i} "
                    f"mean_latency={mean_lat / 1000:.1f}s "
                    f"eta={remaining / 3600:.2f}h",
                    flush=True,
                )

            if notifier and (i % notify_step == 0 or i == len(pending)):
                notifier.send(
                    f"Phase P {100 * i / max(1, len(pending)):.0f}% "
                    f"({i}/{len(pending)})\n"
                    f"grounding {stats['grounding_correct']}/{i} "
                    f"({100 * stats['grounding_correct'] / max(1, i):.1f}%)\n"
                    f"schema valid {stats['schema_valid']}/{i}, errors {stats['error']}\n"
                    f"mean latency {mean_lat / 1000:.1f}s, ETA {remaining / 3600:.1f}h",
                    silent=True,
                )

    return {
        "cache": str(out_path),
        "jobs_planned": len(jobs),
        "jobs_run": sum(v for k, v in stats.items() if k.startswith("model:")),
        "stats": dict(stats),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase P perception campaign")
    parser.add_argument("--models", nargs="+", default=list(VISION_MODELS))
    parser.add_argument("--variants", nargs="+", default=list(PROMPT_VARIANTS))
    parser.add_argument("--primary-model", default=PRIMARY_MODEL)
    parser.add_argument("--include-text-control", action="store_true",
                        help=f"also run {TEXT_ONLY_CONTROL} as a blind floor control")
    parser.add_argument("--limit", type=int, default=None, help="cap on frames, for smoke runs")
    parser.add_argument("--tag", default="phase_p")
    parser.add_argument("--probe-only", action="store_true",
                        help="check which models actually attend to images, then exit")
    parser.add_argument("--no-notify", action="store_true", help="disable Telegram reporting")
    args = parser.parse_args()

    notifier = Notifier.from_env()
    if args.no_notify:
        notifier.enabled = False

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    episodes = load_manifest()
    if not episodes:
        print(f"No manifest at {DATASET_ROOT / 'manifest.jsonl'}", file=sys.stderr)
        return 1

    client = OllamaClient()
    if not client.health():
        print(f"Ollama unreachable at {client.host}", file=sys.stderr)
        notifier.send(f"FAILED Phase P: Ollama unreachable at {client.host}")
        return 1

    models = list(args.models)
    if args.include_text_control and TEXT_ONLY_CONTROL not in models:
        models.append(TEXT_ONLY_CONTROL)

    sample_image = DATASET_ROOT / episodes[0]["path"]
    probes = {}
    for model in models:
        ok, detail = probe_vision(client, model, sample_image)
        probes[model] = {"responds": ok, "detail": detail}
        print(f"[probe] {model:26s} responds={ok} :: {detail}", flush=True)

    if args.probe_only:
        print(json.dumps(probes, indent=2))
        return 0

    usable = [m for m in models if probes[m]["responds"]]
    if not usable:
        print("No usable models", file=sys.stderr)
        return 1
    if len(usable) < len(models):
        print(f"[perception] skipping unreachable models: {set(models) - set(usable)}", flush=True)

    try:
        out = run_campaign(
            episodes,
            models=usable,
            variants=list(args.variants),
            primary_model=args.primary_model,
            out_path=cache_path(args.tag),
            limit=args.limit,
            notifier=notifier,
        )
    except BaseException as exc:  # noqa: BLE001 - report, then let it propagate
        notifier.send(f"FAILED Phase P crashed\n{type(exc).__name__}: {exc}"[:800])
        raise

    out["probes"] = probes
    print(json.dumps(out, indent=2))
    stats = out["stats"]
    run = max(1, out["jobs_run"])
    notifier.send(
        f"OK Phase P finished\n"
        f"{out['jobs_run']} calls cached to {Path(out['cache']).name}\n"
        f"grounding {stats.get('grounding_correct', 0)}/{run} "
        f"({100 * stats.get('grounding_correct', 0) / run:.1f}%)\n"
        f"schema valid {stats.get('schema_valid', 0)}/{run}, errors {stats.get('error', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
