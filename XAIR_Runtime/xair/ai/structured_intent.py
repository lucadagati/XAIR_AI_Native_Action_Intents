"""
Produce Action Intents from a VLM, keeping perception and validity contract separate.

Design constraints that this module exists to enforce:

* The model is never told the inspection verdict or the target action in blind variants.
  Only the ``leaky`` control variant leaks them, and it exists solely as an ablation.
* Preconditions are whatever the model emitted. Nothing is injected silently; a caller
  that wants harness-supplied preconditions must pass them explicitly to
  :func:`build_submission`.
* The temporal fields are NOT set here. Perception yields a capture timestamp and an
  emission timestamp; the gating layer decides which one anchors validity. That choice
  is an experimental factor, not a hidden default.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from xair.ai.ollama_client import OllamaClient, OllamaResponse

_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

BLIND_PROMPT = _PROMPT_DIR / "ais_blind.txt"
LEAKY_PROMPT = _PROMPT_DIR / "ais_leaky_control.txt"

ACTION_TYPES = (
    "ACCEPT",
    "REJECT_TO_BIN",
    "HOLD_FOR_OPERATOR",
    "SLOW_DOWN",
    "RESUME",
    "STOP",
    "E_STOP",
)

USE_CASES = {
    "uc1_triage": (
        "Decide whether this part may continue down the line. Choose ACCEPT, "
        "REJECT_TO_BIN, or HOLD_FOR_OPERATOR."
    ),
    "uc2_restart": (
        "The cell is being considered for restart. Decide whether the line may RESUME "
        "or must STOP."
    ),
    "uc3_speed": (
        "Decide the line speed for the next segment. Choose SLOW_DOWN with a "
        "parameters.target_speed value, or ACCEPT to keep the current speed."
    ),
    "uc4_conflict": (
        "An operator has independently requested a line stop. Propose the action you "
        "believe correct from the image: RESUME or STOP."
    ),
    "uc5_safety": (
        "Judge whether this scene shows an immediate hazard requiring E_STOP, or "
        "whether ACCEPT is appropriate."
    ),
}

# Only variants listed here are permitted to see ground-truth labels.
LEAKY_VARIANTS = frozenset({"leaky"})
# Variants that receive a second, known-good image of the same part category. This is how
# industrial anomaly detection is normally posed to a VLM, and it separates "this model
# cannot judge this part" from "this model was never shown what nominal looks like".
REFERENCE_VARIANTS = frozenset({"blind_ref"})
PROMPT_VARIANTS = ("blind", "blind_cot", "blind_noctx", "blind_ref", "leaky")

_CONTEXT_DOC_PATHS = ("line.state", "gripper.state", "robot.speed", "defect.absent")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _iso(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        return json.loads(m.group())
    raise ValueError("no JSON object in model output")


def _as_float(val, default: float | None = None) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


@dataclass
class PerceptionResult:
    """One VLM decision, with everything needed to replay it through any gate."""

    frame_id: str
    use_case: str
    prompt_variant: str
    model: str
    capture_ts: str
    emitted_ts: str
    latency_ms: float
    action: str
    parameters: dict = field(default_factory=dict)
    preconditions: list[str] = field(default_factory=list)
    preconditions_repaired: list[str] = field(default_factory=list)
    confidence: float | None = None
    observation: str = ""
    defect_judgement: bool | None = None
    severity: str | None = None
    reasoning: str = ""
    parse_ok: bool = False
    schema_valid: bool = False
    schema_valid_repaired: bool = False
    action_valid: bool = False
    precondition_syntax_ok: bool = True
    precondition_syntax_ok_repaired: bool = True
    preconditions_repaired_count: int = 0
    reference_used: bool = False
    raw_content: str = ""
    prompt_hash: str = ""
    evidence: dict = field(default_factory=dict)
    error: str | None = None

    def to_json(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "use_case": self.use_case,
            "prompt_variant": self.prompt_variant,
            "model": self.model,
            "capture_ts": self.capture_ts,
            "emitted_ts": self.emitted_ts,
            "latency_ms": self.latency_ms,
            "action": self.action,
            "parameters": self.parameters,
            "preconditions": self.preconditions,
            "preconditions_repaired": self.preconditions_repaired,
            "confidence": self.confidence,
            "observation": self.observation,
            "defect_judgement": self.defect_judgement,
            "severity": self.severity,
            "reasoning": self.reasoning,
            "parse_ok": self.parse_ok,
            "schema_valid": self.schema_valid,
            "schema_valid_repaired": self.schema_valid_repaired,
            "action_valid": self.action_valid,
            "precondition_syntax_ok": self.precondition_syntax_ok,
            "precondition_syntax_ok_repaired": self.precondition_syntax_ok_repaired,
            "preconditions_repaired_count": self.preconditions_repaired_count,
            "reference_used": self.reference_used,
            "prompt_hash": self.prompt_hash,
            "evidence": self.evidence,
            "error": self.error,
        }

    @classmethod
    def from_json(cls, data: dict) -> PerceptionResult:
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        known.setdefault("frame_id", "")
        known.setdefault("use_case", "uc1_triage")
        known.setdefault("prompt_variant", "blind")
        known.setdefault("model", "")
        known.setdefault("capture_ts", "")
        known.setdefault("emitted_ts", "")
        known.setdefault("latency_ms", 0.0)
        known.setdefault("action", "")
        return cls(**known)


# Mirrors ContextValidator._PATTERN so we can score precondition syntax without a runtime.
_PRECOND_PATTERN = re.compile(
    r"^(?P<path>\w+(?:\.\w+)*)\s*"
    r"(?P<op>==|!=|<=|>=|<|>)\s*"
    r"(?:'(?P<sval>[^']*)'"
    r"|\"(?P<dval>[^\"]*)\""
    r"|(?P<bval>true|false|True|False)"
    r"|(?P<nval>-?\d+(?:\.\d+)?))$"
)


def precondition_syntax_ok(expr: str) -> bool:
    return bool(_PRECOND_PATTERN.match(expr.strip()))


# A bare word where a literal belongs: `line.state != RUN` instead of `line.state != 'RUN'`.
_BARE_LITERAL = re.compile(
    r"^(?P<lhs>\w+(?:\.\w+)*\s*(?:==|!=|<=|>=|<|>)\s*)(?P<lit>[A-Za-z_][\w\-]*)$"
)
# A path with no comparison at all: `defect.absent` instead of `defect.absent == true`.
_BARE_PATH = re.compile(r"^(?P<path>\w+(?:\.\w+)+)$")


def repair_precondition(expr: str) -> str:
    """
    Apply one documented normalisation pass to a model-emitted precondition.

    Two repairs are performed, both of them punctuation rather than meaning:

      * an unquoted string literal is quoted (``line.state != RUN``), and
      * a bare path is read as a truthiness test (``defect.absent`` becomes
        ``defect.absent == true``), which is the only reading such an entry admits.

    No repair changes an operator, a path or a value, so a repaired expression can never
    mean something the model did not write. Expressions that are wrong in substance rather
    than in punctuation, such as compound conditions or copied placeholders, are returned
    unchanged and stay invalid.
    """
    text = expr.strip().rstrip(";")
    if precondition_syntax_ok(text):
        return text

    m = _BARE_LITERAL.match(text)
    if m and m.group("lit").lower() not in ("true", "false"):
        candidate = f"{m.group('lhs')}'{m.group('lit')}'"
        if precondition_syntax_ok(candidate):
            return candidate

    m = _BARE_PATH.match(text)
    if m:
        candidate = f"{m.group('path')} == true"
        if precondition_syntax_ok(candidate):
            return candidate

    return text


class StructuredIntentProducer:
    """Turn an image plus reported plant state into a proposed Action Intent."""

    def __init__(self, client: OllamaClient | None = None):
        self.client = client or OllamaClient()
        self.blind_system = BLIND_PROMPT.read_text()
        self.leaky_system = LEAKY_PROMPT.read_text()

    def system_prompt(self, variant: str) -> str:
        return self.leaky_system if variant in LEAKY_VARIANTS else self.blind_system

    def build_user_prompt(self, episode: dict, *, variant: str, use_case: str) -> str:
        if variant not in PROMPT_VARIANTS:
            raise ValueError(f"unknown prompt variant: {variant}")

        parts: list[str] = []

        if variant != "blind_noctx":
            ctx = episode.get("context", {})
            parts.append("Plant state reported by the MES:")
            parts.append(f"  line.state = {ctx.get('line', {}).get('state', 'RUN')}")
            parts.append(f"  gripper.state = {ctx.get('gripper', {}).get('state', 'OPEN')}")
            parts.append(f"  robot.speed = {ctx.get('robot', {}).get('speed', 0.05)}")

        if variant in REFERENCE_VARIANTS:
            parts.append(
                "You are given two images. The FIRST is a known-good reference part of the "
                "same type. The SECOND is the part currently under inspection. Compare them "
                "and judge the second part only."
            )

        parts.append(f"Task: {USE_CASES.get(use_case, USE_CASES['uc1_triage'])}")

        if variant in LEAKY_VARIANTS:
            # Deliberate leakage, control arm only.
            parts.append(
                f"Visual inspection result: defect_present={episode.get('defect_present', False)}"
            )
            parts.append(f"Target action: {episode.get('ground_truth_action', 'ACCEPT')}")
        else:
            parts.append(
                "You have not been given any inspection verdict. Judge the part from the "
                "image alone."
            )

        if variant == "blind_cot":
            parts.append(
                'Before deciding, fill a "reasoning" string field with a brief '
                "step-by-step justification, then give the decision fields."
            )

        parts.append("Return only the JSON object.")
        return "\n".join(parts)

    def produce(
        self,
        episode: dict,
        *,
        variant: str = "blind",
        use_case: str = "uc1_triage",
        image_path: Path | None = None,
        reference_image_path: Path | None = None,
        model: str | None = None,
    ) -> PerceptionResult:
        """
        Run one live inference. ``capture_ts`` is taken before the call so that the
        caller can anchor validity to evidence acquisition rather than emission.

        ``reference_image_path`` supplies the known-good part for reference variants and is
        sent first, matching the ordering the prompt describes.
        """
        system = self.system_prompt(variant)
        user = self.build_user_prompt(episode, variant=variant, use_case=use_case)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        images: list[str] = []
        if variant in REFERENCE_VARIANTS and reference_image_path:
            ref = Path(reference_image_path)
            if ref.is_file():
                images.append(OllamaClient.encode_image(str(ref)))
        if image_path and Path(image_path).is_file():
            images.append(OllamaClient.encode_image(str(image_path)))
        images = images or None

        capture_dt = datetime.now(timezone.utc)
        base = PerceptionResult(
            frame_id=episode.get("frame_id", ""),
            use_case=use_case,
            prompt_variant=variant,
            model=model or self.client.model,
            capture_ts=_iso(capture_dt),
            emitted_ts="",
            latency_ms=0.0,
            action="",
            prompt_hash=_prompt_hash(system + user),
            evidence=dict(episode.get("evidence") or {}),
            # A reference variant whose reference silently failed to load would look like a
            # weak prompt rather than a broken run, so record what was actually sent.
            reference_used=bool(images and len(images) > 1),
        )

        try:
            resp: OllamaResponse = self.client.chat(
                messages, format_json=True, model=model, images=images
            )
        except RuntimeError as exc:
            base.emitted_ts = _now_iso()
            base.error = str(exc)[:300]
            return base

        base.emitted_ts = _now_iso()
        base.latency_ms = resp.latency_ms
        base.model = resp.model
        base.raw_content = resp.content[:4000]

        try:
            parsed = _extract_json(resp.content)
        except (ValueError, json.JSONDecodeError) as exc:
            base.error = f"invalid_json: {exc}"[:300]
            return base

        base.parse_ok = True
        payload = parsed.get("payload") or {}
        if not isinstance(payload, dict):
            payload = {}

        base.action = str(payload.get("action_type", "")).upper().strip()
        base.action_valid = base.action in ACTION_TYPES
        params = payload.get("parameters")
        base.parameters = params if isinstance(params, dict) else {}

        raw_preconds = parsed.get("preconditions")
        preconds: list[str] = []
        if isinstance(raw_preconds, list):
            for item in raw_preconds:
                if isinstance(item, dict) and item.get("expr"):
                    preconds.append(str(item["expr"]).strip())
                elif isinstance(item, str) and item.strip():
                    preconds.append(item.strip())
        # Keep what the model actually emitted, and record the repaired form alongside it,
        # so strict and repaired precondition quality can both be reported.
        base.preconditions = preconds
        base.preconditions_repaired = [repair_precondition(p) for p in preconds]
        base.precondition_syntax_ok = all(precondition_syntax_ok(p) for p in preconds)
        base.precondition_syntax_ok_repaired = all(
            precondition_syntax_ok(p) for p in base.preconditions_repaired
        )
        base.preconditions_repaired_count = sum(
            1 for raw, fixed in zip(preconds, base.preconditions_repaired) if raw != fixed
        )

        ext = parsed.get("extensions") or {}
        if isinstance(ext, dict):
            base.confidence = _as_float(ext.get("confidence"))
            base.observation = str(ext.get("observation", ""))[:500]
            judgement = ext.get("defect_judgement")
            if isinstance(judgement, bool):
                base.defect_judgement = judgement
            elif isinstance(judgement, str):
                base.defect_judgement = judgement.strip().lower() in ("true", "yes", "1")
            sev = ext.get("severity")
            base.severity = str(sev).strip().lower() if sev is not None else None
        reasoning = parsed.get("reasoning") or (ext.get("reasoning") if isinstance(ext, dict) else "")
        base.reasoning = str(reasoning or "")[:1000]

        # Schema validity means: parseable, a known action, and syntactically usable
        # preconditions. Absence of preconditions is allowed and measured separately.
        base.schema_valid = base.parse_ok and base.action_valid and base.precondition_syntax_ok
        base.schema_valid_repaired = (
            base.parse_ok and base.action_valid and base.precondition_syntax_ok_repaired
        )
        return base


def build_submission(
    result: PerceptionResult,
    *,
    anchor: str = "capture",
    freshness_ms: int = 500,
    deadline_ms: int | None = None,
    preconditions: list[str] | None = None,
    target_entity: str = "robot_3",
    intent_id: str | None = None,
) -> dict:
    """
    Assemble the AIS payload actually submitted to the gate.

    ``anchor`` selects what ``timestamp_decision`` means:
      ``capture``   the instant the evidence was acquired (inference latency counts
                    against the freshness window)
      ``emission``  the instant the model finished (inference latency is invisible to
                    the temporal validator)

    ``preconditions`` overrides the model's own list; pass ``None`` to submit exactly
    what the model emitted.
    """
    if anchor not in ("capture", "emission"):
        raise ValueError(f"unknown anchor: {anchor}")

    ts = result.capture_ts if anchor == "capture" else result.emitted_ts
    exprs = result.preconditions if preconditions is None else preconditions

    intent: dict = {
        "id": intent_id or str(uuid.uuid4()),
        "source": "ai",
        "timestamp_decision": ts,
        "freshness_window_ms": int(freshness_ms),
        "preconditions": [{"expr": e} for e in exprs],
        "payload": {
            "action_type": result.action or "HOLD_FOR_OPERATOR",
            "target_entity": target_entity,
            "parameters": dict(result.parameters),
        },
        "extensions": {
            "model_id": result.model,
            "inference_latency_ms": result.latency_ms,
            "prompt_hash": result.prompt_hash,
            "prompt_variant": result.prompt_variant,
            "validity_anchor": anchor,
            "confidence": result.confidence,
            "observation": result.observation,
            "grounded_preconditions": list(exprs),
            "precondition_source": "model" if preconditions is None else "harness",
        },
    }
    if deadline_ms is not None:
        intent["deadline_ms"] = int(deadline_ms)
    if result.evidence:
        intent["extensions"]["evidence"] = result.evidence
    return intent


# ---------------------------------------------------------------------------
# Legacy ablation
#
# Everything below reproduces the original A1 pipeline on purpose: the prompt is
# handed the verdict and the target action, missing preconditions are back-filled by
# the harness, and validity is anchored to emission so inference latency never counts
# against the freshness window. It is kept runnable because the paper reports it as a
# saturated ablation, but it must be called by its explicit name so that no headline
# result can reach it by default.
# ---------------------------------------------------------------------------

LEGACY_PRECONDITIONS = (
    "line.state == 'RUN'",
    "gripper.state == 'OPEN'",
    "defect.absent == true",
)


def _ensure_ais_envelope(data: dict, *, include_preconditions: bool = True) -> dict:
    out = dict(data)
    out.setdefault("id", str(uuid.uuid4()))
    out.setdefault("source", "ai")
    out.setdefault("timestamp_decision", _now_iso())
    out.setdefault("freshness_window_ms", 500)
    out.setdefault("deadline_ms", 800)
    payload = out.get("payload") or {}
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("action_type", "RESUME")
    payload.setdefault("target_entity", "robot_3")
    payload.setdefault("parameters", {})
    out["payload"] = payload
    if include_preconditions and "preconditions" not in out:
        out["preconditions"] = [{"expr": e} for e in LEGACY_PRECONDITIONS]
    elif not include_preconditions:
        out["preconditions"] = []
    return out


def _legacy_user_prompt(episode: dict, *, arm: str) -> str:
    parts = [
        f"Manufacturing cell scene (dataset {episode.get('source_dataset', 'unknown')}).",
        f"Category: {episode.get('category', 'unknown')}.",
        f"Visual inspection: defect_present={episode.get('defect_present', False)}.",
        f"Line at capture: {episode.get('context', {}).get('line', {}).get('state', 'RUN')}. "
        f"Suggested action: {episode.get('ground_truth_action', 'RESUME')}.",
        f"Frame id: {episode.get('frame_id', 'unknown')}.",
    ]
    if arm == "A1b":
        parts.append(
            'Return raw robot command JSON only: '
            '{"action":"RESUME|STOP|GRASP","x":0,"y":0,"confidence":0.9}'
        )
    elif arm == "A1d":
        parts.append("Return full AIS JSON WITHOUT preconditions array (empty preconditions).")
    else:
        parts.append("Return full AIS JSON WITH grounded preconditions for XAIR validation.")
    return "\n".join(parts)


def _legacy_produce_ais(
    self: StructuredIntentProducer,
    episode: dict,
    *,
    arm: str = "A1c",
    image_path: Path | None = None,
    model: str | None = None,
) -> tuple[dict, OllamaResponse, bool]:
    """Legacy leaky pipeline. Returns ``(intent_or_command, response, schema_valid)``."""
    user = _legacy_user_prompt(episode, arm=arm)
    messages = [
        {"role": "system", "content": self.leaky_system},
        {"role": "user", "content": user},
    ]
    images = None
    if image_path and Path(image_path).is_file():
        images = [OllamaClient.encode_image(str(image_path))]

    resp = self.client.chat(messages, format_json=True, model=model, images=images)
    try:
        parsed = _extract_json(resp.content)
    except (ValueError, json.JSONDecodeError):
        return {"error": "invalid_json", "raw": resp.content}, resp, False

    if arm == "A1b":
        return parsed, resp, "action" in parsed

    include_pre = arm not in ("A1b", "A1d")
    intent = _ensure_ais_envelope(parsed, include_preconditions=include_pre)
    intent["timestamp_decision"] = _now_iso()

    ext = intent.setdefault("extensions", {})
    ext.setdefault("model_id", resp.model)
    ext.setdefault("inference_latency_ms", resp.latency_ms)
    ext.setdefault("prompt_hash", _prompt_hash(self.leaky_system + user))
    ext["validity_anchor"] = "emission"
    ext["prompt_variant"] = "legacy_leaky"
    if episode.get("evidence"):
        ext["evidence"] = episode["evidence"]
    if include_pre and intent.get("preconditions"):
        ext["grounded_preconditions"] = [p["expr"] for p in intent["preconditions"]]

    return intent, resp, _validate_base_ais(intent)


def _raw_to_legacy_intent(raw: dict, episode: dict) -> dict:
    """Convert a legacy A1b raw command into a minimal AIS for the direct gate."""
    action = raw.get("action", "RESUME")
    return {
        "id": str(uuid.uuid4()),
        "source": "ai",
        "timestamp_decision": _now_iso(),
        "freshness_window_ms": 500,
        "preconditions": [],
        "payload": {
            "action_type": str(action).upper(),
            "target_entity": "robot_3",
            "parameters": {
                "x": raw.get("x", 0),
                "y": raw.get("y", 0),
                "confidence": raw.get("confidence", 0.5),
            },
        },
    }


def _validate_base_ais(intent: dict) -> bool:
    required = ("id", "source", "timestamp_decision", "freshness_window_ms", "payload")
    if not all(k in intent for k in required):
        return False
    payload = intent.get("payload", {})
    return bool(payload.get("action_type") and payload.get("target_entity"))


StructuredIntentProducer.produce_ais_legacy = _legacy_produce_ais
StructuredIntentProducer.raw_to_legacy_intent = staticmethod(_raw_to_legacy_intent)
StructuredIntentProducer._validate_base_ais = staticmethod(_validate_base_ais)
