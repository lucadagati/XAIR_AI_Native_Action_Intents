# Design rationale — AI-native action intents

This note summarizes the design choices behind the B1–B5 suites in this
repository. It supersedes an earlier pre-redesign sketch (RQ-A1–A4) that no
longer matches the implemented study; see the paper for the full account.

## Problem

Vision-language models (VLMs) and tool-using agents can propose actions in
industrial cyber-physical systems, but a plausible model output is not the
same thing as an action admissible to publish *right now*. Between evidence
capture and middleware publication, plant state can drift and inference
itself takes time — often several seconds for a VLM.

## Approach

AI producers emit **Action Intents (AIS)**: structured envelopes with a typed
action, explicit preconditions, evidence provenance, and a validity anchor.
They never publish directly to ROS/OPC UA/PLC interlocks — every actuation
passes through **XAIR**, which authorizes or revokes publication against the
current plant context at the middleware boundary. XAIR does not improve
perceptual accuracy; it bounds *authorization to publish*.

Three design choices this repository operationalizes:

1. **Evidence anchoring** — freshness is measured from evidence *capture*
   time, not model *emission* time, so slow inference correctly consumes the
   freshness budget instead of being invisible to the gate
   (`xair.ai.structured_intent.build_submission(anchor=...)`).
2. **Stochastic drift** — plant volatility is a controlled experimental
   factor (probability and timing of context invalidation), not an
   unconditional or asserted event, so publication-time admissibility is
   actually measured rather than guaranteed by construction
   (`experiments/paper2_common.py`).
3. **Validity budgets** — a fixed, learned, or oracle policy selects the
   freshness window and precondition strictness to balance successful
   actuation against hazardous publication and wrongful revocation
   (`experiments/rl/`, suites B3–B5).

## What's implemented here

| Suite | Question |
|-------|----------|
| B1 | Blind grounding, label-leakage ablation, precondition quality across five VLMs |
| B2 | Validity frontier: freshness × drift probability × timing × anchor |
| B3 | Learned validity budget (LinUCB / Q-learning) vs. a train-selected fixed policy |
| B4 | Post-hoc selection among preserved model outputs under increasing volatility |
| B5 | Post-revocation policy simulation (abstain / retry / re-observe / escalate) |

Legacy suites A1–A4 (`run_a1_vlm_ais.py`, `run_paper2_campaign.sh`) predate
this redesign and are retained only as a saturated gate-semantics ablation
(unconditional drift forces every outcome to its floor or ceiling); they do
not support headline claims.

## Non-goals

- XAIR is a publication-boundary authorization layer, not a replacement for
  safety PLCs, robot interlocks, control barrier functions, or certified
  fallback controllers.
- Continuous-control assurance (shields, Simplex, CBFs) is complementary, not
  superseded — XAIR governs discrete, typed intents, not continuous
  trajectories.
