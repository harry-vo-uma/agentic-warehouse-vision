"""Node 2 -- investigation.

Sends the keyframes plus the numeric track context to the VLM and parses a
structured finding. The prompt asks the model to argue *against* the
hypothesis as well as for it; without that instruction the model agrees with
whatever the rule layer proposed, which defeats the purpose of having it.
"""
from __future__ import annotations

from ..nim.client import NIMClient, NIMError, extract_json, get_client
from ..schemas import EventKind, VLMFinding
from .state import GraphState

SYSTEM = (
    "You are a warehouse safety and operations analyst reviewing CCTV keyframes. "
    "You are rigorous and sceptical: your job is to confirm or refute a specific "
    "hypothesis, not to describe the scene in general. You always answer with JSON."
)

PROMPT = """A rule-based detector flagged a possible **{kind}** event.

Camera: {camera}
Window: {t_start:.1f}s - {t_end:.1f}s ({duration:.1f}s)
Zone: {zone}
Tracker context (computed from object trajectories, not from the images):
{context}
Detector signal strength: {strength:.2f}

You are shown {n} keyframes sampled across this window, in time order.

Decide whether the images actually support the "{kind}" hypothesis.
Argue both sides: list concrete contributing factors if it holds, and list any
concrete contradicting evidence you can see if it does not. Do not invent
detail you cannot see. If the frames are ambiguous, say so via a low confidence
rather than by guessing.

Respond with ONLY this JSON object:
{{
  "supports_candidate": true|false,
  "observed_kind": "congestion|blocked_zone|unsafe_interaction|workflow_anomaly|nominal",
  "confidence": 0.0-1.0,
  "description": "one or two sentences on what is actually happening",
  "entities": ["person", "forklift", ...],
  "contributing_factors": ["..."],
  "contradicting_evidence": ["..."]
}}"""


class InvestigationAgent:
    name = "investigation"

    def __init__(self, client: NIMClient | None = None):
        self.client = client or get_client()

    def __call__(self, state: GraphState) -> GraphState:
        c = state.candidate
        context_lines = "\n".join(f"  - {k}: {v}" for k, v in sorted(state.track_context.items()))
        prompt = PROMPT.format(
            kind=c.kind.value,
            camera=c.camera_id,
            t_start=c.t_start,
            t_end=c.t_end,
            duration=c.t_end - c.t_start,
            zone=c.zone_id or "n/a",
            context=context_lines or "  - (none)",
            strength=c.signal_strength,
            n=len(state.keyframes_b64),
        )

        hint = {
            "candidate_id": c.candidate_id,
            "kind": c.kind.value,
            "signal_strength": c.signal_strength,
            "features": c.features,
            "entities": state.track_context.get("labels", []),
        }

        try:
            raw = self.client.vlm(prompt, state.keyframes_b64, system=SYSTEM, mock_hint=hint)
            data = extract_json(raw)
            state.finding = VLMFinding(
                supports_candidate=bool(data.get("supports_candidate", False)),
                observed_kind=_coerce_kind(data.get("observed_kind")),
                confidence=float(data.get("confidence", 0.0)),
                description=str(data.get("description", ""))[:600],
                entities=[str(e) for e in data.get("entities", [])][:8],
                contributing_factors=[str(e) for e in data.get("contributing_factors", [])][:6],
                contradicting_evidence=[str(e) for e in data.get("contradicting_evidence", [])][:6],
            )
        except (NIMError, ValueError, KeyError, TypeError) as exc:
            # A failed vision call must not silently become a confirmed event.
            # Downstream treats finding=None as "geometry only, discounted".
            state.errors.append(f"investigation: {type(exc).__name__}: {exc}")
            state.finding = None

        state.log(
            self.name,
            supports=None if state.finding is None else state.finding.supports_candidate,
            vlm_confidence=None if state.finding is None else state.finding.confidence,
            n_contradictions=0 if state.finding is None else len(state.finding.contradicting_evidence),
        )
        return state


def _coerce_kind(value) -> EventKind:
    try:
        return EventKind(str(value))
    except ValueError:
        return EventKind.NOMINAL
