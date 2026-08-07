"""Node 3 -- recommendation.

Fuses confidence, applies the escalation policy, and (for anything that
survives) asks the LLM for concrete, owner-assigned actions. Suppressed
candidates skip generation entirely -- there is no point spending tokens
writing advice nobody will read.
"""
from __future__ import annotations

from ..nim.client import NIMClient, NIMError, extract_json, get_client
from ..schemas import Disposition, Event, Recommendation, Severity
from .escalation import EscalationPolicy
from .state import GraphState

REC_PROMPT = """You are advising a warehouse floor supervisor.

Confirmed event: {kind} on camera {camera}, {t_start:.1f}s-{t_end:.1f}s, zone {zone}.
Severity: {severity}. Fused confidence: {confidence:.2f}.
What the vision analysis observed: {description}
Contributing factors: {factors}
Tracker context: {context}

Produce 1-3 recommendations. The first must be an immediate action that can be
taken in under 15 minutes by someone on the floor right now. Any others should
address the recurring cause. Be specific to this event -- no generic safety
platitudes.

Respond with ONLY this JSON:
{{"recommendations": [
  {{"action": "...", "rationale": "...", "owner_role": "floor_supervisor|safety_lead|operations|facilities|team_lead",
    "urgency": "info|low|medium|high|critical", "est_minutes": 5}}
]}}"""

SUMMARY_PROMPT = """Summarise this confirmed warehouse event for an operator dashboard.

Kind: {kind}. Camera: {camera}. Zone: {zone}. Window: {t_start:.1f}s-{t_end:.1f}s.
Vision analysis: {description}
Contributing factors: {factors}
Contradicting evidence considered and outweighed: {contradictions}

Respond with ONLY this JSON:
{{"summary": "one line, under 120 characters, no preamble",
  "narrative": "2-3 sentences an operator can read before opening the clip"}}"""


class RecommendationAgent:
    name = "recommendation"

    def __init__(self, client: NIMClient | None = None, policy: EscalationPolicy | None = None):
        self.client = client or get_client()
        self.policy = policy or EscalationPolicy()

    def __call__(self, state: GraphState) -> GraphState:
        c = state.candidate
        result = self.policy.decide(c, state.finding)
        state.fused_confidence = result.confidence

        f = state.finding
        description = f.description if f else "No vision evidence available."
        factors = ", ".join(f.contributing_factors) if f and f.contributing_factors else "none identified"
        contradictions = ", ".join(f.contradicting_evidence) if f and f.contradicting_evidence else "none"

        if result.disposition is Disposition.SUPPRESSED:
            state.summary = f"Suppressed {c.kind.value} candidate on {c.camera_id}"
            state.narrative = "; ".join(result.reasons)
            state.recommendations = []
        else:
            state.summary, state.narrative = self._summarise(c, description, factors, contradictions, state)
            state.recommendations = self._recommend(c, result, description, factors, state)

        state.event = Event(
            event_id=c.candidate_id,
            camera_id=c.camera_id,
            kind=c.kind,
            severity=result.severity,
            t_start=c.t_start,
            t_end=c.t_end,
            confidence=result.confidence,
            disposition=result.disposition,
            summary=state.summary,
            narrative=state.narrative,
            zone_id=c.zone_id,
            track_ids=c.track_ids,
            clip_uri=state.clip_uri,
            keyframe_uris=state.keyframe_uris,
            findings=state.finding,
            recommendations=state.recommendations,
            trace=state.trace,
        )
        state.log(
            self.name,
            disposition=result.disposition.value,
            severity=result.severity.value,
            confidence=result.confidence,
            reasons=result.reasons,
        )
        state.event.trace = list(state.trace)
        return state

    # ---------------------------------------------------------------- helpers
    def _summarise(self, c, description, factors, contradictions, state) -> tuple[str, str]:
        prompt = SUMMARY_PROMPT.format(
            kind=c.kind.value,
            camera=c.camera_id,
            zone=c.zone_id or "n/a",
            t_start=c.t_start,
            t_end=c.t_end,
            description=description,
            factors=factors,
            contradictions=contradictions,
        )
        try:
            data = extract_json(self.client.chat(prompt, temperature=0.2))
            return (
                str(data.get("summary", ""))[:160] or f"{c.kind.value} on {c.camera_id}",
                str(data.get("narrative", ""))[:800],
            )
        except (NIMError, ValueError, KeyError) as exc:
            state.errors.append(f"summary: {type(exc).__name__}: {exc}")
            return (f"{c.kind.value.replace('_', ' ').title()} on {c.camera_id}", description)

    def _recommend(self, c, result, description, factors, state) -> list[Recommendation]:
        prompt = REC_PROMPT.format(
            kind=c.kind.value,
            camera=c.camera_id,
            t_start=c.t_start,
            t_end=c.t_end,
            zone=c.zone_id or "n/a",
            severity=result.severity.value,
            confidence=result.confidence,
            description=description,
            factors=factors,
            context=state.track_context,
        )
        try:
            data = extract_json(self.client.chat(prompt, temperature=0.3))
            out = []
            for r in data.get("recommendations", [])[:3]:
                out.append(
                    Recommendation(
                        action=str(r.get("action", ""))[:280],
                        rationale=str(r.get("rationale", ""))[:400],
                        owner_role=str(r.get("owner_role", "floor_supervisor")),
                        urgency=_coerce_severity(r.get("urgency")),
                        est_minutes=int(r.get("est_minutes", 5)),
                    )
                )
            return [r for r in out if r.action]
        except (NIMError, ValueError, KeyError, TypeError) as exc:
            state.errors.append(f"recommendation: {type(exc).__name__}: {exc}")
            return []


def _coerce_severity(value) -> Severity:
    try:
        return Severity(str(value))
    except ValueError:
        return Severity.MEDIUM
