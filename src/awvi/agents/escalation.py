"""Confidence fusion and the escalation policy.

This module is where the false-positive reduction actually happens. The
proposer's geometric signal and the VLM's reading are fused, then adjusted by
three temporal/structural terms:

  * persistence  -- conditions that hold for several seconds are real; two-frame
                    coincidences are not
  * corroboration -- multiple independent tracks implicated raises confidence
  * contradiction -- the VLM explicitly naming disconfirming evidence is a much
                     stronger negative signal than mere low confidence

Everything is a pure function of the state so it can be swept in
`eval/ablations.py` without touching the graph.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import EscalationSettings, get_settings
from ..schemas import Disposition, EventCandidate, EventKind, Severity, VLMFinding

#: Baseline severity per event class, before confidence weighting.
_BASE_SEVERITY: dict[EventKind, int] = {
    EventKind.UNSAFE_INTERACTION: 4,
    EventKind.BLOCKED_ZONE: 3,
    EventKind.CONGESTION: 2,
    EventKind.WORKFLOW_ANOMALY: 2,
    EventKind.NOMINAL: 0,
}

_SEVERITY_ORDER = [Severity.INFO, Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]


@dataclass
class FusionResult:
    confidence: float
    disposition: Disposition
    severity: Severity
    reasons: list[str]


def score_event(
    candidate: EventCandidate,
    finding: VLMFinding | None,
    settings: EscalationSettings | None = None,
) -> FusionResult:
    cfg = settings or get_settings().escalation
    reasons: list[str] = []

    geo = candidate.signal_strength
    if finding is None:
        # No vision evidence: fall back to geometry, heavily discounted. We do
        # not trust the rule layer on its own -- that is the regime that
        # produced the original false-positive rate.
        conf = geo * 0.55
        reasons.append("no vision evidence; geometry-only score discounted")
    elif not finding.supports_candidate:
        # Symmetric with the agreement path: a refutation held with confidence c
        # is evidence for the event at strength (1 - c). Taking min() here
        # instead collapses every refusal to ~0 and makes the whole policy
        # threshold-invariant, which hides how much the veto actually costs.
        conf = 0.38 * geo + 0.62 * (1.0 - finding.confidence)
        reasons.append(f"vision disagrees with the {candidate.kind.value} hypothesis")
    else:
        # Weighted fusion. Vision carries more weight than geometry because the
        # geometry is what generated the hypothesis in the first place -- using
        # it twice at full weight double-counts the same evidence.
        conf = 0.38 * geo + 0.62 * finding.confidence
        reasons.append("geometry and vision agree")

    duration = max(0.0, candidate.t_end - candidate.t_start)
    if duration >= cfg.min_temporal_persistence_s:
        conf += cfg.persistence_bonus
        reasons.append(f"condition persisted {duration:.1f}s")
    else:
        conf -= cfg.persistence_bonus
        reasons.append(f"condition lasted only {duration:.1f}s")

    if len(candidate.track_ids) >= 3:
        conf += cfg.multi_track_bonus
        reasons.append(f"{len(candidate.track_ids)} tracks implicated")

    if finding and finding.contradicting_evidence:
        conf -= cfg.contradiction_penalty * min(2, len(finding.contradicting_evidence))
        reasons.append(f"{len(finding.contradicting_evidence)} contradicting observations")

    if cfg.require_vlm_agreement and finding is not None and not finding.supports_candidate:
        # A fixed multiplicative penalty, deliberately not a clamp to the
        # suppression threshold: clamping makes the veto threshold-invariant,
        # which hides its cost and makes the threshold sweep meaningless.
        conf *= cfg.veto_penalty
        reasons.append("vision veto applied")

    conf = round(min(0.99, max(0.0, conf)), 4)

    if conf < cfg.suppress_below:
        disposition = Disposition.SUPPRESSED
    elif conf >= cfg.page_above:
        disposition = Disposition.PAGED
    elif conf >= cfg.escalate_above:
        disposition = Disposition.ESCALATED
    else:
        disposition = Disposition.LOGGED

    base = _BASE_SEVERITY.get(candidate.kind, 1)
    idx = max(0, min(len(_SEVERITY_ORDER) - 1, round(base * (0.5 + conf))))
    if disposition is Disposition.SUPPRESSED:
        idx = 0
    return FusionResult(confidence=conf, disposition=disposition, severity=_SEVERITY_ORDER[idx], reasons=reasons)


class EscalationPolicy:
    """Stateful wrapper adding cross-event de-duplication.

    Repeated alerts for the same condition are the second-largest source of
    operator alert fatigue after outright false positives, so a candidate that
    overlaps an already-escalated event of the same kind on the same camera is
    demoted to LOGGED rather than paged again.
    """

    def __init__(self, settings: EscalationSettings | None = None, dedup_window_s: float = 45.0):
        self.settings = settings or get_settings().escalation
        self.dedup_window_s = dedup_window_s
        self._recent: list[tuple[str, EventKind, float]] = []

    def decide(self, candidate: EventCandidate, finding: VLMFinding | None) -> FusionResult:
        result = score_event(candidate, finding, self.settings)
        if result.disposition in (Disposition.ESCALATED, Disposition.PAGED):
            for cam, kind, t_end in self._recent:
                if cam == candidate.camera_id and kind == candidate.kind and candidate.t_start - t_end < self.dedup_window_s:
                    result.disposition = Disposition.LOGGED
                    result.reasons.append("de-duplicated against a recent alert of the same kind")
                    break
            else:
                self._recent.append((candidate.camera_id, candidate.kind, candidate.t_end))
                self._recent = self._recent[-64:]
        return result
