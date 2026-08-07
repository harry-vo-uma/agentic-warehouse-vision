import pytest

from awvi.agents.escalation import EscalationPolicy, score_event
from awvi.config import EscalationSettings
from awvi.schemas import Disposition, EventCandidate, EventKind, VLMFinding


def candidate(strength=0.8, duration=8.0, tracks=2, kind=EventKind.UNSAFE_INTERACTION, t0=10.0):
    return EventCandidate(
        candidate_id=EventCandidate.make_id("cam-test", kind.value, t0),
        camera_id="cam-test",
        kind=kind,
        t_start=t0,
        t_end=t0 + duration,
        track_ids=list(range(1, tracks + 1)),
        signal_strength=strength,
    )


def finding(supports=True, conf=0.85, contradictions=0):
    return VLMFinding(
        supports_candidate=supports,
        observed_kind=EventKind.UNSAFE_INTERACTION if supports else EventKind.NOMINAL,
        confidence=conf,
        description="test",
        contradicting_evidence=["c"] * contradictions,
    )


def test_agreement_escalates():
    r = score_event(candidate(), finding())
    assert r.disposition in (Disposition.ESCALATED, Disposition.PAGED)
    assert r.confidence > 0.7


def test_vision_disagreement_suppresses():
    r = score_event(candidate(), finding(supports=False, conf=0.9, contradictions=2))
    assert r.disposition is Disposition.SUPPRESSED


def test_missing_vision_evidence_is_discounted_not_trusted():
    with_vlm = score_event(candidate(), finding())
    without = score_event(candidate(), None)
    assert without.confidence < with_vlm.confidence


def test_short_events_are_penalised():
    long_ = score_event(candidate(duration=8.0), finding())
    short = score_event(candidate(duration=1.0), finding())
    assert short.confidence < long_.confidence


def test_contradictions_lower_confidence_monotonically():
    scores = [score_event(candidate(), finding(contradictions=n)).confidence for n in (0, 1, 2)]
    assert scores[0] > scores[1] > scores[2]


def test_veto_can_be_disabled():
    cfg = EscalationSettings()
    cfg.require_vlm_agreement = False
    vetoed = score_event(candidate(), finding(supports=False, conf=0.4))
    unvetoed = score_event(candidate(), finding(supports=False, conf=0.4), cfg)
    assert unvetoed.confidence > vetoed.confidence


def test_suppressed_events_are_never_severe():
    r = score_event(candidate(strength=0.05, duration=0.5), finding(supports=False, conf=0.95, contradictions=2))
    assert r.disposition is Disposition.SUPPRESSED
    assert r.severity.value == "info"


def test_policy_deduplicates_repeat_alerts():
    policy = EscalationPolicy()
    first = policy.decide(candidate(t0=10.0), finding(conf=0.95))
    second = policy.decide(candidate(t0=25.0), finding(conf=0.95))
    assert first.disposition in (Disposition.ESCALATED, Disposition.PAGED)
    assert second.disposition is Disposition.LOGGED
    assert any("de-duplicated" in r for r in second.reasons)


def test_dedup_window_expires():
    policy = EscalationPolicy(dedup_window_s=5.0)
    policy.decide(candidate(t0=10.0), finding(conf=0.95))
    later = policy.decide(candidate(t0=100.0), finding(conf=0.95))
    assert later.disposition in (Disposition.ESCALATED, Disposition.PAGED)


@pytest.mark.parametrize("conf", [0.0, 0.5, 1.0])
def test_confidence_always_in_range(conf):
    r = score_event(candidate(strength=conf), finding(conf=conf))
    assert 0.0 <= r.confidence <= 1.0
