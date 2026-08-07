from conftest import hold, line, make_scenario

from awvi.agents.graph import IncidentGraph, should_extract_clip
from awvi.agents.state import GraphState
from awvi.nim.client import NIMClient
from awvi.nim.mock import MockNIMBackend
from awvi.perception.deepstream import SyntheticSource
from awvi.perception.events import CandidateProposer
from awvi.perception.tracks import TrackStore
from awvi.schemas import Disposition


def build(actors, zones, jitter=0.002):
    store = TrackStore("cam-test")
    store.update(list(SyntheticSource(make_scenario(actors), jitter=jitter).stream()))
    return store, CandidateProposer(zones).propose(store)


UNSAFE = [
    {"track_id": 1, "label": "forklift", "waypoints": line(0, 10, (0.05, 0.6), (0.95, 0.6))},
    {"track_id": 2, "label": "person", "waypoints": line(2, 8, (0.5, 0.9), (0.5, 0.3))},
]


def test_graph_produces_an_event_per_candidate(zones):
    store, cands = build(UNSAFE, zones)
    events = IncidentGraph(store).run(cands)
    assert len(events) == len(cands)
    assert all(e.event_id == c.candidate_id for e, c in zip(events, cands, strict=True))


def test_confirmed_event_has_narrative_and_actions(zones):
    store, cands = build(UNSAFE, zones)
    events = [e for e in IncidentGraph(store).run(cands) if e.disposition is not Disposition.SUPPRESSED]
    assert events
    e = events[0]
    assert e.summary and e.narrative
    assert e.recommendations
    assert e.recommendations[0].est_minutes <= 15
    assert e.findings is not None


def test_trace_records_every_node(zones):
    store, cands = build(UNSAFE, zones)
    st = IncidentGraph(store).run_one(cands[0])
    nodes = [t["node"] for t in st.trace]
    assert nodes[0] == "perception"
    assert "investigation" in nodes
    assert nodes[-1] == "recommendation"


def test_clip_gate_skips_extraction_for_weak_candidates(zones):
    store, cands = build(UNSAFE, zones)
    graph = IncidentGraph(store)
    st = graph.investigation(graph.perception(GraphState(candidate=cands[0])))
    assert should_extract_clip(st) in {"clip", "recommendation"}

    st.finding.supports_candidate = False
    st.finding.confidence = 0.95
    st.finding.contradicting_evidence = ["clearly nothing happening", "forks grounded"]
    assert should_extract_clip(st) == "recommendation"


def test_pipeline_is_deterministic_in_mock_mode(zones):
    store, cands = build(UNSAFE, zones)
    a = IncidentGraph(store).run(cands)
    b = IncidentGraph(store).run(cands)
    assert [e.confidence for e in a] == [e.confidence for e in b]
    assert [e.disposition for e in a] == [e.disposition for e in b]


def test_vision_failure_does_not_confirm_an_event(zones):
    """If the VLM call blows up, the candidate must not sail through on
    geometry alone."""

    class Broken(MockNIMBackend):
        def vlm(self, *a, **kw):
            return "the model returned prose and no json at all"

    store, cands = build(UNSAFE, zones)
    graph = IncidentGraph(store, client=NIMClient(mock=Broken()))
    st = graph.run_one(cands[0])
    assert st.finding is None
    assert st.errors
    assert st.event.confidence < 0.7


def test_keyframe_budget_is_respected(zones):
    store, cands = build(UNSAFE, zones)
    st = IncidentGraph(store).run_one(cands[0])
    assert 1 <= len(st.keyframe_uris) <= 4


def test_stationary_scene_yields_no_unsafe_events(zones):
    actors = [
        {"track_id": 1, "label": "forklift", "waypoints": hold(0, 10, (0.5, 0.6))},
        {"track_id": 2, "label": "person", "waypoints": line(2, 8, (0.52, 0.9), (0.52, 0.3))},
    ]
    store, cands = build(actors, zones)
    events = [e for e in IncidentGraph(store).run(cands) if e.disposition is not Disposition.SUPPRESSED]
    assert not [e for e in events if e.kind.value == "unsafe_interaction"]
