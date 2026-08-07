from awvi.index.store import EventIndex, parse_filters
from awvi.schemas import Disposition, Event, EventKind, Severity


def event(eid, kind, camera="cam-aisle-01", t=10.0, conf=0.8, summary="", severity=Severity.MEDIUM):
    return Event(
        event_id=eid, camera_id=camera, kind=kind, severity=severity,
        t_start=t, t_end=t + 5, confidence=conf,
        disposition=Disposition.ESCALATED,
        summary=summary or f"{kind.value} on {camera}",
        narrative="detail",
    )


CORPUS = [
    event("a", EventKind.UNSAFE_INTERACTION, summary="pedestrian passes close to a moving forklift"),
    event("b", EventKind.BLOCKED_ZONE, camera="cam-dock-02", summary="pallet obstructing the fire egress lane"),
    event("c", EventKind.CONGESTION, summary="aisle queue with stalled traffic", severity=Severity.HIGH),
    event("d", EventKind.WORKFLOW_ANOMALY, camera="cam-pack-03", summary="tote idle at the pack station"),
]


def _index():
    idx = EventIndex()
    idx.add_many(CORPUS)
    return idx


def test_index_size_and_dedup():
    idx = _index()
    assert len(idx) == 4
    idx.add_many(CORPUS)
    assert len(idx) == 4


def test_kind_filter_parsing():
    f = parse_filters("any near misses with forklifts?", {"cam-aisle-01"})
    assert EventKind.UNSAFE_INTERACTION in f.kinds


def test_camera_filter_parsing():
    f = parse_filters("what happened on cam-dock-02", {"cam-dock-02", "cam-aisle-01"})
    assert f.cameras == {"cam-dock-02"}


def test_time_range_parsing():
    f = parse_filters("show events between 10 and 40s", set())
    assert f.t_start == 10.0 and f.t_end == 40.0


def test_semantic_query_finds_the_right_event():
    idx = _index()
    assert idx.search("who nearly got hit by a forklift?", top_k=1)[0].event.event_id == "a"
    assert idx.search("what is blocking the fire exit?", top_k=1)[0].event.event_id == "b"
    assert idx.search("where is traffic backed up?", top_k=1)[0].event.event_id == "c"


def test_answer_returns_prose_and_hits():
    resp = _index().answer("any near misses?", top_k=2)
    assert resp.answer
    assert resp.hits
    assert resp.latency_ms >= 0


def test_suppressed_events_are_hidden_by_default():
    idx = EventIndex()
    e = event("s", EventKind.CONGESTION)
    e.disposition = Disposition.SUPPRESSED
    idx.add_many([e, *CORPUS])
    assert all(h.event.event_id != "s" for h in idx.search("congestion", top_k=5))
    assert any(h.event.event_id == "s" for h in idx.search("congestion", top_k=5, include_suppressed=True))


def test_empty_index_returns_no_hits():
    assert EventIndex().answer("anything?").hits == []
