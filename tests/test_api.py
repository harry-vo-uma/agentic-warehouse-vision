import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_session=None):
    from awvi.schemas import Disposition, Event, EventKind, Severity

    events = [
        Event(
            event_id="evt-1", camera_id="cam-aisle-01", kind=EventKind.UNSAFE_INTERACTION,
            severity=Severity.HIGH, t_start=12.0, t_end=18.0, confidence=0.82,
            disposition=Disposition.ESCALATED,
            summary="pedestrian crosses a moving forklift path",
            narrative="Tracker and vision agree.",
        ),
        Event(
            event_id="evt-2", camera_id="cam-dock-02", kind=EventKind.BLOCKED_ZONE,
            severity=Severity.MEDIUM, t_start=40.0, t_end=70.0, confidence=0.71,
            disposition=Disposition.LOGGED,
            summary="pallet in the dock keep-clear area",
            narrative="Dwell exceeded threshold.",
        ),
    ]
    path = tmp_path_factory.mktemp("api") / "events.json"
    path.write_text(json.dumps([json.loads(e.model_dump_json()) for e in events]))

    import os

    os.environ["AWVI_EVENTS"] = str(path)
    from awvi.api import server

    server._index.__init__()
    with TestClient(server.app) as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["events_indexed"] == 2
    assert body["nim"]["mode"] == "mock"


def test_home_serves_ui(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Vision Intelligence" in r.text


def test_list_and_filter_events(client):
    assert client.get("/api/events").json()["count"] == 2
    assert client.get("/api/events?kind=blocked_zone").json()["count"] == 1
    assert client.get("/api/events?camera_id=cam-aisle-01").json()["count"] == 1
    assert client.get("/api/events?min_confidence=0.8").json()["count"] == 1


def test_get_single_event(client):
    assert client.get("/api/events/evt-1").json()["event_id"] == "evt-1"
    assert client.get("/api/events/nope").status_code == 404


def test_ask_returns_answer_and_hits(client):
    r = client.post("/api/ask", json={"question": "any near misses with a forklift?", "top_k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"]
    assert body["hits"][0]["event"]["kind"] == "unsafe_interaction"


def test_summary_aggregates(client):
    body = client.get("/api/summary").json()
    assert body["total"] == 2
    assert body["by_kind"]["blocked_zone"] == 1
    assert 0 < body["mean_confidence"] <= 1


def test_missing_clip_is_404(client):
    assert client.get("/media/clips/does-not-exist.mp4").status_code == 404
