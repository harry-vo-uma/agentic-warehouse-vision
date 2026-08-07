from conftest import hold, line, make_scenario

from awvi.perception.deepstream import SyntheticSource
from awvi.perception.events import CandidateProposer
from awvi.perception.tracks import TrackStore
from awvi.schemas import EventKind


def propose(actors, zones, jitter=0.002):
    store = TrackStore("cam-test")
    store.update(list(SyntheticSource(make_scenario(actors), jitter=jitter).stream()))
    return CandidateProposer(zones).propose(store), store


def kinds(cands):
    return {c.kind for c in cands}


def test_moving_forklift_near_pedestrian_is_proposed(zones):
    actors = [
        {"track_id": 1, "label": "forklift", "waypoints": line(0, 10, (0.05, 0.6), (0.95, 0.6))},
        {"track_id": 2, "label": "person", "waypoints": line(2, 8, (0.5, 0.9), (0.5, 0.3))},
    ]
    cands, _ = propose(actors, zones)
    assert EventKind.UNSAFE_INTERACTION in kinds(cands)


def test_parked_forklift_near_pedestrian_is_not_proposed(zones):
    """Proximity alone is not an incident. This was the dominant false positive."""
    actors = [
        {"track_id": 1, "label": "forklift", "waypoints": hold(0, 10, (0.5, 0.6))},
        {"track_id": 2, "label": "person", "waypoints": line(2, 8, (0.52, 0.9), (0.52, 0.3))},
    ]
    cands, _ = propose(actors, zones)
    assert EventKind.UNSAFE_INTERACTION not in kinds(cands)


def test_long_dwell_in_keep_clear_is_proposed(zones):
    actors = [{"track_id": 1, "label": "pallet", "waypoints": hold(0, 30, (0.35, 0.35))}]
    cands, _ = propose(actors, zones)
    blocked = [c for c in cands if c.kind is EventKind.BLOCKED_ZONE]
    assert blocked and blocked[0].zone_id == "z-keepclear"
    assert blocked[0].signal_strength > 0.6


def test_brief_pass_through_keep_clear_is_not_proposed(zones):
    actors = [{"track_id": 1, "label": "forklift", "waypoints": line(0, 3, (0.1, 0.35), (0.9, 0.35))}]
    cands, _ = propose(actors, zones)
    assert EventKind.BLOCKED_ZONE not in kinds(cands)


def test_stalled_cluster_is_congestion(zones):
    actors = [
        {"track_id": i, "label": "person" if i % 2 else "forklift",
         "waypoints": hold(0, 12, (0.55 + 0.02 * i, 0.55 + 0.01 * i))}
        for i in range(1, 7)
    ]
    cands, _ = propose(actors, zones)
    cong = [c for c in cands if c.kind is EventKind.CONGESTION]
    assert cong
    assert cong[0].features["stall"] > 0.8


def test_flowing_group_is_not_congestion(zones):
    actors = [
        {"track_id": i, "label": "person",
         "waypoints": line(0, 8, (0.05, 0.4 + 0.02 * i), (0.95, 0.45 + 0.02 * i))}
        for i in range(1, 7)
    ]
    cands, _ = propose(actors, zones)
    assert EventKind.CONGESTION not in kinds(cands)


def test_blocked_zone_wins_arbitration_over_workflow_anomaly(zones):
    """One pallet in a keep-clear area should page an operator once, not twice."""
    actors = [{"track_id": 1, "label": "pallet", "waypoints": hold(0, 40, (0.35, 0.35))}]
    actors += [
        {"track_id": 10 + i, "label": "box", "waypoints": hold(i, i + 4, (0.8, 0.8))}
        for i in range(4)
    ]
    cands, _ = propose(actors, zones)
    anomaly_tracks = {t for c in cands if c.kind is EventKind.WORKFLOW_ANOMALY for t in c.track_ids}
    assert 1 not in anomaly_tracks


def test_candidate_ids_are_stable(zones):
    actors = [{"track_id": 1, "label": "pallet", "waypoints": hold(0, 30, (0.35, 0.35))}]
    a, _ = propose(actors, zones)
    b, _ = propose(actors, zones)
    assert [c.candidate_id for c in a] == [c.candidate_id for c in b]
