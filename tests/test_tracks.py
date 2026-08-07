from conftest import hold, line, make_scenario

from awvi.perception.deepstream import SyntheticSource
from awvi.perception.tracks import TrackStore


def _store(actors, jitter=0.0):
    scenario = make_scenario(actors)
    store = TrackStore("cam-test")
    store.update(list(SyntheticSource(scenario, jitter=jitter).stream()))
    return store


def test_net_speed_ignores_jitter_but_mean_speed_does_not():
    """The whole reason net_speed exists: a stationary object with detector
    jitter reads as moving under path-length speed."""
    actors = [{"track_id": 1, "label": "pallet", "waypoints": hold(0, 10, (0.5, 0.5))}]
    noisy = _store(actors, jitter=0.01)
    tr = noisy.get(1)
    assert tr.net_speed() < 0.01
    assert tr.mean_speed() > tr.net_speed()


def test_moving_track_has_real_net_speed():
    actors = [{"track_id": 1, "label": "forklift", "waypoints": line(0, 10, (0.1, 0.5), (0.9, 0.5))}]
    tr = _store(actors).get(1)
    assert 0.07 < tr.net_speed() < 0.09
    assert tr.dwell_ratio() < 0.05


def test_stationary_track_has_high_dwell_ratio():
    actors = [{"track_id": 1, "label": "box", "waypoints": hold(0, 12, (0.3, 0.3))}]
    assert _store(actors).get(1).dwell_ratio() > 0.9


def test_min_separation_finds_closest_approach():
    actors = [
        {"track_id": 1, "label": "person", "waypoints": line(0, 10, (0.5, 0.0), (0.5, 1.0))},
        {"track_id": 2, "label": "forklift", "waypoints": line(0, 10, (0.0, 0.5), (1.0, 0.5))},
    ]
    store = _store(actors)
    dist, t = store.min_separation(1, 2)
    assert dist < 0.05
    assert 4.0 < t < 6.0


def test_min_separation_for_disjoint_tracks_is_large():
    actors = [
        {"track_id": 1, "label": "person", "waypoints": hold(0, 5, (0.1, 0.1))},
        {"track_id": 2, "label": "forklift", "waypoints": hold(0, 5, (0.9, 0.9))},
    ]
    dist, _ = _store(actors).min_separation(1, 2)
    assert dist > 0.5


def test_positions_at_interpolates():
    actors = [{"track_id": 1, "label": "person", "waypoints": line(0, 10, (0.0, 0.0), (1.0, 0.0))}]
    pos = _store(actors).positions_at(5.0)
    assert 1 in pos
    assert abs(pos[1][0] - 0.5) < 0.05
