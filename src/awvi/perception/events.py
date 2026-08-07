"""Rule-based candidate proposal.

This layer is intentionally tuned for *recall*, not precision. It is cheap
enough to run on every frame of every camera, and it exists to decide where
the expensive VLM should look. Precision is the agent graph's job.
"""
from __future__ import annotations

from ..schemas import EventCandidate, EventKind
from .tracks import PEDESTRIANS, STATIC, VEHICLES, TrackStore
from .zones import ZoneIndex


class CandidateProposer:
    """Turns a TrackStore into event hypotheses.

    Thresholds here are deliberately loose. Raising them improves precision at
    the proposer, but the ablation in `eval/ablations.py` shows that costs more
    recall than the reasoning layer costs precision -- so we keep them loose
    and let the escalation policy do the filtering.
    """

    def __init__(
        self,
        zones: ZoneIndex,
        *,
        proximity_threshold: float = 0.09,
        min_vehicle_speed: float = 0.015,
        congestion_min_actors: int = 4,
        congestion_min_duration: float = 3.0,
        congestion_radius: float = 0.14,
        blocked_min_dwell: float = 6.0,
        anomaly_dwell_factor: float = 2.5,
    ):
        self.zones = zones
        self.proximity_threshold = proximity_threshold
        self.min_vehicle_speed = min_vehicle_speed
        self.congestion_min_actors = congestion_min_actors
        self.congestion_min_duration = congestion_min_duration
        self.congestion_radius = congestion_radius
        self.blocked_min_dwell = blocked_min_dwell
        self.anomaly_dwell_factor = anomaly_dwell_factor

    def propose(self, store: TrackStore) -> list[EventCandidate]:
        """Propose candidates, then arbitrate between rules.

        Arbitration matters: a pallet parked in a keep-clear zone satisfies both
        the blocked-zone rule and the long-dwell rule, and shipping both means
        the operator gets paged twice for one pallet. The more specific
        explanation wins.
        """
        blocked = self._blocked_zones(store)
        explained = {tid for c in blocked for tid in c.track_ids}
        out: list[EventCandidate] = []
        out += self._unsafe_interactions(store)
        out += blocked
        out += self._congestion(store)
        out += [c for c in self._workflow_anomalies(store) if not set(c.track_ids) & explained]
        return sorted(out, key=lambda c: c.t_start)

    # ------------------------------------------------------------------ rules
    def _unsafe_interactions(self, store: TrackStore) -> list[EventCandidate]:
        peds = [t for t in store.all() if t.label in PEDESTRIANS]
        vehs = [t for t in store.all() if t.label in VEHICLES]
        out = []
        for p in peds:
            for v in vehs:
                if p.last_seen < v.first_seen or v.last_seen < p.first_seen:
                    continue
                dist, t_close = store.min_separation(p.track_id, v.track_id)
                if dist > self.proximity_threshold:
                    continue
                v_speed = v.net_speed()
                # A parked forklift with a person walking past it is not an
                # unsafe interaction, and treating it as one was the single
                # largest source of false alerts in the first version of this
                # rule. Powered motion is required, not merely proximity.
                if v_speed < self.min_vehicle_speed:
                    continue
                # Closeness dominates; vehicle motion is the aggravating factor.
                closeness = 1.0 - min(1.0, dist / self.proximity_threshold)
                motion = min(1.0, v_speed / 0.12)
                strength = min(1.0, 0.62 * closeness + 0.38 * motion)
                out.append(
                    EventCandidate(
                        candidate_id=EventCandidate.make_id(store.camera_id, "unsafe_interaction", t_close),
                        camera_id=store.camera_id,
                        kind=EventKind.UNSAFE_INTERACTION,
                        t_start=max(0.0, t_close - 3.0),
                        t_end=t_close + 3.0,
                        track_ids=[p.track_id, v.track_id],
                        signal_strength=round(strength, 4),
                        features={
                            "min_distance": round(dist, 4),
                            "closeness": round(closeness, 4),
                            "vehicle_speed": round(v_speed, 4),
                            "overlap_duration": round(
                                max(0.0, min(p.last_seen, v.last_seen) - max(p.first_seen, v.first_seen)), 3
                            ),
                        },
                    )
                )
        return out

    def _blocked_zones(self, store: TrackStore) -> list[EventCandidate]:
        out = []
        for zone in self.zones.keep_clear(store.camera_id):
            for tr in store.all():
                if tr.label not in STATIC and tr.label not in VEHICLES:
                    continue
                inside = [
                    p for p in tr.points
                    if _inside(p.x, p.y, zone.polygon)
                ]
                if len(inside) < 2:
                    continue
                dwell = inside[-1].t - inside[0].t
                if dwell < self.blocked_min_dwell:
                    continue
                coverage = len(inside) / max(1, len(tr.points))
                # Dwell carries most of the weight. Coverage is ~1.0 for any
                # static object that is in the zone at all, so weighting it
                # heavily just means "an object is present", which cannot
                # separate an obstruction from an in-progress put-away.
                strength = min(1.0, 0.75 * min(1.0, dwell / 30.0) + 0.25 * coverage)
                out.append(
                    EventCandidate(
                        candidate_id=EventCandidate.make_id(store.camera_id, "blocked_zone", inside[0].t),
                        camera_id=store.camera_id,
                        kind=EventKind.BLOCKED_ZONE,
                        t_start=inside[0].t,
                        t_end=inside[-1].t,
                        track_ids=[tr.track_id],
                        zone_id=zone.zone_id,
                        signal_strength=round(strength, 4),
                        features={
                            "dwell_s": round(dwell, 3),
                            "zone_coverage": round(coverage, 4),
                            "is_static_object": 1.0 if tr.label in STATIC else 0.0,
                        },
                    )
                )
        return out

    def _congestion(self, store: TrackStore) -> list[EventCandidate]:
        """Spatio-temporal clustering.

        Counting concurrent tracks per camera is not enough -- four people at
        four different ends of a wide-angle view are not a queue. We look for a
        spatial cluster that is both dense and slow-moving, then merge
        consecutive seconds where the cluster stays put.
        """
        tracks = store.all()
        if len(tracks) < self.congestion_min_actors:
            return []
        t_lo = min(t.first_seen for t in tracks)
        t_hi = max(t.last_seen for t in tracks)

        hot: list[tuple[float, tuple[float, float], int, float]] = []
        t = t_lo
        while t <= t_hi:
            positions = store.positions_at(t)
            if len(positions) >= self.congestion_min_actors:
                cluster = _densest_cluster(positions, self.congestion_radius)
                if len(cluster) >= self.congestion_min_actors:
                    ids = set(cluster)
                    speeds = [tr.net_speed() for tr in tracks if tr.track_id in ids]
                    flow = sum(speeds) / len(speeds) if speeds else 0.0
                    if flow < 0.06:
                        cx = sum(positions[i][0] for i in cluster) / len(cluster)
                        cy = sum(positions[i][1] for i in cluster) / len(cluster)
                        hot.append((round(t, 3), (cx, cy), len(cluster), flow))
            t = round(t + 1.0, 3)

        out = []
        for run in _cluster_runs(hot, gap=1.5, max_drift=0.18):
            t0, t1 = run[0][0], run[-1][0]
            if t1 - t0 < self.congestion_min_duration:
                continue
            peak = max(r[2] for r in run)
            flow = sum(r[3] for r in run) / len(run)
            stall = 1.0 - min(1.0, flow / 0.06)
            strength = min(1.0, 0.5 * min(1.0, peak / 8.0) + 0.5 * stall)
            members = [tr.track_id for tr in store.in_window(t0, t1)][:12]
            out.append(
                EventCandidate(
                    candidate_id=EventCandidate.make_id(store.camera_id, "congestion", t0),
                    camera_id=store.camera_id,
                    kind=EventKind.CONGESTION,
                    t_start=t0,
                    t_end=t1,
                    track_ids=members,
                    signal_strength=round(strength, 4),
                    features={
                        "peak_actors": float(peak),
                        "duration_s": round(t1 - t0, 3),
                        "mean_flow": round(flow, 4),
                        "stall": round(stall, 4),
                        "cluster_radius": self.congestion_radius,
                    },
                )
            )
        return out

    def _workflow_anomalies(self, store: TrackStore) -> list[EventCandidate]:
        tracks = [t for t in store.all() if t.label in STATIC]
        if len(tracks) < 2:
            return []
        durations = sorted(t.duration for t in tracks)
        median = durations[len(durations) // 2] or 1.0
        out = []
        for tr in tracks:
            if tr.duration < self.anomaly_dwell_factor * median:
                continue
            ratio = tr.duration / median
            dwell = tr.dwell_ratio()
            # Stationarity is the discriminator, not duration. A cart that is
            # in frame for a long time while travelling is doing its job; a
            # unit that has not moved is the anomaly.
            strength = min(1.0, 0.45 * min(1.0, (ratio - 1) / 4.0) + 0.55 * dwell)
            out.append(
                EventCandidate(
                    candidate_id=EventCandidate.make_id(store.camera_id, "workflow_anomaly", tr.first_seen),
                    camera_id=store.camera_id,
                    kind=EventKind.WORKFLOW_ANOMALY,
                    t_start=tr.first_seen,
                    t_end=tr.last_seen,
                    track_ids=[tr.track_id],
                    signal_strength=round(strength, 4),
                    features={
                        "dwell_ratio_vs_median": round(ratio, 3),
                        "stationarity": round(dwell, 4),
                        "duration_s": round(tr.duration, 3),
                    },
                )
            )
        return out


def _inside(x: float, y: float, poly) -> bool:
    from .zones import point_in_polygon

    return point_in_polygon(x, y, poly)


def _densest_cluster(positions: dict[int, tuple[float, float]], radius: float) -> list[int]:
    """Greedy single-pass cluster: the neighbourhood with the most members.

    O(n^2) in actors-per-frame, which is fine -- n is the number of tracked
    objects visible at one instant, not the corpus size.
    """
    best: list[int] = []
    for _anchor, (ax, ay) in positions.items():
        members = [
            tid for tid, (x, y) in positions.items()
            if ((x - ax) ** 2 + (y - ay) ** 2) ** 0.5 <= radius
        ]
        if len(members) > len(best):
            best = members
    return best


def _cluster_runs(hot, gap: float, max_drift: float):
    """Group hot seconds into runs that stay in roughly the same place."""
    runs: list[list] = []
    for entry in hot:
        if runs:
            prev = runs[-1][-1]
            dt = entry[0] - prev[0]
            drift = ((entry[1][0] - prev[1][0]) ** 2 + (entry[1][1] - prev[1][1]) ** 2) ** 0.5
            if dt <= gap and drift <= max_drift:
                runs[-1].append(entry)
                continue
        runs.append([entry])
    return runs
