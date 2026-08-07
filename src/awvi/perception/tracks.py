"""Accumulates per-frame detections into trajectories and derives the
temporal features the agent graph reasons over.

Frame-level detection alone is what produces the false-positive problem this
system exists to solve: a person who appears near a forklift for two frames is
not an unsafe interaction. Everything here is about giving the reasoning layer
*time* as a first-class input.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from ..schemas import Detection, Track, TrackPoint

VEHICLES = {"forklift", "agv", "tugger", "reach_truck"}
PEDESTRIANS = {"person", "worker"}
STATIC = {"pallet", "box", "cart", "debris", "tote"}


class TrackStore:
    """In-memory trajectory store for one camera.

    Deliberately not a database: at 30fps over a 40h corpus the working set
    that matters is the last few minutes, and completed tracks are handed off
    to the event index as soon as they close.
    """

    def __init__(self, camera_id: str, max_gap_s: float = 2.0):
        self.camera_id = camera_id
        self.max_gap_s = max_gap_s
        self._tracks: dict[int, Track] = {}
        self._conf_sum: dict[int, float] = defaultdict(float)
        self._conf_n: dict[int, int] = defaultdict(int)

    def update(self, detections: Iterable[Detection]) -> None:
        for d in detections:
            tr = self._tracks.get(d.track_id)
            if tr is None:
                tr = Track(
                    track_id=d.track_id,
                    label=d.label,
                    camera_id=self.camera_id,
                    first_seen=d.timestamp,
                    last_seen=d.timestamp,
                )
                self._tracks[d.track_id] = tr
            cx, cy = d.bbox.center
            tr.points.append(TrackPoint(t=d.timestamp, x=cx, y=cy))
            tr.last_seen = max(tr.last_seen, d.timestamp)
            self._conf_sum[d.track_id] += d.confidence
            self._conf_n[d.track_id] += 1
            tr.mean_confidence = self._conf_sum[d.track_id] / self._conf_n[d.track_id]

    def get(self, track_id: int) -> Track | None:
        return self._tracks.get(track_id)

    def all(self) -> list[Track]:
        return list(self._tracks.values())

    def active_at(self, t: float, tol: float = 0.5) -> list[Track]:
        return [
            tr for tr in self._tracks.values()
            if tr.first_seen - tol <= t <= tr.last_seen + tol
        ]

    def in_window(self, t0: float, t1: float) -> list[Track]:
        return [tr for tr in self._tracks.values() if tr.last_seen >= t0 and tr.first_seen <= t1]

    # ---------------------------------------------------------------- features
    def positions_at(self, t: float) -> dict[int, tuple[float, float]]:
        """Linear interpolation of every track's position at time t."""
        out: dict[int, tuple[float, float]] = {}
        for tr in self._tracks.values():
            pts = tr.points
            if not pts or t < pts[0].t or t > pts[-1].t:
                continue
            prev = pts[0]
            for p in pts:
                if p.t >= t:
                    span = p.t - prev.t
                    if span <= 1e-9:
                        out[tr.track_id] = (p.x, p.y)
                    else:
                        a = (t - prev.t) / span
                        out[tr.track_id] = (
                            prev.x + a * (p.x - prev.x),
                            prev.y + a * (p.y - prev.y),
                        )
                    break
                prev = p
        return out

    def min_separation(self, a: int, b: int) -> tuple[float, float]:
        """Closest approach between two tracks: (distance, timestamp)."""
        ta, tb = self._tracks.get(a), self._tracks.get(b)
        if not ta or not tb or not ta.points or not tb.points:
            return (float("inf"), 0.0)
        best, best_t = float("inf"), 0.0
        j = 0
        for p in ta.points:
            while j + 1 < len(tb.points) and tb.points[j + 1].t < p.t:
                j += 1
            q = tb.points[j]
            if abs(q.t - p.t) > 0.5:
                continue
            d = ((p.x - q.x) ** 2 + (p.y - q.y) ** 2) ** 0.5
            if d < best:
                best, best_t = d, p.t
        return (best, best_t)

    def density_series(self, bucket_s: float = 1.0) -> dict[float, int]:
        """Count of concurrently visible tracks per time bucket."""
        counts: dict[float, int] = defaultdict(int)
        for tr in self._tracks.values():
            t = tr.first_seen
            while t <= tr.last_seen:
                counts[round(t // bucket_s * bucket_s, 3)] += 1
                t += bucket_s
        return dict(counts)
