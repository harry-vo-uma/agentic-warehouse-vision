"""Node 1 -- perception.

Gathers the evidence packet for a candidate: the temporal context from the
track store and the keyframes the VLM will actually look at. Frame selection
matters more than people expect; uniformly sampling the window wastes the
budget on empty lead-in, so we bias sampling toward the moment of peak signal.
"""
from __future__ import annotations

from ..config import get_settings
from ..perception.tracks import TrackStore
from ..schemas import EventCandidate
from ..video.frames import KeyframeSampler
from .state import GraphState


class PerceptionAgent:
    name = "perception"

    def __init__(self, store: TrackStore, sampler: KeyframeSampler | None = None):
        self.store = store
        self.sampler = sampler or KeyframeSampler()

    def __call__(self, state: GraphState) -> GraphState:
        cfg = get_settings()
        c: EventCandidate = state.candidate

        tracks = [self.store.get(tid) for tid in c.track_ids]
        tracks = [t for t in tracks if t is not None]
        state.track_context = {
            "n_tracks": len(tracks),
            "labels": sorted({t.label for t in tracks}),
            "window_s": round(c.t_end - c.t_start, 3),
            "mean_speed": round(sum(t.net_speed() for t in tracks) / len(tracks), 4) if tracks else 0.0,
            "max_dwell_ratio": round(max((t.dwell_ratio() for t in tracks), default=0.0), 4),
            "mean_track_confidence": round(
                sum(t.mean_confidence for t in tracks) / len(tracks), 4
            ) if tracks else 0.0,
            "concurrent_actors": len(self.store.active_at((c.t_start + c.t_end) / 2)),
        }

        peak_t = _peak_time(c)
        frames = self.sampler.sample(
            camera_id=c.camera_id,
            t_start=c.t_start,
            t_end=c.t_end,
            focus_t=peak_t,
            n=cfg.keyframes_per_event,
        )
        state.keyframes_b64 = [f.b64 for f in frames]
        state.keyframe_uris = [f.uri for f in frames]
        state.log(
            self.name,
            n_keyframes=len(frames),
            focus_t=round(peak_t, 3),
            **state.track_context,
        )
        return state


def _peak_time(c: EventCandidate) -> float:
    """Best guess at the most informative instant in the window."""
    mid = (c.t_start + c.t_end) / 2
    # Proximity events peak at closest approach, which the proposer centred on.
    if "min_distance" in c.features:
        return mid
    # Dwell-driven events are most legible late, once the condition is obvious.
    if "dwell_s" in c.features or "dwell_ratio_vs_median" in c.features:
        return c.t_start + 0.75 * (c.t_end - c.t_start)
    return mid
