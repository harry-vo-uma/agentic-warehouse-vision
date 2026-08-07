"""Shared state passed between graph nodes."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..schemas import Event, EventCandidate, Recommendation, VLMFinding


@dataclass
class GraphState:
    candidate: EventCandidate
    keyframes_b64: list[str] = field(default_factory=list)
    keyframe_uris: list[str] = field(default_factory=list)
    clip_uri: str | None = None
    track_context: dict[str, Any] = field(default_factory=dict)
    finding: VLMFinding | None = None
    fused_confidence: float = 0.0
    summary: str = ""
    narrative: str = ""
    recommendations: list[Recommendation] = field(default_factory=list)
    event: Event | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def log(self, node: str, **payload: Any) -> None:
        self.trace.append({"node": node, "ts": time.time(), **payload})

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate.candidate_id,
            "kind": self.candidate.kind.value,
            "fused_confidence": self.fused_confidence,
            "n_trace": len(self.trace),
            "errors": self.errors,
        }
