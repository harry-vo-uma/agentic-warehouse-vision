"""Core data contracts shared across perception, agents, index and API layers.

Everything that crosses a module boundary is a pydantic model so that the
mock backend, the DeepStream backend and the HTTP API all agree on shape.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class EventKind(str, Enum):
    CONGESTION = "congestion"
    BLOCKED_ZONE = "blocked_zone"
    UNSAFE_INTERACTION = "unsafe_interaction"
    WORKFLOW_ANOMALY = "workflow_anomaly"
    NOMINAL = "nominal"


class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Disposition(str, Enum):
    """Terminal decision produced by the escalation policy."""

    SUPPRESSED = "suppressed"      # believed false positive
    LOGGED = "logged"              # real but not actionable
    ESCALATED = "escalated"        # routed to an operator
    PAGED = "paged"                # routed to on-call / safety lead


class BBox(BaseModel):
    x: float
    y: float
    w: float
    h: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def iou(self, other: BBox) -> float:
        ax2, ay2 = self.x + self.w, self.y + self.h
        bx2, by2 = other.x + other.w, other.y + other.h
        ix = max(0.0, min(ax2, bx2) - max(self.x, other.x))
        iy = max(0.0, min(ay2, by2) - max(self.y, other.y))
        inter = ix * iy
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


class Detection(BaseModel):
    """One object in one frame, as emitted by the tracker."""

    track_id: int
    label: str                      # person | forklift | pallet | box | agv
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BBox
    frame_idx: int
    timestamp: float                # seconds from stream start


class TrackPoint(BaseModel):
    t: float
    x: float
    y: float


class Track(BaseModel):
    """A trajectory accumulated over time for a single tracked object."""

    track_id: int
    label: str
    camera_id: str
    points: list[TrackPoint] = Field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0
    mean_confidence: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)

    def displacement(self) -> float:
        if len(self.points) < 2:
            return 0.0
        a, b = self.points[0], self.points[-1]
        return ((b.x - a.x) ** 2 + (b.y - a.y) ** 2) ** 0.5

    def net_speed(self) -> float:
        """Net displacement per second.

        Prefer this to `mean_speed` for any decision about whether something is
        *going* anywhere. Path-length speed is dominated by detector jitter: a
        stationary pallet with a few pixels of box wobble reads as moving,
        which is how parked forklifts ended up in an unsafe-interaction alert.
        """
        return self.displacement() / self.duration if self.duration > 0 else 0.0

    def mean_speed(self) -> float:
        """Path-length speed. Includes jitter; use `net_speed` for gating."""
        if self.duration <= 0:
            return 0.0
        path = 0.0
        for p, q in zip(self.points, self.points[1:], strict=False):
            path += ((q.x - p.x) ** 2 + (q.y - p.y) ** 2) ** 0.5
        return path / self.duration

    def dwell_ratio(self) -> float:
        """1.0 == object never left its starting neighbourhood."""
        if len(self.points) < 2:
            return 1.0
        path = sum(
            ((q.x - p.x) ** 2 + (q.y - p.y) ** 2) ** 0.5
            for p, q in zip(self.points, self.points[1:], strict=False)
        )
        if path <= 1e-9:
            return 1.0
        return max(0.0, 1.0 - self.displacement() / path)


class Zone(BaseModel):
    """Named polygon in normalised image coordinates."""

    zone_id: str
    name: str
    camera_id: str
    kind: Literal["keep_clear", "aisle", "dock", "pedestrian", "storage"] = "aisle"
    polygon: list[tuple[float, float]]

    @field_validator("polygon")
    @classmethod
    def _at_least_a_triangle(cls, v: list[tuple[float, float]]):
        if len(v) < 3:
            raise ValueError("a zone polygon needs at least 3 vertices")
        return v


class EventCandidate(BaseModel):
    """Cheap, rule-derived hypothesis. Deliberately high recall / low precision.

    The whole point of the agent graph is to turn these into precise events.
    """

    candidate_id: str
    camera_id: str
    kind: EventKind
    t_start: float
    t_end: float
    track_ids: list[int] = Field(default_factory=list)
    zone_id: str | None = None
    signal_strength: float = Field(0.0, ge=0.0, le=1.0)
    features: dict[str, float] = Field(default_factory=dict)

    @staticmethod
    def make_id(camera_id: str, kind: str, t_start: float) -> str:
        raw = f"{camera_id}:{kind}:{t_start:.3f}"
        return hashlib.sha1(raw.encode()).hexdigest()[:12]


class VLMFinding(BaseModel):
    """Structured output extracted from the VLM's reading of sampled frames."""

    supports_candidate: bool
    observed_kind: EventKind
    confidence: float = Field(ge=0.0, le=1.0)
    description: str
    entities: list[str] = Field(default_factory=list)
    contributing_factors: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    action: str
    rationale: str
    owner_role: str = "floor_supervisor"
    urgency: Severity = Severity.MEDIUM
    est_minutes: int = 5


class Event(BaseModel):
    """The confirmed, operator-facing record. This is what the UI renders."""

    event_id: str
    camera_id: str
    kind: EventKind
    severity: Severity
    t_start: float
    t_end: float
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    confidence: float = Field(ge=0.0, le=1.0)
    disposition: Disposition = Disposition.LOGGED
    summary: str = ""
    narrative: str = ""
    zone_id: str | None = None
    track_ids: list[int] = Field(default_factory=list)
    clip_uri: str | None = None
    keyframe_uris: list[str] = Field(default_factory=list)
    findings: VLMFinding | None = None
    recommendations: list[Recommendation] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)

    def searchable_text(self) -> str:
        parts = [self.kind.value, self.summary, self.narrative, self.zone_id or ""]
        if self.findings:
            parts += self.findings.entities + self.findings.contributing_factors
        parts += [r.action for r in self.recommendations]
        return " ".join(p for p in parts if p)


class QueryHit(BaseModel):
    event: Event
    score: float
    matched_on: list[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    query: str
    answer: str
    hits: list[QueryHit] = Field(default_factory=list)
    latency_ms: float = 0.0
