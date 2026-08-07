"""End-to-end pipeline: source -> tracks -> candidates -> agent graph -> index."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .agents.graph import IncidentGraph
from .config import CONFIG_DIR, get_settings
from .index.store import EventIndex
from .nim.client import get_client
from .perception.deepstream import SyntheticSource, build_source
from .perception.events import CandidateProposer
from .perception.tracks import TrackStore
from .perception.zones import ZoneIndex
from .schemas import Event

log = logging.getLogger(__name__)


@dataclass
class RunReport:
    cameras: int = 0
    detections: int = 0
    tracks: int = 0
    candidates: int = 0
    events: int = 0
    suppressed: int = 0
    escalated: int = 0
    paged: int = 0
    wall_ms: float = 0.0
    nim: dict[str, Any] = field(default_factory=dict)
    graph_backend: str = "builtin"

    def as_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["suppression_rate"] = round(self.suppressed / self.candidates, 4) if self.candidates else 0.0
        return d


class Pipeline:
    def __init__(self, zones: ZoneIndex | None = None, index: EventIndex | None = None):
        self.settings = get_settings()
        self.zones = zones if zones is not None else load_zones()
        self.proposer = CandidateProposer(self.zones)
        self.index = index or EventIndex()
        self.report = RunReport()

    def process_scenario(self, scenario: dict) -> list[Event]:
        t0 = time.perf_counter()
        source = SyntheticSource(scenario, seed=self.settings.seed)
        return self._process_source(source, t0)

    def process_camera(self, camera_id: str, uri: str | None = None, scenario: dict | None = None) -> list[Event]:
        t0 = time.perf_counter()
        source = build_source(camera_id, uri=uri, scenario=scenario)
        return self._process_source(source, t0)

    def _process_source(self, source, t0: float) -> list[Event]:
        store = TrackStore(source.camera_id)
        n_det = 0
        batch = []
        for det in source.stream():
            batch.append(det)
            n_det += 1
            if len(batch) >= 512:
                store.update(batch)
                batch = []
        if batch:
            store.update(batch)

        candidates = self.proposer.propose(store)
        graph = IncidentGraph(store)
        events = graph.run(candidates)

        self.index.add_many([e for e in events if e.disposition.value != "suppressed"])

        self.report.cameras += 1
        self.report.detections += n_det
        self.report.tracks += len(store.all())
        self.report.candidates += len(candidates)
        self.report.events += len(events)
        self.report.suppressed += sum(1 for e in events if e.disposition.value == "suppressed")
        self.report.escalated += sum(1 for e in events if e.disposition.value == "escalated")
        self.report.paged += sum(1 for e in events if e.disposition.value == "paged")
        self.report.wall_ms += (time.perf_counter() - t0) * 1000
        self.report.nim = get_client().stats()
        self.report.graph_backend = graph.backend
        return events

    def process_all(self, scenario_dir: str | Path) -> list[Event]:
        out: list[Event] = []
        for path in sorted(Path(scenario_dir).glob("*.json")):
            out += self.process_scenario(json.loads(path.read_text()))
        return out


def load_zones(path: str | Path | None = None) -> ZoneIndex:
    path = Path(path) if path else CONFIG_DIR / "zones.yaml"
    if not path.exists():
        return ZoneIndex()
    return ZoneIndex.from_config(yaml.safe_load(path.read_text()) or {})
