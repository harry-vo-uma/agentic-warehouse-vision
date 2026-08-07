"""Operator-facing API + single-page UI.

Endpoints exist so an operator can do three things without leaving the page:
ask a question in plain language, look at the supporting clip, and see what
they should do about it. That triad is the whole product.
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from ..config import get_settings
from ..index.store import EventIndex
from ..nim.client import get_client
from ..pipeline import Pipeline, load_zones
from ..schemas import Event

STATIC = Path(__file__).parent / "static"

_index = EventIndex()
_report: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _bootstrap()
    yield


app = FastAPI(
    title="Agentic Warehouse Vision Intelligence",
    version="0.3.0",
    description="Multi-camera Vision AI incident triage: DeepStream tracking + VLM reasoning + agent orchestration.",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


def _bootstrap() -> None:
    """Load precomputed events if present, otherwise run the demo scenarios."""
    global _report
    events_path = Path(os.getenv("AWVI_EVENTS", "data/events.json"))
    if events_path.exists():
        events = [Event.model_validate(e) for e in json.loads(events_path.read_text())]
        _index.add_many([e for e in events if e.disposition.value != "suppressed"])
        _report = {"source": str(events_path), "events": len(events)}
        return
    scenarios = Path(os.getenv("AWVI_SCENARIOS", "data/scenarios"))
    if scenarios.exists():
        pipe = Pipeline(zones=load_zones(), index=_index)
        pipe.process_all(scenarios)
        _report = pipe.report.as_dict()


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return (STATIC / "index.html").read_text()


@app.get("/api/health")
def health() -> dict[str, Any]:
    s = get_settings()
    return {
        "status": "ok",
        "events_indexed": len(_index),
        "nim": get_client().stats(),
        "vlm_model": s.nim.vlm_model,
        "escalation": s.escalation.__dict__,
        "run_report": _report,
    }


@app.get("/api/events")
def list_events(
    kind: str | None = None,
    camera_id: str | None = None,
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(100, ge=1, le=1000),
) -> dict[str, Any]:
    events = _index.all()
    if kind:
        events = [e for e in events if e.kind.value == kind]
    if camera_id:
        events = [e for e in events if e.camera_id == camera_id]
    events = [e for e in events if e.confidence >= min_confidence]
    return {
        "count": len(events),
        "events": [json.loads(e.model_dump_json()) for e in events[:limit]],
    }


@app.get("/api/events/{event_id}")
def get_event(event_id: str) -> dict[str, Any]:
    e = _index.get(event_id)
    if e is None:
        raise HTTPException(404, f"no event {event_id}")
    return json.loads(e.model_dump_json())


@app.post("/api/ask")
def ask(req: AskRequest) -> dict[str, Any]:
    resp = _index.answer(req.question, top_k=req.top_k)
    return json.loads(resp.model_dump_json())


@app.get("/api/summary")
def summary() -> dict[str, Any]:
    events = _index.all()
    by_kind: dict[str, int] = {}
    by_disposition: dict[str, int] = {}
    by_camera: dict[str, int] = {}
    for e in events:
        by_kind[e.kind.value] = by_kind.get(e.kind.value, 0) + 1
        by_disposition[e.disposition.value] = by_disposition.get(e.disposition.value, 0) + 1
        by_camera[e.camera_id] = by_camera.get(e.camera_id, 0) + 1
    return {
        "total": len(events),
        "by_kind": by_kind,
        "by_disposition": by_disposition,
        "by_camera": by_camera,
        "mean_confidence": round(sum(e.confidence for e in events) / len(events), 4) if events else 0.0,
    }


@app.get("/media/clips/{name}")
def clip(name: str):
    path = get_settings().data_dir / "clips" / name
    if not path.exists():
        raise HTTPException(404, "clip not materialised; source media not mounted")
    return FileResponse(path)
