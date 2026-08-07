"""Agent orchestration.

The topology is:

    perception -> investigation -> [gate] -> clip -> recommendation
                                      \\-> recommendation  (suppress path)

Built on LangGraph when it is installed, and on an equivalent built-in runner
when it is not. Both paths execute the *same* node callables and the same
conditional edge, so behaviour -- and therefore eval numbers -- do not depend
on which one you get. The fallback exists because a reference application that
cannot start without a specific orchestration library is not a reference
application.
"""
from __future__ import annotations

import logging
from collections.abc import Callable

from ..config import get_settings
from ..nim.client import NIMClient, get_client
from ..perception.tracks import TrackStore
from ..schemas import Event, EventCandidate
from ..video.clips import ClipExtractor
from ..video.frames import KeyframeSampler
from .escalation import EscalationPolicy, score_event
from .investigation_agent import InvestigationAgent
from .perception_agent import PerceptionAgent
from .recommendation_agent import RecommendationAgent
from .state import GraphState

log = logging.getLogger(__name__)

Node = Callable[[GraphState], GraphState]


def langgraph_available() -> bool:
    try:
        import langgraph  # noqa: F401
        return True
    except Exception:
        return False


class ClipNode:
    """Cuts the evidence clip. Skipped for candidates headed for suppression,
    which is most of them -- clip extraction is the most expensive I/O in the
    pipeline and doing it eagerly was the original throughput bottleneck.
    """

    name = "clip"

    def __init__(self, extractor: ClipExtractor | None = None):
        self.extractor = extractor or ClipExtractor()

    def __call__(self, state: GraphState) -> GraphState:
        cfg = get_settings()
        clip = self.extractor.extract(
            camera_id=state.candidate.camera_id,
            t_start=max(0.0, state.candidate.t_start - cfg.clip_pre_roll_s),
            t_end=state.candidate.t_end + cfg.clip_post_roll_s,
            event_id=state.candidate.candidate_id,
        )
        state.clip_uri = clip.uri
        state.log(self.name, clip_uri=clip.uri, duration_s=round(clip.duration, 2), materialised=clip.materialised)
        return state


def should_extract_clip(state: GraphState) -> str:
    """Conditional edge: cheap pre-check using the same fusion function the
    recommendation node will run, so the gate can never disagree with it.
    """
    cfg = get_settings().escalation
    provisional = score_event(state.candidate, state.finding, cfg)
    return "clip" if provisional.confidence >= cfg.suppress_below else "recommendation"


class IncidentGraph:
    """Executable graph over a single camera's track store."""

    def __init__(
        self,
        store: TrackStore,
        *,
        client: NIMClient | None = None,
        sampler: KeyframeSampler | None = None,
        extractor: ClipExtractor | None = None,
        policy: EscalationPolicy | None = None,
    ):
        client = client or get_client()
        self.store = store
        self.perception = PerceptionAgent(store, sampler)
        self.investigation = InvestigationAgent(client)
        self.clip = ClipNode(extractor)
        self.recommendation = RecommendationAgent(client, policy or EscalationPolicy())
        self._compiled = self._compile()

    # ------------------------------------------------------------- compilation
    def _compile(self):
        if not langgraph_available():
            log.info("langgraph not installed; using built-in sequential runner")
            return None
        from langgraph.graph import END, START, StateGraph

        g = StateGraph(dict)

        def wrap(fn: Node):
            def _node(payload: dict) -> dict:
                st: GraphState = payload["state"]
                return {"state": fn(st)}
            return _node

        g.add_node("perception", wrap(self.perception))
        g.add_node("investigation", wrap(self.investigation))
        g.add_node("clip", wrap(self.clip))
        g.add_node("recommendation", wrap(self.recommendation))

        g.add_edge(START, "perception")
        g.add_edge("perception", "investigation")
        g.add_conditional_edges(
            "investigation",
            lambda payload: should_extract_clip(payload["state"]),
            {"clip": "clip", "recommendation": "recommendation"},
        )
        g.add_edge("clip", "recommendation")
        g.add_edge("recommendation", END)
        return g.compile()

    @property
    def backend(self) -> str:
        return "langgraph" if self._compiled is not None else "builtin"

    # ---------------------------------------------------------------- running
    def run_one(self, candidate: EventCandidate) -> GraphState:
        state = GraphState(candidate=candidate)
        if self._compiled is not None:
            out = self._compiled.invoke({"state": state})
            return out["state"]
        # Built-in runner, identical topology.
        state = self.perception(state)
        state = self.investigation(state)
        if should_extract_clip(state) == "clip":
            state = self.clip(state)
        return self.recommendation(state)

    def run(self, candidates: list[EventCandidate], *, include_suppressed: bool = True) -> list[Event]:
        events: list[Event] = []
        for c in candidates:
            st = self.run_one(c)
            if st.event is None:
                continue
            if not include_suppressed and st.event.disposition.value == "suppressed":
                continue
            events.append(st.event)
        return events


def build_graph(store: TrackStore, **kw) -> IncidentGraph:
    return IncidentGraph(store, **kw)
