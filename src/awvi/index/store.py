"""Natural-language event retrieval.

Hybrid scoring: dense similarity over NIM embeddings plus a lexical BM25-ish
term score, with structured filters (camera / kind / severity / time) parsed
out of the query first. Pure vector search does badly here because operators
ask questions like "forklift near dock 3 yesterday afternoon" -- the useful
constraints are structured, and burying them in an embedding throws them away.
"""
from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from ..nim.client import NIMClient, get_client
from ..schemas import Event, EventKind, QueryHit, QueryResponse, Severity

_KIND_SYNONYMS = {
    EventKind.CONGESTION: ["congestion", "congested", "queue", "queueing", "backed up", "traffic", "jam", "crowded"],
    EventKind.BLOCKED_ZONE: ["blocked", "obstruction", "obstructed", "keep clear", "blocking", "egress", "pallet in"],
    EventKind.UNSAFE_INTERACTION: ["unsafe", "near miss", "near-miss", "close call", "hazard", "danger", "forklift and", "pedestrian"],
    EventKind.WORKFLOW_ANOMALY: ["anomaly", "stuck", "stalled", "idle", "dwell", "delay", "abandoned", "orphan"],
}

_SEVERITY_WORDS = {
    "critical": Severity.CRITICAL,
    "severe": Severity.CRITICAL,
    "high": Severity.HIGH,
    "serious": Severity.HIGH,
    "medium": Severity.MEDIUM,
    "minor": Severity.LOW,
    "low": Severity.LOW,
}

_STOP = {
    "the", "a", "an", "of", "in", "on", "at", "to", "and", "or", "is", "was", "were",
    "did", "do", "any", "show", "me", "what", "when", "where", "which", "that", "there",
    "for", "with", "from", "by", "it", "this", "these", "those", "have", "has", "had",
}


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP and len(t) > 1]


@dataclass
class _Filters:
    kinds: set[EventKind] = field(default_factory=set)
    cameras: set[str] = field(default_factory=set)
    min_severity: Severity | None = None
    t_start: float | None = None
    t_end: float | None = None

    def matches(self, e: Event) -> bool:
        if self.kinds and e.kind not in self.kinds:
            return False
        if self.cameras and e.camera_id not in self.cameras:
            return False
        if self.min_severity is not None:
            order = [Severity.INFO, Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
            if order.index(e.severity) < order.index(self.min_severity):
                return False
        if self.t_start is not None and e.t_end < self.t_start:
            return False
        if self.t_end is not None and e.t_start > self.t_end:
            return False
        return True


def parse_filters(query: str, known_cameras: set[str]) -> _Filters:
    q = query.lower()
    f = _Filters()
    for kind, words in _KIND_SYNONYMS.items():
        if any(w in q for w in words):
            f.kinds.add(kind)
    for cam in known_cameras:
        if cam.lower() in q:
            f.cameras.add(cam)
        tail = cam.lower().split("-")[-1]
        if tail and re.search(rf"\b(camera|cam)\s*{re.escape(tail)}\b", q):
            f.cameras.add(cam)
    for word, sev in _SEVERITY_WORDS.items():
        if word in q:
            f.min_severity = sev
            break
    m = re.search(r"(?:after|since|from)\s+(\d+(?:\.\d+)?)\s*s", q)
    if m:
        f.t_start = float(m.group(1))
    m = re.search(r"(?:before|until|up to)\s+(\d+(?:\.\d+)?)\s*s", q)
    if m:
        f.t_end = float(m.group(1))
    m = re.search(r"between\s+(\d+(?:\.\d+)?)\s*(?:s)?\s*and\s+(\d+(?:\.\d+)?)\s*s", q)
    if m:
        f.t_start, f.t_end = float(m.group(1)), float(m.group(2))
    return f


class EventIndex:
    """In-memory hybrid index. Swap `_vectors` for a real vector DB in prod;
    the interface is deliberately three methods wide.
    """

    def __init__(self, client: NIMClient | None = None, dense_weight: float = 0.6):
        self.client = client or get_client()
        self.dense_weight = dense_weight
        self._events: dict[str, Event] = {}
        self._vectors: dict[str, list[float]] = {}
        self._postings: dict[str, set[str]] = defaultdict(set)
        self._doc_len: dict[str, int] = {}
        self._tf: dict[str, Counter] = {}

    # ------------------------------------------------------------------ writes
    def add(self, event: Event) -> None:
        self.add_many([event])

    def add_many(self, events: list[Event]) -> None:
        events = [e for e in events if e.event_id not in self._events]
        if not events:
            return
        texts = [e.searchable_text() for e in events]
        vectors = self.client.embed(texts, input_type="passage")
        for e, text, vec in zip(events, texts, vectors, strict=True):
            self._events[e.event_id] = e
            self._vectors[e.event_id] = vec
            toks = _tokens(text)
            self._tf[e.event_id] = Counter(toks)
            self._doc_len[e.event_id] = len(toks) or 1
            for t in set(toks):
                self._postings[t].add(e.event_id)

    # ------------------------------------------------------------------- reads
    def __len__(self) -> int:
        return len(self._events)

    def get(self, event_id: str) -> Event | None:
        return self._events.get(event_id)

    def all(self) -> list[Event]:
        return sorted(self._events.values(), key=lambda e: e.t_start)

    @property
    def cameras(self) -> set[str]:
        return {e.camera_id for e in self._events.values()}

    # ------------------------------------------------------------------ search
    def _lexical(self, query_tokens: list[str]) -> dict[str, float]:
        n = len(self._events) or 1
        avg_len = sum(self._doc_len.values()) / n
        k1, b = 1.4, 0.72
        scores: dict[str, float] = defaultdict(float)
        for t in query_tokens:
            docs = self._postings.get(t)
            if not docs:
                continue
            idf = math.log(1 + (n - len(docs) + 0.5) / (len(docs) + 0.5))
            for eid in docs:
                tf = self._tf[eid][t]
                dl = self._doc_len[eid]
                scores[eid] += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_len))
        return scores

    def search(self, query: str, top_k: int = 5, include_suppressed: bool = False) -> list[QueryHit]:
        if not self._events:
            return []
        filters = parse_filters(query, self.cameras)
        pool = [
            e for e in self._events.values()
            if filters.matches(e) and (include_suppressed or e.disposition.value != "suppressed")
        ]
        if not pool:
            pool = [e for e in self._events.values() if include_suppressed or e.disposition.value != "suppressed"]

        qvec = self.client.embed([query], input_type="query")[0]
        lex = self._lexical(_tokens(query))
        lex_max = max(lex.values(), default=0.0) or 1.0

        hits: list[QueryHit] = []
        for e in pool:
            dense = _cosine(qvec, self._vectors[e.event_id])
            lexical = lex.get(e.event_id, 0.0) / lex_max
            score = self.dense_weight * dense + (1 - self.dense_weight) * lexical
            # Confident, high-severity events break ties toward what an
            # operator most likely meant.
            score += 0.05 * e.confidence
            matched = []
            if filters.kinds:
                matched.append("kind")
            if filters.cameras:
                matched.append("camera")
            if lexical > 0:
                matched.append("lexical")
            if dense > 0:
                matched.append("semantic")
            hits.append(QueryHit(event=e, score=round(score, 5), matched_on=matched))

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def answer(self, query: str, top_k: int = 5) -> QueryResponse:
        t0 = time.perf_counter()
        hits = self.search(query, top_k=top_k)
        if not hits:
            return QueryResponse(
                query=query,
                answer="No events in the index match that description.",
                hits=[],
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
        context = "\n".join(
            f"- [{h.event.event_id}] {h.event.kind.value} on {h.event.camera_id} "
            f"({h.event.t_start:.1f}s-{h.event.t_end:.1f}s, {h.event.severity.value}, "
            f"conf {h.event.confidence:.2f}): {h.event.summary}"
            for h in hits
        )
        prompt = (
            "Answer the operator question using ONLY the retrieved events below. "
            "Be concise, reference events by their id, and say plainly if the "
            "evidence does not answer the question.\n\n"
            f"Operator question: {query}\n\nRetrieved events:\n{context}"
        )
        answer = self.client.chat(prompt, temperature=0.2, max_tokens=400)
        return QueryResponse(
            query=query,
            answer=answer.strip(),
            hits=hits,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(v * v for v in a[:n])) or 1.0
    nb = math.sqrt(sum(v * v for v in b[:n])) or 1.0
    return dot / (na * nb)
