"""Deterministic offline stand-in for NIM endpoints.

Why this exists: the agent graph, the escalation policy, the event index and
the API are the interesting parts of this system, and none of them should
require an H100 to exercise. The mock is seeded by content hash, so the same
scene always yields the same reading -- which is what makes `eval/run_eval.py`
reproducible on a laptop.

It is NOT a model. It reads the numeric scene features the perception layer
already computed and writes them back as prose + structured JSON, with a
calibrated error rate so that downstream precision/recall numbers are not
trivially 1.0.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any

_KIND_PHRASES = {
    "congestion": (
        "Multiple people and at least one powered vehicle are occupying the same aisle "
        "segment with little forward progress",
        ["queueing at the aisle mouth", "pallet staged mid-aisle", "cross-traffic from the dock"],
    ),
    "blocked_zone": (
        "A pallet and loose stock are sitting inside a marked keep-clear area",
        ["overflow from adjacent storage", "staging during a shift change", "no open floor location nearby"],
    ),
    "unsafe_interaction": (
        "A pedestrian passes within close range of a moving forklift without visible acknowledgement",
        ["pedestrian shortcut across the travel lane", "forklift approaching a blind corner", "high forklift speed"],
    ),
    "workflow_anomaly": (
        "Stock is dwelling at a station far longer than the surrounding sequence would predict",
        ["upstream starvation", "scanner or system error at the station", "operator stepped away"],
    ),
    "nominal": (
        "Normal traffic flow with adequate clearance between people and equipment",
        [],
    ),
}

_CONTRADICTIONS = [
    "the forklift is stationary with forks grounded",
    "the person is standing behind the yellow safety line",
    "the aisle clears within the observed window",
    "the object in the zone is a floor marking, not stock",
    "occlusion from a rack column makes proximity ambiguous",
]


def _seed_from(*parts: Any) -> int:
    raw = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


class MockNIMBackend:
    """Seeded, dependency-free responder for chat / vlm / embed."""

    #: Simulated per-call reliability of the vision reading. Chosen to sit near
    #: the observed agreement rate of the live VLM on the held-out split so the
    #: mock does not flatter the downstream policy.
    vlm_reliability: float = 0.88

    def __init__(self, reliability: float | None = None):
        if reliability is not None:
            self.vlm_reliability = reliability

    # ------------------------------------------------------------------ vision
    def vlm(self, prompt: str, n_images: int, hint: dict[str, Any] | None = None) -> str:
        hint = hint or {}
        kind = str(hint.get("kind", "nominal"))
        strength = float(hint.get("signal_strength", 0.5))
        features: dict[str, float] = dict(hint.get("features", {}))
        rng = random.Random(_seed_from(hint.get("candidate_id", ""), kind, round(strength, 4), n_images))

        # The mock "sees" what the geometry already implies, then adds
        # calibrated disagreement so evaluation is not degenerate.
        agrees = rng.random() < self.vlm_reliability
        supports = (strength >= 0.5) if agrees else (strength < 0.5)

        base, factors = _KIND_PHRASES.get(kind, _KIND_PHRASES["nominal"])
        observed = kind if supports else "nominal"

        # Confidence is a logistic squash of signal strength, nudged by the
        # strongest supporting feature, then clamped away from 0/1.
        lead = max(features.values()) if features else 0.5
        z = 2.4 * (strength - 0.5) + 0.8 * (lead - 0.5)
        conf = 1.0 / (1.0 + math.exp(-z))
        if not supports:
            conf = 1.0 - conf
        conf = min(0.93, max(0.08, conf + rng.uniform(-0.06, 0.06)))

        entities = list(hint.get("entities", [])) or ["person", "forklift"]
        n_factors = 2 if supports else 0
        contradicting = [] if supports else rng.sample(_CONTRADICTIONS, k=min(2, len(_CONTRADICTIONS)))

        payload = {
            "supports_candidate": supports,
            "observed_kind": observed,
            "confidence": round(conf, 3),
            "description": (
                base if supports else "No hazardous condition is visible across the sampled frames"
            )
            + f" (reviewed {n_images} keyframes).",
            "entities": entities[:6],
            "contributing_factors": factors[:n_factors],
            "contradicting_evidence": contradicting,
        }
        return json.dumps(payload)

    # -------------------------------------------------------------------- text
    def chat(self, prompt: str, system: str = "") -> str:
        rng = random.Random(_seed_from(prompt[:512], system[:128]))
        low = prompt.lower()

        if "recommendation" in low or "recommend" in low:
            return json.dumps({"recommendations": self._recommendations(prompt, rng)})
        if "answer the operator" in low or "operator question" in low:
            return self._answer(prompt)
        if "summar" in low:
            return json.dumps(
                {
                    "summary": self._summary(prompt),
                    "narrative": (
                        "Tracking and vision evidence agree on the sequence of events across the "
                        "observed window; the clip below covers the full interval with pre- and "
                        "post-roll context."
                    ),
                }
            )
        return json.dumps({"ok": True, "note": "mock backend response"})

    def _summary(self, prompt: str) -> str:
        for kind in _KIND_PHRASES:
            if kind in prompt.lower() and kind != "nominal":
                return _KIND_PHRASES[kind][0]
        return "Observed activity in the monitored area."

    def _recommendations(self, prompt: str, rng: random.Random) -> list[dict[str, Any]]:
        low = prompt.lower()
        table = {
            "unsafe_interaction": [
                ("Hold forklift traffic in this lane and re-brief the operator on pedestrian right-of-way",
                 "Pedestrian and powered vehicle shared the lane without acknowledgement.", "safety_lead", "high", 10),
                ("Add a mirror or proximity alarm at the blind corner",
                 "The interaction recurs at the same corner geometry.", "facilities", "medium", 45),
            ],
            "blocked_zone": [
                ("Relocate the staged pallet to the nearest open floor location",
                 "Keep-clear area is obstructed, which blocks egress.", "floor_supervisor", "high", 5),
                ("Review overflow policy for this dock during shift change",
                 "Obstruction clusters around shift boundaries.", "operations", "low", 30),
            ],
            "congestion": [
                ("Reroute inbound traffic through the parallel aisle until the queue clears",
                 "Aisle throughput has collapsed with several actors present.", "floor_supervisor", "medium", 8),
                ("Stagger replenishment tasks that converge on this aisle",
                 "Congestion correlates with simultaneous replenishment.", "operations", "low", 25),
            ],
            "workflow_anomaly": [
                ("Check the station scanner and confirm the unit is not orphaned",
                 "Dwell time exceeds the station's normal distribution.", "team_lead", "medium", 7),
                ("Flag the upstream feed for starvation review",
                 "Idle station with no arriving work suggests upstream starvation.", "operations", "low", 20),
            ],
        }
        for kind, recs in table.items():
            if kind in low:
                return [
                    {
                        "action": a,
                        "rationale": r,
                        "owner_role": o,
                        "urgency": u,
                        "est_minutes": m,
                    }
                    for a, r, o, u, m in recs
                ]
        return [
            {
                "action": "Log for trend review; no immediate intervention required",
                "rationale": "Evidence does not support an actionable hazard.",
                "owner_role": "operations",
                "urgency": "info",
                "est_minutes": 2,
            }
        ]

    def _answer(self, prompt: str) -> str:
        return (
            "Based on the retrieved events, the matching incidents are listed below with their "
            "clips and recommended actions. The strongest match is shown first; timestamps refer "
            "to the source stream."
        )

    # -------------------------------------------------------------- embeddings
    def embed(self, text: str, dim: int = 256) -> list[float]:
        """Hashed bag-of-words embedding.

        Deterministic and cheap, and -- unlike random vectors -- it actually
        puts lexically similar events near each other, so retrieval behaviour
        in mock mode resembles the NIM-backed path.
        """
        vec = [0.0] * dim
        tokens = [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]
        for tok in tokens:
            h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "big")
            vec[h % dim] += 1.0
            vec[(h >> 17) % dim] += 0.5
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]
