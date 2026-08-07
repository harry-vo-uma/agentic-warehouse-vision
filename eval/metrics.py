"""Matching and metrics for temporal event detection.

A predicted event counts as a true positive when it shares a camera and a kind
with a ground-truth incident and their time intervals overlap by at least
`min_tiou`. Each ground-truth incident can be matched once; extra predictions
over the same incident are duplicates, not additional true positives, because
that is exactly how they land on an operator.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


def temporal_iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


@dataclass
class MatchResult:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    duplicates: int = 0
    matched_pairs: list[tuple[str, int]] = field(default_factory=list)
    fp_by_kind: dict[str, int] = field(default_factory=dict)
    fn_by_kind: dict[str, int] = field(default_factory=dict)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "duplicates": self.duplicates,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "fp_by_kind": self.fp_by_kind,
            "fn_by_kind": self.fn_by_kind,
        }


def match(
    predictions: Iterable[dict],
    truth: Iterable[dict],
    min_tiou: float = 0.15,
) -> MatchResult:
    preds = sorted(predictions, key=lambda p: -float(p.get("confidence", 0.0)))
    gts = list(truth)
    used: set[int] = set()
    res = MatchResult()

    for p in preds:
        best_i, best_iou = -1, 0.0
        for i, g in enumerate(gts):
            if g["camera_id"] != p["camera_id"] or g["kind"] != p["kind"]:
                continue
            if p.get("scenario_id") and g.get("scenario_id") and p["scenario_id"] != g["scenario_id"]:
                continue
            iou = temporal_iou(p["t_start"], p["t_end"], g["t_start"], g["t_end"])
            if iou >= min_tiou and iou > best_iou:
                best_i, best_iou = i, iou
        if best_i < 0:
            res.fp += 1
            res.fp_by_kind[p["kind"]] = res.fp_by_kind.get(p["kind"], 0) + 1
        elif best_i in used:
            res.duplicates += 1
            res.fp += 1
            res.fp_by_kind[p["kind"]] = res.fp_by_kind.get(p["kind"], 0) + 1
        else:
            used.add(best_i)
            res.tp += 1
            res.matched_pairs.append((p.get("event_id", ""), best_i))

    for i, g in enumerate(gts):
        if i not in used:
            res.fn += 1
            res.fn_by_kind[g["kind"]] = res.fn_by_kind.get(g["kind"], 0) + 1
    return res


def per_kind(predictions: list[dict], truth: list[dict], min_tiou: float = 0.15) -> dict[str, dict]:
    kinds = sorted({g["kind"] for g in truth} | {p["kind"] for p in predictions})
    return {
        k: match([p for p in predictions if p["kind"] == k],
                 [g for g in truth if g["kind"] == k],
                 min_tiou).as_dict()
        for k in kinds
    }
