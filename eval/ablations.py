#!/usr/bin/env python3
"""Ablation sweep over the escalation policy.

Answers three questions that came up repeatedly in review:
  1. How much of the precision gain comes from the VLM versus from the temporal
     persistence rule alone?
  2. Where should `suppress_below` sit on the precision/recall curve?
  3. Does the vision veto pay for itself, or is it just discarding recall?

Usage:  python eval/ablations.py --scenarios data/scenarios --truth eval/datasets/ground_truth.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import match  # noqa: E402

from awvi.agents.escalation import score_event  # noqa: E402
from awvi.agents.investigation_agent import InvestigationAgent  # noqa: E402
from awvi.agents.perception_agent import PerceptionAgent  # noqa: E402
from awvi.agents.state import GraphState  # noqa: E402
from awvi.config import EscalationSettings  # noqa: E402
from awvi.perception.deepstream import SyntheticSource  # noqa: E402
from awvi.perception.events import CandidateProposer  # noqa: E402
from awvi.perception.tracks import TrackStore  # noqa: E402
from awvi.pipeline import load_zones  # noqa: E402

SUPPRESSED = "suppressed"


def collect(scenarios_dir: Path):
    """Run perception + investigation once; reuse for every policy variant.

    The expensive part is the vision call, and the policy is a pure function of
    its output, so there is no reason to pay for it 30 times.
    """
    proposer = CandidateProposer(load_zones())
    rows = []
    for path in sorted(scenarios_dir.glob("*.json")):
        scenario = json.loads(path.read_text())
        sid = scenario.get("scenario_id", path.stem)
        store = TrackStore(scenario["camera_id"])
        store.update(list(SyntheticSource(scenario).stream()))
        perception = PerceptionAgent(store)
        investigation = InvestigationAgent()
        for c in proposer.propose(store):
            st = investigation(perception(GraphState(candidate=c)))
            rows.append((sid, c, st.finding))
    return rows


def evaluate(rows, truth, settings: EscalationSettings, use_vlm: bool = True) -> dict:
    preds = []
    for sid, c, finding in rows:
        f = finding if use_vlm else None
        r = score_event(c, f, settings)
        if r.disposition.value != SUPPRESSED:
            preds.append({
                "event_id": c.candidate_id, "scenario_id": sid, "camera_id": c.camera_id,
                "kind": c.kind.value, "t_start": c.t_start, "t_end": c.t_end,
                "confidence": r.confidence,
            })
    m = match(preds, truth)
    d = m.as_dict()
    d["delivered"] = len(preds)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="data/scenarios")
    ap.add_argument("--truth", default="eval/datasets/ground_truth.jsonl")
    ap.add_argument("--out", default="eval/ablations.json")
    args = ap.parse_args()

    truth = [json.loads(line) for line in Path(args.truth).read_text().splitlines() if line.strip()]
    rows = collect(Path(args.scenarios))
    print(f"collected {len(rows)} candidates with vision findings\n")

    results: dict = {"threshold_sweep": [], "component_ablation": {}}

    for thr in [0.10, 0.20, 0.30, 0.36, 0.42, 0.48, 0.54, 0.62, 0.70, 0.80]:
        s = EscalationSettings()
        s.suppress_below = thr
        s.escalate_above = max(thr + 0.05, 0.68)
        row = evaluate(rows, truth, s)
        row["suppress_below"] = thr
        results["threshold_sweep"].append(row)
        print(f"suppress_below={thr:<5}  P={row['precision']:.3f}  R={row['recall']:.3f}  "
              f"F1={row['f1']:.3f}  delivered={row['delivered']:<4} fp={row['fp']}")

    print()
    variants = {
        "full_policy": (EscalationSettings(), True),
        "no_vlm_geometry_only": (EscalationSettings(), False),
        "no_vision_veto": (_mut(require_vlm_agreement=False), True),
        "no_persistence_term": (_mut(persistence_bonus=0.0), True),
        "no_contradiction_penalty": (_mut(contradiction_penalty=0.0), True),
        "no_multi_track_bonus": (_mut(multi_track_bonus=0.0), True),
    }
    for name, (s, use_vlm) in variants.items():
        row = evaluate(rows, truth, s, use_vlm=use_vlm)
        results["component_ablation"][name] = row
        print(f"{name:28} P={row['precision']:.3f}  R={row['recall']:.3f}  "
              f"F1={row['f1']:.3f}  fp={row['fp']:<4} fn={row['fn']}")

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")
    return 0


def _mut(**kw) -> EscalationSettings:
    s = EscalationSettings()
    for k, v in kw.items():
        setattr(s, k, v)
    return s


if __name__ == "__main__":
    raise SystemExit(main())
