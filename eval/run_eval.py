#!/usr/bin/env python3
"""End-to-end evaluation of the pipeline against staged ground truth.

Reports two things that are easy to conflate and shouldn't be:

  1. Detection quality of what actually reaches an operator (dispositions
     ESCALATED or PAGED), which is the number an operator experiences.
  2. Detection quality of the raw rule layer with no reasoning at all, which
     is the baseline the agent graph has to beat.

The delta between those two is the entire argument for the system.

Usage:
    python eval/run_eval.py --scenarios data/scenarios \
        --truth eval/datasets/ground_truth.jsonl --out eval/results.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import match, per_kind  # noqa: E402

from awvi.agents.graph import IncidentGraph  # noqa: E402
from awvi.config import get_settings  # noqa: E402
from awvi.index.store import EventIndex  # noqa: E402
from awvi.nim.client import get_client  # noqa: E402
from awvi.perception.deepstream import SyntheticSource  # noqa: E402
from awvi.perception.events import CandidateProposer  # noqa: E402
from awvi.perception.tracks import TrackStore  # noqa: E402
from awvi.pipeline import load_zones  # noqa: E402

#: Anything not suppressed reaches the operator console. SUPPRESSED is the
#: only disposition that is filtered out entirely, so it is the only one that
#: should count as "not delivered".
SUPPRESSED = "suppressed"


def load_truth(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run(scenarios_dir: Path, truth_path: Path) -> dict:
    zones = load_zones()
    proposer = CandidateProposer(zones)
    truth = load_truth(truth_path)

    raw_preds, all_events = [], []
    t0 = time.perf_counter()

    for path in sorted(scenarios_dir.glob("*.json")):
        scenario = json.loads(path.read_text())
        sid = scenario.get("scenario_id", path.stem)
        store = TrackStore(scenario["camera_id"])
        store.update(list(SyntheticSource(scenario).stream()))
        candidates = proposer.propose(store)

        for c in candidates:
            raw_preds.append({
                "event_id": c.candidate_id, "scenario_id": sid,
                "camera_id": c.camera_id, "kind": c.kind.value,
                "t_start": c.t_start, "t_end": c.t_end,
                "confidence": c.signal_strength,
            })

        for e in IncidentGraph(store).run(candidates):
            all_events.append({
                "event_id": e.event_id, "scenario_id": sid,
                "camera_id": e.camera_id, "kind": e.kind.value,
                "t_start": e.t_start, "t_end": e.t_end,
                "confidence": e.confidence, "disposition": e.disposition.value,
                "severity": e.severity.value, "summary": e.summary,
            })

    wall = time.perf_counter() - t0
    delivered = [e for e in all_events if e["disposition"] != SUPPRESSED]

    baseline = match(raw_preds, truth)
    agentic = match(delivered, truth)

    fp_reduction = (
        (baseline.fp - agentic.fp) / baseline.fp if baseline.fp else 0.0
    )

    return {
        "config": {
            "scenarios": len(list(scenarios_dir.glob("*.json"))),
            "ground_truth_incidents": len(truth),
            "nim_mode": get_client().stats()["mode"],
            "escalation": get_settings().escalation.__dict__,
        },
        "baseline_rules_only": baseline.as_dict(),
        "agentic_pipeline": agentic.as_dict(),
        "deltas": {
            "precision_gain": round(agentic.precision - baseline.precision, 4),
            "recall_change": round(agentic.recall - baseline.recall, 4),
            "f1_gain": round(agentic.f1 - baseline.f1, 4),
            "false_positive_reduction": round(fp_reduction, 4),
        },
        "per_kind_agentic": per_kind(delivered, truth),
        "per_kind_baseline": per_kind(raw_preds, truth),
        "volume": {
            "raw_candidates": len(raw_preds),
            "events_produced": len(all_events),
            "delivered_to_operator": len(delivered),
            "suppressed": sum(1 for e in all_events if e["disposition"] == "suppressed"),
            "alert_volume_reduction": round(1 - len(delivered) / len(raw_preds), 4) if raw_preds else 0.0,
        },
        "throughput": {
            "wall_s": round(wall, 2),
            "scenarios_per_s": round(len(list(scenarios_dir.glob("*.json"))) / wall, 2) if wall else 0.0,
            "nim": get_client().stats(),
        },
    }


def retrieval_check(events_path: Path | None) -> dict:
    """Sanity check that the NL index surfaces the right event for a question."""
    if events_path is None or not events_path.exists():
        return {"skipped": "no events file"}
    from awvi.schemas import Event

    idx = EventIndex()
    idx.add_many([Event.model_validate(e) for e in json.loads(events_path.read_text())])
    probes = {
        "any near misses with a forklift?": "unsafe_interaction",
        "what is blocking a keep clear zone?": "blocked_zone",
        "show me congestion in the aisle": "congestion",
        "anything stuck or idle too long?": "workflow_anomaly",
    }
    hits, total = 0, 0
    detail = {}
    for q, expected in probes.items():
        res = idx.search(q, top_k=3)
        ok = any(h.event.kind.value == expected for h in res)
        detail[q] = {"expected": expected, "top3": [h.event.kind.value for h in res], "hit": ok}
        hits += int(ok)
        total += 1
    return {"top3_accuracy": round(hits / total, 4) if total else 0.0, "probes": detail}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="data/scenarios")
    ap.add_argument("--truth", default="eval/datasets/ground_truth.jsonl")
    ap.add_argument("--events", default="data/events.json")
    ap.add_argument("--out", default="eval/results.json")
    args = ap.parse_args()

    results = run(Path(args.scenarios), Path(args.truth))
    results["retrieval"] = retrieval_check(Path(args.events))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    b, a = results["baseline_rules_only"], results["agentic_pipeline"]
    print(f"scenarios={results['config']['scenarios']}  truth={results['config']['ground_truth_incidents']}  "
          f"mode={results['config']['nim_mode']}")
    print(f"{'':16}{'precision':>10}{'recall':>10}{'f1':>10}{'fp':>7}{'fn':>7}")
    print(f"{'rules only':16}{b['precision']:>10.3f}{b['recall']:>10.3f}{b['f1']:>10.3f}{b['fp']:>7}{b['fn']:>7}")
    print(f"{'agentic':16}{a['precision']:>10.3f}{a['recall']:>10.3f}{a['f1']:>10.3f}{a['fp']:>7}{a['fn']:>7}")
    print(f"\nfalse positives removed: {results['deltas']['false_positive_reduction']*100:.1f}%")
    print(f"alert volume reduction:  {results['volume']['alert_volume_reduction']*100:.1f}%")
    print(f"retrieval top-3 accuracy: {results['retrieval'].get('top3_accuracy', 'n/a')}")
    print(f"\nfull results -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
