#!/usr/bin/env python3
"""Generate labelled synthetic warehouse scenarios.

Each scenario is a scripted set of actor trajectories plus a ground-truth list
of the incidents that were deliberately staged in it. Distractors -- geometry
that *looks* like an incident to the rule layer but is benign -- are staged
too, and they are what the precision numbers in the eval are actually
measuring. Without them any policy scores 1.0.

Usage:
    python scripts/generate_synthetic_scenario.py --out data/scenarios \
        --labels eval/datasets/ground_truth.jsonl --n 24 --seed 1337
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

CAMERAS = ["cam-aisle-01", "cam-dock-02", "cam-pack-03", "cam-recv-04"]


def _line(t0, t1, p0, p1, steps=6):
    return [
        [round(t0 + (t1 - t0) * i / steps, 3),
         round(p0[0] + (p1[0] - p0[0]) * i / steps, 4),
         round(p0[1] + (p1[1] - p0[1]) * i / steps, 4)]
        for i in range(steps + 1)
    ]


def _hold(t0, t1, p, steps=4):
    return [[round(t0 + (t1 - t0) * i / steps, 3), p[0], p[1]] for i in range(steps + 1)]


class Builder:
    def __init__(self, camera_id, rng):
        self.camera_id = camera_id
        self.rng = rng
        self.actors = []
        self.truth = []
        self._next_id = 1

    def _id(self):
        self._next_id += 1
        return self._next_id - 1

    def add(self, label, waypoints, size=None, confidence=0.86):
        self.actors.append({
            "track_id": self._id(),
            "label": label,
            "size": size or ([0.05, 0.13] if label == "person" else [0.11, 0.10]),
            "confidence": confidence,
            "waypoints": waypoints,
        })
        return self.actors[-1]["track_id"]

    def label(self, kind, t0, t1, note=""):
        self.truth.append({
            "camera_id": self.camera_id, "kind": kind,
            "t_start": round(t0, 3), "t_end": round(t1, 3), "note": note,
        })

    # ------------------------------------------------------------- incidents
    def unsafe_interaction(self, t0):
        """Pedestrian crosses in front of a moving forklift. True positive."""
        y = self.rng.uniform(0.35, 0.6)
        self.add("forklift", _line(t0, t0 + 9, (0.08, y), (0.92, y), steps=10), confidence=0.9)
        self.add("person", _line(t0 + 2.5, t0 + 7.0, (0.5, 0.12), (0.5, 0.88), steps=8))
        self.label("unsafe_interaction", t0 + 3.5, t0 + 6.0, "pedestrian crosses moving forklift path")
        return t0 + 12

    def blocked_zone(self, t0, zone_point):
        """Pallet parked in a keep-clear area for a long dwell. True positive."""
        self.add("pallet", _hold(t0, t0 + 24, zone_point, steps=10), size=[0.09, 0.08], confidence=0.82)
        self.label("blocked_zone", t0, t0 + 24, "pallet staged inside keep-clear polygon")
        return t0 + 27

    def congestion(self, t0):
        """Six actors bunched with near-zero net motion. True positive."""
        cx, cy = 0.5, 0.5
        for i in range(6):
            px = cx + self.rng.uniform(-0.07, 0.07)
            py = cy + self.rng.uniform(-0.07, 0.07)
            self.add(
                "person" if i % 3 else "forklift",
                _hold(t0, t0 + 11, (round(px, 4), round(py, 4)), steps=8),
            )
        self.label("congestion", t0, t0 + 11, "six actors stalled in one aisle segment")
        return t0 + 14

    def workflow_anomaly(self, t0):
        """One unit dwells far longer than its peers. True positive."""
        # y=0.48 keeps the station row clear of every keep-clear polygon, so a
        # long dwell here is a workflow anomaly and nothing else.
        for i in range(4):
            self.add("box", _hold(t0 + i * 1.5, t0 + i * 1.5 + 6, (0.2 + 0.12 * i, 0.48)), size=[0.05, 0.05])
        self.add("box", _hold(t0, t0 + 46, (0.86, 0.48)), size=[0.05, 0.05])
        self.label("workflow_anomaly", t0, t0 + 46, "unit dwells ~8x the station median")
        return t0 + 50

    # ------------------------------------------------------------ distractors
    def near_pass_safe(self, t0):
        """Moving forklift, pedestrian walking parallel at the edge of the
        proximity threshold. The rule layer proposes it; the geometry is
        marginal enough that the reasoning layer should decline it. This is the
        distractor that the precision numbers actually turn on.
        """
        y = 0.5
        self.add("forklift", _line(t0, t0 + 10, (0.08, y), (0.92, y), steps=10), confidence=0.9)
        self.add("person", _line(t0 + 1, t0 + 9, (0.12, y - 0.085), (0.88, y - 0.085), steps=8))
        return t0 + 12

    def transient_cluster(self, t0):
        """Group walks through together and disperses. Density without stall."""
        for _ in range(5):
            off = self.rng.uniform(-0.04, 0.04)
            self.add("person", _line(t0, t0 + 5, (0.1, 0.4 + off), (0.9, 0.45 + off), steps=8))
        return t0 + 7

    def brief_zone_touch(self, t0, zone_point):
        """Forklift clips the keep-clear zone in transit. Below dwell threshold."""
        self.add("forklift", _line(t0, t0 + 3, (zone_point[0] - 0.2, zone_point[1]), (zone_point[0] + 0.2, zone_point[1])))
        return t0 + 5

    def drifting_huddle(self, t0):
        """Four people clustered but slowly drifting -- a conversation that is
        moving, not a stalled queue. Density fires; net flow is marginal."""
        cx, cy = 0.45, 0.5
        for _ in range(4):
            off = (self.rng.uniform(-0.05, 0.05), self.rng.uniform(-0.05, 0.05))
            self.add("person", _line(t0, t0 + 9, (cx + off[0], cy + off[1]),
                                     (cx + off[0] + 0.28, cy + off[1] + 0.05), steps=8))
        return t0 + 11

    def zone_handling(self, t0, zone_point):
        """Pallet set down in a keep-clear area and picked up again inside the
        handling window. Above the dwell threshold, below the point where an
        operator would want to hear about it."""
        self.add("pallet", _hold(t0, t0 + 7.5, zone_point, steps=6), size=[0.09, 0.08], confidence=0.8)
        return t0 + 10

    def slow_cart(self, t0):
        """A cart in transit for a long time. Long duration, but it is
        travelling, so it is not a stalled unit."""
        self.add("cart", _line(t0, t0 + 40, (0.1, 0.35), (0.9, 0.35), steps=12), size=[0.07, 0.06])
        return t0 + 43

    def normal_traffic(self, t0):
        for i in range(3):
            self.add("person", _line(t0 + i, t0 + i + 8, (0.05, 0.2 + 0.25 * i), (0.95, 0.25 + 0.25 * i), steps=8))
        return t0 + 11


def build_scenario(camera_id: str, seed: int, zone_point) -> tuple[dict, list[dict]]:
    rng = random.Random(seed)
    b = Builder(camera_id, rng)
    t = 1.0
    # Roughly 40% staged incidents, 60% benign/distractor traffic -- close to
    # the class balance in the annotated slice of the source corpus.
    plan = [
        # staged incidents (ground truth)
        "unsafe", "blocked", "congestion", "anomaly",
        # distractors: each one trips exactly one rule and should not survive
        "safe_pass", "zone_handling", "huddle", "slow_cart", "cluster", "touch",
        # benign background
        "normal",
    ]
    rng.shuffle(plan)
    for step in plan:
        if step == "unsafe":
            t = b.unsafe_interaction(t)
        elif step == "blocked":
            t = b.blocked_zone(t, zone_point)
        elif step == "congestion":
            t = b.congestion(t)
        elif step == "anomaly":
            t = b.workflow_anomaly(t)
        elif step == "safe_pass":
            t = b.near_pass_safe(t)
        elif step == "cluster":
            t = b.transient_cluster(t)
        elif step == "touch":
            t = b.brief_zone_touch(t, zone_point)
        elif step == "zone_handling":
            t = b.zone_handling(t, zone_point)
        elif step == "huddle":
            t = b.drifting_huddle(t)
        elif step == "slow_cart":
            t = b.slow_cart(t)
        else:
            t = b.normal_traffic(t)
    return (
        {"camera_id": camera_id, "fps": 10, "duration_s": round(t, 2), "actors": b.actors},
        b.truth,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/scenarios")
    ap.add_argument("--labels", default="eval/datasets/ground_truth.jsonl")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    labels_path = Path(args.labels)
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    zone_points = {
        "cam-aisle-01": (0.30, 0.30),
        "cam-dock-02": (0.70, 0.30),
        "cam-pack-03": (0.30, 0.70),
        "cam-recv-04": (0.70, 0.70),
    }

    all_truth = []
    for i in range(args.n):
        cam = CAMERAS[i % len(CAMERAS)]
        scenario, truth = build_scenario(cam, args.seed + i, zone_points[cam])
        name = f"{cam}-{i:03d}"
        scenario["scenario_id"] = name
        (out / f"{name}.json").write_text(json.dumps(scenario, indent=1))
        for t in truth:
            t["scenario_id"] = name
        all_truth += truth

    with labels_path.open("w") as fh:
        for t in all_truth:
            fh.write(json.dumps(t) + "\n")

    print(f"wrote {args.n} scenarios to {out}")
    print(f"wrote {len(all_truth)} ground-truth incidents to {labels_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
