# Architecture

## The problem this system exists to solve

A warehouse with 40 cameras and a competent object detector produces thousands
of alerts a day. Almost none of them are incidents. An operator who is paged
for every frame in which a person and a forklift share a bounding-box
neighbourhood stops reading the pages within a week, and the system becomes
decoration.

The interesting engineering problem is therefore not detection. It is
**deciding which detections are worth a human's attention, and giving that
human enough context to act in seconds rather than minutes.**

## Pipeline

```
  RTSP / file
       |
       v
+---------------------+     DeepStream 7.x
|  nvstreammux        |     nvinfer -> nvtracker(NvDCF) -> nvdsanalytics
|  nvinfer            |     probe on tracker src pad
|  nvtracker          |     ==> Detection{track_id,label,bbox,t}
|  nvdsanalytics      |
+---------------------+
       |
       v
+---------------------+     Trajectories, not frames.
|  TrackStore         |     net_speed, dwell_ratio, min_separation,
|                     |     positions_at(t), density
+---------------------+
       |
       v
+---------------------+     HIGH RECALL, LOW PRECISION on purpose.
|  CandidateProposer  |     4 geometric rules + cross-rule arbitration
|                     |     ==> EventCandidate{kind, window, features, strength}
+---------------------+
       |
       v
+========================================================+
|                    AGENT GRAPH                          |
|                                                         |
|  perception ---> investigation ---+--> clip --+          |
|   (evidence      (VLM confirms    |           |          |
|    packet,        or refutes)     |           v          |
|    keyframes)                     +----> recommendation  |
|                                    (fuse, escalate,      |
|                                     summarise, advise)   |
+========================================================+
       |
       v
+---------------------+     Hybrid dense + BM25 + structured filters
|  EventIndex         |     "any near misses on the dock?"
+---------------------+
       |
       v
   FastAPI + operator console
```

## Why each layer is shaped the way it is

**The proposer is deliberately bad at precision.** It runs on every frame of
every camera, so it has to be cheap, which means geometry. Geometry cannot
distinguish a forklift reversing carefully into a bay from one about to hit
someone. Tightening its thresholds trades away more recall than the reasoning
layer costs in precision — the sweep in `eval/ablations.py` is the receipt.

**Trajectories, not frames.** Frame-level detection is the root cause of the
false-positive problem. `TrackStore` exposes `net_speed` (displacement over
time) separately from `mean_speed` (path length over time) specifically because
detector jitter makes a stationary pallet read as moving under path length.
Gating unsafe-interaction candidates on *net* vehicle motion removed the single
largest false-positive class in the benchmark, taking rules-only precision from
0.12 to 0.50 on the `unsafe_interaction` class alone.

**The VLM is asked to disagree.** The investigation prompt requires the model
to list contradicting evidence, not just confirm. Without that instruction the
model agrees with whatever the rule layer proposed, and the second stage
becomes an expensive no-op. `contradicting_evidence` is the highest-signal
field the VLM returns — zeroing its weight in the ablation collapses precision
from 0.84 to 0.50.

**Clip extraction sits behind a gate.** It is the most expensive I/O in the
system and most candidates are suppressed, so the graph routes around it via a
conditional edge that reuses the exact fusion function the terminal node will
run. Using a *different*, cheaper heuristic at the gate would let the gate and
the policy disagree, which produces events with no clip and clips with no
event.

**The escalation policy is a pure function.** `score_event(candidate, finding,
settings)` has no I/O and no state, which is what makes a 10-point threshold
sweep across 192 candidates take under a second: perception and vision run
once, and only the policy is re-evaluated. De-duplication is the one stateful
part and it lives in a separate wrapper class.

## Failure behaviour

Every external dependency is optional and degrades rather than crashes:

| Missing | Behaviour |
|---|---|
| DeepStream / pyds | `SyntheticSource` replays scripted scenarios through the same interface |
| `NVIDIA_API_KEY` | `MockNIMBackend` — seeded, deterministic, reproducible evals |
| LangGraph | built-in runner with identical topology and identical results |
| PyAV / source media | deterministic placeholder keyframes, real JPEGs |
| ffmpeg | clip manifests written, `materialised: false` |

A failed VLM call sets `finding = None`, which the fusion function treats as
"geometry only, heavily discounted" — never as confirmation. That is tested
(`test_vision_failure_does_not_confirm_an_event`); a vision outage must not
turn into a wave of unfiltered alerts.

## Extension points

- **Real vector store.** `EventIndex` is three methods wide (`add_many`,
  `search`, `answer`). Swap `_vectors` for Milvus or pgvector without touching
  callers.
- **New event class.** Add a rule to `CandidateProposer`, a phrase entry in the
  mock, and a row in `_BASE_SEVERITY`. The agent graph needs no changes.
- **Cross-camera association.** `TrackStore` is per-camera by design. Re-ID
  across cameras belongs in a store that wraps several, sharing the same
  `Detection` schema.
