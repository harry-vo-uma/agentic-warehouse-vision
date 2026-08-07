# Evaluation

## What is being measured

A prediction is a true positive when it shares a camera and an event class with
a staged incident and their time intervals overlap by at least `tIoU = 0.15`.
Each ground-truth incident can be claimed once; a second alert for the same
incident is a duplicate and is counted as a false positive, because that is
exactly how it lands on an operator.

Two systems are scored on the same 96 staged incidents:

- **rules only** — every candidate the geometric proposer emits. This is the
  behaviour of a conventional analytics pipeline with no reasoning stage.
- **agentic** — everything the agent graph does not suppress. `LOGGED` events
  count as delivered: they appear in the operator console, so they cost
  attention.

## Reproducing

```bash
make data     # 24 scenarios, 96 staged incidents, 96 staged distractors
make run      # full pipeline
make eval     # scores both systems
make ablate   # threshold sweep + component ablation
```

Runs in the deterministic mock backend by default, so the numbers below
reproduce exactly. Set `NVIDIA_API_KEY` to score against live NIM endpoints;
results will differ because a real VLM's judgement is not the mock's calibrated
approximation of it.

## Results on the bundled benchmark

24 scenarios · 96 staged incidents · 96 staged distractors · mock backend

| system | precision | recall | F1 | FP | FN |
|---|---|---|---|---|---|
| rules only | 0.500 | 1.000 | 0.667 | 96 | 0 |
| **agentic** | **0.838** | **0.917** | **0.876** | **17** | 8 |

- **false positives removed: 82.3%**
- **alert volume reduction: 45.3%** (192 candidates → 105 delivered)
- retrieval top-3 accuracy on NL probes: 1.00
- throughput: 24 scenarios (59k detections, 402 model calls) in 0.87 s

Per class, agentic:

| class | P | R | FP | FN |
|---|---|---|---|---|
| unsafe_interaction | 0.96 | 0.83 | 1 | 4 |
| blocked_zone | 0.92 | 0.83 | 2 | 4 |
| congestion | 0.86 | 1.00 | 4 | 0 |
| workflow_anomaly | 0.69 | 0.92 | 10 | 2 |

`workflow_anomaly` is the weakest class and the honest reason is that "this
unit has been sitting here too long" is genuinely ambiguous without knowledge
of the work order it belongs to. That is a data-integration problem, not a
vision problem, and pretending otherwise by tuning the threshold would just
move the errors into recall.

## Threshold sweep

`suppress_below` against delivered precision/recall:

| suppress_below | P | R | F1 | delivered |
|---|---|---|---|---|
| 0.10 | 0.500 | 1.000 | 0.667 | 192 |
| 0.30 | 0.709 | 0.990 | 0.826 | 134 |
| **0.36** | **0.838** | **0.917** | **0.876** | **105** |
| 0.42 | 0.835 | 0.896 | 0.864 | 103 |
| 0.62 | 0.835 | 0.896 | 0.864 | 103 |
| 0.80 | 0.835 | 0.896 | 0.864 | 103 |

Fused confidence is strongly bimodal, so the curve has a wide plateau above
0.42 — the threshold is not the sensitive knob, which is a useful thing to know
before spending a week tuning it. 0.36 sits just past the knee.

## Component ablation

| variant | P | R | F1 | FP |
|---|---|---|---|---|
| full policy | 0.838 | 0.917 | 0.876 | 17 |
| no vision (geometry only) | 0.727 | 1.000 | 0.842 | 36 |
| no vision veto | 0.500 | 1.000 | 0.667 | 96 |
| no contradiction penalty | 0.500 | 1.000 | 0.667 | 96 |
| no persistence term | 0.838 | 0.917 | 0.876 | 17 |
| no multi-track bonus | 0.838 | 0.917 | 0.876 | 17 |

Read this honestly: **the veto and the contradiction penalty do all the work.**
The persistence and multi-track terms move confidences but never across the
threshold on this benchmark, so on this data they are dead weight. They are
retained because the proposer's persistence gate already filters short events
upstream — remove that gate and the persistence term starts earning its place —
but nobody should claim credit for them based on these numbers.

Note also that "no vision, geometry only" scores 0.727 precision at 1.00 recall
with a fixed 0.55 discount applied. That is a real baseline and it is not far
behind on F1 (0.842 vs 0.876). The case for the VLM here is precision at fixed
recall, and the ability to explain *why* an event was raised — not a dramatic
F1 jump.

## Known limits of this benchmark

1. **Synthetic trajectories.** Actors move along piecewise-linear waypoints with
   Gaussian jitter. Real tracks have occlusions, ID switches, and fragmentation
   that this does not model. Track fragmentation in particular would hurt every
   dwell-based rule.
2. **The mock VLM is not a VLM.** It reads the same geometric features the
   proposer computed and returns a calibrated (88% reliability) verdict. It
   therefore cannot catch anything the geometry cannot express — a real VLM's
   main advantage — and it also cannot hallucinate, which is a real VLM's main
   failure mode. Both directions of error are absent.
3. **Distractors are marginal by construction.** They sit near the decision
   boundary because that is the regime in which a reasoning layer can help. Real
   floors also produce distractors that are geometrically identical to
   incidents, and no amount of policy tuning resolves those.
4. **Class balance is 1:1** incidents to distractors. Real footage is far more
   skewed toward benign, which makes precision harder than reported here.
