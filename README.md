# Agentic Warehouse Vision Intelligence

A multi-camera Vision AI reference application that pairs **NVIDIA DeepStream** tracking with
**VLM scene reasoning** on **NVIDIA NIM**, orchestrated as a small agent graph
(**NVIDIA Agent Toolkit / LangGraph**), and exposes the result as an operator console you can
ask questions in plain English.

The problem it addresses: a frame-level detector that fires on every rule match buries the
operator. This pipeline keeps the detector's recall but adds a vision-reasoning stage and an
explicit escalation policy on top, so what reaches the console is smaller and mostly correct.

**It runs on a laptop.** No GPU, no DeepStream install, and no API key are required — every
external dependency has a deterministic fallback. See [Running without a GPU](#running-without-a-gpu).

---

## Results

Reproduce everything below with `make data && make eval`. Numbers are from the committed
synthetic benchmark (24 scenarios, 96 staged incidents, 59,040 detections).

| Pipeline | Precision | Recall | F1 | FP | FN |
|---|---|---|---|---|---|
| Rules only (frame-level geometry) | 0.500 | 1.000 | 0.667 | 96 | 0 |
| **Agentic (VLM + escalation policy)** | **0.838** | **0.917** | **0.876** | 17 | 8 |

- **False positives removed: 82.3%** (96 → 17)
- **Alert volume reduction: 45.3%** (192 raw candidates → 105 delivered, 87 suppressed)
- **Retrieval top-3 accuracy: 1.0** on the natural-language query probes
- **Throughput:** 24 scenarios in 0.87 s (27.6 scenarios/s, 402 model calls) in mock mode

The recall cost is real and reported: the policy trades 8 incidents to remove 79 false alerts.
[`docs/evaluation.md`](docs/evaluation.md) has the threshold sweep, the component ablation, and
an explicit *Known limits of this benchmark* section — including the two policy terms that
measurably do nothing here.

---

## How it works

```
 RTSP / file sources
        │
        ▼
 ┌──────────────────────┐   DeepStream: nvstreammux → nvinfer → nvtracker (NvDCF)
 │  Perception           │   → nvdsanalytics, probe on tracker src pad
 │  (DeepStream or       │   Fallback: SyntheticSource replays recorded detections
 │   SyntheticSource)    │
 └──────────┬───────────┘
            │  Tracks (id, class, trajectory, zone dwell)
            ▼
 ┌──────────────────────┐   High-recall geometry. Four rules: congestion,
 │  CandidateProposer    │   blocked_zone, unsafe_interaction, workflow_anomaly.
 │                       │   Cross-rule arbitration; deliberately over-fires.
 └──────────┬───────────┘
            │  EventCandidate
            ▼
 ┌──────────────────────┐   LangGraph StateGraph (built-in runner if LangGraph absent)
 │  Agent graph          │
 │  perception →         │   investigation: VLM is asked to argue *both* sides and
 │  investigation →      │   return strict JSON. On parse failure → finding = None,
 │  [gate] → clip →      │   which never confirms an event on geometry alone.
 │  recommendation       │
 └──────────┬───────────┘
            │  VLMFinding (supports / refutes + confidence + contradicting evidence)
            ▼
 ┌──────────────────────┐   score_event() — a pure function, so the whole policy
 │  Escalation policy    │   can be swept and ablated cheaply.
 │                       │   SUPPRESSED / LOGGED / ESCALATED / PAGED
 └──────────┬───────────┘
            ▼
   Hybrid index + FastAPI console (dense NIM embeddings + lexical + parsed filters)
```

Two design decisions did most of the work:

**Net speed, not path speed.** Detector jitter inflates path-length speed, so a parked forklift
reads as moving and a stalled crowd reads as flowing. Gating on net displacement per second
instead moved rules-only precision from 0.283 to 0.800 by itself.

**Symmetric fusion.** A refutation held at confidence *c* is evidence for the event at strength
*1 − c*, so disagreement is scored `0.38·geometry + 0.62·(1 − c)` rather than clamped to zero.
Clamping made the policy threshold-invariant and hid what the veto actually costs.

More in [`docs/architecture.md`](docs/architecture.md).

---

## Quickstart

```bash
git clone https://github.com/harry-vo-uma/agentic-warehouse-vision
cd agentic-warehouse-vision

make install      # pip install -e .
make data         # generate the 24-scenario synthetic benchmark
make eval         # reproduce the results table above
make serve        # operator console at http://localhost:8000
```

Then ask the console things like *"show me blocked aisle events on the dock camera"* or
*"what happened near packing station 3 this morning"*.

Other targets: `make ablate` (component ablation + threshold sweep), `make test` (65 tests),
`make lint` (ruff), `make demo` (the rehearsed walkthrough in [`docs/demo-script.md`](docs/demo-script.md)).

---

## Running without a GPU

Every external dependency degrades rather than crashes:

| Missing | Behaviour |
|---|---|
| DeepStream / `pyds` | `SyntheticSource` replays recorded detection streams |
| `NVIDIA_API_KEY` | `MockNIMBackend` — seeded, deterministic, no network |
| `langgraph` | Behaviour-identical built-in graph runner |
| PyAV | Keyframes become placeholder JPEGs |
| ffmpeg | Clips become manifests instead of video files |

This is why `make eval` is reproducible: in mock mode the pipeline is bit-for-bit
deterministic (`test_pipeline_is_deterministic_in_mock_mode`).

## Running against real NVIDIA services

```bash
cp .env.example .env
# NVIDIA_API_KEY=nvapi-...
# AWVI_VLM_MODEL=nvidia/vila
# AWVI_LLM_MODEL=nvidia/llama-3.1-nemotron-70b-instruct
# AWVI_EMBED_MODEL=nvidia/nv-embedqa-e5-v5
```

The NIM client speaks the OpenAI-compatible API, so `AWVI_NIM_BASE_URL` points at either
`https://integrate.api.nvidia.com/v1` or a locally hosted NIM container with no code change.

For real video, install DeepStream 7.x and set `AWVI_SOURCE=deepstream` with your RTSP URIs in
`configs/cameras.yaml`. [`docs/deepstream-integration.md`](docs/deepstream-integration.md) covers
the porting gotchas — batch size must equal source count, normalise bboxes before they leave the
probe, do no work in the probe itself, and why NvDCF beats the IOU tracker here.

---

## Layout

```
src/awvi/
  schemas.py            pydantic contracts shared by every stage
  config.py             env-driven settings incl. escalation thresholds
  perception/           tracks, zones, candidate proposal
  nim/                  NIM client + deterministic mock backend
  agents/               investigation, recommendation, escalation, graph
  index/                hybrid retrieval (dense + lexical + structured filters)
  api/                  FastAPI server + operator console SPA
eval/                   run_eval.py, ablations.py, datasets/
scripts/                synthetic scenario generator
configs/                zones, pipeline, cameras
docs/                   architecture, DeepStream, evaluation, demo script
```

## Tuning the policy

The thresholds are environment variables, so a sweep needs no code edit:

```bash
AWVI_SUPPRESS_BELOW=0.36 AWVI_ESCALATE_ABOVE=0.68 AWVI_PAGE_ABOVE=0.86 make eval
```

`0.36` is the default because the sweep in `docs/evaluation.md` puts the F1 optimum there — it
was moved down from `0.42` once the fusion was made symmetric.

## License

Apache-2.0. See [LICENSE](LICENSE).
