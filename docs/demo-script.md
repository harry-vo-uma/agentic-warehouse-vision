# Demo script — 6 minutes

Rehearsed for a booth or a customer call. Every command runs on a laptop with
no GPU and no API key.

## Setup (before anyone is watching)

```bash
pip install -e ".[dev]"
make data && make run
python -m awvi.cli serve --port 8000
```

## 0:00 — The problem (45 s, no slides)

> "This customer has 40 cameras and a working object detector. It produces
> about four thousand alerts a day. Their floor supervisor reads none of them —
> and I don't blame him, because roughly one in a hundred is real. So the
> question isn't detection. It's triage."

## 0:45 — Show the raw firehose (45 s)

```bash
python eval/run_eval.py | head -6
```

Point at the `rules only` row: precision 0.500 at recall 1.000.

> "That's a conventional analytics pipeline. It finds everything. Half of what
> it hands you is noise."

## 1:30 — Show the pipeline running (60 s)

```bash
python -m awvi.cli run --scenarios data/scenarios --out data/events.json
```

While it scrolls, name the three agents: perception builds the evidence packet
and picks keyframes; investigation asks a VLM to *refute* the hypothesis, not
confirm it; recommendation fuses, escalates, and writes the action.

Point at a `suppressed` line.

> "That one is a person walking past a parked forklift. Geometry says they were
> nine centimetres apart. The forklift hasn't moved in ten seconds — so it isn't
> an incident, and nobody gets paged."

## 2:30 — The operator console (90 s)

Open `http://localhost:8000`. Click the chip **"any near misses with forklifts?"**

Let the answer render, then click the top event. Walk the detail pane top to
bottom: severity and fused confidence, the keyframes the model actually looked
at, the narrative, the contradicting evidence it weighed and overruled, and the
recommended action with an owner and a time estimate.

> "Twenty-two minutes of scrubbing footage becomes about ninety seconds of
> reading. And the operator can see exactly why the system thinks this is real,
> which is the difference between a tool people use and a tool people mute."

## 4:00 — The receipts (90 s)

```bash
python eval/ablations.py
```

> "82% of the false positives are gone, at 92% recall. And here's the part I'd
> want to know if I were you: the ablation says the persistence and multi-track
> terms do nothing on this benchmark. The vision veto and the contradiction
> penalty do all the work. I'd rather tell you that than show you six knobs and
> imply they all matter."

## 5:30 — What I'd build next (30 s)

> "The weakest class is workflow anomaly, at 0.69 precision, and it's weak for a
> reason a bigger model won't fix: 'this pallet has been here too long' is
> unanswerable without the work order. The next version reads WMS task state as
> a tool call inside the investigation agent. That's a product ask, not a
> modelling one."

## If asked "does this need a GPU?"

No — the whole thing runs on the deterministic mock backend. Set
`NVIDIA_API_KEY` and it routes to real VILA and Nemotron NIMs; set
`AWVI_PERCEPTION_BACKEND=deepstream` and the same code reads RTSP through
nvtracker. Same interfaces, three backends.
