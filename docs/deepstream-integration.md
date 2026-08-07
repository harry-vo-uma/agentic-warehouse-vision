# DeepStream integration notes

`SyntheticSource` and `DeepStreamSource` implement the same `FrameSource`
protocol, so everything above the tracker is identical in both modes. This
document covers only the real path.

## Requirements

- DeepStream 7.x, GStreamer 1.20+, `pyds` bindings on the Python path
- An NVIDIA GPU with a driver matching the DeepStream release
- A primary detector engine (PeopleNet, or a custom model for forklifts/pallets)

`DeepStreamSource.available()` checks for `gi` and `pyds` and is what
`backend: auto` uses to decide. Nothing DeepStream-related is imported at
module scope — the repo has to be importable on a laptop.

## Pipeline

```
uridecodebin -> nvstreammux -> nvinfer(pgie) -> nvtracker -> nvdsanalytics -> fakesink
```

The probe sits on the **tracker's src pad**, not the inference element's, so
every object already carries a stable `object_id`.

## Things that cost time when porting this

**Batch size must equal the source count.** `nvstreammux`'s `batch-size` is not
advisory. Set it to 1 with four cameras attached and the tracker will quietly
drop the tail streams — no error, no warning, just missing cameras.

**Normalise the bounding boxes.** `NvOSD_RectParams` are in the *muxer's*
output resolution, not the source's. Every threshold in this repo
(`proximity_threshold`, `congestion_radius`, zone polygons) is in normalised
`[0,1]` coordinates so that swapping a 1080p camera for a 4K one does not
silently change what counts as "close".

**Do no work in the probe.** The probe runs on the GStreamer streaming thread.
Anything expensive there stalls the pipeline and you will see frame drops
attributed to the decoder. Extract meta, push to a queue, reason elsewhere.

**`batched-push-timeout` matters for live sources.** With RTSP inputs and no
timeout, the muxer will wait indefinitely for a batch that never fills when one
camera drops. 40 ms is a reasonable default for 30 fps.

**Tracker choice.** NvDCF gives stable IDs through short occlusions, which the
dwell and trajectory features depend on. IOU-only tracking fragments tracks
behind rack columns, and a fragmented track resets `first_seen`, which destroys
every dwell-based rule. If you must use IOU, raise
`min_temporal_persistence_s` to compensate.

## Zones

`configs/zones.yaml` holds normalised polygons. `nvdsanalytics` also needs ROI
definitions in pixel space in `configs/deepstream/nvdsanalytics.txt`. Keeping
both is redundant but deliberate: nvdsanalytics gives per-frame ROI flags
cheaply on-GPU, while the Python polygons drive the temporal dwell logic that
nvdsanalytics does not model.

## Running against real streams

```bash
export AWVI_PERCEPTION_BACKEND=deepstream
export NVIDIA_API_KEY=nvapi-...
python -m awvi.cli run --scenarios /dev/null   # cameras come from configs/cameras.yaml
```
