"""DeepStream backend + a synthetic backend with the same interface.

The DeepStream path builds a standard `nvstreammux -> nvinfer -> nvtracker ->
nvdsanalytics` pipeline and converts NvDsObjectMeta into our `Detection`
schema via a probe on the tracker's src pad. It requires DeepStream 7.x and
the pyds bindings, so it is imported lazily and never at module scope --
otherwise the whole repo becomes un-runnable on any machine without a GPU.

`SyntheticSource` replays a scripted scenario through the identical interface,
which is what the tests, the eval harness and the laptop demo use.
"""
from __future__ import annotations

import json
import logging
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from ..config import PerceptionSettings, get_settings
from ..schemas import BBox, Detection

log = logging.getLogger(__name__)


class FrameSource(Protocol):
    """Anything that can yield tracked detections in timestamp order."""

    camera_id: str

    def stream(self) -> Iterator[Detection]: ...


# --------------------------------------------------------------------- DeepStream
class DeepStreamSource:
    """Real multi-object tracking over an RTSP/file input.

    Notes for anyone porting this:
      * `nvstreammux` batch size must equal the number of sources or the
        tracker silently drops the tail cameras.
      * Bounding boxes come back in *stream* pixel space; we normalise against
        the muxer output resolution so every downstream threshold is
        resolution-independent.
      * The probe runs on the pipeline thread. Do no work in it beyond meta
        extraction -- push to a queue and reason elsewhere.
    """

    def __init__(self, camera_id: str, uri: str, settings: PerceptionSettings | None = None):
        self.camera_id = camera_id
        self.uri = uri
        self.settings = settings or get_settings().perception
        self._width = 1920
        self._height = 1080

    @staticmethod
    def available() -> bool:
        try:
            import gi  # noqa: F401
            import pyds  # noqa: F401
            return True
        except Exception:
            return False

    def build_pipeline(self):  # pragma: no cover - requires DeepStream runtime
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)
        pipeline = Gst.Pipeline.new(f"awvi-{self.camera_id}")

        def make(factory: str, name: str):
            el = Gst.ElementFactory.make(factory, name)
            if not el:
                raise RuntimeError(f"failed to create GStreamer element {factory}")
            pipeline.add(el)
            return el

        src = make("uridecodebin", "src")
        src.set_property("uri", self.uri)
        mux = make("nvstreammux", "mux")
        mux.set_property("batch-size", 1)
        mux.set_property("width", self._width)
        mux.set_property("height", self._height)
        mux.set_property("live-source", 1 if self.uri.startswith("rtsp") else 0)
        mux.set_property("batched-push-timeout", 40000)

        pgie = make("nvinfer", "pgie")
        pgie.set_property("config-file-path", self.settings.detector_config)

        tracker = make("nvtracker", "tracker")
        tracker.set_property("ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
        tracker.set_property("ll-config-file", f"configs/deepstream/tracker_{self.settings.tracker}.yml")
        tracker.set_property("tracker-width", 960)
        tracker.set_property("tracker-height", 544)

        analytics = make("nvdsanalytics", "analytics")
        analytics.set_property("config-file", "configs/deepstream/nvdsanalytics.txt")
        sink = make("fakesink", "sink")
        sink.set_property("sync", 0)

        def on_pad_added(_el, pad):
            sinkpad = mux.get_request_pad("sink_0")
            if pad.query_caps(None).to_string().startswith("video"):
                pad.link(sinkpad)

        src.connect("pad-added", on_pad_added)
        mux.link(pgie)
        pgie.link(tracker)
        tracker.link(analytics)
        analytics.link(sink)
        return pipeline, tracker

    def stream(self) -> Iterator[Detection]:  # pragma: no cover - requires DeepStream
        import queue
        import threading

        import pyds
        from gi.repository import GLib, Gst

        pipeline, tracker = self.build_pipeline()
        out: queue.Queue[Detection | None] = queue.Queue(maxsize=4096)

        def probe(pad, info, _u):
            buf = info.get_buffer()
            if not buf:
                return Gst.PadProbeReturn.OK
            batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
            l_frame = batch.frame_meta_list
            while l_frame is not None:
                frame = pyds.NvDsFrameMeta.cast(l_frame.data)
                ts = frame.buf_pts / 1e9
                l_obj = frame.obj_meta_list
                while l_obj is not None:
                    obj = pyds.NvDsObjectMeta.cast(l_obj.data)
                    if obj.confidence >= self.settings.min_detection_confidence:
                        r = obj.rect_params
                        out.put(
                            Detection(
                                track_id=int(obj.object_id),
                                label=pyds.get_string(obj.text_params.display_text) or obj.obj_label,
                                confidence=float(obj.confidence),
                                bbox=BBox(
                                    x=r.left / self._width,
                                    y=r.top / self._height,
                                    w=r.width / self._width,
                                    h=r.height / self._height,
                                ),
                                frame_idx=frame.frame_num,
                                timestamp=ts,
                            )
                        )
                    l_obj = l_obj.next
                l_frame = l_frame.next
            return Gst.PadProbeReturn.OK

        tracker.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe, 0)
        loop = GLib.MainLoop()
        pipeline.set_state(Gst.State.PLAYING)
        threading.Thread(target=loop.run, daemon=True).start()
        try:
            while True:
                det = out.get()
                if det is None:
                    break
                yield det
        finally:
            pipeline.set_state(Gst.State.NULL)
            loop.quit()


# ---------------------------------------------------------------------- synthetic
class SyntheticSource:
    """Replays a scripted scenario as if it came off the tracker.

    Scenario JSON format (see scripts/generate_synthetic_scenario.py):
        {"camera_id": "...", "fps": 10, "actors": [
            {"track_id": 1, "label": "person",
             "waypoints": [[t, x, y], ...], "size": [w, h]}]}
    """

    def __init__(self, scenario: dict, seed: int = 1337, jitter: float = 0.004):
        self.scenario = scenario
        self.camera_id = scenario.get("camera_id", "cam-synthetic")
        self.fps = float(scenario.get("fps", 10))
        self.rng = random.Random(seed)
        self.jitter = jitter

    @classmethod
    def from_file(cls, path: str | Path, seed: int = 1337) -> SyntheticSource:
        return cls(json.loads(Path(path).read_text()), seed=seed)

    def stream(self) -> Iterator[Detection]:
        dets: list[Detection] = []
        for actor in self.scenario["actors"]:
            wps = [(float(t), float(x), float(y)) for t, x, y in actor["waypoints"]]
            if len(wps) < 2:
                continue
            w, h = actor.get("size", [0.05, 0.12])
            t = wps[0][0]
            end = wps[-1][0]
            step = 1.0 / self.fps
            i = 0
            while t <= end + 1e-9:
                while i + 1 < len(wps) - 1 and wps[i + 1][0] < t:
                    i += 1
                (t0, x0, y0), (t1, x1, y1) = wps[i], wps[i + 1]
                a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                a = min(1.0, max(0.0, a))
                cx = x0 + a * (x1 - x0) + self.rng.gauss(0, self.jitter)
                cy = y0 + a * (y1 - y0) + self.rng.gauss(0, self.jitter)
                dets.append(
                    Detection(
                        track_id=int(actor["track_id"]),
                        label=actor["label"],
                        confidence=min(0.99, max(0.4, actor.get("confidence", 0.86) + self.rng.gauss(0, 0.03))),
                        bbox=BBox(x=cx - w / 2, y=cy - h / 2, w=w, h=h),
                        frame_idx=int(round(t * self.fps)),
                        timestamp=round(t, 4),
                    )
                )
                t = round(t + step, 6)
        dets.sort(key=lambda d: (d.timestamp, d.track_id))
        yield from dets


def build_source(camera_id: str, uri: str | None = None, scenario: dict | None = None) -> FrameSource:
    """Pick a backend. `auto` prefers DeepStream when the runtime is present."""
    mode = get_settings().perception.backend
    if mode == "synthetic" or (mode == "auto" and not DeepStreamSource.available()):
        if scenario is None:
            raise ValueError("synthetic backend requires a scenario")
        if mode == "auto":
            log.info("DeepStream runtime not found; using synthetic source for %s", camera_id)
        return SyntheticSource(scenario)
    if uri is None:
        raise ValueError("deepstream backend requires a source uri")
    return DeepStreamSource(camera_id, uri)
