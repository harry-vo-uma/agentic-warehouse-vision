"""Keyframe sampling and encoding.

Decodes with PyAV when the source media is present; otherwise emits deterministic
placeholder frames so the graph, the API and the eval harness all still run.
The placeholders are real JPEGs (a solid-colour tile derived from the frame's
identity), so anything that base64-decodes and opens an image still works.
"""
from __future__ import annotations

import base64
import hashlib
import io
import logging
from dataclasses import dataclass
from pathlib import Path

from ..config import get_settings

log = logging.getLogger(__name__)


@dataclass
class Keyframe:
    uri: str
    t: float
    b64: str
    decoded: bool


def _placeholder_jpeg(seed: str, size: int = 64) -> bytes:
    """Tiny deterministic JPEG. Uses Pillow when available, else a minimal
    baseline JPEG byte string so the function never raises.
    """
    digest = hashlib.md5(seed.encode()).digest()
    rgb = (digest[0], digest[1], digest[2])
    try:
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (size, size), rgb).save(buf, format="JPEG", quality=70)
        return buf.getvalue()
    except Exception:
        return _MINIMAL_JPEG


class KeyframeSampler:
    """Selects timestamps and returns encoded frames.

    Sampling is focus-weighted: half the budget lands within +/-20% of the
    focus instant, the rest spreads across the window. Uniform sampling of a
    10s window at 4 frames routinely misses the half-second that matters.
    """

    def __init__(self, media_root: Path | None = None):
        self.media_root = media_root or (get_settings().data_dir / "video")
        self._decoded = 0
        self._placeholders = 0

    def timestamps(self, t_start: float, t_end: float, focus_t: float, n: int) -> list[float]:
        if n <= 1:
            return [focus_t]
        span = max(1e-6, t_end - t_start)
        n_focus = max(1, n // 2)
        n_spread = n - n_focus
        focus_span = 0.2 * span
        ts = [
            min(t_end, max(t_start, focus_t + focus_span * ((i / max(1, n_focus - 1)) - 0.5)))
            for i in range(n_focus)
        ] if n_focus > 1 else [focus_t]
        ts += [t_start + span * (i + 0.5) / n_spread for i in range(n_spread)]
        return sorted(set(round(t, 3) for t in ts))

    def sample(self, camera_id: str, t_start: float, t_end: float, focus_t: float, n: int) -> list[Keyframe]:
        ts = self.timestamps(t_start, t_end, focus_t, n)
        source = self.media_root / f"{camera_id}.mp4"
        decoded = self._decode(source, ts) if source.exists() else {}
        out = []
        for t in ts:
            raw = decoded.get(t)
            if raw is None:
                raw = _placeholder_jpeg(f"{camera_id}:{t}")
                self._placeholders += 1
            else:
                self._decoded += 1
            out.append(
                Keyframe(
                    uri=f"/media/{camera_id}/frame/{t:.3f}.jpg",
                    t=t,
                    b64=base64.b64encode(raw).decode(),
                    decoded=t in decoded,
                )
            )
        return out

    def _decode(self, source: Path, ts: list[float]) -> dict[float, bytes]:
        try:
            import av  # type: ignore
        except Exception:
            return {}
        out: dict[float, bytes] = {}
        try:
            with av.open(str(source)) as container:
                stream = container.streams.video[0]
                for t in ts:
                    container.seek(int(t / float(stream.time_base)), stream=stream)
                    for frame in container.decode(stream):
                        buf = io.BytesIO()
                        frame.to_image().save(buf, format="JPEG", quality=80)
                        out[t] = buf.getvalue()
                        break
        except Exception as exc:  # noqa: BLE001 - decode failures fall back to placeholders
            log.warning("keyframe decode failed for %s: %s", source, exc)
        return out

    def stats(self) -> dict[str, int]:
        return {"decoded": self._decoded, "placeholders": self._placeholders}


#: 1x1 baseline JPEG, used only if Pillow is unavailable.
_MINIMAL_JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0a"
    "HBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFAABAAAAAAAA"
    "AAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AKp//2Q=="
)
