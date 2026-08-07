"""Evidence clip extraction.

Uses ffmpeg stream-copy when the source media exists (no re-encode: an
operator opening a clip should not wait on a transcode). When it does not, it
records the cut points and returns an unmaterialised handle so the rest of the
system is unaffected.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..config import get_settings

log = logging.getLogger(__name__)


@dataclass
class Clip:
    uri: str
    path: Path | None
    t_start: float
    t_end: float
    materialised: bool

    @property
    def duration(self) -> float:
        return max(0.0, self.t_end - self.t_start)


class ClipExtractor:
    def __init__(self, media_root: Path | None = None, out_root: Path | None = None):
        cfg = get_settings()
        self.media_root = media_root or (cfg.data_dir / "video")
        self.out_root = out_root or (cfg.data_dir / "clips")
        self.extracted = 0
        self.skipped = 0

    def extract(self, camera_id: str, t_start: float, t_end: float, event_id: str) -> Clip:
        uri = f"/media/clips/{event_id}.mp4"
        source = self.media_root / f"{camera_id}.mp4"
        target = self.out_root / f"{event_id}.mp4"

        if target.exists():
            return Clip(uri=uri, path=target, t_start=t_start, t_end=t_end, materialised=True)

        if not source.exists() or not shutil.which("ffmpeg"):
            self.skipped += 1
            self._write_manifest(event_id, camera_id, t_start, t_end, materialised=False)
            return Clip(uri=uri, path=None, t_start=t_start, t_end=t_end, materialised=False)

        self.out_root.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{t_start:.3f}",
            "-to", f"{t_end:.3f}",
            "-i", str(source),
            "-c", "copy",
            "-movflags", "+faststart",
            str(target),
        ]
        try:
            subprocess.run(cmd, check=True, timeout=60)
            self.extracted += 1
            self._write_manifest(event_id, camera_id, t_start, t_end, materialised=True)
            return Clip(uri=uri, path=target, t_start=t_start, t_end=t_end, materialised=True)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            log.warning("clip extraction failed for %s: %s", event_id, exc)
            self.skipped += 1
            return Clip(uri=uri, path=None, t_start=t_start, t_end=t_end, materialised=False)

    def _write_manifest(self, event_id, camera_id, t_start, t_end, materialised: bool) -> None:
        self.out_root.mkdir(parents=True, exist_ok=True)
        (self.out_root / f"{event_id}.json").write_text(
            json.dumps(
                {
                    "event_id": event_id,
                    "camera_id": camera_id,
                    "t_start": round(t_start, 3),
                    "t_end": round(t_end, 3),
                    "materialised": materialised,
                },
                indent=2,
            )
        )
