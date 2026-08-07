import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Tests must never hit a live endpoint: results have to be reproducible in CI.
os.environ["AWVI_FORCE_MOCK"] = "1"
os.environ.setdefault("AWVI_PERCEPTION_BACKEND", "synthetic")

import pytest  # noqa: E402

from awvi.perception.tracks import TrackStore  # noqa: E402
from awvi.perception.zones import ZoneIndex  # noqa: E402
from awvi.schemas import Zone  # noqa: E402


@pytest.fixture
def zones() -> ZoneIndex:
    return ZoneIndex([
        Zone(
            zone_id="z-keepclear",
            name="Test keep clear",
            camera_id="cam-test",
            kind="keep_clear",
            polygon=[(0.2, 0.2), (0.5, 0.2), (0.5, 0.5), (0.2, 0.5)],
        )
    ])


@pytest.fixture
def store() -> TrackStore:
    return TrackStore("cam-test")


def make_scenario(actors, camera_id="cam-test", fps=10):
    return {"camera_id": camera_id, "fps": fps, "actors": actors}


def line(t0, t1, p0, p1, steps=8):
    return [
        [t0 + (t1 - t0) * i / steps,
         p0[0] + (p1[0] - p0[0]) * i / steps,
         p0[1] + (p1[1] - p0[1]) * i / steps]
        for i in range(steps + 1)
    ]


def hold(t0, t1, p, steps=6):
    return [[t0 + (t1 - t0) * i / steps, p[0], p[1]] for i in range(steps + 1)]
