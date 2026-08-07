"""Polygon zone geometry in normalised image coordinates."""
from __future__ import annotations

from collections import defaultdict

from ..schemas import BBox, Zone


def point_in_polygon(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    """Standard ray-casting test. Boundary points count as inside."""
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if abs((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)) < 1e-12 and (
            min(x1, x2) - 1e-12 <= x <= max(x1, x2) + 1e-12
            and min(y1, y2) - 1e-12 <= y <= max(y1, y2) + 1e-12
        ):
            return True
        if (y1 > y) != (y2 > y):
            x_cross = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-15) + x1
            if x < x_cross:
                inside = not inside
    return inside


def polygon_area(poly: list[tuple[float, float]]) -> float:
    s = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def bbox_overlap_fraction(bbox: BBox, poly: list[tuple[float, float]], samples: int = 5) -> float:
    """Fraction of a bbox that falls inside the polygon, by grid sampling.

    Cheap and adequate at the resolutions we work at; a full polygon clip is
    overkill for a keep-clear check and costs more than the tracker itself.
    """
    hits = 0
    total = samples * samples
    for i in range(samples):
        for j in range(samples):
            px = bbox.x + bbox.w * (i + 0.5) / samples
            py = bbox.y + bbox.h * (j + 0.5) / samples
            if point_in_polygon(px, py, poly):
                hits += 1
    return hits / total


class ZoneIndex:
    """Lookup of zones by camera, with containment queries."""

    def __init__(self, zones: list[Zone] | None = None):
        self._by_camera: dict[str, list[Zone]] = defaultdict(list)
        self._by_id: dict[str, Zone] = {}
        for z in zones or []:
            self.add(z)

    def add(self, zone: Zone) -> None:
        self._by_camera[zone.camera_id].append(zone)
        self._by_id[zone.zone_id] = zone

    def get(self, zone_id: str) -> Zone | None:
        return self._by_id.get(zone_id)

    def for_camera(self, camera_id: str) -> list[Zone]:
        return list(self._by_camera.get(camera_id, []))

    def zones_containing(self, camera_id: str, x: float, y: float) -> list[Zone]:
        return [z for z in self.for_camera(camera_id) if point_in_polygon(x, y, z.polygon)]

    def keep_clear(self, camera_id: str) -> list[Zone]:
        return [z for z in self.for_camera(camera_id) if z.kind == "keep_clear"]

    def __len__(self) -> int:
        return len(self._by_id)

    @classmethod
    def from_config(cls, raw: dict) -> ZoneIndex:
        zones = []
        for cam_id, entries in (raw.get("cameras") or {}).items():
            for e in entries:
                zones.append(
                    Zone(
                        zone_id=e["zone_id"],
                        name=e.get("name", e["zone_id"]),
                        camera_id=cam_id,
                        kind=e.get("kind", "aisle"),
                        polygon=[tuple(p) for p in e["polygon"]],
                    )
                )
        return cls(zones)
