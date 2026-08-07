from awvi.perception.zones import bbox_overlap_fraction, point_in_polygon, polygon_area
from awvi.schemas import BBox

SQUARE = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
CONCAVE = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 0.4), (0.0, 1.0)]


def test_point_inside_and_outside():
    assert point_in_polygon(0.5, 0.5, SQUARE)
    assert not point_in_polygon(1.5, 0.5, SQUARE)
    assert not point_in_polygon(-0.1, 0.5, SQUARE)


def test_boundary_counts_as_inside():
    assert point_in_polygon(0.0, 0.5, SQUARE)
    assert point_in_polygon(0.5, 1.0, SQUARE)


def test_concave_notch_excluded():
    # (0.5, 0.7) sits inside the notch cut out of the top edge.
    assert not point_in_polygon(0.5, 0.7, CONCAVE)
    assert point_in_polygon(0.1, 0.5, CONCAVE)


def test_polygon_area():
    assert abs(polygon_area(SQUARE) - 1.0) < 1e-9


def test_bbox_overlap_fraction():
    full = BBox(x=0.2, y=0.2, w=0.2, h=0.2)
    assert bbox_overlap_fraction(full, SQUARE) == 1.0
    outside = BBox(x=2.0, y=2.0, w=0.1, h=0.1)
    assert bbox_overlap_fraction(outside, SQUARE) == 0.0
    half = BBox(x=0.9, y=0.4, w=0.2, h=0.2)
    assert 0.0 < bbox_overlap_fraction(half, SQUARE) < 1.0


def test_zone_index_lookup(zones):
    assert len(zones) == 1
    assert zones.for_camera("cam-test")
    assert zones.for_camera("cam-other") == []
    assert [z.zone_id for z in zones.keep_clear("cam-test")] == ["z-keepclear"]
    assert zones.zones_containing("cam-test", 0.3, 0.3)
    assert not zones.zones_containing("cam-test", 0.9, 0.9)
