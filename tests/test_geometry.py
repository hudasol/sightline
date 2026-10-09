import math

import numpy as np
import pytest

from sightline.geometry import GeometryError, Polygon, box_ref_point, iou_matrix

SQUARE = [(0, 0), (100, 0), (100, 100), (0, 100)]


def test_ref_point_is_bottom_centre():
    assert box_ref_point((10, 20, 30, 80)) == (20.0, 80)


def test_contains_inside_outside_and_boundary():
    p = Polygon.create(SQUARE)
    assert p.contains((50, 50))
    assert not p.contains((150, 50))
    assert not p.contains((50, -1))
    assert p.contains((0, 50))  # on an edge counts as inside
    assert p.contains((100, 100))  # corner too


def test_concave_polygon():
    l_shape = Polygon.create([(0, 0), (100, 0), (100, 40), (40, 40), (40, 100), (0, 100)])
    assert l_shape.contains((20, 80))
    assert not l_shape.contains((70, 70))


@pytest.mark.parametrize("verts,msg", [
    ([(0, 0), (1, 1)], "at least 3"),
    ([(0, 0), (1, 1), (2, 2)], "zero area"),
    ([(0, 0), (100, 100), (100, 0), (0, 100)], "self-intersecting"),
    ([(0, 0), (math.nan, 0), (1, 1)], "non-finite"),
    ([("a", 0), (1, 1), (2, 5)], "number pairs"),
])
def test_invalid_polygons_are_rejected(verts, msg):
    with pytest.raises(GeometryError, match=msg):
        Polygon.create(verts)


def test_vertex_outside_frame_is_rejected():
    with pytest.raises(GeometryError, match="outside"):
        Polygon.create([(0, 0), (700, 0), (700, 100)], frame_size=(640, 360))


def test_nearest_edge():
    p = Polygon.create(SQUARE)  # edges: 0 top, 1 right, 2 bottom, 3 left
    assert p.nearest_edge((50, 3)) == 0
    assert p.nearest_edge((97, 50)) == 1
    assert p.nearest_edge((5, 50)) == 3


def test_iou_matrix_known_values():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    iou = iou_matrix(a, b)
    assert iou[0, 0] == pytest.approx(1.0)
    assert iou[0, 1] == pytest.approx(50 / 150)
    assert iou[0, 2] == 0.0
    assert iou_matrix(np.zeros((0, 4)), b).shape == (0, 3)
