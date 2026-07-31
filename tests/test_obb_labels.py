"""Solution H OBB label geometry (review item G2-08).

Two defects, both in the label builder rather than the model:

1. ``_fit_ellipse`` fitted on mean-shifted coordinates and then returned the *mean* as the centre,
   discarding the fitted offset. Correct only for a polygon whose vertices are symmetric about the
   ellipse centre; across 26,830 validation polygons the error was 0.157 px median, 0.702 px p99,
   7.216 px max, with 2.1% above half a pixel.
2. The builder clipped out-of-tile corners to [0, 1] *independently*, which does not clip a
   rectangle — it deforms it into a quadrilateral with the wrong angle and aspect, for 3.4% of
   instances. Training an angle head on deformed boxes defeats the exercise.

Orientation itself was never affected: ``ellipse_angle180`` reads only ``theta``.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from shapely.affinity import rotate, scale, translate
from shapely.geometry import Point, Polygon

from YOLOv8BeyondEarth.orientation import _fit_ellipse, ellipse_angle180, ellipse_obb

_SPEC = importlib.util.spec_from_file_location(
    "build_obb_labels", Path(__file__).resolve().parents[1] / "scripts" / "build_obb_labels.py")
build_obb = importlib.util.module_from_spec(_SPEC)
sys.modules["build_obb_labels"] = build_obb
_SPEC.loader.exec_module(build_obb)


def _ellipse(cx, cy, a, b, deg):
    """An exact ellipse polygon with a known centre, semi-axes and orientation."""
    base = scale(Point(0, 0).buffer(1.0, quad_segs=128), xfact=a, yfact=b)
    return translate(rotate(base, deg, origin=(0, 0)), cx, cy)


# --- the fitted centre ------------------------------------------------------------------------------

@pytest.mark.parametrize("cx,cy", [(0.0, 0.0), (37.5, -12.25), (1000.0, 2000.0)])
def test_fitted_centre_is_recovered(cx, cy):
    xc, yc, a, b, _ = _fit_ellipse(_ellipse(cx, cy, 8.0, 3.0, 25.0), 0.25)
    assert xc == pytest.approx(cx, abs=0.02) and yc == pytest.approx(cy, abs=0.02)
    assert sorted([a, b]) == pytest.approx([3.0, 8.0], abs=0.05)


def test_fit_agrees_with_an_unshifted_ellipse_fit():
    """The direct check that the conditioning shift is undone: fitting the same densified
    coordinates without any shift returns the centre outright, and the two must agree."""
    from shapely import segmentize
    from skimage.measure import EllipseModel

    for cx, cy, deg in ((0.0, 0.0, 0.0), (37.5, -12.25, 25.0), (500.0, 900.0, 115.0)):
        geom = _ellipse(cx, cy, 9.0, 3.5, deg)
        xy = np.asarray(segmentize(geom, 0.25).exterior.coords)
        ref = EllipseModel()
        assert ref.estimate(xy)
        xc, yc, _, _, _ = _fit_ellipse(geom, 0.25)
        assert xc == pytest.approx(ref.params[0], abs=1e-6)
        assert yc == pytest.approx(ref.params[1], abs=1e-6)


def test_centre_does_not_track_the_vertex_mean_on_an_asymmetric_polygon():
    """The failure mode itself. A half-ellipse (arc closed by its chord) has a vertex mean well
    away from the parent ellipse's centre, and densifying cannot symmetrize it because the chord
    stays a straight line."""
    ell = _ellipse(0.0, 0.0, 10.0, 4.0, 0.0)
    xy = np.asarray(ell.exterior.coords)
    half = Polygon(xy[xy[:, 1] >= 0])                    # upper arc + closing chord
    from shapely import segmentize
    mean_y = float(np.asarray(segmentize(half, 0.5).exterior.coords)[:, 1].mean())

    _, yc, _, _, _ = _fit_ellipse(half, 0.5)
    assert mean_y > 0.5, "fixture no longer has an offset vertex mean"
    assert abs(yc - mean_y) > 0.1, f"fitted centre {yc} is still the vertex mean {mean_y}"


def test_orientation_is_unaffected_by_the_centre_fix():
    """ellipse_angle180 reads only theta, so every Goal-2 angle result stands."""
    for deg in (0, 20, 55, 90, 140):
        ang, aspect = ellipse_angle180(_ellipse(500.0, -300.0, 9.0, 3.0, deg), 0.25)
        expected = (90.0 - deg) % 180.0        # math angle -> azimuth from North
        d = abs(ang - expected) % 180.0
        assert min(d, 180.0 - d) < 1.5, f"{deg}: got {ang}, expected {expected}"
        assert aspect == pytest.approx(3.0, abs=0.1)


def test_obb_is_centred_on_the_boulder():
    cx, cy = 61.3, 44.7
    corners, angle, aspect = ellipse_obb(_ellipse(cx, cy, 10.0, 4.0, 30.0), 0.25)
    centre = corners.mean(axis=0)
    assert centre[0] == pytest.approx(cx, abs=0.05) and centre[1] == pytest.approx(cy, abs=0.05)
    assert build_obb.is_rectangle(corners)
    # half-extents are the fitted semi-axes
    edge_long = np.linalg.norm(corners[0] - corners[3])
    edge_short = np.linalg.norm(corners[0] - corners[1])
    assert max(edge_long, edge_short) == pytest.approx(20.0, abs=0.3)
    assert min(edge_long, edge_short) == pytest.approx(8.0, abs=0.3)
    assert aspect == pytest.approx(2.5, abs=0.1)


# --- rectangle validity and the edge policy -----------------------------------------------------------

def test_is_rectangle_rejects_an_independently_clipped_box():
    """Exactly what the old builder emitted for 3.4% of instances. A *rotated* box is the case
    that matters — clipping an axis-aligned one happens to leave a rectangle, which is why the
    deformation went unnoticed."""
    theta = np.deg2rad(35.0)
    u = 0.25 * np.array([np.cos(theta), np.sin(theta)])
    v = 0.10 * np.array([-np.sin(theta), np.cos(theta)])
    c = np.array([0.12, 0.5])                            # close enough to x = 0 to overhang
    corners = np.array([c + u + v, c + u - v, c - u - v, c - u + v])
    assert build_obb.is_rectangle(corners)
    assert (corners < 0).any(), "fixture does not overhang the tile"
    assert not build_obb.is_rectangle(np.clip(corners, 0.0, 1.0))


def test_rotated_rectangle_is_accepted():
    theta = np.deg2rad(37.0)
    u = 0.3 * np.array([np.cos(theta), np.sin(theta)])
    v = 0.1 * np.array([-np.sin(theta), np.cos(theta)])
    c = np.array([0.5, 0.5])
    assert build_obb.is_rectangle(np.array([c + u + v, c + u - v, c - u - v, c - u + v]))


def _write_seg(tmp_path, polys):
    src = tmp_path / "tile.txt"
    src.write_text("\n".join("0 " + " ".join(f"{v:.6f}" for v in np.asarray(p).flatten())
                             for p in polys) + "\n")
    return src


def _norm_ellipse_coords(cx, cy, a, b, deg):
    return np.asarray(_ellipse(cx, cy, a, b, deg).exterior.coords)[::8]


def test_edge_crossing_labels_are_dropped_not_deformed(tmp_path):
    interior = _norm_ellipse_coords(0.50, 0.50, 0.10, 0.04, 20.0)
    crossing = _norm_ellipse_coords(0.02, 0.50, 0.10, 0.04, 20.0)   # sticks out past x = 0
    src = _write_seg(tmp_path, [interior, crossing])
    dst = tmp_path / "out.txt"

    counts = build_obb.convert_file(src, dst, edge_policy="drop")
    assert counts["n_in"] == 2 and counts["n_edge"] == 1
    assert counts["n_edge_dropped"] == 1 and counts["n_out"] == 1
    kept = [l for l in dst.read_text().splitlines() if l.strip()]
    assert len(kept) == 1
    corners = np.asarray(kept[0].split()[1:], dtype=float).reshape(4, 2)
    assert build_obb.is_rectangle(corners)
    assert (corners >= 0).all() and (corners <= 1).all()


def test_keep_policy_emits_a_true_rectangle_outside_the_tile(tmp_path):
    crossing = _norm_ellipse_coords(0.02, 0.50, 0.10, 0.04, 20.0)
    src = _write_seg(tmp_path, [crossing])
    dst = tmp_path / "out.txt"

    counts = build_obb.convert_file(src, dst, edge_policy="keep")
    assert counts["n_edge"] == 1 and counts["n_edge_dropped"] == 0 and counts["n_out"] == 1
    corners = np.asarray(dst.read_text().split()[1:], dtype=float).reshape(4, 2)
    assert build_obb.is_rectangle(corners), "the kept box is not a rectangle"
    assert (corners < 0).any(), "fixture no longer crosses the tile edge"


def test_every_emitted_label_is_a_rectangle(tmp_path):
    rng = np.random.default_rng(0)
    polys = [_norm_ellipse_coords(*rng.uniform([0.2, 0.2, 0.03, 0.02, 0], [0.8, 0.8, 0.12, 0.03, 180]))
             for _ in range(40)]
    src = _write_seg(tmp_path, polys)
    dst = tmp_path / "out.txt"
    counts = build_obb.convert_file(src, dst, edge_policy="drop")
    assert counts["n_out"] > 20
    for line in dst.read_text().splitlines():
        if line.strip():
            assert build_obb.is_rectangle(np.asarray(line.split()[1:], dtype=float).reshape(4, 2))


def test_malformed_rows_are_skipped(tmp_path):
    src = tmp_path / "tile.txt"
    src.write_text("0 0.1 0.1 0.2 0.2\n0 0.1 0.1 0.2 0.2 0.3\n\n")   # too short; odd count
    counts = build_obb.convert_file(src, tmp_path / "out.txt")
    assert counts["n_in"] == 0 and counts["n_out"] == 0
