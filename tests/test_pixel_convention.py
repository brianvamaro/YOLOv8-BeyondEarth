"""The pixel-boundary convention and mask-support gate (review items X-03 and G1-06).

Two things are pinned here.

1. **One documented convention.** Production geometry is the skimage 0.5 level set. These tests
   fix its area/diameter on solid blocks so a future perf change cannot quietly move it, and they
   fix the *size of cv2's disagreement* so nobody promotes the fast tracer into area work by
   accident.
2. **Component selection by area, not vertex count.** A thin ragged artifact has a long perimeter
   and therefore many vertices; the old ``max(contours, key=len)`` let it replace the larger
   compact boulder in the same mask.
"""
import numpy as np
import pytest
from shapely.geometry import Polygon

from YOLOv8BeyondEarth.polygon import (
    MASK_SUPPORT_LEVEL, binary_mask_to_polygon, binary_mask_to_polygon_cv,
    check_mask_validity, mask_support_area,
)

TRACERS = {"skimage": binary_mask_to_polygon, "cv2": binary_mask_to_polygon_cv}


def _area(poly):
    return Polygon(poly).area


def _equiv_diameter(poly):
    return np.sqrt(4.0 * _area(poly) / np.pi)


def _solid(n, pad=3, value=1.0, dtype=np.float32):
    m = np.zeros((n + 2 * pad, n + 2 * pad), dtype=dtype)
    m[pad:pad + n, pad:pad + n] = value
    return m


# --- the convention itself -----------------------------------------------------------------------

@pytest.mark.parametrize("n,expected", [(3, 8.5), (5, 24.5), (10, 99.5), (20, 399.5)])
def test_skimage_area_is_pixel_count_minus_corner_clipping(n, expected):
    """The 0.5 level set of a solid n x n block encloses n**2 - 0.5: marching squares cuts each
    of the four corners by an eighth of a pixel. This IS the production area definition."""
    assert _area(binary_mask_to_polygon(_solid(n))) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("n,sk,cv", [(5, 24.5, 16.0), (10, 99.5, 81.0), (20, 399.5, 361.0)])
def test_cv2_traces_pixel_centres_and_is_biased_low(n, sk, cv):
    """cv2 follows boundary pixel *centres*, so it inscribes an (n-1) x (n-1) square. Pinning the
    exact deficit keeps the module docstring honest and blocks silent adoption for area work."""
    a_sk = _area(binary_mask_to_polygon(_solid(n)))
    a_cv = _area(binary_mask_to_polygon_cv(_solid(n)))
    assert a_sk == pytest.approx(sk, abs=1e-6)
    assert a_cv == pytest.approx(cv, abs=1e-6)
    # equivalent-diameter deficit: ~19% at 5 px, shrinking with size but never negligible
    deficit = 1.0 - _equiv_diameter(binary_mask_to_polygon_cv(_solid(n))) / \
        _equiv_diameter(binary_mask_to_polygon(_solid(n)))
    assert 0.02 < deficit < 0.25
    if n == 5:
        assert deficit == pytest.approx(0.192, abs=0.01)


@pytest.mark.parametrize("deg", [15, 65, 110, 150])
def test_orientation_agrees_across_tracers_even_though_area_does_not(deg):
    """cv2's clearance is angles, not size — the reason it exists. An elongated blob must read the
    same axis under both tracers while its area differs by ~20%."""
    from skimage.draw import ellipse
    from skimage.measure import EllipseModel

    def angle(poly):
        m = EllipseModel()
        xy = np.asarray(poly, dtype=float)
        assert m.estimate(xy - xy.mean(axis=0))
        return np.degrees(m.params[4]) % 180.0

    img = np.zeros((140, 140), dtype=np.float32)
    rr, cc = ellipse(70, 70, 14, 34, shape=img.shape, rotation=np.deg2rad(deg))
    img[rr, cc] = 1.0
    d = abs(angle(binary_mask_to_polygon(img)) - angle(binary_mask_to_polygon_cv(img))) % 180.0
    assert min(d, 180.0 - d) < 5.0
    # the area deficit is a perimeter effect, so it is only ~5% on a blob this large — see
    # test_cv2_traces_pixel_centres_and_is_biased_low for the 36% it reaches on a 5 px boulder
    ratio = _area(binary_mask_to_polygon_cv(img)) / _area(binary_mask_to_polygon(img))
    assert 0.90 < ratio < 0.98, f"cv2/skimage area ratio {ratio:.3f} out of the pinned range"


# --- component selection -------------------------------------------------------------------------

def test_disconnected_mask_keeps_the_largest_area_not_the_most_vertices():
    """The X-03 failure mode. A 1-px-wide filament carries roughly two contour vertices per pixel
    of length, so it out-votes a compact boulder of three times its area on vertex count. Under
    the old ``max(contours, key=len)`` the filament won and replaced the boulder."""
    from skimage.measure import find_contours

    m = np.zeros((60, 60), dtype=np.float32)
    m[5:17, 5:17] = 1.0                                     # compact 12x12 boulder, area 143.5
    m[40, 3:58] = 1.0                                       # 1 px x 55 filament, area ~54.5

    contours = find_contours(np.pad(m, 1, constant_values=0), MASK_SUPPORT_LEVEL)
    assert len(contours) == 2
    by_vertices = max(contours, key=len)
    assert len(by_vertices) > 100                           # the filament, by a wide margin
    assert by_vertices[:, 0].min() > 30                     # ...and it is the filament (row ~40)

    poly = binary_mask_to_polygon(m)
    assert _area(poly) == pytest.approx(143.5, abs=1e-6)    # the boulder, not the filament
    assert poly[:, 1].max() < 20                            # every vertex is in the boulder's rows


def test_outer_ring_wins_over_its_own_hole():
    """find_contours also returns hole boundaries; the enclosing ring must win."""
    m = np.zeros((40, 40), dtype=np.float32)
    m[10:30, 10:30] = 1.0
    m[16:24, 16:24] = 0.0                                  # an 8x8 hole
    poly = binary_mask_to_polygon(m)
    assert _area(poly) == pytest.approx(399.5, abs=1.0)    # the 20x20 outer boundary


def test_cv2_degenerate_single_pixel_line_is_not_dropped():
    """A 1-px-wide blob has zero contour area under cv2; it must still return an outline rather
    than an arbitrary pick (the marginal regime the Test-3d tracer swap measures)."""
    m = np.zeros((20, 20), dtype=np.float32)
    m[10, 4:14] = 1.0
    poly = binary_mask_to_polygon_cv(m)
    assert poly is not None and len(poly) >= 2


# --- fractional / downscaled masks and the area gate (G1-06) --------------------------------------

def test_cv2_no_longer_truncates_fractional_masks_to_zero():
    """After an INTER_AREA downscale, mask values are fractional. cv2 used to cast to uint8
    *before* thresholding, so a mask of 0.9 truncated to 0 and the two tracers disagreed about
    which pixels are even part of the object. Both now threshold at the same 0.5."""
    m = np.zeros((20, 20), dtype=np.float32)
    m[6:14, 6:14] = 0.9                                     # supported, none equal to 1.0
    assert binary_mask_to_polygon(m) is not None
    cv = binary_mask_to_polygon_cv(m)
    assert cv is not None and _area(cv) == pytest.approx(49.0, abs=1e-6)   # 7x7 inscribed


def test_fractional_mask_isoline_shrinks_with_the_fringe_value():
    """A real property of the 0.5-isoline convention, worth pinning because it bites the
    ``downscale_pred=True`` production path.

    The contour is a true isoline, not a pixel boundary: between an interior pixel of value v and
    an exterior 0, marching squares puts the crossing at ``(v - 0.5) / v`` of a pixel outside the
    interior centre. So a downscaled boulder whose fringe pixels are only partly covered traces
    *smaller* than the same boulder at full coverage, even though both have the same supported
    pixel count. Areas stay bracketed by the inscribed (n-1)^2 and the circumscribed n^2.
    """
    areas = [_area(binary_mask_to_polygon(_solid(8, value=v))) for v in (1.0, 0.9, 0.7, 0.55)]
    assert areas[0] == pytest.approx(63.5, abs=1e-6)        # full coverage: n^2 - corner clip
    assert all(np.diff(areas) < 0), f"isoline area must fall with fringe value, got {areas}"
    assert all(49.0 < a < 64.0 for a in areas)
    # ...while the support count — what the gate uses — is identical for all of them
    assert {mask_support_area(_solid(8, value=v)) for v in (1.0, 0.9, 0.7, 0.55)} == {64}


def test_area_gate_counts_support_not_full_coverage():
    """G1-06: the gate must count the pixels the model considers part of the object, at the same
    level the tracer uses. A downscaled 3x3 boulder whose pixels sit at 0.6 has 9 supported pixels
    and 0 fully-covered ones — the old ``== 1`` count dropped it, though the tracer would have
    outlined it happily."""
    m = _solid(3, value=0.6)
    assert mask_support_area(m) == 9
    assert int(np.count_nonzero(m == 1)) == 0               # what the old gate saw
    assert binary_mask_to_polygon(m) is not None            # ...and what the tracer sees
    assert mask_support_area(_solid(3, value=0.4)) == 0     # genuinely unsupported
    assert binary_mask_to_polygon(_solid(3, value=0.4)) is None


def test_area_gate_matches_hard_mask_behaviour_exactly():
    """No change for hard {0, 1} masks — the default `downscale_pred=False` path is untouched."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        m = (rng.random((16, 16)) > 0.5).astype(np.float32)
        assert mask_support_area(m) == int(np.count_nonzero(m == 1))


def test_check_mask_validity_uses_the_same_level():
    assert check_mask_validity(_solid(3, value=0.6), min_area_threshold=4) is True
    assert check_mask_validity(_solid(3, value=0.4), min_area_threshold=4) is False


def test_empty_mask_returns_none_for_both_tracers():
    empty = np.zeros((12, 12), dtype=np.float32)
    sub_threshold = np.full((12, 12), 0.3, dtype=np.float32)
    for name, fn in TRACERS.items():
        assert fn(empty) is None, name
        assert fn(sub_threshold) is None, name
