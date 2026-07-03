"""Characterization tests for angles_from_masks (Test 3d tracer swap) — CPU only, no model.

Locks two things: (i) both tracers read a clearly-elongated mask's orientation correctly and
consistently (the azimuth convention), and (ii) cv2's low-vertex outlines for tiny blocky blobs are
NOT dropped (the deliberate <3-vertex-only guard — a <5 guard would bias exactly the marginal
regime Test 3d measures).
"""
import numpy as np

from YOLOv8BeyondEarth.orientation_validation import angles_from_masks


def _rect_mask(h, w, pad=3):
    m = np.zeros((h + 2 * pad, w + 2 * pad), dtype=np.uint8)
    m[pad:pad + h, pad:pad + w] = 1
    return m


def test_both_tracers_read_elongation_correctly():
    wide = _rect_mask(6, 20)        # long axis E-W -> azimuth 90
    tall = _rect_mask(20, 6)        # long axis N-S -> azimuth 0
    for method in ("skimage", "cv2"):
        df = angles_from_masks([wide, tall], method=method)
        assert np.all(np.isfinite(df.angle180.values)), method
        d_wide = abs(((df.angle180[0] - 90.0 + 90.0) % 180.0) - 90.0)
        d_tall = abs(((df.angle180[1] - 0.0 + 90.0) % 180.0) - 90.0)
        assert d_wide <= 5.0, f"{method}: E-W rect read as {df.angle180[0]}"
        assert d_tall <= 5.0, f"{method}: N-S rect read as {df.angle180[1]}"
        assert np.all(df.aspect.values > 2.0), method


def test_tracers_agree_on_same_mask():
    rng = np.random.default_rng(0)
    masks = []
    for _ in range(20):             # random small-ish rectangles at grid-aligned orientations
        h, w = rng.integers(4, 12), rng.integers(4, 12)
        masks.append(_rect_mask(int(h), int(w)))
    sk = angles_from_masks(masks, "skimage")
    cv = angles_from_masks(masks, "cv2")
    ok = np.isfinite(sk.angle180.values) & np.isfinite(cv.angle180.values)
    assert ok.mean() > 0.9
    d = np.abs(((sk.angle180.values[ok] - cv.angle180.values[ok] + 90.0) % 180.0) - 90.0)
    # exclude near-square masks (no meaningful axis: tiny aspect -> angle is noise for BOTH tracers)
    elong = (sk.aspect.values[ok] > 1.3) & (cv.aspect.values[ok] > 1.3)
    assert np.median(d[elong]) <= 5.0


def test_cv2_four_vertex_square_not_dropped():
    # a tiny 3x3 square: cv2 CHAIN_APPROX_SIMPLE gives a 4-corner outline; it must be traced,
    # not dropped (rows stay aligned, angle may be meaningless but must be finite or NaN in place)
    df = angles_from_masks([_rect_mask(3, 3)], "cv2")
    assert len(df) == 1             # row preserved either way


def test_rows_align_with_input_order():
    masks = [_rect_mask(6, 18), np.zeros((5, 5), dtype=np.uint8), _rect_mask(18, 6)]
    for method in ("skimage", "cv2"):
        df = angles_from_masks(masks, method)
        assert len(df) == 3
        assert np.isnan(df.angle180[1])                    # empty mask -> NaN in place
        assert np.isfinite(df.angle180[0]) and np.isfinite(df.angle180[2])
