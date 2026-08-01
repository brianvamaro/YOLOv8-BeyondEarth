"""The Goal-2 validation path must be the production path (review item G2-02).

Three things were wrong with the T1-T5 runners:

1. The plan registered ``EllipseModel`` angles; the runners used weighted second moments. Those
   are different estimators, and earlier work showed the choice can change chirality — so a null
   measured with one and read against a plan written for the other is not the registered test.
2. The mask cleanup and the row-flip that puts angles in a North-up frame were copy-pasted into
   four functions, free to drift apart.
3. The tiling helper kept tile-edge objects. Production discards them, and for good reason: a
   clipped mask has a spurious straight edge aligned to the tile boundary, which is a *cardinal*
   bias fed straight into tests measuring cardinal bias.
"""
import numpy as np
import pytest
import rasterio as rio
import torch
from rasterio.transform import from_origin
from types import SimpleNamespace

from YOLOv8BeyondEarth import predict as P
from YOLOv8BeyondEarth.orientation import (
    ANGLE_ESTIMATORS, is_mask_within_slice, mask_readout, mask_to_polygon_geom)
from YOLOv8BeyondEarth.orientation_validation import (
    angles_from_masks, detect_binary_masks, detect_orientations)

SLICE = 64


def _axial_gap(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _bar(mask_size, cy, cx, half_long, half_short, deg):
    """An elongated blob at a known azimuth. deg is the math angle of the long axis."""
    from skimage.draw import ellipse
    m = np.zeros((mask_size, mask_size), dtype=np.float32)
    rr, cc = ellipse(cy, cx, half_short, half_long, shape=m.shape, rotation=np.deg2rad(deg))
    m[rr, cc] = 1.0
    return m


# --- the shared readout ------------------------------------------------------------------------

@pytest.mark.parametrize("deg", [0, 20, 55, 90, 130, 160])
def test_mask_readout_recovers_a_known_azimuth(deg):
    ang, aspect, dpx = mask_readout(_bar(120, 60, 60, 30, 10, deg))
    assert _axial_gap(ang, (90.0 - deg) % 180.0) < 3.0     # math angle -> azimuth from North
    assert aspect == pytest.approx(3.0, abs=0.4)
    assert dpx == pytest.approx(2 * np.sqrt(np.pi * 30 * 10 / np.pi), abs=2.0)


def test_readout_cleans_to_the_largest_blob_and_fills_holes():
    m = _bar(120, 60, 60, 30, 10, 30.0)
    m[20:24, 20:24] = 1.0                                   # a speck elsewhere
    m[58:62, 58:62] = 0.0                                   # a hole in the boulder
    geom, area = mask_to_polygon_geom(m)
    assert geom is not None
    assert not geom.interiors, "the hole was not filled"
    assert geom.bounds[0] > 25, "the speck was not removed"
    ang, _, _ = mask_readout(m)
    assert _axial_gap(ang, 60.0) < 4.0


def test_empty_and_degenerate_masks_return_nan_not_an_exception():
    ang, asp, dpx = mask_readout(np.zeros((20, 20), dtype=np.float32))
    assert np.isnan(ang) and np.isnan(asp)
    assert mask_to_polygon_geom(np.zeros((8, 8))) == (None, 0)


def test_estimator_must_be_named_explicitly():
    with pytest.raises(ValueError, match="estimator"):
        mask_readout(_bar(60, 30, 30, 12, 5, 0.0), estimator="moment")
    assert set(ANGLE_ESTIMATORS) == {"ellipse", "moments"}


def test_the_two_estimators_are_not_interchangeable():
    """Both are defensible; they are not the same number, which is the point of registering one.
    On a clean ellipse they agree, so the fixture is a lopsided blob of the kind real masks are."""
    m = _bar(120, 60, 60, 30, 10, 35.0)
    m[62:78, 58:96] = 1.0                                   # an asymmetric lobe
    ell = mask_readout(m, estimator="ellipse")[0]
    mom = mask_readout(m, estimator="moments")[0]
    assert np.isfinite(ell) and np.isfinite(mom)
    assert _axial_gap(ell, mom) > 2.0, (ell, mom)


def test_angles_from_masks_uses_the_shared_readout():
    m = _bar(120, 60, 60, 30, 10, 25.0)
    df = angles_from_masks([m], method="skimage")
    direct = mask_readout(m, contour_method="skimage")
    assert df.angle180[0] == pytest.approx(direct[0], abs=1e-9)
    assert df.diameter_px[0] == pytest.approx(direct[2], abs=1e-9)


# --- the production edge rule --------------------------------------------------------------------

def test_is_mask_within_slice_matches_productions_5_percent_rule():
    interior = np.zeros((SLICE, SLICE), np.float32)
    interior[20:40, 20:40] = 1.0
    assert is_mask_within_slice(interior, SLICE) is True

    clipped = np.zeros((SLICE, SLICE), np.float32)
    clipped[0:20, 20:40] = 1.0                              # touches the top edge
    assert is_mask_within_slice(clipped, SLICE) is False

    just_inside = np.zeros((SLICE, SLICE), np.float32)
    just_inside[4:40, 4:40] = 1.0                           # 3.5 >= 0.05*64 = 3.2
    assert is_mask_within_slice(just_inside, SLICE) is True

    just_outside = np.zeros((SLICE, SLICE), np.float32)
    just_outside[3:40, 3:40] = 1.0                          # 2.5 < 3.2
    assert is_mask_within_slice(just_outside, SLICE) is False


def test_clipped_masks_are_cardinally_biased_and_now_excluded():
    """Why the rule matters for a *cardinal-bias* test specifically: clipping an oblique boulder
    against the tile edge manufactures a straight axis-aligned edge and drags its angle toward
    the grid."""
    whole = _bar(SLICE, 32, 32, 22, 7, 40.0)                # comfortably interior
    clipped = _bar(SLICE, 32, 8, 22, 7, 40.0)               # same boulder, cut by the tile edge
    assert clipped[:, 0].any(), "fixture does not actually reach the tile boundary"
    a_whole = mask_readout(whole)[0]
    a_clipped = mask_readout(clipped)[0]
    assert is_mask_within_slice(whole, SLICE) is True
    assert is_mask_within_slice(clipped, SLICE) is False
    assert _axial_gap(a_whole, a_clipped) > 3.0, (a_whole, a_clipped)


# --- helper vs the public wrapper, detection by detection -------------------------------------------

class _FakeSeg:
    """A model that returns one fixed mask set per tile, at imgsz like ultralytics does."""

    def __init__(self, masks_by_tile, imgsz):
        self.masks_by_tile = masks_by_tile
        self.imgsz = imgsz
        self.calls = 0

    def _result(self, tile_idx):
        masks = self.masks_by_tile.get(tile_idx, [])
        boxes = np.zeros((len(masks), 6), dtype=np.float32)
        boxes[:, 4] = 0.9
        big = np.zeros((len(masks), self.imgsz, self.imgsz), dtype=np.float32)
        for i, m in enumerate(masks):
            rep = np.kron(m, np.ones((self.imgsz // m.shape[0],) * 2, dtype=np.float32))
            big[i] = rep
        return SimpleNamespace(boxes=SimpleNamespace(data=torch.from_numpy(boxes)),
                               masks=SimpleNamespace(data=torch.from_numpy(big)))

    def __call__(self, images, **kw):
        batch = images if isinstance(images, list) else [images]
        out = [self._result(self.calls + i) for i in range(len(batch))]
        self.calls += len(batch)
        return out


def test_helper_and_public_wrapper_agree_detection_by_detection(tmp_path):
    """The G2-02 acceptance criterion. The same masks, through the tiling helper and through
    `get_sliced_prediction`, must give the same angles — otherwise a null measured with the helper
    says nothing about what production emits."""
    from YOLOv8BeyondEarth.orientation import boulder_orientations

    imgsz = 256
    tiles = {0: [_bar(SLICE, 32, 32, 20, 7, 30.0)],
             1: [_bar(SLICE, 30, 34, 18, 6, 115.0)],
             2: [_bar(SLICE, 34, 30, 22, 8, 70.0)],
             3: [_bar(SLICE, 32, 32, 16, 6, 5.0)]}

    # --- helper path (in-memory array, non-overlapping tiles)
    img = np.full((2 * SLICE, 2 * SLICE), 200, dtype=np.uint8)
    helper = detect_orientations(_FakeSeg(tiles, imgsz), img, slice_size=SLICE, imgsz=imgsz,
                                 conf=0.1, device="cpu", aspect_min=0.0)

    # --- production path (real raster, real slicing/edge/NMS/georeferencing)
    scene = tmp_path / "scene.tif"
    with rio.open(scene, "w", driver="GTiff", height=2 * SLICE, width=2 * SLICE, count=1,
                  dtype="uint8", crs="EPSG:32631",
                  transform=from_origin(0.0, 2 * SLICE, 1.0, 1.0)) as dst:
        dst.write(np.full((1, 2 * SLICE, 2 * SLICE), 200, dtype=np.uint8))

    class _Slices:
        images = [np.full((SLICE, SLICE, 3), 200, np.uint8) for _ in range(4)]
        starting_pixels = [(0, 0), (SLICE, 0), (0, SLICE), (SLICE, SLICE)]

        def __len__(self):
            return 4

    import unittest.mock as mock
    with mock.patch.object(P, "slice_image", lambda **_: _Slices()):
        model = SimpleNamespace(model=_FakeSeg(tiles, imgsz), image_size=imgsz,
                                confidence_threshold=0.1, device="cpu",
                                category_mapping={"0": "boulder"}, model_path=None)
        gdf, _ = P.get_sliced_prediction(scene, model, output_dir=tmp_path / "out",
                                         slice_size=SLICE, inference_size=imgsz,
                                         downscale_pred=True, save_prenms=False, save_bbox=False,
                                         output_format="gpkg", postprocess=False)
    prod = boulder_orientations(gdf, res=1.0, n_workers=1)

    assert len(helper) == len(prod) == 4, (len(helper), len(prod))
    got = sorted(helper.angle180.tolist())
    want = sorted(prod.angle180.tolist())
    for a, b in zip(got, want):
        assert _axial_gap(a, b) < 1.0, f"helper {got} vs production {want}"


def test_helper_drops_the_edge_objects_production_would_have_dropped(tmp_path):
    imgsz = 256
    tiles = {0: [_bar(SLICE, 32, 32, 20, 7, 30.0),          # interior: kept
                 _bar(SLICE, 4, 32, 20, 7, 30.0)]}          # against the top edge: dropped
    img = np.full((SLICE, SLICE), 200, dtype=np.uint8)

    kept = detect_orientations(_FakeSeg(tiles, imgsz), img, slice_size=SLICE, imgsz=imgsz,
                               device="cpu", aspect_min=0.0)
    all_of_them = detect_orientations(_FakeSeg(tiles, imgsz), img, slice_size=SLICE, imgsz=imgsz,
                                      device="cpu", aspect_min=0.0, drop_edge=False)
    assert len(kept) == 1 and len(all_of_them) == 2


def test_detect_binary_masks_sees_the_same_population(tmp_path):
    imgsz = 256
    tiles = {0: [_bar(SLICE, 32, 32, 20, 7, 30.0), _bar(SLICE, 4, 32, 20, 7, 30.0)]}
    img = np.full((SLICE, SLICE), 200, dtype=np.uint8)
    masks = detect_binary_masks(_FakeSeg(tiles, imgsz), img, slice_size=SLICE, imgsz=imgsz,
                                device="cpu")
    angles = detect_orientations(_FakeSeg(tiles, imgsz), img, slice_size=SLICE, imgsz=imgsz,
                                 device="cpu", aspect_min=0.0)
    assert len(masks) == len(angles) == 1
