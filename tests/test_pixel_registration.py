"""Absolute georegistration of the rescaling paths (review item G3-10).

Two coordinate rescales happen between a model mask and a map polygon, and both used the naive
``p * s`` instead of the centre-aware ``(p + 0.5) * s - 0.5``:

- the streamed-SR arm mapping SR px back onto the native grid (``s = 1/f``);
- ``downscale_pred=False`` mapping the model's mask frame onto the slice — live in every config,
  because ultralytics returns masks at ``imgsz`` (1024) rather than at ``slice_size``.

Each leaves a constant ``0.5 * (1 - s)`` offset in both axes. It is a rigid translation, so sizes
and angles survive it and only *absolute position* is wrong — which is precisely what a
cross-scene, cross-body map (Goal 4) depends on. These tests check absolute centroids of masks
whose true position is known by construction, which size- or IoU-based checks cannot see.
"""
import numpy as np
import pytest
import rasterio as rio
import torch
from rasterio.transform import from_origin
from types import SimpleNamespace

from YOLOv8BeyondEarth import predict as P
from YOLOv8BeyondEarth.polygon import rescale_pixel_coords


def test_rescale_is_centre_aware():
    """A pixel index names a pixel whose centre is at index + 0.5."""
    # a 2x downscale: mask pixels 0 and 1 both fall inside slice pixel 0, centred on its centre
    np.testing.assert_allclose(rescale_pixel_coords(np.array([[0.0, 0.0]]), 0.5), [[-0.25, -0.25]])
    np.testing.assert_allclose(rescale_pixel_coords(np.array([[1.0, 1.0]]), 0.5), [[0.25, 0.25]])
    # scale 1 is exactly the identity, including object identity (no copy on the hot path)
    coords = np.array([[3.0, 4.0]])
    assert rescale_pixel_coords(coords, 1.0) is coords


@pytest.mark.parametrize("scale", [0.25, 0.5, 2.0, 4.0])
def test_rescale_matches_cv2_resize_convention(scale):
    """cv2.resize maps dst -> src as (d + 0.5)/scale - 0.5; the inverse must be our forward map."""
    src = np.array([0.0, 1.0, 7.0, 63.0])
    dst = rescale_pixel_coords(np.stack([src, src], axis=-1), scale)[:, 0]
    back = (dst + 0.5) / scale - 0.5
    np.testing.assert_allclose(back, src, atol=1e-12)


def test_rescale_round_trips():
    coords = np.array([[0.0, 0.0], [10.0, 3.0], [255.0, 128.0]])
    there = rescale_pixel_coords(coords, 0.25)
    np.testing.assert_allclose(rescale_pixel_coords(there, 4.0), coords, atol=1e-12)


def test_naive_scaling_offset_is_the_documented_amount():
    """Pin the two magnitudes the review names, so a regression is recognisable."""
    p = np.array([[100.0, 100.0]])
    for scale, expected in ((0.5, 0.25),        # streamed SR at f = 2
                            (256 / 1024, 0.375)):   # a 1024 px mask on a 256 px slice
        naive = p * scale
        correct = rescale_pixel_coords(p, scale)
        assert float(naive[0, 0] - correct[0, 0]) == pytest.approx(expected, abs=1e-12)


# --- end to end: a known mask must land on its known ground position -------------------------------

SLICE = 64


class _Slices:
    def __init__(self, image):
        self.images = [image]
        self.starting_pixels = [(0, 0)]

    def __len__(self):
        return 1


def _one_result(mask, score=0.9):
    boxes = np.array([[0, 0, 1, 1, score, 0]], dtype=np.float32)
    return SimpleNamespace(
        boxes=SimpleNamespace(data=torch.from_numpy(boxes)),
        masks=SimpleNamespace(data=torch.from_numpy(mask[None].astype(np.float32))))


@pytest.fixture
def scene(tmp_path):
    """North-up, 1 map unit per px, origin at (0, 128) — so native px (col, row) has its centre at
    map (col + 0.5, 128 - row - 0.5)."""
    path = tmp_path / "scene.tif"
    with rio.open(path, "w", driver="GTiff", height=128, width=128, count=1, dtype="uint8",
                  crs="EPSG:32631", transform=from_origin(0.0, 128.0, 1.0, 1.0)) as dst:
        dst.write(np.full((1, 128, 128), 200, dtype=np.uint8))
    return path


def _centroid_of_single_detection(scene, mask, monkeypatch, tmp_path, **kw):
    image = np.full((SLICE, SLICE, 3), 200, np.uint8)
    monkeypatch.setattr(P, "slice_image", lambda **_: _Slices(image))
    model = SimpleNamespace(model=lambda images, **k: [_one_result(mask)], image_size=256,
                            confidence_threshold=0.1, device="cpu",
                            category_mapping={"0": "boulder"}, model_path=None)
    opts = dict(output_dir=tmp_path / "out", slice_size=SLICE, save_prenms=False,
                save_bbox=False, output_format="gpkg", on_conflict="overwrite")
    opts.update(kw)
    gdf, _ = P.get_sliced_prediction(scene, model, **opts)
    assert len(gdf) == 1, f"expected exactly one detection, got {len(gdf)}"
    return gdf.geometry.iloc[0].centroid


def _blob(size, row, col, extent):
    m = np.zeros((size, size), dtype=np.float32)
    m[row:row + extent, col:col + extent] = 1.0
    return m


def test_absolute_centroid_native_path(scene, tmp_path, monkeypatch):
    """Mask frame == slice frame: no rescale, so this is the reference the others must reproduce."""
    c = _centroid_of_single_detection(scene, _blob(SLICE, 20, 30, 4), monkeypatch, tmp_path)
    # native px cols 30..33, rows 20..23 -> centre at col 32.0, row 22.0 in edge coords
    assert c.x == pytest.approx(32.0, abs=1e-6)
    assert c.y == pytest.approx(128.0 - 22.0, abs=1e-6)


def test_absolute_centroid_survives_a_mask_frame_rescale(scene, tmp_path, monkeypatch):
    """The same ground object, seen through a 4x larger mask frame (ultralytics returns masks at
    imgsz). `downscale_pred=False` rescales coordinates; the centroid must not move."""
    c = _centroid_of_single_detection(scene, _blob(SLICE * 4, 80, 120, 16), monkeypatch, tmp_path,
                                      downscale_pred=False)
    assert c.x == pytest.approx(32.0, abs=1e-6)
    assert c.y == pytest.approx(128.0 - 22.0, abs=1e-6)


def test_absolute_centroid_agrees_between_downscale_modes(scene, tmp_path, monkeypatch):
    """`downscale_pred` resizes the mask (cv2, already centre-correct) while its opposite rescales
    the coordinates. They must land in the same place; before the fix they differed by 0.375 px."""
    big = _blob(SLICE * 4, 80, 120, 16)
    a = _centroid_of_single_detection(scene, big, monkeypatch, tmp_path, downscale_pred=False)
    b = _centroid_of_single_detection(scene, big, monkeypatch, tmp_path, downscale_pred=True)
    assert a.x == pytest.approx(b.x, abs=1e-6)
    assert a.y == pytest.approx(b.y, abs=1e-6)


def test_absolute_centroid_streamed_sr(scene, tmp_path, monkeypatch):
    """A boulder occupying native px (10..11, 15..16) is a 4x4 block in an SR x2 slice. Mapped back
    it must sit on the native pixels it came from, not 0.25 px away."""
    # sr_factor=2 with slice_size=64 means a 32 px native window upscaled to 64 SR px;
    # native cols 15..16 -> SR cols 30..33, native rows 10..11 -> SR rows 20..23
    c = _centroid_of_single_detection(scene, _blob(SLICE, 20, 30, 4), monkeypatch, tmp_path,
                                      sr_factor=2)
    # native cols 15..16 centre 16.0, rows 10..11 centre 11.0
    assert c.x == pytest.approx(16.0, abs=1e-6)
    assert c.y == pytest.approx(128.0 - 11.0, abs=1e-6)


def test_streamed_sr_and_native_agree_on_the_same_ground_object(scene, tmp_path, monkeypatch):
    """The cross-scale check that matters for fusion: the SR arm and the native arm must place one
    ground object at the same absolute position, or every cross-arm DIoU is biased."""
    native = _centroid_of_single_detection(scene, _blob(SLICE, 10, 15, 2), monkeypatch, tmp_path)
    sr = _centroid_of_single_detection(scene, _blob(SLICE, 20, 30, 4), monkeypatch, tmp_path,
                                       sr_factor=2)
    assert sr.x == pytest.approx(native.x, abs=1e-6)
    assert sr.y == pytest.approx(native.y, abs=1e-6)
