"""Levers N1 and N2 change *how* the prediction path reads a raster, and must not change *what it sees*.

N2 replaced three whole-raster materialisations: the TIFF->PNG copy, sahi's decoded RGB scene (all
slices are views into it), and ``rastertools.raster.true_footprint``'s int64 mask. N1 stopped running
the model on slices that are entirely fill. Each test compares a new piece against the **old code
itself** (sahi's ``slice_image`` on the PNG ``tiff_to_png`` writes; ``true_footprint``), not against a
re-derivation of it, because the claim is identity with what shipped.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio as rio
import shapely
import torch
from rasterio.transform import from_origin
from rastertools_BOULDERING import convert, raster as rt_raster
from sahi.slicing import slice_image

from YOLOv8BeyondEarth import coverage
from YOLOv8BeyondEarth import predict as P

MARS_CRS = ("+proj=eqc +lat_ts=45 +lat_0=0 +lon_0=180 +x_0=0 +y_0=0 "
            "+R=3386150.7470034 +units=m +no_defs")
#: ESP_028537_2270's real origin and scale, so coordinate arithmetic runs at realistic magnitudes
ORIGIN = from_origin(-6736886.25, 2755670.0, 0.25, 0.25)


def _write(path, arr, transform=ORIGIN, crs=MARS_CRS):
    with rio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                  dtype="uint8", crs=crs, transform=transform, nodata=0) as dst:
        dst.write(arr, 1)
    return path


def _swath(h, w, angle_deg=13.0, half_width=0.32, seed=0):
    """A HiRISE-like product: a rotated strip of data in a north-up box, fill triangles around it."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(1, 256, size=(h, w)).astype(np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    t = np.deg2rad(angle_deg)
    u = (xx - w / 2) * np.cos(t) + (yy - h / 2) * np.sin(t)
    arr[np.abs(u) > half_width * w] = 0
    return arr


# --- N2: the windows are sahi's slices --------------------------------------------------------------

@pytest.mark.parametrize("shape, slice_size, overlap", [
    ((128, 128), 64, 0.2),
    ((300, 517), 256, 0.2),        # neither dimension a multiple of the stride: shifted last slices
    ((517, 300), 128, 0.2),        # the SR arm's native window
    ((90, 100), 256, 0.2),         # smaller than one slice
    ((257, 513), 256, 0.0),
])
def test_windowed_slices_equal_sahis_slices_of_the_old_png(tmp_path, shape, slice_size, overlap):
    tif = _write(tmp_path / "scene.tif", _swath(*shape))
    png = tmp_path / "scene.png"
    convert.tiff_to_png(tif, png)                       # the old path's whole-raster copy
    old = slice_image(image=str(png), slice_height=slice_size, slice_width=slice_size,
                      overlap_height_ratio=overlap, overlap_width_ratio=overlap)
    new = P._WindowedSlices(tif, slice_size, overlap, overlap)
    try:
        assert len(new) == len(old)
        assert [list(p) for p in new.starting_pixels] == [list(p) for p in old.starting_pixels]
        for i in range(len(old)):
            a, b = np.asarray(old.images[i]), new.images[i]
            assert a.dtype == b.dtype == np.uint8 and a.shape == b.shape
            assert np.array_equal(a, b), f"slice {i} differs"
    finally:
        new.close()


def test_the_prediction_path_refuses_what_the_old_png_hop_silently_mangled(tmp_path):
    """A 10-bit raster used to reach the model clipped at DN 255; a 3-band one crashed in tiff_to_png."""
    ten_bit = tmp_path / "ten_bit.tif"
    with rio.open(ten_bit, "w", driver="GTiff", height=8, width=8, count=1, dtype="uint16",
                  crs=MARS_CRS, transform=ORIGIN) as dst:
        dst.write(np.full((1, 8, 8), 700, np.uint16))
    rgb = tmp_path / "rgb.tif"
    with rio.open(rgb, "w", driver="GTiff", height=8, width=8, count=3, dtype="uint8",
                  crs=MARS_CRS, transform=ORIGIN) as dst:
        dst.write(np.full((3, 8, 8), 7, np.uint8))
    model = SimpleNamespace(model=None, device="cpu", category_mapping={"0": "boulder"})
    for path, what in ((ten_bit, "uint16"), (rgb, "3 band")):
        with pytest.raises(ValueError, match="single-band uint8"):
            P.get_sliced_prediction(path, model, output_dir=tmp_path / "out")


# --- N2: the streamed hull is true_footprint's hull ----------------------------------------------------

def _patterns():
    rng = np.random.default_rng(7)
    swath = _swath(400, 260)
    holes = swath.copy()
    holes[150:190, 100:140] = 0                          # interior fill: invisible to a hull
    specks = swath.copy()
    specks[rng.integers(0, 400, 12), rng.integers(0, 260, 12)] = 9   # stray valid pixels in the fill
    full = np.full((90, 70), 5, np.uint8)
    single = np.zeros((50, 60), np.uint8)
    single[17, 41] = 1
    diagonal = np.zeros((64, 64), np.uint8)
    diagonal[np.arange(5, 60, 3), np.arange(5, 60, 3)] = 3
    return {"swath": swath, "holes": holes, "specks": specks, "full": full, "single": single,
            "diagonal": diagonal}


@pytest.mark.parametrize("name", sorted(_patterns()))
def test_streamed_hull_is_true_footprints_hull_byte_for_byte(tmp_path, name):
    arr = _patterns()[name]
    tif = _write(tmp_path / f"{name}.tif", arr)
    old = rt_raster.true_footprint(tif, tmp_path / f"{name}-fp.shp").geometry.iloc[0]
    for block_rows in (1, 7, 1024):                      # block seams must not matter
        new = coverage.valid_convex_hull(tif, block_rows=block_rows)
        assert shapely.to_wkb(new) == shapely.to_wkb(old), (name, block_rows)


def test_streamed_hull_of_an_all_fill_raster_is_empty(tmp_path):
    tif = _write(tmp_path / "empty.tif", np.zeros((40, 40), np.uint8))
    assert coverage.valid_convex_hull(tif).is_empty


# --- N1: fill slices are skipped, nothing else changes ---------------------------------------------

class _ListSlices:
    def __init__(self, images):
        self.images = images
        self.starting_pixels = [[i, 0] for i in range(len(images))]

    def __len__(self):
        return len(self.images)


def test_kept_batches_skip_exactly_the_slices_that_are_entirely_fill():
    z = np.zeros((4, 4, 3), np.uint8)
    edge = z.copy()
    edge[0, 0] = 1                                       # one valid pixel: must be inferred
    shadow = np.full((4, 4, 3), 3, np.uint8)
    images = [z, edge, z, shadow, z, z, edge, shadow, z, shadow]
    counts = {}
    batches = list(P._kept_batches(_ListSlices(images), 2, True, counts))
    assert [b[0] for b in batches] == [[1, 3], [6, 7], [9]]
    assert counts == {"skipped_fill": 5}
    assert len(batches[-1][1]) == 2 and batches[-1][1][1].max() == 0, "short batch padded with fill"
    everything = list(P._kept_batches(_ListSlices(images), 4, False, {}))
    assert [b[0] for b in everything] == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert [len(b[1]) for b in everything] == [4, 4, 2], "nothing skipped: exactly the old batching"


def test_every_kept_slice_runs_at_the_batch_size_it_had_before():
    """A batch's *size* moves results (~1e-4 in score); its composition does not. So N1 re-batches the
    kept slices but restores each one's original batch size: regular ones in full batches, the grid's
    own final partial batch at its own size."""
    z, v = np.zeros((4, 4, 3), np.uint8), np.full((4, 4, 3), 9, np.uint8)
    #           regular stretch (8 = 2 x 4)    | the grid's tail (old batch of 3)
    images = [v, z, v, z, z, v, z, z, z, v, v]
    batches = list(P._kept_batches(_ListSlices(images), 4, True, {}))
    assert [b[0] for b in batches] == [[0, 2, 5], [9, 10]]
    assert [len(b[1]) for b in batches] == [4, 3]
    for idx, imgs in batches:
        assert all(im.max() == 0 for im in imgs[len(idx):]), "pads are fill"


def _content_model(calls):
    """Detects one 'boulder' wherever a slice has any bright pixel -- and nothing on fill, which is what
    the production weights do on an all-zero slice (measured 2026-10-14)."""
    def model(images, **kw):
        calls.append(len(images))
        out = []
        for img in images:
            ys, xs = np.nonzero(img[..., 0] > 128)
            if len(ys):
                x0, y0 = int(xs.min()), int(ys.min())
                x1, y1 = min(x0 + 6, img.shape[1] - 1), min(y0 + 6, img.shape[0] - 1)
                boxes = torch.tensor([[x0, y0, x1, y1, 0.9, 0]], dtype=torch.float32)
                masks = torch.zeros((1, img.shape[0], img.shape[1]))
                masks[0, y0:y1 + 1, x0:x1 + 1] = 1.0
            else:
                boxes, masks = torch.zeros((0, 6)), torch.zeros((0, img.shape[0], img.shape[1]))
            out.append(SimpleNamespace(boxes=SimpleNamespace(data=boxes),
                                       masks=SimpleNamespace(data=masks)))
        return out
    return SimpleNamespace(model=model, image_size=64, confidence_threshold=0.1, device="cpu",
                           category_mapping={"0": "boulder"}, model_path=None)


def test_skipping_fill_slices_changes_no_detection(tmp_path):
    arr = _swath(320, 240, half_width=0.22)
    tif = _write(tmp_path / "scene.tif", arr)
    runs = {}
    for skip in (False, True):
        calls = []
        gdf, nms = P.get_sliced_prediction(
            tif, _content_model(calls), output_dir=tmp_path / f"skip-{skip}", slice_size=32,
            batch_size=4, save_prenms=False, save_bbox=False, output_format="gpkg", n_workers=1,
            skip_fill_slices=skip)
        runs[skip] = (nms, calls, gdf.attrs["slices"])
    (off, calls_off, s_off), (on, calls_on, s_on) = runs[False], runs[True]
    assert s_off["n_slices"] == s_on["n_slices"] and s_off["n_slices_skipped_fill"] == 0
    assert s_on["n_slices_skipped_fill"] > 0, "the fixture must have fill-only slices"
    kept, tail = s_on["n_slices"] - s_on["n_slices_skipped_fill"], s_on["n_slices"] % 4
    assert all(c == 4 for c in calls_on[:-1]) and calls_on[-1] in (4, tail), \
        "every batch keeps the size it had without N1"
    assert kept <= sum(calls_on) < kept + 2 * 4, "only the pads are extra: under two batches of them"
    assert sum(calls_on) < sum(calls_off), "and N1 still skips"
    assert len(on) == len(off) > 0
    key = lambda g: sorted(zip(shapely.to_wkb(g.geometry.values), g.score, g.is_at_edge))
    assert key(on) == key(off)
