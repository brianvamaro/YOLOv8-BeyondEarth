"""Unit tests for the Goal-3 phase-5a post-fusion window re-masker (``remask.remask_windows``).

No GPU and no SAM weights: a fake SAM stands in for the model and returns masks we control, so
these pin the *plumbing* — pixel<->map round-tripping, window grouping, georeferencing, the
guard/floor bookkeeping, and detection parity. The thing they deliberately do NOT cover is whether
SAM produces the same mask in a window as it does in a 256 px slice — that is entry gate 2, a real
GPU measurement on the dev crops (plan §Phase-5 / 5a).
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
import rasterio as rio
from rasterio.transform import from_origin

from YOLOv8BeyondEarth import remask as R
from YOLOv8BeyondEarth.polygon import add_geometries


# --- fakes -----------------------------------------------------------------------------------

class _FakeTensor:
    """Mimics the ``r.masks.data.cpu().numpy()`` chain of an ultralytics result."""
    def __init__(self, arr):
        self._arr = arr

    def cpu(self):
        return self

    def numpy(self):
        return self._arr

    def __len__(self):
        return len(self._arr)


class _FakeMasks:
    def __init__(self, arr):
        self.data = _FakeTensor(arr)


class _FakeResult:
    def __init__(self, arr):
        self.masks = None if arr is None else _FakeMasks(arr)


class FakeSAM:
    """Returns, for each prompted box, a filled rectangle inset by ``inset`` px inside the box.

    ``blowup_boxes`` (indices into the per-call chunk) instead return a mask covering the whole
    window, which must trip the background-grab guard.
    """
    def __init__(self, inset=1, blowup_all=False, return_none=False, short_by=0, mutate=False):
        self.inset, self.blowup_all, self.return_none = inset, blowup_all, return_none
        self.short_by = short_by
        self.mutate = mutate            # imitate ultralytics' in-place `bboxes *= r` (G2-11)
        self.calls = []
        self.confs = []

    def __call__(self, image, bboxes=None, verbose=False, device=0, conf=None):
        h, w = image.shape[:2]
        if self.mutate and hasattr(bboxes, "__imul__"):
            bboxes *= 2.0               # what `_prepare_prompts` does to a shared buffer
        boxes = np.asarray(bboxes, dtype=float).reshape(-1, 4)
        self.calls.append((image.shape, len(boxes)))
        self.confs.append(conf)
        if self.return_none:
            return [_FakeResult(None)]
        out = np.zeros((len(boxes), h, w), dtype=np.uint8)
        for i, b in enumerate(boxes):
            if self.blowup_all:
                out[i, :, :] = 1
                continue
            x0 = int(np.ceil(b[0])) + self.inset
            y0 = int(np.ceil(b[1])) + self.inset
            x1 = int(np.floor(b[2])) - self.inset
            y1 = int(np.floor(b[3])) - self.inset
            out[i, max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = 1
        # Simulate the count-mismatch that forces per-box realignment. Only on multi-box calls:
        # the per-box retry must succeed, or the test would measure "SAM returned nothing" instead
        # of "the realignment path recovered alignment".
        if self.short_by and len(boxes) > 1:
            out = out[:-self.short_by] if len(out) > self.short_by else out[:0]
        return [_FakeResult(out)]


# --- fixtures --------------------------------------------------------------------------------

RES = 0.5
ORIGIN_X, ORIGIN_Y = 1000.0, 5000.0


@pytest.fixture
def scene(tmp_path):
    """A 600x600, 0.5 m/px, north-up 8-bit scene."""
    path = tmp_path / "scene.tif"
    transform = from_origin(ORIGIN_X, ORIGIN_Y, RES, RES)
    data = np.full((600, 600), 40, dtype=np.uint8)
    with rio.open(path, "w", driver="GTiff", height=600, width=600, count=1,
                  dtype="uint8", crs="EPSG:32633", transform=transform) as dst:
        dst.write(data, 1)
    return path


def _boxes_gdf(scene, pixel_boxes):
    """Build a fused-like catalog from axis-aligned boxes given in PIXEL coords."""
    polys = [np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], dtype=float)
             for (x0, y0, x1, y1) in pixel_boxes]
    df = pd.DataFrame({"polygon": polys,
                       "score": np.linspace(0.5, 0.9, len(polys)),
                       "scale_key": 256, "n_merged": 0})
    gdf = add_geometries(scene, df)
    return gdf.drop(columns=["polygon", "bbox"])


# --- pixel <-> map round trip -----------------------------------------------------------------

def test_pixel_bboxes_inverts_add_geometries(scene):
    pixel_boxes = [(10, 20, 30, 44), (100.0, 200.0, 118.0, 214.0), (500, 30, 540, 70)]
    gdf = _boxes_gdf(scene, pixel_boxes)
    with rio.open(scene) as src:
        pix = R._pixel_bboxes(gdf.geometry.bounds.to_numpy(), src.transform)
    np.testing.assert_allclose(pix, np.array(pixel_boxes, dtype=float), atol=1e-6)


def test_pixel_bboxes_handles_y_flip(scene):
    """A north-up raster flips y: the map miny corner is the pixel maxy corner."""
    gdf = _boxes_gdf(scene, [(10, 20, 30, 40)])
    with rio.open(scene) as src:
        pix = R._pixel_bboxes(gdf.geometry.bounds.to_numpy(), src.transform)
    assert pix[0, 1] < pix[0, 3]          # y0 < y1 despite the flip
    assert pix[0, 0] < pix[0, 2]


# --- window grouping ---------------------------------------------------------------------------

def test_group_windows_covers_every_box_exactly_once():
    boxes = np.array([[10, 10, 20, 20], [300, 12, 320, 30], [11, 400, 25, 420],
                      [260, 260, 275, 280], [12, 12, 18, 19]], dtype=float)
    groups = list(R._group_windows(boxes, window_size=256, pad=8,
                                   raster_width=600, raster_height=600))
    seen = np.concatenate([idx for _, idx in groups])
    assert sorted(seen.tolist()) == list(range(len(boxes)))
    # the two boxes in cell (0,0) share a window
    sizes = sorted(len(idx) for _, idx in groups)
    assert sizes == [1, 1, 1, 2]


def test_group_windows_interior_cell_is_exactly_window_size():
    """A boulder well inside its cell must not grow the window: SAM sees the production scale."""
    boxes = np.array([[300, 300, 320, 320]], dtype=float)
    (win, _), = R._group_windows(boxes, window_size=256, pad=8,
                                 raster_width=600, raster_height=600)
    _, _, w, h = win
    assert (w, h) == (256, 256)


def test_group_windows_grows_for_a_straddling_boulder():
    boxes = np.array([[250, 250, 262, 262]], dtype=float)   # straddles the 256 boundary
    (win, _), = R._group_windows(boxes, window_size=256, pad=4,
                                 raster_width=600, raster_height=600)
    col_off, row_off, w, h = win
    assert col_off + w >= 266 and row_off + h >= 266

def test_group_windows_clips_to_raster():
    boxes = np.array([[590, 590, 598, 598]], dtype=float)
    (win, _), = R._group_windows(boxes, window_size=256, pad=32,
                                 raster_width=600, raster_height=600)
    col_off, row_off, w, h = win
    assert col_off >= 0 and row_off >= 0
    assert col_off + w <= 600 and row_off + h <= 600


def test_group_windows_skips_empty_cells():
    boxes = np.array([[10, 10, 20, 20]], dtype=float)
    groups = list(R._group_windows(boxes, window_size=64, pad=0,
                                   raster_width=600, raster_height=600))
    assert len(groups) == 1        # not 600/64 squared


# --- end to end ----------------------------------------------------------------------------------

def test_remask_replaces_geometry_and_keeps_every_row(scene):
    pixel_boxes = [(20, 20, 40, 40), (300, 310, 326, 330), (100, 480, 124, 500)]
    gdf = _boxes_gdf(scene, pixel_boxes)
    out = R.remask_windows(scene, gdf, FakeSAM(inset=2), progress=False)

    assert len(out) == len(gdf)                      # detection parity
    assert list(out.index) == list(gdf.index)
    assert (out.mask_src == R.SRC_SAM).all()
    assert (out.remask_note == "").all()
    assert out.crs == gdf.crs
    assert "score" in out.columns and "scale_key" in out.columns   # provenance survives

    # the fake insets by 2 px on each side, so every SAM polygon sits strictly inside its box
    for i in range(len(out)):
        assert out.geometry.iloc[i].within(gdf.geometry.iloc[i])
    assert out.attrs["n_sam"] == 3
    assert out.attrs["n_prompted"] == 3


def test_remask_shifts_window_coords_back_to_the_raster(scene):
    """The window-offset shift is the easiest thing to get wrong here.

    A boulder at pixel ~500 sits in window (256, 256); forgetting to shift the traced polygon back
    would land it ~256 px (128 m) away, so a tight centroid check catches it outright.
    """
    gdf = _boxes_gdf(scene, [(500, 500, 524, 524)])
    out = R.remask_windows(scene, gdf, FakeSAM(inset=0), window_size=256, progress=False)
    assert out.attrs["n_sam"] == 1
    c_in, c_out = gdf.geometry.iloc[0].centroid, out.geometry.iloc[0].centroid
    assert abs(c_out.x - c_in.x) < 2 * RES
    assert abs(c_out.y - c_in.y) < 2 * RES
    assert 0.5 < out.geometry.iloc[0].area / gdf.geometry.iloc[0].area < 1.5


def test_remask_georeferencing_uses_the_production_path(scene):
    """Traced pixels must be georeferenced exactly as add_geometries would place them."""
    gdf = _boxes_gdf(scene, [(100, 100, 140, 140)])
    out = R.remask_windows(scene, gdf, FakeSAM(inset=0), progress=False)
    # inset=0 fills pixel rows/cols 100..139; the 0.5 isoline of that blob is 99.5..139.5, which
    # add_geometries maps through the same +0.5 pixel-centre convention.
    expect = add_geometries(scene, pd.DataFrame({"polygon": [
        np.array([[99.5, 99.5], [139.5, 99.5], [139.5, 139.5], [99.5, 139.5], [99.5, 99.5]],
                 dtype=float)]})).geometry.iloc[0]
    got = out.geometry.iloc[0]
    assert got.intersection(expect).area / expect.union(got).area > 0.99


def test_blowup_guard_keeps_yolo_geometry(scene):
    gdf = _boxes_gdf(scene, [(100, 100, 120, 120)])
    out = R.remask_windows(scene, gdf, FakeSAM(blowup_all=True), progress=False)
    assert out.mask_src.iloc[0] == R.SRC_YOLO
    assert out.remask_note.iloc[0] == "blowup"
    assert out.geometry.iloc[0].equals(gdf.geometry.iloc[0])     # untouched, not dropped
    assert out.attrs["n_sam"] == 0


def test_no_mask_keeps_yolo_geometry(scene):
    gdf = _boxes_gdf(scene, [(100, 100, 120, 120)])
    out = R.remask_windows(scene, gdf, FakeSAM(return_none=True), progress=False)
    assert out.mask_src.iloc[0] == R.SRC_YOLO
    assert out.remask_note.iloc[0] == "no_mask"
    assert out.geometry.iloc[0].equals(gdf.geometry.iloc[0])


def test_min_dpx_floor_skips_small_boulders(scene):
    # 8 px box -> ~9 px equivalent diameter; 3 px box -> ~3.4 px
    gdf = _boxes_gdf(scene, [(100, 100, 108, 108), (200, 200, 203, 203)])
    out = R.remask_windows(scene, gdf, FakeSAM(inset=0), min_dpx=6.0, progress=False)
    assert out.mask_src.tolist() == [R.SRC_SAM, R.SRC_YOLO]
    assert out.remask_note.tolist() == ["", "below_floor"]
    assert out.geometry.iloc[1].equals(gdf.geometry.iloc[1])
    assert out.attrs["n_prompted"] == 1        # the small one was never sent to SAM


def test_min_dpx_default_is_off(scene):
    gdf = _boxes_gdf(scene, [(200, 200, 203, 203)])
    out = R.remask_windows(scene, gdf, FakeSAM(inset=0), progress=False)
    assert out.attrs["min_dpx"] is None
    assert out.attrs["n_prompted"] == 1


def test_count_mismatch_triggers_realignment(scene):
    """A short SAM return must re-prompt per box, not silently misalign later boulders."""
    gdf = _boxes_gdf(scene, [(20, 20, 40, 40), (60, 60, 84, 84), (120, 30, 140, 50)])
    sam = FakeSAM(inset=1, short_by=1)
    out = R.remask_windows(scene, gdf, sam, progress=False)
    assert out.attrs["realigned_chunks"] >= 1
    assert (out.mask_src == R.SRC_SAM).all()
    for i in range(len(out)):
        assert out.geometry.iloc[i].within(gdf.geometry.iloc[i])


def test_empty_catalog(scene):
    gdf = _boxes_gdf(scene, [(10, 10, 20, 20)]).iloc[:0]
    out = R.remask_windows(scene, gdf, FakeSAM(), progress=False)
    assert len(out) == 0
    assert "mask_src" in out.columns
    assert out.attrs["n_windows"] == 0


# --- the shared helpers, from the in-slice G side -----------------------------------------------
# `sam2_remasker` had no test coverage before 2026-07-30, which made extracting `largest_blob_fill`
# / `sam_masks_in_order` out of it (so the window re-masker could share them) an unguarded change.
# These pin the in-slice contract that the extraction must not have altered.

def test_sam2_remasker_returns_one_mask_per_box():
    from YOLOv8BeyondEarth.orientation_validation import sam2_remasker
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    boxes = np.array([[20, 20, 40, 40], [100, 100, 130, 130]], dtype=float)
    masks = sam2_remasker(FakeSAM(inset=1))(image, boxes)
    assert len(masks) == 2
    for m, b in zip(masks, boxes):
        assert m.shape == (256, 256)
        assert m.dtype == np.float32
        assert set(np.unique(m)).issubset({0.0, 1.0})
        assert 0 < m.sum() < (b[2] - b[0]) * (b[3] - b[1])


def test_sam2_remasker_drops_guard_failures_in_slice():
    """The in-slice path *drops* (None) where the window path keeps the YOLO mask — deliberate."""
    from YOLOv8BeyondEarth.orientation_validation import sam2_remasker
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    masks = sam2_remasker(FakeSAM(blowup_all=True))(image, np.array([[20, 20, 40, 40]], float))
    assert masks == [None]


def test_sam2_remasker_empty_boxes():
    from YOLOv8BeyondEarth.orientation_validation import sam2_remasker
    assert sam2_remasker(FakeSAM())(np.zeros((64, 64, 3), np.uint8), np.zeros((0, 4))) == []


# --- the SAM box-prompt confidence, and the in-place box mutation (G2-11) -------------------------

def test_box_prompts_do_not_inherit_ultralytics_0_25_conf():
    """Ultralytics applies `conf` to the mask decoder's *predicted-IoU* head, so at its 0.25 default a
    prompted box SAM is merely unsure about comes back unmasked — the mask producer silently
    overruling the detector, in a pipeline whose whole premise is that it does not. It also makes the
    returned count disagree with the prompt count, which sends the chunk down the one-box-at-a-time
    realignment path: 1 + sam_batch encodes instead of 1."""
    from YOLOv8BeyondEarth.orientation_validation import (SAM_PROMPT_CONF, sam2_remasker,
                                                          sam_masks_in_order)
    assert SAM_PROMPT_CONF == 0.0

    sam = FakeSAM()
    sam_masks_in_order(sam, np.zeros((64, 64, 3), np.uint8), np.array([[10, 10, 30, 30]], float))
    assert sam.confs == [0.0], "conf was not forwarded to the model"

    sam2 = FakeSAM()
    sam2_remasker(sam2)(np.zeros((64, 64, 3), np.uint8), np.array([[10, 10, 30, 30]], float))
    assert sam2.confs == [0.0]

    explicit = FakeSAM()
    sam_masks_in_order(explicit, np.zeros((64, 64, 3), np.uint8),
                       np.array([[10, 10, 30, 30]], float), conf=0.25)
    assert explicit.confs == [0.25], "an explicit conf must still be honoured"


def test_prompt_boxes_survive_ultralytics_in_place_scaling():
    """`Predictor._prepare_prompts` does `torch.as_tensor(bboxes, ...)` then `bboxes *= r`, and
    `as_tensor` shares memory when dtype *and* device already match. Production escapes only because a
    host->device transfer copies — an accident of where the array lives, not a stated invariant. With
    more boxes than `sam_batch`, an in-place rescale corrupts the *second* chunk's prompts (G2-11)."""
    from YOLOv8BeyondEarth.orientation_validation import sam_masks_in_order

    boxes = np.array([[10, 10, 30, 30], [12, 12, 32, 32], [14, 14, 34, 34]], dtype=np.float32)
    before = boxes.copy()
    sam = FakeSAM(mutate=True)
    sam_masks_in_order(sam, np.zeros((64, 64, 3), np.uint8), boxes, sam_batch=2)

    np.testing.assert_array_equal(boxes, before)      # caller's array untouched
    assert len(sam.calls) == 2                        # 3 boxes at sam_batch=2 -> two chunks
    # Second chunk must see original-scale coordinates, not doubled ones. The fake doubles whatever
    # buffer it is handed, so a shared buffer would have the third box arrive at 4x.
    assert sam.calls[1][1] == 1


def test_largest_blob_fill_keeps_biggest_component_and_fills():
    from YOLOv8BeyondEarth.orientation_validation import largest_blob_fill
    m = np.zeros((20, 20), dtype=np.uint8)
    m[2:12, 2:12] = 1
    m[5:8, 5:8] = 0          # a hole, must be filled
    m[17:19, 17:19] = 1      # a smaller blob, must be dropped
    out = largest_blob_fill(m)
    assert out[6, 6] == 1
    assert out[17, 17] == 0
    assert out.sum() == 100
    assert largest_blob_fill(np.zeros((5, 5), dtype=np.uint8)) is None


def test_window_size_changes_sam_input_scale(scene):
    """Guards the documented behaviour that window_size sets SAM's input scale (gate-2 relevance)."""
    gdf = _boxes_gdf(scene, [(300, 300, 320, 320)])
    s256, s512 = FakeSAM(inset=1), FakeSAM(inset=1)
    R.remask_windows(scene, gdf, s256, window_size=256, progress=False)
    R.remask_windows(scene, gdf, s512, window_size=512, progress=False)
    assert s256.calls[0][0][:2] == (256, 256)
    assert s512.calls[0][0][:2] == (512, 512)


# --- defects found reviewing the unreviewed phase-5a file (2026-08-01) --------------------------

def test_pixel_bboxes_is_vectorized_and_unchanged(scene):
    """Vectorizing the inverse affine must not move a single corner. Phase 5a runs on full scenes
    (~1e6 boulders), where a Python-level affine call per corner is not a detail."""
    with rio.open(scene) as src:
        transform = src.transform
    rng = np.random.default_rng(0)
    x0 = rng.uniform(ORIGIN_X, ORIGIN_X + 200, 500)
    y0 = rng.uniform(ORIGIN_Y - 200, ORIGIN_Y, 500)
    bounds = np.stack([x0, y0 - 3.0, x0 + 4.0, y0], axis=-1)

    inv = ~transform
    ref = np.array([[*(np.minimum((inv * (b[0], b[1]))[0], (inv * (b[2], b[3]))[0]) - 0.5,),
                     *(np.minimum((inv * (b[0], b[1]))[1], (inv * (b[2], b[3]))[1]) - 0.5,),
                     *(np.maximum((inv * (b[0], b[1]))[0], (inv * (b[2], b[3]))[0]) - 0.5,),
                     *(np.maximum((inv * (b[0], b[1]))[1], (inv * (b[2], b[3]))[1]) - 0.5,)]
                    for b in bounds])
    np.testing.assert_allclose(R._pixel_bboxes(bounds, transform), ref, atol=1e-9)


def test_prior_mask_src_is_not_relabelled_as_yolo(scene):
    """Since G2-03 the sliced path also writes mask_src, so a fused catalog can already hold SAM2
    geometry. A row this pass does not replace must keep its real provenance, not be flattened to
    "yolo"."""
    gdf = _boxes_gdf(scene, [(60, 60, 80, 80), (300, 300, 320, 320)])
    gdf["mask_src"] = ["sam2", "yolo"]                      # first came from the in-slice path
    out = R.remask_windows(scene, gdf, FakeSAM(blowup_all=True), progress=False)

    assert (out.remask_note == "blowup").all(), "fixture should fail the guard for every row"
    assert list(out.mask_src) == ["sam2", "yolo"], "prior provenance was overwritten"


def test_mask_src_defaults_to_yolo_when_the_catalog_has_no_column(scene):
    gdf = _boxes_gdf(scene, [(60, 60, 80, 80)])
    out = R.remask_windows(scene, gdf, FakeSAM(blowup_all=True), progress=False)
    assert list(out.mask_src) == ["yolo"]


def test_successful_remask_still_claims_sam2(scene):
    gdf = _boxes_gdf(scene, [(60, 60, 80, 80)])
    gdf["mask_src"] = ["yolo"]
    out = R.remask_windows(scene, gdf, FakeSAM(inset=1), progress=False)
    assert list(out.mask_src) == ["sam2"] and list(out.remask_note) == [""]


def test_degenerate_prompt_box_is_named_not_mislabelled_as_blowup(scene):
    """A zero-extent box used to be compared against an epsilon area, so any returned mask read as
    a blow-up and the real cause never reached the notes."""
    gdf = _boxes_gdf(scene, [(100.0, 100.0, 100.0, 100.0)])
    out = R.remask_windows(scene, gdf, FakeSAM(inset=0), progress=False)
    assert list(out.remask_note) == ["degenerate_box"]
    assert list(out.mask_src) == ["yolo"]


def test_oversize_boulder_does_not_rescale_its_neighbours_window(scene):
    """SAM's encoder resizes its input, so window size IS the effective resolution. One outsized
    fused polygon used to grow its cell without limit, silently re-masking its neighbours at a
    different scale than the validated 256."""
    small = (20, 20, 34, 34)
    huge = (5, 5, 500, 500)                                  # spans most of the raster
    gdf = _boxes_gdf(scene, [small, huge])

    sam = FakeSAM(inset=1)
    out = R.remask_windows(scene, gdf, sam, window_size=64, pad=4, max_window_growth=2.0,
                           progress=False)
    assert out.remask_note.iloc[1] == "oversize_window"
    assert out.mask_src.iloc[1] == "yolo"
    assert out.geometry.iloc[1].equals(gdf.geometry.iloc[1]), "the huge row's geometry changed"
    # ...and the small neighbour was still re-masked, at a window no bigger than the cap
    assert out.mask_src.iloc[0] == "sam2"
    assert max(shape[0] for shape, _ in sam.calls) <= 2 * 64
    assert out.attrs["n_oversize"] == 1


def test_growth_cap_can_be_disabled(scene):
    gdf = _boxes_gdf(scene, [(20, 20, 34, 34), (5, 5, 500, 500)])
    sam = FakeSAM(inset=1)
    out = R.remask_windows(scene, gdf, sam, window_size=64, pad=4, max_window_growth=None,
                           progress=False)
    assert "oversize_window" not in set(out.remask_note)
    assert max(shape[0] for shape, _ in sam.calls) > 2 * 64   # the old unbounded behaviour
