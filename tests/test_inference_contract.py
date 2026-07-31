"""Acceptance tests for the inference/output contract (review items X-01 and X-02).

X-01: the requested ``confidence_threshold`` must reach ultralytics as ``conf=``. It did not,
so every run used ultralytics' own 0.25 default and every catalog named ``ct-010`` is really a
0.25 catalog.

X-02: ``score`` must be the detector's real confidence for *every* row. It used to be
overwritten with the literal 0.10 for slice-truncated detections, which destroyed the
calibration and put sub-threshold rows in catalogs. Suppression ordering now lives in
``nms_prio``.

No GPU and no real weights: the model is a recording stand-in that filters by ``conf`` exactly
as ultralytics does, but the raster, footprint, edge logic, NMS and writers are the real ones.
"""
import numpy as np
import pytest
import rasterio as rio
import torch
from rasterio.transform import from_origin
from types import SimpleNamespace

from YOLOv8BeyondEarth import predict as P
from YOLOv8BeyondEarth.predict import (INFERENCE_CONTRACT_VERSION, _result_to_df, edge_priority)


SLICE = 64


# --- stand-ins -----------------------------------------------------------------------------------

class FakeUltralytics:
    """Records every call and applies ``conf`` the way ultralytics does.

    ``per_slice`` is a list (one entry per slice, in slice order) of ``[(x0, y0, x1, y1, score)]``
    mask rectangles in *slice* pixel coordinates.
    """

    def __init__(self, per_slice):
        self.per_slice = per_slice
        self.calls = []

    def __call__(self, images, imgsz=None, conf=None, verbose=None, device=None):
        batch = images if isinstance(images, list) else [images]
        self.calls.append({"conf": conf, "imgsz": imgsz, "device": device,
                           "n_images": len(batch), "images": batch})
        results = []
        for _ in batch:
            rects = self.per_slice[len(results) + sum(c["n_images"] for c in self.calls[:-1])]
            # ultralytics drops anything below conf before we ever see it
            rects = [r for r in rects if conf is None or r[4] >= conf]
            boxes = np.zeros((len(rects), 6), dtype=np.float32)
            masks = np.zeros((len(rects), SLICE, SLICE), dtype=np.float32)
            for i, (x0, y0, x1, y1, score) in enumerate(rects):
                boxes[i] = [x0, y0, x1, y1, score, 0]
                masks[i, y0:y1 + 1, x0:x1 + 1] = 1.0
            results.append(SimpleNamespace(
                boxes=SimpleNamespace(data=torch.from_numpy(boxes)),
                masks=SimpleNamespace(data=torch.from_numpy(masks))))
        return results


def _fake_model(per_slice, conf):
    return SimpleNamespace(model=FakeUltralytics(per_slice), image_size=256,
                           confidence_threshold=conf, device="cpu",
                           category_mapping={"0": "boulder"})


class _FakeSlices:
    """Stand-in for ``sahi.slicing.slice_image``'s result, so slice placement is ours to choose."""

    def __init__(self, n, shifts):
        self.images = [np.full((SLICE, SLICE, 3), 200, dtype=np.uint8) for _ in range(n)]
        self.starting_pixels = shifts

    def __len__(self):
        return len(self.images)


@pytest.fixture
def raster_128(tmp_path):
    """A 128x128 north-up GeoTIFF, all non-zero so the true footprint is the whole rectangle."""
    path = tmp_path / "scene.tif"
    with rio.open(path, "w", driver="GTiff", height=128, width=128, count=1, dtype="uint8",
                  crs="EPSG:32631", transform=from_origin(0.0, 128.0, 1.0, 1.0)) as dst:
        dst.write(np.full((1, 128, 128), 200, dtype=np.uint8))
    return path


def _run(raster, per_slice, shifts, conf, monkeypatch, **kw):
    monkeypatch.setattr(P, "slice_image",
                        lambda **_: _FakeSlices(len(per_slice), shifts))
    model = _fake_model(per_slice, conf)
    opts = dict(confidence_threshold=conf, output_dir=raster.parent, slice_size=SLICE,
                inference_size=256, save_prenms=False, save_bbox=False,
                output_format="gpkg", n_workers=1)
    opts.update(kw)
    gdf, gdf_nms = P.get_sliced_prediction(raster, model, **opts)
    return model.model, gdf, gdf_nms


# --- X-01: the requested threshold reaches the model ---------------------------------------------

def test_single_image_path_forwards_conf_and_flips_to_bgr():
    """``YOLOv8`` (the non-batched entry point) must pass ``conf`` and hand ultralytics BGR."""
    model = _fake_model([[(10, 10, 20, 20, 0.4)]], conf=0.17)
    image = np.zeros((SLICE, SLICE, 3), dtype=np.uint8)
    image[..., 0] = 10   # R
    image[..., 1] = 20   # G
    image[..., 2] = 30   # B
    P.YOLOv8(model, image, has_mask=True, shift_amount=(0, 0), slice_size=SLICE,
             min_area_threshold=None, downscale_pred=False)
    call = model.model.calls[0]
    assert call["conf"] == pytest.approx(0.17)
    passed = call["images"][0] if isinstance(call["images"], list) else call["images"]
    assert passed[0, 0].tolist() == [30, 20, 10], "RGB was not flipped to BGR"


def test_batched_path_forwards_conf(raster_128, monkeypatch):
    """The batched call in ``get_sliced_prediction`` is the production path and was the one
    silently running at ultralytics' 0.25 default."""
    per_slice = [[(20, 20, 30, 30, 0.40)], [(20, 20, 30, 30, 0.40)]]
    fake, _, _ = _run(raster_128, per_slice, [(0, 0), (32, 0)], 0.10, monkeypatch)
    assert len(fake.calls) == 1 and fake.calls[0]["n_images"] == 2   # one batched call
    assert fake.calls[0]["conf"] == pytest.approx(0.10)
    assert fake.calls[0]["imgsz"] == 256


def test_detection_between_010_and_025_is_retained(raster_128, monkeypatch):
    """The X-01 acceptance fixture: a 0.15-confidence boulder exists in a ct-010 catalog and
    is absent from a ct-025 one. Before the fix both catalogs were identical, because
    ultralytics filtered at 0.25 either way."""
    per_slice = [[(20, 20, 30, 30, 0.15), (40, 40, 50, 50, 0.80)]]

    _, gdf10, _ = _run(raster_128, per_slice, [(0, 0)], 0.10, monkeypatch)
    assert sorted(np.round(gdf10.score.values, 2)) == [0.15, 0.80]
    assert ((gdf10.score > 0.10) & (gdf10.score < 0.25)).sum() == 1

    _, gdf25, _ = _run(raster_128, per_slice, [(0, 0)], 0.25, monkeypatch)
    assert sorted(np.round(gdf25.score.values, 2)) == [0.80]


# --- X-02: score is real confidence; priority is its own field -----------------------------------

def test_edge_priority_orders_untruncated_above_truncated():
    score = np.array([0.10, 0.50, 1.00])
    interior = edge_priority(score, np.ones(3, bool))
    truncated = edge_priority(score, np.zeros(3, bool))
    assert interior.min() > truncated.max()          # every whole beats every clipped copy
    assert (truncated > 0).all()                     # lsnms silently drops score <= 0
    assert np.all(np.diff(interior) > 0) and np.all(np.diff(truncated) > 0)  # monotone in score


def test_result_to_df_keeps_real_confidence_for_truncated_detections():
    """Per-slice unit check: the truncated detection keeps 0.93 in ``score`` and is demoted
    only in ``nms_prio``. Pre-fix, ``score`` was the literal 0.10."""
    rects = [(0, 20, 15, 35, 0.93),      # touches the slice edge -> truncated
             (25, 25, 40, 40, 0.31)]     # comfortably interior
    boxes = np.zeros((2, 6), dtype=np.float32)
    masks = np.zeros((2, SLICE, SLICE), dtype=np.float32)
    for i, (x0, y0, x1, y1, s) in enumerate(rects):
        boxes[i] = [x0, y0, x1, y1, s, 0]
        masks[i, y0:y1 + 1, x0:x1 + 1] = 1.0
    result = SimpleNamespace(boxes=SimpleNamespace(data=torch.from_numpy(boxes)),
                             masks=SimpleNamespace(data=torch.from_numpy(masks)))
    df = _result_to_df(result, _fake_model([], 0.1), True, (0, 0), SLICE, None, False)

    assert list(df.columns) == ['score', 'nms_prio', 'polygon', 'category_id', 'category_name',
                                'is_within_slice', 'mask_src']
    assert set(df.mask_src) == {"yolo"}
    trunc = df[~df.is_within_slice].iloc[0]
    whole = df[df.is_within_slice].iloc[0]
    assert trunc.score == pytest.approx(0.93, abs=1e-6)
    assert trunc.nms_prio < 0.5 < whole.nms_prio
    assert whole.score == pytest.approx(0.31, abs=1e-6)
    assert whole.nms_prio == pytest.approx(0.5 + 0.5 * 0.31, abs=1e-6)


def test_threshold_invariant_holds_for_edge_and_interior(raster_128, monkeypatch):
    """No row in a catalog may sit below the requested threshold — including the truncated
    rows kept because they touch the raster's true footprint edge. This is the invariant the
    hard-coded 0.10 broke: at conf=0.5 those rows shipped as 0.10."""
    per_slice = [[(0, 20, 15, 35, 0.90),      # truncated at the slice edge == raster edge
                  (25, 25, 40, 40, 0.60)]]    # interior
    _, gdf, gdf_nms = _run(raster_128, per_slice, [(0, 0)], 0.50, monkeypatch)

    assert len(gdf) == 2
    assert (gdf.score >= 0.50).all(), "a sub-threshold score survived into the catalog"
    assert not gdf.is_within_slice.all(), "expected one truncated row to be kept at the edge"
    assert gdf.loc[~gdf.is_within_slice, "score"].iloc[0] == pytest.approx(0.90, abs=1e-6)
    assert (gdf_nms.score >= 0.50).all()


def test_untruncated_member_wins_the_duplicate_cluster(raster_128, monkeypatch):
    """The behaviour the 0.10 sentinel existed to produce, now carried by ``nms_prio``: the
    whole copy of a boulder beats the clipped copy even though the clipped one scored higher."""
    per_slice = [[(0, 20, 15, 35, 0.90),      # clipped by the slice edge, high confidence
                  (4, 20, 16, 35, 0.60)]]     # same boulder, whole, lower confidence
    _, gdf, gdf_nms = _run(raster_128, per_slice, [(0, 0)], 0.10, monkeypatch)

    assert len(gdf) == 2                       # both survive the edge filter (raster-edge buffer)
    assert len(gdf_nms) == 1, "the duplicate pair did not cluster"
    kept = gdf_nms.iloc[0]
    assert bool(kept.is_within_slice) is True, "NMS kept the clipped copy"
    assert kept.score == pytest.approx(0.60, abs=1e-6)   # its real confidence, not a sentinel


def test_nms_prio_survives_the_file_round_trip(raster_128, monkeypatch, tmp_path):
    """Downstream fusion reads catalogs back off disk, so the ordering key has to be in the
    file (and short enough to survive shapefile field-name truncation)."""
    import geopandas as gpd
    per_slice = [[(20, 20, 30, 30, 0.42)]]
    _run(raster_128, per_slice, [(0, 0)], 0.10, monkeypatch, save_prenms=True)
    written = sorted(tmp_path.glob("*-mask-nms.gpkg"))
    assert written, "no NMS mask layer written"
    back = gpd.read_file(written[0])
    assert "nms_prio" in back.columns and len(back.columns) == len(set(back.columns))
    assert back.score.iloc[0] == pytest.approx(0.42, abs=1e-6)


def test_contract_version_is_recorded():
    assert INFERENCE_CONTRACT_VERSION >= 2


# --- G2-03: Solution G must be genuinely detector-preserving --------------------------------------

def _remask_result(rects):
    boxes = np.zeros((len(rects), 6), dtype=np.float32)
    masks = np.zeros((len(rects), SLICE, SLICE), dtype=np.float32)
    for i, (x0, y0, x1, y1, score) in enumerate(rects):
        boxes[i] = [x0, y0, x1, y1, score, 0]
        masks[i, y0:y1 + 1, x0:x1 + 1] = 1.0
    return SimpleNamespace(boxes=SimpleNamespace(data=torch.from_numpy(boxes)),
                           masks=SimpleNamespace(data=torch.from_numpy(masks)))


def test_failed_remask_keeps_the_detection_with_the_yolo_mask():
    """A failed SAM re-mask used to delete the row. A saved wrapper check lost 177 of 26,112 rows
    (0.68%), so the "identical detections, different masks" premise of Solution G did not hold and
    the SAM/YOLO comparison ran over slightly different populations."""
    rects = [(10, 10, 25, 25, 0.80), (35, 35, 50, 50, 0.60), (12, 40, 24, 52, 0.45)]
    result = _remask_result(rects)
    model = _fake_model([], 0.1)

    def remask(image, boxes):
        # SAM succeeds on the first box only; the rest hit the background-grab guard
        out = [None] * len(boxes)
        m = np.zeros((SLICE, SLICE), np.float32)
        m[10:26, 10:26] = 1.0
        out[0] = m
        return out

    baseline = _result_to_df(result, model, True, (0, 0), SLICE, None, False)
    df = _result_to_df(result, model, True, (0, 0), SLICE, None, False,
                       remask_fn=remask, image=np.zeros((SLICE, SLICE, 3), np.uint8),
                       remask_tag="sam2")

    assert len(df) == len(baseline) == 3, "row count is not invariant to re-masking"
    np.testing.assert_allclose(df.score.values, baseline.score.values)
    assert list(df.mask_src) == ["sam2", "yolo", "yolo"]


def test_mask_src_records_a_fully_successful_remask():
    rects = [(10, 10, 25, 25, 0.80)]
    result = _remask_result(rects)

    def remask(image, boxes):
        m = np.zeros((SLICE, SLICE), np.float32)
        m[10:26, 10:26] = 1.0
        return [m]

    df = _result_to_df(result, _fake_model([], 0.1), True, (0, 0), SLICE, None, False,
                       remask_fn=remask, image=np.zeros((SLICE, SLICE, 3), np.uint8),
                       remask_tag="sam2")
    assert list(df.mask_src) == ["sam2"]


def test_mask_src_reaches_the_catalog_on_disk(raster_128, monkeypatch, tmp_path):
    """Downstream analysis needs to be able to split success-only from intention-to-treat."""
    import geopandas as gpd
    per_slice = [[(20, 20, 40, 40, 0.42), (5, 5, 15, 15, 0.31)]]

    def remask(image, boxes):
        out = [None] * len(boxes)
        m = np.zeros((SLICE, SLICE), np.float32)
        m[20:41, 20:41] = 1.0
        out[0] = m
        return out

    _run(raster_128, per_slice, [(0, 0)], 0.10, monkeypatch, save_prenms=True,
         remask_fn=remask, mask_tag="sam2")
    written = sorted(tmp_path.glob("*-mask-sam2-nms.gpkg"))
    assert written, sorted(p.name for p in tmp_path.iterdir())
    back = gpd.read_file(written[0])
    assert set(back.mask_src) == {"sam2", "yolo"}


# --- fusion honours the new ordering key ---------------------------------------------------------

def test_fusion_prefers_nms_prio_over_raw_score():
    """``fuse_predictions`` must sort by ``nms_prio`` when it exists, and fall back to ``score``
    for contract-version-1 catalogs read off disk."""
    import geopandas as gpd
    import shapely
    from YOLOv8BeyondEarth.fusion import _keep_score

    v2 = gpd.GeoDataFrame({"score": [0.9, 0.6], "nms_prio": [0.45, 0.80],
                           "is_within_slice": [False, True]},
                          geometry=[shapely.box(0, 0, 4, 4), shapely.box(0, 0, 4, 4)])
    np.testing.assert_allclose(_keep_score(v2), [0.45, 0.80])

    v1 = v2.drop(columns="nms_prio")
    np.testing.assert_allclose(_keep_score(v1), [0.9, 0.6])
