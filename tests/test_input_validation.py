"""Negative-path coverage for the public inference API (review item X-05).

Two classes of silence are pinned closed here.

**Invalid arguments.** These used to surface as an ``AttributeError`` deep in the batch loop, a
``TypeError`` from ``Path(None)``, or not at all: an unknown ``contour_method`` fell through to
skimage and an unknown ``output_format`` fell through to shapefile, so a run asked for one thing
and quietly delivered another.

**Silently dropped detections.** ``_result_to_df`` wrapped the whole per-detection body in
``except Exception: return None``, so a broken re-masker, an unmapped class or a failing tracer
produced an empty catalog that was indistinguishable from an empty scene.
"""
import warnings
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
import rasterio as rio
import torch
from rasterio.transform import from_origin

from YOLOv8BeyondEarth import predict as P
from YOLOv8BeyondEarth.predict import _result_to_df

SLICE = 64


@pytest.fixture
def raster(tmp_path):
    path = tmp_path / "scene.tif"
    with rio.open(path, "w", driver="GTiff", height=128, width=128, count=1, dtype="uint8",
                  crs="EPSG:32631", transform=from_origin(0.0, 128.0, 1.0, 1.0)) as dst:
        dst.write(np.full((1, 128, 128), 200, dtype=np.uint8))
    return path


@pytest.fixture
def model():
    return SimpleNamespace(model=lambda *a, **k: [], image_size=256, confidence_threshold=0.1,
                           device="cpu", category_mapping={"0": "boulder"})


def _result(rects, mask_size=SLICE):
    boxes = np.zeros((len(rects), 6), dtype=np.float32)
    masks = np.zeros((len(rects), mask_size, mask_size), dtype=np.float32)
    for i, (x0, y0, x1, y1, score, cls) in enumerate(rects):
        boxes[i] = [x0, y0, x1, y1, score, cls]
        masks[i, y0:y1 + 1, x0:x1 + 1] = 1.0
    return SimpleNamespace(boxes=SimpleNamespace(data=torch.from_numpy(boxes)),
                           masks=SimpleNamespace(data=torch.from_numpy(masks)))


# --- invalid arguments ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs,exc,match", [
    (dict(sr_factor=0), ValueError, "sr_factor"),
    (dict(sr_factor=3), ValueError, "divisible"),
    (dict(slice_size=0), ValueError, "slice_size"),
    (dict(slice_size=-256), ValueError, "slice_size"),
    (dict(inference_size=0), ValueError, "inference_size"),
    (dict(batch_size=0), ValueError, "batch_size"),
    (dict(batch_size=-1), ValueError, "batch_size"),
    (dict(confidence_threshold=1.5), ValueError, "confidence_threshold"),
    (dict(confidence_threshold=-0.1), ValueError, "confidence_threshold"),
    (dict(overlap_height_ratio=1.0), ValueError, "overlap_height_ratio"),
    (dict(overlap_width_ratio=-0.1), ValueError, "overlap_width_ratio"),
    (dict(postprocess_match_threshold=0.0), ValueError, "postprocess_match_threshold"),
    (dict(postprocess_match_threshold=1.5), ValueError, "postprocess_match_threshold"),
    (dict(min_area_threshold=-1), ValueError, "min_area_threshold"),
    (dict(n_workers=0), ValueError, "n_workers"),
    (dict(contour_method="marching"), ValueError, "contour_method"),
    (dict(output_format="geojson"), ValueError, "output_format"),
    (dict(remask_fn="not-callable"), TypeError, "remask_fn"),
    (dict(max_det=0), ValueError, "max_det"),
    (dict(max_det=-1), ValueError, "max_det"),
    (dict(on_max_det="ignore"), ValueError, "on_max_det"),
])
def test_invalid_arguments_are_rejected_up_front(raster, model, tmp_path, kwargs, exc, match):
    base = dict(in_raster=raster, detection_model=model, output_dir=tmp_path, slice_size=SLICE)
    base.update(kwargs)
    with pytest.raises(exc, match=match):
        P.get_sliced_prediction(**base)


def test_unknown_modes_no_longer_fall_through_to_a_default():
    """The dangerous ones: both used to be accepted and silently substituted."""
    with pytest.raises(ValueError, match="silently"):
        P._validate_prediction_args(**_valid_args(contour_method="opencv"))
    with pytest.raises(ValueError, match="silently"):
        P._validate_prediction_args(**_valid_args(output_format="parquet"))


def _valid_args(**over):
    args = dict(in_raster=__file__, detection_model=SimpleNamespace(
        model=None, device="cpu", category_mapping={}), confidence_threshold=0.1,
        output_dir=".", slice_size=256, inference_size=1024, overlap_height_ratio=0.2,
        overlap_width_ratio=0.2, min_area_threshold=None, postprocess_match_threshold=0.5,
        batch_size=8, contour_method="skimage", output_format="gpkg", n_workers=None,
        remask_fn=None, sr_factor=1, max_det=300, on_max_det="error")
    args.update(over)
    return args


def test_missing_required_arguments(raster, model, tmp_path):
    with pytest.raises(ValueError, match="output_dir"):
        P.get_sliced_prediction(raster, model, output_dir=None, slice_size=SLICE)
    with pytest.raises(ValueError, match="detection_model"):
        P.get_sliced_prediction(raster, None, output_dir=tmp_path, slice_size=SLICE)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        P.get_sliced_prediction(tmp_path / "nope.tif", model, output_dir=tmp_path,
                                slice_size=SLICE)


def test_detection_model_shape_is_checked(raster, tmp_path):
    with pytest.raises(TypeError, match="category_mapping"):
        P.get_sliced_prediction(raster, SimpleNamespace(model=None, device="cpu"),
                                output_dir=tmp_path, slice_size=SLICE)


# --- cardinality ----------------------------------------------------------------------------------

def test_remask_fn_returning_the_wrong_count_raises(model):
    """``zip(boxes, masks)`` would truncate to the shorter of the two and lose detections with no
    trace — the exact shape of the Solution-G row loss the review found."""
    res = _result([(5, 5, 20, 20, 0.8, 0), (30, 30, 45, 45, 0.7, 0)])
    with pytest.raises(ValueError, match="one entry per box"):
        _result_to_df(res, model, True, (0, 0), SLICE, None, False,
                      remask_fn=lambda img, boxes: [np.ones((SLICE, SLICE), np.float32)],
                      image=np.zeros((SLICE, SLICE, 3), np.uint8))


def test_batch_result_count_mismatch_raises(raster, model, tmp_path, monkeypatch):
    """A short result list would pair every later slice with the wrong shift_amount, silently
    displacing detections across the scene."""
    class _Slices:
        images = [np.full((SLICE, SLICE, 3), 200, np.uint8) for _ in range(3)]
        starting_pixels = [(0, 0), (32, 0), (64, 0)]

        def __len__(self):
            return 3

    monkeypatch.setattr(P, "slice_image", lambda **_: _Slices())
    model.model = lambda images, **kw: [_result([])]          # 1 result for 3 slices
    with pytest.raises(ValueError, match="results for"):
        P.get_sliced_prediction(raster, model, output_dir=tmp_path, slice_size=SLICE,
                                save_prenms=False, save_bbox=False, output_format="gpkg")


# --- failures are counted, not swallowed ------------------------------------------------------------

def test_unmapped_class_raises_instead_of_shrinking_the_catalog(model):
    res = _result([(5, 5, 20, 20, 0.8, 7)])                   # class 7, mapping has only "0"
    with pytest.raises(KeyError, match="category_mapping"):
        _result_to_df(res, model, True, (0, 0), SLICE, None, False)


def test_tracer_failure_is_counted_by_reason(model, monkeypatch):
    """A geometry failure is tolerated — but recorded, so a systematic one is visible."""
    def boom(mask):
        raise ValueError("synthetic tracer failure")

    monkeypatch.setattr(P, "binary_mask_to_polygon", boom)
    res = _result([(5, 5, 20, 20, 0.8, 0), (30, 30, 45, 45, 0.7, 0)])
    stats = {}
    df = _result_to_df(res, model, True, (0, 0), SLICE, None, False, stats=stats)
    assert len(df) == 0
    assert stats == {"trace_failed:ValueError": 2}


def test_area_gate_and_remask_drops_are_counted_separately(model):
    res = _result([(5, 5, 6, 6, 0.8, 0),                      # 2x2 = 4 px, below a 6 px gate
                   (30, 30, 45, 45, 0.7, 0)])                 # comfortably above
    stats = {}
    df = _result_to_df(res, model, True, (0, 0), SLICE, 6, False, stats=stats)
    assert len(df) == 1 and stats == {"below_area_gate": 1}

    # a re-mask that fails on every box falls back to the detector's own mask (G2-03) — the rows
    # survive, and the fallback is counted rather than reported as a drop
    stats = {}
    df = _result_to_df(res, model, True, (0, 0), SLICE, None, False, stats=stats,
                       remask_fn=lambda img, boxes: [None, None],
                       image=np.zeros((SLICE, SLICE, 3), np.uint8), remask_tag="sam2")
    assert len(df) == 2 and stats == {"remask_fallback_to_yolo": 2}
    assert set(df.mask_src) == {"yolo"}

    # with no detector mask to fall back on there is nothing to keep, and that IS a drop
    stats = {}
    df = _result_to_df(res, model, False, (0, 0), SLICE, None, False, stats=stats,
                       remask_fn=lambda img, boxes: [None, None],
                       image=np.zeros((SLICE, SLICE, 3), np.uint8))
    assert len(df) == 0 and stats == {"remask_failed_no_fallback": 2}


def test_discard_tally_is_attached_to_the_result(raster, model, tmp_path, monkeypatch):
    class _Slices:
        images = [np.full((SLICE, SLICE, 3), 200, np.uint8)]
        starting_pixels = [(0, 0)]

        def __len__(self):
            return 1

    monkeypatch.setattr(P, "slice_image", lambda **_: _Slices())
    model.model = lambda images, **kw: [_result([(5, 5, 6, 6, 0.8, 0),      # dropped: 4 px
                                                 (25, 25, 45, 45, 0.7, 0)])]
    gdf, gdf_nms = P.get_sliced_prediction(raster, model, output_dir=tmp_path, slice_size=SLICE,
                                           min_area_threshold=6, save_prenms=False,
                                           save_bbox=False, output_format="gpkg")
    assert gdf.attrs["discarded"]["below_area_gate"] == 1
    assert gdf_nms.attrs["discarded"] == gdf.attrs["discarded"]


def test_systematic_unexpected_failure_warns(model, raster, tmp_path, monkeypatch):
    """The whole point of X-05: a broken tracer must not read as a quiet scene."""
    class _Slices:
        images = [np.full((SLICE, SLICE, 3), 200, np.uint8)]
        starting_pixels = [(0, 0)]

        def __len__(self):
            return 1

    def boom(mask):
        raise ValueError("synthetic tracer failure")

    monkeypatch.setattr(P, "slice_image", lambda **_: _Slices())
    monkeypatch.setattr(P, "binary_mask_to_polygon", boom)
    model.model = lambda images, **kw: [_result([(10, 10, 30, 30, 0.8, 0)] * 5)]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gdf, _ = P.get_sliced_prediction(raster, model, output_dir=tmp_path, slice_size=SLICE,
                                         save_prenms=False, save_bbox=False,
                                         output_format="gpkg")
    assert len(gdf) == 0
    msgs = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert any("pipeline failure" in m and "trace_failed:ValueError" in m for m in msgs), msgs


def test_a_healthy_solution_g_run_does_not_report_a_pipeline_failure():
    """Review item A3, with the reviewer's own reproduction.

    ``remask_fallback_to_yolo`` counts detections that are **in the catalog** carrying the detector's
    mask, but it was summed into both the ``unexpected`` numerator and the ``total`` denominator — so
    a healthy Solution-G run tripped the one warning that is supposed to mean "systematically
    broken", and ``attrs["discarded"]`` reported rows as lost that are present in the file. The rates
    here are the real ones recorded in commit f4d8a22: 777 native-SAM guard failures, ~26,112 rows.
    """
    gdf = gpd.GeoDataFrame({"score": []}, geometry=[])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        P._report_discards(gdf, {"below_area_gate": 120, "remask_fallback_to_yolo": 777},
                           n_kept=26112, truncation={})
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)], \
        [str(w.message) for w in caught]

    # survivors live in their own channel, and are not counted as losses
    assert gdf.attrs["fallbacks"] == {"remask_fallback_to_yolo": 777}
    assert gdf.attrs["discarded"] == {"below_area_gate": 120}


def test_losing_a_detection_with_no_fallback_still_warns():
    """The other half of A3's "decide explicitly". Before G2-03 this drop was named
    ``remask_returned_none`` and was listed as expected, because back then *every* guard failure
    deleted the detection. Now it fires only when there is no mask to fall back to, which means rows
    were lost outright — so it must stay unexpected. The rename left the old key in the expected set,
    which both broke the classification and left a dead string in the code."""
    gdf = gpd.GeoDataFrame({"score": []}, geometry=[])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        P._report_discards(gdf, {"remask_failed_no_fallback": 50}, n_kept=100, truncation={})
    msgs = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert any("pipeline failure" in m and "remask_failed_no_fallback" in m for m in msgs), msgs
    assert "remask_returned_none" not in P._EXPECTED_DISCARDS


def test_intentional_drops_do_not_warn(model, raster, tmp_path, monkeypatch):
    """The area gate discarding almost everything is a configuration choice, not a failure."""
    class _Slices:
        images = [np.full((SLICE, SLICE, 3), 200, np.uint8)]
        starting_pixels = [(0, 0)]

        def __len__(self):
            return 1

    monkeypatch.setattr(P, "slice_image", lambda **_: _Slices())
    model.model = lambda images, **kw: [_result([(5, 5, 6, 6, 0.8, 0)] * 20)]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gdf, _ = P.get_sliced_prediction(raster, model, output_dir=tmp_path, slice_size=SLICE,
                                         min_area_threshold=6, save_prenms=False,
                                         save_bbox=False, output_format="gpkg")
    assert gdf.attrs["discarded"]["below_area_gate"] == 20
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
