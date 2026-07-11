"""Unit tests for the polygon-catalog AP50 evaluator (Goal-3 validation item 9)."""
import numpy as np
import pandas as pd
import pytest
import shapely

from YOLOv8BeyondEarth.ap_eval import catalog_ap, load_yolo_seg_labels, size_gated


def _sq(x, y, s=10):
    return shapely.box(x, y, x + s, y + s)


def test_perfect_match_ap_is_one():
    gts = {"a": [_sq(0, 0), _sq(50, 50)]}
    preds = pd.DataFrame({"image_id": ["a", "a"], "score": [0.9, 0.8],
                          "geometry": [_sq(0, 0), _sq(50, 50)]})
    res = catalog_ap(preds, gts)
    assert res["ap"] == pytest.approx(1.0)
    assert res["n_tp"] == 2 and res["recall"] == pytest.approx(1.0)


def test_high_scoring_fp_caps_ap_at_two_thirds():
    # FP ranked first: precision envelope is 2/3 at every recall level -> AP = 2/3 exactly
    gts = {"a": [_sq(0, 0), _sq(50, 50)]}
    preds = pd.DataFrame({"image_id": ["a", "a", "a"], "score": [0.95, 0.8, 0.7],
                          "geometry": [_sq(200, 200), _sq(0, 0), _sq(50, 50)]})
    res = catalog_ap(preds, gts)
    assert res["ap"] == pytest.approx(2 / 3, abs=1e-6)
    assert res["n_tp"] == 2 and res["n_pred"] == 3


def test_iou_threshold_and_one_to_one_matching():
    # two preds on one GT: only the better one matches; the second is an FP (GT already used)
    gts = {"a": [_sq(0, 0)]}
    preds = pd.DataFrame({"image_id": ["a", "a"], "score": [0.9, 0.8],
                          "geometry": [_sq(0, 0), shapely.box(1, 1, 11, 11)]})
    res = catalog_ap(preds, gts)
    assert res["n_tp"] == 1
    # a shifted box at IoU < 0.5 alone does not match
    preds2 = pd.DataFrame({"image_id": ["a"], "score": [0.9],
                           "geometry": [shapely.box(6, 6, 16, 16)]})
    assert catalog_ap(preds2, gts)["n_tp"] == 0


def test_box_vs_mask_geom():
    # a thin L-shape vs its bounding box: box-IoU matches, mask-IoU does not
    l_shape = shapely.union(shapely.box(0, 0, 10, 2), shapely.box(0, 0, 2, 10))
    gts = {"a": [l_shape]}
    preds = pd.DataFrame({"image_id": ["a"], "score": [0.9],
                          "geometry": [shapely.box(0, 0, 10, 10)]})
    assert catalog_ap(preds, gts, geom="box")["n_tp"] == 1
    assert catalog_ap(preds, gts, geom="mask")["n_tp"] == 0


def test_unknown_image_id_is_fp_and_missing_preds_are_misses():
    gts = {"a": [_sq(0, 0)], "b": [_sq(0, 0)]}
    preds = pd.DataFrame({"image_id": ["a", "zzz"], "score": [0.9, 0.8],
                          "geometry": [_sq(0, 0), _sq(0, 0)]})
    res = catalog_ap(preds, gts)
    assert res["n_gt"] == 2 and res["n_tp"] == 1
    assert res["recall"] == pytest.approx(0.5)


def test_empty_inputs():
    assert catalog_ap(pd.DataFrame(columns=["image_id", "score", "geometry"]),
                      {"a": [_sq(0, 0)]})["ap"] == 0.0
    assert np.isnan(catalog_ap(pd.DataFrame({"image_id": ["a"], "score": [0.9],
                                             "geometry": [_sq(0, 0)]}), {"a": []})["ap"])


def test_load_yolo_seg_labels(tmp_path):
    # a 0.2x0.2 normalized square on a 500-px tile = 100x100 px at (100, 100)
    txt = tmp_path / "tile.txt"
    txt.write_text("0 0.2 0.2 0.4 0.2 0.4 0.4 0.2 0.4\n0 0.5 0.5\n")  # 2nd line degenerate
    polys = load_yolo_seg_labels(txt, 500, 500)
    assert len(polys) == 1
    assert polys[0].area == pytest.approx(100 * 100)
    assert polys[0].bounds == pytest.approx((100, 100, 200, 200))
    # with a north-up transform (pad to 512): y flips about the raster height
    from rasterio.transform import from_origin
    polys_t = load_yolo_seg_labels(txt, 500, 500, transform=from_origin(0, 512, 1, 1))
    assert polys_t[0].bounds == pytest.approx((100, 512 - 200, 200, 512 - 100))


def test_size_gated():
    small, big = _sq(0, 0, 3), _sq(50, 50, 20)   # eq diam ~3.4 px vs ~22.6 px
    preds = pd.DataFrame({"image_id": ["a", "a"], "score": [0.9, 0.8],
                          "geometry": [small, big]})
    kept = size_gated(preds, min_diam_px=5, res=1.0)
    assert len(kept) == 1 and kept.iloc[0].geometry.equals(big)
