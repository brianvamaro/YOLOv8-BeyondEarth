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


# --- size-range AP with COCO ignore semantics (review item G3-04) ---------------------------------

def test_out_of_range_truths_leave_the_recall_denominator():
    """The G3-04 defect: filtering predictions only left every out-of-range truth in the
    denominator, so a detector was charged for missing boulders declared out of scope."""
    gts = {"a": [_sq(0, 0, 20), _sq(50, 50, 3), _sq(80, 80, 3)]}   # one big, two sub-5px
    preds = pd.DataFrame({"image_id": ["a"], "score": [0.9], "geometry": [_sq(0, 0, 20)]})

    unrestricted = catalog_ap(preds, gts)
    assert unrestricted["n_gt_counted"] == 3
    assert unrestricted["recall"] == pytest.approx(1 / 3)

    restricted = catalog_ap(preds, gts, size_range=(5.0, np.inf))
    assert restricted["n_gt"] == 3 and restricted["n_gt_counted"] == 1
    assert restricted["recall"] == pytest.approx(1.0)
    assert restricted["ap"] == pytest.approx(1.0)


def test_below_range_prediction_on_ignored_truth_is_neither_tp_nor_fp():
    """The explicit G3-04 acceptance criterion. Finding an out-of-scope boulder is not an error."""
    gts = {"a": [_sq(0, 0, 20), _sq(50, 50, 3)]}
    preds = pd.DataFrame({"image_id": ["a", "a"], "score": [0.9, 0.8],
                          "geometry": [_sq(0, 0, 20), _sq(50, 50, 3)]})
    res = catalog_ap(preds, gts, size_range=(5.0, np.inf))
    assert res["n_tp"] == 1
    assert res["n_fp"] == 0, "the sub-range detection was charged as a false positive"
    assert res["n_pred_ignored"] == 1
    assert res["ap"] == pytest.approx(1.0)


def test_unmatched_below_range_prediction_is_ignored_not_a_fp():
    gts = {"a": [_sq(0, 0, 20)]}
    preds = pd.DataFrame({"image_id": ["a", "a"], "score": [0.9, 0.8],
                          "geometry": [_sq(0, 0, 20), _sq(300, 300, 2)]})
    res = catalog_ap(preds, gts, size_range=(5.0, np.inf))
    assert res["n_tp"] == 1 and res["n_fp"] == 0 and res["n_pred_ignored"] == 1
    assert res["ap"] == pytest.approx(1.0)


def test_in_range_prediction_missing_an_in_range_truth_still_costs_recall():
    """Ignoring must not become a way to make misses disappear."""
    gts = {"a": [_sq(0, 0, 20), _sq(50, 50, 20)]}
    preds = pd.DataFrame({"image_id": ["a"], "score": [0.9], "geometry": [_sq(0, 0, 20)]})
    res = catalog_ap(preds, gts, size_range=(5.0, np.inf))
    assert res["n_gt_counted"] == 2 and res["recall"] == pytest.approx(0.5)


def test_in_range_prediction_that_is_wrong_is_still_a_fp():
    gts = {"a": [_sq(0, 0, 20)]}
    preds = pd.DataFrame({"image_id": ["a", "a"], "score": [0.9, 0.8],
                          "geometry": [_sq(0, 0, 20), _sq(300, 300, 20)]})
    res = catalog_ap(preds, gts, size_range=(5.0, np.inf))
    assert res["n_tp"] == 1 and res["n_fp"] == 1 and res["n_pred_ignored"] == 0


def test_matching_prefers_an_in_range_truth_over_an_ignored_one():
    """An ignored truth must never consume a detection that an in-range truth needed."""
    big = _sq(0, 0, 20)
    overlapping_small = shapely.box(0, 0, 3, 3)          # inside `big`, below the range
    gts = {"a": [overlapping_small, big]}                # ignored one listed FIRST
    preds = pd.DataFrame({"image_id": ["a"], "score": [0.9], "geometry": [big]})
    res = catalog_ap(preds, gts, size_range=(5.0, np.inf))
    assert res["n_tp"] == 1 and res["recall"] == pytest.approx(1.0)


def test_upper_bound_of_the_size_range_is_exclusive():
    gts = {"a": [_sq(0, 0, 20)]}                          # eq diam ~22.6 px
    preds = pd.DataFrame({"image_id": ["a"], "score": [0.9], "geometry": [_sq(0, 0, 20)]})
    inside = catalog_ap(preds, gts, size_range=(5.0, 30.0))
    outside = catalog_ap(preds, gts, size_range=(5.0, 10.0))
    assert inside["n_gt_counted"] == 1
    assert outside["n_gt_counted"] == 0 and np.isnan(outside["ap"])


def test_size_range_is_validated():
    with pytest.raises(ValueError, match="size_range"):
        catalog_ap(pd.DataFrame({"image_id": ["a"], "score": [0.9], "geometry": [_sq(0, 0)]}),
                   {"a": [_sq(0, 0)]}, size_range=(10.0, 5.0))


def test_precision_sampling_is_cocos_step_lookup():
    """COCOeval samples the envelope at the first recall >= each grid point; the previous linear
    interpolation gave a slightly different, non-COCO number."""
    gts = {"a": [_sq(0, 0), _sq(50, 50)]}
    preds = pd.DataFrame({"image_id": ["a", "a"], "score": [0.9, 0.8],
                          "geometry": [_sq(0, 0), _sq(200, 200)]})
    res = catalog_ap(preds, gts)
    # recall reaches only 0.5, at precision 1.0 -> 51 of the 101 grid points are covered
    assert res["ap"] == pytest.approx(51 / 101, abs=1e-9)


# --- paired uncertainty ---------------------------------------------------------------------------

def test_paired_ap_ci_brackets_a_real_difference_and_covers_zero_for_none():
    from YOLOv8BeyondEarth.ap_eval import paired_ap_ci
    rng = np.random.default_rng(0)
    gts, rows_a, rows_b = {}, [], []
    for i in range(24):
        img = f"t{i}"
        truths = [_sq(20 * j, 20 * j) for j in range(4)]
        gts[img] = truths
        for j, t in enumerate(truths):
            rows_a.append({"image_id": img, "score": float(rng.uniform(0.5, 1)), "geometry": t})
            if j < 2:                                     # arm B finds only half of them
                rows_b.append({"image_id": img, "score": float(rng.uniform(0.5, 1)),
                               "geometry": t})
    preds_a, preds_b = pd.DataFrame(rows_a), pd.DataFrame(rows_b)

    worse = paired_ap_ci(preds_a, preds_b, gts, n_boot=200)
    assert worse["delta"] < 0 and worse["hi"] < 0        # a real deficit, interval excludes 0
    assert worse["n_images"] == 24

    same = paired_ap_ci(preds_a, preds_a.copy(), gts, n_boot=200)
    assert same["delta"] == pytest.approx(0.0)
    assert same["lo"] <= 0 <= same["hi"]
