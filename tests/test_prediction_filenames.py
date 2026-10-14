"""`prediction_filenames` must name the file `get_sliced_prediction` actually writes.

Why this exists. Downstream analysis has to *locate* prediction catalogs, and until 2026-08-09 it
did so by re-deriving the filename pattern by hand. `data/solutionDE/analyze_hseg_acceptance.py`
hardcoded `ct-010` and an untagged mask layer in three places, so it could only ever read catalogs
written at conf 0.10 without a re-masker. The H-seg x G experiment changes both of those — step 0
retunes the confidence and step 1 passes `mask_tag="sam2"` — which meant the scoring half of that
experiment could not read the output of its own GPU half.

The fix was to expose the naming from `predict.py` and have both sides call it. These tests pin the
two together end-to-end: if the naming inside `get_sliced_prediction` ever drifts from the helper,
downstream analysis silently stops finding catalogs, so a tautological "helper equals helper" test
would be worthless. Each case therefore *runs* the pipeline and checks the file on disk.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio as rio
import torch
from rasterio.transform import from_origin

from YOLOv8BeyondEarth import predict as P

SLICE = 64


class _Slices:
    def __init__(self):
        self.images = [np.full((SLICE, SLICE, 3), 200, np.uint8)]
        self.starting_pixels = [(0, 0)]

    def __len__(self):
        return 1


def _result(rects):
    boxes = np.zeros((len(rects), 6), dtype=np.float32)
    masks = np.zeros((len(rects), SLICE, SLICE), dtype=np.float32)
    for i, (x0, y0, x1, y1, s) in enumerate(rects):
        boxes[i] = [x0, y0, x1, y1, s, 0]
        masks[i, y0:y1 + 1, x0:x1 + 1] = 1.0
    return SimpleNamespace(boxes=SimpleNamespace(data=torch.from_numpy(boxes)),
                           masks=SimpleNamespace(data=torch.from_numpy(masks)))


@pytest.fixture
def scene(tmp_path):
    path = tmp_path / "scene.tif"
    with rio.open(path, "w", driver="GTiff", height=128, width=128, count=1, dtype="uint8",
                  crs="EPSG:32631", transform=from_origin(0.0, 128.0, 1.0, 1.0)) as dst:
        dst.write(np.full((1, 128, 128), 200, dtype=np.uint8))
    return path


@pytest.fixture
def model():
    return SimpleNamespace(model=lambda images, **kw: [_result([(20, 20, 40, 40, 0.8)])],
                           image_size=256, confidence_threshold=0.1, device="cpu",
                           category_mapping={"0": "boulder"}, model_path=None)


# (conf, downscale_pred, mask_tag) — the two middle rows are exactly what H-seg x G changes.
CASES = [
    (0.10, True, None),
    (0.15, True, None),          # step 0: retuned confidence  -> ct-015
    (0.10, True, "sam2"),        # step 1: Solution-G re-mask  -> -mask-sam2-
    (0.25, True, "sam2"),        # both at once
    (0.10, False, None),         # no downscale: the "-downscaled" infix disappears
]


@pytest.mark.parametrize("conf,downscale,mask_tag", CASES)
def test_helper_names_every_layer_that_is_actually_written(scene, model, tmp_path, monkeypatch,
                                                           conf, downscale, mask_tag):
    """Assert on the FULL set of written files, not just the NMS layer.

    Checking only `nms` was not enough: a planted mutation that stripped the tag from the
    intermediate mask layer alone survived, because nothing looked at `bbox`/`mask`. Downstream
    code reads those too, so all three are pinned and the comparison is set equality.
    """
    monkeypatch.setattr(P, "_slice_raster", lambda **_: _Slices())
    out = tmp_path / "out"
    P.get_sliced_prediction(
        scene, model, output_dir=out, slice_size=SLICE, confidence_threshold=conf,
        inference_size=1024, overlap_height_ratio=0.2, overlap_width_ratio=0.2,
        downscale_pred=downscale, mask_tag=mask_tag, postprocess=True,
        save_prenms=True, save_bbox=True, save_nms=True, output_format="gpkg")

    expected = P.prediction_filenames(
        scene.stem, confidence_threshold=conf, slice_size=SLICE, inference_size=1024,
        overlap_height_ratio=0.2, downscale_pred=downscale, mask_tag=mask_tag,
        output_format="gpkg")
    written = {p.name for p in out.glob("*.gpkg")}
    assert written == set(expected.values()), (
        f"helper predicted {sorted(expected.values())}, disk has {sorted(written)}")


def test_the_tag_and_confidence_actually_change_the_name():
    """Guards the reverse failure: a helper that ignored its arguments would pass the test above
    for every case while still collapsing two different runs onto one filename."""
    base = dict(confidence_threshold=0.10, slice_size=256, inference_size=1024,
                overlap_height_ratio=0.2, downscale_pred=True, output_format="gpkg")
    plain = P.prediction_filenames("s", **base)["nms"]
    retuned = P.prediction_filenames("s", **{**base, "confidence_threshold": 0.15})["nms"]
    tagged = P.prediction_filenames("s", **base, mask_tag="sam2")["nms"]

    assert plain == "s-predictions-ct-010-ss-256-is-1024-ov-020-downscaled-mask-nms.gpkg"
    assert "ct-015" in retuned and retuned != plain
    assert "-mask-sam2-nms" in tagged and tagged != plain
    # A re-masked layer must not overwrite the plain YOLO one — that is why mask_tag exists.
    assert len({plain, retuned, tagged}) == 3


def test_sr_arm_keeps_its_scale_suffix():
    """`sr_factor` is encoded in the slice field, so the SR arm has its own catalog."""
    base = dict(confidence_threshold=0.10, slice_size=256, inference_size=1024,
                overlap_height_ratio=0.2, downscale_pred=True, output_format="gpkg")
    assert "-ss-256sr2-" in P.prediction_filenames("s", **base, sr_factor=2)["nms"]
    assert "-ss-256-" in P.prediction_filenames("s", **base, sr_factor=1)["nms"]
