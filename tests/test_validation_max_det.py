"""Acceptance tests for X-06 in the *validation* detectors (``orientation_validation``).

X-06 was fixed in the production path (``predict.get_sliced_prediction``) on 2026-08-04, but the
tiled scanners in ``orientation_validation`` were missed: none of them passed ``max_det``, so
ultralytics' silent 300 applied to every tile of every scan they produced — including the SAM2
equal-area rescan behind ``data/summary/fixed_area_scenes.csv``, which is saturated on at least one
control site (``ESP_030222_1220``: 275.6 det/tile, 57.4% of detections inside capped tiles).

The cap keeps the top N *by confidence*, so an angle statistic computed on a high-confidence subset
is far less exposed than a density. That is a reason to record the saturation, not a reason to let
it stay silent: what made X-06 survive two reviews is that nothing on disk said the cap had bound.

No GPU and no real weights: the models are recording stand-ins that apply ``conf`` and ``max_det``
the way ultralytics does.
"""
import numpy as np
import pytest
import torch
from types import SimpleNamespace

from YOLOv8BeyondEarth import orientation_validation as OV
from YOLOv8BeyondEarth import predict as P


SLICE = 32


# --- stand-ins -----------------------------------------------------------------------------------

class _Boxes:
    """Enough of ultralytics' ``Boxes`` for the validation detectors and the truncation counter."""

    def __init__(self, xyxy, conf):
        self.xyxy = torch.from_numpy(np.asarray(xyxy, dtype=np.float32).reshape(-1, 4))
        self.conf = torch.from_numpy(np.asarray(conf, dtype=np.float32))
        self.data = torch.zeros((len(self.conf), 6), dtype=torch.float32)

    def __len__(self):
        return int(self.conf.shape[0])


class FakeYOLO:
    """Returns ``n_per_tile`` identical square detections per tile, applying ``max_det`` as
    ultralytics does (cap after NMS, keep the highest-confidence N)."""

    def __init__(self, n_per_tile, slice_size=SLICE, obb=False):
        self.n_per_tile = n_per_tile
        self.slice_size = slice_size
        self.obb = obb
        self.calls = []

    def __call__(self, image, imgsz=None, conf=None, verbose=None, device=None, max_det=None):
        self.calls.append({"imgsz": imgsz, "conf": conf, "device": device, "max_det": max_det})
        n = self.n_per_tile
        if max_det is not None:
            n = min(n, max_det)
        # A 5x9 rectangle well inside the tile: elongated, resolved, and not touching an edge, so
        # it survives `drop_edge` and any aspect floor.
        xyxy = np.tile(np.array([4.0, 4.0, 9.0, 13.0], dtype=np.float32), (n, 1))
        scores = np.linspace(0.9, 0.5, num=n, dtype=np.float32) if n else np.zeros(0, np.float32)
        if self.obb:
            corners = np.tile(np.array([[4., 4.], [9., 4.], [9., 13.], [4., 13.]], np.float32),
                              (n, 1, 1))
            obb = SimpleNamespace(xyxyxyxy=torch.from_numpy(corners),
                                  xywhr=torch.from_numpy(
                                      np.tile(np.array([6.5, 8.5, 5., 9., 0.], np.float32),
                                              (n, 1))))
            return [SimpleNamespace(boxes=None, obb=_Len(obb, n), masks=None)]
        masks = np.zeros((n, imgsz or self.slice_size, imgsz or self.slice_size), dtype=np.float32)
        up = (imgsz or self.slice_size) / self.slice_size
        for i in range(n):
            masks[i, int(4 * up):int(13 * up), int(4 * up):int(9 * up)] = 1.0
        return [SimpleNamespace(boxes=_Boxes(xyxy, scores), obb=None,
                                masks=SimpleNamespace(data=torch.from_numpy(masks)))]


class _Len:
    """Wraps an object so ``len()`` works, without touching its attributes."""

    def __init__(self, obj, n):
        self._obj, self._n = obj, n

    def __len__(self):
        return self._n

    def __getattr__(self, name):
        return getattr(self._obj, name)


class FakeSAM:
    """One mask per prompted box, in order — the count-matched path ``_sam_masks`` expects."""

    def __call__(self, image, bboxes=None, verbose=None, device=None, conf=None):
        boxes = np.atleast_2d(np.asarray(bboxes, dtype=np.float32))
        h, w = image.shape[:2]
        masks = np.zeros((len(boxes), h, w), dtype=np.float32)
        for i, (x0, y0, x1, y1) in enumerate(boxes[:, :4]):
            masks[i, int(y0):int(y1), int(x0):int(x1)] = 1.0
        return [SimpleNamespace(masks=SimpleNamespace(data=torch.from_numpy(masks)))]


@pytest.fixture
def img():
    """A 2x2 grid of tiles."""
    return np.full((2 * SLICE, 2 * SLICE), 200, dtype=np.uint8)


# --- the constants may not drift ------------------------------------------------------------------

def test_validation_default_matches_predict():
    """``orientation_validation`` duplicates the constant rather than importing it (``predict``
    pulls in cv2/geopandas/torch at module level). This is the guard against the copies drifting."""
    assert OV.ULTRALYTICS_DEFAULT_MAX_DET == P.ULTRALYTICS_DEFAULT_MAX_DET == 300
    assert OV._NEAR_MAX_DET_FRACTION == P._NEAR_MAX_DET_FRACTION


# --- every tiled scanner must pass it -------------------------------------------------------------

def test_detect_orientations_passes_max_det(img):
    model = FakeYOLO(n_per_tile=2)
    OV.detect_orientations(model, img, slice_size=SLICE, imgsz=SLICE * 4, max_det=77)
    assert len(model.calls) == 4
    assert {c["max_det"] for c in model.calls} == {77}


def test_detect_orientations_defaults_to_the_ultralytics_cap(img):
    """Default kept at 300 so an un-parameterised call reproduces the historical artifacts."""
    model = FakeYOLO(n_per_tile=2)
    OV.detect_orientations(model, img, slice_size=SLICE, imgsz=SLICE * 4)
    assert {c["max_det"] for c in model.calls} == {OV.ULTRALYTICS_DEFAULT_MAX_DET}


def test_detect_binary_masks_passes_max_det(img):
    model = FakeYOLO(n_per_tile=2)
    OV.detect_binary_masks(model, img, slice_size=SLICE, imgsz=SLICE * 4, max_det=55)
    assert {c["max_det"] for c in model.calls} == {55}


def test_detect_obb_angles_passes_max_det(img):
    model = FakeYOLO(n_per_tile=2, obb=True)
    df = OV.detect_obb_angles(model, img, slice_size=SLICE, imgsz=SLICE * 4, max_det=44)
    assert {c["max_det"] for c in model.calls} == {44}
    assert df.attrs["truncation"]["max_det"] == 44


def test_detect_sam2_mask_sets_passes_max_det(img):
    """The one the queue names: this is the function behind the equal-area rescan."""
    model, sam = FakeYOLO(n_per_tile=3), FakeSAM()
    df, masks = OV.detect_sam2_mask_sets(model, sam, img, slice_size=SLICE, imgsz=SLICE * 4,
                                         configs=("yolo", "sam256"), max_det=33)
    assert {c["max_det"] for c in model.calls} == {33}
    assert df.attrs["truncation"]["max_det"] == 33
    assert len(df) == len(masks["yolo"]) == len(masks["sam256"])


# --- and record when it binds ---------------------------------------------------------------------

def test_saturated_scan_is_recorded_and_warns(img):
    """4 tiles, each returning the full cap: every tile counted at the cap, and loud about it."""
    model = FakeYOLO(n_per_tile=10)
    with pytest.warns(RuntimeWarning, match="LOWER BOUNDS"):
        df = OV.detect_orientations(model, img, slice_size=SLICE, imgsz=SLICE * 4, max_det=4)
    t = df.attrs["truncation"]
    assert (t["tiles"], t["tiles_at_max_det"], t["max_det_returned"]) == (4, 4, 4)


def test_unsaturated_scan_is_silent(img, recwarn):
    model = FakeYOLO(n_per_tile=2)
    df = OV.detect_orientations(model, img, slice_size=SLICE, imgsz=SLICE * 4, max_det=300)
    t = df.attrs["truncation"]
    assert (t["tiles"], t["tiles_at_max_det"], t["tiles_near_max_det"]) == (4, 0, 0)
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_the_counter_falls_back_to_data_when_boxes_has_no_len(img):
    """`test_orientation_path_parity`'s stand-in exposes `boxes` as a bare namespace holding a
    tensor, with no `__len__`. The counter must read `.data` rather than abort the scan — a
    diagnostic that can kill the run it observes is worse than no diagnostic. It must still *count*,
    not silently return 0, or the truncation record becomes decorative."""
    n = 7
    result = SimpleNamespace(
        boxes=SimpleNamespace(data=torch.zeros((n, 6))), obb=None,
        masks=SimpleNamespace(data=torch.zeros((n, 8, 8))))
    trunc = OV._new_truncation(n)
    OV._note_tile_truncation(result, n, trunc)
    assert (trunc["tiles"], trunc["max_det_returned"], trunc["tiles_at_max_det"]) == (1, n, 1)


def test_near_the_cap_is_recorded_separately(img):
    """Ultralytics' empty-mask filter can drop a row *after* the cap applied, so a truncated tile
    can arrive holding fewer than ``max_det``. Those land in ``tiles_near_max_det``, which is why
    ``tiles_at_max_det`` is documented as a lower bound."""
    model = FakeYOLO(n_per_tile=19)
    df = OV.detect_orientations(model, img, slice_size=SLICE, imgsz=SLICE * 4, max_det=20)
    t = df.attrs["truncation"]
    assert (t["tiles_at_max_det"], t["tiles_near_max_det"]) == (0, 4)
