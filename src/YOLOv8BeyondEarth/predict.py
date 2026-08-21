import contextlib
import json
import logging
import os
import shutil
import tempfile
import time
import warnings

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import torch

from concurrent.futures import ThreadPoolExecutor

from YOLOv8BeyondEarth import presets
from YOLOv8BeyondEarth.polygon import (binary_mask_to_polygon, binary_mask_to_polygon_cv,
                                       shift_polygon, mask_support_area, rescale_pixel_coords,
                                       add_geometries, bboxes_to_shp, outlines_to_shp)
from lsnms import nms

from sahi.slicing import slice_image
from tqdm import tqdm
from pathlib import Path

from rastertools_BOULDERING import raster, convert as raster_convert, metadata as raster_metadata
from shptools_BOULDERING import shp

#from torchvision.ops import (nms as nms_torch, batched_nms as batched_nms_torch)

# --- inference/output contract -------------------------------------------------------------------
# Bumped whenever a change alters the numeric content of a saved catalog, so a file can never be
# ambiguous about which convention produced it. Recorded in every result manifest.
#   1  original: `conf` never reached ultralytics (effective 0.25 regardless of the request) and
#      slice-truncated detections had their confidence overwritten with the literal 0.10.
#   2  (2026-07-30, review item X-01/X-02): the requested threshold is passed to the model, and
#      `score` is the detector's real confidence; truncation priority moved to `nms_prio`.
#   3  (2026-08-04, review item X-06): `max_det` is passed explicitly and recorded. Before this,
#      ultralytics' own 300 applied to every slice, silently deleting the faintest detections in
#      the densest ones — measured engaging on real HiRISE terrain, see `ULTRALYTICS_DEFAULT_MAX_DET`.
INFERENCE_CONTRACT_VERSION = 3

#: Ultralytics' ``cfg/default.yaml`` value, which applied to every slice while this pipeline never
#: passed ``max_det`` (review item X-06). It is applied *after* NMS and keeps the top N **by
#: confidence**, so what it discards is the faintest detections in the densest slices — the
#: SFD-critical population, exactly where completeness matters most.
#:
#: **Measured engaging, not merely suspected** (``scripts/review/probe_max_det.py``, 2026-08-04): on
#: the densest HiRISE 512-arm terrain the model returns a mean of ~606 and a max of 673 detections
#: per slice at conf 0.10, so a cap of 300 was throwing away roughly half of what was found, on
#: every one of those slices. Kept as this module's *function* default so an un-parameterised call
#: reproduces historical behaviour exactly; the production value lives in
#: :data:`presets.GOAL1_PRODUCTION`.
ULTRALYTICS_DEFAULT_MAX_DET = 300

#: Fraction of ``max_det`` above which a slice is recorded as *near* the cap. A slice at the cap
#: proves truncation; a slice just below it means the headroom is thin — and it is also where a
#: slice that truncated and then lost a detection or two to ultralytics' empty-mask filter
#: (``masks.amax() > 0`` in ``SegmentationPredictor.construct_result``) shows up instead of at the
#: cap exactly. That filter is why the at-cap count is a *lower* bound on binding slices.
_NEAR_MAX_DET_FRACTION = 0.9

# Column order of the per-slice detection frame. `score` is the detector's real confidence;
# `nms_prio` is the suppression ordering key (see `edge_priority`); `mask_src` names which
# producer this row's geometry came from ("yolo", or the remask tag, e.g. "sam2").
_DETECTION_COLUMNS = ['score', 'nms_prio', 'polygon', 'category_id', 'category_name',
                      'is_within_slice', 'mask_src']


CONTOUR_METHODS = ("skimage", "cv2")
OUTPUT_FORMATS = ("shp", "gpkg")

logger = logging.getLogger(__name__)

#: Fraction of the model's detections that may be discarded for *unexpected* reasons before the
#: run warns. Area-gate and edge drops are intentional and excluded from the count.
_DISCARD_WARN_FRACTION = 0.01

#: Reasons that are **not discards at all** — the detection is in the catalog, carrying a note. They
#: go to ``attrs["fallbacks"]`` rather than ``attrs["discarded"]``, for the same reason X-06 gave
#: ``truncation`` its own channel: the units differ, so mixing them corrupts the arithmetic. Summing
#: a survivor into "detections discarded for unexpected reasons" made a *healthy* Solution-G run trip
#: the one warning that is supposed to mean "systematically broken" — 777 fallbacks against 26,112
#: kept rows reads as 2.9% unexpected loss, while those 777 rows are present with ``mask_src="yolo"``
#: (review item A3).
_SURVIVOR_REASONS = ("remask_fallback_to_yolo",)

#: Discards that are deliberate, so they never raise the "pipeline failure" warning.
#:
#: ``remask_failed_no_fallback`` is **deliberately absent** — the decision A3 asked for. Before
#: G2-03 (commit 4231b23) this drop was called ``remask_returned_none`` and *was* expected, because
#: back then any re-mask guard failure deleted the detection and ~0.68% of rows did so routinely.
#: G2-03 changed what it means: a guard failure now falls back to the detector's own mask, so this
#: reason fires only when there is **no** mask to fall back to — i.e. a detect-only model driving a
#: ``remask_fn``, never the production seg path. On that path it is unreachable; off it, it means
#: detections were lost outright with no recourse, which is exactly what X-05's warning is for. The
#: rename left the old key here and so silently reclassified the drop; both halves are fixed.
_EXPECTED_DISCARDS = ("below_area_gate", "slice_truncated_interior", "duplicate_geometry")


def _report_discards(gdf, discarded, n_kept, truncation=None):
    """Attach the per-reason discard tally to the result and warn if failures look systematic.

    Detections vanishing one at a time inside a broad ``except Exception`` is how a broken
    re-masker or tracer reads as "a quiet scene" (review item X-05). Intentional drops (the area
    gate, interior slice truncation, exact duplicates) are recorded but never warned about.

    Three channels, deliberately separate because their units differ and no sum across them is
    meaningful:

    - ``attrs["discarded"]`` — detections the model produced that the pipeline did not keep.
    - ``attrs["fallbacks"]`` — detections it *did* keep, with a caveat (:data:`_SURVIVOR_REASONS`).
      No warning threshold is asserted on these: a fallback rate that would undermine a Goal-2
      aggregate is a science judgement, and ``mask_src`` records the truth per row, so a consumer can
      compute it exactly rather than trust a number nobody validated.
    - ``truncation`` — the **slice-level** channel X-06 added: whole groups of detections the model
      never returned, so they cannot appear in a per-detection tally at all. Its values count
      *slices*, not detections, so summing it into the discard denominator would make the "unexpected
      fraction" arithmetic meaningless.
    """
    tally = {k: v for k, v in discarded.items() if v}
    fallbacks = {k: tally.pop(k) for k in _SURVIVOR_REASONS if k in tally}
    gdf.attrs["discarded"] = tally
    gdf.attrs["fallbacks"] = fallbacks
    unexpected = sum(v for k, v in tally.items() if k not in _EXPECTED_DISCARDS)
    total = n_kept + sum(tally.values())
    if unexpected and total and unexpected / total > _DISCARD_WARN_FRACTION:
        warnings.warn(
            f"{unexpected} of {total} detections ({unexpected / total:.1%}) were discarded for "
            f"unexpected reasons: "
            f"{ {k: v for k, v in tally.items() if k not in _EXPECTED_DISCARDS} }. "
            "This is a pipeline failure, not an empty scene.", RuntimeWarning, stacklevel=3)
    if tally:
        logger.info("discarded detections by reason: %s", tally)
    if fallbacks:
        logger.info("detections kept via a fallback producer (see mask_src): %s", fallbacks)

    trunc = {k: v for k, v in (truncation or {}).items() if v}
    gdf.attrs["truncation"] = trunc
    if trunc.get("slices_at_max_det"):
        warnings.warn(
            f"{trunc['slices_at_max_det']} slices returned the full max_det, so the model's "
            "output was cut off in each of them and the faintest detections are missing from this "
            "catalog. Raise max_det (see presets.GOAL1_PRODUCTION) and re-run; this count is a "
            "lower bound.", RuntimeWarning, stacklevel=3)
    if trunc.get("nms_time_limit_batches"):
        warnings.warn(
            f"ultralytics' NMS time limit fired on {trunc['nms_time_limit_batches']} batches, "
            f"abandoning up to {trunc.get('slices_zeroed_after_nms_time_limit', 0)} slices that "
            "therefore returned zero detections.", RuntimeWarning, stacklevel=3)
    if trunc:
        logger.info("slice-level truncation events: %s", trunc)


@contextlib.contextmanager
def _watch_nms_time_limit():
    """Count ultralytics' ``NMS time limit ... exceeded`` warnings (review item X-06).

    ``ultralytics/utils/nms.py`` sets ``time_limit = 2.0 + 0.05 * batch_size`` (**2.4 s** at the
    production ``batch_size=8``) and ``break``s out of its per-image loop once a batch exceeds it.
    The images it has not reached keep their pre-allocated zero tensors, so **every remaining slice
    in that batch returns zero detections** — a whole-slice dropout, not a thinning. It does emit
    ``LOGGER.warning``, but across ~46,000 slices under a tqdm bar that scrolls past unread and
    nothing structured records it, so a run cannot tell you afterwards whether it happened.

    Yields a callable returning the cumulative count. Ultralytics being unimportable (the test
    suite drives fake models) degrades to a constant zero rather than failing.
    """
    counter = {"n": 0}
    try:
        from ultralytics.utils import LOGGER as ultralytics_logger
    except Exception:                        # no ultralytics (fake-model tests): nothing to watch
        yield lambda: 0
        return

    class _Count(logging.Handler):
        def emit(self, record):
            try:
                if "NMS time limit" in record.getMessage():
                    counter["n"] += 1
            except Exception:                # a logging handler must never break the run
                pass

    handler = _Count(level=logging.WARNING)
    ultralytics_logger.addHandler(handler)
    try:
        yield lambda: counter["n"]
    finally:
        ultralytics_logger.removeHandler(handler)


def _note_slice_truncation(results, max_det, truncation, on_max_det, first_slice, new_timeouts):
    """Record — and by default fail fast on — the two ways a slice silently loses detections.

    Called once per batch, immediately after the model returns, so a scene run learns in its first
    minutes that the cap is too low instead of after hours of compute (the G1-07 lesson: a check
    that only runs at write time burns the whole budget first).

    ``slices_at_max_det`` is a **lower** bound. Ultralytics applies ``max_det`` inside NMS and then
    ``SegmentationPredictor.construct_result`` drops detections whose upsampled mask is empty, so a
    slice that truncated at N can come back with fewer than N rows. ``slices_near_max_det`` is where
    those land, which is why both are recorded.
    """
    counts = [int(r.boxes.data.shape[0]) for r in results]
    near = max(1, int(round(_NEAR_MAX_DET_FRACTION * max_det)))
    at_cap = sum(1 for c in counts if c >= max_det)
    if at_cap:
        truncation["slices_at_max_det"] = truncation.get("slices_at_max_det", 0) + at_cap
    n_near = sum(1 for c in counts if near <= c < max_det)
    if n_near:
        truncation["slices_near_max_det"] = truncation.get("slices_near_max_det", 0) + n_near
    if new_timeouts:
        truncation["nms_time_limit_batches"] = (
            truncation.get("nms_time_limit_batches", 0) + new_timeouts)
        # Upper bound: a slice in a timed-out batch that returns nothing may also just be empty
        # ground. We cannot tell the two apart from outside ultralytics, so record the bound.
        truncation["slices_zeroed_after_nms_time_limit"] = (
            truncation.get("slices_zeroed_after_nms_time_limit", 0)
            + sum(1 for c in counts if c == 0))

    if at_cap and on_max_det == "error":
        worst = max(counts)
        raise RuntimeError(
            f"{at_cap} of {len(counts)} slices in the batch starting at slice {first_slice} "
            f"returned max_det={max_det} detections, so the model's output was truncated in each "
            f"of them (worst {worst}). Ultralytics applies the cap after NMS and keeps the top N by "
            "confidence, so what is missing is the faintest detections in the densest slices. "
            "Raise max_det — but check VRAM first: masks are materialised at imgsz, so peak memory "
            "scales with max_det x batch_size (see scripts/review/probe_max_det.py). Pass "
            "on_max_det='warn' to accept the truncation and record it instead of failing.")


def _validate_prediction_args(*, in_raster, detection_model, confidence_threshold, output_dir,
                              slice_size, inference_size, overlap_height_ratio,
                              overlap_width_ratio, min_area_threshold,
                              postprocess_match_threshold, batch_size, contour_method,
                              output_format, n_workers, remask_fn, sr_factor, max_det,
                              on_max_det):
    """Reject every invalid argument up front, before any raster is read (review item X-05).

    These used to surface — if at all — as an ``AttributeError`` deep inside the batch loop, a
    ``TypeError`` from ``Path(None)``, or silently: an unknown ``contour_method`` fell through to
    skimage and an unknown ``output_format`` fell through to shapefile, so a run asked for one
    thing and quietly delivered another. Checked in the order a caller is most likely to have got
    wrong, cheapest first.
    """
    if int(sr_factor) < 1:
        raise ValueError(f"sr_factor must be >= 1, got {sr_factor}")
    if int(slice_size) <= 0:
        raise ValueError(f"slice_size must be positive, got {slice_size}")
    if int(slice_size) % int(sr_factor) != 0:
        raise ValueError(f"slice_size ({slice_size}) must be divisible by sr_factor "
                         f"({sr_factor}) — the native slicing window is slice_size / sr_factor")
    if int(inference_size) <= 0:
        raise ValueError(f"inference_size must be positive, got {inference_size}")
    if int(batch_size) <= 0:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if int(max_det) < 1:
        raise ValueError(f"max_det must be >= 1, got {max_det}")
    if on_max_det not in ("error", "warn"):
        raise ValueError(f"on_max_det must be 'error' or 'warn', got {on_max_det!r}")
    if not 0.0 <= float(confidence_threshold) <= 1.0:
        raise ValueError(f"confidence_threshold must be in [0, 1], got {confidence_threshold}")
    for name, value in (("overlap_height_ratio", overlap_height_ratio),
                        ("overlap_width_ratio", overlap_width_ratio)):
        if not 0.0 <= float(value) < 1.0:
            raise ValueError(f"{name} must be in [0, 1), got {value}")
    if not 0.0 < float(postprocess_match_threshold) <= 1.0:
        raise ValueError("postprocess_match_threshold must be in (0, 1], got "
                         f"{postprocess_match_threshold}")
    if min_area_threshold is not None and int(min_area_threshold) < 0:
        raise ValueError(f"min_area_threshold must be >= 0 or None, got {min_area_threshold}")
    if n_workers is not None and int(n_workers) < 1:
        raise ValueError(f"n_workers must be >= 1 or None, got {n_workers}")
    if contour_method not in CONTOUR_METHODS:
        raise ValueError(f"contour_method must be one of {CONTOUR_METHODS}, got "
                         f"{contour_method!r} — it used to fall through to skimage silently")
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(f"output_format must be one of {OUTPUT_FORMATS}, got "
                         f"{output_format!r} — it used to fall through to shapefile silently")
    if remask_fn is not None and not callable(remask_fn):
        raise TypeError(f"remask_fn must be callable, got {type(remask_fn).__name__}")
    if output_dir is None:
        raise ValueError("output_dir is required")
    if in_raster is None:
        raise ValueError("in_raster is required")
    if not Path(in_raster).exists():
        raise FileNotFoundError(f"in_raster does not exist: {in_raster}")
    if detection_model is None:
        raise ValueError("detection_model is required")
    for attr in ("model", "device", "category_mapping"):
        if not hasattr(detection_model, attr):
            raise TypeError(f"detection_model is missing .{attr} — expected a sahi DetectionModel")


def _ensure_png(in_raster, raster_identity):
    """Convert the raster to 8-bit PNG once, reusing an earlier conversion of the same bytes.

    ``tiff_to_png`` reads the whole TIFF and re-encodes it on every call, then overwrites the same
    output path — so a multiscale run redid the identical conversion once per arm, and two runs on
    different rasters with the same stem would clobber each other's PNG (review item G1-07). A
    sidecar records which source produced the PNG; the conversion is skipped when it still matches.
    """
    out_png = in_raster.with_name(in_raster.stem + ".png")
    stamp = out_png.with_name(out_png.name + ".source.json")
    key = {"sha256": raster_identity.get("sha256"), "bytes": raster_identity.get("bytes"),
           "path": raster_identity.get("path")}
    if out_png.exists() and stamp.exists():
        try:
            if json.loads(stamp.read_text()) == key:
                logger.debug("reusing %s (source unchanged)", out_png.name)
                return out_png
        except (OSError, json.JSONDecodeError):
            pass
    raster_convert.tiff_to_png(in_raster, out_png)   # only work with 8bit
    try:
        stamp.write_text(json.dumps(key))
    except OSError:                                  # read-only data dir: just skip the cache
        logger.debug("could not write %s; PNG will be reconverted next run", stamp.name)
    return out_png


def _to_bgr(image):
    """RGB -> BGR for the ultralytics numpy path, matching ``sahi``'s own model wrapper.

    ``sahi.slicing.slice_image`` reads through PIL and hands back **RGB** arrays, while
    ultralytics treats a numpy input as **BGR** (the ``cv2.imread`` convention) — which is why
    ``sahi``'s ``UltralyticsDetectionModel`` flips before calling the model. This pipeline called
    the model directly and skipped that flip. For the single-band rasters it runs on the PNG is
    grayscale-replicated (R == G == B), so the flip is a no-op and no saved result changes; it
    matters only if a true colour product is ever fed in. ``np.ascontiguousarray`` because the
    reversed view has a negative stride.
    """
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return arr
    return np.ascontiguousarray(arr[:, :, ::-1])


def edge_priority(score, is_within_slice):
    """NMS/fusion ordering key: real confidence for interior detections, demoted for truncated ones.

    Overlapping slices mean a boulder clipped by a slice boundary is usually *also* seen whole by
    the neighbouring slice. Suppression must always keep the whole copy, which the original code
    achieved by overwriting the truncated copy's ``score`` with the literal 0.10 — destroying the
    detector's real confidence and putting sub-threshold rows in every catalog (review X-02).

    The ordering job now lives here instead. Interior detections map onto ``(0.5, 1.0]`` and
    truncated ones onto ``(0, 0.5]``, so:

    - every untruncated detection outranks every truncated one, whatever the confidence threshold
      (the old fixed 0.10 failed this below ``conf=0.10``);
    - ordering *within* each group still follows real confidence — the old code tied every
      truncated row at exactly 0.10 and let array order break it;
    - values stay strictly positive, which ``lsnms.nms`` requires (it silently drops ``score <= 0``).
    """
    score = np.asarray(score, dtype=float)
    within = np.asarray(is_within_slice, dtype=bool)
    return np.where(within, 0.5 + 0.5 * score, 0.5 * score)


def _downscale_masks_on_device(masks, slice_size, downscale_pred):
    """Do the ``INTER_AREA`` downscale on the GPU, before the masks are moved to the host.

    Ultralytics returns ``masks.data`` at ``imgsz`` (1024), not at ``slice_size`` — G3-10's
    discovery — so the production path used to move a **4 MiB float32 mask per detection** across
    PCIe and then shrink it on the CPU. Profiling the production entry point put those two stages
    at a third of the whole run (slice 256: ``cv2.resize`` 19.4% + device-to-host 14.2%, 18.1 GiB
    moved for a single 2048 px crop; slice 512: 15.0% + 19.4%). Pooling first makes the transfer
    16x (slice 256) or 4x (slice 512) smaller and removes the CPU resize entirely.

    **This is bit-identical, not merely close.** The masks are already thresholded to exactly
    {0.0, 1.0}, and for an exact integer ratio ``INTER_AREA`` is a plain box mean, so every output
    value is ``n / k**2`` with ``k`` a power of two — a dyadic rational that float32 holds exactly
    under either implementation. Verified on 4,611 real detections across both arms: zero value
    differences and every traced polygon equal
    (``scripts/review/probe_gpu_mask_downscale.py``, ``tests/test_gpu_mask_downscale.py``).

    Anything outside that guarantee falls through untouched and is downscaled on the CPU exactly
    as before: a non-integer ratio, a mask already at or below ``slice_size``, a non-square mask,
    or ``downscale_pred=False`` (which keeps masks in the model frame on purpose, see G3-10).
    """
    if not downscale_pred or masks.ndim != 3:
        return masks
    height, width = masks.shape[-2], masks.shape[-1]
    if height != width or height <= slice_size or height % slice_size:
        return masks
    factor = height // slice_size
    return torch.nn.functional.avg_pool2d(masks.unsqueeze(1), kernel_size=factor,
                                          stride=factor).squeeze(1)


def _result_to_df(prediction_result, detection_model, has_mask, shift_amount, slice_size,
                  min_area_threshold, downscale_pred, contour_method="skimage", executor=None,
                  remask_fn=None, image=None, coord_scale=1.0, stats=None, remask_tag=None):
    """Convert a single ultralytics Result (one slice) into the detections DataFrame.

    This is the per-slice core extracted from ``YOLOv8`` so the model can be called on a
    *batch* of slices (see ``get_sliced_prediction``) instead of one at a time. Masks are
    thresholded at 0.5, optionally resized to ``slice_size`` with ``INTER_AREA``, and converted
    to polygons.

    ``contour_method`` selects the mask->polygon extractor:
    - ``"skimage"`` (default): ``binary_mask_to_polygon`` (skimage ``find_contours``) — the
      original behaviour, bit-identical geometry.
    - ``"cv2"``: ``binary_mask_to_polygon_cv`` (OpenCV) — much faster, sparser vertices;
      negligible effect on orientation (see Goal-2 bridge notebook), opt-in.

    ``executor``: an optional ``ThreadPoolExecutor``. The per-detection work (resize, contour
    extraction, geometry) is independent and dominated by GIL-releasing C code
    (``find_contours``, ``cv2``, numpy), so mapping it across threads gives a large speedup on
    dense slices. Results are collected in input order, so output is identical to the serial
    path.

    ``remask_fn`` (Solution G, detector-preserving re-masking): if given, YOLO's boxes and scores
    are kept but each detection's **mask** is replaced by ``remask_fn(image, boxes_xyxy)`` — a
    callable returning one slice-resolution mask per box (or ``None`` to drop a failed re-mask),
    e.g. the SAM2 box-prompt re-masker from :func:`YOLOv8BeyondEarth.orientation_validation.sam2_remasker`.
    ``image`` is the slice the boxes are in (needed to prompt the re-masker). Default ``None`` =
    original YOLO-prototype masks (bit-identical).

    ``coord_scale``: multiplier applied to slice-frame polygon coordinates *before* the shift to
    full-image coordinates. Used by the streamed-SR path (Goal 3): a slice upscaled ×f before
    inference yields polygons in SR px, and ``coord_scale = 1/f`` maps them back onto the native
    grid that ``shift_amount`` lives on. Edge logic runs in the (SR) slice frame, before scaling.
    Default ``1.0`` = untouched (bit-identical).

    ``stats``: an optional dict accumulating **why** detections were discarded, keyed by reason
    (``below_area_gate``, ``no_contour``, ``degenerate_polygon``, ``remask_failed_no_fallback``,
    ``trace_failed:<ExcType>``), plus the non-fatal ``remask_fallback_to_yolo``. Without it a
    systematic failure is indistinguishable from an empty scene (review item X-05).

    ``remask_tag``: the name recorded in ``mask_src`` for successfully re-masked rows (e.g.
    ``"sam2"``). Rows that fell back to the detector's own mask are tagged ``"yolo"``, so a
    downstream comparison can report both success-only and intention-to-treat summaries
    (review item G2-03).
    """
    extract_polygon = binary_mask_to_polygon_cv if contour_method == "cv2" else binary_mask_to_polygon
    shift_x = shift_amount[0]
    shift_y = shift_amount[1]

    # if no predictions
    if prediction_result.boxes.data.size()[0] == 0:
        return pd.DataFrame(columns=_DETECTION_COLUMNS)

    conf_mask = prediction_result.boxes.data[:, 4] >= detection_model.confidence_threshold
    result_boxes = prediction_result.boxes.data[conf_mask]
    boxes_np = result_boxes.cpu().detach().numpy()

    def _yolo_masks():
        # Threshold the whole mask batch in one GPU op (equivalent to the old per-mask
        # ``bool_mask[bool_mask>=0.5]=1; [<0.5]=0``) before moving to CPU. Kept as float32 {0.,1.}
        # so the later INTER_AREA downscale yields the same fractional mask the contour extractor
        # sees -> geometry-neutral, just without two full-array writes per detection.
        hard = (prediction_result.masks.data[conf_mask] >= 0.5).to(torch.float32)
        hard = _downscale_masks_on_device(hard, slice_size, downscale_pred)
        out = hard.cpu().detach().numpy()
        if len(out) != len(boxes_np):
            raise ValueError(f"model returned {len(out)} masks for {len(boxes_np)} boxes "
                             "— box/mask cardinality must match")
        return out

    fallback_masks = None
    if remask_fn is not None:
        # Solution G: swap the mask producer, keep YOLO's boxes/scores. ``remask_fn`` returns a
        # per-box list of slice-resolution masks, or None where the re-mask failed its guards.
        masks_np = remask_fn(image, boxes_np[:, :4])
        # X-05: `zip` below would silently truncate to the shorter of the two, losing detections
        # without a trace. A re-masker that returns the wrong count is a bug in the re-masker.
        if len(masks_np) != len(boxes_np):
            raise ValueError(f"remask_fn returned {len(masks_np)} masks for {len(boxes_np)} "
                             "boxes — it must return exactly one entry per box (None to drop)")
        # G2-03: a failed re-mask used to delete the detection, which is not detector-preserving —
        # a saved wrapper check lost 177 of 26,112 rows (0.68%) and one no-overlap run logged 777
        # native-SAM failures, so the "identical detections, different masks" premise of Solution G
        # did not actually hold and the SAM/YOLO comparison was over slightly different
        # populations. The detector's own mask is kept instead, tagged in `mask_src`.
        if has_mask and getattr(prediction_result, "masks", None) is not None:
            fallback_masks = _yolo_masks()
    else:
        # has_mask=False is rejected by the public entry points (polygon pipeline needs masks)
        masks_np = _yolo_masks()

    min_edge_distance = 0.05 * slice_size
    max_edge_distance = 0.95 * slice_size

    def _drop(reason):
        """Record why a detection was discarded. Counts are reported by the public wrapper."""
        if stats is not None:
            stats[reason] = stats.get(reason, 0) + 1
        return None

    def _process_detection(item):
        """Mask -> (score, polygon, category_id, category_name, is_within_slice) or None.

        Every discard is counted by reason (X-05): this used to swallow *all* exceptions and
        return None, so a systematic failure — a broken re-masker, an unmapped class, a bad
        tracer — looked exactly like "there were no boulders here".

        Free of shared mutable state apart from the `stats` counter, whose increments are
        CPython-atomic dict operations, so it stays safe across the thread pool.
        """
        prediction, bool_mask, fallback = item
        mask_src = remask_tag or "yolo"
        if bool_mask is None:          # re-masker failed its guards (e.g. background-grab)
            if fallback is None:
                return _drop("remask_failed_no_fallback")
            bool_mask, mask_src = fallback, "yolo"
            if stats is not None:      # counted, but the detection survives
                stats["remask_fallback_to_yolo"] = stats.get("remask_fallback_to_yolo", 0) + 1
        category_id = int(prediction[5])

        if downscale_pred and bool_mask.shape[0] != slice_size:
            bool_mask = cv2.resize(bool_mask, (slice_size, slice_size), interpolation=cv2.INTER_AREA)

        # Supported pixels at the level the tracer uses; min_area_threshold=None (the documented
        # default) means no area gate. Counting `== 1` instead — as this did before G1-06 — is
        # equivalent for a hard mask but discards downscaled boulders that have ample support at
        # 0.5 yet few *fully covered* pixels. That arises whenever the returned mask is LARGER than
        # slice_size, which is every production config: ultralytics returns `masks.data` at `imgsz`
        # (1024), not at the prototype size, so `downscale_pred=True` is a genuine 4x (slice 256)
        # or 2x (slice 512) downscale and 100% of production masks are fractional. Measured on the
        # NAC dev crop: the gate change rescues +0.58% of detections at the production 6 px
        # threshold, all tiny (median 3.19 native px). An earlier version of this comment claimed
        # the prototype was upscaled and the change was inert — that was wrong; see the correction
        # in `docs/code_review_backlog.md`.
        area = mask_support_area(bool_mask)
        if area <= (min_area_threshold if min_area_threshold is not None else 0):
            return _drop("below_area_gate")

        # Only genuine geometry failures are tolerated here. A KeyError from an unmapped class,
        # or anything else, is a configuration bug and must propagate rather than quietly
        # shrinking the catalog.
        try:
            polygon = extract_polygon(bool_mask)
        except (ValueError, IndexError, cv2.error) as exc:
            return _drop(f"trace_failed:{type(exc).__name__}")
        if polygon is None:
            return _drop("no_contour")
        if len(polygon) < 3:
            return _drop("degenerate_polygon")

        if downscale_pred:
            # the *mask* was resized by cv2, which is already pixel-centre correct
            polygon_slice = polygon
        else:
            # the mask stayed in the model's frame (ultralytics returns masks at `imgsz`, not at
            # slice_size), so the coordinates carry the rescale — centre-aware (G3-10)
            polygon_slice = rescale_pixel_coords(polygon, slice_size / bool_mask.shape[0])
        is_polygon_within_slice = (np.logical_and(polygon_slice[:, 0].min() >= min_edge_distance,
                                                  polygon_slice[:, 0].max() <= max_edge_distance) and
                                   np.logical_and(polygon_slice[:, 1].min() >= min_edge_distance,
                                                  polygon_slice[:, 1].max() <= max_edge_distance))

        # X-02: `score` is always the detector's real confidence. Slice-truncated detections
        # are demoted for *suppression ordering only*, in `nms_prio`.
        score = float(prediction[4])
        if coord_scale != 1.0:   # streamed SR: SR px -> native px (shift is in native px)
            polygon_slice = rescale_pixel_coords(polygon_slice, coord_scale)
        shifted_polygon = shift_polygon(polygon_slice, shift_x, shift_y)
        try:
            category_name = detection_model.category_mapping[str(category_id)]
        except KeyError:
            raise KeyError(
                f"model emitted class id {category_id}, which is not in category_mapping "
                f"({sorted(detection_model.category_mapping)}). Dropping it silently would "
                "shrink the catalog with no record — fix the mapping or the weights.") from None
        return (score, shifted_polygon, category_id, category_name, is_polygon_within_slice,
                mask_src)

    n = len(boxes_np)
    items = list(zip(boxes_np, masks_np,
                     fallback_masks if fallback_masks is not None else [None] * n))
    if executor is not None and len(items) > 1:
        results = executor.map(_process_detection, items)
    else:
        results = map(_process_detection, items)

    scores = []
    polygons = []
    category_ids = []
    category_names = []
    is_polygon_within_slice_list = []
    mask_srcs = []
    for res in results:
        if res is None:
            continue
        scores.append(res[0])
        polygons.append(res[1])  # polygon in absolute (shifted) coordinates
        category_ids.append(res[2])
        category_names.append(res[3])
        is_polygon_within_slice_list.append(res[4])
        mask_srcs.append(res[5])

    data = {'score': scores,
            'nms_prio': edge_priority(scores, is_polygon_within_slice_list).tolist()
                        if scores else [],
            'polygon': polygons,
            'category_id': category_ids, 'category_name': category_names,
            'is_within_slice': is_polygon_within_slice_list,
            'mask_src': mask_srcs}

    return pd.DataFrame(data, columns=_DETECTION_COLUMNS)


def YOLOv8(detection_model, image, has_mask, shift_amount, slice_size, min_area_threshold,
           downscale_pred, contour_method="skimage", max_det=ULTRALYTICS_DEFAULT_MAX_DET):
    """
    Single-image (single-slice) prediction. Kept for backward compatibility; the batched
    pipeline in ``get_sliced_prediction`` calls ``_result_to_df`` directly.

    YOLOv8 expects numpy arrays to have BGR (height, width, 3).

    1. Let's say you want to detect very very small objects, the slice height and width should be
    pretty small, and detection_model.image_size should be increased:
    - slice_height, slice_width = 256
    - detection_model.image_size = 1024.

    2. If you are in the opposite situation, where you realized that most of the large boulders are missed out. You can
    increase the slice height and width.
    - slice_height, slice_width = 1024
    - detection_model.image_size = 512, 1024.

    You can get the best of both worlds by combining predictions (1) and (2) with NMS. Obviously the larger the
    slices height and width, and the larger the detection_model.image_size, the more time it takes to run this
    script.

    If the predictions is starting to be very large compare to the size of the slice, WBF can be advantageous as it
    will merge the overlapping boulders. However, WBF and NMS works better in less dense area (or at least is less
    sensitive to the iou_threshold selected).

    Test Time Augmentation could be included too, but it takes lot of time to run it.. so not sure about that too.
    WBF for instance seg: "https://www.kaggle.com/code/mistag/sartorius-tta-with-weighted-segments-fusion"

    Note that the bboxes (in absolute coordinates) are calculated from the bounds of the polygons after the
    predictions are computed.
    """
    if not has_mask:
        raise NotImplementedError(
            "has_mask=False (detection-only models) is not supported: the pipeline is polygon-based, "
            "and without masks every detection would be silently dropped by the area gate. "
            "Use a -seg model.")

    prediction_results = detection_model.model(_to_bgr(image), imgsz=detection_model.image_size,
                                               conf=detection_model.confidence_threshold,
                                               max_det=max_det,
                                               verbose=False, device=detection_model.device)
    return _result_to_df(prediction_results[0], detection_model, has_mask, shift_amount, slice_size,
                         min_area_threshold, downscale_pred, contour_method)


def prediction_filenames(stem, *, confidence_threshold, slice_size, inference_size,
                         overlap_height_ratio, sr_factor=1, downscale_pred=False,
                         mask_tag=None, output_format="shp"):
    """The output filenames :func:`get_sliced_prediction` writes, for a raster named ``stem``.

    Exposed as a function because **downstream analysis has to locate these catalogs**, and code
    that hardcodes the pattern silently stops finding them the moment any encoded parameter moves.
    That is not hypothetical: three call sites in ``data/solutionDE/analyze_hseg_acceptance.py``
    pinned ``ct-010`` and an untagged mask, so the H-seg x G run — which changes *both*, by
    retuning ``confidence_threshold`` and passing ``mask_tag="sam2"`` — could not be scored at all
    (found 2026-08-09). Import this instead of re-deriving the pattern.

    Returns the four layers the function can write: ``bbox``, ``mask``, ``bbox_nms`` and ``nms``.
    ``nms`` is the post-NMS *mask* layer — the one analysis normally wants. Which subset actually
    lands on disk depends on ``save_bbox`` / ``save_prenms`` / ``save_nms`` and ``postprocess``.
    """
    ext = ".gpkg" if output_format == "gpkg" else ".shp"
    ss_str = str(slice_size) if sr_factor == 1 else f"{slice_size}sr{sr_factor}"
    bbox = (str(stem) + "-predictions-ct-" + str(int(confidence_threshold * 100)).zfill(3) + "-ss-"
            + ss_str + "-is-" + str(inference_size) + "-ov-"
            + str(int(overlap_height_ratio * 100)).zfill(3) + "-bbox" + ext)
    mask = bbox.replace("-bbox" + ext, "-mask" + ext)
    if downscale_pred:
        bbox = bbox.replace("-bbox" + ext, "-downscaled-bbox" + ext)
        mask = mask.replace("-mask" + ext, "-downscaled-mask" + ext)
    if mask_tag:  # keep a re-masked layer (e.g. SAM2) from overwriting the plain YOLO one
        bbox = bbox.replace("-bbox" + ext, f"-bbox-{mask_tag}" + ext)
        mask = mask.replace("-mask" + ext, f"-mask-{mask_tag}" + ext)
    return {"bbox": bbox, "mask": mask,
            "bbox_nms": bbox[: -len(ext)] + "-nms" + ext,
            "nms": mask[: -len(ext)] + "-nms" + ext}


def get_sliced_prediction(in_raster,
                          detection_model=None,
                          confidence_threshold: float = 0.1,
                          has_mask=True,
                          output_dir=None,
                          interim_file_name=None,  # ADDED OUTPUT FILE NAME TO (OPTIONALLY) SAVE SLICES
                          interim_dir=None,  # ADDED INTERIM DIRECTORY TO (OPTIONALLY) SAVE SLICES
                          slice_size: int = 256,
                          inference_size: int = 1024,
                          overlap_height_ratio: float = 0.2,
                          overlap_width_ratio: float = 0.2,
                          min_area_threshold: int = None,
                          downscale_pred: bool = False,
                          postprocess: bool = True,
                          postprocess_match_threshold: float = 0.5,
                          postprocess_class_agnostic: bool = False,
                          batch_size: int = 8,
                          max_det: int = ULTRALYTICS_DEFAULT_MAX_DET,
                          on_max_det: str = "error",
                          contour_method: str = "skimage",
                          save_bbox: bool = True,
                          save_prenms: bool = True,
                          save_nms: bool = True,
                          output_format: str = "shp",
                          n_workers: int = None,
                          remask_fn=None,
                          mask_tag: str = None,
                          sr_factor: int = 1,
                          on_conflict: str = "error"):
    """
    Function for slice image + get predicion for each slice + combine predictions in full image.

    .. warning::
       **These defaults are not a production configuration.** They are the historical generic
       defaults (no area gate, NMS 0.5, Shapefile, every intermediate layer, prototype-resolution
       masks), which reproduce neither the geometry, the count, the cost nor the output safety of
       any benchmarked run. Call :func:`predict_production` — or pass
       ``**presets.GOAL1_PRODUCTION`` — for the configuration Goal 1 actually measured
       (review item G1-01).

    The time to run the script is dependent on the number of predictions over the whole image. This is because we need
    to loop through each prediction and transform the bool_mask to polygon.

    Args:
        in_raster: str or Path()
            Path to raster tif file.
        detection_model: model.DetectionModel
        confidence_threshold: float
            minimum confidence threshold, values below will be automatically filtered away.
            Passed straight through to ultralytics as ``conf=`` — before contract version 2 it
            was not, so every run silently used ultralytics' own 0.25 default no matter what was
            requested, and any catalog named ``ct-010`` is really a 0.25 catalog (review X-01).
        has_mask: bool
            Whether the model produces masks. Must be ``True`` unless ``remask_fn`` is given
            (the polygon pipeline needs a mask source) — ``False`` without a ``remask_fn``
            raises ``NotImplementedError``.
        interim_dir: str or Path()
        slice_size: int
            Height and width of each slice. Defaults to ``256``.
        inference_size: int
            Model input size (``imgsz``) each slice is inferred at. Defaults to ``1024``.
        min_area_threshold: int or None
            Drop detections whose mask area (pixels) is ``<=`` this. ``None`` (default) = no gate.
        downscale_pred: bool
            If ``True``, keep the mask at the model's native prototype resolution instead of
            resizing polygons back up to ``slice_size``. Defaults to ``False``.
        batch_size: int
            Number of slices per batched GPU inference call. Defaults to ``8``.
        max_det: int
            Maximum detections ultralytics returns **per slice**. Defaults to
            :data:`ULTRALYTICS_DEFAULT_MAX_DET` (300), which is the value ultralytics silently
            applied before contract version 3 — so the default is unchanged behaviour, and the
            production value lives in :data:`presets.GOAL1_PRODUCTION` (review item X-06).

            The cap is applied *after* NMS and keeps the top N **by confidence**, so it discards the
            faintest detections in the densest slices. Raising it costs VRAM rather than NMS time:
            ``ops.process_mask`` materialises masks at ``inference_size``, so peak memory scales
            with ``max_det x batch_size`` (~1 MiB per detection retained at imgsz 1024, ~5 MiB
            transient). ``max_nms`` bounds NMS's input at 30,000 independently, so the time side is
            free. Size it with ``scripts/review/probe_max_det.py``.
        on_max_det: str
            What to do when a slice comes back holding exactly ``max_det`` detections, i.e. when the
            model's output was cut off. ``"error"`` (default) raises on the first such batch, so a
            multi-hour scene arm fails in its first minutes rather than writing a quietly truncated
            catalog; ``"warn"`` records the count and continues. Either way the count reaches
            ``attrs["truncation"]`` and the manifest.
        contour_method: str
            Mask->polygon extractor: ``"skimage"`` (default, bit-identical) or ``"cv2"`` (faster).
        overlap_height_ratio: float
            Fractional overlap in height of each window (e.g. an overlap of 0.2 for a window
            of size 512 yields an overlap of 102 pixels).
            Default to ``0.2``.
        overlap_width_ratio: float
            Fractional overlap in width of each window (e.g. an overlap of 0.2 for a window
            of size 512 yields an overlap of 102 pixels).
            Default to ``0.2``.
        postprocess: bool
            Include postprocessing or not.
        postprocess_match_threshold: float
            Sliced predictions having higher iou than postprocess_match_threshold will be
            postprocessed after sliced prediction.
        postprocess_class_agnostic: bool
            If True, postprocess will ignore category ids.
        save_bbox: bool
            Write the bounding-box layers (pre- and post-NMS). Default ``True``. Set ``False``
            to skip them when only the mask outlines are needed.
        save_prenms: bool
            Write the pre-NMS (un-suppressed) layers. Default ``True``. Set ``False`` to write
            only the final post-NMS layer(s) — useful for dense full-image runs.
        save_nms: bool
            Write the post-NMS layer. Default ``True``. The multiscale wrapper sets ``False``
            because its deliverable is the *fused* layer; the per-arm layers were being written
            anyway, contradicting the wrapper's own contract (review item G3-09).
        output_format: str
            ``"shp"`` (default) or ``"gpkg"``. GeoPackage avoids the shapefile 2 GB / field
            limits and is faster to write for very large (~1e6-feature) outputs.
        n_workers: int
            Threads for the per-detection mask->polygon post-processing. Default
            ``min(8, os.cpu_count())``; pass ``1`` to force the serial path. Output is identical
            regardless of worker count (order-preserving).
        remask_fn: callable or None
            Solution G (detector-preserving re-masking). If given, YOLO still supplies every box and
            score, but each detection's **mask** is produced by ``remask_fn(slice_image, boxes_xyxy)``
            (e.g. SAM2 box prompts — see
            :func:`YOLOv8BeyondEarth.orientation_validation.sam2_remasker`) instead of the YOLO
            prototype head. Everything else (slicing, edge handling, dedup, NMS, output) is
            unchanged, so detection is identical and only the mask geometry differs. Default ``None``
            = original behaviour.
        mask_tag: str or None
            Inserted into the output filenames (e.g. ``"sam2"`` -> ``...-mask-sam2.gpkg``) so a
            re-masked layer does not overwrite the plain YOLO one. Default ``None`` = no tag.
        sr_factor: int
            Streamed super-resolution (Goal 3's SR arm). ``sr_factor=f > 1`` slices the *native*
            raster at ``slice_size / f`` px, Lanczos-upscales each slice ×``f`` to ``slice_size``
            **inside the batch loop** (the ×f image is never materialized — a ×2 NAC scene would
            be ~1.2e9 px), and maps polygons back to the native grid. ``slice_size`` keeps its
            meaning of *model-input* slice px (matching the ``(slice_size, sr_factor)`` fusion
            keys), so ``slice_size=256, sr_factor=2`` = the E-arm's "slice 256 SR px, 128-native
            footprint" at the trained 4× internal resize. ``slice_size`` must be divisible by
            ``sr_factor``. Output filenames carry ``-ss-256sr2-``. Default ``1`` = bit-identical
            native path.

    Returns:
        ``(gdf, gdf_nms)`` — a tuple of GeoDataFrames: all kept detections and the post-NMS
        subset. ``gdf_nms`` is ``None`` when ``postprocess=False``.

        Score columns (contract version 2, review X-02):

        - ``score`` — the detector's **real** confidence, never overwritten. Every row therefore
          satisfies ``score >= confidence_threshold``. In contract-version-1 catalogs this column
          instead held the literal 0.10 for any slice-truncated detection.
        - ``nms_prio`` — the suppression ordering key (see :func:`edge_priority`); untruncated
          detections outrank truncated ones. Fusion uses it for the same reason NMS does.
        - ``mask_src`` — which producer this row's geometry came from: ``"yolo"``, or ``mask_tag``
          when a ``remask_fn`` succeeded. A failed re-mask **keeps the detection** with the
          detector's own mask rather than deleting the row, so Solution G is genuinely
          detector-preserving (review item G2-03); the count is in
          ``attrs["fallbacks"]["remask_fallback_to_yolo"]``.

        ``gdf.attrs["discarded"]`` holds the per-reason count of detections the model produced but
        the pipeline did not keep, and ``gdf.attrs["resolved_config"]`` the parameters that
        produced the result (review items X-05 and G1-01).

        ``gdf.attrs["fallbacks"]`` counts detections that **are** in the catalog but were produced by
        a fallback — today only ``remask_fallback_to_yolo``. It is separate from ``discarded``
        because these rows were kept: counting them as losses made a healthy Solution-G run report a
        "pipeline failure" and understated the catalog it had just written (review item A3).

        ``gdf.attrs["truncation"]`` holds the **slice-level** losses — detections the model never
        returned at all, so they cannot appear in the per-detection tally (review item X-06):
        ``slices_at_max_det`` (a lower bound, see :func:`_note_slice_truncation`),
        ``slices_near_max_det``, ``nms_time_limit_batches`` and
        ``slices_zeroed_after_nms_time_limit`` (an upper bound). An empty dict means the run lost
        nothing this way, which is the acceptance condition for X-06.

    Raises:
        ValueError / FileNotFoundError for any invalid argument, before any work is done.
    """
    _validate_prediction_args(
        in_raster=in_raster, detection_model=detection_model,
        confidence_threshold=confidence_threshold, output_dir=output_dir, slice_size=slice_size,
        inference_size=inference_size, overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio, min_area_threshold=min_area_threshold,
        postprocess_match_threshold=postprocess_match_threshold, batch_size=batch_size,
        contour_method=contour_method, output_format=output_format, n_workers=n_workers,
        remask_fn=remask_fn, sr_factor=sr_factor, max_det=max_det, on_max_det=on_max_det)

    if not has_mask and remask_fn is None:
        raise NotImplementedError(
            "has_mask=False (detection-only models) is not supported without a remask_fn: the "
            "pipeline is polygon-based, and without masks every detection would be silently dropped "
            "by the area gate. Use a -seg model, or supply remask_fn (e.g. sam2_remasker) to build "
            "masks from the boxes.")

    sr_factor = int(sr_factor)
    native_slice = slice_size // sr_factor   # window on the native grid; == slice_size when sr=1
    discarded = {}                           # detection -> reason it was not kept (X-05)
    truncation = {}                          # slice-level losses the model never returned (X-06)
    started = time.perf_counter()

    in_raster = Path(in_raster)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Every parameter that defines the result, so the manifest can answer "what made this file"
    # and a differing config cannot silently overwrite it (G1-01/G1-07).
    resolved_config = {
        "inference_contract_version": INFERENCE_CONTRACT_VERSION,
        "confidence_threshold": float(confidence_threshold), "slice_size": int(slice_size),
        "inference_size": int(inference_size), "sr_factor": sr_factor,
        "overlap_height_ratio": float(overlap_height_ratio),
        "overlap_width_ratio": float(overlap_width_ratio),
        "min_area_threshold": min_area_threshold, "downscale_pred": bool(downscale_pred),
        "postprocess": bool(postprocess),
        "postprocess_match_threshold": float(postprocess_match_threshold),
        "postprocess_class_agnostic": bool(postprocess_class_agnostic),
        "contour_method": contour_method, "output_format": output_format,
        "batch_size": int(batch_size), "max_det": int(max_det), "mask_tag": mask_tag,
        # Not `__name__`: a re-masker replaces *every* mask in the catalog, so which one ran is
        # result-defining, and `sam2_remasker`'s closure is always called `remask` (review item A1).
        "remask_fn": presets.describe_callable(remask_fn),
        "model_path": str(getattr(detection_model, "model_path", None)),
    }

    # Output names depend only on parameters known now, so resolve them and refuse a conflicting
    # configuration BEFORE any inference runs. Checking at write time instead means a full scene
    # burns its whole compute budget and then raises on the last line (G1-07 follow-up).
    ext = ".gpkg" if output_format == "gpkg" else ".shp"
    _names = prediction_filenames(
        in_raster.stem, confidence_threshold=confidence_threshold, slice_size=slice_size,
        inference_size=inference_size, overlap_height_ratio=overlap_height_ratio,
        sr_factor=sr_factor, downscale_pred=downscale_pred, mask_tag=mask_tag,
        output_format=output_format)
    bbox_filename, mask_filename = _names["bbox"], _names["mask"]
    out_bbox_shp = output_dir / bbox_filename
    out_mask_shp = output_dir / mask_filename
    nms_mask_shp = output_dir / _names["nms"]
    presets.check_config_conflict(nms_mask_shp if postprocess else out_mask_shp,
                                  resolved_config, on_conflict)

    raster_identity = presets.file_identity(in_raster)
    out_png = _ensure_png(in_raster, raster_identity)

    # set model's confidence_threshold and inference size
    detection_model.image_size = inference_size
    detection_model.confidence_threshold = confidence_threshold

    slice_image_result = slice_image(
        image=out_png.as_posix(),  # need to be path to
        output_file_name=interim_file_name,  # ADDED OUTPUT FILE NAME TO (OPTIONALLY) SAVE SLICES
        output_dir=interim_dir,  # ADDED INTERIM DIRECTORY TO (OPTIONALLY) SAVE SLICES
        slice_height=native_slice,
        slice_width=native_slice,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        out_ext=".png",  # FORMAT OF (OPTIONALLY) SAVED SLICES
    )

    num_slices = len(slice_image_result)
    shift_amounts = slice_image_result.starting_pixels
    slice_images = slice_image_result.images
    frames = []
    # The mask->polygon post-processing is CPU-bound and embarrassingly parallel per detection;
    # run it across a thread pool (find_contours / cv2 / numpy release the GIL). Output order is
    # preserved, so results are identical to the serial path. Default to a modest worker count
    # (returns diminish past ~8 and we don't want to oversubscribe alongside the GPU thread).
    workers = n_workers if n_workers is not None else min(8, os.cpu_count() or 1)
    executor = ThreadPoolExecutor(max_workers=workers) if workers and workers > 1 else None
    try:
        # perform sliced prediction in batches (batched GPU inference instead of one slice at a
        # time). Per-slice geometry is unchanged; only the model call is grouped.
        with _watch_nms_time_limit() as nms_timeouts:
            seen_timeouts = 0
            for start in tqdm(range(0, num_slices, batch_size), total=(num_slices + batch_size - 1) // batch_size):
                batch_images = list(slice_images[start:start + batch_size])
                if sr_factor > 1:
                    # Streamed SR: upscale each native slice to slice_size (SR px) just before
                    # inference — same Lanczos kernel as the whole-crop SR rasters it must be
                    # equivalent to (prep_de_crops.py). shift_amounts stay on the native grid;
                    # coord_scale below maps polygons back onto it.
                    batch_images = [cv2.resize(img, (slice_size, slice_size),
                                               interpolation=cv2.INTER_LANCZOS4)
                                    for img in batch_images]
                prediction_results = detection_model.model([_to_bgr(img) for img in batch_images],
                                                           imgsz=detection_model.image_size,
                                                           conf=detection_model.confidence_threshold,
                                                           max_det=max_det,
                                                           verbose=False, device=detection_model.device)
                if len(prediction_results) != len(batch_images):
                    raise ValueError(f"model returned {len(prediction_results)} results for "
                                     f"{len(batch_images)} slices — results would be paired with the "
                                     "wrong shift_amounts")
                # Before any per-detection work: did the model's own output get cut off? Checked
                # here so a scene arm fails in its first minutes rather than after hours (X-06).
                total_timeouts = nms_timeouts()
                _note_slice_truncation(prediction_results, max_det, truncation, on_max_det,
                                       first_slice=start,
                                       new_timeouts=total_timeouts - seen_timeouts)
                seen_timeouts = total_timeouts
                for j, prediction_result in enumerate(prediction_results):
                    df = _result_to_df(prediction_result, detection_model, has_mask, shift_amounts[start + j],
                                       slice_size, min_area_threshold, downscale_pred, contour_method,
                                       executor=executor, remask_fn=remask_fn, image=batch_images[j],
                                       coord_scale=1.0 / sr_factor, stats=discarded,
                                       remask_tag=mask_tag)
                    if df.shape[0] > 0:
                        frames.append(df)
    finally:
        if executor is not None:
            executor.shutdown()

    if len(frames) == 0:
        df_all = pd.DataFrame(columns=_DETECTION_COLUMNS)
    else:
        df_all = pd.concat(frames, ignore_index=True)
    gdf = add_geometries(in_raster, df_all)

    # keep edge predictions (within 10% of slice size from the true footprint edge)

    # extract true footprint. Per-run temp directory (G1-04): this used to be a fixed
    # ~/tmp/true-footprint.shp shared by every process on the machine, so two concurrent scenes --
    # exactly what Goal 1 proposes and Goal 4 requires -- would overwrite each other's footprint
    # and cross-contaminate their edge classification with no error at all.
    tmp_dir = Path(tempfile.mkdtemp(prefix="bouldernet-run-"))
    try:
        gdf_true_footprint = raster.true_footprint(in_raster, tmp_dir / "true-footprint.shp")
        in_res = raster_metadata.get_resolution(in_raster)[0]
        in_meta = raster_metadata.get_profile(in_raster)
        gpd.GeoDataFrame(geometry=gdf_true_footprint.geometry.boundary.values, crs=in_meta["crs"].to_wkt()).to_file(
            tmp_dir / "true-footprint-as-a-line.shp")
        gdf_line_buffer = shp.buffer(tmp_dir / "true-footprint-as-a-line.shp", native_slice * 0.10 * in_res,
                                     (tmp_dir / "footprint-buffer.shp"))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Flag boulders touching the footprint-edge buffer. We only need the boolean flag, not the
    # materialized intersection geometries, so use a spatial join (intersects predicate)
    # instead of gpd.overlay(how="intersection") — far cheaper and lighter on memory at ~1e6
    # boulders. The buffer was round-tripped through a shapefile whose .prj uses the ESRI-WKT
    # dialect of the same projection; normalize its CRS to the boulders' (identical grid) to
    # avoid a spurious CRS-mismatch and keep the join on one coordinate system.
    gdf_line_buffer = gdf_line_buffer.set_crs(gdf.crs, allow_override=True)
    edge_idx = gpd.sjoin(gdf[["geometry"]], gdf_line_buffer[["geometry"]],
                         predicate="intersects", how="inner").index.unique()
    gdf["is_at_edge"] = False
    gdf.loc[edge_idx, "is_at_edge"] = True

    # keep edge predictions close to the edge of the footprint of the raster, but otherwise remove edge predictions
    n_before_edge = len(gdf)
    gdf = gdf.loc[np.logical_or(gdf.is_at_edge == True, gdf.is_within_slice == True)]
    discarded["slice_truncated_interior"] = int(n_before_edge - len(gdf))

    # remove duplicates
    n_before_dedup = len(gdf)
    gdf = gdf.drop_duplicates(subset="geometry", ignore_index=True)
    discarded["duplicate_geometry"] = int(n_before_dedup - len(gdf))
    gdf["id"] = gdf.index
    _report_discards(gdf, discarded, n_kept=len(gdf), truncation=truncation)

    # output paths and the conflict check were resolved before inference ran
    gdf.attrs["resolved_config"] = resolved_config

    # The pre-NMS files are intermediate, and bbox files aren't needed by many workflows.
    # Gate both so a dense full-image run can avoid writing ~1e6-feature layers it won't use.
    if save_prenms:
        if save_bbox:
            bboxes_to_shp(gdf, out_bbox_shp)
        outlines_to_shp(gdf, out_mask_shp)


    def _finish(gdf_nms, written):
        """Serialize the resolved config and provenance beside the primary deliverable (G1-01)."""
        weights = getattr(detection_model, "model_path", None)
        # Every model whose weights shaped this catalog, not just the detector's. A re-masker built
        # by a factory advertises its own inputs (already-hashed identity dicts, so the manifest does
        # not re-read a 300 MB checkpoint) — review item A1.
        inputs = {"raster": raster_identity,     # already hashed for the PNG cache; don't redo it
                  **({"weights": weights} if weights else {}),
                  **(getattr(remask_fn, "inputs", None) or {} if remask_fn else {})}
        presets.write_manifest(
            nms_mask_shp if postprocess else out_mask_shp,
            config=resolved_config,
            inputs=inputs,
            outputs={"files": [p.name for p in written],
                     "n_detections": int(len(gdf)),
                     "n_after_nms": None if gdf_nms is None else int(len(gdf_nms))},
            crs=str(gdf.crs) if len(gdf) else None,
            discarded=gdf.attrs.get("discarded", {}),
            fallbacks=gdf.attrs.get("fallbacks", {}),
            truncation=gdf.attrs.get("truncation", {}),
            wall_seconds=round(time.perf_counter() - started, 3),
            n_slices=int(num_slices))
        return gdf, gdf_nms

    if postprocess:
        # Zero detections on the whole raster (sparse tiles): NMS input would be an empty
        # stack. Short-circuit to an empty gdf_nms; the *_to_shp writers skip featureless
        # layers, so callers treat a missing NMS file as an empty catalog.
        if len(gdf) == 0:
            return _finish(gdf.copy(), [])
        # Non-maximum suppression (NMS). Ordered by `nms_prio`, not `score`: the truncated copy
        # of a boulder straddling a slice boundary must lose to the whole copy from the
        # neighbouring slice, while `score` keeps the detector's real confidence (X-02).
        nms_scores = gdf.nms_prio.to_numpy(dtype=float)
        # regardless of the classes ids (right now is a class agnoistic not supported)
        if postprocess_class_agnostic:
            keep = nms(boxes=np.stack(gdf.bbox.values), scores=nms_scores,
                       iou_threshold=postprocess_match_threshold, class_ids=None, rtree_leaf_size=32)
        # or taking into account the classes ids
        else:
            keep = nms(boxes=np.stack(gdf.bbox.values), scores=nms_scores,
                       iou_threshold=postprocess_match_threshold, class_ids=gdf.category_id.values, rtree_leaf_size=32)

        # saving post-processed predictions (the NMS mask layer is the primary deliverable)
        gdf_nms = gdf.loc[keep]
        gdf_nms.attrs = dict(gdf.attrs)   # .loc does not reliably carry attrs across
        written = []
        if save_nms:
            if save_bbox:
                nms_bbox = output_dir / _names["bbox_nms"]
                bboxes_to_shp(gdf_nms, nms_bbox)
                written.append(nms_bbox)
            outlines_to_shp(gdf_nms, nms_mask_shp)
            written.append(nms_mask_shp)
        return _finish(gdf_nms, written)
    else:
        return _finish(None, [out_mask_shp] if save_prenms else [])


def predict_production(in_raster, detection_model, output_dir, preset="goal1_production",
                       on_conflict="error", remask_fn=None, mask_tag=None, n_workers=None,
                       on_max_det="error", **overrides):
    """Run single-scale sliced inference in a **named, versioned** configuration (G1-01).

    ``get_sliced_prediction``'s own defaults are the historical generic ones and reproduce no
    benchmarked result; this resolves :data:`presets.GOAL1_PRODUCTION` (or any named preset)
    instead, rejects override keys the preset does not define — a typo would otherwise silently
    fall back to the preset value — and records the resolved configuration in the manifest beside
    the output.

    ``**overrides`` are *preset* parameters, so a typo raises. Arguments that a preset does not own
    — ``on_conflict``, ``remask_fn``, ``mask_tag``, ``n_workers``, ``on_max_det`` — are named
    explicitly and pass straight through. Of those, only ``on_conflict``, ``n_workers`` and
    ``on_max_det`` are genuinely outside the scientific configuration. ``remask_fn`` and ``mask_tag``
    are **not**: a re-masker replaces every mask in the catalog and is the whole content of Solution G
    (diagonal fraction 0.428 -> 0.200). They are recorded in the manifest and folded into the config
    hash by way of ``presets.describe_callable`` — see review item A1, which is what an earlier
    version of this docstring got wrong by asserting they "change nothing about what the result
    means". (``max_det`` is result-defining too, so it lives in the preset; ``on_max_det`` only
    decides whether a truncated run fails or merely records it.)

        gdf, gdf_nms = predict_production(raster, model, out_dir)
        gdf, gdf_nms = predict_production(raster, model, out_dir, slice_size=256)
    """
    config = presets.resolve(preset, **overrides)
    call = {k: v for k, v in config.items() if not k.startswith("_")}
    return get_sliced_prediction(in_raster, detection_model, output_dir=output_dir,
                                 on_conflict=on_conflict, remask_fn=remask_fn, mask_tag=mask_tag,
                                 n_workers=n_workers, on_max_det=on_max_det, **call)


_UNSET = object()   # lets an explicit `engulf_thr=None` mean "off" rather than "use the preset"

#: Largest raster (native pixels) the in-memory multiscale wrapper will accept. The tracked dev
#: crop is 4.2e6 px; the HiRISE scene is 1.9e9. Phase 4 measured 24-78 GB of commit growth on
#: full scenes and adopted a subprocess-per-arm rule for exactly that reason.
MAX_IN_MEMORY_PX = 200_000_000


def _check_in_memory_budget(in_raster, scales, max_px, allow):
    """Refuse a full-scene in-memory multiscale run (review item G3-09).

    ``get_multiscale_prediction`` holds every arm's catalog *and* the pooled copy in one process.
    That is fine on a crop and ruinous on a scene: phase 4 measured 24-78 GB of commit growth and
    concluded one subprocess per arm, but the wrapper never enforced it, so an ordinary call on a
    full scene would thrash the pagefile rather than fail. It now fails, with the alternative named.
    """
    if allow:
        return
    import rasterio
    try:
        with rasterio.open(in_raster) as src:
            n_px = int(src.width) * int(src.height)
    except rasterio.errors.RasterioError:
        # a missing or unreadable raster is not a memory problem; the per-arm call reports it
        # with a better message than this guard could
        return
    if n_px > max_px:
        raise MemoryError(
            f"{Path(in_raster).name} is {n_px / 1e6:.0f} Mpx, over the {max_px / 1e6:.0f} Mpx "
            f"in-memory limit for {len(scales)} arms. This wrapper keeps every arm plus a pooled "
            "copy resident; phase 4 measured 24-78 GB of commit growth on full scenes and adopted "
            "one subprocess per arm. Use scripts/run_multiscale_arms.py, which runs each arm in "
            "its own process and fuses the persisted per-arm layers, or pass allow_in_memory=True "
            "if you have the RAM and mean it.")


#: Per-arm inference parameters the multiscale wrapper owns itself — the ladder decides the slicing
#: and the wrapper decides what gets written, so these may not be inherited from the single-scale
#: preset even though it defines them.
_WRAPPER_OWNED_ARM_KEYS = ("slice_size", "sr_factor", "save_prenms", "save_bbox", "output_format",
                           "postprocess")


def _production_arm_kwargs(preset):
    """Per-arm inference configuration for a named multiscale preset.

    ``GOAL3_PRODUCTION`` documents that "the per-scale arms inherit everything in
    ``GOAL1_PRODUCTION`` except the slicing keys, which the ladder owns", and
    ``get_multiscale_prediction`` documents that a no-argument call runs the production
    configuration. Neither was true: only the *fusion* keys came from the preset and the per-arm
    call fell through to ``get_sliced_prediction``'s generic defaults — ``downscale_pred=False``,
    no area gate, NMS 0.5 — so a no-argument production call fused arms that were not the
    production arms. G3-01 fixed this one level down and the wrapper was left behind.

    It is also why the gap surfaced as a performance mystery rather than a numerical one:
    ``downscale_pred=False`` keeps every mask at ``imgsz`` (1024) through ``_mask_bounds`` and the
    tracer instead of ``slice_size``, which is 16x the array per detection at slice 256.
    """
    if preset is None:
        return {}
    single = presets.resolve("goal1_production")
    return {k: v for k, v in single.items()
            if not k.startswith("_") and k not in _WRAPPER_OWNED_ARM_KEYS}


def get_multiscale_prediction(in_raster,
                              detection_model=None,
                              scales=_UNSET,
                              rule=_UNSET,
                              cluster_thr=_UNSET,
                              metric=_UNSET,
                              bands=_UNSET,
                              engulf_thr=_UNSET,
                              cross_arm_only=_UNSET,
                              preset="goal3_production",
                              output_dir=None,
                              output_format: str = "gpkg",
                              on_conflict: str = "error",
                              allow_in_memory: bool = False,
                              max_in_memory_px: int = MAX_IN_MEMORY_PX,
                              **kwargs):
    """Goal 3: run ``get_sliced_prediction`` once per scale and fuse into one catalog.

    **Called with no fusion arguments this runs the production configuration**
    :data:`presets.GOAL3_PRODUCTION` — the phase-3 ablation winner ``256pref_engulf``: the ladder
    {512 native, 256 native, SR x2 @ 128-native footprint}, ``rule="bands"`` with
    ``bands={256: (0, inf)}`` so 256 masks win contested clusters while 512/SR act as recall nets,
    and ``engulf_thr=0.7`` so fine-arm fragments consolidate into the whole boulder. It used to
    default to ``rule="scale_aware"`` with no engulf pass — a rule the phase-3 ablation had already
    refuted — so an ordinary call produced a catalog nobody had validated (review item G3-01).
    Pass ``preset=None`` plus explicit arguments for a generic run; any argument you do pass
    overrides the preset.

    .. warning::
       **This wrapper is for crops, not scenes.** It keeps every arm's catalog and the pooled copy
       resident in one process. Phase 4 measured 24-78 GB of commit growth on full scenes and
       adopted one subprocess per arm; rasters above ``max_in_memory_px`` are now refused with a
       pointer to ``scripts/run_multiscale_arms.py`` rather than left to thrash the pagefile
       (review item G3-09). ``allow_in_memory=True`` overrides.

    ``scales`` uses the same keys as :func:`YOLOv8BeyondEarth.fusion.fuse_predictions`: an int
    is a native slice size, a ``(slice_size, sr_factor)`` tuple is a streamed-SR arm. Per-scale
    passes run with their own intra-scale NMS on and **all** per-scale file output off
    (``save_prenms=False, save_bbox=False, save_nms=False``) — the per-arm NMS layers used to be
    written regardless, contradicting the wrapper's promise of one fused deliverable. The fused
    mask layer is the one file written, as ``<stem>-predictions-fused-<rule>-...-mask.gpkg``, with
    the suppression audit table next to it as ``...-audit.csv`` and the resolved-config manifest as
    ``...-manifest.json``.

    ``cluster_thr=None`` uses the calibrated ``fusion.DEFAULT_CLUSTER_THR`` (a DIoU value —
    pass an explicit threshold if ``metric="iou"``). ``bands``/``rule``/``metric``/``engulf_thr``/
    ``cross_arm_only`` are passed through to ``fuse_predictions`` (``native_res`` for the bands
    rule is read off the raster); when ``engulf_thr`` is set the fused filename carries an
    ``-engulfNN`` tag.
    Remaining ``**kwargs`` (confidence_threshold, inference_size, overlap ratios,
    contour_method, batch_size, ...) go to every per-scale ``get_sliced_prediction`` call —
    per-scale parameter tuning is deliberately not offered (plan §Open risks: population-shift
    confound magnet).

    Returns ``(fused, audit, per_scale)`` — the fused GeoDataFrame (provenance columns
    included), the audit DataFrame, and the raw ``{scale: gdf_nms}`` dict for validation work.
    """
    from YOLOv8BeyondEarth.fusion import (fuse_predictions, pool_predictions, _empty_audit,
                                          DEFAULT_CLUSTER_THR)

    # Explicit arguments win; anything left unset comes from the preset, and from the generic
    # fallback if there is no preset.
    generic = {"scales": (512, 256, (256, 2)), "rule": "scale_aware", "cluster_thr": None,
               "metric": "diou", "bands": None, "engulf_thr": None, "cross_arm_only": True}
    explicit = {k: v for k, v in
                (("scales", scales), ("rule", rule), ("cluster_thr", cluster_thr),
                 ("metric", metric), ("bands", bands), ("engulf_thr", engulf_thr),
                 ("cross_arm_only", cross_arm_only)) if v is not _UNSET}
    base = dict(generic)
    if preset is not None:
        base.update({k: v for k, v in presets.resolve(preset).items()
                     if not k.startswith("_")})
    fuse_cfg = {**base, **explicit}
    scales = fuse_cfg["scales"]

    in_raster = Path(in_raster)
    output_dir = Path(output_dir)
    for forbidden in ("slice_size", "sr_factor", "save_prenms", "save_bbox", "postprocess"):
        if forbidden in kwargs:
            raise ValueError(f"{forbidden} is owned by the wrapper — set scales=... instead")
    if not scales:
        raise ValueError("scales is empty — nothing to fuse")

    ext = ".gpkg" if output_format == "gpkg" else ".shp"
    rule_str = (fuse_cfg["rule"] if fuse_cfg["engulf_thr"] is None else
                f"{fuse_cfg['rule']}-engulf{int(round(fuse_cfg['engulf_thr'] * 100)):02d}")
    scales_str = "-".join(str(s) if isinstance(s, int) else f"{s[0]}sr{s[1]}" for s in scales)
    out_mask = output_dir / (in_raster.stem + f"-predictions-fused-{rule_str}-{scales_str}-mask" + ext)
    # Resolve the arm configuration BEFORE hashing it. This used to record the caller's bare
    # `**kwargs`, check that against the manifest, and only then replace it with the inherited
    # `GOAL1_PRODUCTION` keys that were actually written — so the hash that was checked was never the
    # hash that was written, and a second byte-identical call raised `FileExistsError` claiming a
    # configuration change that had not happened (review item A2). It fired for every *named* preset,
    # i.e. the default, and the documented workaround — `on_conflict="overwrite"` — switches G1-07 off
    # on the fused deliverable, which is exactly what made A1 dangerous. Nothing may mutate
    # `resolved_config` after the check below.
    arm_kwargs = {**_production_arm_kwargs(preset), **kwargs}
    resolved_config = {"preset": preset if preset is None else str(preset),
                       "fusion": fuse_cfg, "per_scale_kwargs": dict(arm_kwargs),
                       "output_format": output_format,
                       "inference_contract_version": INFERENCE_CONTRACT_VERSION}
    presets.check_config_conflict(out_mask, resolved_config, on_conflict)

    _check_in_memory_budget(in_raster, scales, max_in_memory_px, allow_in_memory)

    per_scale = {}
    for key in scales:
        slice_size, sr_factor = (key if isinstance(key, (tuple, list)) else (key, 1))
        _, gdf_nms = get_sliced_prediction(
            in_raster, detection_model, slice_size=int(slice_size), sr_factor=int(sr_factor),
            output_dir=output_dir, output_format=output_format,
            save_prenms=False, save_bbox=False, save_nms=False, postprocess=True,
            on_conflict=on_conflict, **arm_kwargs)
        per_scale[key] = gdf_nms

    thr = DEFAULT_CLUSTER_THR if fuse_cfg["cluster_thr"] is None else fuse_cfg["cluster_thr"]
    native_res = raster_metadata.get_resolution(in_raster)[0]
    # Fusing one arm against nothing is the identity, so skip it — and skipping is not an
    # optimisation, it is the only correct behaviour. On a sparse input at a strict confidence an arm
    # can legitimately return **zero** detections; `pool_predictions` then never sees that arm, and
    # `rule="bands"` refuses because its band key matches no arm in the pool. That guard is right (a
    # band naming an absent arm is normally a key-spelling bug, and its failure mode is a silent
    # degradation to `scale_aware`) but it cannot tell a spelling bug from an empty tile, so a
    # legitimately sparse tile crashed the run. Measured 2026-08-21 on the boulder2024 test split at
    # conf 0.25: 2 of 280 tiles returned detections from the SR arm alone.
    #
    # Deliberately narrow: only the fewer-than-two-non-empty case is handled, because that is the
    # one with no decision in it. If the banded arm is empty while two or more others are not, there
    # is a real contest and no rule to settle it, so that still raises rather than being guessed.
    present = {k: g for k, g in per_scale.items() if g is not None}
    non_empty = {k: g for k, g in present.items() if len(g)}
    if len(non_empty) < 2:
        fused = pool_predictions(non_empty if non_empty else present)
        audit = _empty_audit()
        if non_empty:
            warnings.warn(
                f"only {list(non_empty)} produced detections, so there was nothing to fuse and the "
                f"fused catalog is that arm alone. Per-arm counts: "
                f"{ {str(k): len(v) for k, v in present.items()} }", RuntimeWarning)
    else:
        fused, audit = fuse_predictions(non_empty, cluster_thr=thr, metric=fuse_cfg["metric"],
                                        rule=fuse_cfg["rule"], bands=fuse_cfg["bands"],
                                        native_res=native_res, engulf_thr=fuse_cfg["engulf_thr"],
                                        cross_arm_only=fuse_cfg["cross_arm_only"])

    outlines_to_shp(fused, out_mask)
    audit.to_csv(out_mask.with_suffix("").as_posix() + "-audit.csv", index=False)
    presets.write_manifest(
        out_mask, config=resolved_config, inputs={"raster": in_raster},
        outputs={"files": [out_mask.name], "n_fused": int(len(fused)),
                 "n_suppressed": int(len(audit)),
                 "n_pooled": int(sum(len(g) for g in per_scale.values())),
                 "per_scale_counts": {str(k): int(len(v)) for k, v in per_scale.items()}},
        same_arm_suppressions=int((~audit.cross_scale).sum()) if len(audit) else 0)
    return fused, audit, per_scale