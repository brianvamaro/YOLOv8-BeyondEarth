"""Goal-3 phase 5a, entry gate 2 (clean version) — window-vs-slice SAM on IDENTICAL boxes.

The first attempt (``run_gate2_window_vs_slice.py``) compared two *independent pipeline runs* and
paired them spatially. That is confounded: the in-slice run's area gate and NMS both operate on
**SAM** masks while the window run's operate on **YOLO** masks, so the two catalogs retain
different members and a "matched pair" within 2x res can be two different detections. It reported
FAIL on HiRISE / MARGINAL on NAC, and the gap was larger on the denser crop — the signature you
would expect from exactly that confound. The gate text says *"re-mask one dev crop both ways on
identical boxes"*; this script does that.

**One box list, three crop sources, index-aligned — no matching step at all:**
  * ``window``   : the non-overlapping 256 grid ``remask.remask_windows`` uses.
  * ``slice``    : the SAHI-style **overlapping** 256 grid the production in-slice re-masker uses
                   (stride 256*(1-overlap)); each box is read in the slice where it is most
                   interior, which is what intra-scale NMS effectively selects.
  * ``window_sh``: the window grid shifted half a cell. **The control.** If ``window`` vs
                   ``slice`` and ``window`` vs ``window_sh`` disagree by similar amounts, the
                   effect is generic crop-translation sensitivity in SAM, not something specific
                   to the window grid — and it is then *already present* inside the validated
                   in-slice path, which is a finding for the validation plan rather than a defect
                   of 5a.

Same SAM model, same prompting helper, same cleanup, same blow-up guard on every arm; only the
crop around the boulder changes.

Usage:
    conda run -n bouldernet --no-capture-output python data/sam2_validation/run_gate2b_identical_boxes.py
Writes gate2b_pairs.csv + gate2b_summary.json next to this script.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import rasterio as rio

from YOLOv8BeyondEarth.predict import get_sliced_prediction
from YOLOv8BeyondEarth.polygon import binary_mask_to_polygon, shift_polygon, add_geometries
from YOLOv8BeyondEarth.orientation_validation import largest_blob_fill, sam_masks_in_order
from YOLOv8BeyondEarth.orientation import attach_orientations, axial_distance
from YOLOv8BeyondEarth.remask import _pixel_bboxes, _read_window_rgb, _group_windows

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent

WEIGHTS = ROOT / "data/weights/yolov8-m-boulder-detection-tmp.pt"
SAM_WEIGHTS = REPO / "sam2.1_b.pt"
CROPS = {
    "hirise_crop4k": ROOT / "data/solutionDE/hirise_crop4k.tif",
    "nac_crop2k": ROOT / "data/raster/crop.tif",
}
OUT_DIR = HERE / "gate2b_out"

SLICE = 256
OVERLAP = 0.2
CONF = 0.10
IMGSZ = 1024
SAM_BATCH = 96
MAX_BLOWUP = 1.5
ASPECT_MIN = 1.35


def _sam_polys_for_groups(src, pix, groups, sam, device=0):
    """Run SAM over ``groups`` of ``(window, member_indices)``; return polygons in raster px."""
    polys = np.full(len(pix), None, dtype=object)
    for (col_off, row_off, w, h), idx in groups:
        image = _read_window_rgb(src, col_off, row_off, w, h)
        boxes = pix[idx].copy()
        boxes[:, [0, 2]] -= col_off
        boxes[:, [1, 3]] -= row_off
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)
        masks, _ = sam_masks_in_order(sam, image, boxes, sam_batch=SAM_BATCH, device=device)
        for k, gi in enumerate(idx):
            mb = None if masks[k] is None else largest_blob_fill(masks[k].astype(np.uint8))
            if mb is None:
                continue
            box_area = max((boxes[k, 2] - boxes[k, 0]) * (boxes[k, 3] - boxes[k, 1]), 1e-9)
            if mb.sum() > MAX_BLOWUP * box_area:
                continue
            p = binary_mask_to_polygon(mb)
            if p is None or len(p) < 3:
                continue
            polys[gi] = shift_polygon(np.asarray(p, float), col_off, row_off)
    return polys


def _slice_groups(pix, width, height, slice_size=SLICE, overlap=OVERLAP):
    """SAHI-style overlapping grid; each box goes to the slice where it is most interior.

    That is what intra-scale NMS effectively selects (a box touching a slice edge is flagged
    ``is_within_slice=False`` and loses to the copy from the neighbouring, overlapping slice).
    """
    stride = int(round(slice_size * (1.0 - overlap)))
    xs = list(range(0, max(width - slice_size, 0) + 1, stride)) or [0]
    ys = list(range(0, max(height - slice_size, 0) + 1, stride)) or [0]
    if xs[-1] + slice_size < width:
        xs.append(max(width - slice_size, 0))
    if ys[-1] + slice_size < height:
        ys.append(max(height - slice_size, 0))

    cx = (pix[:, 0] + pix[:, 2]) / 2.0
    cy = (pix[:, 1] + pix[:, 3]) / 2.0
    assign = {}
    for i in range(len(pix)):
        best, best_margin = None, -np.inf
        for y0 in ys:
            if not (y0 <= cy[i] < y0 + slice_size):
                continue
            for x0 in xs:
                if not (x0 <= cx[i] < x0 + slice_size):
                    continue
                # margin = smallest distance from the BOX (not the centre) to the slice edge
                m = min(pix[i, 0] - x0, pix[i, 1] - y0,
                        x0 + slice_size - pix[i, 2], y0 + slice_size - pix[i, 3])
                if m > best_margin:
                    best_margin, best = m, (x0, y0)
        if best is not None:
            assign.setdefault(best, []).append(i)
    for (x0, y0), idx in assign.items():
        w = min(slice_size, width - x0)
        h = min(slice_size, height - y0)
        yield (x0, y0, w, h), np.array(idx, dtype=np.int64)


def run_crop(name, path, model, sam):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _, gdf_yolo = get_sliced_prediction(
        path, detection_model=model, confidence_threshold=CONF, inference_size=IMGSZ,
        slice_size=SLICE, overlap_height_ratio=OVERLAP, overlap_width_ratio=OVERLAP,
        downscale_pred=True, postprocess=True, output_dir=OUT_DIR, output_format="gpkg",
        save_prenms=False, save_bbox=False)

    with rio.open(path) as src:
        res = abs(src.transform.a)
        W, H = src.width, src.height
        pix = _pixel_bboxes(gdf_yolo.geometry.bounds.to_numpy(), src.transform)
        all_idx = np.arange(len(pix))

        print(f"  window grid ...", flush=True)
        g_win = list(_group_windows(pix, SLICE, 16, W, H))
        p_win = _sam_polys_for_groups(src, pix, g_win, sam)

        print(f"  slice grid (overlapping) ...", flush=True)
        g_sli = list(_slice_groups(pix, W, H))
        p_sli = _sam_polys_for_groups(src, pix, g_sli, sam)

        print(f"  shifted-window control ...", flush=True)
        shifted = pix + np.array([SLICE // 2, SLICE // 2, SLICE // 2, SLICE // 2], float)
        g_shx = [((c - SLICE // 2, r - SLICE // 2, w, h), i)
                 for (c, r, w, h), i in _group_windows(shifted, SLICE, 16, W + SLICE, H + SLICE)]
        g_shx = [((max(c, 0), max(r, 0), min(w, W - max(c, 0)), min(h, H - max(r, 0))), i)
                 for (c, r, w, h), i in g_shx if W - max(c, 0) > 1 and H - max(r, 0) > 1]
        p_shx = _sam_polys_for_groups(src, pix, g_shx, sam)

    ok = np.array([(p_win[i] is not None) and (p_sli[i] is not None) and (p_shx[i] is not None)
                   for i in all_idx])
    print(f"  usable on all three arms: {ok.sum()}/{len(ok)}", flush=True)

    def _gdf(polys):
        g = add_geometries(path, pd.DataFrame({"polygon": list(np.asarray(polys, object)[ok])}))
        return attach_orientations(g.drop(columns=["polygon", "bbox"]), res)

    win, sli, shx = _gdf(p_win), _gdf(p_sli), _gdf(p_shx)
    df = pd.DataFrame({
        "crop": name,
        "angle_win": win.angle180.values, "angle_sli": sli.angle180.values,
        "angle_shx": shx.angle180.values,
        "aspect_win": win.aspect_ra.values, "aspect_sli": sli.aspect_ra.values,
        "dpx_win": win.diameter_m.values / res,
        "area_win": win.geometry.area.values, "area_sli": sli.geometry.area.values,
        "area_shx": shx.geometry.area.values,
    })
    df["dangle_win_sli"] = axial_distance(df.angle_win.values, df.angle_sli.values)
    df["dangle_win_shx"] = axial_distance(df.angle_win.values, df.angle_shx.values)
    df["darea_win_sli"] = (df.area_win - df.area_sli).abs() / df.area_sli.clip(lower=1e-9)
    df["darea_win_shx"] = (df.area_win - df.area_shx).abs() / df.area_shx.clip(lower=1e-9)

    el = df[(df.aspect_win > ASPECT_MIN) & (df.aspect_sli > ASPECT_MIN)]
    q = lambda s, p: float(np.nanquantile(s, p)) if len(s) else None
    s = {
        "crop": name, "res": res, "n_detections": int(len(pix)),
        "n_usable_all_arms": int(ok.sum()), "n_elong": int(len(el)),
        "median_dangle_win_vs_slice": q(el.dangle_win_sli, 0.5),
        "p90_dangle_win_vs_slice": q(el.dangle_win_sli, 0.9),
        "median_dangle_win_vs_shifted_CONTROL": q(el.dangle_win_shx, 0.5),
        "p90_dangle_win_vs_shifted_CONTROL": q(el.dangle_win_shx, 0.9),
        "median_darea_win_vs_slice": q(df.darea_win_sli, 0.5),
        "median_darea_win_vs_shifted_CONTROL": q(df.darea_win_shx, 0.5),
        "frac_angle_within_10deg_win_vs_slice": float((el.dangle_win_sli <= 10).mean()) if len(el) else None,
        "frac_identical_angle": float((el.dangle_win_sli < 1e-6).mean()) if len(el) else None,
    }
    return df, s


def main():
    from sahi import AutoDetectionModel
    from ultralytics import SAM
    model = AutoDetectionModel.from_pretrained(
        model_type="yolov8", model_path=str(WEIGHTS), device="cuda:0",
        image_size=IMGSZ, confidence_threshold=CONF)
    sam = SAM(SAM_WEIGHTS.as_posix())

    frames, summaries = [], []
    for name, path in CROPS.items():
        if not path.exists():
            print(f"[skip] {name}"); continue
        print(f"\n=== {name} ===", flush=True)
        df, s = run_crop(name, path, model, sam)
        frames.append(df); summaries.append(s)
        print(json.dumps(s, indent=2), flush=True)

    if frames:
        pd.concat(frames, ignore_index=True).to_csv(HERE / "gate2b_pairs.csv", index=False)
        (HERE / "gate2b_summary.json").write_text(json.dumps(summaries, indent=2))
        print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
