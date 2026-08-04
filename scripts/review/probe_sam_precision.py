"""Verify the fp16 claim on the SAM path (cost_model_optimization_plan, precision item).

The optimisation research reported, single-source and unverified: **fp16 leaves 80% of masks identical,
median |delta angle| 0.000 deg, p90 3.09 deg** — and concluded that is acceptable because it sits below
the already-ratified 4.65 deg crop-position jitter. bf16 was reported at 30% identity / 1.66 deg median
and is already on the don't-retry list, so this checks **fp32 vs fp16 only**, which is the one that
could still change a decision.

Method: real HiRISE slices, real YOLO boxes, the same boxes re-masked twice — once at the production
fp32 and once with ultralytics' `half=True` (`models/sam/predict.py:459` does
`model.half() if self.args.half else model.float()`). Angles come from the **shared production
readout** `orientation.mask_readout` (review item G2-02), so this measures the quantity Goal 2 actually
publishes rather than a proxy.

Note on what this can and cannot settle: mask identity and angle deltas are deterministic and so are
trustworthy here. The *speedup* is a wall clock and this machine's history of Modern-Standby / power-cap
contamination means a timing from one short run is not quotable — it is reported, flagged, and should
not be used without a controlled repetition.

    conda run -n bouldernet python scripts/review/probe_sam_precision.py --n-slices 12
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_max_det import densest_slice_origins, power_state, read_slices   # noqa: E402


def angle_diff180(a, b):
    """Smallest separation between two undirected axes, in degrees (period 180)."""
    d = np.abs(np.asarray(a, float) - np.asarray(b, float)) % 180.0
    return np.minimum(d, 180.0 - d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raster", default=str(REPO / "data" / "mars_hirise" / "raster"
                                           / "ESP_017355_2260_RED.tif"))
    ap.add_argument("--catalog", default=str(
        REPO / "data" / "solutionDE" / "fullscene" / "hirise"
        / "ESP_017355_2260_RED-predictions-ct-010-ss-256-is-1024-ov-020-downscaled-mask-nms.gpkg"))
    ap.add_argument("--slice-size", type=int, default=256)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--inference-size", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--sam-batch", type=int, default=96)
    ap.add_argument("--n-slices", type=int, default=12)
    ap.add_argument("--weights", default=str(REPO / "data" / "weights"
                                             / "yolov8-m-boulder-detection-tmp.pt"))
    ap.add_argument("--sam-weights", default=str(REPO / "data" / "weights" / "sam2.1_b.pt"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from ultralytics import SAM, YOLO
    from YOLOv8BeyondEarth.orientation import mask_readout
    from YOLOv8BeyondEarth.orientation_validation import SAM_PROMPT_CONF, largest_blob_fill

    batches, _, _ = densest_slice_origins(args.catalog, args.raster, args.slice_size,
                                          args.overlap, args.n_slices, 1)
    origins = [tuple(b["origins"][0]) for b in batches]
    images = read_slices(args.raster, origins, args.slice_size, args.slice_size)

    model = YOLO(args.weights, task="segment")

    # ONE SAM OBJECT PER PRECISION, and each must receive its `half` on its FIRST call.
    # `half` looks like a per-call predict argument but is consumed in `setup_model`
    # (`models/sam/predict.py:459`, `model.half() if self.args.half else model.float()`), which runs
    # only once per predictor. Verified on this environment: calling half=False then half=True on the
    # same object leaves the parameters at torch.float32 and `model.fp16` False while `args.half`
    # flips to True — so the request is silently ignored and a naive A/B measures fp32 twice. That is
    # exactly what the first version of this probe did (it reported 100% mask identity and a 1.008x
    # wall ratio, which is the signature of the bug, not of fp16).
    sams = {}
    for name, half in (("fp32", False), ("fp16", True)):
        s = SAM(args.sam_weights)
        s(images[0], bboxes=np.array([[8, 8, 40, 40]], dtype=np.float64), verbose=False,
          device=args.device, conf=SAM_PROMPT_CONF, half=half)          # forces setup at `half`
        got = next(s.predictor.model.parameters()).dtype
        want = torch.float16 if half else torch.float32
        if got != want:
            raise SystemExit(f"{name}: model is {got}, expected {want} — half did not take effect")
        sams[name] = s
        print(f"{name}: model dtype {got}, fp16 flag {getattr(s.predictor.model, 'fp16', None)}")

    def sam_masks(image, boxes, half):
        """One mask per box at the requested precision, from that precision's own predictor."""
        sam = sams["fp16" if half else "fp32"]
        out = []
        for i in range(0, len(boxes), args.sam_batch):
            chunk = boxes[i:i + args.sam_batch]
            r = sam(image, bboxes=chunk, verbose=False, device=args.device,
                    conf=SAM_PROMPT_CONF)[0]
            md = None if r.masks is None else r.masks.data.cpu().numpy()
            if md is None or len(md) != len(chunk):
                out.extend([None] * len(chunk))          # keep alignment; counted as unusable
            else:
                out.extend(list(md))
        return out

    identical = differing = unusable = 0
    deltas, walls = [], {"fp32": 0.0, "fp16": 0.0}
    n_boxes = 0

    for i, img in enumerate(images):
        res = model(img, imgsz=args.inference_size, conf=args.conf, max_det=1200,
                    verbose=False, device=args.device)[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        boxes = res.boxes.xyxy.cpu().numpy().astype(np.float64)
        n_boxes += len(boxes)

        torch.cuda.synchronize(); t0 = time.perf_counter()
        m32 = sam_masks(img, boxes, half=False)
        torch.cuda.synchronize(); walls["fp32"] += time.perf_counter() - t0

        torch.cuda.synchronize(); t0 = time.perf_counter()
        m16 = sam_masks(img, boxes, half=True)
        torch.cuda.synchronize(); walls["fp16"] += time.perf_counter() - t0

        for a, b in zip(m32, m16):
            if a is None or b is None:
                unusable += 1
                continue
            if np.array_equal(a, b):
                identical += 1
                continue
            differing += 1
            # only the differing masks can move an angle; read both through the production path
            ca, cb = largest_blob_fill(a.astype(np.uint8)), largest_blob_fill(b.astype(np.uint8))
            if ca is None or cb is None:
                continue
            ra, rb = mask_readout(ca), mask_readout(cb)
            if ra is None or rb is None:
                continue
            deltas.append(float(angle_diff180(ra[0], rb[0])))
        print(f"slice {i:>3} boxes={len(boxes):>4}  identical={identical:>5} "
              f"differing={differing:>5} unusable={unusable}")

    comparable = identical + differing
    d = np.asarray(deltas, float)
    # The headline must be over ALL comparable masks, not just the differing ones: an identical mask
    # contributes an angle delta of exactly 0, and dropping those inflates every percentile.
    all_deltas = np.concatenate([d, np.zeros(identical)]) if comparable else np.zeros(0)

    record = {
        "raster": Path(args.raster).name, "n_slices": len(images), "n_boxes": n_boxes,
        "sam_weights": Path(args.sam_weights).name, "power": power_state(),
        "identical": identical, "differing": differing, "unusable": unusable,
        "fraction_identical": round(identical / comparable, 4) if comparable else None,
        "claim_fraction_identical": 0.80,
        "angle_delta_over_all_comparable_masks_deg": {
            "median": round(float(np.median(all_deltas)), 4) if len(all_deltas) else None,
            "p90": round(float(np.percentile(all_deltas, 90)), 4) if len(all_deltas) else None,
            "p99": round(float(np.percentile(all_deltas, 99)), 4) if len(all_deltas) else None,
            "max": round(float(all_deltas.max()), 4) if len(all_deltas) else None,
        },
        "angle_delta_over_differing_masks_only_deg": {
            "n": int(len(d)),
            "median": round(float(np.median(d)), 4) if len(d) else None,
            "p90": round(float(np.percentile(d, 90)), 4) if len(d) else None,
        },
        "claim_median_deg": 0.0, "claim_p90_deg": 3.09,
        "comparison_bar_note": (
            "the plan compares fp16's p90 against the crop-position jitter's MEDIAN (4.65 deg Mars / "
            "2.69 deg Moon). That is not like-for-like, but it is conservative: matching the statistics "
            "would only make fp16 look better."),
        "wall_seconds": {k: round(v, 2) for k, v in walls.items()},
        "wall_ratio_fp32_over_fp16": round(walls["fp32"] / walls["fp16"], 3) if walls["fp16"] else None,
        "wall_caveat": ("single short run on a machine with a Modern-Standby / power-cap history; "
                        "NOT a quotable speedup. Both sides ran back-to-back in one process, so the "
                        "ratio is the only part with any standing."),
    }
    print("\n" + json.dumps({k: v for k, v in record.items() if k != "power"}, indent=2))

    out = Path(args.out) if args.out else REPO / "data" / "review" / "sam_precision_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
