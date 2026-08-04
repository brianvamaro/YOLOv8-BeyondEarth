"""What did ultralytics' default `conf=0.25` cost Solution G? (review of the SAM box-prompt threshold)

`SAM.predict` defaults `conf=0.25` and applies it to the mask decoder's **predicted-IoU** head
(`models/sam/predict.py:523`, `idx = pred_scores > self.args.conf`), so a *prompted* box whose mask SAM
is merely unsure about is returned with no mask at all. `sam_masks_in_order` never passed `conf`, so
every G run inherited it. Two consequences, and this measures both on real slices:

1. **Correctness.** Those boulders fall back to their YOLO mask (`mask_src="yolo"`) — downgraded on
   SAM's *self-report*, not because anything inspected the mask. Solution G exists to replace YOLO
   masks, and the geometric guards (background-grab, `largest_blob_fill`, degenerate-polygon) are what
   is supposed to decide quality.
2. **Cost.** A dropped mask makes the returned count disagree with the prompt count, and the
   realignment path then re-prompts that chunk **one box at a time** — 1 + `sam_batch` encodes (97 at
   the default) where 1 would have done.

It also checks the claim that matters for re-ratification: that `conf=0.0` is **purely additive** —
every box that produced a mask at 0.25 must produce the *same* mask at 0.0, because the threshold is
applied after the masks are generated. If that holds, no existing SAM mask changes and the only
difference is the rescued boulders.

    conda run -n bouldernet python scripts/review/probe_sam_prompt_conf.py --n-slices 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_max_det import densest_slice_origins, power_state, read_slices   # noqa: E402


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
    ap.add_argument("--n-slices", type=int, default=20)
    ap.add_argument("--weights", default=str(REPO / "data" / "weights"
                                             / "yolov8-m-boulder-detection-tmp.pt"))
    ap.add_argument("--sam-weights", default=str(REPO / "data" / "weights" / "sam2.1_b.pt"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-box-dpx", type=float, default=None,
                    help="keep only prompts whose box equivalent diameter is below this (native px). "
                         "The threshold fires on AMBIGUOUS masks, and the densest slices are full of "
                         "well-resolved boulders — so the population at risk is the small/sub-floor "
                         "one the SR arm contributes, not the easy one.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from ultralytics import SAM, YOLO
    from YOLOv8BeyondEarth.orientation_validation import sam_masks_in_order

    batches, saved_max, _ = densest_slice_origins(
        args.catalog, args.raster, args.slice_size, args.overlap, args.n_slices, 1)
    origins = [tuple(b["origins"][0]) for b in batches]
    images = read_slices(args.raster, origins, args.slice_size, args.slice_size)

    model = YOLO(args.weights, task="segment")
    sam = SAM(args.sam_weights)

    record = {
        "raster": Path(args.raster).name, "slice_size": args.slice_size, "conf": args.conf,
        "sam_batch": args.sam_batch, "n_slices": len(images),
        "saved_catalog_max_per_slice": saved_max, "power": power_state(),
        "sam_weights": Path(args.sam_weights).name,
        "slices": [],
    }

    tot = {"boxes": 0, "masks_025": 0, "masks_000": 0, "realigned_025": 0, "realigned_000": 0,
           "encodes_025": 0, "encodes_000": 0, "rescued": 0, "differing_shared_masks": 0,
           "shared_masks": 0}

    for i, img in enumerate(images):
        res = model(img, imgsz=args.inference_size, conf=args.conf, max_det=1200,
                    verbose=False, device=args.device)[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        boxes = res.boxes.xyxy.cpu().numpy().astype(np.float64)
        if args.max_box_dpx is not None:
            wh = np.maximum(boxes[:, 2:4] - boxes[:, 0:2], 0.0)
            dpx = 2.0 * np.sqrt(wh.prod(axis=1) / np.pi)     # equivalent-circle diameter
            boxes = boxes[dpx < args.max_box_dpx]
            if len(boxes) == 0:
                continue
        n = len(boxes)

        m025, r025 = sam_masks_in_order(sam, img, boxes, sam_batch=args.sam_batch,
                                        device=args.device, conf=0.25)
        m000, r000 = sam_masks_in_order(sam, img, boxes, sam_batch=args.sam_batch,
                                        device=args.device, conf=0.0)

        got025 = [k for k, m in enumerate(m025) if m is not None]
        got000 = [k for k, m in enumerate(m000) if m is not None]
        # encodes: one per chunk normally; a realigned chunk costs 1 + its own box count
        n_chunks = int(np.ceil(n / args.sam_batch))
        enc025 = n_chunks + r025 * args.sam_batch
        enc000 = n_chunks + r000 * args.sam_batch

        shared = [k for k in got025 if m000[k] is not None]
        differing = sum(1 for k in shared if not np.array_equal(m025[k], m000[k]))

        row = {"slice": i, "origin": list(origins[i]), "n_boxes": n,
               "masks_at_025": len(got025), "masks_at_000": len(got000),
               "dropped_at_025": n - len(got025),
               "realigned_chunks_at_025": r025, "realigned_chunks_at_000": r000,
               "encodes_at_025": enc025, "encodes_at_000": enc000,
               "shared_masks": len(shared), "differing_shared_masks": differing}
        record["slices"].append(row)
        tot["boxes"] += n
        tot["masks_025"] += len(got025)
        tot["masks_000"] += len(got000)
        tot["realigned_025"] += r025
        tot["realigned_000"] += r000
        tot["encodes_025"] += enc025
        tot["encodes_000"] += enc000
        tot["rescued"] += len(got000) - len(got025)
        tot["shared_masks"] += len(shared)
        tot["differing_shared_masks"] += differing
        print(f"slice {i:>3} n={n:>4}  masks 0.25={len(got025):>4} 0.00={len(got000):>4}  "
              f"realigned {r025}/{r000}  encodes {enc025:>4}/{enc000:>4}  "
              f"shared={len(shared):>4} differing={differing}")

    b = max(tot["boxes"], 1)
    record["totals"] = tot
    record["summary"] = {
        "fraction_of_prompted_boxes_left_unmasked_at_025": round((tot["boxes"] - tot["masks_025"]) / b, 5),
        "fraction_rescued_by_conf_0": round(tot["rescued"] / b, 5),
        "encode_inflation_at_025": round(tot["encodes_025"] / max(tot["encodes_000"], 1), 3),
        "conf_0_is_purely_additive": tot["differing_shared_masks"] == 0,
        "note": ("purely additive means every box that produced a mask at 0.25 produced the SAME mask "
                 "at 0.0, so no existing SAM geometry changes and the delta is only the rescued boxes"),
    }
    print("\n" + json.dumps(record["summary"], indent=2))
    print(json.dumps(tot, indent=2))

    out = Path(args.out) if args.out else REPO / "data" / "review" / "sam_prompt_conf_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
