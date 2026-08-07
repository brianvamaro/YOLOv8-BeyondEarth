# De-risk the G throughput work: find the fastest CORRECT way to re-mask many boxes on one tile.
# Compare, on real HiRISE tiles (256 px, native), producing sam256 masks:
#   A. current: chunked sam(rgb, bboxes=chunk)  -> re-encodes per chunk
#   B. single:  sam(rgb, bboxes=all)            -> one encode per tile (if it fits in VRAM)
#   C. reuse:   predictor.set_image(rgb) once + predictor(bboxes=chunk) per chunk -> one encode, chunked decode
# Verify B and C produce the same masks as A (IoU), and time each. Whichever is fastest+correct
# becomes the production path.
import sys, time
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window

sys.path.insert(0, r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\src")
CROP = r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\data\solutionDE\hirise_crop4k.tif"
W = r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\data\weights\yolov8-m-boulder-detection-tmp.pt"


def masks_from_result(r):
    return (r.masks.data.cpu().numpy() >= 0.5).astype(np.uint8) if r.masks is not None else None


def iou(a, b):
    a = a.astype(bool); b = b.astype(bool)
    u = (a | b).sum()
    return (a & b).sum() / u if u else 1.0


if __name__ == "__main__":
    import torch
    from ultralytics import YOLO, SAM

    model = YOLO(W)
    sam = SAM("sam2.1_b.pt")

    # gather a few dense 256 tiles with many boxes
    tiles = []
    with rasterio.open(CROP) as src:
        for (r0, c0) in [(1024, 1024), (2048, 2048), (1536, 2560), (2560, 1536)]:
            sl = src.read(1, window=Window(c0, r0, 256, 256))
            rgb = np.stack([sl] * 3, axis=-1)
            res = model(rgb, imgsz=1024, conf=0.10, verbose=False, device=0)[0]
            if res.masks is not None and len(res.boxes) >= 40:
                tiles.append((rgb, res.boxes.xyxy.cpu().numpy()))
    print(f"probe tiles: {[(t[1].shape[0]) for t in tiles]} boxes each")

    def warm():
        rgb, bx = tiles[0]
        _ = sam(rgb, bboxes=bx[:8], verbose=False, device=0)
    warm(); torch.cuda.synchronize()

    # A. chunked (current), sam_batch=96
    def run_A():
        out = []
        for rgb, bx in tiles:
            ms = []
            for i in range(0, len(bx), 96):
                r = sam(rgb, bboxes=bx[i:i+96], verbose=False, device=0)[0]
                ms.append(masks_from_result(r))
            out.append(np.concatenate(ms, 0))
        return out

    # C. set_image reuse + chunked decode via predictor(bboxes=)
    def run_C():
        pred = sam.predictor
        out = []
        for rgb, bx in tiles:
            pred.set_image(rgb)               # encode once
            ms = []
            for i in range(0, len(bx), 96):
                r = pred(bboxes=bx[i:i+96])[0]  # reuse self.features
                ms.append(masks_from_result(r))
            pred.reset_image()
            out.append(np.concatenate(ms, 0))
        return out

    results = {}
    for name, fn in [("A_chunked", run_A), ("C_reuse", run_C)]:
        try:
            torch.cuda.synchronize(); t0 = time.time()
            out = fn()
            torch.cuda.synchronize()
            dt = time.time() - t0
            nb = sum(len(o) for o in out)
            results[name] = (dt, out)
            print(f"{name:10s}: {dt*1000:7.1f} ms total, {dt/nb*1000:5.1f} ms/boulder ({nb} boulders)")
        except Exception as e:
            print(f"{name:10s}: FAILED -> {type(e).__name__}: {e}")

    # correctness: mask IoU of B and C vs A, per tile per boulder (median)
    if "A_chunked" in results:
        A = results["A_chunked"][1]
        for name in ("C_reuse",):
            if name in results:
                O = results[name][1]
                ious = [iou(a, o) for ta, to in zip(A, O) for a, o in zip(ta, to)]
                print(f"{name} vs A: median mask IoU {np.median(ious):.3f}  min {np.min(ious):.3f}")
