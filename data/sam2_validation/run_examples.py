# Example boulder detections for the SAM2 angle-validation notebook: pick a few real boulders per
# body and cache raw patch + YOLO mask + SAM mask + human polygon + fitted long-axes, so the notebook
# can draw qualitative "raw | YOLO | SAM | human" panels (illustrating YOLO's looser/snapped mask vs
# SAM's tighter mask vs human truth). GPU (YOLO + SAM on whole tiles); the notebook draws from the
# cached pickle. __main__ guard (Windows spawn). Run from inside YOLOv8-BeyondEarth/.
import pickle, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth")
sys.path.insert(0, str(BASE / "src"))

VAL = Path(r"c:\Users\brian\Documents\PhD\BoulderNet\data_prieur\boulder2024\validation")
OUT = BASE / "data" / "sam2_validation"
WEIGHTS = BASE / "data" / "weights" / "yolov8-m-boulder-detection-tmp.pt"
SAM_WEIGHTS = BASE / "sam2.1_b.pt"

# (body, tile stem, n examples to keep) — dense tiles with human labels, one per body
TILES = [("mars", "ESP_017355_2260_RED_2327_image", 3),
         ("moon", "M104827900_0405_image", 3),
         ("earth", "courtright-reservoir-sept2022-morning_0075_image", 2)]


def _clean(mb):
    from scipy.ndimage import label, binary_fill_holes
    lab, n = label(mb)
    if n == 0:
        return None
    if n > 1:
        mb = (lab == 1 + int(np.argmax([(lab == j).sum() for j in range(1, n + 1)]))).astype(np.uint8)
    return binary_fill_holes(mb).astype(np.uint8)


if __name__ == "__main__":
    import cv2
    from PIL import Image
    from ultralytics import YOLO, SAM
    from shapely.geometry import Polygon
    from YOLOv8BeyondEarth.benchmark import mask_angle180
    from YOLOv8BeyondEarth.orientation import ellipse_angle180

    import torch
    model = YOLO(str(WEIGHTS)); sam = SAM(str(SAM_WEIGHTS))
    examples = []
    t0 = time.time()
    for body, stem, n_keep in TILES:
        arr = np.array(Image.open(VAL / "images" / f"{stem}.png").convert("L"))
        H, W = arr.shape
        rgb = np.stack([arr] * 3, -1)
        r = model(rgb, imgsz=1024, conf=0.10, verbose=False, device=0)[0]
        if r.masks is None:
            continue
        boxes = r.boxes.xyxy.cpu().numpy()
        ymd = r.masks.data.cpu().numpy()
        sm_parts = []                                  # chunk the SAM prompt (dense tiles OOM otherwise)
        for i in range(0, len(boxes), 64):
            sm_parts.append(sam(rgb, bboxes=boxes[i:i + 64], verbose=False,
                                device=0)[0].masks.data.cpu().numpy())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        sm = np.concatenate(sm_parts, axis=0)
        # human polys (tile px)
        humans = []
        lp = VAL / "labels" / f"{stem}.txt"
        if lp.exists():
            for line in lp.read_text().strip().splitlines():
                v = line.split()
                if len(v) < 7:
                    continue
                xy = np.array(v[1:], float).reshape(-1, 2) * [W, H]
                humans.append(xy)
        hcent = np.array([xy.mean(0) for xy in humans]) if humans else np.empty((0, 2))
        # score candidates: well-resolved elongated, matched to a human, YOLO disagrees w/ human more
        cands = []
        for k in range(len(boxes)):
            ym = _clean((cv2.resize(ymd[k], (W, H)) >= 0.5).astype(np.uint8))
            smk = _clean((sm[k] >= 0.5).astype(np.uint8))
            if ym is None or smk is None:
                continue
            bx = boxes[k]
            cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
            if min(cx, cy) < 20 or cx > W - 20 or cy > H - 20:
                continue
            ay, aspy = mask_angle180(ym, res=0.5)
            as_, asps = mask_angle180(smk, res=0.5)
            dpx = 2 * np.sqrt(smk.sum() / np.pi)
            if not (np.isfinite(as_) and asps > 1.5 and 10 <= dpx <= 32):
                continue
            hpoly, ah = None, np.nan
            if len(hcent):
                j = int(np.argmin(np.hypot(hcent[:, 0] - cx, hcent[:, 1] - cy)))
                if np.hypot(hcent[j, 0] - cx, hcent[j, 1] - cy) <= max(dpx, 8):
                    hpoly = humans[j]
                    ah, _ = ellipse_angle180(Polygon(np.c_[hpoly[:, 0], -hpoly[:, 1]]), 1.0)
            score = abs(((ay - (ah if np.isfinite(ah) else as_) + 90) % 180) - 90)  # YOLO-vs-ref gap
            cands.append((score, k, cx, cy, ym, smk, ay, as_, ah, hpoly, dpx))
        cands.sort(reverse=True)                      # prefer the most illustrative (YOLO disagrees)
        for score, k, cx, cy, ym, smk, ay, as_, ah, hpoly, dpx in cands[:n_keep]:
            half = int(max(dpx * 1.6, 18))
            c0, r0 = int(cx - half), int(cy - half)
            c1, r1 = c0 + 2 * half, r0 + 2 * half
            if c0 < 0 or r0 < 0 or c1 > W or r1 > H:
                continue
            sl = slice(r0, r1), slice(c0, c1)
            hp = None if hpoly is None else (hpoly - [c0, r0])
            examples.append(dict(body=body, tile=stem, patch=arr[sl].copy(),
                                 ymask=ym[sl].copy(), smask=smk[sl].copy(), hpoly=hp,
                                 ay=float(ay), as_=float(as_), ah=float(ah), dpx=float(dpx)))
        print(f"[{body}] {stem}: {len([e for e in examples if e['tile']==stem])} examples", flush=True)

    with open(OUT / "examples.pkl", "wb") as f:
        pickle.dump(examples, f)
    print(f"saved {len(examples)} examples in {time.time()-t0:.0f}s", flush=True)
