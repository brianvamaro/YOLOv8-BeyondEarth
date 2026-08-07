"""Box-prompt probe (2026-07-23): is SAM's shadow-ward orientation lean on 2260 induced by the
axis-aligned YOLO BOX prompt, or by SAM responding to the shadow itself?

On a 2260 crop, for each YOLO detection re-mask with three prompts and compare the moment angle:
  box   = tight axis-aligned YOLO box (production sam256 baseline)
  exp   = box expanded 30% (lets SAM include more surrounding shadow)
  point = single positive point at the box centroid (NO box geometry at all)

If the lean is box-induced it should weaken/change under `point`; if SAM tracks the shadow it should
persist. Shadow axis of 2260 = 75.3 deg. __main__ guard (Windows spawn).
  conda run -n bouldernet python data/sam2_validation/run_box_prompt_probe.py
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(r"c:/Users/brian/Documents/PhD/BoulderNet/YOLOv8-BeyondEarth")
sys.path.insert(0, str(BASE / "src"))
OUT = BASE / "data" / "sam2_validation"
WEIGHTS = BASE / "data" / "weights" / "yolov8-m-boulder-detection-tmp.pt"
SAM_WEIGHTS = BASE / "sam2.1_b.pt"
TIF = BASE / "data/mars_hirise/raster/ESP_017355_2260_RED.tif"
SHADOW_AXIS = 75.3
TILE = 512; STRIP_W = 3072; TARGET = 700; MAX_TILES = 400; BLOWUP = 1.5


def clean(mb):
    from scipy.ndimage import label, binary_fill_holes
    lab, n = label(mb)
    if n == 0:
        return None
    if n > 1:
        big = 1 + int(np.argmax([(lab == j).sum() for j in range(1, n + 1)]))
        mb = (lab == big).astype(np.uint8)
    return binary_fill_holes(mb).astype(np.uint8)


def one_mask(sam, rgb, box=None, point=None):
    r = (sam(rgb, bboxes=box[None], verbose=False, device=0)[0] if box is not None
         else sam(rgb, points=[list(point)], labels=[1], verbose=False, device=0)[0])
    md = None if r.masks is None else r.masks.data.cpu().numpy()
    if md is None or len(md) == 0:
        return None
    return clean(md[0].astype(np.uint8))


def keep(mask, box):
    if mask is None:
        return None
    ba = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
    return None if mask.sum() > BLOWUP * ba else mask.astype(np.float32)


if __name__ == "__main__":
    import cv2, rasterio
    from rasterio.windows import Window
    from ultralytics import YOLO, SAM
    from YOLOv8BeyondEarth.orientation_validation import angles_from_moments

    model = YOLO(str(WEIGHTS)); sam = SAM(str(SAM_WEIGHTS))
    Lbox, Lpt, Lexp, dpx = [], [], [], []
    t0 = time.time()
    with rasterio.open(TIF) as src:
        H, W = src.height, src.width
        c0 = max(0, W // 2 - STRIP_W // 2)
        nt = 0
        for r0 in range(0, H - TILE, TILE):
            if nt >= MAX_TILES or len(Lbox) >= TARGET:
                break
            for cc in range(c0, c0 + STRIP_W - TILE, TILE):
                if nt >= MAX_TILES or len(Lbox) >= TARGET:
                    break
                tile = src.read(1, window=Window(cc, r0, TILE, TILE)); nt += 1
                if tile.max() == tile.min():
                    continue
                rgb = cv2.cvtColor(tile, cv2.COLOR_GRAY2RGB)
                res = model(rgb, imgsz=1024, conf=0.10, verbose=False, device=0)[0]
                if res.boxes is None or len(res.boxes) == 0:
                    continue
                boxes = res.boxes.xyxy.cpu().numpy()
                for b in boxes:
                    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                    w, h = b[2] - b[0], b[3] - b[1]
                    eb = np.array([b[0] - 0.15 * w, b[1] - 0.15 * h, b[2] + 0.15 * w, b[3] + 0.15 * h])
                    eb = np.clip(eb, 0, TILE - 1)
                    mb = keep(one_mask(sam, rgb, box=b), b)
                    mp = keep(one_mask(sam, rgb, point=(cx, cy)), b)
                    me = keep(one_mask(sam, rgb, box=eb), b)
                    if mb is None or mp is None or me is None:
                        continue
                    Lbox.append(mb); Lpt.append(mp); Lexp.append(me)
                    dpx.append(2 * np.sqrt(mb.sum() / np.pi))
            if len(Lbox) % 100 < 5:
                print(f"  tiles={nt} kept={len(Lbox)} {time.time()-t0:.0f}s", flush=True)

    abx = angles_from_moments(Lbox); apt = angles_from_moments(Lpt); aex = angles_from_moments(Lexp)
    d = pd.DataFrame(dict(dpx=dpx,
                          angle_box=abx.angle180, aspect_box=abx.aspect,
                          angle_pt=apt.angle180, aspect_pt=apt.aspect,
                          angle_exp=aex.angle180, aspect_exp=aex.aspect))
    d.to_csv(OUT / "box_prompt_probe.csv", index=False)
    print(f"\nkept {len(d)} boulders in {(time.time()-t0)/60:.1f} min. Shadow axis={SHADOW_AXIS}")
    from YOLOv8BeyondEarth.orientation import (half_plane_fraction as fl, asymmetry_ratio as asym,
                                               axial_distance, axis_alignment_fraction as align)
    for tag, ac, pc in [("box", "angle_box", "aspect_box"), ("point", "angle_pt", "aspect_pt"),
                        ("exp30", "angle_exp", "aspect_exp")]:
        a = d.loc[d[pc] > 1.35, ac].to_numpy()
        print(f"  {tag:6s} n={len(a):4d}  f_lower={fl(a):.3f}  asym135/45={asym(a):.2f}  "
              f"along-shadow={align(a, SHADOW_AXIS, 45):.3f}  "
              f"mean|d-shadow|={np.nanmean(axial_distance(a, SHADOW_AXIS)):.1f}")
