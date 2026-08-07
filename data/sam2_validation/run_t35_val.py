# T3 + T5 — cross-body cardinal signature at scale + ground-truth angle/area vs human labels
# (docs/sam2_orientation_validation_plan.md). Runs the production detect -> SAM2 re-mask on every
# boulder2024 validation tile (739; Mars ESP_*, Moon M*/NAC_DTM, Earth courtright/stonegarden),
# reading yolo + sam256 with the moment estimator, and matches detections to the human polygon
# labels. One per-detection table feeds T3 (SAM cardinal grid_frac + f_lower per body, by band) and
# T5 (|dAngle| SAM/YOLO vs human, and area ratios SAM/human, YOLO/human).
#
# Detection config = the CANONICAL val consumption (whole tile at imgsz=1024, as model.val /
# seg_map_eval use it): pad each 500px tile to 512 and run detect_sam2_mask_sets with slice_size=512
# so there is ONE slice = the whole tile -> NO internal slice edges (which would inject spurious
# cardinal fragments and confound the cardinal-artifact study). Border-touching detections dropped.
#
# __main__ guard required (Windows spawn). Run from inside YOLOv8-BeyondEarth/.
import json, re, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

BASE = Path(r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth")
sys.path.insert(0, str(BASE / "src"))

VAL = Path(r"c:\Users\brian\Documents\PhD\BoulderNet\data_prieur\boulder2024\validation")
OUT = BASE / "data" / "sam2_validation"
WEIGHTS = BASE / "data" / "weights" / "yolov8-m-boulder-detection-tmp.pt"
SAM_WEIGHTS = BASE / "sam2.1_b.pt"
PAD = 512               # pad target (multiple of slice; one whole-tile slice)
BORDER = 8              # drop detections whose centroid is within BORDER px of the real tile edge
MATCH_MIN_R = 6.0       # min match radius (px)


def body_of(stem):
    if stem.startswith("ESP"):
        return "mars"
    if re.match(r"^M\d", stem) or stem.startswith("NAC_DTM"):
        return "moon"
    return "earth"


def load_labels(txt_path, W, H):
    """Human polygon labels -> list of dicts (angle180 North-up, aspect, area_px, cx, cy in tile px)."""
    from shapely.geometry import Polygon
    from YOLOv8BeyondEarth.orientation import ellipse_angle180
    out = []
    if not txt_path.exists():
        return out
    for line in txt_path.read_text().strip().splitlines():
        v = line.split()
        if len(v) < 7:
            continue
        xy = np.array(v[1:], dtype=float).reshape(-1, 2)
        px = xy[:, 0] * W
        py = xy[:, 1] * H
        if len(px) < 3:
            continue
        poly = Polygon(np.c_[px, py])                      # tile px (col,row) for area/centroid
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area <= 0:
            continue
        ang, asp = ellipse_angle180(Polygon(np.c_[px, -py]), 1.0)   # y-flip -> North-up, like masks
        c = poly.centroid
        out.append(dict(cx=float(c.x), cy=float(c.y), angle=float(ang), aspect=float(asp),
                        area=float(poly.area), dpx=float(2 * np.sqrt(poly.area / np.pi))))
    return out


if __name__ == "__main__":
    import cv2
    from PIL import Image
    from ultralytics import YOLO, SAM
    from YOLOv8BeyondEarth.orientation_validation import detect_sam2_mask_sets, angles_from_moments

    OUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(WEIGHTS))
    sam = SAM(str(SAM_WEIGHTS))

    imgs = sorted((VAL / "images").glob("*.png"))
    print(f"{len(imgs)} val tiles", flush=True)
    t0 = time.time()
    rows = []
    realigned = 0
    for ti, ip in enumerate(imgs):
        stem = ip.stem                                     # e.g. ESP_017355_2260_RED_2327_image
        body = body_of(stem)
        scene = re.sub(r"_\d+_image$", "", stem)
        arr = np.array(Image.open(ip).convert("L"))        # (H,W) uint8
        H, W = arr.shape
        pad = np.zeros((PAD, PAD), np.uint8)
        pad[:H, :W] = arr
        df, masks = detect_sam2_mask_sets(model, sam, pad, slice_size=PAD, imgsz=1024, conf=0.10,
                                          device=0, configs=("yolo", "sam256"))
        realigned += int(df.attrs.get("realigned_chunks", 0))
        humans = load_labels(VAL / "labels" / f"{stem}.txt", W, H)
        if len(df) == 0:
            continue
        ay = angles_from_moments(masks["yolo"]); asm = angles_from_moments(masks["sam256"])
        df["angle_yolo"] = ay["angle180"].values; df["aspect_yolo"] = ay["aspect"].values
        df["angle_sam"] = asm["angle180"].values; df["aspect_sam"] = asm["aspect"].values
        df["dpx_sam"] = asm["diameter_px"].values
        # keep interior detections only (drop border-split / pad-region)
        keep = (df.cx >= BORDER) & (df.cx < W - BORDER) & (df.cy >= BORDER) & (df.cy < H - BORDER)
        df = df[keep].reset_index(drop=True)
        if len(df) == 0:
            continue
        # match to humans by centroid (greedy nearest within max(0.75*human_dpx, MATCH_MIN_R))
        hmap = {}
        if humans:
            hc = np.array([[h["cx"], h["cy"]] for h in humans])
            hr = np.maximum(0.75 * np.array([h["dpx"] for h in humans]), MATCH_MIN_R)
            dc = df[["cx", "cy"]].to_numpy()
            pairs = cKDTree(hc).query_ball_tree(cKDTree(dc), r=float(hr.max()))
            cand = sorted((float(np.hypot(*(hc[hi] - dc[di]))), hi, di)
                          for hi, dis in enumerate(pairs) for di in dis
                          if np.hypot(*(hc[hi] - dc[di])) <= hr[hi])
            ut, ud = set(), set()
            for d, hi, di in cand:
                if hi in ut or di in ud:
                    continue
                ut.add(hi); ud.add(di); hmap[di] = hi
        for di in range(len(df)):
            r = df.iloc[di]
            h = humans[hmap[di]] if di in hmap else None
            rows.append(dict(
                body=body, scene=scene, tile=stem, cx=float(r.cx), cy=float(r.cy),
                angle_sam=float(r.angle_sam), aspect_sam=float(r.aspect_sam), dpx_sam=float(r.dpx_sam),
                area_sam=float(r.area_sam256), angle_yolo=float(r.angle_yolo),
                aspect_yolo=float(r.aspect_yolo), area_yolo=float(r.area_yolo),
                h_angle=(np.nan if h is None else h["angle"]),
                h_aspect=(np.nan if h is None else h["aspect"]),
                h_area=(np.nan if h is None else h["area"]),
                h_dpx=(np.nan if h is None else h["dpx"])))
        if (ti + 1) % 100 == 0:
            print(f"  {ti+1}/{len(imgs)} tiles  rows={len(rows)}  realigned={realigned}  "
                  f"{time.time()-t0:.0f}s", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "t35_val_detections.csv", index=False)
    wall = time.time() - t0
    counts = out.groupby("body").size().to_dict()
    matched = int(out["h_angle"].notna().sum())
    meta = dict(wall_s=wall, n_tiles=len(imgs), n_detections=len(out), n_matched=matched,
                by_body=counts, realigned_chunks=realigned, pad=PAD, imgsz=1024, slice_size=PAD,
                border=BORDER)
    (OUT / "t35_run_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    print(f"wall {wall:.0f}s", flush=True)
