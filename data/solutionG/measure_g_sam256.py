# Real production cost of G: sam256-only re-masking (drop the 1024px pass; encoder-reuse refuted).
# Times SAM separately from YOLO, because in production the detection GDF already exists -- the G
# post-pass only pays the SAM re-mask. Reports ms/boulder for (a) SAM-only (the production number)
# and (b) YOLO+SAM (if you re-detect too). Mirrors detect_sam2_mask_sets tiling/cleaning exactly.
import sys, time, json
from pathlib import Path
import numpy as np
import rasterio

sys.path.insert(0, r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\src")
CROP = r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\data\solutionDE\hirise_crop4k.tif"
W = r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\data\weights\yolov8-m-boulder-detection-tmp.pt"
OUT = Path(r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\data\solutionG")
SLICE, IMGSZ, CONF, SAM_BATCH, MAXBLOW = 256, 1024, 0.10, 96, 1.5
SCENE_PX, CROP_PX = 22490 * 85922, 4096 * 4096

if __name__ == "__main__":
    import torch
    from scipy.ndimage import label, binary_fill_holes
    from ultralytics import YOLO, SAM

    model = YOLO(W)
    sam = SAM("sam2.1_b.pt")

    def clean(mb):
        lab, n = label(mb)
        if n == 0:
            return None
        if n > 1:
            big = 1 + int(np.argmax([(lab == j).sum() for j in range(1, n + 1)]))
            mb = (lab == big).astype(np.uint8)
        return binary_fill_holes(mb).astype(np.uint8)

    with rasterio.open(CROP) as src:
        img = src.read(1)
    H, Wd = img.shape

    yolo_t = sam_t = 0.0
    n_boul = n_kept = 0
    n_tiles = 0
    # warmup
    sl = np.stack([img[:256, :256]] * 3, -1)
    _ = model(sl, imgsz=IMGSZ, conf=CONF, verbose=False, device=0)
    torch.cuda.synchronize()

    t_wall = time.time()
    for y0 in range(0, H - SLICE + 1, SLICE):
        for x0 in range(0, Wd - SLICE + 1, SLICE):
            rgb = np.stack([img[y0:y0+SLICE, x0:x0+SLICE]] * 3, -1)
            n_tiles += 1
            torch.cuda.synchronize(); t0 = time.time()
            r0 = model(rgb, imgsz=IMGSZ, conf=CONF, verbose=False, device=0)[0]
            torch.cuda.synchronize(); yolo_t += time.time() - t0
            if r0.masks is None or len(r0.boxes) == 0:
                continue
            boxes = r0.boxes.xyxy.cpu().numpy()
            n_boul += len(boxes)
            # --- SAM sam256-only re-mask (timed) ---
            torch.cuda.synchronize(); t0 = time.time()
            outs = []
            for i in range(0, len(boxes), SAM_BATCH):
                rr = sam(rgb, bboxes=boxes[i:i+SAM_BATCH], verbose=False, device=0)[0]
                outs.append(rr.masks.data.cpu().numpy())
            s = np.concatenate(outs, 0)
            for k in range(len(boxes)):
                mb = clean(s[k].astype(np.uint8))
                ba = (boxes[k, 2]-boxes[k, 0]) * (boxes[k, 3]-boxes[k, 1])
                if mb is None or mb.sum() > MAXBLOW * ba:
                    continue
                n_kept += 1
            torch.cuda.synchronize(); sam_t += time.time() - t0
            if n_tiles % 64 == 0:
                torch.cuda.empty_cache()
                print(f"tile {n_tiles}: cum {n_boul} boul, sam {sam_t:.0f}s yolo {yolo_t:.0f}s", flush=True)
    wall = time.time() - t_wall

    res = dict(n_tiles=n_tiles, n_boulders=n_boul, n_kept=n_kept, wall_s=wall,
               yolo_s=yolo_t, sam_s=sam_t,
               sam_ms_per_boulder=sam_t / max(n_boul, 1) * 1000,
               total_ms_per_boulder=(yolo_t + sam_t) / max(n_boul, 1) * 1000,
               scene_h_sam_only=sam_t * (SCENE_PX / CROP_PX) / 3600,
               scene_h_total=(yolo_t + sam_t) * (SCENE_PX / CROP_PX) / 3600)
    (OUT / "g_sam256_cost.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
