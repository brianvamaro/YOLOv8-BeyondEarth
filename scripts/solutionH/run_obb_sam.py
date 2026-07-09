"""H x G: SAM2 re-masking on H-obb's detections. For each H-obb box, compare the angle from the
SAM MASK (should be clean ≈ G's 0.200) vs from the OBB HEAD (the cardinal-prior artifact), paired on
identical detections. Tests whether SAM rescues the per-boulder orientation while H-obb supplies the
box. Caches paired angles to data/solutionH/obb_sam_paired.csv."""
import sys, numpy as np, pandas as pd, rasterio
sys.path.insert(0, "YOLOv8-BeyondEarth/src")
from ultralytics import YOLO, SAM
from scipy.ndimage import label, binary_fill_holes
from YOLOv8BeyondEarth.orientation import obb_angle180, diagonal_fraction, grid_fraction, grid_fraction_uniform
from YOLOv8BeyondEarth.orientation_validation import angles_from_moments

OUT = "YOLOv8-BeyondEarth/data/solutionH"
UNI = grid_fraction_uniform()
img = rasterio.open("YOLOv8-BeyondEarth/data/solutionDE/hirise_crop4k.tif").read(1)
obb = YOLO("YOLOv8-BeyondEarth/data/weights/yolo26m-h-obb.pt", task="obb")
sam = SAM("YOLOv8-BeyondEarth/data/weights/sam2.1_b.pt")
SLICE, IMGSZ, CONF, DEV, BLOWUP, CHUNK = 256, 1024, 0.10, 0, 1.5, 96


def clean(mb):
    lab, n = label(mb)
    if n == 0:
        return None
    if n > 1:
        mb = (lab == 1 + int(np.argmax([(lab == j).sum() for j in range(1, n + 1)]))).astype(np.uint8)
    return binary_fill_holes(mb).astype(np.uint8)


rows = []
for y0 in range(0, img.shape[0] - SLICE + 1, SLICE):
    for x0 in range(0, img.shape[1] - SLICE + 1, SLICE):
        rgb = np.stack([img[y0:y0 + SLICE, x0:x0 + SLICE]] * 3, -1)
        r = obb.predict(rgb, imgsz=IMGSZ, conf=CONF, verbose=False, device=DEV)[0]
        if r.obb is None or len(r.obb) == 0:
            continue
        xyxy = r.obb.xyxy.cpu().numpy()
        corners = r.obb.xyxyxyxy.cpu().numpy()
        sm = []
        for i in range(0, len(xyxy), CHUNK):
            sm.append(sam(rgb, bboxes=xyxy[i:i + CHUNK], verbose=False, device=DEV)[0].masks.data.cpu().numpy())
        sm = np.concatenate(sm, 0)
        for k in range(len(xyxy)):
            mb = clean((sm[k] >= 0.5).astype(np.uint8))
            box_area = max((xyxy[k, 2] - xyxy[k, 0]) * (xyxy[k, 3] - xyxy[k, 1]), 1)
            if mb is None or mb.sum() > BLOWUP * box_area:      # background-grab guard (as in G)
                continue
            sa, sasp, sdpx = angles_from_moments([mb]).iloc[0]
            oa, oasp = obb_angle180(corners[k])
            if np.isfinite(sa) and np.isfinite(oa):
                rows.append((sa, sasp, oa, oasp, sdpx))

df = pd.DataFrame(rows, columns=["sam_angle", "sam_aspect", "obb_angle", "obb_aspect", "dpx"])
df.to_csv(f"{OUT}/obb_sam_paired.csv", index=False)
print(f"paired detections: {len(df)}  (uniform={UNI:.3f}; G/SAM ref diag 0.200)")
for tag, ang, asp in [("SAM-mask angle (H×G)", df.sam_angle, df.sam_aspect),
                      ("OBB-head angle", df.obb_angle, df.obb_aspect)]:
    el = df[asp > 1.35]
    a = el.sam_angle if "SAM" in tag else el.obb_angle
    print(f"  {tag:22} n_elong={len(el):>5}  diag={diagonal_fraction(a):.3f}  grid={grid_fraction(a):.3f}")
