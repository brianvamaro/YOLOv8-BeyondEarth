# T4b (structure-tensor sub-probe) — illumination attribution of the scene-locked half-plane
# (docs/sam2_orientation_validation_plan.md). T4a separated pixel-vs-scene (verdict: mostly scene +
# a small SAM delta~0.04); this confirms the scene part is ILLUMINATION by measuring the raw-pixel
# gradient (structure-tensor) axis per boulder — the bright-cap->shadow terminator, i.e. the
# illumination edge, measured with NO mask in the loop — and checking whether SAM's half-plane
# imbalance IS that illumination axis (f_lower(st) ~ f_lower(SAM) ~ f_lower(human), and per-boulder
# SAM angle aligns with st).
#
# CPU-only (image gradients) and reuses the detections already cached by run_t35_val.py, so no GPU
# and no re-detection: for each detection, read a patch around its centroid from the source tile and
# run orientation.structure_tensor_orientation. Adds st_ang/st_coh columns.
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth")
sys.path.insert(0, str(BASE / "src"))

VAL = Path(r"c:\Users\brian\Documents\PhD\BoulderNet\data_prieur\boulder2024\validation")
OUT = BASE / "data" / "sam2_validation"
PAD_FRAC, PAD_MIN, SMOOTH = 1.5, 6.0, 1.0


if __name__ == "__main__":
    from PIL import Image
    from YOLOv8BeyondEarth.orientation import structure_tensor_orientation

    det = pd.read_csv(OUT / "t35_val_detections.csv")
    print(f"{len(det)} detections across {det.tile.nunique()} tiles", flush=True)
    t0 = time.time()
    st_ang = np.full(len(det), np.nan)
    st_coh = np.full(len(det), np.nan)
    # group by tile so each PNG is read once
    for tile, idx in det.groupby("tile").groups.items():
        ip = VAL / "images" / f"{tile}.png"
        if not ip.exists():
            continue
        img = np.array(Image.open(ip).convert("L")).astype(float)
        H, W = img.shape
        for i in idx:
            cx, cy = det.at[i, "cx"], det.at[i, "cy"]
            rad = max(det.at[i, "dpx_sam"] / 2.0, 2.0)
            pad = max(PAD_FRAC * rad, PAD_MIN)
            c0, c1 = int(round(cx - pad)), int(round(cx + pad))
            r0, r1 = int(round(cy - pad)), int(round(cy + pad))
            c0, r0 = max(c0, 0), max(r0, 0)
            c1, r1 = min(c1, W), min(r1, H)
            patch = img[r0:r1, c0:c1]
            if patch.shape[0] < 7 or patch.shape[1] < 7:
                continue
            h, w = patch.shape
            yy, xx = np.mgrid[0:h, 0:w]
            sig = max(rad, 2.0)
            wgt = np.exp(-(((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (2 * sig ** 2)))
            a, coh = structure_tensor_orientation(patch, weight=wgt, smooth=SMOOTH)
            st_ang[i] = a
            st_coh[i] = coh
    det["st_ang"] = st_ang
    det["st_coh"] = st_coh
    det.to_csv(OUT / "t4b_st_detections.csv", index=False)
    wall = time.time() - t0
    meta = dict(wall_s=wall, n=len(det), n_st=int(np.isfinite(st_ang).sum()),
                pad_frac=PAD_FRAC, pad_min=PAD_MIN, smooth=SMOOTH)
    (OUT / "t4b_run_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    print(f"wall {wall:.0f}s", flush=True)
