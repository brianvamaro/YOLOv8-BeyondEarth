"""T1 (SAM) — control locations: does SAM show ANY preferred boulder orientation on
geologically-null ground?

The original Test 1 ran YOLO on 4 far-S Mars controls (+ N-off-coast + Moon) chosen so that NO
depositional fabric is expected; YOLO's rose there peaked ~44-55° (near the 45° pixel diagonal) at
dpx>=8 — i.e. an artifact signature on fabric-free ground. This is the SAM version of that test, and
it is the cleanest available "is there an artifact at all?" probe: on null ground ANY preferred axis
must be an artifact (or an unexpected fabric), and WHERE the peak sits discriminates the cause:
  peak at the pixel grid (0/90) or diagonal (45/135)  -> decoder / grid artifact
  peak at that scene's SUN axis                       -> illumination
  no peak (~uniform)                                  -> no SAM artifact
Sites span 53-64°S plus 69°N, so their sun axes differ from the 46°N treatment scenes -> the sun and
the pixel grid are DE-CONFOUNDED across sites (unlike a single north-up scene).

Reuses the validated whole-tile methodology (run_sam_chirality.process_scene: 512-px tiles, YOLO
detect -> SAM2 sam256 re-mask, moment angles, border drop). __main__ guard (Windows spawn).
  conda run -n bouldernet python data/sam2_validation/run_t1_controls_sam.py
"""
import json, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(r"c:/Users/brian/Documents/PhD/BoulderNet/YOLOv8-BeyondEarth")
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE / "data" / "sam2_validation"))
OUT = BASE / "data" / "sam2_validation"

# 4 far-S Mars controls + the N-off-coast control (all geologically-null for the tsunami fabric)
SITES = ["ESP_030222_1220", "ESP_076499_1160", "ESP_021862_1175", "ESP_013249_1270",
         "TRA_000828_2495"]


def tif(o):
    for sub in (f"test1_controls/{o}", "test1_controls/n_offcoast"):
        p = BASE / f"data/{sub}/{o}_RED.tif"
        if p.exists():
            return p
    raise FileNotFoundError(o)


def sun_of(o):
    t = tif(o).with_suffix(".LBL").read_text(errors="replace")
    def g(k):
        m = re.search(rf"{k}\s*=\s*([-\d.]+)", t)
        return float(m.group(1)) if m else np.nan
    ss, na, inc = g("SUB_SOLAR_AZIMUTH"), g("NORTH_AZIMUTH"), g("INCIDENCE_ANGLE")
    return (ss - na) % 180.0, inc, na


if __name__ == "__main__":
    from ultralytics import YOLO, SAM
    from YOLOv8BeyondEarth.orientation_validation import detect_sam2_mask_sets, angles_from_moments
    import run_sam_chirality as R
    R.TARGET_ELONG = 900          # aim for ~900 elongated (pre-size-floor) per site
    R.MAX_TILES = 700
    R.STRIP_W = 4096

    model = YOLO(str(R.WEIGHTS)); sam = SAM(str(R.SAM_WEIGHTS))
    all_rows, meta = [], {}
    for o in SITES:
        t0 = time.time()
        try:
            p = tif(o)
        except FileNotFoundError:
            print(f"SKIP {o}: no tif", flush=True); continue
        sun, inc, north = sun_of(o)
        df, info = R.process_scene(model, sam, str(p), angles_from_moments, detect_sam2_mask_sets)
        df["scene"] = o; df["sun"] = sun; df["incidence"] = inc; df["north_az"] = north
        all_rows.append(df)
        ne8 = int(((df.aspect_sam > 1.35) & (df.dpx_sam >= 8)).sum())
        meta[o] = dict(sun_axis=round(float(sun), 1), incidence=round(float(inc), 1),
                       north_az=north, n=len(df), n_elong8=ne8, tiles=info["tiles"],
                       res=info["res"], wall_s=round(time.time() - t0, 1))
        print(f"  {o}: sun={sun:.0f} inc={inc:.0f} res={info['res']:.2f} n={len(df)} "
              f"elong8={ne8} tiles={info['tiles']} {time.time()-t0:.0f}s", flush=True)
        pd.concat(all_rows, ignore_index=True).to_csv(OUT / "t1_controls_sam.csv", index=False)
    (OUT / "t1_controls_meta.json").write_text(json.dumps(meta, indent=2))
    print("DONE\n" + json.dumps(meta, indent=2), flush=True)
