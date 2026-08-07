"""SAM orientation-artifact generality run (2026-07-23).

Question: is SAM's aggregate orientation lean a HiRISE-shadow-wide pattern, and does it track each
scene's shadow axis? Runs the VALIDATED whole-tile methodology (run_t35_val): cut 512-px tiles from a
central strip of each HiRISE scene, YOLO-detect, SAM2 re-mask (sam256, production), read moment angles
for both. No human labels on these scenes -> SAM-vs-YOLO-vs-sun-axis only (weaker than the val, per
the plan). Shadow (=sun) axis per scene from the LBLs: geographic = (SUB_SOLAR - NORTH) mod 180.

Idempotent per scene. __main__ guard (Windows spawn). Run from inside YOLOv8-BeyondEarth/:
  conda run -n bouldernet python data/sam2_validation/run_sam_chirality.py
"""
import json, re, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(r"c:/Users/brian/Documents/PhD/BoulderNet/YOLOv8-BeyondEarth")
sys.path.insert(0, str(BASE / "src"))
OUT = BASE / "data" / "sam2_validation"
WEIGHTS = BASE / "data" / "weights" / "yolov8-m-boulder-detection-tmp.pt"
SAM_WEIGHTS = BASE / "sam2.1_b.pt"

TILE = 512            # whole-tile slice = sam256 native-tile px budget (matches run_t35)
BORDER = 8            # drop detections within BORDER px of a tile edge (split boulders)
STRIP_W = 3072        # central vertical strip width (px) to scan
TARGET_ELONG = 800    # stop a scene once it has this many elongated SAM detections
MAX_TILES = 500       # or after this many tiles, whichever first
ASPECT = 1.35

SCENES = [  # (obsid, relative tif path). 2260 base first (the reference scene), then Test-2 stack.
    ("ESP_017355_2260", "data/mars_hirise/raster/ESP_017355_2260_RED.tif"),
    ("ESP_076235_2260", "data/test2_pairs/ESP_076235_2260/ESP_076235_2260_RED.tif"),
    ("ESP_067214_2270", "data/test2_pairs/ESP_067214_2270/ESP_067214_2270_RED.tif"),
    ("ESP_059499_2270", "data/test2_pairs/ESP_059499_2270/ESP_059499_2270_RED.tif"),
    ("ESP_055833_2260", "data/test2_pairs/ESP_055833_2260/ESP_055833_2260_RED.tif"),
    ("ESP_055714_2270", "data/test2_pairs/ESP_055714_2270/ESP_055714_2270_RED.tif"),
    ("PSP_007492_2265", "data/test2_pairs/PSP_007492_2265/PSP_007492_2265_RED.tif"),
    ("PSP_007703_2265", "data/test2_pairs/PSP_007703_2265/PSP_007703_2265_RED.tif"),
    ("ESP_069088_2270", "data/test2_pairs/ESP_069088_2270/ESP_069088_2270_RED.tif"),
    ("ESP_071923_2260", "data/test2_pairs/ESP_071923_2260/ESP_071923_2260_RED.tif"),
    ("ESP_071699_2260", "data/test2_pairs/ESP_071699_2260/ESP_071699_2260_RED.tif"),
]


def sun_table():
    """geographic shadow (=sun) axis per scene = (SUB_SOLAR - NORTH) mod 180, from the LBLs."""
    out = {}
    lbls = list((BASE / "data/test2_pairs").rglob("*.LBL"))
    lbls.append(BASE / "data/mars_hirise/raster/ESP_017355_2260_RED.LBL")
    for p in set(lbls):
        t = p.read_text(errors="replace")
        def g(k):
            m = re.search(rf"{k}\s*=\s*([-\d.]+)", t)
            return float(m.group(1)) if m else None
        ss, na, inc = g("SUB_SOLAR_AZIMUTH"), g("NORTH_AZIMUTH"), g("INCIDENCE_ANGLE")
        if ss is not None and na is not None:
            out[p.stem.replace("_RED", "")] = dict(shadow_axis=round((ss - na) % 180.0, 1),
                                                   incidence=inc)
    return out


def process_scene(model, sam, tif, angles_from_moments, detect_sam2_mask_sets):
    import rasterio
    from rasterio.windows import Window
    with rasterio.open(tif) as src:
        H, W = src.height, src.width
        res = src.res[0]
        c0 = max(0, W // 2 - STRIP_W // 2)
        cw = min(STRIP_W, W - c0)
        rows, n_elong, ntiles, realigned = [], 0, 0, 0
        for r0 in range(0, H - TILE, TILE):
            for cc in range(c0, c0 + cw - TILE, TILE):
                if ntiles >= MAX_TILES or n_elong >= TARGET_ELONG:
                    break
                tile = src.read(1, window=Window(cc, r0, TILE, TILE))
                ntiles += 1
                if tile.shape != (TILE, TILE) or tile.max() == tile.min():
                    continue
                # X-06: pass the cap explicitly. The 2026-07 run behind `sam_chirality*.csv` did
                # not, so ultralytics' silent 300 applied — one of the two defects (with the
                # raster-order/quota sampler) that retired that sample.
                df, masks = detect_sam2_mask_sets(model, sam, tile, slice_size=TILE, imgsz=1024,
                                                  conf=0.10, device=0, configs=("yolo", "sam256"),
                                                  sam_batch=64, empty_cache_every=64, max_det=600)
                realigned += int(df.attrs.get("realigned_chunks", 0))
                if len(df) == 0:
                    continue
                ay = angles_from_moments(masks["yolo"]); asm = angles_from_moments(masks["sam256"])
                keep = ((df.cx >= BORDER) & (df.cx < TILE - BORDER) &
                        (df.cy >= BORDER) & (df.cy < TILE - BORDER)).to_numpy()
                for di in np.where(keep)[0]:
                    rows.append(dict(scene=Path(tif).stem.replace("_RED", ""), res=res,
                                     angle_yolo=float(ay["angle180"].iloc[di]),
                                     aspect_yolo=float(ay["aspect"].iloc[di]),
                                     angle_sam=float(asm["angle180"].iloc[di]),
                                     aspect_sam=float(asm["aspect"].iloc[di]),
                                     dpx_sam=float(asm["diameter_px"].iloc[di])))
                n_elong = sum(1 for r in rows if r["aspect_sam"] > ASPECT)
            if ntiles >= MAX_TILES or n_elong >= TARGET_ELONG:
                break
    return pd.DataFrame(rows), dict(tiles=ntiles, res=res, realigned=realigned)


if __name__ == "__main__":
    from ultralytics import YOLO, SAM
    from YOLOv8BeyondEarth.orientation_validation import detect_sam2_mask_sets, angles_from_moments

    suns = sun_table()
    model = YOLO(str(WEIGHTS))
    sam = SAM(str(SAM_WEIGHTS))
    all_rows, meta = [], {}
    t0 = time.time()
    for obsid, rel in SCENES:
        tif = BASE / rel
        if not tif.exists():
            print(f"SKIP {obsid}: no tif", flush=True); continue
        ts = time.time()
        df, info = process_scene(model, sam, str(tif), angles_from_moments, detect_sam2_mask_sets)
        df["shadow_axis"] = suns.get(obsid, {}).get("shadow_axis", np.nan)
        df["incidence"] = suns.get(obsid, {}).get("incidence", np.nan)
        all_rows.append(df)
        ne = int((df.aspect_sam > ASPECT).sum())
        meta[obsid] = dict(n=len(df), n_elong=ne, shadow_axis=suns.get(obsid, {}).get("shadow_axis"),
                           incidence=suns.get(obsid, {}).get("incidence"), wall_s=round(time.time()-ts, 1),
                           **info)
        print(f"  {obsid}: n={len(df)} elong={ne} tiles={info['tiles']} shadow={df.shadow_axis.iloc[0] if len(df) else '?'} "
              f"{time.time()-ts:.0f}s", flush=True)
        pd.concat(all_rows, ignore_index=True).to_csv(OUT / "sam_chirality_hirise.csv", index=False)
    (OUT / "sam_chirality_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nDONE {len(SCENES)} scenes in {(time.time()-t0)/60:.1f} min", flush=True)
    print(json.dumps(meta, indent=2), flush=True)
