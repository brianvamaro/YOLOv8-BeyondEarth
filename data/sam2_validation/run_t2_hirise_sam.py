"""T2 (SAM) on HiRISE repeat pairs — does the boulder rose rotate with the sun on FIXED ground?

The '_2260/_2270' suffix is a latitude band, not one footprint; the genuine overlapping repeat pairs
are below. For each pair we scan the SAME geographic overlap of both acquisitions (no reprojection ->
sidesteps the lat_0 bug; the WKT may differ but the numeric coords coincide for these repeat-target
acquisitions), run detect->SAM2 + YOLO, and compare the population rose to each acquisition's sun axis.
Ground is FIXED, so: rose rotates with the sun => illumination; stays put => fabric/decoder.
Weak arm (Mars sun-synchronous: Δsun only 13-29°/pair) -> direction-of-effect, not a slope.

Scans the overlap tile-by-tile (on-demand reads) until TARGET elongated boulders, robust to
boulder-poor patches. __main__ guard (Windows spawn).
  conda run -n bouldernet python data/sam2_validation/run_t2_hirise_sam.py
"""
import json, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(r"c:/Users/brian/Documents/PhD/BoulderNet/YOLOv8-BeyondEarth")
sys.path.insert(0, str(BASE / "src"))
OUT = BASE / "data" / "sam2_validation"
WEIGHTS = BASE / "data/weights/yolov8-m-boulder-detection-tmp.pt"
SAM_WEIGHTS = BASE / "sam2.1_b.pt"
TILE = 512; BORDER = 8; TARGET = 700; MAX_TILES = 600

PAIRS = [("2270a", "ESP_067214_2270", "ESP_069088_2270"),   # Δsun ~29 (best), res 0.25
         ("2260b", "ESP_076235_2260", "ESP_055833_2260"),   # Δsun ~18, res 0.5
         ("2270b", "ESP_059499_2270", "ESP_055714_2270")]   # Δsun ~13, res 0.5


def tif(o):
    p = BASE / f"data/test2_pairs/{o}/{o}_RED.tif"
    return p if p.exists() else BASE / f"data/mars_hirise/raster/{o}_RED.tif"


def sun_of(o):
    t = tif(o).with_suffix(".LBL").read_text(errors="replace")
    g = lambda k: float(re.search(rf"{k}\s*=\s*([-\d.]+)", t).group(1))
    return (g("SUB_SOLAR_AZIMUTH") - g("NORTH_AZIMUTH")) % 180.0, g("INCIDENCE_ANGLE")


def scan_overlap(src, sub_bounds, model, sam, detect_fn, ang_fn):
    """Iterate 512-tiles across the overlap (on-demand reads) until TARGET elong8 / MAX_TILES."""
    from rasterio.windows import from_bounds, Window
    w = from_bounds(*sub_bounds, transform=src.transform)
    c_off, r_off = int(round(w.col_off)), int(round(w.row_off))
    W = min(int(round(w.width)), src.width - c_off); H = min(int(round(w.height)), src.height - r_off)
    rows, ne, nt = [], 0, 0
    for r0 in range(r_off, r_off + H - TILE, TILE):
        if ne >= TARGET or nt >= MAX_TILES:
            break
        for c0 in range(c_off, c_off + W - TILE, TILE):
            if ne >= TARGET or nt >= MAX_TILES:
                break
            tile = src.read(1, window=Window(c0, r0, TILE, TILE)); nt += 1
            if tile.shape != (TILE, TILE) or tile.max() == tile.min():
                continue
            df, masks = detect_fn(model, sam, tile, slice_size=TILE, imgsz=1024, conf=0.10, device=0,
                                  configs=("yolo", "sam256"), sam_batch=64, empty_cache_every=64)
            if len(df) == 0:
                continue
            ay = ang_fn(masks["yolo"]); asm = ang_fn(masks["sam256"])
            keep = ((df.cx >= BORDER) & (df.cx < TILE - BORDER) & (df.cy >= BORDER) & (df.cy < TILE - BORDER)).to_numpy()
            for di in np.where(keep)[0]:
                rows.append(dict(angle_sam=float(asm["angle180"].iloc[di]), aspect_sam=float(asm["aspect"].iloc[di]),
                                 dpx_sam=float(asm["diameter_px"].iloc[di]), angle_yolo=float(ay["angle180"].iloc[di]),
                                 aspect_yolo=float(ay["aspect"].iloc[di])))
            ne = sum(1 for r in rows if r["aspect_sam"] > 1.35 and r["dpx_sam"] >= 8)
    return rows, nt


if __name__ == "__main__":
    import rasterio
    from ultralytics import YOLO, SAM
    from YOLOv8BeyondEarth.orientation_validation import detect_sam2_mask_sets, angles_from_moments
    model = YOLO(str(WEIGHTS)); sam = SAM(str(SAM_WEIGHTS))
    all_rows, meta = [], {}
    for pid, a, b in PAIRS:
        with rasterio.open(tif(a)) as sa, rasterio.open(tif(b)) as sb:
            note = "" if str(sa.crs) == str(sb.crs) else " [CRS-WKT differ; numeric-bounds overlap]"
            L = max(sa.bounds.left, sb.bounds.left); B = max(sa.bounds.bottom, sb.bounds.bottom)
            R = min(sa.bounds.right, sb.bounds.right); T = min(sa.bounds.top, sb.bounds.top)
            if R <= L or T <= B:
                print(f"SKIP {pid}: no numeric-bounds overlap", flush=True); continue
            print(f"{pid}: overlap {R-L:.0f}x{T-B:.0f} m{note}", flush=True)
            sub = (L, B, R, T)
            for o, src in [(a, sa), (b, sb)]:
                t0 = time.time()
                rows, nt = scan_overlap(src, sub, model, sam, detect_sam2_mask_sets, angles_from_moments)
                sun, inc = sun_of(o)
                for r in rows:
                    r.update(pair=pid, scene=o, sun=sun, inc=inc, res=src.res[0])
                all_rows += rows
                ne = sum(1 for r in rows if r["aspect_sam"] > 1.35 and r["dpx_sam"] >= 8)
                meta[o] = dict(pair=pid, sun=round(sun, 1), inc=round(inc, 1), n=len(rows), n_elong8=ne,
                               tiles=nt, wall_s=round(time.time() - t0, 1))
                print(f"  {pid} {o}: sun={sun:.0f} inc={inc:.0f} n={len(rows)} elong8={ne} tiles={nt} "
                      f"{time.time()-t0:.0f}s", flush=True)
                pd.DataFrame(all_rows).to_csv(OUT / "t2_hirise_sam.csv", index=False)
    (OUT / "t2_hirise_meta.json").write_text(json.dumps(meta, indent=2))
    print("DONE\n" + json.dumps(meta, indent=2), flush=True)
