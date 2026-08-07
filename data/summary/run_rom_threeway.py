"""YOLO / SAM2 / human orientations over the SAME GROUND, on Mars and the Moon.

What this produces. The summary notebook's flagship figure needs the three producers read two ways,
and the existing caches only cover one of them:

* **matched detections** — same boulder, all three producers. `data/sam2_validation/t35_val_detections.csv`
  already has this for the boulder2024 *validation tiles*.
* **same area** — each producer's *own* population inside one identical footprint. Nothing had this
  on a real scene, so it is measured here.

The "same area" read matters because a matched comparison silently conditions on the boulders all
three happen to agree on, which is exactly the selection effect the archive found doing most of the
work behind the scary-looking population roses. Over a fixed footprint nothing is conditioned away:
if the model reports a preferred axis the humans do not, that is visible.

Footprint = the Prieur **ROM** (region of measurement) polygons, the ground the annotators actually
mapped, on both labelled scenes:

* Mars HiRISE `ESP_017355_2260` — 20 x 1024^2 crops around the 20 ROM tiles, already cut by
  `data/solutionDE/prep_rom_crops.py` (5,324 labels).
* Moon LRO NAC `M139694087LE` (Censorinus) — 54 crops cut by `prep_moon_rom_crops.py`
  (6,271 in-ROM labels of 12,753 mapped; the rest lie outside the ROM tiles).

Pipeline per crop: production sliced detection (`slice_size=256, imgsz=1024, conf=0.10`) with the
production **G** re-masker (`detect_sam2_mask_sets(..., configs=("yolo","sam256"),
empty_cache_every=64)`), moment angles for both mask sets, then a greedy centroid match to the human
polygons using the same radius rule as the validation run (`max(0.75 * human_dpx, 6 px)`).

Three correctness details worth knowing:

* **Crops overlap** (padding, and HiRISE windows can be within 1024 px of each other), so detections
  are de-duplicated in *world* coordinates before anything is counted — otherwise a boulder in a seam
  votes twice.
* **CRS**: the Prieur shapefiles are attached with `set_crs(..., allow_override=True)` and never
  reprojected; `to_crs` invents a datum shift that clips the whole set away.
* **Angle frames**: mask angles come from `angles_from_moments` (geographic azimuth, North = 0, in the
  raster's own north-up frame); human angles from `ellipse_angle180` on the world-coordinate polygon
  (same convention, since projected y increases north). No y-flip is applied here — that flip is only
  needed for *tile-pixel* label polygons, as in `run_t35_val.py`.

Idempotent per scene (skips if the CSV already covers it unless --force). GPU. __main__ guard
(Windows spawn). Run from inside YOLOv8-BeyondEarth/:
  C:\\Users\\brian\\anaconda3\\Scripts\\conda.exe run -n bouldernet --no-capture-output \\
      python data/summary/run_rom_threeway.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"c:\Users\brian\Documents\PhD\BoulderNet")
ROOT = BASE / "YOLOv8-BeyondEarth"
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "data" / "summary"
DE = ROOT / "data" / "solutionDE"
PRIEUR = BASE / "data_prieur" / "raw_data"
WEIGHTS = ROOT / "data" / "weights" / "yolov8-m-boulder-detection-tmp.pt"
SAM_WEIGHTS = ROOT / "sam2.1_b.pt"

SLICE, IMGSZ, CONF = 256, 1024, 0.10      # the production detection config
MATCH_MIN_R_PX = 6.0                      # min centroid match radius, as in run_t35_val.py
DEDUP_PX = 1.0                            # world-space dedup radius, in native pixels


def scene_specs():
    return [
        dict(body="mars", scene="ESP_017355_2260", res=0.5, sun_axis=75.3,
             crops=sorted(DE.glob("hirise_rom*.tif")),
             rom=PRIEUR / "mars/ESP_017355_2260/shp/ESP_017355_2260-ROM.shp",
             human=PRIEUR / "mars/ESP_017355_2260/shp/ESP_017355_2260-boulder-mapping.shp"),
        dict(body="moon", scene="M139694087LE", res=0.48357597846618, sun_axis=98.6,
             crops=sorted(OUT.glob("moon_rom_*.tif")),
             rom=PRIEUR / "moon/censorinus/shp/M139694087LE-censorinus-ROM.shp",
             human=PRIEUR / "moon/censorinus/shp/M139694087LE-censorinus-boulder-mapping.shp"),
    ]


def load_humans(spec, crs):
    """Human polygons with orientation, plus the ROM union, all in the raster CRS."""
    import geopandas as gpd
    from YOLOv8BeyondEarth.orientation import ellipse_angle180
    rom = (gpd.read_file(spec["rom"]).set_crs(crs, allow_override=True)
           .explode(index_parts=False).reset_index(drop=True))
    hum = gpd.read_file(spec["human"]).set_crs(crs, allow_override=True)
    rom_union = rom.union_all()
    c = hum.geometry.centroid
    inside = c.within(rom_union)
    hum = hum[inside].reset_index(drop=True)
    res = spec["res"]
    ang, asp = zip(*(ellipse_angle180(g, res) for g in hum.geometry))
    out = pd.DataFrame(dict(
        x=hum.geometry.centroid.x.values, y=hum.geometry.centroid.y.values,
        h_angle=np.asarray(ang, float), h_aspect=np.asarray(asp, float),
        h_area=hum.geometry.area.values / res ** 2,                     # px^2, like the mask areas
        h_dpx=2 * np.sqrt(hum.geometry.area.values / np.pi) / res))
    return out, rom_union


def dedup_world(df, res):
    """Drop repeat detections of the same boulder from overlapping crop windows.

    Greedy, keeping the first occurrence within ``DEDUP_PX`` native pixels — enough to catch the
    same rock seen in two padded windows, far below inter-boulder spacing.
    """
    from scipy.spatial import cKDTree
    if df.empty:
        return df, 0
    xy = df[["x", "y"]].to_numpy()
    tree = cKDTree(xy)
    keep = np.ones(len(df), bool)
    for i, j in tree.query_pairs(r=DEDUP_PX * res):
        if keep[i] and keep[j]:
            keep[j] = False
    return df[keep].reset_index(drop=True), int((~keep).sum())


def match_humans(det, hum):
    """Greedy nearest-centroid match, radius max(0.75*h_dpx, MATCH_MIN_R_PX) in metres."""
    from scipy.spatial import cKDTree
    for c in ("h_angle", "h_aspect", "h_area", "h_dpx"):
        det[c] = np.nan
    if det.empty or hum.empty:
        return det
    hc = hum[["x", "y"]].to_numpy()
    dc = det[["x", "y"]].to_numpy()
    hr = np.maximum(0.75 * hum.h_dpx.to_numpy(), MATCH_MIN_R_PX) * det.attrs["res"]
    pairs = cKDTree(hc).query_ball_tree(cKDTree(dc), r=float(hr.max()))
    cand = sorted((float(np.hypot(*(hc[hi] - dc[di]))), hi, di)
                  for hi, dis in enumerate(pairs) for di in dis
                  if np.hypot(*(hc[hi] - dc[di])) <= hr[hi])
    uh, ud = set(), set()
    for _, hi, di in cand:
        if hi in uh or di in ud:
            continue
        uh.add(hi)
        ud.add(di)
        for c in ("h_angle", "h_aspect", "h_area", "h_dpx"):
            det.iat[di, det.columns.get_loc(c)] = hum.at[hi, c]
    return det


def run_scene(spec, model, sam, max_crops=None):
    import rasterio
    from YOLOv8BeyondEarth.orientation_validation import detect_sam2_mask_sets, angles_from_moments
    res = spec["res"]
    if max_crops:
        spec = dict(spec, crops=spec["crops"][:max_crops])
    with rasterio.open(spec["crops"][0]) as s:
        crs = s.crs
    hum, rom_union = load_humans(spec, crs)
    print(f"[{spec['scene']}] {len(spec['crops'])} crops | {len(hum):,} human labels in ROM",
          flush=True)

    rows, failed, realigned, t0 = [], {}, 0, time.time()
    for ci, path in enumerate(spec["crops"]):
        with rasterio.open(path) as src:
            arr = src.read(1)
            T = src.transform
        if arr.max() == 0:
            continue
        df, masks = detect_sam2_mask_sets(model, sam, arr, slice_size=SLICE, imgsz=IMGSZ,
                                          conf=CONF, device=0, configs=("yolo", "sam256"),
                                          empty_cache_every=64)
        realigned += int(df.attrs.get("realigned_chunks", 0))
        for k, v in (df.attrs.get("failed") or {}).items():
            failed[k] = failed.get(k, 0) + int(v)
        if len(df) == 0:
            continue
        ay, asm = angles_from_moments(masks["yolo"]), angles_from_moments(masks["sam256"])
        wx, wy = T * (df.cx.to_numpy() + 0.5, df.cy.to_numpy() + 0.5)
        rows.append(pd.DataFrame(dict(
            crop=path.stem, x=wx, y=wy, score=df.score.to_numpy(),
            angle_yolo=ay.angle180.to_numpy(), aspect_yolo=ay.aspect.to_numpy(),
            dpx_yolo=ay.diameter_px.to_numpy(), area_yolo=df.area_yolo.to_numpy(),
            angle_sam=asm.angle180.to_numpy(), aspect_sam=asm.aspect.to_numpy(),
            dpx_sam=asm.diameter_px.to_numpy(), area_sam=df.area_sam256.to_numpy())))
        if (ci + 1) % 10 == 0 or ci + 1 == len(spec["crops"]):
            n = sum(len(r) for r in rows)
            print(f"  {ci+1}/{len(spec['crops'])} crops  dets={n:,}  {time.time()-t0:.0f}s",
                  flush=True)

    det = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    n_raw = len(det)
    det, n_dupes = dedup_world(det, res)
    if not det.empty:
        import geopandas as gpd
        from shapely.geometry import Point
        pts = gpd.GeoSeries([Point(a, b) for a, b in zip(det.x, det.y)], crs=crs)
        det = det[pts.within(rom_union).to_numpy()].reset_index(drop=True)
    det.attrs["res"] = res
    n_in_rom = len(det)
    det = match_humans(det, hum)
    det["body"], det["scene"], det["res"], det["sun_axis"] = (
        spec["body"], spec["scene"], res, spec["sun_axis"])

    meta = dict(scene=spec["scene"], body=spec["body"], res_m=res, sun_axis=spec["sun_axis"],
                n_crops=len(spec["crops"]), n_det_raw=n_raw, n_dupes_dropped=n_dupes,
                n_det_in_rom=n_in_rom, n_human_in_rom=int(len(hum)),
                n_matched=int(det.h_angle.notna().sum()),
                remask_failed=failed, realigned_chunks=realigned,
                wall_s=round(time.time() - t0, 1),
                config=dict(slice_size=SLICE, imgsz=IMGSZ, conf=CONF, sam="sam256"))
    fail_rate = (failed.get("sam256", 0) / max(1, n_raw))
    print(f"[{spec['scene']}] dets {n_raw:,} raw -> {n_in_rom:,} in ROM (dropped {n_dupes:,} dupes) "
          f"| matched {meta['n_matched']:,}/{len(hum):,} human "
          f"| SAM re-mask failures {failed.get('sam256', 0):,} ({fail_rate:.1%}) "
          f"| {meta['wall_s']:.0f}s", flush=True)
    if fail_rate > 0.05:
        print(f"  WARNING SAM re-mask failure rate {fail_rate:.1%} exceeds the ~0.5-2% seen on the "
              f"benchmarked crops — check the box-blowup guard before trusting this scene.",
              flush=True)
    return det, meta


def main():
    from ultralytics import SAM, YOLO
    OUT.mkdir(parents=True, exist_ok=True)
    force = "--force" in sys.argv
    # --max-crops N: smoke-test a couple of crops per scene before committing to the full run.
    # It writes to a separate file so a partial run can never be mistaken for the real cache.
    max_crops = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--max-crops=")), None)
    csv = OUT / ("rom_threeway_smoke.csv" if max_crops else "rom_threeway.csv")
    have = set()
    if csv.exists() and not force and not max_crops:
        have = set(pd.read_csv(csv, usecols=["scene"]).scene.unique())
        print(f"existing {csv.name} covers {sorted(have)} (use --force to redo)")

    specs = [s for s in scene_specs() if s["scene"] not in have]
    if not specs:
        print("nothing to do")
        return
    for s in specs:
        if not s["crops"]:
            print(f"[{s['scene']}] NO CROPS FOUND — run the prep script first"); return

    model, sam = YOLO(str(WEIGHTS)), SAM(str(SAM_WEIGHTS))
    frames, metas = [], []
    if have:
        frames.append(pd.read_csv(csv))
        metas = json.loads((OUT / "rom_threeway_meta.json").read_text()).get("scenes", [])
    for s in specs:
        det, meta = run_scene(s, model, sam, max_crops)
        frames.append(det)
        metas.append(meta)

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(csv, index=False)
    meta_path = OUT / ("rom_threeway_smoke_meta.json" if max_crops else "rom_threeway_meta.json")
    meta_path.write_text(json.dumps(dict(smoke_max_crops=max_crops, scenes=metas), indent=2))
    print(f"\nwrote {csv} ({len(out):,} rows)")
    for m in metas:
        print(f"  {m['scene']:16s} in-ROM {m['n_det_in_rom']:>7,}  human {m['n_human_in_rom']:>6,}  "
              f"matched {m['n_matched']:>6,}")


if __name__ == "__main__":
    main()
