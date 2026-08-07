"""Cut LRO NAC crops covering the Censorinus human-mapped ROM (for the same-area three-way rose).

Why. The summary notebook's flagship figure compares YOLO / SAM2 / human orientations **over the same
ground**. On Mars that footprint already exists (`data/solutionDE/prep_rom_crops.py` cut 20 x 1024^2
windows around the Test-6 HiRISE ROM tiles). The Moon needs the same thing, and its ROM is shaped
differently: **338 contiguous-ish 500 x 500-px tiles** rather than 20 scattered ones. Cutting 338
padded windows would triple-count the overlaps, so tiles are grouped into **connected clusters** and
one padded window is cut per cluster (split if a cluster is huge).

Footprint accounting, stated because it is not obvious. `M139694087LE-censorinus-boulder-mapping.shp`
holds 12,753 polygons but only **6,271 centroids fall inside the ROM** — the rest sit a median 172 m
away, i.e. in mapped ground that is genuinely outside the region-of-measurement tiles (checked
2026-07-26; not an edge/containment artifact). The ROM is the footprint of record here, matching
Test 6 and the Goal-3 recall work, so the other 6,482 are out of scope and the manifest records both
numbers.

Padding. Each cluster window is grown by `PAD` px so boulders at the ROM edge still get full slice
context (same reasoning as the HiRISE prep, which used a 262-px margin). Detections are clipped back
to the exact ROM geometry downstream, so the padding never leaks into the statistics.

CRS gotcha: the Prieur shapefiles carry a datum whose name differs from the raster's by punctuation,
so they are attached with `set_crs(..., allow_override=True)` and **never** reprojected with
`to_crs` (that produces a bogus datum shift and clips everything away).

Idempotent (skips crops that exist). Run from inside YOLOv8-BeyondEarth/:
  C:\\Users\\brian\\anaconda3\\Scripts\\conda.exe run -n bouldernet python data/summary/prep_moon_rom_crops.py
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import label as cc_label

BASE = Path(r"c:\Users\brian\Documents\PhD\BoulderNet")
OUT = BASE / "YOLOv8-BeyondEarth" / "data" / "summary"
PRIEUR = BASE / "data_prieur" / "raw_data" / "moon" / "censorinus" / "shp"
NAC = BASE / "YOLOv8-BeyondEarth" / "data" / "raster" / "M139694087LE.tif"

TILE = 500          # ROM tile size in px (the Prieur mapping grid)
PAD = 256           # slice-context margin around each cluster (cf. HiRISE prep's 262)
MAX_SIDE = 3072     # split clusters larger than this, to keep single reads bounded
MIN_VALID = 0.50    # skip a window that is mostly nodata border, and say so


def tile_windows(rom, T):
    """Pixel (col0, row0) of each ROM tile, snapped to the TILE grid."""
    out = []
    for i, geom in enumerate(rom.geometry):
        b = geom.bounds
        col, row = (~T) * (b[0], b[3])
        out.append((i, int(round(col / TILE)), int(round(row / TILE))))
    return out


def clusters(tiles):
    """Group ROM tiles into 8-connected clusters on the TILE grid; yields tile-index lists."""
    cols = [c for _, c, _ in tiles]
    rows = [r for _, _, r in tiles]
    c0, r0 = min(cols), min(rows)
    grid = np.zeros((max(rows) - r0 + 1, max(cols) - c0 + 1), bool)
    for _, c, r in tiles:
        grid[r - r0, c - c0] = True
    lab, n = cc_label(grid, structure=np.ones((3, 3), int))
    groups = {}
    for idx, c, r in tiles:
        groups.setdefault(int(lab[r - r0, c - c0]), []).append((idx, c, r))
    print(f"{len(tiles)} ROM tiles -> {n} connected clusters")
    return list(groups.values())


def split(c0, r0, w, h):
    """Chop a window into <= MAX_SIDE pieces with PAD overlap so nothing falls in a seam."""
    step = MAX_SIDE - PAD
    for cc in range(c0, c0 + w, step):
        for rr in range(r0, r0 + h, step):
            yield cc, rr, min(MAX_SIDE, c0 + w - cc), min(MAX_SIDE, r0 + h - rr)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with rasterio.open(NAC) as src:
        crs, T, W, H = src.crs, src.transform, src.width, src.height
        res = abs(T.a)

    rom = (gpd.read_file(PRIEUR / "M139694087LE-censorinus-ROM.shp")
           .set_crs(crs, allow_override=True).explode(index_parts=False).reset_index(drop=True))
    hum = (gpd.read_file(PRIEUR / "M139694087LE-censorinus-boulder-mapping.shp")
           .set_crs(crs, allow_override=True))
    cent = hum.geometry.centroid
    rom_union = rom.union_all()
    n_in_rom = int(cent.within(rom_union).sum())
    print(f"raster {W}x{H} @ {res:.4f} m/px | ROM parts {len(rom)} | human polys {len(hum):,} "
          f"| inside ROM {n_in_rom:,} (the {len(hum) - n_in_rom:,} outside are mapped ground beyond "
          f"the ROM tiles and are OUT OF SCOPE for the same-area comparison)")

    tiles = tile_windows(rom, T)
    in_rom = cent.within(rom_union)
    # Coverage is tracked as a SET of label ids, not a running sum: crop windows are padded and
    # therefore overlap, so summing per-crop counts double-counts every label near a seam (it
    # reported 7,019 of 6,271 before this was fixed).
    covered = set()
    manifest, skipped = {}, []
    with rasterio.open(NAC) as src:
        for grp in clusters(tiles):
            cs = [c for _, c, _ in grp]
            rs = [r for _, _, r in grp]
            c0 = int(np.clip(min(cs) * TILE - PAD, 0, W - 1))
            r0 = int(np.clip(min(rs) * TILE - PAD, 0, H - 1))
            c1 = int(np.clip((max(cs) + 1) * TILE + PAD, 0, W))
            r1 = int(np.clip((max(rs) + 1) * TILE + PAD, 0, H))
            for cc, rr, ww, hh in split(c0, r0, c1 - c0, r1 - r0):
                ww, hh = min(ww, W - cc), min(hh, H - rr)
                if ww < TILE or hh < TILE:
                    continue
                name = f"moon_rom_c{cc}_r{rr}"
                path = OUT / f"{name}.tif"
                win = Window(cc, rr, ww, hh)
                arr = src.read(1, window=win)
                valid = float((arr > 0).mean())
                if valid < MIN_VALID:
                    skipped.append(dict(name=name, reason="mostly nodata", valid_frac=valid))
                    continue
                if not path.exists():
                    prof = src.profile.copy()
                    prof.update(width=ww, height=hh, count=1, transform=src.window_transform(win))
                    for k in ("blockxsize", "blockysize"):
                        prof.pop(k, None)
                    prof["tiled"] = False
                    with rasterio.open(path, "w", **prof) as dst:
                        dst.write(arr, 1)
                bx = rasterio.windows.bounds(win, T)
                hit = in_rom & cent.x.between(bx[0], bx[2]) & cent.y.between(bx[1], bx[3])
                covered |= set(hum.index[hit])
                manifest[name] = dict(window=[cc, rr, ww, hh], bounds=list(bx),
                                      n_human_in_rom=int(hit.sum()),
                                      valid_frac=round(valid, 4), n_rom_tiles=len(grp))

    meta = dict(raster=str(NAC), res_m=res, tile_px=TILE, pad_px=PAD, max_side=MAX_SIDE,
                n_rom_tiles=len(rom), n_human_total=int(len(hum)), n_human_in_rom=n_in_rom,
                n_human_covered_by_crops=len(covered), n_crops=len(manifest), skipped=skipped,
                crops=manifest)
    (OUT / "moon_rom_manifest.json").write_text(json.dumps(meta, indent=2))
    px = sum(m["window"][2] * m["window"][3] for m in manifest.values())
    print(f"\nwrote {len(manifest)} crops ({px / 1e6:.0f} Mpx incl. padding) covering "
          f"{len(covered):,}/{n_in_rom:,} in-ROM human labels")
    if skipped:
        print(f"SKIPPED {len(skipped)} windows (mostly nodata): "
              + ", ".join(f"{s['name']} valid={s['valid_frac']:.2f}" for s in skipped[:8])
              + (" ..." if len(skipped) > 8 else ""))
    if len(covered) < n_in_rom:
        print(f"NOTE {n_in_rom - len(covered):,} in-ROM labels are not inside any written crop "
              f"(nodata-skipped windows) — recorded in the manifest.")


if __name__ == "__main__":
    main()
