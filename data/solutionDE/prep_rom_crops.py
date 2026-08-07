# Goal-3 phase 3: HiRISE ROM-tile crops for the recall-vs-human item (plan §Validation 4).
# The Test-6 ROM is 20 scattered 500x500-px tiles; the phase-3 hirise_crop4k (scene centre)
# overlaps NONE of them (checked 2026-07-10), so recall-vs-ROM gets its own crops: one 1024^2
# window centred on each ROM tile (262-px context margin around the mapped square, so boulders
# at the ROM edge keep full slice context). All 20 tiles = the full 5,324-label reference.
# NAC needs no extra prep: nac_crop2k already overlaps its ROM (1,167 human polys inside).
import json
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from pathlib import Path

BASE = Path(r"c:\Users\brian\Documents\PhD\BoulderNet")
DE = BASE / "YOLOv8-BeyondEarth" / "data" / "solutionDE"
PRIEUR = BASE / "data_prieur" / "raw_data"
HIRISE = BASE / "YOLOv8-BeyondEarth" / "data" / "mars_hirise" / "raster" / "ESP_017355_2260_RED.tif"
SIZE = 1024

with rasterio.open(HIRISE) as src:
    crs, T, W, H = src.crs, src.transform, src.width, src.height

rom = (gpd.read_file(PRIEUR / "mars/ESP_017355_2260/shp/ESP_017355_2260-ROM.shp")
       .set_crs(crs, allow_override=True).explode(index_parts=False).reset_index(drop=True))
hum = (gpd.read_file(PRIEUR / "mars/ESP_017355_2260/shp/ESP_017355_2260-boulder-mapping.shp")
       .set_crs(crs, allow_override=True))
cent = hum.geometry.centroid

manifest = {}
for i, r in rom.iterrows():
    cx, cy = r.geometry.centroid.x, r.geometry.centroid.y
    col, row = (~T) * (cx, cy)
    col0 = int(np.clip(round(col - SIZE / 2), 0, W - SIZE))
    row0 = int(np.clip(round(row - SIZE / 2), 0, H - SIZE))
    out = DE / f"hirise_rom{i:02d}.tif"
    with rasterio.open(HIRISE) as src:
        w = Window(col0, row0, SIZE, SIZE)
        arr = src.read(1, window=w)
        prof = src.profile.copy()
        prof.update(width=SIZE, height=SIZE, count=1, transform=src.window_transform(w))
        prof.pop("blockxsize", None); prof.pop("blockysize", None)
        prof["tiled"] = False
        if not out.exists():
            with rasterio.open(out, "w", **prof) as dst:
                dst.write(arr, 1)
    n_in = int(cent.within(r.geometry).sum())
    manifest[out.stem] = dict(window=[col0, row0, SIZE], rom_index=i, n_human=n_in,
                              nonzero_frac=float((arr > 0).mean()))
    print(f"{out.name}: window=({col0},{row0})  n_human={n_in}  "
          f"nonzero={manifest[out.stem]['nonzero_frac']:.3f}")

(DE / "hirise_rom_manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"\ntotal human labels across tiles: {sum(m['n_human'] for m in manifest.values())}")
