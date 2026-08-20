#!/bin/bash
#SBATCH --job-name=bnet-clip
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=logs/clip-%j.out
#SBATCH --error=logs/clip-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bamaro@stanford.edu
#
# Clip the three HiRISE arm layers to the human-mapped ROM footprint. Submit from ~/BoulderNet.
#
# WHY: two independent reasons, and either alone justifies it.
#
#  1. **Purge insurance.** The arms (4.6 GB) exist only on $SCRATCH, which is purged on an inactivity
#     timer. The SR/512 ratification needs them. Clipping to ROM turns 4.6 GB into megabytes, so the
#     inputs can be preserved locally without moving the full layers — which decouples "keep the data"
#     from "make the decision" and removes the deadline.
#  2. **ROM is the only place the ablation is meaningful.** Humans mapped inside the ROM only, so a
#     detection outside it is not a false positive, it is unmapped ground. Scoring the whole scene
#     against a partial reference would manufacture precision loss out of nothing.
#
# NO GPU. This is a spatial clip.
#   ⚠ If `normal` is rejected, fall back to `--partition=gpu --gres=gpu:1` (proven here).
#
# BEFORE SUBMITTING, stage the ROM footprint (a few KB, all four sidecar files):
#   mkdir -p $SCRATCH/hirise/rom
#   # upload from data_prieur/raw_data/mars/ESP_017355_2260/shp/:
#   #   ESP_017355_2260-ROM.shp .shx .dbf .prj
#
# ⚠ CRS: the Prieur shapefiles must be **set_crs**-overridden to the raster's CRS, never `to_crs`.
# Reprojecting them applies a bogus datum shift and clips to zero rows (memory
# `prieur-crs-override-gotcha`). This script does the override, and asserts a non-empty result so the
# failure mode cannot pass silently.
set -euo pipefail

PYMODULE="${PYMODULE:-python/3.12.1}"
ENV_DIR="${ENV_DIR:-/home/groups/mlapotre/bamaro/envs/bouldernet-infer}"
REPO="${REPO:-$HOME/BoulderNet/YOLOv8-BeyondEarth}"
OUT="${OUT:-$SCRATCH/hirise/out}"
ROM="${ROM:-$SCRATCH/hirise/rom/ESP_017355_2260-ROM.shp}"
DEST="${DEST:-$SCRATCH/hirise/rom_clipped}"

ml "$PYMODULE"
source "$ENV_DIR/bin/activate"
cd "$REPO"

[ -f "$ROM" ] || { echo "MISSING ROM shapefile: $ROM -- see this script's header." >&2; exit 2; }
mkdir -p "$DEST"

OUT="$OUT" ROM="$ROM" DEST="$DEST" python - <<'PY'
import glob
import os
from pathlib import Path

import geopandas as gpd

out, rom_path, dest = Path(os.environ["OUT"]), Path(os.environ["ROM"]), Path(os.environ["DEST"])
layers = sorted(glob.glob(str(out / "*-downscaled-mask-nms.gpkg")))
assert len(layers) == 3, f"expected 3 arm layers, found {len(layers)}"

rom = gpd.read_file(rom_path)
print(f"ROM: {len(rom)} polygon(s), declared crs {rom.crs}")

for src in layers:
    arm = gpd.read_file(src)
    # set_crs override, never to_crs -- see the header. Assert afterwards rather than trusting it.
    r = rom.set_crs(arm.crs, allow_override=True)
    footprint = r.union_all() if hasattr(r, "union_all") else r.unary_union
    keep = arm[arm.geometry.representative_point().within(footprint)]
    assert len(keep) > 0, (f"clip produced 0 rows for {Path(src).name} -- the CRS override is wrong or "
                           f"the ROM does not overlap this raster")
    target = dest / Path(src).name.replace("-downscaled-mask-nms", "-rom-downscaled-mask-nms")
    keep.to_file(target, driver="GPKG")
    mb = target.stat().st_size / 1e6
    print(f"{Path(src).name[:58]:58} {len(arm):>9,} -> {len(keep):>8,}  ({mb:7.1f} MB)")
    # carry the arm's manifest across so provenance survives the clip
    man = Path(str(src).replace(".gpkg", "-manifest.json"))
    if man.exists():
        (dest / man.name).write_text(man.read_text())
PY

echo
echo "Clipped layers + manifests in: $DEST"
du -sh "$DEST"
echo
echo "Pack for download:"
echo "  tar czf ~/hirise_rom_arms.tgz -C $(dirname "$DEST") $(basename "$DEST")"
