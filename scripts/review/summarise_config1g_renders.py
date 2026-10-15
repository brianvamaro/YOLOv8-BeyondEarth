"""Provenance for the configuration study's C1G renders (256 arm + SAM2), 2026-10-15.

``scripts/render_rom_configs.py --configs 1g`` re-masks configuration 1's catalog on every ROM crop.
This records what came out: rows in and out per tile (the re-masker never drops a row), how many
fell back to the YOLO mask, the realised SAM logit resolution (5a gate 3 is open), and SAM2's
seconds per row, which is the cost side of the study.

    conda run -n bouldernet python scripts/review/summarise_config1g_renders.py --out data/review/rom_renders_config1g.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import geopandas as gpd                                                    # noqa: E402
from YOLOv8BeyondEarth import presets                                      # noqa: E402

ROM_DIR = REPO / "data" / "rom_label_error"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tiles, ratios = [], []
    for path in sorted(ROM_DIR.glob("*/config1g/*-predictions-256-sam2-nms.gpkg")):
        man = json.loads(presets.manifest_path(path).read_text())
        base = path.parents[1] / man["config"]["base"]
        g, b = gpd.read_file(path), gpd.read_file(base)
        src = g["mask_src"].value_counts().to_dict()
        r = g.loc[g["mask_src"] == "sam2", "logit_px_per_native_px"].dropna().to_numpy()
        ratios.append(r)
        tiles.append({"tile": path.name.split("-predictions-")[0], "obs": path.parents[1].name,
                      "n_base": int(len(b)), "n_out": int(len(g)),
                      "n_sam2": int(src.get("sam2", 0)), "n_yolo_fallback": int(src.get("yolo", 0)),
                      "sam_seconds": man.get("sam_seconds"),
                      "logit_px_per_native_px_median": float(np.median(r)) if len(r) else None})
    allr = np.concatenate(ratios) if ratios else np.zeros(0)
    n_rows = sum(t["n_out"] for t in tiles)
    secs = sum(t["sam_seconds"] or 0.0 for t in tiles)
    out = {"what": "C1G renders: configuration 1's 256 catalog re-masked by SAM2 (remask_windows, "
                   "window 256, max_window_growth 1.0), as configuration 3 re-masks configuration 2",
           "script": "scripts/render_rom_configs.py --configs 1g",
           "n_tiles": len(tiles), "rows_in_equals_rows_out": all(t["n_base"] == t["n_out"]
                                                                 for t in tiles),
           "n_rows": n_rows, "n_sam2": sum(t["n_sam2"] for t in tiles),
           "n_yolo_fallback": sum(t["n_yolo_fallback"] for t in tiles),
           "logit_px_per_native_px": {"min": float(allr.min()), "median": float(np.median(allr)),
                                      "max": float(allr.max()),
                                      "n_at_or_above_1": int((allr >= 1.0).sum())},
           "sam_seconds_total": round(secs, 1),
           "sam_ms_per_row": round(1000 * secs / n_rows, 2) if n_rows else None,
           "code": presets.code_identity(), "tiles": tiles}
    Path(args.out).write_text(json.dumps(out, indent=1, default=str))
    print({k: out[k] for k in ("n_tiles", "rows_in_equals_rows_out", "n_rows", "n_sam2",
                               "n_yolo_fallback", "logit_px_per_native_px", "sam_ms_per_row")})


if __name__ == "__main__":
    main()
