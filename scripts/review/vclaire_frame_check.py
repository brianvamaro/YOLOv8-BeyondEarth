"""Which georeferencing frame did v2's vClaire run use: the PDS label's, or the JP2's? (2026-10-17)

hirise2ctx's pilot answers (§5, question 6) ask this for v2's record. One PDS product opens in two
frames (X-07): through its ``.LBL`` (GDAL's PDS driver, the frame of the Prieur rasters and the human
labels) or through its ``.JP2`` (the GeoJP2 box), whose origin is 1.5 px east and 1.5 px south of the
label's on older products such as ESP_017355_2260. At 0.5 m/px that is 0.75 m on each axis.

The test is a matched-centroid offset between vClaire's export of ESP_017355_2260 and a catalog of the
same image known to be in the JP2 frame: the June run in ``data/mars_hirise/predictions_new/``, made on
a raster with the JP2's geotransform (``data/mars_hirise/README.md``). Same detector, same slicing, so
matched boulders should coincide.
- vClaire in the JP2 frame: offset (0, 0).
- vClaire in the label frame: vClaire minus ours = (-0.75, +0.75) m (west and north of ours).

Matching: each catalog filtered to score >= 0.25 and equivalent diameter >= 1.5 m, then mutual nearest
centroids within 3 m whose areas agree within 25%. vClaire's ``.shp`` is byte-truncated (hirise2ctx's
R23), so only its surviving geometries take part; their scores are the export's highest.

    conda run -n bouldernet python -I scripts/review/vclaire_frame_check.py --out data/review/vclaire_frame_check.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyogrio
import shapely
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from YOLOv8BeyondEarth import presets                                       # noqa: E402
from YOLOv8BeyondEarth.runstate import file_digest                          # noqa: E402

STEM = "ESP_017355_2260_RED-predictions-ct-010-ss-256-is-1024-ov-020-downscaled-mask-nms"
VCLAIRE = Path(r"C:/Users/brian/Documents/PhD/HiRiseToCTXBoulders/hirise_40_vClaire/ESP_017355_2260") / f"{STEM}.shp"
OURS = REPO / "data" / "mars_hirise" / "predictions_new" / "ESP_017355_2260_RED-predictions-skimage-mask-nms.gpkg"
SCALE = 0.5                     # m/px for ESP_017355_2260
JP2_MINUS_LABEL_PX = (1.5, 1.5)  # tif_minus_label_origin_px of a JP2-frame raster of this product (+x east, +y south)


def load(path):
    g = pyogrio.read_dataframe(path, columns=["score"])
    geoms = np.asarray(g.geometry.values, dtype=object)
    ok = ~shapely.is_missing(geoms)
    geoms, score = geoms[ok], g["score"].to_numpy()[ok]
    area = shapely.area(geoms)
    cen = shapely.get_coordinates(shapely.centroid(geoms))
    keep = (score >= 0.25) & (2.0 * np.sqrt(area / np.pi) >= 1.5)
    return dict(n_rows=int(len(g)), n_geom=int(ok.sum()), n_kept=int(keep.sum()), cen=cen[keep],
                area=area[keep], score=score[keep], bounds=pyogrio.read_info(path)["total_bounds"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vclaire", default=str(VCLAIRE))
    ap.add_argument("--ours", default=str(OURS))
    ap.add_argument("--radius-m", type=float, default=3.0)
    ap.add_argument("--out", default=str(REPO / "data" / "review" / "vclaire_frame_check.json"))
    args = ap.parse_args()
    V, O = load(args.vclaire), load(args.ours)
    tree_o, tree_v = cKDTree(O["cen"]), cKDTree(V["cen"])
    d, j = tree_o.query(V["cen"], distance_upper_bound=args.radius_m)
    hit = np.isfinite(d)
    i = np.flatnonzero(hit)
    j = j[hit]
    _, back = tree_v.query(O["cen"][j])
    mutual = back == i
    i, j = i[mutual], j[mutual]
    ratio = V["area"][i] / O["area"][j]
    good = (ratio > 0.8) & (ratio < 1.25)
    i, j = i[good], j[good]
    off = V["cen"][i] - O["cen"][j]                     # vClaire minus ours, metres (x east, y north)
    label_frame_expect = [-JP2_MINUS_LABEL_PX[0] * SCALE, JP2_MINUS_LABEL_PX[1] * SCALE]
    res = {
        "what": "matched-centroid offset, vClaire's ESP_017355_2260 export minus a JP2-frame catalog of "
                "the same image",
        "code": presets.code_identity(),
        "inputs": {"vclaire_shp": {"path": args.vclaire, "sha256": file_digest(args.vclaire),
                                   "prj": Path(args.vclaire).with_suffix(".prj").read_text().strip()},
                   "ours_gpkg": {"path": args.ours, "sha256": file_digest(args.ours),
                                 "crs": pyogrio.read_info(args.ours)["crs"]}},
        "filters": {"score_min": 0.25, "equivalent_diameter_min_m": 1.5, "radius_m": args.radius_m,
                    "area_ratio": [0.8, 1.25], "matching": "mutual nearest centroid"},
        "vclaire": {k: V[k] for k in ("n_rows", "n_geom", "n_kept")} | {"total_bounds": list(V["bounds"])},
        "ours": {k: O[k] for k in ("n_rows", "n_geom", "n_kept")} | {"total_bounds": list(O["bounds"])},
        "n_matched": int(len(off)),
        "offset_m": {"median_dx": float(np.median(off[:, 0])), "median_dy": float(np.median(off[:, 1])),
                     "mean_dx": float(off[:, 0].mean()), "mean_dy": float(off[:, 1].mean()),
                     "p05_p95_dx": [float(x) for x in np.percentile(off[:, 0], [5, 95])],
                     "p05_p95_dy": [float(x) for x in np.percentile(off[:, 1], [5, 95])],
                     "share_within_0.05m": float((np.hypot(*off.T) < 0.05).mean()),
                     "share_within_0.05m_of_label_frame_expect": float(
                         (np.hypot(off[:, 0] - label_frame_expect[0], off[:, 1] - label_frame_expect[1])
                          < 0.05).mean())},
        "expected_m": {"jp2_frame": [0.0, 0.0], "label_frame": label_frame_expect},
    }
    med = np.array([res["offset_m"]["median_dx"], res["offset_m"]["median_dy"]])
    res["verdict"] = ("jp2_frame" if np.hypot(*med) < 0.2 else
                      "label_frame" if np.hypot(*(med - label_frame_expect)) < 0.2 else "neither")
    Path(args.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res[k] for k in ("vclaire", "ours", "n_matched", "offset_m", "verdict")},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
