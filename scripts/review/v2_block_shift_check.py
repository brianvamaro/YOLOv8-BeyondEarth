"""v2's vClaire export of ESP_045550_2180 (S3) has a displaced block (2026-10-18). Evidence, and its scope.

**How it was found.** The pilot's v2 comparison (`sherlock_pilot_analysis.py --sections v2`) matched
counts and score distributions within 0.1 %. But after the JP2-frame offset, only 72.6 % of vClaire's S3
polygons had one of ours within 1 m, against 100 % on S2.

**This script, per image:**
1. The match rate of vClaire's polygons (score ≥ 0.25, shifted by the measured JP2-frame offset of
   (+1.5, +1.5) px, i.e. into the label frame) to the pilot's catalog, in 512-row bands; and the row
   where it collapses.
2. Below that row, the shift that re-aligns the two. It is a cross-correlation of 2 m centroid-density
   maps.
3. Which catalog sits on the image's boulders. That uses the same polygon-on-feature contrast as
   `sherlock_pilot_analysis.py --sections placement`: |mean DN inside − mean DN in a 1 m ring| per
   4096-row band, against the same polygons moved 50 m east.

**Scope.**
- S2 has a pilot catalog; its vClaire `.shp` survives only above score 0.407.
- ESP_017355_2260 has none, but its label-frame raster (`data/pds_scenes/`) tests vClaire's polygons and
  BoulderNet's own June catalog (old code) directly.

v2's other images cannot be tested here without their rasters. Everything is read-only, and v2's files
are not touched.

    conda run -n bouldernet python -I scripts/review/v2_block_shift_check.py --out data/review/v2_block_shift_check.json
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
sys.path.insert(0, str(REPO / "scripts" / "review"))
sys.path.insert(0, str(REPO / "src"))
from sherlock_pilot_analysis import DELIVERY, VCLAIRE, contrast_by_rows    # noqa: E402
from YOLOv8BeyondEarth import presets                                       # noqa: E402
from YOLOv8BeyondEarth.runstate import file_digest                          # noqa: E402

STEM = "_RED-predictions-ct-010-ss-256-is-1024-ov-020-downscaled-mask-nms"
JUNE = REPO / "data" / "mars_hirise" / "predictions_new" / "ESP_017355_2260_RED-predictions-skimage-mask-nms.gpkg"
LABEL_RASTER_017355 = REPO / "data" / "pds_scenes" / "ESP_017355_2260" / "ESP_017355_2260_RED.tif"
JP2_MINUS_LABEL_PX = (1.5, 1.5)          # these products' JP2 box (pds_driver_frames.json): +x east, +y south


def polygons(path, floor=0.25):
    g = pyogrio.read_dataframe(path, columns=["score"])
    geoms = np.asarray(g.geometry.values, dtype=object)
    keep = ~shapely.is_missing(geoms) & (g["score"].to_numpy() >= floor)
    return geoms[keep]


def to_label_frame(geoms, scale):
    """JP2-frame geometry -> label frame: x - dx*s, y + dy*s (HANDOFF.md §7)."""
    d = np.array([-JP2_MINUS_LABEL_PX[0] * scale, JP2_MINUS_LABEL_PX[1] * scale])
    return shapely.transform(geoms, lambda c: c + d)


def with_pilot(obs, delivery):
    h = json.loads((delivery / "out" / obs / "handoff.json").read_text())
    (lay,) = h["layers"]
    prep = json.loads((delivery / "scenes" / obs / f"{h['product_id']}.prep.json").read_text())
    t, H = prep["output"]["transform"], prep["output"]["height"]
    tif = delivery / "scenes" / obs / h["raster"]["path"]
    ours = polygons(delivery / "out" / obs / lay["layer"])
    shp = VCLAIRE / obs / f"{obs}{STEM}.shp"
    v = to_label_frame(polygons(shp), t[0])
    oc, vc = (shapely.get_coordinates(shapely.centroid(x)) for x in (ours, v))
    hit = np.isfinite(cKDTree(oc).query(vc, distance_upper_bound=1.0)[0])
    rv = (vc[:, 1] - t[5]) / t[4]
    bands = []
    for b0 in range(0, H, 512):
        m = (rv >= b0) & (rv < b0 + 512)
        if m.sum() >= 50:
            bands.append({"rows": [b0, b0 + 512], "n": int(m.sum()), "matched": float(hit[m].mean())})
    transition = next((b["rows"][0] for i, b in enumerate(bands)
                       if b["matched"] < 0.5 and i and bands[i - 1]["matched"] > 0.9), None)
    rec = {"vclaire_shp": str(shp), "vclaire_shp_sha256": file_digest(shp), "n_vclaire": int(len(v)),
           "n_ours": int(len(ours)), "matched_within_1m": float(hit.mean()), "bands_512": bands,
           "transition_row": transition}
    if transition is not None:
        so, sv = (oc[:, 1] - t[5]) / t[4] > transition + 512, rv > transition + 512
        co, cv = (oc[:, 0] - t[2]) / t[0], (vc[:, 0] - t[2]) / t[0]
        ro = (oc[:, 1] - t[5]) / t[4]
        b, x0, y0 = 8, min(co[so].min(), cv[sv].min()), transition + 512
        nx, ny = int((max(co.max(), cv.max()) - x0) / b) + 1, int((H - y0) / b) + 1

        def grid(c, r):
            g = np.zeros((ny, nx))
            np.add.at(g, (((r - y0) / b).astype(int).clip(0, ny - 1), ((c - x0) / b).astype(int).clip(0, nx - 1)), 1)
            return g - g.mean()
        go, gv = grid(co[so], ro[so]), grid(cv[sv], rv[sv])
        xc = np.fft.irfft2(np.fft.rfft2(go) * np.conj(np.fft.rfft2(gv)), s=go.shape)
        k = np.unravel_index(np.argmax(xc), xc.shape)
        dy, dx = (k[0] if k[0] < ny // 2 else k[0] - ny), (k[1] if k[1] < nx // 2 else k[1] - nx)
        nrm = np.sqrt((go ** 2).sum() * (gv ** 2).sum())
        rec["below_transition_shift_ours_minus_vclaire_px"] = [int(dx * b), int(dy * b)]
        rec["xcorr_peak"], rec["xcorr_zero_lag"] = float(xc.max() / nrm), float(xc[0, 0] / nrm)
    rec["contrast_ours"] = contrast_by_rows(tif, t, ours)
    rec["contrast_vclaire"] = contrast_by_rows(tif, t, v)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delivery", default=str(DELIVERY))
    ap.add_argument("--out", default=str(REPO / "data" / "review" / "v2_block_shift_check.json"))
    args = ap.parse_args()
    out = {"what": "a displaced block in v2's vClaire export of ESP_045550_2180: evidence and scope",
           "code": presets.code_identity(), "images": {}}
    for obs in ("ESP_045550_2180", "ESP_068483_2280"):
        r = with_pilot(obs, Path(args.delivery))
        out["images"][obs] = r
        lo = lambda key: min((b["ratio"] for b in r[key] if b["ratio"] is not None), default=None)
        print(f"{obs}: matched {r['matched_within_1m']:.3f}; transition row {r['transition_row']}; shift "
              f"{r.get('below_transition_shift_ours_minus_vclaire_px')} (peak {r.get('xcorr_peak')}); "
              f"min contrast/null ours {lo('contrast_ours'):.2f}, vClaire {lo('contrast_vclaire'):.2f}", flush=True)
    import rasterio
    with rasterio.open(LABEL_RASTER_017355) as r:
        t = list(r.transform)[:6]
    e = {}
    for name, path, shift in (("vclaire", VCLAIRE / "ESP_017355_2260" / f"ESP_017355_2260{STEM}.shp", True),
                              ("june_catalog_old_code", JUNE, True)):
        g = polygons(path)
        g = to_label_frame(g, t[0]) if shift else g
        bands = contrast_by_rows(LABEL_RASTER_017355, t, g, band=8192, n=200)
        e[name] = {"path": str(path), "n": int(len(g)), "bands": bands,
                   "min_ratio": min(b["ratio"] for b in bands if b["ratio"] is not None)}
        print(f"ESP_017355_2260 {name}: min contrast/null over {len(bands)} bands {e[name]['min_ratio']:.2f}", flush=True)
    out["images"]["ESP_017355_2260"] = e
    s3 = out["images"]["ESP_045550_2180"]
    out["summary"] = {
        "s3_transition_row": s3["transition_row"],
        "s3_shift_ours_minus_vclaire_px": s3.get("below_transition_shift_ours_minus_vclaire_px"),
        "s3_vclaire_rows_below_transition_share": float(sum(b["n"] for b in s3["bands_512"]
                                                            if s3["transition_row"] is not None and b["rows"][0] >= s3["transition_row"])
                                                        / s3["n_vclaire"]) if s3["transition_row"] else None}
    Path(args.out).write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
