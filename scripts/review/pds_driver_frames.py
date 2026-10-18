"""Does GDAL's PDS driver give the label frame on every product, PSP-era and ESP alike? (2026-10-17)

hirise2ctx's pilot answers (§5, question 4) ask this before porting their Stages 2-3 to open each JP2
through its `.LBL`. For every product whose label and JP2 sit side by side on this laptop, this
opens the product two ways and compares each origin with the label's own numbers:
- through the `.LBL` (GDAL's PDS driver), the route `pds_prep.prepare_scene` takes;
- through the `.JP2` (its GeoJP2 box).

Two v3 products have their JP2 only in hirise2ctx's cache, S2 (ESP_068483_2280) among them. For
those, only the JP2 box is read: a header read, against the label `pilot_images_check.py` fetched.

The label frame is `pds_prep.label_frame_origin`, `(-(SPO + 0.5) * s, (LPO + 0.5) * s)`. Offsets are
written as `tif_minus_label_origin_px` is in `incidence_pairs_label.py`: `((ulx - label_ulx) / s,
(label_uly - uly) / s)`, so +x is east (columns) and +y is south (rows). The CRS parameters are
recorded too (X-04: the box transposes `lat_ts` and `lat_0` on older products).

The PDS driver reads its origin convention from four GDAL config options
(`PDS_SampleProjOffset_Shift` / `_Mult`, `PDS_LineProjOffset_Shift` / `_Mult`). The check re-opens one
product with a non-default shift, to show that the label frame depends on those options staying at
their defaults.

    conda run -n bouldernet python -I scripts/review/pds_driver_frames.py --out data/review/pds_driver_frames.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import rasterio
from rasterio.env import GDALVersion

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from YOLOv8BeyondEarth import pds_prep, presets                             # noqa: E402

ROOTS = [REPO / "data" / "test2_pairs", REPO / "data" / "pds_scenes", REPO / "data" / "mars_hirise" / "raster"]
#: JP2s held only in hirise2ctx's cache (read-only; header reads with GDAL_PAM_ENABLED=NO, so no
#: .aux.xml is written beside them), checked against labels fetched by pilot_images_check.py
JP2_CACHE = Path(r"C:/Users/brian/Documents/PhD/HiRiseToCTXBoulders/v3_data/cache/hirise_jp2")
LABEL_CACHE = REPO / "data" / "pds_labels"


def crs_params(crs):
    d = crs.to_dict() if crs is not None else {}
    return {k: d.get(k) for k in ("proj", "lat_ts", "lat_0", "lon_0", "R")}


def offset_px(ul, label_ul, s):
    return [(ul[0] - label_ul[0]) / s, (label_ul[1] - ul[1]) / s]


def products():
    for root in ROOTS:
        for lbl in sorted(root.rglob("*_RED.LBL")):
            jp2 = lbl.with_name(pds_prep.label_summary(pds_prep.read_label(lbl))["jp2_name"])
            if jp2.exists():
                yield lbl, jp2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "data" / "review" / "pds_driver_frames.json"))
    args = ap.parse_args()
    rows = []
    for lbl, jp2 in products():
        f = pds_prep.read_label(lbl)
        s = pds_prep.label_summary(f)
        scale, label_ul = float(s["map_scale"]), pds_prep.label_frame_origin(s)
        rec = {"product_id": s["product_id"], "product_version": pds_prep._lookup(f, "PRODUCT_VERSION_ID", False),
               "map_scale": scale, "center_latitude": s["center_latitude"], "label_ul_m": list(label_ul)}
        with rasterio.open(lbl) as src:
            rec["pds"] = {"driver": src.driver, "ul_m": [src.transform.c, src.transform.f],
                          "pixel": [src.transform.a, src.transform.e],
                          "minus_label_px": offset_px((src.transform.c, src.transform.f), label_ul, scale),
                          "crs": crs_params(src.crs)}
        with rasterio.open(jp2) as src:
            rec["jp2"] = {"driver": src.driver, "ul_m": [src.transform.c, src.transform.f],
                          "minus_label_px": offset_px((src.transform.c, src.transform.f), label_ul, scale),
                          "crs": crs_params(src.crs)}
        p = rec["pds"]
        rec["pds_is_label_frame"] = (p["driver"] == "PDS" and max(map(abs, p["minus_label_px"])) <= 1e-6
                                     and abs(float(p["crs"]["lat_ts"] or 0) - float(s["center_latitude"])) < 1e-9
                                     and float(p["crs"]["lat_0"] or 0) == 0.0)
        rows.append(rec)
        print(f"{rec['product_id']:22s} v{rec['product_version']} {scale} m/px  PDS {p['driver']} "
              f"d={[round(x, 6) for x in p['minus_label_px']]} lat_ts={p['crs']['lat_ts']} lat_0={p['crs']['lat_0']} | "
              f"JP2 d={[round(x, 3) for x in rec['jp2']['minus_label_px']]} lat_ts={rec['jp2']['crs']['lat_ts']} "
              f"lat_0={rec['jp2']['crs']['lat_0']}  label_frame={rec['pds_is_label_frame']}")
    jp2_only = []
    with rasterio.Env(GDAL_PAM_ENABLED="NO"):
        for jp2 in sorted(JP2_CACHE.glob("*_RED.JP2")):
            lbl = LABEL_CACHE / jp2.with_suffix(".LBL").name
            if not lbl.exists():
                continue
            s = pds_prep.label_summary(pds_prep.read_label(lbl))
            scale, label_ul = float(s["map_scale"]), pds_prep.label_frame_origin(s)
            with rasterio.open(jp2) as src:
                rec = {"product_id": s["product_id"], "map_scale": scale, "center_latitude": s["center_latitude"],
                       "jp2": {"driver": src.driver, "ul_m": [src.transform.c, src.transform.f],
                               "minus_label_px": offset_px((src.transform.c, src.transform.f), label_ul, scale),
                               "crs": crs_params(src.crs)}}
            jp2_only.append(rec)
            print(f"{rec['product_id']:22s} (JP2 only) d={[round(x, 3) for x in rec['jp2']['minus_label_px']]} "
                  f"lat_ts={rec['jp2']['crs']['lat_ts']} lat_0={rec['jp2']['crs']['lat_0']}")
    # the driver's convention is configurable: show that a non-default shift moves the origin
    lbl0 = next(products())[0]
    s0 = pds_prep.label_summary(pds_prep.read_label(lbl0))
    with rasterio.Env(PDS_SampleProjOffset_Shift="-0.5", PDS_LineProjOffset_Shift="-0.5"):
        with rasterio.open(lbl0) as src:
            moved = offset_px((src.transform.c, src.transform.f), pds_prep.label_frame_origin(s0),
                              float(s0["map_scale"]))
    out = {"what": "GDAL's PDS driver vs the label frame, and the JP2 box, on every local product",
           "code": presets.code_identity(), "gdal": str(GDALVersion.runtime()),
           "convention": "minus_label_px = ((ulx - label_ulx) / s, (label_uly - uly) / s): +x east, +y south",
           "products": rows, "jp2_box_only": jp2_only,
           "all_pds_label_frame": all(r["pds_is_label_frame"] for r in rows),
           "config_option_probe": {"product": s0["product_id"],
                                   "options": {"PDS_SampleProjOffset_Shift": "-0.5", "PDS_LineProjOffset_Shift": "-0.5"},
                                   "minus_label_px": moved}}
    Path(args.out).write_text(json.dumps(out, indent=1, default=str))
    print("GDAL", out["gdal"], "| all PDS opens in the label frame:", out["all_pds_label_frame"],
          "| non-default shift moves the origin by", [round(x, 3) for x in moved], "px")


if __name__ == "__main__":
    main()
