"""The pilot's prep cross-check: does Sherlock's prep give the laptop's pixels? (2026-10-18)

It stands in for the laptop dress rehearsal and the ESP_028537_2270 pre-flight, both skipped by
Brian's call on 2026-10-18 (`docs/sherlock_pilot_plan.md` §5 step 0).

**The idea.** The five Test-2 pairs' JP2s are on the laptop already (`data/test2_pairs/<obs>/`), and they
are byte-identical to the ones the cluster fetched (sha256 checked against each scene's
`fetched.json`). So a laptop `pds_prep` of the same product, compared pixel for pixel with the
prepared raster the cluster delivered, tests the cluster's decoder and conversion (GDAL 3.9.3, pip
wheel) against the laptop's (GDAL 3.10, conda), whose route reproduces the Prieur training rasters
pixel for pixel. The compare is `pds_prep.compare_rasters`: pixels, transform and CRS.

**The products.** Four, to span the variation:
- one PSP (v2.0, 0.25 m/px);
- an old-style ESP at 0.5 m/px;
- a new-style ESP at 0.5 m/px;
- a new-style, mixed-binning ESP at 0.25 m/px.

Laptop preps go to `data/pilot_crosscheck/<obs>/` (gitignored).

    conda run -n bouldernet python -I scripts/review/pilot_prep_crosscheck.py --out data/review/pilot_prep_crosscheck.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from YOLOv8BeyondEarth import pds_prep, presets                             # noqa: E402
from YOLOv8BeyondEarth.runstate import file_digest                          # noqa: E402

DELIVERY = Path(r"C:/Users/brian/Documents/PhD/BoulderNet/data/deliveries/hirise2ctx_v3/pilot-1")
LOCAL = REPO / "data" / "test2_pairs"
WORK = REPO / "data" / "pilot_crosscheck"
PRODUCTS = ("PSP_007492_2265", "ESP_059499_2270", "ESP_071699_2260", "ESP_069088_2270")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", nargs="+", default=list(PRODUCTS))
    ap.add_argument("--delivery", default=str(DELIVERY))
    ap.add_argument("--out", default=str(REPO / "data" / "review" / "pilot_prep_crosscheck.json"))
    args = ap.parse_args()
    rows = []
    for obs in args.products:
        lbl = LOCAL / obs / f"{obs}_RED.LBL"
        jp2 = LOCAL / obs / f"{obs}_RED.JP2"
        cluster_dir = Path(args.delivery) / "scenes" / obs
        cluster_tif = cluster_dir / f"{obs}_RED.tif"
        cluster_prep = json.loads((cluster_dir / f"{obs}_RED.prep.json").read_text())
        fetched = json.loads((cluster_dir / "fetched.json").read_text())
        out_tif = WORK / obs / f"{obs}_RED.tif"
        side = out_tif.with_name(out_tif.stem + ".prep.json")
        out_tif.parent.mkdir(parents=True, exist_ok=True)
        if not (out_tif.exists() and side.exists()):
            pds_prep.prepare_scene(lbl, out_tif, overwrite=True)
        local_prep = json.loads(side.read_text())
        jp2_same = file_digest(jp2) == fetched["jp2"]["sha256"]
        cmp = pds_prep.compare_rasters(out_tif, cluster_tif)
        row = {"obs_id": obs, "jp2_identical": jp2_same,
               "gdal": {"laptop": local_prep["gdal"], "cluster": cluster_prep["gdal"]},
               "identical": cmp["identical"], "n_pixels": cmp.get("n_pixels"), "n_differ": cmp.get("n_differ"),
               "diff_histogram": cmp.get("diff_histogram"), "transform_equal": cmp["transform_equal"],
               "crs_equal": cmp["crs_equal"],
               "dn_histogram_equal": local_prep["dn_counts"]["histogram_nonzero"]
               == {int(k): v for k, v in cluster_prep["dn_counts"]["histogram_nonzero"].items()},
               "frame_check": {"laptop": [local_prep["frame_check"]["dx_px"], local_prep["frame_check"]["dy_px"]],
                               "cluster": [cluster_prep["frame_check"]["dx_px"], cluster_prep["frame_check"]["dy_px"]]}}
        rows.append(row)
        print(f"{obs}: JP2 identical {jp2_same}; {'IDENTICAL' if cmp['identical'] else 'DIFFERENT'} -- "
              f"{cmp.get('n_differ')} of {cmp.get('n_pixels')} px differ, transform {cmp['transform_equal']}, "
              f"CRS {cmp['crs_equal']}, DN histogram {row['dn_histogram_equal']} "
              f"(GDAL laptop {row['gdal']['laptop']}, cluster {row['gdal']['cluster']})", flush=True)
    out = {"what": "the pilot's prep cross-check: laptop pds_prep vs the cluster's delivered prepared raster",
           "code": presets.code_identity(), "delivery": args.delivery, "products": rows,
           "all_identical": all(r["identical"] and r["jp2_identical"] for r in rows)}
    Path(args.out).write_text(json.dumps(out, indent=1, default=str))
    print("all identical:", out["all_identical"])


if __name__ == "__main__":
    main()
