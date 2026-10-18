"""The Sherlock pilot's images, checked before anything runs (2026-10-17).

Two read-only checks for `docs/sherlock_pilot_plan.md`, written to one JSON.

1. **Index check.** Every pilot image is looked up in v3's frozen PDS index (hirise2ctx's
   `RDRCUMINDEX_2026-10-06.TAB`, sha256 `50c8eb38...`). That covers S1-S3, the dress-rehearsal scene,
   the five Test-2 pairs and the five same-ground gap sets Brian added on 2026-10-17. For each image
   it reads the product id, the JP2 and LBL URLs, the viewing angles, Ls, the date and Gpx (lines x
   samples, fill included). The gap sets' overlaps are recomputed the way hirise2ctx's answers (§3.1)
   say they made them: the convex hull of the index's four corners, with geodesic areas on the Mars 2015
   sphere (R = 3,396,190 m). The answers' table is compared value by value. Each gap-set image is also
   checked against v3's manifest and candidate pool, v2's images (vClaire 40, priority 10) and the ROM
   images.
2. **CCD binning from the RDR labels** (pilot answers §5, question 2). `MRO:BINNING` and `MRO:CCD_FLAG`
   come from each product's `.LBL`, for v3's 62 scenes and every pilot image. Labels already on disk
   (`data/test2_pairs/`, `data/pds_scenes/`) are read in place. The rest are fetched whole into
   `data/pds_labels/` (gitignored), with sha256 recorded. A 0.25 m/px product is **mixed** when any RED
   CCD that is on is binned 2x or more. Those strips are lower-resolution data resampled onto the
   0.25 m grid. The label names the CCDs (RED0-RED9), not where their strips fall in the raster.

    conda run -n bouldernet python -I scripts/review/pilot_images_check.py --out data/review/pilot_images_check.json
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import shapely
from pyproj import Geod

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from YOLOv8BeyondEarth import pds_prep, presets                             # noqa: E402
from YOLOv8BeyondEarth.runstate import file_digest                          # noqa: E402

H = Path(r"C:/Users/brian/Documents/PhD/HiRiseToCTXBoulders")
INDEX = H / "v3_data" / "cache" / "pds_index" / "RDRCUMINDEX_2026-10-06.TAB"
SELECTION = H / "v3_data" / "dataset" / "selection"
PDS = "https://hirise-pds.lpl.arizona.edu/PDS/"

SCENES = {"S1": "PSP_009116_2240", "S2": "ESP_068483_2280", "S3": "ESP_045550_2180"}
REHEARSAL = {"rehearsal": "ESP_046328_2180", "rehearsal_withdrawn": "ESP_043417_2255"}
PAIRS5 = {1: ("ESP_071923_2260", "ESP_071699_2260"), 2: ("ESP_076235_2260", "ESP_055833_2260"),
          3: ("ESP_069088_2270", "ESP_067214_2270"), 4: ("ESP_059499_2270", "ESP_055714_2270"),
          5: ("PSP_007703_2265", "PSP_007492_2265")}
#: set -> (reference, gap images); each gap image is scored against the reference as its own pair
GAP_SETS = {"G1": ("ESP_059475_2245", ("ESP_058974_2245",)),
            "G2": ("ESP_036978_2150", ("ESP_037545_2150", "ESP_029818_2150")),   # the second is G2+
            "U1": ("ESP_041871_2205", ("ESP_037796_2205",)),
            "U2": ("ESP_070475_2225", ("ESP_049730_2225", "ESP_076224_2225"))}
#: hirise2ctx's answers §3.1, as published: (inc, em, Ls, date, Gpx); overlaps (km2, % of the smaller)
ANSWERS_TABLE = {
    "ESP_058974_2245": (56.2, 4.1, 346, "2019-02-24", 1.06), "ESP_059475_2245": (49.1, 12.6, 6, "2019-04-04", 1.18),
    "ESP_037545_2150": (58.4, 5.5, 170, "2014-07-31", 1.03), "ESP_036978_2150": (51.9, 0.4, 147, "2014-06-16", 1.19),
    "ESP_029818_2150": (69.5, 5.8, 220, "2012-12-05", 1.02), "ESP_037796_2205": (64.4, 5.2, 181, "2014-08-19", 1.16),
    "ESP_041871_2205": (49.2, 8.1, 7, "2015-07-03", 0.81), "ESP_049730_2225": (60.5, 4.6, 328, "2017-03-06", 1.88),
    "ESP_070475_2225": (46.6, 0.0, 83, "2021-08-08", 1.32), "ESP_076224_2225": (62.1, 4.8, 330, "2022-10-30", 1.45)}
ANSWERS_OVERLAP = {"ESP_058974_2245": (46.9, 94), "ESP_037545_2150": (47.4, 99), "ESP_037796_2205": (41.1, 99),
                   "ESP_049730_2225": (57.8, 100), "ESP_076224_2225": (51.5, 89)}
ANSWERS_SHARED = {"G2": 94, "U2": 51.4}   # G2+ covers 94% of G2's overlap; U2's three share 51.4 km2
ROM = ("ESP_017355_2260", "ESP_025650_2165", "ESP_028537_2270", "ESP_037018_2170")
LOCAL_LABELS = [REPO / "data" / "test2_pairs", REPO / "data" / "pds_scenes"]
GEOD = Geod(a=3396190.0, f=0.0)


def read_index(path, want):
    """RED rows of the index for the wanted observation ids, sliced by the label's byte layout."""
    cols, cur = [], {}
    for line in Path(path).with_suffix(".LBL").read_text().splitlines():
        m = re.match(r"\s*(NAME|START_BYTE|BYTES)\s*=\s*(\S+)", line)
        if m:
            cur[m.group(1)] = m.group(2)
            if m.group(1) == "BYTES" and {"NAME", "START_BYTE"} <= set(cur):
                cols.append((cur["NAME"], int(cur["START_BYTE"]), int(cur["BYTES"])))
                cur = {}
    rows = {}
    with open(path, "rb") as fh:
        for raw in fh:
            line = raw.decode("ascii", "replace")
            rec = {n: line[s - 1:s - 1 + b].strip().strip('"').strip() for n, s, b in cols}
            if rec["PRODUCT_ID"].endswith("_RED") and rec["OBSERVATION_ID"] in want:
                rows.setdefault(rec["OBSERVATION_ID"], []).append(rec)
    return rows


def footprint(rec):
    pts = [(((float(rec[f"CORNER{i}_LONGITUDE"]) + 180) % 360) - 180, float(rec[f"CORNER{i}_LATITUDE"]))
           for i in range(1, 5)]
    return shapely.convex_hull(shapely.multipoints(pts))


def km2(g):
    return abs(GEOD.geometry_area_perimeter(g)[0]) / 1e6


def index_entry(recs):
    r = recs[0]
    spec = r["FILE_NAME_SPECIFICATION"]
    return {"product_id": r["PRODUCT_ID"], "version": r["PRODUCT_VERSION_ID"], "n_index_rows": len(recs),
            "jp2_url": PDS + spec, "lbl_url": PDS + re.sub(r"\.JP2$", ".LBL", spec),
            "incidence": float(r["INCIDENCE_ANGLE"]), "emission": float(r["EMISSION_ANGLE"]),
            "ls": float(r["SOLAR_LONGITUDE"]), "date": r["START_TIME"][:10],
            "map_scale": float(r["MAP_SCALE"]), "lines": int(r["IMAGE_LINES"]),
            "samples": int(r["LINE_SAMPLES"]), "gpx": int(r["IMAGE_LINES"]) * int(r["LINE_SAMPLES"]) / 1e9,
            "rationale": r["RATIONALE_DESC"], "footprint_km2": km2(footprint(r))}


def _ints(s):
    return [int(x) for x in re.findall(r"-?\d+", str(s or ""))]


def binning(obs, label_dir, fetch):
    """CCD binning from the product's label: a local copy if there is one, else fetched."""
    name = f"{obs}_RED.LBL"
    local = next((p / obs / name for p in LOCAL_LABELS if (p / obs / name).exists()), None)
    if local is None:
        dest = Path(label_dir) / name
        if not dest.exists():
            if not fetch:
                return {"label": None}
            pds_prep.fetch(pds_prep.rdr_urls(obs)[0], dest)
        local = dest
    f = pds_prep.read_label(local)
    s = pds_prep.label_summary(f)
    bins, flags = _ints(s["binning"]), re.findall(r"ON|OFF", str(pds_prep._lookup(f, "MRO:CCD_FLAG", False)))
    red = {f"RED{k}": bins[k] for k in range(10) if k < len(flags) and flags[k] == "ON" and bins[k] > 0}
    binned = [c for c, b in red.items() if b > 1]
    scale = float(s["map_scale"])
    return {"label": {"path": str(local.relative_to(REPO)) if local.is_relative_to(REPO) else str(local),
                      "sha256": file_digest(local)},
            "product_version": pds_prep._lookup(f, "PRODUCT_VERSION_ID", False),
            "map_scale": scale, "center_latitude": s["center_latitude"],
            "label_frame_ul_m": list(pds_prep.label_frame_origin(s)),
            "red_on": list(red), "red_binning": red, "red_binned": binned,
            "mixed": bool(scale <= 0.25 + 1e-9 and binned),
            "uniform_binning": sorted(set(red.values())) if len(set(red.values())) == 1 else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(INDEX))
    ap.add_argument("--label-dir", default=str(REPO / "data" / "pds_labels"))
    ap.add_argument("--no-fetch", action="store_true", help="read only labels already on disk")
    ap.add_argument("--out", default=str(REPO / "data" / "review" / "pilot_images_check.json"))
    args = ap.parse_args()

    manifest = {r["obs_id"]: r for r in csv.DictReader(open(SELECTION / "manifest_v3.csv", encoding="utf-8"))}
    pool = {r["obs_id"] for r in csv.DictReader(open(SELECTION / "pool.csv", encoding="utf-8"))}
    v2 = {p.name for d in ("hirise_40_vClaire", "hirise_priority10_detections") for p in (H / d).iterdir()
          if p.is_dir()}
    gap_imgs = [o for ref, gs in GAP_SETS.values() for o in (ref, *gs)]
    pair_imgs = [o for p in PAIRS5.values() for o in p]
    pilot = list(SCENES.values()) + list(REHEARSAL.values()) + pair_imgs + gap_imgs
    rows = read_index(args.index, set(pilot) | set(manifest) | v2 | set(ROM))
    fp = {o: footprint(r[0]) for o, r in rows.items()}

    images = {o: index_entry(rows[o]) for o in pilot}
    url_check = {o: (images[o]["lbl_url"], images[o]["jp2_url"]) == pds_prep.rdr_urls(o) for o in pilot}
    vs_answers = {}
    for o, (inc, em, ls, date, gpx) in ANSWERS_TABLE.items():
        x = images[o]
        vs_answers[o] = {"incidence": round(x["incidence"], 1) == inc, "emission": round(x["emission"], 1) == em,
                         "ls": round(x["ls"]) % 360 == ls, "date": x["date"] == date,
                         "gpx": round(x["gpx"], 2) == gpx}
    sets = {}
    for name, (ref, gaps) in GAP_SETS.items():
        s = {"reference": ref, "pairs": {}}
        for g in gaps:
            ov = shapely.intersection(fp[ref], fp[g])
            a, small = km2(ov), min(km2(fp[ref]), km2(fp[g]))
            s["pairs"][g] = {"overlap_km2": a, "pct_of_smaller": 100 * a / small,
                             "centre_lat": ov.centroid.y, "centre_lon_w": -ov.centroid.x,
                             "answers": ANSWERS_OVERLAP.get(g)}
        if len(gaps) == 2:
            first = shapely.intersection(fp[ref], fp[gaps[0]])
            tri = shapely.intersection(first, fp[gaps[1]])
            s["shared_by_all_three_km2"] = km2(tri)
            s["shared_pct_of_first_overlap"] = 100 * km2(tri) / km2(first)
            s["answers"] = ANSWERS_SHARED[name]
        sets[name] = s
    clear = {}
    for o in gap_imgs:
        hits = [x for x in sorted(set(manifest) | v2 | set(ROM)) if x != o and x in fp
                and shapely.intersects(fp[o], fp[x])]
        clear[o] = {"in_v3_manifest": o in manifest, "in_v3_pool": o in pool, "in_v2": o in v2,
                    "overlaps_v3_v2_or_rom": hits}
    totals = {"S1-S3": sum(images[o]["gpx"] for o in SCENES.values()),
              "pairs5": sum(images[o]["gpx"] for o in pair_imgs),
              "gap_sets": sum(images[o]["gpx"] for o in gap_imgs)}
    totals["pilot_on_sherlock"] = sum(totals.values())
    totals["rehearsal_laptop"] = images[REHEARSAL["rehearsal"]]["gpx"]

    bins = {o: binning(o, args.label_dir, not args.no_fetch) for o in sorted(set(manifest) | set(pilot))}
    v3_mixed = sorted(o for o in manifest if bins[o].get("mixed"))
    out = {"what": "the Sherlock pilot's images: index check and CCD binning (read-only)",
           "code": presets.code_identity(),
           "index": {"path": str(args.index), "sha256": file_digest(args.index)},
           "manifest_v3_sha256": file_digest(SELECTION / "manifest_v3.csv"),
           "roles": {**SCENES, **REHEARSAL, **{f"pair{k}": list(v) for k, v in PAIRS5.items()},
                     **{k: {"reference": r, "gap": list(g)} for k, (r, g) in GAP_SETS.items()}},
           "images": images, "urls_match_pds_prep_rdr_urls": url_check,
           "gap_table_vs_answers": vs_answers, "gap_sets": sets, "gap_images_clear_of": clear,
           "gpx_totals": totals, "binning": bins,
           "v3_summary": {"n": len(manifest),
                          "mixed": [{"obs_id": o, "role": manifest[o]["role"], "gpx": float(manifest[o]["gpx"]),
                                     "red_binned": bins[o]["red_binned"], "red_on": bins[o]["red_on"]}
                                    for o in v3_mixed],
                          "all_bin1": sorted(o for o in manifest if bins[o].get("uniform_binning") == [1]),
                          "no_label": sorted(o for o in manifest if not bins[o].get("label"))}}
    Path(args.out).write_text(json.dumps(out, indent=1, default=str))

    bad = {o: [k for k, v in c.items() if not v] for o, c in vs_answers.items() if not all(c.values())}
    print("answers' table mismatches:", bad or "none")
    print("URLs match rdr_urls:", all(url_check.values()))
    for n, s in sets.items():
        for g, v in s["pairs"].items():
            print(f"{n} {g}: {v['overlap_km2']:.2f} km2 ({v['pct_of_smaller']:.1f}%), answers {v['answers']}")
        if "shared_by_all_three_km2" in s:
            print(f"{n} shared {s['shared_by_all_three_km2']:.2f} km2 ({s['shared_pct_of_first_overlap']:.1f}% of "
                  f"the first overlap), answers {s['answers']}")
    print("gap images clear:", all(not c["in_v3_manifest"] and c["in_v3_pool"] and not c["in_v2"]
                                   and not c["overlaps_v3_v2_or_rom"] for c in clear.values()))
    print("Gpx:", {k: round(v, 3) for k, v in totals.items()})
    for o in pilot:
        b = bins[o]
        print(f"  {o} {images[o]['map_scale']} m/px v{b.get('product_version')} on={len(b.get('red_on', []))} "
              f"binned={b.get('red_binned')} mixed={b.get('mixed')}")
    print(f"v3: {len(v3_mixed)} of {len(manifest)} mixed; no label: {out['v3_summary']['no_label']}")
    for m in out["v3_summary"]["mixed"]:
        print(f"  {m['obs_id']} {m['role']:8s} {m['gpx']:.2f} Gpx binned {m['red_binned']} of {m['red_on']}")


if __name__ == "__main__":
    main()
