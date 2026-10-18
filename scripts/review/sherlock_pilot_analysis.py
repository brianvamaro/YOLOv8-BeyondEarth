"""The Sherlock pilot's analysis (`docs/sherlock_pilot_plan.md` §8): one tracked script, one JSON.

The pilot ran release `pilot-1` (`dfcaa26`) on Sherlock on 2026-10-18, over 23 images. It ran
configuration 1 at conf 0.10, with escalation. The hand-offs were pulled into the stable delivery root
(`HANDOFF.md` §1). Each section below reads only that delivery, the manifests and the tracked `sacct`
record, and writes its own key in the output JSON. `--sections` re-runs some sections and keeps the
others.

- **run**:
  - the delivery re-verified scene by scene: `HANDOFF.md` §8's refusals, plus every listed file's
    sha256, plus the prepared raster's bytes against `handoff.json`;
  - the cost and RAM table (§8.A): the delivered attempt's arm wall time, the job's elapsed time
    from `sacct`, peak RSS, escalation attempts, slices and fill skipped, and prep conversion speed;
  - two descriptive fits: GPU-h per Gpx against detections per Gpx, and peak RSS against detections.
- **kept** (§8.B–C):
  - per scene, the share of ≥ 1.41 m detections kept at each floor relative to 0.10, by count and by
    area, by size in metres and in image pixels, with score histograms;
  - for the emission reading, S2's kept share at 0.40 by effective native pixel size, D / (s · k) with
    k = 1, 1/cos e and 1/cos² e, against the low-emission scenes' spread (emission ≤ 13°).

  The size is notebook 35's equivalent-circle diameter on the native-frame polygon, with its
  1.4105 m floor.
- **v2** (§8.E): S2 and S3 against v2's vClaire exports (read-only).
  - counts at ≥ 0.25 and ≥ 0.40, from the `.dbf`, with the contract-1 rows overwritten to a literal
    0.10 (`isin_slice = 0`) held apart;
  - score quantiles and the KS distance;
  - where geometry survives, the matched-centroid offset, which should be the JP2 box's (+1.5, +1.5)
    px, i.e. vClaire 0.375 m east and 0.375 m south of ours. After that shift, the matched shares and
    the matched area ratio.

  vClaire's coordinates are compared numerically, never reprojected: its `.prj` carries the JP2
  box's mislabelled CRS (SP1 = 0).
- **steps** (§8.D): detection density across S2's CCD-channel calibration steps. The steps are found
  in the prepared raster itself, not from instrument geometry.
  - **Axes.** The swath's long axis is the principal axis of the valid-data mask, which gives each
    pixel an along-track t and an across-track u.
  - **Finding the steps.** The mean DN either side of each u, in 1 km along-track segments. A step is
    a jump of at least 0.5 DN with the same sign in ≥ 85 % of ≥ 10 segments; terrain edges do not
    recur like that.
    - S2 has no step anywhere near 1.5 DN in the prepared 8-bit raster.
    - Its persistent ones are 0.5–1 DN, i.e. 2–4 DN in 10 bits: visible in a stretched browse image,
      small in what the detector sees.
  - **The comparison.** Detection density in 128 m bands either side of each step (4 m guard), by
    floor and size, with a block bootstrap over 1 km along-track blocks.
  - **The null.** The same at 40 random flat positions (|median jump| < 0.2 DN), at least 150 m from
    any step.
- **placement**: do the delivered polygons sit on image features in every band of rows? This catches
  a displaced block, which is how v2's export of S3 fails (`v2_block_shift_check.py`).
  - **The test.** Per 4096-row band, the median |mean DN inside − mean DN in a 1 m ring| over 300
    polygons (score ≥ 0.25), against the same polygons moved 50 m east.
  - **Why it matters for this release.** The memory fix's windowed reads have never had a full-scene
    check other than this one. S1 is 137,876 rows tall.

Results are reported as measured. Nothing here applies a score floor to the delivered catalogs, or
changes what a label is.

    conda run -n bouldernet python -I scripts/review/sherlock_pilot_analysis.py --sections run
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from YOLOv8BeyondEarth import pds_prep, presets                             # noqa: E402
from YOLOv8BeyondEarth.runstate import file_digest                          # noqa: E402

DELIVERY = Path(r"C:/Users/brian/Documents/PhD/BoulderNet/data/deliveries/hirise2ctx_v3/pilot-1")
MANIFESTS = {"scenes": REPO / "sherlock" / "manifests" / "pilot_scenes.csv",
             "pairs": REPO / "sherlock" / "manifests" / "pilot_pairs.csv"}
SACCT = REPO / "data" / "review" / "pilot_sacct_2026-10-18.json"
#: Slurm job id -> (stage, manifest) as submitted on 2026-10-18; array task N = manifest row N
JOBS = {"48319948": ("prep", "scenes"), "48319950": ("prep", "pairs"),
        "48319951": ("detect", "scenes"), "48319952": ("detect", "scenes"),
        "48319953": ("detect", "pairs")}
SCENE_SETS = {"PSP_009116_2240": "S1", "ESP_068483_2280": "S2", "ESP_045550_2180": "S3"}
RELEASE_SHA = "dfcaa26acc233923cbeed541e23421e943017f82"
FRAME = "PDS label via GDAL's PDS driver (not the JP2's GeoJP2 box)"


def manifest_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def elapsed_s(text):
    parts = [int(x) for x in text.replace("-", ":").split(":")]
    while len(parts) < 4:
        parts.insert(0, 0)
    d, h, m, s = parts
    return ((d * 24 + h) * 60 + m) * 60 + s


def scene_index():
    """ObsId -> {set, reference, rows}, from the two manifests."""
    idx = {}
    for key, path in MANIFESTS.items():
        for i, r in enumerate(manifest_rows(path)):
            obs = r["ObsId"]
            idx[obs] = {"manifest": key, "row": i,
                        "set": SCENE_SETS.get(obs) or r.get("Set") or "",
                        "reference": r.get("Reference") or ""}
    return idx


def sacct_times():
    """ObsId -> {prep_job_s, detect_job_s, states}, from the tracked sacct record."""
    rec = json.loads(SACCT.read_text())
    rows = {k: manifest_rows(p) for k, p in MANIFESTS.items()}
    out = {}
    for r in rec["rows"]:
        job, task = r["job_task"].split("_")
        stage, man = JOBS[job]
        obs = rows[man][int(task)]["ObsId"]
        o = out.setdefault(obs, {"states": {}})
        o[f"{stage}_job_s"] = elapsed_s(r["elapsed"])
        o["states"][stage] = f"{r['state']} {r['exit_code']}"
    return out


def section_run(delivery):
    idx, times = scene_index(), sacct_times()
    scenes = []
    for d in sorted((delivery / "out").iterdir()):
        obs, checks = d.name, {}
        h = json.loads((d / "handoff.json").read_text())
        checks["handoff_ok"] = bool(h["ok"]) and not h["problems"]
        checks["stages_config1"] = h["stages"] == ["arm:256"]
        checks["nothing_not_delivered"] = not h["not_delivered"]
        sums = dict(reversed(line.split(None, 1)) for line in (d / "SHA256SUMS").read_text().splitlines()
                    if line.strip())
        sums = {k.strip(): v for k, v in sums.items()}
        checks["sha256sums_pass"] = all(file_digest(d / name) == digest for name, digest in sums.items())
        verified = datetime.fromisoformat(h["verified_utc"])
        stamps = [r[k] for line in (d / "run_ledger.jsonl").read_text().splitlines() if line.strip()
                  for r in [json.loads(line)] for k in ("started_utc", "finished_utc") if r.get(k)]
        checks["not_verify_stale"] = not any(datetime.fromisoformat(s) > verified for s in stamps)
        raster = h["raster"] or {}
        checks["label_frame"] = raster.get("frame") == FRAME
        (lay,) = h["layers"]
        man = json.loads((d / lay["manifest"]).read_text())
        cfg, code = man["config"], man["code"]
        checks["conf_0.10"] = cfg["confidence_threshold"] == 0.1
        checks["contract_3"] = cfg.get("inference_contract_version") == 3
        checks["untruncated"] = man.get("truncation") == {}
        checks["max_det_matches"] = cfg["max_det"] == lay["max_det"]
        checks["release_named"] = (code.get("git_sha") == RELEASE_SHA and code.get("git_sha_from") == "CODE_STAMP.json"
                                   and code.get("stamp_matches_source") is True)
        checks["ran_on_prepared_raster"] = (man.get("inputs") or {}).get("raster", {}).get("sha256") == raster.get("sha256")
        sd = delivery / "scenes" / obs
        prep = json.loads((sd / f"{h['product_id']}.prep.json").read_text())
        fc = prep["frame_check"]
        checks["frame_check_zero"] = max(abs(fc["dx_px"]), abs(fc["dy_px"])) <= 1e-6
        checks["raster_bytes_intact"] = file_digest(sd / raster["path"]) == raster.get("sha256") == prep["output"]["sha256"]
        checks["label_and_fetch_record_present"] = (sd / f"{h['product_id']}.LBL").exists() and (sd / "fetched.json").exists()
        s = pds_prep.label_summary(pds_prep.read_label(sd / f"{h['product_id']}.LBL"))
        f = pds_prep.read_label(sd / f"{h['product_id']}.LBL")
        gpx = prep["output"]["width"] * prep["output"]["height"] / 1e9
        att = lay["attempts"]
        t = times.get(obs, {})
        scenes.append({
            "obs_id": obs, **idx.get(obs, {}), "gpx": gpx, "map_scale": float(s["map_scale"]),
            "incidence": float(s["incidence_angle"]), "emission": float(pds_prep._lookup(f, "EMISSION_ANGLE")),
            "n_features": lay["n_features"], "detections_per_gpx": lay["n_features"] / gpx,
            "max_det": lay["max_det"], "batch_size": lay["batch_size"], "n_attempts": len(att),
            "stopped": [a.get("stopped") for a in att if a.get("stopped")],
            "arm_wall_s": att[-1].get("wall_seconds"), "arm_h_per_gpx": att[-1].get("wall_seconds") / 3600 / gpx,
            "detect_job_s": t.get("detect_job_s"), "prep_job_s": t.get("prep_job_s"), "sacct": t.get("states"),
            "prep_convert_s": prep.get("wall_seconds"),
            "prep_convert_mpx_s": gpx * 1e3 / prep["wall_seconds"] if prep.get("wall_seconds") else None,
            "peak_rss_gib": (lay.get("peak_rss_bytes") or 0) / 2 ** 30,
            "n_slices": lay.get("n_slices"), "n_slices_skipped_fill": lay.get("n_slices_skipped_fill"),
            "gdal": prep.get("gdal"), "torch": lay["code"].get("torch"), "ultralytics": lay["code"].get("ultralytics"),
            "high_saturation_px": prep["dn_counts"]["high_saturation_dn_1022_1023"],
            "checks": checks, "all_checks_pass": all(checks.values())})
        print(f"{obs} {scenes[-1]['set']:4s} {gpx:5.2f} Gpx {lay['n_features']:>9,d} det, arm "
              f"{att[-1].get('wall_seconds')/60:5.1f} min, RSS {scenes[-1]['peak_rss_gib']:5.2f} GiB, "
              f"{len(att)} attempt(s); checks {'pass' if scenes[-1]['all_checks_pass'] else [k for k, v in checks.items() if not v]}",
              flush=True)
    x = np.array([r["detections_per_gpx"] / 1e6 for r in scenes])
    y = np.array([r["arm_h_per_gpx"] for r in scenes])
    b, a = np.polyfit(x, y, 1)
    n = np.array([r["n_features"] / 1e6 for r in scenes])
    m = np.array([r["peak_rss_gib"] for r in scenes])
    d_, c_ = np.polyfit(n, m, 1)
    arm_total = sum(r["arm_wall_s"] for r in scenes)
    job_total = sum(r["detect_job_s"] or 0 for r in scenes)
    gpx_total = sum(r["gpx"] for r in scenes)
    totals = {"n_scenes": len(scenes), "all_checks_pass": all(r["all_checks_pass"] for r in scenes),
              "gpx": gpx_total, "n_features": sum(r["n_features"] for r in scenes),
              "arm_gpu_h": arm_total / 3600, "detect_job_gpu_h": job_total / 3600,
              "arm_h_per_gpx": arm_total / 3600 / gpx_total,
              "escalations": sum(r["n_attempts"] - 1 for r in scenes),
              "peak_rss_gib_max": max(r["peak_rss_gib"] for r in scenes),
              "fit_arm_h_per_gpx": {"intercept": float(a), "per_million_detections_per_gpx": float(b),
                                    "r": float(np.corrcoef(x, y)[0, 1])},
              "fit_peak_rss_gib": {"intercept": float(c_), "per_million_detections": float(d_),
                                   "r": float(np.corrcoef(n, m)[0, 1])}}
    print(json.dumps(totals, indent=1))
    return {"scenes": scenes, "totals": totals}


MIN_D = 1.4105                                    # notebook 35's STAGE4_MIN_DIAMETER_M
FLOORS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
SIZE_M = ((1.4105, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, np.inf))
SIZE_PX = ((2.8, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, 12.0), (12.0, 16.0), (16.0, 24.0), (24.0, np.inf))
VCLAIRE = Path(r"C:/Users/brian/Documents/PhD/HiRiseToCTXBoulders/hirise_40_vClaire")
STEM = "-predictions-ct-010-ss-256-is-1024-ov-020-downscaled-mask-nms"


def load_layer(delivery, obs, with_geometry=True):
    """score, equivalent diameter (m) and centroid of every feature of a scene's delivered layer."""
    import pyogrio
    import shapely

    h = json.loads((delivery / "out" / obs / "handoff.json").read_text())
    (lay,) = h["layers"]
    g = pyogrio.read_dataframe(delivery / "out" / obs / lay["layer"], columns=["score"],
                               read_geometry=with_geometry)
    out = {"score": g["score"].to_numpy(), "product_id": h["product_id"]}
    if with_geometry:
        geoms = np.asarray(g.geometry.values, dtype=object)
        area = shapely.area(geoms)
        out.update(area=area, diam=2.0 * np.sqrt(area / np.pi),
                   cen=shapely.get_coordinates(shapely.centroid(geoms)))
    return out


def _share(num, den):
    return float(num / den) if den else None


def section_kept(delivery):
    run = section_run_meta(delivery)
    scenes, curves = [], {}
    for obs, meta in sorted(run.items()):
        L = load_layer(delivery, obs)
        big = L["diam"] >= MIN_D
        s, a, d = L["score"][big], L["area"][big], L["diam"][big]
        n10, a10 = int(len(s)), float(a.sum())
        rec = {"obs_id": obs, **meta, "n_features": int(len(L["score"])), "n_ge_1.41m": n10,
               "n_by_floor": {f"{f:.2f}": int((s >= f).sum()) for f in FLOORS},
               "kept_count": {f"{f:.2f}": _share((s >= f).sum(), n10) for f in FLOORS[1:]},
               "kept_area": {f"{f:.2f}": _share(a[s >= f].sum(), a10) for f in FLOORS[1:]},
               "median_score": float(np.median(s)) if n10 else None,
               "score_hist": {"edges": [round(x, 2) for x in np.arange(0.10, 1.0001, 0.05)],
                              "counts": np.histogram(s, bins=np.arange(0.10, 1.0001, 0.05))[0].tolist()},
               "by_size_m": [], "by_size_px": []}
        for lo, hi in SIZE_M:
            m = (d >= lo) & (d < hi)
            rec["by_size_m"].append({"bin_m": [lo, hi if np.isfinite(hi) else None], "n_0.10": int(m.sum()),
                                     "kept_0.25": _share((m & (s >= 0.25)).sum(), m.sum()),
                                     "kept_0.40": _share((m & (s >= 0.40)).sum(), m.sum())})
        px = L["diam"] / meta["map_scale"]                 # every size, in image pixels
        ss = L["score"]
        for lo, hi in SIZE_PX:
            m = (px >= lo) & (px < hi)
            rec["by_size_px"].append({"bin_px": [lo, hi if np.isfinite(hi) else None], "n_0.10": int(m.sum()),
                                      "kept_0.25": _share((m & (ss >= 0.25)).sum(), m.sum()),
                                      "kept_0.40": _share((m & (ss >= 0.40)).sum(), m.sum())})
        # emission: kept share at 0.40 against effective native pixels, for three scale factors
        e = np.radians(meta["emission"])
        curves[obs] = {}
        for tag, k in (("k1", 1.0), ("k_cos", 1 / np.cos(e)), ("k_cos2", 1 / np.cos(e) ** 2)):
            eff = px / k
            curves[obs][tag] = [{"bin_px": [lo, hi if np.isfinite(hi) else None],
                                 "n_0.10": int(((eff >= lo) & (eff < hi)).sum()),
                                 "kept_0.40": _share(((eff >= lo) & (eff < hi) & (ss >= 0.40)).sum(),
                                                     ((eff >= lo) & (eff < hi)).sum())}
                                for lo, hi in SIZE_PX]
        scenes.append(rec)
        print(f"{obs} {meta['set']:4s} em {meta['emission']:4.1f}: n>=1.41m {n10:>9,d}, kept 0.25 "
              f"{rec['kept_count']['0.25']:.3f}, 0.40 {rec['kept_count']['0.40']:.3f} (area "
              f"{rec['kept_area']['0.40']:.3f}), median score {rec['median_score']:.3f}", flush=True)
    low = [o for o, m in run.items() if m["emission"] <= 13.0]
    ref = []
    for i, (lo, hi) in enumerate(SIZE_PX):
        vals = [curves[o]["k1"][i]["kept_0.40"] for o in low if curves[o]["k1"][i]["n_0.10"] >= 200]
        ref.append({"bin_px": [lo, hi if np.isfinite(hi) else None], "n_scenes": len(vals),
                    "median": float(np.median(vals)) if vals else None,
                    "range": [float(min(vals)), float(max(vals))] if vals else None})
    s2 = "ESP_068483_2280"
    emission = {"reference_scenes": sorted(low), "reference_kept_0.40_by_px": ref,
                "s2": curves[s2], "s2_emission": run[s2]["emission"],
                "other_oblique": {o: curves[o] for o, m in run.items() if m["emission"] > 13.0 and o != s2},
                "rom_kept_0.40_all_tiles": {"ESP_017355_2260": 0.49, "ESP_025650_2165": 0.34,
                                            "ESP_028537_2270": 0.56, "ESP_037018_2170": 0.33,
                                            "source": "data/review/config_study_corrections.json, scene"}}
    return {"scenes": scenes, "emission": emission}


def section_run_meta(delivery):
    """set, map scale and viewing angles per scene, from the delivered labels (no hashing)."""
    idx, meta = scene_index(), {}
    for d in sorted((delivery / "out").iterdir()):
        h = json.loads((d / "handoff.json").read_text())
        lbl = delivery / "scenes" / d.name / f"{h['product_id']}.LBL"
        f = pds_prep.read_label(lbl)
        s = pds_prep.label_summary(f)
        meta[d.name] = {"set": idx.get(d.name, {}).get("set", ""), "map_scale": float(s["map_scale"]),
                        "incidence": float(s["incidence_angle"]),
                        "emission": float(pds_prep._lookup(f, "EMISSION_ANGLE"))}
    return meta


def section_v2(delivery):
    import pyogrio
    import shapely
    from scipy.spatial import cKDTree
    from scipy.stats import ks_2samp

    out = {}
    for obs in ("ESP_068483_2280", "ESP_045550_2180"):
        ours = load_layer(delivery, obs)
        stem = VCLAIRE / obs / f"{obs}_RED{STEM}"
        dbf = pyogrio.read_dataframe(f"{stem}.dbf", read_geometry=False)
        vs = dbf["score"].to_numpy()
        overwritten = np.isclose(vs, 0.10)
        real = vs[~overwritten]
        rec = {"vclaire": {"dbf": str(stem) + ".dbf", "dbf_sha256": file_digest(f"{stem}.dbf"),
                           "n_rows": int(len(vs)), "n_at_exactly_0.10": int(overwritten.sum()),
                           "isin_slice_0_rows": int((dbf["isin_slice"].to_numpy() == 0).sum()),
                           "min_other_score": float(real.min()) if len(real) else None},
               "counts": {}, "scores": {}}
        for f in (0.25, 0.40):
            n_o = int((ours["score"] >= f).sum())
            n_v = int((real >= f).sum())
            rec["counts"][f"{f:.2f}"] = {"ours": n_o, "vclaire_real": n_v,
                                         "ratio_ours_over_vclaire_real": n_o / n_v if n_v else None}
        n_v_all = int(len(real) + overwritten.sum())
        rec["counts"]["0.25"]["vclaire_with_overwritten_rows"] = n_v_all
        rec["counts"]["0.25"]["ratio_ours_over_vclaire_with_overwritten"] = (
            rec["counts"]["0.25"]["ours"] / n_v_all if n_v_all else None)
        q = [0.05, 0.25, 0.5, 0.75, 0.95]
        o25, v25 = ours["score"][ours["score"] >= 0.25], real[real >= 0.25]
        rec["scores"] = {"quantiles": q, "ours": np.quantile(o25, q).tolist(), "vclaire": np.quantile(v25, q).tolist(),
                         "ks_statistic": float(ks_2samp(o25, v25).statistic)}
        # geometry: whatever of vClaire's .shp survives (complete for S3; truncated for S2)
        g = pyogrio.read_dataframe(f"{stem}.shp", columns=["score"])
        geoms = np.asarray(g.geometry.values, dtype=object)
        okg = ~shapely.is_missing(geoms)
        vg, vsc = geoms[okg], g["score"].to_numpy()[okg]
        va = shapely.area(vg)
        vc = shapely.get_coordinates(shapely.centroid(vg))
        keep_v = (vsc >= 0.25) & (2 * np.sqrt(va / np.pi) >= 1.0)
        keep_o = (ours["score"] >= 0.25) & (ours["diam"] >= 1.0)
        Vc, Va, Oc, Oa = vc[keep_v], va[keep_v], ours["cen"][keep_o], ours["area"][keep_o]
        to, tv = cKDTree(Oc), cKDTree(Vc)
        dist, j = to.query(Vc, distance_upper_bound=3.0)
        i = np.flatnonzero(np.isfinite(dist))
        j = j[i]
        _, back = tv.query(Oc[j])
        mutual = back == i
        i, j = i[mutual], j[mutual]
        good = (Va[i] / Oa[j] > 0.8) & (Va[i] / Oa[j] < 1.25)
        off = Vc[i[good]] - Oc[j[good]]
        med = np.median(off, axis=0) if len(off) else np.array([np.nan, np.nan])
        # after removing the measured offset: matched shares within 1 m, and the matched area ratio
        dist2, j2 = to.query(Vc - med, distance_upper_bound=1.0)
        hit = np.isfinite(dist2)
        rec["geometry"] = {"n_vclaire_geometries": int(okg.sum()), "n_vclaire_used": int(keep_v.sum()),
                           "n_ours_used": int(keep_o.sum()), "n_offset_pairs": int(len(off)),
                           "offset_vclaire_minus_ours_m": med.tolist(),
                           "offset_px": (med / 0.25).tolist(),
                           "expected_if_jp2_frame_m": [0.375, -0.375],
                           "share_vclaire_matched_within_1m_after_shift": float(hit.mean()),
                           "matched_area_ratio_ours_over_vclaire_median": float(np.median(Oa[j2[hit]] / Va[hit]))
                           if hit.any() else None}
        out[obs] = rec
        c = rec["counts"]
        print(f"{obs}: >=0.25 ours {c['0.25']['ours']:,d} vs vClaire {c['0.25']['vclaire_real']:,d} real "
              f"(+{rec['vclaire']['n_at_exactly_0.10']:,d} overwritten): ratio "
              f"{c['0.25']['ratio_ours_over_vclaire_real']:.4f} / {c['0.25']['ratio_ours_over_vclaire_with_overwritten']:.4f}; "
              f">=0.40 ratio {c['0.40']['ratio_ours_over_vclaire_real']:.4f}; KS {rec['scores']['ks_statistic']:.4f}; "
              f"offset {np.round(med, 3).tolist()} m on {len(off):,d} pairs; matched "
              f"{rec['geometry']['share_vclaire_matched_within_1m_after_shift']:.3f}", flush=True)
    return out


def section_steps(delivery, obs="ESP_068483_2280", f=8, seg_m=1000.0, win_m=20.0, band_m=128.0, guard_m=4.0,
                  n_boot=2000, seed=0):
    """Detection density either side of the CCD-channel calibration steps of one scene (S2 by default)."""
    import rasterio
    from rasterio.enums import Resampling

    h = json.loads((delivery / "out" / obs / "handoff.json").read_text())
    tif = delivery / "scenes" / obs / h["raster"]["path"]
    with rasterio.open(tif) as r:
        T, H, W = r.transform, r.height, r.width
        dn = r.read(1, out_shape=(H // f, W // f), resampling=Resampling.nearest)
    s = T.a                                                     # m per image pixel
    valid = dn > 0
    rr, cc = np.nonzero(valid)
    xy = np.column_stack([cc, rr]).astype(float) * f            # image pixel coordinates
    mu = xy.mean(axis=0)
    _, _, vt = np.linalg.svd((xy - mu)[:: max(1, len(xy) // 200000)], full_matrices=False)
    along, across = vt[0], vt[1]
    t = (xy - mu) @ along * s                                   # metres along track
    u = (xy - mu) @ across * s                                  # metres across track
    vals = dn[rr, cc].astype(float)
    ub = np.floor((u - u.min()) / 1.0).astype(int)               # 1 m across-track bins
    tb = np.floor((t - t.min()) / seg_m).astype(int)            # along-track segments
    nU, nT = ub.max() + 1, tb.max() + 1
    S_ = np.zeros((nT, nU))
    N_ = np.zeros((nT, nU))
    np.add.at(S_, (tb, ub), vals)
    np.add.at(N_, (tb, ub), 1.0)
    w = int(win_m)
    cS, cN = np.cumsum(np.pad(S_, ((0, 0), (1, 0))), axis=1), np.cumsum(np.pad(N_, ((0, 0), (1, 0))), axis=1)
    idx = np.arange(w, nU - w)
    left = (cS[:, idx] - cS[:, idx - w]) / np.maximum(cN[:, idx] - cN[:, idx - w], 1)
    right = (cS[:, idx + w] - cS[:, idx]) / np.maximum(cN[:, idx + w] - cN[:, idx], 1)
    full = ((cN[:, idx] - cN[:, idx - w]) >= 0.9 * w * (seg_m / (f * s)) * 0.5) & \
           ((cN[:, idx + w] - cN[:, idx]) >= 0.9 * w * (seg_m / (f * s)) * 0.5)
    jump = np.where(full, right - left, np.nan)                 # DN jump at u, per segment
    fin = np.isfinite(jump)
    nseg = fin.sum(axis=0)
    med = np.full(jump.shape[1], np.nan)
    med[nseg > 0] = np.nanmedian(jump[:, nseg > 0], axis=0)
    agree = np.where(nseg > 0, (fin & (np.sign(np.nan_to_num(jump)) == np.sign(med)[None, :])).sum(axis=0)
                     / np.maximum(nseg, 1), 0.0)
    # Steps: local maxima of |median jump| that persist (same sign in >= 85 % of >= 10 segments), at least
    # 0.5 DN, and at least 150 m apart and from the swath edges. On S2 no step reaches 1.5 DN in the prepared
    # 8-bit raster; the persistent ones are 0.5-1 DN (2-4 DN in 10 bits).
    min_jump, min_agree, min_seg, sep = 0.5, 0.85, 10, 150
    edge_ok = (idx >= sep) & (idx <= nU - sep)
    score_ = np.where((agree >= min_agree) & (nseg >= min_seg) & edge_ok, np.abs(np.nan_to_num(med)), 0.0)
    order = np.argsort(score_)[::-1]
    steps = []
    for k in order:
        if score_[k] < min_jump:
            break
        if all(abs(k - j) >= sep for j in steps):
            steps.append(k)
    steps = sorted(steps)
    u0 = u.min()
    step_u = [float(u0 + idx[k]) for k in steps]
    # pseudo-steps: random flat positions (|median jump| < 0.2 DN), >= sep from every step and edge
    flat = np.flatnonzero(edge_ok & (nseg >= min_seg) & (np.abs(np.nan_to_num(med, nan=9.0)) < 0.2))
    flat = [k for k in flat if all(abs(k - j) >= sep for j in steps)]
    prng = np.random.default_rng(seed + 1)
    pseudo_k = []
    for k in prng.permutation(flat):
        if all(abs(k - j) >= sep for j in pseudo_k):
            pseudo_k.append(int(k))
        if len(pseudo_k) == 40:
            break
    pseudo_u = sorted(float(u0 + idx[k]) for k in pseudo_k)
    # detections in the same (t, u) frame
    L = load_layer(delivery, obs)
    cpx = np.column_stack([(L["cen"][:, 0] - T.c) / T.a, (L["cen"][:, 1] - T.f) / T.e])
    dt = (cpx - mu) @ along * s
    du = (cpx - mu) @ across * s
    score, diam = L["score"], L["diam"]
    tblk = np.floor((dt - t.min()) / seg_m).astype(int)
    vblk = np.floor((t - t.min()) / seg_m).astype(int)

    def band_stats(c):
        """per along-track block: counts on each side (floors, sizes) and valid area per side."""
        sides = {"left": (c - guard_m - band_m, c - guard_m), "right": (c + guard_m, c + guard_m + band_m)}
        out = {}
        for side, (lo, hi) in sides.items():
            vm = (u >= lo) & (u < hi)
            area = np.bincount(vblk[vm], minlength=nT) * (f * s) ** 2
            dm = (du >= lo) & (du < hi)
            cnt = {}
            for fl in (0.10, 0.25, 0.40):
                for name, (a_, b_) in {"ge1.41": (MIN_D, np.inf), "1.41-2": (MIN_D, 2.0),
                                       "2-4": (2.0, 4.0), "ge4": (4.0, np.inf)}.items():
                    m = dm & (score >= fl) & (diam >= a_) & (diam < b_)
                    cnt[f"{fl:.2f}_{name}"] = np.bincount(tblk[m].clip(0, nT - 1), minlength=nT)
            med_score = float(np.median(score[dm & (diam >= MIN_D)])) if (dm & (diam >= MIN_D)).any() else None
            out[side] = {"area": area, "cnt": cnt, "median_score": med_score}
        return out

    rng = np.random.default_rng(seed)

    def ratio_ci(bs, key):
        """right/left density ratio over blocks where both sides are mostly valid, with a block bootstrap."""
        aL, aR = bs["left"]["area"], bs["right"]["area"]
        good = (aL >= 0.8 * band_m * seg_m) & (aR >= 0.8 * band_m * seg_m)
        cL, cR = bs["left"]["cnt"][key][good], bs["right"]["cnt"][key][good]
        aL, aR = aL[good], aR[good]
        if not good.any() or cL.sum() == 0:
            return {"n_blocks": int(good.sum()), "ratio": None, "ci95": None}
        pt = (cR.sum() / aR.sum()) / (cL.sum() / aL.sum())
        nb = len(cL)
        reps = []
        for _ in range(n_boot):
            i = rng.integers(0, nb, nb)
            if cL[i].sum() > 0:
                reps.append((cR[i].sum() / aR[i].sum()) / (cL[i].sum() / aL[i].sum()))
        return {"n_blocks": int(nb), "n_left": int(cL.sum()), "n_right": int(cR.sum()), "ratio": float(pt),
                "ci95": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]}

    keys = [f"{fl:.2f}_{n}" for fl in (0.10, 0.25, 0.40) for n in ("ge1.41", "1.41-2", "2-4", "ge4")]
    rows = []
    for c in step_u:
        k = int(round(c - u0)) - w
        bs = band_stats(c)
        rows.append({"u_m": c, "dn_jump_right_minus_left": float(med[k]), "sign_agreement": float(agree[k]),
                     "n_segments": int(nseg[k]),
                     "median_score": {sd: bs[sd]["median_score"] for sd in ("left", "right")},
                     "ratio_right_over_left": {key: ratio_ci(bs, key) for key in keys}})
    pseudo = []
    for c in pseudo_u:
        bs = band_stats(c)
        pseudo.append({"u_m": c, "ratio_right_over_left": {key: ratio_ci(bs, key) for key in keys}})
    null = {}
    for key in keys:
        lr = [abs(np.log(p["ratio_right_over_left"][key]["ratio"])) for p in pseudo
              if p["ratio_right_over_left"][key]["ratio"]]
        null[key] = {"n": len(lr), "abs_log_ratio_p95": float(np.percentile(lr, 95)) if lr else None,
                     "abs_log_ratio_median": float(np.median(lr)) if lr else None}
    from scipy.stats import spearmanr
    jumps = [r["dn_jump_right_minus_left"] for r in rows]
    lrat = [np.log(r["ratio_right_over_left"]["0.10_ge1.41"]["ratio"] or np.nan) for r in rows]
    okk = np.isfinite(lrat)
    rho = (float(spearmanr(np.array(jumps)[okk], np.array(lrat)[okk]).statistic) if okk.sum() > 3 else None)
    print(f"{obs}: {len(rows)} steps found, {len(pseudo)} pseudo-steps; |DN jump| "
          f"{[round(abs(j), 1) for j in jumps]}", flush=True)
    for r in rows:
        x = r["ratio_right_over_left"]["0.10_ge1.41"]
        print(f"  u {r['u_m']:8.0f} m  jump {r['dn_jump_right_minus_left']:+5.1f} DN  density R/L at 0.10 "
              f"{x['ratio']:.3f} {x['ci95']}  (null |log| p95 {null['0.10_ge1.41']['abs_log_ratio_p95']})", flush=True)
    big = np.argsort(np.abs(np.nan_to_num(med)) * edge_ok)[::-1][:5]
    return {"obs_id": obs, "params": {"decimation_px": f, "segment_m": seg_m, "window_m": win_m, "band_m": band_m,
                                      "guard_m": guard_m, "min_jump_dn": min_jump, "min_sign_agreement": min_agree,
                                      "min_segments": min_seg, "min_separation_m": sep, "n_segments": int(nT)},
            "largest_median_jumps_dn": [{"u_m": float(u0 + idx[k]), "jump": float(med[k]), "agree": float(agree[k])}
                                        for k in big],
            "swath": {"along_axis": along.tolist(), "across_axis": across.tolist(),
                      "width_m": float(u.max() - u.min()), "length_m": float(t.max() - t.min())},
            "steps": rows, "pseudo_steps": pseudo, "null": null,
            "spearman_log_density_ratio_vs_dn_jump_0.10_ge1.41": rho}


def polygon_contrast(r, t, geoms, n, rng):
    """Median |mean DN inside - mean DN in a 1 m ring| over up to ``n`` polygons, on open raster ``r`` with
    affine ``t`` ([a, b, c, d, e, f]). Polygons whose ring touches fill (DN 0) are skipped."""
    import shapely
    from rasterio.windows import Window

    if len(geoms) > n:
        geoms = geoms[rng.choice(len(geoms), size=n, replace=False)]
    vals = []
    for g in geoms:
        ring = shapely.difference(shapely.buffer(g, 1.0), g)
        b = shapely.bounds(shapely.buffer(g, 1.0))
        c0, r0 = int((b[0] - t[2]) / t[0]), int((b[3] - t[5]) / t[4])
        c1, r1 = int(np.ceil((b[2] - t[2]) / t[0])), int(np.ceil((b[1] - t[5]) / t[4]))
        if c0 < 0 or r0 < 0 or c1 >= r.width or r1 >= r.height:
            continue
        win = r.read(1, window=Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1)).astype(float)
        yy, xx = np.mgrid[r0:r0 + win.shape[0], c0:c0 + win.shape[1]]
        px = shapely.points(t[2] + (xx.ravel() + 0.5) * t[0], t[5] + (yy.ravel() + 0.5) * t[4])
        inside, inring = shapely.contains(g, px), shapely.contains(ring, px)
        if inside.sum() >= 3 and inring.sum() >= 3 and (win.ravel()[inring] > 0).all():
            vals.append(abs(win.ravel()[inside].mean() - win.ravel()[inring].mean()))
    return (float(np.median(vals)) if vals else None), len(vals)


def contrast_by_rows(tif, t, geoms, band=4096, n=300, seed=0):
    """polygon_contrast per band of rows, with the null of the same polygons moved 50 m east."""
    import rasterio
    import shapely

    rows = (shapely.get_coordinates(shapely.centroid(geoms))[:, 1] - t[5]) / t[4]
    rng = np.random.default_rng(seed)
    out = []
    with rasterio.open(tif) as r:
        for b0 in range(0, r.height, band):
            m = (rows >= b0) & (rows < b0 + band)
            if m.sum() < 50:
                continue
            c, k = polygon_contrast(r, t, geoms[m], n, rng)
            c0, _ = polygon_contrast(r, t, shapely.transform(geoms[m], lambda x: x + np.array([50.0, 0.0])), n, rng)
            out.append({"rows": [b0, b0 + band], "n": int(m.sum()), "sample": k, "contrast": c, "null": c0,
                        "ratio": (c / c0) if (c is not None and c0) else None})
    return out


def section_placement(delivery, scenes=None):
    import pyogrio

    scenes = scenes or sorted(p.name for p in (delivery / "out").iterdir())
    res = {}
    for obs in scenes:
        h = json.loads((delivery / "out" / obs / "handoff.json").read_text())
        (lay,) = h["layers"]
        t = json.loads((delivery / "scenes" / obs / f"{h['product_id']}.prep.json").read_text())["output"]["transform"]
        g = pyogrio.read_dataframe(delivery / "out" / obs / lay["layer"], columns=["score"])
        geoms = np.asarray(g.geometry.values, dtype=object)[g["score"].to_numpy() >= 0.25]
        bands = contrast_by_rows(delivery / "scenes" / obs / h["raster"]["path"], t, geoms)
        ratios = [b["ratio"] for b in bands if b["ratio"] is not None]
        res[obs] = {"bands": bands, "min_ratio": float(min(ratios)) if ratios else None, "n_bands": len(bands)}
        print(f"{obs}: {len(bands)} bands, contrast/null ratio min {res[obs]['min_ratio']:.2f}", flush=True)
    return res


SECTIONS = {"run": section_run, "kept": section_kept, "v2": section_v2, "steps": section_steps,
            "placement": section_placement}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delivery", default=str(DELIVERY))
    ap.add_argument("--sections", nargs="+", default=list(SECTIONS), choices=list(SECTIONS))
    ap.add_argument("--out", default=str(REPO / "data" / "review" / "sherlock_pilot.json"))
    args = ap.parse_args()
    out_path = Path(args.out)
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    out.update({"what": "the Sherlock pilot's analysis (docs/sherlock_pilot_plan.md section 8)",
                "delivery": args.delivery, "release": RELEASE_SHA})
    out.setdefault("code", {})
    for name in args.sections:
        out[name] = SECTIONS[name](Path(args.delivery))
        out["code"][name] = presets.code_identity()
    out_path.write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
