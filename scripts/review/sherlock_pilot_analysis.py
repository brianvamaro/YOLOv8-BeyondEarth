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


SECTIONS = {"run": section_run}


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
