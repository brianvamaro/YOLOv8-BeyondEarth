"""The v3 campaign's delivery, re-verified on the laptop: one tracked script, one JSON.

The campaign ran release `pilot-1` (`dfcaa26`) on Sherlock on 2026-10-18 and 19, over hirise2ctx's final 62
(`sherlock/manifests/campaign_v3.csv`), with both of configuration 2's arms and no fusion (`ARMS="256 512"`,
`HANDOFF.md` §9). Its hand-offs were pulled into the campaign's own laptop root (`HANDOFF.md` §1). This
reads that root, the manifest, `pilot-1`'s root (S1–S3's 256 layers were delivered there) and the release
tarball, and writes nothing but its JSON.

- **Per scene:**
  - `HANDOFF.md` §8's six refusals, per layer; rule 4's arm set is `["arm:256", "arm:512"]`;
  - every layer's bytes against `handoff.json` and `SHA256SUMS`, and the prepared raster's against
    `handoff.json` and its sidecar. `handoff.json`'s own sha256 is recorded (§3);
  - each layer's configuration against its arm: slice 256 or 512, inference 1024, overlap 0.2, skimage
    contours, no SAM2, no SR;
  - the release in every code block: `dfcaa26` through `CODE_STAMP.json`, the stamp matching the source;
  - the scene against its manifest row: product, URLs and size, and the label against its fetch record;
  - S1–S3: the 256 layer and its manifest, byte for byte against `pilot-1`'s delivery.
- **Per arm, the run:** features, GPU-h (every attempt this campaign ran, from the ledger), escalations,
  peak RSS and slices skipped as fill. Attempts already in `pilot-1`'s ledger are the pilot's, not the
  campaign's.
- **`CODE_STAMP.json`** in the root, against the release tarball and every manifest's embedded stamp.

Results are reported as measured. Nothing here applies a score floor or fuses the layers.

    conda run -n bouldernet python -I scripts/review/campaign_v3_delivery_check.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tarfile
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from YOLOv8BeyondEarth import pds_prep, presets                             # noqa: E402
from YOLOv8BeyondEarth.runstate import file_digest                          # noqa: E402

DELIVERIES = Path(r"C:/Users/brian/Documents/PhD/BoulderNet/data/deliveries/hirise2ctx_v3")
MANIFEST = REPO / "sherlock" / "manifests" / "campaign_v3.csv"
MANIFEST_SHA = "bc370cb5efcc5e1e401b06684d7a8a828d16a6eed807cef1a0f8a9a0b8a968b7"
PACK = REPO.parent / "sherlock_upload" / "bnet_code_pilot-1.tgz"
RELEASE_SHA = "dfcaa26acc233923cbeed541e23421e943017f82"
SOURCE_SHA = "db7f3e58344142212f46486578c8f9b8e302d99d640efcfef10edcb01a751d6d"
FRAME = "PDS label via GDAL's PDS driver (not the JP2's GeoJP2 box)"
STAGES = ["arm:256", "arm:512"]
SLICE = {"arm:256": 256, "arm:512": 512}
SCENE_SETS = {"PSP_009116_2240": "S1", "ESP_068483_2280": "S2", "ESP_045550_2180": "S3"}


def manifest_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def read_sums(path):
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            out[name.strip().lstrip("*")] = digest
    return out


def ledger(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def crs_ok(wkt, center_latitude):
    from rasterio.crs import CRS

    p = pds_prep._crs_params(CRS.from_wkt(wkt))
    return (p.get("lat_ts") is not None and abs(float(p["lat_ts"]) - float(center_latitude)) <= 1e-9
            and float(p.get("lat_0") or 0.0) == 0.0)


def code_stamp(root, manifests):
    """The root's CODE_STAMP.json against the release tarball's member and each manifest's embedded stamp."""
    path = root / "CODE_STAMP.json"
    rec = {"path": str(path), "present": path.exists(), "pack": str(PACK),
           "pack_sha256": file_digest(PACK) if PACK.exists() else None}
    if not path.exists():
        return rec
    raw = path.read_bytes()
    with tarfile.open(PACK, "r:gz") as tf:
        (member,) = [m for m in tf.getmembers() if m.name.endswith("CODE_STAMP.json")]
        packed = tf.extractfile(member).read()
    stamp = json.loads(raw)
    rec.update(sha256=file_digest(path), pack_member=member.name, equals_pack_member=raw == packed,
               git_sha=stamp.get("git_sha"), source_sha256=stamp.get("source_sha256"),
               equals_every_manifest_stamp=all(m["code"].get("stamp") == stamp for m in manifests))
    return rec


def check_scene(row, i, root, pilot):
    obs, prod = row["ObsId"], row["ProductId"]
    d, sd = root / "out" / obs, root / "scenes" / obs
    c = {}
    h = json.loads((d / "handoff.json").read_text())
    # rule 1, rule 4 and the hand-off's shape
    c["rule1_ok"] = h["ok"] is True and h["problems"] == []
    c["rule4_stages"] = h["stages"] == STAGES and sorted(lay["stage"] for lay in h["layers"]) == STAGES
    c["nothing_not_delivered"] = not h["not_delivered"]
    c["product_matches_row"] = h["obs_id"] == obs and h["product_id"] == prod
    # rule 2, the VERIFY-STALE test
    lines = ledger(d / "run_ledger.jsonl")
    verified = datetime.fromisoformat(h["verified_utc"])
    c["rule2_not_verify_stale"] = not any(datetime.fromisoformat(r[k]) > verified for r in lines
                                          for k in ("started_utc", "finished_utc") if r.get(k))
    # rule 3, plus each layer against handoff.json
    sums = read_sums(d / "SHA256SUMS")
    listed = {lay["layer"] for lay in h["layers"]} | {lay["manifest"] for lay in h["layers"]}
    c["rule3_sha256sums_cover_layers"] = set(sums) == listed
    c["rule3_sha256sums_pass"] = all(file_digest(d / name) == digest for name, digest in sums.items())
    # the prepared raster, the label and the fetch record
    raster = h["raster"] or {}
    c["rule6_label_frame"] = raster.get("frame") == FRAME
    prep = json.loads((sd / f"{prod}.prep.json").read_text())
    fc = prep["frame_check"]
    c["frame_check_zero"] = max(abs(fc["dx_px"]), abs(fc["dy_px"])) <= 1e-6
    raster_sha = file_digest(sd / raster["path"]) if raster.get("path") else None
    c["raster_bytes_intact"] = raster_sha == raster.get("sha256") == prep["output"]["sha256"]
    fetched = json.loads((sd / "fetched.json").read_text())
    c["label_matches_fetch_record"] = file_digest(sd / f"{prod}.LBL") == fetched["lbl"]["sha256"]
    c["jp2_matches_fetch_record"] = raster.get("jp2_sha256") == fetched["jp2"]["sha256"]
    c["urls_match_row"] = (fetched["lbl"]["url"] == row["LBL URL"] and fetched["jp2"]["url"] == row["JP2 URL"])
    fields = pds_prep.read_label(sd / f"{prod}.LBL")
    s = pds_prep.label_summary(fields)
    gpx = prep["output"]["width"] * prep["output"]["height"] / 1e9
    c["gpx_matches_row"] = abs(gpx - float(row["Gpx"])) < 1e-4
    c["map_scale_0.25"] = abs(float(s["map_scale"]) - 0.25) < 1e-9
    # per layer: rule 5, configuration, code and inputs
    pilot_lines = ledger(pilot / "out" / obs / "run_ledger.jsonl")
    pilot_keys = {json.dumps(r, sort_keys=True) for r in pilot_lines}
    arms, manifests = {}, []
    for lay in sorted(h["layers"], key=lambda x: x["stage"]):
        st = lay["stage"]
        man = json.loads((d / lay["manifest"]).read_text())
        manifests.append(man)
        cfg, code = man["config"], man["code"]
        c[f"{st}_layer_sha_matches_handoff"] = file_digest(d / lay["layer"]) == lay.get("sha256") == sums.get(lay["layer"])
        c[f"{st}_rule5"] = (man.get("truncation") == {} and lay.get("truncation") == {}
                            and cfg["confidence_threshold"] == 0.1 and cfg["max_det"] == lay["max_det"]
                            and cfg["batch_size"] == lay["batch_size"]
                            and cfg.get("inference_contract_version") == 3)
        c[f"{st}_config_is_arm"] = (cfg["slice_size"] == SLICE[st] and cfg["inference_size"] == 1024
                                    and cfg["overlap_height_ratio"] == 0.2 and cfg["overlap_width_ratio"] == 0.2
                                    and cfg["contour_method"] == "skimage" and cfg["remask_fn"] is None
                                    and cfg["sr_factor"] == 1 and SLICE[st] == int(lay["layer"].split("-ss-")[1].split("-")[0]))
        c[f"{st}_release"] = (code.get("git_sha") == RELEASE_SHA and code.get("git_sha_from") == "CODE_STAMP.json"
                              and code.get("stamp_matches_source") is True and code.get("source_sha256") == SOURCE_SHA
                              and lay["code"].get("source_sha256") == SOURCE_SHA)
        c[f"{st}_ran_on_prepared_raster"] = (man.get("inputs") or {}).get("raster", {}).get("sha256") == raster.get("sha256")
        c[f"{st}_crs_lat_ts_center_latitude"] = crs_ok(man["crs"], s["center_latitude"])
        outcomes = [r for r in lines if r.get("stage") == st and r.get("state") != "running"]
        ours = [r for r in outcomes if json.dumps(r, sort_keys=True) not in pilot_keys]
        markers = [r for r in lines if r.get("stage") == st and r.get("state") == "running"]
        att = lay["attempts"]
        arms[st] = {
            "n_features": lay["n_features"], "detections_per_gpx": lay["n_features"] / gpx,
            "max_det": lay["max_det"], "batch_size": lay["batch_size"], "n_attempts": len(att),
            "stopped": [a.get("stopped") for a in att if a.get("stopped")],
            "delivered_wall_s": att[-1].get("wall_seconds"),
            "ran_in": "pilot-1" if not ours else "campaign",
            "campaign_outcomes": len(ours), "campaign_wall_s": sum(r.get("wall_seconds") or 0 for r in ours),
            "unmatched_running_markers": len(markers) - len(outcomes),
            "peak_rss_gib": (lay.get("peak_rss_bytes") or 0) / 2 ** 30,
            "n_slices": lay.get("n_slices"), "n_slices_skipped_fill": lay.get("n_slices_skipped_fill"),
            "host": lay["code"].get("host"), "torch": lay["code"].get("torch"),
            "ultralytics": lay["code"].get("ultralytics"), "layer_sha256": lay.get("sha256")}
    if obs in SCENE_SETS:
        pd = pilot / "out" / obs
        ph = json.loads((pd / "handoff.json").read_text())
        (pl,) = ph["layers"]
        lay = next(x for x in h["layers"] if x["stage"] == "arm:256")
        c["arm:256_identical_to_pilot-1"] = (pl["layer"] == lay["layer"] and pl.get("sha256") == lay.get("sha256")
                                             and file_digest(pd / pl["layer"]) == file_digest(d / lay["layer"])
                                             and file_digest(pd / pl["manifest"]) == file_digest(d / lay["manifest"]))
        c["arm:256_not_rerun"] = arms["arm:256"]["ran_in"] == "pilot-1"
    rec = {"row": i, "obs_id": obs, "set": SCENE_SETS.get(obs, ""), "role": row["Role"],
           "binned_ccds": row["Binned CCDs"], "gpx": gpx, "incidence": float(s["incidence_angle"]),
           "emission": float(pds_prep._lookup(fields, "EMISSION_ANGLE")),
           "handoff_sha256": file_digest(d / "handoff.json"), "verified_utc": h["verified_utc"],
           "prep_convert_s": prep.get("wall_seconds"), "gdal": prep.get("gdal"),
           "arms": arms, "checks": c, "all_checks_pass": all(c.values())}
    return rec, manifests


def arm_totals(scenes, st):
    camp = [r for r in scenes if r["arms"][st]["ran_in"] == "campaign"]
    a = [r["arms"][st] for r in camp]
    gpx = sum(r["gpx"] for r in camp)
    wall = sum(x["campaign_wall_s"] for x in a)
    worst = max(scenes, key=lambda r: r["arms"][st]["peak_rss_gib"])
    return {"n_scenes_run_in_campaign": len(camp), "gpx_run_in_campaign": gpx,
            "n_features_delivered": sum(r["arms"][st]["n_features"] for r in scenes),
            "campaign_gpu_h": wall / 3600, "campaign_h_per_gpx": wall / 3600 / gpx if gpx else None,
            "escalated_scenes": sorted(r["obs_id"] for r in scenes if r["arms"][st]["n_attempts"] > 1),
            "stopped": dict(Counter(s for r in scenes for s in r["arms"][st]["stopped"])),
            "delivered_cap_batch": {f"{k[0]}/{k[1]}": v for k, v in Counter(
                (r["arms"][st]["max_det"], r["arms"][st]["batch_size"]) for r in scenes).items()},
            "unmatched_running_markers": sum(x["unmatched_running_markers"] for x in a),
            "peak_rss_gib_max": worst["arms"][st]["peak_rss_gib"], "peak_rss_scene": worst["obs_id"],
            "slices_skipped_fill_share": (sum(x["n_slices_skipped_fill"] for x in a) / sum(x["n_slices"] for x in a)
                                          if a else None),
            "hosts": dict(Counter(x["host"] for x in a)),
            "torch": sorted({x["torch"] for x in a}), "ultralytics": sorted({x["ultralytics"] for x in a})}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--delivery", default=str(DELIVERIES / "campaign-v3"))
    ap.add_argument("--pilot", default=str(DELIVERIES / "pilot-1"))
    ap.add_argument("--out", default=str(REPO / "data" / "review" / "campaign_v3_delivery.json"))
    args = ap.parse_args()
    root, pilot = Path(args.delivery), Path(args.pilot)
    msha = file_digest(MANIFEST)
    if msha != MANIFEST_SHA:
        raise SystemExit(f"{MANIFEST} is {msha}, not the campaign's {MANIFEST_SHA}")
    rows = manifest_rows(MANIFEST)
    on_disk = sorted(p.name for p in (root / "out").iterdir())
    scenes, manifests = [], []
    for i, row in enumerate(rows):
        rec, mans = check_scene(row, i, root, pilot)
        scenes.append(rec)
        manifests += mans
        a, b = rec["arms"]["arm:256"], rec["arms"]["arm:512"]
        print(f"{i:2d} {rec['obs_id']} {rec['set']:2s} {rec['gpx']:5.2f} Gpx  256: {a['n_features']:>9,d} det "
              f"{(a['delivered_wall_s'] or 0) / 60:5.1f} min ({a['ran_in']})  512: {b['n_features']:>9,d} det "
              f"{(b['delivered_wall_s'] or 0) / 60:5.1f} min, RSS {max(a['peak_rss_gib'], b['peak_rss_gib']):5.2f} GiB, "
              f"attempts {a['n_attempts']}/{b['n_attempts']}; checks "
              f"{'pass' if rec['all_checks_pass'] else [k for k, v in rec['checks'].items() if not v]}", flush=True)
    both = [r for r in scenes if all(r["arms"][st]["ran_in"] == "campaign" for st in STAGES)]
    w256 = sum(r["arms"]["arm:256"]["campaign_wall_s"] for r in both)
    w512 = sum(r["arms"]["arm:512"]["campaign_wall_s"] for r in both)
    n256 = sum(r["arms"]["arm:256"]["n_features"] for r in both)
    n512 = sum(r["arms"]["arm:512"]["n_features"] for r in both)
    ratios = sorted((r["arms"]["arm:256"]["campaign_wall_s"] + r["arms"]["arm:512"]["campaign_wall_s"])
                    / r["arms"]["arm:256"]["campaign_wall_s"] for r in both)
    totals = {
        "n_scenes": len(scenes), "n_all_checks_pass": sum(r["all_checks_pass"] for r in scenes),
        "failing": {r["obs_id"]: [k for k, v in r["checks"].items() if not v] for r in scenes if not r["all_checks_pass"]},
        "out_dirs_match_manifest": on_disk == sorted(r["ObsId"] for r in rows),
        "gpx": sum(r["gpx"] for r in scenes),
        "arms": {st: arm_totals(scenes, st) for st in STAGES},
        "campaign_gpu_h_both_arms": sum(r["arms"][st]["campaign_wall_s"] for r in scenes for st in STAGES) / 3600,
        "config2_over_config1_arm_time": {
            "scenes": len(both), "pooled": (w256 + w512) / w256 if w256 else None,
            "per_scene_min_median_max": [ratios[0], ratios[len(ratios) // 2], ratios[-1]] if ratios else None,
            "note": "both arms' wall time over the 256 arm's, on scenes where the campaign ran both; "
                    "fusion is not included (it is not built)"},
        "arm512_over_arm256_features": {"pooled": n512 / n256 if n256 else None, "scenes": len(both)},
        "prep_convert_h": sum(r["prep_convert_s"] or 0 for r in scenes if r["set"] == "") / 3600,
        "prep_convert_s_max": max((r["prep_convert_s"] or 0, r["obs_id"]) for r in scenes if r["set"] == ""),
    }
    out = {"code": presets.code_identity(), "delivery": str(root), "pilot_delivery": str(pilot),
           "manifest": {"path": str(MANIFEST.relative_to(REPO)), "sha256": msha, "rows": len(rows)},
           "code_stamp": code_stamp(root, manifests), "scenes": scenes, "totals": totals}
    print(json.dumps({"code_stamp": out["code_stamp"], "totals": totals}, indent=1))
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
