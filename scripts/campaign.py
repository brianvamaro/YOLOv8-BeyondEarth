"""A multi-scene BoulderNet campaign (hirise2ctx PLAN_v3 P3): one scene manifest, one hand-off per scene.

Three steps per manifest row, each resumable, the first two run as Slurm array jobs
(``sherlock/campaign_prep.sbatch``, ``sherlock/campaign_detect.sbatch``; runbook ``sherlock/CAMPAIGN.md``):

    prep    scripts/prep_pds_scene.py --manifest M --row N --out-root SCENES     (CPU; fetch + 8-bit)
    detect  scripts/campaign.py detect --manifest M --row N --scenes SCENES --out OUT --arms 256
    verify  scripts/campaign.py verify --manifest M --row N --scenes SCENES --out OUT

``detect`` runs ``run_multiscale_arms.py`` for the scene (one subprocess per arm, ledger-backed resume,
``--max-det`` explicit, ``--on-max-det error``) and then ``verify``. ``verify`` is the acceptance step,
and it fails the scene unless:

- every arm manifest reports ``"truncation": {}`` -- a truncated catalog undercounts exactly the
  densest ground, which for density labels is a bias, not noise;
- the detections ran on the prepared raster: the manifest's input sha256 equals the prep sidecar's
  output sha256 (or, when the manifest skipped hashing, its size and name);

and then writes ``SHA256SUMS`` (``sha256sum -c`` format) and ``handoff.json`` beside the layers. The
hand-off directory is ``OUT/<obs_id>/`` holding ``<product_id>-predictions-...-nms.gpkg`` per arm, as
the D10 proposal has it. GeoPackage bytes are not deterministic, so the checksums certify transfer
integrity, not reproducibility; reproducibility is the manifest's job.

``status`` prints one line per manifest row (prepped / detected / verified / failed).

**Not decided here, deliberately:** which arms (D9); fusing a ``{512, 256}`` ladder (the production
runner refuses any arm set but its own three, and a two-arm preset is a product decision); a score
floor (D36, applied downstream to the conf-0.10 superset). Everything is ``--no-fuse`` until then.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from prep_pds_scene import manifest_row  # noqa: E402
from YOLOv8BeyondEarth.runstate import COMPLETE, FAILED, StageLedger, file_digest  # noqa: E402

DEFAULT_WEIGHTS = REPO / "data" / "weights" / "yolov8-m-boulder-detection-tmp.pt"


def scene_paths(manifest, row, scenes, out):
    scene = manifest_row(manifest, row)
    scene_dir = Path(scenes) / scene["obs_id"]
    raster = scene_dir / f"{scene['product_id']}.tif"
    return {**scene, "raster": raster, "prep": raster.with_name(raster.stem + ".prep.json"),
            "out_dir": Path(out) / scene["obs_id"]}


def _layer_manifests(out_dir):
    return sorted(p for p in Path(out_dir).glob("*-manifest.json")
                  if p.name.endswith("-nms-manifest.json") or "-fused-" in p.name)


def verify(paths):
    """Check one scene's catalogs and write its hand-off files. Returns ``(ok, report)``."""
    out_dir, problems, layers = Path(paths["out_dir"]), [], []
    prep = json.loads(Path(paths["prep"]).read_text()) if Path(paths["prep"]).exists() else None
    if prep is None:
        problems.append(f"no prep sidecar at {paths['prep']}")
    manifests = _layer_manifests(out_dir)
    if not manifests:
        problems.append(f"no layer manifests in {out_dir}")
    for mpath in manifests:
        man = json.loads(mpath.read_text())
        layer = mpath.with_name(mpath.name.replace("-manifest.json", ".gpkg"))
        rec = {"layer": layer.name, "manifest": mpath.name,
               "config_hash": man.get("config_hash"), "truncation": man.get("truncation"),
               "n_features": (man.get("outputs") or {}).get("n_after_nms"),
               "max_det": (man.get("config") or {}).get("max_det"),
               "peak_rss_bytes": man.get("peak_rss_bytes"),
               "n_slices": man.get("n_slices"),
               "n_slices_skipped_fill": man.get("n_slices_skipped_fill"),
               "code": {k: (man.get("code") or {}).get(k)
                        for k in ("git_sha", "git_dirty", "source_sha256", "scripts_sha256", "host",
                                  "torch", "ultralytics")}}
        if man.get("truncation") not in ({}, None) and "-fused-" not in mpath.name:
            problems.append(f"{layer.name}: truncation {man['truncation']} -- a lower bound, not a "
                            "catalog; raise max_det (and check VRAM) and re-run this arm")
        if not layer.exists():
            problems.append(f"{layer.name}: manifest without its layer")
        else:
            rec["sha256"] = file_digest(layer)
        raster_in = (man.get("inputs") or {}).get("raster") or {}
        if prep is not None:
            want = prep["output"]["sha256"]
            if raster_in.get("sha256"):
                if raster_in["sha256"] != want:
                    problems.append(f"{layer.name}: ran on a raster with sha256 "
                                    f"{raster_in['sha256'][:12]}, not the prepared {want[:12]}")
            elif (raster_in.get("bytes"), Path(str(raster_in.get("path", ""))).name) != \
                    (prep["output"]["bytes"], prep["output"]["path"]):
                problems.append(f"{layer.name}: input raster unhashed and its size/name do not "
                                "match the prepared raster")
        layers.append(rec)
    report = {"obs_id": paths["obs_id"], "product_id": paths["product_id"],
              "verified_utc": datetime.now(timezone.utc).isoformat(), "ok": not problems,
              "problems": problems, "layers": layers,
              "raster": None if prep is None else {
                  "path": prep["output"]["path"], "sha256": prep["output"]["sha256"],
                  "rule": prep["rule_expr"], "frame": prep["frame"],
                  "jp2_sha256": ((prep.get("inputs") or {}).get("fetched", {}).get("jp2") or
                                 (prep.get("inputs") or {}).get("jp2") or {}).get("sha256")}}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "handoff.json").write_text(json.dumps(report, indent=2, default=str))
    sums = [f"{r['sha256']}  {r['layer']}" for r in layers if r.get("sha256")]
    sums += [f"{file_digest(m)}  {m.name}" for m in manifests]
    (out_dir / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    return not problems, report


def detect(args, paths):
    if not Path(paths["raster"]).exists():
        raise SystemExit(f"{paths['obs_id']}: no prepared raster at {paths['raster']} (run prep first)")
    cmd = [sys.executable, str(REPO / "scripts" / "run_multiscale_arms.py"),
           "--raster", str(paths["raster"]), "--out-dir", str(paths["out_dir"]),
           "--weights", str(args.weights), "--arms", *args.arms,
           "--max-det", str(args.max_det), "--on-max-det", "error", "--no-fuse",
           "--device", args.device]
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def status(manifest, scenes, out):
    import csv
    with open(manifest, newline="", encoding="utf-8-sig") as fh:
        n = sum(1 for _ in csv.DictReader(fh))
    counts = {}
    for row in range(n):
        p = scene_paths(manifest, row, scenes, out)
        state = "pending"
        if Path(p["prep"]).exists():
            state = "prepped"
        ledger = StageLedger(Path(p["out_dir"]) / "run_ledger.jsonl")
        arms = {r["stage"]: r.get("state") for r in ledger.entries() if r["stage"].startswith("arm:")}
        if any(s == FAILED for s in arms.values()):
            state = "FAILED"
        elif arms and all(s == COMPLETE for s in arms.values()):
            state = "detected"
        hand = Path(p["out_dir"]) / "handoff.json"
        if hand.exists():
            state = "verified" if json.loads(hand.read_text()).get("ok") else "VERIFY-FAILED"
        counts[state] = counts.get(state, 0) + 1
        print(f"{row:4d}  {p['obs_id']:16s} {state:14s} arms={arms or '-'}")
    print("totals:", counts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    obs = sub.add_parser("obs", help="print the ObsId of a manifest row (for shell scripts)")
    obs.add_argument("--manifest", required=True)
    obs.add_argument("--row", type=int, required=True)
    for name in ("detect", "verify", "status"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", required=True)
        p.add_argument("--scenes", required=True, help="prep_pds_scene.py's --out-root")
        p.add_argument("--out", required=True, help="hand-off root: one directory per ObsId")
        if name != "status":
            p.add_argument("--row", type=int, required=True, help="0-based; $SLURM_ARRAY_TASK_ID")
        if name == "detect":
            p.add_argument("--arms", nargs="+", default=["256"],
                           help="arms to run; '256' is configuration 1 (D9 is not decided here)")
            p.add_argument("--max-det", type=int, default=1100)
            p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
            p.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)
    if args.cmd == "obs":
        print(manifest_row(args.manifest, args.row)["obs_id"])
        return 0
    if args.cmd == "status":
        return status(args.manifest, args.scenes, args.out)
    paths = scene_paths(args.manifest, args.row, args.scenes, args.out)
    if args.cmd == "detect":
        detect(args, paths)
    ok, report = verify(paths)
    print(json.dumps({"obs_id": report["obs_id"], "ok": ok, "problems": report["problems"]}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
