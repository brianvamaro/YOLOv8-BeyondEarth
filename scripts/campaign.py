"""A multi-scene BoulderNet campaign (hirise2ctx PLAN_v3 P3): one scene manifest, one hand-off per scene.

Three steps per manifest row, each resumable, the first two run as Slurm array jobs
(``sherlock/campaign_prep.sbatch``, ``sherlock/campaign_detect.sbatch``; runbook ``sherlock/CAMPAIGN.md``):

    prep    scripts/prep_pds_scene.py --manifest M --row N --out-root SCENES     (CPU; fetch + 8-bit)
    detect  scripts/campaign.py detect --manifest M --row N --scenes SCENES --out OUT --arms 256
    verify  scripts/campaign.py verify --manifest M --row N --scenes SCENES --out OUT [--arms 256]

``detect`` runs ``run_multiscale_arms.py`` for the scene (one subprocess per arm, ledger-backed resume,
``--max-det`` explicit) and then ``verify``. It escalates by default (``--on-max-det escalate``,
maintainer decision 2026-10-15): an arm with a slice within 10% of its cap is re-run at twice the cap
and half the batch until none is, so a dense scene yields the uncapped catalog instead of a failure.
``--on-max-det error`` restores fail-and-stop. ``verify`` is the acceptance step, and it fails the scene
unless:

- the latest ledger entry of every arm it checks is complete, and the layer still checksums to what
  that entry recorded -- the ledger, not the directory listing, says what the last run produced, so a
  re-run that failed or was killed cannot pass on an older layer;
- every arm manifest reports ``"truncation": {}`` -- a truncated catalog undercounts exactly the
  densest ground, which for density labels is a bias, not noise;
- the detections ran on the prepared raster: the manifest's input sha256 equals the prep sidecar's
  output sha256 (or, when the manifest skipped hashing, its size and name);

and then writes ``SHA256SUMS`` (``sha256sum -c`` format) and ``handoff.json`` beside the layers. The
hand-off directory is ``OUT/<obs_id>/`` holding ``<product_id>-predictions-...-nms.gpkg`` per arm, as
the D10 proposal has it. GeoPackage bytes are not deterministic, so the checksums certify transfer
integrity, not reproducibility; reproducibility is the manifest's job. ``handoff.json`` lists the
attempts behind each layer (its cap, batch and why it stopped), so an escalated arm says so, and names
any layer in the directory that it does not deliver.

``status`` prints one line per manifest row (prepped / detected / verified / failed), and
``VERIFY-STALE`` where the ledger moved on after the last ``verify``. It only reads. ``count`` prints
the manifest's row count, so an array range can be derived rather than typed.

**Not decided here, deliberately:** which arms (D9); fusing a ``{512, 256}`` ladder (the production
runner refuses any arm set but its own three, and a two-arm preset is a product decision); a score
floor (D36, applied downstream to the conf-0.10 superset). Everything is ``--no-fuse`` until then.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from prep_pds_scene import manifest_row  # noqa: E402
from YOLOv8BeyondEarth.presets import manifest_path  # noqa: E402
from YOLOv8BeyondEarth.runstate import COMPLETE, FAILED, StageLedger, file_digest  # noqa: E402

DEFAULT_WEIGHTS = REPO / "data" / "weights" / "yolov8-m-boulder-detection-tmp.pt"

#: Truncation counts that say the cap bound (as in ``run_multiscale_arms.CAP_TRUNCATION``).
CAP_TRUNCATION = ("slices_at_max_det", "slices_near_max_det")


def manifest_count(manifest):
    """Data rows in a scene manifest, i.e. array tasks ``0 .. n-1`` (read as ``manifest_row`` reads)."""
    with open(manifest, newline="", encoding="utf-8-sig") as fh:
        return sum(1 for _ in csv.DictReader(fh))


def scene_paths(manifest, row, scenes, out):
    scene = manifest_row(manifest, row)
    scene_dir = Path(scenes) / scene["obs_id"]
    raster = scene_dir / f"{scene['product_id']}.tif"
    return {**scene, "raster": raster, "prep": raster.with_name(raster.stem + ".prep.json"),
            "out_dir": Path(out) / scene["obs_id"]}


def _layer_manifests(out_dir):
    return sorted(p for p in Path(out_dir).glob("*-manifest.json")
                  if p.name.endswith("-nms-manifest.json") or "-fused-" in p.name)


def _ledger_stages(out_dir):
    """The scene ledger's entries, oldest first, grouped by stage.

    Only reads: with no ledger there are no stages. ``StageLedger`` itself creates the ledger's
    directory, which made ``status`` leave an empty directory for every row it looked at.
    """
    path = Path(out_dir) / "run_ledger.jsonl"
    stages = {}
    for rec in (StageLedger(path).entries() if path.exists() else []):
        stages.setdefault(rec.get("stage", ""), []).append(rec)
    return stages


def _attempts(entries):
    """The finished attempts behind a stage's latest entry, oldest first: those of its configuration.

    An escalating arm's attempts share a ``base_key``, which covers everything but the cap and the
    batch. Entries with another ``base_key`` (other code, raster, weights or settings) did not produce
    this layer, and neither did entries without one, which come from runs before escalation recorded
    it. Both are left out here; ``run_ledger.jsonl`` travels with the hand-off and still has them.
    """
    latest = entries[-1]
    base = latest.get("base_key")
    picked = [r for r in entries if r.get("state") != "running"
              and (r is latest or (base is not None and r.get("base_key") == base))]
    return [{"state": r.get("state"), "max_det": r.get("max_det"),
             "batch_size": r.get("batch_size"), "stopped": r.get("stopped"),
             "worst_slice": (r.get("bound") or {}).get("worst"),
             "wall_seconds": r.get("wall_seconds"), "finished_utc": r.get("finished_utc")}
            for r in picked]


def _lossy(truncation, batch_size):
    """The part of a layer's truncation record that can have cost detections.

    An NMS time-limit record is dropped when it cannot have. Ultralytics writes each image's output
    before it checks the limit and keeps pre-allocated empty outputs for the images it abandons
    (``ultralytics/utils/nms.py``). So an abandoned slice comes back empty, and
    ``slices_zeroed_after_nms_time_limit`` (the timed-out batches' empty slices) bounds the loss: 0
    means none. At batch 1 the one slice is written before the check, so nothing can be lost there,
    and an empty slice in that batch is just empty ground.
    """
    t = {k: v for k, v in (truncation or {}).items() if v}
    if t.get("nms_time_limit_batches") and (not t.get("slices_zeroed_after_nms_time_limit")
                                            or batch_size == 1):
        t.pop("nms_time_limit_batches")
        t.pop("slices_zeroed_after_nms_time_limit", None)
    return t


def _truncation_problem(layer, stage, truncation, batch_size, max_det):
    """A truncated layer, and the remedy that actually changes it."""
    if any(truncation.get(k) for k in CAP_TRUNCATION):
        return (f"{layer}: truncation {truncation} -- a lower bound, not a catalog; re-run detect with "
                "--on-max-det escalate (the default), which re-runs this arm at twice the cap")
    if truncation.get("nms_time_limit_batches"):
        retry = (f"BATCH_SIZE={max(1, int(batch_size) // 2)} and MAX_DET={max_det}"
                 if batch_size and max_det else "a smaller BATCH_SIZE and this layer's MAX_DET")
        return (f"{layer}: truncation {truncation} -- ultralytics' NMS time limit fired, so up to "
                f"{truncation.get('slices_zeroed_after_nms_time_limit', 0)} slices returned no "
                "detections. No cap fixes that, so escalation does not retry it, and re-running detect "
                f"as it is skips this arm as complete. To retry at a smaller batch, remove the {stage} "
                f"lines from run_ledger.jsonl (keep a copy) and re-run detect with {retry} "
                "(campaign_detect.sbatch). The limit is 2.0 + 0.05 x batch seconds per batch, so a "
                "smaller batch gives each slice more of it. Without MAX_DET the runner starts the "
                "escalation ladder again at its first cap. Deleting the layer instead re-runs the same "
                "attempt at the same batch. Runbook: sherlock/CAMPAIGN.md, 'NMS time limit'")
    return f"{layer}: truncation {truncation} -- a lower bound, not a catalog"


def verify(paths, arms=None):
    """Check one scene's catalogs and write its hand-off files. Returns ``(ok, report)``.

    The ledger says what is delivered, not the directory listing (configuration-study audit,
    2026-10-16). A layer counts only when the latest ledger entry of the stage that wrote it is
    complete and the file still checksums to what that entry recorded. A stage whose latest attempt
    failed, or never finished, fails the scene, whatever an older layer of it on disk says. ``arms``
    (``detect`` passes its ``--arms``) are the arms this hand-off is for. Any other layer in the
    directory, e.g. from a trial of another configuration, is listed under ``not_delivered`` and kept
    out of ``SHA256SUMS``. Without ``arms``, every stage in the scene's ledger is checked.
    """
    out_dir, problems, layers = Path(paths["out_dir"]), [], []
    prep = json.loads(Path(paths["prep"]).read_text()) if Path(paths["prep"]).exists() else None
    if prep is None:
        problems.append(f"no prep sidecar at {paths['prep']}")
    stages = _ledger_stages(out_dir)
    wanted = sorted({f"arm:{a}" for a in arms}) if arms else sorted(stages)
    if not wanted:
        problems.append(f"no ledger entries in {out_dir / 'run_ledger.jsonl'}")
    delivered = {}                            # layer file name -> (stage, its entries, recorded sha256)
    for stage in wanted:
        entries = stages.get(stage)
        if not entries:
            problems.append(f"{stage}: no ledger entry, so this scene has no layer for it")
            continue
        latest = entries[-1]
        if latest.get("state") != COMPLETE:
            problems.append(f"{stage}: its latest attempt is {latest.get('state')!r} "
                            f"({latest.get('error') or 'killed, or still running'}), so no layer of it "
                            "is delivered; any layer of it on disk is from an earlier attempt")
            continue
        for path, digest in (latest.get("artifacts") or {}).items():
            delivered[Path(path).name] = (stage, entries, digest)

    for name, (stage, entries, digest) in sorted(delivered.items()):
        layer = out_dir / name
        mpath = manifest_path(layer)
        man = json.loads(mpath.read_text()) if mpath.exists() else {}
        rec = {"layer": name, "stage": stage, "manifest": mpath.name,
               "config_hash": man.get("config_hash"), "truncation": man.get("truncation"),
               "n_features": (man.get("outputs") or {}).get("n_after_nms"),
               "max_det": (man.get("config") or {}).get("max_det"),
               "batch_size": (man.get("config") or {}).get("batch_size"),
               "attempts": _attempts(entries),
               "peak_rss_bytes": man.get("peak_rss_bytes"),
               "n_slices": man.get("n_slices"),
               "n_slices_skipped_fill": man.get("n_slices_skipped_fill"),
               "code": {k: (man.get("code") or {}).get(k)
                        for k in ("git_sha", "git_dirty", "source_sha256", "scripts_sha256", "host",
                                  "torch", "ultralytics")}}
        if not mpath.exists():
            problems.append(f"{name}: no manifest beside it ({mpath.name})")
        lossy = _lossy(man.get("truncation"), rec["batch_size"])
        if lossy and "-fused-" not in name:
            problems.append(_truncation_problem(name, stage, lossy, rec["batch_size"], rec["max_det"]))
        if not layer.exists():
            problems.append(f"{name}: its ledger entry is complete, but the layer is not on disk")
        else:
            rec["sha256"] = file_digest(layer)
            if digest is not None and rec["sha256"] != digest:
                problems.append(f"{name}: changed on disk since its ledger entry (sha256 "
                                f"{rec['sha256'][:12]}, recorded {digest[:12]})")
        raster_in = (man.get("inputs") or {}).get("raster") or {}
        if prep is not None and man:
            want = prep["output"]["sha256"]
            if raster_in.get("sha256"):
                if raster_in["sha256"] != want:
                    problems.append(f"{name}: ran on a raster with sha256 "
                                    f"{raster_in['sha256'][:12]}, not the prepared {want[:12]}")
            elif (raster_in.get("bytes"), Path(str(raster_in.get("path", ""))).name) != \
                    (prep["output"]["bytes"], prep["output"]["path"]):
                problems.append(f"{name}: input raster unhashed and its size/name do not "
                                "match the prepared raster")
        layers.append(rec)

    # Layers in the directory that this hand-off does not deliver, and why, so nobody takes them for it.
    owner = {Path(p).name: s for s, recs in stages.items() for r in recs
             for p in (r.get("artifacts") or {})}
    not_delivered = []
    for mpath in _layer_manifests(out_dir):
        name = mpath.name.replace("-manifest.json", ".gpkg")
        if name in delivered:
            continue
        stage = owner.get(name)
        if stage is None:
            why = "no ledger entry wrote it"
        elif stage not in wanted:
            why = f"{stage} is not one of this hand-off's arms"
        elif stages[stage][-1].get("state") != COMPLETE:
            why = f"the latest attempt of {stage} is {stages[stage][-1].get('state')!r}"
        else:
            why = f"the latest attempt of {stage} wrote another layer"
        not_delivered.append({"layer": name, "stage": stage, "why": why})

    report = {"obs_id": paths["obs_id"], "product_id": paths["product_id"],
              "verified_utc": datetime.now(timezone.utc).isoformat(), "ok": not problems,
              "problems": problems, "stages": wanted, "layers": layers,
              "not_delivered": not_delivered,
              "raster": None if prep is None else {
                  "path": prep["output"]["path"], "sha256": prep["output"]["sha256"],
                  "rule": prep["rule_expr"], "frame": prep["frame"],
                  "jp2_sha256": ((prep.get("inputs") or {}).get("fetched", {}).get("jp2") or
                                 (prep.get("inputs") or {}).get("jp2") or {}).get("sha256")}}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "handoff.json").write_text(json.dumps(report, indent=2, default=str))
    sums = [f"{r['sha256']}  {r['layer']}" for r in layers if r.get("sha256")]
    sums += [f"{file_digest(out_dir / r['manifest'])}  {r['manifest']}" for r in layers
             if (out_dir / r["manifest"]).exists()]
    (out_dir / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    return not problems, report


def _cpus():
    """CPUs this process may run on: its affinity mask, which a Slurm step sets, else the node's."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:                    # Windows and macOS have no sched_getaffinity
        return os.cpu_count()


def detect(args, paths):
    if not Path(paths["raster"]).exists():
        raise SystemExit(f"{paths['obs_id']}: no prepared raster at {paths['raster']} (run prep first)")
    cmd = [sys.executable, str(REPO / "scripts" / "run_multiscale_arms.py"),
           "--raster", str(paths["raster"]), "--out-dir", str(paths["out_dir"]),
           "--weights", str(args.weights), "--arms", *args.arms,
           "--max-det", str(args.max_det), "--on-max-det", args.on_max_det, "--no-fuse",
           "--device", args.device]
    if args.batch_size is not None:
        cmd += ["--batch-size", str(args.batch_size)]
    # Logged from inside the step: Slurm 22.05-23.02 does not pass sbatch's --cpus-per-task on to
    # srun, so a step can be pinned to fewer CPUs than the job has, and predict sizes its worker pool
    # from this affinity mask.
    print(f"step cpus: {_cpus()} in this process's affinity mask, {os.cpu_count()} on the node",
          flush=True)
    print("+ " + " ".join(cmd), flush=True)
    rc = subprocess.run(cmd).returncode
    if rc:
        # The runner's own status rather than a generic 1, so the job script can tell 137 (SIGKILL;
        # on the cluster, the host-RAM out-of-memory killer) from a crash. A signal reads 128 + signal.
        raise SystemExit(128 - rc if rc < 0 else rc)


def _verified_before_ledger_moved(report, stages):
    """True when the ledger has an entry newer than this hand-off: a re-run after ``verify`` that then
    failed or was killed before ``verify`` could run again leaves exactly that."""
    try:
        verified = datetime.fromisoformat(report["verified_utc"])
    except (KeyError, TypeError, ValueError):
        return True
    stamps = (r.get("finished_utc") or r.get("started_utc") for recs in stages.values() for r in recs)
    return any(datetime.fromisoformat(s) > verified for s in stamps if s)


def status(manifest, scenes, out):
    """One line per manifest row. Only reads: a scene with no ledger yet is reported, not created."""
    counts = {}
    for row in range(manifest_count(manifest)):
        p = scene_paths(manifest, row, scenes, out)
        state = "prepped" if Path(p["prep"]).exists() else "pending"
        stages = _ledger_stages(p["out_dir"])
        arms = {s: recs[-1].get("state") for s, recs in stages.items() if s.startswith("arm:")}
        if any(s == FAILED for s in arms.values()):
            state = "FAILED"
        elif any(s != COMPLETE for s in arms.values()):
            state = "unfinished"             # a `running` marker: still running, or killed
        elif arms:
            state = "detected"
        hand = Path(p["out_dir"]) / "handoff.json"
        if hand.exists():
            report = json.loads(hand.read_text())
            if _verified_before_ledger_moved(report, stages):
                state = "VERIFY-STALE"       # the ledger state above is the current one
            else:
                state = "verified" if report.get("ok") else "VERIFY-FAILED"
        counts[state] = counts.get(state, 0) + 1
        print(f"{row:4d}  {p['obs_id']:16s} {state:14s} arms={arms or '-'}")
    print("totals:", counts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    obs = sub.add_parser("obs", help="print the ObsId of a manifest row (for shell scripts)")
    obs.add_argument("--manifest", required=True)
    obs.add_argument("--row", type=int, required=True)
    count = sub.add_parser("count", help="print the manifest's row count N: array tasks 0..N-1")
    count.add_argument("--manifest", required=True)
    for name in ("detect", "verify", "status"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", required=True)
        p.add_argument("--scenes", required=True, help="prep_pds_scene.py's --out-root")
        p.add_argument("--out", required=True, help="hand-off root: one directory per ObsId")
        if name != "status":
            p.add_argument("--row", type=int, required=True, help="0-based; $SLURM_ARRAY_TASK_ID")
        if name == "verify":
            p.add_argument("--arms", nargs="+", default=None,
                           help="the arms this hand-off is for (detect passes its own); default "
                                "every stage in the scene's ledger")
        if name == "detect":
            p.add_argument("--arms", nargs="+", default=["256"],
                           help="arms to run; '256' is configuration 1 (D9 is not decided here)")
            p.add_argument("--max-det", type=int, default=1100,
                           help="starting per-slice cap (escalation doubles it as needed)")
            p.add_argument("--on-max-det", default="escalate", choices=("escalate", "error"),
                           help="'escalate' re-runs a binding arm at twice the cap and half the "
                                "batch until no slice is within 10%% of it; 'error' stops the scene")
            p.add_argument("--batch-size", type=int, default=None,
                           help="starting batch; default the preset's 8 (escalation halves it)")
            p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
            p.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)
    if args.cmd == "obs":
        print(manifest_row(args.manifest, args.row)["obs_id"])
        return 0
    if args.cmd == "count":
        print(manifest_count(args.manifest))
        return 0
    if args.cmd == "status":
        return status(args.manifest, args.scenes, args.out)
    paths = scene_paths(args.manifest, args.row, args.scenes, args.out)
    if args.cmd == "detect":
        detect(args, paths)
    ok, report = verify(paths, arms=args.arms)
    print(json.dumps({"obs_id": report["obs_id"], "ok": ok, "problems": report["problems"]}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
