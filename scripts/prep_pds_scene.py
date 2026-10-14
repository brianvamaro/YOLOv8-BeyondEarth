"""Fetch a HiRISE RED RDR from PDS and convert it to the 8-bit GeoTIFF BoulderNet infers on.

The rules (label frame, ``min((DN + 2) >> 2, 255)``, native grid) live in
``YOLOv8BeyondEarth.pds_prep``; this is the command line around them, in three modes:

    # a product already on disk (the label and its JP2 side by side, PDS names)
    python scripts/prep_pds_scene.py --lbl data/pds_scenes/ESP_028537_2270/ESP_028537_2270_RED.LBL

    # one observation, fetched into <out-root>/<obs_id>/
    python scripts/prep_pds_scene.py --obs-id ESP_028537_2270 --out-root $SCRATCH/campaign/scenes

    # row N of a scene manifest -- what the Slurm array job runs, N = $SLURM_ARRAY_TASK_ID
    python scripts/prep_pds_scene.py --manifest scenes.csv --row 7 --out-root $SCRATCH/campaign/scenes

The manifest is a CSV with an observation id column and, optionally, the two URLs. Column names are
matched loosely (case and punctuation ignored), so hirise2ctx's ``obs_id, product_id, jp2_url,
lbl_url`` and the brief's ``ObsId, ProductId, JP2 URL, LBL URL`` both work; missing URLs are derived
from the observation id.

``--compare-to`` checks the result against a reference raster pixel for pixel. That is how the
conversion was validated against the Prieur training rasters, and the comparison lands in the
sidecar beside the output.

A scene whose output and sidecar both exist is skipped, so re-running an array job after a timeout
only does the scenes that did not finish. Exit status is non-zero on any failure.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from YOLOv8BeyondEarth import pds_prep  # noqa: E402


def _norm(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def manifest_row(path, row):
    """``{obs_id, product_id, lbl_url, jp2_url}`` for 0-based data row ``row`` of a scene manifest."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not 0 <= row < len(rows):
        raise SystemExit(f"{path}: row {row} requested, manifest has {len(rows)} rows (0-based)")
    rec = {_norm(k): (v or "").strip() for k, v in rows[row].items() if k}
    obs = rec.get("obsid") or rec.get("observationid")
    if not obs:
        raise SystemExit(f"{path}: no observation id column (looked for ObsId / obs_id)")
    lbl_url, jp2_url = pds_prep.rdr_urls(obs)
    return {"obs_id": obs, "product_id": rec.get("productid") or f"{obs}_RED",
            "lbl_url": rec.get("lblurl") or lbl_url, "jp2_url": rec.get("jp2url") or jp2_url}


def run_scene(scene, out_root, *, rule, overwrite, compare_to=None, fetch="yes"):
    """``fetch``: ``"yes"`` (download what is missing, then convert), ``"only"`` (download and stop --
    for a login/DTN node with internet), ``"no"`` (convert files already there -- for a compute node
    without it). The fetch identities are kept in ``fetched.json`` so an offline conversion still
    records where its inputs came from."""
    scene_dir = Path(out_root) / scene["obs_id"]
    lbl = scene_dir / Path(scene["lbl_url"]).name
    out_tif = scene_dir / f"{scene['product_id']}.tif"
    sidecar = out_tif.with_name(out_tif.stem + ".prep.json")
    fetched = scene_dir / "fetched.json"
    if fetch != "only" and out_tif.exists() and sidecar.exists() and not overwrite:
        print(f"[skip] {scene['obs_id']}: {out_tif.name} and its sidecar exist")
        return json.loads(sidecar.read_text())
    if fetch == "no":
        sources = json.loads(fetched.read_text()) if fetched.exists() else None
        if not lbl.exists():
            raise SystemExit(f"--no-fetch: {lbl} is not there; fetch it first (--fetch-only)")
    else:
        sources = {"lbl": pds_prep.fetch(scene["lbl_url"], lbl)}
        jp2_name = pds_prep.label_summary(pds_prep.read_label(lbl))["jp2_name"]
        sources["jp2"] = pds_prep.fetch(scene["jp2_url"], scene_dir / jp2_name)
        fetched.write_text(json.dumps(sources, indent=2))
        if fetch == "only":
            print(f"[fetched] {scene['obs_id']}: {sources['jp2']['bytes'] / 1e6:.0f} MB JP2")
            return {"fetched": sources}
    return prepare(lbl, out_tif, rule=rule, overwrite=True, sources=sources, compare_to=compare_to)


def prepare(lbl, out_tif, *, rule, overwrite, sources=None, compare_to=None):
    rec = pds_prep.prepare_scene(lbl, out_tif, rule=rule, overwrite=overwrite, sources=sources)
    if compare_to:
        cmp = pds_prep.compare_rasters(compare_to, out_tif)
        rec["reference_comparison"] = cmp
        side = Path(out_tif).with_name(Path(out_tif).stem + ".prep.json")
        side.write_text(json.dumps(rec, indent=2, default=str))
        verdict = "IDENTICAL" if cmp["identical"] else "DIFFERENT"
        print(f"vs {Path(compare_to).name}: {verdict} -- {cmp.get('n_differ')} of "
              f"{cmp.get('n_pixels')} px differ {cmp.get('diff_histogram')}, transform equal "
              f"{cmp['transform_equal']}, CRS equal {cmp['crs_equal']}")
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--lbl", help="a local PDS label with its JP2 beside it")
    src.add_argument("--obs-id", help="observation id to fetch, e.g. ESP_028537_2270")
    src.add_argument("--manifest", help="scene manifest CSV (use with --row)")
    ap.add_argument("--row", type=int, help="0-based manifest data row (Slurm: $SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--out-root", help="fetch into <out-root>/<obs_id>/ (required with --obs-id / "
                                       "--manifest)")
    ap.add_argument("--out-tif", help="with --lbl: output path (default: beside the label)")
    ap.add_argument("--rule", default=pds_prep.DEFAULT_RULE, choices=sorted(pds_prep.RULES))
    ap.add_argument("--compare-to", help="reference raster to check the result against, pixel for pixel")
    ap.add_argument("--overwrite", action="store_true")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--fetch-only", action="store_true",
                      help="download the label and JP2 and stop (a node with internet)")
    mode.add_argument("--no-fetch", action="store_true",
                      help="convert files already downloaded (a node without internet)")
    args = ap.parse_args(argv)
    fetch = "only" if args.fetch_only else "no" if args.no_fetch else "yes"

    if args.lbl and fetch != "yes":
        ap.error("--fetch-only / --no-fetch apply to --obs-id and --manifest, not to a local --lbl")
    if args.lbl:
        rec = prepare(Path(args.lbl), args.out_tif, rule=args.rule, overwrite=args.overwrite,
                      compare_to=args.compare_to)
    else:
        if not args.out_root:
            ap.error("--out-root is required with --obs-id or --manifest")
        if args.manifest:
            if args.row is None:
                ap.error("--manifest needs --row")
            scene = manifest_row(args.manifest, args.row)
        else:
            lbl_url, jp2_url = pds_prep.rdr_urls(args.obs_id)
            scene = {"obs_id": args.obs_id, "product_id": f"{args.obs_id}_RED",
                     "lbl_url": lbl_url, "jp2_url": jp2_url}
        rec = run_scene(scene, args.out_root, rule=args.rule, overwrite=args.overwrite,
                        compare_to=args.compare_to, fetch=fetch)
    cmp = rec.get("reference_comparison")
    return 1 if cmp is not None and not cmp["identical"] else 0


if __name__ == "__main__":
    sys.exit(main())
