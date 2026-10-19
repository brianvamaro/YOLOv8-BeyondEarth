"""The campaign's scene manifest, built from hirise2ctx's final selection (read-only), with its checks.

hirise2ctx's ``v3_data/dataset/selection/manifest_v3.csv`` is the scene list (DECISIONS 2026-10-18c: 62
scenes, D35 enacted). This writes ``sherlock/manifests/campaign_v3.csv`` in the pilot manifests' layout,
one row per scene in the source's order, so row *i* is array task *i* on both sides. Beside it,
``campaign_v3.json`` records the source, the checks and what changed against a previous selection.

The checks refuse rather than warn:
- the source's sha256 must equal ``--expect-sha256``;
- ObsIds are unique, every product is ``<ObsId>_RED``, and every URL is what ``pds_prep.rdr_urls``
  derives (the runner derives them the same way when a column is empty);
- every scene is at 0.25 m/px and inside the ruled incidence band (``--band``, D35).

Recorded, not refused:
- CCD binning per scene from its RDR label (``pilot_images_check.binning``: a local copy if there is
  one, else fetched into ``data/pds_labels/``, which is gitignored), and so which scenes are mixed;
- the scenes already delivered in a release (``--delivered``), with their ``handoff.json`` verdict;
- the rows that changed against ``--previous``.

    conda run -n bouldernet python -I scripts/build_campaign_manifest.py \
        --expect-sha256 13d22888b08b344cdd86c4e910a9b90dfa32d8fa021b06a57f8a17cdced43a2f \
        --previous <hirise2ctx's superseded manifest_v3.csv> --delivered <delivery root>/pilot-1
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "review"))
from YOLOv8BeyondEarth import pds_prep, presets                             # noqa: E402
from YOLOv8BeyondEarth.runstate import file_digest                          # noqa: E402
import pilot_images_check as pic                                            # noqa: E402

SELECTION = pic.SELECTION
OUT_CSV = REPO / "sherlock" / "manifests" / "campaign_v3.csv"
COLUMNS = ("ObsId", "ProductId", "JP2 URL", "LBL URL", "Role", "Gpx", "Incidence", "Binned CCDs")


def read(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def delivered(root, obs):
    """The verdict a delivered scene's hand-off carries, or None if it is not in the release."""
    h = Path(root) / "out" / obs / "handoff.json"
    if not h.exists():
        return None
    j = json.loads(h.read_text())
    return {"handoff": str(h), "ok": j.get("ok"), "problems": j.get("problems"),
            "stages": j.get("stages"), "verified_utc": j.get("verified_utc"),
            # a later run resumes a layer only on the same package digest and config
            "layers": [{"stage": layer.get("stage"), "config_hash": layer.get("config_hash"),
                        "git_sha": (layer.get("code") or {}).get("git_sha"),
                        "source_sha256": (layer.get("code") or {}).get("source_sha256")}
                       for layer in j.get("layers", [])]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", default=str(SELECTION / "manifest_v3.csv"))
    ap.add_argument("--expect-sha256", required=True)
    ap.add_argument("--band", type=float, nargs=2, default=(40.0, 65.0),
                    help="the ruled incidence band in degrees (D35, 2026-10-18)")
    ap.add_argument("--previous", help="a superseded manifest to diff against")
    ap.add_argument("--delivered", action="append", default=[],
                    help="a delivery root (holding out/<ObsId>/handoff.json); repeatable")
    ap.add_argument("--label-dir", default=str(REPO / "data" / "pds_labels"))
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--out", default=str(OUT_CSV))
    args = ap.parse_args()
    code = presets.code_identity()             # before anything is written, so the record is of a clean tree

    src = Path(args.source)
    sha = file_digest(src)
    if sha != args.expect_sha256:
        raise SystemExit(f"{src}: sha256 {sha}, expected {args.expect_sha256}; not building")
    rows = read(src)
    problems = []
    obs = [r["obs_id"] for r in rows]
    if len(set(obs)) != len(obs):
        problems.append("duplicate obs_id")
    for r in rows:
        o = r["obs_id"]
        lbl, jp2 = pds_prep.rdr_urls(o)
        if r["product_id"] != f"{o}_RED":
            problems.append(f"{o}: product_id {r['product_id']}")
        if (r["lbl_url"], r["jp2_url"]) != (lbl, jp2):
            problems.append(f"{o}: URLs differ from pds_prep.rdr_urls")
        if abs(float(r["map_scale"]) - 0.25) > 1e-9:
            problems.append(f"{o}: map_scale {r['map_scale']}")
        inc = float(r["incidence"])
        if not args.band[0] <= inc <= args.band[1]:
            problems.append(f"{o}: incidence {inc} outside {args.band}")
    if problems:
        raise SystemExit("refused:\n  " + "\n  ".join(problems))

    bins = {o: pic.binning(o, args.label_dir, not args.no_fetch) for o in obs}
    missing = [o for o, b in bins.items() if b.get("label") is None]
    if missing:
        raise SystemExit(f"no label for {missing} (fetching disabled)")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(COLUMNS)
        for r in rows:
            o = r["obs_id"]
            w.writerow([o, r["product_id"], r["jp2_url"], r["lbl_url"], r["role"], r["gpx"],
                        r["incidence"], " ".join(bins[o]["red_binned"])])

    gpx = {o: float(r["gpx"]) for o, r in zip(obs, rows)}
    deliv = {}
    for root in args.delivered:
        for o in obs:
            d = delivered(root, o)
            if d is not None:
                deliv.setdefault(o, []).append({"release_root": str(root), **d})
    record = {
        "what": "the campaign's scene manifest, from hirise2ctx's final v3 selection (read-only)",
        "code": code,
        "source": {"path": str(src), "sha256": sha, "rows": len(rows)},
        "manifest": {"path": (str(out.resolve().relative_to(REPO)).replace("\\", "/")
                              if out.resolve().is_relative_to(REPO) else str(out)),
                     "sha256": file_digest(out),
                     "columns": list(COLUMNS),
                     "note": "the runner reads ObsId, ProductId, JP2 URL and LBL URL; the other columns "
                             "are for the reader"},
        "checks_passed": ["source sha256", "unique ObsIds", "product ids", "URLs equal pds_prep.rdr_urls",
                          "0.25 m/px", f"incidence within {list(args.band)}"],
        "totals": {"scenes": len(rows), "gpx": round(sum(gpx.values()), 4),
                   "gpx_not_yet_delivered": round(sum(v for o, v in gpx.items() if o not in deliv), 4),
                   "incidence_range": [min(float(r["incidence"]) for r in rows),
                                       max(float(r["incidence"]) for r in rows)],
                   "roles": {k: sum(1 for r in rows if r["role"] == k)
                             for k in sorted({r["role"] for r in rows})}},
        "delivered": deliv,
        "binning": {"mixed": [{"obs_id": o, "role": r["role"], "gpx": gpx[o],
                               "red_binned": bins[o]["red_binned"], "red_on": bins[o]["red_on"]}
                              for o, r in zip(obs, rows) if bins[o]["mixed"]],
                    "labels": {o: bins[o]["label"] for o in obs}},
    }
    if args.previous:
        prev = {r["obs_id"]: r for r in read(args.previous)}
        cur = {r["obs_id"]: r for r in rows}
        same = [o for o in obs if o in prev]
        record["against_previous"] = {
            "path": str(args.previous), "sha256": file_digest(Path(args.previous)),
            "out": [{"obs_id": o, "incidence": float(prev[o]["incidence"]), "cell": f"{prev[o]['cell_lon']}"
                     f"_{prev[o]['cell_lat']}", "row": list(prev).index(o)} for o in prev if o not in cur],
            "in": [{"obs_id": o, "incidence": float(cur[o]["incidence"]), "cell": f"{cur[o]['cell_lon']}"
                    f"_{cur[o]['cell_lat']}", "queue_rank": cur[o]["queue_rank"], "row": obs.index(o),
                    "binned": bins[o]["red_binned"]} for o in obs if o not in prev],
            "unchanged_rows_identical_in_every_column": all(prev[o] == cur[o] for o in same),
            "n_unchanged": len(same)}
    js = out.with_suffix(".json")
    js.write_text(json.dumps(record, indent=1))
    print(f"wrote {out} ({len(rows)} scenes, {record['totals']['gpx']} Gpx; "
          f"{len(record['binning']['mixed'])} mixed; delivered already: {sorted(deliv)}) and {js}")


if __name__ == "__main__":
    main()
