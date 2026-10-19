"""Do the scripts committed at 08c94df reproduce the pilot's analysis JSONs? (hirise2ctx's Stage 1 note, 6.2.)

Every analysis section's code block records ``git_dirty: true``: the sections ran before their scripts were
committed. This answers the question two ways and writes one record.

1. **By digest.** Each code block records ``scripts_sha256``, presets.source_digest over every ``.py`` under
   ``scripts/`` (paths and LF-normalised bytes). The same digest is computed from git's blobs for each commit
   from ``--first`` to ``--last``, so a section either ran from exactly some commit's scripts, or from a state
   that was never committed.
2. **By re-running.** Each script, as committed at ``--last``, re-run on a clean tree (outputs elsewhere), and
   compared with the committed JSON leaf by leaf, code blocks apart. A file with no re-run in ``--rerun-dir`` is
   recorded as not re-run. With ``--incidence-backup`` and ``--crosscheck-orig``, the gitignored intermediates
   are compared too (the incidence script's per-comparison files, the prep cross-check's laptop rasters).

    conda run -n bouldernet python -I scripts/review/check_pilot_reproduction.py --rerun-dir <dir> \
        --incidence-backup <dir> --crosscheck-orig <dir> --out data/review/pilot_reproduction_08c94df.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
REVIEW = REPO / "data" / "review"
JSONS = ("sherlock_pilot.json", "incidence_pairs_pilot.json", "v2_block_shift_check.json",
         "pilot_prep_crosscheck.json")


def git(*a, binary=False):
    r = subprocess.run(["git", *a], cwd=REPO, capture_output=True, check=True)
    return r.stdout if binary else r.stdout.decode()


def tree_digest(commit, prefix):
    """presets.source_digest's algorithm over a commit's blobs under ``prefix``."""
    files = [f for f in git("ls-tree", "-r", "--name-only", commit, "--", prefix).splitlines() if f.endswith(".py")]
    h = hashlib.sha256()
    for rel, f in sorted((f[len(prefix):].lstrip("/"), f) for f in files):
        h.update(rel.encode())
        h.update(b"\0")
        h.update(git("show", f"{commit}:{f}", binary=True).replace(b"\r\n", b"\n"))
        h.update(b"\0")
    return h.hexdigest()


def leaves(o, path=()):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, path + (str(k),))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, path + (f"[{i}]",))
    else:
        yield path, o


def compare(old, new, skip=("code",)):
    out = {}
    for top in sorted(set(old) | set(new)):
        if top in skip:
            continue
        a, b = dict(leaves(old.get(top))), dict(leaves(new.get(top)))
        differ, worst = [], 0.0
        for k in sorted(set(a) & set(b)):
            x, y = a[k], b[k]
            if x == y or (isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y)):
                continue
            num = isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool)
            if num:
                worst = max(worst, abs(x - y) / max(abs(x), abs(y), 1e-300))
            differ.append({"leaf": "/".join(k), "committed": x, "rerun": y})
        out[top] = {"leaves": len(set(a) | set(b)), "differ": len(differ), "only_committed": len(set(a) - set(b)),
                    "only_rerun": len(set(b) - set(a)), "worst_relative_numeric": worst, "first_differences": differ[:20]}
    return out


def code_blocks(d):
    c = d.get("code")
    if not isinstance(c, dict):
        return {}
    pick = lambda x: {k: x.get(k) for k in ("git_sha", "git_dirty", "scripts_sha256", "source_sha256")}   # noqa: E731
    if all(isinstance(v, dict) for v in c.values()):
        return {k: pick(v) for k, v in c.items()}
    return {"(file)": pick(c)}


def npz_equal(a, b):
    x, y = np.load(a, allow_pickle=False), np.load(b, allow_pickle=False)
    return sorted(x.files) == sorted(y.files) and all(np.array_equal(x[k], y[k], equal_nan=True) for k in x.files)


def rasters_equal(a, b):
    import rasterio
    from rasterio.windows import Window
    with rasterio.open(a) as ra, rasterio.open(b) as rb:
        same_meta = (ra.width, ra.height, ra.count, ra.dtypes) == (rb.width, rb.height, rb.count, rb.dtypes)
        out = {"transform_equal": ra.transform == rb.transform, "crs_equal": ra.crs == rb.crs, "shape_equal": same_meta}
        if not same_meta:
            return {**out, "identical": False}
        n_diff, step = 0, 1024
        for y in range(0, ra.height, step):
            w = Window(0, y, ra.width, min(step, ra.height - y))
            n_diff += int(np.count_nonzero(ra.read(1, window=w) != rb.read(1, window=w)))
        return {**out, "n_pixels": ra.width * ra.height, "n_differ": n_diff,
                "identical": n_diff == 0 and out["transform_equal"] and out["crs_equal"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rerun-dir", required=True, help="where the re-runs wrote their JSONs (same names)")
    ap.add_argument("--first", default="e007bb4")
    ap.add_argument("--last", default="08c94df")
    ap.add_argument("--incidence-backup", help="a copy of data/incidence_pairs_pilot taken before the re-run")
    ap.add_argument("--crosscheck-orig", help="the prep cross-check's laptop rasters from the original run")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from YOLOv8BeyondEarth import presets
    code = presets.code_identity()
    commits = [c for c in git("rev-list", "--reverse", f"{args.first}~1..{args.last}").split()]
    by_digest = {}
    digests = []
    for c in commits:
        d = {"commit": c[:7], "subject": git("log", "-1", "--format=%s", c).strip()[:90],
             "scripts_sha256": tree_digest(c, "scripts"), "source_sha256": tree_digest(c, "src/YOLOv8BeyondEarth")}
        digests.append(d)
        by_digest.setdefault(d["scripts_sha256"], []).append(c[:7])
    rerun = Path(args.rerun_dir)
    files = {}
    for name in JSONS:
        old = json.loads((REVIEW / name).read_text())
        cb = code_blocks(old)
        files[name] = {
            "committed_sha256": hashlib.sha256((REVIEW / name).read_bytes()).hexdigest(),
            "code_committed": {k: {**v, "scripts_equal_commits": by_digest.get(v["scripts_sha256"], [])}
                               for k, v in cb.items()}}
        if (rerun / name).exists():
            new = json.loads((rerun / name).read_text())
            files[name].update({"rerun_sha256": hashlib.sha256((rerun / name).read_bytes()).hexdigest(),
                                "code_rerun": code_blocks(new), "sections": compare(old, new)})
        else:
            files[name]["rerun"] = None
    inter = {}
    if args.incidence_backup:
        bdir, ndir = Path(args.incidence_backup), REPO / "data" / "incidence_pairs_pilot"
        for f in sorted(bdir.iterdir()):
            g = ndir / f.name
            if not g.exists():
                inter[f.name] = {"rerun_missing": True}
            elif f.suffix == ".json":
                c = compare(json.loads(f.read_text()), json.loads(g.read_text()), skip=())
                inter[f.name] = {"differ": sum(v["differ"] for v in c.values()),
                                 "leaves": sum(v["leaves"] for v in c.values())}
            elif f.suffix == ".npz":
                inter[f.name] = {"arrays_equal": bool(npz_equal(f, g))}
    rasters = {}
    if args.crosscheck_orig:
        odir, ndir = Path(args.crosscheck_orig), REPO / "data" / "pilot_crosscheck"
        for d in sorted(p for p in ndir.iterdir() if p.is_dir()):
            orig = odir / d.name / f"{d.name}_RED.tif"
            rasters[d.name] = (rasters_equal(orig, d / f"{d.name}_RED.tif") if orig.exists()
                               else {"original_absent": True})
    out = {"what": "whether the scripts committed at the last commit reproduce the pilot's analysis JSONs, by "
                   "digest and by re-running on a clean tree", "asked_by":
               "hirise2ctx-pub/docs/bouldernet_stage1_results_2026-10-18.md, section 6, question 2",
           "code": code, "commits": digests, "files": files,
           "incidence_intermediates": inter, "crosscheck_rasters": rasters}
    Path(args.out).write_text(json.dumps(out, indent=1, default=str))
    for name, r in files.items():
        print(name, {k: (v["leaves"], v["differ"]) for k, v in r.get("sections", {}).items()} or "not re-run",
              {k: v["scripts_equal_commits"] for k, v in r["code_committed"].items()})
    print("intermediates differing:", [k for k, v in inter.items() if v.get("differ") or v.get("arrays_equal") is False
                                       or v.get("rerun_missing")])
    print("rasters:", {k: v.get("identical") for k, v in rasters.items()})


if __name__ == "__main__":
    main()
