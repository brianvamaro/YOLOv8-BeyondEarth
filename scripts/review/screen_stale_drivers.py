"""Which drivers were fixed and then never re-run? The execution screen, not a review pass.

The July/August review was reading-based, and reading does not catch this class. `run_ap50_test.py`
was repaired on 2026-08-06 and then sat in the queue for two weeks as "run-ready" while its gate
block raised ``TypeError`` on every invocation — the script read perfectly and had produced nothing
since. Three of the four defects found in it on 2026-08-21 were invisible until it ran.

The signal is cheap: **a driver whose last code fix is NEWER than its newest output artifact has not
been run since it was fixed**, so whatever it last wrote describes the pre-fix code. That is exactly
the state that let the backlog carry a false claim about the ap50 cache for weeks.

Output paths are recovered heuristically — literal filename strings, plus the literal tails of
f-strings — so a driver whose outputs are fully computed at runtime reports ``no outputs found``
rather than a verdict. That is a *finding to check by hand*, not a pass: it means the screen cannot
see what this script produces.

Dates: the code side uses the last **commit** touching the file (authoritative); the artifact side
uses **mtime**, which a copy or a re-download can move forward. So ``STALE`` is the reliable verdict
and ``ok`` is the weaker one — a false ``ok`` is possible, a false ``STALE`` much less so.

    conda run -n bouldernet python scripts/review/screen_stale_drivers.py
    conda run -n bouldernet python scripts/review/screen_stale_drivers.py --since 2026-08-04
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: Extensions that count as a result. `.py`/`.md` are excluded deliberately — a driver committed
#: alongside its own docs would otherwise look fresh because the doc moved.
ART_EXT = (".json", ".csv", ".gpkg", ".shp", ".png", ".pdf", ".parquet", ".tif", ".jsonl", ".npz")

#: Directory trees searched for artifacts.
ART_ROOTS = ("data", "docs/figures")

#: Names that are inputs or caches rather than results, so their freshness says nothing about
#: whether the driver ran. `tiles` is the ap50 input cache; weights and rasters are downloads.
SKIP_DIRS = {"weights", "raster", "tiles", "__pycache__", ".git", "labels", "images"}

#: How long after an artifact a commit may land and still count as "ran, then committed" — the
#: normal workflow, and the second false-positive mode this screen had.
TOLERANCE_HOURS = 24.0

#: Drivers, by name. A module that only defines helpers is not a driver and has no outputs to check.
DRIVER_PREFIXES = ("run_", "analyze_", "measure_", "probe_", "check_", "prep_", "calibrate_",
                   "validate_", "make_", "audit_", "screen_", "bench_", "stress_", "profile_",
                   "cut_", "fetch_", "build_", "chase_", "tune_")

_LITERAL = re.compile(r"""['"]([^'"\n]*?\.(?:%s))['"]""" % "|".join(e[1:] for e in ART_EXT))
_FSTRING_TAIL = re.compile(r"""\}([A-Za-z0-9_.\-]*\.(?:%s))['"]""" % "|".join(e[1:] for e in ART_EXT))
#: Literal *prefix* before an interpolation, e.g. `obb_crop_angles` in
#: ``f"{OUT}/obb_crop_angles{SUF}.csv"``. Without this the screen sees only the ``.csv`` tail and
#: cannot find the file — which is how it first reported `run_obb_orientation.py` as stale while its
#: real outputs were current, the exact false positive it exists to avoid producing.
_FSTRING_PREFIX = re.compile(r"""[/"']([A-Za-z0-9_\-]{6,})\{""")


def _log1(path, *flags):
    """``(timestamp, subject)`` of the last commit matching ``flags`` for ``path``."""
    out = subprocess.run(["git", "-C", str(REPO), "log", "-1", "--format=%cI %s", *flags,
                          "--", str(path)], capture_output=True, text=True)
    text = out.stdout.strip()
    if not text:
        return None, None
    # %cI contains no spaces, so the first one separates the timestamp from the subject.
    stamp, subject = text.partition(" ")[::2]
    return datetime.fromisoformat(stamp), subject


def last_code_change(path):
    """When this driver's code last **changed** — modifications only, not the commit that first
    tracked it.

    This distinction is the whole screen. 69 drivers under ``data/`` became tracked in one commit on
    2026-08-06 (``a3ad4f5``, the ``!data/**/*.py`` decision), because leaving them untracked meant
    fixes to them were not durable. Treating that ADD as a code change flagged **64 of 83** drivers
    as stale on the first run — noise, not signal, and it would have sent someone re-running Solution
    G because its script became tracked. A file whose only commit is its ADD has not been modified
    since, so it is not a candidate at all.
    """
    when, subject = _log1(path, "--diff-filter=M")
    return when, subject


#: Calls that mean "this filename is an OUTPUT". Without this the screen cannot tell a file the
#: driver reads from one it writes, and an *input's* date then decides the verdict — which is what
#: made `run_obb_orientation.py` read as stale off a 2026-07-06 input while its own outputs were
#: current. `read_file`/`read_csv` are deliberately absent.
WRITE_VERBS = ("to_csv", "to_json", "to_file", "write_text", "write_bytes", "savefig", "savez",
               "savez_compressed", "json.dump", "to_parquet", "np.save", "outlines_to_shp",
               "write_manifest", '"w"', "'w'", "to_excel")


def declared_outputs(script):
    """``(exact basenames, filename prefixes)`` this driver appears to **write**.

    Two heuristics, both narrow on purpose. Names are taken from lines that also contain a write
    call; prefixes come from the literal part before an interpolation, because most drivers here
    build every output name at runtime (``f"{OUT}/obb_crop_angles{SUF}.csv"``) so the finished
    basename is nowhere in the source.

    A driver that passes its path through a variable (``p = OUT / "x.csv"`` … ``df.to_csv(p)``) has
    no write verb on the naming line, so if the write-scoped scan finds nothing the whole-file scan
    is used instead. That fallback is looser and can pick up an input; it is why ``STALE`` here is a
    triage signal to check rather than a verdict to act on.
    """
    text = script.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    written = [ln for ln in lines if any(verb in ln for verb in WRITE_VERBS)]

    def scan(source):
        names, prefixes = set(), set()
        for chunk in source:
            for match in _LITERAL.findall(chunk):
                name = match.replace("\\", "/").rsplit("/", 1)[-1]
                if name and not name.startswith("."):
                    names.add(name.lower())
            for match in _FSTRING_TAIL.findall(chunk):
                if not match.startswith("."):
                    names.add(match.lower())
            for match in _FSTRING_PREFIX.findall(chunk):
                prefixes.add(match.lower())
        # A driver that reads a raster or weights is not producing it.
        return ({n for n in names if not n.endswith((".pt", ".yaml", ".yml"))}, prefixes)

    names, prefixes = scan(written)
    if not names and not prefixes:
        names, prefixes = scan(lines)
    return names, prefixes


def artifact_index(roots):
    """``{basename: (newest mtime, path)}`` over every artifact under ``roots``."""
    by_name = {}
    for root in roots:
        base = REPO / root
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for filename in filenames:
                if not filename.lower().endswith(ART_EXT):
                    continue
                path = Path(dirpath) / filename
                try:
                    mtime = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
                except OSError:
                    continue
                key = filename.lower()
                if key not in by_name or mtime > by_name[key][0]:
                    by_name[key] = (mtime, path)
    return by_name


def newest_match(names, prefixes, by_name):
    """Newest artifact for this driver: exact/tail matches if there are any, else prefix matches.

    **Exact evidence beats fuzzy evidence, and the fallback is only a fallback.** A prefix is shared
    property — ``operating_point`` matches both ``measure_operating_point.py``'s own
    ``operating_point.json`` and the unrelated ``operating_point_test_split.json`` — so letting a
    prefix hit override an exact one lets *another script's* fresh output vouch for this one. That
    turns a true STALE into a false ``ok``, which is the one direction this screen must not fail in.
    Prefixes are still needed for the many drivers that build every output name at runtime.
    """
    best, how = None, None

    def consider(hit, kind):
        nonlocal best, how
        if hit and (best is None or hit[0] > best[0]):
            best, how = hit, kind

    for name in names:
        hit = by_name.get(name)
        if hit is None and len(name) >= 8:
            # the declared name may be the literal tail of an f-string
            for key, value in by_name.items():
                if key.endswith(name) and (hit is None or value[0] > hit[0]):
                    hit = value
        consider(hit, "exact")
    if best is not None:
        return best, how
    for prefix in prefixes:
        hit = None
        for key, value in by_name.items():
            if key.startswith(prefix) and (hit is None or value[0] > hit[0]):
                hit = value
        consider(hit, "prefix")
    return best, how


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--since", default="2026-08-04",
                    help="only report drivers whose last fix is on or after this date "
                         "(default: the 2026-08-05/06 fixing session)")
    ap.add_argument("--out", default=str(REPO / "data/review/stale_driver_screen.json"))
    args = ap.parse_args()
    cutoff = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)

    scripts = sorted(
        [p for p in (REPO / "data").rglob("*.py") if p.name.startswith(DRIVER_PREFIXES)]
        + [p for p in (REPO / "scripts").rglob("*.py") if p.name.startswith(DRIVER_PREFIXES)],
        key=lambda p: p.as_posix())
    print(f"{len(scripts)} drivers found; indexing artifacts...", flush=True)
    by_name = artifact_index(ART_ROOTS)
    print(f"{len(by_name)} artifact basenames indexed\n", flush=True)

    rows = []
    skipped_add_only = 0
    for script in scripts:
        fixed, subject = last_code_change(script)
        if fixed is None:
            skipped_add_only += 1       # tracked but never modified since: nothing to re-run for
            continue
        if fixed < cutoff:
            continue
        names, prefixes = declared_outputs(script)
        hit, how = newest_match(names, prefixes, by_name)
        if hit is None:
            verdict = "NO OUTPUTS FOUND"
            newest, where = None, None
        else:
            newest, where = hit
            # A fix is normally committed *after* the run that verified it, so an artifact from
            # within a day of the commit is evidence the driver ran, not that it did not. Only a
            # gap wider than that is the "fixed and never run" state this screen is looking for.
            gap_hours = (fixed - newest).total_seconds() / 3600.0
            verdict = "STALE" if gap_hours > TOLERANCE_HOURS else (
                "same-day" if gap_hours > 0 else "ok")
        rows.append(dict(driver=script.relative_to(REPO).as_posix(),
                         last_fix=fixed.date().isoformat(), last_fix_subject=subject,
                         newest_output=None if newest is None else newest.date().isoformat(),
                         output=None if where is None else where.relative_to(REPO).as_posix(),
                         n_declared=len(names), n_prefixes=len(prefixes),
                         matched_by=how, verdict=verdict))

    order = {"STALE": 0, "NO OUTPUTS FOUND": 1, "same-day": 2, "ok": 3}
    rows.sort(key=lambda r: (order[r["verdict"]], r["driver"]))
    width = max((len(r["driver"]) for r in rows), default=20)
    for row in rows:
        print(f"{row['verdict']:17} {row['driver']:<{width}}  changed {row['last_fix']}  "
              f"newest output {row['newest_output'] or '-'}"
              f"{'' if row['matched_by'] in (None, 'exact') else ' (prefix match)'}")
        if row["verdict"] in ("STALE", "NO OUTPUTS FOUND"):
            print(f"{'':17} {'':<{width}}  └ {(row['last_fix_subject'] or '')[:96]}")

    counts = {v: sum(1 for r in rows if r["verdict"] == v) for v in order}
    print(f"\n{counts.get('STALE', 0)} STALE, {counts.get('NO OUTPUTS FOUND', 0)} unseeable, "
          f"{counts.get('ok', 0)} ok  (of {len(rows)} drivers fixed since {args.since})")
    print("STALE = fixed after its newest output, so that output describes the pre-fix code.")
    print("NO OUTPUTS FOUND = the screen cannot see what this writes; check by hand, not a pass.")
    Path(args.out).write_text(json.dumps(
        dict(since=args.since, n_drivers_considered=len(scripts),
             n_tracked_but_never_modified=skipped_add_only,
             tolerance_hours=TOLERANCE_HOURS, counts=counts, rows=rows,
             caveat=("code side = last commit (authoritative); artifact side = mtime, which a copy "
                     "can move forward. STALE is the reliable verdict; ok is the weaker one.")),
        indent=1))
    print("written", args.out)


if __name__ == "__main__":
    main()
