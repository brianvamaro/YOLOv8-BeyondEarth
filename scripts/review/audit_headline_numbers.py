"""Check the numbers the documents quote against the tables they came from (review item G2-09).

Three headline figures were wrong or inconsistent, and none of them was checkable without
re-deriving it by hand:

- documents called **26,830** objects "matched". The saved T35 table has **30,756** detections and
  **20,896** matched to a human label; 26,830 is the total number of human *labels*, so the figure
  was both the wrong number and the wrong word;
- the citable notebook said SAM beats YOLO on **every** body while the summary recorded a Mars
  advantage for YOLO. Recomputed from the table, the summary is right — **Mars favours YOLO** in
  both cohorts;
- illumination wording survived in older plans after the attribution was narrowed to sub-degree.

This is the mechanical part of the audit: it re-derives each number from the saved table and fails
if a tracked document quotes something else. Run it after any re-run.

    conda run -n bouldernet python scripts/review/audit_headline_numbers.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
TOP = REPO.parent
T35 = REPO / "data" / "sam2_validation" / "t35_val_detections.csv"

#: Documents that face outward and must agree with the tables.
TRACKED_DOCS = [
    TOP / "docs" / "orientation_results_summary.md",
    TOP / "docs" / "sam2_orientation_validation_plan.md",
    # The two documents that *define* the Goal-4 fabric gate. Added 2026-08-06: the retracted
    # "R ~ 0.35" bar lived here for two review passes precisely because nothing checked them.
    TOP / "docs" / "goal4_global_maps_plan.md",
    TOP / "docs" / "solutions" / "G_sam2_remasking.md",
    TOP / "notebooks" / "Goal2_SAM2_angle_validation.py",
    TOP / "notebooks" / "Results_summary.py",
    # The rendered hand-out. Generated from Results_summary.py, so in principle redundant — but it
    # is the artifact people are actually given, it can be edited by hand, and a claim can also be
    # assembled at render time from a table the source only describes.
    TOP / "docs" / "results_summary.md",
]


def axial(a, b):
    d = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) % 180.0
    return np.minimum(d, 180.0 - d)


#: A line that quotes the disclaimed wording in order to correct it is not itself a claim.
_CORRECTION = re.compile(r"(⚠|corrected|caveat|but NOT|no longer|used to read|this read"
                         r"|withdrawn|retracted|WRONG|does not reproduce)", re.IGNORECASE)


def normalise(line):
    """Strip markdown emphasis and possessives so a claim cannot hide behind formatting.

    This is not cosmetic. ``audit_headline_numbers.py`` exited **0** while
    ``Goal2_SAM2_angle_validation.py`` — the citable notebook — asserted "SAM's median |Δangle| is
    lower than YOLO's on **every** body", contradicted by its own synthesis table twenty lines
    later. The audit missed it purely because the pattern wanted the literal ``lower than YOLO on
    every body`` and the text had a possessive and two asterisks in the way. A claim checker that
    can be defeated by bold text is not a checker.
    """
    line = re.sub(r"[*_`\\]", "", line)
    line = re.sub(r"\b(\w+)'s\b", r"\1", line)
    return re.sub(r"\s+", " ", line)


#: Claims that were verified wrong and must not come back. Each entry is
#: ``(id, compiled pattern, why it is wrong)``; the pattern runs against the **normalised** line.
#: A line that also matches ``_CORRECTION`` is exempt, so a document may quote a retired claim in
#: order to retract it.
RETIRED_CLAIMS = [
    ("G2-09", re.compile(r"(lower than YOLO on every body"
                         r"|beats YOLO (?:everywhere|on every body)"
                         r"|closer to human than YOLO,? everywhere"
                         r"|SAM (?:is )?closer (?:to human )?than YOLO on every body)",
                         re.IGNORECASE),
     "the saved T5 table has Mars favouring YOLO (SAM 10.97 vs YOLO 9.22 well-resolved)"),
    ("GATE-R", re.compile(r"R\s*(?:≈|~|=)\s*0\.35", re.IGNORECASE),
     "the R ~ 0.35 control-site bar came from the retracted quota sample (8-103 boulders/site, "
     "where an isotropic R of 0.09-0.31 is expected for free) and is withdrawn; the equal-area "
     "replacement is R 0.196-0.443 at n 271-2,388 — see data/review/control_site_null.json"),
    ("GATE-PEAKS", re.compile(r"peaks?\s+(?:at\s+)?43\s*[-–]\s*76", re.IGNORECASE),
     "same retracted sample; the equal-area replacement is peaks 17.3-87.3 deg"),
    ("G2-01", re.compile(r"not something the re-?masker introduces"
                         r"|cardinal .{0,40}is not SAM-specific"
                         r"|no cardinal signature", re.IGNORECASE),
     "on uniform-random synthetic truth with no tracer in the loop SAM returns grid 0.273 against "
     "a truth of 0.227 (n=11,252, generator unbiased to +/-0.0039), and SAM's cardinal share "
     "exceeds isotropic at 5/5 control sites — never state the signature as absent"),
    ("T3-01", re.compile(r"12\.9\s*%|12\.88\s*%"),
     "the Test-3 near-circular pile-up is 4.23% on the production new-skimage set; 12.9% pooled "
     "three stacked catalogs, one of which (new-cv2) reads 35.5% on its own"),
    # The two claims THIS SESSION withdrew. Review pass 9's skeptic found the registry had no entry
    # for either — the guard built to stop the hand-out quoting withdrawn claims was empty exactly
    # where it was needed, and both claims were still live in the citable document.
    ("PEAKS-CARDINAL", re.compile(r"peaks?\s+(?:sit|sits|land|lands|lie|lies)\s+on\s+the\s+"
                                  r"(?:pixel\s+)?cardinals?"
                                  r"|peaks?\s+(?:are\s+)?on\s+the\s+cardinal\s+ax",
                                  re.IGNORECASE),
     "estimator-dependent and NOT established: under refine_peak the control-site peaks average "
     "8.7 deg from a cardinal, under the doubled-angle circular mean 17.4 deg (MC p 0.20 vs 0.007); "
     "on the 11 HiRISE scenes fig07 flips 13.5 deg / 6-of-11 / p 0.010 to 19.6 deg / 2-of-11 / "
     "p 0.227. Cite the grid FRACTIONS, which need no peak estimator"),
    ("NOT-INHERITED", re.compile(r"each\s+(?:producer|masker)\s+imprints\s+its\s+own"
                                 r"|not\s+a\s+single\s+inherited\s+effect"
                                 r"|exchanges?\s+YOLO'?s?\s+diagonal", re.IGNORECASE),
     "pass 5 reported the residual as manufactured-not-inherited; pass 6 RETRACTED that (the "
     "statistic conditioned each producer's rose on its own aspect column) and the standing verdict "
     "is 'inherited (SAM ~ YOLO)' — the intersection has the two producers agreeing at 5/5 sites "
     "under both estimators. What SAM does add is a cardinal signature on FRACTIONS; say that "
     "instead. See fig09_controls._population_note"),
    ("T2-RECALL", re.compile(r"7\s*[-–]\s*24\s*(?:vs|against)\s*211\s*[-–]\s*711"
                             r"|211\s*[-–]\s*711", re.IGNORECASE),
     "the pooled high-incidence recall collapse is a CROSS-PAIR MIXTURE: of the six repeat-pair "
     "members, 24 is a low-incidence member and 498 a high-incidence one, so the two ranges are not "
     "high-vs-low at all. Same-ground the three pairs give 30x (2270a), 1.43x and inverted on raw "
     "counts (2270b), and cannot-tell (2260b, high-vs-high) — and the no-collapse pair is the only "
     "statistically usable one. Source data/sam2_validation/t2_hirise_meta.json; table in "
     "orientation_results_summary.md section T2"),
]


def derive(table=None):
    """Re-derive the T5 headline numbers from the saved detections table.

    **Cohort:** matched boulders with ``h_aspect > 1.35`` and ``h_dpx >= 8`` — i.e. elongated and
    well-resolved *according to the human label*. That is deliberate and it is not the cohort the
    2026-07-24 caveat in ``orientation_results_summary.md`` used: that one required *both* SAM and
    YOLO to read elongated, which selects on the treatments being compared and is exactly what
    review item G2-05 rules out. The two give different numbers (Mars n=176 here against n=106
    there) and agree on the direction: Mars favours YOLO.
    """
    df = pd.read_csv(T35 if table is None else table)
    matched = df.dropna(subset=["h_angle"])
    well = matched[(matched.h_aspect > 1.35) & (matched.h_dpx >= 8)]
    out = {"n_detections": int(len(df)), "n_matched": int(len(matched)),
           "per_body": {}}
    for body, sub in well.groupby("body"):
        out["per_body"][body] = {
            "n": int(len(sub)),
            "sam": float(np.median(axial(sub.angle_sam, sub.h_angle))),
            "yolo": float(np.median(axial(sub.angle_yolo, sub.h_angle))),
        }
    out["sam_wins_every_body"] = all(v["sam"] < v["yolo"] for v in out["per_body"].values())
    return out


def check_docs(facts, docs=None):
    """Fail on claims the tables contradict, and on claims already verified wrong.

    ``docs`` is resolved at call time, not bound as a default, so the list stays overridable.
    """
    docs = TRACKED_DOCS if docs is None else docs
    problems = []
    matched_pat = re.compile(r"([\d,]{4,})\s+(?:matched\s+val\s+boulders|objects?\s+.?matched)",
                             re.IGNORECASE)

    for path in docs:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        # A retraction is usually a paragraph, not a line: "(Corrected 2026-08-06: this read <the
        # retired wording>...)" wraps, and the wording lands on a continuation line carrying no
        # marker of its own. So the exemption persists to the end of the paragraph — a blank line,
        # or a bare comment marker, ends it.
        in_correction = False
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.strip().lstrip("#").strip()
            if not stripped:
                in_correction = False
            if _CORRECTION.search(line):
                in_correction = True
            if in_correction:
                continue          # the paragraph is disclaiming the wording, not asserting it
            flat = normalise(line)
            # Only runnable with the table. `main()` deliberately calls check_docs with
            # n_matched=None first, so the retired-claim scan works on a fresh clone — without this
            # guard that pre-scan raised TypeError from the f-string the moment any document quoted
            # a matched count, i.e. the check named after the defect this script was written for
            # could never report. Found by review pass 9.
            if facts.get("n_matched") is not None:
                for m in matched_pat.finditer(flat):
                    value = int(m.group(1).replace(",", ""))
                    if value != facts["n_matched"]:
                        problems.append(
                            f"{path.name}:{lineno}: quotes {value:,} as a matched count; the table "
                            f"has {facts['n_matched']:,} matched of {facts['n_detections']:,}")
            for cid, pat, why in RETIRED_CLAIMS:
                if cid == "G2-09" and facts.get("sam_wins_every_body"):
                    continue      # only a defect while the table says otherwise
                if pat.search(flat):
                    problems.append(f"{path.name}:{lineno}: [{cid}] retired claim — {why}\n"
                                    f"      {flat.strip()[:150]}")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allow-missing-table", action="store_true",
                    help="exit 0 instead of 2 when the gitignored T35 table is absent. Only for a "
                         "fresh clone with no data/; a CI job that passes this is not auditing "
                         "anything.")
    args = ap.parse_args(argv)

    # The retired-claim scan reads only tracked documents, so it runs with or without the table —
    # and it is the half that caught the citable notebook. Run it first, unconditionally.
    problems = check_docs({"n_matched": None, "n_detections": None,
                           "sam_wins_every_body": False})
    if not T35.exists():
        if problems:
            print("INCONSISTENT CLAIMS (from the document scan; the table is absent):")
            for p in problems:
                print("  " + p)
            return 1
        # An audit that cannot audit must not report success. `data/` is gitignored, so returning 0
        # here made this script pass vacuously on every fresh clone and in any CI without the data.
        print(f"CANNOT AUDIT: {T35} is not present (gitignored data), so no number was checked "
              "against its table. Fetch the data, or pass --allow-missing-table to accept a "
              "documents-only scan.")
        return 0 if args.allow_missing_table else 2

    facts = derive()
    print(f"detections {facts['n_detections']:,}   matched to a human label "
          f"{facts['n_matched']:,}")
    print(f"{'body':7s} {'n':>6s} {'SAM':>7s} {'YOLO':>7s}  winner")
    for body, v in sorted(facts["per_body"].items()):
        print(f"{body:7s} {v['n']:6d} {v['sam']:7.2f} {v['yolo']:7.2f}  "
              f"{'SAM' if v['sam'] < v['yolo'] else 'YOLO'}")
    print(f"SAM beats YOLO on every body: {facts['sam_wins_every_body']}")

    problems = check_docs(facts)
    if problems:
        print("\nINCONSISTENT CLAIMS:")
        for p in problems:
            print("  " + p)
        return 1
    print(f"\nall tracked documents agree with the table, and none of the "
          f"{len(RETIRED_CLAIMS)} retired claims has come back")
    return 0


if __name__ == "__main__":
    sys.exit(main())
