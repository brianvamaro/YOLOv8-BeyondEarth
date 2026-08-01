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
    TOP / "notebooks" / "Goal2_SAM2_angle_validation.py",
    TOP / "notebooks" / "Results_summary.py",
]


def axial(a, b):
    d = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) % 180.0
    return np.minimum(d, 180.0 - d)


#: A line that quotes the disclaimed wording in order to correct it is not itself a claim.
_CORRECTION = re.compile(r"(⚠|corrected|caveat|but NOT|no longer|used to read|this read)",
                         re.IGNORECASE)


def derive(table=T35):
    """Re-derive the T5 headline numbers from the saved detections table.

    **Cohort:** matched boulders with ``h_aspect > 1.35`` and ``h_dpx >= 8`` — i.e. elongated and
    well-resolved *according to the human label*. That is deliberate and it is not the cohort the
    2026-07-24 caveat in ``orientation_results_summary.md`` used: that one required *both* SAM and
    YOLO to read elongated, which selects on the treatments being compared and is exactly what
    review item G2-05 rules out. The two give different numbers (Mars n=176 here against n=106
    there) and agree on the direction: Mars favours YOLO.
    """
    df = pd.read_csv(table)
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


def check_docs(facts, docs=TRACKED_DOCS):
    """Fail on claims the table contradicts."""
    problems = []
    matched_pat = re.compile(r"([\d,]{4,})\s+(?:matched\s+val\s+boulders|objects?\s+.?matched)",
                             re.IGNORECASE)
    every_body = re.compile(r"(beats YOLO (?:everywhere|on every body)"
                            r"|lower than YOLO on every body"
                            r"|closer to human than YOLO, everywhere)", re.IGNORECASE)

    for path in docs:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            if _CORRECTION.search(line):
                continue          # the line is disclaiming the wording, not asserting it
            for m in matched_pat.finditer(line):
                value = int(m.group(1).replace(",", ""))
                if value != facts["n_matched"]:
                    problems.append(
                        f"{path.name}:{lineno}: quotes {value:,} as a matched count; the table "
                        f"has {facts['n_matched']:,} matched of {facts['n_detections']:,}")
            if not facts["sam_wins_every_body"] and every_body.search(line):
                problems.append(
                    f"{path.name}:{lineno}: claims SAM beats YOLO on every body, but the table "
                    "has Mars favouring YOLO")
    return problems


def main():
    if not T35.exists():
        print(f"skip: {T35} not present (gitignored data)")
        return 0
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
    print("\nall tracked documents agree with the table")
    return 0


if __name__ == "__main__":
    sys.exit(main())
