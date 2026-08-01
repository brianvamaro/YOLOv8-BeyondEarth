"""Development versus confirmation, kept apart (review item G3-11).

The same 20 HiRISE ROMs and the NAC overlap chose ``256pref_engulf`` *and* supplied the recall
gains reported for it. A number selected on a set and quoted on the same set is a development
number: it carries the selection, and it is optimistic by an amount nobody measured. That is not a
reason to throw it away — it is a reason to label it, and to reserve ground that the selection
never saw before the gain is called confirmed.

Two things live here. :func:`group_split` makes the reservation deterministic, reproducible and
recorded, so "held out" is a checkable claim rather than an assertion. :func:`labelled_result`
makes the label travel with the number, so a table cannot quietly present one as the other.
:func:`threshold_sensitivity` covers the other half of the item: the SFD completeness limit is read
off a roll-off threshold that was chosen, not derived, so the result has to be shown as a function
of that choice.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

SPLITS = ("development", "held_out")


def group_split(keys, holdout_fraction=0.3, seed=0, salt=""):
    """Assign each group to ``development`` or ``held_out``, deterministically and reproducibly.

    Assignment is by a hash of the *key*, not by a shuffled index, so adding or removing groups
    later does not reshuffle the ones already assigned — a split that moves when the dataset grows
    is not a held-out set. Returns ``(mapping, manifest)``; the manifest carries the parameters and
    a digest of the assignment so a later run can prove it used the same one.
    """
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError(f"holdout_fraction must be in (0, 1), got {holdout_fraction}")
    keys = [str(k) for k in keys]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate group keys — a group must land in exactly one split")

    mapping = {}
    for key in keys:
        digest = hashlib.sha256(f"{salt}|{seed}|{key}".encode()).hexdigest()
        u = int(digest[:16], 16) / float(1 << 64)
        mapping[key] = "held_out" if u < holdout_fraction else "development"

    payload = json.dumps(mapping, sort_keys=True, separators=(",", ":"))
    manifest = {"seed": seed, "salt": salt, "holdout_fraction": holdout_fraction,
                "n_groups": len(keys),
                "n_held_out": sum(v == "held_out" for v in mapping.values()),
                "assignment_sha256": hashlib.sha256(payload.encode()).hexdigest()[:16],
                "assignment": mapping}
    return mapping, manifest


def assign_split(frame, group_col, mapping, out_col="split"):
    """Attach the split label to every row, refusing to guess for an unassigned group."""
    unknown = set(frame[group_col].astype(str)) - set(mapping)
    if unknown:
        raise ValueError(f"groups with no split assignment: {sorted(unknown)[:5]}"
                         f"{' ...' if len(unknown) > 5 else ''}")
    out = frame.copy()
    out[out_col] = frame[group_col].astype(str).map(mapping)
    return out


def labelled_result(value, split, lo=None, hi=None, n_groups=None, note=None):
    """A reported number with its provenance welded on.

    ``split="development"`` is not a failure — most of the phase-3 work legitimately is
    development, and saying so costs nothing. Presenting it *unlabelled* is what the review
    objected to.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    return {"value": float(value), "split": split, "lo": lo, "hi": hi,
            "n_groups": n_groups, "note": note,
            "is_confirmatory": split == "held_out"}


def evaluate_by_split(frame, metric_fn, split_col="split", group_col=None):
    """Run ``metric_fn`` separately on each split and report both, never a pooled figure.

    Pooling development and held-out rows produces a number that is neither, and it is the easiest
    mistake to make once the labels exist.
    """
    rows = []
    for split in SPLITS:
        sub = frame[frame[split_col] == split]
        if len(sub) == 0:
            continue
        rows.append({"split": split, "n": int(len(sub)),
                     "n_groups": int(sub[group_col].nunique()) if group_col else None,
                     "value": float(metric_fn(sub)),
                     "is_confirmatory": split == "held_out"})
    return pd.DataFrame(rows)


def threshold_sensitivity(diameters_px, thresholds=None, counts=None):
    """How the completeness limit moves with the roll-off threshold that defines it.

    The SFD completeness limit is read off where the size distribution turns over, against a
    threshold that was chosen rather than derived. Quoting one limit hides that choice; this
    reports the limit as a function of it, which is what a sensitivity analysis is for
    (review item G3-11).

    ``diameters_px`` is the detected equivalent-diameter population. For each threshold — the
    fraction of the peak bin count below which a bin is called incomplete — the returned limit is
    the smallest diameter still above it.
    """
    d = np.asarray(diameters_px, dtype=float)
    d = d[np.isfinite(d) & (d > 0)]
    if len(d) == 0:
        return pd.DataFrame(columns=["threshold", "limit_px", "n_above_limit"])
    if thresholds is None:
        thresholds = (0.5, 0.6, 0.7, 0.8, 0.9)

    edges = np.geomspace(max(d.min(), 1e-6), d.max(), 25)
    hist, _ = np.histogram(d, bins=edges)
    centres = np.sqrt(edges[:-1] * edges[1:])
    peak_idx = int(np.argmax(hist))
    peak = hist[peak_idx]

    rows = []
    for t in thresholds:
        ok = np.flatnonzero(hist[:peak_idx + 1] >= t * peak)
        # walk down from the peak until a bin falls below the threshold; the limit is the last
        # bin that did not
        limit = float(centres[ok[0]]) if len(ok) else float(centres[peak_idx])
        rows.append({"threshold": float(t), "limit_px": limit,
                     "n_above_limit": int((d >= limit).sum())})
    out = pd.DataFrame(rows)
    out.attrs["peak_bin_px"] = float(centres[peak_idx])
    out.attrs["limit_spread_px"] = float(out.limit_px.max() - out.limit_px.min())
    return out
