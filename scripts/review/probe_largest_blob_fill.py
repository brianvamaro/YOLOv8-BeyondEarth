"""Prove the cropped ``largest_blob_fill`` is bit-identical, and measure what the crop buys.

The 2026-08-02/03 optimisation research measured ``largest_blob_fill`` at **1.97 ms of Solution G's
6.54 ms/boulder** — ~30% of G's CPU wall — because ``scipy.ndimage.label`` and ``binary_fill_holes``
both ran over the whole ``slice_size**2`` mask for a boulder occupying a few hundred pixels of it.

Two things need settling before that change can be trusted, and neither is a wall clock:

1. **Bit-identity**, adversarially. The hazards are the equal-area tie-break between components
   (``argmax`` takes the lowest label index, which assumes ``label`` numbers in first-encounter
   raster order) and hole classification at the crop border. Random masks hit those far harder than
   real SAM output does, which is nearly always one convex blob — so the check is a large randomised
   sweep with generators chosen to *manufacture* ties, holes touching the bbox border, components
   touching the array border, single-pixel components, and fractional (non-{0,1}) values.
2. **The speedup at production mask sizes**, which depends entirely on the blob-to-array ratio.

    conda run -n bouldernet python scripts/review/probe_largest_blob_fill.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from YOLOv8BeyondEarth.orientation_validation import largest_blob_fill   # noqa: E402
from YOLOv8BeyondEarth.polygon import _mask_bounds                       # noqa: E402


def reference_largest_blob_fill(mb):
    """The pre-2026-08-04 implementation, verbatim, as the equivalence oracle."""
    from scipy.ndimage import label, binary_fill_holes
    lab, n = label(mb)
    if n == 0:
        return None
    if n > 1:
        big = 1 + int(np.argmax([(lab == j).sum() for j in range(1, n + 1)]))
        mb = (lab == big).astype(np.uint8)
    return binary_fill_holes(mb).astype(np.uint8)


# --- mask generators -----------------------------------------------------------------------------

def gen_production_like(rng, size=256):
    """One compact blob of production size (5-15 native px) somewhere in a slice-sized mask."""
    m = np.zeros((size, size), np.uint8)
    r = rng.integers(2, 8)
    cy, cx = rng.integers(r + 1, size - r - 1, size=2)
    yy, xx = np.ogrid[:size, :size]
    m[((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r] = 1
    return m


def gen_multi_blob(rng, size=64):
    """Several blobs, deliberately often the *same* area so the tie-break is exercised."""
    m = np.zeros((size, size), np.uint8)
    r = int(rng.integers(1, 5))
    for _ in range(int(rng.integers(2, 6))):
        rr = r if rng.random() < 0.6 else int(rng.integers(1, 5))   # 60% exact ties
        cy, cx = rng.integers(rr, size - rr, size=2)
        yy, xx = np.ogrid[:size, :size]
        m[((yy - cy) ** 2 + (xx - cx) ** 2) <= rr * rr] = 1
    return m


def gen_holed(rng, size=48):
    """A ring, so the fill has real work, with the hole often abutting the tight bbox border."""
    m = np.zeros((size, size), np.uint8)
    r = int(rng.integers(4, size // 2 - 1))
    cy, cx = rng.integers(r, size - r, size=2)
    yy, xx = np.ogrid[:size, :size]
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2
    m[d2 <= r * r] = 1
    m[d2 <= max(1, r - int(rng.integers(1, 3))) ** 2] = 0
    if rng.random() < 0.5:                      # notch the ring open to the bbox edge
        m[cy, cx:] = 0
    return m


def gen_speckle(rng, size=32):
    """Bernoulli noise: many components, many ties, components touching the array border."""
    return (rng.random((size, size)) < rng.uniform(0.15, 0.6)).astype(np.uint8)


def gen_border_touching(rng, size=32):
    """Foreground on the array border, so the crop bbox coincides with the array edge."""
    m = gen_speckle(rng, size)
    m[0, :] = m[:, 0] = 0
    m[rng.integers(0, size), 0] = 1
    m[0, rng.integers(0, size)] = 1
    return m


def gen_fractional(rng, size=32):
    """Non-{0,1} values: ``label`` treats any nonzero as foreground and so must ``_mask_bounds``."""
    m = gen_speckle(rng, size).astype(np.float32)
    return m * rng.uniform(0.1, 0.9, size=m.shape).astype(np.float32)


def gen_degenerate(rng, size=8):
    """Empty, single-pixel, and all-foreground masks."""
    kind = int(rng.integers(0, 3))
    if kind == 0:
        return np.zeros((size, size), np.uint8)
    if kind == 1:
        m = np.zeros((size, size), np.uint8)
        m[rng.integers(0, size), rng.integers(0, size)] = 1
        return m
    return np.ones((size, size), np.uint8)


GENERATORS = {"production_like": gen_production_like, "multi_blob": gen_multi_blob,
              "holed": gen_holed, "speckle": gen_speckle,
              "border_touching": gen_border_touching, "fractional": gen_fractional,
              "degenerate": gen_degenerate}


# --- checks --------------------------------------------------------------------------------------

def equivalence_sweep(n_per_generator, seed=0):
    rng = np.random.default_rng(seed)
    report = {}
    for name, gen in GENERATORS.items():
        diffs, n_none, n_multi, n_tie = 0, 0, 0, 0
        first_bad = None
        for i in range(n_per_generator):
            m = gen(rng)
            want = reference_largest_blob_fill(m)
            got = largest_blob_fill(m)
            if want is None or got is None:
                n_none += 1
                if (want is None) != (got is None):
                    diffs += 1
                    first_bad = first_bad or {"i": i, "reason": "None mismatch"}
                continue
            if want.shape != got.shape or want.dtype != got.dtype or not np.array_equal(want, got):
                diffs += 1
                if first_bad is None:
                    first_bad = {"i": i, "shape": [list(want.shape), list(got.shape)],
                                 "dtype": [str(want.dtype), str(got.dtype)],
                                 "n_differing_px": int((want != got).sum())
                                 if want.shape == got.shape else None}
            # bookkeeping: how often did the sweep actually reach the risky branches?
            from scipy.ndimage import label
            lab, n = label(m)
            if n > 1:
                n_multi += 1
                counts = np.bincount(lab.ravel())[1:]
                if (counts == counts.max()).sum() > 1:
                    n_tie += 1
        report[name] = {"n": n_per_generator, "differing": diffs, "returned_none": n_none,
                        "multi_component": n_multi, "equal_area_ties": n_tie,
                        "first_bad": first_bad}
        print(f"{name:<17} n={n_per_generator:<7} diffs={diffs:<4} "
              f"multi={n_multi:<7} ties={n_tie:<7} none={n_none}")
    return report


def timing(size, radius, reps, cases):
    """Wall per call at a production-like blob-to-array ratio, old vs new."""
    rng = np.random.default_rng(7)
    masks = []
    for _ in range(cases):
        m = np.zeros((size, size), np.uint8)
        cy, cx = rng.integers(radius + 1, size - radius - 1, size=2)
        yy, xx = np.ogrid[:size, :size]
        m[((yy - cy) ** 2 + (xx - cx) ** 2) <= radius * radius] = 1
        masks.append(m)

    def bench(fn):
        fn(masks[0])                                   # warm the scipy import / first-call cost
        best = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            for m in masks:
                fn(m)
            best = min(best, (time.perf_counter() - t0) / len(masks))
        return best * 1e3                              # ms per mask

    old = bench(reference_largest_blob_fill)
    new = bench(largest_blob_fill)
    blob_px = int(np.count_nonzero(masks[0]))
    return {"mask_size": size, "blob_radius": radius, "blob_px": blob_px,
            "blob_fraction_of_array": round(blob_px / size ** 2, 5),
            "old_ms": round(old, 4), "new_ms": round(new, 4),
            "speedup": round(old / new, 2) if new else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-generator", type=int, default=30_000)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--cases", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print("=== equivalence (old implementation as oracle) ===")
    equivalence = equivalence_sweep(args.n_per_generator)
    total = sum(v["n"] for v in equivalence.values())
    total_diff = sum(v["differing"] for v in equivalence.values())
    total_ties = sum(v["equal_area_ties"] for v in equivalence.values())
    print(f"\n{total} masks, {total_diff} differing, {total_ties} reached an equal-area tie")

    print("\n=== timing (production-like blob in a slice-sized mask) ===")
    timings = []
    for size, radius in ((256, 4), (256, 8), (256, 24), (128, 4), (64, 4), (512, 8)):
        t = timing(size, radius, args.reps, args.cases)
        timings.append(t)
        print(f"{size:>4}^2 r={radius:<3} blob={t['blob_px']:<5} "
              f"({t['blob_fraction_of_array']*100:5.2f}% of array)  "
              f"old={t['old_ms']:.3f} ms  new={t['new_ms']:.3f} ms  {t['speedup']}x")

    record = {"equivalence": equivalence, "total_masks": total, "total_differing": total_diff,
              "total_equal_area_ties": total_ties, "timings": timings,
              "note": ("timings are CPU-only and therefore not affected by the GPU power state; "
                       "the 1.97 ms/boulder figure they are compared against came from the "
                       "2026-08-02/03 research profile of Solution G")}
    out = Path(args.out) if args.out else REPO / "data" / "review" / "largest_blob_fill_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    print(f"\nwrote {out}")
    if total_diff:
        raise SystemExit(f"NOT bit-identical: {total_diff} differing outputs")


if __name__ == "__main__":
    main()
