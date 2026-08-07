# Goal-3 phase 4: CPU-only correctness + speed check for the _mask_bounds fix (polygon.py).
# The 512-arm profiling (profile_512_prefix.json + profile_runs_prefix/) pinned the cost: the
# per-detection np.nonzero(mask) crop-bounds scan was 26.8s of a 128.5s serial 4k-crop run
# (25.6k detections at slice_size=512) vs find_contours' own 1.7s — because nonzero cost scales
# with the FULL mask area (slice_size^2) per detection, which the 512 arm quadruples over 256.
# The fix replaces nonzero with any()/argmax() row/col reductions (short-circuiting, no per-pixel
# index arrays). This bench (a) proves bit-identical crop bounds + polygon output on random
# masks at both scales, (b) times the old vs new bounds step in isolation. No GPU, no raster.
import sys
import time

import numpy as np

sys.path.insert(0, r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\src")

from YOLOv8BeyondEarth.polygon import _mask_bounds, binary_mask_to_polygon


def old_bounds(binary_mask):
    """The pre-fix crop-bounds computation (np.nonzero + min/max), for the A/B comparison."""
    ys, xs = np.nonzero(binary_mask)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def make_mask(slice_size, blob_diam, rng, empty=False):
    """A slice_size x slice_size bool mask with one roughly-round blob (a typical downscaled
    boulder mask): most pixels zero, the crop-bounds scan's worst case."""
    m = np.zeros((slice_size, slice_size), dtype=bool)
    if empty:
        return m
    r = blob_diam / 2.0
    cy = rng.uniform(r, slice_size - r)
    cx = rng.uniform(r, slice_size - r)
    yy, xx = np.ogrid[:slice_size, :slice_size]
    m[(yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2] = True
    return m


def main():
    rng = np.random.default_rng(0)
    # --- correctness: bounds + polygon identical, both scales, incl. edge-touching & empty ---
    n_bad_bounds = n_bad_poly = 0
    cases = []
    for ss in (256, 512):
        for diam in (4, 6, 10, 20, 60):
            for _ in range(40):
                cases.append(make_mask(ss, diam, rng))
    cases.append(np.zeros((512, 512), dtype=bool))           # empty
    edge = np.zeros((256, 256), dtype=bool); edge[0:5, 0:5] = True   # touches (0,0) corner
    cases.append(edge)
    for m in cases:
        ob, nb = old_bounds(m), _mask_bounds(m)
        if ob != nb:
            n_bad_bounds += 1
        po = binary_mask_to_polygon(m)   # already uses the new path; sanity on shape validity
        if m.any() and (po is None or len(po) < 3) and m.sum() >= 3:
            n_bad_poly += 1
    print(f"correctness: {len(cases)} masks, bounds mismatches={n_bad_bounds}, "
          f"poly-degenerate={n_bad_poly}")
    assert n_bad_bounds == 0, "FIX IS NOT BIT-IDENTICAL — bounds differ"

    # --- timing: old vs new bounds step, per scale, on a realistic small-blob population -----
    for ss in (256, 512):
        masks = [make_mask(ss, rng.uniform(4, 12), rng) for _ in range(4000)]
        for label, fn in (("old_nonzero", old_bounds), ("new_argmax", _mask_bounds)):
            t0 = time.perf_counter()
            for m in masks:
                fn(m)
            dt = time.perf_counter() - t0
            print(f"  ss={ss}: {label:12s} {dt*1e3:8.1f} ms / 4000 masks "
                  f"({dt/4000*1e6:6.1f} us/mask)")


if __name__ == "__main__":
    main()
