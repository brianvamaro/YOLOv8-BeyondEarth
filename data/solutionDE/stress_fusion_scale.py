# Goal-3 phase 4: de-risk the full-scene fusion step. Full HiRISE fusion pools ~3.3M polygons
# across {512, 256, SR x2}; the likely bottleneck is fusion._greedy_suppress (a Python
# list-of-lists adjacency + loop over every intersecting pair). Rather than one full-scale RAM
# spike concurrent with the GPU detection passes, this measures the SCALING CURVE at safe sizes
# (<=800k polys, ~1GB, no tracemalloc) so we can extrapolate the empirical complexity to 3.3M
# and decide if the real fusion needs spatial chunking. CPU-only, low RAM by design.
import os
import sys
import time

import numpy as np
import geopandas as gpd
import shapely

sys.path.insert(0, r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\src")

from YOLOv8BeyondEarth.fusion import fuse_predictions, matched_recall, _candidate_pairs

try:
    import psutil
    _PROC = psutil.Process(os.getpid())
    def rss_gb():
        return _PROC.memory_info().rss / 1e9
except Exception:
    def rss_gb():
        return float("nan")


def synth_arm(n, scene_w, scene_h, diam_px_range, res, rng, overlap_frac=0.15):
    cx = rng.uniform(0, scene_w, n); cy = rng.uniform(0, scene_h, n)
    d = rng.uniform(*diam_px_range, n) * res
    n_dup = int(n * overlap_frac)
    if n_dup:
        src = rng.integers(0, n, n_dup)
        cx[:n_dup] = cx[src] + rng.normal(0, 0.5 * res, n_dup)
        cy[:n_dup] = cy[src] + rng.normal(0, 0.5 * res, n_dup)
        d[:n_dup] = d[src] * rng.uniform(0.8, 1.2, n_dup)
    r = d / 2.0
    return gpd.GeoDataFrame(
        {"score": rng.uniform(0.1, 0.95, n), "category_id": 0, "category_name": "boulder",
         "is_within_slice": rng.random(n) > 0.1},
        geometry=shapely.box(cx - r, cy - r, cx + r, cy + r), crs="EPSG:32633")


def build_pool(total, res, rng):
    """Three arms in the observed full-HiRISE proportions (512:256:SR ~ 0.21:0.34:0.45),
    scaled to `total`. Density is held at the real full-HiRISE value by shrinking the scene
    extent with `total` (so pair-count/detection — what drives _greedy_suppress — matches)."""
    W, H = 22490 * res, 85922 * res
    frac_full = total / 3_350_000.0
    W *= np.sqrt(frac_full); H *= np.sqrt(frac_full)          # keep areal density constant
    fr = {512: (0.21, (8, 60)), 256: (0.34, (4, 30)), (256, 2): (0.45, (3, 16))}
    return {k: synth_arm(max(1, int(total * f)), W, H, dr, res, rng) for k, (f, dr) in fr.items()}


def main():
    rng = np.random.default_rng(0)
    res = 0.5
    print(f"{'total':>10} {'n_pairs':>12} {'fuse_s':>8} {'parity_s':>9} {'rss_GB':>7} {'fused':>10}",
          flush=True)
    fits = []
    for total in (100_000, 300_000, 500_000, 800_000):
        per_scale = build_pool(total, res, rng)
        n = sum(len(g) for g in per_scale.values())
        # pair count drives the greedy loop — measure it directly
        boxes = np.concatenate([g.geometry.bounds.to_numpy(float) for g in per_scale.values()])
        ii, jj, _ = _candidate_pairs(boxes, 0.30, "diou")
        n_pairs = len(ii)
        t0 = time.time()
        fused, audit = fuse_predictions(per_scale, rule="bands", bands={256: (0, np.inf)},
                                        engulf_thr=0.7, native_res=res)
        fuse_s = time.time() - t0
        t0 = time.time()
        rec, _ = matched_recall(per_scale[256], fused, engulf_thr=0.7)
        parity_s = time.time() - t0
        print(f"{n:>10,} {n_pairs:>12,} {fuse_s:>8.2f} {parity_s:>9.2f} {rss_gb():>7.2f} "
              f"{len(fused):>10,}  parity={rec:.4f}", flush=True)
        fits.append((n, n_pairs, fuse_s))
        del per_scale, fused, audit

    # extrapolate to 3.35M assuming fuse_s ~ a*n_pairs + b*n (greedy loop + STRtree builds)
    ns = np.array([f[0] for f in fits], float)
    nps = np.array([f[1] for f in fits], float)
    ts = np.array([f[2] for f in fits], float)
    A = np.column_stack([nps, ns, np.ones_like(ns)])
    coef, *_ = np.linalg.lstsq(A, ts, rcond=None)
    pairs_per_det = nps[-1] / ns[-1]
    n_full = 3_350_000
    pred_pairs = pairs_per_det * n_full
    pred_s = float(coef @ [pred_pairs, n_full, 1.0])
    print(f"\npairs/detection ~ {pairs_per_det:.2f}; extrapolated full 3.35M: "
          f"~{pred_pairs/1e6:.1f}M pairs, ~{pred_s:.0f}s fuse", flush=True)
    print("VERDICT:", "flat fusion scales — no chunking needed" if pred_s < 300
          else "flat fusion too slow/heavy — prepare spatial chunking", flush=True)


if __name__ == "__main__":
    main()
