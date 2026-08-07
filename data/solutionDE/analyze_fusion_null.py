# Goal-3 phase 3: analysis half of the §4c fusion null (plan §Validation 1; GPU half =
# run_fusion_null.py). Loads the per-scale detections on the content-free fields (true
# orientations uniformly random by construction), fuses under all three rules, and checks the
# standing gate: every fused anisotropy metric (cardinal grid-frac, diagonal-frac, 135/45) must
# sit inside its input arms' envelope (min-max across arms, with bootstrap CIs) — fusion must
# not manufacture orientation structure the inputs don't have. Composition shift toward the
# finest arm is expected and allowed. Writes goal3_null.csv.
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

sys.path.insert(0, r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\src")
from YOLOv8BeyondEarth.fusion import fuse_predictions, pool_predictions
from YOLOv8BeyondEarth.orientation import (attach_orientations, grid_fraction,
                                           diagonal_fraction, asymmetry_ratio)

DE = Path(r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\data\solutionDE")
RES = 0.5
SEEDS = [200, 201, 202, 203]
SCALES = [512, 256, (256, 2)]
RULES = ("scale_aware", "score", "bands")
PROVISIONAL_BANDS = {(256, 2): (0.0, 10.0), 256: (10.0, 40.0), 512: (40.0, np.inf)}


def nms_gpkg(seed, ss, sr):
    tag = f"{ss}sr{sr}" if sr > 1 else f"{ss}"
    return (DE / "preds" / f"null_seed{seed}" /
            (f"null_field_seed{seed}-predictions-ct-010-ss-{tag}-is-1024-ov-020"
             "-downscaled-mask-nms.gpkg"))


def boot_ci(a, fn, nb=1000, seed=0):
    a = np.asarray(a)
    if len(a) < 10:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    stats = [fn(a[rng.integers(0, len(a), len(a))]) for _ in range(nb)]
    return tuple(np.percentile(stats, [2.5, 97.5]))


def metric_row(source, a):
    gl, gh = boot_ci(a, grid_fraction)
    dl, dh = boot_ci(a, diagonal_fraction)
    rl, rh = boot_ci(a, asymmetry_ratio)
    return dict(source=source, n=len(a),
                grid_frac=grid_fraction(a) if len(a) else np.nan, grid_lo=gl, grid_hi=gh,
                diag_frac=diagonal_fraction(a) if len(a) else np.nan, diag_lo=dl, diag_hi=dh,
                r135=asymmetry_ratio(a) if len(a) else np.nan, r135_lo=rl, r135_hi=rh)


if __name__ == "__main__":
    # pool the 4 seed fields per arm (one detection population per arm; §4c did the same)
    per_arm = {}
    for key in SCALES:
        ss, sr = key if isinstance(key, tuple) else (key, 1)
        frames = [gpd.read_file(nms_gpkg(seed, ss, sr)) for seed in SEEDS]
        # shift each seed's field apart so pooled fusion can't cluster across seeds
        shifted = []
        for i, g in enumerate(frames):
            g = g.copy()
            g["geometry"] = g.geometry.translate(xoff=i * 1e6)
            shifted.append(g)
        per_arm[key] = gpd.GeoDataFrame(pd.concat(shifted, ignore_index=True),
                                        geometry="geometry", crs=frames[0].crs)
        print(f"arm {key}: {len(per_arm[key])} detections over {len(SEEDS)} fields", flush=True)

    pool = pool_predictions(per_arm)
    pool["pool_index"] = np.arange(len(pool))
    ori = attach_orientations(pool, res=RES, n_workers=8)

    rows = []
    for key in SCALES:
        a = ori[ori.scale_key == str(key)].angle180.dropna().values
        rows.append(metric_row(f"input_{key}", a))
    for rule in RULES:
        kw = dict(bands=PROVISIONAL_BANDS, native_res=RES) if rule == "bands" else {}
        fused, _ = fuse_predictions(per_arm, rule=rule, **kw)
        a = ori[ori.pool_index.isin(fused.pool_index)].angle180.dropna().values
        row = metric_row(f"fused_{rule}", a)
        comp = fused.scale_key.value_counts()
        row["composition"] = "/".join(f"{comp.get(str(k), 0)}" for k in SCALES)
        rows.append(row)

    out = pd.DataFrame(rows)
    # gate: fused point estimates inside the input min-max envelope per metric
    inputs = out[out.source.str.startswith("input_")]
    for m in ("grid_frac", "diag_frac", "r135"):
        lo, hi = inputs[m].min(), inputs[m].max()
        out[f"{m}_in_envelope"] = out[m].between(lo, hi) | out.source.str.startswith("input_")
    out.to_csv(DE / "goal3_null.csv", index=False)
    print(out.to_string(index=False))
    fused_rows = out[out.source.str.startswith("fused_")]
    ok = fused_rows[[c for c in out.columns if c.endswith("_in_envelope")]].all().all()
    print(f"\nGATE {'PASS' if ok else 'CHECK CI OVERLAP'}: fused metrics vs input envelope")
