"""Calibrate the synthetic cast-shadow model against a REAL HiRISE scene at known incidence.

Component A of docs/illumination_artifact_exploration_plan.md needs the synthetic shadow to match
real Mars boulders, not the old hand-set constants (which under-modelled: T1's shadow arm shifted
f_lower only +0.016 vs the real ~0.05). This measures, on a sample of well-resolved boulders in the
treatment scene ESP_017355_2260 (INCIDENCE 42.135 deg; SUB_SOLAR 165.341, NORTH_AZIMUTH 270 ->
geographic sun-from 255.34 deg, so cast shadows point toward bearing 75.34 deg), the three numbers
the `render_boulder_field(inc_deg=...)` model consumes:

    cap         = bright-rock DN above local background (the ellipse fill is bg + cap),
    shadow      = darkest cast-shadow DN *below* background, signed negative (fill is bg + shadow),
    height_frac = cast-shadow length / (equiv_diameter * tan(incidence)),  the assumed height/diam.

Method (per boulder, from the raw raster + the skimage prediction gpkg): local background from an
annulus; cap contrast from a bright percentile inside the centre; then a bilinear intensity profile
along the anti-sun ray, from which the shadow depth (bg - min) and the contiguous dark-run length
are read. Robust medians over the sample set the calibration; an IQR is reported for spread.

Writes shadow_calibration.json (medians + spread + full provenance) for the sweep driver, and a QA
figure (sample crops with the anti-sun ray + the median ray profile). READ-ONLY on the scene. Run
from inside YOLOv8-BeyondEarth/ in the bouldernet env.
"""
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[2]                    # .../YOLOv8-BeyondEarth
sys.path.insert(0, str(BASE / "src"))

RASTER = BASE / "data" / "mars_hirise" / "raster" / "ESP_017355_2260_RED.tif"
GPKG = (BASE / "data" / "mars_hirise" / "predictions_new" /
        "ESP_017355_2260_RED-predictions-skimage-mask-nms.gpkg")
OUT = Path(__file__).resolve().parent / "shadow_calibration.json"
FIG = BASE.parent / "docs" / "figures" / "illumination" / "shadow_calibration.png"

INCIDENCE = 42.135349
SUB_SOLAR = 165.341400
NORTH_AZIMUTH = 270.0
RES = 0.5                                                      # m/px

# geographic sun-FROM azimuth; cast shadow points opposite (anti-sun bearing), in the image N-up
# frame (North = -row, East = +col), matching render_boulder_field's convention.
SUN_FROM = (SUB_SOLAR - NORTH_AZIMUTH) % 360.0                 # 255.34
ANTISUN = (SUN_FROM + 180.0) % 360.0                           # 75.34
SDX, SDY = np.sin(np.deg2rad(ANTISUN)), -np.cos(np.deg2rad(ANTISUN))   # (col,row) toward shadow

N_SAMPLE = 1200
DPX_LO, DPX_HI = 8.0, 40.0                                    # well-resolved, not huge
MIN_DEPTH_DN = 6.0                                           # skip boulders with no real shadow
SEED = 7


def main():
    import geopandas as gpd
    import rasterio
    from scipy.ndimage import map_coordinates
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(SEED)
    gdf = gpd.read_file(GPKG)
    area = gdf.geometry.area.to_numpy()                       # m^2 (projected, metres)
    diam_m = 2.0 * np.sqrt(np.clip(area, 1e-9, None) / np.pi)
    diam_px = diam_m / RES
    cent = gdf.geometry.centroid
    cx_m, cy_m = cent.x.to_numpy(), cent.y.to_numpy()

    keep = np.where((diam_px >= DPX_LO) & (diam_px <= DPX_HI))[0]
    rng.shuffle(keep)

    caps, depths, hfracs, prof_stack = [], [], [], []
    n_used, n_seen = 0, 0
    examples = []                                             # (crop, r0, c0, rad, profile)
    with rasterio.open(RASTER) as ds:
        H, W = ds.height, ds.width
        for idx in keep:
            if n_used >= N_SAMPLE:
                break
            n_seen += 1
            rad = 0.5 * float(diam_px[idx])
            r0, c0 = ds.index(cx_m[idx], cy_m[idx])           # centroid -> (row,col)
            half = int(np.ceil(6.5 * rad))
            r_lo, r_hi = r0 - half, r0 + half + 1
            c_lo, c_hi = c0 - half, c0 + half + 1
            if r_lo < 0 or c_lo < 0 or r_hi > H or c_hi > W:
                continue
            win = rasterio.windows.Window(c_lo, r_lo, c_hi - c_lo, r_hi - r_lo)
            crop = ds.read(1, window=win).astype(float)
            if crop.size == 0 or not np.isfinite(crop).all():
                continue
            cyc, cxc = r0 - r_lo, c0 - c_lo                   # centre in crop coords
            yy, xx = np.mgrid[0:crop.shape[0], 0:crop.shape[1]]
            dist = np.hypot(yy - cyc, xx - cxc)
            bg_ring = (dist >= 3.5 * rad) & (dist <= 6.0 * rad)
            if bg_ring.sum() < 30:
                continue
            bg = float(np.median(crop[bg_ring]))
            inner = dist <= 0.7 * rad
            if inner.sum() < 4:
                continue
            cap_dn = float(np.percentile(crop[inner], 80)) - bg

            # anti-sun intensity profile (bilinear), distances in px
            dd = np.arange(0.0, 6.0 * rad, 0.5)
            rs = cyc + SDY * dd
            csamp = cxc + SDX * dd
            prof = map_coordinates(crop, [rs, csamp], order=1, mode="nearest")
            seg = (dd >= rad) & (dd <= 5.0 * rad)             # beyond the cap
            if seg.sum() < 3:
                continue
            shadow_min = float(prof[seg].min())
            depth = bg - shadow_min
            if depth < MIN_DEPTH_DN or cap_dn <= 0:
                continue

            # contiguous dark run beyond the boulder edge: length of shadow
            thr = bg - 0.4 * depth
            length = 0.0
            started = False
            for d, p in zip(dd, prof):
                if d < rad:
                    continue
                if p < thr:
                    length = d - rad + 0.5
                    started = True
                elif started:
                    break                                     # end of the contiguous shadow
            hf = length / (2.0 * rad * np.tan(np.deg2rad(INCIDENCE)))

            caps.append(cap_dn)
            depths.append(depth)
            hfracs.append(hf)
            # resample profile to a fixed grid in units of rad for stacking (0..5 rad)
            grid = np.linspace(0.0, 5.0, 60)
            prof_rad = np.interp(grid, dd / rad, (prof - bg) / max(cap_dn, 1e-6))
            prof_stack.append(prof_rad)
            if len(examples) < 6 and depth > 15:
                examples.append((crop.copy(), cyc, cxc, rad, prof, dd, bg))
            n_used += 1

    caps = np.array(caps); depths = np.array(depths); hfracs = np.array(hfracs)

    def med_iqr(a):
        return (float(np.median(a)), float(np.percentile(a, 25)), float(np.percentile(a, 75)))

    cap_m, cap_lo, cap_hi = med_iqr(caps)
    dep_m, dep_lo, dep_hi = med_iqr(depths)
    hf_m, hf_lo, hf_hi = med_iqr(np.clip(hfracs, 0, 3))

    result = {
        "scene": "ESP_017355_2260_RED",
        "incidence_deg": INCIDENCE, "sun_from_deg": SUN_FROM, "antisun_bearing_deg": ANTISUN,
        "res_m_per_px": RES, "n_used": int(n_used), "n_seen": int(n_seen),
        "dpx_range": [DPX_LO, DPX_HI],
        "cap": {"median": round(cap_m, 2), "iqr": [round(cap_lo, 2), round(cap_hi, 2)]},
        "shadow": {"median": round(-dep_m, 2), "iqr": [round(-dep_hi, 2), round(-dep_lo, 2)]},
        "height_frac": {"median": round(hf_m, 3), "iqr": [round(hf_lo, 3), round(hf_hi, 3)]},
        "note": ("cap = bright-rock DN above local bg; shadow = signed DN below bg (negative); "
                 "height_frac = shadow_length / (equiv_diam * tan(incidence)). Feed these to "
                 "render_boulder_field(inc_deg=incidence, cap=cap, shadow=shadow, "
                 "height_frac=height_frac, sun_az_deg=sun_from)."),
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    # --- QA figure ---
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(15, 8))
    for i, (crop, cyc, cxc, rad, prof, dd, bg) in enumerate(examples[:6]):
        ax = fig.add_subplot(2, 4, i + 1)
        ax.imshow(crop, cmap="gray")
        ax.plot([cxc, cxc + SDX * 4 * rad], [cyc, cyc + SDY * 4 * rad], "r-", lw=1.4)
        ax.plot(cxc, cyc, "c+", ms=8)
        ax.set_title(f"boulder {i} (rad~{rad:.1f}px)", fontsize=8); ax.axis("off")
    axp = fig.add_subplot(2, 4, 7)
    if prof_stack:
        ps = np.array(prof_stack)
        g = np.linspace(0, 5, 60)
        axp.plot(g, np.median(ps, 0), "b-", lw=2, label="median")
        axp.fill_between(g, np.percentile(ps, 25, 0), np.percentile(ps, 75, 0), alpha=0.25)
        axp.axhline(0, color="k", lw=0.6); axp.axvline(1, color="gray", ls=":", lw=0.8)
    axp.set_xlabel("distance along anti-sun / boulder-radius")
    axp.set_ylabel("(DN - bg) / cap"); axp.set_title("median anti-sun ray profile", fontsize=9)
    axp.legend(fontsize=8)
    axh = fig.add_subplot(2, 4, 8)
    axh.hist(np.clip(hfracs, 0, 2), bins=30, color="tab:orange")
    axh.axvline(hf_m, color="k", lw=1.2)
    axh.set_title(f"height_frac (median {hf_m:.2f})", fontsize=9); axh.set_xlabel("height_frac")
    fig.suptitle(f"Shadow calibration — {result['scene']} inc={INCIDENCE:.1f} "
                 f"cap={cap_m:.0f} shadow={-dep_m:.0f} hf={hf_m:.2f} (n={n_used})", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG, dpi=120, bbox_inches="tight")
    print("wrote", FIG)


if __name__ == "__main__":
    main()
