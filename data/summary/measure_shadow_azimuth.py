"""Measure each scene's cast-shadow direction and depth straight from the imagery.

Why this exists. Two claims in the summary notebook rest on illumination geometry: the sun axis drawn
on every rose, and the statement that the Moon scene is *weakly* shadowed while the Mars scene is
strongly shadowed (which is why the Moon panel reads as a low-illumination control). Both currently
come from metadata — HiRISE `.LBL` files and, for NAC, an ODE/CUMINDEX lookup that reports **incidence
9.25 deg**, i.e. near-zero-phase and almost shadow-free. A stretched grey crop *looked* shadowed to the
eye, which is exactly the situation where a number beats an impression.

So this measures it. For each human-labelled boulder (the labels give position and size without any
model in the loop), the mean DN is taken in an annulus just outside the boulder, in 24 azimuth bins,
normalised by that boulder's own local background. A cast shadow shows up as a dark lobe at a
consistent azimuth across boulders; its direction is the **anti-sun** direction and its depth measures
how strongly the scene is shadowed at all.

Outputs, per scene: the resultant (mean) shadow azimuth and its concentration R, the shadow depth
(1 - darkest-bin brightness), and the implied sun azimuth (shadow + 180) with its axial difference from
the metadata sun axis. That last number is the independent check on the metadata.

No GPU, no model. Run from inside YOLOv8-BeyondEarth/:
  C:\\Users\\brian\\anaconda3\\Scripts\\conda.exe run -n bouldernet python data/summary/measure_shadow_azimuth.py
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio

BASE = Path(r"c:\Users\brian\Documents\PhD\BoulderNet")
OUT = BASE / "YOLOv8-BeyondEarth" / "data" / "summary"
DE = BASE / "YOLOv8-BeyondEarth" / "data" / "solutionDE"
PRIEUR = BASE / "data_prieur" / "raw_data"
SAMV = BASE / "YOLOv8-BeyondEarth" / "data" / "sam2_validation"

NBINS = 24
R_IN, R_OUT = 0.65, 1.45      # annulus radii, in units of boulder equivalent radius
DPX_MIN = 8                   # only well-resolved boulders carry a measurable shadow lobe
MAX_BOULDERS = 4000           # per scene, for runtime; sampling is random and reported


def scenes():
    """(name, body, sun_axis_meta, source, crop paths, human shp) for the two labelled scenes."""
    # mars_sun.json stores the raw image-frame azimuths; the geographic axis is the difference
    # (project convention, see `hirise-subsolar-azimuth-frame`): 165.3414 - 270 = -104.66 -> 75.3.
    m = json.loads((SAMV / "mars_sun.json").read_text())["ESP_017355_2260_RED"]
    mars_axis = (m["sub_solar"] - m["north"]) % 360.0
    moon = json.loads((OUT / "moon_sun.json").read_text())
    return [
        dict(name="HiRISE ESP_017355_2260", body="mars", sun_axis=mars_axis,
             source="HiRISE LBL (SUB_SOLAR - NORTH)", incidence=m["incidence"],
             crops=sorted(DE.glob("hirise_rom*.tif")),
             human=PRIEUR / "mars/ESP_017355_2260/shp/ESP_017355_2260-boulder-mapping.shp"),
        dict(name="NAC M139694087LE", body="moon", sun_axis=moon["sun_axis"],
             source=f"ODE/CUMINDEX ({moon.get('ode_pdsid')})", incidence=moon.get("incidence"),
             crops=sorted(OUT.glob("moon_rom_*.tif")),
             human=PRIEUR / "moon/censorinus/shp/M139694087LE-censorinus-boulder-mapping.shp"),
    ]


def annulus_profile(arr, cy, cx, rpx):
    """Mean DN per azimuth bin in the annulus around (cy, cx), normalised by the annulus mean.

    Azimuth convention matches the rest of the project: 0 = North (up, decreasing row), 90 = East,
    clockwise. Returns ``None`` if the annulus leaves the array or is degenerate.
    """
    ro = int(np.ceil(R_OUT * rpx)) + 1
    y0, y1, x0, x1 = int(cy) - ro, int(cy) + ro + 1, int(cx) - ro, int(cx) + ro + 1
    if y0 < 0 or x0 < 0 or y1 > arr.shape[0] or x1 > arr.shape[1]:
        return None
    sub = arr[y0:y1, x0:x1].astype(float)
    if sub.size == 0 or not np.isfinite(sub).all():
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dy, dx = yy - cy, xx - cx
    r = np.hypot(dy, dx) / rpx
    ring = (r >= R_IN) & (r <= R_OUT) & (sub > 0)
    if ring.sum() < 3 * NBINS:
        return None
    az = np.degrees(np.arctan2(dx, -dy)) % 360.0          # North=0, East=90, clockwise
    idx = np.clip((az / (360.0 / NBINS)).astype(int), 0, NBINS - 1)
    prof = np.full(NBINS, np.nan)
    for b in range(NBINS):
        sel = ring & (idx == b)
        if sel.sum() >= 2:
            prof[b] = sub[sel].mean()
    if np.isnan(prof).any():
        return None
    m = prof.mean()
    return prof / m if m > 0 else None


def measure(scene, rng):
    hum = gpd.read_file(scene["human"])
    profiles, used = [], 0
    for path in scene["crops"]:
        with rasterio.open(path) as src:
            arr = src.read(1)
            T, crs = src.transform, src.crs
            res = abs(T.a)
            b = src.bounds
        h = hum.set_crs(crs, allow_override=True)          # never to_crs: bogus datum shift
        c = h.geometry.centroid
        sel = h[c.x.between(b.left, b.right) & c.y.between(b.bottom, b.top)]
        if sel.empty:
            continue
        cs = sel.geometry.centroid
        dpx = np.sqrt(sel.geometry.area.values / np.pi) * 2 / res
        for (x, y), d in zip(zip(cs.x.values, cs.y.values), dpx):
            if d < DPX_MIN:
                continue
            col, row = (~T) * (x, y)
            p = annulus_profile(arr, row, col, d / 2.0)
            if p is not None:
                profiles.append(p)
                used += 1
        if used >= MAX_BOULDERS:
            break
    if not profiles:
        return None
    P = np.vstack(profiles)
    mean_prof = P.mean(0)
    centers = (np.arange(NBINS) + 0.5) * (360.0 / NBINS)
    # Resultant of the *darkness* (1 - brightness) over full-circle azimuth: a cast shadow is a
    # one-sided lobe, so this is a directional (not axial) mean.
    w = np.clip(1.0 - mean_prof, 0, None)
    th = np.deg2rad(centers)
    sx, sy = (w * np.sin(th)).sum(), (w * np.cos(th)).sum()
    shadow_az = np.degrees(np.arctan2(sx, sy)) % 360.0
    R = float(np.hypot(sx, sy) / w.sum()) if w.sum() else np.nan
    depth = float(1.0 - mean_prof.min())
    contrast = float(mean_prof.max() - mean_prof.min())
    sun_from_image = (shadow_az + 180.0) % 360.0
    ax = abs(((sun_from_image - scene["sun_axis"] + 90.0) % 180.0) - 90.0)
    # Two conventions coexist upstream (HiRISE stores a full 0-360 direction, the NAC lookup already
    # reduced to 0-180), so both are written explicitly. `sun_axis_axial` is the one the roses use.
    return dict(scene=scene["name"], body=scene["body"], n_boulders=len(profiles),
                sun_axis_meta=round(float(scene["sun_axis"]), 1),
                sun_axis_axial=round(float(scene["sun_axis"]) % 180.0, 1),
                sun_axis_axial_from_image=round(float(sun_from_image % 180.0), 1),
                meta_source=scene["source"], incidence_meta=scene["incidence"],
                shadow_azimuth_measured=round(float(shadow_az), 1),
                shadow_concentration_R=round(R, 3), shadow_depth=round(depth, 3),
                ring_contrast=round(contrast, 3),
                sun_azimuth_from_image=round(float(sun_from_image), 1),
                axial_diff_vs_meta_deg=round(float(ax), 1),
                mean_profile=[round(v, 4) for v in mean_prof],
                bin_centers_deg=[round(v, 1) for v in centers])


def main():
    rng = np.random.default_rng(0)
    rows = [r for r in (measure(s, rng) for s in scenes()) if r]
    (OUT / "shadow_azimuth.json").write_text(json.dumps(
        dict(n_bins=NBINS, r_in=R_IN, r_out=R_OUT, dpx_min=DPX_MIN, scenes=rows), indent=2))
    print(f"{'scene':26s} {'n':>6s} {'depth':>6s} {'R':>5s} {'axis(img)':>10s} "
          f"{'axis(meta)':>11s} {'d_axial':>8s} {'inc':>6s}")
    print("-" * 88)
    for r in rows:
        inc = r["incidence_meta"]
        print(f"{r['scene'][:26]:26s} {r['n_boulders']:6d} {r['shadow_depth']:6.3f} "
              f"{r['shadow_concentration_R']:5.2f} {r['sun_axis_axial_from_image']:10.1f} "
              f"{r['sun_axis_axial']:11.1f} {r['axial_diff_vs_meta_deg']:8.1f} "
              f"{inc if inc is None else round(float(inc), 1):>6}")
    print("\nRead: `depth` is how dark the darkest azimuth bin around a boulder gets relative to its\n"
          "own ring mean (0 = no shadow). `d_axial` is the axial disagreement between the\n"
          "image-measured sun direction and the metadata sun axis — small values corroborate the\n"
          "metadata independently of any ephemeris.")


if __name__ == "__main__":
    main()
