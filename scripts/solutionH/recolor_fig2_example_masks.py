"""Regenerate the Test-3c 'diagonal example masks' figure (Fig 2 of the orientation results
summary) with the document color palette.

The original is produced deep in notebooks/HiRISE_Test3_different_model (§3c II). All heavy compute
(YOLO detect + SAM2 re-mask per example) is disk-cached under data/test3_cache, so this reproduces
the exact same 12 panels WITHOUT a GPU by loading that cache directly by its known key. Only the
outline/axis colors change, to match the rest of the summary doc:
    YOLO = blue, SAM2 = green, human = black (the reference), MRCNN = orange.
(In the original, SAM2 was red and human was green — inconsistent with red=OBB-d0 elsewhere.)

The color literals in the notebook source were updated to match, so a future notebook rebuild
produces the same thing; this script just refreshes the committed PNG now.

Run from the repo root: python YOLOv8-BeyondEarth/scripts/solutionH/recolor_fig2_example_masks.py
"""
import os
import pickle
import sys

import matplotlib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
INNER = os.path.join(REPO, "YOLOv8-BeyondEarth")
DATA = os.path.join(INNER, "data")
PRIEUR = os.path.join(REPO, "data_prieur", "raw_data")
INF = os.path.join(REPO, "data_prieur", "maskrcnn_runs", "test3_boulder2024", "inference")
CACHE3 = os.path.join(DATA, "test3_cache")
T3CT = os.path.join(CACHE3, "t3c_treatment")
RAS_T = os.path.join(DATA, "mars_hirise", "raster", "ESP_017355_2260_RED.tif")
MRCNN_T = os.path.join(INF, "treatment-mrcnn-ss500-native.gpkg")
HUM_SHP = os.path.join(PRIEUR, "mars", "ESP_017355_2260", "shp",
                       "ESP_017355_2260-boulder-mapping.shp")
ROM_SHP = os.path.join(PRIEUR, "mars", "ESP_017355_2260", "shp", "ESP_017355_2260-ROM.shp")
OUT = os.path.join(REPO, "docs", "figures", "test3", "t3c_example_masks_diag.png")

RES_T = 0.5
SCALE3C = {"yolo": 1.0, "sam2": 1.0}
# the 12-panel selection, verbatim from the cached key (9 near-135° + 3 near-45°, size-spanning):
PICKD = [4799, 4114, 550, 4698, 567, 2102, 935, 1246, 4474, 3221, 481, 4407]
MASK_CACHE = os.path.join(CACHE3, "t3c_example_masks_diag_" + "-".join(map(str, PICKD)) + ".pkl")

# document palette
C_YOLO, C_SAM, C_HUMAN, C_MRCNN = "tab:blue", "tab:green", "k", "tab:orange"


def main():
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    import rasterio.windows
    import geopandas as gpd
    from scipy.spatial import cKDTree
    from pyogrio import read_dataframe as rdf
    from skimage.measure import find_contours as fc
    from YOLOv8BeyondEarth.orientation import ellipse_angle180, axial_distance
    from YOLOv8BeyondEarth.orientation_validation import angles_from_masks

    # --- reconstruct rom_t (cached CSVs), hum_t (shapefile), and the m3ct crossmatch -------------
    rom_t = pd.concat([pd.read_csv(os.path.join(T3CT, f)) for f in sorted(os.listdir(T3CT))
                       if f.startswith("rom") and f.endswith(".csv")
                       and os.path.getsize(os.path.join(T3CT, f)) > 0], ignore_index=True)
    with rasterio.open(RAS_T) as src:
        T = src.transform
    hum_t = gpd.read_file(HUM_SHP)
    rom_poly = gpd.read_file(ROM_SHP)
    hum_t = hum_t[hum_t.geometry.centroid.within(rom_poly.union_all())].copy()
    hum_t["_area"] = hum_t.geometry.area
    hum_t["_dia"] = 2 * np.sqrt(hum_t["_area"].to_numpy() / np.pi)

    xs, ys = T * ((rom_t.col0 + rom_t.cx).to_numpy(), (rom_t.row0 + rom_t.cy).to_numpy())
    tree = cKDTree(np.c_[xs, ys])
    hc = np.c_[hum_t.geometry.centroid.x.to_numpy(), hum_t.geometry.centroid.y.to_numpy()]
    dist, idx = tree.query(hc, k=1)
    ok = dist <= np.maximum(hum_t["_dia"].to_numpy() / 2, 1.0)
    m3ct = pd.DataFrame({"ref_area": hum_t["_area"].to_numpy()[ok],
                         "ref_dia": hum_t["_dia"].to_numpy()[ok],
                         "det_idx": idx[ok]})

    with open(MASK_CACHE, "rb") as f:
        exD = pickle.load(f)
    print(f"loaded {len(exD)} cached example masks; {len(PICKD)} panels; m3ct n={len(m3ct)}")
    if not os.path.exists(MRCNN_T):
        raise FileNotFoundError(f"MRCNN treatment gpkg missing: {MRCNN_T}")
    n_mrcnn = 0

    def axis_line(ax, cx, cy, ang, L, color):
        dx, dy = np.sin(np.radians(ang)) * L / 2, -np.cos(np.radians(ang)) * L / 2
        ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], color=color, lw=1.3, ls="--", alpha=.95)

    fig, axes = plt.subplots(3, 4, figsize=(14, 11.6))
    with rasterio.open(RAS_T) as src:
        for ax, i in zip(axes.ravel(), PICKD):
            r = m3ct.loc[i]
            det = rom_t.iloc[int(r.det_idx)]
            gx, gy = det.col0 + det.cx, det.row0 + det.cy
            pad = max(int(r.ref_dia / RES_T), 12)
            c0, r0 = int(gx) - pad, int(gy) - pad
            patch = src.read(1, window=rasterio.windows.Window(c0, r0, 2 * pad, 2 * pad))
            ax.imshow(patch, cmap="gray", vmin=patch.min(), vmax=patch.max())
            xm, ym = src.transform * (gx, gy)
            L = 1.5 * r.ref_dia / RES_T
            angs = {}
            bbox = (xm - pad * RES_T, ym - pad * RES_T, xm + pad * RES_T, ym + pad * RES_T)
            try:
                g = rdf(MRCNN_T, bbox=bbox)
            except Exception:
                g = []
            if len(g):
                g = g.iloc[[np.argmin(np.hypot(g.geometry.centroid.x - xm,
                                               g.geometry.centroid.y - ym))]]
                if np.hypot(g.geometry.centroid.x.iloc[0] - xm,
                            g.geometry.centroid.y.iloc[0] - ym) <= max(r.ref_dia / 2, 2.0):
                    geom = g.geometry.iloc[0]
                    xy = np.asarray(geom.exterior.coords)
                    px = (~src.transform) * (xy[:, 0], xy[:, 1])
                    ax.plot(px[0] - c0, px[1] - r0, color=C_MRCNN, lw=1.5, alpha=.7, label="MRCNN")
                    am, _asp = ellipse_angle180(geom, RES_T)
                    n_mrcnn += 1
                    if np.isfinite(am):
                        mpx = (~src.transform) * (geom.centroid.x, geom.centroid.y)
                        axis_line(ax, mpx[0] - c0, mpx[1] - r0, am, L, C_MRCNN)
                        angs["M"] = am
            cand = hum_t[np.hypot(hum_t.geometry.centroid.x - xm, hum_t.geometry.centroid.y - ym)
                         <= max(r.ref_dia / 2, 1.0)]
            if len(cand):
                geom = cand.iloc[0].geometry
                xy = np.asarray(geom.exterior.coords)
                px = (~src.transform) * (xy[:, 0], xy[:, 1])
                ax.plot(px[0] - c0, px[1] - r0, color=C_HUMAN, lw=1.5, alpha=.85, label="human")
                ah, _asp = ellipse_angle180(geom, RES_T)
                if np.isfinite(ah):
                    hpx = (~src.transform) * (geom.centroid.x, geom.centroid.y)
                    axis_line(ax, hpx[0] - c0, hpx[1] - r0, ah, L, C_HUMAN)
                    angs["H"] = ah
            e = exD.get(i)
            if e is not None:
                tc0, tr0 = e["tile_origin"]
                for key_, color, lab, tag in (("yolo", C_YOLO, "YOLO", "Y"),
                                              ("sam2", C_SAM, "SAM2", "S")):
                    mkk = e[key_]
                    if mkk is None:
                        continue
                    sc = mkk.shape[0] / 256.0
                    cont = max(fc(mkk, .5), key=len)
                    ax.plot(cont[:, 1] / sc + tc0 - c0, cont[:, 0] / sc + tr0 - r0,
                            color=color, lw=1.5, alpha=.85, label=lab)
                    aa = angles_from_masks([mkk]).angle180.iloc[0]
                    if np.isfinite(aa):
                        ys_, xs_ = np.nonzero(mkk)
                        axis_line(ax, xs_.mean() / sc + tc0 - c0, ys_.mean() / sc + tr0 - r0,
                                  aa, L, color)
                        angs[tag] = aa
            ttl = f"{r.ref_dia:.1f} m" + "".join(
                f"  {k}{angs[k]:.0f}°" for k in ("H", "Y", "S", "M") if k in angs)
            ax.set_title(ttl, fontsize=8, pad=4)
            ax.set_xticks([]), ax.set_yticks([])
    hl = {}
    for ax in axes.ravel():
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            hl.setdefault(ll, hh)
    order = [l for l in ("human", "YOLO", "SAM2", "MRCNN") if l in hl]
    fig.legend([hl[l] for l in order], order, loc="lower center", ncol=4, fontsize=10)
    fig.suptitle("§3c II: boulders whose YOLO angle sits on a pixel diagonal — outlines + measured "
                 "long-axis lines (dashed) per source", fontsize=11)
    fig.tight_layout(rect=(0, .04, 1, 0.985))
    fig.subplots_adjust(hspace=0.28)
    if n_mrcnn == 0:
        raise RuntimeError("no MRCNN outline matched any panel — check MRCNN_T / bbox filtering")
    print(f"MRCNN outlines drawn on {n_mrcnn}/{len(PICKD)} panels")
    fig.savefig(OUT, dpi=150)
    print("wrote:", OUT)


if __name__ == "__main__":
    main()
