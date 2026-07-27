"""Build every figure for `notebooks/Results_summary` from cached artifacts. No GPU.

One entry point, one house style, one place where each figure's data provenance is written down.
Figures land in `docs/figures/summary_v2/` (a new directory, so the archive document's figures keep
working) and every figure also returns the numbers it computed, which are collected into
`figure_stats.json` — that file is the claim-audit backbone: any number quoted in the notebook must
appear there, traceable to a cached input.

Usage (from the repo root or anywhere):
  python YOLOv8-BeyondEarth/scripts/summary/make_summary_figs.py            # all figures
  python .../make_summary_figs.py --only fig04a,fig10                       # iterate on a few
  python .../make_summary_figs.py --list                                    # names + inputs

Style rules are enforced by `YOLOv8BeyondEarth.figstyle`, not by convention here: every rose and
histogram states its minimum aspect ratio and minimum size, multi-panel figures share axes, the sun
axis is drawn in ink rather than a series colour, and outlines over imagery get a contrasting casing
(series colours sit at 1.1-2.2:1 against mid-grey regolith, i.e. invisible without one).
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "YOLOv8-BeyondEarth"
sys.path.insert(0, str(ROOT / "src"))

import matplotlib                                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402

from YOLOv8BeyondEarth import figstyle as fs                           # noqa: E402
from YOLOv8BeyondEarth.orientation import (asymmetry_ratio, axial_distance,      # noqa: E402
                                           axis_alignment_fraction, bootstrap_peak_ci,
                                           diagonal_fraction, grid_fraction, half_plane_fraction,
                                           refine_peak)

DATA = ROOT / "data"
SUM = DATA / "summary"
SAMV = DATA / "sam2_validation"
DE = DATA / "solutionDE"
SG = DATA / "solutionG"
SH = DATA / "solutionH"
PRIEUR = REPO / "data_prieur" / "raw_data"
FIG = REPO / "docs" / "figures" / "summary_v2"

ASPECT_MIN = 1.35     # project standard: below this a mask has no meaningful long axis
DPX_MIN = 8           # project standard well-resolved cut
UNIF = fs.UNIFORM_FRACTION

REGISTRY = {}

# --- measured provenance -----------------------------------------------------------------------
# Every figure's inputs used to be a hand-written string in its decorator. That is a claim, not a
# fact: change a figure's data source without updating the decorator and the document confidently
# prints a false provenance line — worse than printing nothing. So all reads go through the helpers
# below, which record what was actually opened. `build()` returns that record, and the CLI checks it
# against the declared list and complains if a real input was never declared.
_STATE = {"fig": None, "opened": [], "save_png": True}


def _record(path):
    p = Path(path)
    try:
        rel = str(p.resolve().relative_to(REPO))
    except ValueError:
        rel = str(p)
    entry = dict(path=rel, bytes=p.stat().st_size if p.exists() else None,
                 mtime=(pd.Timestamp(p.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M")
                        if p.exists() else None))
    if not any(e["path"] == rel for e in _STATE["opened"]):
        _STATE["opened"].append(entry)
    return entry


def read_csv(path, **kw):
    """`pd.read_csv` that records the file (and its row count) as a measured input."""
    e = _record(path)
    df = pd.read_csv(path, **kw)
    e["rows"] = len(df)
    return df


def load_json(path):
    """`load_json(Path)` that records the file as a measured input."""
    _record(path)
    return json.loads(Path(path).read_text())


def load_pickle(path):
    """`pickle.load` that records the file as a measured input."""
    _record(path)
    with open(path, "rb") as fh:
        return pickle.load(fh)


def figure(name, inputs, title):
    def deco(fn):
        REGISTRY[name] = dict(fn=fn, inputs=inputs, title=title)
        return fn
    return deco


def save(fig, name):
    """Hand the finished figure back to whoever asked for it.

    Two callers with different needs, hence the indirection: the CLI wants a PNG on disk and the
    figure closed (16 open figures is a lot of memory), while the notebook wants the live `Figure`
    object to display inline — reading a PNG off disk would let the document show something the data
    no longer supports. `build()` flips `save_png` and picks the figure up from `_STATE`.
    """
    _STATE["fig"] = fig
    p = FIG / f"{name}.png"
    if _STATE["save_png"]:
        FIG.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"    -> {p.relative_to(REPO)}")
    return str(p.relative_to(REPO))


def build(name, save_png=False):
    """Build one figure **now**, from its cached inputs. Returns ``(fig, stats, provenance)``.

    This is the notebook's entry point. Calling it means the document renders from the data every
    time it is executed, so a figure cannot silently disagree with the numbers quoted beside it, and
    a stale PNG cannot survive on disk unnoticed. ``provenance`` lists the files actually opened,
    with row counts and modification times.
    """
    if name not in REGISTRY:
        raise KeyError(f"unknown figure {name!r}; available: {list(REGISTRY)}")
    fs.apply_rcparams()
    _STATE.update(fig=None, opened=[], save_png=save_png)
    _, st = REGISTRY[name]["fn"]()
    prov = dict(name=name, title=REGISTRY[name]["title"],
                declared_inputs=list(REGISTRY[name]["inputs"]),
                measured_inputs=list(_STATE["opened"]))
    return _STATE["fig"], st, prov


def run_metadata():
    """Recorded configuration of the long GPU runs behind these figures.

    The measurement runs take hours and can never execute inside a document, so their provenance has
    to be *recorded metadata shown* rather than re-execution. This surfaces the parameters that decide
    what the numbers mean — sampling area, tile order, random seed, per-scene wall clock — so a reader
    can check the sampling design without taking it on trust.
    """
    out = {}
    p = SUM / "fixed_area_scenes_meta.json"
    if p.exists():
        m = load_json(p)
        sc = [s for s in m.get("scenes", []) if "error" not in s]
        out["multi-scene / control-site rescan"] = dict(
            target_km2=m.get("target_km2"), tile_px=m.get("tile_px"), tile_order=m.get("order"),
            seed=m.get("seed"), n_scenes=len(sc),
            km2_per_scene=sorted({round(s.get("km2_valid", 0)) for s in sc}),
            total_detections=int(sum(s.get("n_det", 0) for s in sc)),
            gpu_hours=round(sum(s.get("wall_s", 0) for s in sc) / 3600, 2),
            all_reached_target=all(s.get("reached_target") for s in sc))
    return out


def _null_line(ax, y, label, x=0.985):
    """Dashed isotropic-null reference with a legible label (boxed, so it never sits on a bar)."""
    ax.axhline(y, ls=(0, (4, 3)), lw=1.0, color=fs.NULL_GREY, zorder=1)
    ax.annotate(label, xy=(x, y), xycoords=("axes fraction", "data"), fontsize=6.6,
                color=fs.NULL_GREY, va="center", ha="right", zorder=6,
                bbox=dict(fc="white", ec="none", pad=1.2))


def stat_block(a, sun=None):
    """The standard statistics recorded for any azimuth set, so claims stay traceable."""
    a = np.asarray(a, float)
    a = a[np.isfinite(a)] % 180.0
    if a.size < 5:
        return dict(n=int(a.size))
    out = dict(n=int(a.size), diag=round(diagonal_fraction(a), 3),
               grid=round(grid_fraction(a), 3), chirality=round(asymmetry_ratio(a), 3),
               f_lower=round(half_plane_fraction(a), 3), peak=round(refine_peak(a), 1))
    if sun is not None:
        out["along_sun"] = round(axis_alignment_fraction(a, sun, 45.0), 3)
        out["peak_minus_sun"] = round(float(axial_distance(refine_peak(a), sun)), 1)
    return out


def sun_axes():
    """Sun axis (axial, degrees) + measured shadow depth per labelled scene."""
    m = load_json(SAMV / "mars_sun.json")["ESP_017355_2260_RED"]
    mars = (m["sub_solar"] - m["north"]) % 180.0
    out = {"mars": dict(sun_axis=mars, incidence=m["incidence"]),
           "moon": dict(sun_axis=None, incidence=None)}
    p = SUM / "moon_sun.json"
    if p.exists():
        j = load_json(p)
        out["moon"] = dict(sun_axis=j.get("sun_axis"), incidence=j.get("incidence"),
                           source=j.get("source"))
    sp = SUM / "shadow_azimuth.json"
    if sp.exists():
        for s in load_json(sp)["scenes"]:
            out[s["body"]]["shadow_depth"] = s["shadow_depth"]
            out[s["body"]]["sun_axis_from_image"] = s["sun_axis_axial_from_image"]
    return out


# =================================================================================================
# 1. The problem
# =================================================================================================

@figure("fig01_artifact", ["data/mars_hirise/hirise_orientations.csv", "illumination geometry: data/sam2_validation/mars_sun.json, data/summary/moon_sun.json, data/summary/shadow_azimuth.json"],
        "The artifact: YOLO long axes lock to the pixel diagonal")
def fig01_artifact():
    h = read_csv(DATA / "mars_hirise" / "hirise_orientations.csv")
    h = h[h["set"] == "new-skimage"].copy()          # the production skimage-contour catalogue
    res = 0.5
    h["dpx"] = h.diameter_m / res
    sun = sun_axes()["mars"]["sun_axis"]
    e = h[h.aspect_ra > ASPECT_MIN]
    # Two panels, two different size cuts — so each panel carries its own `dpx_min` and the captions
    # follow the panel. The floored panel pre-empts "it is only the near-pixel-scale blobs".
    panels = [
        dict(angles=e.angle180.values, color="yolo", label="all elongated boulders", sun_axis=sun,
             stats=("diag", "grid", "chir"), dpx_min=None),
        dict(angles=e[e.dpx >= DPX_MIN].angle180.values, color="yolo",
             label="well-resolved only", sun_axis=sun, stats=("diag", "grid", "chir"),
             dpx_min=DPX_MIN),
    ]
    fig, axes = fs.orientation_grid(
        [panels], aspect_min=ASPECT_MIN, dpx_min=None, res_m=res, share_hist=None,
        suptitle="Mars HiRISE ESP_017355_2260 — YOLOv8-seg long-axis azimuths "
                 "(150,000-boulder random sample of the 1.1M-boulder scene)",
        footer_extra="each panel states its own size cut")
    st = dict(no_floor=stat_block(e.angle180.values, sun),
              well_resolved=stat_block(e[e.dpx >= DPX_MIN].angle180.values, sun),
              uniform_reference=round(UNIF, 3), sun_axis=sun,
              n_sample=int(len(h)), median_dpx=round(float(h.dpx.median()), 2))
    return save(fig, "fig01_artifact"), st


@figure("fig02_segmenter_view",
        ["data/summary/segmenter_view.pkl + segmenter_view_meta.json"],
        "What the segmenter sees: one prototype cell per native pixel")
def fig02_segmenter_view():
    p = SUM / "segmenter_view.pkl"
    if not p.exists():
        raise FileNotFoundError(f"{p} — run data/summary/run_segmenter_view.py first")
    d = load_pickle(p)
    recs, meta = d["picked"], d["meta"]
    ppn = meta["proto_px_per_native_px"]
    cols = [("native", f"native raster\n({meta['slice_size']} px slice)"),
            ("input", f"network input\n(upscaled to {meta['imgsz']} px)"),
            ("proto", f"seg-head mask field\n(prototype res {meta['proto_res']:.0f} px)"),
            ("final", "final mask + reported axis")]
    fig, axes = plt.subplots(len(recs), 4, figsize=(9.4, 2.35 * len(recs)))
    axes = np.atleast_2d(axes)
    for i, r in enumerate(recs):
        for j, (key, _) in enumerate(cols):
            ax = axes[i, j]
            img = r[key].astype(float)
            if key == "proto":
                ax.imshow(img, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
                ax.set_title(f"{r['proto_cells']} cells > 0.5", fontsize=6.5, color=fs.INK_SOFT,
                             pad=2)
            elif key == "final":
                ax.imshow(r["native"], cmap="gray", interpolation="nearest")
                ys, xs = np.nonzero(img)
                if len(xs):
                    ax.contour(img, levels=[0.5], colors=[fs.PALETTE["yolo"]], linewidths=1.4)
                    cy, cx = ys.mean(), xs.mean()
                    L = max(img.shape) * 0.42
                    t = np.deg2rad(r["angle"])
                    ax.plot([cx - L * np.sin(t), cx + L * np.sin(t)],
                            [cy + L * np.cos(t), cy - L * np.cos(t)],
                            color=fs.PALETTE["yolo"], lw=1.3, ls="--",
                            path_effects=fs.outline_effects(1.3))
                ax.set_title(f"{r['angle']:.0f}°", fontsize=6.5, color=fs.INK_SOFT, pad=2)
            else:
                ax.imshow(img, cmap="gray", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.text(0.5, 1.30, cols[j][1], transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=8, color=fs.INK)
        axes[i, 0].set_ylabel(f"{r['dpx']:.1f} px", fontsize=8, color=fs.INK)
    fig.suptitle("What the segmentation head actually sees — Mars HiRISE boulders\n"
                 f"at the production config ({meta['slice_size']} px slice → {meta['imgsz']} px "
                 f"input) the prototype basis runs at {meta['proto_res']:.0f} px, i.e. "
                 f"{ppn:.0f} prototype cell per native pixel",
                 fontsize=10.5, y=1.0)
    fig.text(0.5, -0.012,
             "A boulder a few pixels across is described by a few mask cells, so its long axis can "
             "only point where that grid allows.\nEverything to the right of the third column is "
             "interpolation of the third column.",
             ha="center", va="top", fontsize=7.4, color=fs.INK_SOFT)
    fig.tight_layout()
    st = dict(meta=meta, panels=[dict(dpx=round(r["dpx"], 2), angle=round(r["angle"], 1),
                                      proto_angle=round(r["proto_angle"], 1),
                                      proto_cells=r["proto_cells"]) for r in recs])
    extra = SUM / "segmenter_view_meta.json"
    if extra.exists():
        j = load_json(extra)
        st["proto_vs_final_median_abs_deg"] = j["proto_vs_final_median_abs_deg"]
        st["proto_cells_median"] = j["proto_cells_median"]
        st["n_candidates"] = j["n_candidates"]
    return save(fig, "fig02_segmenter_view"), st


def _draw_outlines(ax, patch, layers, title=None, subtitle=None):
    """Grey patch with coloured outlines + dashed fitted axes; every stroke gets a casing."""
    p = patch.astype(float)
    lo, hi = np.percentile(p, [1, 99]) if p.max() > p.min() else (0, 1)
    ax.imshow(np.clip((p - lo) / max(hi - lo, 1e-6), 0, 1), cmap="gray", interpolation="nearest")
    for lab, geom, ang, col in layers:
        c = fs.color_of(col)
        if geom is None:
            continue
        if geom.ndim == 2 and geom.shape[1] == 2:                       # polygon vertices
            xy = np.vstack([geom, geom[:1]])
            ax.plot(xy[:, 0], xy[:, 1], color=c, lw=1.5,
                    path_effects=fs.outline_effects(1.5))
            cx, cy = geom[:, 0].mean(), geom[:, 1].mean()
        else:                                                            # binary mask
            ax.contour(geom, levels=[0.5], colors=[c], linewidths=1.5)
            ys, xs = np.nonzero(geom)
            if not len(xs):
                continue
            cx, cy = xs.mean(), ys.mean()
        if ang is not None and np.isfinite(ang):
            L = max(patch.shape) * 0.30
            t = np.deg2rad(ang)
            ax.plot([cx - L * np.sin(t), cx + L * np.sin(t)],
                    [cy + L * np.cos(t), cy - L * np.cos(t)],
                    color=c, lw=1.1, ls=(0, (5, 2)),
                    path_effects=fs.outline_effects(1.1, casing_extra=1.2))
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=7.6, color=fs.INK, pad=3)
    if subtitle:
        ax.text(0.5, -0.03, subtitle, transform=ax.transAxes, ha="center", va="top",
                fontsize=6.8, color=fs.INK_SOFT, linespacing=1.35)


@figure("fig03_same_boulder",
        ["data/summary/boulder_examples.pkl + boulder_examples_meta.json",
         "data/solutionH/synth_examples.pkl"],
        "The same boulder, three producers — real and against known truth")
def fig03_same_boulder():
    real_p, syn_p = SUM / "boulder_examples.pkl", SH / "synth_examples.pkl"
    if not real_p.exists():
        raise FileNotFoundError(f"{real_p} — run data/summary/run_boulder_examples.py first")
    NCOL = 4                      # fewer, larger panels: at 6 across the boulders were unreadable
    real = load_pickle(real_p)["diag_locked"][:NCOL]
    syn = load_pickle(syn_p)
    sb = syn["boulders"]
    # synthetic: the boulders YOLO mis-measures most, spread over size (same intent as the real row)
    sb = sorted(sb, key=lambda r: -abs(r["yolo_err"]))[:NCOL]
    n = max(len(real), len(sb))
    # Explicit inch-based geometry: each row needs its image plus a 3-line subtitle underneath, and
    # `tight_layout` packs rows until row 1's subtitle sits on row 2's title.
    PW, IMG, SUB, TTL = 2.35, 2.35, 0.52, 0.28
    head, foot = 0.85, 0.42
    rowh = IMG + SUB + TTL
    height = 2 * rowh + head + foot
    fig = plt.figure(figsize=(PW * n + 0.55, height))
    gs = fig.add_gridspec(2, n, left=0.055, right=0.99, top=1 - head / height,
                          bottom=foot / height, hspace=(SUB + TTL) / IMG, wspace=0.10)
    axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(n)] for i in range(2)])
    for j in range(n):
        if j < len(real):
            r = real[j]
            _draw_outlines(axes[0, j], r["patch"],
                           [("human", r["hpoly"], r["angle_h"], "human"),
                            ("yolo", r["ymask"], r["angle_y"], "yolo"),
                            ("sam", r["smask"], r["angle_s"], "sam")],
                           title=f"{r['dpx']:.0f} px  ·  {r['body'].capitalize()}",
                           subtitle=f"human {r['angle_h']:.0f}°\n"
                                    f"YOLO {r['angle_y']:.0f}°  (Δ{r['gap_y_h']:.0f}°)\n"
                                    f"SAM {r['angle_s']:.0f}°  (Δ{r['gap_s_h']:.0f}°)")
        else:
            axes[0, j].axis("off")
        if j < len(sb):
            r = sb[j]
            th = np.linspace(0, 2 * np.pi, 90)
            t = np.deg2rad(r["true_angle"])
            ell = np.c_[r["cx"] + r["a_px"] * np.cos(th) * np.sin(t)
                        + r["b_px"] * np.sin(th) * np.cos(t),
                        r["cy"] - r["a_px"] * np.cos(th) * np.cos(t)
                        + r["b_px"] * np.sin(th) * np.sin(t)]
            _draw_outlines(axes[1, j], r["patch"],
                           [("truth", ell, r["true_angle"], "truth"),
                            ("yolo", r["yolo_mask"], r["yolo_angle"], "yolo"),
                            ("sam", r["sam_mask"], r["sam_angle"], "sam")],
                           title=f"{r['true_dpx']:.0f} px  ·  synthetic",
                           subtitle=f"true {r['true_angle']:.0f}°\n"
                                    f"YOLO {r['yolo_angle']:.0f}°  (Δ{abs(r['yolo_err']):.0f}°)\n"
                                    f"SAM {r['sam_angle']:.0f}°  (Δ{abs(r['sam_err']):.0f}°)")
        else:
            axes[1, j].axis("off")
    axes[0, 0].set_ylabel("real boulders\nhuman = reference", fontsize=9, color=fs.INK)
    axes[1, 0].set_ylabel("synthetic boulders\nknown truth = reference", fontsize=9, color=fs.INK)
    handles = [
        plt.Line2D([], [], color=fs.PALETTE["human"], lw=2,
                   label="reference: human tracing (top) / known truth (bottom)"),
        plt.Line2D([], [], color=fs.PALETTE["yolo"], lw=2, label="YOLOv8-seg mask"),
        plt.Line2D([], [], color=fs.PALETTE["sam"], lw=2, label="SAM2 re-mask")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.002))
    fig.suptitle("One boulder, three producers — solid = outline, dashed = that producer's fitted "
                 "long axis\nreal boulders selected where YOLO's axis has snapped onto a pixel "
                 "diagonal; synthetic where YOLO's angle error is largest",
                 fontsize=10.5, y=1 - 0.10 / height, va="top")
    st = dict(real=[dict(body=r["body"], dpx=round(r["dpx"], 1), angle_h=round(r["angle_h"], 1),
                         angle_y=round(r["angle_y"], 1), angle_s=round(r["angle_s"], 1),
                         gap_yolo=round(r["gap_y_h"], 1), gap_sam=round(r["gap_s_h"], 1))
                    for r in real],
              synthetic_median_yolo_err=round(float(syn["median_yolo_err"]), 1),
              synthetic_median_sam_err=round(float(syn["median_sam_err"]), 1),
              synthetic_sun_az=syn["sun_az_deg"])
    mp = SUM / "boulder_examples_meta.json"
    if mp.exists():
        st["pool"] = load_json(mp)["pool_by_body"]
    return save(fig, "fig03_same_boulder"), st


# =================================================================================================
# 2. The fix
# =================================================================================================

def _rom():
    p = SUM / "rom_threeway.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} — run data/summary/run_rom_threeway.py first")
    return read_csv(p)


def _synth_rows(cond="shadow"):
    t = read_csv(SAMV / "t1_synth.csv")
    return t[t.cond == cond].copy()


@figure("fig04a_threeway_matched",
        ["data/summary/rom_threeway.csv", "data/sam2_validation/t1_synth.csv",
         "illumination geometry: data/sam2_validation/mars_sun.json, data/summary/moon_sun.json, data/summary/shadow_azimuth.json"],
        "YOLO vs SAM2 vs reference on the SAME boulders")
def fig04a_threeway_matched():
    rom, sun = _rom(), sun_axes()
    rows, labels, st = [], [], {}
    for body, scene, nice in (("mars", "ESP_017355_2260", "Mars HiRISE ESP_017355_2260"),
                              ("moon", "M139694087LE", "Moon LRO NAC M139694087LE")):
        g = rom[(rom.body == body) & (rom.scene == scene)]
        # Population defined by the REFERENCE, never by a model: the humans decide which rocks are
        # elongated and well-resolved, then all three producers are read on exactly those rocks.
        # Conditioning on a model's own aspect would bias the comparison toward that model.
        m = g[g.h_angle.notna() & (g.h_aspect > ASPECT_MIN) & (g.h_dpx >= DPX_MIN)]
        sx = sun[body]["sun_axis"]
        rows.append([
            dict(angles=m.angle_yolo.values, color="yolo", label="YOLOv8-seg mask", sun_axis=sx),
            dict(angles=m.angle_sam.values, color="sam", label="SAM2 re-mask", sun_axis=sx),
            dict(angles=m.h_angle.values, color="human", label="human tracing", sun_axis=sx),
        ])
        labels.append(f"{nice}\n(shadow depth {sun[body].get('shadow_depth', float('nan')):.2f}, "
                      f"incidence {sun[body].get('incidence', float('nan')):.0f}°)")
        st[body] = dict(n_matched=int(len(m)),
                        yolo=stat_block(m.angle_yolo.values, sx),
                        sam=stat_block(m.angle_sam.values, sx),
                        human=stat_block(m.h_angle.values, sx),
                        median_abs_dangle_sam_vs_human=round(float(np.median(
                            np.abs(((m.angle_sam - m.h_angle + 90) % 180) - 90))), 2),
                        median_abs_dangle_yolo_vs_human=round(float(np.median(
                            np.abs(((m.angle_yolo - m.h_angle + 90) % 180) - 90))), 2))
    s = _synth_rows()
    sm = s[(s.aspect_true > ASPECT_MIN) & (s.dpx_sam256 >= DPX_MIN)]
    rows.append([
        dict(angles=sm.angle_yolo.values, color="yolo", label="YOLOv8-seg mask"),
        dict(angles=sm.angle_sam256.values, color="sam", label="SAM2 re-mask"),
        dict(angles=sm.truth_angle.values, color="truth", label="known truth (uniform)"),
    ])
    labels.append("synthetic boulders,\nshadowed, uniform truth")
    st["synthetic"] = dict(n=int(len(sm)), yolo=stat_block(sm.angle_yolo.values),
                           sam=stat_block(sm.angle_sam256.values),
                           truth=stat_block(sm.truth_angle.values))
    fig, _ = fs.orientation_grid(
        rows, aspect_min=ASPECT_MIN, dpx_min=DPX_MIN, row_labels=labels,
        suptitle="Matched detections — the SAME boulders through three producers\n"
                 "(population set by the reference: reference aspect > 1.35 and ≥ 8 px, "
                 "never by a model's own output)",
        footer_extra="sun axis = reference line, not an attribution")
    st["uniform_reference"] = round(UNIF, 3)
    return save(fig, "fig04a_threeway_matched"), st


@figure("fig04b_threeway_samearea", ["data/summary/rom_threeway.csv", "illumination geometry: data/sam2_validation/mars_sun.json, data/summary/moon_sun.json, data/summary/shadow_azimuth.json"],
        "YOLO vs SAM2 vs reference over the SAME AREA")
def fig04b_threeway_samearea():
    rom, sun = _rom(), sun_axes()
    rows, labels, st = [], [], {}
    for body, scene, nice in (("mars", "ESP_017355_2260", "Mars HiRISE ESP_017355_2260"),
                              ("moon", "M139694087LE", "Moon LRO NAC M139694087LE")):
        g = rom[(rom.body == body) & (rom.scene == scene)]
        sx = sun[body]["sun_axis"]
        y = g[(g.aspect_yolo > ASPECT_MIN) & (g.dpx_yolo >= DPX_MIN)]
        sa = g[(g.aspect_sam > ASPECT_MIN) & (g.dpx_sam >= DPX_MIN)]
        # Human labels come from the whole ROM, not only where a detection matched — that is the
        # point of a same-area read.
        h = g[g.h_angle.notna() & (g.h_aspect > ASPECT_MIN) & (g.h_dpx >= DPX_MIN)]
        rows.append([
            dict(angles=y.angle_yolo.values, color="yolo", label="YOLOv8-seg mask", sun_axis=sx),
            dict(angles=sa.angle_sam.values, color="sam", label="SAM2 re-mask", sun_axis=sx),
            dict(angles=h.h_angle.values, color="human", label="human tracing", sun_axis=sx,
                 rescale_hist=True),
        ])
        labels.append(nice)
        st[body] = dict(yolo=stat_block(y.angle_yolo.values, sx),
                        sam=stat_block(sa.angle_sam.values, sx),
                        human=stat_block(h.h_angle.values, sx))
    fig, _ = fs.orientation_grid(
        rows, aspect_min=ASPECT_MIN, dpx_min=DPX_MIN, row_labels=labels,
        suptitle="Same area, each producer's own population — one identical human-mapped footprint "
                 "(ROM)\nnothing is conditioned on agreement between producers",
        footer_extra="human panel counts are rescaled where marked (the model reports more "
                     "boulders than were mapped)")
    st["uniform_reference"] = round(UNIF, 3)
    return save(fig, "fig04b_threeway_samearea"), st


@figure("fig05_axis_fractions", ["data/solutionG/G_detections.csv"],
        "Per-axis share and size-resolved diagonal fraction")
def fig05_axis_fractions():
    g = read_csv(SG / "G_detections.csv")
    tol, per_axis_unif = 10.0, 10.0 / 90.0
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.9))

    axes_deg = [0, 45, 90, 135]
    width = 0.36
    st = dict(per_axis={}, by_band={}, tol_deg=tol, uniform_per_axis=round(per_axis_unif, 3),
              uniform_two_axis=round(UNIF, 3))
    for k, (tag, acol, pcol) in enumerate((("YOLOv8-seg mask", "angle_yolo", "aspect_yolo"),
                                           ("SAM2 re-mask", "angle_sam256", "aspect_sam256"))):
        sel = g[g[pcol] > ASPECT_MIN]
        a = sel[acol].dropna().values % 180.0
        fr = [float((axial_distance(a, ax_) <= tol).mean()) for ax_ in axes_deg]
        ax[0].bar(np.arange(4) + (k - 0.5) * width, fr, width,
                  color=fs.color_of("yolo" if k == 0 else "sam"), edgecolor=fs.INK, lw=0.3,
                  label=f"{tag}  (n = {len(a):,})")
        for x, v in zip(np.arange(4) + (k - 0.5) * width, fr):
            ax[0].text(x, v + 0.004, f"{v:.3f}", ha="center", fontsize=6.4, color=fs.INK_SOFT)
        st["per_axis"][tag] = dict(zip([f"{d}deg" for d in axes_deg], [round(v, 3) for v in fr]))
        st["per_axis"][tag]["n"] = int(len(a))
    _null_line(ax[0], per_axis_unif, "isotropic  0.111")
    ax[0].set_xticks(np.arange(4), ["0°\n(N–S)", "45°", "90°\n(E–W)", "135°"])
    ax[0].set_ylabel(f"share of azimuths within ±{tol:.0f}° of the axis", fontsize=8)
    ax[0].set_title("Which pixel axis absorbs the excess", fontsize=9.5)
    ax[0].legend(fontsize=7.5, loc="upper left")

    bands = [(4, 6), (6, 8), (8, 10), (10, 14), (14, 60)]
    lab = [f"{lo}–{hi if hi < 60 else '+'}" for lo, hi in bands]
    for tag, acol, pcol, col in (("YOLOv8-seg mask", "angle_yolo", "aspect_yolo", "yolo"),
                                 ("SAM2 re-mask", "angle_sam256", "aspect_sam256", "sam")):
        ys, ns = [], []
        for lo, hi in bands:
            s = g[(g[pcol] > ASPECT_MIN) & (g.dpx_native >= lo) & (g.dpx_native < hi)]
            a = s[acol].dropna().values
            ys.append(diagonal_fraction(a) if len(a) > 30 else np.nan)
            ns.append(len(a))
        ax[1].plot(range(len(bands)), ys, "o-", color=fs.color_of(col), lw=1.8, ms=6,
                   label=tag, path_effects=fs.outline_effects(1.8))
        st["by_band"][tag] = dict(bands=lab, diag=[None if not np.isfinite(v) else round(v, 3)
                                                  for v in ys], n=ns)
    _null_line(ax[1], UNIF, "isotropic  0.222")
    ax[1].set_xticks(range(len(bands)), lab)
    ax[1].set_xlabel("equivalent diameter (native px)", fontsize=8)
    ax[1].set_ylabel("diagonal fraction (45°/135°, ±10°)", fontsize=8)
    ax[1].set_title("The excess is carried by the 6–14 px bulk,\nand every band returns to "
                    "isotropic", fontsize=9.5)
    ax[1].legend(fontsize=7.5)
    for a_ in ax:
        for s_ in ("top", "right"):
            a_.spines[s_].set_visible(False)
    fig.suptitle("Mars HiRISE 4096² crop, matched detections (same boxes, only the mask changes)  ·  "
                 f"aspect > {ASPECT_MIN}", fontsize=10, y=1.03)
    fig.tight_layout()
    return save(fig, "fig05_axis_fractions"), st


@figure("fig06_quantization", ["data/sam2_validation/t35_val_detections.csv"],
        "Why an average-angle metric is blind to this")
def fig06_quantization():
    t = read_csv(SAMV / "t35_val_detections.csv")
    bodies = ["mars", "moon", "earth"]
    TOL = 0.01
    prods = (("YOLOv8-seg mask", "angle_yolo", "aspect_yolo", "yolo"),
             ("SAM2 re-mask", "angle_sam", "aspect_sam", "sam"),
             ("human tracing", "h_angle", "h_aspect", "human"))
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.8))
    width = 0.26

    def measure(b, acol, pcol, dpx_min):
        s = t[(t.body == b) & (t[pcol] > ASPECT_MIN)]
        if dpx_min:
            s = s[s.dpx_sam >= dpx_min]
        a = s[acol].dropna().values % 180.0
        if not len(a):
            return np.nan, np.nan, 0
        d = np.minimum.reduce([axial_distance(a, v) for v in (0, 45, 90, 135)])
        return float((d <= TOL).mean()) * 100, float(d.mean()), int(len(a))

    # No size floor here, deliberately: this is a statement about the whole elongated catalogue (and
    # it is the subset the published quantization table used). The well-resolved subset is measured
    # too and reported in the caption, because "does it survive a size cut?" is the obvious question.
    st = dict(exact_tol_deg=TOL, exact_fraction={}, mean_distance_to_axis={},
              uniform_mean_distance=11.25, exact_fraction_well_resolved={},
              well_resolved_cut_px=DPX_MIN)
    for k, (tag, acol, pcol, col) in enumerate(prods):
        ex, md, ns, exw = [], [], [], []
        for b in bodies:
            e, m_, n_ = measure(b, acol, pcol, None)
            ex.append(e); md.append(m_); ns.append(n_)
            exw.append(measure(b, acol, pcol, DPX_MIN)[0])
        x = np.arange(len(bodies)) + (k - 1) * width
        ax[0].bar(x, ex, width, color=fs.color_of(col), edgecolor=fs.INK, lw=0.3, label=tag)
        for xi, v in zip(x, ex):
            if np.isfinite(v):
                ax[0].text(xi, v + 0.3, f"{v:.1f}%", ha="center", fontsize=6.4, color=fs.INK_SOFT)
        ax[1].bar(x, md, width, color=fs.color_of(col), edgecolor=fs.INK, lw=0.3, label=tag)
        st["exact_fraction"][tag] = dict(zip(bodies, [None if not np.isfinite(v) else round(v, 2)
                                                      for v in ex]))
        st["exact_fraction_well_resolved"][tag] = dict(
            zip(bodies, [None if not np.isfinite(v) else round(v, 2) for v in exw]))
        st["mean_distance_to_axis"][tag] = dict(zip(bodies, [None if not np.isfinite(v)
                                                             else round(v, 2) for v in md]))
        st["exact_fraction"][tag]["n"] = ns
    yw = st["exact_fraction_well_resolved"]["YOLOv8-seg mask"]
    ax[0].text(0.02, 0.97, "survives a size cut: for ≥ 8 px boulders alone YOLO is still\n"
                           f"{yw['mars']:.1f}% / {yw['moon']:.1f}% / {yw['earth']:.1f}% "
                           "(Mars / Moon / Earth)",
               transform=ax[0].transAxes, va="top", fontsize=6.6, color=fs.INK_SOFT)
    ax[0].set_ylabel("% of angles landing within 0.01° of\na pixel axis (0/45/90/135°)", fontsize=8)
    ax[0].set_title("The artifact: a minority of angles snap EXACTLY onto a pixel axis",
                    fontsize=9.5)
    _null_line(ax[1], 11.25, "isotropic  11.25°")
    ax[1].set_ylabel("mean distance to the nearest pixel axis (°)", fontsize=8)
    ax[1].set_title("...which an average-error metric barely registers", fontsize=9.5)
    for i, a_ in enumerate(ax):
        a_.set_xticks(np.arange(len(bodies)), [b.capitalize() for b in bodies])
        if i == 0:            # one legend per figure: the right panel uses the same three colours
            a_.legend(fontsize=7.5, loc="upper right", framealpha=1.0)
        for s_ in ("top", "right"):
            a_.spines[s_].set_visible(False)
    ax[0].set_ylim(top=max(v for d in st["exact_fraction"].values()
                           for v in d.values() if isinstance(v, float)) * 1.55)
    ax[1].set_ylim(0, 13.2)
    fig.suptitle(f"boulder2024 validation tiles, matched boulders  ·  aspect > {ASPECT_MIN}  ·  "
                 "no size floor  —  this is why the rose, not median |Δangle|, is the metric of "
                 "record", fontsize=9.6, y=1.04)
    fig.tight_layout()
    return save(fig, "fig06_quantization"), st


# =================================================================================================
# 3. How far to trust a rose
# =================================================================================================

FIXED_AREA = SUM / "fixed_area_scenes.csv"
FIXED_AREA_META = SUM / "fixed_area_scenes_meta.json"
CI_POWER_LIMIT = 90.0     # a bootstrap peak interval wider than this carries no directional info


def _fixed_area(group):
    """Per-scene detections + metadata from the fixed-area rescan.

    Returns ``(df, meta_by_scene, source_label)``. Falls back to the earlier detection-quota run if
    the rescan has not been done, and says which was used — the two are NOT interchangeable: the old
    one walked tiles in raster order and stopped on a detection count, so its `n` is not comparable
    between scenes and its sample is a spatial prefix rather than a random sample of the scene.
    """
    # Prefer the combined file the run writes at the end; otherwise assemble from the per-scene
    # files it writes as it goes, so a partially-complete rescan can still be inspected. The number
    # of scenes present is always reported, so a partial run is never mistaken for a complete one.
    if FIXED_AREA.exists() and FIXED_AREA_META.exists():
        meta = {m["scene"]: m for m in load_json(FIXED_AREA_META)["scenes"]
                if m.get("group") == group and "error" not in m}
        d = read_csv(FIXED_AREA)
        d = d[d.scene.isin(meta)]
        if len(meta):
            return d, meta, f"fixed area, shuffled tiles ({len(meta)} scenes)"
    part_dir = SUM / "fixed_area"
    if part_dir.exists():
        metas = [load_json(p) for p in part_dir.glob(f"{group}__*.json")]
        meta = {m["scene"]: m for m in metas if "error" not in m}
        frames = [read_csv(p) for p in part_dir.glob(f"{group}__*.csv")]
        if meta and frames:
            d = pd.concat(frames, ignore_index=True)
            d = d[d.scene.isin(meta)]
            return d, meta, (f"fixed area, shuffled tiles — PARTIAL: {len(meta)} scenes complete "
                             f"so far")
    if group == "hirise":
        d = read_csv(SAMV / "sam_chirality_hirise.csv")
        ps = read_csv(SAMV / "sam_chirality_perscene.csv").set_index("scene")
        meta = {sc: dict(scene=sc, res_m=float(d[d.scene == sc].res.iloc[0]),
                         shadow_axis=float(ps.at[sc, "shadow"]), incidence=float(ps.at[sc, "inc"]),
                         km2_valid=None, per_km2_elong_ge8px=None) for sc in ps.index}
    else:
        d = read_csv(SAMV / "t1_controls_sam.csv")
        meta = {sc: dict(scene=sc, res_m=float(g.res.iloc[0]), shadow_axis=float(g.sun.iloc[0]),
                         incidence=float(g.incidence.iloc[0]), km2_valid=None,
                         per_km2_elong_ge8px=None) for sc, g in d.groupby("scene")}
    return d, meta, "detection quota, raster order (LEGACY — n not comparable between scenes)"


DM_MIN = 4.0     # metres: a COMMON physical size floor, so cross-scene panels count the same rocks


def _scene_panels(d, meta, colour="sam", angle_col="angle_sam", aspect_col="aspect_sam",
                  dm_min=DM_MIN):
    """One rose panel per scene, annotated with density and the peak's bootstrap interval.

    ``dm_min`` imposes a **metric** size floor on top of the pixel floor. This is what makes a
    cross-scene comparison legitimate: >= 8 px is 2 m at 0.25 m/px but 4 m at 0.50 m/px, so a
    pixel-only cut would have the fine-resolution scenes counting a different (much more numerous)
    physical population — contaminating both the roses and the boulders/km² densities. Requiring
    ``max(8 px, 4 m)`` costs power on the fine scenes and buys a comparison that means something.
    Pass ``dm_min=None`` to reproduce the pixel-only cut.
    """
    panels, st = [], {}
    order = sorted(meta, key=lambda s: meta[s].get("incidence") or 0)
    for sc in order:
        m = meta[sc]
        res = m.get("res_m")
        g = d[(d.scene == sc) & (d[aspect_col] > ASPECT_MIN) & (d.dpx_sam >= DPX_MIN)]
        dm = g.dm_sam if "dm_sam" in g else g.dpx_sam * res      # legacy CSVs lack the metric column
        if dm_min:
            g = g[dm >= dm_min]
        sx, inc = m.get("shadow_axis"), m.get("incidence")
        a = g[angle_col].dropna().values
        km2 = m.get("km2_valid")
        # Density is recomputed at the figure's own cut rather than taken from the run metadata,
        # which recorded the pixel-only count.
        dens = round(len(a) / km2, 1) if km2 else m.get("per_km2_elong_ge8px")
        # Two short lines, not one long one: at five columns a single note line overran into the
        # neighbouring panel's title (seen on a rendered draft). |peak-sun| is dropped from the panel
        # because it is tabulated under the figure — the panel keeps only what the rose cannot show.
        note = f"sun {sx:.0f}°  ·  inc {inc:.0f}°"
        second = f"{res:.2f} m/px" + (f"  ·  {dens:g}/km²" if dens is not None else "")
        note += f"\n{second}"
        panels.append(dict(angles=a, color=colour, label=sc, sun_axis=sx, note=note,
                           stats=("peakci",)))
        blk = stat_block(a, sx)
        if len(a) > 20:
            pk, lo, hi = bootstrap_peak_ci(a, n_boot=400)
            blk.update(peak_ci=[round(float(lo), 1), round(float(hi), 1)],
                       peak_ci_width=round(float(hi - lo), 1),
                       has_power=bool((hi - lo) <= CI_POWER_LIMIT))
        else:
            blk.update(peak_ci=None, peak_ci_width=None, has_power=False)
        st[sc] = dict(sun_axis=None if sx is None else round(sx, 1),
                      incidence=None if inc is None else round(inc, 1), res_m=res,
                      km2_valid=km2, per_km2_elong_ge8px=dens, **blk)
    return panels, st, order


def _scene_summary(st, order):
    """Aggregate |peak − sun|, and the confound checks that aggregate invites."""
    def arr(key):
        return np.array([st[s].get(key) if st[s].get(key) is not None else np.nan
                         for s in order], float)
    dsun, inc, res = arr("peak_minus_sun"), arr("incidence"), arr("res_m")
    powered = np.array([bool(st[s].get("has_power")) for s in order])
    # Which axis do the peaks actually sit on? The three candidates are the sun, the pixel cardinals
    # and the pixel diagonals, and comparing all three is what identifies the mechanism — the same
    # test the control sites use, so the two sections can be read against each other.
    pk = arr("peak")
    dcard = np.array([min(axial_distance(p, 0.0), axial_distance(p, 90.0)) if np.isfinite(p)
                      else np.nan for p in pk])
    ddiag = np.array([min(axial_distance(p, 45.0), axial_distance(p, 135.0)) if np.isfinite(p)
                      else np.nan for p in pk])
    out = dict(n_scenes=len(order), n_with_power=int(powered.sum()),
               ci_power_limit_deg=CI_POWER_LIMIT, random_expectation=45.0,
               mean_peak_minus_sun=round(float(np.nanmean(dsun)), 1),
               mean_peak_minus_sun_powered_only=(round(float(np.nanmean(dsun[powered])), 1)
                                                if powered.any() else None),
               mean_peak_minus_cardinal=round(float(np.nanmean(dcard)), 1),
               mean_peak_minus_diagonal=round(float(np.nanmean(ddiag)), 1),
               mean_peak_minus_cardinal_powered_only=(round(float(np.nanmean(dcard[powered])), 1)
                                                     if powered.any() else None),
               n_peaks_within_10deg_of_cardinal=int(np.nansum(dcard <= 10)),
               mean_along_sun=round(float(np.nanmean(arr("along_sun"))), 3),
               median_peak_ci_width=round(float(np.nanmedian(arr("peak_ci_width"))), 1))

    def corr(x, y, mask=None):
        m = np.isfinite(x) & np.isfinite(y) & (np.ones_like(x, bool) if mask is None else mask)
        return None if m.sum() < 4 else round(float(np.corrcoef(x[m], y[m])[0, 1]), 3)

    sun = arr("sun_axis")
    out["corr_incidence_vs_peak_minus_sun"] = corr(inc, dsun)
    out["corr_sun_axis_vs_incidence"] = corr(sun, inc)
    # The >=8 px cut selects 2 m boulders at 0.25 m/px and 4 m at 0.5 m/px, so resolution is a
    # candidate confound for anything read off this figure. Checked explicitly, and the incidence
    # correlation is re-run within the single largest resolution group.
    out["corr_resolution_vs_peak_minus_sun"] = corr(res, dsun)
    out["corr_resolution_vs_incidence"] = corr(res, inc)
    for r in np.unique(res[np.isfinite(res)]):
        sub = res == r
        out[f"corr_incidence_vs_peak_minus_sun_at_{r:g}mpx"] = corr(inc, dsun, sub)
        out[f"n_scenes_at_{r:g}mpx"] = int(sub.sum())
    return out


def _sampling_validity(order):
    """Measure how unrepresentative the earlier raster-order sample was, per scene.

    The rerun's justification should be a number, not an assertion. Both passes measured the same
    quantity — elongated well-resolved boulders per km² — on the same scenes, one by walking a
    contiguous prefix of a central strip, the other by sampling tiles at random over the whole
    footprint. Where the two disagree, the prefix was not representative of its scene; the ratio is
    the size of that error, and its *scene-dependence* is why no global correction could have fixed
    the old numbers after the fact.
    """
    try:
        old = read_csv(SAMV / "sam_chirality_hirise.csv")
        om = load_json(SAMV / "sam_chirality_meta.json")
        new_meta = {m["scene"]: m for m in load_json(FIXED_AREA_META)["scenes"]} \
            if FIXED_AREA_META.exists() else \
            {load_json(p)["scene"]: load_json(p)
             for p in (SUM / "fixed_area").glob("hirise__*.json")}
    except Exception as e:                                            # noqa: BLE001
        return dict(error=f"could not compare: {type(e).__name__}: {e}")
    rows = {}
    for sc in order:
        if sc not in om or sc not in new_meta:
            continue
        o, n = om[sc], new_meta[sc]
        okm2 = o["tiles"] * (512 * o["res"]) ** 2 / 1e6
        g = old[(old.scene == sc) & (old.aspect_sam > ASPECT_MIN) & (old.dpx_sam >= DPX_MIN)]
        o_dens, n_dens = len(g) / okm2, n.get("per_km2_elong_ge8px")
        if not n_dens:
            continue
        rows[sc] = dict(old_km2=round(okm2, 1), old_per_km2=round(o_dens, 1),
                        new_km2=n.get("km2_valid"), new_per_km2=n_dens,
                        ratio_new_over_old=round(n_dens / max(o_dens, 1e-6), 2))
    if not rows:
        return dict(note="no overlapping scenes to compare yet")
    r = [v["ratio_new_over_old"] for v in rows.values()]
    return dict(
        per_scene=rows, n_scenes=len(rows),
        ratio_min=round(min(r), 2), ratio_max=round(max(r), 2),
        ratio_median=round(float(np.median(r)), 2),
        note="new (equal-area, randomly-sampled) boulders per km² divided by old (detection-quota, "
             "raster-order prefix). Values far from 1 mean the old prefix was not representative of "
             "its scene; the spread across scenes is why the old per-scene n could not be corrected "
             "after the fact.")


@figure("fig07_hirise_scenes",
        ["data/summary/fixed_area_scenes.csv + fixed_area_scenes_meta.json (the equal-area rescan)",
         "data/sam2_validation/sam_chirality_hirise.csv + sam_chirality_meta.json "
         "(the superseded quota sample, used only for the sampling-validity comparison)"],
        "SAM orientations across the HiRISE scenes, against each scene's sun axis")
def fig07_hirise_scenes():
    d, meta, source = _fixed_area("hirise")
    panels, st, order = _scene_panels(d, meta, dm_min=DM_MIN)
    # Round before comparing: 153 tiles at 0.50 m/px is 10.03 km2 and 611 at 0.25 m/px is 10.01, so
    # an exact-value set makes an equal-area run look unequal (it did, in the title of a draft).
    km2 = {round(m["km2_valid"]) for m in meta.values() if m.get("km2_valid")}
    area_txt = (f"{next(iter(km2))} km² of valid ground per scene" if len(km2) == 1
                else "unequal area per scene (legacy sampling)")
    fig, _ = fs.rose_grid(
        panels, aspect_min=ASPECT_MIN, dpx_min=DPX_MIN, dm_min=DM_MIN, ncols=4,
        suptitle=f"SAM2 orientations on {len(order)} Mars HiRISE scenes, ordered by solar "
                 f"incidence — {area_txt}, tiles sampled at random across each scene",
        footer_extra="brackets = bootstrap 95% interval on the peak  ·  the metric floor puts every "
                     "scene on the same physical population despite the 0.25/0.50 m/px mix  ·  each "
                     "scene is a DIFFERENT footprint, so 'sun axis' also indexes different ground")
    st["_summary"] = dict(source=source, size_cut=f"≥ {DPX_MIN:g} px and ≥ {DM_MIN:g} m",
                          **_scene_summary(st, order))
    st["_sampling_validity"] = _sampling_validity(order)
    # Same figure's statistics under the pixel-only cut, so the effect of imposing a common physical
    # population is visible rather than asserted (it changes n on the 0.25 m/px scenes by ~10x).
    _, st_px, order_px = _scene_panels(d, meta, dm_min=None)
    st["_summary_pixel_cut_only"] = dict(
        size_cut=f"≥ {DPX_MIN:g} px only (2 m at 0.25 m/px, 4 m at 0.50 m/px)",
        n_by_scene={s: st_px[s]["n"] for s in order_px}, **_scene_summary(st_px, order_px))
    return save(fig, "fig07_hirise_scenes"), st


@figure("fig08_repeat_pairs",
        ["data/sam2_validation/t2_hirise_sam.csv", "data/summary/t2_thumbnails.npz",
         "data/summary/t2_thumbnails_meta.json"],
        "Same ground, different sun: the rose does not follow the sun")
def fig08_repeat_pairs():
    t = read_csv(SAMV / "t2_hirise_sam.csv")
    npz_p, meta_p = SUM / "t2_thumbnails.npz", SUM / "t2_thumbnails_meta.json"
    if not npz_p.exists():
        raise FileNotFoundError(f"{npz_p} — run data/summary/cut_t2_thumbnails.py first")
    npz = np.load(npz_p)
    tm = load_json(meta_p)

    # Order pairs by statistical power: the usable pair first, the recall-collapsed ones after, and
    # labelled as such. Presenting a n=7 rose next to a n=711 rose without saying so would be the
    # figure lying by omission.
    rows, st = [], {}
    for pid in tm:
        members = [k for k in tm[pid] if k.startswith(("ESP", "PSP"))]
        if len(members) != 2:
            continue
        members.sort(key=lambda o: tm[pid][o]["incidence"])
        ns = []
        for o in members:
            g = t[(t.scene == o) & (t.aspect_sam > ASPECT_MIN) & (t.dpx_sam >= DPX_MIN)]
            ns.append(len(g))
        rows.append((pid, members, min(ns)))
    rows.sort(key=lambda r: -r[2])

    # Each pair gets its own block: a thin full-width banner axis for the pair's caption, then the
    # four panels. A `fig.text` placed by hand landed on the panel titles.
    ROWH, HEAD, FOOT = 3.35, 0.95, 0.55
    height = ROWH * len(rows) + HEAD + FOOT
    fig = plt.figure(figsize=(11.4, height))
    outer = fig.add_gridspec(len(rows), 1, hspace=0.30, top=1 - HEAD / height,
                             bottom=FOOT / height, left=0.055, right=0.98)
    # shared radial limit across every rose in the figure
    allang = []
    for pid, members, _ in rows:
        for o in members:
            g = t[(t.scene == o) & (t.aspect_sam > ASPECT_MIN) & (t.dpx_sam >= DPX_MIN)]
            allang.append(g.angle_sam.values)
    rmax = max((fs._rose_max(a, 36) for a in allang if len(a) > 5), default=0.1) * 1.08

    for i, (pid, members, nmin) in enumerate(rows):
        weak = nmin < 60
        inner = outer[i].subgridspec(2, 4, height_ratios=[0.17, 1.0], hspace=0.62, wspace=0.10)
        banner = fig.add_subplot(inner[0, :])
        banner.axis("off")
        # Two short lines rather than one long one: a single line ran under the rose column and
        # collided with its title.
        tag = ("the only well-powered pair" if not weak else
               f"RECALL COLLAPSE at high incidence (n = {nmin})\nshapes are not interpretable")
        banner.text(0.0, 1.0, f"pair {pid}   ·   Δsun "
                    f"{tm[pid].get('d_sun_deg', float('nan')):.0f}°, Δincidence "
                    f"{tm[pid].get('d_inc_deg', float('nan')):.0f}°\n{tag}",
                    fontsize=8.4, color=fs.INK if not weak else fs.PALETTE["human"],
                    va="top", ha="left", transform=banner.transAxes, linespacing=1.4)
        st[pid] = dict(usable=not weak, d_sun_deg=tm[pid].get("d_sun_deg"),
                       d_inc_deg=tm[pid].get("d_inc_deg"), members={})
        for j, o in enumerate(members):
            info = tm[pid][o]
            g = t[(t.scene == o) & (t.aspect_sam > ASPECT_MIN) & (t.dpx_sam >= DPX_MIN)]
            gy = t[(t.scene == o) & (t.aspect_yolo > ASPECT_MIN) & (t.dpx_sam >= DPX_MIN)]
            axi = fig.add_subplot(inner[1, 2 * j])
            img = npz[f"{pid}__{o}"].astype(float)
            axi.imshow(np.clip((img - info["stretch"][0])
                               / max(info["stretch"][1] - info["stretch"][0], 1e-6), 0, 1),
                       cmap="gray", interpolation="antialiased")
            axi.set_xticks([]); axi.set_yticks([])
            axi.set_title(f"{o.replace('_RED','')}\nsun {info['sun_axis']:.0f}°  ·  "
                          f"incidence {info['incidence']:.0f}°", fontsize=7.4, color=fs.INK, pad=3)
            # sun arrow, drawn in ink on the image so the reader can see the shadows agree with it
            h, w = img.shape
            L = 0.30 * w
            th = np.deg2rad(info["sun_axis"])
            axi.annotate("", xy=(w / 2 + L * np.sin(th), h / 2 - L * np.cos(th)),
                         xytext=(w / 2 - L * np.sin(th), h / 2 + L * np.cos(th)),
                         arrowprops=dict(arrowstyle="-", color="white", lw=2.4, alpha=0.85))
            axi.annotate("", xy=(w / 2 + L * np.sin(th), h / 2 - L * np.cos(th)),
                         xytext=(w / 2 - L * np.sin(th), h / 2 + L * np.cos(th)),
                         arrowprops=dict(arrowstyle="-", color=fs.INK, lw=1.3,
                                         linestyle=(0, (7, 2, 1.5, 2))))
            axr = fig.add_subplot(inner[1, 2 * j + 1], projection="polar")
            fs.rose(axr, g.angle_sam.values, color="sam", rmax=rmax, show_rticks=(i == 0 and j == 0))
            fs.draw_sun_axis(axr, sun_axis=info["sun_axis"], rmax=rmax)
            pk = refine_peak(g.angle_sam.values) if len(g) > 5 else np.nan
            axr.set_title(f"SAM2  n = {len(g):,}\n"
                          + (f"peak {pk:.0f}°" if np.isfinite(pk) else "too few boulders"),
                          fontsize=7.4, color=fs.INK, pad=6)
            st[pid]["members"][o] = dict(
                sun_axis=info["sun_axis"], incidence=info["incidence"],
                sam=stat_block(g.angle_sam.values, info["sun_axis"]),
                yolo=stat_block(gy.angle_yolo.values, info["sun_axis"]))
    fig.suptitle("Same ground, different sun — genuine HiRISE repeat pairs\n"
                 "an illumination-driven mask would swing the rose TOWARD the sun between the two "
                 "members; it does not", fontsize=10.8, y=1 - 0.10 / height, va="top")
    fig.text(0.5, 0.10 / height,
             f"aspect > {ASPECT_MIN}  ·  d$_{{eq}}$ ≥ {DPX_MIN} px  ·  radial density shared across "
             "all roses  ·  dash-dot = that member's own sun axis  ·  thumbnails are the SAME "
             "geographic window, contrast-stretched per member",
             ha="center", va="bottom", fontsize=7.2, color=fs.INK_SOFT)
    return save(fig, "fig08_repeat_pairs"), st


@figure("fig09_controls",
        ["data/summary/fixed_area_scenes.csv + fixed_area_scenes_meta.json (the equal-area rescan)",
         "data/sam2_validation/t1_controls_sam.csv (superseded quota sample; fallback only)"],
        "Control sites: on fabric-free ground each masker imprints its own grid axis — SAM2 cardinal, "
        "YOLO diagonal — and neither follows the sun")
def fig09_controls():
    d, meta, source = _fixed_area("controls")
    st, panels = {}, []
    for prod, acol, pcol, col in (("SAM2 re-mask", "angle_sam", "aspect_sam", "sam"),
                                  ("YOLOv8-seg mask", "angle_yolo", "aspect_yolo", "yolo")):
        p, s, order = _scene_panels(d, meta, colour=col, angle_col=acol, aspect_col=pcol,
                                    dm_min=DM_MIN)
        panels += p
        st[prod] = s
    scenes = order
    # Round before comparing: 153 tiles at 0.50 m/px is 10.03 km2 and 611 at 0.25 m/px is 10.01, so
    # an exact-value set makes an equal-area run look unequal (it did, in the title of a draft).
    km2 = {round(m["km2_valid"]) for m in meta.values() if m.get("km2_valid")}
    area_txt = (f"{next(iter(km2))} km² of valid ground per site" if len(km2) == 1
                else "unequal area per site (legacy sampling)")
    fig, _ = fs.rose_grid(
        panels, aspect_min=ASPECT_MIN, dpx_min=DPX_MIN, dm_min=DM_MIN, ncols=len(scenes),
        row_labels=["SAM2 re-mask", "YOLOv8-seg mask"],
        suptitle="Geologically-null control sites (no depositional fabric expected), sun axes "
                 f"spanning 55–130° — {area_txt}, tiles sampled at random",
        footer_extra="brackets = bootstrap 95% interval on the peak  ·  no fabric is expected here, "
                     "yet no rose is flat: SAM2's peaks sit on the CARDINAL axis and YOLO's on the "
                     "pixel DIAGONAL, so each masker imprints its own grid signature — re-masking "
                     "exchanges one for the other rather than removing it")
    for prod in ("SAM2 re-mask", "YOLOv8-seg mask"):
        have = [s for s in scenes if st[prod][s].get("peak") is not None]
        pk = [st[prod][s]["peak_minus_sun"] for s in have]
        cd = [min(axial_distance(st[prod][s]["peak"], 0.0),
                  axial_distance(st[prod][s]["peak"], 90.0)) for s in have]
        dd = [min(axial_distance(st[prod][s]["peak"], 45.0),
                  axial_distance(st[prod][s]["peak"], 135.0)) for s in have]
        st[prod]["_summary"] = dict(
            source=source, n_sites=len(have),
            n_with_power=int(sum(bool(st[prod][s].get("has_power")) for s in have)),
            mean_peak_minus_sun=round(float(np.mean(pk)), 1),
            mean_peak_minus_cardinal=round(float(np.mean(cd)), 1),
            mean_peak_minus_diagonal=round(float(np.mean(dd)), 1), random_expectation=45.0)
    # The load-bearing claim of this figure is that the residual is INHERITED, i.e. the two producers
    # agree site by site. Aggregate means cannot show that, so the per-site agreement is recorded.
    agree = {sc: round(float(axial_distance(st["SAM2 re-mask"][sc]["peak"],
                                            st["YOLOv8-seg mask"][sc]["peak"])), 1)
             for sc in scenes
             if st["SAM2 re-mask"][sc].get("peak") is not None
             and st["YOLOv8-seg mask"][sc].get("peak") is not None}
    both_powered = {sc: v for sc, v in agree.items()
                    if st["SAM2 re-mask"][sc].get("has_power")
                    and st["YOLOv8-seg mask"][sc].get("has_power")}
    st["_sam_vs_yolo_peak_agreement"] = dict(
        per_site_deg=agree, median_deg=round(float(np.median(list(agree.values()))), 1),
        max_deg=round(float(max(agree.values())), 1),
        median_deg_both_powered=(round(float(np.median(list(both_powered.values()))), 1)
                                 if both_powered else None),
        n_both_powered=len(both_powered),
        note="axial distance between the two producers' fitted peaks at the same site; small values "
             "mean the residual is inherited from detection/imagery rather than added by re-masking. "
             "The `both_powered` variant drops sites where either producer's peak interval exceeds "
             f"{CI_POWER_LIMIT:.0f}deg, since agreement between two undetermined peaks is not "
             "evidence of anything.")
    return save(fig, "fig09_controls"), st


# =================================================================================================
# 4. Area / mask fidelity
# =================================================================================================

@figure("fig10_area",
        ["data/sam2_validation/t35_val_detections.csv",
         "data/summary/boulder_examples.pkl + boulder_examples_meta.json"],
        "Mask area against human tracing, with examples")
def fig10_area():
    t = read_csv(SAMV / "t35_val_detections.csv")
    m = t[t.h_area.notna() & (t.h_area > 0)].copy()
    m["r_yolo"] = m.area_yolo / m.h_area
    m["r_sam"] = m.area_sam / m.h_area
    bands = [(4, 6), (6, 8), (8, 10), (10, 14), (14, 60)]
    lab = [f"{lo}–{hi if hi < 60 else '+'}" for lo, hi in bands]
    bodies = ["mars", "moon", "earth"]

    ex_p = SUM / "boulder_examples.pkl"
    have_ex = ex_p.exists()
    ex = load_pickle(ex_p)["area_spread"] if have_ex else []
    nex = min(len(ex), 6)

    fig = plt.figure(figsize=(11.2, 6.6 if nex else 3.6))
    gs = fig.add_gridspec(2 if nex else 1, 1, height_ratios=[1.05, 1.0] if nex else [1],
                          hspace=0.62, top=0.86, bottom=0.07, left=0.06, right=0.985)
    top = gs[0].subgridspec(1, 3, wspace=0.10)
    st = dict(by_body={}, overall={})
    axs = []
    for k, b in enumerate(bodies):
        # One shared y-axis across the three bodies: the point is that the pattern is the same
        # everywhere, which is only readable if the panels are on one scale.
        ax = fig.add_subplot(top[0, k], sharey=axs[0] if axs else None)
        axs.append(ax)
        g = m[m.body == b]
        for tag, col, colr in (("YOLO / human", "r_yolo", "yolo"), ("SAM2 / human", "r_sam", "sam")):
            ys, ns = [], []
            for lo, hi in bands:
                s = g[(g.h_dpx >= lo) & (g.h_dpx < hi)]
                ys.append(float(s[col].median()) if len(s) > 15 else np.nan)
                ns.append(int(len(s)))
            off = -0.18 if col == "r_yolo" else 0.18
            ax.bar(np.arange(len(bands)) + off, ys, 0.36, color=fs.color_of(colr),
                   edgecolor=fs.INK, lw=0.3, label=tag)      # labels are per-axes; legend on the last
            st["by_body"].setdefault(b, {})[tag] = dict(
                bands=lab, median_ratio=[None if not np.isfinite(v) else round(v, 3) for v in ys],
                n=ns)
        ax.axhline(1.0, ls=(0, (4, 3)), lw=1.0, color=fs.NULL_GREY)
        ax.annotate("= human area", xy=(0.985, 1.0), xycoords=("axes fraction", "data"),
                    fontsize=6.6, color=fs.NULL_GREY, ha="right", va="center",
                    bbox=dict(fc="white", ec="none", pad=1.2))
        ax.set_xticks(np.arange(len(bands)), lab, fontsize=7)
        ax.set_title(f"{b.capitalize()}  (n = {len(g):,})", fontsize=9)
        ax.set_xlabel("human equivalent diameter (px)", fontsize=7.5)
        if k == 0:
            ax.set_ylabel("median mask area ÷ human area", fontsize=8)
        else:
            ax.tick_params(labelleft=False)
        if k == len(bodies) - 1:
            ax.legend(fontsize=7.5, loc="upper right")
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    for b in bodies:
        g = m[m.body == b]
        st["overall"][b] = dict(n=int(len(g)), median_r_yolo=round(float(g.r_yolo.median()), 3),
                                median_r_sam=round(float(g.r_sam.median()), 3))

    if nex:
        bot = gs[1].subgridspec(1, nex, wspace=0.10)
        for j, r in enumerate(ex[:nex]):
            ax = fig.add_subplot(bot[0, j])
            _draw_outlines(ax, r["patch"],
                           [("human", r["hpoly"], None, "human"),
                            ("yolo", r["ymask"], None, "yolo"),
                            ("sam", r["smask"], None, "sam")],
                           title=f"{r['body']}  ·  {r['dpx']:.0f} px",
                           subtitle=f"YOLO {r['r_yolo']:.1f}×  ·  SAM {r['r_sam']:.1f}×")
        st["examples"] = [dict(body=r["body"], dpx=round(r["dpx"], 1),
                              r_yolo=round(r["r_yolo"], 2), r_sam=round(r["r_sam"], 2))
                          for r in ex[:nex]]
    fig.suptitle("Mask area against hand tracing — YOLO's blocky masks over-report area, SAM2's do "
                 "not\n(the same blockiness that quantizes the angle also inflates the area; both "
                 "feed the size–frequency distribution)", fontsize=10.6, y=0.985, va="top")
    if nex:
        fig.text(0.5, 0.012, "lower row: representative examples (median-to-85th percentile of the "
                             "YOLO/human ratio, not the extreme tail)  ·  "
                             "human = orange, YOLO = blue, SAM2 = green",
                 ha="center", va="bottom", fontsize=7.2, color=fs.INK_SOFT)
    return save(fig, "fig10_area"), st


# =================================================================================================
# 5. Multi-scale fusion
# =================================================================================================

ARMS = [("512", "arm_512", "512-px slices"),
        ("256", "arm_256", "256-px slices (production)"),
        ("256sr2", "arm_sr2", "super-resolved ×2")]


@figure("fig11_fusion_gallery",
        ["data/solutionDE/preds/rom_ladder/hirise_rom10/*.gpkg",
         "data_prieur/.../ESP_017355_2260-boulder-mapping.shp"],
        "Fused detections coloured by source arm, against human mapping")
def fig11_fusion_gallery():
    import geopandas as gpd
    import rasterio
    import shapely
    from rasterio.windows import from_bounds as win_from_bounds
    from YOLOv8BeyondEarth.fusion import fuse_predictions, pool_predictions

    TILE, RES = "hirise_rom10", 0.5
    tif = DE / f"{TILE}.tif"
    base = DE / "preds" / "rom_ladder" / TILE
    arms = {}
    for tag, _, _ in ARMS:
        p = base / (f"{TILE}-predictions-ct-010-ss-{tag}-is-1024-ov-020-downscaled-mask-nms.gpkg")
        if not p.exists():
            raise FileNotFoundError(p)
        arms[(256, 2) if tag == "256sr2" else int(tag)] = gpd.read_file(p)
    hum = (gpd.read_file(PRIEUR / "mars/ESP_017355_2260/shp/ESP_017355_2260-boulder-mapping.shp")
           .set_crs(arms[256].crs, allow_override=True))            # override, never to_crs
    fused, audit = fuse_predictions(arms, rule="bands", bands={256: (0.0, np.inf)},
                                    native_res=RES, engulf_thr=0.7)

    with rasterio.open(tif) as src:
        tb = src.bounds
    hin = hum[hum.geometry.centroid.within(shapely.box(*tb))]
    cx, cy = hin.geometry.centroid.x.round(-2), hin.geometry.centroid.y.round(-2)
    cell = (cx.astype(str) + "_" + cy.astype(str)).value_counts().idxmax().split("_")
    half = 100 * RES
    box = shapely.box(float(cell[0]) - half, float(cell[1]) - half,
                      float(cell[0]) + half, float(cell[1]) + half)

    def bg(ax):
        with rasterio.open(tif) as src:
            w = win_from_bounds(*box.bounds, src.transform)
            img = src.read(1, window=w, boundless=True, fill_value=0)
        p1, p99 = np.percentile(img[img > 0], [1, 99])
        ax.imshow(np.clip((img.astype(float) - p1) / (p99 - p1), 0, 1), cmap="gray",
                  extent=(box.bounds[0], box.bounds[2], box.bounds[1], box.bounds[3]))
        ax.set_xticks([]); ax.set_yticks([])

    def draw(ax, gdf, color, lw=1.1, ls="-"):
        sel = gdf[gdf.geometry.intersects(box)]
        for g in sel.geometry:
            xs, ys = g.exterior.xy
            ax.plot(xs, ys, color=fs.color_of(color), lw=lw, ls=ls,
                    path_effects=fs.outline_effects(lw))
        return len(sel)

    key_of = {"512": 512, "256": 256, "256sr2": (256, 2)}
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.6))
    bg(axes[0])
    n_h = draw(axes[0], hin, "human", lw=1.3, ls=(0, (4, 2)))
    axes[0].set_title(f"human mapping — {n_h} boulders", fontsize=9.5)
    bg(axes[1])
    counts = {}
    for tag, colr, _ in ARMS:
        sub = fused[fused.scale_key.astype(str) == str(key_of[tag])]
        counts[tag] = draw(axes[1], sub, colr)
    axes[1].set_title(f"fused catalogue — {sum(counts.values())} boulders, coloured by which "
                      f"arm found them", fontsize=9.5)
    handles = [plt.Line2D([], [], color=fs.PALETTE[c], lw=2, label=f"{lab}: {counts[t]}")
               for t, c, lab in ARMS]
    handles.append(plt.Line2D([], [], color=fs.PALETTE["human"], lw=2, ls=(0, (4, 2)),
                              label=f"human mapping: {n_h}"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 0.035))
    fig.text(0.5, 0.005, "This window is small-boulder dominated, so the 512 arm contributes almost "
                         "nothing here; its payoff is large boulders that every 256 slice cuts in "
                         "two (14+ px recall 0.83 → 0.91, next figure).",
             ha="center", va="bottom", fontsize=7.2, color=fs.INK_SOFT)
    fig.suptitle(f"Multi-scale fusion on the densest human-mapped patch of {TILE} "
                 f"(200 native px ≈ 100 m across)\nthe production 256 pass keeps its own masks; "
                 f"the other arms only ADD boulders it missed", fontsize=10.4, y=0.995, va="top")
    fig.tight_layout(rect=(0, 0.09, 1, 0.92))
    st = dict(tile=TILE, window_m=[round(v, 1) for v in box.bounds], n_human_in_window=int(n_h),
              per_arm_in_window=counts, n_fused_total=int(len(fused)),
              n_pool_total=int(len(pool_predictions(arms))),
              n_engulfed=int(audit.get("engulfed", pd.Series(dtype=bool)).sum())
              if hasattr(audit, "get") else None)
    return save(fig, "fig11_fusion_gallery"), st


@figure("fig12_size_recall",
        ["data/solutionDE/goal3_fullscene_sfd.csv", "data/solutionDE/goal3_rom_recall.csv"],
        "Where the added boulders are, and recall against human mapping")
def fig12_size_recall():
    sfd = read_csv(DE / "goal3_fullscene_sfd.csv")
    rec = read_csv(DE / "goal3_rom_recall.csv")
    bands = [(2, 4), (4, 6), (6, 8), (8, 10), (10, 14), (14, 60)]
    lab = [f"{lo}–{hi if hi < 60 else '+'}" for lo, hi in bands]

    def band_counts(scene, catalog):
        g = sfd[(sfd.scene == scene) & (sfd.catalog == catalog)].set_index("d_px").n_gt
        return [int(g.loc[float(lo)] - g.loc[float(hi)]) for lo, hi in bands]

    fig, ax = plt.subplots(1, 2, figsize=(11.0, 3.9))
    st = dict(band_counts={}, recall={})
    scene, nice = "hirise", "Mars HiRISE ESP_017355_2260 (full scene)"
    base, fused = band_counts(scene, "256_only"), band_counts(scene, "fused")
    added = [f - b for f, b in zip(fused, base)]
    x = np.arange(len(bands))
    ax[0].bar(x, base, 0.64, color=fs.color_of("arm_256"), edgecolor=fs.INK, lw=0.3,
              label="found by the production 256 pass")
    ax[0].bar(x, added, 0.64, bottom=base, color=fs.color_of("arm_sr2"), edgecolor=fs.INK, lw=0.3,
              label="added by the 512 + super-resolved arms")
    ax[0].set_yscale("log")
    ax[0].set_xticks(x, lab)
    ax[0].set_xlabel("equivalent diameter (native px)", fontsize=8)
    ax[0].set_ylabel("boulders in band", fontsize=8)
    ax[0].set_title(f"{nice}\ncatalogue grows {sum(base):,} → {sum(fused):,} "
                    f"(+{100*sum(added)/sum(base):.0f}%)", fontsize=9)
    ax[0].set_ylim(top=max(fused) * 60)          # headroom for the legend + caveat
    ax[0].legend(fontsize=7.3, loc="upper right", framealpha=1.0)
    ax[0].text(0.02, 0.97, "the 2–6 px additions are precision-gated:\nno size-frequency claim is "
                           "made below the old floor", transform=ax[0].transAxes, fontsize=6.6,
               va="top", ha="left", color=fs.PALETTE["human"])
    st["band_counts"][scene] = dict(bands=lab, baseline=base, fused=fused, added=added)

    PROD = "fused_256pref_engulf"
    for arm, colr, nm in (("arm_256", "arm_256", "production 256 pass"),
                          (PROD, "arm_sr2", "fused ladder (production rule)")):
        g = rec[(rec.scene == "hirise") & (rec.catalog == arm)]
        agg = g.groupby("band").apply(
            lambda s: pd.Series(dict(n_human=s.n_human.sum(), n_matched=s.n_matched.sum())),
            include_groups=False)
        agg["recall"] = agg.n_matched / agg.n_human
        order = [b for b in ["3-4", "4-6", "6-8", "8-10", "10-14", "14-inf"] if b in agg.index]
        ax[1].plot(range(len(order)), agg.loc[order, "recall"].values, "o-",
                   color=fs.color_of(colr), lw=1.8, ms=6, label=nm,
                   path_effects=fs.outline_effects(1.8))
        st["recall"][arm] = dict(bands=order,
                                 recall=[round(float(v), 3) for v in agg.loc[order, "recall"]],
                                 n_human=[int(v) for v in agg.loc[order, "n_human"]])
    ax[1].set_xticks(range(len(order)), order)
    ax[1].set_xlabel("human equivalent diameter (native px)", fontsize=8)
    ax[1].set_ylabel("fraction of human-mapped boulders recovered", fontsize=8)
    ax[1].set_title("Recall against the human mapping\n(20 ROM tiles, 5,324 labels)", fontsize=9)
    ax[1].legend(fontsize=7.3, loc="lower right")
    for a_ in ax:
        for s_ in ("top", "right"):
            a_.spines[s_].set_visible(False)
    fig.tight_layout()
    return save(fig, "fig12_size_recall"), st


@figure("fig13_sfd", ["data/solutionDE/goal3_fullscene_sfd.csv",
                      "data/solutionDE/goal3_fullscene_sfd_fit.csv"],
        "Size–frequency distribution and the completeness roll-off")
def fig13_sfd():
    sfd = read_csv(DE / "goal3_fullscene_sfd.csv")
    fit = read_csv(DE / "goal3_fullscene_sfd_fit.csv")
    scenes = [("hirise", "Mars HiRISE ESP_017355_2260"), ("nac", "Moon LRO NAC M139694087LE")]
    cols = {"256_only": ("arm_256", "production 256 pass"),
            "fused": ("arm_sr2", "fused multi-scale ladder")}
    fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.1))
    st = {}
    for a_, (sc, nice) in zip(ax, scenes):
        for cat, (colr, nm) in cols.items():
            d = sfd[(sfd.scene == sc) & (sfd.catalog == cat) & (sfd.per_km2 > 0)]
            a_.plot(d.d_px, d.per_km2, color=fs.color_of(colr), lw=1.8, label=nm,
                    path_effects=fs.outline_effects(1.8))
            f = fit[(fit.scene == sc) & (fit.catalog == cat)].iloc[0]
            e = dict(rolloff_px=None if not np.isfinite(f.rolloff_px) else float(f.rolloff_px),
                     lam=None if not np.isfinite(f.lam) else round(float(f.lam), 3))
            if np.isfinite(f.lam):
                anchor = d[d.d_px.between(10, 30)].iloc[0]
                a_.plot(d.d_px, anchor.per_km2 * np.exp(f.lam * (d.d_px - anchor.d_px)),
                        color=fs.color_of(colr), ls=":", lw=1.0)
            if np.isfinite(f.rolloff_px):
                a_.axvline(f.rolloff_px, color=fs.color_of(colr), ls="--", lw=1.0, alpha=0.75)
                a_.annotate(f"roll-off {f.rolloff_px:g} px", xy=(f.rolloff_px, 0.98),
                            xycoords=("data", "axes fraction"), rotation=90, fontsize=6.6,
                            color=fs.color_of(colr), ha="right", va="top")
            st.setdefault(sc, {})[cat] = e
        a_.set_xlim(2, 40)
        a_.set_yscale("log")
        a_.set_xlabel("equivalent diameter (native px)", fontsize=8)
        a_.set_ylabel("N(>D) per km²", fontsize=8)
        a_.set_title(nice, fontsize=9.5)
        a_.legend(fontsize=7.3)
        for s_ in ("top", "right"):
            a_.spines[s_].set_visible(False)
    ax[1].text(0.03, 0.06, "no roll-off is detectable on this scene:\nits size distribution is too "
                           "flat (λ −0.17 vs −0.36)\nfor the fit to separate — the estimator is "
                           "uninformative here", transform=ax[1].transAxes, fontsize=6.8,
               va="bottom", color=fs.INK_SOFT)
    fig.suptitle("Cumulative size–frequency distribution — fusion extends completeness by ~2 px on "
                 "the dense scene\n(dotted = exponential law fitted on the clearly-complete "
                 "10–30 px range; dashed = where the measurement falls 10% below it)",
                 fontsize=10.2, y=1.02, va="top")
    fig.tight_layout()
    return save(fig, "fig13_sfd"), st


# =================================================================================================
# 6. Speed
# =================================================================================================

@figure("fig14_speed", ["notebooks/Goal1_inference_performance (measured tables)",
                        "data/solutionDE/goal3_fullscene_cost.csv",
                        "docs/solutions/G_sam2_remasking.md (measured ms/boulder)"],
        "Where the time goes, and what each optional pass costs")
def fig14_speed():
    # Measured values, transcribed from the Goal-1 profiling notebook (cProfile on a 4096² HiRISE
    # crop at ~16.5k detections) and its full-image timing table. Kept here rather than recomputed
    # because re-running a multi-hour full-scene benchmark to redraw a bar chart is not justified;
    # the provenance is recorded in figure_stats.json.
    profile = [("per-detection mask → polygon loop", 73.0),
               ("GPU inference", 17.0),
               ("vector output + CRS", 6.0),
               ("everything else", 4.0)]
    # (label, before_min, after_min, before_is_measured). The HiRISE "before" was never run to
    # completion at full scene scale — it is the measured 2.23x crop speed-up extrapolated — so it is
    # drawn hatched and labelled as an estimate rather than passed off as a measurement.
    scenes = [("Moon NAC M139694087LE\n303 Mpx", 4.79, 3.46, True),
              ("Mars HiRISE ESP_017355_2260\n1,930 Mpx", 123.0, 55.1, False)]
    cost = read_csv(DE / "goal3_fullscene_cost.csv")

    fig, ax = plt.subplots(1, 3, figsize=(12.4, 3.9))
    lbl = [p[0] for p in profile]
    val = [p[1] for p in profile]
    greys = ["#3f3f3c", "#6f6e6a", "#9d9c97", "#c9c8c4"]
    left = 0.0
    for l, v, c in zip(lbl, val, greys):
        ax[0].barh([0], [v], left=left, color=c, edgecolor="white", lw=1.2,
                   label=f"{l} — {v:.0f}%")
        if v > 12:                                     # only wide segments get an in-place label
            ax[0].text(left + v / 2, 0, f"{l}\n{v:.0f}%", ha="center", va="center", fontsize=7,
                       color="white" if c == greys[0] else fs.INK)
        left += v
    ax[0].set_xlim(0, 100)
    ax[0].set_ylim(-1.5, 0.5)
    ax[0].set_yticks([])
    ax[0].set_xlabel("% of wall clock", fontsize=8)
    ax[0].legend(fontsize=6.6, loc="lower center", ncol=2, frameon=False,
                 handlelength=1.1, columnspacing=1.0)
    ax[0].set_title("The bottleneck is CPU polygon extraction,\nnot the GPU (≈1e6 detections)",
                    fontsize=9)
    for s_ in ("top", "right", "left"):
        ax[0].spines[s_].set_visible(False)

    x = np.arange(len(scenes))
    for i, s in enumerate(scenes):
        ax[1].bar([x[i] - 0.19], [s[1]], 0.38, color=fs.NULL_GREY, edgecolor=fs.INK, lw=0.3,
                  hatch=None if s[3] else "///",
                  label=("before optimisation" if i == 0 else None))
    ax[1].bar(x + 0.19, [s[2] for s in scenes], 0.38, color=fs.color_of("arm_256"),
              edgecolor=fs.INK, lw=0.3, label="after (bit-identical output)")
    for i, s in enumerate(scenes):
        ax[1].text(i + 0.19, s[2] * 1.06, f"{s[1]/s[2]:.1f}×", ha="center", fontsize=8,
                   color=fs.INK)
        if not s[3]:
            ax[1].text(i - 0.19, s[1] * 1.06, "estimated", ha="center", fontsize=6.4,
                       color=fs.INK_SOFT)
    ax[1].set_yscale("log")
    ax[1].set_xticks(x, [s[0] for s in scenes], fontsize=7.5)
    ax[1].set_ylabel("whole-scene detection time (min)", fontsize=8)
    ax[1].set_title("Whole-scene detection, before vs after", fontsize=9)
    ax[1].legend(fontsize=7.3, loc="upper left")
    for s_ in ("top", "right"):
        ax[1].spines[s_].set_visible(False)

    h = cost[cost.scene == "hirise"].set_index("arm")
    # These bars are cumulative pipeline STAGES, not competing categories, so they take an ordinal
    # one-hue ramp (light -> dark) rather than the categorical arm colours. Reusing the arm hues here
    # would put the super-resolution arm and the SAM2 re-mask in the same colour.
    passes = [("detection\n256 px", float(h.at["ss256", "wall_s"]) / 3600, "large boulders"),
              ("+ 512 arm", float(h.at["ss512", "wall_s"]) / 3600, "large-boulder recall"),
              ("+ super-res", float(h.at["ss256sr2", "wall_s"]) / 3600, "sub-floor completeness"),
              ("+ SAM2\nre-mask", 9.0, "orientation + area")]
    ramp = ["#bcd7f2", "#7fb3e5", "#3f8ad4", "#1b4f8f"]
    ax[2].bar(range(len(passes)), [p[1] for p in passes], 0.62, color=ramp, edgecolor=fs.INK, lw=0.3)
    for i, p in enumerate(passes):
        ax[2].text(i, p[1] * 1.08, f"{p[1]:.2f} h" if p[1] < 1 else f"{p[1]:.1f} h", ha="center",
                   fontsize=7.4, color=fs.INK)
        ax[2].text(i, 0.052, p[2], ha="center", va="bottom", fontsize=6.2, rotation=90,
                   color=fs.INK_SOFT)
    ax[2].set_yscale("log")
    ax[2].set_ylim(0.05, 20)
    ax[2].set_xticks(range(len(passes)), [p[0] for p in passes], fontsize=7.2)
    ax[2].set_ylabel("added wall clock per HiRISE scene (h)", fontsize=8)
    ax[2].set_title("What each optional pass costs\n(1.93 Gpx scene, one laptop GPU)", fontsize=9)
    for s_ in ("top", "right"):
        ax[2].spines[s_].set_visible(False)
    fig.suptitle("Throughput: the detection pass is 1.4–2.2× faster with identical output; every "
                 "quality upgrade is a separately-priced pass", fontsize=10.4, y=1.03, va="top")
    fig.tight_layout()
    st = dict(profile_percent=dict(profile),
              scene_minutes=[dict(scene=s[0].replace("\n", " "), before=s[1], after=s[2],
                                  speedup=round(s[1] / s[2], 2),
                                  before_is_measured=s[3]) for s in scenes],
              added_hours_per_hirise_scene={p[0].replace("\n", " "): round(p[1], 2)
                                           for p in passes},
              sam_ms_per_boulder=6.5,
              note="profile + before/after are the Goal-1 measured tables; per-arm wall clocks are "
                   "goal3_fullscene_cost.csv; the SAM2 figure is the measured 6.5 ms/boulder "
                   "extrapolated ungated over the scene")
    return save(fig, "fig14_speed"), st


# =================================================================================================
# Appendix A — the OBB angle head
# =================================================================================================

@figure("figA1_obb", ["data/solutionH/obb_null_truth_rot.csv",
                      "data/solutionH/obb_null_pred_rot.csv",
                      "data/solutionH/obb_null_pred.csv"],
        "Appendix: a direct angle head, and why it is not the fix")
def figA1_obb():
    truth = read_csv(SH / "obb_null_truth_rot.csv")
    d0 = read_csv(SH / "obb_null_pred.csv")
    ra = read_csv(SH / "obb_null_pred_rot.csv")
    panels = [
        dict(angles=truth[truth.a_px / truth.b_px > ASPECT_MIN].angle180.values
             if "b_px" in truth else truth.angle180.values,
             color="truth", label="input: known uniform truth", stats=("grid", "diag")),
        dict(angles=d0[d0.aspect > ASPECT_MIN].angle180.values, color="obb_d0",
             label="angle head as trained", stats=("grid", "diag")),
        dict(angles=ra[ra.aspect > ASPECT_MIN].angle180.values, color="obb_ra",
             label="after rotation augmentation", stats=("grid", "diag")),
    ]
    fig, _ = fs.orientation_grid(
        [panels], aspect_min=ASPECT_MIN, dpx_min=None, share_hist=None,
        suptitle="Appendix A — a direct angle head must first pass a content-free null: feed it "
                 "boulders drawn at uniform-random angles and see what it returns",
        footer_extra="rotation augmentation removes most of the head's own cardinal prior but not "
                     "all of it, so it is a cheap per-boulder angle, not a population rose")
    st = dict(truth=stat_block(panels[0]["angles"]), as_trained=stat_block(panels[1]["angles"]),
              rotation_augmented=stat_block(panels[2]["angles"]),
              uniform_reference=round(UNIF, 3))
    return save(fig, "figA1_obb"), st


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=None, help="comma-separated figure names")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, v in REGISTRY.items():
            print(f"{k:30s} {v['title']}")
            for i in v["inputs"]:
                print(f"{'':30s}   <- {i}")
        return
    names = ([n.strip() for n in a.only.split(",")] if a.only else list(REGISTRY))
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise SystemExit(f"unknown figure(s): {unknown}\navailable: {list(REGISTRY)}")
    fs.apply_rcparams()
    FIG.mkdir(parents=True, exist_ok=True)
    sp = FIG / "figure_stats.json"
    allstats = json.loads(sp.read_text()) if sp.exists() else {}
    failed = []
    undeclared = []
    for n in names:
        print(f"[{n}] {REGISTRY[n]['title']}")
        _STATE.update(fig=None, opened=[], save_png=True)
        try:
            path, st = REGISTRY[n]["fn"]()
        except Exception as e:                                    # noqa: BLE001 - report, continue
            print(f"    !! FAILED: {type(e).__name__}: {e}")
            failed.append((n, f"{type(e).__name__}: {e}"))
            continue
        # Cross-check the hand-written `inputs` against what the function actually opened. A figure
        # that reads a file nobody declared is a provenance claim that has drifted from the code, so
        # it is reported rather than left to be discovered by a reader.
        measured = list(_STATE["opened"])
        declared = " ".join(REGISTRY[n]["inputs"])
        for e in measured:
            stem = Path(e["path"]).name
            if stem not in declared:
                undeclared.append((n, e["path"]))
        allstats[n] = dict(figure=path, inputs=REGISTRY[n]["inputs"],
                           measured_inputs=measured, title=REGISTRY[n]["title"], stats=st)
    sp.write_text(json.dumps(allstats, indent=2, default=str))
    print(f"\nstats -> {sp.relative_to(REPO)}")
    if undeclared:
        print("\nUNDECLARED INPUTS (a figure read a file its decorator does not mention —\n"
              "update the decorator so the document's provenance line stays true):")
        for n, p in undeclared:
            print(f"  {n}: {p}")
    if failed:
        print("\nFAILED figures (nothing was silently skipped):")
        for n, e in failed:
            print(f"  {n}: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
