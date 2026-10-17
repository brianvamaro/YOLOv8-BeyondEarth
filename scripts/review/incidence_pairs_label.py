"""Same-ground HiRISE repeat pairs: how much does incidence move hirise2ctx's label, and does the score
floor widen or narrow that? (2026-10-16)

This is the configuration-study audit's analysis (finding A3), promoted from its scratch script into
tracked code. The audit found that the study never mentioned illumination, and that on repeat pairs it
outweighs every configuration and floor difference.

For each pair:
- **Frame.** Both members are mapped into the low-incidence member's PDS-label frame through
  longitude/latitude (X-04 CRS, X-07 origin).
- **Cells.** A 40 m grid is laid over the ground valid in both members (5x5 samples per cell). The
  E-W width is corrected for lat_ts versus latitude.
- **Label.** Each member's per-cell area fraction is computed the way notebook 35's
  `areal_abundance` does it: polygons are unioned, and boulders >= the size floor are counted at each
  score floor. A spot check against `areal_abundance` is recorded.

Outputs per pair:
- the lo/hi label ratio at 40 and 160 m, with and without a co-registration shift (the cross-correlation
  lag of centroid density, within +-200 m), and on all cells or only cells clear of near-cap slices;
- Spearman, count ratios, and the share of each member's 0.25 label kept at 0.40;
- a block bootstrap over 160 m blocks, for the ratio and for how the floor changes it (r0.40 / r0.25).

Limits, carried into the JSON:
- The catalogs (`data/test2_pairs/`) are contract 1, with an effective conf floor of 0.25 (X-01) and
  max_det 300 per 256 slice, so floors below 0.25 are untested. Cells touched by a slice at >= 270
  detections are reported apart.
- There is no human truth here. This measures how consistent two labels of the same ground are, not
  which one is right.

    conda run -n bouldernet python scripts/review/incidence_pairs_label.py --out data/review/incidence_pairs_label.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyogrio
import rasterio
import shapely
from pyproj import Transformer
from rasterio.enums import Resampling
from sahi.slicing import get_slice_bboxes
from scipy import ndimage, stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import config_study as cs                                                   # noqa: E402
from YOLOv8BeyondEarth import crs_fix, pds_prep                             # noqa: E402

D = REPO / "data" / "test2_pairs"
WORK = REPO / "data" / "incidence_pairs"          # per-pair intermediates (gitignored)
#: (first, second): the lower-incidence member first where incidence differs; the controls' members
#: differ by under 1 degree, so their order is arbitrary
PAIRS = {1: ("ESP_071923_2260", "ESP_071699_2260"), 2: ("ESP_076235_2260", "ESP_055833_2260"),
         3: ("ESP_069088_2270", "ESP_067214_2270"), 4: ("ESP_059499_2270", "ESP_055714_2270"),
         5: ("PSP_007703_2265", "PSP_007492_2265")}
#: names the earlier record (orientation_results_summary §T2) used for the same pairs
T2_NAMES = {2: "2260b", 3: "2270a", 4: "2270b"}
FLOORS = (0.25, 0.30, 0.40, 0.50)
DMINS = (1.4105, 2.0)
CAP = 270                       # a 256 slice at >= 270 of contract 1's 300 is treated as near the cap
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


class Scene:
    def __init__(self, obs):
        self.obs = obs
        d = D / obs
        self.tif, self.lbl = d / f"{obs}_RED.tif", d / f"{obs}_RED.LBL"
        self.gpkg = d / f"{obs}_RED-predictions-ct-010-ss-256-is-1024-ov-020-downscaled-mask-nms.gpkg"
        f = pds_prep.read_label(self.lbl)
        s = pds_prep.label_summary(f)
        self.inc = float(s["incidence_angle"])
        self.emi = float(pds_prep._lookup(f, "EMISSION_ANGLE"))
        self.crs = crs_fix.crs_from_pds_label(self.lbl)           # authoritative (X-04)
        self.ox, self.oy = pds_prep.label_frame_origin(s)          # label frame (X-07)
        self.scale = float(s["map_scale"])
        self.center_lat = float(s["center_latitude"])
        with rasterio.open(self.tif) as r:
            self.W, self.H, self.T = r.width, r.height, r.transform
        assert abs(self.T.a - self.scale) < 1e-9 and abs(-self.T.e - self.scale) < 1e-9
        self.dpx = ((self.T.c - self.ox) / self.scale, (self.oy - self.T.f) / self.scale)

    def tif_to_px(self, xy):
        return np.column_stack([(xy[:, 0] - self.T.c) / self.T.a, (xy[:, 1] - self.T.f) / self.T.e])

    def px_to_lab(self, cr):
        return np.column_stack([self.ox + cr[:, 0] * self.scale, self.oy - cr[:, 1] * self.scale])

    def lab_to_px(self, xy):
        return np.column_stack([(xy[:, 0] - self.ox) / self.scale, (self.oy - xy[:, 1]) / self.scale])


def make_maps(A, B):
    """Map each member's pixel coordinates to and from the reference (A's label) frame via lon/lat."""
    A_inv = Transformer.from_crs(A.crs, A.crs.geodetic_crs, always_xy=True)
    A_fwd = Transformer.from_crs(A.crs.geodetic_crs, A.crs, always_xy=True)
    B_inv = Transformer.from_crs(B.crs, B.crs.geodetic_crs, always_xy=True)
    B_fwd = Transformer.from_crs(B.crs.geodetic_crs, B.crs, always_xy=True)

    def px2ref(S, cr):
        lab = S.px_to_lab(cr)
        if S is A:
            return lab
        lon, lat = B_inv.transform(lab[:, 0], lab[:, 1])
        return np.column_stack(A_fwd.transform(lon, lat))

    def ref2px(S, xy):
        if S is A:
            return S.lab_to_px(xy)
        lon, lat = A_inv.transform(xy[:, 0], xy[:, 1])
        return S.lab_to_px(np.column_stack(B_fwd.transform(lon, lat)))
    return px2ref, ref2px


def valid_mask(S):
    f = max(1, int(round(2.0 / S.scale)))
    with rasterio.open(S.tif) as r:
        a = r.read(1, out_shape=(S.H // f, S.W // f), resampling=Resampling.nearest)
    return ndimage.binary_erosion(a > 0, iterations=2), f


def slice_counts(S, cx, cy):
    """Detections per production 256 slice (centroid membership), to find slices near the cap."""
    bb = np.asarray(get_slice_bboxes(image_height=S.H, image_width=S.W, slice_height=256,
                                     slice_width=256, overlap_height_ratio=0.2,
                                     overlap_width_ratio=0.2))
    xs, ys = np.unique(bb[:, [0, 2]], axis=0), np.unique(bb[:, [1, 3]], axis=0)
    cnt = np.zeros((len(ys), len(xs)), dtype=np.int64)

    def memb(c, ax):
        lo = np.searchsorted(ax[:, 1], c, side="right")
        out = []
        for off in range(4):
            k = lo + off
            ok = k < len(ax)
            kk = np.where(ok, k, 0)
            out.append((kk, ok & (ax[kk, 0] <= c) & (ax[kk, 1] > c)))
        return out
    for kx, okx in memb(cx, xs):
        for ky, oky in memb(cy, ys):
            ok = okx & oky
            np.add.at(cnt, (ky[ok], kx[ok]), 1)
    return xs, ys, cnt


def load_preds(le, S, px2ref):
    g = pyogrio.read_dataframe(S.gpkg, columns=["score"])
    score = g["score"].to_numpy()
    geoms = np.asarray(g.geometry.values, dtype=object)
    keep = ~shapely.is_missing(geoms)
    geoms, score = geoms[keep], score[keep]
    inv = ~shapely.is_valid(geoms)
    for i in np.flatnonzero(inv):
        r = le.repair([geoms[i]])
        geoms[i] = r[0] if r.size else None
    keep = ~shapely.is_missing(geoms) & np.isin(shapely.get_type_id(geoms), (3, 6))
    n_drop = int((~keep).sum())
    geoms, score = geoms[keep], score[keep]
    cpx = S.tif_to_px(shapely.get_coordinates(shapely.centroid(geoms)))
    geoms_ref = shapely.transform(geoms, lambda c: px2ref(S, S.tif_to_px(c)))
    bad = ~shapely.is_valid(geoms_ref)
    if bad.any():
        geoms_ref[bad] = shapely.make_valid(geoms_ref[bad])
    return dict(geoms=geoms_ref, score=score, diam=le.equivalent_diameter_m(geoms_ref), cpx=cpx,
                cref=shapely.get_coordinates(shapely.centroid(geoms_ref)), n_raw=int(len(g)),
                n_invalid=int(inv.sum()), n_drop=n_drop)


def overlap_pairs(geoms):
    i, j = shapely.STRtree(geoms).query(geoms, predicate="intersects")
    m = i < j
    return i[m], j[m]


def pieces_for(P, pi, pj, mask):
    """Polygons above the floors, with overlapping ones unioned (as areal_abundance does)."""
    pm = mask[pi] & mask[pj]
    inv = np.zeros(len(mask), bool)
    inv[pi[pm]] = True
    inv[pj[pm]] = True
    solo = P["geoms"][mask & ~inv]
    if inv.any():
        parts = shapely.get_parts(shapely.union_all(P["geoms"][inv]))
        return np.concatenate([solo, parts[np.isin(shapely.get_type_id(parts), (3,))]])
    return solo


def cell_fa(pieces, cells, ctree, cell_area, shift=(0.0, 0.0)):
    if shift != (0.0, 0.0):
        pieces = shapely.transform(pieces, lambda c: c + np.asarray(shift))
    if not len(pieces):
        return np.zeros(len(cells))
    p, c = ctree.query(pieces, predicate="intersects")
    a = shapely.area(shapely.intersection(pieces[p], cells[c]))
    return np.bincount(c, weights=a, minlength=len(cells)) / cell_area


def run_pair(pno):
    le, sha = cs.load_label_error()
    lo_obs, hi_obs = PAIRS[pno]
    A, B = Scene(lo_obs), Scene(hi_obs)
    px2ref, ref2px = make_maps(A, B)
    res = dict(pair=pno, t2_name=T2_NAMES.get(pno), members={}, label_error_sha256=sha)
    for S in (A, B):
        res["members"][S.obs] = dict(incidence=S.inc, emission=S.emi, scale=S.scale,
                                     tif_minus_label_origin_px=list(map(float, S.dpx)))
    log(pno, A.obs, A.inc, B.obs, B.inc)
    VM = {S.obs: valid_mask(S) for S in (A, B)}

    def ref_hull(S):
        v, f = VM[S.obs]
        rr, cc = np.nonzero(v[::4, ::4])
        pts = np.column_stack([cc * 4 * f, rr * 4 * f]).astype(float)
        return shapely.convex_hull(shapely.multipoints(px2ref(S, pts)))
    inter = shapely.intersection(ref_hull(A), ref_hull(B))
    if inter.is_empty:
        raise SystemExit("no overlap")
    xmin, ymin, xmax, ymax = inter.bounds
    lat = le.latitude_of(A.crs, (xmin + xmax) / 2, (ymin + ymax) / 2)
    kx = np.cos(np.radians(A.center_lat)) / np.cos(np.radians(lat))
    dx, dy = 40.0 * kx, 40.0
    x0, y0 = np.floor(xmin / (4 * dx)) * 4 * dx, np.floor(ymin / (4 * dy)) * 4 * dy
    nx, ny = int(np.ceil((xmax - x0) / (4 * dx))) * 4, int(np.ceil((ymax - y0) / (4 * dy))) * 4
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny))
    gx, gy = gx.ravel(), gy.ravel()
    cells_all = shapely.box(x0 + gx * dx, y0 + gy * dy, x0 + (gx + 1) * dx, y0 + (gy + 1) * dy)
    fr = (np.arange(5) + 0.5) / 5
    sx, sy = (a.ravel() for a in np.meshgrid(fr, fr))
    ok = np.ones(len(gx), bool)
    for S in (A, B):
        v, f = VM[S.obs]
        for a_, b_ in zip(sx, sy):
            cr = ref2px(S, np.column_stack([x0 + (gx + a_) * dx, y0 + (gy + b_) * dy])) / f
            c, r = np.floor(cr[:, 0]).astype(int), np.floor(cr[:, 1]).astype(int)
            inside = (c >= 0) & (c < v.shape[1]) & (r >= 0) & (r < v.shape[0])
            vv = np.zeros(len(c), bool)
            vv[inside] = v[r[inside], c[inside]]
            ok &= vv
    cells, cgx, cgy = cells_all[ok], gx[ok], gy[ok]
    ctree, cell_area = shapely.STRtree(cells), dx * dy
    res["n_cells40_valid"] = int(ok.sum())
    P, CAPPED = {}, {}
    for S in (A, B):
        P[S.obs] = Pm = load_preds(le, S, px2ref)
        xs, ys, cnt = slice_counts(S, Pm["cpx"][:, 0], Pm["cpx"][:, 1])
        boxes = [shapely.polygons(px2ref(S, np.array([[xs[b_][0], ys[a_][0]], [xs[b_][1], ys[a_][0]],
                                                     [xs[b_][1], ys[a_][1]], [xs[b_][0], ys[a_][1]]],
                                                    float)))
                 for a_, b_ in zip(*np.nonzero(cnt >= CAP))]
        capc = np.zeros(len(cells), bool)
        if boxes:
            _, ci = ctree.query(np.asarray(boxes, dtype=object), predicate="intersects")
            capc[np.unique(ci)] = True
        CAPPED[S.obs] = capc
        res["members"][S.obs].update(n_det=len(Pm["geoms"]), slices_ge270=int((cnt >= CAP).sum()),
                                     slices_ge300=int((cnt >= 300).sum()), slice_count_max=int(cnt.max()),
                                     cells40_near_cap=int(capc.sum()), n_drop=Pm["n_drop"])
    clear = ~(CAPPED[A.obs] | CAPPED[B.obs])
    # co-registration lag from centroid density (5 m bins, score >= 0.25, >= 1.41 m)
    for S in (A, B):
        q, _ = ctree.query(shapely.points(P[S.obs]["cref"]), predicate="intersects")
        m = np.zeros(len(P[S.obs]["geoms"]), bool)
        m[q] = True
        P[S.obs]["in_common"] = m

    def dens(S):
        Pm = P[S.obs]
        c = Pm["cref"][Pm["in_common"] & (Pm["score"] >= 0.25) & (Pm["diam"] >= DMINS[0])]
        nbx, nby = int(np.ceil(nx * dx / 5.0)), int(np.ceil(ny * dy / 5.0))
        h = np.zeros((nby, nbx))
        np.add.at(h, (np.clip(((c[:, 1] - y0) / 5.0).astype(int), 0, nby - 1),
                      np.clip(((c[:, 0] - x0) / 5.0).astype(int), 0, nbx - 1)), 1.0)
        return ndimage.gaussian_filter(h, 1.0)
    ha, hb = dens(A), dens(B)
    mask = (ha + hb) > 0
    a_, b_ = np.where(mask, ha - ha[mask].mean(), 0), np.where(mask, hb - hb[mask].mean(), 0)
    xc = np.fft.fftshift(np.fft.irfft2(np.fft.rfft2(a_) * np.conj(np.fft.rfft2(b_)), s=a_.shape))
    L = 40
    cy_, cx_ = np.array(xc.shape) // 2
    win = xc[cy_ - L:cy_ + L + 1, cx_ - L:cx_ + L + 1]
    iy, ix = np.unravel_index(np.argmax(win), win.shape)
    lag = ((ix - L) * 5.0, (iy - L) * 5.0)
    norm = np.sqrt((a_ ** 2).sum() * (b_ ** 2).sum()) + 1e-12
    res["coreg"] = dict(lag_B_to_A_m=list(lag), lag_norm_m=float(np.hypot(*lag)),
                        xcorr_peak=float(win.max() / norm), xcorr_zero=float(win[L, L] / norm))
    OV = {S.obs: overlap_pairs(P[S.obs]["geoms"]) for S in (A, B)}
    key160 = (cgy // 4) * (nx // 4) + (cgx // 4)

    def agg160(v, sel):
        _, inv_, cnt = np.unique(key160[sel], return_inverse=True, return_counts=True)
        return (np.bincount(inv_, weights=v[sel]) / 16.0)[cnt == 16]
    rows, FA, checks = [], {}, []
    for floor in FLOORS:
        for dmin in DMINS:
            fa = {}
            for S in (A, B):
                Pm = P[S.obs]
                pi, pj = OV[S.obs]
                m = (Pm["score"] >= floor) & (Pm["diam"] >= dmin)
                pcs = pieces_for(Pm, pi, pj, m)
                fa[S.obs] = cell_fa(pcs, cells, ctree, cell_area)
                if S is B:
                    fa[S.obs + "_shift"] = cell_fa(pcs, cells, ctree, cell_area, shift=lag)
                if floor in (0.25, 0.40) and dmin == DMINS[0]:
                    rng = np.random.default_rng(0)
                    nz = np.flatnonzero(fa[S.obs] > 0)
                    pick = rng.choice(nz, size=min(60, len(nz)), replace=False) if len(nz) else []
                    gtree, err = shapely.STRtree(Pm["geoms"]), 0.0
                    for c in pick:
                        idx = gtree.query(cells[c])
                        ref, _ = le.areal_abundance(Pm["geoms"][idx[m[idx]]], cells[c], min_diameter_m=0.0)
                        err = max(err, abs(ref - fa[S.obs][c]))
                    checks.append(dict(obs=S.obs, floor=floor, n=len(pick), max_abs_err=err))
            FA[(floor, dmin)] = fa
            for sub_name, sel in (("all", np.ones(len(cells), bool)), ("clear_of_cap", clear)):
                for shifted in (False, True):
                    a = fa[A.obs]
                    b = fa[B.obs + "_shift"] if shifted else fa[B.obs]
                    for cm in (40, 160):
                        aa, bb = (a[sel], b[sel]) if cm == 40 else (agg160(a, sel), agg160(b, sel))
                        n = len(aa)
                        rows.append(dict(floor=floor, dmin=dmin, subset=sub_name, shifted=shifted,
                                         cell_m=cm, n_cells=n,
                                         ratio_lo_over_hi=(float(aa.sum() / bb.sum())
                                                           if n and bb.sum() > 0 else None),
                                         spearman=float(stats.spearmanr(aa, bb).statistic) if n > 3 else None,
                                         frac_zero_lo=float((aa == 0).mean()) if n else None,
                                         frac_zero_hi=float((bb == 0).mean()) if n else None))
        log("floor", floor, "done")
    counts = []
    for floor in FLOORS:
        d = {}
        for S in (A, B):
            Pm = P[S.obs]
            q, ci = ctree.query(shapely.points(Pm["cref"]), predicate="intersects")
            m = (Pm["score"][q] >= floor) & (Pm["diam"][q] >= DMINS[0])
            d[S.obs] = (int(m.sum()), int((m & clear[ci]).sum()))
        counts.append(dict(floor=floor, n_lo=d[A.obs], n_hi=d[B.obs],
                           count_ratio_all=d[A.obs][0] / max(1, d[B.obs][0]),
                           count_ratio_clear=d[A.obs][1] / max(1, d[B.obs][1])))
    res.update(rows=rows, counts=counts, areal_abundance_check=checks,
               n_cells40_clear_of_cap=int(clear.sum()))
    WORK.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(WORK / f"pair{pno}_cells.npz", key160=key160,
                        **{f"{o}__{fl}__{dm}": v for (fl, dm), d_ in FA.items() for o, v in d_.items()})
    (WORK / f"pair{pno}.json").write_text(json.dumps(res, indent=1))
    log("done")


def bootstrap(pno, n_boot=2000, seed=1):
    """Block bootstrap over 160 m blocks: the lo/hi ratio at 0.25 and 0.40, and r0.40 / r0.25."""
    a, b = PAIRS[pno]
    z = np.load(WORK / f"pair{pno}_cells.npz")
    _, inv = np.unique(z["key160"], return_inverse=True)
    nb = inv.max() + 1
    rng = np.random.default_rng(seed)
    S = {f: (np.bincount(inv, weights=z[f"{a}__{f}__{DMINS[0]}"], minlength=nb),
             np.bincount(inv, weights=z[f"{b}__{f}__{DMINS[0]}"], minlength=nb)) for f in FLOORS}
    reps = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, nb, nb), minlength=nb)
        r = {f: (w * S[f][0]).sum() / (w * S[f][1]).sum() for f in FLOORS}
        reps.append((r[0.25], r[0.40], r[0.40] / r[0.25]))
    reps = np.array(reps)
    pt = {f: S[f][0].sum() / S[f][1].sum() for f in FLOORS}
    q = lambda c: [float(np.percentile(reps[:, c], 2.5)), float(np.percentile(reps[:, c], 97.5))]
    kept = {o: float(z[f"{o}__0.4__{DMINS[0]}"].sum() / z[f"{o}__0.25__{DMINS[0]}"].sum()) for o in (a, b)}
    return {"n_blocks_160m": int(nb),
            "ratio_lo_over_hi_at_0.25": {"point": float(pt[0.25]), "ci95": q(0)},
            "ratio_lo_over_hi_at_0.40": {"point": float(pt[0.40]), "ci95": q(1)},
            "widening_r040_over_r025": {"point": float(pt[0.40] / pt[0.25]), "ci95": q(2)},
            "share_of_0.25_label_kept_at_0.40": kept}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", type=int, help=argparse.SUPPRESS)              # internal: one pair
    ap.add_argument("--pairs", type=int, nargs="+", default=sorted(PAIRS))
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.pair:
        return run_pair(args.pair)
    if not args.aggregate_only:
        for p in args.pairs:          # one process per pair: full-scene catalogs, no RAM ratchet
            subprocess.run([sys.executable, __file__, "--pair", str(p)], check=True)
    from YOLOv8BeyondEarth import presets
    summary = {"what": "same-ground repeat pairs: incidence vs hirise2ctx's label, by score floor",
               "code": presets.code_identity(),
               "label_error_sha256": cs.load_label_error()[1],
               "limits": ["contract-1 catalogs (effective conf floor 0.25, max_det 300 per 256 slice): "
                          "floors below 0.25 untested",
                          "no human truth: label consistency between the two images, not accuracy",
                          "pairs are misregistered by tens of metres: 'shifted' applies the "
                          "cross-correlation lag to the high-incidence member"],
               "pairs": {}}
    for p in args.pairs:
        r = json.loads((WORK / f"pair{p}.json").read_text())
        pick = lambda fl, sub, sh, cm: next((x for x in r["rows"] if x["floor"] == fl and
                                             x["dmin"] == DMINS[0] and x["subset"] == sub and
                                             x["shifted"] == sh and x["cell_m"] == cm), None)
        summary["pairs"][str(p)] = {
            "t2_name": r["t2_name"], "members": r["members"], "n_cells40_valid": r["n_cells40_valid"],
            "label_error_sha256_matches": r["label_error_sha256"] == summary["label_error_sha256"],
            "n_cells40_clear_of_cap": r["n_cells40_clear_of_cap"], "coreg": r["coreg"],
            "areal_abundance_check_max_abs_err": max(c["max_abs_err"] for c in r["areal_abundance_check"]),
            "ratio_lo_over_hi": {f"{fl:.2f}": {f"{sub}{'_shifted' if sh else ''}_{cm}m":
                                               (pick(fl, sub, sh, cm) or {}).get("ratio_lo_over_hi")
                                               for sub in ("all", "clear_of_cap") for sh in (False, True)
                                               for cm in (40, 160)} for fl in FLOORS},
            "spearman_40m_shifted": {f"{fl:.2f}": (pick(fl, "all", True, 40) or {}).get("spearman")
                                     for fl in FLOORS},
            "spearman_40m_unshifted": {f"{fl:.2f}": (pick(fl, "all", False, 40) or {}).get("spearman")
                                       for fl in FLOORS},
            "counts": r["counts"], "bootstrap": bootstrap(p)}
    out = Path(args.out or REPO / "data" / "review" / "incidence_pairs_label.json")
    out.write_text(json.dumps(summary, indent=1))
    for p, s in summary["pairs"].items():
        m = list(s["members"].items())
        b = s["bootstrap"]
        print(f"pair {p} ({s['t2_name'] or 'control'}): inc {m[0][1]['incidence']:.1f} vs "
              f"{m[1][1]['incidence']:.1f}; lo/hi at 0.25 {b['ratio_lo_over_hi_at_0.25']['point']:.2f} "
              f"{[round(x, 2) for x in b['ratio_lo_over_hi_at_0.25']['ci95']]}, at 0.40 "
              f"{b['ratio_lo_over_hi_at_0.40']['point']:.2f}; widening x"
              f"{b['widening_r040_over_r025']['point']:.2f} "
              f"{[round(x, 2) for x in b['widening_r040_over_r025']['ci95']]}; lag "
              f"{s['coreg']['lag_norm_m']:.0f} m; cells {s['n_cells40_valid']}")


if __name__ == "__main__":
    main()
