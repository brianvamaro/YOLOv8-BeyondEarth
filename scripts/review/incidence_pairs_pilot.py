"""Same-ground incidence comparisons on the Sherlock pilot's catalogs (docs/sherlock_pilot_plan.md §8.F).

`incidence_pairs_label.py` built this method on 2026-10-16, for the five Test-2 pairs' contract-1
catalogs. Here the same method, through that module's own functions, runs on the pilot's contract-3
deliveries (release `pilot-1`):
- **Inputs.** Conf 0.10, escalation, label-frame rasters, pulled into the delivery root (`HANDOFF.md` §1).
- **Comparisons.** 11, read from `sherlock/manifests/pilot_pairs.csv`. Every row with a `Reference` is
  scored against it: the five pairs P1–P5, and the gap sets G1, G2, G2+, U1 and U2. G2 and G2+ are
  each scored against G2's 51.9° reference, and U2's two gap images each against its 46.6° reference.
- **Frame and cells.** Each member is mapped into the reference's PDS-label frame through lon/lat, with
  40 m cells over the ground valid in both. The label is notebook 35's per-cell area fraction.
- **Floors.** Score floors 0.10, 0.15, 0.20, 0.25, 0.30, 0.40 and 0.50; size floors 1.41 and 2.0 m.
- **Outputs.**
  - the co-registration lag, with ratios unshifted and shifted;
  - Spearman at 40 m, the count ratio, and each member's kept share relative to 0.10 and to 0.25;
  - a block bootstrap over 160 m blocks, for the ratio at every floor and for its change from 0.25.

**Two differences from the contract-1 run.**
- Floors below 0.25 are tested now, since these catalogs hold everything down to 0.10.
- "Near the cap" means a slice within 10 % of its layer's own recorded `max_det`, counted by centroid
  after merging, as before. Escalation leaves no delivered slice there, so this is a check, not a
  correction.

There is no human truth here: this measures how consistently two images label the same ground, not
which one is right. Nothing here changes what a label is.

    conda run -n bouldernet python -I scripts/review/incidence_pairs_pilot.py --out data/review/incidence_pairs_pilot.json
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import shapely
from scipy import ndimage, stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "review"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import config_study as cs                                                   # noqa: E402
import incidence_pairs_label as ipl                                         # noqa: E402

DELIVERY = Path(r"C:/Users/brian/Documents/PhD/BoulderNet/data/deliveries/hirise2ctx_v3/pilot-1")
MANIFEST = REPO / "sherlock" / "manifests" / "pilot_pairs.csv"
WORK = REPO / "data" / "incidence_pairs_pilot"      # per-comparison intermediates (gitignored)
FLOORS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
DMINS = ipl.DMINS
NEAR_CAP = 0.9                                       # escalation's own "within 10 % of the cap"


def comparisons(manifest=MANIFEST):
    """[(key, set, reference, other)] for every manifest row that names a Reference, in file order."""
    with open(manifest, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    return [(f"{r['Set']}:{r['ObsId']}", r["Set"], r["Reference"], r["ObsId"]) for r in rows if r.get("Reference")]


class PilotScene(ipl.Scene):
    """`incidence_pairs_label.Scene` on a delivered scene: the prepared raster and label from `scenes/`,
    the layer `handoff.json` lists from `out/`. `Scene.__init__` reads `<ipl.D>/<obs>/<obs>_RED.{tif,LBL}`,
    so `ipl.D` points at the delivery's `scenes/` while it runs."""

    def __init__(self, obs, delivery):
        saved, ipl.D = ipl.D, Path(delivery) / "scenes"
        try:
            super().__init__(obs)
        finally:
            ipl.D = saved
        h = json.loads((Path(delivery) / "out" / obs / "handoff.json").read_text())
        (lay,) = h["layers"]
        self.gpkg = Path(delivery) / "out" / obs / lay["layer"]
        self.max_det = int(lay["max_det"])


def run_comparison(key, delivery):
    le, sha = cs.load_label_error()
    _, set_name, ref_obs, other_obs = next(c for c in comparisons() if c[0] == key)
    A, B = PilotScene(ref_obs, delivery), PilotScene(other_obs, delivery)
    px2ref, ref2px = ipl.make_maps(A, B)
    res = dict(key=key, set=set_name, members={}, label_error_sha256=sha)
    for S in (A, B):
        res["members"][S.obs] = dict(incidence=S.inc, emission=S.emi, scale=S.scale, max_det=S.max_det,
                                     tif_minus_label_origin_px=list(map(float, S.dpx)))
    ipl.log(key, A.obs, A.inc, B.obs, B.inc)
    VM = {S.obs: ipl.valid_mask(S) for S in (A, B)}

    def ref_hull(S):
        v, f = VM[S.obs]
        rr, cc = np.nonzero(v[::4, ::4])
        pts = np.column_stack([cc * 4 * f, rr * 4 * f]).astype(float)
        return shapely.convex_hull(shapely.multipoints(px2ref(S, pts)))
    inter = shapely.intersection(ref_hull(A), ref_hull(B))
    if inter.is_empty:
        raise SystemExit(f"{key}: no overlap")
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
    res["shared_km2"] = float(ok.sum() * 40.0 * 40.0 / 1e6)     # each cell is 40 m x 40 m on the ground
    P, NEAR = {}, {}
    for S in (A, B):
        P[S.obs] = Pm = ipl.load_preds(le, S, px2ref)
        xs, ys, cnt = ipl.slice_counts(S, Pm["cpx"][:, 0], Pm["cpx"][:, 1])
        cap = NEAR_CAP * S.max_det
        boxes = [shapely.polygons(px2ref(S, np.array([[xs[b_][0], ys[a_][0]], [xs[b_][1], ys[a_][0]],
                                                     [xs[b_][1], ys[a_][1]], [xs[b_][0], ys[a_][1]]], float)))
                 for a_, b_ in zip(*np.nonzero(cnt >= cap))]
        near = np.zeros(len(cells), bool)
        if boxes:
            _, ci = ctree.query(np.asarray(boxes, dtype=object), predicate="intersects")
            near[np.unique(ci)] = True
        NEAR[S.obs] = near
        res["members"][S.obs].update(n_det=len(Pm["geoms"]), slices_near_cap=int((cnt >= cap).sum()),
                                     slice_count_max=int(cnt.max()), cells40_near_cap=int(near.sum()),
                                     n_drop=Pm["n_drop"], n_invalid_repaired=Pm["n_invalid"])
    clear = ~(NEAR[A.obs] | NEAR[B.obs])
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
    OV = {S.obs: ipl.overlap_pairs(P[S.obs]["geoms"]) for S in (A, B)}
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
                pcs = ipl.pieces_for(Pm, pi, pj, m)
                fa[S.obs] = ipl.cell_fa(pcs, cells, ctree, cell_area)
                if S is B:
                    fa[S.obs + "_shift"] = ipl.cell_fa(pcs, cells, ctree, cell_area, shift=lag)
                if floor in (0.10, 0.25, 0.40) and dmin == DMINS[0]:
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
                        rows.append(dict(floor=floor, dmin=dmin, subset=sub_name, shifted=shifted, cell_m=cm,
                                         n_cells=n, ratio_lo_over_hi=(float(aa.sum() / bb.sum())
                                                                      if n and bb.sum() > 0 else None),
                                         spearman=float(stats.spearmanr(aa, bb).statistic) if n > 3 else None,
                                         frac_zero_lo=float((aa == 0).mean()) if n else None,
                                         frac_zero_hi=float((bb == 0).mean()) if n else None))
        ipl.log(key, "floor", floor, "done")
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
    res.update(rows=rows, counts=counts, areal_abundance_check=checks, n_cells40_clear_of_cap=int(clear.sum()),
               reference=A.obs, other=B.obs)
    WORK.mkdir(parents=True, exist_ok=True)
    tag = key.replace(":", "__").replace("+", "plus")
    np.savez_compressed(WORK / f"{tag}_cells.npz", key160=key160,
                        **{f"{o}__{fl}__{dm}": v for (fl, dm), d_ in FA.items() for o, v in d_.items()})
    (WORK / f"{tag}.json").write_text(json.dumps(res, indent=1))
    ipl.log(key, "done")


def bootstrap(key, ref, other, n_boot=2000, seed=1):
    """Block bootstrap over 160 m blocks: the ref/other ratio at every floor, its change from 0.25, and the
    kept shares relative to 0.10 and to 0.25 (1.41 m size floor)."""
    tag = key.replace(":", "__").replace("+", "plus")
    z = np.load(WORK / f"{tag}_cells.npz")
    _, inv = np.unique(z["key160"], return_inverse=True)
    nb = inv.max() + 1
    rng = np.random.default_rng(seed)
    S = {f: (np.bincount(inv, weights=z[f"{ref}__{f}__{DMINS[0]}"], minlength=nb),
             np.bincount(inv, weights=z[f"{other}__{f}__{DMINS[0]}"], minlength=nb)) for f in FLOORS}
    reps = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, nb, nb), minlength=nb)
        reps.append([(w * S[f][0]).sum() / max((w * S[f][1]).sum(), 1e-300) for f in FLOORS])
    reps = np.array(reps)
    pt = np.array([S[f][0].sum() / max(S[f][1].sum(), 1e-300) for f in FLOORS])
    i25 = FLOORS.index(0.25)
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    tot = {o: {f: float(z[f"{o}__{f}__{DMINS[0]}"].sum()) for f in FLOORS} for o in (ref, other)}
    return {"n_blocks_160m": int(nb),
            "ratio_ref_over_other": {f"{f:.2f}": {"point": float(pt[k]), "ci95": ci(reps[:, k])}
                                     for k, f in enumerate(FLOORS)},
            "change_from_0.25": {f"{f:.2f}": {"point": float(pt[k] / pt[i25]), "ci95": ci(reps[:, k] / reps[:, i25])}
                                 for k, f in enumerate(FLOORS)},
            "kept_share_of_0.10_label": {o: {f"{f:.2f}": (tot[o][f] / tot[o][0.10] if tot[o][0.10] else None)
                                             for f in FLOORS} for o in (ref, other)},
            "kept_share_of_0.25_label": {o: {f"{f:.2f}": (tot[o][f] / tot[o][0.25] if tot[o][0.25] else None)
                                             for f in FLOORS} for o in (ref, other)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison", help=argparse.SUPPRESS)             # internal: one comparison
    ap.add_argument("--only", nargs="+", help="comparison keys to run (default all)")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--delivery", default=str(DELIVERY))
    ap.add_argument("--out", default=str(REPO / "data" / "review" / "incidence_pairs_pilot.json"))
    args = ap.parse_args()
    if args.comparison:
        return run_comparison(args.comparison, Path(args.delivery))
    comps = [c for c in comparisons() if not args.only or c[0] in args.only]
    if not args.aggregate_only:
        for key, *_ in comps:            # one process per comparison: full-scene catalogs, no RAM ratchet
            subprocess.run([sys.executable, __file__, "--comparison", key, "--delivery", args.delivery], check=True)
    from YOLOv8BeyondEarth import presets
    summary = {"what": "same-ground incidence comparisons on the Sherlock pilot's catalogs (release pilot-1)",
               "code": presets.code_identity(), "delivery": args.delivery,
               "label_error_sha256": cs.load_label_error()[1],
               "limits": ["no human truth: label consistency between two images of the same ground, not accuracy",
                          "images are misregistered by tens of metres: 'shifted' applies the cross-correlation lag",
                          "one region per set; seasons and atmospheres differ (no opacity correction)",
                          "P3's reference ESP_069088_2270 bins RED0-3 and RED7-8 2x; read P3 at the 2.0 m size floor too"],
               "comparisons": {}}
    for key, set_name, ref, other in comps:
        tag = key.replace(":", "__").replace("+", "plus")
        r = json.loads((WORK / f"{tag}.json").read_text())
        pick = lambda fl, dm, sub, sh, cm: next((x for x in r["rows"] if x["floor"] == fl and x["dmin"] == dm and
                                                 x["subset"] == sub and x["shifted"] == sh and x["cell_m"] == cm), None)
        summary["comparisons"][key] = {
            "set": set_name, "reference": ref, "other": other, "members": r["members"],
            "n_cells40_valid": r["n_cells40_valid"], "shared_km2": r["shared_km2"],
            "n_cells40_clear_of_cap": r["n_cells40_clear_of_cap"], "coreg": r["coreg"],
            "label_error_sha256_matches": r["label_error_sha256"] == summary["label_error_sha256"],
            "areal_abundance_check_max_abs_err": max(c["max_abs_err"] for c in r["areal_abundance_check"]),
            "ratio_ref_over_other": {f"{fl:.2f}": {f"{'d2.0_' if dm == 2.0 else ''}{sub}{'_shifted' if sh else ''}_{cm}m":
                                                   (pick(fl, dm, sub, sh, cm) or {}).get("ratio_lo_over_hi")
                                                   for dm in DMINS for sub in ("all", "clear_of_cap")
                                                   for sh in (False, True) for cm in (40, 160)} for fl in FLOORS},
            "spearman_40m_shifted": {f"{fl:.2f}": (pick(fl, DMINS[0], "all", True, 40) or {}).get("spearman")
                                     for fl in FLOORS},
            "counts": r["counts"], "bootstrap": bootstrap(key, ref, other)}
    out = Path(args.out)
    out.write_text(json.dumps(summary, indent=1))
    print(f"{'comparison':28s} {'inc ref/other':>14s} {'km2':>6s} {'r@0.10':>8s} {'r@0.25':>8s} {'r@0.40':>8s} "
          f"{'r@0.25 shifted':>14s} {'lag m':>6s} {'peak':>5s}")
    for key, s in summary["comparisons"].items():
        m = s["members"]
        b = s["bootstrap"]["ratio_ref_over_other"]
        print(f"{key:28s} {m[s['reference']]['incidence']:6.1f}/{m[s['other']]['incidence']:5.1f} "
              f"{s['shared_km2']:6.1f} {b['0.10']['point']:8.2f} {b['0.25']['point']:8.2f} {b['0.40']['point']:8.2f} "
              f"{s['ratio_ref_over_other']['0.25']['all_shifted_40m'] or float('nan'):14.2f} "
              f"{s['coreg']['lag_norm_m']:6.0f} {s['coreg']['xcorr_peak']:5.2f}")


if __name__ == "__main__":
    main()
