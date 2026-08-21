"""The 2x2 arm ratification: does the 512 arm earn its place, does the SR arm, and do they interact.

Both arms' inclusion in ``GOAL3_PRODUCTION["scales"]`` is unratified. This scores the **four**
fusions -- ``{256}``, ``{512,256}``, ``{256,SR}``, ``{512,256,SR}`` -- against a human reference per
size band, so one design yields both marginal contributions *and* their interaction. Two tests would
not: the value of the SR arm is allowed to depend on whether the 512 arm is present, and a pair of
one-arm-at-a-time ablations cannot see that.

Three design rules are welded in rather than left to the caller.

**The reference is a parameter, and its digest travels with the result.** New human labels are weeks
away and must re-run this in one command. More importantly a recall number is meaningless without
saying what it was recall *of*, and the way that goes wrong is not malice but a second reference
appearing on disk. :func:`ratification_provenance` hashes every reference file, every arm layer, the
fusion preset and this module's own matching rule into the output.

**Every cell goes through the same fusion call, including the one-arm cell.** ``{256}`` is fused too
-- with ``cross_arm_only=True`` every pair inside a single arm is same-arm, so fusion suppresses
nothing and the cell reduces to that arm's catalog plus provenance columns. That is deliberate: a
one-arm cell that *bypassed* fusion would differ from the others by two things (the missing arm and
the missing code path) and the comparison would not be an ablation. ``test_arm_ratification.py``
pins the no-op.

**Results are stratified by the reference tile's dataset split, never only pooled.** The ROM is not
an independently mapped region: it is exactly the set of tiles that became ``boulder2024``, so most
of it is *training* ground (measured 2026-08-21: Moon 338 ROM tiles = 270 train / 51 validation /
17 test; Mars 20 = 16 / 3 / 1). Absolute recall on the pooled ROM is therefore optimistic, and the
direction of the bias on an *ablation* is not benign either -- a memorised boulder is found by
whichever arm sees it, so leakage compresses the marginal value of adding an arm and biases the
answer toward "it contributes nothing". Reporting per split costs nothing here and is the only way
that shows.

The matching rule is the established one -- ">50% overlap of the smaller polygon", the admissibility
gate behind every published recall number in ``goal3_multiscale_fusion_plan.md`` -- but it is
resolved **both** ways and both are reported. The published tables took admissible pairs greedily in
descending-overlap order; ``matching.py`` exists because greedy pairing can consume a prediction
another human needed and report a recall below the achievable one (review item G2-05). Since the
optimal resolution can only ever match at least as many, the pair brackets the answer instead of
asserting one. Never average them: they are two definitions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import presets
from .benchmark import DEFAULT_SIZE_BANDS, _band_label
from .fusion import equivalent_diameter_px, fuse_predictions
from .matching import min_cost_matching

#: The four cells, as ``name -> ((slice_size, sr_factor), ...)``. Read as a 2x2 in
#: (512 in/out) x (SR in/out): each arm's marginal contribution appears twice and the difference
#: between its two appearances is the interaction.
ARM_2X2 = {
    "256": ((256, 1),),
    "512_256": ((512, 1), (256, 1)),
    "256_sr": ((256, 1), (256, 2)),
    "512_256_sr": ((512, 1), (256, 1), (256, 2)),
}

#: Sidecars that make up a shapefile. All of them are hashed: the geometry lives in ``.shp``, the
#: attributes in ``.dbf`` and the CRS label in ``.prj``, so a digest over ``.shp`` alone would call
#: two references identical after an attribute edit.
SHAPEFILE_PARTS = (".shp", ".dbf", ".shx", ".prj", ".cpg")

#: How this module resolves a set of admissible pairs into a one-to-one matching. See the module
#: docstring: report both, never average.
RESOLUTIONS = ("optimal", "greedy")


def arm_key(slice_size, sr_factor=1):
    """``(256, 1) -> 256``; ``(256, 2) -> (256, 2)`` -- the scale-key spelling ``fusion`` uses."""
    slice_size, sr_factor = int(slice_size), int(sr_factor)
    return slice_size if sr_factor == 1 else (slice_size, sr_factor)


def arm_tag(slice_size, sr_factor=1):
    """``(256, 2) -> "256sr2"`` -- the spelling that appears in a layer filename."""
    slice_size, sr_factor = int(slice_size), int(sr_factor)
    return f"{slice_size}" if sr_factor == 1 else f"{slice_size}sr{sr_factor}"


# --- the reference -------------------------------------------------------------------------------

def load_aligned(path, ref_crs, gpd=None):
    """Read a vector file and **override** its CRS label with the raster's.

    ``set_crs(allow_override=True)``, never ``to_crs``. The Prieur shapefiles carry a
    ``D_MOON``/``D_MARS`` datum whose definition does not round-trip: reprojecting applies a bogus
    datum shift that moves the geometry out of the raster's coordinate range and clips to **zero
    rows**. The files are already in the raster's grid; only the label is untrustworthy.
    """
    if gpd is None:
        import geopandas as gpd
    return gpd.read_file(path).set_crs(ref_crs, allow_override=True)


def rom_tiles(rom_path, ref_crs, split_map=None):
    """The ROM as one row per mapped tile, carrying ``tile_id``, ``file_name`` and ``split``.

    Exploded, because a ROM tile is stored as a part of a multipolygon and the per-tile readouts
    below (which tile is a human in, which split is that tile) need one geometry each.

    ``split_map`` maps a tile's image stem to its ``boulder2024`` split; see :func:`dataset_splits`.
    Tiles absent from it are labelled ``"absent"`` rather than dropped or guessed -- a reference tile
    that is not in the training dataset is the *useful* case, and silently calling it training would
    invert the caveat this column exists to carry.
    """
    tiles = load_aligned(rom_path, ref_crs).explode(index_parts=False).reset_index(drop=True)
    if split_map is None:
        tiles["split"] = "unknown"
    else:
        stems = tiles["file_name"].astype(str).str.replace(r"\.[^.]*$", "", regex=True)
        tiles["split"] = [split_map.get(s, "absent") for s in stems]
    return tiles


def dataset_splits(dataset_root, subdirs=("train", "validation", "test")):
    """``{image stem: split}`` for a YOLO dataset laid out as ``<root>/<split>/images/*``.

    This is what makes the leakage caveat in the module docstring a measured column rather than a
    remark. It reads filenames only -- it never opens an image or a label.
    """
    from pathlib import Path

    out = {}
    for split in subdirs:
        images = Path(dataset_root) / split / "images"
        if not images.is_dir():
            continue
        for path in images.iterdir():
            if path.is_file():
                out[path.stem] = split
    return out


def clip_to_tiles(gdf, tiles, tile_cols=("tile_id", "split")):
    """Keep the rows whose **centroid** falls in a ROM tile, tagged with that tile's columns.

    Centroid-in-tile, not intersects: a tile boundary cuts boulders, and an ``intersects`` rule
    would count one straddling boulder in two tiles -- inflating the recall denominator and letting
    a single prediction be credited twice. It is the rule the ROM ladder already used.
    """
    import geopandas as gpd

    if len(gdf) == 0:
        empty = gdf.copy()
        for col in tile_cols:
            empty[col] = pd.Series(dtype=object)
        return empty
    centroids = gpd.GeoDataFrame(geometry=gdf.geometry.centroid, crs=gdf.crs)
    hit = gpd.sjoin(centroids, tiles[[*tile_cols, "geometry"]], predicate="within", how="inner")
    # One centroid can land in two tiles only if the ROM parts overlap, which would be a defect in
    # the reference rather than something to average over -- keep the first and say how many.
    hit = hit[~hit.index.duplicated(keep="first")]
    out = gdf.loc[hit.index].copy()
    for col in tile_cols:
        out[col] = hit[col].to_numpy()
    return out.reset_index(drop=True)


# --- matching ------------------------------------------------------------------------------------

def overlap_candidates(human, pred, hybrid=False):
    """Admissible (human, prediction) pairs and their overlap fraction.

    Admissible = the intersection covers more than half of the **smaller** polygon. Half-of-smaller
    rather than IoU because the reference and the prediction disagree systematically about extent
    (measured area factors 1.06-1.27x SAM/human, 1.58-2.13x YOLO/human), and an IoU gate would score
    that disagreement as a detection failure.

    ``hybrid=True`` additionally accepts centroid-in-polygon either way -- the localisation-tolerant
    reading for the <6 px regime, where a ~1 px mask offset defeats the pure overlap rule (probed
    2026-07-10: the "lost" humans all had an intersecting prediction at overlap 0.37-0.46). It is a
    strictly wider gate, so it is reported alongside the strict one, never instead of it.

    Returns ``(human_idx, pred_idx, frac)`` over admissible pairs only.
    """
    import shapely

    empty = (np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, float))
    if len(human) == 0 or len(pred) == 0:
        return empty
    hg = human.geometry.values
    pg = pred.geometry.values
    hi, pi = shapely.STRtree(pg).query(hg, predicate="intersects")
    if len(hi) == 0:
        return empty
    inter = shapely.area(shapely.intersection(hg[hi], pg[pi]))
    smaller = np.minimum(shapely.area(hg[hi]), shapely.area(pg[pi]))
    frac = np.where(smaller > 0, inter / smaller, 0.0)
    ok = frac > 0.5
    if hybrid:
        ok = (ok | shapely.contains(hg[hi], shapely.centroid(pg[pi]))
                 | shapely.contains(pg[pi], shapely.centroid(hg[hi])))
    return hi[ok].astype(np.int64), pi[ok].astype(np.int64), frac[ok]


def match_overlap(human, pred, hybrid=False, resolve="optimal"):
    """One-to-one match under :func:`overlap_candidates`. Returns ``(human_mask, pred_mask)``.

    ``resolve="greedy"`` reproduces the published tables exactly: admissible pairs consumed in
    descending-overlap order. ``resolve="optimal"`` maximises the number of pairs first and prefers
    higher overlap only to break ties, so it can never match fewer. The gap between them is a
    property of the catalog, not a free parameter -- report both (module docstring).
    """
    h_mask = np.zeros(len(human), dtype=bool)
    p_mask = np.zeros(len(pred), dtype=bool)
    hi, pi, frac = overlap_candidates(human, pred, hybrid=hybrid)
    if len(hi) == 0:
        return h_mask, p_mask
    if resolve == "greedy":
        for k in np.argsort(-frac):
            if not h_mask[hi[k]] and not p_mask[pi[k]]:
                h_mask[hi[k]] = True
                p_mask[pi[k]] = True
        return h_mask, p_mask
    if resolve != "optimal":
        raise ValueError(f"resolve must be one of {RESOLUTIONS}, got {resolve!r}")
    # cost = 1 - frac: min_cost_matching is cardinality-first, so this only orders the ties.
    rows, cols = min_cost_matching(hi, pi, 1.0 - frac, len(human), len(pred))
    h_mask[rows] = True
    p_mask[cols] = True
    return h_mask, p_mask


# --- scoring -------------------------------------------------------------------------------------

def _f1(precision, recall):
    """Harmonic mean, NaN-safe. NaN when either side is undefined or both are zero -- an F1 of 0
    would read as "measured and bad" where the truth is "nothing to measure"."""
    if not np.isfinite(precision) or not np.isfinite(recall) or (precision + recall) == 0:
        return np.nan
    return 2.0 * precision * recall / (precision + recall)


def score_bands(human, pred, native_res, bands=DEFAULT_SIZE_BANDS, hybrid=False,
                resolve="optimal", split_col="split"):
    """Per-band recall / precision / F1 in one pass, optionally also split by ``split_col``.

    **The two denominators are different populations by construction.** Recall is over humans in the
    band; precision is over *predictions* in the band, because an unmatched false positive has no
    human whose size could band it -- its own size is the only definition available. F1 is a summary
    of the pair, not a quantity over a single population. Matching itself is done **once over the
    whole region** and only then split by band and tile: matching inside a band would let a
    prediction match a human it is not the best partner for, and matching inside a split would let
    the same prediction be credited in two.
    """
    h_mask, p_mask = match_overlap(human, pred, hybrid=hybrid, resolve=resolve)
    h_band = _band_of(equivalent_diameter_px(human, native_res), bands)
    p_band = _band_of(equivalent_diameter_px(pred, native_res), bands)
    groups = [("all", np.ones(len(human), bool), np.ones(len(pred), bool))]
    if split_col in getattr(human, "columns", ()) and split_col in getattr(pred, "columns", ()):
        hs, ps = human[split_col].to_numpy(), pred[split_col].to_numpy()
        for value in sorted(set(hs.tolist()) | set(ps.tolist())):
            groups.append((str(value), hs == value, ps == value))
    rows = []
    for label, h_in, p_in in groups:
        for lo, hi in (("all", None),) + tuple(bands):
            if lo == "all":
                band, h_sel, p_sel = "all", h_in, p_in
            else:
                band = _band_label(lo, hi)
                h_sel = h_in & (h_band == band)
                p_sel = p_in & (p_band == band)
            recall = float(h_mask[h_sel].mean()) if h_sel.sum() else np.nan
            precision = float(p_mask[p_sel].mean()) if p_sel.sum() else np.nan
            rows.append(dict(split=label, band=band, n_human=int(h_sel.sum()),
                             n_pred=int(p_sel.sum()), n_matched=int(h_mask[h_sel].sum()),
                             recall=recall, precision=precision, f1=_f1(precision, recall)))
    return pd.DataFrame(rows)


def _band_of(dpx, bands=DEFAULT_SIZE_BANDS):
    """Band label per object; anything below the first edge gets ``"<lo"`` and is never pooled in."""
    dpx = np.asarray(dpx, dtype=float)
    out = np.full(len(dpx), f"<{bands[0][0]}", dtype=object)
    for lo, hi in bands:
        out[(dpx >= lo) & (dpx < hi)] = _band_label(lo, hi)
    return out


def fuse_combo(per_arm, arms, native_res, preset=None):
    """Fuse one 2x2 cell with production's rule. ``per_arm`` maps :func:`arm_key` -> GeoDataFrame.

    A one-arm cell goes through this too and comes back unsuppressed, because ``cross_arm_only``
    makes every within-arm pair ineligible. That keeps the reference cell on the same code path as
    the others (module docstring).
    """
    cfg = presets.GOAL3_PRODUCTION if preset is None else preset
    missing = [a for a in arms if arm_key(*a) not in per_arm]
    if missing:
        raise KeyError(f"missing arm layer(s) for this cell: {missing}; have {sorted(map(str, per_arm))}")
    gdfs = {arm_key(*a): per_arm[arm_key(*a)] for a in arms}
    fused, audit = fuse_predictions(
        gdfs, rule=cfg["rule"], bands=cfg["bands"], engulf_thr=cfg["engulf_thr"],
        metric=cfg["metric"], cross_arm_only=cfg["cross_arm_only"], native_res=native_res)
    return fused, audit


def ratify_arms(per_arm, human, native_res, combos=None, bands=DEFAULT_SIZE_BANDS,
                resolutions=RESOLUTIONS, hybrids=(False, True), preset=None, body=None,
                on_cell=None, post_fuse=None):
    """Score every 2x2 cell against ``human``. Returns one long DataFrame.

    Columns: ``body, combo, has_512, has_sr, resolve, hybrid, split, band, n_human, n_pred,
    n_matched, recall, precision, f1, n_fused, n_scored, n_suppressed``. ``has_512``/``has_sr`` are
    spelled out so the 2x2 can be read as a factorial design without re-parsing ``combo``.

    ``post_fuse`` is applied to each fused catalog before scoring -- the clip to the reference's
    footprint. It runs **after** fusion because that is production's order: an arm's detection just
    outside the reference region can still suppress or engulf one just inside, and removing it
    first would quietly change the catalog being scored.

    Per body, never pooled across bodies: the Moon reference outnumbers the Mars one ~12:1 at the
    large end, so a pooled row would be the Moon's answer wearing both names, and per-body
    differences are already established (orientation accuracy 12.1 deg Mars / 8.7 deg Moon).
    """
    combos = ARM_2X2 if combos is None else combos
    out = []
    for name, arms in combos.items():
        raw, audit = fuse_combo(per_arm, arms, native_res, preset=preset)
        fused = raw if post_fuse is None else post_fuse(raw)
        if on_cell is not None:
            on_cell(name, raw, fused, audit)
        for resolve in resolutions:
            for hybrid in hybrids:
                part = score_bands(human, fused, native_res, bands=bands, hybrid=hybrid,
                                   resolve=resolve)
                part.insert(0, "hybrid", bool(hybrid))
                part.insert(0, "resolve", resolve)
                part.insert(0, "has_sr", any(sr > 1 for _, sr in arms))
                part.insert(0, "has_512", any(ss == 512 for ss, _ in arms))
                part.insert(0, "combo", name)
                part.insert(0, "body", body)
                part["n_fused"] = int(len(raw))
                part["n_scored"] = int(len(fused))
                part["n_suppressed"] = int(len(audit))
                out.append(part)
    return pd.concat(out, ignore_index=True)


def marginal_table(scores, band="14-inf", split="all", resolve="optimal", hybrid=False,
                   metric="recall"):
    """The 2x2 read out as two marginal contributions and their interaction, at one cell definition.

    ``d512_without_sr`` = ``{512,256}`` - ``{256}``; ``d512_with_sr`` = ``{512,256,SR}`` -
    ``{256,SR}``; likewise for SR; ``interaction`` = the difference between an arm's two marginals
    (identical whichever arm you compute it from -- that identity is what makes the 2x2 worth
    running instead of two ablations, and ``test_arm_ratification.py`` pins it).

    Every number carries the ``band``/``split``/``resolve``/``hybrid`` it was read at, because these
    differences are small and all four choices move them.
    """
    sel = scores[(scores["band"] == band) & (scores["split"] == split)
                 & (scores["resolve"] == resolve) & (scores["hybrid"] == bool(hybrid))]
    value = {row["combo"]: row[metric] for _, row in sel.iterrows()}
    need = ("256", "512_256", "256_sr", "512_256_sr")
    if any(k not in value for k in need):
        raise KeyError(f"need all four cells at band={band} split={split}; have {sorted(value)}")
    d512_without_sr = value["512_256"] - value["256"]
    d512_with_sr = value["512_256_sr"] - value["256_sr"]
    dsr_without_512 = value["256_sr"] - value["256"]
    dsr_with_512 = value["512_256_sr"] - value["512_256"]
    return dict(metric=metric, band=band, split=split, resolve=resolve, hybrid=bool(hybrid),
                n_human=int(sel["n_human"].iloc[0]), base=value["256"],
                d512_without_sr=d512_without_sr, d512_with_sr=d512_with_sr,
                dsr_without_512=dsr_without_512, dsr_with_512=dsr_with_512,
                interaction=d512_with_sr - d512_without_sr)


# --- provenance ----------------------------------------------------------------------------------

def shapefile_identity(path):
    """:func:`presets.file_identity` for every sidecar of a shapefile, keyed by extension.

    A reference is a *set* of files. Hashing only ``.shp`` would call two references identical after
    an attribute edit, and the human tables carry attributes this harness reads.
    """
    from pathlib import Path

    path = Path(path)
    if path.suffix.lower() != ".shp":
        return {path.suffix or "<none>": presets.file_identity(path)}
    return {ext: presets.file_identity(path.with_suffix(ext)) for ext in SHAPEFILE_PARTS
            if path.with_suffix(ext).exists()}


def ratification_provenance(reference_paths, arm_paths, native_res, combos=None,
                            bands=DEFAULT_SIZE_BANDS, preset=None, extra=None):
    """Everything needed to say what a ratification table is a table *of*.

    Deliberately includes the **matching rule** and the band edges, not just the file digests: the
    inputs alone do not pin a recall number, and the two published cases where a "validated" result
    turned out to describe a different configuration than the shipped one (review items X-01, G3-01)
    were both invisible in the inputs.
    """
    cfg = presets.GOAL3_PRODUCTION if preset is None else preset
    rule = {"admissibility": "intersection > 0.5 * area(smaller polygon)",
            "resolutions": list(RESOLUTIONS), "hybrid_gate": "centroid-in-polygon, either side",
            "assignment": "centroid within ROM tile",
            "size_definition": "equivalent-area diameter in native px",
            "bands": [[lo, None if not np.isfinite(hi) else hi] for lo, hi in bands]}
    prov = {
        "harness": "arm_ratification",
        "reference": {name: shapefile_identity(p) for name, p in dict(reference_paths).items()},
        "arms": {str(name): presets.file_identity(p) for name, p in dict(arm_paths).items()},
        "combos": {k: [list(a) for a in v] for k, v in (combos or ARM_2X2).items()},
        "fusion": presets._jsonable({k: v for k, v in cfg.items()}),
        "preset_version": presets.PRESET_VERSION,
        "native_res": float(native_res),
        "matching_rule": rule,
        "code": presets.code_identity(),
    }
    prov["config_hash"] = presets.config_hash(
        {k: prov[k] for k in ("combos", "fusion", "matching_rule", "native_res")})
    if extra:
        prov.update(extra)
    return prov
