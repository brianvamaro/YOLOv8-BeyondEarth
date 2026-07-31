"""Cross-scale fusion of sliced-inference detections (Goal 3, phase 1).

Fuses per-scale ``get_sliced_prediction`` outputs (each already intra-scale NMS'd) into one
catalog, keeping each boulder from its best scale. Structurally this is generalized NMS: pool,
cluster overlapping detections, keep one per cluster — with the keep criterion swapped from
"highest score" (scale-agnostic, wrong for best-scale selection) to a rule. Design + gates:
``docs/goal3_multiscale_fusion_plan.md``; run-1 evidence: ``docs/solutions/D_multiscale_fusion.md``.

Cluster metric is **DIoU** (IoU minus normalized center-distance penalty; Zheng et al. 2020,
AAAI, https://doi.org/10.1609/aaai.v34i07.6999) — one mechanism covering the tiny-box mode
where a ~1 px localization shift collapses plain IoU. A DIoU threshold is NOT an IoU threshold
(range (-1, 1], systematically below IoU). ``DEFAULT_CLUSTER_THR = 0.30`` was calibrated on the
three-arm cross-scale matched pairs vs distinct-neighbour negatives
(``data/solutionDE/calibrate_fusion_threshold.py``, 2026-07-10): true pairs have DIoU p01
≈ 0.42–0.52 in every size band incl. 4–6 px, distinct neighbours p99 ≈ 0.00–0.03 in the small
bands; 0.30 sits in the dead zone (pair capture 1.000, neighbour contamination 0.0032).
Caveat: the positive pairs are themselves centroid-gated (≤ 2×res), so the capture number is
partly by construction — the evidence is the width of the gap.

Fusion operates in **map coordinates**: all inputs must share a CRS. Everything here is CPU-only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely

DEFAULT_CLUSTER_THR = 0.30  # DIoU; see module docstring for calibration provenance

# on-disk layers (polygon.outlines_to_shp) carry shapefile-truncated names; normalize back
_FILE_RENAMES = {"cat_id": "category_id", "cat_name": "category_name",
                 "isin_slice": "is_within_slice"}

_RULES = ("scale_aware", "score", "bands")


# --- pairwise box metrics -----------------------------------------------------------------------

def pairwise_box_metric(boxes_a, boxes_b, metric="diou"):
    """Element-wise IoU or DIoU for aligned ``(n, 4)`` ``[xmin, ymin, xmax, ymax]`` arrays."""
    ba = np.asarray(boxes_a, dtype=float).reshape(-1, 4)
    bb = np.asarray(boxes_b, dtype=float).reshape(-1, 4)
    ixmin = np.maximum(ba[:, 0], bb[:, 0]); iymin = np.maximum(ba[:, 1], bb[:, 1])
    ixmax = np.minimum(ba[:, 2], bb[:, 2]); iymax = np.minimum(ba[:, 3], bb[:, 3])
    inter = np.clip(ixmax - ixmin, 0, None) * np.clip(iymax - iymin, 0, None)
    area_a = (ba[:, 2] - ba[:, 0]) * (ba[:, 3] - ba[:, 1])
    area_b = (bb[:, 2] - bb[:, 0]) * (bb[:, 3] - bb[:, 1])
    union = area_a + area_b - inter
    iou = np.where(union > 0, inter / union, 0.0)
    if metric == "iou":
        return iou
    if metric != "diou":
        raise ValueError(f"metric must be 'diou' or 'iou', got {metric!r}")
    cxa = (ba[:, 0] + ba[:, 2]) / 2; cya = (ba[:, 1] + ba[:, 3]) / 2
    cxb = (bb[:, 0] + bb[:, 2]) / 2; cyb = (bb[:, 1] + bb[:, 3]) / 2
    d2 = (cxa - cxb) ** 2 + (cya - cyb) ** 2
    exmin = np.minimum(ba[:, 0], bb[:, 0]); eymin = np.minimum(ba[:, 1], bb[:, 1])
    exmax = np.maximum(ba[:, 2], bb[:, 2]); eymax = np.maximum(ba[:, 3], bb[:, 3])
    c2 = (exmax - exmin) ** 2 + (eymax - eymin) ** 2
    return iou - np.where(c2 > 0, d2 / c2, 0.0)


# --- pooling ------------------------------------------------------------------------------------

def _parse_scale_key(key):
    """``512`` -> (512, 1.0); ``(256, 2)`` -> (256, 2.0) i.e. slice 256 on a ×2-SR'd image.

    A **third element is an opaque tag** (phase 5c, seam fusion): ``(256, 1, "tile_03")`` is still
    slice 256 at sr 1 — the tag does not enter ``footprint`` or any priority, it only makes the dict
    key unique and lands in the ``scale_key`` provenance string. Seam fusion needs it because
    neighbouring tiles are at the *same* scale, so scale alone cannot key them apart and a dict
    cannot hold the same key twice.
    """
    if isinstance(key, (tuple, list)):
        if len(key) == 3:
            slice_size, sr_factor, _tag = key
        elif len(key) == 2:
            slice_size, sr_factor = key
        else:
            raise ValueError(f"scale key {key!r}: expected slice_size, (slice_size, sr_factor) "
                             "or (slice_size, sr_factor, tag)")
        return int(slice_size), float(sr_factor)
    return int(key), 1.0


def pool_predictions(gdfs):
    """Pool per-scale GeoDataFrames into one, attaching the provenance columns.

    ``gdfs``: dict mapping scale key -> GeoDataFrame (in-memory ``gdf_nms`` or a read-back
    ``-nms`` layer; on-disk column names are normalized). Keys are ``slice_size`` ints,
    ``(slice_size, sr_factor)`` tuples for SR arms, or ``(slice_size, sr_factor, tag)`` for
    same-scale seam fusion across neighbouring tiles (phase 5c). Provenance attached per detection —
    ``scale_key, slice_size, sr_factor, footprint, src_index`` (``footprint`` =
    slice_size / sr_factor = native-px context window; *finest scale = smallest footprint*) —
    and these survive into the fused output (G re-masking and all audits key off them).
    """
    if not gdfs:
        raise ValueError("gdfs is empty")
    frames, crss = [], []
    for key, gdf in gdfs.items():
        slice_size, sr_factor = _parse_scale_key(key)
        g = gdf.rename(columns=_FILE_RENAMES).copy()
        if "is_within_slice" not in g.columns:
            raise ValueError(f"scale {key!r}: no is_within_slice/isin_slice column — the "
                             "truncation guard needs it (present in all pipeline outputs)")
        g["scale_key"] = str(key)
        g["slice_size"] = slice_size
        g["sr_factor"] = sr_factor
        g["footprint"] = slice_size / sr_factor
        g["src_index"] = np.arange(len(g), dtype=np.int64)
        frames.append(g)
        if len(g) > 0:
            crss.append(g.crs)
    if crss and any(c != crss[0] for c in crss[1:]):
        raise ValueError("inputs disagree on CRS — fusion is done in map coordinates; "
                         "reproject to one grid first")
    pool = pd.concat(frames, ignore_index=True, sort=False)
    return gpd.GeoDataFrame(pool, geometry="geometry", crs=crss[0] if crss else None)


# --- clustering core ----------------------------------------------------------------------------

def _same_arm(arm_of, ii, jj):
    """Mask of pairs whose two members came from the **same arm**.

    "Arm" is the full ``scale_key`` — deliberately *not* the parsed ``(slice_size, sr_factor)``.
    Seam fusion (phase 5c) keys neighbouring tiles of one scene as ``(256, 1, "A")`` and
    ``(256, 1, "B")``: same scale, different arm. Keying same-arm on the parsed scale would make
    every cross-tile pair look same-arm, silently turning seam dedup into a no-op and letting
    duplicates flow into the regional catalog at every seam with no error at all.
    """
    return arm_of[ii] == arm_of[jj]


def _candidate_pairs(boxes, cluster_thr, metric, arm_of=None, cross_arm_only=False):
    """All unordered index pairs with metric >= cluster_thr, via one bulk STRtree query.

    The spatial query only sees *intersecting* boxes, which is complete for thresholds > 0:
    disjoint boxes have IoU 0 and DIoU < 0.

    ``cross_arm_only`` drops same-arm pairs before suppression: see :func:`fuse_predictions`.
    """
    if cluster_thr <= 0:
        raise ValueError("cluster_thr must be > 0 (the intersects-based candidate search is "
                         "only complete for positive thresholds)")
    geoms = shapely.box(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3])
    tree = shapely.STRtree(geoms)
    ii, jj = tree.query(geoms, predicate="intersects")
    keep = ii < jj  # drop self-pairs and mirrored duplicates
    if cross_arm_only:
        if arm_of is None:
            raise ValueError("cross_arm_only needs arm_of")
        keep &= ~_same_arm(arm_of, ii, jj)
    ii, jj = ii[keep], jj[keep]
    if len(ii) == 0:
        return ii, jj, np.empty(0)
    vals = pairwise_box_metric(boxes[ii], boxes[jj], metric=metric)
    sel = vals >= cluster_thr
    return ii[sel], jj[sel], vals[sel]


def _greedy_suppress(n, order, ii, jj, vals):
    """Priority-ordered greedy suppression (generalized NMS).

    Returns ``keeper`` (len n): -1 for kept detections, else the pool index of the kept
    detection that suppressed it, and ``pair_val``, the metric between the two. Suppressed
    detections never suppress others, so no transitive chain-merging.
    """
    neighbors = [[] for _ in range(n)]
    for a, b, v in zip(ii, jj, vals):
        neighbors[a].append((b, v))
        neighbors[b].append((a, v))
    keeper = np.full(n, -1, dtype=np.int64)
    pair_val = np.full(n, np.nan)
    suppressed = np.zeros(n, dtype=bool)
    for i in order:
        if suppressed[i]:
            continue
        for j, v in neighbors[i]:
            if not suppressed[j] and j != i:
                suppressed[j] = True
                keeper[j] = i
                pair_val[j] = v
    return keeper, pair_val


def _keep_score(pool):
    """The confidence used as fusion's tiebreak, honouring the X-02 score contract.

    Contract version 2 keeps the detector's real confidence in ``score`` and puts truncation
    demotion in ``nms_prio``; version 1 catalogs have only ``score``, with truncated rows already
    overwritten to 0.10. Preferring ``nms_prio`` when present makes both read the same way here —
    truncated members sort last — while ``score`` stays the honest confidence for everything else.
    """
    col = "nms_prio" if "nms_prio" in pool.columns else "score"
    return pool[col].to_numpy(dtype=float)


def _priority_order(pool, rule, in_band=None):
    """Pool indices in keep-priority order (highest priority first).

    ``scale_aware``: non-slice-truncated first (``is_within_slice`` — footprint-edge keeps have
    clipped masks, last resort), then finest footprint, then score. ``score``: score only —
    pooled keep-highest-score, the control rule (known inadequate for best-scale selection).
    ``bands``: band membership as the *primary* key on top of the scale_aware ordering — NOT an
    admission filter: hard exclusion structurally fails the parity gate (measured 0.79/0.47 on
    the hirise/nac treatment pools, 2026-07-10) because a boulder found only by an out-of-band
    arm gets deleted. As a priority, in-band members win their clusters and an out-of-band
    detection with no in-band partner is rescued — parity-exact by construction.
    """
    score = _keep_score(pool)
    if rule == "score":
        return np.argsort(-score, kind="stable")
    within = pool["is_within_slice"].to_numpy(dtype=bool)
    footprint = pool["footprint"].to_numpy(dtype=float)
    # np.lexsort: last key is primary
    if in_band is None:
        return np.lexsort((-score, footprint, ~within))
    return np.lexsort((-score, footprint, ~within, ~in_band))


# --- the main entry point -----------------------------------------------------------------------

def fuse_predictions(gdfs, cluster_thr=DEFAULT_CLUSTER_THR, metric="diou", rule="scale_aware",
                     bands=None, native_res=None, engulf_thr=None, cross_arm_only=True):
    """Fuse per-scale detection sets into one catalog. Returns ``(fused, audit)``.

    Parameters
    ----------
    gdfs : dict — scale key (``slice_size`` int or ``(slice_size, sr_factor)``) -> GeoDataFrame.
    cluster_thr : float — cluster metric threshold (> 0). The default is a **DIoU** threshold;
        do not port IoU thresholds across (see module docstring).
    metric : "diou" | "iou".
    rule : "scale_aware" (keep finest non-truncated member), "score" (pooled NMS control),
        "bands" (deterministic size->scale assignment as a cluster *priority*: in-band members
        win, out-of-band detections without an in-band partner are rescued — see
        ``_priority_order`` for why it is not an admission filter).
    bands : required for ``rule="bands"``: dict scale key -> ``(lo, hi)`` half-open
        equivalent-diameter interval in **native px**. Boundaries are ours to derive (Liu & Fa's
        bins are theirs — plan §Open risks).
    native_res : map units per native px — required for ``rule="bands"`` size banding.
    engulf_thr : None | float in (0, 1] — opt-in fragment-consolidation post-pass (phase-3
        finding, 2026-07-10). A partial mask of a big boulder (fine arm sees only a piece; the
        footprint wall) sits almost wholly *inside* the whole-boulder mask but at DIoU far below
        any sane cluster threshold (measured: poly containment median 0.99, DIoU median 0.14 on
        the HiRISE dev crop, 84% of large-band cross-scale near-pairs) — the double-count mode
        the duplicate-rate audit flagged at 8-25% in the >=10 px bands. With ``engulf_thr``
        set, kept members whose polygon intersection / own polygon area >= the threshold w.r.t.
        a larger kept member are suppressed into that member (container wins, regardless of
        rule priority: at the large end whole masks are trustworthy and fragments are
        footprint-wall artifacts). Suppressions land in the audit with ``engulfed=True``.
        Default None = off (phase-1 pinned behaviour). Use :func:`matched_recall` with the same
        ``engulf_thr`` so consolidated fragments count as represented.
    cross_arm_only : bool, default True — only let detections from *different* arms suppress each
        other (review item G3-02). Candidate generation was arm-agnostic, so fusion also
        re-suppressed pairs *within* one arm that that arm's own NMS had already deliberately
        kept — 5,043 such suppressions for HiRISE and 223 for NAC in the saved audits. Two
        overlapping detections that survived a calibrated IoU-0.2 NMS are the arm's considered
        answer; a DIoU-0.30 second opinion on them is a different, uncalibrated operation. Pass
        ``False`` to request it explicitly.

        "Arm" is the full ``scale_key``, **not** the parsed ``(slice_size, sr_factor)``: seam
        fusion keys neighbouring tiles as ``(256, 1, "A")`` / ``(256, 1, "B")``, which are the
        same scale but different arms, and keying on the parsed scale would silently turn
        cross-tile dedup into a no-op.

    Returns
    -------
    fused : GeoDataFrame — the kept detections with provenance columns (``scale_key,
        slice_size, sr_factor, footprint, src_index``) plus ``n_merged`` (cluster size - 1;
        consolidations show up here) and, for ``bands``, ``in_band``. No detection is
        *deleted*: every pooled detection is either kept or suppressed by a kept one — parity
        by construction for every rule (the gate in validation re-checks it independently).
    audit : DataFrame — one row per suppressed detection: its provenance, its keeper's pool row
        (``keeper`` indexes ``fused``'s original pool index — join on ``pool_index``), the pair
        ``metric``, ``cross_scale`` (the keeper came from a different ``scale_key``, i.e. a
        different arm — for seam fusion, a different tile), ``engulfed``, and ``relation``:
        ``"cluster"`` for a DIoU suppression, ``"engulf"`` for containment, ``"engulf_chain"``
        for a member redirected to an outer container. ``metric`` always describes the pair the
        row names.
    """
    if rule not in _RULES:
        raise ValueError(f"rule must be one of {_RULES}, got {rule!r}")
    pool = pool_predictions(gdfs)
    n = len(pool)
    pool["pool_index"] = np.arange(n, dtype=np.int64)
    if n == 0:
        return pool, _empty_audit()

    in_band = None
    if rule == "bands":
        if bands is None or native_res is None:
            raise ValueError("rule='bands' needs bands={scale_key: (lo_px, hi_px)} and native_res")
        dpx = equivalent_diameter_px(pool, native_res)
        in_band = np.zeros(n, dtype=bool)
        for key, (lo, hi) in bands.items():
            in_scale = pool["scale_key"].to_numpy() == str(key)
            in_band |= in_scale & (dpx >= lo) & (dpx < hi)
        pool["in_band"] = in_band

    boxes = pool.geometry.bounds.to_numpy(dtype=float)
    scale_of = pool["scale_key"].to_numpy()
    ii, jj, vals = _candidate_pairs(boxes, cluster_thr, metric, arm_of=scale_of,
                                    cross_arm_only=cross_arm_only)
    order = _priority_order(pool, rule, in_band=in_band)
    keeper, pair_val = _greedy_suppress(n, order, ii, jj, vals)

    if engulf_thr is not None:
        keeper, pair_val, relation = _engulf_pass(pool, keeper, pair_val, engulf_thr,
                                                  arm_of=scale_of,
                                                  cross_arm_only=cross_arm_only)
    else:
        relation = np.where(keeper == -1, "", "cluster").astype(object)

    kept_mask = keeper == -1
    fused = pool.loc[kept_mask].copy()
    counts = np.bincount(keeper[keeper >= 0], minlength=n)
    fused["n_merged"] = counts[kept_mask]

    dropped = pool.loc[~kept_mask]
    audit = pd.DataFrame({
        "pool_index": dropped["pool_index"].to_numpy(),
        "scale_key": dropped["scale_key"].to_numpy(),
        "src_index": dropped["src_index"].to_numpy(),
        "score": dropped["score"].to_numpy(),
        "keeper": keeper[~kept_mask],
        "keeper_scale_key": scale_of[keeper[~kept_mask]],
        "metric": pair_val[~kept_mask],
        "cross_scale": scale_of[keeper[~kept_mask]] != dropped["scale_key"].to_numpy(),
        "engulfed": np.isin(relation[~kept_mask], ("engulf", "engulf_chain")),
        "relation": relation[~kept_mask],
    })
    return fused, audit


def _poly_containment(geoms_small, geoms_big):
    """Vectorized intersection area / area of the first (smaller) geometry."""
    inter = shapely.area(shapely.intersection(geoms_small, geoms_big))
    a = shapely.area(geoms_small)
    return np.where(a > 0, inter / a, 0.0)


def _engulf_pass(pool, keeper, pair_val, engulf_thr, arm_of=None, cross_arm_only=False):
    """Fragment-consolidation post-pass: among KEPT members, suppress any whose polygon is
    ``engulf_thr``-contained in a larger kept member's polygon (container wins). Greedy in
    descending polygon area, so a suppressed fragment never suppresses others and chains
    (fragment in fragment in whole) resolve to the outermost container.

    Returns ``(keeper, pair_val, relation)``, where ``relation`` is one of ``"cluster"`` (untouched
    by this pass), ``"engulf"`` (contained in its keeper) or ``"engulf_chain"``.

    **Chain provenance (review item G3-03).** When a fragment is engulfed, the members the DIoU
    pass had already suppressed *into* that fragment are redirected to the outer container. This
    used to leave their recorded metric as the old DIoU against the fragment and their
    ``engulfed`` flag False — so the audit named one pair and measured another, and a
    child -> fragment -> container chain produced a row claiming a relationship that had never
    been evaluated. Every redirected member's metric is now recomputed against the container it
    actually ends up attributed to, and flagged ``engulf_chain`` so a weak final relationship is
    visible rather than disguised as a strong one.
    """
    if not (0 < engulf_thr <= 1):
        raise ValueError("engulf_thr must be in (0, 1]")
    relation = np.where(keeper == -1, "", "cluster").astype(object)
    kept_idx = np.flatnonzero(keeper == -1)
    geoms = pool.geometry.values
    areas = shapely.area(geoms)
    kg = geoms[kept_idx]
    tree = shapely.STRtree(kg)
    ai, bi = tree.query(kg, predicate="intersects")
    off_diag = ai != bi
    ai, bi = ai[off_diag], bi[off_diag]
    if len(ai) == 0:
        return keeper, pair_val, relation
    # orient pairs small -> big, dedupe mirrored copies
    small_first = areas[kept_idx[ai]] <= areas[kept_idx[bi]]
    si = np.where(small_first, ai, bi)
    bg = np.where(small_first, bi, ai)
    uniq = np.unique(np.stack([si, bg], axis=1), axis=0)
    si, bg = uniq[:, 0], uniq[:, 1]
    if cross_arm_only:
        cross = ~_same_arm(arm_of, kept_idx[si], kept_idx[bg])
        si, bg = si[cross], bg[cross]
        if len(si) == 0:
            return keeper, pair_val, relation
    contain = _poly_containment(kg[si], kg[bg])
    sel = contain >= engulf_thr
    keeper = keeper.copy()
    pair_val = pair_val.copy()
    redirected = np.zeros(len(pool), dtype=bool)
    # process containers in descending area so chains resolve to surviving containers
    order = np.argsort(-areas[kept_idx[bg[sel]]], kind="stable")
    for s, b, c in zip(kept_idx[si[sel]][order], kept_idx[bg[sel]][order],
                       contain[sel][order]):
        if keeper[b] == -1 and keeper[s] == -1 and s != b:
            keeper[s] = b
            pair_val[s] = c
            relation[s] = "engulf"
            # members previously suppressed into the fragment follow it to the container; their
            # recorded relationship is no longer the one they were measured against
            children = np.flatnonzero(keeper == s)
            keeper[children] = b
            redirected[children] = True

    # Re-measure every redirected member against the container it actually landed on, so the
    # audit's metric always describes the pair the audit names.
    moved = np.flatnonzero(redirected)
    if len(moved):
        pair_val[moved] = _poly_containment(geoms[moved], geoms[keeper[moved]])
        relation[moved] = "engulf_chain"
    return keeper, pair_val, relation


def _empty_audit():
    return pd.DataFrame(columns=["pool_index", "scale_key", "src_index", "score", "keeper",
                                 "keeper_scale_key", "metric", "cross_scale", "engulfed",
                                 "relation"])


# --- validation helpers (the standing gates; see plan §Validation) -------------------------------

def equivalent_diameter_px(gdf, native_res):
    """Equivalent-area diameter in native px — THE size definition, pinned identically across
    arms (Golombek 2003 MarsMap lesson: never mix size definitions across a fused catalog)."""
    return np.sqrt(4.0 * gdf.geometry.area.to_numpy(dtype=float) / np.pi) / native_res


def matched_recall(reference, fused, cluster_thr=DEFAULT_CLUSTER_THR, metric="diou",
                   engulf_thr=None, unique=True):
    """Parity gate: fraction of ``reference`` detections *represented* in ``fused`` — i.e. paired
    with a fused detection at metric >= cluster_thr (representation, not verbatim survival: a
    baseline member may be replaced or consolidated). Must be 1.0 for the 256-only baseline.

    When the fusion ran with ``engulf_thr``, pass the same value here: a reference fragment is
    also *represented* by a fused polygon that contains ``engulf_thr`` of it (the consolidation
    case — the fragment's boulder is in the catalog, as its whole).

    ``unique=True`` (default) requires a **one-to-one** pairing, computed as a maximum-cardinality
    bipartite matching (review item G3-02). The old behaviour was many-to-one: one surviving
    keeper could credit *every* reference it happened to overlap, so a fusion that collapsed two
    neighbouring boulders into one still scored parity 1.0 — the gate could not see the loss it
    exists to catch. Maximum-cardinality (rather than greedy) matching means the gate never fails
    merely because a greedy pass consumed a keeper that a different reference needed.

    Returns ``(recall, unmatched_positional_indices)`` — a non-empty second element on the
    baseline is a gate failure and points at the offending detections.
    """
    if len(reference) == 0:
        return 1.0, np.empty(0, dtype=np.int64)
    if len(fused) == 0:
        return 0.0, np.arange(len(reference), dtype=np.int64)
    rb = reference.geometry.bounds.to_numpy(dtype=float)
    fb = fused.geometry.bounds.to_numpy(dtype=float)
    tree = shapely.STRtree(shapely.box(fb[:, 0], fb[:, 1], fb[:, 2], fb[:, 3]))
    ri, fi = tree.query(shapely.box(rb[:, 0], rb[:, 1], rb[:, 2], rb[:, 3]),
                        predicate="intersects")
    matched = np.zeros(len(reference), dtype=bool)
    if len(ri) > 0:
        vals = pairwise_box_metric(rb[ri], fb[fi], metric=metric)
        ok = vals >= cluster_thr
        if engulf_thr is not None:
            contain = _poly_containment(reference.geometry.values[ri],
                                        fused.geometry.values[fi])
            ok |= contain >= engulf_thr
        if unique:
            matched[_max_bipartite_matching(ri[ok], fi[ok], len(reference), len(fused))] = True
        else:
            matched[np.unique(ri[ok])] = True
    return float(matched.mean()), np.flatnonzero(~matched)


def _max_bipartite_matching(rows, cols, n_rows, n_cols):
    """Row indices in a maximum-cardinality matching of the sparse bipartite graph rows->cols."""
    from YOLOv8BeyondEarth.matching import max_cardinality_matching
    return max_cardinality_matching(rows, cols, n_rows, n_cols)


def duplicate_rate(fused, audit_thr=0.0, cluster_thr=DEFAULT_CLUSTER_THR, metric="diou",
                   native_res=None, bands=((3, 4), (4, 6), (6, 8), (8, 10), (10, 14),
                                           (14, np.inf))):
    """Duplicate-rate audit (plan §Validation 3): near-pairs *surviving* fusion, by size band.

    Escaped duplicates are pairs whose metric fell in ``[audit_thr, cluster_thr)`` — same
    boulder, localization shift below the cluster threshold (the tiny-box leakage mode that
    inflates SFD counts). Pairs >= cluster_thr in a fused output are clustering bugs by
    construction and are reported separately. Returns a per-band DataFrame with ``n_dets,
    n_near_pairs, n_bug_pairs, near_pair_rate`` (pairs / detections; target < ~1–2%).
    Rates are an upper bound: adjacent distinct boulders can land in the audit window too —
    the phase-3 spot-check is what separates them.
    """
    boxes = fused.geometry.bounds.to_numpy(dtype=float)
    ii, jj, vals = _candidate_pairs(boxes, max(audit_thr, 1e-9), metric)
    if native_res is not None:
        dpx = equivalent_diameter_px(fused, native_res)
    else:
        dpx = np.full(len(fused), np.nan)
        bands = ((-np.inf, np.inf),)
    rows = []
    for lo, hi in bands:
        in_band = (dpx >= lo) & (dpx < hi) if np.isfinite(lo) else np.ones(len(fused), bool)
        pair_in = in_band[ii] | in_band[jj] if len(ii) else np.zeros(0, bool)
        near = pair_in & (vals < cluster_thr)
        bug = pair_in & (vals >= cluster_thr)
        n = int(in_band.sum())
        rows.append(dict(band=f"{lo}-{'inf' if not np.isfinite(hi) else hi}", n_dets=n,
                         n_near_pairs=int(near.sum()), n_bug_pairs=int(bug.sum()),
                         near_pair_rate=(float(near.sum()) / n if n else np.nan)))
    return pd.DataFrame(rows)
