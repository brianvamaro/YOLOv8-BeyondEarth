"""One-to-one matching between two sets of detections (review items G2-05 and G3-02).

Every comparison in this project — model against human truth, fused against baseline, SAM against
YOLO — rests on deciding which detection corresponds to which. Doing that with a nearest-neighbour
query is **many-to-one**: several references can all claim the same prediction, so a model that
merges two neighbouring boulders into one still scores full recall, and the merged mask's area is
counted twice in an area-ratio distribution. The gate cannot see the failure it exists to catch.

The helpers here are one-to-one and *optimal*, not greedy. Greedy-by-distance can consume a
prediction that another reference needed and report a recall below the achievable one, which would
make a matching artefact look like a detector deficit.
"""
from __future__ import annotations

import numpy as np


def max_cardinality_matching(rows, cols, n_rows, n_cols):
    """Row indices covered by a maximum-cardinality matching of the sparse graph ``rows -> cols``.

    Use when only *how many* references can be paired matters (recall).
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching

    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    if len(rows) == 0:
        return np.empty(0, dtype=np.int64)
    graph = coo_matrix((np.ones(len(rows), dtype=bool), (rows, cols)),
                       shape=(n_rows, n_cols)).tocsr()
    pairing = maximum_bipartite_matching(graph, perm_type="column")
    return np.flatnonzero(pairing >= 0)


#: Above this many nodes on either side of one connected component, fall back to greedy rather
#: than a dense assignment. Components are normally 1-3 nodes, because admissibility is a tight
#: spatial radius; anything near this cap means the gate is far too loose to be meaningful.
MAX_DENSE_COMPONENT = 400


def min_cost_matching(rows, cols, costs, n_rows, n_cols):
    """Optimal one-to-one pairing over admissible edges: maximum cardinality, least total cost.

    Returns ``(matched_rows, matched_cols)``. Both are index arrays into the original sets, and
    neither contains a repeat — that is the whole point.

    The two objectives are lexicographic, not interchangeable: pair as many references as possible
    *first*, and only then prefer the closer pairing. Doing it in two separate stages does not
    work, because there are usually several maximum matchings covering different row sets and the
    first stage would arbitrarily fix the wrong one. Instead the admissible graph is split into
    connected components — tiny, because admissibility is a tight spatial radius — and each is
    solved exactly with a dense assignment whose non-edges cost more than any complete real
    alternative, so an extra real pair always beats a cheaper-but-smaller one.
    """
    from scipy.optimize import linear_sum_assignment
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    costs = np.asarray(costs, dtype=float)
    empty = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
    if len(rows) == 0:
        return empty

    # one undirected graph over rows and (offset) columns, so a component holds both sides
    total = n_rows + n_cols
    adj = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols + n_rows)),
                     shape=(total, total)).tocsr()
    _, labels = connected_components(adj, directed=False)

    big = float(costs.max()) * (min(n_rows, n_cols) + 1) + 1.0
    order = np.argsort(labels[rows], kind="stable")
    rows, cols, costs = rows[order], cols[order], costs[order]
    boundaries = np.flatnonzero(np.diff(labels[rows])) + 1

    out_r, out_c = [], []
    for chunk_r, chunk_c, chunk_w in zip(np.split(rows, boundaries), np.split(cols, boundaries),
                                         np.split(costs, boundaries)):
        uniq_r, local_r = np.unique(chunk_r, return_inverse=True)
        uniq_c, local_c = np.unique(chunk_c, return_inverse=True)
        if max(len(uniq_r), len(uniq_c)) > MAX_DENSE_COMPONENT:
            pr, pc = _greedy_component(chunk_r, chunk_c, chunk_w)
            out_r.extend(pr); out_c.extend(pc)
            continue
        cost = np.full((len(uniq_r), len(uniq_c)), big, dtype=float)
        cost[local_r, local_c] = np.minimum(cost[local_r, local_c], chunk_w)
        ri, ci = linear_sum_assignment(cost)
        real = cost[ri, ci] < big
        out_r.extend(uniq_r[ri[real]].tolist())
        out_c.extend(uniq_c[ci[real]].tolist())
    return np.asarray(out_r, dtype=np.int64), np.asarray(out_c, dtype=np.int64)


def _greedy_component(rows, cols, costs):
    """Cheapest-edge-first fallback for a pathologically large component (see the cap above)."""
    taken_r, taken_c, pr, pc = set(), set(), [], []
    for k in np.argsort(costs, kind="stable"):
        r, c = int(rows[k]), int(cols[k])
        if r not in taken_r and c not in taken_c:
            taken_r.add(r); taken_c.add(c); pr.append(r); pc.append(c)
    return pr, pc


def unique_centroid_match(ref_xy, model_xy, max_dist):
    """Pair each reference with at most one model detection, within a per-reference radius.

    ``max_dist`` is a scalar or a per-reference array. Returns ``(ref_idx, model_idx)`` — an
    optimal one-to-one pairing, so ``ref_idx`` and ``model_idx`` are both duplicate-free.
    """
    from scipy.spatial import cKDTree

    ref_xy = np.asarray(ref_xy, dtype=float).reshape(-1, 2)
    model_xy = np.asarray(model_xy, dtype=float).reshape(-1, 2)
    empty = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
    if len(ref_xy) == 0 or len(model_xy) == 0:
        return empty
    radius = np.broadcast_to(np.asarray(max_dist, dtype=float), (len(ref_xy),))

    tree = cKDTree(model_xy)
    # every model detection within the largest admissible radius, then filtered per reference
    neighbours = tree.query_ball_point(ref_xy, float(radius.max()))
    rows, cols, costs = [], [], []
    for i, cand in enumerate(neighbours):
        for j in cand:
            d = float(np.hypot(*(ref_xy[i] - model_xy[j])))
            if d <= radius[i]:
                rows.append(i); cols.append(j); costs.append(d)
    if not rows:
        return empty
    return min_cost_matching(rows, cols, costs, len(ref_xy), len(model_xy))
