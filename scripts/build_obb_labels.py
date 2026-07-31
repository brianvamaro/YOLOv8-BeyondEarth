"""Solution H pre-gate prep: build oriented-bounding-box (OBB) labels for boulder2024.

Reads the shipped YOLO-**seg** polygon labels and converts each instance to an ultralytics
YOLO-**OBB** label (``class x1 y1 x2 y2 x3 y3 x4 y4``, normalized) via the *unbiased*
``EllipseModel`` fit (``orientation.ellipse_obb``) — NOT a minimum-rotated-rectangle of the
pixelated polygon, which would train the exact 0/90 grid bias Goal 2 is removing (H plan,
label caveat #1).

boulder2024 tiles are 500x500 (square), so normalized coordinates are isotropic and the ellipse
angle is preserved without any aspect correction; the fit is done directly in normalized units
(densify spacing = 1 px = 1/500).

Output mirrors the source tree at ``data_prieur/boulder2024_obb/{split}/`` with ``labels/`` written
fresh (overwrite, never append) and ``images/`` wired as a directory **junction** to the original
images (zero-copy, no admin needed on Windows). Also writes ``boulder2024_obb.yaml`` and an audit
JSON.

**Edge policy (review item G2-08).** A box that extends past the tile edge used to have its
corners clipped to [0, 1] *independently*, which does not clip a rectangle — it deforms it into a
quadrilateral with the wrong angle and aspect, for 3.4% of instances. Since the OBB head is being
trained precisely to read angle, teaching it deformed boxes defeats the exercise. The policy is now
explicit:

``drop``  (default) — omit boxes that leave the tile. Honest, at the cost of under-representing
          edge boulders; the counts are written to the audit so the bias is quantified.
``keep``  — emit the true rectangle with out-of-range coordinates intact.

Run:  python scripts/build_obb_labels.py [--edge-policy drop|keep]
"""
import argparse
import json
import os
import sys
import subprocess
import numpy as np
from shapely.geometry import Polygon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from YOLOv8BeyondEarth.orientation import ellipse_obb

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(REPO, "data_prieur", "boulder2024")
DST = os.path.join(REPO, "data_prieur", "boulder2024_obb")
SPLITS = ("train", "validation", "test")
TILE_PX = 500                      # boulder2024 tile size (square)
DENSIFY = 1.0 / TILE_PX            # ellipse densify spacing = 1 px, in normalized units
EDGE_POLICIES = ("drop", "keep")


def is_rectangle(corners, atol=1e-5, rtol=1e-3):
    """True if the 4 ordered corners form a rectangle: equal diagonals sharing a midpoint.

    Tolerances allow for the 6-decimal label format, whose rounding is ~5e-7 per coordinate and
    therefore a relative error of ~1e-5 on a box a few hundredths of a tile across.
    """
    c = np.asarray(corners, dtype=float).reshape(4, 2)
    d1, d2 = c[2] - c[0], c[3] - c[1]                      # the two diagonals
    n1, n2 = float(np.linalg.norm(d1)), float(np.linalg.norm(d2))
    tol = atol + rtol * max(n1, n2)
    mid_ok = bool(np.allclose((c[0] + c[2]) / 2, (c[1] + c[3]) / 2, atol=tol, rtol=0))
    return bool(mid_ok and abs(n1 - n2) <= tol)


def parse_seg_row(row):
    """A seg label row -> (cls, Nx2 normalized polygon) or (None, None) if too short to be a polygon."""
    v = row.split()
    if len(v) < 7:                 # class + >=3 (x,y) pairs to be a polygon
        return None, None
    cls = v[0]
    xy = np.array(v[1:], dtype=float)
    if xy.size % 2:                # odd count -> malformed
        return None, None
    return cls, xy.reshape(-1, 2)


def convert_file(src_txt, dst_txt, edge_policy="drop"):
    """Convert one seg label file to OBB. Returns a per-file count dict."""
    with open(src_txt) as f:
        rows = [r.strip() for r in f if r.strip()]
    out_lines = []
    counts = {"n_in": 0, "n_bad_fit": 0, "n_edge": 0, "n_edge_dropped": 0}
    for row in rows:
        cls, poly = parse_seg_row(row)
        if cls is None:
            continue
        counts["n_in"] += 1
        # fit in normalized space (isotropic since tiles are square)
        corners, _, _ = ellipse_obb(Polygon(poly), DENSIFY)
        if corners is None:
            counts["n_bad_fit"] += 1
            continue               # degenerate polygon / failed fit -> dropped
        if (corners < 0).any() or (corners > 1).any():
            counts["n_edge"] += 1
            if edge_policy == "drop":
                counts["n_edge_dropped"] += 1
                continue
        # every emitted label must still be a rectangle — this is what independent corner
        # clipping silently broke
        assert is_rectangle(corners), f"non-rectangular OBB emitted from {src_txt}"
        out_lines.append(cls + " " + " ".join(f"{c:.6f}" for c in corners.flatten()))
    with open(dst_txt, "w") as f:                # overwrite (never append)
        f.write("\n".join(out_lines) + ("\n" if out_lines else ""))
    counts["n_out"] = len(out_lines)
    return counts


def ensure_images_junction(src_images, dst_images):
    """Wire dst_images -> src_images as a directory junction (Windows, no admin). Idempotent."""
    if os.path.exists(dst_images):
        return
    if sys.platform == "win32":
        subprocess.run(["cmd", "/c", "mklink", "/J", dst_images, src_images], check=True,
                       capture_output=True)
    else:
        os.symlink(src_images, dst_images)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge-policy", choices=EDGE_POLICIES, default="drop",
                    help="what to do with boxes extending past the tile edge (see the docstring)")
    args = ap.parse_args()

    audit = {"edge_policy": args.edge_policy, "splits": {}}
    total = {"n_in": 0, "n_out": 0, "n_bad_fit": 0, "n_edge": 0, "n_edge_dropped": 0}
    for split in SPLITS:
        s_lbl = os.path.join(SRC, split, "labels")
        s_img = os.path.join(SRC, split, "images")
        d_lbl = os.path.join(DST, split, "labels")
        d_img = os.path.join(DST, split, "images")
        os.makedirs(d_lbl, exist_ok=True)
        ensure_images_junction(s_img, d_img)
        files = [f for f in os.listdir(s_lbl) if f.endswith(".txt")]
        acc = {k: 0 for k in total}
        for fn in files:
            counts = convert_file(os.path.join(s_lbl, fn), os.path.join(d_lbl, fn),
                                  edge_policy=args.edge_policy)
            for k in acc:
                acc[k] += counts[k]
        acc["n_files"] = len(files)
        audit["splits"][split] = acc
        for k in total:
            total[k] += acc[k]
        print(f"{split:>11}: {len(files):>4} files  {acc['n_in']:>7} seg -> {acc['n_out']:>7} obb  "
              f"({acc['n_bad_fit']} bad fits, {acc['n_edge']} edge-crossing, "
              f"{acc['n_edge_dropped']} dropped by policy)")

    yaml_path = os.path.join(DST, "boulder2024_obb.yaml")
    with open(yaml_path, "w") as f:
        f.write("# OBB labels for boulder2024, built from the shipped seg polygons via the unbiased\n"
                "# EllipseModel fit (scripts/build_obb_labels.py). Solution H (docs/solutions/H_...).\n"
                f"# edge policy: {args.edge_policy}; see boulder2024_obb_audit.json\n"
                f"path: {DST.replace(os.sep, '/')}\n"
                "train: train/images\nval: validation/images\ntest: test/images\n"
                "nc: 1\nnames:\n  0: boulder\n")
    audit["total"] = total
    audit_path = os.path.join(DST, "boulder2024_obb_audit.json")
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)

    edge_frac = 100 * total["n_edge"] / max(total["n_in"], 1)
    print(f"\nTOTAL: {total['n_in']} seg -> {total['n_out']} obb  "
          f"({total['n_bad_fit']} bad fits, {total['n_edge']} edge-crossing = {edge_frac:.2f}%, "
          f"{total['n_edge_dropped']} dropped)")
    print(f"wrote {yaml_path}\nwrote {audit_path}")


if __name__ == "__main__":
    main()
