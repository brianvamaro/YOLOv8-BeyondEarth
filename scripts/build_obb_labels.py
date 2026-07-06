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
images (zero-copy, no admin needed on Windows). Also writes ``boulder2024_obb.yaml``.

Run:  python scripts/build_obb_labels.py
"""
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


def convert_file(src_txt, dst_txt):
    """Convert one seg label file to OBB. Returns (n_in, n_out, n_outside) counts."""
    with open(src_txt) as f:
        rows = [r.strip() for r in f if r.strip()]
    out_lines, n_in, n_outside = [], 0, 0
    for row in rows:
        cls, poly = parse_seg_row(row)
        if cls is None:
            continue
        n_in += 1
        # fit in normalized space (isotropic since tiles are square)
        corners, _, _ = ellipse_obb(Polygon(poly), DENSIFY)
        if corners is None:
            continue               # degenerate polygon / failed fit -> dropped
        if (corners < 0).any() or (corners > 1).any():
            n_outside += 1
        corners = np.clip(corners, 0.0, 1.0)   # ultralytics expects normalized in [0,1]
        out_lines.append(cls + " " + " ".join(f"{c:.6f}" for c in corners.flatten()))
    with open(dst_txt, "w") as f:                # overwrite (never append)
        f.write("\n".join(out_lines) + ("\n" if out_lines else ""))
    return n_in, len(out_lines), n_outside


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
    tot_in = tot_out = tot_drop = tot_outside = 0
    for split in SPLITS:
        s_lbl = os.path.join(SRC, split, "labels")
        s_img = os.path.join(SRC, split, "images")
        d_lbl = os.path.join(DST, split, "labels")
        d_img = os.path.join(DST, split, "images")
        os.makedirs(d_lbl, exist_ok=True)
        ensure_images_junction(s_img, d_img)
        files = [f for f in os.listdir(s_lbl) if f.endswith(".txt")]
        s_in = s_out = s_drop = s_outside = 0
        for fn in files:
            n_in, n_out, n_outside = convert_file(os.path.join(s_lbl, fn), os.path.join(d_lbl, fn))
            s_in += n_in; s_out += n_out; s_drop += (n_in - n_out); s_outside += n_outside
        print(f"{split:>11}: {len(files):>4} files  {s_in:>7} seg -> {s_out:>7} obb  "
              f"(dropped {s_drop} bad fits, {s_outside} clipped to edge)")
        tot_in += s_in; tot_out += s_out; tot_drop += s_drop; tot_outside += s_outside

    yaml_path = os.path.join(DST, "boulder2024_obb.yaml")
    with open(yaml_path, "w") as f:
        f.write("# OBB labels for boulder2024, built from the shipped seg polygons via the unbiased\n"
                "# EllipseModel fit (scripts/build_obb_labels.py). Solution H (docs/solutions/H_...).\n"
                f"path: {DST.replace(os.sep, '/')}\n"
                "train: train/images\nval: validation/images\ntest: test/images\n"
                "nc: 1\nnames:\n  0: boulder\n")
    print(f"\nTOTAL: {tot_in} seg -> {tot_out} obb  (dropped {tot_drop} = "
          f"{100*tot_drop/max(tot_in,1):.2f}%, {tot_outside} edge-clipped)")
    print(f"wrote {yaml_path}")


if __name__ == "__main__":
    main()
