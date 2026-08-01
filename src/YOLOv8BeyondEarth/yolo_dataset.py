"""Mask -> YOLO-seg label conversion, done correctly (review item G5-06).

The legacy converter lives in ``resources/nb/shapefiles_to_YOLOv8_custom_dataset.ipynb`` and is a
hand-maintained tutorial from the original fork, so it is not edited. It has three defects, and
anything built with it inherits them:

1. **Both coordinates are normalised by the image height** (``arr = cc.flatten() / row.height``).
   The flattened array interleaves x and y, so x is divided by the wrong dimension. Harmless on the
   square 500x500 boulder2024 tiles and silently corrupting on anything else — every x is scaled by
   ``height / width``, which shears the labels.
2. **Label files are opened in append mode** (``open(txt, "a")``), so re-running the converter
   duplicates every annotation instead of replacing it. Nothing about the output says how many
   times it has been run.
3. **Small and large targets are dropped with no record.** Instances below 6 px and above
   ``(0.75 * 500)^2`` vanish; there is no count, so a dataset cannot report its own size limits.

This module is the replacement: width- and height-aware, idempotent, and explicit about what it
drops. It is deliberately independent of Goal 5's undecided scope — it is the correct version of a
conversion the project already does.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .orientation import mask_to_polygon_geom
from .polygon import binary_mask_to_polygon, mask_support_area

#: Instances outside these bounds are dropped *and counted*. ``None`` disables a bound.
DEFAULT_MIN_AREA_PX = 6
DEFAULT_MAX_AREA_FRACTION = 0.75 ** 2      # the legacy (0.75*size)^2 rule, as a fraction


def normalize_polygon(polygon_xy, width, height):
    """Pixel ``(x, y)`` vertices -> YOLO-normalised, x by **width** and y by **height**.

    The one-line fix for defect 1. Kept as its own function precisely because the bug was a
    plausible-looking flat division on an interleaved array.
    """
    xy = np.asarray(polygon_xy, dtype=float).reshape(-1, 2)
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got {width}x{height}")
    return np.stack([xy[:, 0] / float(width), xy[:, 1] / float(height)], axis=-1)


def polygon_to_label_row(polygon_xy, width, height, class_id=0, decimals=6, clip=True):
    """One YOLO-seg label line: ``class x1 y1 x2 y2 ...`` normalised to ``[0, 1]``."""
    norm = normalize_polygon(polygon_xy, width, height)
    if clip:
        norm = np.clip(norm, 0.0, 1.0)
    flat = norm.reshape(-1).round(decimals)
    return " ".join([str(int(class_id))] + [f"{v:.{decimals}f}" for v in flat])


def masks_to_label_lines(masks, width, height, class_id=0,
                         min_area_px=DEFAULT_MIN_AREA_PX,
                         max_area_fraction=DEFAULT_MAX_AREA_FRACTION,
                         keep_edge=True, decimals=6):
    """Convert a tile's instance masks to label lines, returning ``(lines, counts)``.

    ``counts`` names every reason an instance did not make it — ``too_small``, ``too_large``,
    ``no_contour``, ``at_edge`` — so a dataset can state its own inclusion limits instead of
    quietly having them.

    Tracing goes through the production convention (:func:`orientation.mask_to_polygon_geom`'s
    cleanup plus :func:`polygon.binary_mask_to_polygon`), so labels and predictions describe
    boundaries the same way.
    """
    lines = []
    counts = {"n_in": 0, "empty": 0, "too_small": 0, "too_large": 0, "no_contour": 0,
              "at_edge": 0}
    max_area = None if max_area_fraction is None else max_area_fraction * width * height

    for mask in masks:
        counts["n_in"] += 1
        arr = np.asarray(mask)
        area = mask_support_area(arr)
        if area == 0:
            # an empty mask is not "below the size floor" — it is an upstream failure, and
            # folding the two together hides it behind a legitimate-looking size cut
            counts["empty"] += 1
            continue
        if min_area_px is not None and area <= min_area_px:
            counts["too_small"] += 1
            continue
        if max_area is not None and area >= max_area:
            counts["too_large"] += 1
            continue
        geom, _ = mask_to_polygon_geom(arr)          # largest blob, holes filled
        if geom is None:
            counts["no_contour"] += 1
            continue
        # mask_to_polygon_geom returns a North-up frame (y = -row); labels are in image rows
        poly = np.asarray(geom.exterior.coords, dtype=float)
        poly[:, 1] *= -1.0
        if not keep_edge:
            touches = (poly[:, 0].min() <= 0 or poly[:, 1].min() <= 0
                       or poly[:, 0].max() >= width - 1 or poly[:, 1].max() >= height - 1)
            if touches:
                counts["at_edge"] += 1
                continue
        lines.append(polygon_to_label_row(poly, width, height, class_id=class_id,
                                          decimals=decimals))
    counts["n_out"] = len(lines)
    return lines, counts


def write_label_file(path, lines):
    """Write a label file **idempotently** — the fix for defect 2.

    The legacy converter opened in append mode, so a second run doubled every annotation and a
    third tripled it, with nothing in the output to say so. Written to a temp file and moved into
    place, so an interrupted run cannot leave a half-file that looks finished.
    """
    import os
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(lines) + ("\n" if lines else ""))
        os.replace(tmp, path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def read_label_file(path):
    """Parse a YOLO-seg label file back to ``[(class_id, (N, 2) normalised polygon), ...]``."""
    out = []
    text = Path(path).read_text(encoding="utf-8") if Path(path).exists() else ""
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 7:                    # class + at least 3 vertices
            continue
        coords = np.asarray(parts[1:], dtype=float)
        if coords.size % 2:
            continue
        out.append((int(parts[0]), coords.reshape(-1, 2)))
    return out


def denormalize_polygon(norm_xy, width, height):
    """Inverse of :func:`normalize_polygon` — used by the round-trip test."""
    xy = np.asarray(norm_xy, dtype=float).reshape(-1, 2)
    return np.stack([xy[:, 0] * float(width), xy[:, 1] * float(height)], axis=-1)


def write_dataset_audit(path, counts, extra=None):
    """Record what the conversion included and excluded, next to the labels."""
    record = {"counts": counts, "min_area_px": DEFAULT_MIN_AREA_PX,
              "max_area_fraction": DEFAULT_MAX_AREA_FRACTION, **(extra or {})}
    Path(path).write_text(json.dumps(record, indent=2))
    return Path(path)
