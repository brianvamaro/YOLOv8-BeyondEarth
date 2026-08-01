"""YOLO-seg label conversion (review item G5-06).

The legacy notebook converter normalises both coordinates by the image *height*, appends on rerun,
and drops small and large targets with no record. All three are pinned here against the corrected
implementation, including a demonstration of what the height-normalisation bug does to a
non-square frame — harmless on the square 500x500 boulder2024 tiles, silently shearing on anything
else.
"""
import json

import numpy as np
import pytest

from YOLOv8BeyondEarth.yolo_dataset import (
    denormalize_polygon, masks_to_label_lines, normalize_polygon, polygon_to_label_row,
    read_label_file, write_dataset_audit, write_label_file)


def _disc(size_h, size_w, cy, cx, r):
    yy, xx = np.mgrid[0:size_h, 0:size_w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2 <= r * r).astype(np.uint8)


# --- defect 1: normalisation ---------------------------------------------------------------------

def test_x_is_normalised_by_width_and_y_by_height():
    poly = np.array([[100.0, 50.0], [300.0, 50.0], [300.0, 150.0], [100.0, 150.0]])
    norm = normalize_polygon(poly, width=400, height=200)
    np.testing.assert_allclose(norm[:, 0], [0.25, 0.75, 0.75, 0.25])
    np.testing.assert_allclose(norm[:, 1], [0.25, 0.25, 0.75, 0.75])


def test_the_legacy_height_only_normalisation_shears_a_non_square_frame():
    """What the bug actually does. On a 400x200 frame every x is scaled by height/width = 0.5, so
    a square boulder is written as a 2:1 rectangle."""
    poly = np.array([[100.0, 50.0], [200.0, 50.0], [200.0, 150.0], [100.0, 150.0]])
    correct = normalize_polygon(poly, width=400, height=200)
    legacy = poly / 200.0                                   # arr = cc.flatten() / row.height

    w_correct = correct[:, 0].max() - correct[:, 0].min()
    h_correct = correct[:, 1].max() - correct[:, 1].min()
    w_legacy = legacy[:, 0].max() - legacy[:, 0].min()
    h_legacy = legacy[:, 1].max() - legacy[:, 1].min()
    assert w_correct == pytest.approx(h_correct * 0.5)      # 100/400 vs 100/200
    assert w_legacy == pytest.approx(h_legacy)              # the shear: both read 0.5
    assert legacy[:, 0].max() > 0.9 and correct[:, 0].max() == pytest.approx(0.5)


def test_square_frames_are_unaffected_which_is_why_it_went_unnoticed():
    poly = np.array([[100.0, 50.0], [300.0, 50.0], [300.0, 150.0], [100.0, 150.0]])
    np.testing.assert_allclose(normalize_polygon(poly, 500, 500), poly / 500.0)


def test_normalisation_round_trips():
    poly = np.array([[10.0, 20.0], [310.0, 25.0], [300.0, 190.0], [15.0, 180.0]])
    back = denormalize_polygon(normalize_polygon(poly, 400, 200), 400, 200)
    np.testing.assert_allclose(back, poly, atol=1e-9)


def test_zero_sized_frame_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        normalize_polygon(np.zeros((4, 2)), 0, 100)


def test_label_row_is_clipped_into_range():
    poly = np.array([[-5.0, -5.0], [405.0, -5.0], [405.0, 205.0], [-5.0, 205.0]])
    row = polygon_to_label_row(poly, 400, 200)
    values = np.asarray(row.split()[1:], dtype=float)
    assert row.split()[0] == "0"
    assert values.min() >= 0.0 and values.max() <= 1.0


# --- defect 2: idempotence -------------------------------------------------------------------------

def test_rewriting_a_label_file_replaces_it(tmp_path):
    """The legacy converter opened in append mode, so a second run doubled every annotation."""
    path = tmp_path / "tile.txt"
    lines = ["0 0.1 0.1 0.2 0.1 0.2 0.2", "0 0.5 0.5 0.6 0.5 0.6 0.6"]
    for _ in range(3):
        write_label_file(path, lines)
    assert len(read_label_file(path)) == 2, "rerunning the conversion duplicated annotations"


def test_write_leaves_no_temp_files(tmp_path):
    path = tmp_path / "tile.txt"
    write_label_file(path, ["0 0.1 0.1 0.2 0.1 0.2 0.2"])
    assert [p.name for p in tmp_path.iterdir()] == ["tile.txt"]


def test_empty_label_file_is_written_not_skipped(tmp_path):
    """A tile with no boulders needs an empty label file; a missing one reads as unlabelled."""
    path = tmp_path / "tile.txt"
    write_label_file(path, [])
    assert path.exists() and path.read_text() == "" and read_label_file(path) == []


def test_a_failed_write_does_not_destroy_the_previous_labels(tmp_path, monkeypatch):
    path = tmp_path / "tile.txt"
    write_label_file(path, ["0 0.1 0.1 0.2 0.1 0.2 0.2"])
    import os
    monkeypatch.setattr(os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("nope")))
    with pytest.raises(OSError):
        write_label_file(path, ["0 0.9 0.9 0.8 0.9 0.8 0.8"])
    assert len(read_label_file(path)) == 1
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]


# --- defect 3: silent drops ------------------------------------------------------------------------

def test_dropped_instances_are_counted_by_reason():
    h, w = 200, 400
    masks = [_disc(h, w, 100, 200, 20),          # kept
             _disc(h, w, 50, 50, 1),             # too small (~5 px)
             np.ones((h, w), np.uint8),          # too large
             np.zeros((h, w), np.uint8)]         # no contour
    lines, counts = masks_to_label_lines(masks, width=w, height=h)
    assert counts["n_in"] == 4 and counts["n_out"] == 1
    assert counts["too_small"] == 1 and counts["too_large"] == 1
    # an empty mask is an upstream failure, not a size cut; folding the two together would hide it
    assert counts["empty"] == 1 and counts["no_contour"] == 0
    assert len(lines) == 1


def test_bounds_can_be_disabled():
    h, w = 100, 100
    tiny = _disc(h, w, 50, 50, 1)
    _, counts = masks_to_label_lines([tiny], w, h, min_area_px=None)
    assert counts["too_small"] == 0 and counts["n_out"] == 1


def test_edge_policy_is_explicit():
    h, w = 100, 200
    edge = np.zeros((h, w), np.uint8)
    edge[0:20, 0:20] = 1                          # touches the top-left corner
    interior = _disc(h, w, 50, 100, 15)

    kept, counts = masks_to_label_lines([edge, interior], w, h, keep_edge=True)
    assert counts["n_out"] == 2 and counts["at_edge"] == 0
    dropped, counts = masks_to_label_lines([edge, interior], w, h, keep_edge=False)
    assert counts["n_out"] == 1 and counts["at_edge"] == 1


def test_labels_are_traced_through_the_production_convention():
    """Cleanup is largest-blob + fill, so a speck and a hole do not become extra vertices."""
    h, w = 120, 240
    m = _disc(h, w, 60, 120, 25)
    m[58:62, 118:122] = 0                         # a hole
    m[5:8, 5:8] = 1                               # a speck
    lines, counts = masks_to_label_lines([m], w, h)
    assert counts["n_out"] == 1
    poly = denormalize_polygon(read_label_file_lines(lines)[0], w, h)
    assert poly[:, 0].min() > 50, "the speck leaked into the polygon"


def read_label_file_lines(lines):
    out = []
    for line in lines:
        coords = np.asarray(line.split()[1:], dtype=float)
        out.append(coords.reshape(-1, 2))
    return out


def test_non_square_tile_round_trips_through_a_real_file(tmp_path):
    """The end-to-end check: a boulder at a known place in a 400x200 tile comes back where it was."""
    h, w = 200, 400
    mask = _disc(h, w, 60, 300, 18)
    lines, counts = masks_to_label_lines([mask], width=w, height=h)
    assert counts["n_out"] == 1
    path = write_label_file(tmp_path / "tile.txt", lines)

    (_, norm), = read_label_file(path)
    poly = denormalize_polygon(norm, w, h)
    assert poly[:, 0].mean() == pytest.approx(300, abs=2)
    assert poly[:, 1].mean() == pytest.approx(60, abs=2)


def test_audit_records_the_inclusion_limits(tmp_path):
    _, counts = masks_to_label_lines([_disc(100, 100, 50, 50, 10)], 100, 100)
    path = write_dataset_audit(tmp_path / "audit.json", counts, extra={"split": "train"})
    rec = json.loads(path.read_text())
    assert rec["counts"]["n_out"] == 1 and rec["min_area_px"] == 6 and rec["split"] == "train"
