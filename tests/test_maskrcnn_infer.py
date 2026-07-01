"""Tests for the sliced-inference tiling logic (Test 3, step 3).

Fast + data-free: exercises the tile-offset grid (edge coverage / overlap), the property that
guarantees the full raster is covered with no gap at the trailing edge.
"""
from YOLOv8BeyondEarth.maskrcnn.infer import _slice_offsets


def test_offsets_single_tile_when_smaller_than_slice():
    assert _slice_offsets(300, 500, 0.2) == [0]
    assert _slice_offsets(500, 500, 0.2) == [0]


def test_offsets_cover_to_edge():
    for full in (501, 1000, 1234, 22490):
        size, overlap = 500, 0.2
        offs = _slice_offsets(full, size, overlap)
        assert offs[0] == 0
        # last window ends exactly at the raster edge (full coverage, no gap)
        assert offs[-1] == full - size
        # monotonic, and every tile fits inside the raster
        assert all(b > a for a, b in zip(offs, offs[1:]))
        assert all(0 <= o <= full - size for o in offs)


def test_offsets_step_matches_overlap():
    offs = _slice_offsets(10000, 500, 0.2)  # step = 500*(1-0.2) = 400
    # interior spacing is the requested step (last step may be shorter due to edge clamp)
    interior_steps = [b - a for a, b in zip(offs, offs[1:])][:-1]
    assert all(s == 400 for s in interior_steps)


def test_offsets_no_overlap():
    offs = _slice_offsets(1500, 500, 0.0)  # step = 500
    assert offs == [0, 500, 1000]
