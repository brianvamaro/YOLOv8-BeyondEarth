"""Tests for the stale-driver screen — specifically the two false-positive modes it actually hit.

The screen's first run flagged 64 of 83 drivers, which is noise. Both causes are pinned here,
because a screen that cries wolf gets ignored and then the one real finding is lost with it.
"""
import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SCREEN = Path(__file__).resolve().parents[1] / "scripts" / "review" / "screen_stale_drivers.py"


@pytest.fixture(scope="module")
def screen():
    spec = importlib.util.spec_from_file_location("screen_stale_drivers", SCREEN)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_only_write_lines_count_as_outputs(screen, tmp_path):
    """A file the driver READS must not be mistaken for one it writes.

    This was the live failure: `run_obb_orientation.py` reads a 2026-07-06 input and writes current
    outputs, and scanning the whole file let the *input's* date decide the verdict.
    """
    script = tmp_path / "run_thing.py"
    script.write_text(
        'ref = pd.read_csv("some_old_input.csv")\n'
        'out.to_csv("the_real_output.csv", index=False)\n')
    names, prefixes = screen.declared_outputs(script)
    assert "the_real_output.csv" in names
    assert "some_old_input.csv" not in names


def test_falls_back_to_whole_file_when_no_write_line_names_a_file(screen, tmp_path):
    """A path built into a variable and written later still has to be found, or the driver reports
    'no outputs' and the screen goes blind."""
    script = tmp_path / "run_indirect.py"
    script.write_text('p = OUT / "deferred_name.csv"\ndf.to_csv(p)\n')
    names, _ = screen.declared_outputs(script)
    assert "deferred_name.csv" in names


def test_interpolated_names_are_found_by_prefix(screen, tmp_path):
    """Most drivers here build every output name at runtime, so the finished basename is nowhere in
    the source. Without prefixes the screen cannot see them at all."""
    script = tmp_path / "run_suffixed.py"
    script.write_text('df.to_csv(f"{OUT}/obb_crop_angles{SUF}.csv", index=False)\n')
    names, prefixes = screen.declared_outputs(script)
    assert "obb_crop_angles" in prefixes


def test_exact_match_beats_prefix_match(screen):
    """A prefix is shared property, so a prefix hit must never override an exact one.

    `operating_point` matches both `measure_operating_point.py`'s own `operating_point.json` and the
    unrelated `operating_point_test_split.json`. If the fresher unrelated file could win, a true
    STALE becomes a false ok — the one direction this screen must not fail in.
    """
    old = datetime(2026, 8, 3, tzinfo=timezone.utc)
    new = datetime(2026, 8, 21, tzinfo=timezone.utc)
    index = {"operating_point.json": (old, Path("a")),
             "operating_point_test_split.json": (new, Path("b"))}
    hit, how = screen.newest_match({"operating_point.json"}, {"operating_point"}, index)
    assert how == "exact"
    assert hit[0] == old               # its OWN output, not the newer neighbour's


def test_prefix_is_used_when_there_is_no_exact_match(screen):
    old = datetime(2026, 7, 6, tzinfo=timezone.utc)
    new = datetime(2026, 8, 9, tzinfo=timezone.utc)
    index = {"obb_crop_angles.csv": (old, Path("a")),
             "obb_crop_angles_rot_de.csv": (new, Path("b"))}
    hit, how = screen.newest_match(set(), {"obb_crop_angles"}, index)
    assert how == "prefix"
    assert hit[0] == new               # the newest member of the family


def test_no_match_at_all_is_reported_not_guessed(screen):
    hit, how = screen.newest_match({"nothing_like_this.csv"}, set(), {"other.csv": (
        datetime(2026, 8, 1, tzinfo=timezone.utc), Path("x"))})
    assert hit is None and how is None


def test_tolerance_covers_run_then_commit(screen):
    """Committing a fix after the run that verified it is the normal workflow and must not read as
    stale — it was the second false-positive mode."""
    assert screen.TOLERANCE_HOURS >= 12
    ran = datetime(2026, 8, 21, 10, tzinfo=timezone.utc)
    committed = ran + timedelta(hours=3)
    assert (committed - ran).total_seconds() / 3600.0 <= screen.TOLERANCE_HOURS
