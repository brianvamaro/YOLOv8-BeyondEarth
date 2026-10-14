"""Resumable runs that do not lie about being finished (review items G3-08 and G3-09).

The Goal-3 runners treated an arm's GeoPackage existing as proof the arm finished. A run killed
between writing the layer and writing its timing JSON leaves an artifact that looks complete and a
record that is missing — which is how AP ended up with 280 done-markers and 223 timing entries.
"""
import json
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio as rio
from rasterio.transform import from_origin

from YOLOv8BeyondEarth import predict as P
from YOLOv8BeyondEarth.runstate import (
    COMPLETE, FAILED, INCOMPLETE, MISSING, STALE, StageLedger, file_digest, stage_key)


@pytest.fixture
def ledger(tmp_path):
    return StageLedger(tmp_path / "run.jsonl")


def _artifact(tmp_path, name="arm256.gpkg", body=b"layer"):
    p = tmp_path / name
    p.write_bytes(body)
    return p


# --- the states ------------------------------------------------------------------------------------

def test_unseen_stage_is_missing(ledger):
    assert ledger.status("arm256", "k1")[0] == MISSING


def test_successful_stage_is_complete_and_records_timing(ledger, tmp_path):
    art = _artifact(tmp_path)
    with ledger.stage("arm256", "k1", artifacts=[art], n_detections=1234):
        pass
    state, _ = ledger.status("arm256", "k1")
    assert state == COMPLETE
    rec = ledger.latest("arm256")
    assert rec["n_detections"] == 1234 and rec["wall_seconds"] >= 0
    assert rec["artifacts"][str(art)] == file_digest(art)


def test_artifact_without_a_ledger_entry_is_not_complete(ledger, tmp_path):
    """The exact defect: the GeoPackage exists, so the old runner called it done."""
    _artifact(tmp_path)
    assert ledger.status("arm256", "k1")[0] == MISSING


def test_crash_between_writes_leaves_incomplete_not_complete(ledger, tmp_path):
    art = _artifact(tmp_path)
    with pytest.raises(RuntimeError):
        with ledger.stage("arm256", "k1", artifacts=[art]):
            raise RuntimeError("killed mid-stage")
    state, detail = ledger.status("arm256", "k1")
    assert state == FAILED and "killed mid-stage" in detail
    assert art.exists(), "the artifact is still there, which is why existence proves nothing"


def test_running_marker_survives_a_hard_kill(ledger, tmp_path):
    """Simulate the process vanishing after __enter__: the entry says running, and the next run
    must treat that as incomplete rather than never-attempted."""
    stage = ledger.stage("arm256", "k1", artifacts=[_artifact(tmp_path)])
    stage.__enter__()                                   # no __exit__ - the process died
    state, _ = ledger.status("arm256", "k1")
    assert state == INCOMPLETE


def test_declared_artifact_that_was_never_produced_raises(ledger, tmp_path):
    with pytest.raises(FileNotFoundError, match="were not produced"):
        with ledger.stage("arm256", "k1", artifacts=[tmp_path / "never.gpkg"]):
            pass
    assert ledger.status("arm256", "k1")[0] == INCOMPLETE


def test_changed_config_is_stale(ledger, tmp_path):
    art = _artifact(tmp_path)
    with ledger.stage("arm256", "k1", artifacts=[art]):
        pass
    assert ledger.status("arm256", "k2")[0] == STALE


def test_artifact_modified_on_disk_is_stale(ledger, tmp_path):
    art = _artifact(tmp_path)
    with ledger.stage("arm256", "k1", artifacts=[art]):
        pass
    art.write_bytes(b"something else")
    state, detail = ledger.status("arm256", "k1")
    assert state == STALE and "changed on disk" in detail


def test_deleted_artifact_is_incomplete(ledger, tmp_path):
    art = _artifact(tmp_path)
    with ledger.stage("arm256", "k1", artifacts=[art]):
        pass
    art.unlink()
    assert ledger.status("arm256", "k1")[0] == INCOMPLETE


def test_upstream_change_invalidates_the_downstream_stage(ledger, tmp_path):
    """Dependency-aware invalidation: re-running one arm must re-fuse, not reuse the old fuse."""
    arm = _artifact(tmp_path, "arm.gpkg")
    fused = _artifact(tmp_path, "fused.gpkg")
    with ledger.stage("arm256", "armkey1", artifacts=[arm]):
        pass
    with ledger.stage("fuse", "fusekey", artifacts=[fused],
                      depends_on={"arm256": "armkey1"}):
        pass
    assert ledger.is_complete("fuse", "fusekey", depends_on={"arm256": "armkey1"})
    state, detail = ledger.status("fuse", "fusekey", depends_on={"arm256": "armkey2"})
    assert state == STALE and "upstream" in detail


def test_reexecuting_an_arm_with_unchanged_config_still_invalidates_the_fuse(ledger, tmp_path):
    """The case the key-only dependency missed, and the one a resumable campaign actually hits.

    ``stage_key`` hashes the *configuration*, so an arm re-run with the same code and inputs keeps
    its key. Keying the fuse on that alone let it skip while its input had been regenerated
    underneath it. Found for real on 2026-08-03: a smoke run of ``run_multiscale_arms.py`` deleted
    one arm's layer, the arm correctly re-ran, and the fuse reported "up to date" — leaving a fused
    catalog descending from an arm layer that no longer existed on disk.
    """
    arm = _artifact(tmp_path, "arm.gpkg", body=b"first")
    fused = _artifact(tmp_path, "fused.gpkg")
    with ledger.stage("arm:256", "armkey", artifacts=[arm]):
        pass
    dep = ledger.dependency_value("arm:256", "armkey")
    with ledger.stage("fuse", "fusekey", artifacts=[fused], depends_on={"256": dep}):
        pass
    assert ledger.is_complete("fuse", "fusekey", depends_on={"256": dep})

    # the arm re-runs with the SAME key and writes different bytes (gpkg is not byte-deterministic)
    arm.write_bytes(b"second")
    with ledger.stage("arm:256", "armkey", artifacts=[arm]):
        pass
    dep2 = ledger.dependency_value("arm:256", "armkey")
    assert dep2 != dep, "the dependency value must change when the arm is re-executed"
    state, detail = ledger.status("fuse", "fusekey", depends_on={"256": dep2})
    assert state == STALE and "upstream" in detail


def test_dependency_value_is_stable_when_the_arm_is_skipped(ledger, tmp_path):
    """The other half: an untouched arm must not make the fuse re-run every time."""
    arm = _artifact(tmp_path, "arm.gpkg", body=b"same")
    with ledger.stage("arm:256", "armkey", artifacts=[arm]):
        pass
    first = ledger.dependency_value("arm:256", "armkey")
    assert ledger.dependency_value("arm:256", "armkey") == first
    assert first.startswith("armkey") and len(first) > len("armkey")


def test_history_is_appended_never_overwritten(ledger, tmp_path):
    art = _artifact(tmp_path)
    for i in range(3):
        with ledger.stage("arm256", f"k{i}", artifacts=[art], wall_note=i):
            pass
    assert len(ledger.entries("arm256")) == 6           # a running + a final entry each
    finals = [e for e in ledger.entries("arm256") if e.get("state") == COMPLETE]
    assert [e["wall_note"] for e in finals] == [0, 1, 2], "prior timing history was lost"


def test_a_torn_final_line_does_not_break_reading(ledger, tmp_path):
    art = _artifact(tmp_path)
    with ledger.stage("arm256", "k1", artifacts=[art]):
        pass
    with open(ledger.path, "a") as fh:
        fh.write('{"stage": "arm512", "sta')                # killed mid-write
    assert ledger.status("arm256", "k1")[0] == COMPLETE


def test_stage_key_covers_the_code_and_inputs():
    a = stage_key(raster="scene.tif", slice_size=256, model="w.pt")
    b = stage_key(raster="scene.tif", slice_size=512, model="w.pt")
    c = stage_key(raster="scene.tif", slice_size=256, model="other.pt")
    assert a != b and a != c
    assert a == stage_key(model="w.pt", slice_size=256, raster="scene.tif")   # order-independent


# --- G3-09: the wrapper's own artifacts and memory bound -----------------------------------------

def test_multiscale_wrapper_writes_only_the_fused_deliverable(tmp_path, monkeypatch):
    """Per-arm NMS layers were being written despite the wrapper promising one fused file."""
    import geopandas as gpd
    import shapely

    def fake_gsp(in_raster, detection_model, slice_size, sr_factor, output_dir, output_format,
                 save_prenms, save_bbox, postprocess, save_nms=True, **kw):
        assert save_nms is False, "the wrapper still asks for per-arm NMS layers"
        g = gpd.GeoDataFrame({"score": [0.7], "category_id": 0, "category_name": "boulder",
                              "is_within_slice": True},
                             geometry=[shapely.box(0, 0, 8, 8)])
        return g, g

    monkeypatch.setattr(P, "get_sliced_prediction", fake_gsp)
    monkeypatch.setattr(P.raster_metadata, "get_resolution", lambda p: (1.0, 1.0))
    monkeypatch.setattr(P, "_check_in_memory_budget", lambda *a, **k: None)

    P.get_multiscale_prediction(tmp_path / "scene.tif", None, output_dir=tmp_path)
    layers = sorted(p.name for p in tmp_path.glob("*.gpkg"))
    assert layers == ["scene-predictions-fused-bands-engulf70-512-256-256sr2-mask.gpkg"], layers


def _raster(path, size):
    with rio.open(path, "w", driver="GTiff", height=size, width=size, count=1, dtype="uint8",
                  crs="EPSG:32631", transform=from_origin(0, size, 1, 1)) as dst:
        dst.write(np.zeros((1, size, size), dtype=np.uint8))
    return path


def test_full_scene_in_memory_run_is_refused_with_the_alternative_named(tmp_path):
    scene = _raster(tmp_path / "big.tif", 600)
    with pytest.raises(MemoryError, match="run_multiscale_arms"):
        P._check_in_memory_budget(scene, (512, 256), max_px=1000, allow=False)
    P._check_in_memory_budget(scene, (512, 256), max_px=1000, allow=True)     # explicit override
    P._check_in_memory_budget(scene, (512, 256), max_px=10 ** 9, allow=False)  # a crop is fine


def test_the_default_limit_admits_a_crop_and_refuses_a_scene():
    assert P.MAX_IN_MEMORY_PX > 4_200_000            # the tracked 2048x2048 dev crop
    assert P.MAX_IN_MEMORY_PX < 1_932_385_780        # the HiRISE scene


# --- partial ladders must not be fused under a production name --------------------------------------
#
# The 512 arm is deferred to a larger GPU (its densest HiRISE slices spill even at batch 1 on the
# 8 GiB laptop card), so the laptop runs `{256, 256:2}`. The fused output is named
# `-fused-production-mask.gpkg` and manifested `preset=goal3_production`, neither of which encodes
# the arm set -- so a subset fuse would claim the shipped three-arm rule while the large-boulder net
# is absent. That is silent, and it is the failure class this review keeps finding.

def _runner():
    import importlib.util
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_multiscale_arms.py"
    spec = importlib.util.spec_from_file_location("run_multiscale_arms", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_multiscale_arms"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_normalise_arm_puts_strings_ints_and_pairs_in_one_form():
    r = _runner()
    assert r.normalise_arm("256") == (256, 1)
    assert r.normalise_arm("256:2") == (256, 2)
    assert r.normalise_arm(512) == (512, 1)          # a preset `scales` entry
    assert r.normalise_arm((256, 2)) == (256, 2)
    assert r.normalise_arm([256, 2]) == (256, 2)     # json round-trips the tuple to a list


def test_the_production_arm_set_is_exactly_what_the_fuse_will_accept():
    """If GOAL3_PRODUCTION's scales change, this is the test that notices the guard went stale."""
    r = _runner()
    from YOLOv8BeyondEarth import presets
    assert {r.normalise_arm(x) for x in presets.GOAL3_PRODUCTION["scales"]} == {
        (512, 1), (256, 1), (256, 2)}


def test_fusing_a_subset_of_the_production_arms_is_refused(tmp_path, monkeypatch):
    r = _runner()
    monkeypatch.setattr(r, "run_one_arm", lambda args: None)
    # every arm already complete, so the run reaches the fuse decision without touching a GPU
    monkeypatch.setattr(r.StageLedger, "status", lambda self, *a, **k: (COMPLETE, "skip"))
    monkeypatch.setattr(r.presets, "file_identity", lambda p: {"sha256": "x"})
    args = SimpleNamespace(
        arms=["256", "256:2"], arm=None, raster=str(tmp_path / "s.tif"), weights="w.pt",
        out_dir=str(tmp_path / "out"), conf=0.10, inference_size=1024, device="cpu",
        on_conflict="error", on_max_det="error", max_det=None, no_fuse=False)
    with pytest.raises(SystemExit, match="refusing to fuse"):
        r.orchestrate(args)


def test_no_fuse_returns_before_the_fuse_and_says_so(tmp_path, monkeypatch, capsys):
    r = _runner()
    monkeypatch.setattr(r.StageLedger, "status", lambda self, *a, **k: (COMPLETE, "skip"))
    monkeypatch.setattr(r.presets, "file_identity", lambda p: {"sha256": "x"})
    args = SimpleNamespace(
        arms=["256", "256:2"], arm=None, raster=str(tmp_path / "s.tif"), weights="w.pt",
        out_dir=str(tmp_path / "out"), conf=0.10, inference_size=1024, device="cpu",
        on_conflict="error", on_max_det="error", max_det=None, no_fuse=True)
    r.orchestrate(args)                                     # no SystemExit, no fuse
    assert "--no-fuse" in capsys.readouterr().out
    assert not list(pathlib.Path(args.out_dir).glob("*fused*"))


def test_the_full_arm_set_is_not_refused(tmp_path, monkeypatch):
    """The guard must not block the real ladder -- it fails past the check, not at it."""
    r = _runner()
    monkeypatch.setattr(r.StageLedger, "status", lambda self, *a, **k: (COMPLETE, "skip"))
    monkeypatch.setattr(r.presets, "file_identity", lambda p: {"sha256": "x"})
    args = SimpleNamespace(
        arms=["512", "256", "256:2"], arm=None, raster=str(tmp_path / "s.tif"), weights="w.pt",
        out_dir=str(tmp_path / "out"), conf=0.10, inference_size=1024, device="cpu",
        on_conflict="error", on_max_det="error", max_det=None, no_fuse=False)
    r.orchestrate(args)     # all stages COMPLETE -> returns at the fuse skip, never reaches the guard


# --- the code slot is content, not a commit -------------------------------------------------------
#
# `stage_key` used to fold in the repo HEAD, which identified which commit you were on rather than
# what code would run, and it failed in both directions. Measured 2026-08-18: six commits that never
# touched the predict path invalidated the banked NAC arms; and on a dirty tree a *second, different*
# uncommitted edit left `(HEAD, True)` unchanged, so the stage was skipped and a stale answer reused.

def test_moving_head_does_not_rekey_when_the_package_is_unchanged(monkeypatch):
    """A doc or test commit must not invalidate hours of banked arms."""
    from YOLOv8BeyondEarth import presets as P

    def ident(sha):
        return {"git_sha": sha, "git_dirty": False, "source_sha256": "same-content-digest"}

    monkeypatch.setattr(P, "code_identity", lambda: ident("commit-one"))
    before = stage_key(raster="scene.tif", slice_size=256)
    monkeypatch.setattr(P, "code_identity", lambda: ident("commit-two"))
    assert stage_key(raster="scene.tif", slice_size=256) == before


def test_changed_package_content_does_rekey(monkeypatch):
    """The protection G3-08 asked for has to survive the change: different code, different key."""
    from YOLOv8BeyondEarth import presets as P
    monkeypatch.setattr(P, "code_identity",
                        lambda: {"git_sha": "same", "source_sha256": "digest-a"})
    a = stage_key(raster="scene.tif", slice_size=256)
    monkeypatch.setattr(P, "code_identity",
                        lambda: {"git_sha": "same", "source_sha256": "digest-b"})
    assert stage_key(raster="scene.tif", slice_size=256) != a


def test_two_different_dirty_edits_are_distinguishable(monkeypatch):
    """The old hole: identical (HEAD, dirty=True) for two genuinely different working trees."""
    from YOLOv8BeyondEarth import presets as P
    keys = set()
    for digest in ("edit-one", "edit-two"):
        monkeypatch.setattr(P, "code_identity",
                            lambda d=digest: {"git_sha": "H", "git_dirty": True,
                                              "source_sha256": d})
        keys.add(stage_key(raster="scene.tif"))
    assert len(keys) == 2


def test_a_missing_source_digest_falls_back_to_the_git_sha(monkeypatch):
    """An unreadable tree must still key on something, not collapse to None for every stage."""
    from YOLOv8BeyondEarth import presets as P
    monkeypatch.setattr(P, "code_identity", lambda: {"git_sha": "abc", "source_error": "boom"})
    a = stage_key(raster="scene.tif")
    monkeypatch.setattr(P, "code_identity", lambda: {"git_sha": "def", "source_error": "boom"})
    assert stage_key(raster="scene.tif") != a


# --- source_digest itself -------------------------------------------------------------------------

def test_source_digest_is_deterministic_and_content_sensitive(tmp_path):
    from YOLOv8BeyondEarth.presets import source_digest
    pkg = tmp_path / "pkg"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "a.py").write_bytes(b"x = 1\n")
    (pkg / "sub" / "b.py").write_bytes(b"y = 2\n")
    first = source_digest(pkg)
    assert first == source_digest(pkg)                       # deterministic
    (pkg / "sub" / "b.py").write_bytes(b"y = 3\n")
    assert source_digest(pkg) != first                       # content change registers
    assert len(first) == 64


def test_source_digest_ignores_line_endings_but_not_renames(tmp_path):
    """CRLF vs LF is not a code difference; a rename is.

    Without normalisation a Windows checkout under core.autocrlf and a Linux checkout of the same
    commit would disagree, which would defeat the cross-platform comparison this digest is for.
    """
    from YOLOv8BeyondEarth.presets import source_digest
    lf, crlf = tmp_path / "lf", tmp_path / "crlf"
    lf.mkdir(); crlf.mkdir()
    (lf / "a.py").write_bytes(b"x = 1\ny = 2\n")
    (crlf / "a.py").write_bytes(b"x = 1\r\ny = 2\r\n")
    assert source_digest(lf) == source_digest(crlf)

    renamed = tmp_path / "renamed"
    renamed.mkdir()
    (renamed / "b.py").write_bytes(b"x = 1\ny = 2\n")
    assert source_digest(renamed) != source_digest(lf)


def test_source_digest_of_the_real_package_is_recorded_in_code_identity():
    from YOLOv8BeyondEarth import presets as P
    info = P.code_identity()
    assert len(info.get("source_sha256", "")) == 64
    assert info["source_sha256"] == P.source_digest()


# --- a deployment that is not a checkout (the Sherlock tarball) ------------------------------------

def test_git_is_asked_with_cwd_not_dash_C(monkeypatch):
    """el7's git 1.8.3 predates `git -C` (exit 129), so every Sherlock manifest said git_sha: None."""
    from YOLOv8BeyondEarth import presets as PR
    seen = []
    real = PR.subprocess.run

    def spy(cmd, *a, **k):
        seen.append((cmd, k.get("cwd")))
        return real(cmd, *a, **k)

    monkeypatch.setattr(PR.subprocess, "run", spy)
    PR.code_identity()
    git_calls = [(c, cwd) for c, cwd in seen if c and c[0] == "git"]
    assert git_calls and all("-C" not in c and cwd is not None for c, cwd in git_calls)


def _stamp(tmp_path, source_sha256):
    (tmp_path / "CODE_STAMP.json").write_text(json.dumps(
        {"git_sha": "f" * 40, "git_dirty": False, "source_sha256": source_sha256}))
    return tmp_path


def test_a_tarball_names_its_commit_through_the_pack_time_stamp(tmp_path):
    from YOLOv8BeyondEarth import presets as PR
    info = PR.code_identity(repo=_stamp(tmp_path, PR.source_digest()))   # tmp_path is no checkout
    assert info["git_sha"] == "f" * 40 and info["git_sha_from"] == "CODE_STAMP.json"
    assert info["stamp_matches_source"] is True


def test_a_stamp_beside_edited_files_is_recorded_but_not_believed(tmp_path):
    from YOLOv8BeyondEarth import presets as PR
    info = PR.code_identity(repo=_stamp(tmp_path, "0" * 64))
    assert info["git_sha"] is None and info["stamp_matches_source"] is False
    assert info["stamp"]["git_sha"] == "f" * 40, "kept for the record, not promoted"
