"""The campaign's acceptance step (``scripts/campaign.py verify``) and its scene manifest.

``verify`` is what lets a scene's catalog leave the cluster, so its refusals are the contract: a
truncated arm is a lower bound rather than a catalog, a catalog made on some other raster than the
prepared one is not this scene's, and a layer whose arm's latest ledger entry is not complete is not
what the last run produced. All of them must fail the scene, not warn. The rest pin what the job
scripts rely on: ``count`` for the array range, a ``status`` that only reads, and ``detect``'s exit
status and CPU report.
"""
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from YOLOv8BeyondEarth.runstate import COMPLETE, FAILED, StageLedger, file_digest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


campaign = _load("campaign")
prep_cli = _load("prep_pds_scene")

STEM = "ESP_028537_2270_RED-predictions-ct-010-ss-256-is-1024-ov-020-downscaled-mask-nms"


def _complete(odir, stem, stage="arm:256", base_key="b", **fields):
    """The ledger entry a finished arm leaves behind: complete, naming its layer and that digest."""
    layer = odir / f"{stem}.gpkg"
    StageLedger(odir / "run_ledger.jsonl").append(
        {"stage": stage, "state": COMPLETE, "base_key": base_key, "max_det": 1100, "batch_size": 8,
         "finished_utc": datetime.now(timezone.utc).isoformat(),
         "artifacts": {str(layer): file_digest(layer)}, **fields})


@pytest.fixture
def scene(tmp_path):
    manifest = tmp_path / "scenes.csv"
    manifest.write_text("ObsId,ProductId,JP2 URL,LBL URL\nESP_028537_2270,ESP_028537_2270_RED,,\n")
    scenes, out = tmp_path / "scenes", tmp_path / "out"
    sdir, odir = scenes / "ESP_028537_2270", out / "ESP_028537_2270"
    sdir.mkdir(parents=True)
    odir.mkdir(parents=True)
    raster = sdir / "ESP_028537_2270_RED.tif"
    raster.write_bytes(b"pretend raster")
    (sdir / "ESP_028537_2270_RED.prep.json").write_text(json.dumps({
        "rule_expr": "min((DN + 2) >> 2, 255)", "frame": "PDS label",
        "inputs": {"jp2": {"sha256": "j" * 64}},
        "output": {"path": raster.name, "bytes": raster.stat().st_size,
                   "sha256": file_digest(raster)}}))
    (odir / f"{STEM}.gpkg").write_bytes(b"pretend layer")

    def write_manifest(truncation=None, raster_sha=None, config=None, ledger=True):
        (odir / f"{STEM}-manifest.json").write_text(json.dumps({
            "config_hash": "abc", "truncation": {} if truncation is None else truncation,
            "outputs": {"n_after_nms": 12}, "config": config or {"max_det": 1100},
            "inputs": {"raster": {"path": str(raster), "bytes": raster.stat().st_size,
                                  "sha256": raster_sha or file_digest(raster)}},
            "code": {"git_sha": "f" * 40, "source_sha256": "s" * 64}}))
        if ledger:                            # what detect leaves: the arm's entry, complete
            _complete(odir, STEM)
    return manifest, scenes, out, write_manifest


def test_a_clean_scene_verifies_and_gets_checksums(scene):
    manifest, scenes, out, write_manifest = scene
    write_manifest()
    paths = campaign.scene_paths(manifest, 0, scenes, out)
    ok, report = campaign.verify(paths)
    assert ok, report["problems"]
    sums = (out / "ESP_028537_2270" / "SHA256SUMS").read_text().splitlines()
    assert any(line.endswith(f"  {STEM}.gpkg") and len(line.split()[0]) == 64 for line in sums)
    hand = json.loads((out / "ESP_028537_2270" / "handoff.json").read_text())
    assert hand["ok"] and hand["raster"]["jp2_sha256"] == "j" * 64
    assert hand["layers"][0]["n_features"] == 12


def test_a_truncated_arm_fails_the_scene(scene):
    manifest, scenes, out, write_manifest = scene
    write_manifest(truncation={"slices_at_max_det": 3})
    ok, report = campaign.verify(campaign.scene_paths(manifest, 0, scenes, out))
    assert not ok and "truncation" in report["problems"][0]
    assert "--on-max-det escalate" in report["problems"][0]


def test_an_nms_time_limit_layer_names_the_remedy_that_works(scene):
    """Escalation accepts it as complete, since no cap fixes it, so 're-run detect' changed nothing."""
    manifest, scenes, out, write_manifest = scene
    write_manifest(truncation={"nms_time_limit_batches": 2, "slices_zeroed_after_nms_time_limit": 5},
                   config={"max_det": 2200, "batch_size": 4})
    ok, report = campaign.verify(campaign.scene_paths(manifest, 0, scenes, out))
    (problem,) = report["problems"]
    assert not ok and "No cap fixes that" in problem and "up to 5 slices" in problem
    assert "remove the arm:256 lines from run_ledger.jsonl" in problem
    # the cap travels with the retry: once the arm's entries are gone, the runner restarts at MAX_DET
    assert "BATCH_SIZE=2 and MAX_DET=2200" in problem


@pytest.mark.parametrize("truncation, batch", [
    ({"nms_time_limit_batches": 2}, 8),                                       # no empty slice
    ({"nms_time_limit_batches": 1, "slices_zeroed_after_nms_time_limit": 1}, 1),   # batch 1
])
def test_an_nms_time_limit_that_cannot_have_lost_a_slice_does_not_fail_the_scene(scene, truncation,
                                                                                 batch):
    """An abandoned slice comes back empty, and at batch 1 the one slice is written before the limit
    is checked (ultralytics/utils/nms.py), so neither record lost anything (pilot-fix review)."""
    manifest, scenes, out, write_manifest = scene
    write_manifest(truncation=truncation, config={"max_det": 8800, "batch_size": batch})
    ok, report = campaign.verify(campaign.scene_paths(manifest, 0, scenes, out))
    assert ok, report["problems"]
    assert report["layers"][0]["truncation"] == truncation, "the raw record stays in the hand-off"


def test_a_catalog_from_another_raster_fails_the_scene(scene):
    manifest, scenes, out, write_manifest = scene
    write_manifest(raster_sha="0" * 64)
    ok, report = campaign.verify(campaign.scene_paths(manifest, 0, scenes, out))
    assert not ok and "not the prepared" in report["problems"][0]


def test_an_escalated_arm_says_so_in_the_handoff(scene):
    """The catalog is the uncapped one, but how it got there -- which cap bound, what batch the final
    attempt ran at -- travels with it rather than staying in a ledger on the cluster."""
    manifest, scenes, out, write_manifest = scene
    write_manifest(config={"max_det": 2200, "batch_size": 4}, ledger=False)
    odir = out / "ESP_028537_2270"
    layer = odir / f"{STEM}.gpkg"
    ledger = StageLedger(odir / "run_ledger.jsonl")
    ledger.append({"stage": "arm:256", "state": "running"})
    ledger.append({"stage": "arm:256", "state": "failed", "base_key": "b", "max_det": 1100,
                   "batch_size": 8, "stopped": "max_det", "bound": {"worst": 1012},
                   "wall_seconds": 61.0})
    ledger.append({"stage": "arm:256", "state": "running"})
    ledger.append({"stage": "arm:256", "state": "complete", "base_key": "b", "max_det": 2200,
                   "batch_size": 4, "wall_seconds": 3600.0,
                   "artifacts": {str(layer): file_digest(layer)}})
    ok, report = campaign.verify(campaign.scene_paths(manifest, 0, scenes, out))
    assert ok, report["problems"]
    rec = report["layers"][0]
    assert (rec["max_det"], rec["batch_size"]) == (2200, 4)
    assert [(a["state"], a["max_det"], a["batch_size"], a["stopped"], a["worst_slice"])
            for a in rec["attempts"]] == [("failed", 1100, 8, "max_det", 1012),
                                          ("complete", 2200, 4, None, None)]


def test_the_handoff_lists_only_the_attempts_of_the_delivered_configuration(scene):
    """'attempts' used to mix in every finished entry of the stage: other code, rasters or weights,
    and runs from before escalation recorded a base_key. None of those produced this layer."""
    manifest, scenes, out, write_manifest = scene
    write_manifest(ledger=False)
    odir = out / "ESP_028537_2270"
    ledger = StageLedger(odir / "run_ledger.jsonl")
    ledger.append({"stage": "arm:256", "state": "complete", "max_det": 600, "batch_size": 8})
    ledger.append({"stage": "arm:256", "state": "failed", "base_key": "old", "max_det": 1100,
                   "batch_size": 8, "stopped": "max_det"})
    ledger.append({"stage": "arm:256", "state": "complete", "base_key": "old", "max_det": 2200,
                   "batch_size": 4})
    _complete(odir, STEM, base_key="new")
    ok, report = campaign.verify(campaign.scene_paths(manifest, 0, scenes, out))
    assert ok, report["problems"]
    assert [(a["state"], a["max_det"]) for a in report["layers"][0]["attempts"]] == \
        [("complete", 1100)]


def test_a_failed_rerun_is_not_hidden_by_the_last_handoff(scene, monkeypatch, capsys):
    """A re-run that failed before verify could run left the previous ok:true hand-off in place:
    status said 'verified', and a standalone verify passed the previous run's layer."""
    class Clock(datetime):                    # campaign's clock, set by hand: no resolution race
        t = datetime(2100, 1, 1, tzinfo=timezone.utc)

        @classmethod
        def now(cls, tz=None):
            return cls.t

    monkeypatch.setattr(campaign, "datetime", Clock)
    manifest, scenes, out, write_manifest = scene
    write_manifest()
    odir = out / "ESP_028537_2270"
    paths = campaign.scene_paths(manifest, 0, scenes, out)
    assert campaign.verify(paths)[0]
    rerun = "2100-01-01T01:00:00+00:00"       # after that verify
    StageLedger(odir / "run_ledger.jsonl").append({"stage": "arm:256", "state": "running",
                                                   "started_utc": rerun})
    StageLedger(odir / "run_ledger.jsonl").append(
        {"stage": "arm:256", "state": FAILED, "base_key": "b2", "finished_utc": rerun,
         "error": "CalledProcessError: the re-run's arm crashed"})
    campaign.status(manifest, scenes, out)
    assert "VERIFY-STALE" in capsys.readouterr().out
    monkeypatch.setattr(Clock, "t", datetime(2100, 1, 1, 2, tzinfo=timezone.utc))
    ok, report = campaign.verify(paths)
    assert not ok and "latest attempt is 'failed'" in report["problems"][0]
    assert report["layers"] == []
    assert report["not_delivered"] == [{"layer": f"{STEM}.gpkg", "stage": "arm:256",
                                        "why": "the latest attempt of arm:256 is 'failed'"}]
    assert STEM not in (odir / "SHA256SUMS").read_text()
    campaign.status(manifest, scenes, out)
    assert "VERIFY-FAILED" in capsys.readouterr().out


def test_a_killed_run_reads_as_unfinished_and_does_not_verify(scene, capsys):
    """A walltime kill leaves only a `running` marker; status used to report the scene as 'prepped'."""
    manifest, scenes, out, write_manifest = scene
    write_manifest()
    StageLedger(out / "ESP_028537_2270" / "run_ledger.jsonl").append(
        {"stage": "arm:256", "state": "running"})
    campaign.status(manifest, scenes, out)
    assert "unfinished" in capsys.readouterr().out
    ok, report = campaign.verify(campaign.scene_paths(manifest, 0, scenes, out))
    assert not ok and "'running' (killed, or still running)" in report["problems"][0]


def test_layers_of_arms_outside_the_handoff_are_listed_not_delivered(scene):
    """A configuration-2 trial left a 512 layer in the same OUT dir; a configuration-1 hand-off
    must not checksum and ship it as its own."""
    manifest, scenes, out, write_manifest = scene
    write_manifest()
    odir = out / "ESP_028537_2270"
    stem512 = STEM.replace("-ss-256-", "-ss-512-")
    (odir / f"{stem512}.gpkg").write_bytes(b"pretend 512 layer")
    (odir / f"{stem512}-manifest.json").write_text((odir / f"{STEM}-manifest.json").read_text())
    _complete(odir, stem512, stage="arm:512")
    ok, report = campaign.verify(campaign.scene_paths(manifest, 0, scenes, out), arms=["256"])
    assert ok, report["problems"]
    assert [r["layer"] for r in report["layers"]] == [f"{STEM}.gpkg"]
    assert report["not_delivered"] == [{"layer": f"{stem512}.gpkg", "stage": "arm:512",
                                        "why": "arm:512 is not one of this hand-off's arms"}]
    assert "-ss-512-" not in (odir / "SHA256SUMS").read_text()


def test_a_layer_rewritten_since_its_ledger_entry_fails_the_scene(scene):
    manifest, scenes, out, write_manifest = scene
    write_manifest()
    (out / "ESP_028537_2270" / f"{STEM}.gpkg").write_bytes(b"not what the ledger recorded")
    ok, report = campaign.verify(campaign.scene_paths(manifest, 0, scenes, out))
    assert not ok and "changed on disk since its ledger entry" in report["problems"][0]


def test_detect_escalates_by_default(scene, monkeypatch):
    manifest, scenes, out, _ = scene
    seen = []
    monkeypatch.setattr(campaign.subprocess, "run",
                        lambda cmd, **kw: seen.append(cmd) or SimpleNamespace(returncode=0))
    args = SimpleNamespace(
        weights="w.pt", arms=["512", "256"], max_det=1100, on_max_det="escalate", batch_size=None,
        device="cpu")
    campaign.detect(args, campaign.scene_paths(manifest, 0, scenes, out))
    (cmd,) = seen
    assert cmd[cmd.index("--on-max-det") + 1] == "escalate"
    assert "--batch-size" not in cmd and "--no-fuse" in cmd


@pytest.mark.parametrize("returncode, status", [(1, 1), (-9, 137)])
def test_detect_exits_with_the_runners_status(scene, monkeypatch, returncode, status):
    """Not a generic 1: campaign_detect.sbatch tells 137 (SIGKILL, on the cluster the host-RAM OOM
    killer) from a crash, and only the first needs a bigger --mem."""
    manifest, scenes, out, _ = scene
    monkeypatch.setattr(campaign.subprocess, "run",
                        lambda cmd, **kw: SimpleNamespace(returncode=returncode))
    args = SimpleNamespace(
        weights="w.pt", arms=["256"], max_det=1100, on_max_det="escalate", batch_size=None,
        device="cpu")
    with pytest.raises(SystemExit) as stop:
        campaign.detect(args, campaign.scene_paths(manifest, 0, scenes, out))
    assert stop.value.code == status


def test_the_cpus_are_read_from_the_affinity_mask_not_the_node(monkeypatch):
    """os.cpu_count() counts the node, so a Slurm step pinned to one CPU got eight worker threads."""
    from YOLOv8BeyondEarth import predict as P
    monkeypatch.setattr(P.os, "sched_getaffinity", lambda pid: {3}, raising=False)
    monkeypatch.setattr(P.os, "cpu_count", lambda: 64)
    assert P._available_cpus() == campaign._cpus() == 1
    monkeypatch.delattr(P.os, "sched_getaffinity")       # Windows and macOS have none
    assert P._available_cpus() == campaign._cpus() == 64


# --- what the job scripts run before and around detect -------------------------------------------

def test_count_is_the_manifest_row_count_for_the_array_range(tmp_path, capsys):
    """`--array=0-62` launched 63 tasks for a 62-row manifest. The range is derived from this now."""
    m = tmp_path / "m.csv"
    m.write_text("obs_id\nESP_011111_1111\nESP_022222_2222\nESP_033333_3333\n\n")
    assert campaign.main(["count", "--manifest", str(m)]) == 0
    assert capsys.readouterr().out.strip() == "3"
    prep_cli.manifest_row(m, 2)
    with pytest.raises(SystemExit):           # so 0..N-1 is the whole manifest, and no more
        prep_cli.manifest_row(m, 3)


def test_status_only_reads(scene, capsys):
    """Pointed at a real --out, it left an empty directory for every row (StageLedger mkdirs)."""
    manifest, scenes, out, _ = scene
    manifest.write_text(manifest.read_text() + "ESP_011111_1111,ESP_011111_1111_RED,,\n")
    campaign.status(manifest, scenes, out)
    assert not (out / "ESP_011111_1111").exists()
    assert "ESP_011111_1111  pending" in capsys.readouterr().out


def test_manifest_columns_are_matched_loosely_and_urls_derived(tmp_path):
    a = tmp_path / "a.csv"
    a.write_text("obs_id,product_id,jp2_url,lbl_url\nPSP_007492_2265,PSP_007492_2265_RED,,\n")
    b = tmp_path / "b.csv"
    b.write_text("ObsId,ProductId,JP2 URL,LBL URL\nPSP_007492_2265,,,\n")
    ra, rb = prep_cli.manifest_row(a, 0), prep_cli.manifest_row(b, 0)
    assert ra == rb
    assert ra["lbl_url"].endswith("/PSP/ORB_007400_007499/PSP_007492_2265/PSP_007492_2265_RED.LBL")
    with pytest.raises(SystemExit):
        prep_cli.manifest_row(a, 1)
