"""The campaign's acceptance step (``scripts/campaign.py verify``) and its scene manifest.

``verify`` is what lets a scene's catalog leave the cluster, so its refusals are the contract: a
truncated arm is a lower bound rather than a catalog, and a catalog made on some other raster than the
prepared one is not this scene's. Both must fail the scene, not warn.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

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
    from YOLOv8BeyondEarth.runstate import file_digest
    (sdir / "ESP_028537_2270_RED.prep.json").write_text(json.dumps({
        "rule_expr": "min((DN + 2) >> 2, 255)", "frame": "PDS label",
        "inputs": {"jp2": {"sha256": "j" * 64}},
        "output": {"path": raster.name, "bytes": raster.stat().st_size,
                   "sha256": file_digest(raster)}}))
    (odir / f"{STEM}.gpkg").write_bytes(b"pretend layer")

    def write_manifest(truncation=None, raster_sha=None):
        (odir / f"{STEM}-manifest.json").write_text(json.dumps({
            "config_hash": "abc", "truncation": {} if truncation is None else truncation,
            "outputs": {"n_after_nms": 12}, "config": {"max_det": 1100},
            "inputs": {"raster": {"path": str(raster), "bytes": raster.stat().st_size,
                                  "sha256": raster_sha or file_digest(raster)}},
            "code": {"git_sha": "f" * 40, "source_sha256": "s" * 64}}))
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


def test_a_catalog_from_another_raster_fails_the_scene(scene):
    manifest, scenes, out, write_manifest = scene
    write_manifest(raster_sha="0" * 64)
    ok, report = campaign.verify(campaign.scene_paths(manifest, 0, scenes, out))
    assert not ok and "not the prepared" in report["problems"][0]


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
