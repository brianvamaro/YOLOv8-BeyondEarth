"""`run_multiscale_arms.py --fuse-only`: assemble a ladder from layers already on disk.

`orchestrate` can only fuse arms it produced itself, because an arm's ledger entry is keyed on the
machine's own code identity. On a heterogeneous ladder -- which is what the 2026-08-04 "defer the 512
arm to Sherlock" decision created -- that is wrong in a way nothing reports: fuse on the laptop and it
tries to recompute the 512 arm that does not fit there; fuse on the cluster and it recomputes the
load-bearing 256 and SR arms, silently relocating them to the other stack.

So `--fuse-only` validates from the manifests instead. These tests pin what it refuses, because a
validator that waves through an incoherent set is worse than no validator: the output would be a
production-named catalog nobody could reconstruct.
"""
import importlib.util
import json
import sys
from types import SimpleNamespace

import pytest

from YOLOv8BeyondEarth import presets

_SCRIPTS = None


def _runner():
    global _SCRIPTS
    import pathlib
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_multiscale_arms.py"
    spec = importlib.util.spec_from_file_location("run_multiscale_arms", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_multiscale_arms"] = mod
    spec.loader.exec_module(mod)
    _SCRIPTS = mod
    return mod


PRODUCTION_ARMS = ["512", "256", "256:2"]


def _config(**over):
    cfg = {"confidence_threshold": 0.1, "inference_size": 1024, "min_area_threshold": 6,
           "downscale_pred": True, "postprocess": True, "postprocess_match_threshold": 0.2,
           "postprocess_class_agnostic": False, "contour_method": "skimage",
           "overlap_height_ratio": 0.2, "overlap_width_ratio": 0.2, "mask_tag": None,
           "remask_fn": None, "inference_contract_version": 3, "max_det": 600, "batch_size": 8,
           "slice_size": 256, "sr_factor": 1, "model_path": "/w.pt"}
    cfg.update(over)
    return cfg


def _build(tmp_path, r, arms=PRODUCTION_ARMS, raster_sha="RS", weights_sha="WS",
           per_arm_config=None, per_arm_code=None, skip_manifest=(), skip_layer=()):
    """An out-dir holding one layer + manifest per arm, as a real run would leave it."""
    raster = tmp_path / "scene.tif"
    raster.write_bytes(b"raster")
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    for arm in arms:
        ss, sr = r.parse_arm(arm)
        layer = r.arm_layer_path(out, raster, ss, sr, 0.1, 1024, 0.2)
        if arm in skip_layer:
            continue
        layer.write_bytes(f"layer-{arm}".encode())
        if arm in skip_manifest:
            continue
        cfg = _config(slice_size=ss, sr_factor=sr)
        cfg.update((per_arm_config or {}).get(arm, {}))
        code = {"host": "laptop", "torch": "2.11.0+cu128", "python": "3.10.15",
                "platform": "Windows", "source_sha256": "d" * 64, "git_sha": "abc"}
        code.update((per_arm_code or {}).get(arm, {}))
        presets.manifest_path(layer).write_text(json.dumps({
            "written_utc": "2026-08-18T00:00:00+00:00", "config": cfg, "code": code,
            "inputs": {"raster": {"sha256": raster_sha}, "weights": {"sha256": weights_sha}},
            "outputs": {"n_after_nms": 10}}))
    return SimpleNamespace(arms=list(arms), raster=str(raster), out_dir=str(out), conf=0.1,
                           inference_size=1024, arm=None, fuse_only=True, no_fuse=False)


# --- what it refuses ------------------------------------------------------------------------------

def test_a_missing_arm_layer_is_refused_by_name(tmp_path):
    r = _runner()
    args = _build(tmp_path, r, skip_layer=("512",))
    with pytest.raises(SystemExit, match="missing arm layer"):
        r.fuse_only(args)


def test_a_layer_without_a_manifest_is_refused(tmp_path):
    """B2's rule: a product that cannot say what produced it is recomputed, not trusted."""
    r = _runner()
    args = _build(tmp_path, r, skip_manifest=("256",))
    with pytest.raises(SystemExit, match="missing manifest"):
        r.fuse_only(args)


def test_arms_from_different_rasters_are_refused(tmp_path):
    r = _runner()
    args = _build(tmp_path, r)
    # rewrite one manifest to claim a different raster
    ss, sr = r.parse_arm("512")
    layer = r.arm_layer_path(args.out_dir, args.raster, ss, sr, 0.1, 1024, 0.2)
    m = json.loads(presets.manifest_path(layer).read_text())
    m["inputs"]["raster"]["sha256"] = "DIFFERENT"
    presets.manifest_path(layer).write_text(json.dumps(m))
    with pytest.raises(SystemExit, match="disagree on raster sha256"):
        r.fuse_only(args)


def test_arms_from_different_weights_are_refused(tmp_path):
    r = _runner()
    args = _build(tmp_path, r)
    ss, sr = r.parse_arm("256")
    layer = r.arm_layer_path(args.out_dir, args.raster, ss, sr, 0.1, 1024, 0.2)
    m = json.loads(presets.manifest_path(layer).read_text())
    m["inputs"]["weights"]["sha256"] = "OTHER"
    presets.manifest_path(layer).write_text(json.dumps(m))
    with pytest.raises(SystemExit, match="disagree on weights sha256"):
        r.fuse_only(args)


@pytest.mark.parametrize("key,value", [
    ("confidence_threshold", 0.25),          # X-01: a different operating point
    ("min_area_threshold", 0),               # G1-06: a different area gate
    ("contour_method", "cv2"),               # a different tracer, so different polygons
    ("inference_contract_version", 2),       # pre-X-01/X-02 semantics
    ("mask_tag", "sam2"),                    # one arm re-masked and the others not
])
def test_config_that_must_match_is_fatal(tmp_path, key, value):
    r = _runner()
    args = _build(tmp_path, r, per_arm_config={"512": {key: value}})
    with pytest.raises(SystemExit, match="must match"):
        r.fuse_only(args)


def test_a_subset_arm_set_is_refused(tmp_path):
    """Same guard as the compute path: the output is named goal3_production."""
    r = _runner()
    args = _build(tmp_path, r, arms=["256", "256:2"])
    with pytest.raises(SystemExit, match="refusing to fuse: production fusion is"):
        r.fuse_only(args)


def test_a_missing_raster_is_refused_because_fusion_needs_its_resolution(tmp_path):
    r = _runner()
    args = _build(tmp_path, r)
    import pathlib
    pathlib.Path(args.raster).unlink()
    with pytest.raises(SystemExit, match="must be present"):
        r.fuse_only(args)


# --- what it permits, and records -----------------------------------------------------------------

def _patch_fusion(monkeypatch, captured):
    """Isolate the provenance logic from fusion internals."""
    import geopandas as gpd
    import pandas as pd
    import rastertools_BOULDERING.metadata as md
    import YOLOv8BeyondEarth.fusion as fusion
    import YOLOv8BeyondEarth.polygon as polygon

    monkeypatch.setattr(gpd, "read_file", lambda p, **k: gpd.GeoDataFrame({"score": [0.5]},
                                                                          geometry=[None]))
    monkeypatch.setattr(md, "get_resolution", lambda p: (0.5, 0.5))
    monkeypatch.setattr(fusion, "fuse_predictions",
                        lambda per_scale, **kw: (gpd.GeoDataFrame({"score": [0.5]},
                                                                  geometry=[None]),
                                                 pd.DataFrame({"why": ["dupe"]})))

    def _write(gdf, path, *a, **k):
        captured["written"] = str(path)
        __import__("pathlib").Path(path).write_bytes(b"fused")
    monkeypatch.setattr(polygon, "outlines_to_shp", _write)


def test_per_machine_differences_are_permitted_and_recorded(tmp_path, monkeypatch, capsys):
    """max_det and the producing stack may differ -- that is the whole point of the mode."""
    r = _runner()
    captured = {}
    _patch_fusion(monkeypatch, captured)
    args = _build(
        tmp_path, r,
        per_arm_config={"512": {"max_det": 1100}},                    # raised on the big GPU
        per_arm_code={"512": {"host": "sherlock", "torch": "2.6.0+cu124",
                              "source_sha256": "e" * 64, "git_sha": None}})
    r.fuse_only(args)

    fused = __import__("pathlib").Path(captured["written"])
    man = json.loads(presets.manifest_path(fused).read_text())
    prov = man["provenance"]
    assert prov["mode"] == "fuse_only"
    assert prov["heterogeneous"] is True                              # two stacks, said out loud
    assert prov["arms"]["512"]["max_det"] == 1100
    assert prov["arms"]["512"]["host"] == "sherlock"
    assert prov["arms"]["256"]["host"] == "laptop"
    assert "max_det" in prov["permitted_differences"]
    assert all(a["layer_sha256"] for a in prov["arms"].values())       # each layer pinned by digest
    assert "MIXED-STACK" in capsys.readouterr().out                   # and visible at the terminal


def test_a_single_stack_fuse_is_not_flagged_heterogeneous(tmp_path, monkeypatch):
    r = _runner()
    captured = {}
    _patch_fusion(monkeypatch, captured)
    r.fuse_only(_build(tmp_path, r))
    man = json.loads(presets.manifest_path(
        __import__("pathlib").Path(captured["written"])).read_text())
    assert man["provenance"]["heterogeneous"] is False


def test_refusing_to_fuse_writes_nothing(tmp_path, monkeypatch):
    """A refusal must not leave a half-built production-named catalog behind."""
    r = _runner()
    captured = {}
    _patch_fusion(monkeypatch, captured)
    args = _build(tmp_path, r, per_arm_config={"512": {"confidence_threshold": 0.25}})
    with pytest.raises(SystemExit):
        r.fuse_only(args)
    assert "written" not in captured
    assert not list(__import__("pathlib").Path(args.out_dir).glob("*fused*"))
