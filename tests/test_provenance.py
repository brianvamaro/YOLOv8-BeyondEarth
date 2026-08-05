"""Presets, provenance, transactional writes and run isolation (G1-01, G1-03, G1-04, G1-07).

The common thread: a result on disk used to be unable to say what produced it, could be a
half-written file that resume logic accepted as complete, could be overwritten by a *different*
configuration under the same name, and could have its edge classification silently corrupted by a
concurrent run through a shared temporary path.
"""
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import geopandas as gpd
import numpy as np
import pytest
import rasterio as rio
import shapely
import torch
from rasterio.transform import from_origin

from YOLOv8BeyondEarth import polygon
from YOLOv8BeyondEarth import predict as P
from YOLOv8BeyondEarth import presets
from YOLOv8BeyondEarth.polygon import VectorWriteError, outlines_to_shp, write_vector

SLICE = 64


@pytest.fixture
def gdf():
    return gpd.GeoDataFrame(
        {"score": [0.5, 0.9], "nms_prio": [0.75, 0.95], "cat_id": [0, 0]},
        geometry=[shapely.box(0, 0, 2, 2), shapely.box(5, 5, 7, 7)], crs="EPSG:32631")


# --- presets (G1-01) ------------------------------------------------------------------------------

def test_production_preset_differs_from_the_generic_defaults():
    """The whole point of G1-01: the function defaults are not the benchmarked configuration."""
    import inspect
    defaults = {k: v.default for k, v in
                inspect.signature(P.get_sliced_prediction).parameters.items()}
    preset = presets.GOAL1_PRODUCTION
    differing = {k for k in preset if k in defaults and defaults[k] != preset[k]}
    assert {"downscale_pred", "min_area_threshold", "postprocess_match_threshold",
            "output_format", "save_bbox", "save_prenms"} <= differing


def test_preset_resolution_rejects_typos():
    resolved = presets.resolve("goal1_production", slice_size=256)
    assert resolved["slice_size"] == 256
    assert resolved["min_area_threshold"] == 6          # preset value survives
    assert resolved["_preset"] == "goal1_production"
    with pytest.raises(ValueError, match="slize_size"):
        presets.resolve("goal1_production", slize_size=256)
    with pytest.raises(ValueError, match="unknown preset"):
        presets.resolve("goal9")


def test_config_hash_is_stable_and_discriminating():
    a = presets.resolve("goal1_production")
    b = presets.resolve("goal1_production")
    c = presets.resolve("goal1_production", contour_method="cv2")
    assert presets.config_hash(a) == presets.config_hash(b)
    assert presets.config_hash(a) != presets.config_hash(c)
    # key order must not matter
    assert presets.config_hash({"x": 1, "y": 2}) == presets.config_hash({"y": 2, "x": 1})


# --- transactional writes (G1-03) -------------------------------------------------------------------

def test_write_vector_validates_and_promotes(tmp_path, gdf):
    out = tmp_path / "layer.gpkg"
    write_vector(gdf, out)
    assert out.exists() and len(gpd.read_file(out)) == 2
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")], "staging left behind"


def test_staging_does_not_lengthen_the_path_by_the_layer_name(tmp_path, gdf):
    """Staging must add a bounded prefix, not a second copy of the (long) layer name.

    It used to be ``mkdtemp(prefix=f".{out_path.stem}-")``, which put the stem in the path twice.
    A fused multiscale catalog name is 78 characters before the tile stem, so an output directory
    a few levels deep pushed the staged path past Windows' 260-character MAX_PATH and *every*
    write failed with a bare ``sqlite3_open ... unable to open database file``. Caught while
    preparing the re-run, on a real run that produced zero catalogs.
    """
    name = ("M127090695RE_00821_image-predictions-fused-bands-engulf70-512-256-256sr2-mask.gpkg")
    deep = tmp_path / "data" / "review" / "operating_point_validation" / "preds" / "conf010"
    deep.mkdir(parents=True)
    staged = []

    real_mkdtemp = tempfile.mkdtemp

    def record(*a, **k):
        path = real_mkdtemp(*a, **k)
        staged.append(Path(path))
        return path

    with mock.patch.object(polygon.tempfile, "mkdtemp", record):
        out = write_vector(gdf, deep / name)
    assert out.exists()
    # the staged file may exceed the destination only by the short staging directory component
    overhead = len(str(staged[0] / name)) - len(str(deep / name))
    assert overhead <= 24, f"staging added {overhead} characters to the path"


def test_failed_write_leaves_no_artifact(tmp_path, gdf, monkeypatch):
    """The failure the shapefile postmortem describes: a partial file that resume logic accepts.
    Nothing may appear at the destination."""
    def broken_to_file(self, *a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(gpd.GeoDataFrame, "to_file", broken_to_file)
    out = tmp_path / "layer.gpkg"
    with pytest.raises(VectorWriteError, match="disk full"):
        write_vector(gdf, out)
    assert not out.exists()
    assert list(tmp_path.iterdir()) == [], "staging directory survived the failure"


def test_null_geometry_readback_is_caught(tmp_path, gdf, monkeypatch):
    """Injected corruption: the layer writes 'successfully' but reads back with null geometry."""
    real_read = gpd.read_file

    def corrupt_read(path, *a, **k):
        out = real_read(path, *a, **k)
        out.loc[out.index[0], "geometry"] = None
        return out

    monkeypatch.setattr(gpd, "read_file", corrupt_read)
    out = tmp_path / "layer.gpkg"
    with pytest.raises(VectorWriteError, match="null or empty"):
        write_vector(gdf, out)
    assert not out.exists()


def test_truncated_feature_count_is_caught(tmp_path, gdf, monkeypatch):
    real_read = gpd.read_file
    monkeypatch.setattr(gpd, "read_file",
                        lambda path, *a, **k: real_read(path, *a, **k).iloc[:1])
    with pytest.raises(VectorWriteError, match="read back"):
        write_vector(gdf, tmp_path / "layer.gpkg")


def test_shapefile_sidecars_are_all_promoted(tmp_path, gdf):
    out = tmp_path / "layer.shp"
    outlines_to_shp(gdf, out)
    for suffix in (".shp", ".shx", ".dbf", ".prj"):
        assert out.with_suffix(suffix).exists(), suffix
    assert len(gpd.read_file(out)) == 2
    assert not [p for p in tmp_path.iterdir() if p.is_dir()], "staging directory left behind"


def test_existing_output_is_not_destroyed_by_a_failed_rewrite(tmp_path, gdf, monkeypatch):
    out = tmp_path / "layer.gpkg"
    write_vector(gdf, out)
    monkeypatch.setattr(gpd.GeoDataFrame, "to_file",
                        lambda self, *a, **k: (_ for _ in ()).throw(OSError("nope")))
    with pytest.raises(VectorWriteError):
        write_vector(gdf, out)
    assert len(gpd.read_file(out)) == 2, "the previous good layer was destroyed"


# --- end-to-end provenance and conflict refusal ------------------------------------------------------

class _Slices:
    def __init__(self):
        self.images = [np.full((SLICE, SLICE, 3), 200, np.uint8)]
        self.starting_pixels = [(0, 0)]

    def __len__(self):
        return 1


def _result(rects):
    boxes = np.zeros((len(rects), 6), dtype=np.float32)
    masks = np.zeros((len(rects), SLICE, SLICE), dtype=np.float32)
    for i, (x0, y0, x1, y1, s) in enumerate(rects):
        boxes[i] = [x0, y0, x1, y1, s, 0]
        masks[i, y0:y1 + 1, x0:x1 + 1] = 1.0
    return SimpleNamespace(boxes=SimpleNamespace(data=torch.from_numpy(boxes)),
                           masks=SimpleNamespace(data=torch.from_numpy(masks)))


@pytest.fixture
def scene(tmp_path):
    path = tmp_path / "scene.tif"
    with rio.open(path, "w", driver="GTiff", height=128, width=128, count=1, dtype="uint8",
                  crs="EPSG:32631", transform=from_origin(0.0, 128.0, 1.0, 1.0)) as dst:
        dst.write(np.full((1, 128, 128), 200, dtype=np.uint8))
    return path


@pytest.fixture
def model():
    return SimpleNamespace(model=lambda images, **kw: [_result([(20, 20, 40, 40, 0.8)])],
                           image_size=256, confidence_threshold=0.1, device="cpu",
                           category_mapping={"0": "boulder"}, model_path=None)


def _run(scene, model, out_dir, monkeypatch, **kw):
    monkeypatch.setattr(P, "slice_image", lambda **_: _Slices())
    opts = dict(output_dir=out_dir, slice_size=SLICE, save_prenms=False, save_bbox=False,
                output_format="gpkg")
    opts.update(kw)
    return P.get_sliced_prediction(scene, model, **opts)


def test_manifest_records_the_resolved_config(scene, model, tmp_path, monkeypatch):
    out = tmp_path / "out"
    _run(scene, model, out, monkeypatch, contour_method="cv2", min_area_threshold=6)
    manifests = list(out.glob("*-manifest.json"))
    assert len(manifests) == 1
    record = json.loads(manifests[0].read_text())
    assert record["config"]["contour_method"] == "cv2"
    assert record["config"]["min_area_threshold"] == 6
    assert record["config"]["inference_contract_version"] == P.INFERENCE_CONTRACT_VERSION
    assert record["inputs"]["raster"]["sha256"]
    assert record["outputs"]["n_detections"] == 1
    assert "git_sha" in record["code"] and "wall_seconds" in record
    assert record["crs"]


def test_a_different_config_cannot_silently_overwrite(scene, model, tmp_path, monkeypatch):
    """Filenames encode confidence/slice/imgsz/overlap only. A run differing in contour method,
    NMS threshold or area gate lands on the same name and used to replace it silently."""
    out = tmp_path / "out"
    _run(scene, model, out, monkeypatch)
    _run(scene, model, out, monkeypatch)                       # identical config: fine

    with pytest.raises(FileExistsError, match="different configuration"):
        _run(scene, model, out, monkeypatch, contour_method="cv2")
    with pytest.raises(FileExistsError):
        _run(scene, model, out, monkeypatch, postprocess_match_threshold=0.2)

    _run(scene, model, out, monkeypatch, contour_method="cv2", on_conflict="overwrite")
    record = json.loads(next(out.glob("*-manifest.json")).read_text())
    assert record["config"]["contour_method"] == "cv2"


def test_conflict_is_detected_before_any_inference_runs(scene, model, tmp_path, monkeypatch):
    """The check has to happen up front. Doing it at write time means a full scene burns hours of
    GPU and *then* raises on the last line, throwing the whole run away."""
    out = tmp_path / "out"
    _run(scene, model, out, monkeypatch)

    calls = []

    def counting_model(images, **kw):
        calls.append(len(images) if isinstance(images, list) else 1)
        return [_result([(20, 20, 40, 40, 0.8)])]

    model.model = counting_model
    with pytest.raises(FileExistsError, match="different configuration"):
        _run(scene, model, out, monkeypatch, contour_method="cv2")
    assert calls == [], "inference ran before the conflict was detected"


def test_manifest_does_not_rehash_an_already_identified_input(scene, model, tmp_path,
                                                              monkeypatch):
    """Hashing the raster is not free at scene scale (1.9 GB for HiRISE), and a three-arm run
    would otherwise re-read it six times."""
    seen = []
    real = presets.file_identity

    def counting(path):
        seen.append(str(path))
        return real(path)

    monkeypatch.setattr(presets, "file_identity", counting)
    _run(scene, model, tmp_path / "out", monkeypatch)
    assert sum(1 for s in seen if s.endswith("scene.tif")) == 1, seen


# --- the mask producer's identity (review item A1) ------------------------------------------------

def _stub_ckpt(tmp_path, name, payload):
    path = tmp_path / name
    path.write_bytes(payload)
    return path


def _remask_hash(sam_ckpt, **kw):
    from YOLOv8BeyondEarth.orientation_validation import sam2_remasker
    fn = sam2_remasker(SimpleNamespace(ckpt_path=str(sam_ckpt)), **kw)
    return presets.config_hash({"remask_fn": presets.describe_callable(fn)})


def test_solution_g_parameterisations_do_not_share_a_config_hash(tmp_path):
    """A1: ``_jsonable`` reduced any callable to its ``__name__``, and ``sam2_remasker`` always
    returns an inner function called ``remask`` — so *every* Solution-G configuration hashed
    identically and ``check_config_conflict`` saw two different mask producers as the same run.

    This is not a cosmetic provenance gap. Solution G **is** the re-masking configuration — it is the
    entire content of the Goal-2 fix (diagonal fraction 0.428 -> 0.200) — and two of its parameters
    changed inside a single delta (``SAM_PROMPT_CONF`` 0.25 -> 0.0, and ``max_box_blowup``). Since
    ``mask_tag`` is the only remask knob that reaches the filename, a pre-fix and a post-fix run
    resolved to the same name *and* the same hash, and the second silently overwrote the first.
    """
    ckpt_b = _stub_ckpt(tmp_path, "sam2.1_b.pt", b"base checkpoint")
    ckpt_l = _stub_ckpt(tmp_path, "sam2.1_l.pt", b"large checkpoint, different weights")

    base = _remask_hash(ckpt_b)
    assert _remask_hash(ckpt_b) == base                        # same parameterisation, same hash
    variants = {
        "a different SAM checkpoint": _remask_hash(ckpt_l),
        "a different background-grab guard": _remask_hash(ckpt_b, max_box_blowup=3.0),
        "the pre-fix ultralytics prompt confidence": _remask_hash(ckpt_b, sam_conf=0.25),
    }
    for why, digest in variants.items():
        assert digest != base, why
    assert len(set(variants.values()) | {base}) == 4, "two parameterisations collided"


def test_throughput_knobs_are_deliberately_outside_the_config_hash(tmp_path):
    """``sam_batch`` / ``device`` / ``empty_cache_every`` are excluded on purpose.

    ``sam_masks_in_order`` guarantees the same masks in the same order whichever chunking it used
    (the count-mismatch path re-prompts one box at a time precisely to preserve that), so these are
    throughput knobs, not result-defining ones. Folding them in would make a resume at a different
    batch size trip G1-07 and demand ``on_conflict="overwrite"`` — which is the escape hatch that
    disables the guard entirely, so a false conflict here costs real provenance elsewhere.
    """
    ckpt = _stub_ckpt(tmp_path, "sam2.1_b.pt", b"base checkpoint")
    assert (_remask_hash(ckpt, sam_batch=96) == _remask_hash(ckpt, sam_batch=32)
            == _remask_hash(ckpt, empty_cache_every=8) == _remask_hash(ckpt, device="cpu"))


def test_manifest_records_which_mask_producer_made_the_masks(scene, model, tmp_path, monkeypatch):
    """The other half of A1: ``write_manifest``'s ``inputs`` were ``{raster, YOLO weights}`` only, so
    nothing on disk could say which SAM produced a published Goal-2 catalog. A factory advertises its
    resolved identity on the closure; the pipeline forwards it to both the hash and the manifest."""
    ckpt = _stub_ckpt(tmp_path, "sam2.1_b.pt", b"base checkpoint")

    def remask(image, boxes_xyxy):
        return [np.ones((SLICE, SLICE), np.float32) for _ in range(len(boxes_xyxy))]
    remask.config = {"factory": "stub_remasker", "checkpoint": {"name": ckpt.name}}
    remask.inputs = {"sam_checkpoint": presets.file_identity(ckpt)}

    out = tmp_path / "out"
    _run(scene, model, out, monkeypatch, remask_fn=remask, mask_tag="sam2")
    record = json.loads(next(out.glob("*-manifest.json")).read_text())

    assert record["config"]["remask_fn"]["config"]["factory"] == "stub_remasker"
    assert record["config"]["mask_tag"] == "sam2"
    assert record["inputs"]["sam_checkpoint"]["sha256"], "the SAM checkpoint is not in the manifest"
    assert record["config_hash"] == presets.config_hash(record["config"])


def test_a_different_sam_cannot_silently_overwrite_a_remasked_catalog(scene, model, tmp_path,
                                                                      monkeypatch):
    """A1's failure mode end to end: same ``mask_tag``, different mask producer, same filename."""
    def _fn(tag):
        def remask(image, boxes_xyxy):
            return [np.ones((SLICE, SLICE), np.float32) for _ in range(len(boxes_xyxy))]
        remask.config = {"factory": "stub_remasker", "checkpoint": {"name": tag}}
        return remask

    out = tmp_path / "out"
    _run(scene, model, out, monkeypatch, remask_fn=_fn("sam2.1_b.pt"), mask_tag="sam2")
    _run(scene, model, out, monkeypatch, remask_fn=_fn("sam2.1_b.pt"), mask_tag="sam2")   # same: ok
    with pytest.raises(FileExistsError, match="different configuration"):
        _run(scene, model, out, monkeypatch, remask_fn=_fn("sam2.1_l.pt"), mask_tag="sam2")


def test_no_shared_temporary_files_between_runs(scene, model, tmp_path, monkeypatch):
    """G1-04: the footprint used to be written to a fixed ~/tmp/true-footprint.shp, shared by
    every process on the machine."""
    home_tmp = os.path.expanduser("~/tmp")
    before = set(os.listdir(home_tmp)) if os.path.isdir(home_tmp) else set()
    _run(scene, model, tmp_path / "out", monkeypatch)
    after = set(os.listdir(home_tmp)) if os.path.isdir(home_tmp) else set()
    assert not {f for f in after - before if "footprint" in f}


def test_concurrent_runs_keep_their_own_footprints(tmp_path, monkeypatch):
    """Two scenes with *different* footprints, run at the same time, must each classify their own
    edges. With the shared path, whichever finished second decided for both."""
    monkeypatch.setattr(P, "slice_image", lambda **_: _Slices())

    def make_scene(name, valid_cols):
        path = tmp_path / f"{name}.tif"
        data = np.zeros((1, 128, 128), dtype=np.uint8)
        data[0, :, :valid_cols] = 200                          # nodata beyond `valid_cols`
        with rio.open(path, "w", driver="GTiff", height=128, width=128, count=1, dtype="uint8",
                      crs="EPSG:32631", transform=from_origin(0.0, 128.0, 1.0, 1.0)) as dst:
            dst.write(data)
        return path

    def run(name, valid_cols):
        scene = make_scene(name, valid_cols)
        m = SimpleNamespace(model=lambda images, **kw: [_result([(20, 20, 40, 40, 0.8)])],
                            image_size=256, confidence_threshold=0.1, device="cpu",
                            category_mapping={"0": "boulder"}, model_path=None)
        gdf, _ = P.get_sliced_prediction(scene, m, output_dir=tmp_path / name, slice_size=SLICE,
                                         save_prenms=False, save_bbox=False, output_format="gpkg")
        return bool(gdf.is_at_edge.iloc[0])

    with ThreadPoolExecutor(max_workers=2) as ex:
        narrow = ex.submit(run, "narrow", 45)     # boulder at cols 20-40 sits near this edge
        wide = ex.submit(run, "wide", 128)        # ...and far from this one
        near_edge, far_from_edge = narrow.result(), wide.result()
    assert near_edge is True and far_from_edge is False


def test_png_conversion_is_reused_when_the_source_is_unchanged(scene, model, tmp_path,
                                                               monkeypatch):
    """G1-07: tiff_to_png reread and re-encoded the whole raster on every call, so a multiscale
    run repeated the identical conversion once per arm."""
    calls = []
    real = P.raster_convert.tiff_to_png

    def counting(in_raster, out_png=False, **kw):
        calls.append(in_raster)
        return real(in_raster, out_png, **kw)

    monkeypatch.setattr(P.raster_convert, "tiff_to_png", counting)
    _run(scene, model, tmp_path / "a", monkeypatch)
    _run(scene, model, tmp_path / "b", monkeypatch)
    assert len(calls) == 1, "the identical conversion ran twice"

    scene.write_bytes(scene.read_bytes())            # same content, new mtime -> still reused
    _run(scene, model, tmp_path / "c", monkeypatch)
    assert len(calls) == 1

    with rio.open(scene, "w", driver="GTiff", height=128, width=128, count=1, dtype="uint8",
                  crs="EPSG:32631", transform=from_origin(0.0, 128.0, 1.0, 1.0)) as dst:
        dst.write(np.full((1, 128, 128), 100, dtype=np.uint8))   # different content
    _run(scene, model, tmp_path / "d", monkeypatch)
    assert len(calls) == 2, "a changed raster must be reconverted"
