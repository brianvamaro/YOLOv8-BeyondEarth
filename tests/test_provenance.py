"""Presets, provenance, transactional writes and run isolation (G1-01, G1-03, G1-04, G1-07).

The common thread: a result on disk used to be unable to say what produced it, could be a
half-written file that resume logic accepted as complete, could be overwritten by a *different*
configuration under the same name, and could have its edge classification silently corrupted by a
concurrent run through a shared temporary path.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
import rasterio as rio
import shapely
import torch
from rasterio.transform import from_origin

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
