"""Unit tests for the Goal-3 phase-2 pieces: the streamed-SR knob validation and the
get_multiscale_prediction wrapper (per-scale runs monkeypatched — no GPU, no raster; the
real streamed-SR vs whole-crop-SR equivalence is a separate GPU check on the dev crops)."""
import geopandas as gpd
import numpy as np
import pytest
import shapely

from YOLOv8BeyondEarth import predict as P


def test_sr_factor_validation():
    with pytest.raises(ValueError, match="divisible"):
        P.get_sliced_prediction("scene.tif", None, output_dir=".", slice_size=256, sr_factor=3)
    with pytest.raises(ValueError, match="sr_factor"):
        P.get_sliced_prediction("scene.tif", None, output_dir=".", slice_size=256, sr_factor=0)


def test_wrapper_rejects_owned_kwargs(tmp_path):
    with pytest.raises(ValueError, match="slice_size"):
        P.get_multiscale_prediction(tmp_path / "scene.tif", None, output_dir=tmp_path,
                                    slice_size=256)


def _fake_scale_outputs(slice_size, sr_factor):
    # one shared boulder seen by all arms (finest must win) + one 512-only large boulder
    shared = {(512, 1): shapely.box(0, 0, 8.4, 8.2),
              (256, 1): shapely.box(0.2, 0.1, 8.3, 8.1),
              (256, 2): shapely.box(0.1, 0.2, 8.2, 8.3)}[(slice_size, sr_factor)]
    geoms, scores = [shared], [0.7]
    if (slice_size, sr_factor) == (512, 1):
        geoms.append(shapely.box(30, 0, 50, 20)); scores.append(0.9)
    return gpd.GeoDataFrame({"score": scores, "category_id": 0, "category_name": "boulder",
                             "is_within_slice": True}, geometry=geoms)


def test_generic_scale_aware_run_keeps_the_finest_footprint(tmp_path, monkeypatch):
    """The generic (non-preset) path, requested explicitly. This used to be the *default*, which
    is what G3-01 objected to: `scale_aware` with no engulf pass was refuted by the phase-3
    ablation, so an ordinary no-argument call produced an unvalidated catalog."""
    calls = []

    def fake_gsp(in_raster, detection_model, slice_size, sr_factor, output_dir, output_format,
                 save_prenms, save_bbox, postprocess, **kwargs):
        assert save_prenms is False and save_bbox is False and postprocess is True
        calls.append((slice_size, sr_factor))
        g = _fake_scale_outputs(slice_size, sr_factor)
        return g, g

    monkeypatch.setattr(P, "get_sliced_prediction", fake_gsp)
    monkeypatch.setattr(P.raster_metadata, "get_resolution", lambda p: (1.0, 1.0))

    fused, audit, per_scale = P.get_multiscale_prediction(
        tmp_path / "scene.tif", None, output_dir=tmp_path, preset=None)

    assert calls == [(512, 1), (256, 1), (256, 2)]           # the ladder, in order
    assert len(fused) == 2                                    # shared boulder + 512-only large
    kept = fused.sort_values("src_index").scale_key.tolist()
    assert "(256, 2)" in kept and "512" in kept               # finest wins; 512-only survives
    assert len(audit) == 2 and set(audit.keeper_scale_key) == {"(256, 2)"}
    assert set(per_scale) == {512, 256, (256, 2)}
    out = tmp_path / "scene-predictions-fused-scale_aware-512-256-256sr2-mask.gpkg"
    assert out.exists()
    assert (tmp_path / (out.stem + "-audit.csv")).exists()
    # provenance survives the round-trip (isin_slice rename included)
    back = gpd.read_file(out)
    assert {"scale_key", "slice_size", "sr_factor", "footprint", "n_merged",
            "isin_slice"} <= set(back.columns)


def test_no_argument_call_is_the_production_preset(tmp_path, monkeypatch):
    """G3-01 acceptance: a no-argument call must select 256 in contested clusters and consolidate
    contained fragments — i.e. be `256pref_engulf`, not the refuted `scale_aware`."""
    def fake_gsp(in_raster, detection_model, slice_size, sr_factor, output_dir, output_format,
                 save_prenms, save_bbox, postprocess, **kwargs):
        g = _fake_scale_outputs(slice_size, sr_factor)
        if (slice_size, sr_factor) == (256, 1):   # fragment wholly inside the 512 large boulder
            g = gpd.GeoDataFrame({"score": list(g.score) + [0.5], "category_id": 0,
                                  "category_name": "boulder", "is_within_slice": True},
                                 geometry=list(g.geometry) + [shapely.box(31, 1, 36, 6)])
        return g, g

    monkeypatch.setattr(P, "get_sliced_prediction", fake_gsp)
    monkeypatch.setattr(P.raster_metadata, "get_resolution", lambda p: (1.0, 1.0))

    fused, audit, _ = P.get_multiscale_prediction(tmp_path / "scene.tif", None,
                                                  output_dir=tmp_path)

    assert set(fused.scale_key) == {"256", "512"}             # 256 beats the finer SR arm
    eng = audit[audit.engulfed]
    assert len(eng) == 1 and eng.iloc[0].scale_key == "256"   # fragment consolidated
    assert eng.iloc[0].keeper_scale_key == "512"
    out = tmp_path / "scene-predictions-fused-bands-engulf70-512-256-256sr2-mask.gpkg"
    assert out.exists(), sorted(p.name for p in tmp_path.iterdir())

    # ...and the resolved fusion configuration is on disk beside it
    import json
    record = json.loads((tmp_path / (out.stem + "-manifest.json")).read_text())
    assert record["config"]["fusion"]["rule"] == "bands"
    assert record["config"]["fusion"]["engulf_thr"] == 0.7
    assert record["config"]["fusion"]["cross_arm_only"] is True
    assert record["same_arm_suppressions"] == 0               # G3-02 production invariant


def test_empty_layer_writers_skip(tmp_path):
    """pyogrio cannot write featureless layers; the writers must skip and leave no file
    (callers treat a missing NMS/fused layer as an empty catalog — the sparse-tile path)."""
    from YOLOv8BeyondEarth.polygon import outlines_to_shp, bboxes_to_shp
    empty = gpd.GeoDataFrame({"score": [], "category_id": [], "category_name": [],
                              "is_within_slice": [], "bbox": [], "polygon": []}, geometry=[])
    outlines_to_shp(empty, tmp_path / "empty-mask.gpkg")
    bboxes_to_shp(empty, tmp_path / "empty-bbox.gpkg")
    assert not (tmp_path / "empty-mask.gpkg").exists()
    assert not (tmp_path / "empty-bbox.gpkg").exists()


def test_wrapper_256pref_engulf(tmp_path, monkeypatch):
    """The phase-3 winner config plumbs through: bands rule + trivial banding + engulf_thr.
    A 256 fragment inside the 512-only large boulder (DIoU below cluster_thr) must be
    consolidated by the engulf pass, and the filename must carry the engulf tag."""

    def fake_gsp(in_raster, detection_model, slice_size, sr_factor, output_dir, output_format,
                 save_prenms, save_bbox, postprocess, **kwargs):
        g = _fake_scale_outputs(slice_size, sr_factor)
        if (slice_size, sr_factor) == (256, 1):   # fragment wholly inside the 512 large boulder
            g = gpd.GeoDataFrame({"score": list(g.score) + [0.5], "category_id": 0,
                                  "category_name": "boulder", "is_within_slice": True},
                                 geometry=list(g.geometry) + [shapely.box(31, 1, 36, 6)])
        return g, g

    monkeypatch.setattr(P, "get_sliced_prediction", fake_gsp)
    monkeypatch.setattr(P.raster_metadata, "get_resolution", lambda p: (1.0, 1.0))

    fused, audit, per_scale = P.get_multiscale_prediction(
        tmp_path / "scene.tif", None, output_dir=tmp_path,
        rule="bands", bands={256: (0, np.inf)}, engulf_thr=0.7)

    assert len(fused) == 2                                    # shared (256 wins) + 512 large
    kept = set(fused.scale_key)
    assert kept == {"256", "512"}                             # 256pref: 256 beats finer SR arm
    eng = audit[audit.engulfed]
    assert len(eng) == 1 and eng.iloc[0].scale_key == "256"   # fragment engulfed by the 512
    assert eng.iloc[0].keeper_scale_key == "512"
    out = tmp_path / "scene-predictions-fused-bands-engulf70-512-256-256sr2-mask.gpkg"
    assert out.exists()
