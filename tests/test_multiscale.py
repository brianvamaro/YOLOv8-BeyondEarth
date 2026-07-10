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


def test_wrapper_runs_ladder_fuses_and_writes(tmp_path, monkeypatch):
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
        tmp_path / "scene.tif", None, output_dir=tmp_path)

    assert calls == [(512, 1), (256, 1), (256, 2)]           # the default ladder, in order
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
