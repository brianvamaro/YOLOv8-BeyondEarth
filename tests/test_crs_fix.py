"""Truthful CRS from the PDS label (review item X-04).

The HiRISE scene's GeoTIFF declares ``latitude_of_origin=45, standard_parallel_1=0``. The PDS
label's own pixel offsets show the two are transposed: the longitude scale carries a ``cos(45)``
factor, so ``lat_ts=45, lat_0=0``. The human label files already used that; the raster did not.

The distinction these tests protect is between *coordinates* and *labels*. The affine came from the
true projection, so map coordinates are right and everything within one scene is fine. Only the
declared CRS is wrong, which is why the repair is a relabel and not a reprojection.
"""
import json

import geopandas as gpd
import numpy as np
import pytest
import rasterio as rio
import shapely
from rasterio.transform import Affine

from YOLOv8BeyondEarth import crs_fix

# the real scene's numbers, so the fixture exercises the actual failure
R = 3386150.7470034
MAP_SCALE = 0.5
MAP_RES = 118198.95900815
SPO, LPO = 13678766.5, 5458445.5
CENTRE_LAT, CENTRE_LON = 45.0, 180.0

LABEL = f"""
OBJECT = IMAGE_MAP_PROJECTION
    MAP_PROJECTION_TYPE          = "EQUIRECTANGULAR"
    A_AXIS_RADIUS                = {R / 1000.0} <KM>
    B_AXIS_RADIUS                = {R / 1000.0} <KM>
    C_AXIS_RADIUS                = {R / 1000.0} <KM>
    POSITIVE_LONGITUDE_DIRECTION = EAST
    CENTER_LATITUDE              = {CENTRE_LAT:.3f} <DEG>
    CENTER_LONGITUDE             = {CENTRE_LON:.3f} <DEG>
    MAP_PROJECTION_ROTATION      = 0.0 <DEG>
    MAP_RESOLUTION               = {MAP_RES} <PIX/DEG>
    MAP_SCALE                    = {MAP_SCALE} <METERS/PIXEL>
    LINE_PROJECTION_OFFSET       = {LPO} <PIXEL>
    SAMPLE_PROJECTION_OFFSET     = {SPO} <PIXEL>
END_OBJECT = IMAGE_MAP_PROJECTION
"""

#: what the GeoTIFF wrongly declares: the two latitude parameters transposed
WRONG_WKT = (
    'PROJCS["Equirectangular MARS",GEOGCS["GCS_MARS",DATUM["unnamed",'
    f'SPHEROID["unnamed",{R},0]],PRIMEM["Reference meridian",0],'
    'UNIT["degree",0.0174532925199433]],PROJECTION["Equirectangular"],'
    f'PARAMETER["latitude_of_origin",{CENTRE_LAT}],PARAMETER["standard_parallel_1",0],'
    f'PARAMETER["central_meridian",{CENTRE_LON}],PARAMETER["false_easting",0],'
    'PARAMETER["false_northing",0],UNIT["metre",1]]')


@pytest.fixture
def scene(tmp_path):
    """A tiny raster whose affine is the real scene's, plus its PDS label."""
    tif = tmp_path / "ESP_test_RED.tif"
    lbl = tmp_path / "ESP_test_RED.LBL"
    lbl.write_text(LABEL)
    transform = Affine(MAP_SCALE, 0.0, -SPO * MAP_SCALE, 0.0, -MAP_SCALE, LPO * MAP_SCALE)
    with rio.open(tif, "w", driver="GTiff", height=32, width=32, count=1, dtype="uint8",
                  crs=rio.crs.CRS.from_wkt(WRONG_WKT), transform=transform) as dst:
        dst.write(np.full((1, 32, 32), 200, dtype=np.uint8))
    return tif, lbl


# --- reading the label -----------------------------------------------------------------------------

def test_label_parsing_converts_km_to_metres(scene):
    _, lbl = scene
    pds = crs_fix.read_pds_label(lbl)
    assert pds["A_AXIS_RADIUS"] == pytest.approx(R)
    assert pds["CENTER_LATITUDE"] == pytest.approx(45.0)
    assert pds["MAP_SCALE"] == pytest.approx(0.5)
    assert pds["MAP_PROJECTION_TYPE"] == "EQUIRECTANGULAR"


def test_missing_fields_are_rejected(tmp_path):
    bad = tmp_path / "bad.LBL"
    bad.write_text('MAP_PROJECTION_TYPE = "EQUIRECTANGULAR"\n')
    with pytest.raises(ValueError, match="missing"):
        crs_fix.read_pds_label(bad)


def test_unhandled_projections_raise(tmp_path):
    lbl = tmp_path / "polar.LBL"
    lbl.write_text(LABEL.replace("EQUIRECTANGULAR", "POLAR_STEREOGRAPHIC"))
    with pytest.raises(NotImplementedError, match="EQUIRECTANGULAR"):
        crs_fix.crs_from_pds_label(lbl)


# --- the correction itself --------------------------------------------------------------------------

def test_centre_latitude_becomes_lat_ts_not_lat_0(scene):
    """The whole bug in one assertion."""
    _, lbl = scene
    d = crs_fix.crs_from_pds_label(lbl).to_dict()
    assert d["proj"] == "eqc"
    assert d["lat_ts"] == pytest.approx(45.0)
    assert d.get("lat_0", 0.0) == pytest.approx(0.0)
    assert d["lon_0"] == pytest.approx(180.0)
    assert d["R"] == pytest.approx(R) if "R" in d else d["a"] == pytest.approx(R)


def test_label_offsets_discriminate_between_the_two_parameterisations():
    """Why we can be sure which way round it goes: with lat_ts=45 the label's own longitudes
    reproduce SAMPLE_PROJECTION_OFFSET to 1e-7; with lat_ts=0 they are wrong by 41%."""
    west = 16.337856391215
    with_ts45 = (CENTRE_LON - west) * MAP_RES * np.cos(np.radians(45.0))
    with_ts0 = (CENTRE_LON - west) * MAP_RES
    assert with_ts45 == pytest.approx(SPO, rel=1e-6)
    assert abs(with_ts0 - SPO) / SPO > 0.4


def test_affine_matches_the_label_so_only_the_crs_was_wrong(scene):
    """The distinction that decides relabel-vs-reproject: the coordinates are already right."""
    tif, lbl = scene
    check = crs_fix.verify_against_label(tif, lbl)
    assert check["ok"] and check["dx_px"] < 1.0 and check["dy_px"] < 1.0
    assert "latitude_of_origin" in check["declared_crs"]


def test_a_shifted_affine_is_detected(scene, tmp_path):
    """If the coordinates themselves were wrong, a relabel would not be enough — say so."""
    tif, lbl = scene
    moved = tmp_path / "moved.tif"
    with rio.open(tif) as src:
        prof = src.profile
        t = src.transform
        prof["transform"] = Affine(t.a, t.b, t.c + 500.0, t.d, t.e, t.f)
        with rio.open(moved, "w", **prof) as dst:
            dst.write(src.read())
    assert crs_fix.verify_against_label(moved, lbl)["ok"] is False


# --- sidecars, and leaving the originals alone -------------------------------------------------------

def test_sidecar_records_the_correction_without_touching_the_raster(scene):
    tif, lbl = scene
    before = tif.read_bytes()
    path = crs_fix.write_crs_sidecar(tif, lbl)
    assert path.name == "ESP_test_RED.crs.json"
    assert tif.read_bytes() == before, "the source raster was modified"

    rec = json.loads(path.read_text())
    assert rec["originals_modified"] is False
    assert rec["coordinates_trustworthy"] is True
    assert "lat_ts=45" in rec["true_crs_proj4"]
    assert "latitude_of_origin" in rec["declared_crs_wkt"]
    assert "transposed" in rec["correction"]


def test_scene_crs_prefers_the_sidecar(scene):
    tif, lbl = scene
    crs_fix.write_crs_sidecar(tif, lbl)
    assert crs_fix.scene_crs(tif).to_dict()["lat_ts"] == pytest.approx(45.0)


def test_scene_crs_refuses_to_fall_back_when_a_label_exists(scene):
    """An un-generated sidecar is a setup mistake, not a reason to use the wrong CRS."""
    tif, _ = scene
    with pytest.raises(FileNotFoundError, match="X-04"):
        crs_fix.scene_crs(tif)
    assert crs_fix.scene_crs(tif, strict=False) is not None      # explicit opt-out still works


def test_plain_raster_without_a_label_is_unaffected(tmp_path):
    from rasterio.transform import from_origin
    plain = tmp_path / "plain.tif"
    with rio.open(plain, "w", driver="GTiff", height=8, width=8, count=1, dtype="uint8",
                  crs="EPSG:32631", transform=from_origin(0, 8, 1, 1)) as dst:
        dst.write(np.zeros((1, 8, 8), dtype=np.uint8))
    assert crs_fix.scene_crs(plain).to_epsg() == 32631


# --- catalogs ----------------------------------------------------------------------------------------

def test_with_true_crs_relabels_and_does_not_move_geometry(scene):
    """set_crs override, never to_crs: the coordinates are already correct, so reprojecting them
    would apply a bogus datum shift on top of a good grid."""
    tif, lbl = scene
    crs_fix.write_crs_sidecar(tif, lbl)
    gdf = gpd.GeoDataFrame(geometry=[shapely.box(-6.8e6, 2.7e6, -6.8e6 + 5, 2.7e6 + 5)],
                           crs=rio.crs.CRS.from_wkt(WRONG_WKT).to_wkt())
    out = crs_fix.with_true_crs(gdf, tif)
    assert out.geometry.iloc[0].equals(gdf.geometry.iloc[0])       # untouched
    assert out.crs.to_dict()["lat_ts"] == pytest.approx(45.0)
    assert out.attrs["crs_corrected"] is True


def test_equirectangular_north_is_true_north_before_and_after(scene):
    """Both parameterisations are eqc, so Goal 2's no-convergence-correction assumption survives
    the fix — worth asserting rather than assuming, since Goal 4 may change frames."""
    from pyproj import CRS
    tif, lbl = scene
    assert crs_fix.meridian_convergence_is_zero(crs_fix.crs_from_pds_label(lbl))
    assert crs_fix.meridian_convergence_is_zero(CRS.from_wkt(WRONG_WKT))
