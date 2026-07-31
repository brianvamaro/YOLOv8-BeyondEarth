"""Truthful CRS for PDS-derived products, applied at read time (review item X-04).

The problem
-----------
The HiRISE test scene's GeoTIFF declares ``latitude_of_origin=45, standard_parallel_1=0``. Its PDS
label says the projection is equirectangular with ``CENTER_LATITUDE = 45``, and the label's own
pixel offsets prove which parameter that is:

- ``SAMPLE_PROJECTION_OFFSET`` = 13,678,766.5 px. With the longitude scale reduced by
  ``cos(45)`` — i.e. ``lat_ts = 45`` — the label's own longitudes predict 13,678,765.0, a relative
  agreement of 1e-7. With ``lat_ts = 0`` they predict 19,344,695: wrong by 41%.
- ``LINE_PROJECTION_OFFSET`` = 5,458,445.5 px matches ``lat_0 = 0`` to 1.7e-7.

So **the two parameters are transposed** in the raster: the truth is ``lat_ts = 45, lat_0 = 0``,
which is exactly what the human label files already carry. The labels were right and the raster was
wrong, not the other way round.

Why it has not broken anything visible
--------------------------------------
Map coordinates come from the GeoTIFF's affine, which was built from the *true* projection. A wrong
CRS label does not move them, so predictions and raster stay mutually aligned and every within-scene
analysis is fine. It breaks the moment anything leaves the local grid: latitude/longitude, true-north
azimuths, reprojection, scene seams, cross-body maps — the whole of Goal 4.

What this module does
---------------------
Derives the correct CRS from the PDS label, verifies it against the raster's own affine, and records
it in a ``<stem>.crs.json`` sidecar. **Source files are never modified** — the 1.9 GB TIFF is
gitignored and fetched from Drive, so it is not reproducible from the repository, and a correction
that cannot be reverted is a correction that cannot be reviewed. Readers call :func:`scene_crs` or
:func:`with_true_crs` and get the truth with provenance attached.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

#: Fields we need out of the PDS ``IMAGE_MAP_PROJECTION`` object.
_PDS_KEYS = ("MAP_PROJECTION_TYPE", "A_AXIS_RADIUS", "B_AXIS_RADIUS", "C_AXIS_RADIUS",
             "CENTER_LATITUDE", "CENTER_LONGITUDE", "MAP_SCALE", "MAP_RESOLUTION",
             "LINE_PROJECTION_OFFSET", "SAMPLE_PROJECTION_OFFSET",
             "POSITIVE_LONGITUDE_DIRECTION", "MAP_PROJECTION_ROTATION")

_VALUE = re.compile(r'^\s*([A-Z_0-9]+)\s*=\s*"?([^"<\r\n]+?)"?\s*(<[A-Z/]+>)?\s*$')


def read_pds_label(lbl_path):
    """Parse the projection fields out of a PDS label. Values are floats where numeric."""
    out = {}
    for line in Path(lbl_path).read_text(errors="replace").splitlines():
        m = _VALUE.match(line)
        if not m:
            continue
        key, raw, unit = m.group(1), m.group(2).strip(), m.group(3)
        if key not in _PDS_KEYS or key in out:
            continue
        try:
            value = float(raw)
            if unit == "<KM>":
                value *= 1000.0
        except ValueError:
            value = raw
        out[key] = value
    missing = {"MAP_PROJECTION_TYPE", "CENTER_LATITUDE", "CENTER_LONGITUDE",
               "A_AXIS_RADIUS"} - set(out)
    if missing:
        raise ValueError(f"{Path(lbl_path).name}: PDS label is missing {sorted(missing)}")
    return out


def crs_from_pds_label(lbl_path):
    """The authoritative CRS for a PDS equirectangular product, as a pyproj CRS.

    ``CENTER_LATITUDE`` is the latitude of true scale (``lat_ts``), **not** the latitude of origin.
    The radii are already the *local* radius at that latitude — the label says so explicitly — so
    the sphere is defined by ``A_AXIS_RADIUS`` and no flattening is applied.
    """
    from pyproj import CRS

    pds = read_pds_label(lbl_path)
    proj = str(pds["MAP_PROJECTION_TYPE"]).upper()
    if proj != "EQUIRECTANGULAR":
        raise NotImplementedError(f"only EQUIRECTANGULAR is handled, label says {proj!r}")
    if float(pds.get("MAP_PROJECTION_ROTATION", 0.0)) != 0.0:
        raise NotImplementedError("rotated map projections are not handled")
    radius = float(pds["A_AXIS_RADIUS"])
    return CRS.from_proj4(
        f"+proj=eqc +lat_ts={float(pds['CENTER_LATITUDE'])} +lat_0=0 "
        f"+lon_0={float(pds['CENTER_LONGITUDE'])} +x_0=0 +y_0=0 "
        f"+a={radius} +b={radius} +units=m +no_defs")


def verify_against_label(raster, lbl_path, tol_px=1.0):
    """Check the raster's affine against the PDS pixel offsets — the acceptance test for the fix.

    The label states where pixel (0, 0) sits in projected metres, independently of whatever CRS the
    GeoTIFF happens to declare. If the affine agrees, the *coordinates* are trustworthy and only the
    CRS label was wrong; that is the distinction that decides whether a re-projection is needed or
    only a relabel.
    """
    import rasterio

    pds = read_pds_label(lbl_path)
    scale = float(pds["MAP_SCALE"])
    with rasterio.open(raster) as src:
        ul_x, ul_y = src.transform.c, src.transform.f
        res = (abs(src.transform.a), abs(src.transform.e))
        declared = src.crs
    exp_x = -float(pds["SAMPLE_PROJECTION_OFFSET"]) * scale
    exp_y = float(pds["LINE_PROJECTION_OFFSET"]) * scale
    dx_px, dy_px = abs(ul_x - exp_x) / scale, abs(ul_y - exp_y) / scale
    return {"ok": bool(dx_px <= tol_px and dy_px <= tol_px),
            "dx_px": float(dx_px), "dy_px": float(dy_px),
            "raster_ul": (float(ul_x), float(ul_y)), "label_ul": (float(exp_x), float(exp_y)),
            "res": res, "map_scale": scale, "declared_crs": str(declared)}


def sidecar_path(raster):
    return Path(raster).with_suffix(".crs.json")


def write_crs_sidecar(raster, lbl_path, tol_px=1.0):
    """Record the corrected CRS beside the raster. Does not touch the raster itself."""
    import rasterio

    true_crs = crs_from_pds_label(lbl_path)
    check = verify_against_label(raster, lbl_path, tol_px=tol_px)
    with rasterio.open(raster) as src:
        declared = src.crs
    record = {
        "raster": Path(raster).name,
        "pds_label": Path(lbl_path).name,
        "declared_crs_wkt": declared.to_wkt() if declared else None,
        "true_crs_wkt": true_crs.to_wkt(),
        "true_crs_proj4": true_crs.to_proj4(),
        "correction": ("latitude_of_origin and standard_parallel_1 are transposed in the file; "
                       "the PDS label's own pixel offsets confirm lat_ts=CENTER_LATITUDE, lat_0=0"),
        "affine_check": check,
        "coordinates_trustworthy": check["ok"],
        "originals_modified": False,
    }
    path = sidecar_path(raster)
    path.write_text(json.dumps(record, indent=2))
    return path


def scene_crs(raster, strict=True):
    """The CRS a reader should use: the sidecar's corrected one if present, else the file's.

    ``strict=True`` refuses to silently fall back for a raster that has a PDS label sitting next to
    it — an un-generated sidecar is a setup mistake, not a reason to use the wrong CRS.
    """
    import rasterio
    from pyproj import CRS

    raster = Path(raster)
    side = sidecar_path(raster)
    if side.exists():
        return CRS.from_wkt(json.loads(side.read_text())["true_crs_wkt"])
    if strict and raster.with_suffix(".LBL").exists():
        raise FileNotFoundError(
            f"{raster.name} has a PDS label but no {side.name}; its declared CRS is not "
            "trustworthy (X-04). Run crs_fix.write_crs_sidecar() first.")
    with rasterio.open(raster) as src:
        return CRS.from_wkt(src.crs.to_wkt()) if src.crs else None


def with_true_crs(gdf, raster, strict=True):
    """Relabel a catalog with the scene's truthful CRS, and say so in ``gdf.attrs``.

    ``set_crs(..., allow_override=True)``, never ``to_crs``: the coordinates are already correct
    (:func:`verify_against_label` is what establishes that), so reprojecting them would apply a
    bogus datum shift on top of a good grid. This is the same override the analyses were already
    doing by hand — the difference is that the target is now derived from the PDS label rather
    than borrowed from whichever layer happened to be loaded first, and it is recorded.
    """
    crs = scene_crs(raster, strict=strict)
    out = gdf.set_crs(crs, allow_override=True)
    out.attrs = dict(getattr(gdf, "attrs", {}))
    out.attrs["crs_source"] = f"crs_fix.scene_crs({Path(raster).name})"
    out.attrs["crs_corrected"] = sidecar_path(raster).exists()
    return out


def meridian_convergence_is_zero(crs):
    """True for a north-up equirectangular frame, where grid north is true north by construction.

    Worth asserting explicitly rather than assuming: it is the property that lets Goal 2 report
    azimuths without a convergence correction, and it survives the CRS fix (both the wrong and the
    right parameterisation are ``eqc``) — but it would *not* survive a switch to a conformal or
    polar frame, which Goal 4's per-region CRS choice may want.
    """
    d = crs.to_dict() if hasattr(crs, "to_dict") else {}
    return d.get("proj") == "eqc" and float(d.get("alpha", 0.0) or 0.0) == 0.0
