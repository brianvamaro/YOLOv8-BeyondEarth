"""Illumination geometry for the LRO NAC scene M139694087LE (Censorinus), for the Moon rose panels.

Every rose in the summary notebook that can carry a sun axis does, and each one must be **sourced** —
an unsourced sun line is worse than none. Mars scenes get theirs from the HiRISE `.LBL` files
(`data/sam2_validation/mars_sun.json`); the NAC product has no local label, so this queries ODE REST
and pulls `SUB_SOLAR_AZIMUTH` + `NORTH_AZIMUTH` out of the CUMINDEX record ODE ships inside
`ODE_notes` — the same trick `data/sam2_validation/recon_linne_nac.py` used for the Component-B
recon, narrowed from a lat/lon box to this one product.

Geographic sun axis = ``(SUB_SOLAR_AZIMUTH - NORTH_AZIMUTH) mod 180`` (the project's convention: the
LBL/index azimuths are in the **image** frame, so North must be subtracted to get a geographic
azimuth comparable to a north-up scene's rose).

Fallback, if ODE is unreachable or the record lacks the fields: measure the scene's illumination axis
from the imagery itself via the gradient structure tensor (`orientation.orientation_field`), which on
sunlit terrain is dominated by the bright-cap -> shadow terminator. That is a *different* kind of
number, so it is written with ``source: "image-derived"`` and the notebook labels the panel
accordingly rather than passing it off as ephemeris.

Writes ``moon_sun.json``, mirroring ``mars_sun.json``. Needs network for the primary path.
Run from inside YOLOv8-BeyondEarth/:
  C:\\Users\\brian\\anaconda3\\Scripts\\conda.exe run -n bouldernet python data/summary/fetch_nac_sun.py
"""
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

BASE = Path(r"c:\Users\brian\Documents\PhD\BoulderNet")
sys.path.insert(0, str(BASE / "YOLOv8-BeyondEarth" / "src"))
OUT = BASE / "YOLOv8-BeyondEarth" / "data" / "summary"
NAC = BASE / "YOLOv8-BeyondEarth" / "data" / "raster" / "M139694087LE.tif"
PRODUCT = "M139694087LE"
ODE = "https://oderest.rsl.wustl.edu/live2/"


def scene_centre():
    """Centre lat/lon of the raster, for the ODE box query."""
    import rasterio
    from pyproj import CRS, Transformer
    with rasterio.open(NAC) as ds:
        cx = (ds.bounds.left + ds.bounds.right) / 2.0
        cy = (ds.bounds.top + ds.bounds.bottom) / 2.0
        crs = CRS.from_wkt(ds.crs.to_wkt())
    lon, lat = Transformer.from_crs(crs, crs.geodetic_crs, always_xy=True).transform(cx, cy)
    return float(lat), float(lon % 360.0)


def fetch(params):
    url = ODE + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "boulder-summary/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def index_record(prod):
    """The widest comma-separated CUMINDEX row embedded in a product's ODE_notes."""
    notes = prod.get("ODE_notes", {}).get("ODE_note", [])
    notes = notes if isinstance(notes, list) else [notes]
    cand = [n for n in notes if isinstance(n, str) and n.count(",") > 20]
    return max(cand, key=lambda s: s.count(",")) if cand else None


def azimuths_from_record(rec, emi, inc, pha):
    """Pull (north_azimuth, sub_solar_azimuth) out of a positional CUMINDEX row.

    The row is headerless CSV, so the columns are located by *content*: ODE also reports the
    emission / incidence / phase angles as named JSON fields, and in the LROC index those three sit
    adjacent and are immediately followed by NORTH_AZIMUTH and SUB_SOLAR_AZIMUTH. Matching the
    triple pins the offset. (Same method as `data/sam2_validation/recon_linne_nac.py`, which used it
    for the Component-B recon.)
    """
    nums = []
    for t in rec.split(","):
        try:
            nums.append(float(t.strip().strip('"')))
        except ValueError:
            nums.append(None)
    for i in range(len(nums) - 4):
        w = nums[i:i + 3]
        if None in w or nums[i + 3] is None or nums[i + 4] is None:
            continue
        if abs(w[0] - emi) < 0.02 and abs(w[1] - inc) < 0.02 and abs(w[2] - pha) < 0.02:
            return nums[i + 3], nums[i + 4]
    return None, None


def from_ode():
    """(sun_axis, incidence, sub_solar, north, pdsid) from ODE, or None.

    Note the lookup is by **observation id**, not by the Prieur filename: ODE indexes this scene as
    ``nac.m139694087lc`` (left camera, CDR) while the raster is named ``M139694087LE``. A
    ``productid=`` query therefore finds nothing; a small box around the scene centre plus a substring
    match on the observation number does.
    """
    lat, lon = scene_centre()
    obs = re.sub(r"^M|[LR][EC]$", "", PRODUCT)         # M139694087LE -> 139694087
    cam = PRODUCT[-2]                                   # 'L' or 'R'
    print(f"scene centre lat {lat:.3f} lon {lon:.3f} | looking for obs {obs}, camera {cam}")
    for d in (0.35, 1.0):
        try:
            r = fetch(dict(query="product", target="moon", ihid="LRO", iid="LROC", pt="CDRNAC4",
                           results="mp", output="JSON", limit=1000, minlat=lat - d, maxlat=lat + d,
                           westernlon=(lon - d) % 360, easternlon=(lon + d) % 360))
        except Exception as e:                                  # noqa: BLE001 - network path
            print(f"  ODE query (box +-{d}) failed: {e}")
            continue
        pr = r.get("ODEResults", {}).get("Products")
        if not isinstance(pr, dict):
            print(f"  ODE box +-{d}: no products")
            continue
        plist = pr["Product"]
        plist = plist if isinstance(plist, list) else [plist]
        hits = [p for p in plist
                if obs in str(p.get("pdsid", "")) and str(p.get("pdsid", ""))[-2] == cam.lower()]
        print(f"  ODE box +-{d}: {len(plist)} products, {len(hits)} matching this observation")
        for p in hits:
            rec = index_record(p)
            if not rec:
                continue
            emi, inc, pha = (float(p["Emission_angle"]), float(p["Incidence_angle"]),
                             float(p["Phase_angle"]))
            na, ss = azimuths_from_record(rec, emi, inc, pha)
            if na is not None:
                return (ss - na) % 180.0, inc, ss, na, str(p.get("pdsid"))
            print(f"  {p.get('pdsid')}: could not locate the azimuth columns in its index row")
    return None


def from_image(n_windows=24, win=1024):
    """Image-derived illumination axis: the modal structure-tensor orientation over bright terrain.

    On a sunlit surface the dominant intensity gradient is the cap->shadow terminator, which runs
    **across** the sun line, so the illumination axis is that orientation rotated by 90 deg. This is a
    fallback, and it is recorded as such.
    """
    import rasterio
    from YOLOv8BeyondEarth.orientation import structure_tensor_orientation
    rng = np.random.default_rng(0)
    with rasterio.open(NAC) as ds:
        angles, weights = [], []
        for _ in range(n_windows * 4):
            if len(angles) >= n_windows:
                break
            c = int(rng.integers(0, max(1, ds.width - win)))
            r = int(rng.integers(0, max(1, ds.height - win)))
            a = ds.read(1, window=rasterio.windows.Window(c, r, win, win)).astype(float)
            if (a > 0).mean() < 0.99 or a.std() < 4:
                continue
            th, coh = structure_tensor_orientation(a)
            angles.append(th)
            weights.append(coh)
    if not angles:
        return None
    a = np.deg2rad(2 * np.asarray(angles))                     # axial -> doubled-angle mean
    w = np.asarray(weights)
    mean = (np.degrees(np.arctan2((w * np.sin(a)).sum(), (w * np.cos(a)).sum())) / 2.0) % 180.0
    return (mean + 90.0) % 180.0, float(w.mean()), len(angles)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    got = from_ode()
    if got:
        sun, inc, ss, na, pdsid = got
        rec = dict(product=PRODUCT, ode_pdsid=pdsid, sun_axis=round(sun, 1), incidence=inc,
                   sub_solar_azimuth=ss, north_azimuth=na, source="ODE/CUMINDEX",
                   convention="(SUB_SOLAR - NORTH) mod 180, geographic")
        print(f"\n{PRODUCT} (= ODE {pdsid}): sub-solar {ss}  north {na}  -> geographic sun axis "
              f"{sun:.1f} deg  (incidence {inc} deg)")
        if inc is not None and inc < 20:
            print("  NOTE incidence < 20 deg: this scene is near-nadir-lit, i.e. almost "
                  "shadow-free. Worth saying in the figure caption — it makes the Moon panel a "
                  "low-illumination control rather than a second shadowed example.")
    else:
        img = from_image()
        if not img:
            print("\nFAILED: neither ODE nor the image fallback produced a sun axis. "
                  "The Moon panels must then be drawn with NO sun line (do not guess one).")
            (OUT / "moon_sun.json").write_text(json.dumps(
                dict(product=PRODUCT, sun_axis=None, source="unavailable"), indent=2))
            return
        sun, coh, n = img
        rec = dict(product=PRODUCT, sun_axis=round(sun, 1), incidence=None,
                   source="image-derived", coherence=round(coh, 3), n_windows=n,
                   convention="structure-tensor terminator axis + 90 deg; NOT ephemeris")
        print(f"\n{PRODUCT}: ODE unavailable -> image-derived illumination axis {sun:.1f} deg "
              f"(mean coherence {coh:.3f}, {n} windows). Label the panel as image-derived.")
    (OUT / "moon_sun.json").write_text(json.dumps(rec, indent=2))
    print(f"wrote {OUT / 'moon_sun.json'}")


if __name__ == "__main__":
    main()
