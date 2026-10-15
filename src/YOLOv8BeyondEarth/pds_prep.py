"""PDS HiRISE RDR -> the 8-bit GeoTIFF BoulderNet infers on (hirise2ctx PLAN_v3 P3, item 1).

BoulderNet's Mars weights were trained on 8-bit rasters, and a PDS RED RDR is a 10-bit JPEG 2000.
Three choices turn one into the other. Each changes something downstream, so each is fixed here and
recorded in a ``<stem>.prep.json`` sidecar rather than left to whichever tool was to hand.

**1. The georeferencing comes from the label, not the JP2.** GDAL can open the same pixels two ways,
and they disagree. Opening the ``.JP2`` reads its embedded GeoJP2 box, which carries the transposed
``lat_ts``/``lat_0`` of review item X-04 *and* an origin 1.5 px from the label's, in x and in y.
Opening the ``.LBL`` (GDAL's PDS driver, which follows the label's ``COMPRESSED_FILE`` pointer to
the JP2 for the pixels) gives the label's CRS and the label's origin. The label route is the one the
Prieur training rasters took: it reproduces their origins to 0.00 px on all three PDS training scenes
(ESP_017355_2260, ESP_025650_2165, ESP_028537_2270; measured 2026-10-14), and those are the rasters
the human labels were drawn on. A catalog made on the JP2 route sits +0.75 m E / -0.75 m S of the
humans at 0.5 m/px (ESP_017355_2260, ~3,800 matched boulders).

Which origin convention is *absolutely* right does not matter at this level. HiRISE's absolute
geolocation is good to tens of metres; what matters is that rasters, detections and human truth share
one frame, and the label frame is the one the truth is in.

**2. The bit-depth rule** is ``min((DN + 2) >> 2, 255)``, i.e. round-half-up of DN/4, clipped. Through
this module it reproduces two Prieur training rasters **pixel for pixel**, at both binnings:
ESP_017355_2260 (0.5 m/px; 0 of 1,932,385,780 px differ) and ESP_028537_2270 (0.25 m/px; 0 of
1,875,276,700), transform and CRS equal (2026-10-14). The hirise2ctx session first measured the
unclipped form on a sample of the first scene: slope 0.25000, intercept 0.12, residual sd 0.28 DN,
which is exactly round-half-up's error distribution. The clip is not cosmetic. DN 1022 and 1023 (the label's ``CORE_HIGH_INSTR_SATURATION`` and
``CORE_HIGH_REPR_SATURATION``) round to 256, and a bare ``uint8`` cast wraps them to **0**, the
nodata value, punching fill-valued holes into the brightest ground.

Two collisions are counted, not fixed, because the training rasters carry them too. DN 1
(``CORE_LOW_REPR_SATURATION``) rounds to 0 and becomes indistinguishable from fill, and DN 0
(``CORE_NULL``) *is* fill. The repo's earlier converter (``data/test2_pairs/run_test2_pairs.py``,
which made the Test-1/2 rasters) used ``x255/1023`` instead. It differs by at most 1 DN, and it is
available here by name so those products can be reproduced, not because it is a candidate.

**3. Native grid.** No resampling and no reprojection; nodata 0 (``CORE_NULL``); tiled, compressed
GeoTIFF. Any resampling of the input moves predictions.

The label route needs the label and the JP2 side by side under the exact names the label gives;
Linux is case-sensitive, so they are kept as PDS ships them. Downloads are whole files, never
``/vsicurl/`` range reads. hirise2ctx measured those at 140x slower, and on 2026-10-14 the PDS host
was serving ~50 KB/s regardless of parallelism, so a campaign fetch must be planned as hours, not
minutes.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .runstate import file_digest

#: Rule name -> (function on a DN array, human-readable expression). The default is the training
#: rasters' rule; the other exists only to reproduce the Test-1/2 rasters.
RULES = {
    "round_half_up_div4_clip255": (
        lambda dn: np.minimum((dn + 2) >> 2, 255).astype(np.uint8),
        "min((DN + 2) >> 2, 255)"),
    "x255_over_1023": (
        lambda dn: np.clip(np.round(dn.astype(np.float32) * (255.0 / 1023.0)), 0, 255)
        .astype(np.uint8),
        "clip(round(DN * 255 / 1023), 0, 255)"),
}
DEFAULT_RULE = "round_half_up_div4_clip255"

#: PDS RDR host. The archive layout is ``RDR/<phase>/ORB_<lo>_<hi>/<obs>/<product>.<ext>``.
PDS_ROOT = "https://hirise-pds.lpl.arizona.edu/PDS/RDR"

_LINE = re.compile(r"^\s*([A-Za-z_0-9:^]+)\s*=\s*(.*?)\s*$")
_RADIX = re.compile(r"^(\d+)#([0-9A-Fa-f]+)#$")
_NUMBER_WITH_UNIT = re.compile(r"^([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(<[^>]*>)?$")


# --- the label ------------------------------------------------------------------------------------

def _parse_value(raw):
    raw = raw.strip()
    if raw.startswith('"') and raw.endswith('"') and len(raw) >= 2:
        return raw[1:-1]
    m = _RADIX.match(raw)
    if m:
        return int(m.group(2), int(m.group(1)))
    m = _NUMBER_WITH_UNIT.match(raw)
    if m:
        num = m.group(1)
        return int(num) if re.fullmatch(r"[-+]?\d+", num) else float(num)
    return raw


def read_label(lbl_path):
    """Every ``KEY = value`` in a PDS3 label, keyed by its object path (``"UNCOMPRESSED_FILE.IMAGE.LINES"``).

    Objects and groups nest; comments (``/* ... */``) are stripped; quoted strings are unquoted;
    radix integers such as ``2#0000001111111111#`` become ints; numbers lose their ``<UNIT>``. A
    parenthesised list that continues onto following lines (``MRO:BINNING``) is joined and kept as
    its string. The first occurrence of a key wins.
    """
    out, stack = {}, []
    lines = iter(Path(lbl_path).read_text(errors="replace").splitlines())
    for raw in lines:
        line = raw.split("/*", 1)[0].rstrip()
        m = _LINE.match(line)
        if not m:
            continue
        key, value = m.group(1), m.group(2)
        while value.startswith("(") and ")" not in value:
            nxt = next(lines, None)
            if nxt is None:
                break
            value += " " + nxt.split("/*", 1)[0].strip()
        if key in ("OBJECT", "GROUP"):
            stack.append(value.strip('"'))
            continue
        if key in ("END_OBJECT", "END_GROUP"):
            if stack:
                stack.pop()
            continue
        out.setdefault(".".join(stack + [key]), _parse_value(value))
    return out


def _lookup(fields, suffix, required=True):
    """The value whose dotted key ends with ``suffix`` (shortest path wins), or ``None``."""
    hits = sorted((k for k in fields if k == suffix or k.endswith("." + suffix)), key=len)
    if not hits:
        if required:
            raise KeyError(f"PDS label has no {suffix}")
        return None
    return fields[hits[0]]


def label_summary(fields):
    """The fields this module depends on, validated. Raises on anything it does not understand."""
    s = {
        "observation_id": _lookup(fields, "OBSERVATION_ID", required=False),
        "product_id": _lookup(fields, "PRODUCT_ID", required=False),
        "jp2_name": fields.get("COMPRESSED_FILE.FILE_NAME"),
        "lines": _lookup(fields, "LINES"),
        "line_samples": _lookup(fields, "LINE_SAMPLES"),
        "sample_bits": _lookup(fields, "SAMPLE_BITS"),
        "sample_bit_mask": _lookup(fields, "SAMPLE_BIT_MASK"),
        "core_null": _lookup(fields, "CORE_NULL"),
        "core_low_repr_saturation": _lookup(fields, "CORE_LOW_REPR_SATURATION", required=False),
        "core_low_instr_saturation": _lookup(fields, "CORE_LOW_INSTR_SATURATION", required=False),
        "core_high_instr_saturation": _lookup(fields, "CORE_HIGH_INSTR_SATURATION", required=False),
        "core_high_repr_saturation": _lookup(fields, "CORE_HIGH_REPR_SATURATION", required=False),
        "scaling_factor": _lookup(fields, "SCALING_FACTOR", required=False),
        "offset": _lookup(fields, "UNCOMPRESSED_FILE.IMAGE.OFFSET", required=False),
        "map_projection_type": _lookup(fields, "MAP_PROJECTION_TYPE"),
        "center_latitude": _lookup(fields, "CENTER_LATITUDE"),
        "center_longitude": _lookup(fields, "CENTER_LONGITUDE"),
        "map_scale": _lookup(fields, "MAP_SCALE"),
        "line_projection_offset": _lookup(fields, "LINE_PROJECTION_OFFSET"),
        "sample_projection_offset": _lookup(fields, "SAMPLE_PROJECTION_OFFSET"),
        "binning": _lookup(fields, "MRO:BINNING", required=False),
        "incidence_angle": _lookup(fields, "INCIDENCE_ANGLE", required=False),
    }
    if not s["jp2_name"]:
        raise ValueError("label has no COMPRESSED_FILE.FILE_NAME, so it does not point at a JP2")
    if str(s["map_projection_type"]).upper() != "EQUIRECTANGULAR":
        raise NotImplementedError(f"only EQUIRECTANGULAR products are handled, label says "
                                  f"{s['map_projection_type']!r}")
    if s["sample_bit_mask"] != 0b1111111111:
        raise ValueError(f"expected a 10-bit product (SAMPLE_BIT_MASK 2#0000001111111111#), label "
                         f"says {s['sample_bit_mask']!r}; the 8-bit rule is defined for 10-bit DN")
    if s["core_null"] != 0:
        raise ValueError(f"CORE_NULL is {s['core_null']!r}; the pipeline assumes fill is 0")
    return s


def label_frame_origin(summary):
    """Upper-left corner the label implies, in GDAL's PDS-driver convention: ``-(SPO+0.5)*s, (LPO+0.5)*s``.

    This is the convention that reproduces the training rasters' origins to 0.00 px. Note it is
    **not** :func:`YOLOv8BeyondEarth.crs_fix.verify_against_label`'s convention (``-SPO*s, LPO*s``),
    which sits 0.5 px from it and 1.0 px from the GeoJP2 box, and so with its 1-px tolerance
    accepts both frames. That check cannot tell them apart; this one is what this module asserts.
    """
    s = float(summary["map_scale"])
    return (-(float(summary["sample_projection_offset"]) + 0.5) * s,
            (float(summary["line_projection_offset"]) + 0.5) * s)


def rdr_urls(observation_id, product="RED"):
    """``(lbl_url, jp2_url)`` for an RDR on the PDS host, e.g. ``ESP_028537_2270``."""
    obs = str(observation_id).strip()
    m = re.fullmatch(r"([A-Z]{3})_(\d{6})_(\d{4})", obs)
    if not m:
        raise ValueError(f"not a HiRISE observation id: {obs!r}")
    lo = int(m.group(2)) // 100 * 100
    base = f"{PDS_ROOT}/{m.group(1)}/ORB_{lo:06d}_{lo + 99:06d}/{obs}/{obs}_{product}"
    return base + ".LBL", base + ".JP2"


# --- fetching -------------------------------------------------------------------------------------

def fetch(url, dest, *, timeout=120, retries=6, chunk=1 << 20, log=print):
    """Download ``url`` to ``dest`` as a whole file, resuming a partial ``.part`` across attempts.

    Returns the file's identity (``url, path, bytes, sha256, fetched_utc``). An existing ``dest`` is
    trusted and not re-downloaded: completion is marked by the atomic rename from ``.part``, which
    only happens after the byte count matched the server's ``Content-Length``.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    if not dest.exists():
        last_error = None
        for attempt in range(retries):
            have = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={have}-"} if have else {}
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                            timeout=timeout) as resp:
                    if have and resp.status == 200:          # server ignored the range: restart
                        have = 0
                    expected = have + int(resp.headers.get("Content-Length") or 0)
                    with open(part, "ab" if have else "wb") as fh:
                        for block in iter(lambda: resp.read(chunk), b""):
                            fh.write(block)
                size = part.stat().st_size
                if expected and size != expected:
                    raise OSError(f"short read: {size} of {expected} bytes")
                os.replace(part, dest)
                break
            except urllib.error.HTTPError as exc:
                # 416 on a resume means the range starts at or past the end: the .part may already be
                # complete. Believe it only if the server's own total says so.
                total = re.search(r"/(\d+)$", exc.headers.get("Content-Range") or "")
                if (exc.code == 416 and part.exists() and total
                        and part.stat().st_size == int(total.group(1))):
                    os.replace(part, dest)
                    break
                last_error = exc
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                last_error = exc
            wait = min(300, 15 * 2 ** attempt)
            log(f"fetch {dest.name}: attempt {attempt + 1}/{retries} failed ({last_error}); "
                f"retrying in {wait}s")
            time.sleep(wait)
        else:
            raise OSError(f"could not fetch {url} after {retries} attempts: {last_error}")
    return {"url": url, "path": str(dest), "bytes": dest.stat().st_size,
            "sha256": file_digest(dest), "fetched_utc": datetime.now(timezone.utc).isoformat()}


# --- conversion -----------------------------------------------------------------------------------

def convert_dn(dn, rule=DEFAULT_RULE):
    """Apply a named 10-bit -> 8-bit rule. Arithmetic is done in int32, so nothing can wrap."""
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; known: {sorted(RULES)}")
    return RULES[rule][0](np.asarray(dn).astype(np.int32))


def _crs_params(crs):
    d = crs.to_dict() if crs is not None else {}
    return {k: d.get(k) for k in ("proj", "lat_ts", "lat_0", "lon_0", "R", "a", "b")}


def prepare_scene(lbl_path, out_tif=None, *, rule=DEFAULT_RULE, overwrite=False,
                  sources=None, log=print):
    """Convert one PDS RDR (label + JP2 beside it) to BoulderNet's 8-bit GeoTIFF. Returns the sidecar.

    ``sources`` is optional provenance (e.g. the :func:`fetch` identities) copied into the sidecar.
    The output is written to a temporary name and renamed, so a killed run never leaves a raster
    that looks finished. Raises, rather than writes, if the PDS driver did not open the product, if
    the frame does not match the label, or if any DN exceeds the 10-bit mask.
    """
    import rasterio
    from rasterio.env import GDALVersion

    lbl_path = Path(lbl_path)
    summary = label_summary(read_label(lbl_path))
    jp2 = lbl_path.with_name(summary["jp2_name"])
    if summary["jp2_name"] not in os.listdir(lbl_path.parent):
        raise FileNotFoundError(f"{lbl_path.name} points at {summary['jp2_name']!r}, which is not in "
                                f"{lbl_path.parent} under that exact name (the PDS driver needs it)")
    out_tif = Path(out_tif) if out_tif else lbl_path.with_suffix(".tif")
    sidecar = out_tif.with_name(out_tif.stem + ".prep.json")
    if out_tif.exists() and not overwrite:
        raise FileExistsError(f"{out_tif} exists; pass overwrite=True to replace it")

    t0 = time.perf_counter()
    counts = np.zeros(1 << 16, dtype=np.int64)
    tmp = out_tif.with_name(out_tif.name + f".{os.getpid()}.tmp")
    with rasterio.Env(GDAL_NUM_THREADS="ALL_CPUS"):
        with rasterio.open(lbl_path) as src:
            if src.driver != "PDS":
                raise RuntimeError(f"{lbl_path.name} opened with driver {src.driver!r}, not PDS; the "
                                   "label frame is only guaranteed through GDAL's PDS driver")
            if src.count != 1 or src.dtypes[0] != "uint16":
                raise ValueError(f"expected 1 band of uint16, got {src.count} of {src.dtypes[0]}")
            if (src.height, src.width) != (summary["lines"], summary["line_samples"]):
                raise ValueError(f"raster is {src.width}x{src.height}, label says "
                                 f"{summary['line_samples']}x{summary['lines']}")
            ul = (src.transform.c, src.transform.f)
            expect = label_frame_origin(summary)
            scale = float(summary["map_scale"])
            frame = {"convention": "gdal_pds_driver: -(SPO+0.5)*s, (LPO+0.5)*s",
                     "raster_ul": list(ul), "label_ul": list(expect),
                     "dx_px": (ul[0] - expect[0]) / scale, "dy_px": (ul[1] - expect[1]) / scale}
            if max(abs(frame["dx_px"]), abs(frame["dy_px"])) > 1e-6:
                raise RuntimeError(f"raster origin is ({frame['dx_px']:+.3f}, {frame['dy_px']:+.3f}) "
                                   "px from the label frame; refusing to write a raster in another "
                                   "frame from the training data and the human labels")
            params = _crs_params(src.crs)
            if (params.get("lat_ts") is None
                    or abs(float(params["lat_ts"]) - float(summary["center_latitude"])) > 1e-9
                    or float(params.get("lat_0") or 0.0) != 0.0):
                raise RuntimeError(f"CRS is not lat_ts=CENTER_LATITUDE, lat_0=0: {params}")
            profile = {
                "driver": "GTiff", "dtype": "uint8", "count": 1,
                "width": src.width, "height": src.height,
                "crs": src.crs, "transform": src.transform, "nodata": 0,
                "tiled": True, "blockxsize": 512, "blockysize": 512,
                "compress": "deflate", "predictor": 2, "zlevel": 6,
                "BIGTIFF": "IF_SAFER", "NUM_THREADS": "ALL_CPUS",
            }
            tags = {k: v for k, v in src.tags().items() if isinstance(v, str)}
            with rasterio.open(tmp, "w", **profile) as dst:
                dst.update_tags(**tags, BOULDERNET_PREP_RULE=RULES[rule][1],
                                BOULDERNET_PREP_FRAME="PDS label (GDAL PDS driver)")
                for _, window in src.block_windows(1):
                    dn = src.read(1, window=window)
                    counts += np.bincount(dn.ravel(), minlength=counts.size)[:counts.size]
                    if dn.max(initial=0) > summary["sample_bit_mask"]:
                        raise ValueError(f"DN {int(dn.max())} exceeds the 10-bit mask in {window}")
                    dst.write(convert_dn(dn, rule), 1, window=window)
            crs_wkt = src.crs.to_wkt()
            transform = list(src.transform)[:6]
            width, height = src.width, src.height
    os.replace(tmp, out_tif)

    lut = convert_dn(np.arange(1024), rule)
    valid = np.arange(1024) != summary["core_null"]
    record = {
        "written_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": round(time.perf_counter() - t0, 1),
        "rule": rule, "rule_expr": RULES[rule][1],
        "frame": "PDS label via GDAL's PDS driver (not the JP2's GeoJP2 box)",
        "gdal": str(GDALVersion.runtime()),
        "label": {k: v for k, v in summary.items()},
        "inputs": {"lbl": {"path": lbl_path.name, "sha256": file_digest(lbl_path)},
                   "jp2": {"path": jp2.name, "bytes": jp2.stat().st_size},
                   **({"fetched": sources} if sources else {})},
        "output": {"path": out_tif.name, "width": width, "height": height,
                   "bytes": out_tif.stat().st_size, "sha256": file_digest(out_tif),
                   "crs_wkt": crs_wkt, "transform": transform, "nodata": 0},
        "frame_check": frame,
        "crs_params": {k: v for k, v in params.items() if v is not None},
        "dn_counts": {
            "total": int(counts.sum()),
            "null_fill": int(counts[summary["core_null"]]),
            "valid_mapped_to_0": int(counts[:1024][valid & (lut == 0)].sum()),
            # 1022/1023 are the label's high-saturation codes; the default rule clips them to 255
            "high_saturation_dn_1022_1023": int(counts[1022] + counts[1023]),
            "histogram_nonzero": {int(i): int(c) for i, c in enumerate(counts[:1024]) if c},
        },
    }
    if summary.get("jp2_name") and not (sources and "jp2" in sources):
        # Hashing a GB-scale JP2 is ~10 s; worth it, since it is the campaign's input identity.
        record["inputs"]["jp2"]["sha256"] = file_digest(jp2)
    sidecar.write_text(json.dumps(record, indent=2, default=str))
    log(f"{out_tif.name}: {width}x{height}, {record['wall_seconds']} s, "
        f"{record['dn_counts']['high_saturation_dn_1022_1023']} px at high saturation, "
        f"{record['dn_counts']['valid_mapped_to_0']} valid px mapped to 0")
    return record


def compare_rasters(a, b, block_rows=2048):
    """Pixel-for-pixel and georeferencing comparison of two single-band rasters (block-wise, low RAM)."""
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(a) as ra, rasterio.open(b) as rb:
        out = {"a": str(a), "b": str(b),
               "shape_a": [ra.height, ra.width], "shape_b": [rb.height, rb.width],
               "transform_equal": tuple(ra.transform)[:6] == tuple(rb.transform)[:6],
               "transform_a": list(ra.transform)[:6], "transform_b": list(rb.transform)[:6],
               "crs_params_a": _crs_params(ra.crs), "crs_params_b": _crs_params(rb.crs),
               "dtype_a": ra.dtypes[0], "dtype_b": rb.dtypes[0]}
        out["crs_equal"] = out["crs_params_a"] == out["crs_params_b"]
        if out["shape_a"] != out["shape_b"]:
            out["identical"] = False
            return out
        diffs = np.zeros(511, dtype=np.int64)                 # index = (b - a) + 255
        for r0 in range(0, ra.height, block_rows):
            win = Window(0, r0, ra.width, min(block_rows, ra.height - r0))
            d = rb.read(1, window=win).astype(np.int16) - ra.read(1, window=win).astype(np.int16)
            diffs += np.bincount((d + 255).ravel(), minlength=511)
        n = int(diffs.sum())
        out["n_pixels"] = n
        out["n_differ"] = n - int(diffs[255])
        out["diff_histogram"] = {int(i) - 255: int(c) for i, c in enumerate(diffs) if c and i != 255}
        out["identical"] = (out["n_differ"] == 0 and out["transform_equal"] and out["crs_equal"])
        return out
