"""PDS RDR -> 8-bit GeoTIFF prep (hirise2ctx P3 item 1).

What these tests protect, in the order it would hurt:

- **The frame.** The prepared raster must sit in the label frame, the one the Prieur training rasters
  and the human labels are in. The convention is pinned to the training rasters' *measured* origins
  rather than re-derived, because the derivation is where the repo has already gone wrong twice
  (X-04's transposed CRS; the 1.5-px GeoJP2 origin of the ``data/mars_hirise`` copy).
- **The rule.** ``min((DN + 2) >> 2, 255)``. The clip is the part a reimplementation forgets, and
  forgetting it turns saturated pixels into nodata.
- **The fetch.** Whole files, resumable, and never renamed into place short.
"""
import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
import rasterio as rio
import rasterio.shutil

from YOLOv8BeyondEarth import pds_prep


# --- the rule -------------------------------------------------------------------------------------

def test_default_rule_is_round_half_up_of_quarter_and_clips():
    dn = np.array([0, 1, 2, 3, 4, 5, 6, 512, 1019, 1020, 1021, 1022, 1023])
    want = np.array([0, 0, 1, 1, 1, 1, 2, 128, 255, 255, 255, 255, 255])
    np.testing.assert_array_equal(pds_prep.convert_dn(dn), want)


def test_saturated_pixels_never_become_nodata():
    """(1022 + 2) >> 2 == 256; a bare uint8 cast of that is 0 -- the fill value."""
    out = pds_prep.convert_dn(np.arange(1024))
    assert out.dtype == np.uint8
    assert np.all(np.diff(out.astype(int)) >= 0), "the rule must be monotone"
    assert out[1022] == out[1023] == 255
    naive = ((np.arange(1024) + 2) >> 2).astype(np.uint8)
    assert naive[1022] == 0, "documents the failure the clip exists for"


def test_legacy_rule_reproduces_the_test2_converter_and_is_within_one_dn():
    dn = np.arange(1024)
    # data/test2_pairs/run_test2_pairs.py:48, verbatim
    test2 = np.clip(np.round(dn.astype(np.float32) * (255.0 / 1023.0)), 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(pds_prep.convert_dn(dn, "x255_over_1023"), test2)
    gap = np.abs(pds_prep.convert_dn(dn).astype(int) - test2.astype(int))
    assert gap.max() == 1 and gap.sum() > 0


def test_unknown_rule_is_refused():
    with pytest.raises(ValueError, match="unknown rule"):
        pds_prep.convert_dn(np.arange(4), "stretch_to_percentiles")


# --- the label ------------------------------------------------------------------------------------

LABEL_TEMPLATE = """PDS_VERSION_ID            = PDS3
OBSERVATION_ID            = "{obs}"
PRODUCT_ID                = "{obs}_RED"
SOURCE_PRODUCT_ID         = ({obs}_RED0_0, {obs}_RED0_1,
                            {obs}_RED1_0)
OBJECT = IMAGE_MAP_PROJECTION
    ^DATA_SET_MAP_PROJECTION     = "DSMAP.CAT"
    MAP_PROJECTION_TYPE          = "{proj}"
    A_AXIS_RADIUS                = 3386.1507470034 <KM>
    B_AXIS_RADIUS                = 3386.1507470034 <KM>
    C_AXIS_RADIUS                = 3386.1507470034 <KM>
    POSITIVE_LONGITUDE_DIRECTION = EAST
    CENTER_LATITUDE              = 45.000 <DEG>
    CENTER_LONGITUDE             = 180.000 <DEG>
    LINE_FIRST_PIXEL             = 1
    LINE_LAST_PIXEL              = {h}
    SAMPLE_FIRST_PIXEL           = 1
    SAMPLE_LAST_PIXEL            = {w}
    MAP_PROJECTION_ROTATION      = 0.0 <DEG>
    MAP_RESOLUTION               = 236397.9180163 <PIX/DEG>
    MAP_SCALE                    = 0.25 <METERS/PIXEL>
    LINE_PROJECTION_OFFSET       = 11022679.5 <PIXEL>
    SAMPLE_PROJECTION_OFFSET     = 26947544.5 <PIXEL>
END_OBJECT = IMAGE_MAP_PROJECTION
GROUP = INSTRUMENT_SETTING_PARAMETERS
    MRO:BINNING                 = (2, 2, 2, 1, 1, 1, 1, 2, 2, -9998, -9998,
                                  -9998, -9998, -9998)
END_GROUP = INSTRUMENT_SETTING_PARAMETERS
GROUP = VIEWING_PARAMETERS
    INCIDENCE_ANGLE   = 58.270411 <DEG>
END_GROUP = VIEWING_PARAMETERS
/* The JPEG2000 image data file associated with this label. */
OBJECT = COMPRESSED_FILE
    FILE_NAME                  = "{obs}_RED.JP2"
    RECORD_TYPE                = UNDEFINED
    ENCODING_TYPE              = "JP2"
    ENCODING_TYPE_VERSION_NAME = "ISO/IEC15444-1:2004"
    INTERCHANGE_FORMAT         = BINARY
    UNCOMPRESSED_FILE_NAME     = "{obs}_RED.IMG"
    REQUIRED_STORAGE_BYTES     = {nbytes} <BYTES>
END_OBJECT = COMPRESSED_FILE
OBJECT = UNCOMPRESSED_FILE
    FILE_NAME    = "{obs}_RED.IMG"
    RECORD_TYPE  = FIXED_LENGTH
    RECORD_BYTES = {rb} <BYTES>
    FILE_RECORDS = {h}
    ^IMAGE       = "{obs}_RED.IMG"
    OBJECT = IMAGE
        DESCRIPTION                = "HiRISE projected and mosaicked product"
        LINES                      = {h}
        LINE_SAMPLES               = {w}
        BANDS                      = 1
        SAMPLE_TYPE                = MSB_UNSIGNED_INTEGER
        SAMPLE_BITS                = 16
        SAMPLE_BIT_MASK            = {mask}
        /* I/F = (DN * SCALING_FACTOR) + OFFSET */
        SCALING_FACTOR             = 6.72430508092682e-05
        OFFSET                     = 0.0349
        CORE_NULL                  = 0
        CORE_LOW_REPR_SATURATION   = 1
        CORE_LOW_INSTR_SATURATION  = 2
        CORE_HIGH_REPR_SATURATION  = 1023
        CORE_HIGH_INSTR_SATURATION = 1022
    END_OBJECT = IMAGE
END_OBJECT = UNCOMPRESSED_FILE
END
"""


def _label(tmp_path, obs="ESP_028537_2270", h=64, w=96, proj="EQUIRECTANGULAR",
           mask="2#0000001111111111#"):
    path = tmp_path / f"{obs}_RED.LBL"
    path.write_text(LABEL_TEMPLATE.format(obs=obs, h=h, w=w, proj=proj, mask=mask,
                                          nbytes=h * w * 2, rb=w * 2))
    return path


def test_read_label_nests_unquotes_and_parses_radix_and_multiline(tmp_path):
    f = pds_prep.read_label(_label(tmp_path))
    assert f["COMPRESSED_FILE.FILE_NAME"] == "ESP_028537_2270_RED.JP2"
    assert f["UNCOMPRESSED_FILE.IMAGE.SAMPLE_BIT_MASK"] == 1023
    assert f["UNCOMPRESSED_FILE.IMAGE.LINES"] == 64
    assert f["IMAGE_MAP_PROJECTION.MAP_SCALE"] == 0.25
    assert f["IMAGE_MAP_PROJECTION.CENTER_LATITUDE"] == 45.0
    # a comment line between objects must not leak, and a list continued on the next line is whole
    assert "-9998, -9998, -9998)" in f["INSTRUMENT_SETTING_PARAMETERS.MRO:BINNING"]
    assert f["UNCOMPRESSED_FILE.IMAGE.OFFSET"] == 0.0349


def test_label_summary_refuses_what_the_rule_is_not_defined_for(tmp_path):
    s = pds_prep.label_summary(pds_prep.read_label(_label(tmp_path)))
    assert (s["sample_bit_mask"], s["core_null"], s["jp2_name"]) == (1023, 0, "ESP_028537_2270_RED.JP2")
    polar, eight_bit = tmp_path / "polar", tmp_path / "eight_bit"
    polar.mkdir()
    eight_bit.mkdir()
    with pytest.raises(NotImplementedError, match="EQUIRECTANGULAR"):
        pds_prep.label_summary(pds_prep.read_label(_label(polar, proj="POLAR STEREOGRAPHIC")))
    with pytest.raises(ValueError, match="10-bit"):
        pds_prep.label_summary(pds_prep.read_label(_label(eight_bit, mask="2#0000000011111111#")))


@pytest.mark.parametrize("scale, spo, lpo, training_ul", [
    # data_prieur/raw_data/mars/*/raster/*.tif origins, read 2026-10-14
    (0.5, 13678766.5, 5458445.5, (-6839383.5, 2729223.0)),            # ESP_017355_2260
    (0.25, 26947544.5, 11022679.5, (-6736886.25, 2755670.0)),         # ESP_028537_2270
    (0.25, -33043975.5, 8642528.5, (8260993.75, 2160632.25)),         # ESP_025650_2165
])
def test_label_frame_is_the_frame_the_training_rasters_are_in(scale, spo, lpo, training_ul):
    ul = pds_prep.label_frame_origin({"map_scale": scale, "sample_projection_offset": spo,
                                      "line_projection_offset": lpo})
    assert ul == training_ul


def test_rdr_urls_follow_the_archive_layout():
    lbl, jp2 = pds_prep.rdr_urls("ESP_028537_2270")
    assert lbl == ("https://hirise-pds.lpl.arizona.edu/PDS/RDR/ESP/ORB_028500_028599/"
                   "ESP_028537_2270/ESP_028537_2270_RED.LBL")
    assert jp2.endswith("ESP_028537_2270_RED.JP2")
    assert "/PSP/ORB_007400_007499/PSP_007492_2265/" in pds_prep.rdr_urls("PSP_007492_2265")[0]
    with pytest.raises(ValueError):
        pds_prep.rdr_urls("ESP_28537_2270")


# --- end to end on a synthetic product ------------------------------------------------------------

@pytest.fixture
def product(tmp_path):
    """A small 10-bit JP2 with a HiRISE-shaped label beside it, so the real PDS driver opens it."""
    h, w = 64, 96
    rng = np.random.default_rng(0)
    dn = rng.integers(3, 1020, size=(h, w)).astype(np.uint16)
    dn[:, :20] = 0                                     # a fill wedge
    dn[5, 30], dn[5, 31], dn[6, 30], dn[6, 31] = 1, 2, 1022, 1023   # the special codes
    lbl = _label(tmp_path, h=h, w=w)
    jp2 = tmp_path / "ESP_028537_2270_RED.JP2"
    with rio.MemoryFile() as mem:
        with mem.open(driver="GTiff", width=w, height=h, count=1, dtype="uint16") as m:
            m.write(dn, 1)
            rio.shutil.copy(m, jp2, driver="JP2OpenJPEG", QUALITY=100, REVERSIBLE="YES", NBITS=10)
    return lbl, dn


def test_prepare_scene_converts_in_the_label_frame_and_records_it(product, tmp_path):
    lbl, dn = product
    out = tmp_path / "out" / "ESP_028537_2270_RED.tif"
    out.parent.mkdir()
    rec = pds_prep.prepare_scene(lbl, out, log=lambda *_: None)
    with rio.open(out) as src:
        got = src.read(1)
        assert src.dtypes[0] == "uint8" and src.nodata == 0
        assert (src.transform.c, src.transform.f) == (-6736886.25, 2755670.0)
        assert src.crs.to_dict()["lat_ts"] == 45.0 and src.crs.to_dict()["lat_0"] == 0
    np.testing.assert_array_equal(got, pds_prep.convert_dn(dn))
    assert got[5, 30] == 0 and got[6, 31] == 255, "DN 1 -> fill-valued 0; DN 1023 -> 255, not 0"
    assert rec["frame_check"]["dx_px"] == rec["frame_check"]["dy_px"] == 0.0
    assert rec["dn_counts"]["valid_mapped_to_0"] == 1           # the one DN-1 pixel (DN 2 -> 1)
    assert rec["dn_counts"]["high_saturation_dn_1022_1023"] == 2
    assert rec["dn_counts"]["null_fill"] == int((dn == 0).sum())
    side = json.loads((out.parent / "ESP_028537_2270_RED.prep.json").read_text())
    assert side["rule_expr"] == "min((DN + 2) >> 2, 255)" and side["output"]["sha256"]
    with pytest.raises(FileExistsError):
        pds_prep.prepare_scene(lbl, out, log=lambda *_: None)


def test_prepare_scene_refuses_a_jp2_under_another_name(product, tmp_path):
    lbl, _ = product
    (tmp_path / "ESP_028537_2270_RED.JP2").rename(tmp_path / "esp_028537_2270_red.jp2")
    with pytest.raises(FileNotFoundError, match="exact name"):
        pds_prep.prepare_scene(lbl, tmp_path / "x.tif", log=lambda *_: None)


def test_compare_rasters_reports_pixel_and_transform_differences(product, tmp_path):
    lbl, _ = product
    a = tmp_path / "a.tif"
    pds_prep.prepare_scene(lbl, a, log=lambda *_: None)
    b = tmp_path / "b.tif"
    with rio.open(a) as src:
        prof, arr = src.profile, src.read(1)
    arr[10, 50] ^= 1
    with rio.open(b, "w", **prof) as dst:
        dst.write(arr, 1)
    res = pds_prep.compare_rasters(a, b)
    assert res["n_differ"] == 1 and res["transform_equal"] and not res["identical"]
    assert pds_prep.compare_rasters(a, a)["identical"]


# --- fetching -------------------------------------------------------------------------------------

class _RangeHandler(SimpleHTTPRequestHandler):
    """Static files with single-range support -- what the PDS host does and http.server does not."""
    honour_ranges = True

    def log_message(self, *args):
        pass

    def do_GET(self):
        path = self.translate_path(self.path)
        try:
            data = open(path, "rb").read()
        except OSError:
            self.send_error(404)
            return
        rng = self.headers.get("Range")
        if rng and self.honour_ranges:
            start = int(rng.split("=")[1].split("-")[0])
            if start >= len(data):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{len(data)}")
                self.end_headers()
                return
            body = data[start:]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        else:
            body = data
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "srv"
    root.mkdir()
    payload = np.random.default_rng(1).integers(0, 256, 300_000, dtype=np.uint8).tobytes()
    (root / "f.JP2").write_bytes(payload)
    handler = type("H", (_RangeHandler,), {})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), partial(handler, directory=str(root)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/f.JP2", payload, handler
    httpd.shutdown()


def test_fetch_resumes_a_partial_download(server, tmp_path):
    url, payload, _ = server
    dest = tmp_path / "dl" / "f.JP2"
    dest.parent.mkdir()
    (dest.parent / "f.JP2.part").write_bytes(payload[:123_456])     # a killed earlier attempt
    ident = pds_prep.fetch(url, dest, log=lambda *_: None)
    assert dest.read_bytes() == payload and not (dest.parent / "f.JP2.part").exists()
    assert ident["bytes"] == len(payload) and len(ident["sha256"]) == 64


def test_fetch_restarts_when_the_server_ignores_the_range(server, tmp_path):
    url, payload, handler = server
    handler.honour_ranges = False
    dest = tmp_path / "f.JP2"
    (tmp_path / "f.JP2.part").write_bytes(b"garbage that must not survive")
    pds_prep.fetch(url, dest, log=lambda *_: None)
    assert dest.read_bytes() == payload


def test_fetch_accepts_a_part_that_is_already_complete(server, tmp_path):
    url, payload, _ = server
    dest = tmp_path / "f.JP2"
    (tmp_path / "f.JP2.part").write_bytes(payload)                  # killed after the last byte
    pds_prep.fetch(url, dest, log=lambda *_: None)
    assert dest.read_bytes() == payload
