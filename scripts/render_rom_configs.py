"""Render the Prieur ROM tiles under detector configurations 2 and 3 (hirise2ctx notebook 35, D9).

Configuration 1 (the vClaire/Goal-1 256 arm) was rendered by
``data_prieur/rom_label_error/render_rom_predictions.py`` into ``data/rom_label_error/<obs>/``. This
renders the other two on **the same padded crops** (its ``<obs>_romNN_tTTT.tif`` files, cut from the
``data_prieur`` rasters, so the label frame of the human labels -- X-07), so the three configurations
differ only in the detector path:

- **config 2** -- the ``{512, 256}`` core ladder: both arms at the production preset (conf 0.10,
  imgsz 1024, ``max_det`` 1100), fused by ``arm_ratification.fuse_combo``, the exact rule the 2x2
  ratification used for its ``{512, 256}`` cell. Nothing new is decided here.
- **config 3** -- config 2's fused catalog re-masked by SAM2 (``remask.remask_windows``, phase 5a)
  at ``window_size`` 256 and ``max_window_growth`` 1.0, the setting closest to the validated
  ``sam256`` regime. ⚠ 5a's **entry gate 3 is open**: a window grows past 256 px whenever a member
  straddles its cell, which lowers SAM's logit resolution below the validated 1.0 logit px per
  native px. Every row therefore carries ``logit_px_per_native_px``, so the analysis can stratify by
  it or restrict to 1.0, rather than mix two treatments under one ``mask_src="sam2"``.

**Layout, chosen so notebook 35 cannot pick the wrong file.** Its loader globs
``<obs>/*romNN_tTTT-predictions-*-nms.gpkg`` non-recursively and takes the first match, so new layers
beside config 1's would compete with it. Each configuration therefore gets its own subdirectory,
holding exactly one ``-nms.gpkg`` per tile; the arm layers sit one level further down:

    data/rom_label_error/<obs>/config2/<stem>-predictions-fused-512-256-nms.gpkg
    data/rom_label_error/<obs>/config2/arms/<stem>-predictions-ct-010-ss-{256,512}-...-nms.gpkg
    data/rom_label_error/<obs>/config3/<stem>-predictions-fused-512-256-sam2-nms.gpkg

Every arm must report ``truncation {}``, and a tile whose arm truncated is recorded in its manifest
and in the run summary as a lower bound rather than passed off as a catalog. Resumable per tile.
GPU; never alongside another GPU job. On a laptop, check it is on AC first: on battery it parks at
~8.5 W with no error.

**Each tile renders in its own process.** A CUDA out-of-memory poisons the context: after it, even
``torch.cuda.empty_cache()`` raises. So a tile that does not fit this card is recorded with its
cause and the rest go on, rather than one dense tile ending the run. That happened 2026-10-14:
ESP_017355_2260's 512 slices reach ~1,000 detections, which at ~5 MiB of transient mask each does
not fit 8 GiB even at batch 1.

    conda run -n bouldernet --no-capture-output python -u scripts/render_rom_configs.py [--obs ESP_028537_2270] [--limit 1]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from YOLOv8BeyondEarth import presets  # noqa: E402

ROM_DIR = REPO / "data" / "rom_label_error"
WEIGHTS = REPO / "data" / "weights" / "yolov8-m-boulder-detection-tmp.pt"
SAM_WEIGHTS = REPO / "data" / "weights" / "sam2.1_b.pt"
ARMS = ((512, 1), (256, 1))
MAX_DET = 1100
#: Masks are materialised at imgsz, ~5 MiB transient per detection, so a dense 512 slice (~1,000
#: detections on ESP_017355_2260) needs ~5 GB on its own. Batch size is recorded in every arm
#: manifest; it moves scores at the ~1e-4 level (measured 2026-10-14), far below anything notebook
#: 35 reads.
BATCH = {512: 1, 256: 8}


def crops(obs_filter=None):
    for obs_dir in sorted(p for p in ROM_DIR.iterdir() if p.is_dir()):
        if obs_filter and obs_dir.name not in obs_filter:
            continue
        for tif in sorted(obs_dir.glob(f"{obs_dir.name}_rom??_t*.tif")):
            yield obs_dir.name, tif


def outputs(crop):
    """``(config 2 fused layer, config 3 re-masked layer)`` for one crop."""
    stem = crop.stem
    return (crop.parent / "config2" / f"{stem}-predictions-fused-512-256-nms.gpkg",
            crop.parent / "config3" / f"{stem}-predictions-fused-512-256-sam2-nms.gpkg")


def output_1g(crop):
    """Configuration "1g", the 256 arm re-masked by SAM2 (the configuration study's C1G)."""
    return crop.parent / "config1g" / f"{crop.stem}-predictions-256-sam2-nms.gpkg"


def config1_layer(crop):
    """Configuration 1's catalog for this crop: the one ``-nms.gpkg`` beside it (non-recursive)."""
    hits = sorted(crop.parent.glob(f"{crop.stem}-predictions-*-nms.gpkg"))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one configuration-1 layer for {crop.stem}, found {hits}")
    return hits[0]


def done(crop, configs):
    fused_path, sam_path = outputs(crop)
    paths = {"2": fused_path, "3": sam_path, "1g": output_1g(crop)}
    return all(presets.manifest_path(paths[c]).exists() for c in configs)


def run_arm(model, crop, out_dir, slice_size):
    """One arm at the production preset. Returns ``(layer read back from disk, truncation)``.

    Read back rather than used in memory because that is the path the ratification validated: it
    fused the written layers (renamed columns and all), not the in-process frames.
    """
    import geopandas as gpd
    from YOLOv8BeyondEarth.predict import get_sliced_prediction, prediction_filenames
    cfg = {k: v for k, v in presets.GOAL1_PRODUCTION.items()
           if k not in ("slice_size", "sr_factor", "save_bbox", "save_prenms", "output_format",
                        "max_det", "batch_size")}
    _, nms = get_sliced_prediction(
        crop, model, output_dir=out_dir, slice_size=slice_size, sr_factor=1, max_det=MAX_DET,
        batch_size=BATCH[slice_size], save_bbox=False, save_prenms=False, save_nms=True,
        output_format="gpkg", on_conflict="overwrite", on_max_det="warn", **cfg)
    layer = Path(out_dir) / prediction_filenames(
        crop.stem, confidence_threshold=cfg["confidence_threshold"], slice_size=slice_size,
        inference_size=cfg["inference_size"], overlap_height_ratio=cfg["overlap_height_ratio"],
        sr_factor=1, downscale_pred=cfg["downscale_pred"], output_format="gpkg")["nms"]
    # a featureless layer is not written (the writers skip it): that is an empty catalog
    gdf = gpd.read_file(layer) if layer.exists() else nms.iloc[0:0]
    return gdf, dict(nms.attrs.get("truncation", {}))


def render_tile(crop, configs, device):
    """The child: render one crop's missing configurations, print a ``RESULT`` line, exit."""
    import geopandas as gpd
    from YOLOv8BeyondEarth.arm_ratification import arm_key, fuse_combo
    from YOLOv8BeyondEarth.polygon import outlines_to_shp

    t0 = time.monotonic()
    rec = {"tile": crop.stem, "obs": crop.parent.name}
    if "2" in configs or "3" in configs:
        _render_ladder(crop, configs, device, rec)
    if "1g" in configs and not presets.manifest_path(output_1g(crop)).exists():
        _render_1g(crop, rec)
    rec["seconds"] = round(time.monotonic() - t0, 1)
    print("RESULT " + json.dumps(rec, default=str), flush=True)


def _render_1g(crop, rec):
    """Configuration "1g": configuration 1's 256 catalog re-masked by SAM2, exactly as configuration 3
    re-masks configuration 2 (``remask_windows``, window 256, growth 1.0), so the two differ only by
    the 512 arm. YOLO is not run: the input is the catalog configuration 1 already rendered."""
    import geopandas as gpd
    from ultralytics import SAM
    from YOLOv8BeyondEarth import remask
    from YOLOv8BeyondEarth.polygon import outlines_to_shp

    base = config1_layer(crop)
    catalog = gpd.read_file(base)
    out_path = output_1g(crop)
    t0 = time.monotonic()
    out = remask.remask_windows(crop, catalog, SAM(str(SAM_WEIGHTS)), window_size=256,
                                max_window_growth=1.0, progress=False)
    sam_seconds = round(time.monotonic() - t0, 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    outlines_to_shp(out, out_path)
    presets.write_manifest(
        out_path,
        config={"configuration": "1g", "base": base.name, "remask": "remask_windows",
                "window_size": 256, "max_window_growth": 1.0,
                "note": "5a entry gate 3 open: see logit_px_per_native_px per row"},
        inputs={"crop": crop, "base": base, "sam_weights": SAM_WEIGHTS},
        outputs={"n": int(len(out))}, sam_seconds=sam_seconds,
        remask={k: v for k, v in out.attrs.items() if isinstance(v, (int, float, str, type(None)))})
    rec.update(n_config1=int(len(catalog)), n_config1g=int(len(out)), n_sam_1g=out.attrs.get("n_sam"),
               sam_seconds_1g=sam_seconds,
               min_ratio_1g=out.attrs.get("min_logit_px_per_native_px"))


def _render_ladder(crop, configs, device, rec):
    """Configurations 2 and 3 for one crop (2 is built whenever 3 needs it)."""
    import geopandas as gpd
    from YOLOv8BeyondEarth.arm_ratification import arm_key, fuse_combo
    from YOLOv8BeyondEarth.polygon import outlines_to_shp

    fused_path, sam_path = outputs(crop)
    if presets.manifest_path(fused_path).exists():
        fused = gpd.read_file(fused_path)
        rec["truncated_arms"] = json.loads(
            presets.manifest_path(fused_path).read_text()).get("truncated_arms", [])
    else:
        from sahi import AutoDetectionModel
        model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=str(WEIGHTS), device=device,
            image_size=presets.GOAL1_PRODUCTION["inference_size"],
            confidence_threshold=presets.GOAL1_PRODUCTION["confidence_threshold"])
        per_arm, truncated = {}, []
        for slice_size, sr in ARMS:
            layer, trunc = run_arm(model, crop, fused_path.parent / "arms", slice_size)
            if trunc:
                truncated.append({"arm": slice_size, "truncation": trunc})
            per_arm[arm_key(slice_size, sr)] = layer
        nonempty = [k for k, g in per_arm.items() if len(g)]
        if len(nonempty) >= 2:
            fused, _ = fuse_combo(per_arm, ARMS, native_res=abs(_res(crop)))
        else:                                   # aaf2d73: fewer than two arms found anything
            fused = per_arm[nonempty[0]] if nonempty else per_arm[arm_key(*ARMS[1])]
        fused_path.parent.mkdir(parents=True, exist_ok=True)
        outlines_to_shp(fused, fused_path)
        cfg = presets.GOAL3_PRODUCTION
        presets.write_manifest(
            fused_path,
            config={"configuration": 2, "arms": [list(a) for a in ARMS], "max_det": MAX_DET,
                    "batch_size": BATCH, "fusion": "arm_ratification.fuse_combo",
                    **{k: cfg[k] for k in ("rule", "bands", "engulf_thr", "metric",
                                           "cross_arm_only")}},
            inputs={"crop": crop, "weights": WEIGHTS},
            outputs={"n_fused": int(len(fused))}, truncated_arms=truncated)
        rec["truncated_arms"] = truncated
        fused = gpd.read_file(fused_path) if fused_path.exists() else fused   # what config 3 reads
    rec["n_config2"] = int(len(fused))

    if "3" in configs and not presets.manifest_path(sam_path).exists():
        from ultralytics import SAM
        from YOLOv8BeyondEarth import remask
        out = remask.remask_windows(crop, fused, SAM(str(SAM_WEIGHTS)), window_size=256,
                                    max_window_growth=1.0, progress=False)
        sam_path.parent.mkdir(parents=True, exist_ok=True)
        outlines_to_shp(out, sam_path)
        presets.write_manifest(
            sam_path,
            config={"configuration": 3, "base": fused_path.name, "remask": "remask_windows",
                    "window_size": 256, "max_window_growth": 1.0,
                    "note": "5a entry gate 3 open: see logit_px_per_native_px per row"},
            inputs={"crop": crop, "fused": fused_path, "sam_weights": SAM_WEIGHTS},
            outputs={"n": int(len(out))},
            remask={k: v for k, v in out.attrs.items()
                    if isinstance(v, (int, float, str, type(None)))},
            truncated_arms=rec["truncated_arms"])
        rec.update(n_config3=int(len(out)), n_sam=out.attrs.get("n_sam"),
                   below_unit_ratio=out.attrs.get("n_prompted_below_unit_logit_ratio"),
                   min_ratio=out.attrs.get("min_logit_px_per_native_px"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--obs", nargs="*", help="only these observations")
    ap.add_argument("--limit", type=int, help="at most this many tiles (smoke test)")
    ap.add_argument("--configs", nargs="+", default=["2", "3"], choices=["2", "3", "1g"],
                    help="'1g' = configuration 1's 256 catalog re-masked by SAM2 (no YOLO run)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tile", help=argparse.SUPPRESS)            # internal: the child
    args = ap.parse_args(argv)
    if args.tile:
        return render_tile(Path(args.tile), args.configs, args.device)

    summary, t_all = [], time.monotonic()
    todo = list(crops(args.obs))[: args.limit]
    print(f"{len(todo)} tiles, configurations {args.configs}", flush=True)
    for obs, crop in todo:
        if done(crop, args.configs):
            print(f"  [skip] {crop.stem}", flush=True)
            continue
        res = subprocess.run([sys.executable, "-u", __file__, "--tile", str(crop), "--device",
                              args.device, "--configs", *map(str, args.configs)],
                             capture_output=True, text=True)
        line = [x for x in res.stdout.splitlines() if x.startswith("RESULT ")]
        if res.returncode == 0 and line:
            rec = json.loads(line[-1][len("RESULT "):])
            flag = f" TRUNCATED {rec['truncated_arms']}" if rec.get("truncated_arms") else ""
            print(f"  [ok]   {crop.stem}  c2={rec.get('n_config2', '-')}  "
                  f"c3={rec.get('n_config3', '-')}  c1g={rec.get('n_config1g', '-')}  "
                  f"ratio>={rec.get('min_ratio', rec.get('min_ratio_1g', '-'))}  "
                  f"{rec['seconds']}s{flag}", flush=True)
        else:
            tail = (res.stderr or "").strip().splitlines()
            oom = "out of memory" in (res.stderr or "").lower()
            rec = {"tile": crop.stem, "obs": obs,
                   "failed": ("CUDA out of memory on this card: render on a 24 GiB GPU (Sherlock)"
                              if oom else (tail[-1] if tail else f"exit {res.returncode}"))}
            print(f"  [FAIL] {crop.stem}: {rec['failed']}", flush=True)
        summary.append(rec)
    # merge with earlier runs, so a resumed run (which skips finished tiles) does not erase them
    path = ROM_DIR / "render_rom_configs_summary.json"
    tiles = {}
    if path.exists():
        tiles = {t["tile"]: t for t in json.loads(path.read_text()).get("tiles", [])}
    for t in summary:            # merge per tile: a "1g" run must not erase the tile's 2/3 record
        tiles[t["tile"]] = {**tiles.get(t["tile"], {}), **t}
    path.write_text(json.dumps(
        {"written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "tiles": [tiles[k] for k in sorted(tiles)], "code": presets.code_identity()},
        indent=2, default=str))
    print(f"TOTAL {time.monotonic() - t_all:.1f}s", flush=True)


def _res(path):
    import rasterio
    with rasterio.open(path) as src:
        return src.transform.a


if __name__ == "__main__":
    main()
