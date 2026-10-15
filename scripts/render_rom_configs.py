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

    conda run -n bouldernet --no-capture-output python -u scripts/render_rom_configs.py [--obs ESP_028537_2270] [--limit 1]
"""
from __future__ import annotations

import argparse
import json
import os
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
#: Masks are materialised at imgsz, ~5 MiB transient per detection, so a batch of dense 512 slices
#: (~1,000 detections each on ESP_017355_2260) is ~5 GB *per slice*: batch 2 hard-OOMed the 8 GiB
#: laptop card on that scene's second tile (2026-10-14). Batch size is recorded in every arm manifest;
#: it moves scores at the ~1e-4 level (measured the same day), far below anything notebook 35 reads.
BATCH = {512: 1, 256: 8}


class ArmOutOfMemory(RuntimeError):
    """An arm that does not fit this GPU even at batch 1 -- record the tile, do not kill the run."""


def _is_cuda_oom(exc):
    return "out of memory" in str(exc).lower()


def crops(obs_filter=None):
    for obs_dir in sorted(p for p in ROM_DIR.iterdir() if p.is_dir()):
        if obs_filter and obs_dir.name not in obs_filter:
            continue
        for tif in sorted(obs_dir.glob(f"{obs_dir.name}_rom??_t*.tif")):
            yield obs_dir.name, tif


def run_arm(model, crop, out_dir, slice_size):
    """One arm at the production preset. Returns ``(layer read back from disk, truncation)``.

    Read back rather than used in memory because that is the path the ratification validated: it
    fused the written layers (renamed columns and all), not the in-process frames.
    """
    import geopandas as gpd
    from YOLOv8BeyondEarth.predict import get_sliced_prediction, prediction_filenames
    import torch
    cfg = {k: v for k, v in presets.GOAL1_PRODUCTION.items()
           if k not in ("slice_size", "sr_factor", "save_bbox", "save_prenms", "output_format",
                        "max_det", "batch_size")}
    for batch in dict.fromkeys((BATCH[slice_size], 1)):         # once more at batch 1 on OOM
        try:
            _, nms = get_sliced_prediction(
                crop, model, output_dir=out_dir, slice_size=slice_size, sr_factor=1,
                max_det=MAX_DET, batch_size=batch, save_bbox=False, save_prenms=False,
                save_nms=True, output_format="gpkg", on_conflict="overwrite", on_max_det="warn",
                **cfg)
            break
        except Exception as exc:                                  # torch raises two OOM types
            if not _is_cuda_oom(exc):
                raise
            torch.cuda.empty_cache()
            print(f"    OOM: {crop.stem} arm {slice_size} at batch {batch}", flush=True)
    else:
        raise ArmOutOfMemory(f"arm {slice_size} does not fit this GPU even at batch 1")
    layer = Path(out_dir) / prediction_filenames(
        crop.stem, confidence_threshold=cfg["confidence_threshold"], slice_size=slice_size,
        inference_size=cfg["inference_size"], overlap_height_ratio=cfg["overlap_height_ratio"],
        sr_factor=1, downscale_pred=cfg["downscale_pred"], output_format="gpkg")["nms"]
    # a featureless layer is not written (the writers skip it): that is an empty catalog
    gdf = gpd.read_file(layer) if layer.exists() else nms.iloc[0:0]
    return gdf, dict(nms.attrs.get("truncation", {}))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--obs", nargs="*", help="only these observations")
    ap.add_argument("--limit", type=int, help="at most this many tiles (smoke test)")
    ap.add_argument("--configs", nargs="+", type=int, default=[2, 3], choices=[2, 3])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)

    import geopandas as gpd
    from sahi import AutoDetectionModel
    from YOLOv8BeyondEarth.arm_ratification import arm_key, fuse_combo
    from YOLOv8BeyondEarth.polygon import outlines_to_shp

    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=str(WEIGHTS), device=args.device,
        image_size=presets.GOAL1_PRODUCTION["inference_size"],
        confidence_threshold=presets.GOAL1_PRODUCTION["confidence_threshold"])
    sam = None
    summary, t_all = [], time.monotonic()
    todo = list(crops(args.obs))[: args.limit]
    print(f"{len(todo)} tiles, configurations {args.configs}", flush=True)
    for obs, crop in todo:
        stem = crop.stem
        c2_dir, c3_dir = crop.parent / "config2", crop.parent / "config3"
        fused_path = c2_dir / f"{stem}-predictions-fused-512-256-nms.gpkg"
        sam_path = c3_dir / f"{stem}-predictions-fused-512-256-sam2-nms.gpkg"
        need2 = 2 in args.configs and not presets.manifest_path(fused_path).exists()
        need3 = 3 in args.configs and not presets.manifest_path(sam_path).exists()
        if not (need2 or need3):
            print(f"  [skip] {stem}", flush=True)
            continue
        t0 = time.monotonic()
        rec = {"tile": stem, "obs": obs}

        if presets.manifest_path(fused_path).exists():
            fused = gpd.read_file(fused_path)
            rec["truncated_arms"] = json.loads(
                presets.manifest_path(fused_path).read_text()).get("truncated_arms", [])
        else:
            per_arm, truncated = {}, []
            try:
                for slice_size, sr in ARMS:
                    layer, trunc = run_arm(model, crop, c2_dir / "arms", slice_size)
                    if trunc:
                        truncated.append({"arm": slice_size, "truncation": trunc})
                    per_arm[arm_key(slice_size, sr)] = layer
            except ArmOutOfMemory as exc:
                rec["failed"] = f"{exc} -- render this tile on a 24 GiB card (Sherlock)"
                summary.append(rec)
                print(f"  [OOM]  {stem}: {exc}", flush=True)
                continue
            nonempty = [k for k, g in per_arm.items() if len(g)]
            if len(nonempty) >= 2:
                fused, _ = fuse_combo(per_arm, ARMS, native_res=abs(_res(crop)))
            else:                               # aaf2d73: fewer than two arms found anything
                fused = per_arm[nonempty[0]] if nonempty else per_arm[arm_key(*ARMS[1])]
            c2_dir.mkdir(parents=True, exist_ok=True)
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

        if need3:
            from ultralytics import SAM
            from YOLOv8BeyondEarth import remask
            if sam is None:
                sam = SAM(str(SAM_WEIGHTS))
            out = remask.remask_windows(crop, fused, sam, window_size=256, max_window_growth=1.0,
                                        progress=False)
            c3_dir.mkdir(parents=True, exist_ok=True)
            outlines_to_shp(out, sam_path)
            presets.write_manifest(
                sam_path,
                config={"configuration": 3, "base": fused_path.name, "remask": "remask_windows",
                        "window_size": 256, "max_window_growth": 1.0,
                        "note": "5a entry gate 3 open: see logit_px_per_native_px per row"},
                inputs={"crop": crop, "fused": fused_path, "sam_weights": SAM_WEIGHTS},
                outputs={"n": int(len(out))},
                remask={k: v for k, v in out.attrs.items() if isinstance(v, (int, float, str, type(None)))},
                truncated_arms=rec["truncated_arms"])
            rec.update(n_config3=int(len(out)), n_sam=out.attrs.get("n_sam"),
                       below_unit_ratio=out.attrs.get("n_prompted_below_unit_logit_ratio"))
        rec["seconds"] = round(time.monotonic() - t0, 1)
        summary.append(rec)
        flag = " TRUNCATED " + str(rec["truncated_arms"]) if rec["truncated_arms"] else ""
        print(f"  [ok]   {stem}  c2={rec['n_config2']}  c3={rec.get('n_config3', '-')}  "
              f"sam={rec.get('n_sam', '-')} <1.0={rec.get('below_unit_ratio', '-')}  "
              f"{rec['seconds']}s{flag}", flush=True)
    (ROM_DIR / "render_rom_configs_summary.json").write_text(json.dumps(
        {"written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tiles": summary,
         "code": presets.code_identity()}, indent=2, default=str))
    print(f"TOTAL {time.monotonic() - t_all:.1f}s", flush=True)


def _res(path):
    import rasterio
    with rasterio.open(path) as src:
        return src.transform.a


if __name__ == "__main__":
    main()
