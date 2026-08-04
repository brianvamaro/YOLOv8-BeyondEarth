"""Full-scene multiscale inference, one subprocess per arm (review items G3-08 and G3-09).

``get_multiscale_prediction`` keeps every arm's catalog and the pooled copy resident in one
process. That is fine on a crop and ruinous on a scene: phase 4 measured 24-78 GB of commit growth
and concluded one subprocess per arm, because CPython does not return freed pymalloc arenas to the
OS and a long-lived process ratchets. This runner enforces that — each arm runs in its own
interpreter, writes its layer, and exits, so the memory goes back at the process boundary — then
fuses the persisted layers in a final pass.

Resume is ledger-backed (``YOLOv8BeyondEarth.runstate``), not existence-backed: an arm counts as
done only when its layer *and* its ledger entry agree and the layer still checksums to what was
recorded. Re-running one arm marks the fuse stale, so the fused catalog cannot silently descend
from a mixture of old and new arms.

    conda run -n bouldernet python scripts/run_multiscale_arms.py \
        --raster data/mars_hirise/raster/ESP_017355_2260_RED.tif --out-dir data/fullscene

    # one arm, in its own process (what --raster above spawns for you)
    conda run -n bouldernet python scripts/run_multiscale_arms.py --arm 256:1 ...
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from YOLOv8BeyondEarth import presets                                   # noqa: E402
from YOLOv8BeyondEarth.runstate import COMPLETE, StageLedger, stage_key  # noqa: E402


def parse_arm(text):
    """``"256"`` -> (256, 1); ``"256:2"`` -> (256, 2)."""
    slice_size, _, sr = text.partition(":")
    return int(slice_size), int(sr or 1)


def arm_layer_path(out_dir, raster, slice_size, sr_factor, conf, imgsz, overlap, ext=".gpkg"):
    ss = str(slice_size) if sr_factor == 1 else f"{slice_size}sr{sr_factor}"
    stem = (f"{Path(raster).stem}-predictions-ct-{int(conf * 100):03d}-ss-{ss}"
            f"-is-{imgsz}-ov-{int(overlap * 100):03d}")
    return Path(out_dir) / f"{stem}-downscaled-mask-nms{ext}"


def run_one_arm(args):
    """The child process: one arm, one layer, then exit so the memory is actually returned."""
    from sahi import AutoDetectionModel
    from YOLOv8BeyondEarth.predict import get_sliced_prediction

    slice_size, sr_factor = parse_arm(args.arm)
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=args.weights, device=args.device,
        image_size=args.inference_size, confidence_threshold=args.conf)
    cfg = {k: v for k, v in presets.GOAL1_PRODUCTION.items()
           if k not in ("slice_size", "sr_factor", "confidence_threshold", "inference_size",
                        "save_bbox", "save_prenms", "output_format")}
    gdf, gdf_nms = get_sliced_prediction(
        args.raster, model, confidence_threshold=args.conf, output_dir=args.out_dir,
        slice_size=slice_size, sr_factor=sr_factor, inference_size=args.inference_size,
        save_bbox=False, save_prenms=False, save_nms=True, output_format="gpkg",
        on_conflict=args.on_conflict, on_max_det=args.on_max_det, **cfg)
    print(json.dumps({"arm": args.arm, "n": int(len(gdf_nms if gdf_nms is not None else gdf)),
                      "truncation": dict(gdf.attrs.get("truncation", {}))}))


def orchestrate(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = StageLedger(out_dir / "run_ledger.jsonl")
    raster_id = presets.file_identity(args.raster)
    weights_id = presets.file_identity(args.weights)

    arm_keys, arm_layers = {}, {}
    for arm in args.arms:
        slice_size, sr_factor = parse_arm(arm)
        # `preset="goal1_production"` is a *name*, so on its own it cannot notice that the preset's
        # values changed underneath a resumed campaign. `presets.PRESET_VERSION` is bumped whenever
        # they do (that is its documented contract), and `max_det` is spelled out because X-06 made
        # it result-defining and a stale ledger entry would otherwise let a 300-capped arm satisfy a
        # run configured for a higher cap.
        key = stage_key(raster=raster_id.get("sha256"), weights=weights_id.get("sha256"),
                        slice_size=slice_size, sr_factor=sr_factor, conf=args.conf,
                        imgsz=args.inference_size, preset="goal1_production",
                        preset_version=presets.PRESET_VERSION,
                        max_det=presets.GOAL1_PRODUCTION["max_det"])
        layer = arm_layer_path(out_dir, args.raster, slice_size, sr_factor, args.conf,
                               args.inference_size, presets.GOAL1_PRODUCTION["overlap_height_ratio"])
        arm_keys[arm], arm_layers[arm] = key, layer

        state, detail = ledger.status(f"arm:{arm}", key)
        if state == COMPLETE:
            print(f"[skip] arm {arm}: {detail}")
            continue
        print(f"[run ] arm {arm}: {state} ({detail})")
        cmd = [sys.executable, str(Path(__file__).resolve()), "--arm", arm,
               "--raster", str(args.raster), "--weights", str(args.weights),
               "--out-dir", str(out_dir), "--conf", str(args.conf),
               "--inference-size", str(args.inference_size), "--device", args.device,
               "--on-conflict", "overwrite"]
        with ledger.stage(f"arm:{arm}", key, artifacts=[layer], command=cmd):
            subprocess.run(cmd, check=True)

    # Depend on each arm's key *and* the artifact digest its ledger entry recorded, not on the key
    # alone: the key is a hash of the configuration, so an arm re-executed with unchanged config
    # keeps it and the fuse would skip while its input had been regenerated underneath it.
    arm_deps = {arm: ledger.dependency_value(f"arm:{arm}", key) for arm, key in arm_keys.items()}
    fuse_key = stage_key(arms=sorted(arm_deps.values()), preset="goal3_production")
    fused_path = out_dir / f"{Path(args.raster).stem}-predictions-fused-production-mask.gpkg"
    state, detail = ledger.status("fuse", fuse_key, depends_on=arm_deps)
    if state == COMPLETE:
        print(f"[skip] fuse: {detail}")
        return
    print(f"[run ] fuse: {state} ({detail})")

    import geopandas as gpd
    from rastertools_BOULDERING import metadata as raster_metadata
    from YOLOv8BeyondEarth.fusion import fuse_predictions
    from YOLOv8BeyondEarth.polygon import outlines_to_shp

    with ledger.stage("fuse", fuse_key, artifacts=[fused_path], depends_on=arm_deps):
        per_scale = {parse_arm(a): gpd.read_file(p) for a, p in arm_layers.items()}
        cfg = presets.GOAL3_PRODUCTION
        fused, audit = fuse_predictions(
            per_scale, rule=cfg["rule"], bands=cfg["bands"], engulf_thr=cfg["engulf_thr"],
            metric=cfg["metric"], cross_arm_only=cfg["cross_arm_only"],
            native_res=raster_metadata.get_resolution(args.raster)[0])
        outlines_to_shp(fused, fused_path)
        audit.to_csv(fused_path.with_suffix("").as_posix() + "-audit.csv", index=False)
        presets.write_manifest(fused_path, config={"preset": "goal3_production", **cfg},
                               inputs={"raster": args.raster, "weights": args.weights},
                               outputs={"n_fused": int(len(fused)),
                                        "n_suppressed": int(len(audit))},
                               arm_keys=arm_keys)
    print(f"wrote {fused_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raster", required=True)
    ap.add_argument("--weights",
                    default=str(REPO / "data" / "weights" / "yolov8-m-boulder-detection-tmp.pt"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arms", nargs="+", default=["512", "256", "256:2"])
    ap.add_argument("--arm", default=None, help="internal: run this single arm and exit")
    ap.add_argument("--conf", type=float, default=presets.GOAL1_PRODUCTION["confidence_threshold"])
    ap.add_argument("--inference-size", type=int,
                    default=presets.GOAL1_PRODUCTION["inference_size"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--on-conflict", default="error", choices=("error", "overwrite"))
    ap.add_argument("--on-max-det", default="error", choices=("error", "warn"),
                    help="'error' fails in the first minutes of an arm rather than writing a "
                         "silently truncated catalog (review item X-06)")
    args = ap.parse_args()
    run_one_arm(args) if args.arm else orchestrate(args)


if __name__ == "__main__":
    main()
