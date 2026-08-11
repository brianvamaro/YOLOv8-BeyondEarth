"""One ladder runner, shared by `run_hseg_ladder.py` and `run_goal3_ladder.py`.

**Why this is shared rather than copied.** H-seg x G needs a *matched* production control — the
production detector put through an identical configuration, so a SAM gain cannot be misattributed
to the detector. Until 2026-08-09 the two ladders were separate scripts that happened to contain
**byte-identical** `get_sliced_prediction` calls, which made "matched" a property of two files
agreeing rather than a property of the code. Anyone adding `remask_fn` to one and not the other, or
retuning conf in one, would have produced a silently unmatched comparison. One runner makes it
structural: the arms can differ only in what the caller is *supposed* to vary — weights, confidence,
re-masker, output location.

It also centralises the resume check, which has to know the output filename and therefore has to
follow `predict.prediction_filenames`. The old copies hardcoded `ct-010` and an untagged mask, so a
retuned-conf or re-masked run could not be resumed correctly.
"""
import json
import time
from pathlib import Path

DATA = Path(r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\data")
DE = DATA / "solutionDE"
SAM_WEIGHTS = DATA / "weights" / "sam2.1_b.pt"

#: The configuration both ladders have used since 2026-07-10. It is deliberately **not** the
#: production preset: `min_area_threshold` None vs 6, `postprocess_match_threshold` 0.5 vs 0.2 and
#: `max_det` 300 vs 600 (G1-01; found again 2026-08-09). It stays the default so every catalog
#: already on disk remains reproducible and the existing H-seg verdict keeps its baseline — pass
#: `--preset goal1_production` to run the shipped configuration instead. Whichever is used gets
#: printed at the top of the run and written into the timings record.
LEGACY_LADDER = dict(inference_size=1024, overlap_height_ratio=0.2, overlap_width_ratio=0.2,
                     downscale_pred=True, postprocess=True)


def add_ladder_args(p):
    """The flags both ladders expose. Identical by construction, for the same reason as above."""
    p.add_argument("--conf", type=float, default=0.10,
                   help="confidence threshold (default 0.10). Retuning this for a retrained weight "
                        "is a PREREQUISITE of the H-seg x G comparison, not a refinement — a "
                        "retrained model's score distribution shifts.")
    p.add_argument("--remask", action="store_true",
                   help="apply Solution G (SAM2 detector-preserving re-masking). Writes to a "
                        "distinct '-mask-sam2-' catalog, so it cannot overwrite a plain run.")
    p.add_argument("--dir-tag", default="",
                   help="inserted into each output subdirectory name, e.g. '_g' for a G run")
    p.add_argument("--preset", default=None, choices=[None, "goal1_production"],
                   help="run the named production preset instead of the legacy ladder config. "
                        "Default (None) reproduces every catalog currently on disk.")
    p.add_argument("--device", default="cuda:0")
    return p


def _remasker(device):
    from ultralytics import SAM
    from YOLOv8BeyondEarth.orientation_validation import sam2_remasker
    dev = 0 if str(device).startswith("cuda") else device
    return sam2_remasker(SAM(str(SAM_WEIGHTS)), device=dev)


def run_ladder(*, weights, arms, timings_path, args, key_fn):
    """Run every (subdir, raster, slice_size, sr_factor) in ``arms``; skip what already exists.

    ``key_fn(sub, ss, sr) -> str`` names the timings entry; the two ladders spell it differently
    and both spellings are already on disk, so it stays caller-supplied.
    """
    from sahi import AutoDetectionModel
    from YOLOv8BeyondEarth.predict import (get_sliced_prediction, predict_production,
                                           prediction_filenames)

    mask_tag = "sam2" if args.remask else None
    cfg_name = args.preset or "legacy_ladder"
    print(f"[config] weights={Path(weights).name} conf={args.conf} preset={cfg_name} "
          f"remask={bool(args.remask)} mask_tag={mask_tag} dir_tag={args.dir_tag!r}", flush=True)
    if args.preset is None:
        print("[config] NOTE this is the historical ladder config, NOT the shipped production one "
              "(min_area_threshold None vs 6, nms 0.5 vs 0.2, max_det 300 vs 600). Intentional — "
              "it is what every catalog on disk was made with. Pass --preset goal1_production to "
              "run the shipped config.", flush=True)

    model = AutoDetectionModel.from_pretrained(
        model_type="yolov8", model_path=str(weights), device=args.device,
        image_size=1024, confidence_threshold=args.conf)
    remask_fn = _remasker(args.device) if args.remask else None

    timings = json.loads(Path(timings_path).read_text()) if Path(timings_path).exists() else {}
    for sub, raster, ss, sr in arms:
        key = key_fn(sub, ss, sr)
        out_dir = DE / "preds" / sub
        out_dir.mkdir(parents=True, exist_ok=True)
        # The resume check must follow the SAME naming the writer uses, or a re-masked / retuned
        # run either re-does finished work or, worse, is skipped against a catalog that is not it.
        nms = out_dir / prediction_filenames(
            Path(raster).stem, confidence_threshold=args.conf, slice_size=ss, inference_size=1024,
            overlap_height_ratio=0.2, sr_factor=sr, downscale_pred=True, mask_tag=mask_tag,
            output_format="gpkg")["nms"]
        if nms.exists():
            print(f"[skip] {key}", flush=True)
            continue
        print(f"[run ] {key}: {Path(raster).name} ss={ss} sr={sr}", flush=True)
        t0 = time.time()
        common = dict(output_dir=out_dir, remask_fn=remask_fn, mask_tag=mask_tag,
                      save_prenms=False, save_bbox=False, output_format="gpkg")
        if args.preset is None:
            gdf, gdf_nms = get_sliced_prediction(
                raster, detection_model=model, confidence_threshold=args.conf,
                slice_size=ss, sr_factor=sr, **LEGACY_LADDER, **common)
        else:
            gdf, gdf_nms = predict_production(
                raster, model, out_dir, preset=args.preset, confidence_threshold=args.conf,
                slice_size=ss, sr_factor=sr,
                **{k: v for k, v in common.items() if k != "output_dir"})
        # X-06: slice-level truncation is invisible in the row count, so record it beside the wall
        # clock. An empty dict is the acceptance condition.
        timings[key] = {"wall_s": round(time.time() - t0, 2), "slice_size": ss, "sr_factor": sr,
                        "raster": Path(raster).name, "n_raw": int(len(gdf)),
                        "n_nms": int(len(gdf_nms)), "conf": args.conf, "config": cfg_name,
                        "mask_tag": mask_tag, "weights": Path(weights).name,
                        "truncation": gdf.attrs.get("truncation", {})}
        if timings[key]["truncation"]:
            print(f"[WARN] {key} lost detections to max_det: {timings[key]['truncation']}",
                  flush=True)
        Path(timings_path).write_text(json.dumps(timings, indent=2))
        print(f"[done] {key}: {timings[key]}", flush=True)

    print(json.dumps(timings, indent=2))
    return timings
