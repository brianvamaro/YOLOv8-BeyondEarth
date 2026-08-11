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

#: Every ladder catalog is named `...-is-1024-...`, and so is the analyzer's reader, so the ladder
#: owns this rather than inheriting it. Asserted against the preset below: if production ever moves
#: off 1024 that must be a loud failure, not a silently unfindable catalog.
LADDER_INFERENCE_SIZE = 1024

#: The configuration both ladders have used since 2026-07-10. It is deliberately **not** the
#: production preset: `min_area_threshold` None vs 6, `postprocess_match_threshold` 0.5 vs 0.2 and
#: `max_det` 300 vs 600 (G1-01; found again 2026-08-09). It stays the default so every catalog
#: already on disk remains reproducible and the existing H-seg verdict keeps its baseline — pass
#: `--preset goal1_production` to run the shipped configuration instead. Whichever is used gets
#: printed at the top of the run and written into the timings record.
#: The output-shape keys are spelled out here because the production preset defines its own, and
#: the two must be comparable key-for-key rather than one of them arriving as a function default.
LEGACY_LADDER = dict(overlap_height_ratio=0.2, overlap_width_ratio=0.2,
                     downscale_pred=True, postprocess=True,
                     save_bbox=False, save_prenms=False, output_format="gpkg")

#: Keys the ladder supplies itself, so they must be stripped from a preset before it is forwarded.
_LADDER_OWNED = ("slice_size", "sr_factor", "confidence_threshold", "inference_size")


def ladder_config(preset):
    """The `get_sliced_prediction` kwargs for this run, minus the ones the ladder owns.

    Follows `scripts/run_multiscale_arms.py:78-85` rather than `predict_production`. `resolve()`
    rejects any override key the preset does not define, and **`sr_factor` is not a preset key** —
    so the SR arm cannot go through `predict_production` at all. (`output_format`, `save_bbox` and
    `save_prenms` *are* preset keys, which is why they are spelled out in `LEGACY_LADDER` too: the
    two configurations have to be comparable key-for-key.)
    """
    if preset is None:
        return dict(LEGACY_LADDER)
    from YOLOv8BeyondEarth import presets
    base = presets.PRESETS[preset]
    if base.get("inference_size", LADDER_INFERENCE_SIZE) != LADDER_INFERENCE_SIZE:
        raise ValueError(
            f"preset {preset!r} has inference_size={base['inference_size']}, but every ladder "
            f"catalog is named '-is-{LADDER_INFERENCE_SIZE}-' and the analyzer reads that name. "
            "Update LADDER_INFERENCE_SIZE and the analyzer's catalog_name together.")
    return {k: v for k, v in base.items() if k not in _LADDER_OWNED and not k.startswith("_")}


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
    p.add_argument("--batch-size", type=int, default=None,
                   help="override the config's batch size. Needed at low confidence: this is an "
                        "8 GiB card and max_det=600 at batch 8 already peaks near 7.7 GiB, so a "
                        "conf-0.02 sweep point OOMs. Count parity is exact across batch sizes "
                        "(G1-02) but geometry differs by ~0.70%%, so it is treated as "
                        "result-defining and a run cannot silently mix batches.")
    p.add_argument("--trust-existing", action="store_true",
                   help="skip catalogs that have no manifest even under a non-default --preset. "
                        "Only pass this if you know the files on disk were made with the "
                        "configuration you are asking for; otherwise they get relabelled.")
    return p


#: Config keys that define the result but are **absent from the filename**, so two runs that differ
#: only in these collide by name. Exactly the legacy-vs-production deltas, which is the point.
#: `batch_size` is here because G1-02 measured batch 1 vs 8 at **exact count parity but 0.70%
#: differing geometry** — so it is result-defining for anything that matches polygons, and it is
#: no more present in the filename than the others.
_SILENT_KEYS = ("min_area_threshold", "postprocess_match_threshold", "max_det",
                "contour_method", "postprocess_class_agnostic", "batch_size")


#: Returned when an existing catalog carries no usable record of how it was made.
UNVERIFIABLE = "<no manifest>"


def _config_mismatch(nms_path, cfg, trust_existing=False):
    """Compare an existing catalog's manifest against the config this run would use.

    The resume check short-circuits before `get_sliced_prediction` can call
    `presets.check_config_conflict`, so without this a `--preset goal1_production` run would
    happily 'skip' a legacy catalog sitting under the same name and report it as its own.

    Returns None when they agree, :data:`UNVERIFIABLE` when nothing is recorded (the caller decides
    — that is not automatically safe), or a human-readable diff string.
    """
    from YOLOv8BeyondEarth import presets
    if trust_existing:
        return None
    mpath = presets.manifest_path(nms_path)
    if not mpath.exists():
        return UNVERIFIABLE
    try:
        prev = json.loads(mpath.read_text()).get("config", {})
    except (OSError, ValueError):
        return UNVERIFIABLE
    if not prev:
        return UNVERIFIABLE
    from YOLOv8BeyondEarth.predict import ULTRALYTICS_DEFAULT_MAX_DET
    # What `get_sliced_prediction` uses when the config does not name the key — i.e. what the
    # legacy ladder actually ran with. Must cover every _SILENT_KEYS entry or the lookup below
    # raises KeyError on a legacy run; `test_ladder_config.py` asserts the two stay in step.
    fallback = {"min_area_threshold": None, "postprocess_match_threshold": 0.5,
                "max_det": ULTRALYTICS_DEFAULT_MAX_DET, "contour_method": "skimage",
                "postprocess_class_agnostic": False, "batch_size": 8}
    diffs = []
    for k in _SILENT_KEYS:
        if k not in prev:
            continue
        want = cfg.get(k, fallback[k])
        got = prev[k]
        if isinstance(want, float) or isinstance(got, float):
            if want is not None and got is not None and abs(float(want) - float(got)) < 1e-9:
                continue
        if want != got:
            diffs.append(f"{k} on disk={got!r} this run={want!r}")
    return "; ".join(diffs) or None


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
    from YOLOv8BeyondEarth.predict import get_sliced_prediction, prediction_filenames

    mask_tag = "sam2" if args.remask else None
    cfg_name = args.preset or "legacy_ladder"
    cfg = ladder_config(args.preset)
    if getattr(args, "batch_size", None):
        cfg["batch_size"] = int(args.batch_size)
    print(f"[config] weights={Path(weights).name} conf={args.conf} preset={cfg_name} "
          f"remask={bool(args.remask)} mask_tag={mask_tag} dir_tag={args.dir_tag!r}", flush=True)
    print(f"[config] {json.dumps({k: str(v) for k, v in sorted(cfg.items())})}", flush=True)
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
            Path(raster).stem, confidence_threshold=args.conf, slice_size=ss,
            inference_size=LADDER_INFERENCE_SIZE,
            overlap_height_ratio=cfg["overlap_height_ratio"], sr_factor=sr,
            downscale_pred=cfg["downscale_pred"], mask_tag=mask_tag, output_format="gpkg")["nms"]
        if nms.exists():
            stale = _config_mismatch(nms, cfg, trust_existing=args.trust_existing)
            if stale == UNVERIFIABLE and args.preset is None:
                # The legacy default IS what every un-manifested catalog on disk was made with,
                # so "no manifest" is the expected, correct state for a bare run.
                stale = None
            if stale == UNVERIFIABLE:
                # Every catalog written before the manifest machinery — which is ALL of the
                # 2026-07-10 ladder output — carries no record of its configuration. So for
                # precisely the files most at risk there is nothing to compare against, and a
                # non-default run would otherwise "skip" a legacy catalog and report it as its own.
                # Refuse instead. In the intended workflow the new config goes to a fresh
                # --dir-tag, so this never fires.
                raise SystemExit(
                    f"[STOP] {key}: {nms.name} exists but has NO manifest, so its configuration "
                    f"cannot be verified, and this run is not the legacy default "
                    f"(preset={cfg_name}). Skipping it would silently score a legacy catalog as "
                    f"{cfg_name}. Use a fresh --dir-tag, or pass --trust-existing if you are "
                    "certain the file on disk was made with this configuration.")
            if stale:
                raise SystemExit(
                    f"[STOP] {key}: {nms.name} already exists but was produced by a DIFFERENT "
                    f"configuration: {stale}.\nThe filename does not encode these parameters, so "
                    "resuming here would score a legacy catalog as if it were this run — which is "
                    "exactly the mislabelling `presets.check_config_conflict` exists to prevent, "
                    "and the resume check runs before the writer can enforce it. Use a fresh "
                    "--dir-tag for the new configuration, or delete the old catalog deliberately.")
            print(f"[skip] {key}", flush=True)
            continue
        print(f"[run ] {key}: {Path(raster).name} ss={ss} sr={sr}", flush=True)
        t0 = time.time()
        # One call for both configurations — they differ only in `cfg`, which is the whole point of
        # naming the choice instead of having two code paths that can drift.
        gdf, gdf_nms = get_sliced_prediction(
            raster, detection_model=model, confidence_threshold=args.conf,
            slice_size=ss, sr_factor=sr, inference_size=LADDER_INFERENCE_SIZE,
            output_dir=out_dir, remask_fn=remask_fn, mask_tag=mask_tag, **cfg)
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
