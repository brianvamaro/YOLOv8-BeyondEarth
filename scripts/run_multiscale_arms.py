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

Three modes, and the third exists because resume is deliberately machine-local:

* default — run any arm that is not already complete, then fuse.
* ``--no-fuse`` — run the arms and stop, for a ladder that cannot be completed here.
* ``--fuse-only`` — fuse the layers already on disk, computing nothing. The default mode can only
  fuse arms *it* produced, since an arm's ledger entry is keyed on this machine's code identity; a
  layer computed elsewhere never matches and would be silently recomputed here, which on a
  heterogeneous ladder means either recomputing the 512 arm where it does not fit or relocating the
  load-bearing arms to the other stack. ``--fuse-only`` validates coherence from each layer's
  manifest instead and records per-arm provenance in the fused one. It is also the cheap way to
  compare fusion choices — re-fusing is ~112 s against arms that cost hours.

    # assemble a ladder whose 512 arm was produced on a larger GPU
    conda run -n bouldernet python scripts/run_multiscale_arms.py --fuse-only \
        --raster ... --out-dir data/fullscene
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
from YOLOv8BeyondEarth.runstate import (COMPLETE, StageLedger, file_digest,  # noqa: E402
                                        stage_key)


def parse_arm(text):
    """``"256"`` -> (256, 1); ``"256:2"`` -> (256, 2)."""
    slice_size, _, sr = text.partition(":")
    return int(slice_size), int(sr or 1)


def normalise_arm(value):
    """``"256:2"`` / ``(256, 2)`` / ``256`` -> ``(256, 2)``.

    ``--arms`` carries strings and ``GOAL3_PRODUCTION["scales"]`` carries ints and pairs, so the two
    cannot be compared until both are put in the same form.
    """
    if isinstance(value, str):
        return parse_arm(value)
    if isinstance(value, (list, tuple)):
        slice_size, sr_factor = (list(value) + [1])[:2]
        return int(slice_size), int(sr_factor)
    return int(value), 1


def arm_layer_path(out_dir, raster, slice_size, sr_factor, conf, imgsz, overlap, ext=".gpkg"):
    ss = str(slice_size) if sr_factor == 1 else f"{slice_size}sr{sr_factor}"
    stem = (f"{Path(raster).stem}-predictions-ct-{int(conf * 100):03d}-ss-{ss}"
            f"-is-{imgsz}-ov-{int(overlap * 100):03d}")
    return Path(out_dir) / f"{stem}-downscaled-mask-nms{ext}"


def effective_max_det(args):
    """The per-slice detection cap this run should use (review item X-06).

    ``presets.GOAL1_PRODUCTION["max_det"] = 600`` is sized for the **256 production arm** on the
    8 GiB laptop GPU: it clears that arm's measured worst slice (370 at conf 0.10) with headroom, and
    costs almost nothing because ``max_det`` is a ceiling rather than an allocation.

    It deliberately does **not** clear the **512 arm**, which returns up to **1013** detections on the
    densest HiRISE slices. That arm cannot be run uncapped on this card at all — 10.5 GiB peak at
    *batch 1*, and Windows' WDDM driver spills into host memory instead of raising, so the failure
    would be a silent throughput collapse. Per the maintainer's 2026-08-04 decision the 512 arm is
    **deferred to Sherlock**, where the VRAM wall does not exist and no geometry has to change; this
    flag is how that run raises the cap without editing the preset the laptop depends on.

    ``--max-det`` therefore overrides, and whatever it resolves to goes into the arm's stage key, so
    a Sherlock arm and a laptop arm can never satisfy each other's ledger entry.
    """
    return int(args.max_det) if args.max_det is not None else presets.GOAL1_PRODUCTION["max_det"]


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
                        "save_bbox", "save_prenms", "output_format", "max_det")}
    cfg["max_det"] = effective_max_det(args)
    gdf, gdf_nms = get_sliced_prediction(
        args.raster, model, confidence_threshold=args.conf, output_dir=args.out_dir,
        slice_size=slice_size, sr_factor=sr_factor, inference_size=args.inference_size,
        save_bbox=False, save_prenms=False, save_nms=True, output_format="gpkg",
        on_conflict=args.on_conflict, on_max_det=args.on_max_det, **cfg)
    print(json.dumps({"arm": args.arm, "n": int(len(gdf_nms if gdf_nms is not None else gdf)),
                      "truncation": dict(gdf.attrs.get("truncation", {}))}))


def require_production_arm_set(arms):
    """Refuse an arm set that is not production's, because the output claims to be production.

    The fused path and its manifest both say `goal3_production`, and neither encodes the arm set. A
    subset fuse would therefore claim the shipped three-arm rule while an arm's contribution is simply
    absent -- the 512 arm's large-boulder net, which is ~3% of large HiRISE humans and the only reason
    that arm exists. Refuse rather than relabel; `--no-fuse` banks a partial ladder instead.
    """
    want = {normalise_arm(x) for x in presets.GOAL3_PRODUCTION["scales"]}
    have = {normalise_arm(a) for a in arms}
    if have != want:
        raise SystemExit(
            f"refusing to fuse: production fusion is {sorted(want)}, this run has {sorted(have)}. "
            f"The fused output is named and manifested as goal3_production, so fusing a subset "
            f"would misrepresent it. Pass --no-fuse to keep the arm layers, or run the full set.")


#: Config keys that must agree across arms or the fused catalog is incoherent. Everything else is
#: allowed to differ and is *recorded* -- `slice_size`/`sr_factor` differ by definition, `max_det` and
#: `batch_size` are sized per machine, and `model_path` is a path so it differs across platforms while
#: the weights digest (checked separately, and authoritative) does not.
FUSE_MUST_MATCH = ("confidence_threshold", "inference_size", "min_area_threshold", "downscale_pred",
                   "postprocess", "postprocess_match_threshold", "postprocess_class_agnostic",
                   "contour_method", "overlap_height_ratio", "overlap_width_ratio", "mask_tag",
                   "remask_fn", "inference_contract_version")


def fuse_only(args):
    """Fuse arm layers that already exist on disk, computing nothing.

    Why this exists. `orchestrate` can only fuse arms it produced itself: an arm's ledger entry is
    keyed on the machine's own code identity, so a layer computed elsewhere never matches and the
    runner recomputes it. On a heterogeneous ladder that is not merely wasteful, it is wrong in a way
    nothing reports -- fuse on the laptop and it tries to recompute the 512 arm that does not fit
    there; fuse on the cluster and it recomputes the 256 and SR arms, silently moving the
    load-bearing arms onto the other stack, which is exactly what the 2026-08-04 "a larger GPU, not a
    different pipeline" decision was protecting.

    So this path ignores arm ledger entries entirely and validates from the **manifests** each layer
    already carries. That is a better provenance record than the ledger for this purpose: it states
    what produced each layer rather than whether this machine produced it. Differences that do not
    threaten coherence -- per-machine `max_det`, host, torch build, code digest -- are permitted and
    written into the fused manifest's `provenance` block, so a mixed-stack catalog says so out loud
    instead of looking single-sourced.

    It is also the cheap way to answer questions that only concern fusion: re-fusing costs ~112 s
    against arms that cost hours, so the open "SR arm default-on vs opt-in" decision is a couple of
    runs of this rather than a re-run of the ladder.
    """
    import geopandas as gpd
    from rastertools_BOULDERING import metadata as raster_metadata
    from YOLOv8BeyondEarth.fusion import fuse_predictions
    from YOLOv8BeyondEarth.polygon import outlines_to_shp

    require_production_arm_set(args.arms)
    out_dir = Path(args.out_dir)
    if not Path(args.raster).exists():
        raise SystemExit(f"--raster {args.raster} must be present: fusion needs its native "
                         f"resolution to interpret the DIoU threshold in metres.")

    # 1. every layer and every manifest must be there. A layer with no manifest cannot say what
    #    produced it, and B2's lesson is that the honest answer then is to refuse, not to assume.
    layers, manifests = {}, {}
    for arm in args.arms:
        layer = arm_layer_path(out_dir, args.raster, *parse_arm(arm), args.conf,
                               args.inference_size, presets.GOAL1_PRODUCTION["overlap_height_ratio"])
        mpath = presets.manifest_path(layer)
        if not layer.exists():
            raise SystemExit(f"missing arm layer: {layer}\nRun that arm, or copy it in from the "
                             f"machine that produced it.")
        if not mpath.exists():
            raise SystemExit(f"missing manifest for {layer.name}: {mpath.name}\nWithout it this "
                             f"layer cannot state what produced it, so it will not be fused.")
        layers[arm] = layer
        manifests[arm] = json.loads(mpath.read_text())

    # 2. the inputs must be the same raster and the same weights, by digest rather than by path.
    ref = args.arms[0]
    for field in ("raster", "weights"):
        digests = {arm: (m.get("inputs", {}).get(field) or {}).get("sha256")
                   for arm, m in manifests.items()}
        if len(set(digests.values())) > 1:
            raise SystemExit(f"refusing to fuse: arms disagree on {field} sha256 -> {digests}")
        if digests[ref] is None:
            raise SystemExit(f"refusing to fuse: no {field} sha256 recorded in the manifests, so "
                             f"the arms cannot be shown to share an input.")

    # 3. surface every config difference in one pass; fail on the ones that break coherence.
    keys = sorted({k for m in manifests.values() for k in m.get("config", {})})
    differing = {k: {arm: manifests[arm].get("config", {}).get(k) for arm in args.arms}
                 for k in keys
                 if len({json.dumps(manifests[a].get("config", {}).get(k), sort_keys=True)
                         for a in args.arms}) > 1}
    fatal = {k: v for k, v in differing.items() if k in FUSE_MUST_MATCH}
    if fatal:
        raise SystemExit("refusing to fuse: arms disagree on configuration that must match:\n"
                         + "\n".join(f"  {k}: {v}" for k, v in fatal.items()))

    # 4. provenance, per arm, including whichever stack produced it.
    provenance = {}
    for arm in args.arms:
        code = manifests[arm].get("code", {})
        cfg = manifests[arm].get("config", {})
        provenance[arm] = {
            "layer": layers[arm].name,
            "layer_sha256": file_digest(layers[arm]),
            "written_utc": manifests[arm].get("written_utc"),
            "max_det": cfg.get("max_det"),
            "batch_size": cfg.get("batch_size"),
            "host": code.get("host"), "platform": code.get("platform"),
            "torch": code.get("torch"), "python": code.get("python"),
            "code_source_sha256": code.get("source_sha256"), "git_sha": code.get("git_sha"),
        }
    stacks = {(p["host"], p["torch"], p["code_source_sha256"]) for p in provenance.values()}
    heterogeneous = len(stacks) > 1

    print(f"fusing {len(args.arms)} arms from disk, computing nothing:")
    for arm in args.arms:
        p = provenance[arm]
        print(f"  {arm:8} {p['layer']}  host={p['host']} torch={p['torch']} max_det={p['max_det']}")
    if differing:
        print("permitted per-arm differences (recorded in the fused manifest): "
              f"{', '.join(sorted(differing))}")
    if heterogeneous:
        print("NOTE: this is a MIXED-STACK fuse -- arms were produced on different stacks. That is "
              "recorded in the manifest as provenance.heterogeneous=true. Whether it is acceptable "
              "is a science question (see the stack-parity check), not one this script can settle.")

    # 5. the ledger keys on the *layer digests*, so re-fusing is skippable and no arm entry is needed.
    ledger = StageLedger(out_dir / "run_ledger.jsonl")
    fuse_key = stage_key(arm_layers=sorted(p["layer_sha256"] for p in provenance.values()),
                         preset="goal3_production", mode="fuse_only")
    fused_path = out_dir / f"{Path(args.raster).stem}-predictions-fused-production-mask.gpkg"
    state, detail = ledger.status("fuse", fuse_key)
    if state == COMPLETE:
        print(f"[skip] fuse: {detail}")
        return
    print(f"[run ] fuse: {state} ({detail})")

    cfg = presets.GOAL3_PRODUCTION
    with ledger.stage("fuse", fuse_key, artifacts=[fused_path]):
        per_scale = {parse_arm(a): gpd.read_file(p) for a, p in layers.items()}
        fused, audit = fuse_predictions(
            per_scale, rule=cfg["rule"], bands=cfg["bands"], engulf_thr=cfg["engulf_thr"],
            metric=cfg["metric"], cross_arm_only=cfg["cross_arm_only"],
            native_res=raster_metadata.get_resolution(args.raster)[0])
        outlines_to_shp(fused, fused_path)
        audit.to_csv(fused_path.with_suffix("").as_posix() + "-audit.csv", index=False)
        presets.write_manifest(
            fused_path, config={"preset": "goal3_production", **cfg},
            inputs={"raster": manifests[ref]["inputs"]["raster"],
                    "weights": manifests[ref]["inputs"]["weights"]},
            outputs={"n_fused": int(len(fused)), "n_suppressed": int(len(audit))},
            provenance={"mode": "fuse_only", "heterogeneous": heterogeneous, "arms": provenance,
                        "permitted_differences": differing})
    print(f"wrote {fused_path}  ({len(fused)} fused, {len(audit)} suppressed)")


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
                        max_det=effective_max_det(args))
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
               "--max-det", str(effective_max_det(args)),
               "--on-max-det", args.on_max_det, "--on-conflict", "overwrite"]
        with ledger.stage(f"arm:{arm}", key, artifacts=[layer], command=cmd):
            subprocess.run(cmd, check=True)

    # Depend on each arm's key *and* the artifact digest its ledger entry recorded, not on the key
    # alone: the key is a hash of the configuration, so an arm re-executed with unchanged config
    # keeps it and the fuse would skip while its input had been regenerated underneath it.
    if args.no_fuse:
        print(f"[skip] fuse: --no-fuse ({len(arm_keys)} arm layer(s) written, nothing fused)")
        return

    require_production_arm_set(args.arms)

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
    ap.add_argument("--fuse-only", action="store_true",
                    help="fuse the arm layers already on disk and compute nothing. The way to "
                         "assemble a ladder whose arms were produced on different machines, and the "
                         "cheap way to compare fusion choices (~112 s vs hours for the arms). "
                         "Validates coherence from each layer's manifest rather than from this "
                         "machine's ledger, and records per-arm provenance in the fused manifest.")
    ap.add_argument("--no-fuse", action="store_true",
                    help="run the arms and stop. For a partial ladder: the fused output is named "
                         "and manifested as goal3_production, so fusing an incomplete arm set is "
                         "refused rather than relabelled.")
    ap.add_argument("--conf", type=float, default=presets.GOAL1_PRODUCTION["confidence_threshold"])
    ap.add_argument("--inference-size", type=int,
                    default=presets.GOAL1_PRODUCTION["inference_size"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--on-conflict", default="error", choices=("error", "overwrite"))
    ap.add_argument("--on-max-det", default="error", choices=("error", "warn"),
                    help="'error' fails in the first minutes of an arm rather than writing a "
                         "silently truncated catalog (review item X-06)")
    ap.add_argument("--max-det", type=int, default=None,
                    help="per-slice detection cap; defaults to GOAL1_PRODUCTION's 600, which is "
                         "sized for the 256 arm on an 8 GiB card. The 512 arm needs ~1100 and is "
                         "deferred to a larger GPU — raise it there with this flag, which also "
                         "enters the stage key so the two runs cannot satisfy each other's ledger.")
    args = ap.parse_args()
    if args.fuse_only and args.no_fuse:
        ap.error("--fuse-only and --no-fuse are contradictory")
    if args.arm:
        run_one_arm(args)
    elif args.fuse_only:
        fuse_only(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
