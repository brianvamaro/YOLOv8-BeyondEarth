"""Is this pipeline's geometry **stack-neutral**? (the Sherlock / Goal-4 prerequisite)

Every parity proof this project rests on was established on exactly one machine: Windows, conda,
Python 3.10.15, torch 2.11.0+cu128, an RTX 5070, opencv 4.13.0.92. The device-side mask downscale
(1,987/1,987 and 2,669/2,669, 100% geometry identical, max |dscore| 0.0) and batch neutrality
(99.30% identical, max |dscore| 3.6e-4) are both single-stack results.

Two facts make that a live problem rather than a hypothetical. The 512 arm already cannot run on this
card (its densest HiRISE slices spill even at batch 1), and Goal 4 cannot run on a laptop at all. So
the production path has to move, and moving it asks a question nobody has measured: does the same
code, same weights, same raster and same preset produce the same catalog on different hardware and a
different dependency set?

This measures it. It runs on BOTH stacks, writes a catalog plus a full software identity per arm, and
compares two such runs with ``compare_catalogs`` -- the same one-to-one spatial matcher the
device-downscale and batch-neutrality proofs used, so the standard is inherited rather than invented.

Two things a naive A/B would get wrong, and this does not:

1. **The within-stack floor comes first** (``--repeat 2``). A cross-stack delta is uninterpretable
   until you know whether a single stack reproduces *itself*: cuDNN autotuning and non-deterministic
   reductions can move a borderline detection between two runs of the same binary. A cross-stack
   difference is only evidence if it exceeds the same-stack difference.
2. **The bar is one the project already accepts, not perfection.** ``batch_size=8`` ships today at
   99.30% geometry-identical with max |dscore| 3.6e-4. A cross-stack result at least that tight is no
   worse than a variation already in production. Exact bit-identity across GPU architectures is not
   expected and is deliberately not the bar -- but it IS reported, because getting it would be a much
   stronger statement and worth knowing.

    # on each stack, from the repo root
    python scripts/review/measure_stack_parity.py --mode produce --label laptop   --repeat 2
    python scripts/review/measure_stack_parity.py --mode produce --label sherlock --repeat 2

    # then, with both label directories present on one machine
    python scripts/review/measure_stack_parity.py --mode compare \
        --a data/review/stack_parity/laptop --b data/review/stack_parity/sherlock

The comparison reads only the written GeoPackages, so it does not matter which stack runs it or
whether that stack can run inference at all.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from measure_batch_neutrality import compare_catalogs                    # noqa: E402

#: The tightest variation this project already ships, from G1-02's batch_size=1-vs-8 measurement.
#: A cross-stack result at least this tight is not a new risk; one worse than it is.
ACCEPTED_BAR = {"frac_geometry_identical": 0.9930, "max_abs_score_delta": 3.6e-4}

#: Every arm of the production ladder. 512 because it is the arm actually going to another GPU; 256
#: because it is the arm every published number rests on; 256:2 because G3-10's pixel-centre rescale
#: is the one that bit it, so it is the arm most likely to disagree.
DEFAULT_ARMS = ("512", "256", "256:2")


def parse_arm(text):
    """``"256"`` -> (256, 1); ``"256:2"`` -> (256, 2). Mirrors ``run_multiscale_arms.parse_arm``."""
    slice_size, _, sr = text.partition(":")
    return int(slice_size), int(sr or 1)


def arm_tag(arm):
    slice_size, sr_factor = parse_arm(arm)
    return f"ss{slice_size}" if sr_factor == 1 else f"ss{slice_size}sr{sr_factor}"


def produce(args):
    """Run every arm ``--repeat`` times on this stack and write catalogs + a software identity."""
    from sahi import AutoDetectionModel
    from YOLOv8BeyondEarth import presets
    from YOLOv8BeyondEarth.predict import predict_production

    out = Path(args.out) / args.label
    out.mkdir(parents=True, exist_ok=True)

    identity = {
        "label": args.label,
        "code": presets.code_identity(),
        "raster": presets.file_identity(args.raster),
        "weights": presets.file_identity(args.weights),
        "preset": args.preset,
        "preset_version": presets.PRESET_VERSION,
        "device": args.device,
        "arms": list(args.arms),
        "repeats": args.repeat,
        "counts": {},
    }

    for rep in range(args.repeat):
        for arm in args.arms:
            slice_size, sr_factor = parse_arm(arm)
            rep_dir = out / f"rep{rep}"
            rep_dir.mkdir(parents=True, exist_ok=True)
            # A fresh model per arm, so a stateful detector cannot leak one arm's work into the next.
            model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics", model_path=args.weights, device=args.device,
                image_size=presets.GOAL1_PRODUCTION["inference_size"],
                confidence_threshold=presets.GOAL1_PRODUCTION["confidence_threshold"])
            _, gdf_nms = predict_production(
                args.raster, model, rep_dir, preset=args.preset, on_conflict="overwrite",
                on_max_det="error", slice_size=slice_size, sr_factor=sr_factor)
            dest = rep_dir / f"arm-{arm_tag(arm)}.gpkg"
            gdf_nms.to_file(dest, driver="GPKG")
            identity["counts"][f"rep{rep}:{arm_tag(arm)}"] = int(len(gdf_nms))
            print(f"[{args.label}] rep{rep} {arm_tag(arm)}: {len(gdf_nms)} boulders -> {dest.name}",
                  flush=True)

    (out / "identity.json").write_text(json.dumps(identity, indent=2))
    print(f"\nwrote {out / 'identity.json'}")
    print("Bring this whole directory to the machine that will run --mode compare.")


def _verdict(cmp):
    """Classify one comparison against the inherited standards. Reports, does not editorialise."""
    if cmp["n_unmatched_a"] or cmp["n_unmatched_b"]:
        return "DIFFERENT_DETECTIONS"
    if cmp["frac_geometry_identical"] == 1.0 and cmp["max_abs_score_delta"] == 0.0:
        return "EXACT"
    if (cmp["frac_geometry_identical"] >= ACCEPTED_BAR["frac_geometry_identical"]
            and cmp["max_abs_score_delta"] <= ACCEPTED_BAR["max_abs_score_delta"]):
        return "WITHIN_ACCEPTED_BAR"
    return "EXCEEDS_ACCEPTED_BAR"


def _load(d, rep, arm):
    import geopandas as gpd
    p = Path(d) / f"rep{rep}" / f"arm-{arm_tag(arm)}.gpkg"
    return gpd.read_file(p) if p.exists() else None


def compare(args):
    import rasterio

    id_a = json.loads((Path(args.a) / "identity.json").read_text())
    id_b = json.loads((Path(args.b) / "identity.json").read_text())

    # A parity check on two different rasters or two different weight files measures nothing.
    for field in ("raster", "weights"):
        sa, sb = id_a[field].get("sha256"), id_b[field].get("sha256")
        if sa != sb:
            raise SystemExit(f"refusing to compare: {field} sha256 differs ({sa} vs {sb}). "
                             "A parity check requires identical inputs on both stacks.")
    if id_a["preset_version"] != id_b["preset_version"]:
        raise SystemExit("refusing to compare: preset_version "
                         f"{id_a['preset_version']} vs {id_b['preset_version']}.")

    res = abs(rasterio.open(args.raster).transform.a)
    tol_m = 2.0 * res                      # a 2 px centroid tolerance: same boulder, small shift
    arms = [a for a in id_a["arms"] if a in id_b["arms"]]

    report = {
        "tolerance_m": tol_m,
        "accepted_bar": ACCEPTED_BAR,
        "bar_provenance": "G1-02 batch_size=1-vs-8, the tightest variation already shipped",
        "stacks": {id_a["label"]: id_a["code"], id_b["label"]: id_b["code"]},
        "stack_differences": sorted(
            k for k in set(id_a["code"]) | set(id_b["code"])
            if id_a["code"].get(k) != id_b["code"].get(k)),
        "within_stack": {},
        "cross_stack": {},
    }

    # 1. the floor: does each stack reproduce itself?
    for label, d, ident in ((id_a["label"], args.a, id_a), (id_b["label"], args.b, id_b)):
        if ident["repeats"] < 2:
            report["within_stack"][label] = "NOT MEASURED (--repeat 1)"
            continue
        report["within_stack"][label] = {}
        for arm in arms:
            r0, r1 = _load(d, 0, arm), _load(d, 1, arm)
            if r0 is None or r1 is None:
                continue
            cmp = compare_catalogs(r0, r1, tol_m)
            cmp["verdict"] = _verdict(cmp)
            report["within_stack"][label][arm_tag(arm)] = cmp

    # 2. the question: does one stack reproduce the other?
    for arm in arms:
        a0, b0 = _load(args.a, 0, arm), _load(args.b, 0, arm)
        if a0 is None or b0 is None:
            continue
        cmp = compare_catalogs(a0, b0, tol_m)
        cmp["verdict"] = _verdict(cmp)
        # The interpretation that matters: is the cross-stack delta bigger than the same-stack one?
        floors = [report["within_stack"].get(lbl, {}) for lbl in (id_a["label"], id_b["label"])]
        floors = [f[arm_tag(arm)] for f in floors if isinstance(f, dict) and arm_tag(arm) in f]
        if floors:
            worst_floor = min(f["frac_geometry_identical"] for f in floors)
            cmp["exceeds_same_stack_noise"] = bool(
                cmp["frac_geometry_identical"] < worst_floor
                or cmp["max_abs_score_delta"] > max(f["max_abs_score_delta"] for f in floors))
        report["cross_stack"][arm_tag(arm)] = cmp

    out = Path(args.out) / "stack_parity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(f"stack difference: {', '.join(report['stack_differences']) or 'none recorded'}\n")
    for arm in arms:
        t = arm_tag(arm)
        if t not in report["cross_stack"]:
            continue
        c = report["cross_stack"][t]
        noise = (f"  (exceeds same-stack noise: {c['exceeds_same_stack_noise']})"
                 if "exceeds_same_stack_noise" in c else "")
        print(f"{t}: n {c['n_a']} vs {c['n_b']}, matched {c['n_matched']}, "
              f"geometry identical {c['frac_geometry_identical']:.4f}, "
              f"max |dscore| {c['max_abs_score_delta']:.3e} -> {c['verdict']}{noise}")
    print(f"\nwrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("produce", "compare"))
    ap.add_argument("--raster", default=str(REPO / "data" / "raster" / "crop.tif"),
                    help="the tracked 2048x2048 NAC dev crop: the raster the other parity proofs used")
    ap.add_argument("--weights",
                    default=str(REPO / "data" / "weights" / "yolov8-m-boulder-detection-tmp.pt"))
    ap.add_argument("--out", default=str(REPO / "data" / "review" / "stack_parity"))
    ap.add_argument("--label", default=None, help="produce mode: names this stack, e.g. 'laptop'")
    ap.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    ap.add_argument("--repeat", type=int, default=2,
                    help="runs per arm. 2 establishes the within-stack floor; 1 makes any "
                         "cross-stack delta uninterpretable and is recorded as NOT MEASURED.")
    ap.add_argument("--preset", default="goal1_production")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--a", default=None, help="compare mode: first produce directory")
    ap.add_argument("--b", default=None, help="compare mode: second produce directory")
    args = ap.parse_args()

    if args.mode == "produce":
        if not args.label:
            ap.error("--mode produce needs --label (e.g. --label laptop)")
        produce(args)
    else:
        if not (args.a and args.b):
            ap.error("--mode compare needs --a and --b")
        compare(args)


if __name__ == "__main__":
    main()
