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

**Cap escalation** (``--on-max-det escalate``, maintainer decision 2026-10-15). A truncated arm is a
lower bound, not a catalog, so instead of failing the scene the runner re-runs the arm with the cap
doubled and the batch halved. It repeats until an attempt completes with no slice within 10% of its
cap (the child runs with ``on_max_det="error_near"``, so each failed attempt stops at its first such
batch rather than at the end of the arm). Escalation ends at :func:`model_max_boxes`, the most boxes
the model can return for one slice, since no cap above that can bind.

What an escalation step costs (corrected after the configuration-study audit, 2026-10-16):

- GPU memory. Halving the batch keeps ``max_det x batch_size``, and with it the retained masks (~1 MiB
  per detection at imgsz 1024), where they were. The transient for the slice being processed does not
  stay put: ultralytics holds ~5 MiB per detection of that slice while it builds masks, which follows
  the cap alone, ~5.4 / 10.7 / 21.5 / 43 GiB worst case at caps 1,100 / 2,200 / 4,400 / 8,800. So a
  later attempt can run out of memory. A CUDA out-of-memory stop halves the batch at the same cap, and
  at batch 1 the arm stops and asks for a card with more memory.
- Time. A failed attempt runs until the first near-cap batch in slice order. That took seconds on the
  25-slice tile of the end-to-end check; on a scene it can be most of a pass.
- Walltime. An attempt has no checkpoint. A resubmit resumes at the attempt the ledger points to, but
  an attempt that needs longer than the job's ``--time`` restarts from zero every time and never
  finishes, so that scene needs a longer ``--time``.

What it produces is the uncapped catalog: a run that does not bind does not depend on the cap (279
tiles verified inert). The halved batch is the one thing that moves, at the ~1e-4 level a batch's size
is measured to move results (G1-02's 1-vs-8 comparison).

Every attempt is a ledger entry of the arm's stage, with its ``max_det``, ``batch_size`` and why it
stopped, so a resubmitted job resumes at the attempt the last one pointed to, rather than paying again
for the passes that already bound. An arm that ``--on-max-det error`` or ``warn`` completed with a
slice at or near its cap is a lower bound as well, so escalation does not accept that entry as done: it
counts as a ``max_det`` stop, and the arm goes on at the next attempt.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Fragmentation, not demand, is what kills scene-scale arms. Both OOMs measured 2026-08-18/19 report a
# large *reserved but unallocated* pool: on the 24 GiB 3090 the 512 arm died asking for 2.40 GiB with
# 7.23 GiB allocated and **14.15 GiB reserved-unallocated**, i.e. the caching allocator could not find a
# contiguous block among its fragments. `max_det` in the high hundreds makes per-slice mask tensors
# vary enormously in size, and a few thousand slices of that shreds the pool. `expandable_segments`
# lets the allocator grow segments instead of hoarding fixed-size blocks, which is exactly the case
# torch's own error message recommends it for.
#
# It is numerically inert -- a virtual-memory strategy, not a computation change -- so no geometry or
# orientation gate has to be re-cleared for it. Set before torch is imported anywhere (every torch
# import in this file is lazy, inside a function, so module scope is early enough), and via setdefault
# so an explicit environment value still wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from YOLOv8BeyondEarth import presets                                   # noqa: E402
from YOLOv8BeyondEarth.runstate import (COMPLETE, FAILED, INCOMPLETE,       # noqa: E402
                                        StageLedger, file_digest, stage_key)

#: Exit codes an arm's child uses to say *why* it stopped, so the orchestrator can tell a stop that a
#: higher cap or a smaller batch fixes from a crash. Anything else non-zero is a crash.
EXIT_MAX_DET = 3
EXIT_CUDA_OOM = 4

#: The truncation counts that say the cap bound: a slice at it, or within 10% of it (where a truncated
#: slice lands after ultralytics drops empty masks). The NMS time-limit counts are a different failure,
#: which no cap fixes.
CAP_TRUNCATION = ("slices_at_max_det", "slices_near_max_det")


class ArmStopped(Exception):
    """An arm attempt stopped for a reason the next attempt fixes; recorded, then retried."""


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


def effective_batch_size(args):
    """Slices per model call: ``--batch-size``, else the preset's. Escalation halves it."""
    return (int(args.batch_size) if args.batch_size is not None
            else presets.GOAL1_PRODUCTION["batch_size"])


def model_max_boxes(inference_size, strides=(8, 16, 32)):
    """The most detections the model can return for one slice: one per grid cell, per stride.

    YOLOv8 is anchor-free with a single prediction per cell at each stride, so at ``imgsz`` 1024 that
    is 128^2 + 64^2 + 32^2 = 21,504 (below ultralytics' ``max_nms`` of 30,000, so NMS sees them
    all). A cap at or above this cannot bind, so it is where escalation stops.
    """
    return sum((int(inference_size) // s) ** 2 for s in strides)


def next_attempt(max_det, batch_size, reason, ceiling):
    """``(max_det, batch_size)`` for the attempt after a recoverable stop, or ``None`` if none is left.

    The cap doubles and the batch halves together, so ``max_det x batch_size`` -- what the retained
    masks scale with -- does not grow. The transient for the slice being processed scales with the cap
    alone and doubles at every step (see the module docstring), so a later attempt can still run out of
    memory. An out-of-memory stop halves the batch at the same cap.
    """
    max_det, batch_size = int(max_det), int(batch_size)
    if reason == "max_det":
        if max_det >= ceiling:
            return None
        return min(2 * max_det, int(ceiling)), max(1, batch_size // 2)
    if reason == "cuda_oom":
        return (max_det, batch_size // 2) if batch_size > 1 else None
    raise ValueError(f"unknown stop reason {reason!r}")


def arm_base_parts(args, raster_id, weights_id, slice_size, sr_factor):
    """Everything that defines an arm's output except the cap and the batch."""
    # `preset="goal1_production"` is a *name*, so on its own it cannot notice that the preset's
    # values changed underneath a resumed campaign. `presets.PRESET_VERSION` is bumped whenever they
    # do (that is its documented contract).
    return dict(raster=raster_id.get("sha256"), weights=weights_id.get("sha256"),
                slice_size=slice_size, sr_factor=sr_factor, conf=args.conf,
                imgsz=args.inference_size, preset="goal1_production",
                preset_version=presets.PRESET_VERSION)


def attempt_key(base, max_det, batch_size):
    """The stage key of one arm attempt.

    ``max_det`` is spelled out because X-06 made it result-defining: a stale entry would otherwise
    let a 300-capped arm satisfy a run configured for a higher cap. ``batch_size`` moves results at
    ~1e-4, so it is keyed too, but only when it differs from the preset's. A preset-batch key is then
    the key the formula gave before escalation existed, so the batch adds no re-key of its own. That
    does not keep older entries valid: every stage key also embeds the package's source digest
    (``runstate.stage_key``), so any change under ``src/`` -- the escalation change itself among them --
    re-keys every banked arm anyway.
    """
    parts = {**base, "max_det": int(max_det)}
    if int(batch_size) != presets.GOAL1_PRODUCTION["batch_size"]:
        parts["batch_size"] = int(batch_size)
    return stage_key(**parts)


def cap_truncation(rec):
    """The cap-side truncation a ledger entry recorded, ``{}`` if none.

    An escalating run never completes with any, because its child stops on them. An attempt that
    ``--on-max-det error`` or ``warn`` completed can, and to escalation that layer is a lower bound.
    """
    return {k: v for k, v in ((rec or {}).get("truncation") or {}).items()
            if k in CAP_TRUNCATION and v}


def _no_higher_cap(arm, max_det, ceiling, inference_size):
    return (f"arm {arm}: a slice still came within 10% of max_det={max_det}, the escalation ceiling "
            f"({ceiling}; the model returns at most {model_max_boxes(inference_size)} boxes per slice "
            f"at imgsz {inference_size}). There is no higher cap to try.")


def _bound(rec):
    """Whether a ledger entry's cap already bound, so that running the same attempt again would
    only bind again: complete with a slice at or near the cap (which only 'error' or 'warn' can
    leave), or stopped on the cap. The latter includes a stop that recorded no next attempt: one
    written under 'error', which fails before computing one, or one at an exhausted ceiling."""
    rec = rec or {}
    return ((rec.get("state") == COMPLETE and bool(cap_truncation(rec)))
            or (rec.get("state") == FAILED and rec.get("stopped") == "max_det"))


def resume_point(ledger, arm, base_key, start, ceiling):
    """Where an escalating arm stands: the ``(max_det, batch_size)`` to check or try first.

    An arm's attempts share its stage name, and each terminal entry records its ``base_key`` and its
    own ``max_det`` and ``batch_size``. The latest entry for this configuration decides. A complete
    attempt is re-checked as it is, so a finished escalated arm is not redone at the starting cap. A
    recoverable stop resumes at the attempt it pointed to. An attempt whose cap already bound
    (:func:`_bound`) but that points nowhere resumes at the attempt after it, up to ``ceiling``; at
    the ceiling it is returned as it is, and :func:`orchestrate` refuses it. Any other ending retries
    the same attempt. A killed attempt leaves only its ``running`` marker, which carries no
    ``base_key``, so the entry before it decides. No entry for this configuration means ``start``.
    A stored next attempt may lie above a ceiling lowered since; :func:`orchestrate` refuses that too.
    """
    for rec in reversed(ledger.entries(f"arm:{arm}")):
        if rec.get("base_key") != base_key:
            continue
        nxt = rec.get("next_attempt")
        if rec.get("state") == FAILED and nxt:
            return int(nxt["max_det"]), int(nxt["batch_size"])
        if _bound(rec):
            nxt = next_attempt(rec["max_det"], rec["batch_size"], "max_det", ceiling)
            if nxt is not None:
                why = (f"recorded {cap_truncation(rec)}, a lower bound" if rec["state"] == COMPLETE
                       else "stopped on the cap")
                print(f"[esc ] arm {arm}: its attempt at max_det {rec['max_det']}, batch "
                      f"{rec['batch_size']} (on_max_det={rec.get('on_max_det')!r}) {why}; next "
                      f"max_det {nxt[0]}, batch {nxt[1]}", flush=True)
                return nxt
        if rec.get("state") in (COMPLETE, FAILED, INCOMPLETE):
            return int(rec["max_det"]), int(rec["batch_size"])
    return start


def _run_child(cmd):
    """Run one arm's child. Returns ``(returncode, summary)``, ``summary`` being the child's one JSON
    stdout line (progress and logs go to stderr), echoed for the job log, or ``None``."""
    res = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    sys.stdout.write(res.stdout)
    for line in reversed(res.stdout.splitlines()):
        if line.startswith("{"):
            try:
                return res.returncode, json.loads(line)
            except json.JSONDecodeError:
                break
    return res.returncode, None


def _is_cuda_oom(exc):
    import torch
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    # a driver-level allocation failure surfaces as a plain RuntimeError ("CUDA error: out of memory")
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def run_one_arm(args):
    """The child process: one arm, one layer, then exit so the memory is actually returned.

    A stop the orchestrator can fix exits with :data:`EXIT_MAX_DET` or :data:`EXIT_CUDA_OOM` after
    printing its summary line; anything else raises as before.
    """
    from sahi import AutoDetectionModel
    from YOLOv8BeyondEarth.predict import MaxDetBound, get_sliced_prediction
    from YOLOv8BeyondEarth.runstate import peak_rss_bytes

    slice_size, sr_factor = parse_arm(args.arm)
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=args.weights, device=args.device,
        image_size=args.inference_size, confidence_threshold=args.conf)
    cfg = {k: v for k, v in presets.GOAL1_PRODUCTION.items()
           if k not in ("slice_size", "sr_factor", "confidence_threshold", "inference_size",
                        "save_bbox", "save_prenms", "output_format", "max_det", "batch_size")}
    cfg["max_det"] = effective_max_det(args)
    cfg["batch_size"] = effective_batch_size(args)
    try:
        gdf, gdf_nms = get_sliced_prediction(
            args.raster, model, confidence_threshold=args.conf, output_dir=args.out_dir,
            slice_size=slice_size, sr_factor=sr_factor, inference_size=args.inference_size,
            save_bbox=False, save_prenms=False, save_nms=True, output_format="gpkg",
            on_conflict=args.on_conflict, on_max_det=args.on_max_det, **cfg)
    except MaxDetBound as exc:
        print(f"arm {args.arm} stopped: {exc}", file=sys.stderr, flush=True)
        print(json.dumps({"arm": args.arm, "stopped": "max_det", "bound": exc.as_dict(),
                          "peak_rss_bytes": peak_rss_bytes()}), flush=True)
        sys.exit(EXIT_MAX_DET)
    except Exception as exc:
        if not _is_cuda_oom(exc):
            raise
        print(f"arm {args.arm} stopped: CUDA out of memory at max_det {cfg['max_det']}, batch "
              f"{cfg['batch_size']}: {exc}", file=sys.stderr, flush=True)
        print(json.dumps({"arm": args.arm, "stopped": "cuda_oom",
                          "peak_rss_bytes": peak_rss_bytes()}), flush=True)
        sys.exit(EXIT_CUDA_OOM)
    print(json.dumps({"arm": args.arm, "n": int(len(gdf_nms if gdf_nms is not None else gdf)),
                      "truncation": dict(gdf.attrs.get("truncation", {})),
                      "slices": gdf.attrs.get("slices"), "peak_rss_bytes": peak_rss_bytes()}),
          flush=True)


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


def _attempt_arm(args, ledger, arm, key, base_key, layer, max_det, batch, child_mode, escalate,
                 ceiling):
    """Run one attempt of one arm in its own process, and record it in the ledger either way.

    Returns ``None`` when the arm is done, or, in escalate mode, the ``(max_det, batch_size)`` to try
    after a recoverable stop. Any other ending raises, after being recorded as a failed attempt.
    """
    cmd = [sys.executable, str(Path(__file__).resolve()), "--arm", arm,
           "--raster", str(args.raster), "--weights", str(args.weights),
           "--out-dir", str(args.out_dir), "--conf", str(args.conf),
           "--inference-size", str(args.inference_size), "--device", args.device,
           "--max-det", str(max_det), "--batch-size", str(batch),
           "--on-max-det", child_mode, "--on-conflict", "overwrite"]
    nxt = None
    try:
        with ledger.stage(f"arm:{arm}", key, artifacts=[layer], command=cmd, base_key=base_key,
                          max_det=int(max_det), batch_size=int(batch),
                          on_max_det=args.on_max_det) as st:
            # The child's one JSON stdout line goes into the ledger: peak RSS per arm per scene is
            # the RAM measurement lever N2 asked for, and a stop says why it stopped.
            code, child = _run_child(cmd)
            st.result.update({k: v for k, v in (child or {}).items()
                              if k in ("n", "truncation", "slices", "peak_rss_bytes", "stopped",
                                       "bound")})
            if code == 0:
                return None
            reason = {EXIT_MAX_DET: "max_det", EXIT_CUDA_OOM: "cuda_oom"}.get(code)
            if not (escalate and reason):
                raise subprocess.CalledProcessError(code, cmd)
            nxt = next_attempt(max_det, batch, reason, ceiling)
            if nxt is None:
                st.result["escalation_exhausted"] = reason
                if reason == "max_det":
                    raise SystemExit(_no_higher_cap(arm, max_det, ceiling, args.inference_size))
                raise SystemExit(
                    f"arm {arm}: CUDA out of memory at batch 1 with max_det={max_det}. This GPU "
                    "cannot hold the scene's densest slice at that cap, so run this arm on a card "
                    "with more memory; a resubmitted job resumes at this attempt.")
            st.result["next_attempt"] = {"max_det": nxt[0], "batch_size": nxt[1]}
            raise ArmStopped(f"stopped on {reason} at max_det {max_det}, batch {batch}")
    except ArmStopped as exc:
        print(f"[esc ] arm {arm}: {exc}; next max_det {nxt[0]}, batch {nxt[1]}", flush=True)
        return nxt


def orchestrate(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = StageLedger(out_dir / "run_ledger.jsonl")
    raster_id = presets.file_identity(args.raster)
    weights_id = presets.file_identity(args.weights)

    escalate = args.on_max_det == "escalate"
    child_mode = "error_near" if escalate else args.on_max_det
    ceiling = (int(args.max_det_ceiling) if args.max_det_ceiling is not None
               else model_max_boxes(args.inference_size))

    arm_keys, arm_layers = {}, {}
    for arm in args.arms:
        slice_size, sr_factor = parse_arm(arm)
        base = arm_base_parts(args, raster_id, weights_id, slice_size, sr_factor)
        base_key = stage_key(**base)
        layer = arm_layer_path(out_dir, args.raster, slice_size, sr_factor, args.conf,
                               args.inference_size, presets.GOAL1_PRODUCTION["overlap_height_ratio"])
        start = (effective_max_det(args), effective_batch_size(args))
        max_det, batch = resume_point(ledger, arm, base_key, start, ceiling) if escalate else start
        if escalate and max_det > ceiling:
            raise SystemExit(f"arm {arm}: the ledger points at max_det {max_det}, above the escalation "
                             f"ceiling ({ceiling}). A lower cap already bound, so no attempt at or "
                             "below the ceiling can complete; raise --max-det-ceiling to go on.")
        while True:
            key = attempt_key(base, max_det, batch)
            state, detail = ledger.status(f"arm:{arm}", key)
            last = ledger.latest(f"arm:{arm}")
            if escalate and last and last.get("key") == key and _bound(last):
                # This very attempt already bound. resume_point moves past such an entry while a
                # higher cap is left, so this one is at the ceiling, and a rerun would bind again.
                raise SystemExit(_no_higher_cap(arm, max_det, ceiling, args.inference_size))
            if state == COMPLETE:
                print(f"[skip] arm {arm}: {detail} (max_det {max_det}, batch {batch})", flush=True)
                break
            print(f"[run ] arm {arm}: {state} ({detail}); max_det {max_det}, batch {batch}",
                  flush=True)
            nxt = _attempt_arm(args, ledger, arm, key, base_key, layer, max_det, batch,
                               child_mode, escalate, ceiling)
            if nxt is None:
                break
            max_det, batch = nxt
        arm_keys[arm], arm_layers[arm] = key, layer

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
    ap.add_argument("--on-max-det", default="error",
                    choices=("error", "error_near", "warn", "escalate"),
                    help="'error' fails in the first minutes of an arm rather than writing a "
                         "silently truncated catalog (review item X-06); 'error_near' also fails "
                         "within 10%% of the cap; 'escalate' re-runs an arm that gets that close at "
                         "twice the cap and half the batch until it does not (see the docstring)")
    ap.add_argument("--max-det", type=int, default=None,
                    help="per-slice detection cap; defaults to GOAL1_PRODUCTION's 600, which is "
                         "sized for the 256 arm on an 8 GiB card. The 512 arm needs ~1100 and is "
                         "deferred to a larger GPU — raise it there with this flag, which also "
                         "enters the stage key so the two runs cannot satisfy each other's ledger. "
                         "With --on-max-det escalate it is the starting cap.")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="slices per model call; defaults to GOAL1_PRODUCTION's 8. Keyed when it "
                         "differs from that. With --on-max-det escalate it is the starting batch.")
    ap.add_argument("--max-det-ceiling", type=int, default=None,
                    help="highest cap escalation may reach; defaults to the most boxes the model "
                         "can return per slice (21,504 at imgsz 1024), where no cap can bind")
    args = ap.parse_args()
    if args.fuse_only and args.no_fuse:
        ap.error("--fuse-only and --no-fuse are contradictory")
    if args.arm and args.on_max_det == "escalate":
        ap.error("'escalate' is the orchestrator's mode; one arm's child takes 'error_near'")
    if args.arm:
        run_one_arm(args)
    elif args.fuse_only:
        fuse_only(args)
    else:
        try:
            orchestrate(args)
        except subprocess.CalledProcessError as exc:
            if exc.returncode >= 0:
                raise
            # An arm killed by a signal exits the way a shell reports one, 128 + signal, so the job
            # script can tell SIGKILL (137) -- on a cluster node, the host-RAM out-of-memory killer --
            # from a crash, which exits 1. The ledger has already recorded the attempt as failed.
            print(exc, file=sys.stderr, flush=True)
            if exc.returncode == -9:
                print("SIGKILL is usually the out-of-memory killer. The same attempt at the same "
                      "memory limit dies the same way, so raise the job's --mem.",
                      file=sys.stderr, flush=True)
            sys.exit(128 - exc.returncode)


if __name__ == "__main__":
    main()
