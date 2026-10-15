"""Run the SR/512 arm ratification 2x2 against the human reference, one body at a time.

    conda run -n bouldernet python scripts/ratify_arms.py --body moon
    conda run -n bouldernet python scripts/ratify_arms.py --body mars
    conda run -n bouldernet python scripts/ratify_arms.py --body moon --clip-first

The harness itself is :mod:`YOLOv8BeyondEarth.arm_ratification`; this file is only the wiring —
which reference, which arm layers, which raster defines the CRS and the native resolution. Keeping
the wiring here and the readout in ``src/`` is what lets a new human reference re-run the whole
thing by editing one table.

**Why the two bodies are not symmetric, and what is done about it.** The Moon arms are full-scene
layers, so the four cells are fused over the whole scene and *then* clipped to the ROM tiles —
production's own order. The Mars arms only exist ROM-clipped (the full-scene layers lived on
``$SCRATCH`` and are gone), so the Mars cells are necessarily fused over clipped inputs. Fusing
after a clip can differ near a tile edge: a 512-arm detection just outside a tile can engulf or
suppress one just inside, and clipping first removes it from the contest. ``--clip-first`` runs the
Moon both ways so that difference is a measured number rather than an assumption carried into the
Mars column.

Nothing here filters the human reference. Every human inside a ROM tile is in the denominator, and
the size bands do the rest — a sub-floor human that no arm can be expected to find shows up as a
low-recall row in a band that is labelled, not as a quietly dropped object.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
BASE = REPO.parent
sys.path.insert(0, str(REPO / "src"))

from YOLOv8BeyondEarth import arm_ratification as ar        # noqa: E402
from YOLOv8BeyondEarth import presets                       # noqa: E402
from YOLOv8BeyondEarth.fusion import equivalent_diameter_px  # noqa: E402

#: One entry per body. ``arms`` maps the scale key to its layer, spelled out rather than derived,
#: because the Mars layers carry a ``-rom-`` infix from the clip and the Moon ones do not.
BODIES = {
    "moon": dict(
        raster=REPO / "data/raster/M139694087LE.tif",
        rom=BASE / "data_prieur/raw_data/moon/censorinus/shp/M139694087LE-censorinus-ROM.shp",
        humans=(BASE / "data_prieur/raw_data/moon/censorinus/shp"
                / "M139694087LE-censorinus-boulder-mapping.shp"),
        arms={
            512: REPO / ("data/rerun_2026-08-17/nac/M139694087LE-predictions-ct-010-ss-512"
                         "-is-1024-ov-020-downscaled-mask-nms.gpkg"),
            256: REPO / ("data/rerun_2026-08-17/nac/M139694087LE-predictions-ct-010-ss-256"
                         "-is-1024-ov-020-downscaled-mask-nms.gpkg"),
            (256, 2): REPO / ("data/rerun_2026-08-17/nac/M139694087LE-predictions-ct-010"
                              "-ss-256sr2-is-1024-ov-020-downscaled-mask-nms.gpkg"),
        },
        arms_are_clipped=False,
    ),
    "mars": dict(
        raster=REPO / "data/mars_hirise/raster/ESP_017355_2260_RED.tif",
        rom=BASE / "data_prieur/raw_data/mars/ESP_017355_2260/shp/ESP_017355_2260-ROM.shp",
        humans=(BASE / "data_prieur/raw_data/mars/ESP_017355_2260/shp"
                / "ESP_017355_2260-boulder-mapping.shp"),
        arms={
            512: REPO / ("data/review/rom_arms/hirise/ESP_017355_2260_RED-predictions-ct-010"
                         "-ss-512-is-1024-ov-020-rom-downscaled-mask-nms.gpkg"),
            256: REPO / ("data/review/rom_arms/hirise/ESP_017355_2260_RED-predictions-ct-010"
                         "-ss-256-is-1024-ov-020-rom-downscaled-mask-nms.gpkg"),
            (256, 2): REPO / ("data/review/rom_arms/hirise/ESP_017355_2260_RED-predictions-ct-010"
                              "-ss-256sr2-is-1024-ov-020-rom-downscaled-mask-nms.gpkg"),
        },
        arms_are_clipped=True,
    ),
}

DATASET = BASE / "data_prieur/boulder2024"


def load_body(body, clip_first=False, pred_shift_m=None, erode_tiles_m=0.0):
    """Reference and arm layers for one body, plus the raster's CRS and native resolution.

    ``pred_shift_m`` and ``erode_tiles_m`` exist for the X-07 measurement (2026-10-14). The Mars
    arms were made on a raster in the JP2 frame, which sits (+0.75, -0.75) m from the label frame the
    human labels are in. So ``pred_shift_m=(-0.75, 0.75)`` translates every prediction into the
    humans' frame, which is what repairing the raster's transform would do. The arms were pre-clipped
    to the *unshifted* ROM with no margin, so any shift loses a ~0.75 m strip along two tile edges;
    ``erode_tiles_m`` scores both runs on tiles shrunk by that much plus a margin, so the comparison
    sees the frame and not the strip.
    """
    import rasterio

    cfg = BODIES[body]
    with rasterio.open(cfg["raster"]) as src:
        crs, res = src.crs, src.res[0]
    split_map = ar.dataset_splits(DATASET)
    tiles = ar.rom_tiles(cfg["rom"], crs, split_map=split_map)
    if erode_tiles_m:
        tiles = tiles.set_geometry(tiles.geometry.buffer(-float(erode_tiles_m)))
    humans = ar.clip_to_tiles(ar.load_aligned(cfg["humans"], crs), tiles)
    arms = {}
    for key, path in cfg["arms"].items():
        layer = gpd.read_file(path).set_crs(crs, allow_override=True)
        if pred_shift_m:
            layer = layer.set_geometry(layer.geometry.translate(*map(float, pred_shift_m)))
        arms[key] = ar.clip_to_tiles(layer, tiles) if clip_first else layer
    return dict(crs=crs, res=res, tiles=tiles, humans=humans, arms=arms, cfg=cfg)


def run_body(body, out_dir, clip_first=False, pred_shift_m=None, erode_tiles_m=0.0):
    data = load_body(body, clip_first=clip_first, pred_shift_m=pred_shift_m,
                     erode_tiles_m=erode_tiles_m)
    tiles, humans, arms, res = data["tiles"], data["humans"], data["arms"], data["res"]
    print(f"[{body}] res {res:.6f} m/px | ROM tiles {len(tiles)} "
          f"({tiles.split.value_counts().to_dict()})")
    print(f"[{body}] humans in ROM {len(humans)} | by split "
          f"{humans.split.value_counts().to_dict()}")
    dpx_h = equivalent_diameter_px(humans, res)
    print(f"[{body}] humans >=4 px {int((dpx_h >= 4).sum())} | >=8 px {int((dpx_h >= 8).sum())} "
          f"| >=14 px {int((dpx_h >= 14).sum())}")
    for key, layer in arms.items():
        print(f"[{body}] arm {key}: {len(layer)} detections"
              f"{' (pre-clipped input)' if clip_first or data['cfg']['arms_are_clipped'] else ''}")

    # Fuse over whatever footprint the arms came in on, then clip the *fused* catalog to the ROM
    # tiles. Clipping the fused product rather than the arms is production's order (see the module
    # docstring); `--clip-first` inverts it so the difference can be measured.
    cell_sizes = {}

    def on_cell(name, raw, scored, audit):
        cell_sizes[name] = dict(n_fused=int(len(raw)), n_in_rom=int(len(scored)),
                                n_suppressed=int(len(audit)))
        print(f"[{body}] cell {name:12} fused {len(raw):7} -> in ROM {len(scored):7} "
              f"(suppressed {len(audit)})", flush=True)

    scores = ar.ratify_arms(arms, humans, res, body=body, on_cell=on_cell,
                            post_fuse=lambda fused: ar.clip_to_tiles(fused, tiles))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = (f"{body}{'-clipfirst' if clip_first else ''}"
           f"{'-predshift' if pred_shift_m else ''}"
           f"{f'-erode{erode_tiles_m:g}m' if erode_tiles_m else ''}")
    scores.to_csv(out_dir / f"arm_ratification_{tag}.csv", index=False)

    marg = []
    for band in ("all", "4-6", "6-8", "8-10", "10-14", "14-inf"):
        for split in ["all", *sorted(set(humans.split))]:
            for resolve in ar.RESOLUTIONS:
                for hybrid in (False, True):
                    for metric in ("recall", "precision", "f1"):
                        try:
                            row = ar.marginal_table(scores, band=band, split=split,
                                                    resolve=resolve, hybrid=hybrid, metric=metric)
                        except (KeyError, IndexError):
                            continue
                        row["body"] = body
                        marg.append(row)
    marginals = pd.DataFrame(marg)
    marginals.to_csv(out_dir / f"arm_ratification_marginals_{tag}.csv", index=False)

    prov = ar.ratification_provenance(
        reference_paths={"rom": data["cfg"]["rom"], "humans": data["cfg"]["humans"]},
        arm_paths={str(k): v for k, v in data["cfg"]["arms"].items()},
        native_res=res,
        extra=dict(body=body, raster=presets.file_identity(data["cfg"]["raster"]),
                   clip_order="clip-then-fuse" if clip_first else "fuse-then-clip",
                   pred_shift_m=list(pred_shift_m) if pred_shift_m else None,
                   erode_tiles_m=float(erode_tiles_m),
                   arms_supplied_clipped=bool(data["cfg"]["arms_are_clipped"]),
                   dataset_root=DATASET.as_posix(),
                   rom_tiles=int(len(tiles)),
                   rom_tiles_by_split={str(k): int(v) for k, v
                                       in tiles.split.value_counts().items()},
                   humans_in_rom=int(len(humans)),
                   humans_by_split={str(k): int(v) for k, v
                                    in humans.split.value_counts().items()},
                   cells=cell_sizes))
    (out_dir / f"arm_ratification_{tag}_provenance.json").write_text(
        json.dumps(presets._jsonable(prov), indent=1))

    # The gate-citing artifact, in the one place `.gitignore` tracks (`!data/review/*.json`): the
    # plan text quotes these marginals, so the evidence has to travel with the claim rather than
    # sitting in an ignored CSV. The full per-band/per-split tables stay as CSV next door -- they
    # regenerate in minutes from the same command.
    headline = scores[(scores.split == "all") & (scores.resolve == "optimal") & (~scores.hybrid)]
    summary = dict(
        provenance=presets._jsonable(prov),
        marginals_recall_pooled_optimal_strict=[
            {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
             for k, v in row.items()}
            for row in marginals[(marginals.metric == "recall") & (marginals.split == "all")
                                 & (marginals.resolve == "optimal")
                                 & (~marginals.hybrid)].to_dict("records")],
        scores_pooled_optimal_strict=headline[
            ["combo", "band", "n_human", "n_pred", "n_matched", "recall", "precision",
             "f1"]].to_dict("records"),
        splits_14plus=scores[(scores.band == "14-inf") & (scores.resolve == "optimal")
                             & (~scores.hybrid)][
            ["combo", "split", "n_human", "n_matched", "recall", "precision"]].to_dict("records"),
    )
    (REPO / "data/review" / f"arm_ratification_{tag}.json").write_text(
        json.dumps(summary, indent=1, default=str))

    headline = scores[(scores.band == "14-inf") & (scores.split == "all")
                      & (scores.resolve == "optimal") & (~scores.hybrid)]
    print(f"\n[{body}] 14+ px, pooled ROM, optimal/strict:")
    print(headline[["combo", "n_human", "n_pred", "n_matched", "recall", "precision",
                    "f1"]].to_string(index=False))
    return scores, marginals, prov


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--body", default="moon", choices=(*BODIES, "both"))
    ap.add_argument("--out-dir", default=str(REPO / "data/review/ratification"))
    ap.add_argument("--clip-first", action="store_true",
                    help="clip the ARM layers to the ROM before fusing, instead of clipping the "
                         "fused catalog after — the Mars-side order, run on the Moon to measure it")
    ap.add_argument("--pred-shift-m", nargs=2, type=float, metavar=("DX", "DY"),
                    help="X-07 diagnostic: translate every prediction by (DX, DY) map units "
                         "before scoring; outputs are tagged -predshift")
    ap.add_argument("--erode-tiles-m", type=float, default=0.0,
                    help="score on ROM tiles shrunk by this many map units (tagged -erodeNm)")
    args = ap.parse_args()
    bodies = list(BODIES) if args.body == "both" else [args.body]
    for body in bodies:
        run_body(body, args.out_dir, clip_first=args.clip_first, pred_shift_m=args.pred_shift_m,
                 erode_tiles_m=args.erode_tiles_m)


if __name__ == "__main__":
    main()
