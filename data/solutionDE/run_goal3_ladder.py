# Goal-3 phase 3: fill in the missing production-ladder arms (plan §Phases 3).
# The ladder is {512 native, 256 native, SR x2 streamed @ 128-native footprint}. Already on disk
# from phases 1-2 (same settings: ct 0.1, is 1024, ov 0.2, downscale_pred, gpkg):
#   preds/{nac,hirise}_baseline (256) and preds/{nac,hirise}_e2stream (256sr2).
# This run adds: the 512 arm on both dev crops, and all three arms on the 20 hirise_rom tiles
# (recall-vs-human item; see prep_rom_crops.py). Idempotent; per-arm wall clocks appended to
# goal3_timings.json (the benchmark-card time axis).
#
# 2026-08-09: the ladder body moved to _ladder.py, shared with run_hseg_ladder.py. That is what
# makes the H-seg x G *matched control* below possible at all: production detections put through
# the identical configuration, rather than two scripts that happen to agree. No default changed.
#
#   # the existing production arms, unchanged:
#   python data/solutionDE/run_goal3_ladder.py
#   # the matched production+G control for H-seg x G (step 2 of the plan) — note conf stays at
#   # production's 0.10 while the H-seg arm runs at its own retuned value:
#   python data/solutionDE/run_goal3_ladder.py --remask --dir-tag _g
import argparse
import sys
from pathlib import Path

sys.path.insert(0, r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\src")
sys.path.insert(0, str(Path(__file__).parent))

from _ladder import DATA, DE, add_ladder_args, run_ladder

WEIGHTS = DATA / "weights" / "yolov8-m-boulder-detection-tmp.pt"
TIMINGS = DE / "goal3_timings.json"


def arms_for(dir_tag):
    # Without a dir_tag the dev-crop 256 / 256sr2 arms already exist under their phase-1/2 names
    # (`*_baseline`, `*_e2stream`), so only the 512 arm is listed — that is the historical layout
    # and it must not change. A tagged run is a fresh set, so it needs all three.
    arms = []
    for scene, raster in [("nac", DE / "nac_crop2k.tif"), ("hirise", DE / "hirise_crop4k.tif")]:
        if dir_tag:
            # All three arms in one directory — their filenames already differ by slice size, and
            # a flat layout means the analyzer needs a single --prod-dev-dir instead of three
            # per-arm overrides.
            for ss, sr in ((512, 1), (256, 1), (256, 2)):
                arms.append((f"{scene}{dir_tag}_ladder", raster, ss, sr))
        else:
            arms.append((f"{scene}_512", raster, 512, 1))
    for tile in sorted(DE.glob("hirise_rom??.tif")):
        for ss, sr in ((512, 1), (256, 1), (256, 2)):
            arms.append((f"rom{dir_tag}_ladder/{tile.stem}", tile, ss, sr))
    return arms


def _key(sub, ss, sr):
    return f"{sub.split('/')[-1] if '/' in sub else sub}_ss{ss}" + (f"sr{sr}" if sr > 1 else "")


if __name__ == "__main__":
    args = add_ladder_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    run_ladder(weights=WEIGHTS, arms=arms_for(args.dir_tag),
               timings_path=TIMINGS, args=args, key_fn=_key)
