# H-seg adoption check, GPU half (plan: "H-seg adoption: decision point = end of phase 3";
# the phase-3 crop suite is the acceptance harness for a weight swap). Re-runs the FULL ladder
# under the H-seg weights (yolo26m-h-seg.pt) on both dev crops + the 20 ROM tiles — including
# the 256 arm (the parity baseline resets with the weights). Outputs under preds/hseg_* /
# preds/hseg_rom_ladder mirror the production layout; analysis = analyze_hseg_acceptance.py.
# Context: H-seg = mask mAP50 0.467->0.569 but box mAP 0.762->0.749 (slightly down) — under a
# phase-5 G-remask future SAM2 is prompted from the BOXES, so the acceptance axes are recall +
# box localization, not the mask headline. Idempotent; timings to hseg_timings.json.
#
# 2026-08-09: the ladder body moved to _ladder.py, shared with run_goal3_ladder.py, so the H-seg arm
# and its matched production control cannot drift apart in configuration (they were byte-identical
# copies, which made "matched" an accident). No default changed — a bare run still reproduces the
# 2026-07-10 catalogs and skips them.
#
#   # the 2026-07-10 arm, unchanged:
#   python data/solutionDE/run_hseg_ladder.py
#   # H-seg x G, at a retuned operating point (step 0) with SAM2 re-masking (step 1):
#   python data/solutionDE/run_hseg_ladder.py --conf 0.15 --remask --dir-tag _g
import argparse
import sys
from pathlib import Path

sys.path.insert(0, r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\src")
sys.path.insert(0, str(Path(__file__).parent))

from _ladder import DATA, DE, add_ladder_args, run_ladder

WEIGHTS = DATA / "weights" / "yolo26m-h-seg.pt"
TIMINGS = DE / "hseg_timings.json"


def arms_for(dir_tag):
    arms = []
    for scene, raster in [("nac", DE / "nac_crop2k.tif"), ("hirise", DE / "hirise_crop4k.tif")]:
        for ss, sr in ((512, 1), (256, 1), (256, 2)):
            arms.append((f"hseg{dir_tag}_{scene}", raster, ss, sr))
    for tile in sorted(DE.glob("hirise_rom??.tif")):
        for ss, sr in ((512, 1), (256, 1), (256, 2)):
            arms.append((f"hseg{dir_tag}_rom_ladder/{tile.stem}", tile, ss, sr))
    return arms


def _key(sub, ss, sr):
    return f"{sub.replace('/', '_')}_ss{ss}" + (f"sr{sr}" if sr > 1 else "")


if __name__ == "__main__":
    args = add_ladder_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    run_ladder(weights=WEIGHTS, arms=arms_for(args.dir_tag),
               timings_path=TIMINGS, args=args, key_fn=_key)
