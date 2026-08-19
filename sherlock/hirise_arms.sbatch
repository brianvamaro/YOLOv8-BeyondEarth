#!/bin/bash
#SBATCH --job-name=bnet-hirise
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_MEM:24GB
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/hirise-%j.out
#SBATCH --error=logs/hirise-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bamaro@stanford.edu
#
# The full HiRISE production ladder {512, 256, 256:2} at conf 0.10. Submit from ~/BoulderNet.
#
# WHY THIS RUNS HERE AND NOT ON THE LAPTOP. Measured 2026-08-18: the HiRISE **256** arm dies with a hard
# `CUDA error: out of memory` after 75.5 min on the 8 GiB laptop card at the shipped batch_size=8. The
# 7,754 MiB "0.95x of the card" figure that made it look like it fit is a **dev-crop** measurement, and
# scene density is not crop density. So it is not only the 512 arm that needs a bigger GPU -- the
# production arm does too, at the declared operating point. The laptop also paged badly (commit 52.7 GB
# of 92.3, ~9 GB of the arm process swapped out), which is why `--mem=128G` is requested here: on this
# node the ratchet simply fits, and the fix is memory rather than cleverness.
#
# RESOURCE NOTES, each from a measurement rather than a guess:
#   GPU_MEM:24GB  the 512 arm needs ~13 GiB at max_det=1100, batch 8.
#   --mem=128G    the HiRISE 256 arm reached 22.9 GB commit on the laptop at ~57% of the scene; the SR
#                 arm carries ~1.9x its detections, projecting ~65 GB. 128 G leaves real headroom.
#                 If the queue will not grant it, 96 G is the next thing to try.
#   --time=12:00  the laptop ladder projected 5.2 h while power-capped at 24 W. This card is not capped
#                 and will not page, so it should be well under, but a timeout would waste the night.
#
# WHY --max-det 1100 AND WHY THAT IS NOT A DEVIATION THAT MATTERS:
#   The 512 arm needs it -- 1013 measured on the densest HiRISE slices, against a preset 600 it cannot
#   clear. `effective_max_det` is run-global, so all three arms get 1100. For the 256 arm that is
#   **nominal, not substantive**: its worst HiRISE slice is 370 (X-06, conf 0.10), so neither 600 nor
#   1100 binds and the output is identical to a production-preset run. Verify rather than trust it: the
#   manifests must show `"truncation": {}`. If any arm reports truncation, that arm's catalog is a lower
#   bound and must not be used.
#
# WHY --no-fuse:
#   Whether a Sherlock-produced *production* catalog is acceptable is an open decision, and the
#   stack-parity result (sub-pixel geometry differences, membership differences only at the conf floor)
#   informs it but does not settle it. Fusing costs ~112 s and can be done later, anywhere, with
#   `--fuse-only`, which records per-arm provenance. Do not bake that decision into an overnight job.
#
# BEFORE SUBMITTING:
#   1. mkdir -p ~/BoulderNet/logs
#   2. Stage the raster into $SCRATCH (1.9 GB; NOT home, which has a small quota):
#        mkdir -p $SCRATCH/hirise
#        # upload ESP_017355_2260_RED.tif there via OnDemand Files
#      The 8-bit PNG conversion is written *beside the raster* (~715 MB) with a `.source.json` stamp,
#      so it happens once and is reused on a resume. Budget ~7 GB in $SCRATCH for raster + PNG + layers.
#   3. The code and weights are already at ~/BoulderNet/YOLOv8-BeyondEarth from the parity staging.
#
# RESUME: the ledger is per-arm and content-keyed, so re-submitting after a timeout or a preemption
# skips whatever finished and continues. Completed arms are not recomputed.
set -euo pipefail

PYMODULE="${PYMODULE:-python/3.12.1}"
ENV_DIR="${ENV_DIR:-/home/groups/mlapotre/bamaro/envs/bouldernet-infer}"
REPO="${REPO:-$HOME/BoulderNet/YOLOv8-BeyondEarth}"
RASTER="${RASTER:-$SCRATCH/hirise/ESP_017355_2260_RED.tif}"
OUT="${OUT:-$SCRATCH/hirise/out}"

ml "$PYMODULE"
source "$ENV_DIR/bin/activate"
cd "$REPO"

echo "host $(hostname)  job ${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total,enforced.power.limit --format=csv || true
echo "mem requested: ${SLURM_MEM_PER_NODE:-?} MB   cpus: ${SLURM_CPUS_PER_TASK:-?}"

# Fail before the GPU is spent, not after.
[ -f "$RASTER" ] || { echo "MISSING raster: $RASTER -- see this script's header." >&2; exit 2; }
[ -f "$REPO/data/weights/yolov8-m-boulder-detection-tmp.pt" ] || {
    echo "MISSING weights under $REPO/data/weights" >&2; exit 2; }
mkdir -p "$OUT"
echo "scratch space:"; df -h "$(dirname "$RASTER")" | tail -1

python scripts/run_multiscale_arms.py \
    --raster "$RASTER" \
    --out-dir "$OUT" \
    --arms 512 256 256:2 \
    --max-det 1100 \
    --on-max-det error \
    --no-fuse

echo
echo "Arm layers and manifests are in: $OUT"
echo "Check every manifest reports \"truncation\": {} before using any of them:"
echo "  grep -h '\"truncation\"' $OUT/*-manifest.json"
echo "Ledger (for resume): $OUT/run_ledger.jsonl"
