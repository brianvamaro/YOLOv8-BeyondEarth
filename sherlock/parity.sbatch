#!/bin/bash
#SBATCH --job-name=bnet-parity
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=GPU_MEM:24GB
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/home/users/%u/BoulderNet/logs/parity-%j.out
#SBATCH --error=/home/users/%u/BoulderNet/logs/parity-%j.err
#
# The Sherlock half of the stack-parity check (scripts/review/measure_stack_parity.py).
#
# Runs the full production ladder {512, 256, 256:2} twice on the tracked 2048x2048 NAC dev crop and
# writes catalogs plus a software identity, for comparison against the laptop's `laptop/` directory.
# The laptop half is already recorded and its within-stack floor is EXACT on all three arms, so any
# difference this run produces is attributable to this stack rather than to run-to-run noise.
#
# WHY THIS IS CHEAP. The crop is 4.2 MB and the weights 55 MB, so staging is ~60 MB rather than the
# 1.9 GB HiRISE scene, and the whole ladder takes minutes. The point of using the crop is that it is
# the raster every other parity proof in this project used -- the device-side mask downscale and batch
# neutrality were both measured on it -- so the comparison inherits their standard.
#
# BEFORE SUBMITTING:
#   1. mkdir -p ~/BoulderNet/logs        # SLURM opens --output before the script runs (gotcha 7)
#   2. Stage, preserving these paths under $REPO:
#        data/raster/crop.tif                              (4.2 MB)
#        data/weights/yolov8-m-boulder-detection-tmp.pt    (55 MB)
#      Both are gitignored, so the code tarball does NOT contain them.
#   3. Build the env once, in a dev session, because BATCH NODES ARE OFFLINE:
#        sh_dev -p gpu -G 1 -t 1:00:00
#        bash YOLOv8-BeyondEarth/sherlock/setup_sherlock_inference_env.sh
#
# A 24 GB card is ample for the crop. The eventual real 512-arm scene run needs ~13 GiB at
# max_det=1100 batch 8, so it wants the same constraint or larger -- but that is a different job.
set -euo pipefail

PYMODULE="${PYMODULE:-python/3.12.1}"
ENV_DIR="${ENV_DIR:-/home/groups/mlapotre/bamaro/envs/bouldernet-infer}"
REPO="${REPO:-$HOME/BoulderNet/YOLOv8-BeyondEarth}"

ml "$PYMODULE"
source "$ENV_DIR/bin/activate"
cd "$REPO"

echo "host $(hostname)"
nvidia-smi --query-gpu=name,memory.total,clocks.sm,power.draw --format=csv || true

# Fail early and loudly if the two staged inputs are missing, rather than after a GPU allocation.
for f in data/raster/crop.tif data/weights/yolov8-m-boulder-detection-tmp.pt; do
    [ -f "$f" ] || { echo "MISSING staged input: $f -- see this script's header." >&2; exit 2; }
done

# --repeat 2 establishes this stack's own determinism floor. Without it a cross-stack delta cannot be
# separated from cuDNN autotuning noise, and the comparator records the run as NOT MEASURED.
python scripts/review/measure_stack_parity.py \
    --mode produce \
    --label sherlock \
    --repeat 2 \
    --device cuda:0

echo
echo "Done. Bring this directory back to the laptop:"
echo "  $REPO/data/review/stack_parity/sherlock/"
echo "Then run there:"
echo "  python scripts/review/measure_stack_parity.py --mode compare \\"
echo "      --a data/review/stack_parity/laptop --b data/review/stack_parity/sherlock"
