#!/bin/bash
#SBATCH --job-name=bnet-fuse
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=2:00:00
#SBATCH --output=logs/fuse-%j.out
#SBATCH --error=logs/fuse-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bamaro@stanford.edu
#
# Fuse the three HiRISE arm layers that hirise_arms.sbatch produced. Submit from ~/BoulderNet.
#
# NO GPU. `--fuse-only` reads three GeoPackages and writes one; it never touches the model. So this asks
# for the CPU partition, which should also schedule far faster than the gpu queue.
#   ⚠ If `normal` is rejected, fall back to `--partition=gpu --gres=gpu:1` (proven to work here) and
#   accept the wait. `bigmem` is the other option if 100 G cannot be granted on a normal node.
#
# WHY NOT FUSE ON THE LAPTOP: it needs all three layers resident -- 5.73M polygons across 4.6 GB of
# GeoPackage -- and that machine has 31 GB and paged badly on a *single* 1.7M-detection arm
# (commit 52.7 GB of 92.3, ~9 GB swapped out). 100 G here is comfortable; the arms peaked around 21 GB.
#
# WHY FUSING HERE DOES NOT CHANGE THE SCIENCE: all three arms were produced on this stack, so the fused
# catalog is a Sherlock product wherever the fuse itself runs -- the arms determine the content. The
# fused manifest records `provenance.heterogeneous`, which will be **false** here: a coherent
# single-stack ladder, which is the cleanest outcome available and avoids the mixed-stack seam the
# original 512-only plan would have created.
#
# WHAT IT CHECKS BEFORE WRITING: `--fuse-only` validates from each layer's manifest rather than from a
# ledger -- same raster and weights digests, same operating point, area gate, tracer, contract version
# and mask tag. Per-machine differences (max_det, host, torch) are permitted and recorded. It refuses
# rather than relabelling if the arm set is not the production three.
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
echo "mem requested: ${SLURM_MEM_PER_NODE:-?} MB   cpus: ${SLURM_CPUS_PER_TASK:-?}"

# The raster is needed for its native resolution: the DIoU threshold is interpreted in metres.
[ -f "$RASTER" ] || { echo "MISSING raster: $RASTER" >&2; exit 2; }
n=$(ls -1 "$OUT"/*-downscaled-mask-nms.gpkg 2>/dev/null | wc -l)
[ "$n" -eq 3 ] || { echo "expected 3 arm layers in $OUT, found $n" >&2; exit 2; }

python scripts/run_multiscale_arms.py --fuse-only --raster "$RASTER" --out-dir "$OUT"

echo
echo "Fused catalog, audit and manifest are in: $OUT"
ls -la "$OUT"/*fused* 2>/dev/null || true
echo
echo "Bring these three down (the audit and manifest are small; the catalog is the deliverable):"
echo "  $OUT/*-predictions-fused-production-mask.gpkg"
echo "  $OUT/*-predictions-fused-production-mask-audit.csv"
echo "  $OUT/*-predictions-fused-production-mask-manifest.json"
