#!/bin/bash
#SBATCH --job-name=bnet-detect
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_MEM:24GB
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/detect-%A_%a.out
#SBATCH --error=logs/detect-%A_%a.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=bamaro@stanford.edu
#
# Campaign step 2 (runbook: sherlock/CAMPAIGN.md). One manifest row per array task: detection on the
# prepared raster, then `campaign.py verify`, which fails the task unless every arm reports
# "truncation": {} and ran on the prepared raster, and writes SHA256SUMS + handoff.json.
#
#   PREP=$(sbatch --parsable --array=0-62%8 --export=ALL,MANIFEST=$M sherlock/campaign_prep.sbatch)
#   sbatch --array=0-62 --dependency=aftercorr:$PREP --export=ALL,MANIFEST=$M,ARMS=256 \
#          sherlock/campaign_detect.sbatch
#
# `aftercorr` starts task i when prep task i succeeded, so a slow download does not hold up the rest.
#
# ARMS (env): "256" is configuration 1. "512 256" runs configuration 2's two arms UNFUSED: the
#   production runner refuses to fuse any set but its own three, and a {512,256} preset is D9's call.
#   Configuration 3 (SAM2 re-mask) is not wired here: it has never run on this stack.
# MAX_DET (env, default 1100): the starting cap. ON_MAX_DET (env, default escalate): an arm with a slice
#   within 10% of its cap is re-run at 2x the cap and 1/2 the batch until none is, so the worst-case
#   mask memory (~1 MiB per retained detection x batch at imgsz 1024) stays put; a resubmit resumes at
#   the attempt the ledger was on. ON_MAX_DET=error fails the scene instead (verify still refuses any
#   truncated arm either way). Runbook: "When an arm escalates".
# --mem=64G: hirise_arms.sbatch asked 128G for a 1.93 Gpx scene under the old whole-raster path. Lever
#   N2 removed the three whole-scene copies (the PNG, the RGB scene, the int64 footprint mask), so what
#   is left is the baseline plus the detections. Every manifest now records peak_rss_bytes and sacct
#   reports MaxRSS below, so the pilot measures this rather than assuming it.
# DEST (env, optional): a durable directory (Oak / group storage). $SCRATCH is purged on inactivity,
#   and the 2026-08-19 HiRISE arms already live only there. Unset = the catalogs stay on $SCRATCH.
set -euo pipefail

PYMODULE="${PYMODULE:-python/3.12.1}"
ENV_DIR="${ENV_DIR:-/home/groups/mlapotre/bamaro/envs/bouldernet-infer}"
REPO="${REPO:-$HOME/BoulderNet/YOLOv8-BeyondEarth}"
SCENES="${SCENES:-$SCRATCH/campaign/scenes}"
OUT="${OUT:-$SCRATCH/campaign/out}"
ARMS="${ARMS:-256}"
MAX_DET="${MAX_DET:-1100}"
ON_MAX_DET="${ON_MAX_DET:-escalate}"
: "${MANIFEST:?set MANIFEST to the scene-manifest CSV}"
ROW="${SLURM_ARRAY_TASK_ID:?run as an array job}"

ml "$PYMODULE"
source "$ENV_DIR/bin/activate"
cd "$REPO"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# After lever N2 the only raster-sized memory left is GDAL's block cache, whose default is 5% of the
# node's *physical* RAM (not of this job's --mem). Slices are read row-major, so a few tile rows are
# all it ever reuses (~50 MB at 45k px wide); 512 MB makes the job's RAM independent of the node it
# lands on. Measured: capped at 128 MB, slicing a 0.51 Gpx band peaks 0.14 GiB above baseline vs
# 4.78 GiB for the old path (data/review/n2_memory_cachemax128.json).
export GDAL_CACHEMAX="${GDAL_CACHEMAX:-512}"

OBS=$(python scripts/campaign.py obs --manifest "$MANIFEST" --row "$ROW")
echo "host $(hostname)  job ${SLURM_ARRAY_JOB_ID:-none}_${ROW}  obs $OBS  arms $ARMS  max_det $MAX_DET ($ON_MAX_DET)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
[ -f "$REPO/data/weights/yolov8-m-boulder-detection-tmp.pt" ] || {
    echo "MISSING weights under $REPO/data/weights" >&2; exit 2; }

# srun makes the detection its own job step, so sacct can report its MaxRSS below
# shellcheck disable=SC2086
srun --ntasks=1 python scripts/campaign.py detect --manifest "$MANIFEST" --row "$ROW" \
    --scenes "$SCENES" --out "$OUT" --arms $ARMS --max-det "$MAX_DET" --on-max-det "$ON_MAX_DET"

sacct -j "${SLURM_JOB_ID}" --format=JobID,MaxRSS,Elapsed,State --units=G 2>/dev/null || true

if [ -n "${DEST:-}" ]; then
    mkdir -p "$DEST/$OBS"
    rsync -a "$OUT/$OBS/" "$DEST/$OBS/"
    (cd "$DEST/$OBS" && sha256sum -c --quiet SHA256SUMS) && echo "copied + verified -> $DEST/$OBS"
else
    echo "DEST unset: pull $OUT/$OBS/ through the DTN now and run 'sha256sum -c SHA256SUMS' on arrival"
    echo "  (\$SCRATCH purges on inactivity -- see sherlock/CAMPAIGN.md, 'Getting the catalogs off the cluster')"
fi
