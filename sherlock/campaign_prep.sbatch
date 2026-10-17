#!/bin/bash
#SBATCH --job-name=bnet-prep
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=logs/prep-%A_%a.out
#SBATCH --error=logs/prep-%A_%a.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=bamaro@stanford.edu
#
# Campaign step 1 (runbook: sherlock/CAMPAIGN.md). One scene-manifest row per array task -> the 8-bit
# GeoTIFF BoulderNet infers on, in the PDS label frame, plus its .prep.json sidecar. No GPU.
#
#   N=$(python scripts/campaign.py count --manifest $M)        # array tasks 0..N-1: derive, never type
#   sbatch --array=0-$((N-1))%8 --export=ALL,MANIFEST=$M sherlock/campaign_prep.sbatch
#
# (M is the manifest CSV; the runbook's shell setup defines it. A range one past the manifest's last row
# launches a task that can only fail.)
#
# FETCH (env):  yes  download whatever is missing, then convert (default)
#               no   convert files already fetched (a partition without outbound internet)
#               only download and stop (what to run on the DTN / a node that has internet)
#
# RESOURCE NOTES, measured on the laptop 2026-10-14 -- re-measure here before trusting them:
#   conversion  ESP_017355_2260 (1.93 Gpx) took 956 s, ~2 Mpx/s, OpenJPEG on all cores with the CPU
#               contended; a 7 Gpx scene would be ~1 h. The read is windowed, so 16G is generous.
#   download    the PDS host served ~50 KB/s to the laptop that day, and 8 parallel connections got
#               29 KB/s between them, so it is a server limit, not a per-connection one. At that rate a
#               1 GB JP2 takes ~6 h, which is the reason for --time 8:00 -- and the reason fetching
#               once on the DTN (FETCH=only there, FETCH=no here) may be the better plan. Measure first:
#                 curl -o /dev/null -w '%{speed_download}\n' --max-time 60 <a JP2 URL>
set -euo pipefail

PYMODULE="${PYMODULE:-python/3.12.1}"
ENV_DIR="${ENV_DIR:-/home/groups/mlapotre/bamaro/envs/bouldernet-infer}"
REPO="${REPO:-$HOME/BoulderNet/YOLOv8-BeyondEarth}"
SCENES="${SCENES:-$SCRATCH/campaign/scenes}"
FETCH="${FETCH:-yes}"
: "${MANIFEST:?set MANIFEST to the scene-manifest CSV}"

ml "$PYMODULE"
source "$ENV_DIR/bin/activate"
cd "$REPO"
case "$FETCH" in
  yes)  MODE=() ;;
  no)   MODE=(--no-fetch) ;;
  only) MODE=(--fetch-only) ;;
  *)    echo "FETCH must be yes|no|only, got $FETCH" >&2; exit 2 ;;
esac

echo "host $(hostname)  job ${SLURM_ARRAY_JOB_ID:-none}_${SLURM_ARRAY_TASK_ID:-?}  fetch=$FETCH"
mkdir -p "$SCENES"
df -h "$SCENES" | tail -1
# ${MODE[@]+"${MODE[@]}"}: el7's bash 4.2 treats an empty "${MODE[@]}" as unbound under `set -u`
GDAL_NUM_THREADS=ALL_CPUS python scripts/prep_pds_scene.py --manifest "$MANIFEST" \
    --row "${SLURM_ARRAY_TASK_ID:?run as an array job}" --out-root "$SCENES" ${MODE[@]+"${MODE[@]}"}
