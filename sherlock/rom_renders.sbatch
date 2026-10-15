#!/bin/bash
#SBATCH --job-name=bnet-rom
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_MEM:24GB
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/rom-%j.out
#SBATCH --error=logs/rom-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bamaro@stanford.edu
#
# Configuration-2 ROM renders for the tiles the 8 GiB laptop card cannot hold
# (scripts/render_rom_configs.py records them as "CUDA out of memory": ESP_017355_2260's densest 512
# slices need ~5 GB of transient masks each, even at batch 1).
#
# ONLY CONFIGURATION 2 (the YOLO arms + fusion) RUNS HERE. Configuration 3's SAM2 re-mask has never
# run on this stack, while YOLO's has a measured stack-equivalence result (notebook Stack_equivalence).
# So: render config 2 here, copy the config2/ outputs back, then run config 3 on the laptop:
#   laptop:   python scripts/render_rom_configs.py --obs ESP_017355_2260 --configs 3
#
# STAGING: copy ONLY the failed tiles' crops (data/rom_label_error/<obs>/<stem>.tif, ~1 MB each, the
# names are in data/rom_label_error/render_rom_configs_summary.json under "failed") into
# $REPO/data/rom_label_error/<obs>/ here, with the weights in $REPO/data/weights/. The renderer only
# visits crops that exist, and skips any tile that is already done.
set -euo pipefail

PYMODULE="${PYMODULE:-python/3.12.1}"
ENV_DIR="${ENV_DIR:-/home/groups/mlapotre/bamaro/envs/bouldernet-infer}"
REPO="${REPO:-$HOME/BoulderNet/YOLOv8-BeyondEarth}"
OBS="${OBS:-ESP_017355_2260}"

ml "$PYMODULE"
source "$ENV_DIR/bin/activate"
cd "$REPO"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export GDAL_CACHEMAX="${GDAL_CACHEMAX:-512}"

echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  obs $OBS"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
ls "$REPO/data/rom_label_error/$OBS/"*_rom??_t*.tif >/dev/null || {
    echo "no crops staged in data/rom_label_error/$OBS/ -- see this script's header" >&2; exit 2; }

python -u scripts/render_rom_configs.py --obs "$OBS" --configs 2

echo
echo "Copy back: $REPO/data/rom_label_error/$OBS/config2/  (then run --configs 3 on the laptop)"
