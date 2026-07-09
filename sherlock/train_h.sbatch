#!/bin/bash
# train_h.sbatch -- Solution H retrain on Sherlock's GPU partition (mlapotre group).
# Trains BOTH variants at the production recipe (imgsz 1024, 200 ep, m-scale, degrees=0):
#   H-seg  (yolo26m-seg, boulder2024 seg labels)  and  H-obb (yolo26m-obb, EllipseModel OBB labels).
# SUBMIT FROM ~/BoulderNet, and create the log dir FIRST -- SLURM opens --output BEFORE the script
# runs (so the in-script `mkdir -p logs` is too late; a missing logs/ dir = instant exit 1, no log):
#   mkdir -p ~/BoulderNet/logs   # once; setup_sherlock_env.sh also does this
# Logs land in ~/BoulderNet/logs/ (relative to the submission dir, not the YOLOv8-BeyondEarth cd).
# Override anything via env vars, e.g.:
#   sbatch --export=ALL,VARIANT=obb,BATCH=16 YOLOv8-BeyondEarth/sherlock/train_h.sbatch
#   sbatch --export=ALL,EPOCHS=200,IMGSZ=1024 YOLOv8-BeyondEarth/sherlock/train_h.sbatch
#SBATCH --job-name=h-retrain
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_MEM:24GB
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=36:00:00
#SBATCH --output=logs/h-retrain-%j.out
#SBATCH --error=logs/h-retrain-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bamaro@stanford.edu

set -euo pipefail
mkdir -p logs
ml python/3.12.1
source "/home/groups/mlapotre/bamaro/envs/bouldernet26/bin/activate"

# Repo root on Sherlock: the BoulderNet tree ($HOME/BoulderNet/{YOLOv8-BeyondEarth,data_prieur}).
# train_h.py resolves data_prieur relative to this, so keep the two-level layout intact.
REPO="${REPO:-$HOME/BoulderNet}"
cd "$REPO/YOLOv8-BeyondEarth"

# Heavy training outputs -> $SCRATCH (big/fast, not backed up); scp best.pt back to the laptop.
OUT="${H_OUT:-$SCRATCH/bouldernet/h_training}"
mkdir -p "$OUT"

VARIANT="${VARIANT:-both}"; EPOCHS="${EPOCHS:-200}"; BATCH="${BATCH:-8}"
IMGSZ="${IMGSZ:-1024}"; NAME="${NAME:-h_full}"; MASK_RATIO="${MASK_RATIO:-1}"
# mask_ratio=1 (Prieur, faithful) OOMs a 24GB 3090 on dense tiles even at batch 1 (the seg mask loss
# builds full-res [n_inst,1024,1024] tensors). Options: (a) request a >=40GB GPU and keep it 1
# (--constraint=GPU_MEM:40GB overrides the header); (b) MASK_RATIO=4 to fit 24GB (16x less mask mem;
# minor deviation from the baseline's mask_ratio=1). OBB ignores mask_ratio.
# BATCH=8 default: on a 24 GB 3090, batch 16 at imgsz 1024 pins VRAM at ~23 GB and the
# TaskAlignedAssigner spills to CPU every step (huge slowdown -> ~15 min/epoch). batch 8 keeps the
# assigner on GPU. Prefer submitting seg and obb as SEPARATE jobs (VARIANT=seg / VARIANT=obb) so
# each fits comfortably under --time and a failure doesn't take both down.

echo "Host: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "variant=$VARIANT epochs=$EPOCHS batch=$BATCH imgsz=$IMGSZ out=$OUT"

# The OBB dataset's images/ is a symlink tree built by build_obb_labels.py (its non-Windows path
# uses os.symlink). If data_prieur/boulder2024_obb is missing/stale after staging, (re)build it:
[ -d "$REPO/data_prieur/boulder2024_obb/train/images" ] || python scripts/build_obb_labels.py

# The committed boulder2024_local.yaml carries a machine-specific absolute `path:` (a Windows path
# on the laptop), which ultralytics can't resolve on Linux. Rewrite it to the cluster location so
# the seg dataset's images are found. (The OBB yaml is generated on-cluster, so it's already correct.)
sed -i "s|^path:.*|path: $REPO/data_prieur/boulder2024|" \
    "$REPO/data_prieur/boulder2024/boulder2024_local.yaml"

# AMP=0 disables mixed precision (numerically stable; the first obb run diverged to NaN with AMP on,
# because the offline node skipped the AMP self-check). Default AMP=1 (on).
AMP_FLAG=""; [ "${AMP:-1}" = "0" ] && AMP_FLAG="--no-amp"
DEGREES="${DEGREES:-0}"      # DEGREES=180 = R-A rotation aug (H-obb cardinal-prior salvage)
python scripts/train_h.py --variant "$VARIANT" --epochs "$EPOCHS" --batch "$BATCH" \
    --imgsz "$IMGSZ" --mask-ratio "$MASK_RATIO" --degrees "$DEGREES" $AMP_FLAG \
    --project "$OUT" --name "$NAME" --device 0

echo "done -- runs in $OUT/${NAME}_{seg,obb}; scp weights/best.pt back to the laptop."
