#!/bin/bash
# setup_sherlock_env.sh -- build the Solution-H training venv on Sherlock (mlapotre group).
# GPU build: the yolo26m retrain runs on the `gpu` partition. No conda (discouraged on Sherlock);
# venv on $GROUP_HOME per the hirise2ctx convention. Idempotent.
#   Run INSIDE a GPU session so torch sees CUDA during the smoke test:
#     sh_dev -p gpu -G 1 -t 1:00:00
#     bash YOLOv8-BeyondEarth/sherlock/setup_sherlock_env.sh
set -euo pipefail

# --- Python module (Lmod). `ml` can exit 0 on a missing version, so load then VERIFY venv. ---
PYMODULE="${PYMODULE:-python/3.12.1}"          # `ml spider python` to list real versions (>=3.10)
ml "$PYMODULE" 2>/dev/null || true
PYBIN="$(command -v python3 || true)"
if [ -z "$PYBIN" ] || ! "$PYBIN" -c "import venv, ensurepip" >/dev/null 2>&1; then
    echo "ERROR: module '$PYMODULE' gave no venv-capable python3 (got: ${PYBIN:-none})." >&2
    echo "  Fix: 'ml spider python' to find the name, then PYMODULE=python/<ver> bash $0" >&2
    exit 1
fi
echo "using $PYMODULE -> $PYBIN ($($PYBIN --version))"

ENV_DIR="${ENV_DIR:-/home/groups/mlapotre/bamaro/envs/bouldernet26}"
[ -d "$ENV_DIR" ] || "$PYBIN" -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel

# --- CUDA PyTorch: default index ships a CUDA wheel that bundles its own CUDA runtime (no system
# CUDA module needed). ultralytics pulls torch+torchvision; pin ultralytics to the LOCAL version so
# the retrain matches the pipeline env (CLAUDE.md: 8.4.75 is pinned against the production stack).
# --only-binary forbids source builds (Sherlock has old glibc / no system OpenBLAS), so pip picks
# the newest glibc-compatible wheels. ultralytics' deps (numpy, opencv, pillow, scipy, matplotlib,
# pandas, torch, torchvision) are all wheels. shapely + scikit-image are the EXTRA deps that the
# on-cluster OBB-label rebuild needs (build_obb_labels.py -> orientation.ellipse_obb uses
# skimage.measure.EllipseModel + shapely); both ship manylinux wheels. No geo/GDAL stack needed. ---
pip install --only-binary=:all: "ultralytics==8.4.75" "shapely>=2.0" "scikit-image>=0.22"

# --- Smoke test: torch sees CUDA and both YOLO26 archs build in-env (mirrors the local scouting
# that closed the env question -- Segment26 27.1M / OBB26 23.6M on the RTX 5070). ---
python - <<'PY'
import torch
from ultralytics import YOLO
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(),
      "device", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))
for cfg, task in [("yolo26m-seg.yaml", "segment"), ("yolo26m-obb.yaml", "obb")]:
    m = YOLO(cfg, task=task)
    n = sum(p.numel() for p in m.model.parameters())
    print(f"OK  {cfg:16} task={m.task:8} params={n/1e6:.1f}M head={type(m.model.model[-1]).__name__}")
PY

echo "OK -- activate later with:  ml $PYMODULE && source $ENV_DIR/bin/activate"
echo "Next: stage data (see sherlock/README.md), then  sbatch YOLOv8-BeyondEarth/sherlock/train_h.sbatch"
