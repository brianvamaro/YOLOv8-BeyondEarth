#!/bin/bash
# setup_sherlock_inference_env.sh -- build an INFERENCE venv on Sherlock (mlapotre group).
#
# `setup_sherlock_env.sh` builds the *training* env and says outright "No geo/GDAL stack needed".
# Inference needs that stack: predict.py imports cv2, geopandas, lsnms, sahi.slicing, rasterio,
# rastertools_BOULDERING and shptools_BOULDERING. So this is a separate env at a separate path --
# it deliberately does NOT touch the training venv, which is a known-good artifact.
#
# WHY EVERY VERSION IS PINNED. This env exists to answer whether the pipeline is stack-neutral
# (scripts/review/measure_stack_parity.py). That question is only answerable if the libraries are
# held fixed, so the one remaining difference is the hardware and its cuDNN kernels. Pinning is
# therefore the experiment, not fussiness: a difference found here is attributable, and a
# difference found against a freely-resolved env would not be.
#
# The pins are the laptop's installed versions as of 2026-08-17, and every one satisfies the bounds
# `pyproject.toml` declares (queue item 8 / a4a24da). Note especially opencv: the cap is
# `opencv-python>=4.7.0.72,<5` because `cv2.resize(..., INTER_AREA)` sits on the production
# mask-downscale path, and an unpinned install here would now select OpenCV 5. We pin the HEADLESS
# build at the laptop's exact version -- headless differs from the GUI build only in GUI/GL linkage,
# not in the resize kernel, and the GUI build cannot import on a headless node (libGL.so.1).
#
#   Run INSIDE a dev session -- BATCH NODES ARE OFFLINE, so pip must run here, not in the job:
#     sh_dev -p gpu -G 1 -t 1:00:00
#     bash YOLOv8-BeyondEarth/sherlock/setup_sherlock_inference_env.sh
set -euo pipefail

PYMODULE="${PYMODULE:-python/3.12.1}"
ml "$PYMODULE" 2>/dev/null || true
PYBIN="$(command -v python3 || true)"
if [ -z "$PYBIN" ] || ! "$PYBIN" -c "import venv, ensurepip" >/dev/null 2>&1; then
    echo "ERROR: module '$PYMODULE' gave no venv-capable python3 (got: ${PYBIN:-none})." >&2
    echo "  Fix: 'ml spider python' to find the name, then PYMODULE=python/<ver> bash $0" >&2
    exit 1
fi
echo "using $PYMODULE -> $PYBIN ($($PYBIN --version))"

# A DIFFERENT path from the training env on purpose: bouldernet26 is a working training artifact and
# adding a geo stack to it risks a resolver change nobody asked for.
ENV_DIR="${ENV_DIR:-/home/groups/mlapotre/bamaro/envs/bouldernet-infer}"
[ -d "$ENV_DIR" ] || "$PYBIN" -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel

# --- torch first, from the CUDA index, so ultralytics cannot pull a different build ---
# The laptop runs 2.11.0+cu128. cu128 wheels are forward-compatible with Ampere/Hopper, so the same
# build runs on a 3090 / A100 / H100. Matching it removes torch as a variable.
pip install --only-binary=:all: --index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.11.0" "torchvision==0.26.0"

# --- the rest of the stack, at the laptop's exact versions ---
# --only-binary forbids source builds (old glibc, no system OpenBLAS/GDAL). rasterio and geopandas
# ship manylinux wheels that bundle their own GDAL/PROJ, so no GDAL module is needed.
pip install --only-binary=:all: \
    "ultralytics==8.4.75" \
    "sahi==0.12.1" \
    "numpy==2.2.6" \
    "scipy==1.15.3" \
    "shapely==2.1.2" \
    "scikit-image==0.25.2" \
    "rasterio==1.4.4" \
    "geopandas==1.1.3" \
    "pyogrio==0.12.1" \
    "pandas==2.3.3" \
    "pillow==12.2.0" \
    "lsnms==0.4.5"

# HEADLESS OpenCV at the laptop's exact version. ultralytics depends on `opencv-python`, whose native
# lib needs libGL.so.1 (absent on Sherlock's headless nodes). Uninstall the GUI build first -- the two
# conflict if both are installed.
pip uninstall -y opencv-python opencv-contrib-python >/dev/null 2>&1 || true
pip install --only-binary=:all: "opencv-python-headless==4.13.0.92"

# --- the two BOULDERING helpers live on test.pypi (CLAUDE.md), so that index is required. The
# extra-index keeps their ordinary dependencies resolving from real PyPI. ---
pip install --only-binary=:all: \
    --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple \
    "rastertools_BOULDERING==1.0.0" "shptools_BOULDERING==0.0.5"

# --- Smoke test: every import on the predict path, plus CUDA, plus the identity this env will
# stamp into the parity artifact. If this passes, the parity run cannot fail on environment. ---
python - <<'PY'
import sys, pathlib
repo = pathlib.Path.home() / "BoulderNet" / "YOLOv8-BeyondEarth"
sys.path.insert(0, str(repo / "src"))
import torch, cv2, geopandas, rasterio, shapely, skimage, scipy, numpy
from lsnms import nms                                             # noqa: F401
from sahi.slicing import slice_image                              # noqa: F401
from rastertools_BOULDERING import raster, convert, metadata      # noqa: F401
from shptools_BOULDERING import shp                               # noqa: F401
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "device", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))
print("cv2", cv2.__version__, "| numpy", numpy.__version__, "| rasterio", rasterio.__version__,
      "| geopandas", geopandas.__version__, "| shapely", shapely.__version__,
      "| skimage", skimage.__version__, "| scipy", scipy.__version__)
from YOLOv8BeyondEarth import presets
print("code_identity:", presets.code_identity())
print("PRESET_VERSION:", presets.PRESET_VERSION)
PY

echo
echo "OK -- activate later with:  ml $PYMODULE && source $ENV_DIR/bin/activate"
echo "Next: stage crop.tif + the boulder weights (see sherlock/README.md), then"
echo "  sbatch YOLOv8-BeyondEarth/sherlock/parity.sbatch"
