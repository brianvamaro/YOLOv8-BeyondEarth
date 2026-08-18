#!/bin/bash
# setup_sherlock_inference_env.sh -- build an INFERENCE venv on Sherlock (mlapotre group).
#
# `setup_sherlock_env.sh` builds the *training* env and says outright "No geo/GDAL stack needed".
# Inference needs that stack: predict.py imports cv2, geopandas, lsnms, sahi.slicing, rasterio,
# rastertools_BOULDERING and shptools_BOULDERING. So this is a separate env at a separate path --
# it deliberately does NOT touch the training venv, which is a known-good artifact.
#
# WHY VERSIONS ARE PINNED. This env exists to answer whether the pipeline is stack-neutral
# (scripts/review/measure_stack_parity.py), and that is only answerable if the libraries are held
# fixed, so the remaining difference is the hardware and its cuDNN kernels. Pinning is the experiment,
# not fussiness.
#
# One pin could NOT be matched: torch. This cluster's wheel ceiling is 2.6.0 against the laptop's
# 2.11.0 -- see the long note at the torch install below for the measurement and for what it costs the
# experiment. Everything else on this list is the laptop's exact version, so torch is the single known
# library difference rather than one of many, which is what keeps the result interpretable.
#
# The pins are the laptop's installed versions as of 2026-08-17, and every one satisfies the bounds
# `pyproject.toml` declares (queue item 8 / a4a24da). Note especially opencv: the cap is
# `opencv-python>=4.7.0.72,<5` because `cv2.resize(..., INTER_AREA)` sits on the production
# mask-downscale path, and an unpinned install here would now select OpenCV 5. We pin the HEADLESS
# build at the laptop's exact version -- headless differs from the GUI build only in GUI/GL linkage,
# not in the resize kernel, and the GUI build cannot import on a headless node (libGL.so.1).
#
#   Run INSIDE a dev session -- BATCH NODES ARE OFFLINE, so pip must run here, not in the job:
#     sh_dev -p gpu -g 1 -c 4 -m 16000 -t 2:00:00
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

# --- torch first, so ultralytics cannot pull a different version underneath it ---
#
# THIS CLUSTER CANNOT RUN THE LAPTOP'S TORCH, so exact library parity is off the table. Measured on a
# dev node, 2026-08-18, `pip install torch==2.11.0` reports:
#   (from versions: 2.2.0, 2.2.1, 2.2.2, 2.3.0, 2.3.1, 2.4.0, 2.4.1, 2.5.0, 2.5.1, 2.6.0)
# 2.6.0 is the ceiling. That is the signature of an older glibc: torch moved to `manylinux_2_28`
# wheels at ~2.7, so anything newer is filtered as incompatible and only a source build would bridge
# it -- which `--only-binary` correctly forbids. Two earlier drafts failed here, first pointing at
# download.pytorch.org/whl/cu128 (not reachable from the compute nodes; PyPI is) and then asking PyPI
# for 2.11.0 (does not exist for this platform).
#
# 2.6.0 is also almost certainly what the *training* env resolved to, since
# `setup_sherlock_env.sh` installs ultralytics unpinned and lets the resolver choose -- and the H
# retrain ran on it. So this is the proven configuration, not a guess. It satisfies
# `pyproject.toml`'s `torch>=2.2.0,<3`, so no declared bound is breached.
#
# WHAT THIS COSTS THE EXPERIMENT, stated plainly: a laptop-vs-Sherlock difference now has two possible
# causes, the GPU and a five-minor-version torch gap, so a *failure* would not be attributable on its
# own. That is why the parity doc registers a 2x2 decomposition to run only if needed: install torch
# 2.6.0 in a throwaway env on the laptop, and laptop-2.11 vs laptop-2.6 isolates torch while
# laptop-2.6 vs sherlock-2.6 isolates hardware. Cheap, because it is all on the crop. A *pass* needs no
# decomposition and is in fact a stronger result, since parity would hold across both variables at once.
#
# torchvision 0.21.0 is the release paired with torch 2.6.0; mismatching that pair is the next thing
# that breaks. TORCH_VERSION / TORCHVISION_VERSION override, and code_identity() records what actually
# got installed, so `--mode compare` lists it under `stack_differences` either way.
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.21.0}"
pip install --only-binary=:all: "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"

# --- the rest of the stack: the laptop's exact version where a wheel exists here, else the bound ---
#
# `--only-binary` forbids source builds (old glibc, no system OpenBLAS/GDAL), and that same old glibc
# means some of the laptop's exact versions simply have no wheel for this platform. Measured: rasterio
# stops at **1.4.3** against the laptop's 1.4.4, the same way torch stops at 2.6.0.
#
# Discovering those one per run wasted several rounds, so this tries the exact pin and falls back to
# the bound `pyproject.toml` declares -- the project's own statement of what is safe, and every bound
# was verified to admit the working environment (queue item 8). Each fallback is announced, collected,
# and summarised at the end, so ONE run tells us everything that had to move. `code_identity()` then
# stamps the versions actually installed into the parity artifact, and `--mode compare` lists them
# under `stack_differences`, so nothing here is silent.
#
# pyogrio and pillow are deliberately NOT pinned: neither is declared in pyproject, both arrive as
# dependencies (of geopandas and torchvision), and pinning them only adds failure modes.
RELAXED=()

pin_install() {           # pin_install <exact spec> <declared bound>
    local exact="$1" bound="$2" pkg got
    pkg="${exact%%==*}"
    if pip install --only-binary=:all: "$exact" >/dev/null 2>&1; then
        echo "  pinned    $exact"
    elif pip install --only-binary=:all: "$bound"; then
        got="$(pip show "$pkg" 2>/dev/null | awk '/^Version:/{print $2}')"
        echo "  RELAXED   $pkg: wanted ${exact#*==}, no wheel here -> installed $got (bound $bound)"
        RELAXED+=("$pkg: wanted ${exact#*==}, got $got")
    else
        echo "  FAILED    $pkg: neither $exact nor $bound is installable" >&2
        return 1
    fi
}

echo "installing the geo/ML stack (exact pin where available, declared bound otherwise):"
pin_install "ultralytics==8.4.75"   "ultralytics>=8.1.15,<9"
pin_install "sahi==0.12.1"          "sahi>=0.12.1,<0.13"
pin_install "numpy==2.2.6"          "numpy>=1.24.4,<3"
pin_install "scipy==1.15.3"         "scipy>=1.10.0,<2"
pin_install "shapely==2.1.2"        "Shapely>=2.0.3,<3"
pin_install "scikit-image==0.25.2"  "scikit-image>=0.22.0,<0.26"
pin_install "rasterio==1.4.4"       "rasterio>=1.3.9,<2"
pin_install "geopandas==1.1.3"      "geopandas>=0.14.1,<2"
pin_install "pandas==2.3.3"         "pandas>=2.2.1,<3"
pin_install "lsnms==0.4.5"          "lsnms>=0.4.4,<0.5"

# HEADLESS OpenCV. ultralytics depends on `opencv-python`, whose native lib needs libGL.so.1 (absent on
# headless nodes). Uninstall the GUI build first -- the two conflict if both are installed. The cap
# matters here more than anywhere: `cv2.resize(..., INTER_AREA)` is on the production mask-downscale
# path and OpenCV 5 would be selected without it.
pip uninstall -y opencv-python opencv-contrib-python >/dev/null 2>&1 || true
pin_install "opencv-python-headless==4.13.0.92" "opencv-python-headless>=4.7.0.72,<5"

# --- the two BOULDERING helpers live on test.pypi (CLAUDE.md), so that index is required. The
# extra-index keeps their ordinary dependencies resolving from real PyPI. Neither is version-pinned in
# pyproject, so the fallback here is simply "unpinned" rather than a declared bound. ---
TESTPYPI=(--index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple)
if pip install --only-binary=:all: "${TESTPYPI[@]}" \
        "rastertools_BOULDERING==1.0.0" "shptools_BOULDERING==0.0.5" >/dev/null 2>&1; then
    echo "  pinned    rastertools_BOULDERING==1.0.0, shptools_BOULDERING==0.0.5"
else
    pip install --only-binary=:all: "${TESTPYPI[@]}" rastertools_BOULDERING shptools_BOULDERING
    for pkg in rastertools_BOULDERING shptools_BOULDERING; do
        got="$(pip show "$pkg" 2>/dev/null | awk '/^Version:/{print $2}')"
        echo "  RELAXED   $pkg: exact pin unavailable -> installed $got"
        RELAXED+=("$pkg: got $got")
    done
fi

# --- say plainly what could not be matched, so it is read now rather than found in the artifact ---
echo
if [ ${#RELAXED[@]} -eq 0 ]; then
    echo "All non-torch pins matched the laptop exactly. torch is the only library difference."
else
    echo "RELAXED PINS (${#RELAXED[@]}) -- these differ from the laptop, on top of torch:"
    printf '  - %s\n' "${RELAXED[@]}"
    echo "Expected and recorded: code_identity() stamps the real versions into the parity artifact and"
    echo "--mode compare lists them under stack_differences. More differences weaken ATTRIBUTION of a"
    echo "failure, not the validity of a pass."
fi

# --- Smoke test: every import on the predict path, plus CUDA, plus the identity this env will
# stamp into the parity artifact. If this passes, the parity run cannot fail on environment. ---
# glibc is the reason torch is capped at 2.6.0 here, so record it rather than leave it to be
# rediscovered the next time someone wonders why a newer wheel will not install.
echo "glibc: $(ldd --version | head -1)"

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
