"""Declared dependency bounds must admit the environment that actually produced the results.

Two failure directions, and this catches both:

* **No upper bound** — until 2026-08-09 every geometry dependency was lower-only, so a fresh resolve
  was free to take the next major. The live trigger was OpenCV **5.0.0.93**, which sits on the
  production `cv2.resize(..., INTER_AREA)` mask-downscale path: a numerical change there moves every
  polygon in every catalog and nothing raises.
* **A cap that excludes the working environment** — equally bad, and easy to introduce by capping at
  the version you happen to remember rather than the one installed.

The bounds are not a reproducibility record; `presets.write_manifest` -> `code` is. They only keep a
fresh install from being obviously wrong.
"""
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest
from packaging.requirements import Requirement

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

#: Result-defining libraries whose breaking releases change numbers rather than raising. Each needs
#: an upper bound; the reason is recorded next to the dependency list in pyproject.toml.
MUST_BE_CAPPED = {"opencv-python", "scikit-image", "Shapely", "scipy", "numpy", "geopandas",
                  "pandas", "rasterio", "lsnms", "torch", "ultralytics", "sahi"}


def _declared():
    text = PYPROJECT.read_text(encoding="utf-8")
    out = []
    for block in re.findall(r"^(?:dependencies|dev)\s*=\s*\[(.*?)^\]", text, re.S | re.M):
        out += re.findall(r'"([^"]+)"', block)
    assert out, "parsed no dependencies out of pyproject.toml"
    return [Requirement(s) for s in out]


@pytest.mark.parametrize("req", _declared(), ids=lambda r: r.name)
def test_bound_admits_the_installed_version(req):
    try:
        have = version(req.name)
    except PackageNotFoundError:
        pytest.skip(f"{req.name} not installed (local fork or optional)")
    assert req.specifier.contains(have, prereleases=True), (
        f"pyproject declares {req.name}{req.specifier} but {have} is installed — the declared "
        "bound excludes the environment the results were produced in")


@pytest.mark.parametrize("name", sorted(MUST_BE_CAPPED))
def test_result_defining_dependencies_have_an_upper_bound(name):
    req = next((r for r in _declared() if r.name == name), None)
    assert req is not None, f"{name} is result-defining but is not declared at all"
    assert any(s.operator in ("<", "<=", "==", "~=") for s in req.specifier), (
        f"{name} has no upper bound: a fresh resolve may take the next major, and for these "
        "libraries that changes numbers silently rather than raising")
