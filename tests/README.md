# Tests

Test suite for `YOLOv8BeyondEarth` (the package under [`../src/`](../src/)).

## Running

Conda is **not** on PATH; always go through the absolute `conda run`. Run from inside
`YOLOv8-BeyondEarth/`:

```bash
# Fast (skip slow/GPU/data-heavy tests)
C:\Users\brian\anaconda3\Scripts\conda.exe run -n bouldernet pytest -m "not slow"

# Full suite
C:\Users\brian\anaconda3\Scripts\conda.exe run -n bouldernet pytest
```

## Conventions

- Config lives in [`../pyproject.toml`](../pyproject.toml) under `[tool.pytest.ini_options]`:
  `testpaths = ["tests"]`, `pythonpath = ["src"]` (so `import YOLOv8BeyondEarth` works with no
  editable install), and the registered `slow` marker.
- Mark long-running, GPU-bound, or large-data tests with `@pytest.mark.slow` so the fast lane
  (`-m "not slow"`) stays quick.
- Put shared fixtures (sample polygons, a tiny test raster, known-angle synthetic ellipses) in
  [`conftest.py`](conftest.py).

## Planned

- **Orientation characterization test** pinning the *current* angle output **before** any
  Goal-1 cleanup — see [`../../docs/orientation_investigation.md`](../../docs/orientation_investigation.md).
  This locks behavior so refactors are provably non-breaking, then the rotation-test harness
  documented there becomes the regression test once the bias is fixed.
