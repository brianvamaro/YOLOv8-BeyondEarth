"""Tiny disk cache for notebook orchestration.

The HiRISE test notebooks re-run thousands of sliced-inference passes, so a full rebuild is slow
(~15 min). This lets the **expensive results** (sweep tables, residuals, detection angles) be computed
**once** and reloaded on every later rebuild — so you can edit markdown/plots and re-execute in
seconds, only paying for inference when the inputs actually change.

Usage in a notebook::

    from YOLOv8BeyondEarth.nbcache import disk_cache
    CACHE = Path("../YOLOv8-BeyondEarth/data/test4_cache")        # under the gitignored data/
    sweep = disk_cache(CACHE, "readoutB", lambda: expensive(...))  # computes once, then loads
    sweep = disk_cache(CACHE, "readoutB", lambda: expensive(...), force=True)   # force recompute

To recompute just one result, delete its ``<key>.pkl`` (or pass ``force=True``); to recompute
everything, delete the cache dir. Pickled, so it stores DataFrames / numpy arrays / dicts uniformly.
The cache does **not** auto-invalidate on parameter changes — that is the caller's responsibility
(change the ``key`` or force) and is why every cached block keeps its parameters next to its key.
"""
import pickle
from pathlib import Path


def disk_cache(cache_dir, key, fn, force=False, verbose=True):
    """Return ``fn()``, caching the result to ``cache_dir/<key>.pkl``.

    Loads the cached object on subsequent calls unless ``force`` is set or the cache is missing /
    unreadable. ``fn`` is a zero-arg callable so the expensive work is skipped entirely on a hit.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"{key}.pkl"
    if p.exists() and not force:
        try:
            with open(p, "rb") as f:
                obj = pickle.load(f)
            if verbose:
                print(f"[cache] loaded '{key}'  ({p.stat().st_size/1e3:.0f} kB)")
            return obj
        except Exception as e:                         # corrupt / version-skew pickle -> recompute
            if verbose:
                print(f"[cache] '{key}' unreadable ({e}); recomputing")
    obj = fn()
    with open(p, "wb") as f:
        pickle.dump(obj, f)
    if verbose:
        print(f"[cache] computed + saved '{key}'")
    return obj
