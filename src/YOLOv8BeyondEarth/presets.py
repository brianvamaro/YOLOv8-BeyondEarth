"""Named production configurations and the provenance record that travels with every result.

Why this module exists (review items G1-01, G1-07, R-02)
--------------------------------------------------------
The public defaults of :func:`predict.get_sliced_prediction` were never the configuration any
published number was produced with. Defaults said ``downscale_pred=False``, no area gate, NMS 0.5,
Shapefile output and every intermediate layer; the declared Goal-1 production benchmark used
``downscale_pred=True``, ``min_area_threshold=6``, NMS 0.2, GeoPackage and the final layer only.
So an ordinary API call reproduced neither the geometry, the count, the cost, nor the output
safety of the benchmark, and nothing on disk recorded which of the two you had.

Presets fix the first half of that: a named, versioned configuration you can ask for by name.
:func:`write_manifest` fixes the second: the *resolved* configuration is serialized beside every
result together with code, model and input identity, so a file can always answer "what made me".
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

#: Bumped when a preset's *values* change, so two results from the same preset name are only
#: comparable if they also share this number.
PRESET_VERSION = 1

#: Files larger than this are fingerprinted by (size, mtime) rather than hashed in full.
MAX_HASH_BYTES = 4 * 1024 ** 3


#: Single-scale sliced inference as Goal 1 actually benchmarked it. This is the configuration
#: behind the Goal-1 throughput tables and the Goal-2 catalogs, *not* the bare function defaults.
GOAL1_PRODUCTION = {
    "confidence_threshold": 0.10,
    "slice_size": 512,
    "inference_size": 1024,
    "overlap_height_ratio": 0.2,
    "overlap_width_ratio": 0.2,
    "min_area_threshold": 6,
    "downscale_pred": True,
    "postprocess": True,
    "postprocess_match_threshold": 0.2,
    "postprocess_class_agnostic": False,
    "batch_size": 8,
    "contour_method": "skimage",
    "save_bbox": False,
    "save_prenms": False,
    "output_format": "gpkg",
    "sr_factor": 1,
}

#: The Goal-3 phase-3 ablation winner, ``256pref_engulf``: the {512, 256, SR x2} ladder fused with
#: 256 masks winning contested clusters and fragments consolidated into their containers. The
#: per-scale arms inherit everything in :data:`GOAL1_PRODUCTION` except the slicing keys, which the
#: ladder owns.
GOAL3_PRODUCTION = {
    "scales": (512, 256, (256, 2)),
    "rule": "bands",
    "bands": {256: (0.0, float("inf"))},
    "engulf_thr": 0.7,
    "metric": "diou",
    "cluster_thr": None,          # None -> fusion.DEFAULT_CLUSTER_THR
    "cross_arm_only": True,
}

PRESETS = {"goal1_production": GOAL1_PRODUCTION, "goal3_production": GOAL3_PRODUCTION}


def resolve(preset, **overrides):
    """Return a preset's parameters with ``overrides`` applied, plus the identity of what was used.

    Unknown override keys are rejected: a typo would otherwise be silently ignored and the run
    would quietly use the preset value instead.
    """
    if isinstance(preset, str):
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}; known: {sorted(PRESETS)}")
        name, base = preset, PRESETS[preset]
    elif preset is None:
        name, base = None, {}
    else:
        name, base = getattr(preset, "__name__", "<dict>"), dict(preset)

    unknown = set(overrides) - set(base)
    if base and unknown:
        raise ValueError(f"override keys not in preset {name!r}: {sorted(unknown)}")
    resolved = {**base, **overrides}
    resolved["_preset"] = name
    resolved["_preset_version"] = PRESET_VERSION
    return resolved


# --- identity ------------------------------------------------------------------------------------

def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (str, int, bool, type(None))):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if callable(value):
        return f"<callable {getattr(value, '__name__', type(value).__name__)}>"
    return repr(value)


def config_hash(config, length=8):
    """Stable short hash of a resolved configuration — two differing configs cannot collide."""
    payload = json.dumps(_jsonable(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:length]


def file_identity(path):
    """``{path, bytes, mtime, sha256}`` for an input, hashing in full unless it is very large."""
    path = Path(path)
    if not path.exists():
        return {"path": path.as_posix(), "exists": False}
    stat = path.stat()
    out = {"path": path.as_posix(), "bytes": stat.st_size,
           "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()}
    if stat.st_size <= MAX_HASH_BYTES:
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        out["sha256"] = digest.hexdigest()
    else:
        out["sha256"] = None
        out["sha256_skipped"] = f"file larger than {MAX_HASH_BYTES} bytes"
    return out


def code_identity():
    """Git SHA and dirty state of the package, plus the interpreter and key package versions."""
    repo = Path(__file__).resolve().parents[2]
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "host": socket.gethostname()}
    try:
        info["git_sha"] = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        info["git_dirty"] = bool(subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10, check=True).stdout.strip())
    except Exception as exc:                                    # not a checkout, no git, ...
        info["git_sha"] = None
        info["git_error"] = str(exc)
    for pkg in ("ultralytics", "sahi", "torch", "geopandas", "shapely", "rasterio", "opencv-python"):
        try:
            from importlib.metadata import version
            info[pkg] = version(pkg)
        except Exception:
            pass
    return info


# --- the manifest --------------------------------------------------------------------------------

def manifest_path(out_path):
    out_path = Path(out_path)
    return out_path.with_name(out_path.stem + "-manifest.json")


def check_config_conflict(out_path, config, on_conflict="error"):
    """Refuse to overwrite a result that a *different* configuration produced (G1-07).

    Result filenames carry only a few of the parameters that define them — not the contour method,
    not the NMS threshold, not the area gate, not the model. Two different runs therefore resolve
    to the same name, and the second silently replaces the first with something that is not what
    the name says. The manifest makes the collision detectable; this makes it fatal.
    """
    if on_conflict not in ("error", "overwrite"):
        raise ValueError("on_conflict must be 'error' or 'overwrite'")
    path = manifest_path(out_path)
    if on_conflict == "overwrite" or not path.exists():
        return
    try:
        previous = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return                                                  # unreadable manifest: treat as absent
    old, new = previous.get("config_hash"), config_hash(config)
    if old and old != new:
        raise FileExistsError(
            f"{Path(out_path).name} already exists and was produced by a different configuration "
            f"(hash {old}, this run {new}). Result filenames do not encode every parameter that "
            "defines them, so overwriting would mislabel the file. Pass on_conflict='overwrite' "
            f"to replace it, or choose a different output_dir. Previous config: {path.name}")


def write_manifest(out_path, *, config, inputs=None, outputs=None, **extra):
    """Serialize the resolved config and full provenance beside a result. Written atomically."""
    path = manifest_path(out_path)
    record = {
        "written_utc": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash(config),
        "config": _jsonable(config),
        "code": code_identity(),
        "inputs": {k: file_identity(v) for k, v in (inputs or {}).items()},
        "outputs": _jsonable(outputs or {}),
        **{k: _jsonable(v) for k, v in extra.items()},
    }
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(record, indent=2))
    os.replace(tmp, path)
    return path
