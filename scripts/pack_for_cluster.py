"""Build the code tarball for Sherlock, stamped with the commit it came from.

The cluster runs an extracted tarball, not a checkout, so every manifest written there recorded
``git_sha: null``. This packs ``src/``, ``scripts/``, ``sherlock/`` and ``pyproject.toml`` (plus any
``--weights``) under ``YOLOv8-BeyondEarth/``, the layout ``sherlock/README.md``'s ``tar czf`` makes,
and adds ``CODE_STAMP.json`` at that root: HEAD, dirty state, the uncommitted paths, and the content
digests of the package and of ``scripts/``. ``presets.code_identity()`` falls back to the stamp when
git is unusable and trusts the stamped commit only while the package's own digest still matches it.

    conda run -n bouldernet python scripts/pack_for_cluster.py --out ../sherlock_upload/bnet_code.tgz \
        --weights data/weights/yolov8-m-boulder-detection-tmp.pt

A dirty tree is refused unless ``--allow-dirty``; the stamp then records ``git_dirty: true`` and the
paths, so the result is at least honest about it. Extract on the cluster into a **clean** directory
(sherlock-inference-env gotcha 5: ``tar x`` over an old tree leaves deleted modules importable).
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from YOLOv8BeyondEarth import presets  # noqa: E402

TREES = ("src", "scripts", "sherlock")
FILES = ("pyproject.toml",)
SKIP = ("__pycache__", ".pytest_cache", ".ipynb_checkpoints")


def _git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout.strip()


def make_stamp():
    dirty = [line[3:] for line in _git("status", "--porcelain").splitlines() if line.strip()]
    return {"git_sha": _git("rev-parse", "HEAD"), "git_describe": _git("log", "-1", "--format=%h %s"),
            "git_dirty": bool(dirty), "dirty_paths": dirty,
            "source_sha256": presets.source_digest(),
            "scripts_sha256": presets.source_digest(REPO / "scripts"),
            "packed_utc": datetime.now(timezone.utc).isoformat()}


def _keep(info):
    return None if any(part in SKIP for part in Path(info.name).parts) or info.name.endswith(".pyc") \
        else info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", nargs="*", default=[], help="repo-relative weight files to include")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args(argv)

    stamp = make_stamp()
    if stamp["git_dirty"] and not args.allow_dirty:
        raise SystemExit(f"refusing to pack a dirty tree ({len(stamp['dirty_paths'])} paths, e.g. "
                         f"{stamp['dirty_paths'][:3]}); commit first or pass --allow-dirty")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    root = "YOLOv8-BeyondEarth"
    with tarfile.open(out, "w:gz") as tar:
        for name in TREES + FILES:
            tar.add(REPO / name, arcname=f"{root}/{name}", filter=_keep)
        for w in args.weights:
            tar.add(REPO / w, arcname=f"{root}/{Path(w).as_posix()}")
        blob = json.dumps(stamp, indent=2).encode()
        info = tarfile.TarInfo(f"{root}/{presets.CODE_STAMP}")
        info.size = len(blob)
        tar.addfile(info, io.BytesIO(blob))
    print(f"{out} ({out.stat().st_size / 1e6:.1f} MB) stamped {stamp['git_describe']}"
          f"{' DIRTY' if stamp['git_dirty'] else ''}; source {stamp['source_sha256'][:12]}")
    print("on the cluster:  mkdir -p ~/BoulderNet/releases/<tag> && tar xzf <tgz> -C "
          "~/BoulderNet/releases/<tag>   (a clean directory, never over an old tree)")


if __name__ == "__main__":
    main()
