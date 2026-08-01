"""Stage ledger: a run that can be resumed without lying to itself (review item G3-08).

The Goal-3 runners treated the *existence* of an arm's GeoPackage as proof the arm finished. It is
not: a run killed between writing the layer and writing its timing JSON leaves an artifact that
looks complete and a record that is missing, which is how AP ended up with 280 done-markers and
223 timing entries. Metadata writes were not atomic either, and the cache key omitted the code, the
model and the source, so changing any of them silently reused the old answer.

A stage here is complete only when its artifact **and** its ledger entry both exist, the entry's
key matches the current inputs, and the artifact still checksums to what was recorded. Anything
else is ``incomplete`` or ``stale`` and recomputes. Entries are appended, so a recompute never
erases the previous run's timings.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from . import presets

#: Terminal and non-terminal stage states. Only COMPLETE means "do not recompute".
COMPLETE, FAILED, INCOMPLETE, STALE, MISSING = "complete", "failed", "incomplete", "stale", "missing"


def file_digest(path, chunk=1 << 20):
    """sha256 of a file, or ``None`` if it is not there."""
    path = Path(path)
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


class StageLedger:
    """Append-only JSONL record of stage attempts, with dependency-aware invalidation.

    ``ledger.status(name, key)`` answers whether a stage needs to run;
    ``with ledger.stage(name, key, artifacts=[...]) as s:`` records the attempt either way.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # --- reading -------------------------------------------------------------------------------
    def entries(self, name=None):
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue                      # a torn final line from a kill mid-write
            if name is None or rec.get("stage") == name:
                out.append(rec)
        return out

    def latest(self, name):
        entries = self.entries(name)
        return entries[-1] if entries else None

    def status(self, name, key, depends_on=()):
        """Why this stage does or does not need to run.

        Returns ``(state, detail)``. ``depends_on`` are upstream stage keys; if any has changed
        since this stage last ran, the stage is ``stale`` — the dependency-aware invalidation the
        review asked for, so re-running one arm re-fuses rather than silently reusing the old fuse.
        """
        rec = self.latest(name)
        if rec is None:
            return MISSING, "no ledger entry"
        if rec.get("state") == FAILED:
            return FAILED, rec.get("error", "previous attempt failed")
        if rec.get("state") != COMPLETE:
            return INCOMPLETE, f"previous attempt state={rec.get('state')!r}"
        if rec.get("key") != key:
            return STALE, "configuration or inputs changed"
        if dict(rec.get("depends_on") or {}) != dict(depends_on or {}):
            return STALE, "an upstream stage changed"
        for path, digest in (rec.get("artifacts") or {}).items():
            if not Path(path).exists():
                return INCOMPLETE, f"artifact missing: {Path(path).name}"
            if digest is not None and file_digest(path) != digest:
                return STALE, f"artifact changed on disk: {Path(path).name}"
        return COMPLETE, "up to date"

    def is_complete(self, name, key, depends_on=()):
        return self.status(name, key, depends_on)[0] == COMPLETE

    # --- writing -------------------------------------------------------------------------------
    def append(self, record):
        """Append one entry, flushed and fsynced, so a kill cannot leave it half-written."""
        line = json.dumps(record, default=str) + "\n"
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())

    def stage(self, name, key, artifacts=(), depends_on=(), **extra):
        return _Stage(self, name, key, list(artifacts), dict(depends_on or {}), extra)


class _Stage:
    """Context manager recording one stage attempt — success, failure, or a crash in between."""

    def __init__(self, ledger, name, key, artifacts, depends_on, extra):
        self.ledger, self.name, self.key = ledger, name, key
        self.artifacts, self.depends_on, self.extra = artifacts, depends_on, extra
        self.result = {}

    def __enter__(self):
        self._t0 = time.perf_counter()
        # a "running" marker first: if the process dies here, the next run sees `incomplete`
        # rather than a missing entry it might mistake for "never attempted"
        self.ledger.append({"stage": self.name, "key": self.key, "state": "running",
                            "started_utc": datetime.now(timezone.utc).isoformat()})
        return self

    def __exit__(self, exc_type, exc, tb):
        record = {"stage": self.name, "key": self.key,
                  "finished_utc": datetime.now(timezone.utc).isoformat(),
                  "wall_seconds": round(time.perf_counter() - self._t0, 3),
                  "depends_on": self.depends_on, **self.extra, **self.result}
        if exc_type is not None:
            record["state"] = FAILED
            record["error"] = f"{exc_type.__name__}: {exc}"
            self.ledger.append(record)
            return False                      # never swallow the exception
        missing = [str(p) for p in self.artifacts if not Path(p).exists()]
        if missing:
            record["state"] = INCOMPLETE
            record["error"] = f"declared artifacts were not produced: {missing}"
            self.ledger.append(record)
            raise FileNotFoundError(record["error"])
        record["state"] = COMPLETE
        record["artifacts"] = {str(p): file_digest(p) for p in self.artifacts}
        self.ledger.append(record)
        return False


def stage_key(**parts):
    """A cache key over *everything* that defines a stage's output.

    Always includes the code SHA and dirty flag: the old keys omitted code, model and source, so a
    changed tracer or a changed checkpoint silently reused the previous answer.
    """
    code = presets.code_identity()
    payload = {"code_sha": code.get("git_sha"), "code_dirty": code.get("git_dirty"), **parts}
    return presets.config_hash(payload, length=16)
