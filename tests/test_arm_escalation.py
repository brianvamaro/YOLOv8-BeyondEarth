"""Cap escalation (maintainer decision 2026-10-15): a binding arm is re-run, not failed.

A truncated arm is a lower bound, not a catalog. ``run_multiscale_arms.py --on-max-det escalate``
re-runs an arm whose slice comes within 10% of the cap at twice the cap and half the batch, until no
slice does, so a dense scene yields the uncapped catalog. These pin the ladder, the stop it escalates
on, and the resume rule that keeps a resubmitted job from paying again for passes that already bound.
The child process is faked: the runner's decisions are what is under test, not inference.
"""
import importlib.util
import json
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from YOLOv8BeyondEarth import predict as P
from YOLOv8BeyondEarth import presets
from YOLOv8BeyondEarth.runstate import COMPLETE, FAILED, StageLedger, stage_key


def _runner():
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_multiscale_arms.py"
    spec = importlib.util.spec_from_file_location("run_multiscale_arms", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_multiscale_arms"] = mod
    spec.loader.exec_module(mod)
    return mod


R = _runner()


def _results(*counts):
    return [SimpleNamespace(boxes=SimpleNamespace(data=torch.zeros((n, 6)))) for n in counts]


# --- the stop it escalates on ---------------------------------------------------------------------

def test_error_near_stops_within_ten_percent_of_the_cap():
    """Near the cap is where a truncated slice lands after ultralytics drops its empty masks, so a run
    that only stopped *at* the cap could complete with a truncated slice in it."""
    with pytest.raises(P.MaxDetBound) as stop:
        P._note_slice_truncation(_results(5, 95), max_det=100, truncation={},
                                 on_max_det="error_near", first_slice=16, new_timeouts=0)
    assert stop.value.as_dict() == {"max_det": 100, "threshold": 90, "worst": 95, "n_hit": 1,
                                    "n_slices": 2, "first_slice": 16, "mode": "error_near"}


def test_plain_error_is_unchanged_it_stops_only_at_the_cap():
    truncation = {}
    P._note_slice_truncation(_results(95), max_det=100, truncation=truncation, on_max_det="error",
                             first_slice=0, new_timeouts=0)
    assert truncation == {"slices_near_max_det": 1}
    with pytest.raises(P.MaxDetBound, match="returned max_det=100"):
        P._note_slice_truncation(_results(100), max_det=100, truncation={}, on_max_det="error",
                                 first_slice=0, new_timeouts=0)


def test_the_stop_is_still_a_runtime_error():
    """Callers that caught the plain RuntimeError before must keep working."""
    assert issubclass(P.MaxDetBound, RuntimeError)


# --- the ladder -----------------------------------------------------------------------------------

def test_the_cap_doubles_and_the_batch_halves_together_up_to_the_model_maximum():
    ceiling = R.model_max_boxes(1024)
    assert ceiling == 128 ** 2 + 64 ** 2 + 32 ** 2 == 21504
    steps, cap, batch = [], 1100, 8
    while (nxt := R.next_attempt(cap, batch, "max_det", ceiling)) is not None:
        cap, batch = nxt
        steps.append(nxt)
    assert steps == [(2200, 4), (4400, 2), (8800, 1), (17600, 1), (21504, 1)]


def test_out_of_memory_halves_the_batch_at_the_same_cap_down_to_one():
    assert R.next_attempt(2200, 4, "cuda_oom", 21504) == (2200, 2)
    assert R.next_attempt(2200, 1, "cuda_oom", 21504) is None


def test_the_preset_batch_keeps_the_old_stage_key():
    """Every ledger written before escalation existed ran at the preset's batch; its entries must
    still satisfy an unescalated arm, or every banked arm would recompute."""
    base = dict(raster="r", weights="w", slice_size=512, sr_factor=1, conf=0.1, imgsz=1024,
                preset="goal1_production", preset_version=presets.PRESET_VERSION)
    old = stage_key(**base, max_det=1100)
    assert R.attempt_key(base, 1100, presets.GOAL1_PRODUCTION["batch_size"]) == old
    assert R.attempt_key(base, 1100, 4) != old


# --- the runner -----------------------------------------------------------------------------------

class FakeChild:
    """Stands in for one arm's process: stops on the cap below ``needs`` (or on memory above
    ``oom_above_batch``), and otherwise writes the layer and reports success."""

    def __init__(self, needs=2200, oom_above_batch=None):
        self.needs, self.oom_above_batch, self.calls = needs, oom_above_batch, []

    def __call__(self, cmd):
        val = lambda flag: cmd[cmd.index(flag) + 1]                         # noqa: E731
        cap, batch, mode = int(val("--max-det")), int(val("--batch-size")), val("--on-max-det")
        self.calls.append((cap, batch, mode))
        if self.oom_above_batch is not None and batch > self.oom_above_batch:
            return R.EXIT_CUDA_OOM, {"stopped": "cuda_oom"}
        if cap < self.needs:
            return R.EXIT_MAX_DET, {"stopped": "max_det", "bound": {"worst": int(0.95 * cap)}}
        layer = R.arm_layer_path(pathlib.Path(val("--out-dir")), val("--raster"),
                                 *R.parse_arm(val("--arm")), 0.10, 1024, 0.2)
        layer.write_bytes(b"layer")
        return 0, {"n": 7, "truncation": {}}


def _args(tmp_path, **overrides):
    args = dict(arms=["512"], arm=None, raster=str(tmp_path / "s.tif"), weights="w.pt",
                out_dir=str(tmp_path / "out"), conf=0.10, inference_size=1024, device="cpu",
                on_conflict="error", on_max_det="escalate", max_det=1100, batch_size=None,
                max_det_ceiling=None, no_fuse=True)
    args.update(overrides)
    return SimpleNamespace(**args)


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(R.presets, "file_identity", lambda p: {"sha256": "x"})

    def go(child, **overrides):
        monkeypatch.setattr(R, "_run_child", child)
        args = _args(tmp_path, **overrides)
        try:
            R.orchestrate(args)
        finally:
            go.ledger = StageLedger(pathlib.Path(args.out_dir) / "run_ledger.jsonl")
        return go.ledger
    return go


def _finished(ledger, arm="512"):
    return [r for r in ledger.entries(f"arm:{arm}") if r["state"] != "running"]


def test_a_binding_arm_is_rerun_at_twice_the_cap_and_half_the_batch(run):
    child = FakeChild(needs=2200)
    ledger = run(child)
    assert child.calls == [(1100, 8, "error_near"), (2200, 4, "error_near")]
    first, second = _finished(ledger)
    assert (first["state"], first["stopped"]) == (FAILED, "max_det")
    assert first["next_attempt"] == {"max_det": 2200, "batch_size": 4}
    assert (second["state"], second["max_det"], second["batch_size"]) == (COMPLETE, 2200, 4)


def test_a_resubmitted_job_does_not_redo_a_finished_escalated_arm(run):
    run(FakeChild(needs=2200))
    again = FakeChild(needs=2200)
    run(again)
    assert again.calls == [], "the complete 2200 attempt must be recognised, not re-bound at 1100"


def test_a_job_killed_mid_escalation_resumes_at_the_attempt_it_was_on(run, tmp_path):
    """A kill leaves only a `running` marker; the bound attempt before it says where to go."""
    args = _args(tmp_path)
    base_key = stage_key(**R.arm_base_parts(args, {"sha256": "x"}, {"sha256": "x"}, 512, 1))
    ledger = StageLedger(pathlib.Path(args.out_dir) / "run_ledger.jsonl")
    ledger.append({"stage": "arm:512", "key": "k1", "state": "running"})
    ledger.append({"stage": "arm:512", "key": "k1", "state": FAILED, "base_key": base_key,
                   "max_det": 1100, "batch_size": 8, "stopped": "max_det",
                   "next_attempt": {"max_det": 2200, "batch_size": 4}})
    ledger.append({"stage": "arm:512", "key": "k2", "state": "running"})        # killed here
    child = FakeChild(needs=2200)
    run(child)
    assert child.calls == [(2200, 4, "error_near")]


def test_out_of_memory_halves_the_batch_and_keeps_the_cap(run):
    child = FakeChild(needs=1100, oom_above_batch=4)
    run(child)
    assert child.calls == [(1100, 8, "error_near"), (1100, 4, "error_near")]


def test_out_of_memory_at_batch_one_stops_and_says_why(run):
    child = FakeChild(needs=1100, oom_above_batch=0)
    with pytest.raises(SystemExit, match="batch 1"):
        run(child, batch_size=1)
    (only,) = _finished(run.ledger)
    assert only["escalation_exhausted"] == "cuda_oom"


def test_escalation_stops_at_the_ceiling(run):
    child = FakeChild(needs=10 ** 9)
    with pytest.raises(SystemExit, match="ceiling"):
        run(child, max_det_ceiling=4400)
    assert [c[0] for c in child.calls] == [1100, 2200, 4400]


def test_without_escalation_a_binding_arm_still_fails_and_records_why(run):
    child = FakeChild(needs=2200)
    with pytest.raises(subprocess.CalledProcessError):
        run(child, on_max_det="error")
    assert child.calls == [(1100, 8, "error")]
    (only,) = _finished(run.ledger)
    assert (only["state"], only["stopped"]) == (FAILED, "max_det")


def test_the_child_summary_line_is_found_among_log_lines(monkeypatch):
    out = "loading model\n{\"arm\": \"512\", \"stopped\": \"max_det\"}\n"
    monkeypatch.setattr(R.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(returncode=3, stdout=out))
    assert R._run_child(["x"]) == (3, {"arm": "512", "stopped": "max_det"})
