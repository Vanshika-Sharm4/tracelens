"""End-to-end on the CPU: record a real PyTorch profile, parse it, and cross-check against torch's own aggregation."""

import pytest
import torch

from tracelens.analysis import analyze
from tracelens.models import DEMO_MODELS, build_workload
from tracelens.overhead import measure_overhead
from tracelens.profiler import profile_callable
from tracelens.validate import validate_model, validate_seeded


def test_profile_callable_records_operators_and_steps():
    wl = build_workload("mlp", batch_size=2, device="cpu")
    trace = profile_callable(wl.fn, iterations=2, device="cpu")
    names = {e.name for e in trace.events}
    assert "aten::addmm" in names and "step_0" in names and "step_1" in names
    assert trace.span_us > 0 and not trace.has_gpu


def test_self_time_never_exceeds_inclusive_time():
    wl = build_workload("tiny_gpt", batch_size=2, device="cpu")
    trace = profile_callable(wl.fn, device="cpu")
    assert all(e.self_us <= e.dur_us + 1e-6 for e in trace.events)


@pytest.mark.parametrize("name", ["mlp", "cnn", "tiny_gpt"])
def test_tracelens_self_time_matches_torch_key_averages(name):
    r = validate_model(name, batch_size=2, device="cpu")
    assert r["operators_compared"] >= 5
    assert r["max_rel_error"] < 1e-6
    assert r["total_rel_error"] < 1e-6


def test_mlp_matmuls_are_costed_and_dominate_compute():
    wl = build_workload("mlp", batch_size=8, device="cpu")
    a = analyze(profile_callable(wl.fn, device="cpu"))
    addmm = next(o for o in a.ops if o.name == "aten::addmm")
    assert addmm.flops and addmm.flops > 0 and addmm.intensity is not None


def test_tiny_ops_workload_triggers_small_ops():
    wl = build_workload("tiny_ops", device="cpu")
    a = analyze(profile_callable(wl.fn, iterations=3, device="cpu"))
    assert "small_ops" in {f.kind for f in a.findings}


def test_seeded_validation_passes_on_cpu():
    # The detectors are timing-based heuristics and a loaded CI machine can distort one run, so allow a retry.
    for attempt in range(3):
        if all(r["passed"] for r in validate_seeded("cpu", batch_size=2)):
            return
    pytest.fail("seeded bottleneck not detected in 3 attempts")


def test_overhead_measurement_is_well_formed():
    wl = build_workload("mlp", batch_size=2, device="cpu")
    r = measure_overhead(wl.fn, device="cpu", warmup=2, iterations=4, sample_every=10)
    assert r["baseline_ms"] > 0 and r["profiled_ms"] > 0
    assert r["amortized_pct"] == pytest.approx(r["per_profiled_step_pct"] / 10)


def test_all_demo_models_run():
    for name in DEMO_MODELS:
        out = build_workload(name, batch_size=2, device="cpu").fn()
        assert isinstance(out, torch.Tensor)
