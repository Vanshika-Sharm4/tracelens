"""Check TraceLens against independent references.

``validate_model``   compares per-operator self time computed by TraceLens from the exported Chrome trace with the
                     aggregation PyTorch itself computes (``prof.key_averages()``) for the same profiling run.
``validate_seeded``  profiles workloads with a planted bottleneck and checks that the matching detector fires.

The first checks the parser and the self-time arithmetic. The second checks the detectors on real executions; on a CPU only
a subset of the detectors apply, because GPU timelines do not exist.
"""

from __future__ import annotations

import os
import statistics
from typing import Any

from tracelens.analysis import AnalysisConfig, analyze
from tracelens.models import DEMO_MODELS, build_workload
from tracelens.profiler import resolve_device, run_profile
from tracelens.trace import parse_chrome_trace


def validate_model(name: str, batch_size: int = 8, device: str = "auto", top_k: int = 15) -> dict[str, Any]:
    device = resolve_device(device)
    workload = build_workload(name, batch_size, device)
    prof, path = run_profile(workload.fn, iterations=1, warmup=3, device=device)
    try:
        trace = parse_chrome_trace(path)
    finally:
        os.remove(path)

    ours = {o.name: o.self_us for o in analyze(trace, config=AnalysisConfig(top_ops=10**6)).ops}
    reference = {
        e.key: float(e.self_cpu_time_total)
        for e in prof.key_averages()
        if e.key in ours  # only operators TraceLens treats as host operators
    }
    ranked = sorted(reference.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    errors = [abs(ours[k] - v) / v for k, v in ranked if v > 0]
    total_ours = sum(ours[k] for k in reference)
    total_ref = sum(reference.values())
    return {
        "model": name,
        "device": device,
        "operators_compared": len(ranked),
        "median_rel_error": statistics.median(errors) if errors else 0.0,
        "max_rel_error": max(errors) if errors else 0.0,
        "total_self_ms_tracelens": total_ours / 1000,
        "total_self_ms_torch": total_ref / 1000,
        "total_rel_error": abs(total_ours - total_ref) / total_ref if total_ref else 0.0,
    }


def validate_seeded(device: str = "auto", batch_size: int = 8) -> list[dict[str, Any]]:
    """Run the two bottleneck-seeded workloads and report which findings TraceLens raised."""
    device = resolve_device(device)
    on_gpu = device.startswith("cuda")
    expectations = {
        # workload -> findings of which at least one must appear
        "tiny_ops": {"launch_overhead", "small_ops"},
        "sync_heavy": {"idle_gap"} if on_gpu else set(),
    }
    results = []
    for name, expected in expectations.items():
        workload = build_workload(name, batch_size, device)
        prof, path = run_profile(workload.fn, iterations=3, warmup=3, device=device)
        try:
            trace = parse_chrome_trace(path)
        finally:
            os.remove(path)
        found = {f.kind for f in analyze(trace).findings}
        results.append(
            {
                "model": name,
                "device": device,
                "expected_any_of": sorted(expected),
                "found": sorted(found),
                "passed": (not expected) or bool(expected & found),
                "note": "" if on_gpu or expected else "no GPU timeline on CPU: host-sync detection needs a GPU",
            }
        )
    return results


def validate_all(device: str = "auto", models: list[str] | None = None) -> dict[str, Any]:
    names = models or list(DEMO_MODELS)
    return {
        "reference_check": [validate_model(n, device=device) for n in names],
        "seeded_bottlenecks": validate_seeded(device),
    }
