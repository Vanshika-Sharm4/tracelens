"""Measure how much slower a workload runs while it is being profiled.

TraceLens records traces with ``torch.profiler``, so it inherits that profiler's cost. This module measures it instead of
assuming it. Two numbers are reported, because they answer different questions:

``per_profiled_step_pct``  slowdown of a step that is being profiled, including starting and stopping the profiler
``amortized_pct``          the same cost spread over a run where only 1 step in ``sample_every`` is profiled

Short, op-heavy workloads (like small models on a CPU) show a large per-step overhead because the profiler records every
operator. Sampling a window of steps is how profiling stays cheap in a long-running job. Export of the trace file happens
after the measured region and is not counted.
"""

from __future__ import annotations

import statistics
import time
from typing import Callable

from torch.profiler import ProfilerActivity, profile

import torch

from tracelens.profiler import _quiet_stderr, resolve_device, synchronize


def _time_once(fn: Callable[[], object], device: str) -> float:
    synchronize(device)
    start = time.perf_counter()
    fn()
    synchronize(device)
    return time.perf_counter() - start


def measure_overhead(
    fn: Callable[[], object],
    device: str = "auto",
    warmup: int = 5,
    iterations: int = 20,
    sample_every: int = 20,
    record_shapes: bool = True,
    profile_memory: bool = True,
) -> dict[str, float]:
    device = resolve_device(device)
    for _ in range(warmup):
        fn()

    activities = [ProfilerActivity.CPU]
    if device.startswith("cuda"):
        activities.append(ProfilerActivity.CUDA)

    baseline, profiled = [], []
    # Interleave baseline and profiled runs so drift in machine load affects both equally.
    for _ in range(iterations):
        baseline.append(_time_once(fn, device))
        with _quiet_stderr():
            synchronize(device)
            start = time.perf_counter()
            with profile(activities=activities, record_shapes=record_shapes, profile_memory=profile_memory):
                fn()
                synchronize(device)
            profiled.append(time.perf_counter() - start)

    base = statistics.median(baseline)
    prof = statistics.median(profiled)
    per_step = 100.0 * (prof - base) / base
    return {
        "baseline_ms": base * 1000,
        "profiled_ms": prof * 1000,
        "per_profiled_step_pct": per_step,
        "sample_every": sample_every,
        "amortized_pct": per_step / sample_every,
        "iterations": iterations,
        "device": device,
        "torch": torch.__version__,
    }
