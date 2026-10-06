"""Record a trace of a PyTorch callable with ``torch.profiler`` and parse it into a :class:`~tracelens.trace.Trace`."""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from typing import Callable, Iterator

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from tracelens.trace import Trace, parse_chrome_trace


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is available")
    return device


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


@contextmanager
def _quiet_stderr() -> Iterator[None]:
    """Kineto prints noisy status lines to stderr when it starts and stops; hide them."""
    fd = os.dup(2)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(fd, 2)
        os.close(fd)


def run_profile(
    fn: Callable[[], object],
    iterations: int = 1,
    warmup: int = 3,
    device: str = "auto",
    record_shapes: bool = True,
    profile_memory: bool = True,
):
    """Profile ``fn``. Returns ``(torch_profiler, chrome_trace_path)``; the caller removes the file."""
    device = resolve_device(device)
    for _ in range(warmup):  # first calls pay one-time costs (allocator, kernel loading); keep them out of the trace
        fn()
    synchronize(device)

    activities = [ProfilerActivity.CPU]
    if device.startswith("cuda"):
        activities.append(ProfilerActivity.CUDA)

    fd, path = tempfile.mkstemp(suffix=".json", prefix="tracelens_")
    os.close(fd)
    with _quiet_stderr():
        with profile(activities=activities, record_shapes=record_shapes, profile_memory=profile_memory) as prof:
            for i in range(iterations):
                with record_function(f"step_{i}"):
                    fn()
                    synchronize(device)
        prof.export_chrome_trace(path)
    return prof, path


def profile_callable(
    fn: Callable[[], object],
    iterations: int = 1,
    warmup: int = 3,
    device: str = "auto",
    record_shapes: bool = True,
    profile_memory: bool = True,
) -> Trace:
    """Profile ``fn`` for ``iterations`` calls and return the parsed trace."""
    device = resolve_device(device)
    _, path = run_profile(fn, iterations, warmup, device, record_shapes, profile_memory)
    try:
        trace = parse_chrome_trace(path)
    finally:
        os.remove(path)
    if device.startswith("cuda") and not trace.meta.get("device_name"):
        trace.meta["device_name"] = torch.cuda.get_device_name(0)
    trace.meta["torch_version"] = torch.__version__
    trace.meta["device"] = device
    trace.meta["iterations"] = iterations
    return trace
