"""Build synthetic Chrome traces with known, injected bottlenecks.

They give the detectors ground truth: we know exactly what pattern was planted, so a test can assert it is flagged (and a
healthy trace is not). They also let the GPU-only detectors run in CI, where no GPU is available.
"""

from __future__ import annotations

from typing import Any


class SyntheticTrace:
    def __init__(self, device_name: str = "Tesla T4"):
        self.device_name = device_name
        self._events: list[dict[str, Any]] = []
        self._memory: list[dict[str, Any]] = []
        self._ext = 0

    # ----------------------------------------------------------------- primitives
    def cpu_op(self, name: str, ts: float, dur: float, tid: int = 1, dims=None, types=None, concrete=None) -> int:
        """Add a host operator and return its External id (pass it to :meth:`kernel` to link a launch)."""
        self._ext += 1
        args: dict[str, Any] = {"External id": self._ext}
        if dims is not None:
            args["Input Dims"] = dims
            args["Input type"] = types or ["float"] * len(dims)
            args["Concrete Inputs"] = concrete or [""] * len(dims)
        self._events.append({"ph": "X", "cat": "cpu_op", "name": name, "pid": 100, "tid": tid, "ts": ts, "dur": dur, "args": args})
        return self._ext

    def kernel(self, name: str, ts: float, dur: float, ext: int | None = None, stream: int = 7, device: int = 0) -> None:
        args: dict[str, Any] = {"stream": stream, "device": device}
        if ext is not None:
            args["External id"] = ext
        self._events.append({"ph": "X", "cat": "kernel", "name": name, "pid": device, "tid": stream, "ts": ts, "dur": dur, "args": args})

    def runtime(self, name: str, ts: float, dur: float, tid: int = 1) -> None:
        self._events.append({"ph": "X", "cat": "cuda_runtime", "name": name, "pid": 100, "tid": tid, "ts": ts, "dur": dur, "args": {}})

    def memory(self, ts: float, nbytes: int, total: int, device_type: int = 1) -> None:
        self._memory.append(
            {"ph": "i", "name": "[memory]", "ts": ts, "pid": 100, "tid": 1, "s": "t",
             "args": {"Bytes": nbytes, "Total Allocated": total, "Device Type": device_type, "Device Id": 0}}
        )

    def to_chrome(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "deviceProperties": [{"id": 0, "name": self.device_name}],
            "traceEvents": self._events + self._memory,
        }


# ------------------------------------------------------------------------------ scenarios
def healthy(n: int = 40) -> dict[str, Any]:
    """Large kernels back to back; host runs ahead. Nothing should be flagged except possibly a hotspot."""
    t = SyntheticTrace()
    for i in range(n):
        ext = t.cpu_op("aten::mm", i * 20.0, 15.0, dims=[[512, 512], [512, 512]])
        t.kernel("gemm_kernel", 5.0 + i * 500.0, 500.0, ext)
    return t.to_chrome()


def launch_bound(n: int = 200) -> dict[str, Any]:
    """Each 5 us kernel waits ~25 us for the next launch: the classic launch-overhead signature."""
    t = SyntheticTrace()
    for i in range(n):
        ext = t.cpu_op("aten::add", i * 30.0, 8.0, dims=[[64, 64], [64, 64]])
        t.kernel("elementwise_add", i * 30.0 + 10.0, 5.0, ext)
    return t.to_chrome()


def sync_gap() -> dict[str, Any]:
    """20 busy kernels, a 3 ms host synchronization during which the GPU idles, then 20 more kernels."""
    t = SyntheticTrace()
    ts = 0.0
    for i in range(20):
        ext = t.cpu_op("aten::mm", ts, 10.0, dims=[[256, 256], [256, 256]])
        t.kernel("gemm_kernel", ts + 5.0, 200.0, ext)
        ts += 200.0
    sync_start = ts
    item = t.cpu_op("aten::item", sync_start, 3000.0)
    t.runtime("cudaStreamSynchronize", sync_start + 10.0, 2990.0)
    ts = sync_start + 3000.0
    for i in range(20):
        ext = t.cpu_op("aten::mm", ts, 10.0, dims=[[256, 256], [256, 256]])
        t.kernel("gemm_kernel", ts + 5.0, 200.0, ext)
        ts += 200.0
    return t.to_chrome()


def memory_bound_mix() -> dict[str, Any]:
    """Big elementwise/softmax kernels dominate GPU time while a modest matmul takes the rest."""
    t = SyntheticTrace()
    ts = 0.0
    for i in range(10):
        mm = t.cpu_op("aten::mm", ts, 10.0, dims=[[4096, 4096], [4096, 4096]])
        t.kernel("gemm_kernel", ts + 5.0, 300.0, mm)
        ts += 320.0
        sm = t.cpu_op("aten::softmax", ts, 10.0, dims=[[8192, 8192]])
        t.kernel("softmax_kernel", ts + 5.0, 900.0, sm)
        ts += 920.0
        ad = t.cpu_op("aten::add", ts, 10.0, dims=[[8192, 8192], [8192, 8192]])
        t.kernel("elementwise_add", ts + 5.0, 700.0, ad)
        ts += 720.0
    return t.to_chrome()


def with_memory() -> dict[str, Any]:
    t = SyntheticTrace()
    t.cpu_op("aten::empty", 0.0, 50.0)
    t.cpu_op("aten::mm", 100.0, 200.0, dims=[[64, 64], [64, 64]])
    t.memory(10.0, 1_000_000, 1_000_000)
    t.memory(150.0, 9_000_000, 10_000_000)
    t.memory(400.0, -9_000_000, 1_000_000)
    return t.to_chrome()
