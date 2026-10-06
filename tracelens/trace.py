"""Trace data model and a parser for the Chrome trace format.

PyTorch's profiler (Kineto) exports Chrome-trace JSON, and so do many other tools. TraceLens uses that format as its
single input, which means it can analyse a trace it recorded itself *or* any ``trace.json`` you already have.

Only complete events (``ph == "X"``) and ``[memory]`` instant events are read. Everything else is ignored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# Chrome-trace category -> TraceLens category.
CATEGORY_MAP = {
    "cpu_op": "cpu_op",
    "kernel": "kernel",
    "gpu_memcpy": "memcpy",
    "gpu_memset": "memset",
    "cuda_runtime": "runtime",
    "cuda_driver": "runtime",
    "python_function": "python",
    "user_annotation": "annotation",
    "gpu_user_annotation": "gpu_annotation",
}

GPU_CATEGORIES = ("kernel", "memcpy", "memset")
HOST_NESTING_CATEGORIES = ("cpu_op", "annotation")

# Tolerance (in microseconds) when deciding whether one event is contained in another.
_EPS_US = 0.01


@dataclass
class Event:
    """One timed interval on a host thread or a GPU stream. Times are microseconds since the start of the trace."""

    id: int
    name: str
    cat: str
    start_us: float
    dur_us: float
    pid: int
    tid: int
    ext_id: int | None = None  # links a kernel to the CPU op that launched it
    dims: list | None = None  # input tensor shapes (needs record_shapes)
    dtypes: list | None = None
    concrete: list | None = None  # non-tensor arguments, as strings
    parent: int = -1
    depth: int = 0
    self_us: float = 0.0

    @property
    def end_us(self) -> float:
        return self.start_us + self.dur_us

    @property
    def on_gpu(self) -> bool:
        return self.cat in GPU_CATEGORIES


@dataclass
class MemoryEvent:
    ts_us: float
    bytes: int  # positive = allocation, negative = free
    total_allocated: int
    device_type: int  # 0 = CPU, 1 = CUDA
    device_id: int


@dataclass
class Trace:
    events: list[Event]
    memory: list[MemoryEvent] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ views
    def of(self, *cats: str) -> list[Event]:
        return [e for e in self.events if e.cat in cats]

    @property
    def cpu_ops(self) -> list[Event]:
        return self.of("cpu_op")

    @property
    def gpu_events(self) -> list[Event]:
        return self.of(*GPU_CATEGORIES)

    @property
    def kernels(self) -> list[Event]:
        return self.of("kernel")

    @property
    def has_gpu(self) -> bool:
        return bool(self.kernels)

    @property
    def span_us(self) -> float:
        timed = [e for e in self.events if e.cat in ("cpu_op", "kernel", "memcpy", "memset")]
        if not timed:
            return 0.0
        return max(e.end_us for e in timed) - min(e.start_us for e in timed)

    @property
    def start_us(self) -> float:
        timed = [e for e in self.events if e.cat in ("cpu_op", "kernel", "memcpy", "memset")]
        return min((e.start_us for e in timed), default=0.0)

    def by_id(self) -> dict[int, Event]:
        return {e.id: e for e in self.events}


# ---------------------------------------------------------------------------- parsing
def load_json(source: str | Path | dict | list) -> Any:
    if isinstance(source, (dict, list)):
        return source
    with open(source, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_chrome_trace(source: str | Path | dict | list) -> Trace:
    """Parse a Chrome-trace JSON file (or an already loaded dict/list) into a :class:`Trace`."""
    data = load_json(source)
    if isinstance(data, dict):
        raw: Iterable[dict] = data.get("traceEvents", [])
        header = {k: v for k, v in data.items() if k != "traceEvents"}
    elif isinstance(data, list):
        raw, header = data, {}
    else:
        raise ValueError("not a Chrome trace: expected an object with 'traceEvents' or a list of events")

    events: list[Event] = []
    memory: list[MemoryEvent] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        ph = item.get("ph")
        if ph == "X":
            cat = CATEGORY_MAP.get(item.get("cat", ""))
            if cat is None or "ts" not in item:
                continue
            args = item.get("args") or {}
            ext = args.get("External id")
            events.append(
                Event(
                    id=len(events),
                    name=str(item.get("name", "?")),
                    cat=cat,
                    start_us=float(item["ts"]),
                    dur_us=max(float(item.get("dur", 0.0)), 0.0),
                    pid=_as_int(item.get("pid")),
                    tid=_as_int(args.get("stream", item.get("tid"))) if cat in GPU_CATEGORIES else _as_int(item.get("tid")),
                    ext_id=int(ext) if isinstance(ext, (int, float)) else None,
                    dims=args.get("Input Dims"),
                    dtypes=args.get("Input type"),
                    concrete=args.get("Concrete Inputs"),
                )
            )
        elif ph == "i" and item.get("name") == "[memory]" and "ts" in item:
            args = item.get("args") or {}
            memory.append(
                MemoryEvent(
                    ts_us=float(item["ts"]),
                    bytes=int(args.get("Bytes", 0)),
                    total_allocated=int(args.get("Total Allocated", 0)),
                    device_type=int(args.get("Device Type", 0)),
                    device_id=int(args.get("Device Id", -1)),
                )
            )

    if events:
        t0 = min(e.start_us for e in events)
        for e in events:
            e.start_us -= t0
        for m in memory:
            m.ts_us -= t0
    else:
        t0 = 0.0

    _assign_hierarchy(events)

    props = header.get("deviceProperties") or []
    device_name = props[0].get("name") if props and isinstance(props[0], dict) else None
    meta = {
        "device_name": device_name,
        "schema_version": header.get("schemaVersion"),
        "record_shapes": bool(header.get("record_shapes", any(e.dims for e in events))),
        "profile_memory": bool(header.get("profile_memory", bool(memory))),
    }
    return Trace(events=events, memory=sorted(memory, key=lambda m: m.ts_us), meta=meta)


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _assign_hierarchy(events: list[Event]) -> None:
    """Fill ``parent``, ``depth`` and ``self_us`` for host-side events using time containment per thread."""
    lanes: dict[tuple[int, int], list[Event]] = {}
    for e in events:
        e.parent, e.depth, e.self_us = -1, 0, e.dur_us
        if e.cat in HOST_NESTING_CATEGORIES:
            lanes.setdefault((e.pid, e.tid), []).append(e)

    for lane in lanes.values():
        lane.sort(key=lambda e: (e.start_us, -e.dur_us, e.id))
        stack: list[Event] = []
        for e in lane:
            while stack and stack[-1].end_us <= e.start_us + _EPS_US:
                stack.pop()
            # A partially overlapping event cannot be a child; treat it as a sibling.
            while stack and e.end_us > stack[-1].end_us + _EPS_US:
                stack.pop()
            if stack:
                parent = stack[-1]
                e.parent, e.depth = parent.id, parent.depth + 1
                parent.self_us -= e.dur_us
            stack.append(e)

    for e in events:
        if e.self_us < 0:
            e.self_us = 0.0


# ---------------------------------------------------------------------------- helpers
def merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    """Union of ``(start, end)`` intervals, sorted and non-overlapping."""
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def total_length(intervals: Iterable[tuple[float, float]]) -> float:
    return sum(end - start for start, end in intervals)
