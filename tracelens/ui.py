"""Compact timeline payload for the web UI."""

from __future__ import annotations

from typing import Any

from tracelens.trace import Trace

_CATS = ["cpu_op", "annotation", "kernel", "memcpy", "memset", "runtime"]


def to_ui(trace: Trace, max_events: int = 20000) -> dict[str, Any]:
    """Events as ``[name_idx, cat_idx, start_us, dur_us, lane, depth]`` rows plus lane and name tables.

    When a trace holds more than ``max_events`` events, the longest ones are kept (so the picture stays faithful about what
    matters) and ``truncated`` is set.
    """
    keep = [
        e
        for e in trace.events
        if e.cat in ("cpu_op", "annotation", "kernel", "memcpy", "memset")
        or (e.cat == "runtime" and "Synchronize" in e.name)
    ]
    total = len(keep)
    truncated = total > max_events
    if truncated:
        keep = sorted(keep, key=lambda e: e.dur_us, reverse=True)[:max_events]
    keep.sort(key=lambda e: (e.start_us, e.depth))

    lane_ids: dict[tuple[str, int, int], int] = {}
    ops_per_thread: dict[int, int] = {}
    for e in keep:
        if not e.on_gpu:
            ops_per_thread[e.tid] = ops_per_thread.get(e.tid, 0) + 1
    main_tid = max(ops_per_thread, key=ops_per_thread.get) if ops_per_thread else None

    def lane_of(e) -> int:
        key = ("gpu" if e.on_gpu else "cpu", e.pid if e.on_gpu else 0, e.tid)
        if key not in lane_ids:
            lane_ids[key] = len(lane_ids)
        return lane_ids[key]

    names: dict[str, int] = {}
    rows = []
    for e in keep:
        idx = names.setdefault(e.name, len(names))
        rows.append([idx, _CATS.index(e.cat), round(e.start_us, 3), round(e.dur_us, 3), lane_of(e), e.depth])

    lanes = []
    for (kind, pid, tid), lane in sorted(lane_ids.items(), key=lambda kv: (kv[0][0] != "cpu", kv[1])):
        if kind == "cpu":
            label = "CPU main thread" if tid == main_tid else f"CPU thread {tid}"
        else:
            label = f"GPU {pid} stream {tid}"
        lanes.append({"id": lane, "label": label, "kind": kind})

    return {
        "names": list(names),
        "cats": _CATS,
        "lanes": lanes,
        "events": rows,
        "span_us": trace.span_us,
        "total_events": total,
        "truncated": truncated,
    }
