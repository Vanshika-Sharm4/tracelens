"""The analysis engine: aggregate a trace and flag likely bottlenecks.

Detectors (each returns zero or more :class:`Finding` objects ranked by how much of the run they affect):

``hotspot``          one operator accounts for a large share of the time
``launch_overhead``  the GPU runs many tiny kernels and sits idle: the host cannot launch work fast enough
``idle_gap``         long stretches where the GPU is idle, with the likely cause (host synchronization or host work)
``memory_bound``     a large share of time is spent in operators that a roofline model says are bandwidth-limited
``small_ops``        many very short operators: dispatch/framework overhead dominates
``peak_memory``      peak allocated memory and the operator running at that moment (informational)

Every detector is a heuristic with documented thresholds (see :class:`AnalysisConfig`). They point you at where to look;
they do not prove a cause.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from typing import Any

from tracelens.cost import Hardware, classify, detect_hardware, estimate_cost
from tracelens.trace import (
    Event,
    Trace,
    merge_intervals,
    total_length,
)


@dataclass
class AnalysisConfig:
    gap_threshold_us: float = 50.0  # GPU idle gaps shorter than this are ignored
    tiny_kernel_us: float = 20.0  # kernels shorter than this count as "tiny"
    tiny_kernel_fraction: float = 0.5  # launch overhead needs at least this fraction of tiny kernels...
    launch_util_ceiling: float = 70.0  # ...and GPU utilisation below this percentage
    hotspot_pct: float = 25.0  # report an operator above this share of time
    small_op_us: float = 10.0  # an operator shorter than this is "small"
    small_op_count: int = 50  # small_ops needs at least this many operators
    small_op_count_fraction: float = 0.6
    small_op_time_fraction: float = 0.3
    memory_bound_pct: float = 15.0  # report memory-bound share above this percentage
    top_ops: int = 25
    top_gaps: int = 10


@dataclass
class OpStat:
    name: str
    count: int
    total_us: float  # inclusive time (children included); not additive across nested ops
    self_us: float  # exclusive time; additive
    gpu_us: float  # GPU kernel time attributed to this op via the launch link
    mean_self_us: float
    pct: float  # share of total self time, or of attributed GPU time when a GPU is present
    flops: float | None
    bytes: float | None
    intensity: float | None  # FLOPs per byte
    bound: str
    example_dims: list | None


@dataclass
class Gap:
    device: int
    start_us: float
    dur_us: float
    after: str  # kernel that ran just before the gap
    before: str  # kernel that ran just after it
    cause: str  # "host synchronization (...)", "host busy in aten::...", or "no host activity"


@dataclass
class Finding:
    kind: str
    severity: str  # high | medium | low | info
    title: str
    detail: str
    impact_pct: float  # rough share of the run this affects
    suggestion: str
    ops: list[str] = field(default_factory=list)
    range_us: tuple[float, float] | None = None  # region to focus in the UI
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class Analysis:
    summary: dict[str, Any]
    ops: list[OpStat]
    gaps: list[Gap]
    findings: list[Finding]
    flame_cpu: dict[str, Any]
    flame_gpu: dict[str, Any] | None
    hardware: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "ops": [asdict(o) for o in self.ops],
            "gaps": [asdict(g) for g in self.gaps],
            "findings": [asdict(f) for f in self.findings],
            "flame_cpu": self.flame_cpu,
            "flame_gpu": self.flame_gpu,
            "hardware": self.hardware,
        }


def _severity(impact_pct: float) -> str:
    if impact_pct >= 30:
        return "high"
    if impact_pct >= 10:
        return "medium"
    return "low"


# ------------------------------------------------------------------------------ main entry
def analyze(trace: Trace, hardware: Hardware | None = None, config: AnalysisConfig | None = None) -> Analysis:
    cfg = config or AnalysisConfig()
    hw = hardware or detect_hardware(trace.meta.get("device_name"), trace.has_gpu)

    wall = trace.span_us
    gpu_events = trace.gpu_events
    by_id = trace.by_id()

    # ---- GPU utilisation: busy time over each device's active window
    per_device: dict[int, list[Event]] = {}
    for e in gpu_events:
        per_device.setdefault(e.pid, []).append(e)
    busy_us = window_us = 0.0
    merged_by_device: dict[int, list[tuple[float, float]]] = {}
    for dev, evs in per_device.items():
        merged = merge_intervals((e.start_us, e.end_us) for e in evs)
        merged_by_device[dev] = merged
        busy_us += total_length(merged)
        window_us += merged[-1][1] - merged[0][0]
    gpu_util = 100.0 * busy_us / window_us if window_us > 0 else None

    cpu_busy = total_length(merge_intervals((e.start_us, e.end_us) for e in trace.cpu_ops))

    # ---- per-operator statistics
    ops = _op_stats(trace, hw, cfg)

    peak_bytes, peak_op = _peak_memory(trace)
    summary = {
        "wall_ms": wall / 1000.0,
        "cpu_busy_ms": cpu_busy / 1000.0,
        "n_cpu_ops": len(trace.cpu_ops),
        "n_kernels": len(trace.kernels),
        "n_memcpy": len(trace.of("memcpy", "memset")),
        "has_gpu": trace.has_gpu,
        "gpu_busy_ms": busy_us / 1000.0 if trace.has_gpu else None,
        "gpu_util_pct": gpu_util,
        "device_name": trace.meta.get("device_name"),
        "peak_memory_mb": peak_bytes / 1e6 if peak_bytes else None,
        "peak_memory_op": peak_op,
        "record_shapes": trace.meta.get("record_shapes"),
    }

    gaps = _find_gaps(trace, merged_by_device, cfg) if trace.has_gpu else []

    findings: list[Finding] = []
    findings += _hotspots(ops, trace.has_gpu, cfg)
    if trace.has_gpu:
        findings += _launch_overhead(trace, gpu_util, gaps, window_us, cfg)
        findings += _idle_gaps(gaps, window_us, cfg)
    findings += _memory_bound(ops, trace.has_gpu, hw, cfg)
    findings += _small_ops(trace, cfg)
    if peak_bytes:
        findings.append(
            Finding(
                kind="peak_memory",
                severity="info",
                title=f"Peak allocated memory {peak_bytes / 1e6:.1f} MB",
                detail=f"Reached while running {peak_op or 'an unknown operator'}.",
                impact_pct=0.0,
                suggestion="If memory is a constraint, look at activation sizes around this operator.",
                ops=[peak_op] if peak_op else [],
            )
        )
    findings.sort(key=lambda f: f.impact_pct, reverse=True)

    return Analysis(
        summary=summary,
        ops=ops[: cfg.top_ops],
        gaps=gaps[: cfg.top_gaps],
        findings=findings,
        flame_cpu=build_flame(trace),
        flame_gpu=build_gpu_flame(trace) if trace.has_gpu else None,
        hardware={"name": hw.name, "peak_tflops": hw.peak_tflops, "peak_gbps": hw.peak_gbps, "ridge": hw.ridge},
    )


# ------------------------------------------------------------------------------ operators
def _op_stats(trace: Trace, hw: Hardware, cfg: AnalysisConfig) -> list[OpStat]:
    name_by_ext = {e.ext_id: e.name for e in trace.cpu_ops if e.ext_id is not None}
    gpu_by_op: dict[str, float] = {}
    for k in trace.kernels:
        if k.ext_id is not None and k.ext_id in name_by_ext:
            n = name_by_ext[k.ext_id]
            gpu_by_op[n] = gpu_by_op.get(n, 0.0) + k.dur_us

    groups: dict[str, dict[str, Any]] = {}
    for e in trace.cpu_ops:
        g = groups.setdefault(
            e.name, {"count": 0, "total": 0.0, "self": 0.0, "flops": 0.0, "bytes": 0.0, "costed": 0, "dims": None}
        )
        g["count"] += 1
        g["total"] += e.dur_us
        g["self"] += e.self_us
        flops, nbytes = estimate_cost(e.name, e.dims, e.dtypes, e.concrete)
        if flops is not None and nbytes:
            g["flops"] += flops
            g["bytes"] += nbytes
            g["costed"] += 1
        if g["dims"] is None and e.dims:
            g["dims"] = e.dims

    total_self = sum(g["self"] for g in groups.values()) or 1.0
    total_gpu = sum(gpu_by_op.values())
    stats: list[OpStat] = []
    for name, g in groups.items():
        flops = g["flops"] if g["costed"] else None
        nbytes = g["bytes"] if g["costed"] else None
        gpu_us = gpu_by_op.get(name, 0.0)
        pct = 100.0 * gpu_us / total_gpu if total_gpu > 0 else 100.0 * g["self"] / total_self
        stats.append(
            OpStat(
                name=name,
                count=g["count"],
                total_us=g["total"],
                self_us=g["self"],
                gpu_us=gpu_us,
                mean_self_us=g["self"] / g["count"],
                pct=pct,
                flops=flops,
                bytes=nbytes,
                intensity=(flops / nbytes) if flops and nbytes else None,
                bound=classify(name, flops, nbytes, hw),
                example_dims=g["dims"],
            )
        )
    stats.sort(key=lambda s: (s.gpu_us if total_gpu > 0 else s.self_us), reverse=True)
    return stats


# ------------------------------------------------------------------------------ GPU gaps
def _find_gaps(trace: Trace, merged_by_device: dict[int, list[tuple[float, float]]], cfg: AnalysisConfig) -> list[Gap]:
    host_ops = [e for e in trace.cpu_ops if e.depth == 0]
    sync_calls = [e for e in trace.of("runtime") if "Synchronize" in e.name or e.name.startswith("cudaMemcpy")]
    kernels_by_dev: dict[int, list[Event]] = {}
    for e in trace.gpu_events:
        kernels_by_dev.setdefault(e.pid, []).append(e)

    gaps: list[Gap] = []
    for dev, merged in merged_by_device.items():
        evs = sorted(kernels_by_dev[dev], key=lambda e: e.start_us)
        for (_, prev_end), (next_start, _) in zip(merged, merged[1:]):
            dur = next_start - prev_end
            if dur < cfg.gap_threshold_us:
                continue
            after = max((e for e in evs if e.end_us <= prev_end + 1e-6), key=lambda e: e.end_us, default=None)
            before = next((e for e in evs if e.start_us >= next_start - 1e-6), None)
            gaps.append(
                Gap(
                    device=dev,
                    start_us=prev_end,
                    dur_us=dur,
                    after=after.name if after else "?",
                    before=before.name if before else "?",
                    cause=_gap_cause(prev_end, next_start, host_ops, sync_calls),
                )
            )
    gaps.sort(key=lambda g: g.dur_us, reverse=True)
    return gaps


def _gap_cause(start: float, end: float, host_ops: list[Event], sync_calls: list[Event]) -> str:
    for s in sync_calls:
        if s.start_us < end and s.end_us > start and "Synchronize" in s.name:
            return f"host synchronization ({s.name})"
    overlapping = [o for o in host_ops if o.start_us < end and o.end_us > start]
    if overlapping:
        top = max(overlapping, key=lambda o: min(o.end_us, end) - max(o.start_us, start))
        return f"host busy in {top.name}"
    return "no host activity recorded"


# ------------------------------------------------------------------------------ detectors
def _hotspots(ops: list[OpStat], has_gpu: bool, cfg: AnalysisConfig) -> list[Finding]:
    found = []
    basis = "GPU time" if has_gpu else "CPU self time"
    for o in ops[:3]:
        if o.pct >= cfg.hotspot_pct:
            found.append(
                Finding(
                    kind="hotspot",
                    severity=_severity(o.pct),
                    title=f"{o.name} takes {o.pct:.0f}% of {basis}",
                    detail=f"{o.count} call(s), {o.self_us / 1000:.2f} ms self time"
                    + (f", {o.gpu_us / 1000:.2f} ms on the GPU" if o.gpu_us else "")
                    + ".",
                    impact_pct=o.pct,
                    suggestion="Optimizing this operator (fusion, a better kernel, lower precision, or a smaller shape) "
                    "has the largest upside.",
                    ops=[o.name],
                    evidence={"count": o.count, "example_dims": o.example_dims},
                )
            )
    return found


def _launch_overhead(trace: Trace, util: float | None, gaps: list[Gap], window_us: float, cfg: AnalysisConfig) -> list[Finding]:
    kernels = trace.kernels
    if len(kernels) < 20 or util is None:
        return []
    durs = [k.dur_us for k in kernels]
    tiny = sum(1 for d in durs if d < cfg.tiny_kernel_us)
    frac = tiny / len(durs)
    if frac < cfg.tiny_kernel_fraction or util > cfg.launch_util_ceiling:
        return []
    impact = max(0.0, 100.0 - util)
    return [
        Finding(
            kind="launch_overhead",
            severity=_severity(impact),
            title=f"Kernel-launch overhead: {frac * 100:.0f}% of kernels run under {cfg.tiny_kernel_us:.0f} us",
            detail=f"{len(kernels)} kernels, median duration {statistics.median(durs):.1f} us, GPU busy only "
            f"{util:.0f}% of its active window. The GPU finishes each kernel faster than the host can launch the next.",
            impact_pct=impact,
            suggestion="Fuse small operators (torch.compile), capture the step in a CUDA Graph, or increase the batch "
            "size so each kernel does more work.",
            evidence={"n_kernels": len(kernels), "median_kernel_us": statistics.median(durs), "tiny_fraction": frac},
        )
    ]


def _idle_gaps(gaps: list[Gap], window_us: float, cfg: AnalysisConfig) -> list[Finding]:
    if not gaps or window_us <= 0:
        return []
    total = sum(g.dur_us for g in gaps)
    impact = 100.0 * total / window_us
    worst = gaps[0]
    causes: dict[str, float] = {}
    for g in gaps:
        key = "host synchronization" if g.cause.startswith("host synchronization") else (
            "host work" if g.cause.startswith("host busy") else "no host activity"
        )
        causes[key] = causes.get(key, 0.0) + g.dur_us
    main_cause = max(causes, key=causes.get)
    return [
        Finding(
            kind="idle_gap",
            severity=_severity(impact),
            title=f"GPU idle for {total / 1000:.2f} ms across {len(gaps)} gap(s) over {cfg.gap_threshold_us:.0f} us",
            detail=f"Longest gap {worst.dur_us / 1000:.2f} ms between {worst.after} and {worst.before}; "
            f"most idle time is attributable to {main_cause}. Worst gap: {worst.cause}.",
            impact_pct=impact,
            suggestion="Remove host-device synchronization from the hot path (.item(), .cpu(), print of tensors), move "
            "data loading off the critical path, or overlap host work with GPU work.",
            ops=[worst.after, worst.before],
            range_us=(worst.start_us, worst.start_us + worst.dur_us),
            evidence={"gaps": len(gaps), "causes_us": causes},
        )
    ]


def _memory_bound(ops: list[OpStat], has_gpu: bool, hw: Hardware, cfg: AnalysisConfig) -> list[Finding]:
    weight = (lambda o: o.gpu_us) if has_gpu and any(o.gpu_us for o in ops) else (lambda o: o.self_us)
    total = sum(weight(o) for o in ops)
    if total <= 0:
        return []
    mem = [o for o in ops if o.bound == "memory-bound"]
    share = 100.0 * sum(weight(o) for o in mem) / total
    if share < cfg.memory_bound_pct:
        return []
    top = sorted(mem, key=weight, reverse=True)[:3]
    return [
        Finding(
            kind="memory_bound",
            severity=_severity(share),
            title=f"{share:.0f}% of time is in memory-bound operators",
            detail="Roofline classification against "
            f"{hw.name} (ridge {hw.ridge:.0f} FLOP/B): " + ", ".join(o.name for o in top) + ".",
            impact_pct=share,
            suggestion="These operators are limited by memory traffic, not arithmetic. Fuse them with neighbours so "
            "tensors make fewer round trips to DRAM (torch.compile, fused kernels), or use lower precision.",
            ops=[o.name for o in top],
            evidence={"hardware": hw.name, "ridge": hw.ridge},
        )
    ]


def _small_ops(trace: Trace, cfg: AnalysisConfig) -> list[Finding]:
    ops = trace.cpu_ops
    if len(ops) < cfg.small_op_count:
        return []
    small = [o for o in ops if o.self_us < cfg.small_op_us]
    total_self = sum(o.self_us for o in ops) or 1.0
    count_frac = len(small) / len(ops)
    time_frac = sum(o.self_us for o in small) / total_self
    if count_frac < cfg.small_op_count_fraction or time_frac < cfg.small_op_time_fraction:
        return []
    return [
        Finding(
            kind="small_ops",
            severity=_severity(100 * time_frac),
            title=f"{count_frac * 100:.0f}% of operators are shorter than {cfg.small_op_us:.0f} us",
            detail=f"{len(small)} of {len(ops)} operators, {time_frac * 100:.0f}% of host time. Per-operator dispatch "
            "overhead dominates over the work itself.",
            impact_pct=100 * time_frac,
            suggestion="Batch work into larger operators, fuse chains of elementwise ops, or use torch.compile / CUDA Graphs.",
            evidence={"small": len(small), "total": len(ops)},
        )
    ]


# ------------------------------------------------------------------------------ memory
def _peak_memory(trace: Trace) -> tuple[int, str | None]:
    if not trace.memory:
        return 0, None
    # Prefer CUDA memory when present, otherwise CPU.
    cuda = [m for m in trace.memory if m.device_type == 1]
    series = cuda or trace.memory
    peak = max(series, key=lambda m: m.total_allocated)
    if peak.total_allocated <= 0:
        return 0, None
    containing = [
        e for e in trace.cpu_ops if e.start_us <= peak.ts_us <= e.end_us
    ]
    op = max(containing, key=lambda e: e.depth).name if containing else None
    return peak.total_allocated, op


# ------------------------------------------------------------------------------ flame graphs
def build_flame(trace: Trace, max_nodes: int = 3000) -> dict[str, Any]:
    """Merge host call stacks (cpu ops and annotations) into one aggregated tree, flame-graph style."""
    nested = [e for e in trace.events if e.cat in ("cpu_op", "annotation")]
    children: dict[int, list[Event]] = {}
    roots: list[Event] = []
    for e in nested:
        if e.parent >= 0:
            children.setdefault(e.parent, []).append(e)
        else:
            roots.append(e)

    budget = [max_nodes]

    def aggregate(nodes: list[Event], depth: int) -> list[dict[str, Any]]:
        groups: dict[str, dict[str, Any]] = {}
        for e in nodes:
            g = groups.setdefault(e.name, {"value": 0.0, "self": 0.0, "count": 0, "kids": []})
            g["value"] += e.dur_us
            g["self"] += e.self_us
            g["count"] += 1
            g["kids"].extend(children.get(e.id, []))
        out = []
        for name, g in sorted(groups.items(), key=lambda kv: kv[1]["value"], reverse=True):
            if budget[0] <= 0:
                break
            budget[0] -= 1
            kids = aggregate(g["kids"], depth + 1) if g["kids"] and depth < 60 else []
            out.append({"name": name, "value": g["value"], "self": g["self"], "count": g["count"], "children": kids})
        return out

    top = aggregate(roots, 0)
    return {"name": "all", "value": sum(n["value"] for n in top), "self": 0.0, "count": 1, "children": top}


def build_gpu_flame(trace: Trace) -> dict[str, Any]:
    """GPU kernel time grouped by the host operator that launched it, then by kernel name."""
    name_by_ext = {e.ext_id: e.name for e in trace.cpu_ops if e.ext_id is not None}
    tree: dict[str, dict[str, dict[str, float]]] = {}
    for k in trace.gpu_events:
        op = name_by_ext.get(k.ext_id, "(unattributed)") if k.ext_id is not None else "(unattributed)"
        leaf = tree.setdefault(op, {}).setdefault(k.name, {"value": 0.0, "count": 0})
        leaf["value"] += k.dur_us
        leaf["count"] += 1
    children = []
    for op, kernels in tree.items():
        kids = [
            {"name": n, "value": v["value"], "self": v["value"], "count": v["count"], "children": []}
            for n, v in sorted(kernels.items(), key=lambda kv: kv[1]["value"], reverse=True)
        ]
        children.append(
            {"name": op, "value": sum(c["value"] for c in kids), "self": 0.0, "count": sum(c["count"] for c in kids), "children": kids}
        )
    children.sort(key=lambda c: c["value"], reverse=True)
    return {"name": "GPU", "value": sum(c["value"] for c in children), "self": 0.0, "count": 1, "children": children}
