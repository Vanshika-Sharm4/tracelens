"""Plain-text rendering of an :class:`~tracelens.analysis.Analysis` for the terminal."""

from __future__ import annotations

from tracelens.analysis import Analysis


def format_report(a: Analysis, top: int = 10) -> str:
    s = a.summary
    lines = ["TraceLens report", "=" * 16]
    lines.append(f"Hardware profile : {a.hardware['name']} (ridge {a.hardware['ridge']:.0f} FLOP/B)")
    if s.get("device_name"):
        lines.append(f"Device           : {s['device_name']}")
    lines.append(f"Wall time        : {s['wall_ms']:.2f} ms   CPU busy: {s['cpu_busy_ms']:.2f} ms")
    lines.append(f"Operators        : {s['n_cpu_ops']}   Kernels: {s['n_kernels']}   Memcpy/memset: {s['n_memcpy']}")
    if s["has_gpu"] and s["gpu_util_pct"] is not None:
        lines.append(f"GPU busy         : {s['gpu_busy_ms']:.2f} ms ({s['gpu_util_pct']:.0f}% of its active window)")
    if s.get("peak_memory_mb"):
        lines.append(f"Peak memory      : {s['peak_memory_mb']:.1f} MB (in {s.get('peak_memory_op')})")

    lines += ["", "Findings (ranked by estimated impact)", "-" * 37]
    if not a.findings:
        lines.append("No bottlenecks flagged.")
    for i, f in enumerate(a.findings, 1):
        lines.append(f"{i}. [{f.severity.upper()}] {f.title}")
        lines.append(f"   {f.detail}")
        lines.append(f"   Suggestion: {f.suggestion}")

    label = "GPU time" if s["has_gpu"] and any(o.gpu_us for o in a.ops) else "self time"
    lines += ["", f"Top operators by {label}", "-" * 26]
    header = f"{'operator':36} {'calls':>6} {'self ms':>9} {'gpu ms':>8} {'share':>6}  bound"
    lines.append(header)
    for o in a.ops[:top]:
        lines.append(
            f"{o.name[:36]:36} {o.count:>6} {o.self_us / 1000:>9.3f} {o.gpu_us / 1000:>8.3f} {o.pct:>5.1f}%  {o.bound}"
        )
    return "\n".join(lines)
