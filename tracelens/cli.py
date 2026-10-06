"""Command-line interface: ``tracelens profile | analyze | overhead | validate | serve``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tracelens.analysis import analyze
from tracelens.cost import HARDWARE, parse_hardware
from tracelens.models import DEMO_MODELS, build_workload
from tracelens.profiler import profile_callable, resolve_device
from tracelens.report import format_report
from tracelens.trace import parse_chrome_trace


def _add_hardware(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--hardware",
        default="auto",
        help=f"roofline peaks: auto, {', '.join(HARDWARE)}, or custom:TFLOPS:GBPS (default: auto-detect)",
    )


def cmd_profile(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    workload = build_workload(args.model, args.batch_size, device)
    print(f"Profiling {args.model} on {device} ({args.iterations} iteration(s))...", file=sys.stderr)
    trace = profile_callable(workload.fn, iterations=args.iterations, device=device)
    analysis = analyze(trace, hardware=parse_hardware(args.hardware))
    print(format_report(analysis))
    if args.out:
        Path(args.out).write_text(json.dumps(analysis.to_dict()))
        print(f"\nAnalysis JSON written to {args.out}", file=sys.stderr)
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    trace = parse_chrome_trace(args.trace)
    if not trace.events:
        print("No timed events found. Is this a Chrome trace from torch.profiler?", file=sys.stderr)
        return 1
    analysis = analyze(trace, hardware=parse_hardware(args.hardware))
    print(json.dumps(analysis.to_dict(), indent=2) if args.json else format_report(analysis))
    return 0


def cmd_overhead(args: argparse.Namespace) -> int:
    from tracelens.overhead import measure_overhead

    device = resolve_device(args.device)
    workload = build_workload(args.model, args.batch_size, device)
    r = measure_overhead(workload.fn, device=device, iterations=args.iterations, sample_every=args.sample_every)
    print(f"Workload: {args.model} on {r['device']} (torch {r['torch']}, {r['iterations']} paired runs)")
    print(f"  baseline step           : {r['baseline_ms']:.3f} ms")
    print(f"  profiled step           : {r['profiled_ms']:.3f} ms")
    print(f"  overhead, profiled step : {r['per_profiled_step_pct']:.1f}%")
    print(f"  overhead, amortized 1/{r['sample_every']:<3}: {r['amortized_pct']:.2f}%")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from tracelens.validate import validate_all

    report = validate_all(device=args.device, models=args.models or None)
    print("Per-operator self time: TraceLens vs torch.profiler key_averages()")
    print(f"{'model':22} {'ops':>4} {'median err':>11} {'max err':>9} {'total err':>10}")
    for r in report["reference_check"]:
        print(
            f"{r['model']:22} {r['operators_compared']:>4} {r['median_rel_error'] * 100:>10.2f}% "
            f"{r['max_rel_error'] * 100:>8.2f}% {r['total_rel_error'] * 100:>9.2f}%"
        )
    print("\nSeeded bottlenecks")
    for r in report["seeded_bottlenecks"]:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  {status} {r['model']:12} expected any of {r['expected_any_of'] or 'none'}; found {r['found']} {r['note']}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nWritten to {args.out}")
    return 0 if all(r["passed"] for r in report["seeded_bottlenecks"]) else 1


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("tracelens.server.app:app", host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tracelens", description="ML inference profiler and bottleneck visualizer")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("profile", help="profile a demo model and print a report")
    p.add_argument("--model", choices=list(DEMO_MODELS), default="tiny_gpt")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--device", default="auto", help="auto, cpu or cuda")
    p.add_argument("--out", help="write the full analysis as JSON to this path")
    _add_hardware(p)
    p.set_defaults(func=cmd_profile)

    p = sub.add_parser("analyze", help="analyze an existing Chrome trace (e.g. from torch.profiler)")
    p.add_argument("trace")
    p.add_argument("--json", action="store_true")
    _add_hardware(p)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("overhead", help="measure profiling overhead for a demo model")
    p.add_argument("--model", choices=list(DEMO_MODELS), default="tiny_gpt")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--sample-every", type=int, default=20, help="profile 1 step in N for the amortized figure")
    p.add_argument("--device", default="auto")
    p.set_defaults(func=cmd_overhead)

    p = sub.add_parser("validate", help="check TraceLens against torch.profiler and seeded bottlenecks")
    p.add_argument("--device", default="auto")
    p.add_argument("--models", nargs="*", choices=list(DEMO_MODELS))
    p.add_argument("--out", help="write the validation report as JSON")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("serve", help="run the API and web UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
