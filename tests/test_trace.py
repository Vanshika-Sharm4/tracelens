"""Parser and hierarchy tests, on hand-built traces where the right answer is known."""

import pytest

from tracelens import synthetic
from tracelens.trace import merge_intervals, parse_chrome_trace, total_length


def _chrome(*events):
    return {"traceEvents": list(events)}


def op(name, ts, dur, tid=1, **args):
    return {"ph": "X", "cat": "cpu_op", "name": name, "pid": 1, "tid": tid, "ts": ts, "dur": dur, "args": args}


def test_nesting_and_self_time():
    # outer (0..100) contains a (10..40) and b (50..90); a contains c (15..25)
    trace = parse_chrome_trace(_chrome(op("outer", 0, 100), op("a", 10, 30), op("b", 50, 40), op("c", 15, 10)))
    by_name = {e.name: e for e in trace.events}
    assert by_name["outer"].depth == 0 and by_name["a"].depth == 1 and by_name["c"].depth == 2
    assert by_name["c"].parent == by_name["a"].id
    assert by_name["outer"].self_us == pytest.approx(100 - 30 - 40)
    assert by_name["a"].self_us == pytest.approx(30 - 10)
    assert by_name["c"].self_us == pytest.approx(10)


def test_threads_do_not_nest_into_each_other():
    trace = parse_chrome_trace(_chrome(op("x", 0, 100, tid=1), op("y", 10, 20, tid=2)))
    assert all(e.depth == 0 for e in trace.events)
    assert {e.self_us for e in trace.events} == {100.0, 20.0}


def test_times_are_relative_to_trace_start():
    trace = parse_chrome_trace(_chrome(op("x", 5_000_000, 10), op("y", 5_000_100, 10)))
    assert min(e.start_us for e in trace.events) == 0.0
    assert trace.span_us == pytest.approx(110)


def test_unknown_categories_and_non_complete_events_are_ignored():
    raw = _chrome(
        op("keep", 0, 10),
        {"ph": "X", "cat": "Trace", "name": "PyTorch Profiler", "pid": 1, "tid": 1, "ts": 0, "dur": 999},
        {"ph": "M", "name": "process_name", "pid": 1, "tid": 0, "args": {"name": "python"}},
        {"ph": "s", "cat": "ac2g", "name": "ac2g", "id": 1, "pid": 1, "tid": 1, "ts": 1},
    )
    trace = parse_chrome_trace(raw)
    assert [e.name for e in trace.events] == ["keep"]


def test_kernel_stream_becomes_lane_and_ext_id_links_to_op():
    trace = parse_chrome_trace(synthetic.launch_bound(n=3))
    kernels = trace.kernels
    assert len(kernels) == 3 and all(k.tid == 7 for k in kernels)
    op_ext = {e.ext_id for e in trace.cpu_ops}
    assert {k.ext_id for k in kernels} == op_ext
    assert trace.meta["device_name"] == "Tesla T4"


def test_memory_events_are_parsed():
    trace = parse_chrome_trace(synthetic.with_memory())
    assert [m.total_allocated for m in trace.memory] == [1_000_000, 10_000_000, 1_000_000]


def test_not_a_trace_raises(tmp_path):
    path = tmp_path / "x.json"
    path.write_text("42")  # valid JSON, but not a trace
    with pytest.raises(ValueError):
        parse_chrome_trace(path)


def test_merge_intervals():
    merged = merge_intervals([(0, 5), (3, 8), (10, 12), (12, 13)])
    assert merged == [(0, 8), (10, 13)]
    assert total_length(merged) == 11
