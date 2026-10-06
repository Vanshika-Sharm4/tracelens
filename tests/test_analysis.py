"""Each detector is checked against a synthetic trace with a planted bottleneck, and against one without."""

import pytest

from tracelens import synthetic
from tracelens.analysis import analyze
from tracelens.cost import HARDWARE, classify, estimate_cost, parse_hardware
from tracelens.trace import parse_chrome_trace


def kinds(raw):
    return {f.kind for f in analyze(parse_chrome_trace(raw)).findings}


def test_healthy_trace_has_no_gpu_pathologies():
    a = analyze(parse_chrome_trace(synthetic.healthy()))
    assert a.summary["gpu_util_pct"] > 95
    assert not {f.kind for f in a.findings} & {"launch_overhead", "idle_gap", "small_ops"}


def test_launch_bound_trace_is_flagged_with_low_utilisation():
    a = analyze(parse_chrome_trace(synthetic.launch_bound()))
    f = next(f for f in a.findings if f.kind == "launch_overhead")
    assert a.summary["gpu_util_pct"] < 30
    assert f.severity == "high"
    assert f.evidence["tiny_fraction"] == 1.0


def test_sync_gap_is_found_and_attributed_to_synchronization():
    a = analyze(parse_chrome_trace(synthetic.sync_gap()))
    assert len(a.gaps) == 1
    gap = a.gaps[0]
    assert 2900 <= gap.dur_us <= 3100
    assert gap.cause.startswith("host synchronization")
    assert "cudaStreamSynchronize" in gap.cause
    finding = next(f for f in a.findings if f.kind == "idle_gap")
    assert finding.range_us is not None and finding.range_us[1] - finding.range_us[0] == pytest.approx(gap.dur_us)


def test_gap_threshold_is_respected():
    raw = synthetic.sync_gap()
    from tracelens.analysis import AnalysisConfig

    a = analyze(parse_chrome_trace(raw), config=AnalysisConfig(gap_threshold_us=5000))
    assert a.gaps == []


def test_memory_bound_operators_are_identified():
    a = analyze(parse_chrome_trace(synthetic.memory_bound_mix()))
    bound = {o.name: o.bound for o in a.ops}
    assert bound["aten::softmax"] == "memory-bound" and bound["aten::add"] == "memory-bound"
    assert bound["aten::mm"] == "compute-bound"  # 4096x4096x4096 is well above the T4 ridge point
    f = next(f for f in a.findings if f.kind == "memory_bound")
    assert f.impact_pct > 70
    assert a.hardware["name"] == "NVIDIA T4"  # detected from the device name in the trace


def test_gpu_time_is_attributed_to_launching_operator():
    a = analyze(parse_chrome_trace(synthetic.memory_bound_mix()))
    gpu = {o.name: o.gpu_us for o in a.ops}
    assert gpu["aten::softmax"] == pytest.approx(10 * 900)
    assert a.ops[0].name == "aten::softmax"  # ranked by GPU time when a GPU is present


def test_peak_memory_reports_the_running_operator():
    a = analyze(parse_chrome_trace(synthetic.with_memory()))
    assert a.summary["peak_memory_mb"] == pytest.approx(10.0)
    assert a.summary["peak_memory_op"] == "aten::mm"


def test_findings_are_sorted_by_impact():
    a = analyze(parse_chrome_trace(synthetic.memory_bound_mix()))
    impacts = [f.impact_pct for f in a.findings]
    assert impacts == sorted(impacts, reverse=True)


def test_flame_graph_aggregates_repeated_calls():
    a = analyze(parse_chrome_trace(synthetic.launch_bound(n=10)))
    (child,) = a.flame_cpu["children"]
    assert child["name"] == "aten::add" and child["count"] == 10
    assert child["value"] == pytest.approx(80)
    gpu = a.flame_gpu
    assert gpu["children"][0]["name"] == "aten::add"
    assert gpu["children"][0]["children"][0]["name"] == "elementwise_add"


def test_analysis_serializes_to_plain_json():
    import json

    json.dumps(analyze(parse_chrome_trace(synthetic.sync_gap())).to_dict())


# ---------------------------------------------------------------------------- cost model
def test_matmul_cost():
    flops, nbytes = estimate_cost("aten::mm", [[128, 256], [256, 64]], ["float", "float"], None)
    assert flops == 2 * 128 * 64 * 256
    assert nbytes == 4 * (128 * 256 + 256 * 64 + 128 * 64)


def test_addmm_skips_the_bias_input():
    flops, _ = estimate_cost("aten::addmm", [[64], [8, 32], [32, 64]], ["float"] * 3, None)
    assert flops == 2 * 8 * 64 * 32


def test_conv2d_cost_uses_concrete_stride_and_padding():
    flops, _ = estimate_cost(
        "aten::conv2d",
        [[2, 3, 8, 8], [8, 3, 3, 3], [8], [], [], [], []],
        ["float"] * 3 + ["ScalarList"] * 3 + ["Scalar"],
        ["", "", "", "[2, 2]", "[1, 1]", "[1, 1]", "1"],
    )
    out = (8 + 2 - 2 - 1) // 2 + 1  # 4
    assert flops == 2 * (2 * 8 * out * out) * 3 * 9


def test_wrapper_ops_are_not_costed_to_avoid_double_counting():
    assert estimate_cost("aten::linear", [[8, 64], [128, 64], [128]], ["float"] * 3, None) == (None, None)


def test_classify_uses_ridge_point():
    t4 = HARDWARE["t4"]
    assert classify("aten::mm", 1e12, 1e9, t4) == "compute-bound"  # AI 1000 > 203
    assert classify("aten::mm", 1e9, 1e9, t4) == "memory-bound"  # AI 1
    assert classify("aten::gelu", None, None, t4) == "memory-bound"
    assert classify("aten::something_else", None, None, t4) == "unknown"


def test_parse_hardware():
    assert parse_hardware("auto") is None
    assert parse_hardware("A100").name == "NVIDIA A100"
    custom = parse_hardware("custom:50:500")
    assert custom.ridge == pytest.approx(100)
    with pytest.raises(ValueError):
        parse_hardware("nonsense")
