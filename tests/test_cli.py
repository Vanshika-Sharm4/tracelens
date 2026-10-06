import json

from tracelens import synthetic
from tracelens.cli import main


def test_analyze_command_prints_a_report(tmp_path, capsys):
    path = tmp_path / "t.json"
    path.write_text(json.dumps(synthetic.sync_gap()))
    assert main(["analyze", str(path), "--hardware", "t4"]) == 0
    out = capsys.readouterr().out
    assert "GPU idle" in out and "NVIDIA T4" in out


def test_analyze_json_output_is_valid(tmp_path, capsys):
    path = tmp_path / "t.json"
    path.write_text(json.dumps(synthetic.launch_bound()))
    assert main(["analyze", str(path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["summary"]["n_kernels"] == 200


def test_analyze_empty_trace_fails(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"traceEvents": []}))
    assert main(["analyze", str(path)]) == 1
