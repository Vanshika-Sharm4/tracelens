import io
import json

import pytest
from fastapi.testclient import TestClient

from tracelens import synthetic
from tracelens.server.app import app

client = TestClient(app)


def test_health_and_listings():
    assert client.get("/api/health").json()["status"] == "ok"
    models = client.get("/api/models").json()
    assert {m["name"] for m in models} >= {"mlp", "tiny_gpt", "sync_heavy"}
    assert any(h["key"] == "t4" for h in client.get("/api/hardware").json())


def test_profile_a_demo_model_returns_analysis_and_timeline():
    r = client.post("/api/profile", json={"model": "mlp", "batch_size": 2, "iterations": 1, "device": "cpu"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["analysis"]["summary"]["n_cpu_ops"] > 0
    assert body["timeline"]["events"] and body["timeline"]["lanes"][0]["kind"] == "cpu"
    # the result can be fetched again by id and appears in the list
    assert client.get(f"/api/traces/{body['id']}").json()["id"] == body["id"]
    assert body["id"] in [t["id"] for t in client.get("/api/traces").json()]


def test_profile_rejects_bad_input():
    assert client.post("/api/profile", json={"model": "nope"}).status_code == 400
    assert client.post("/api/profile", json={"model": "mlp", "batch_size": 0}).status_code == 422
    assert client.post("/api/profile", json={"model": "mlp", "device": "tpu"}).status_code == 422


def _upload(payload: bytes, **data):
    return client.post("/api/analyze", files={"file": ("trace.json", io.BytesIO(payload), "application/json")}, data=data)


def test_upload_a_trace_and_get_findings():
    r = _upload(json.dumps(synthetic.sync_gap()).encode(), hardware="t4")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["label"] == "trace.json"
    assert "idle_gap" in {f["kind"] for f in body["analysis"]["findings"]}
    assert body["analysis"]["hardware"]["name"] == "NVIDIA T4"
    assert any(lane["kind"] == "gpu" for lane in body["timeline"]["lanes"])


def test_upload_errors_are_clear():
    assert _upload(b"not json").status_code == 400
    assert _upload(json.dumps({"traceEvents": []}).encode()).status_code == 400
    assert _upload(json.dumps(synthetic.healthy()).encode(), hardware="bogus").status_code == 400


def test_unknown_trace_is_404():
    assert client.get("/api/traces/doesnotexist").status_code == 404


def test_timeline_is_truncated_when_huge():
    from tracelens.trace import parse_chrome_trace
    from tracelens.ui import to_ui

    ui = to_ui(parse_chrome_trace(synthetic.launch_bound(n=500)), max_events=100)
    assert ui["truncated"] and len(ui["events"]) == 100 and ui["total_events"] == 1000
