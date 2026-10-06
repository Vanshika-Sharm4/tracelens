"""FastAPI service: profile a demo model or upload a trace, get the analysis and a timeline back.

    uvicorn tracelens.server.app:app --reload

The built React app is served from ``frontend/dist`` when it exists (override with ``TRACELENS_STATIC``).
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from tracelens import __version__
from tracelens.analysis import analyze
from tracelens.cost import HARDWARE, parse_hardware
from tracelens.models import DEMO_MODELS, build_workload
from tracelens.profiler import profile_callable
from tracelens.trace import Trace, parse_chrome_trace
from tracelens.ui import to_ui

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_STORED = 20

app = FastAPI(title="TraceLens", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# torch.profiler is process-global state: two simultaneous profiles would corrupt each other.
_profile_lock = threading.Lock()
_store: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_store_lock = threading.Lock()


class ProfileRequest(BaseModel):
    model: str
    batch_size: int = Field(8, ge=1, le=64)
    iterations: int = Field(3, ge=1, le=20)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    hardware: str | None = None


def _remember(label: str, trace: Trace, hardware: str | None) -> dict[str, Any]:
    try:
        hw = parse_hardware(hardware)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    result = {
        "id": uuid.uuid4().hex[:12],
        "label": label,
        "created": time.time(),
        "analysis": analyze(trace, hardware=hw).to_dict(),
        "timeline": to_ui(trace),
        "meta": trace.meta,
    }
    with _store_lock:
        _store[result["id"]] = result
        while len(_store) > MAX_STORED:
            _store.popitem(last=False)
    return result


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "version": __version__, "cuda": torch.cuda.is_available(), "torch": torch.__version__}


@app.get("/api/models")
def models() -> list[dict[str, str]]:
    return [{"name": k, "description": v} for k, v in DEMO_MODELS.items()]


@app.get("/api/hardware")
def hardware() -> list[dict[str, Any]]:
    return [{"key": k, "name": h.name, "peak_tflops": h.peak_tflops, "peak_gbps": h.peak_gbps} for k, h in HARDWARE.items()]


@app.post("/api/profile")
async def profile_model(req: ProfileRequest) -> dict[str, Any]:
    if req.model not in DEMO_MODELS:
        raise HTTPException(status_code=400, detail=f"unknown model {req.model!r}")
    if req.device == "cuda" and not torch.cuda.is_available():
        raise HTTPException(status_code=400, detail="CUDA requested but no GPU is available on the server")

    def work() -> Trace:
        with _profile_lock:
            device = "cuda" if (req.device == "cuda" or (req.device == "auto" and torch.cuda.is_available())) else "cpu"
            workload = build_workload(req.model, req.batch_size, device)
            return profile_callable(workload.fn, iterations=req.iterations, device=device)

    trace = await run_in_threadpool(work)
    return _remember(f"{req.model} (batch {req.batch_size})", trace, req.hardware)


@app.post("/api/analyze")
async def analyze_upload(file: UploadFile = File(...), hardware: str | None = Form(None)) -> dict[str, Any]:
    size = 0
    chunks = []
    while chunk := await file.read(1024 * 1024):
        size += len(chunk)
        if size > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="trace larger than 200 MB")
        chunks.append(chunk)
    import json

    try:
        data = json.loads(b"".join(chunks))
        trace = await run_in_threadpool(parse_chrome_trace, data)
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"could not read a Chrome trace from that file: {exc}") from exc
    if not trace.events:
        raise HTTPException(status_code=400, detail="the file contains no timed events TraceLens understands")
    return _remember(file.filename or "uploaded trace", trace, hardware)


@app.get("/api/traces")
def list_traces() -> list[dict[str, Any]]:
    with _store_lock:
        return [{"id": r["id"], "label": r["label"], "created": r["created"]} for r in reversed(_store.values())]


@app.get("/api/traces/{trace_id}")
def get_trace(trace_id: str) -> dict[str, Any]:
    with _store_lock:
        result = _store.get(trace_id)
    if result is None:
        raise HTTPException(status_code=404, detail="no such trace (the server keeps only the last %d)" % MAX_STORED)
    return result


def _static_dir() -> Path | None:
    configured = os.environ.get("TRACELENS_STATIC")
    candidates = [Path(configured)] if configured else []
    candidates.append(Path(__file__).resolve().parents[2] / "frontend" / "dist")
    return next((p for p in candidates if (p / "index.html").exists()), None)


_static = _static_dir()
if _static is not None:
    app.mount("/", StaticFiles(directory=_static, html=True), name="ui")
