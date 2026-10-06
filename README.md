# TraceLens

An ML inference profiler and bottleneck visualizer for PyTorch. It captures a `torch.profiler` trace, finds where the time goes, and shows it in a browser: a zoomable timeline, a flame graph, and a ranked list of findings with suggested fixes.

- **Python package + CLI**: profile a model, analyze a saved trace, measure profiling overhead, validate the analysis.
- **Analysis engine**: per-operator self time, GPU utilization, idle-gap attribution, roofline classification, and detectors for the common bottlenecks.
- **FastAPI service + React/TypeScript UI**: profile a demo model or upload any Chrome-trace JSON.

## Quick start

```bash
pip install torch            # or the CPU wheel from pytorch.org
pip install -e ".[dev]"
tracelens profile --model tiny_gpt          # text report
cd frontend && npm install && npm run build && cd ..
tracelens serve                              # UI at http://127.0.0.1:8000
```

Or with Docker: `docker compose up --build` (CPU torch; see below for GPU).

Try the UI without profiling anything: upload `examples/sample_trace.json`.

For frontend development, run `tracelens serve` and, in `frontend/`, `npm run dev` (Vite proxies `/api` to port 8000).

## What it detects

| Finding | Meaning |
|---|---|
| `hotspot` | One operator takes a large share of self time |
| `launch_overhead` | GPU kernels are tiny and the GPU is mostly idle: the CPU cannot launch fast enough |
| `idle_gap` | Gaps on the GPU timeline, each attributed to host synchronization, a busy host, or no host activity |
| `memory_bound` | Large share of time in operators below the roofline ridge point |
| `small_ops` | Thousands of sub-10 us operators: dispatch overhead dominates |
| `peak_memory` | Peak allocated memory and the operator where it occurs |

Findings are ranked by estimated impact (share of the run) and each carries a suggestion. Thresholds live in `AnalysisConfig`.

## How it works

1. `trace.py` parses Chrome-trace JSON, reconstructs host call nesting by time containment per thread, and links CPU launches to GPU kernels through `External id`.
2. `analysis.py` computes self time, merges GPU busy intervals, finds idle gaps, estimates FLOPs/bytes (`cost.py`) and classifies operators against a hardware roofline, then runs the detectors.
3. `server/app.py` exposes it over HTTP; `frontend/` renders it.

Design notes and limits are in [docs/DESIGN.md](docs/DESIGN.md).

## Validation (what is and is not measured)

Run `tracelens validate` (CPU or `--device cuda`). Outputs checked into `results/`.

- **Against torch.profiler**: per-operator self time from TraceLens matches `prof.key_averages()` on 6 demo models (mlp, cnn, tiny_gpt, transformer_encoder, tiny_ops, sync_heavy) with 0.00% error on the CPU run. This checks the parser and nesting logic, not the detectors. It is expected to match closely because both read the same profiler data.
- **Seeded bottlenecks**: `tiny_ops` (thousands of tiny ops) must trigger `launch_overhead` or `small_ops`; on a GPU, `sync_heavy` must trigger `idle_gap`. The CPU run passes; the GPU-only check needs a GPU (see below).
- **Synthetic traces**: `tests/` plants known patterns (launch-bound, sync gaps, memory-bound mix) and checks they are found, and that a healthy trace is not flagged.

### Profiling overhead

Measured with paired, interleaved baseline and profiled runs (`tracelens overhead`), CPU, torch 2.14:

| Workload | Per profiled step | Amortized, 1 step in 20 profiled |
|---|---|---|
| tiny_gpt | 12.9% | 0.65% |
| mlp | 9.2% | 0.46% |

Profiling every step costs roughly 5-20% on CPU here, not under 5%. Under 5% holds only when you sample (profile 1 step in N), which is how it would be used in serving. GPU numbers: run `tracelens overhead --device cuda` and record them in `results/`.

### Not yet done

- GPU validation numbers (needs a GPU; command below).
- A peer study of time-to-find-a-bottleneck. [docs/USER_TEST.md](docs/USER_TEST.md) is the protocol; no results are claimed.
- No C++ component. The profiler wraps `torch.profiler`.

## GPU run (Colab or any CUDA machine)

```bash
git clone <your repo> && cd tracelens
pip install -e ".[dev]"
tracelens validate --device cuda --out results/validation_cuda.json
tracelens overhead --model tiny_gpt --device cuda
tracelens overhead --model transformer_encoder --device cuda
```

## Tests

```bash
pytest -q                          # 44 Python tests, CPU only
cd frontend && npm test            # view maths, layout, hit testing
```

## Layout

```
tracelens/        trace parsing, cost model, analysis, profiler, CLI, server
frontend/         Vite + React + TypeScript UI
tests/            pytest suite (synthetic traces, real CPU profiles, API, CLI)
examples/         sample_trace.json
results/          validation and overhead outputs
docs/             design notes, interview prep, user-test protocol
```

## Limitations

- FLOPs and bytes are estimates (matmul, conv, a list of elementwise ops); other ops are "unknown". Roofline peaks are datasheet figures, so classification is indicative.
- Roofline "memory-bound" on CPU uses a generic profile unless you pass `--hardware custom:TFLOPS:GBPS`.
- The server keeps the last 20 traces in memory and has no auth; do not expose it publicly.
- The timeline draws at most 20,000 events.

MIT licensed.
