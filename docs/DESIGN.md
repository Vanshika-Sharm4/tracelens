# Design notes

## Input: one format
Everything consumes Chrome-trace JSON as written by `torch.profiler`. That keeps the tool usable on traces from anywhere (training jobs, other people's runs), and the demo-model profiler is just one producer.

## Parsing
- Event categories used: `cpu_op`, `kernel`, `gpu_memcpy`, `gpu_memset`, `cuda_runtime`, `python_function`, `user_annotation`; `[memory]` instant events for allocation.
- Host nesting is rebuilt by time containment per (pid, tid) with a stack. Self time = duration minus children. GPU events use the stream as their lane.
- CPU launches link to kernels via `External id`, which gives the "launching op -> kernel" GPU flame graph.

## Analysis
- **GPU utilization** = merged busy intervals / the device's active window (first to last event), not wall time of the whole trace.
- **Idle gaps** above `gap_threshold_us` are attributed: host synchronization (a `cudaStreamSynchronize`-like call spans the gap), host busy in an op, or no host activity.
- **Roofline**: arithmetic intensity = FLOPs / bytes vs ridge = peak FLOP/s / peak B/s. Wrapper ops (`aten::linear`, `aten::matmul`) are not costed, because the leaf `mm`/`addmm` inside is, and counting both double counts.
- Detectors are plain threshold rules, deliberately explainable. Thresholds are in `AnalysisConfig`.

## Overhead measurement
Paired, interleaved baseline/profiled runs cancel drift. Reported both per profiled step (worst case) and amortized for 1-in-N sampling.

## Server
FastAPI. `torch.profiler` is process-global, so profile requests take a lock. Uploads are capped at 200 MB. Results are held in memory (last 20). The built UI is served as static files from the same process.

## UI
Canvas timeline (many thousands of events; wheel zoom anchored at the cursor, drag pan, hit-testing with one pixel of slack) and an SVG icicle flame graph (click to zoom into a subtree). View maths is in `frontend/src/lib/layout.ts` and unit-tested.

## Known weak spots
- Cost model covers a limited operator set.
- Idle-gap cause attribution is heuristic.
- No multi-GPU or distributed view.
- Validation against `key_averages()` shares a data source with the parser, so it checks parsing, not ground truth.
