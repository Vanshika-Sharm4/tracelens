# Interview prep

## 60-second pitch
TraceLens takes a PyTorch profiler trace and tells you why inference is slow: it rebuilds the call tree, measures GPU utilization and idle gaps, classifies operators on a roofline, and ranks findings by impact. A React UI shows a zoomable timeline and flame graph.

## Questions to be ready for
- **How do you compute self time?** Stack-based nesting by time containment per thread; self = duration minus children.
- **How do you know it's correct?** Matches `key_averages()` exactly on 6 models, but that shares a data source with the parser. The detectors are checked with seeded workloads and synthetic traces with planted patterns.
- **Overhead?** Per profiled step is ~9-13% on CPU (measured), amortized under 1% at 1-in-20 sampling. Say both.
- **Why is a GPU idle?** Attribute gaps: host sync, host busy, or nothing running on the host.
- **What's the roofline limitation?** FLOP/byte are estimates, datasheet peaks, no cache modelling.
- **What would you do next?** GPU validation at scale, wider cost model, multi-GPU view, diffing two traces.
- **Why no C++?** The profiler wraps torch.profiler; I did not write native code, so I don't claim it.

## Don't say
- Anything about a peer study unless you ran it.
- "Under 5% overhead" without "amortized".
