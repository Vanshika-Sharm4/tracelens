# Peer study protocol: time to locate a bottleneck

Run this before putting any "time saved" number on a resume. Report only what you measure, with N.

**Participants**: 5+ peers who have used PyTorch. Note their experience.

**Tasks**: 3 traces with a known planted bottleneck (generate with `tracelens profile --model tiny_ops`, `--model sync_heavy` on a GPU, and an upload of your own). Don't reveal the answer.

**Conditions** (within-subject, counterbalance order):
- A: baseline tooling: `prof.key_averages().table()` output, or chrome://tracing / Perfetto with the raw JSON.
- B: TraceLens UI.

**Measure**: seconds from "start" until the participant names the right operator or cause. Stop at 10 minutes (record as a timeout). Record whether the answer was correct.

**Report**: median time per condition, N, number correct, number of timeouts. With 5 people, say so. Don't call it significant.

Template:
| Participant | Task | Condition | Seconds | Correct |
|---|---|---|---|---|
