import type { Analysis } from "../types";
import { formatDuration } from "../lib/layout";

export default function Summary({ analysis }: { analysis: Analysis }) {
  const s = analysis.summary;
  const items: [string, string][] = [
    ["Wall time", formatDuration(s.wall_ms * 1000)],
    ["CPU ops", String(s.n_cpu_ops)],
  ];
  if (s.has_gpu) {
    items.push(
      ["GPU kernels", String(s.n_kernels)],
      ["GPU busy", s.gpu_busy_ms != null ? formatDuration(s.gpu_busy_ms * 1000) : "–"],
      ["GPU utilization", s.gpu_util_pct != null ? `${s.gpu_util_pct.toFixed(1)}%` : "–"],
    );
  }
  if (s.peak_memory_mb != null) items.push(["Peak memory", `${s.peak_memory_mb.toFixed(1)} MB`]);
  items.push(["Roofline", `${analysis.hardware.name} (ridge ${analysis.hardware.ridge.toFixed(0)} FLOP/B)`]);
  return (
    <section className="summary" aria-label="Summary">
      {items.map(([k, v]) => (
        <div className="stat" key={k}>
          <span className="stat-label">{k}</span>
          <span className="stat-value">{v}</span>
        </div>
      ))}
      {s.device_name && <div className="stat"><span className="stat-label">Device</span><span className="stat-value">{s.device_name}</span></div>}
    </section>
  );
}
