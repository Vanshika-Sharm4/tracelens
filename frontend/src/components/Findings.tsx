import type { Finding } from "../types";

interface Props {
  findings: Finding[];
  onFocus: (range: [number, number]) => void;
}

export default function Findings({ findings, onFocus }: Props) {
  if (findings.length === 0) {
    return (
      <section aria-label="Findings">
        <h2>Findings</h2>
        <p className="hint">No bottlenecks crossed the detection thresholds.</p>
      </section>
    );
  }
  return (
    <section aria-label="Findings">
      <h2>Findings</h2>
      <ul className="findings">
        {findings.map((f, i) => (
          <li key={i} className={`finding sev-${f.severity}`}>
            <div className="finding-head">
              <span className="badge">{f.severity}</span>
              <strong>{f.title}</strong>
              <span className="impact">{f.impact_pct.toFixed(1)}% of run</span>
            </div>
            <p>{f.detail}</p>
            <p className="suggestion">{f.suggestion}</p>
            {f.range_us && (
              <button className="link" onClick={() => onFocus(f.range_us as [number, number])}>
                Show on timeline
              </button>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}
