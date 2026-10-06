import { useMemo, useState } from "react";
import type { OpStat } from "../types";
import { formatDuration } from "../lib/layout";

type SortKey = "self_us" | "total_us" | "count" | "gpu_us";

export default function OpTable({ ops }: { ops: OpStat[] }) {
  const [sort, setSort] = useState<SortKey>("self_us");
  const rows = useMemo(() => [...ops].sort((a, b) => b[sort] - a[sort]), [ops, sort]);
  const th = (key: SortKey, label: string) => (
    <th>
      <button className={`link ${sort === key ? "active" : ""}`} onClick={() => setSort(key)}>
        {label}
      </button>
    </th>
  );
  return (
    <section aria-label="Operators">
      <h2>Operators</h2>
      <table className="ops">
        <thead>
          <tr>
            <th>Operator</th>
            {th("count", "Calls")}
            {th("self_us", "Self time")}
            {th("total_us", "Total")}
            {th("gpu_us", "GPU time")}
            <th>%</th>
            <th>Bound</th>
            <th>AI (FLOP/B)</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((o) => (
            <tr key={o.name}>
              <td className="mono">{o.name}</td>
              <td>{o.count}</td>
              <td>{formatDuration(o.self_us)}</td>
              <td>{formatDuration(o.total_us)}</td>
              <td>{o.gpu_us ? formatDuration(o.gpu_us) : "–"}</td>
              <td>{o.pct.toFixed(1)}</td>
              <td className={`bound-${o.bound}`}>{o.bound}</td>
              <td>{o.intensity != null ? o.intensity.toFixed(1) : "–"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
