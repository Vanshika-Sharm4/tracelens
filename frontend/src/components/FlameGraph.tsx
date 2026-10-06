import { useEffect, useMemo, useState } from "react";
import type { FlameNode } from "../types";
import { ROW_H, colorFor, descend, formatDuration, layoutFlame } from "../lib/layout";

interface Props {
  cpu: FlameNode;
  gpu: FlameNode | null;
}

export default function FlameGraph({ cpu, gpu }: Props) {
  const [which, setWhich] = useState<"cpu" | "gpu">("cpu");
  const [path, setPath] = useState<string[]>([]);
  const [hover, setHover] = useState<FlameNode | null>(null);
  const full = which === "gpu" && gpu ? gpu : cpu;

  useEffect(() => setPath([]), [cpu, gpu, which]);

  const root = useMemo(() => descend(full, path), [full, path]);
  const rects = useMemo(() => layoutFlame(root), [root]);
  const depth = rects.reduce((m, r) => Math.max(m, r.depth), 0);
  const parentOf = useMemo(() => {
    const map = new Map<FlameNode, FlameNode>();
    const walk = (n: FlameNode) => n.children.forEach((c) => (map.set(c, n), walk(c)));
    walk(root);
    return map;
  }, [root]);

  const pathTo = (node: FlameNode): string[] => {
    const names: string[] = [];
    for (let n: FlameNode | undefined = node; n && n !== root; n = parentOf.get(n)) names.unshift(n.name);
    return [...path, ...names];
  };

  return (
    <section aria-label="Flame graph">
      <div className="section-head">
        <h2>Flame graph</h2>
        <div className="toggle">
          <button className={which === "cpu" ? "on" : ""} onClick={() => setWhich("cpu")}>CPU ops</button>
          <button className={which === "gpu" ? "on" : ""} disabled={!gpu} onClick={() => setWhich("gpu")}>GPU kernels</button>
          {path.length > 0 && <button onClick={() => setPath([])}>Reset zoom</button>}
        </div>
      </div>
      <p className="hint">
        {hover
          ? `${hover.name}: ${formatDuration(hover.value)} total, ${formatDuration(hover.self)} self, ${hover.count} call(s)`
          : "Width is time. Click a block to zoom in."}
      </p>
      <svg className="flame" width="100%" height={(depth + 1) * ROW_H + 2} role="img" aria-label="Flame graph">
        {rects.map((r, i) => (
          <g key={i} onMouseEnter={() => setHover(r.node)} onMouseLeave={() => setHover(null)} onClick={() => r.depth > 0 && setPath(pathTo(r.node))}>
            <rect
              x={`${r.x * 100}%`}
              y={r.depth * ROW_H}
              width={`${r.w * 100}%`}
              height={ROW_H - 1}
              fill={colorFor(r.node.name, which === "gpu" ? "kernel" : "cpu_op")}
              stroke="var(--bg)"
            />
            {r.w > 0.04 && (
              <text x={`${r.x * 100 + 0.4}%`} y={r.depth * ROW_H + 13} className="flame-label">
                {r.node.name.length > Math.floor(r.w * 140) ? r.node.name.slice(0, Math.max(3, Math.floor(r.w * 140))) + "…" : r.node.name}
              </text>
            )}
          </g>
        ))}
      </svg>
    </section>
  );
}
