import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Timeline as TimelineData } from "../types";
import {
  AXIS_H, LANE_LABEL_H, ROW_H, type View, clampView, colorFor, formatDuration, hitTest, layoutLanes, panView, timeToX, xToTime, zoomView,
} from "../lib/layout";

interface Props {
  data: TimelineData;
  focus: [number, number] | null;
}

const HEIGHT_CAP = 520;

export default function Timeline({ data, focus }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(900);
  const [view, setView] = useState<View>({ start: 0, end: Math.max(data.span_us, 1) });
  const [hover, setHover] = useState<{ index: number; x: number; y: number } | null>(null);
  const drag = useRef<{ x: number; start: number } | null>(null);
  const span = Math.max(data.span_us, 1);
  const { lanes: layout, height } = useMemo(() => layoutLanes(data.lanes, data.events), [data]);
  const canvasH = Math.min(height, HEIGHT_CAP);

  useEffect(() => setView({ start: 0, end: span }), [data, span]);
  useEffect(() => {
    if (!focus) return;
    const pad = Math.max((focus[1] - focus[0]) * 0.25, 1);
    setView(clampView({ start: focus[0] - pad, end: focus[1] + pad }, span, 1));
  }, [focus, span]);
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(Math.max(300, el.clientWidth)));
    ro.observe(el);
    setWidth(Math.max(300, el.clientWidth));
    return () => ro.disconnect();
  }, []);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = canvasH * dpr;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const css = getComputedStyle(document.documentElement);
    const fg = css.getPropertyValue("--fg").trim() || "#222";
    const muted = css.getPropertyValue("--muted").trim() || "#888";
    ctx.clearRect(0, 0, width, canvasH);
    ctx.font = "11px system-ui, sans-serif";
    // axis
    const step = niceStep((view.end - view.start) / 8);
    ctx.fillStyle = muted;
    ctx.strokeStyle = muted;
    ctx.globalAlpha = 0.25;
    for (let t = Math.ceil(view.start / step) * step; t < view.end; t += step) {
      const x = timeToX(t, view, width);
      ctx.beginPath(); ctx.moveTo(x, AXIS_H - 4); ctx.lineTo(x, canvasH); ctx.stroke();
    }
    ctx.globalAlpha = 1;
    for (let t = Math.ceil(view.start / step) * step; t < view.end; t += step) {
      ctx.fillText(formatDuration(t), timeToX(t, view, width) + 3, 14);
    }
    for (const l of layout) {
      ctx.fillStyle = fg;
      ctx.fillText(l.lane.label, 4, l.top + 11);
    }
    const laneTop = new Map(layout.map((l) => [l.lane.id, l.top]));
    for (const e of data.events) {
      const x0 = timeToX(e[2], view, width);
      const x1 = timeToX(e[2] + e[3], view, width);
      if (x1 < 0 || x0 > width) continue;
      const top = laneTop.get(e[4]);
      if (top === undefined) continue;
      const y = top + LANE_LABEL_H + e[5] * ROW_H;
      if (y > canvasH) continue;
      const w = Math.max(x1 - x0, 1);
      ctx.fillStyle = colorFor(data.names[e[0]], data.cats[e[1]]);
      ctx.fillRect(x0, y, w, ROW_H - 2);
      if (w > 50) {
        ctx.fillStyle = "#fff";
        const label = data.names[e[0]];
        const max = Math.floor((w - 6) / 6);
        ctx.fillText(label.length > max ? label.slice(0, Math.max(max - 1, 1)) + "…" : label, Math.max(x0, 0) + 3, y + 12);
      }
    }
  }, [data, layout, view, width, canvasH]);

  useEffect(() => draw(), [draw]);

  const local = (ev: { clientX: number; clientY: number }) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    return { x: ev.clientX - rect.left, y: ev.clientY - rect.top };
  };

  // Wheel must be non-passive to preventDefault, so attach manually.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const onWheel = (ev: WheelEvent) => {
      ev.preventDefault();
      const { x } = local(ev);
      setView((v) => zoomView(v, ev.deltaY < 0 ? 0.8 : 1.25, xToTime(x, v, width), span));
    };
    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", onWheel);
  }, [width, span]);

  const onMove = (ev: React.MouseEvent) => {
    const { x, y } = local(ev);
    if (drag.current) {
      const dt = ((drag.current.x - x) / width) * (view.end - view.start);
      setView(panView({ start: drag.current.start, end: drag.current.start + (view.end - view.start) }, dt, span));
      return;
    }
    const index = hitTest(data.events, layout, view, width, x, y);
    setHover(index >= 0 ? { index, x, y } : null);
  };

  const hovered = hover ? data.events[hover.index] : null;
  return (
    <section aria-label="Timeline">
      <div className="section-head">
        <h2>Timeline</h2>
        <div className="toggle">
          <button onClick={() => setView((v) => zoomView(v, 0.5, (v.start + v.end) / 2, span))}>Zoom in</button>
          <button onClick={() => setView((v) => zoomView(v, 2, (v.start + v.end) / 2, span))}>Zoom out</button>
          <button onClick={() => setView({ start: 0, end: span })}>Reset</button>
        </div>
      </div>
      <p className="hint">
        Scroll to zoom, drag to pan. Showing {formatDuration(view.end - view.start)} of {formatDuration(span)}
        {data.truncated && ` (first ${data.events.length.toLocaleString()} of ${data.total_events.toLocaleString()} events)`}.
      </p>
      <div className="timeline-wrap" ref={wrapRef}>
        <canvas
          ref={canvasRef}
          style={{ width: "100%", height: canvasH, cursor: drag.current ? "grabbing" : "grab" }}
          onMouseDown={(ev) => (drag.current = { x: local(ev).x, start: view.start })}
          onMouseUp={() => (drag.current = null)}
          onMouseLeave={() => { drag.current = null; setHover(null); }}
          onMouseMove={onMove}
        />
        {hovered && hover && (
          <div className="tooltip" style={{ left: Math.min(hover.x + 12, width - 260), top: hover.y + 12 }}>
            <strong>{data.names[hovered[0]]}</strong>
            <div>{formatDuration(hovered[3])} · starts at {formatDuration(hovered[2])}</div>
            <div className="muted">{data.cats[hovered[1]]}</div>
          </div>
        )}
      </div>
    </section>
  );
}

function niceStep(raw: number): number {
  const pow = Math.pow(10, Math.floor(Math.log10(Math.max(raw, 1e-9))));
  const f = raw / pow;
  return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * pow;
}
