import type { EventRow, FlameNode, Lane } from "../types";

export const ROW_H = 18;
export const LANE_LABEL_H = 16;
export const LANE_GAP = 10;
export const AXIS_H = 24;

/** The visible window of the timeline, in microseconds. */
export interface View {
  start: number;
  end: number;
}

export interface LaneLayout {
  lane: Lane;
  top: number;
  height: number;
  maxDepth: number;
}

export function layoutLanes(lanes: Lane[], events: EventRow[]): { lanes: LaneLayout[]; height: number } {
  const maxDepth = new Map<number, number>();
  for (const e of events) maxDepth.set(e[4], Math.max(maxDepth.get(e[4]) ?? 0, e[5]));
  let top = AXIS_H;
  const out: LaneLayout[] = [];
  for (const lane of lanes) {
    const depth = maxDepth.get(lane.id) ?? 0;
    const height = LANE_LABEL_H + (depth + 1) * ROW_H;
    out.push({ lane, top, height, maxDepth: depth });
    top += height + LANE_GAP;
  }
  return { lanes: out, height: top };
}

export const timeToX = (t: number, view: View, width: number): number => ((t - view.start) / (view.end - view.start)) * width;

export const xToTime = (x: number, view: View, width: number): number => view.start + (x / width) * (view.end - view.start);

/** Keep a view inside [0, span] and no narrower than ``minSpan`` microseconds. */
export function clampView(view: View, span: number, minSpan = 1): View {
  let width = Math.min(Math.max(view.end - view.start, minSpan), span);
  let start = Math.min(Math.max(view.start, 0), span - width);
  if (start < 0) start = 0;
  width = Math.min(width, span);
  return { start, end: start + width };
}

/** Zoom keeping the time under ``anchor`` fixed. ``factor`` below 1 zooms in. */
export function zoomView(view: View, factor: number, anchor: number, span: number): View {
  const start = anchor - (anchor - view.start) * factor;
  const end = anchor + (view.end - anchor) * factor;
  return clampView({ start, end }, span);
}

export function panView(view: View, deltaUs: number, span: number): View {
  return clampView({ start: view.start + deltaUs, end: view.end + deltaUs }, span);
}

/** Index of the event under pixel (x, y), or -1. */
export function hitTest(
  events: EventRow[],
  layout: LaneLayout[],
  view: View,
  width: number,
  x: number,
  y: number,
): number {
  const lane = layout.find((l) => y >= l.top + LANE_LABEL_H && y < l.top + l.height);
  if (!lane) return -1;
  const depth = Math.floor((y - lane.top - LANE_LABEL_H) / ROW_H);
  const t = xToTime(x, view, width);
  const slack = (view.end - view.start) / width; // one pixel of tolerance so tiny events can be hit
  let best = -1;
  for (let i = 0; i < events.length; i++) {
    const e = events[i];
    if (e[4] !== lane.lane.id || e[5] !== depth) continue;
    if (t >= e[2] - slack && t <= e[2] + e[3] + slack) best = i;
  }
  return best;
}

export function formatDuration(us: number): string {
  if (us < 1) return `${(us * 1000).toFixed(0)} ns`;
  if (us < 1000) return `${us.toFixed(us < 10 ? 2 : 1)} us`;
  if (us < 1_000_000) return `${(us / 1000).toFixed(us < 10_000 ? 2 : 1)} ms`;
  return `${(us / 1_000_000).toFixed(2)} s`;
}

function hash(text: string): number {
  let h = 0;
  for (let i = 0; i < text.length; i++) h = (h * 31 + text.charCodeAt(i)) >>> 0;
  return h;
}

/** Stable colour per name, in a hue family chosen by category. */
export function colorFor(name: string, cat: string): string {
  const h = hash(name);
  switch (cat) {
    case "kernel":
      return `hsl(${20 + (h % 30)}, 75%, 52%)`;
    case "memcpy":
    case "memset":
      return `hsl(${280 + (h % 30)}, 55%, 55%)`;
    case "runtime":
      return `hsl(${350 + (h % 10)}, 70%, 52%)`;
    case "annotation":
      return `hsl(${200 + (h % 20)}, 15%, 55%)`;
    default:
      return `hsl(${195 + (h % 55)}, 62%, 50%)`;
  }
}

export interface FlameRect {
  node: FlameNode;
  x: number; // fraction of the root width
  w: number;
  depth: number;
}

/** Icicle layout: a node's width is its time divided by the root's time; children sit left to right under it. */
export function layoutFlame(root: FlameNode, minFraction = 0.002): FlameRect[] {
  const total = root.value || 1;
  const out: FlameRect[] = [];
  const place = (node: FlameNode, x: number, depth: number): void => {
    const w = node.value / total;
    if (w < minFraction) return;
    out.push({ node, x, w, depth });
    let cursor = x;
    for (const child of node.children) {
      place(child, cursor, depth + 1);
      cursor += child.value / total;
    }
  };
  place(root, 0, 0);
  return out;
}

/** Follow ``path`` (a list of names) down from ``root``; stops early if a name is missing. */
export function descend(root: FlameNode, path: string[]): FlameNode {
  let node = root;
  for (const name of path) {
    const next = node.children.find((c) => c.name === name);
    if (!next) break;
    node = next;
  }
  return node;
}
