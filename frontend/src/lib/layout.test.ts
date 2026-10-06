import { describe, expect, it } from "vitest";
import type { EventRow, FlameNode, Lane } from "../types";
import {
  AXIS_H,
  LANE_LABEL_H,
  ROW_H,
  clampView,
  descend,
  formatDuration,
  hitTest,
  layoutFlame,
  layoutLanes,
  panView,
  timeToX,
  xToTime,
  zoomView,
} from "./layout";

const lanes: Lane[] = [
  { id: 0, label: "CPU main thread", kind: "cpu" },
  { id: 1, label: "GPU 0 stream 7", kind: "gpu" },
];
// [name, cat, start, dur, lane, depth]
const events: EventRow[] = [
  [0, 0, 0, 100, 0, 0],
  [1, 0, 10, 30, 0, 1],
  [2, 2, 50, 20, 1, 0],
];

describe("layoutLanes", () => {
  it("sizes each lane by its deepest event and stacks them", () => {
    const { lanes: l, height } = layoutLanes(lanes, events);
    expect(l[0].maxDepth).toBe(1);
    expect(l[0].height).toBe(LANE_LABEL_H + 2 * ROW_H);
    expect(l[1].top).toBeGreaterThan(l[0].top + l[0].height - 1);
    expect(height).toBeGreaterThan(AXIS_H);
  });
});

describe("view maths", () => {
  const view = { start: 0, end: 100 };
  it("converts between time and pixels both ways", () => {
    expect(timeToX(25, view, 400)).toBe(100);
    expect(xToTime(100, view, 400)).toBe(25);
  });
  it("zooms around the anchor without moving it", () => {
    const z = zoomView(view, 0.5, 40, 100);
    expect(z.end - z.start).toBeCloseTo(50);
    expect((40 - z.start) / (z.end - z.start)).toBeCloseTo(0.4);
  });
  it("clamps to the trace and never zooms out past it", () => {
    expect(zoomView(view, 4, 50, 100)).toEqual({ start: 0, end: 100 });
    expect(panView({ start: 10, end: 30 }, -50, 100)).toEqual({ start: 0, end: 20 });
    expect(panView({ start: 70, end: 90 }, 50, 100)).toEqual({ start: 80, end: 100 });
    expect(clampView({ start: 5, end: 5 }, 100, 2).end - 5).toBe(2);
  });
});

describe("hitTest", () => {
  const { lanes: layout } = layoutLanes(lanes, events);
  const view = { start: 0, end: 100 };
  const yAt = (laneIdx: number, depth: number) => layout[laneIdx].top + LANE_LABEL_H + depth * ROW_H + 2;

  it("finds the event under the cursor at the right depth", () => {
    expect(hitTest(events, layout, view, 1000, 200, yAt(0, 1))).toBe(1); // t = 20, depth 1
    expect(hitTest(events, layout, view, 1000, 800, yAt(0, 0))).toBe(0); // t = 80, depth 0
    expect(hitTest(events, layout, view, 1000, 600, yAt(1, 0))).toBe(2); // GPU lane, t = 60
  });
  it("returns -1 for empty space and for the lane label", () => {
    expect(hitTest(events, layout, view, 1000, 900, yAt(1, 0))).toBe(-1);
    expect(hitTest(events, layout, view, 1000, 200, layout[0].top + 2)).toBe(-1);
  });
});

describe("flame layout", () => {
  const tree: FlameNode = {
    name: "all", value: 100, self: 0, count: 1,
    children: [
      { name: "a", value: 60, self: 20, count: 3, children: [{ name: "c", value: 40, self: 40, count: 3, children: [] }] },
      { name: "b", value: 30, self: 30, count: 1, children: [] },
    ],
  };
  it("sizes nodes by value relative to the root and places siblings side by side", () => {
    const rects = layoutFlame(tree);
    const by = Object.fromEntries(rects.map((r) => [r.node.name, r]));
    expect(by.a.w).toBeCloseTo(0.6);
    expect(by.b.x).toBeCloseTo(0.6);
    expect(by.c.x).toBe(by.a.x);
    expect(by.c.depth).toBe(2);
  });
  it("drops nodes that would be invisibly thin", () => {
    const thin: FlameNode = { ...tree, children: [...tree.children, { name: "tiny", value: 0.01, self: 0.01, count: 1, children: [] }] };
    expect(layoutFlame(thin).some((r) => r.node.name === "tiny")).toBe(false);
  });
  it("descends a path and stops at a missing name", () => {
    expect(descend(tree, ["a", "c"]).name).toBe("c");
    expect(descend(tree, ["a", "zzz"]).name).toBe("a");
  });
});

describe("formatDuration", () => {
  it("picks a readable unit", () => {
    expect(formatDuration(0.5)).toBe("500 ns");
    expect(formatDuration(12.34)).toBe("12.3 us");
    expect(formatDuration(1500)).toBe("1.50 ms");
    expect(formatDuration(2_500_000)).toBe("2.50 s");
  });
});
