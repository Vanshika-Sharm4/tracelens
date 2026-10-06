export interface Summary {
  wall_ms: number;
  cpu_busy_ms: number;
  n_cpu_ops: number;
  n_kernels: number;
  n_memcpy: number;
  has_gpu: boolean;
  gpu_busy_ms: number | null;
  gpu_util_pct: number | null;
  device_name: string | null;
  peak_memory_mb: number | null;
  peak_memory_op: string | null;
  record_shapes: boolean | null;
}

export interface OpStat {
  name: string;
  count: number;
  total_us: number;
  self_us: number;
  gpu_us: number;
  mean_self_us: number;
  pct: number;
  flops: number | null;
  bytes: number | null;
  intensity: number | null;
  bound: "compute-bound" | "memory-bound" | "unknown";
  example_dims: number[][] | null;
}

export interface Gap {
  device: number;
  start_us: number;
  dur_us: number;
  after: string;
  before: string;
  cause: string;
}

export type Severity = "high" | "medium" | "low" | "info";

export interface Finding {
  kind: string;
  severity: Severity;
  title: string;
  detail: string;
  impact_pct: number;
  suggestion: string;
  ops: string[];
  range_us: [number, number] | null;
  evidence: Record<string, unknown>;
}

export interface FlameNode {
  name: string;
  value: number;
  self: number;
  count: number;
  children: FlameNode[];
}

export interface HardwareInfo {
  name: string;
  peak_tflops: number;
  peak_gbps: number;
  ridge: number;
}

export interface Analysis {
  summary: Summary;
  ops: OpStat[];
  gaps: Gap[];
  findings: Finding[];
  flame_cpu: FlameNode;
  flame_gpu: FlameNode | null;
  hardware: HardwareInfo;
}

export interface Lane {
  id: number;
  label: string;
  kind: "cpu" | "gpu";
}

/** One row per event: [name index, category index, start us, duration us, lane id, depth]. */
export type EventRow = [number, number, number, number, number, number];

export interface Timeline {
  names: string[];
  cats: string[];
  lanes: Lane[];
  events: EventRow[];
  span_us: number;
  total_events: number;
  truncated: boolean;
}

export interface TraceResult {
  id: string;
  label: string;
  created: number;
  analysis: Analysis;
  timeline: Timeline;
}

export interface ModelInfo {
  name: string;
  description: string;
}

export interface HardwareOption {
  key: string;
  name: string;
}

export interface ProfileRequest {
  model: string;
  batch_size: number;
  iterations: number;
  device: "auto" | "cpu" | "cuda";
  hardware: string | null;
}
