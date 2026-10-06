import type { HardwareOption, ModelInfo, ProfileRequest, TraceResult } from "./types";

async function handle<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body && typeof body.detail === "string") message = body.detail;
      else if (body && Array.isArray(body.detail)) message = body.detail.map((d: { msg: string }) => d.msg).join("; ");
    } catch {
      /* keep the status text */
    }
    throw new Error(message);
  }
  return (await response.json()) as T;
}

export const getModels = (): Promise<ModelInfo[]> => fetch("/api/models").then((r) => handle<ModelInfo[]>(r));

export const getHardware = (): Promise<HardwareOption[]> => fetch("/api/hardware").then((r) => handle<HardwareOption[]>(r));

export const getHealth = (): Promise<{ cuda: boolean; torch: string }> => fetch("/api/health").then((r) => handle(r));

export const profileModel = (req: ProfileRequest): Promise<TraceResult> =>
  fetch("/api/profile", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  }).then((r) => handle<TraceResult>(r));

export function uploadTrace(file: File, hardware: string | null): Promise<TraceResult> {
  const form = new FormData();
  form.append("file", file);
  if (hardware) form.append("hardware", hardware);
  return fetch("/api/analyze", { method: "POST", body: form }).then((r) => handle<TraceResult>(r));
}
