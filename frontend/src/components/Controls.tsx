import { useRef, useState } from "react";
import type { HardwareOption, ModelInfo, ProfileRequest } from "../types";

interface Props {
  models: ModelInfo[];
  hardware: HardwareOption[];
  cuda: boolean;
  busy: boolean;
  onProfile: (req: ProfileRequest) => void;
  onUpload: (file: File, hardware: string | null) => void;
}

export default function Controls({ models, hardware, cuda, busy, onProfile, onUpload }: Props) {
  const [model, setModel] = useState("tiny_gpt");
  const [batch, setBatch] = useState(8);
  const [iterations, setIterations] = useState(3);
  const [device, setDevice] = useState<ProfileRequest["device"]>("auto");
  const [hw, setHw] = useState("auto");
  const fileRef = useRef<HTMLInputElement>(null);
  const hwValue = hw === "auto" ? null : hw;

  return (
    <section className="controls" aria-label="Profile or upload">
      <div className="control-group">
        <h2>Profile a demo model</h2>
        <label>
          Model
          <select value={model} onChange={(e) => setModel(e.target.value)}>
            {models.map((m) => (
              <option key={m.name} value={m.name}>
                {m.name}
              </option>
            ))}
          </select>
        </label>
        <label>
          Batch
          <input type="number" min={1} max={64} value={batch} onChange={(e) => setBatch(Number(e.target.value) || 1)} />
        </label>
        <label>
          Steps
          <input type="number" min={1} max={20} value={iterations} onChange={(e) => setIterations(Number(e.target.value) || 1)} />
        </label>
        <label>
          Device
          <select value={device} onChange={(e) => setDevice(e.target.value as ProfileRequest["device"])}>
            <option value="auto">auto</option>
            <option value="cpu">cpu</option>
            <option value="cuda" disabled={!cuda}>
              cuda{cuda ? "" : " (no GPU on server)"}
            </option>
          </select>
        </label>
        <label>
          Roofline
          <select value={hw} onChange={(e) => setHw(e.target.value)}>
            <option value="auto">auto-detect</option>
            {hardware.map((h) => (
              <option key={h.key} value={h.key}>
                {h.name}
              </option>
            ))}
          </select>
        </label>
        <button
          disabled={busy}
          onClick={() => onProfile({ model, batch_size: batch, iterations, device, hardware: hwValue })}
        >
          {busy ? "Working…" : "Profile"}
        </button>
      </div>
      <div className="control-group">
        <h2>Or upload a trace</h2>
        <p className="hint">Chrome-trace JSON from torch.profiler (prof.export_chrome_trace).</p>
        <input
          ref={fileRef}
          type="file"
          accept=".json,application/json"
          disabled={busy}
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) onUpload(file, hwValue);
            if (fileRef.current) fileRef.current.value = "";
          }}
        />
      </div>
    </section>
  );
}
