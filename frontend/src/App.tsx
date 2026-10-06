import { useCallback, useEffect, useState } from "react";
import { getHardware, getHealth, getModels, profileModel, uploadTrace } from "./api";
import type { HardwareOption, ModelInfo, ProfileRequest, TraceResult } from "./types";
import Controls from "./components/Controls";
import Summary from "./components/Summary";
import Findings from "./components/Findings";
import Timeline from "./components/Timeline";
import FlameGraph from "./components/FlameGraph";
import OpTable from "./components/OpTable";

export default function App() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [hardware, setHardware] = useState<HardwareOption[]>([]);
  const [cuda, setCuda] = useState(false);
  const [result, setResult] = useState<TraceResult | null>(null);
  const [focus, setFocus] = useState<[number, number] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([getModels(), getHardware(), getHealth()])
      .then(([m, h, health]) => {
        setModels(m);
        setHardware(h);
        setCuda(health.cuda);
      })
      .catch((e: Error) => setError(`Cannot reach the TraceLens server: ${e.message}`));
  }, []);

  const run = useCallback(async (work: () => Promise<TraceResult>) => {
    setBusy(true);
    setError(null);
    try {
      setResult(await work());
      setFocus(null);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }, []);

  return (
    <main>
      <header>
        <h1>TraceLens</h1>
        <p>Profile PyTorch inference, then find the bottleneck.</p>
      </header>
      <Controls
        models={models}
        hardware={hardware}
        cuda={cuda}
        busy={busy}
        onProfile={(req: ProfileRequest) => run(() => profileModel(req))}
        onUpload={(file, hw) => run(() => uploadTrace(file, hw))}
      />
      {error && <div className="error" role="alert">{error}</div>}
      {result && (
        <>
          <h2 className="result-title">{result.label}</h2>
          <Summary analysis={result.analysis} />
          <Findings findings={result.analysis.findings} onFocus={setFocus} />
          <Timeline data={result.timeline} focus={focus} />
          <FlameGraph cpu={result.analysis.flame_cpu} gpu={result.analysis.flame_gpu} />
          <OpTable ops={result.analysis.ops} />
        </>
      )}
    </main>
  );
}
