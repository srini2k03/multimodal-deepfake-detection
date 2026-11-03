import { useCallback, useEffect, useRef, useState } from "react";
import { getHealth, predict } from "./api.js";
import UploadZone from "./components/UploadZone.jsx";
import ProgressBar from "./components/ProgressBar.jsx";
import ResultDashboard from "./components/ResultDashboard.jsx";
import StatusBadge from "./components/StatusBadge.jsx";

export default function App() {
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState(null);
  const [file, setFile] = useState(null);
  const [stage, setStage] = useState("idle"); // idle | uploading | analysing | done | error
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const abortRef = useRef(null);

  useEffect(() => {
    const ctrl = new AbortController();
    getHealth(ctrl.signal)
      .then(setHealth)
      .catch((e) => e.name !== "AbortError" && setHealthError(e.message));
    return () => ctrl.abort();
  }, []);

  const analyse = useCallback(async (selected) => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setFile(selected);
    setResult(null);
    setError(null);
    setProgress(0);
    try {
      const res = await predict(selected, {
        signal: ctrl.signal,
        onStage: setStage,
        onUploadProgress: setProgress,
      });
      setResult(res);
      setStage("done");
    } catch (e) {
      if (e.name === "AbortError") return;
      setError(e.message);
      setStage("error");
    }
  }, []);

  const cancel = () => {
    abortRef.current?.abort();
    setStage("idle");
  };

  const busy = stage === "uploading" || stage === "analysing";

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>Multi-Modal Deepfake Detector</h1>
          <p className="subtitle">
            Xception + InceptionV3 face ensemble, temporal aggregation and an audio branch with late fusion.
          </p>
        </div>
        <StatusBadge health={health} error={healthError} />
      </header>

      <main>
        <UploadZone onFile={analyse} disabled={busy} />

        {busy && (
          <section className="card">
            <ProgressBar
              label={stage === "uploading" ? `Uploading ${file?.name}` : `Analysing ${file?.name}...`}
              value={stage === "uploading" ? progress : null}
            />
            <button className="btn-secondary" onClick={cancel}>
              Cancel
            </button>
          </section>
        )}

        {error && (
          <section className="card error" role="alert">
            <strong>Analysis failed:</strong> {error}
          </section>
        )}

        {result && <ResultDashboard result={result} />}
      </main>

      <footer className="app-footer">
        Scores are model estimates, not proof. Always corroborate with provenance and human review.
      </footer>
    </div>
  );
}
