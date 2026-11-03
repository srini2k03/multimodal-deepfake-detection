import { useCallback, useEffect, useRef, useState } from "react";
import { getHealth, predict } from "./api.js";
import UploadZone from "./components/UploadZone.jsx";
import ProgressBar from "./components/ProgressBar.jsx";
import ResultDashboard from "./components/ResultDashboard.jsx";
import StatusBadge from "./components/StatusBadge.jsx";
import ThemeToggle from "./components/ThemeToggle.jsx";
import Overview, { Hero } from "./components/Overview.jsx";
import { BROWSER_DEMO, REPO_URL, SETUP_URL } from "./demo/config.js";

const NAV = [
  ["#demo", "Demo"],
  ["#architecture", "Architecture"],
  ["#pipeline", "Pipeline"],
  ["#models", "Models"],
  ["#benchmark", "Benchmark"],
  ["#api", "API"],
];

function BrowserDemoBanner() {
  return (
    <div className="banner banner-warn banner-lg" role="note">
      <strong>This is a browser demo, not a deepfake detector.</strong> It runs the real preprocessing pipeline (frame
      sampling, face detection, alignment and 299×299 cropping) in your browser, and nothing is uploaded. No trained
      model runs here: every score on this page is a <b>placeholder</b> computed from simple pixel or loudness
      statistics and says nothing about whether a file is manipulated. Real detection needs the Python backend with
      trained weights;{" "}
      <a href={SETUP_URL} target="_blank" rel="noreferrer">
        see the setup guide
      </a>
      .
    </div>
  );
}

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

  const workspace = (
    <>
      <UploadZone onFile={analyse} disabled={busy} />

      {busy && (
        <section className="card">
          <ProgressBar
            label={
              BROWSER_DEMO
                ? `Running the preprocessing pipeline in your browser: ${file?.name}`
                : stage === "uploading"
                  ? `Uploading ${file?.name}`
                  : `Analysing ${file?.name}...`
            }
            value={stage === "uploading" || BROWSER_DEMO ? progress : null}
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
    </>
  );

  if (!BROWSER_DEMO) {
    return (
      <div className="app">
        <header className="app-header">
          <div>
            <h1>Multi-Modal Deepfake Detector</h1>
            <p className="subtitle">
              Xception + InceptionV3 face ensemble, temporal aggregation and an audio branch with late fusion.
            </p>
          </div>
          <div className="header-actions">
            <StatusBadge health={health} error={healthError} />
            <ThemeToggle />
          </div>
        </header>

        <main>{workspace}</main>

        <footer className="app-footer">
          Scores are model estimates, not proof. Always corroborate with provenance and human review.
        </footer>
      </div>
    );
  }

  return (
    <>
      <nav className="topnav" aria-label="Sections">
        <div className="topnav-inner">
          <a className="brand" href="#top">
            <img src={`${import.meta.env.BASE_URL}favicon.svg`} alt="" width="22" height="22" />
            <span>Deepfake Detection</span>
          </a>
          <div className="topnav-links">
            {NAV.map(([href, label]) => (
              <a key={href} href={href}>
                {label}
              </a>
            ))}
            <a href={REPO_URL} target="_blank" rel="noreferrer">
              GitHub
            </a>
          </div>
          <ThemeToggle />
        </div>
      </nav>

      <div className="app">
        <Hero />

        <section className="section" id="demo">
          <div className="section-head">
            <h2>Browser demo</h2>
            <StatusBadge health={health} error={healthError} />
          </div>
          <BrowserDemoBanner />
          <main>{workspace}</main>
        </section>

        <Overview />

        <footer className="app-footer">
          <p>
            Browser demo scores are placeholders. Scores from the trained backend are model estimates, not proof: always
            corroborate with provenance and human review.
          </p>
          <p>
            <a href={REPO_URL} target="_blank" rel="noreferrer">
              Source on GitHub
            </a>{" "}
            &middot; MIT licence &middot; face detection by MediaPipe BlazeFace (Apache-2.0)
          </p>
        </footer>
      </div>
    </>
  );
}
