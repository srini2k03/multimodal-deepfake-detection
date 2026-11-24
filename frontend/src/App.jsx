import { useCallback, useEffect, useRef, useState } from "react";
import { getHealth, predict } from "./api.js";
import UploadZone from "./components/UploadZone.jsx";
import ProgressBar from "./components/ProgressBar.jsx";
import ResultDashboard from "./components/ResultDashboard.jsx";
import StatusBadge from "./components/StatusBadge.jsx";
import ThemeToggle from "./components/ThemeToggle.jsx";
import Overview, { Hero, SectionHead } from "./components/Overview.jsx";
import { AlertIcon, ArrowUpRight, GitHubIcon } from "./components/Icons.jsx";
import { BROWSER_DEMO, PORTFOLIO_URL, REPO_URL, SETUP_URL } from "./demo/config.js";
import { useKitEffects } from "./kit.js";

const NAV = [
  ["#demo", "Demo"],
  ["#architecture", "Architecture"],
  ["#pipeline", "Pipeline"],
  ["#models", "Models"],
  ["#benchmark", "Benchmark"],
  ["#api", "API"],
];

/** Sticky top bar shared with the portfolio's other project sites. */
function TopBar({ links = [] }) {
  return (
    <header className="sb-nav">
      <div className="sb-container sb-nav__inner">
        <a className="sb-brand" href={PORTFOLIO_URL} title="Back to the portfolio">
          <span className="sb-brand__mark" aria-hidden="true">
            SB
          </span>
          <span className="brand-text">
            <span className="brand-name">Srinivasan B.</span>
            <span className="sb-brand__sub">
              <span className="brand-slash" aria-hidden="true">
                /{" "}
              </span>
              Deepfake detection
            </span>
          </span>
        </a>
        <nav className="sb-nav__links" aria-label="Sections">
          {links.map(([href, label]) => (
            <a key={href} className="sb-nav__link nav-section-link" href={href}>
              {label}
            </a>
          ))}
          <a className="sb-nav__link nav-code" href={REPO_URL} target="_blank" rel="noreferrer">
            Code <ArrowUpRight size={12} />
          </a>
          <ThemeToggle />
        </nav>
      </div>
    </header>
  );
}

function SiteFooter({ children }) {
  return (
    <footer className="sb-footer">
      <div className="sb-container sb-footer__inner">
        <div className="footer-copy">
          <p className="footer-name">© 2026 Srinivasan Balasubramanian</p>
          {children}
        </div>
        <div className="footer-links">
          <a href={PORTFOLIO_URL}>Portfolio</a>
          <a href={REPO_URL} target="_blank" rel="noreferrer" className="footer-icon" aria-label="Source on GitHub">
            <GitHubIcon />
          </a>
        </div>
      </div>
    </footer>
  );
}

function BrowserDemoBanner() {
  return (
    <div className="sb-callout sb-callout--warn banner banner-warn banner-lg" role="note">
      <span className="callout-icon" aria-hidden="true">
        <AlertIcon size={20} />
      </span>
      <div>
        <p className="callout-label">Placeholder scores, not a verdict</p>
        <p className="callout-body">
          <strong>This is a browser demo, not a deepfake detector.</strong> It runs the real preprocessing pipeline
          (frame sampling, face detection, alignment and 299×299 cropping) in your browser, and nothing is uploaded. No
          trained model runs here: every score on this page is a <b>placeholder</b> computed from simple pixel or
          loudness statistics and says nothing about whether a file is manipulated. Real detection needs the Python
          backend with trained weights;{" "}
          <a href={SETUP_URL} target="_blank" rel="noreferrer">
            see the setup guide
          </a>
          .
        </p>
      </div>
    </div>
  );
}

// What src/demo/browserAnalysis.js really runs, versus what needs the Python backend.
const IN_BROWSER = [
  "Frame sampling",
  "Face detection (BlazeFace)",
  "Eye alignment + 299² crop",
  "4 s audio segments",
  "Fusion arithmetic",
];
const BACKEND_ONLY = ["Xception", "InceptionV3", "Temporal attention", "Audio CNN", "Grad-CAM"];

function WhereItRuns() {
  return (
    <div className="where-runs" aria-label="Which stages run on this page">
      <div className="where-runs__group">
        <p className="sb-label">Runs here, in your browser</p>
        <ul className="where-runs__list">
          {IN_BROWSER.map((s) => (
            <li key={s} className="sb-tag stage-tag stage-tag--live">
              {s}
            </li>
          ))}
        </ul>
      </div>
      <div className="where-runs__group">
        <p className="sb-label">Needs the Python backend + trained weights</p>
        <ul className="where-runs__list">
          {BACKEND_ONLY.map((s) => (
            <li key={s} className="sb-tag stage-tag stage-tag--off">
              {s}
            </li>
          ))}
        </ul>
      </div>
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

  useKitEffects();

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

  // Short screen-reader announcement for the async states (the dashboard itself is not a live region).
  const announcement = busy
    ? `Analysing ${file?.name ?? "file"}`
    : error
      ? `Analysis failed: ${error}`
      : result
        ? result.mode === "browser-demo"
          ? `Done. Placeholder score ${(result.probability_fake * 100).toFixed(1)} percent, not a verdict.`
          : `Done. Manipulation probability ${(result.probability_fake * 100).toFixed(1)} percent.`
        : "";

  const workspace = (
    <div className="workspace">
      <p className="sr-only" role="status" aria-live="polite">
        {announcement}
      </p>

      <UploadZone onFile={analyse} disabled={busy} />

      {busy && (
        <section className="sb-card progress-card" aria-label="Analysis progress">
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
          <button type="button" className="sb-btn sb-btn--sm btn-secondary" onClick={cancel}>
            Cancel
          </button>
        </section>
      )}

      {error && (
        <section className="sb-callout sb-callout--error analysis-error" role="alert">
          <span className="callout-icon" aria-hidden="true">
            <AlertIcon size={18} />
          </span>
          <div>
            <p className="callout-body">
              <strong>Analysis failed:</strong> {error}
            </p>
            <p className="callout-hint">Drop another file to try again.</p>
          </div>
        </section>
      )}

      {result && <ResultDashboard result={result} />}
    </div>
  );

  if (!BROWSER_DEMO) {
    return (
      <div className="site sb-grain">
        <div className="sb-backdrop" aria-hidden="true" />
        <a className="skip-link" href="#main">
          Skip to content
        </a>
        <TopBar />
        <main id="main" className="sb-container app">
          <header className="app-header">
            <div>
              <p className="sb-kicker">
                <span className="sb-kicker__index">Dashboard</span>
                <span className="sb-kicker__rule" />
                Flask API
              </p>
              <h1 className="sb-h1 app-title">
                Multi-Modal Deepfake <span className="sb-serif">Detector</span>
              </h1>
              <p className="sb-lead subtitle">
                Xception + InceptionV3 face ensemble, temporal aggregation and an audio branch with late fusion.
              </p>
            </div>
            <div className="header-actions">
              <StatusBadge health={health} error={healthError} />
            </div>
          </header>

          {workspace}
        </main>

        <SiteFooter>
          <p>Scores are model estimates, not proof. Always corroborate with provenance and human review.</p>
          <p className="sb-faint">React + Vite dashboard for the Flask API in this repository.</p>
        </SiteFooter>
      </div>
    );
  }

  return (
    <div className="site sb-grain">
      <div className="sb-backdrop" aria-hidden="true" />
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <TopBar links={NAV} />

      <main id="main">
        <Hero />

        <section className="section-block" id="demo" aria-labelledby="demo-title">
          <div className="sb-container">
            <div className="section-inner">
              <SectionHead
                id="demo"
                index="01"
                label="Browser demo"
                title={
                  <>
                    Try the pipeline on <span className="sb-serif">your own</span> media.
                  </>
                }
                aside={<StatusBadge health={health} error={healthError} />}
              >
                Pick an image, a short video or an audio clip. The page samples frames, finds and aligns faces and cuts
                the 299×299 crops the models would receive, all on your device.
              </SectionHead>
              <BrowserDemoBanner />
              <WhereItRuns />
              {workspace}
            </div>
          </div>
        </section>

        <Overview />
      </main>

      <SiteFooter>
        <p>
          Browser demo scores are placeholders. Scores from the trained backend are model estimates, not proof: always
          corroborate with provenance and human review.
        </p>
        <p className="sb-faint">
          React + Vite static build (<code>VITE_DEMO=browser</code>) deployed to GitHub Pages by GitHub Actions · MIT
          licence · face detection by MediaPipe BlazeFace (Apache-2.0).{" "}
          <a href={REPO_URL} target="_blank" rel="noreferrer">
            Source on GitHub
          </a>
          .
        </p>
      </SiteFooter>
    </div>
  );
}
