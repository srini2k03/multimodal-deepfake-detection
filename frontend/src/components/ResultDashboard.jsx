import ProbabilityGauge from "./ProbabilityGauge.jsx";
import FrameTimeline from "./FrameTimeline.jsx";
import GradCamViewer from "./GradCamViewer.jsx";
import ModelBreakdown from "./ModelBreakdown.jsx";
import AudioSegments from "./AudioSegments.jsx";
import BrowserPreview, { CropGallery } from "./BrowserPreview.jsx";
import { AlertIcon } from "./Icons.jsx";
import { SETUP_URL } from "../demo/config.js";

function formatBytes(n) {
  if (n == null) return "";
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 ** 2).toFixed(1)} MB`;
}

/** One-line reminder shown on every result card in the browser demo. */
function PlaceholderNote({ children }) {
  return (
    <p className="placeholder-note">
      <span className="tag-placeholder">Placeholder</span> {children}
    </p>
  );
}

/** Card with a mono label above the title, matching the portfolio's result panels. */
function ResultCard({ label, title, aside, className = "", children }) {
  return (
    <div className={`sb-card result-card ${className}`}>
      <div className="result-card__head">
        <div>
          {label && <p className="sb-label">{label}</p>}
          <h3 className="sb-card__title">{title}</h3>
        </div>
        {aside}
      </div>
      {children}
    </div>
  );
}

function stripImages(result) {
  const out = { ...result, gradcam: result.gradcam ? "[base64 images omitted]" : null };
  if (result.browser) {
    out.browser = {
      ...result.browser,
      samples: result.browser.samples.map((s) => ({ ...s, annotated: "[image omitted]", crop: "[image omitted]" })),
    };
  }
  return out;
}

export default function ResultDashboard({ result }) {
  const visual = result.modalities.visual;
  const audio = result.modalities.audio;
  const frames = visual?.frames || [];
  const browser = result.mode === "browser-demo";

  return (
    <section className="dashboard" aria-label="Analysis results">
      {browser ? (
        <div className="sb-callout sb-callout--warn banner banner-warn" role="status">
          <span className="callout-icon" aria-hidden="true">
            <AlertIcon size={18} />
          </span>
          <p className="callout-body">
            <strong>Browser demo: placeholder scores, not a deepfake verdict.</strong> {result.warning}{" "}
            <a href={SETUP_URL} target="_blank" rel="noreferrer">
              Set up the backend
            </a>
            .
          </p>
        </div>
      ) : (
        result.warning && (
          <div className="sb-callout sb-callout--warn banner banner-warn" role="status">
            <span className="callout-icon" aria-hidden="true">
              <AlertIcon size={18} />
            </span>
            <p className="callout-body">
              <strong>Untrained / demo weights.</strong> {result.warning}
            </p>
          </div>
        )
      )}

      {browser && visual && (
        <ResultCard
          label="Real pipeline output"
          title="Preprocessing pipeline (real, in your browser)"
          aside={<span className="sb-pill pill-real">runs on your device</span>}
        >
          <BrowserPreview result={result} />
          <p className="sb-small result-foot">
            Detector: {result.browser.detector}. Faces found in {visual.faces_detected} of {visual.frames_analyzed}{" "}
            sampled frame{visual.frames_analyzed === 1 ? "" : "s"}.
          </p>
        </ResultCard>
      )}

      <div className="grid-2">
        <ResultCard label="Score" title={browser ? "Overall placeholder score" : "Verdict"}>
          {browser && (
            <PlaceholderNote>Mean of pixel and loudness heuristics. Not a probability of manipulation.</PlaceholderNote>
          )}
          <ProbabilityGauge
            probability={result.probability_fake}
            threshold={result.threshold}
            label={result.label}
            placeholder={browser}
          />
          <dl className="facts">
            <dt>File</dt>
            <dd>
              {result.file?.name} <span className="sb-muted">{formatBytes(result.file?.size_bytes)}</span>
            </dd>
            <dt>Media</dt>
            <dd>{result.media_type}</dd>
            {result.file?.duration != null && (
              <>
                <dt>Duration</dt>
                <dd>{result.file.duration}s</dd>
              </>
            )}
            <dt>Model mode</dt>
            <dd>{browser ? "browser demo (no models run)" : result.mode}</dd>
            <dt>Processing</dt>
            <dd>{(result.processing_ms / 1000).toFixed(2)}s</dd>
          </dl>
        </ResultCard>

        <ResultCard label="Models" title={browser ? "Per-model breakdown (placeholders)" : "Per-model breakdown"}>
          {browser && <PlaceholderNote>Heuristic values standing in for model outputs.</PlaceholderNote>}
          <ModelBreakdown result={result} />
        </ResultCard>
      </div>

      {frames.length > 1 && (
        <ResultCard
          label="Timeline"
          title={browser ? "Per-frame timeline (placeholder scores)" : "Per-frame timeline"}
          className="sb-card--chart"
        >
          {browser && (
            <PlaceholderNote>
              Frame sampling and face/no-face flags are real; the plotted values are heuristics, not detections.
            </PlaceholderNote>
          )}
          <FrameTimeline frames={frames} threshold={result.threshold} showThreshold={!browser} />
        </ResultCard>
      )}

      <div className="grid-2">
        {visual && browser && (
          <ResultCard
            label="Model input"
            title="Face crops fed to the models"
            aside={<span className="sb-pill sb-pill--pending">Grad-CAM needs trained weights</span>}
          >
            <PlaceholderNote>
              These crops are what Xception and InceptionV3 would score. Grad-CAM needs the trained models, so it is
              only available from the Python backend.
            </PlaceholderNote>
            <CropGallery samples={result.browser.samples} />
          </ResultCard>
        )}
        {visual && !browser && (
          <ResultCard label="Explainability" title="Explainability (Grad-CAM)">
            <GradCamViewer gradcam={result.gradcam} />
          </ResultCard>
        )}
        {audio?.available && (
          <ResultCard
            label="Audio"
            title={browser ? "Audio segments (placeholder scores)" : "Audio segments"}
            className="sb-card--chart"
          >
            {browser && (
              <PlaceholderNote>Decoding and 4 s segmentation are real; bar heights are loudness only.</PlaceholderNote>
            )}
            <AudioSegments segments={audio.segments} threshold={result.threshold} placeholder={browser} />
          </ResultCard>
        )}
      </div>

      <details className="sb-card raw">
        <summary>{browser ? "Raw result object (generated in the browser)" : "Raw JSON response"}</summary>
        <pre className="sb-code">{JSON.stringify(stripImages(result), null, 2)}</pre>
      </details>
    </section>
  );
}
