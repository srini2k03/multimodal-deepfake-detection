import ProbabilityGauge from "./ProbabilityGauge.jsx";
import FrameTimeline from "./FrameTimeline.jsx";
import GradCamViewer from "./GradCamViewer.jsx";
import ModelBreakdown from "./ModelBreakdown.jsx";
import AudioSegments from "./AudioSegments.jsx";
import BrowserPreview, { CropGallery } from "./BrowserPreview.jsx";
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
    <section className="dashboard">
      {browser ? (
        <div className="banner banner-warn" role="status">
          <strong>Browser demo: placeholder scores, not a deepfake verdict.</strong> {result.warning}{" "}
          <a href={SETUP_URL} target="_blank" rel="noreferrer">
            Set up the backend
          </a>
          .
        </div>
      ) : (
        result.warning && (
          <div className="banner banner-warn" role="status">
            <strong>Untrained / demo weights.</strong> {result.warning}
          </div>
        )
      )}

      {browser && visual && (
        <div className="card">
          <h3>Preprocessing pipeline (real, in your browser)</h3>
          <BrowserPreview result={result} />
          <p className="muted small">
            Detector: {result.browser.detector}. Faces found in {visual.faces_detected} of {visual.frames_analyzed}{" "}
            sampled frame{visual.frames_analyzed === 1 ? "" : "s"}.
          </p>
        </div>
      )}

      <div className="grid-2">
        <div className="card">
          <h3>{browser ? "Overall placeholder score" : "Verdict"}</h3>
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
              {result.file?.name} <span className="muted">{formatBytes(result.file?.size_bytes)}</span>
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
        </div>

        <div className="card">
          <h3>{browser ? "Per-model breakdown (placeholders)" : "Per-model breakdown"}</h3>
          {browser && <PlaceholderNote>Heuristic values standing in for model outputs.</PlaceholderNote>}
          <ModelBreakdown result={result} />
        </div>
      </div>

      {frames.length > 1 && (
        <div className="card">
          <h3>{browser ? "Per-frame timeline (placeholder scores)" : "Per-frame timeline"}</h3>
          {browser && (
            <PlaceholderNote>
              Frame sampling and face/no-face flags are real; the plotted values are heuristics, not detections.
            </PlaceholderNote>
          )}
          <FrameTimeline frames={frames} threshold={result.threshold} showThreshold={!browser} />
        </div>
      )}

      <div className="grid-2">
        {visual && browser && (
          <div className="card">
            <h3>Face crops fed to the models</h3>
            <PlaceholderNote>
              These crops are what Xception and InceptionV3 would score. Grad-CAM needs the trained models, so it is
              only available from the Python backend.
            </PlaceholderNote>
            <CropGallery samples={result.browser.samples} />
          </div>
        )}
        {visual && !browser && (
          <div className="card">
            <h3>Explainability (Grad-CAM)</h3>
            <GradCamViewer gradcam={result.gradcam} />
          </div>
        )}
        {audio?.available && (
          <div className="card">
            <h3>{browser ? "Audio segments (placeholder scores)" : "Audio segments"}</h3>
            {browser && (
              <PlaceholderNote>Decoding and 4 s segmentation are real; bar heights are loudness only.</PlaceholderNote>
            )}
            <AudioSegments segments={audio.segments} threshold={result.threshold} placeholder={browser} />
          </div>
        )}
      </div>

      <details className="card raw">
        <summary>{browser ? "Raw result object (generated in the browser)" : "Raw JSON response"}</summary>
        <pre>{JSON.stringify(stripImages(result), null, 2)}</pre>
      </details>
    </section>
  );
}
