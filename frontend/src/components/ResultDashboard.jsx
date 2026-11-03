import ProbabilityGauge from "./ProbabilityGauge.jsx";
import FrameTimeline from "./FrameTimeline.jsx";
import GradCamViewer from "./GradCamViewer.jsx";
import ModelBreakdown from "./ModelBreakdown.jsx";
import AudioSegments from "./AudioSegments.jsx";

function formatBytes(n) {
  if (n == null) return "";
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 ** 2).toFixed(1)} MB`;
}

export default function ResultDashboard({ result }) {
  const visual = result.modalities.visual;
  const audio = result.modalities.audio;
  const frames = visual?.frames || [];

  return (
    <section className="dashboard">
      {result.warning && (
        <div className="banner banner-warn" role="status">
          <strong>Untrained / demo weights.</strong> {result.warning}
        </div>
      )}

      <div className="grid-2">
        <div className="card">
          <h3>Verdict</h3>
          <ProbabilityGauge probability={result.probability_fake} threshold={result.threshold} label={result.label} />
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
            <dd>{result.mode}</dd>
            <dt>Processing</dt>
            <dd>{(result.processing_ms / 1000).toFixed(2)}s</dd>
          </dl>
        </div>

        <div className="card">
          <h3>Per-model breakdown</h3>
          <ModelBreakdown result={result} />
        </div>
      </div>

      {frames.length > 1 && (
        <div className="card">
          <h3>Per-frame timeline</h3>
          <FrameTimeline frames={frames} threshold={result.threshold} />
        </div>
      )}

      <div className="grid-2">
        {visual && (
          <div className="card">
            <h3>Explainability (Grad-CAM)</h3>
            <GradCamViewer gradcam={result.gradcam} />
          </div>
        )}
        {audio?.available && (
          <div className="card">
            <h3>Audio segments</h3>
            <AudioSegments segments={audio.segments} threshold={result.threshold} />
          </div>
        )}
      </div>

      <details className="card raw">
        <summary>Raw JSON response</summary>
        <pre>{JSON.stringify({ ...result, gradcam: result.gradcam ? "[base64 images omitted]" : null }, null, 2)}</pre>
      </details>
    </section>
  );
}
