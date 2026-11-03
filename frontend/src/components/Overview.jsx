import ArchitectureDiagram from "./ArchitectureDiagram.jsx";
import { REPO_URL, SETUP_URL, TRAINING_URL } from "../demo/config.js";

const STEPS = [
  ["Frame sampling", "N frames spread evenly across the video (centre of each of N segments), decoded in one sequential pass."],
  ["Face detection", "OpenCV ResNet-10 SSD with a Haar-cascade fallback. The largest face in each frame is kept."],
  ["Alignment and crop", "Box expanded by a 30% margin and made square; eye-line roll correction; resized to 299×299."],
  ["Frame classifiers", "Xception and InceptionV3 fine-tuned in two stages: frozen-backbone head, then partial unfreeze."],
  ["Ensemble and temporal", "Weighted per-frame ensemble, then mean / median / top-k aggregation or an attention-pooling head."],
  ["Audio branch", "Mono 16 kHz audio split into 4 s clips, 64-band log-mel spectrograms scored by a compact CNN."],
  ["Late fusion", "Weighted average in probability or logit space over the modalities that are present."],
  ["Explainability", "Grad-CAM on the last convolutional block of the most suspicious frame, returned as an overlay."],
];

const MODELS = [
  ["Xception", "ImageNet transfer learning, 299×299 face crops, single-logit head", "frame"],
  ["InceptionV3", "ImageNet transfer learning, same input pipeline, weighted ensemble with Xception", "frame"],
  ["Temporal attention head", "TimeDistributed backbone + additive attention, α = softmax(vᵀ tanh(W hₜ))", "video"],
  ["Audio CNN", "3-block CNN on standardised log-mel spectrograms with SpecAugment-style masking", "audio"],
];

const BENCHMARK = [
  ["FaceForensics++ (all four manipulations)", "Raw", "XceptionNet (face crops)", "99.26%"],
  ["FaceForensics++ (all four manipulations)", "c23 (HQ)", "XceptionNet (face crops)", "95.73%"],
  ["FaceForensics++ (all four manipulations)", "c40 (LQ)", "XceptionNet (face crops)", "81.00%"],
];

const ENDPOINTS = [
  ["GET", "/api/health", "Liveness, version and model mode (trained / partial / demo)"],
  ["GET", "/api/models", "Loaded models and their status, ensemble weights and fusion settings"],
  ["POST", "/api/predict", "multipart/form-data with a file field; ?async=1 returns 202 {job_id, status_url}"],
  ["GET", "/api/jobs/<job_id>", "{status: queued | running | done | error, result, error}"],
];

export function Hero() {
  return (
    <section className="hero" id="top">
      <p className="eyebrow">Xception + InceptionV3 · temporal attention · audio CNN · late fusion</p>
      <h1>Multi-Modal Deepfake Detection</h1>
      <p className="hero-lead">
        An end-to-end system for detecting manipulated media: OpenCV face preprocessing, a transfer-learning ensemble,
        temporal and audio branches, a Flask REST API and this React dashboard.
      </p>
      <div className="hero-actions">
        <a className="btn-primary" href="#demo">
          Try the browser demo
        </a>
        <a className="btn-secondary" href={REPO_URL} target="_blank" rel="noreferrer">
          View on GitHub
        </a>
        <a className="btn-secondary" href={SETUP_URL} target="_blank" rel="noreferrer">
          Run it locally
        </a>
      </div>
    </section>
  );
}

export default function Overview() {
  return (
    <>
      <section className="section" id="architecture">
        <h2>Architecture</h2>
        <div className="card">
          <ArchitectureDiagram />
        </div>
      </section>

      <section className="section" id="pipeline">
        <h2>Pipeline</h2>
        <ol className="steps">
          {STEPS.map(([title, body], i) => (
            <li key={title} className="step">
              <span className="step-num">{i + 1}</span>
              <div>
                <h3>{title}</h3>
                <p className="small muted">{body}</p>
              </div>
            </li>
          ))}
        </ol>
      </section>

      <section className="section" id="models">
        <h2>Models</h2>
        <div className="model-grid">
          {MODELS.map(([name, body, level]) => (
            <div key={name} className="card model-card">
              <div className="model-head">
                <h3>{name}</h3>
                <span className="chip">{level}</span>
              </div>
              <p className="small muted">{body}</p>
            </div>
          ))}
        </div>
        <p className="small muted">
          No trained weights are published with this project. Train them with the{" "}
          <a href={TRAINING_URL} target="_blank" rel="noreferrer">
            training commands
          </a>{" "}
          to get real scores from the API.
        </p>
      </section>

      <section className="section" id="benchmark">
        <h2>Reference benchmark (published)</h2>
        <div className="card">
          <p className="small">
            Published results for the Xception backbone used in this project, from the FaceForensics++ benchmark
            (Rössler et al., <i>FaceForensics++: Learning to Detect Manipulated Facial Images</i>, ICCV 2019).{" "}
            <strong>Published, not reproduced here:</strong> these come from the paper. This repository did not produce
            them, and the browser demo on this page does not run Xception.
          </p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Dataset</th>
                  <th>Compression</th>
                  <th>Model</th>
                  <th className="num">Accuracy</th>
                </tr>
              </thead>
              <tbody>
                {BENCHMARK.map((r) => (
                  <tr key={r[1]}>
                    <td>{r[0]}</td>
                    <td>{r[1]}</td>
                    <td>{r[2]}</td>
                    <td className="num">{r[3]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section className="section" id="api">
        <h2>REST API</h2>
        <div className="card">
          <p className="small">
            Flask API at <code>http://localhost:5000</code>. Errors have the shape{" "}
            <code>{'{"error": {"status", "message", "request_id"}}'}</code> and every response carries an{" "}
            <code>X-Request-ID</code> header.
          </p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Method</th>
                  <th>Path</th>
                  <th>Description</th>
                </tr>
              </thead>
              <tbody>
                {ENDPOINTS.map(([m, p, d]) => (
                  <tr key={p}>
                    <td>
                      <span className={`method method-${m.toLowerCase()}`}>{m}</span>
                    </td>
                    <td>
                      <code>{p}</code>
                    </td>
                    <td>{d}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="small muted">
            Status codes: 400 no file · 413 too large or too long · 415 unsupported type · 422 undecodable · 500
            server error. Accepted: images .jpg .jpeg .png .bmp .webp, videos .mp4 .avi .mov .mkv .webm, audio .wav
            .flac .mp3 .m4a .ogg.
          </p>
          <pre className="code">curl -F file=@clip.mp4 http://localhost:5000/api/predict</pre>
        </div>
      </section>
    </>
  );
}
