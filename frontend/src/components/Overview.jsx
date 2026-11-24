import ArchitectureDiagram from "./ArchitectureDiagram.jsx";
import { AlertIcon, ArrowDown, ArrowUpRight, GitHubIcon } from "./Icons.jsx";
import { REPO_URL, SETUP_URL, TRAINING_URL } from "../demo/config.js";

const STEPS = [
  [
    "Frame sampling",
    "N frames spread evenly across the video (centre of each of N segments), decoded in one sequential pass.",
  ],
  [
    "Face detection",
    "OpenCV Haar cascade by default; the ResNet-10 SSD is opt-in, with Haar as its fallback. The largest face in each frame is kept.",
  ],
  ["Alignment and crop", "Box expanded by a 30% margin and made square; eye-line roll correction; resized to 299×299."],
  [
    "Frame classifiers",
    "Xception and InceptionV3 fine-tuned in two stages: frozen-backbone head, then partial unfreeze.",
  ],
  [
    "Ensemble and temporal",
    "Weighted per-frame ensemble, then mean / median / top-k aggregation or an attention-pooling head.",
  ],
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

// Facts taken from the repository: modalities and backbones from the model code, the test count from
// backend/tests (61 `def test_` functions; pytest collects more cases through parametrisation) and the
// crop size from configs/default.yaml (face.crop_size).
const STATS = [
  ["3", "Modalities", "video, image, audio"],
  ["2", "CNN backbones ensembled", "Xception + InceptionV3"],
  ["61", "Test functions", "CI runs with and without TensorFlow"],
  ["299²", "Aligned face crops", "eye-based roll correction"],
];

/** Kicker (index, rule, label), heading and optional lead, as on the portfolio's case studies. */
export function SectionHead({ id, index, label, title, aside, children }) {
  return (
    <div className="section-head">
      <div className="section-head__main">
        <p className="sb-kicker">
          <span className="sb-kicker__index">{index}</span>
          <span className="sb-kicker__rule" aria-hidden="true" />
          {label}
        </p>
        <h2 className="sb-h2" id={`${id}-title`}>
          {title}
        </h2>
        {children && <p className="sb-lead">{children}</p>}
      </div>
      {aside && <div className="section-head__aside">{aside}</div>}
    </div>
  );
}

function Section({ id, index, label, title, lead, children }) {
  return (
    <section className="section-block" id={id} aria-labelledby={`${id}-title`}>
      <div className="sb-container">
        <div className="section-inner">
          <div className="sb-reveal">
            <SectionHead id={id} index={index} label={label} title={title}>
              {lead}
            </SectionHead>
          </div>
          <div className="sb-reveal">{children}</div>
        </div>
      </div>
    </section>
  );
}

export function Hero() {
  return (
    <section className="sb-hero hero" id="top" aria-labelledby="hero-title">
      <div className="sb-container">
        <p className="sb-kicker">
          <span className="sb-kicker__index">Deep learning</span>
          <span className="sb-kicker__rule" aria-hidden="true" />
          Multi-modal · 2025
        </p>
        <h1 className="sb-h1" id="hero-title">
          Multi-Modal Deepfake <span className="sb-serif">Detection</span>
        </h1>
        <p className="sb-lead hero-lead">
          An end-to-end system for detecting manipulated media: OpenCV face preprocessing, a transfer-learning ensemble,
          temporal and audio branches, a Flask REST API and this React dashboard.
        </p>
        <div className="sb-hero__meta">
          <span className="sb-pill">Pipeline &amp; serving complete</span>
          <span className="sb-pill sb-pill--pending">Benchmark training pending</span>
          <span className="sb-pill sb-pill--plain">Face preprocessing runs in your browser</span>
        </div>
        <div className="sb-hero__actions">
          <a className="sb-btn sb-btn--primary btn-primary" href="#demo">
            Try the browser demo <ArrowDown />
          </a>
          <a className="sb-btn btn-secondary" href={REPO_URL} target="_blank" rel="noreferrer">
            <GitHubIcon size={15} /> View on GitHub
          </a>
          <a className="sb-btn sb-btn--ghost btn-secondary" href={SETUP_URL} target="_blank" rel="noreferrer">
            Run it locally <ArrowUpRight />
          </a>
        </div>
        <dl className="sb-stats hero-stats">
          {STATS.map(([value, label, ctx]) => (
            <div className="sb-stat" key={label}>
              <dt className="sb-stat__label">{label}</dt>
              <dd className="sb-stat__value">{value}</dd>
              <dd className="sb-stat__ctx">{ctx}</dd>
            </div>
          ))}
        </dl>
      </div>
    </section>
  );
}

export default function Overview() {
  return (
    <>
      <Section
        id="architecture"
        index="02"
        label="Architecture"
        title={
          <>
            One API, three <span className="sb-serif">modalities</span>.
          </>
        }
        lead="The Flask API validates the upload, then the inference engine runs the visual and audio branches and fuses whatever modalities the file contains."
      >
        <div className="sb-card arch-card">
          <ArchitectureDiagram />
        </div>
      </Section>

      <Section
        id="pipeline"
        index="03"
        label="Pipeline"
        title={
          <>
            From upload to <span className="sb-serif">late fusion</span>.
          </>
        }
      >
        <ol className="steps">
          {STEPS.map(([title, body], i) => (
            <li key={title} className="sb-card sb-spotlight step">
              <span className="step-num">Stage {String(i + 1).padStart(2, "0")}</span>
              <h3 className="sb-h3">{title}</h3>
              <p>{body}</p>
            </li>
          ))}
        </ol>
      </Section>

      <Section
        id="models"
        index="04"
        label="Models"
        title={
          <>
            Four networks, two <span className="sb-serif">branches</span>.
          </>
        }
      >
        <div className="model-grid">
          {MODELS.map(([name, body, level]) => (
            <div key={name} className="sb-card sb-spotlight model-card">
              <div className="model-head">
                <h3 className="sb-h3">{name}</h3>
                <span className="sb-tag">{level}</span>
              </div>
              <p>{body}</p>
            </div>
          ))}
        </div>
        <p className="sb-small models-note">
          No trained weights are published with this project. Train them with the{" "}
          <a href={TRAINING_URL} target="_blank" rel="noreferrer">
            training commands
          </a>{" "}
          to get real scores from the API.
        </p>
      </Section>

      <Section
        id="benchmark"
        index="05"
        label="Reference benchmark (published)"
        title={
          <>
            Published results, <span className="sb-serif">not reproduced</span> here.
          </>
        }
      >
        <div className="sb-card benchmark-card">
          <p className="card-intro">
            Published results for the Xception backbone used in this project, from the FaceForensics++ benchmark
            (Rössler et al., <i>FaceForensics++: Learning to Detect Manipulated Facial Images</i>, ICCV 2019).{" "}
            <strong>Published, not reproduced here:</strong> these come from the paper. This repository did not produce
            them, and the browser demo on this page does not run Xception.
          </p>
          <div className="sb-table-wrap table-wrap" tabIndex={0} role="region" aria-label="Published benchmark table">
            <table className="sb-table stack-sm">
              <thead>
                <tr>
                  <th scope="col">Dataset</th>
                  <th scope="col">Compression</th>
                  <th scope="col">Model</th>
                  <th scope="col" className="num">
                    Accuracy
                  </th>
                </tr>
              </thead>
              <tbody>
                {BENCHMARK.map((r) => (
                  <tr key={r[1]}>
                    <td data-label="Dataset">{r[0]}</td>
                    <td data-label="Compression">{r[1]}</td>
                    <td data-label="Model">{r[2]}</td>
                    <td data-label="Accuracy" className="num">
                      {r[3]}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="sb-callout sb-callout--warn repo-results">
            <span className="callout-icon" aria-hidden="true">
              <AlertIcon size={16} />
            </span>
            <p>
              <strong>This repository&apos;s own results: TBD.</strong> No model has been trained yet, so the
              README&apos;s results table (AUC, accuracy, F1, EER on FaceForensics++ and Celeb-DF) stays empty until
              training runs.
            </p>
          </div>
          <p className="sb-card__foot">Source: Rössler et al., ICCV 2019 · README › Reference benchmark (published)</p>
        </div>
      </Section>

      <Section
        id="api"
        index="06"
        label="REST API"
        title={
          <>
            Four endpoints, one <span className="sb-serif">JSON</span> shape.
          </>
        }
      >
        <div className="sb-card api-card">
          <p className="card-intro">
            Flask API at <code>http://localhost:5000</code>. Errors have the shape{" "}
            <code>{'{"error": {"status", "message", "request_id"}}'}</code> and every response carries an{" "}
            <code>X-Request-ID</code> header.
          </p>
          <div className="sb-table-wrap table-wrap" tabIndex={0} role="region" aria-label="API endpoints">
            <table className="sb-table stack-sm">
              <thead>
                <tr>
                  <th scope="col">Method</th>
                  <th scope="col">Path</th>
                  <th scope="col">Description</th>
                </tr>
              </thead>
              <tbody>
                {ENDPOINTS.map(([m, p, d]) => (
                  <tr key={p}>
                    <td data-label="Method">
                      <span className={`method method-${m.toLowerCase()}`}>{m}</span>
                    </td>
                    <td data-label="Path">
                      <code>{p}</code>
                    </td>
                    <td data-label="Description">{d}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="sb-small">
            Status codes: 400 no file · 413 too large or too long · 415 unsupported type · 422 undecodable · 500 server
            error. Accepted: images .jpg .jpeg .png .bmp .webp, videos .mp4 .avi .mov .mkv .webm, audio .wav .flac .mp3
            .m4a .ogg.
          </p>
          <pre className="sb-code code">
            <span className="k">curl</span> -F file=@clip.mp4{" "}
            <span className="s">http://localhost:5000/api/predict</span>
          </pre>
        </div>
      </Section>
    </>
  );
}
