import ArchitectureDiagram from "./ArchitectureDiagram.jsx";
import BenchmarkResults, { benchmarkData } from "./BenchmarkResults.jsx";
import { ArrowDown, ArrowUpRight, GitHubIcon } from "./Icons.jsx";
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
  [
    "Alignment and crop",
    "Box expanded by a 30% margin and made square; eye-line roll correction; resized to the model input (224² or 299²).",
  ],
  [
    "Frame classifiers",
    "EfficientNetV2-B0 fine-tuned on DeepFakeFace; the Xception + InceptionV3 two-stage ensemble is implemented too.",
  ],
  [
    "Frame to video",
    "Per-frame scores aggregated by mean, median, max or top-k mean; an attention-pooling head is implemented.",
  ],
  [
    "Audio branch",
    "Mono 16 kHz, optional silence trimming, 4 s repeat-padded clips, 64-band log-mel, CNN trained on ASVspoof 2019 LA.",
  ],
  [
    "Calibration and fusion",
    "Platt scaling from the validation split; modalities reported separately; heuristic fusion over those present.",
  ],
  ["Explainability", "Grad-CAM on the last convolutional block of the most suspicious frame, returned as an overlay."],
];

const TRAINED_AUDIO = Boolean(benchmarkData.audio);
const TRAINED_VISUAL = Boolean(benchmarkData.visual);
const MODELS = [
  [
    "EfficientNetV2-B0",
    "ImageNet transfer learning on 224×224 face crops: frozen-feature probe and top-30% fine-tuning on DeepFakeFace.",
    TRAINED_VISUAL ? "frame · trained" : "frame",
  ],
  [
    "Audio CNN",
    "Three conv blocks (16-32-64) on standardised log-mel with SpecAugment, trained on ASVspoof 2019 LA.",
    TRAINED_AUDIO ? "audio · trained" : "audio",
  ],
  [
    "Xception + InceptionV3",
    "Two-stage fine-tuning ensemble. Implemented and tested; not trained, because FaceForensics++ needs an access request.",
    "frame · untrained",
  ],
  [
    "Temporal attention head",
    "TimeDistributed backbone + additive attention, α = softmax(vᵀ tanh(W hₜ)). Needs labelled video; not trained.",
    "video · untrained",
  ],
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

// Facts from the repository: the test count is the number of `def test_` functions in backend/tests
// (pytest collects a few more cases through parametrisation); benchmark numbers come from
// src/data/benchmarks.json (built from committed results by backend/scripts/make_report.py).
const TEST_FUNCTIONS = "96";
const fmtPct = (v) => `${(100 * v).toFixed(1)}%`;
const audioMain = benchmarkData.audio?.systems?.cnn_trim || benchmarkData.audio?.systems?.cnn;
const visualMain = benchmarkData.visual?.systems?.effnet_ft || benchmarkData.visual?.systems?.effnet_probe;
const STATS = [
  audioMain
    ? [fmtPct(audioMain.eval_eer), "Audio eval EER", "ASVspoof 2019 LA, unseen attacks"]
    : ["3", "Modalities", "video, image, audio"],
  visualMain
    ? [visualMain.test_auc.toFixed(3), "Visual test ROC-AUC", "DeepFakeFace, identity-disjoint"]
    : ["2", "Branches", "visual and audio"],
  [TEST_FUNCTIONS, "Test functions", "CI runs with and without TensorFlow"],
  ["3", "Public datasets", "ASVspoof 2019 LA · In-the-Wild · DeepFakeFace"],
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
          Face and voice manipulation detectors trained and evaluated on lawfully available public data, with speaker-
          and identity-disjoint splits, unseen-attack tests, robustness and calibration measurements, a Flask API and
          this dashboard. Every score is a probabilistic model assessment, not a verdict.
        </p>
        <div className="sb-hero__meta">
          <span className={`sb-pill${TRAINED_AUDIO ? "" : " sb-pill--pending"}`}>
            {TRAINED_AUDIO ? "Audio: trained on ASVspoof 2019 LA" : "Audio benchmark pending"}
          </span>
          <span className={`sb-pill${TRAINED_VISUAL ? "" : " sb-pill--pending"}`}>
            {TRAINED_VISUAL ? "Visual: trained on DeepFakeFace" : "Visual benchmark pending"}
          </span>
          <span className="sb-pill sb-pill--pending">Fusion not evaluated</span>
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
          {TRAINED_AUDIO || TRAINED_VISUAL ? (
            <>
              Trained weights are GitHub release assets (not in git); <code>python scripts/fetch_weights.py</code>{" "}
              verifies their checksums and writes the calibration model card for the API.{" "}
            </>
          ) : (
            <>No trained weights are published yet. </>
          )}
          See the{" "}
          <a href={TRAINING_URL} target="_blank" rel="noreferrer">
            training and benchmark commands
          </a>
          .
        </p>
      </Section>

      <Section
        id="benchmark"
        index="05"
        label="Benchmark"
        title={
          <>
            Measured on lawful, <span className="sb-serif">public</span> data.
          </>
        }
        lead="Each branch is trained and scored on its own dataset with leakage-free splits. Thresholds and calibration come from the validation split; test numbers are computed once."
      >
        <BenchmarkResults
          published={
            <details className="sb-card published-ref">
              <summary>Published reference: FaceForensics++ (not reproduced here)</summary>
              <p className="card-intro">
                Published results for the Xception backbone, from Rössler et al.,{" "}
                <i>FaceForensics++: Learning to Detect Manipulated Facial Images</i>, ICCV 2019. These come from the
                paper; this repository did not produce them, because FaceForensics++ requires an access request.
              </p>
              <div
                className="sb-table-wrap table-wrap"
                tabIndex={0}
                role="region"
                aria-label="Published benchmark table"
              >
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
              <p className="sb-card__foot">Source: Rössler et al., ICCV 2019</p>
            </details>
          }
        />
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
