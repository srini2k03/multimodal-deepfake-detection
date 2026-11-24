import { riskColor } from "./ProbabilityGauge.jsx";

function Bar({ name, probability, meta, threshold, placeholder }) {
  return (
    <div className="bar-row">
      <div className="bar-head">
        <span className="bar-name">{name}</span>
        <span className="bar-value">{(probability * 100).toFixed(1)}%</span>
      </div>
      <div className="bar-track">
        <div
          className="bar-fill"
          style={{
            width: `${probability * 100}%`,
            background: placeholder ? "var(--neutral)" : riskColor(probability),
          }}
        />
        {!placeholder && <div className="bar-threshold" style={{ left: `${threshold * 100}%` }} />}
      </div>
      {meta && <div className="bar-meta">{meta}</div>}
    </div>
  );
}

/** Per-model and per-modality scores, plus fusion settings. */
export default function ModelBreakdown({ result }) {
  const { visual, audio } = result.modalities;
  const t = result.threshold;
  const placeholder = result.mode === "browser-demo";
  const audioReal = audio?.status === "trained";
  const heuristics = result.browser?.heuristics || {};
  return (
    <div className="breakdown">
      {placeholder && (
        <p className="placeholder-note">
          {visual &&
            "No face classifier runs in the browser demo: the visual bars below are placeholder heuristics in their slots, not model outputs. "}
          {audioReal
            ? "The audio bar is real output of the trained audio CNN."
            : "The audio CNN is not deployed here, so the audio bar is a loudness placeholder."}
        </p>
      )}
      {visual && (
        <>
          <h4>{placeholder ? "Visual slots (placeholder heuristics)" : "Visual models"}</h4>
          {Object.entries(visual.per_model).map(([name, m]) => (
            <Bar
              key={name}
              name={name}
              probability={m.probability}
              threshold={t}
              placeholder={placeholder}
              meta={heuristics[name] || `ensemble weight ${m.weight} · ${m.status.replace("_", " ")}`}
            />
          ))}
          {visual.temporal_probability != null && (
            <Bar
              name="temporal head"
              probability={visual.temporal_probability}
              threshold={t}
              meta="attention pooling"
            />
          )}
          <Bar
            name={placeholder ? "visual (mean of placeholders)" : "visual (ensemble)"}
            probability={visual.probability}
            threshold={t}
            placeholder={placeholder}
            meta={`aggregation: ${visual.aggregation} over ${visual.faces_detected}/${visual.frames_analyzed} face frames${
              visual.calibrated ? " · calibrated (per-model bars are raw)" : ""
            }`}
          />
        </>
      )}
      <h4>Audio</h4>
      {audio?.available ? (
        <Bar
          name={placeholder && !audioReal ? "audio (loudness placeholder)" : "audio CNN"}
          probability={audio.probability}
          threshold={t}
          placeholder={placeholder && !audioReal}
          meta={`${audio.segments.length} segment(s) · ${audio.duration}s · ${audio.status.replace("_", " ")}${
            audio.calibrated ? " · calibrated" : ""
          }${audio.raw_probability != null ? ` · raw ${(100 * audio.raw_probability).toFixed(1)}%` : ""}`}
        />
      ) : (
        <p className="muted small">{audio?.reason ? `Not used: ${audio.reason}` : "Not applicable for images."}</p>
      )}
      <h4>Late fusion</h4>
      <p className="small">
        {result.fusion.method} &middot; weights visual {result.fusion.weights.visual} / audio{" "}
        {result.fusion.weights.audio} &middot; modalities used: {result.fusion.used.join(", ")}
      </p>
      {result.fusion.note && <p className="small muted">{result.fusion.note}</p>}
    </div>
  );
}
