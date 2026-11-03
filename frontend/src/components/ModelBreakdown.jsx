import { riskColor } from "./ProbabilityGauge.jsx";

function Bar({ name, probability, meta, threshold }) {
  return (
    <div className="bar-row">
      <div className="bar-head">
        <span className="bar-name">{name}</span>
        <span className="bar-value">{(probability * 100).toFixed(1)}%</span>
      </div>
      <div className="bar-track">
        <div className="bar-fill" style={{ width: `${probability * 100}%`, background: riskColor(probability) }} />
        <div className="bar-threshold" style={{ left: `${threshold * 100}%` }} />
      </div>
      {meta && <div className="bar-meta">{meta}</div>}
    </div>
  );
}

/** Per-model and per-modality scores, plus fusion settings. */
export default function ModelBreakdown({ result }) {
  const { visual, audio } = result.modalities;
  const t = result.threshold;
  return (
    <div className="breakdown">
      {visual && (
        <>
          <h4>Visual models</h4>
          {Object.entries(visual.per_model).map(([name, m]) => (
            <Bar
              key={name}
              name={name}
              probability={m.probability}
              threshold={t}
              meta={`ensemble weight ${m.weight} · ${m.status.replace("_", " ")}`}
            />
          ))}
          {visual.temporal_probability != null && (
            <Bar name="temporal head" probability={visual.temporal_probability} threshold={t} meta="attention pooling" />
          )}
          <Bar
            name="visual (ensemble)"
            probability={visual.probability}
            threshold={t}
            meta={`aggregation: ${visual.aggregation} over ${visual.faces_detected}/${visual.frames_analyzed} face frames`}
          />
        </>
      )}
      <h4>Audio</h4>
      {audio?.available ? (
        <Bar
          name="audio CNN"
          probability={audio.probability}
          threshold={t}
          meta={`${audio.segments.length} segment(s) · ${audio.duration}s · ${audio.status.replace("_", " ")}`}
        />
      ) : (
        <p className="muted small">{audio?.reason ? `Not used: ${audio.reason}` : "Not applicable for images."}</p>
      )}
      <h4>Late fusion</h4>
      <p className="small">
        {result.fusion.method} &middot; weights visual {result.fusion.weights.visual} / audio{" "}
        {result.fusion.weights.audio} &middot; modalities used: {result.fusion.used.join(", ")}
      </p>
    </div>
  );
}
