import { useMemo, useState } from "react";

const W = 640;
const H = 220;
const PAD = { l: 40, r: 16, t: 14, b: 32 };
const MODEL_COLORS = ["var(--series-2)", "var(--series-3)", "var(--series-4)"];

/** Per-frame P(fake) over time: ensemble line, faint per-model lines, threshold and hover readout. */
export default function FrameTimeline({ frames, threshold = 0.5, showThreshold = true }) {
  const [hover, setHover] = useState(null);
  const [showModels, setShowModels] = useState(true);

  const models = useMemo(() => Object.keys(frames[0]?.per_model || {}), [frames]);
  const tMax = Math.max(...frames.map((f) => f.timestamp), 1e-6);
  const x = (t) => PAD.l + (t / tMax) * (W - PAD.l - PAD.r);
  const y = (p) => PAD.t + (1 - p) * (H - PAD.t - PAD.b);
  const path = (get) => frames.map((f, i) => `${i ? "L" : "M"} ${x(f.timestamp).toFixed(1)} ${y(get(f)).toFixed(1)}`).join(" ");

  const onMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    let best = 0;
    frames.forEach((f, i) => {
      if (Math.abs(x(f.timestamp) - px) < Math.abs(x(frames[best].timestamp) - px)) best = i;
    });
    setHover(best);
  };

  const h = hover !== null ? frames[hover] : null;

  return (
    <div className="timeline">
      <div className="legend">
        <span className="legend-item">
          <i style={{ background: "var(--series-1)" }} /> {showThreshold ? "ensemble" : "mean of placeholders"}
        </span>
        {showModels &&
          models.map((m, i) => (
            <span className="legend-item" key={m}>
              <i style={{ background: MODEL_COLORS[i % MODEL_COLORS.length] }} /> {m}
            </span>
          ))}
        <span className="legend-item">
          <i className="hollow" /> no face found
        </span>
        <label className="toggle">
          <input type="checkbox" checked={showModels} onChange={(e) => setShowModels(e.target.checked)} /> per-model
        </label>
      </div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="timeline-svg"
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
        role="img"
        aria-label="Per-frame manipulation probability timeline"
      >
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <g key={v}>
            <line x1={PAD.l} x2={W - PAD.r} y1={y(v)} y2={y(v)} className="grid" />
            <text x={PAD.l - 6} y={y(v) + 4} textAnchor="end" className="axis-label">
              {v.toFixed(2)}
            </text>
          </g>
        ))}
        {showThreshold && (
          <>
            <line x1={PAD.l} x2={W - PAD.r} y1={y(threshold)} y2={y(threshold)} className="threshold-line" />
            <text x={W - PAD.r} y={y(threshold) - 4} textAnchor="end" className="axis-label">
              threshold
            </text>
          </>
        )}
        {showModels &&
          models.map((m, i) => (
            <path
              key={m}
              d={path((f) => f.per_model[m])}
              fill="none"
              stroke={MODEL_COLORS[i % MODEL_COLORS.length]}
              strokeWidth="1.5"
              opacity="0.55"
            />
          ))}
        <path d={path((f) => f.probability)} fill="none" stroke="var(--series-1)" strokeWidth="2.5" />
        {frames.map((f, i) => (
          <circle
            key={f.index}
            cx={x(f.timestamp)}
            cy={y(f.probability)}
            r={hover === i ? 5.5 : 3.5}
            fill={f.face_found ? "var(--series-1)" : "var(--surface)"}
            stroke="var(--series-1)"
            strokeWidth="1.5"
          />
        ))}
        {h && <line x1={x(h.timestamp)} x2={x(h.timestamp)} y1={PAD.t} y2={H - PAD.b} className="hover-line" />}
        <text x={(W + PAD.l) / 2} y={H - 6} textAnchor="middle" className="axis-label">
          time (s)
        </text>
        <text x={PAD.l} y={H - 18} className="axis-label">
          0
        </text>
        <text x={W - PAD.r} y={H - 18} textAnchor="end" className="axis-label">
          {tMax.toFixed(1)}
        </text>
      </svg>
      <div className="timeline-readout" aria-live="polite">
        {h ? (
          <>
            Frame {h.index} @ {h.timestamp.toFixed(2)}s &middot; {showThreshold ? "ensemble" : "mean placeholder"} <b>{(h.probability * 100).toFixed(1)}%</b>
            {models.map((m) => (
              <span key={m}>
                {" "}
                &middot; {m} {(h.per_model[m] * 100).toFixed(1)}%
              </span>
            ))}
            {!h.face_found && <span className="muted"> &middot; no face (centre crop)</span>}
          </>
        ) : (
          <span className="muted">Hover the chart to inspect individual frames.</span>
        )}
      </div>
    </div>
  );
}
