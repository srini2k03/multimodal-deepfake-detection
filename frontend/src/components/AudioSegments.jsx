import { riskColor } from "./ProbabilityGauge.jsx";

/** Column chart of P(fake) per fixed-length audio segment. */
export default function AudioSegments({ segments, threshold = 0.5, placeholder = false }) {
  const W = 320;
  const H = 140;
  const pad = 24;
  // Bars are at most 56 units wide and centred, so a single 4 s clip does not fill the whole chart.
  const bw = Math.min((W - 2 * pad) / Math.max(segments.length, 1), 56);
  const x0 = pad + (W - 2 * pad - bw * segments.length) / 2;
  const y = (p) => 8 + (1 - p) * (H - pad - 8);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="audio-svg" role="img" aria-label="Audio segment scores">
      {[0, 0.5, 1].map((v) => (
        <g key={v}>
          <line x1={pad} x2={W - pad} y1={y(v)} y2={y(v)} className="grid" />
          <text x={pad - 4} y={y(v) + 3} textAnchor="end" className="axis-label">
            {v}
          </text>
        </g>
      ))}
      {!placeholder && <line x1={pad} x2={W - pad} y1={y(threshold)} y2={y(threshold)} className="threshold-line" />}
      {segments.map((s, i) => (
        <g key={i}>
          <rect
            x={x0 + i * bw + 2}
            y={y(s.probability)}
            width={Math.max(bw - 4, 2)}
            height={H - pad - y(s.probability)}
            fill={placeholder ? "var(--neutral)" : riskColor(s.probability)}
            rx="3"
          >
            <title>{`${s.start}-${s.end}s: ${(s.probability * 100).toFixed(1)}%`}</title>
          </rect>
          {segments.length <= 12 && (
            <text x={x0 + i * bw + bw / 2} y={H - 8} textAnchor="middle" className="axis-label">
              {s.start}s
            </text>
          )}
        </g>
      ))}
    </svg>
  );
}
