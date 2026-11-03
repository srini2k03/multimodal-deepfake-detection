import { riskColor } from "./ProbabilityGauge.jsx";

/** Column chart of P(fake) per fixed-length audio segment. */
export default function AudioSegments({ segments, threshold = 0.5, placeholder = false }) {
  const W = 320;
  const H = 140;
  const pad = 24;
  const bw = (W - 2 * pad) / Math.max(segments.length, 1);
  const y = (p) => 8 + (1 - p) * (H - pad - 8);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="audio-svg" role="img" aria-label="Audio segment scores">
      {!placeholder && <line x1={pad} x2={W - pad} y1={y(threshold)} y2={y(threshold)} className="threshold-line" />}
      {segments.map((s, i) => (
        <g key={i}>
          <rect
            x={pad + i * bw + 2}
            y={y(s.probability)}
            width={Math.max(bw - 4, 2)}
            height={H - pad - y(s.probability)}
            fill={placeholder ? "var(--neutral)" : riskColor(s.probability)}
            rx="3"
          >
            <title>{`${s.start}-${s.end}s: ${(s.probability * 100).toFixed(1)}%`}</title>
          </rect>
          {segments.length <= 12 && (
            <text x={pad + i * bw + bw / 2} y={H - 8} textAnchor="middle" className="axis-label">
              {s.start}s
            </text>
          )}
        </g>
      ))}
    </svg>
  );
}
