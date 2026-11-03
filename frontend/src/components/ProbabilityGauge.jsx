const R = 80;
const CX = 100;
const CY = 100;

function polar(fraction) {
  // fraction 0 -> left (180deg), 1 -> right (0deg)
  const angle = Math.PI * (1 - fraction);
  return [CX + R * Math.cos(angle), CY - R * Math.sin(angle)];
}

function arc(from, to) {
  const [x1, y1] = polar(from);
  const [x2, y2] = polar(to);
  return `M ${x1} ${y1} A ${R} ${R} 0 0 1 ${x2} ${y2}`;
}

export function riskColor(p) {
  if (p >= 0.7) return "var(--fake)";
  if (p >= 0.4) return "var(--warn)";
  return "var(--real)";
}

/**
 * Semicircular gauge for P(fake) with the decision threshold marked.
 * `placeholder` renders a neutral colour, no verdict and a "placeholder score" caption.
 */
export default function ProbabilityGauge({ probability, threshold = 0.5, label, placeholder = false }) {
  const p = Math.min(1, Math.max(0, probability ?? 0));
  const [nx, ny] = polar(p);
  const [tx1, ty1] = polar(threshold);
  const color = placeholder ? "var(--neutral)" : riskColor(p);
  return (
    <figure className="gauge">
      <svg
        viewBox="0 0 200 120"
        role="img"
        aria-label={
          placeholder
            ? `Placeholder score ${(p * 100).toFixed(1)} percent, not a manipulation probability`
            : `Manipulation probability ${(p * 100).toFixed(1)} percent`
        }
      >
        <path d={arc(0, 1)} className="gauge-track" />
        {p > 0.001 && <path d={arc(0, p)} stroke={color} className="gauge-value" />}
        {!placeholder && <line x1={CX} y1={CY} x2={tx1} y2={ty1} className="gauge-threshold" />}
        <circle cx={nx} cy={ny} r="6" fill={color} stroke="var(--surface)" strokeWidth="2" />
        <text x={CX} y={CY - 18} textAnchor="middle" className="gauge-number">
          {(p * 100).toFixed(1)}%
        </text>
        <text x={CX} y={CY + 2} textAnchor="middle" className="gauge-caption">
          {placeholder ? "placeholder score" : "P(manipulated)"}
        </text>
        <text x={CX - R} y={CY + 16} textAnchor="middle" className="gauge-tick">
          0
        </text>
        <text x={CX + R} y={CY + 16} textAnchor="middle" className="gauge-tick">
          1
        </text>
      </svg>
      {placeholder && <figcaption className="verdict verdict-none">No verdict: placeholder score only</figcaption>}
      {!placeholder && label && (
        <figcaption className={`verdict ${label === "FAKE" ? "verdict-fake" : "verdict-real"}`}>
          {label === "FAKE" ? "Likely manipulated" : "Likely authentic"}
        </figcaption>
      )}
    </figure>
  );
}
