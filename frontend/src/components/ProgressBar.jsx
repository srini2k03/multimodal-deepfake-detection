/** Determinate bar when `value` (0..1) is given, otherwise an indeterminate animation. */
export default function ProgressBar({ label, value }) {
  const determinate = typeof value === "number";
  return (
    <div className="progress">
      <div className="progress-label">
        <span className="progress-text">
          <span className="sb-spinner" aria-hidden="true" />
          <span>{label}</span>
        </span>
        {determinate && <span className="progress-pct">{Math.round(value * 100)}%</span>}
      </div>
      <div
        className="sb-meter progress-track"
        role="progressbar"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={determinate ? Math.round(value * 100) : undefined}
      >
        <span
          className={`progress-fill ${determinate ? "" : "indeterminate"}`}
          style={determinate ? { width: `${value * 100}%` } : undefined}
        />
      </div>
    </div>
  );
}
