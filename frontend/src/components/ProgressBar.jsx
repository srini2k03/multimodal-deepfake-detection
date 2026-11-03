/** Determinate bar when `value` (0..1) is given, otherwise an indeterminate animation. */
export default function ProgressBar({ label, value }) {
  const determinate = typeof value === "number";
  return (
    <div className="progress">
      <div className="progress-label">
        <span>{label}</span>
        {determinate && <span>{Math.round(value * 100)}%</span>}
      </div>
      <div
        className="progress-track"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={determinate ? Math.round(value * 100) : undefined}
      >
        <div
          className={`progress-fill ${determinate ? "" : "indeterminate"}`}
          style={determinate ? { width: `${value * 100}%` } : undefined}
        />
      </div>
    </div>
  );
}
