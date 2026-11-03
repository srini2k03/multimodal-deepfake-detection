export default function StatusBadge({ health, error }) {
  if (error) return <span className="badge badge-bad">API offline</span>;
  if (!health) return <span className="badge">Connecting...</span>;
  const mode = health.mode || "loading";
  const cls = mode === "trained" ? "badge-ok" : "badge-warn";
  return (
    <span className={`badge ${cls}`} title={health.warning || `API v${health.version}`}>
      API v{health.version} &middot; {mode === "trained" ? "trained models" : `${mode} mode`}
    </span>
  );
}
