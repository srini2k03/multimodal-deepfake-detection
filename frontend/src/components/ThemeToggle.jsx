import { useEffect, useState } from "react";

const KEY = "dfd-theme";

function initialTheme() {
  const set = document.documentElement.dataset.theme;
  if (set === "light" || set === "dark") return set;
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

/** Light/dark switch; follows the OS setting until the viewer picks one. */
export default function ThemeToggle() {
  const [theme, setTheme] = useState(initialTheme);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  const toggle = () => {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    try {
      localStorage.setItem(KEY, next);
    } catch {
      /* storage unavailable: the choice lasts for this page view only */
    }
  };

  return (
    <button
      type="button"
      className="theme-toggle"
      onClick={toggle}
      aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
      title={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
    >
      {theme === "dark" ? (
        <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
          <circle cx="12" cy="12" r="4.5" fill="currentColor" />
          <g stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
          </g>
        </svg>
      ) : (
        <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
          <path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z" fill="currentColor" />
        </svg>
      )}
    </button>
  );
}
