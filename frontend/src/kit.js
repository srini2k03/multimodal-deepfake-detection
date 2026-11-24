// React port of sb-kit.js (the portfolio's progressive enhancements): cursor spotlight on
// .sb-spotlight cards and reveal-on-scroll for .sb-reveal sections. The theme toggle lives in
// components/ThemeToggle.jsx. Content is never left hidden: reduced motion, a missing
// IntersectionObserver or a 2.5 s timeout all reveal everything.
import { useEffect } from "react";

const reducedMotion = () => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;

export function useKitEffects() {
  useEffect(() => {
    const cleanups = [];

    if (!reducedMotion() && window.matchMedia?.("(pointer: fine)").matches) {
      const onMove = (e) => {
        const el = e.target.closest?.(".sb-spotlight");
        if (!el) return;
        const r = el.getBoundingClientRect();
        el.style.setProperty("--mx", `${e.clientX - r.left}px`);
        el.style.setProperty("--my", `${e.clientY - r.top}px`);
      };
      document.addEventListener("pointermove", onMove, { passive: true });
      cleanups.push(() => document.removeEventListener("pointermove", onMove));
    }

    const items = document.querySelectorAll(".sb-reveal");
    const revealAll = () => items.forEach((el) => el.classList.add("is-in"));
    if (reducedMotion() || !("IntersectionObserver" in window)) {
      revealAll();
    } else {
      const io = new IntersectionObserver(
        (entries) =>
          entries.forEach((en) => {
            if (en.isIntersecting) {
              en.target.classList.add("is-in");
              io.unobserve(en.target);
            }
          }),
        { rootMargin: "0px 0px -8% 0px" },
      );
      items.forEach((el) => io.observe(el));
      const timer = setTimeout(revealAll, 2500);
      cleanups.push(() => {
        io.disconnect();
        clearTimeout(timer);
      });
    }

    return () => cleanups.forEach((fn) => fn());
  }, []);
}
