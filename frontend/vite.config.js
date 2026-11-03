import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// During development requests to /api are proxied to the Flask backend.
const apiTarget = process.env.VITE_API_PROXY || "http://localhost:5000";

export default defineConfig(({ mode }) => {
  // `vite build --mode pages` reads .env.pages: VITE_DEMO=browser and the GitHub Pages base path.
  const env = { ...loadEnv(mode, process.cwd(), "VITE_"), ...process.env };
  return {
    base: env.VITE_BASE || "/",
    plugins: [react()],
    server: {
      port: 5173,
      proxy: { "/api": { target: apiTarget, changeOrigin: true } },
    },
    build: { outDir: "dist", sourcemap: false },
  };
});
