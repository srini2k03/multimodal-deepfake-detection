import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// During development requests to /api are proxied to the Flask backend.
const apiTarget = process.env.VITE_API_PROXY || "http://localhost:5000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": { target: apiTarget, changeOrigin: true } },
  },
  build: { outDir: "dist", sourcemap: false },
});
