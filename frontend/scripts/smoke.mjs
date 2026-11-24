// Browser smoke test for the static Pages build (run after `npm run build:pages`).
// Serves dist/ under the Pages base path, loads the site in Chromium at desktop and phone widths in
// both themes, and fails on page errors, horizontal overflow or missing sections. It then uploads a
// generated WAV and checks that the result is labelled correctly: real model output when the audio
// model is deployed (public/models/audio), a placeholder otherwise.
import { createServer } from "node:http";
import { existsSync, readFileSync } from "node:fs";
import { extname, join, resolve } from "node:path";
import { chromium } from "playwright";

const BASE = "/multimodal-deepfake-detection/";
const ROOT = resolve("dist");
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml",
  ".json": "application/json", ".png": "image/png", ".wasm": "application/wasm" }; // prettier-ignore

const server = createServer((req, res) => {
  const url = decodeURIComponent(req.url.split("?")[0]);
  if (!url.startsWith(BASE)) return res.writeHead(404).end();
  let file = join(ROOT, url.slice(BASE.length));
  if (!file.startsWith(ROOT)) return res.writeHead(403).end();
  if (url.endsWith("/")) file = join(file, "index.html");
  if (!existsSync(file)) return res.writeHead(404).end();
  res.writeHead(200, { "content-type": TYPES[extname(file)] || "application/octet-stream" }).end(readFileSync(file));
});
await new Promise((r) => server.listen(4179, "127.0.0.1", r));
const URL = `http://127.0.0.1:4179${BASE}`;

function wav(seconds = 3, sr = 16000) {
  const n = seconds * sr;
  const buf = Buffer.alloc(44 + 2 * n);
  buf.write("RIFF", 0); buf.writeUInt32LE(36 + 2 * n, 4); buf.write("WAVEfmt ", 8);
  buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20); buf.writeUInt16LE(1, 22);
  buf.writeUInt32LE(sr, 24); buf.writeUInt32LE(2 * sr, 28); buf.writeUInt16LE(2, 32); buf.writeUInt16LE(16, 34);
  buf.write("data", 36); buf.writeUInt32LE(2 * n, 40);
  for (let i = 0; i < n; i++) {
    const v = 0.3 * Math.sin((2 * Math.PI * 180 * i) / sr) * (1 + 0.5 * Math.sin((2 * Math.PI * 3 * i) / sr));
    buf.writeInt16LE(Math.round(v * 32767), 44 + 2 * i);
  }
  return buf;
} // prettier-ignore

const failures = [];
const browser = await chromium.launch();
for (const [theme, width] of [["dark", 1280], ["light", 390]]) {
  const ctx = await browser.newContext({ viewport: { width, height: 900 }, reducedMotion: "reduce" });
  await ctx.addInitScript((t) => localStorage.setItem("theme", t), theme);
  const page = await ctx.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto(URL, { waitUntil: "load" });
  const tag = `${theme}@${width}`;
  for (const id of ["#demo", "#architecture", "#benchmark", "#api"]) {
    if (!(await page.locator(id).count())) failures.push(`${tag}: missing ${id}`);
  }
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  if (overflow > 1) failures.push(`${tag}: horizontal overflow ${overflow}px`);
  if (theme === "dark") {
    await page.setInputFiles('input[type="file"]', { name: "tone.wav", mimeType: "audio/wav", buffer: wav() });
    await page.waitForSelector(".dashboard, .analysis-error", { timeout: 60000 });
    const text = await page.locator("main").innerText();
    const modelDeployed = existsSync(join(ROOT, "models/audio/audio_cnn.json"));
    if (modelDeployed && !/trained model/.test(text)) failures.push(`${tag}: audio model deployed but not used`);
    if (!modelDeployed && !/placeholder/i.test(text)) failures.push(`${tag}: placeholder audio not labelled`);
    if (/definitely/i.test(text)) failures.push(`${tag}: categorical wording`);
  }
  if (errors.length) failures.push(`${tag}: page errors ${errors.join(" | ")}`);
  await ctx.close();
}
await browser.close();
server.close();
if (failures.length) {
  console.error(failures.join("\n"));
  process.exit(1);
}
console.log("browser smoke test passed");
