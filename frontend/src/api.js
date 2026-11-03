import { BROWSER_DEMO } from "./demo/config.js";

const API_BASE = (import.meta.env.VITE_API_BASE || "").replace(/\/$/, "");

export const ACCEPTED = {
  image: [".jpg", ".jpeg", ".png", ".bmp", ".webp"],
  video: [".mp4", ".avi", ".mov", ".mkv", ".webm"],
  audio: [".wav", ".flac", ".mp3", ".m4a", ".ogg"],
};
export const ACCEPT_ATTR = Object.values(ACCEPTED).flat().join(",");

export function mediaTypeOf(name) {
  const ext = name.slice(name.lastIndexOf(".")).toLowerCase();
  return Object.keys(ACCEPTED).find((k) => ACCEPTED[k].includes(ext)) || null;
}

async function parseError(res) {
  try {
    const body = await res.json();
    return body?.error?.message || `Request failed (${res.status})`;
  } catch {
    return `Request failed (${res.status})`;
  }
}

export async function getHealth(signal) {
  if (BROWSER_DEMO) return { status: "ok", mode: "browser-demo", version: "static" };
  const res = await fetch(`${API_BASE}/api/health`, { signal });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

/**
 * Upload a file with progress reporting (XMLHttpRequest exposes upload progress; fetch does not).
 * Resolves with the parsed JSON body and HTTP status.
 */
function uploadWithProgress(url, file, onProgress, signal) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url);
    xhr.responseType = "json";
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress?.(e.loaded / e.total);
    };
    xhr.onload = () => {
      const body = xhr.response;
      if (xhr.status >= 200 && xhr.status < 300) resolve({ status: xhr.status, body });
      else reject(new Error(body?.error?.message || `Upload failed (${xhr.status})`));
    };
    xhr.onerror = () => reject(new Error("Network error - is the API running?"));
    xhr.onabort = () => reject(new DOMException("Aborted", "AbortError"));
    signal?.addEventListener("abort", () => xhr.abort());
    const form = new FormData();
    form.append("file", file);
    xhr.send(form);
  });
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/**
 * Run a prediction. Videos use the async job API (upload -> poll) so long analyses don't
 * hit proxy timeouts; images and audio are analysed synchronously.
 */
export async function predict(file, { onUploadProgress, onStage, signal } = {}) {
  if (BROWSER_DEMO) {
    // Static build: run the preprocessing pipeline client-side; no request leaves the browser.
    const { analyseInBrowser } = await import("./demo/browserAnalysis.js");
    return analyseInBrowser(file, { onStage, onProgress: onUploadProgress, signal });
  }
  const useAsync = mediaTypeOf(file.name) === "video";
  const url = `${API_BASE}/api/predict${useAsync ? "?async=1" : ""}`;
  onStage?.("uploading");
  const { status, body } = await uploadWithProgress(url, file, onUploadProgress, signal);
  onStage?.("analysing");
  if (status !== 202) return body;

  const started = Date.now();
  while (Date.now() - started < 10 * 60 * 1000) {
    if (signal?.aborted) throw new DOMException("Aborted", "AbortError");
    await sleep(800);
    const res = await fetch(`${API_BASE}/api/jobs/${body.job_id}`, { signal });
    if (!res.ok) throw new Error(await parseError(res));
    const job = await res.json();
    if (job.status === "done") return job.result;
    if (job.status === "error") throw new Error(job.error || "Analysis failed");
  }
  throw new Error("Timed out waiting for the analysis to finish");
}
