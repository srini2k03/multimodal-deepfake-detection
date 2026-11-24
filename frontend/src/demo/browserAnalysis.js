/**
 * Client-side version of the preprocessing pipeline for the static browser demo.
 *
 * What is real: frame sampling (same "centre of N segments" rule as the backend), face
 * detection (MediaPipe BlazeFace), eye-based roll alignment, square margin crops at 299x299,
 * centre-crop fallback, audio decoding and 4 s segmentation, and late-fusion arithmetic.
 *
 * What is NOT real: the visual scores. No face classifier runs here; each visual "score" is a
 * squashed pixel statistic, used only so the dashboard has numbers to plot.
 *
 * Audio: when the exported audio CNN is deployed (public/models/audio, see AUDIO_MODEL_URL) the
 * audio score is real model output computed in the browser; otherwise it is a loudness placeholder.
 * A real audio score is never averaged with placeholder visual scores.
 */
import { mediaTypeOf } from "../api.js";
import { calibrate, loadAudioModel, scoreWaveformAsync, sigmoid as sig } from "./audioModel.js";
import { AUDIO_MODEL_NOTE, AUDIO_MODEL_URL, BROWSER_AUDIO_MAX_SECONDS, BROWSER_WARNING, PIPELINE } from "./config.js";
import { DETECTOR_NAME, detectFaces } from "./faceDetector.js";

export const HEURISTICS = {
  sharpness_heuristic: "Placeholder A: crop sharpness (variance of the Laplacian), squashed to 0..1",
  saturation_heuristic: "Placeholder B: mean colour saturation of the crop, squashed to 0..1",
  loudness_heuristic: "Placeholder: RMS loudness of each 4 s audio segment, squashed to 0..1",
};

const sigmoid = (z) => 1 / (1 + Math.exp(-z));
const placeholder = (z) => Math.round(Math.min(0.98, Math.max(0.02, sigmoid(z))) * 1e5) / 1e5;
const round = (v, d = 3) => Math.round(v * 10 ** d) / 10 ** d;
const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);

function abortable(signal) {
  if (signal?.aborted) throw new DOMException("Aborted", "AbortError");
}

function makeCanvas(w, h) {
  const c = document.createElement("canvas");
  c.width = Math.max(1, Math.round(w));
  c.height = Math.max(1, Math.round(h));
  return c;
}

/** Copy a drawable source to a canvas whose longest side is at most `maxSide`. */
function toCanvas(source, srcW, srcH, maxSide) {
  const s = Math.min(1, maxSide / Math.max(srcW, srcH));
  const c = makeCanvas(srcW * s, srcH * s);
  c.getContext("2d").drawImage(source, 0, 0, c.width, c.height);
  return c;
}

/** Square, margin-expanded box shifted back inside the frame (mirrors alignment.square_box). */
function squareBox(face, W, H) {
  const cx = face.x + face.w / 2;
  const cy = face.y + face.h / 2;
  const side = Math.min(Math.max(face.w, face.h) * (1 + 2 * PIPELINE.margin), Math.max(W, H));
  let x1 = cx - side / 2;
  let y1 = cy - side / 2;
  let x2 = cx + side / 2;
  let y2 = cy + side / 2;
  if (x1 < 0) [x2, x1] = [x2 - x1, 0];
  if (y1 < 0) [y2, y1] = [y2 - y1, 0];
  if (x2 > W) [x1, x2] = [Math.max(0, x1 - (x2 - W)), W];
  if (y2 > H) [y1, y2] = [Math.max(0, y1 - (y2 - H)), H];
  return { x1, y1, x2, y2, cx, cy };
}

/** Roll angle from the two eye keypoints, with the same sanity limits as alignment.estimate_roll. */
function estimateRoll(face) {
  if (!face.eyes) return null;
  const [a, b] = [...face.eyes].sort((p, q) => p.x - q.x);
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  if (dx < face.w * 0.15) return null;
  const angle = (Math.atan2(dy, dx) * 180) / Math.PI;
  return Math.abs(angle) <= 30 ? angle : null;
}

/** Rotate about the face centre so the eye line is level, then crop and resize to 299x299. */
function alignAndCrop(src, face) {
  const S = PIPELINE.cropSize;
  const out = makeCanvas(S, S);
  const ctx = out.getContext("2d", { willReadFrequently: true });
  const angle = estimateRoll(face);
  const { x1, y1, x2, y2, cx, cy } = squareBox(face, src.width, src.height);
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, S, S);
  ctx.scale(S / (x2 - x1), S / (y2 - y1));
  ctx.translate(-x1, -y1);
  if (angle !== null && Math.abs(angle) > 1) {
    // OpenCV's positive angle is counter-clockwise on screen; canvas rotate() is clockwise.
    ctx.translate(cx, cy);
    ctx.rotate((-angle * Math.PI) / 180);
    ctx.translate(-cx, -cy);
  }
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(src, 0, 0);
  return { canvas: out, angle: angle ?? 0, aligned: angle !== null, box: { x1, y1, x2, y2 } };
}

/** Fallback when no face is found: central square crop (mirrors alignment.center_crop). */
function centreCrop(src) {
  const S = PIPELINE.cropSize;
  const out = makeCanvas(S, S);
  const side = Math.min(src.width, src.height);
  const sx = (src.width - side) / 2;
  const sy = (src.height - side) / 2;
  out.getContext("2d", { willReadFrequently: true }).drawImage(src, sx, sy, side, side, 0, 0, S, S);
  return { canvas: out, angle: 0, aligned: false, box: null };
}

/** Placeholder scores from simple pixel statistics of the crop. Not a classifier. */
function heuristicScores(crop) {
  const { data, width, height } = crop
    .getContext("2d", { willReadFrequently: true })
    .getImageData(0, 0, crop.width, crop.height);
  const gray = new Float32Array(width * height);
  let sat = 0;
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    gray[p] = 0.299 * r + 0.587 * g + 0.114 * b;
    const mx = Math.max(r, g, b);
    const mn = Math.min(r, g, b);
    sat += mx === 0 ? 0 : (mx - mn) / mx;
  }
  let sum = 0;
  let sumSq = 0;
  let n = 0;
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x;
      const lap = gray[i - 1] + gray[i + 1] + gray[i - width] + gray[i + width] - 4 * gray[i];
      sum += lap;
      sumSq += lap * lap;
      n++;
    }
  }
  const lapVar = sumSq / n - (sum / n) ** 2;
  const meanSat = sat / (width * height);
  return {
    sharpness_heuristic: placeholder(Math.log1p(Math.max(lapVar, 0)) - 5),
    saturation_heuristic: placeholder((meanSat - 0.3) / 0.12),
  };
}

/** Downscaled copy of the frame with every detected face box, eye keypoints and crop region drawn on (kit accent colours). */
function annotate(src, faces, crop, maxSide) {
  const c = toCanvas(src, src.width, src.height, maxSide);
  const s = c.width / src.width;
  const ctx = c.getContext("2d");
  const lw = Math.max(1.5, c.width / 220);
  ctx.font = `600 ${Math.max(10, Math.round(c.width / 32))}px system-ui, sans-serif`;
  faces.forEach((f, i) => {
    const primary = i === 0;
    ctx.strokeStyle = primary ? "#5eeaa8" : "#ffb547";
    ctx.lineWidth = primary ? lw * 1.4 : lw;
    ctx.setLineDash([]);
    ctx.strokeRect(f.x * s, f.y * s, f.w * s, f.h * s);
    const label = `face ${(f.score * 100).toFixed(0)}%`;
    const tw = ctx.measureText(label).width + 8;
    const th = parseInt(ctx.font.match(/(\d+)px/)[1], 10) + 6;
    const ty = f.y * s - th >= 0 ? f.y * s - th : f.y * s;
    ctx.fillStyle = ctx.strokeStyle;
    ctx.fillRect(f.x * s - lw / 2, ty, tw, th);
    ctx.fillStyle = "#04140c";
    ctx.fillText(label, f.x * s + 4, ty + th - 5);
    if (primary && f.eyes) {
      ctx.fillStyle = "#5eeaa8";
      f.eyes.forEach((e) => {
        ctx.beginPath();
        ctx.arc(e.x * s, e.y * s, lw * 1.6, 0, Math.PI * 2);
        ctx.fill();
      });
    }
  });
  if (crop.box) {
    ctx.strokeStyle = "#eef1f5";
    ctx.lineWidth = lw;
    ctx.setLineDash([lw * 3, lw * 2]);
    const { x1, y1, x2, y2 } = crop.box;
    ctx.strokeRect(x1 * s, y1 * s, (x2 - x1) * s, (y2 - y1) * s);
  }
  return c.toDataURL("image/jpeg", 0.85);
}

function once(target, ok, timeoutMs = 15000) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => done(reject, new Error(`Timed out waiting for the video (${ok})`)), timeoutMs);
    const onOk = () => done(resolve);
    const onErr = () => done(reject, new Error("UNDECODABLE"));
    function done(fn, arg) {
      clearTimeout(timer);
      target.removeEventListener(ok, onOk);
      target.removeEventListener("error", onErr);
      fn(arg);
    }
    target.addEventListener(ok, onOk);
    target.addEventListener("error", onErr);
  });
}

async function seek(video, t) {
  const p = once(video, "seeked");
  video.currentTime = t;
  await p;
}

/** Uniformly sample `n` frames (centre of each of n equal time segments) with a <video> element. */
async function sampleVideo(file, n, onProgress, signal) {
  const url = URL.createObjectURL(file);
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.preload = "auto";
  try {
    const meta = once(video, "loadeddata");
    video.src = url;
    try {
      await meta;
    } catch (e) {
      if (e.message !== "UNDECODABLE") throw e;
      throw new Error(
        "Your browser cannot decode this video container or codec. In the browser demo use MP4 (H.264) or WebM " +
          "(VP8/VP9); the Python backend decodes AVI and MKV with OpenCV.",
      );
    }
    let duration = video.duration;
    if (!Number.isFinite(duration)) {
      // Some WebM files (e.g. from MediaRecorder) only report their length after a seek to the end.
      await seek(video, 1e7);
      duration = video.duration;
    }
    if (!Number.isFinite(duration) || duration <= 0 || !video.videoWidth) {
      throw new Error("Could not read the video duration or frame size.");
    }
    if (duration > PIPELINE.maxVideoSeconds) {
      throw new Error(`Video is longer than ${PIPELINE.maxVideoSeconds}s.`);
    }
    const frames = [];
    for (let i = 0; i < n; i++) {
      abortable(signal);
      const t = ((i + 0.5) * duration) / n;
      await seek(video, Math.min(t, Math.max(0, duration - 0.001)));
      frames.push({
        index: i,
        timestamp: round(t, 3),
        canvas: toCanvas(video, video.videoWidth, video.videoHeight, 960),
      });
      onProgress?.((i + 1) / n);
    }
    return { frames, duration: round(duration, 2), width: video.videoWidth, height: video.videoHeight };
  } finally {
    video.removeAttribute("src");
    video.load();
    URL.revokeObjectURL(url);
  }
}

async function loadImage(file) {
  let bitmap;
  try {
    bitmap = await createImageBitmap(file);
  } catch {
    throw new Error("The image could not be decoded by your browser.");
  }
  const canvas = toCanvas(bitmap, bitmap.width, bitmap.height, 1600);
  const info = { width: bitmap.width, height: bitmap.height };
  bitmap.close?.();
  return { canvas, info };
}

let audioModelPromise = null;
/** Load the exported audio CNN once; resolves to null when it is not deployed. */
export function getAudioModel() {
  if (!audioModelPromise) audioModelPromise = loadAudioModel(AUDIO_MODEL_URL).catch(() => null);
  return audioModelPromise;
}

// Viridis control points (perceptually uniform, readable in both themes).
const VIRIDIS = [
  [68, 1, 84],
  [59, 82, 139],
  [33, 145, 140],
  [94, 201, 98],
  [253, 231, 37],
];

/** Log-mel (n_mels x frames, row 0 = lowest band) -> PNG data URL, low frequencies at the bottom. */
function melToDataUrl(mel) {
  const { data, nMels, frames } = mel;
  const sorted = Float32Array.from(data).sort();
  const lo = sorted[Math.floor(0.01 * (sorted.length - 1))];
  const hi = sorted[Math.floor(0.99 * (sorted.length - 1))];
  const canvas = makeCanvas(frames, nMels);
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(frames, nMels);
  for (let m = 0; m < nMels; m++) {
    for (let f = 0; f < frames; f++) {
      const t = Math.min(1, Math.max(0, (data[m * frames + f] - lo) / Math.max(hi - lo, 1e-6))) * 4;
      const k = Math.min(3, Math.floor(t));
      const a = VIRIDIS[k];
      const b = VIRIDIS[k + 1];
      const w = t - k;
      const o = ((nMels - 1 - m) * frames + f) * 4;
      img.data[o] = a[0] + (b[0] - a[0]) * w;
      img.data[o + 1] = a[1] + (b[1] - a[1]) * w;
      img.data[o + 2] = a[2] + (b[2] - a[2]) * w;
      img.data[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL("image/png");
}

async function scoreWithModel(model, mono, sr, onProgress, signal) {
  const limit = BROWSER_AUDIO_MAX_SECONDS * sr;
  const x = mono.length > limit ? mono.subarray(0, limit) : mono;
  const clips = await scoreWaveformAsync(model, x, { onClip: (i, n) => onProgress?.(i / n), signal });
  if (!clips.length) return { available: false, reason: "audio too short to analyse" };
  const meanLogit = mean(clips.map((c) => c.logit));
  const b = model.manifest.benchmark || {};
  const top = clips.reduce((best, c, i) => (c.logit > clips[best].logit ? i : best), 0);
  return {
    available: true,
    probability: round(calibrate(model, meanLogit), 5),
    raw_probability: round(sig(meanLogit), 5),
    calibrated: Boolean(b.platt),
    duration: round(mono.length / sr, 2),
    analysed_seconds: round(x.length / sr, 2),
    truncated: x.length < mono.length,
    silence_trimmed: Boolean(model.cfg.trim_silence),
    segments: clips.map((c) => ({
      start: round(c.start, 2),
      end: round(c.end, 2),
      probability: round(calibrate(model, c.logit), 5),
      raw_probability: round(sig(c.logit), 5),
    })),
    explanation: {
      clip_index: top,
      start: round(clips[top].start, 2),
      end: round(clips[top].end, 2),
      spectrogram: melToDataUrl(clips[top].mel),
      method: "log-mel spectrogram, computed in the browser (no saliency map here)",
    },
    status: "trained",
    model: {
      name: "Audio CNN (ASVspoof 2019 LA)",
      system: b.system,
      eval_eer: b.eval_eer,
      in_the_wild_eer: b.in_the_wild_eer,
      run_id: b.run_id,
      release: b.release,
    },
    note: AUDIO_MODEL_NOTE,
  };
}

/** Decode, mix to mono 16 kHz, then score with the trained audio CNN or a loudness placeholder. */
async function analyseAudio(file, { onProgress, signal } = {}) {
  const Offline = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  if (!Offline) return { available: false, reason: "Web Audio is not available in this browser" };
  let buffer;
  try {
    const ctx = new Offline(1, 1, 16000);
    buffer = await ctx.decodeAudioData(await file.arrayBuffer());
  } catch {
    return { available: false, reason: "no decodable audio stream found" };
  }
  const sr = buffer.sampleRate;
  const mono = new Float32Array(buffer.length);
  for (let ch = 0; ch < buffer.numberOfChannels; ch++) {
    const d = buffer.getChannelData(ch);
    for (let i = 0; i < d.length; i++) mono[i] += d[i] / buffer.numberOfChannels;
  }
  const clip = Math.round(PIPELINE.audioClipSeconds * sr);
  const segments = [];
  let total = 0;
  for (let start = 0; start < mono.length; start += clip) {
    const end = Math.min(mono.length, start + clip);
    let sq = 0;
    for (let i = start; i < end; i++) sq += mono[i] * mono[i];
    total += sq;
    const db = 10 * Math.log10(sq / Math.max(1, end - start) + 1e-10);
    segments.push({ start: round(start / sr, 2), end: round(end / sr, 2), probability: placeholder((db + 30) / 6) });
  }
  if (!segments.length || total / mono.length < 1e-8) {
    return { available: false, reason: "audio track is silent" };
  }
  const model = await getAudioModel();
  if (model && sr === model.cfg.sample_rate) return scoreWithModel(model, mono, sr, onProgress, signal);
  return {
    available: true,
    probability: round(mean(segments.map((s) => s.probability)), 5),
    duration: round(mono.length / sr, 2),
    segments,
    status: "placeholder_heuristic",
    heuristic: HEURISTICS.loudness_heuristic,
  };
}

/**
 * Run the browser pipeline and return an object shaped like the Flask /api/predict response,
 * with mode "browser-demo", no label, and extra `browser` details for the preview components.
 */
export async function analyseInBrowser(file, { onStage, onProgress, signal } = {}) {
  const started = performance.now();
  const mediaType = mediaTypeOf(file.name);
  if (!mediaType) throw new Error(`Unsupported file type: ${file.name}`);
  if (file.size > PIPELINE.maxUploadMb * 1024 * 1024) {
    throw new Error(`File too large. Maximum size is ${PIPELINE.maxUploadMb} MB.`);
  }
  onStage?.("analysing");
  onProgress?.(0);

  let frames = [];
  let fileMeta = {};
  if (mediaType === "image") {
    const { canvas, info } = await loadImage(file);
    frames = [{ index: 0, timestamp: 0, canvas }];
    fileMeta = info;
  } else if (mediaType === "video") {
    const v = await sampleVideo(file, PIPELINE.framesPerVideo, (p) => onProgress?.(p * 0.5), signal);
    frames = v.frames;
    fileMeta = { duration: v.duration, width: v.width, height: v.height, frames_sampled: v.frames.length };
  }

  const samples = [];
  for (let i = 0; i < frames.length; i++) {
    abortable(signal);
    const f = frames[i];
    const faces = await detectFaces(f.canvas);
    const crop = faces.length ? alignAndCrop(f.canvas, faces[0]) : centreCrop(f.canvas);
    const scores = heuristicScores(crop.canvas);
    const probability = round(mean(Object.values(scores)), 5);
    samples.push({
      index: f.index,
      timestamp: f.timestamp,
      face_found: faces.length > 0,
      faces_in_frame: faces.length,
      detector_confidence: faces.length ? round(faces[0].score, 3) : null,
      aligned: crop.aligned,
      roll_deg: round(crop.angle, 1),
      probability,
      per_model: scores,
      annotated: annotate(f.canvas, faces, crop, mediaType === "image" ? 900 : 360),
      crop: crop.canvas.toDataURL("image/jpeg", 0.9),
    });
    onProgress?.(mediaType === "video" ? 0.5 + ((i + 1) / frames.length) * 0.45 : (i + 1) / frames.length);
  }

  const tAudio = performance.now();
  const audioStart = mediaType === "video" ? 0.95 : 0;
  const audio =
    mediaType === "image"
      ? null
      : await analyseAudio(file, { onProgress: (p) => onProgress?.(audioStart + p * (1 - audioStart)), signal });
  if (mediaType === "audio" && !audio.available) {
    throw new Error(`Audio could not be analysed: ${audio.reason}.`);
  }
  onProgress?.(1);

  const frameRows = samples.map(({ annotated: _a, crop: _c, ...rest }) => rest);
  const visual = samples.length
    ? {
        probability: round(mean(samples.map((s) => s.probability)), 5),
        per_model: Object.fromEntries(
          Object.keys(samples[0].per_model).map((k) => [
            k,
            {
              probability: round(mean(samples.map((s) => s.per_model[k])), 5),
              weight: 0.5,
              status: "placeholder_heuristic",
            },
          ]),
        ),
        temporal_probability: null,
        aggregation: "mean",
        frames: frameRows,
        faces_detected: samples.filter((s) => s.face_found).length,
        frames_analyzed: samples.length,
      }
    : null;

  const used = [];
  let num = 0;
  let den = 0;
  if (visual) {
    used.push("visual");
    num += PIPELINE.visualWeight * visual.probability;
    den += PIPELINE.visualWeight;
  }
  const audioReal = audio?.status === "trained";
  // Never average a real model score with placeholder heuristics: with a placeholder visual branch
  // the overall number is the visual placeholder alone, and the real audio score is shown on its own.
  if (audio?.available && (!visual || !audioReal)) {
    used.push("audio");
    num += PIPELINE.audioWeight * audio.probability;
    den += PIPELINE.audioWeight;
  }

  const audioOnlyReal = audioReal && !visual;
  return {
    media_type: mediaType,
    mode: "browser-demo",
    audio_model: audioReal ? "trained" : "placeholder",
    warning: BROWSER_WARNING,
    label: audioOnlyReal ? (audio.probability >= PIPELINE.threshold ? "FAKE" : "REAL") : null,
    probability_fake: round(num / den, 5),
    threshold: PIPELINE.threshold,
    modalities: {
      ...(visual ? { visual } : {}),
      audio: audio || { available: false, reason: "not applicable for images" },
    },
    fusion: {
      method: "weighted",
      weights: { visual: PIPELINE.visualWeight, audio: PIPELINE.audioWeight },
      used,
      validated: false,
      note:
        audioReal && visual
          ? "The real audio score is not fused with placeholder visual scores; it is reported on its own."
          : "Fusion weights are a heuristic and have not been evaluated on paired audio-visual data.",
    },
    gradcam: null,
    processing_ms: round(performance.now() - started, 1),
    timings_ms: {
      ...(frames.length ? { decode_and_face_crop: round(tAudio - started, 1) } : {}),
      ...(audio ? { [audioReal ? "audio_model" : "audio_decode"]: round(performance.now() - tAudio, 1) } : {}),
    },
    file: { name: file.name, size_bytes: file.size, ...fileMeta },
    browser: {
      detector: mediaType === "audio" ? null : DETECTOR_NAME,
      heuristics: HEURISTICS,
      crop_size: PIPELINE.cropSize,
      margin: PIPELINE.margin,
      samples: samples.map(({ per_model: _p, ...rest }) => rest),
    },
  };
}
