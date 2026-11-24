/**
 * Browser port of the backend audio branch: log-mel front end + the small CNN trained on
 * ASVspoof 2019 LA. Pure JavaScript (no DOM, no dependencies) so Node can run the parity check
 * in scripts/audio-parity.mjs against values computed by Python/Keras.
 *
 * Mirrors backend/deepfake_detector/preprocessing/audio.py:
 *   silence trim (optional) -> 4 s segments (repeat padding) -> reflect-padded STFT with a
 *   symmetric Hann window -> power -> mel filterbank (exported from Python) -> log(x + 1e-6)
 *   -> per-clip standardisation -> CNN (BatchNorm folded into the convolutions) -> spoof logit.
 */

const SILENCE_FRAME_S = 0.025;
const SILENCE_HOP_S = 0.01;

// ------------------------------------------------------------------ model loading
export async function loadAudioModel(baseUrl, fetchImpl = fetch) {
  const base = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
  const res = await fetchImpl(`${base}audio_cnn.json`);
  if (!res.ok) throw new Error(`audio model manifest not found (${res.status})`);
  const manifest = await res.json();
  if (manifest.format !== "dfd-audio-cnn-v1") throw new Error(`unsupported model format ${manifest.format}`);
  const bin = await fetchImpl(`${base}${manifest.weights_file}`);
  if (!bin.ok) throw new Error(`audio model weights not found (${bin.status})`);
  return buildModel(manifest, await bin.arrayBuffer());
}

export function buildModel(manifest, buffer) {
  const all = new Float32Array(buffer);
  const view = (ref) => all.subarray(ref.offset / 4, ref.offset / 4 + ref.length);
  const layers = manifest.layers.map((l) => ({
    ...l,
    kernel: l.kernel ? view(l.kernel) : null,
    kernelShape: l.kernel ? l.kernel.shape : null,
    bias: l.bias ? view(l.bias) : null,
  }));
  return { manifest, cfg: manifest.audio, filterbank: view(manifest.mel_filterbank), layers };
}

// ------------------------------------------------------------------ waveform helpers
export function frameEnergyDb(x, sr) {
  const frame = Math.trunc(SILENCE_FRAME_S * sr);
  const hop = Math.trunc(SILENCE_HOP_S * sr);
  if (x.length === 0) return new Float64Array(0);
  if (x.length < frame) return Float64Array.of(rmsDb(x, 0, x.length));
  const n = 1 + Math.floor((x.length - frame) / hop);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = rmsDb(x, i * hop, i * hop + frame);
  return out;
}

function rmsDb(x, start, end) {
  let s = 0;
  for (let i = start; i < end; i++) s += x[i] * x[i];
  return 20 * Math.log10(Math.max(Math.sqrt(s / (end - start)), 1e-10));
}

export function silenceProfile(x, sr, topDb = 40) {
  const e = frameEnergyDb(x, sr);
  const duration = x.length / sr;
  if (!e.length) return { duration_s: 0, lead_silence_s: 0, trail_silence_s: 0 };
  let peak = -Infinity;
  for (const v of e) peak = Math.max(peak, v);
  let first = -1;
  let last = -1;
  for (let i = 0; i < e.length; i++) {
    if (e[i] > peak - topDb) {
      if (first < 0) first = i;
      last = i;
    }
  }
  return {
    duration_s: duration,
    lead_silence_s: first * SILENCE_HOP_S,
    trail_silence_s: Math.max(0, duration - (last * SILENCE_HOP_S + SILENCE_FRAME_S)),
  };
}

export function trimSilence(x, sr, topDb = 40) {
  const p = silenceProfile(x, sr, topDb);
  const start = Math.trunc(p.lead_silence_s * sr);
  const end = x.length - Math.trunc(p.trail_silence_s * sr);
  const out = x.subarray(start, end);
  return out.length >= Math.trunc(0.1 * sr) ? out : x;
}

export function fixLength(x, length, padMode = "repeat") {
  if (x.length >= length) return x.subarray(0, length);
  const out = new Float32Array(length);
  if (x.length === 0 || padMode === "zero") {
    out.set(x);
    return out;
  }
  for (let i = 0; i < length; i++) out[i] = x[i % x.length];
  return out;
}

/** Same rule as segment_waveform: a short tail is kept only if >= half a clip (or the only clip). */
export function segment(x, cfg) {
  const clip = Math.trunc(cfg.clip_seconds * cfg.sample_rate);
  const segs = [];
  for (let start = 0; start < x.length; start += clip) {
    let seg = x.subarray(start, Math.min(x.length, start + clip));
    if (seg.length < clip) {
      if (seg.length < clip * 0.5 && segs.length) break;
      seg = fixLength(seg, clip, cfg.pad_mode);
    }
    segs.push({ start: start / cfg.sample_rate, end: (start + clip) / cfg.sample_rate, samples: seg });
  }
  return segs;
}

// ------------------------------------------------------------------ log-mel
const fftCache = new Map();

function fftTables(n) {
  if (fftCache.has(n)) return fftCache.get(n);
  const bits = Math.log2(n);
  if (!Number.isInteger(bits)) throw new Error("n_fft must be a power of two");
  const rev = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    let r = 0;
    for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
    rev[i] = r;
  }
  const cos = new Float64Array(n / 2);
  const sin = new Float64Array(n / 2);
  for (let i = 0; i < n / 2; i++) {
    cos[i] = Math.cos((2 * Math.PI * i) / n);
    sin[i] = -Math.sin((2 * Math.PI * i) / n);
  }
  const hann = new Float64Array(n); // numpy.hanning: symmetric window
  for (let i = 0; i < n; i++) hann[i] = Math.fround(0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1)));
  const t = { rev, cos, sin, hann };
  fftCache.set(n, t);
  return t;
}

/** In-place iterative radix-2 FFT on (re, im). */
function fft(re, im, t) {
  const n = re.length;
  for (let i = 0; i < n; i++) {
    const j = t.rev[i];
    if (j > i) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1;
    const step = n / size;
    for (let start = 0; start < n; start += size) {
      for (let k = 0; k < half; k++) {
        const wr = t.cos[k * step];
        const wi = t.sin[k * step];
        const a = start + k;
        const b = a + half;
        const xr = re[b] * wr - im[b] * wi;
        const xi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - xr;
        im[b] = im[a] - xi;
        re[a] += xr;
        im[a] += xi;
      }
    }
  }
}

function reflectPad(x, pad) {
  const n = x.length;
  const out = new Float64Array(n + 2 * pad);
  if (n <= pad) {
    for (let i = 0; i < n; i++) out[pad + i] = x[i]; // numpy falls back to constant (zero) padding
    return out;
  }
  for (let i = 0; i < n; i++) out[pad + i] = x[i];
  for (let i = 0; i < pad; i++) {
    out[pad - 1 - i] = x[i + 1];
    out[pad + n + i] = x[n - 2 - i];
  }
  return out;
}

/** Standardised log-mel of one fixed-length clip, row-major (n_mels x frames). */
export function logMel(samples, cfg, filterbank) {
  const nFft = cfg.n_fft;
  const hop = cfg.hop_length;
  const nBins = nFft / 2 + 1;
  const t = fftTables(nFft);
  const padded = reflectPad(samples, nFft / 2);
  const frames = 1 + Math.max(0, Math.floor((padded.length - nFft) / hop));
  const power = new Float64Array(nBins * frames); // (bins, frames)
  const re = new Float64Array(nFft);
  const im = new Float64Array(nFft);
  for (let f = 0; f < frames; f++) {
    const off = f * hop;
    for (let i = 0; i < nFft; i++) {
      re[i] = Math.fround(padded[off + i] * t.hann[i]);
      im[i] = 0;
    }
    fft(re, im, t);
    for (let k = 0; k < nBins; k++) power[k * frames + f] = re[k] * re[k] + im[k] * im[k];
  }
  const nMels = cfg.n_mels;
  const out = new Float32Array(nMels * frames);
  let sum = 0;
  for (let m = 0; m < nMels; m++) {
    for (let f = 0; f < frames; f++) {
      let acc = 0;
      for (let k = 0; k < nBins; k++) acc += filterbank[m * nBins + k] * power[k * frames + f];
      const v = Math.log(acc + 1e-6);
      out[m * frames + f] = v;
      sum += v;
    }
  }
  const mean = sum / out.length;
  let sq = 0;
  for (let i = 0; i < out.length; i++) sq += (out[i] - mean) ** 2;
  const std = Math.sqrt(sq / out.length);
  for (let i = 0; i < out.length; i++) out[i] = (out[i] - mean) / (std + 1e-6);
  return { data: out, nMels, frames };
}

// ------------------------------------------------------------------ CNN forward pass
function conv3x3(input, c, h, w, layer) {
  const [cout, cin] = layer.kernelShape;
  if (cin !== c) throw new Error(`channel mismatch in ${layer.name}: ${cin} vs ${c}`);
  const k = layer.kernel;
  const out = new Float32Array(cout * h * w);
  const acc = new Float64Array(h * w);
  for (let o = 0; o < cout; o++) {
    acc.fill(layer.bias[o]);
    for (let i = 0; i < cin; i++) {
      const src = i * h * w;
      for (let dy = 0; dy < 3; dy++) {
        for (let dx = 0; dx < 3; dx++) {
          const wv = k[((o * cin + i) * 3 + dy) * 3 + dx];
          if (wv === 0) continue;
          const oy0 = Math.max(0, 1 - dy);
          const oy1 = Math.min(h, h + 1 - dy);
          const ox0 = Math.max(0, 1 - dx);
          const ox1 = Math.min(w, w + 1 - dx);
          for (let y = oy0; y < oy1; y++) {
            const row = src + (y + dy - 1) * w + (dx - 1);
            const orow = y * w;
            for (let x = ox0; x < ox1; x++) acc[orow + x] += wv * input[row + x];
          }
        }
      }
    }
    const base = o * h * w;
    for (let p = 0; p < h * w; p++) out[base + p] = layer.relu && acc[p] < 0 ? 0 : acc[p];
  }
  return { data: out, c: cout, h, w };
}

function maxPool2(t) {
  const h = Math.floor(t.h / 2);
  const w = Math.floor(t.w / 2);
  const out = new Float32Array(t.c * h * w);
  for (let c = 0; c < t.c; c++) {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = c * t.h * t.w + 2 * y * t.w + 2 * x;
        out[c * h * w + y * w + x] = Math.max(t.data[i], t.data[i + 1], t.data[i + t.w], t.data[i + t.w + 1]);
      }
    }
  }
  return { data: out, c: t.c, h, w };
}

/** Spoof logit for one standardised log-mel clip. */
export function forward(model, mel) {
  let t = { data: mel.data, c: 1, h: mel.nMels, w: mel.frames };
  let vec = null;
  for (const layer of model.layers) {
    if (layer.type === "conv3x3") t = conv3x3(t.data, t.c, t.h, t.w, layer);
    else if (layer.type === "maxpool2") t = maxPool2(t);
    else if (layer.type === "gap") {
      vec = new Float64Array(t.c);
      for (let c = 0; c < t.c; c++) {
        let s = 0;
        for (let p = 0; p < t.h * t.w; p++) s += t.data[c * t.h * t.w + p];
        vec[c] = s / (t.h * t.w);
      }
    } else if (layer.type === "dense") {
      const [nOut, nIn] = layer.kernelShape;
      const out = new Float64Array(nOut);
      for (let o = 0; o < nOut; o++) {
        let s = layer.bias[o];
        for (let i = 0; i < nIn; i++) s += layer.kernel[o * nIn + i] * vec[i];
        out[o] = layer.relu && s < 0 ? 0 : s;
      }
      vec = out;
    } else throw new Error(`unknown layer type ${layer.type}`);
  }
  return vec[0];
}

export const sigmoid = (z) => 1 / (1 + Math.exp(-z));

/**
 * Score a mono waveform already at cfg.sample_rate. Returns per-clip logits/probabilities and
 * the mean probability, using the same preprocessing as the Python API.
 */
export function scoreWaveform(model, signal, { onClip } = {}) {
  const cfg = model.cfg;
  const x = cfg.trim_silence ? trimSilence(signal, cfg.sample_rate, cfg.trim_top_db) : signal;
  const segs = segment(x, cfg);
  const clips = segs.map((s, i) => {
    const mel = logMel(s.samples, cfg, model.filterbank);
    const logit = forward(model, mel);
    onClip?.(i + 1, segs.length);
    return { start: s.start, end: s.end, logit, probability: sigmoid(logit), mel };
  });
  return clips;
}
