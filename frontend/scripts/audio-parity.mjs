// Numerical parity check: the browser audio model (src/demo/audioModel.js) must reproduce the
// Python log-mel features and Keras logits stored in audio_fixture.json by
// backend/scripts/export_audio_web.py.  Usage: node scripts/audio-parity.mjs <model-dir>
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { buildModel, scoreWaveform } from "../src/demo/audioModel.js";

const dir = process.argv[2] || "public/models/audio";
const f32 = (b64) => new Float32Array(new Uint8Array(Buffer.from(b64, "base64")).buffer);
const manifest = JSON.parse(readFileSync(join(dir, "audio_cnn.json"), "utf8"));
const bin = readFileSync(join(dir, manifest.weights_file));
const model = buildModel(manifest, new Uint8Array(bin).buffer);
const fixture = JSON.parse(readFileSync(join(dir, "audio_fixture.json"), "utf8"));

const t0 = performance.now();
const clips = scoreWaveform(model, f32(fixture.signal_f32));
const ms = performance.now() - t0;
const refMel = f32(fixture.first_clip_log_mel_f32);
let melDiff = 0;
for (let i = 0; i < refMel.length; i++) melDiff = Math.max(melDiff, Math.abs(refMel[i] - clips[0].mel.data[i]));
const logitDiff = Math.max(...clips.map((c, i) => Math.abs(c.logit - fixture.clip_logits[i])));
const result = {
  n_clips: clips.length,
  expected_clips: fixture.n_clips,
  max_abs_log_mel_diff: melDiff,
  max_abs_logit_diff: logitDiff,
  js_ms_per_clip: Math.round((ms / clips.length) * 10) / 10,
  tolerance: fixture.tolerance,
};
console.log(JSON.stringify(result));
if (process.argv[3]) writeFileSync(process.argv[3], JSON.stringify(result, null, 1) + "\n");
const ok =
  clips.length === fixture.n_clips &&
  melDiff <= fixture.tolerance.log_mel &&
  logitDiff <= fixture.tolerance.logit;
if (!ok) {
  console.error("audio parity FAILED");
  process.exit(1);
}
