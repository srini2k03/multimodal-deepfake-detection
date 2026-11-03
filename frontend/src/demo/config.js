// Static "browser demo" build (VITE_DEMO=browser): the analysis runs client-side and no
// Flask API is contacted. Scores in this mode are placeholders, never deepfake verdicts.
export const BROWSER_DEMO = import.meta.env.VITE_DEMO === "browser";

export const REPO_URL = "https://github.com/srini2k03/multimodal-deepfake-detection";
export const SETUP_URL = `${REPO_URL}#quick-start`;
export const TRAINING_URL = `${REPO_URL}#training`;

// MediaPipe Tasks Vision, loaded at runtime from a CDN with an exact version pin.
export const MEDIAPIPE_VERSION = "0.10.35";
export const MEDIAPIPE_CDN = (
  import.meta.env.VITE_MEDIAPIPE_CDN || `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}`
).replace(/\/$/, "");

// BlazeFace short-range model (Apache-2.0), committed under public/models.
export const FACE_MODEL_URL = `${import.meta.env.BASE_URL}models/blaze_face_short_range.tflite`;

// Mirrors backend/configs/default.yaml so the browser pipeline behaves like the Python one.
export const PIPELINE = {
  framesPerVideo: 16, // api.frames_per_video
  maxVideoSeconds: 300, // api.max_video_seconds
  maxUploadMb: 100, // api.max_upload_mb
  margin: 0.3, // face.margin
  cropSize: 299, // face.crop_size
  minConfidence: 0.6, // face.confidence
  audioClipSeconds: 4, // audio.clip_seconds
  visualWeight: 0.7, // fusion.visual_weight
  audioWeight: 0.3, // fusion.audio_weight
  threshold: 0.5, // fusion.threshold
};

export const BROWSER_WARNING =
  "This page shows the preprocessing pipeline (frame sampling, face detection, alignment and " +
  "cropping) running in your browser. The scores are placeholder heuristics computed from simple pixel and " +
  "loudness statistics. They are not produced by a deepfake classifier and say nothing about whether the media " +
  "is manipulated. Real detection needs the Python backend with trained weights.";
