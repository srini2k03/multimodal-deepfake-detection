import { FACE_MODEL_URL, MEDIAPIPE_CDN, MEDIAPIPE_VERSION, PIPELINE } from "./config.js";

let detectorPromise = null;
let firstRunDone = false;

// The MediaPipe WASM runtime prints this informational line through console.error when the
// TFLite graph starts. Drop only that exact message, only during start-up.
const STARTUP_INFO = /^INFO: Created TensorFlow Lite XNNPACK delegate for CPU\.?$/;

async function quietStartup(fn) {
  const original = console.error;
  console.error = (...args) => {
    if (!(args.length === 1 && STARTUP_INFO.test(String(args[0])))) original(...args);
  };
  try {
    return await fn();
  } finally {
    console.error = original;
  }
}

/** Lazily load MediaPipe Tasks Vision from the CDN and create one shared BlazeFace detector. */
export function getFaceDetector() {
  if (!detectorPromise) {
    detectorPromise = (async () => {
      const vision = await import(/* @vite-ignore */ `${MEDIAPIPE_CDN}/vision_bundle.mjs`);
      const fileset = await vision.FilesetResolver.forVisionTasks(`${MEDIAPIPE_CDN}/wasm`);
      return quietStartup(() =>
        vision.FaceDetector.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: FACE_MODEL_URL, delegate: "CPU" },
          runningMode: "IMAGE",
          minDetectionConfidence: PIPELINE.minConfidence,
        }),
      );
    })().catch((err) => {
      detectorPromise = null; // allow a retry after a network failure
      throw new Error(`Could not load the in-browser face detector (${err.message || err}).`);
    });
  }
  return detectorPromise;
}

export const DETECTOR_NAME = `MediaPipe BlazeFace short-range (@mediapipe/tasks-vision ${MEDIAPIPE_VERSION})`;

/**
 * Detect faces on a canvas. Returns pixel-space boxes sorted largest first, each with the
 * two eye keypoints (when present) and the detector confidence.
 */
export async function detectFaces(canvas) {
  const detector = await getFaceDetector();
  const run = () => detector.detect(canvas);
  const { detections = [] } = firstRunDone ? run() : await quietStartup(run);
  firstRunDone = true;
  const W = canvas.width;
  const H = canvas.height;
  return detections
    .filter((d) => d.boundingBox)
    .map((d) => {
      const b = d.boundingBox;
      const eyes = (d.keypoints || []).slice(0, 2).map((k) => ({ x: k.x * W, y: k.y * H }));
      return {
        x: b.originX,
        y: b.originY,
        w: b.width,
        h: b.height,
        score: d.categories?.[0]?.score ?? 0,
        eyes: eyes.length === 2 ? eyes : null,
      };
    })
    .sort((a, b) => b.w * b.h - a.w * a.h);
}
