import { useState } from "react";

/**
 * Log-mel spectrogram of the most suspicious audio clip, with the audio CNN's Grad-CAM blended on
 * top when the backend provides it. Diagnostic evidence about what the model reacts to, not proof.
 */
export default function AudioExplanation({ explanation }) {
  const [opacity, setOpacity] = useState(0.7);
  if (!explanation?.spectrogram) return null;
  const { overlay } = explanation;
  return (
    <figure className="audio-explain">
      <div className="spectro-stack">
        <img src={explanation.spectrogram} alt="Log-mel spectrogram of the most suspicious clip" />
        {overlay && (
          <img src={overlay} alt="Grad-CAM over the spectrogram" style={{ opacity }} className="gradcam-overlay" />
        )}
      </div>
      {overlay && (
        <label className="slider">
          Saliency opacity
          <input
            className="sb-range"
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={opacity}
            onChange={(e) => setOpacity(Number(e.target.value))}
          />
        </label>
      )}
      <figcaption className="muted small">
        Clip {explanation.start}–{explanation.end}s (highest score) · {explanation.method}. Time runs left to right,
        frequency bottom to top. Warm regions raised the spoof score; this shows what the model reacts to (often
        silence or high-frequency bands), not proof that a region is synthetic.
      </figcaption>
    </figure>
  );
}
