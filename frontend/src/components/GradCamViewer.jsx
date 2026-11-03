import { useState } from "react";

/** Face crop with an adjustable Grad-CAM overlay (opacity slider blends the two PNGs). */
export default function GradCamViewer({ gradcam }) {
  const [opacity, setOpacity] = useState(0.8);
  if (!gradcam) return <p className="muted">No explanation available for this input.</p>;
  return (
    <div className="gradcam">
      <div className="gradcam-stack">
        <img src={gradcam.face} alt="Analysed face crop" />
        <img src={gradcam.overlay} alt="Grad-CAM heatmap overlay" style={{ opacity }} className="gradcam-overlay" />
      </div>
      <label className="slider">
        Heatmap opacity
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={opacity}
          onChange={(e) => setOpacity(Number(e.target.value))}
        />
      </label>
      <p className="muted small">
        {gradcam.method} &middot; model: {gradcam.model} &middot; frame #{gradcam.frame_index} (most suspicious frame).
        Warm regions contributed most to the "manipulated" score.
      </p>
    </div>
  );
}
