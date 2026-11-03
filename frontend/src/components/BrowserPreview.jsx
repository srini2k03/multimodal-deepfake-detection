import { useState } from "react";

function SampleFacts({ s }) {
  if (!s.face_found) return <span className="chip chip-muted">no face &middot; centre crop</span>;
  return (
    <>
      <span className="chip chip-ok">face {(s.detector_confidence * 100).toFixed(0)}%</span>
      {s.faces_in_frame > 1 && <span className="chip">{s.faces_in_frame} faces, largest kept</span>}
      <span className="chip">{s.aligned ? `roll ${s.roll_deg}°` : "no roll fix"}</span>
    </>
  );
}

/** Image input: detection overlay next to the 299x299 crop the models would receive. */
function ImagePreview({ sample, cropSize, margin }) {
  return (
    <div className="preview-image">
      <figure className="preview-figure">
        <img src={sample.annotated} alt="Uploaded image with detected face boxes" />
        <figcaption>
          Detected faces (solid), eye keypoints and the {Math.round(margin * 100)}% margin square crop (dashed).
        </figcaption>
      </figure>
      <figure className="preview-crop">
        <img src={sample.crop} alt="Aligned face crop" width={cropSize} height={cropSize} />
        <figcaption>
          Model input: {cropSize}&times;{cropSize} aligned crop
          <div className="chips">
            <SampleFacts s={sample} />
          </div>
        </figcaption>
      </figure>
    </div>
  );
}

/** Video input: the sampled frame timeline with face boxes, and the crop for the selected frame. */
function VideoPreview({ samples, cropSize }) {
  const [sel, setSel] = useState(() => Math.max(0, samples.findIndex((s) => s.face_found)));
  const s = samples[sel];
  return (
    <div className="preview-video">
      <ol className="frame-strip" aria-label="Sampled frames">
        {samples.map((f, i) => (
          <li key={f.index}>
            <button
              type="button"
              className={`frame-thumb ${i === sel ? "selected" : ""} ${f.face_found ? "" : "faceless"}`}
              onClick={() => setSel(i)}
              aria-pressed={i === sel}
              aria-label={`Frame ${f.index + 1} at ${f.timestamp.toFixed(2)} seconds${f.face_found ? "" : ", no face"}`}
            >
              <img src={f.annotated} alt="" loading="lazy" />
              <span className="frame-time">{f.timestamp.toFixed(2)}s</span>
            </button>
          </li>
        ))}
      </ol>
      <div className="preview-image">
        <figure className="preview-figure">
          <img src={s.annotated} alt={`Sampled frame ${s.index + 1} with detected face boxes`} />
          <figcaption>
            Sample {s.index + 1} of {samples.length} at {s.timestamp.toFixed(2)}s (centre of segment {s.index + 1}).
          </figcaption>
        </figure>
        <figure className="preview-crop">
          <img src={s.crop} alt="Aligned face crop for the selected frame" width={cropSize} height={cropSize} />
          <figcaption>
            Model input: {cropSize}&times;{cropSize} crop
            <div className="chips">
              <SampleFacts s={s} />
            </div>
          </figcaption>
        </figure>
      </div>
    </div>
  );
}

export function CropGallery({ samples }) {
  return (
    <ol className="crop-gallery" aria-label="Face crops fed to the models">
      {samples.map((s) => (
        <li key={s.index} className={s.face_found ? "" : "faceless"}>
          <img src={s.crop} alt={`Crop for sample ${s.index + 1}`} loading="lazy" />
          <span>
            {s.timestamp.toFixed(1)}s{s.face_found ? "" : " · centre"}
          </span>
        </li>
      ))}
    </ol>
  );
}

export default function BrowserPreview({ result }) {
  const { samples, crop_size: cropSize, margin } = result.browser;
  if (!samples.length) return null;
  return result.media_type === "video" ? (
    <VideoPreview samples={samples} cropSize={cropSize} />
  ) : (
    <ImagePreview sample={samples[0]} cropSize={cropSize} margin={margin} />
  );
}
