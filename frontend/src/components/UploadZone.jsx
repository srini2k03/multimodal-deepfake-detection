import { useRef, useState } from "react";
import { ACCEPT_ATTR, mediaTypeOf } from "../api.js";
import { BROWSER_DEMO, PIPELINE } from "../demo/config.js";
import { AudioIcon, ImageIcon, UploadIcon, VideoIcon } from "./Icons.jsx";

const KINDS = [
  [ImageIcon, "Image", "JPG · PNG · WebP"],
  [VideoIcon, "Video", "MP4 · AVI · MOV · MKV · WebM"],
  [AudioIcon, "Audio", "WAV · FLAC · MP3"],
];

export default function UploadZone({ onFile, disabled }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const [localError, setLocalError] = useState(null);

  const handle = (file) => {
    if (!file) return;
    if (!mediaTypeOf(file.name)) {
      setLocalError(`Unsupported file type: ${file.name}`);
      return;
    }
    setLocalError(null);
    onFile(file);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    if (!disabled) handle(e.dataTransfer.files?.[0]);
  };

  return (
    <section
      className={`dropzone sb-spotlight ${dragging ? "dragging" : ""} ${disabled ? "disabled" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setDragging(true);
      }}
      onDragLeave={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget)) setDragging(false);
      }}
      onDrop={onDrop}
      onClick={() => !disabled && inputRef.current?.click()}
      onKeyDown={(e) => {
        if ((e.key === "Enter" || e.key === " ") && !disabled) {
          e.preventDefault();
          inputRef.current?.click();
        }
      }}
      role="button"
      tabIndex={0}
      aria-disabled={disabled}
      aria-label="Upload an image, video or audio file"
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT_ATTR}
        hidden
        onChange={(e) => {
          handle(e.target.files?.[0]);
          e.target.value = "";
        }}
      />
      <div className="dropzone-icon" aria-hidden>
        <UploadIcon size={28} />
      </div>
      <p className="dropzone-title">
        {dragging ? "Release to analyse" : <>Drag &amp; drop a file, or click to browse</>}
      </p>
      <p className="dropzone-hint sr-only">
        Images (JPG, PNG, WebP) &middot; Videos (MP4, AVI, MOV, MKV, WebM) &middot; Audio (WAV, FLAC, MP3)
      </p>
      <ul className="dropzone-kinds" aria-hidden="true">
        {KINDS.map(([Icon, label, exts]) => (
          <li key={label}>
            <Icon />
            <span className="kind-label">{label}</span>
            <span className="kind-exts">{exts}</span>
          </li>
        ))}
      </ul>
      <p className="dropzone-meta">
        Up to {PIPELINE.maxUploadMb} MB &middot; videos up to {PIPELINE.maxVideoSeconds} s &middot;{" "}
        {BROWSER_DEMO ? "processed on your device, nothing is uploaded" : "sent to the detection API"}
      </p>
      {localError && (
        <p className="dropzone-error" role="alert">
          {localError}
        </p>
      )}
    </section>
  );
}
