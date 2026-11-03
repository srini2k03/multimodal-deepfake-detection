import { useRef, useState } from "react";
import { ACCEPT_ATTR, mediaTypeOf } from "../api.js";

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
      className={`dropzone ${dragging ? "dragging" : ""} ${disabled ? "disabled" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      onClick={() => !disabled && inputRef.current?.click()}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && !disabled && inputRef.current?.click()}
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
        &#8682;
      </div>
      <p className="dropzone-title">Drag &amp; drop a file, or click to browse</p>
      <p className="dropzone-hint">Images (JPG, PNG, WebP) &middot; Videos (MP4, AVI, MOV, MKV, WebM) &middot; Audio (WAV, FLAC, MP3)</p>
      {localError && <p className="dropzone-error">{localError}</p>}
    </section>
  );
}
