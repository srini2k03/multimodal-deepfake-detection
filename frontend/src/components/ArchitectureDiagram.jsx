/**
 * Hand-drawn SVG version of the README's mermaid architecture flowchart.
 * Two layouts share one renderer: a wide left-to-right one and a tall one for phones.
 */

const WIDE = {
  viewBox: "0 0 1040 400",
  groups: [
    { x: 10, y: 120, w: 152, h: 120, label: "Client" },
    { x: 175, y: 60, w: 160, h: 200, label: "Flask API" },
    { x: 350, y: 10, w: 678, h: 325, label: "Inference engine" },
  ],
  nodes: {
    UI: [20, 150, 132, 70, ["React dashboard", "upload · gauge", "timeline · Grad-CAM"]],
    V: [185, 90, 140, 62, ["Validation", "type · size · decode"]],
    J: [185, 190, 140, 50, ["sync / async job"], "decision"],
    S: [365, 45, 115, 50, ["Frame sampler", "OpenCV"]],
    FD: [495, 45, 115, 50, ["Face detector", "DNN SSD / Haar"]],
    AL: [625, 45, 125, 50, ["Align + crop", "eye roll · 299²"]],
    X: [765, 34, 110, 36, ["Xception"], "model"],
    I: [765, 82, 110, 36, ["InceptionV3"], "model"],
    GC: [765, 150, 110, 36, ["Grad-CAM"]],
    E: [885, 35, 135, 80, ["Weighted ensemble", "per-frame P(fake)"]],
    AG: [885, 135, 135, 55, ["Temporal aggregation", "mean / top-k / attention"]],
    A: [365, 245, 115, 50, ["Audio track", "ffmpeg / wav"]],
    M: [495, 245, 115, 50, ["Log-mel", "spectrogram"]],
    AC: [625, 245, 125, 50, ["Audio CNN", "per 4 s segment"], "model"],
    F: [885, 240, 135, 50, ["Late fusion", "weighted / logit"]],
    R: [480, 352, 400, 40, ["JSON: label · probability · per-model · per-frame · heatmap"], "output"],
  },
  edges: [
    [[150, 170, 167, 170, 167, 121, 185, 121]],
    [[255, 152, 255, 190]],
    [[325, 205, 345, 205, 345, 70, 365, 70]],
    [[325, 225, 345, 225, 345, 270, 365, 270]],
    [[480, 70, 495, 70]],
    [[610, 70, 625, 70]],
    [[750, 62, 757, 62, 757, 52, 765, 52]],
    [[750, 78, 757, 78, 757, 100, 765, 100]],
    [[875, 52, 885, 52]],
    [[875, 100, 885, 100]],
    [[875, 64, 880, 64, 880, 168, 875, 168], true],
    [[952, 115, 952, 135]],
    [[952, 190, 952, 240]],
    [[480, 270, 495, 270]],
    [[610, 270, 625, 270]],
    [[750, 270, 885, 270]],
    [[952, 290, 952, 372, 880, 372]],
    [[820, 186, 820, 352], true],
    [[480, 372, 86, 372, 86, 220]],
  ],
};

const NARROW = {
  viewBox: "0 0 360 850",
  groups: [{ x: 10, y: 222, w: 340, h: 530, label: "Inference engine", bottom: true }],
  nodes: {
    UI: [100, 10, 160, 56, ["React dashboard", "upload · gauge · timeline"]],
    V: [100, 96, 160, 50, ["Flask API: validation", "type · size · decode"]],
    J: [100, 170, 160, 40, ["sync / async job"], "decision"],
    S: [20, 260, 150, 46, ["Frame sampler", "OpenCV"]],
    FD: [20, 326, 150, 46, ["Face detector", "DNN SSD / Haar"]],
    AL: [20, 392, 150, 46, ["Align + crop", "eye roll · 299²"]],
    I: [20, 458, 70, 40, ["Inception"], "model"],
    X: [100, 458, 70, 40, ["Xception"], "model"],
    E: [20, 518, 150, 46, ["Weighted ensemble", "per-frame P(fake)"]],
    AG: [20, 584, 150, 46, ["Temporal aggregation", "mean / top-k / attention"]],
    A: [190, 260, 150, 46, ["Audio track", "ffmpeg / wav"]],
    M: [190, 326, 150, 46, ["Log-mel", "spectrogram"]],
    AC: [190, 392, 150, 46, ["Audio CNN", "per 4 s segment"], "model"],
    GC: [190, 458, 150, 40, ["Grad-CAM"]],
    F: [100, 660, 160, 46, ["Late fusion", "weighted / logit"]],
    R: [60, 780, 240, 56, ["JSON response", "label · probability · per-frame · heatmap"], "output"],
  },
  edges: [
    [[180, 66, 180, 96]],
    [[180, 146, 180, 170]],
    [[140, 210, 140, 240, 95, 240, 95, 260]],
    [[220, 210, 220, 240, 265, 240, 265, 260]],
    [[95, 306, 95, 326]],
    [[95, 372, 95, 392]],
    [[55, 438, 55, 458]],
    [[135, 438, 135, 458]],
    [[55, 498, 55, 518]],
    [[135, 498, 135, 518]],
    [[170, 478, 190, 478], true],
    [[95, 564, 95, 584]],
    [[95, 630, 95, 683, 100, 683]],
    [[265, 306, 265, 326]],
    [[265, 372, 265, 392]],
    [[340, 415, 345, 415, 345, 683, 260, 683]],
    [[180, 706, 180, 780]],
    [[265, 498, 265, 640, 280, 640, 280, 780], true],
    [[60, 808, 4, 808, 4, 38, 100, 38]],
  ],
};

function Diagram({ spec, className, id }) {
  return (
    <svg viewBox={spec.viewBox} className={`arch ${className}`} role="img" aria-labelledby={`${id}-title`}>
      <title id={`${id}-title`}>
        System architecture: the React dashboard uploads to the Flask API, which runs frame sampling, face detection,
        alignment, the Xception and InceptionV3 ensemble, temporal aggregation and an audio CNN, combined by late
        fusion into a JSON response.
      </title>
      <defs>
        <marker id={`${id}-arrow`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
          <path d="M0 0 L10 5 L0 10 z" className="arch-arrowhead" />
        </marker>
      </defs>
      {spec.groups.map((g) => (
        <g key={g.label}>
          <rect x={g.x} y={g.y} width={g.w} height={g.h} rx="12" className="arch-group" />
          <text x={g.x + 10} y={g.bottom ? g.y + g.h - 8 : g.y + 16} className="arch-group-label">
            {g.label}
          </text>
        </g>
      ))}
      {spec.edges.map(([pts, dashed], i) => (
        <polyline
          key={i}
          points={pts.join(" ")}
          className={`arch-edge ${dashed ? "dashed" : ""}`}
          markerEnd={`url(#${id}-arrow)`}
        />
      ))}
      {Object.entries(spec.nodes).map(([key, [x, y, w, h, lines, kind]]) => {
        const lineH = 14;
        const top = y + h / 2 - ((lines.length - 1) * lineH) / 2;
        return (
          <g key={key} className={`arch-node ${kind || ""}`}>
            <rect x={x} y={y} width={w} height={h} rx="8" />
            {lines.map((t, i) => (
              <text key={i} x={x + w / 2} y={top + i * lineH + 4} textAnchor="middle" className={i ? "sub" : "title"}>
                {t}
              </text>
            ))}
          </g>
        );
      })}
    </svg>
  );
}

export default function ArchitectureDiagram() {
  return (
    <figure className="arch-figure">
      <Diagram spec={WIDE} className="arch-wide" id="arch-wide" />
      <Diagram spec={NARROW} className="arch-narrow" id="arch-narrow" />
      <figcaption className="muted small">
        Dashed arrows: Grad-CAM explanation path. Missing modalities are dropped from fusion and the remaining weights
        renormalised.
      </figcaption>
    </figure>
  );
}
