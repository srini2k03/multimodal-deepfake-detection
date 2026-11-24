import { useEffect, useRef, useState } from "react";
import data from "../data/benchmarks.json";
import { AlertIcon, ArrowUpRight } from "./Icons.jsx";
import { REPO_URL } from "../demo/config.js";

// Every number on this panel comes from src/data/benchmarks.json, which
// backend/scripts/make_report.py builds from the result files the benchmark workflows commit
// (results/audio/asvspoof2019_la, results/visual/dff). Nothing here is typed in by hand.

const pct = (v, d = 1) => (v == null ? "n/a" : `${(100 * v).toFixed(d)}%`);
const fix = (v, d = 3) => (v == null ? "n/a" : v.toFixed(d));
const SERIES = ["var(--chart-1)", "var(--chart-2)", "var(--chart-3)", "var(--chart-4)"];
const SHAPES = ["circle", "square", "diamond", "triangle"];

function useWidth(ref, fallback = 640) {
  const [width, setWidth] = useState(fallback);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") return undefined;
    const ro = new ResizeObserver(([entry]) => setWidth(Math.max(260, Math.round(entry.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref]);
  return width;
}

function Mark({ shape, x, y, color }) {
  const r = 5;
  if (shape === "square") return <rect x={x - r} y={y - r} width={2 * r} height={2 * r} rx={1.5} fill={color} />;
  if (shape === "diamond")
    return <path d={`M${x} ${y - r - 1}L${x + r + 1} ${y}L${x} ${y + r + 1}L${x - r - 1} ${y}Z`} fill={color} />;
  if (shape === "triangle") return <path d={`M${x} ${y - r - 1}L${x + r + 1} ${y + r}L${x - r - 1} ${y + r}Z`} fill={color} />;
  return <circle cx={x} cy={y} r={r} fill={color} />;
}

/**
 * Horizontal dot plot: one row per category, one colour + marker shape per series, a dashed
 * reference line (chance), hover/focus tooltips and a table fallback.
 */
export function DotPlot({ title, rows, series, value, domain, format, axisLabel, refLine }) {
  const wrap = useRef(null);
  const width = useWidth(wrap);
  const [tip, setTip] = useState(null);
  const narrow = width < 520;
  const left = narrow ? 104 : 150;
  const right = 16;
  const rowH = 30;
  const top = 22;
  const height = top + rows.length * rowH + 30;
  const [lo, hi] = domain;
  const x = (v) => left + ((Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo)) * (width - left - right);
  const ticks = Array.from({ length: 5 }, (_, i) => lo + ((hi - lo) * i) / 4);
  const offsets = series.map((_, k) => (k - (series.length - 1) / 2) * 7);

  return (
    <figure className="dotplot" ref={wrap}>
      <figcaption className="dotplot-title">{title}</figcaption>
      {series.length > 1 && (
        <div className="legend" aria-hidden="true">
          {series.map((s, k) => (
            <span className="legend-item" key={s.key}>
              <svg width="12" height="12" viewBox="0 0 12 12" className="legend-mark">
                <Mark shape={SHAPES[k]} x={6} y={6} color={SERIES[k]} />
              </svg>
              {s.label}
            </span>
          ))}
        </div>
      )}
      <div className="dotplot-canvas">
        <svg width={width} height={height} role="img" aria-label={`${title}. Values are listed in the table below.`}>
          {ticks.map((t) => (
            <g key={t}>
              <line x1={x(t)} x2={x(t)} y1={top - 6} y2={height - 26} className="chart-grid" />
              <text x={x(t)} y={height - 10} textAnchor="middle" className="chart-tick">
                {format(t)}
              </text>
            </g>
          ))}
          {refLine && (
            <g>
              <line x1={x(refLine.value)} x2={x(refLine.value)} y1={top - 12} y2={height - 26} className="chart-ref" />
              <text x={x(refLine.value) + 4} y={top - 10} className="chart-tick">
                {refLine.label}
              </text>
            </g>
          )}
          {rows.map((row, i) => {
            const cy = top + i * rowH + rowH / 2;
            return (
              <g key={row.key}>
                <text x={left - 10} y={cy + 4} textAnchor="end" className="chart-row-label">
                  {row.label}
                </text>
                {series.map((s, k) => {
                  const v = value(row.key, s.key);
                  if (v == null) return null;
                  const cx = x(v);
                  const label = `${row.label}, ${s.label}: ${format(v)}`;
                  const show = () => setTip({ cx, cy: cy + offsets[k], label });
                  return (
                    <g
                      key={s.key}
                      tabIndex={0}
                      role="img"
                      aria-label={label}
                      className="chart-point"
                      onMouseEnter={show}
                      onFocus={show}
                      onMouseLeave={() => setTip(null)}
                      onBlur={() => setTip(null)}
                    >
                      <circle cx={cx} cy={cy + offsets[k]} r={11} fill="transparent" />
                      <Mark shape={SHAPES[k]} x={cx} y={cy + offsets[k]} color={SERIES[k]} />
                    </g>
                  );
                })}
              </g>
            );
          })}
        </svg>
        {tip && (
          <div className="chart-tip" style={{ left: Math.min(width - 150, Math.max(0, tip.cx - 75)), top: tip.cy + 12 }}>
            {tip.label}
          </div>
        )}
      </div>
      <p className="chart-axis">{axisLabel}</p>
      <details className="chart-table">
        <summary>Show as table</summary>
        <div className="sb-table-wrap table-wrap">
          <table className="sb-table">
            <thead>
              <tr>
                <th scope="col">{title.split(":")[0]}</th>
                {series.map((s) => (
                  <th scope="col" className="num" key={s.key}>
                    {s.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.key}>
                  <td>{row.label}</td>
                  {series.map((s) => (
                    <td className="num" key={s.key}>
                      {value(row.key, s.key) == null ? "n/a" : format(value(row.key, s.key))}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </figure>
  );
}

function Provenance({ meta, path }) {
  if (!meta) return null;
  const run = String(meta.run_id);
  return (
    <p className="sb-card__foot provenance">
      Run{" "}
      <a href={`${REPO_URL}/actions/runs/${run}`} target="_blank" rel="noreferrer">
        {run}
      </a>{" "}
      · commit{" "}
      <a href={`${REPO_URL}/commit/${meta.git_sha}`} target="_blank" rel="noreferrer">
        <code>{String(meta.git_sha).slice(0, 7)}</code>
      </a>{" "}
      · {String(meta.date_utc).slice(0, 10)} · {meta.runner} ·{" "}
      <a href={`${REPO_URL}/tree/main/${path}`} target="_blank" rel="noreferrer">
        {path} <ArrowUpRight />
      </a>
    </p>
  );
}

function Stats({ items }) {
  return (
    <dl className="sb-stats bench-stats">
      {items.map(([value, label, ctx]) => (
        <div className="sb-stat" key={label}>
          <dt className="sb-stat__label">{label}</dt>
          <dd className="sb-stat__value">{value}</dd>
          <dd className="sb-stat__ctx">{ctx}</dd>
        </div>
      ))}
    </dl>
  );
}

function Pending({ what }) {
  return (
    <div className="sb-callout sb-callout--warn">
      <span className="callout-icon" aria-hidden="true">
        <AlertIcon size={16} />
      </span>
      <p>
        <strong>{what}: no committed results yet.</strong> This panel fills in from the files the benchmark workflow
        commits; until then it shows nothing rather than an estimate.
      </p>
    </div>
  );
}

const AUDIO_MAIN = "cnn_trim";

function AudioBench({ a }) {
  const s = a.systems;
  const main = s[AUDIO_MAIN] || s.cnn;
  const itw = a.in_the_wild?.systems || {};
  const attacks = Object.keys(main.per_attack);
  const shown = ["cnn", "cnn_trim", "silence_lr"].filter((k) => s[k]);
  return (
    <article className="sb-card bench-card" aria-labelledby="bench-audio-title">
      <p className="sb-label">Audio branch · ASVspoof 2019 LA (ODC-By)</p>
      <h3 className="sb-h3" id="bench-audio-title">
        Speaker-disjoint eval with 11 attack systems never seen in training
      </h3>
      <Stats
        items={[
          [pct(main.eval_eer, 2), "Eval EER", `${main.label}; lower is better`],
          [fix(main.eval_min_tdcf, 3), "Eval min t-DCF", "ASVspoof 2019 cost model"],
          [pct(s.silence_lr?.eval_eer, 1), "Silence-only baseline EER", "no speech features at all"],
          [pct(itw[AUDIO_MAIN] ?? itw.cnn, 1), "In-the-Wild EER", "cross-dataset, no tuning"],
        ]}
      />
      <DotPlot
        title="Per-attack EER on eval"
        rows={attacks.map((k) => ({ key: k, label: k }))}
        series={shown.map((k) => ({ key: k, label: s[k].label }))}
        value={(row, sys) => s[sys].per_attack[row]}
        domain={[0, 0.6]}
        format={(v) => pct(v, 1)}
        axisLabel="Equal error rate (lower is better). A16 and A19 reuse training-set systems A04 and A06."
        refLine={{ value: 0.5, label: "chance" }}
      />
      {a.robustness && (
        <DotPlot
          title="Robustness: EER under perturbation"
          rows={Object.keys(a.robustness).map((k) => ({ key: k, label: k.replaceAll("_", " ") }))}
          series={["cnn", "cnn_trim"]
            .filter((k) => k in Object.values(a.robustness)[0])
            .map((k) => ({ key: k, label: s[k].label }))}
          value={(row, sys) => a.robustness[row][sys]}
          domain={[0, 0.6]}
          format={(v) => pct(v, 1)}
          axisLabel="Stratified eval subset; same models, perturbed input."
          refLine={{ value: 0.5, label: "chance" }}
        />
      )}
      <Provenance meta={a.meta} path="results/audio/asvspoof2019_la" />
    </article>
  );
}

const TYPE_LABEL = { inpainting: "inpainting (SD)", insight: "face swap (InsightFace)", text2img: "text-to-image (SD)" };

function VisualBench({ v }) {
  const s = v.systems;
  const main = s.effnet_ft || s.effnet_probe;
  const logo = s.effnet_ft_no_insight;
  const shown = ["meta_lr", "effnet_probe", "effnet_ft", "effnet_ft_no_insight"].filter((k) => s[k]);
  return (
    <article className="sb-card bench-card" aria-labelledby="bench-visual-title">
      <p className="sb-label">Visual branch · DeepFakeFace (image level)</p>
      <h3 className="sb-h3" id="bench-visual-title">
        Identity-disjoint split; one face-swap generator held out
      </h3>
      <Stats
        items={[
          [fix(main.test_auc, 3), "Test ROC-AUC", main.label],
          [pct(main.test_eer, 1), "Test EER", "threshold-free error rate"],
          [fix(logo?.per_type_auc?.insight, 3), "Unseen face-swap AUC", "model never saw InsightFace swaps"],
          [fix(s.meta_lr?.test_auc, 3), "Metadata-only AUC", "shortcut probe: no pixels"],
        ]}
      />
      <DotPlot
        title="Per-manipulation ROC-AUC on test"
        rows={Object.keys(TYPE_LABEL).map((k) => ({ key: k, label: TYPE_LABEL[k] }))}
        series={shown.map((k) => ({ key: k, label: s[k].label }))}
        value={(row, sys) => s[sys].per_type_auc[row]}
        domain={[0.4, 1]}
        format={(x) => fix(x, 3)}
        axisLabel="Real test photos against one manipulation type (higher is better)."
        refLine={{ value: 0.5, label: "chance" }}
      />
      {v.robustness && (
        <DotPlot
          title="Robustness: ROC-AUC under perturbation"
          rows={Object.keys(v.robustness).map((k) => ({ key: k, label: k.replaceAll("_", " ") }))}
          series={["effnet_probe", "effnet_ft"]
            .filter((k) => k in Object.values(v.robustness)[0])
            .map((k) => ({ key: k, label: s[k].label }))}
          value={(row, sys) => v.robustness[row][sys]}
          domain={[0.4, 1]}
          format={(x) => fix(x, 3)}
          axisLabel="Stratified test subset; same models, perturbed input."
          refLine={{ value: 0.5, label: "chance" }}
        />
      )}
      <Provenance meta={v.meta} path="results/visual/dff" />
    </article>
  );
}

export default function BenchmarkResults({ published }) {
  return (
    <div className="bench">
      {data.audio ? <AudioBench a={data.audio} /> : <Pending what="Audio benchmark" />}
      {data.visual ? <VisualBench v={data.visual} /> : <Pending what="Visual benchmark" />}
      <div className="sb-callout sb-callout--warn fusion-note">
        <span className="callout-icon" aria-hidden="true">
          <AlertIcon size={16} />
        </span>
        <p>
          <strong>Audio-visual fusion is not evaluated.</strong> Every dataset with labelled, paired audio and video
          (FakeAVCeleb, LAV-DF, AV-Deepfake1M, DeepSpeak) needs a request form or gated access, so the two branches are
          trained and scored on different datasets. The API reports each modality separately; its fused number uses
          fixed heuristic weights and is marked as unvalidated.{" "}
          <a href={`${REPO_URL}/blob/main/docs/DATASETS.md`} target="_blank" rel="noreferrer">
            Dataset decisions <ArrowUpRight />
          </a>
        </p>
      </div>
      {published}
    </div>
  );
}

export const hasResults = Boolean(data.audio || data.visual);
export const benchmarkData = data;
