"""Build benchmark figures, Markdown tables and the site's data file from committed results.

    python scripts/make_report.py [--results ../results] [--figures ../docs/figures]
                                  [--site ../frontend/src/data/benchmarks.json]

Reads only files that benchmark workflows committed (results/audio/..., results/visual/...), so
every number in docs/BENCHMARKS.md, the README and the website traces back to a run ID and
commit. Figures use the site's validated chart palette on a light chart surface.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Validated categorical order (scripts/validate_palette: lightness, chroma, CVD, contrast all pass).
SERIES = ["#0f9e6a", "#e0662f", "#2f5bd3", "#c23a5c"]
MARKERS = ["o", "s", "D", "^"]
INK, MUTED, GRID = "#0c0f14", "#5b6573", "#e6e7ea"
AUDIO_NAMES = {
    "silence_lr": "Silence/duration only (LR)",
    "logmel_lr": "Log-mel statistics (LR)",
    "cnn": "Log-mel CNN",
    "cnn_trim": "Log-mel CNN, silence trimmed",
    "cnn_on_trimmed_audio": "Log-mel CNN on trimmed audio",
}
VISUAL_NAMES = {
    "meta_lr": "File metadata only (LR)",
    "effnet_probe": "EfficientNetV2-B0 frozen + LR",
    "effnet_ft": "EfficientNetV2-B0 fine-tuned",
    "effnet_ft_no_insight": "Fine-tuned without face swaps",
    "effnet_probe_no_insight": "Frozen + LR without face swaps",
    "effnet_probe_no_inpainting": "Frozen + LR without inpainting",
    "effnet_probe_no_text2img": "Frozen + LR without text2img",
}


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def pct(v: float | None, d: int = 2) -> str:
    return "n/a" if v is None else f"{100 * v:.{d}f}%"


def num(v: float | None, d: int = 4) -> str:
    return "n/a" if v is None else f"{v:.{d}f}"


# --------------------------------------------------------------------------- figures
def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff", "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": INK, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "font.size": 9.5, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    })  # fmt: skip
    return plt


def dot_plot(
    rows: list[str],
    series: dict[str, list[float | None]],
    xlabel: str,
    title: str,
    path: Path,
    *,
    xlim: tuple[float, float] | None = None,
    ref: float | None = None,
    ref_label: str = "",
) -> None:
    """Horizontal dot plot: one row per category, one marker shape + colour per series."""
    plt = _style()
    height = 0.9 + 0.32 * len(rows)
    fig, ax = plt.subplots(figsize=(7.2, height))
    n = len(series)
    for k, (name, vals) in enumerate(series.items()):
        offset = (k - (n - 1) / 2) * 0.16
        ys = [i + offset for i, v in enumerate(vals) if v is not None]
        xs = [v for v in vals if v is not None]
        ax.scatter(xs, ys, s=34, color=SERIES[k], marker=MARKERS[k], label=name, zorder=3,
                   edgecolors="#ffffff", linewidths=1.2)  # fmt: skip
    if ref is not None:
        ax.axvline(ref, color=MUTED, linewidth=1, linestyle=(0, (3, 3)), zorder=2)
        ax.text(ref, len(rows) - 0.4, f" {ref_label}", color=MUTED, fontsize=8, va="bottom")
    ax.set_yticks(range(len(rows)), rows)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18 - 0.6 / height), ncol=min(n, 3), fontsize=8.5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- audio
def audio_section(res: Path, figs: Path) -> tuple[str, dict]:
    m = _load(res / "metrics.json")
    if m is None:
        return "", {}
    rob, lat = _load(res / "robustness.json"), _load(res / "latency.json")
    itw, audit = _load(res / "cross_dataset_in_the_wild.json"), _load(res / "data_audit.json")
    S = m["systems"]
    meta = m["meta"]
    lines = [
        f"_Run {meta['run_id']} at commit `{meta['git_sha'][:7]}` on {meta['date_utc'][:10]} "
        f"({meta['runner']}); scores in `results/audio/asvspoof2019_la/eval_scores.csv.gz`._",
        "",
        "| System | Dev EER | Eval EER | Eval min t-DCF | Eval ROC-AUC | Eval PR-AUC (spoof) | FPR @ dev thr. "
        "| FNR @ dev thr. | ECE raw → Platt |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in S.items():
        e = s["eval"]
        cal = e["calibration"]
        label = AUDIO_NAMES.get(name, name)
        lines.append(
            f"| {label} | {pct(s['dev']['eer'])} | **{pct(e['eer'])}** | {num(e.get('min_tdcf'))} "
            f"| {num(e['roc_auc'])} | {num(e['pr_auc_spoof'])} | {pct(e['at_dev_threshold']['fpr'])} "
            f"| {pct(e['at_dev_threshold']['fnr'])} | {num(cal['raw']['ece'], 3)} → {num(cal['platt']['ece'], 3)} |"
        )
    site: dict = {"meta": meta, "systems": {}}
    for name, s in S.items():
        site["systems"][name] = {
            "label": AUDIO_NAMES.get(name, name),
            "dev_eer": s["dev"]["eer"],
            "eval_eer": s["eval"]["eer"],
            "eval_min_tdcf": s["eval"].get("min_tdcf"),
            "eval_auc": s["eval"]["roc_auc"],
            "per_attack": {a: v["eer"] for a, v in s["eval"]["per_attack"].items()},
        }
    attacks = sorted(next(iter(S.values()))["eval"]["per_attack"])
    shown = [k for k in ("cnn", "cnn_trim", "silence_lr") if k in S]
    rows = [f"{a}{' (seen)' if S[shown[0]]['eval']['per_attack'][a]['seen_in_training'] else ''}" for a in attacks]
    dot_plot(rows, {AUDIO_NAMES[k]: [100 * S[k]["eval"]["per_attack"][a]["eer"] for a in attacks] for k in shown},
             "EER on ASVspoof 2019 LA eval (%, lower is better)", "Per-attack EER (eval: unseen speakers)",
             figs / "asvspoof_per_attack.png", ref=50, ref_label="chance")  # fmt: skip
    lines += ["", "Per-attack EER (eval):", "", "| Attack | " + " | ".join(AUDIO_NAMES[k] for k in shown) + " |",
              "|---|" + "---|" * len(shown)]  # fmt: skip
    for a in attacks:
        lines.append(f"| {a} | " + " | ".join(pct(S[k]["eval"]["per_attack"][a]["eer"]) for k in shown) + " |")
    if rob:
        conds = list(rob["conditions"])
        systems = [k for k in ("cnn", "cnn_trim") if k in rob["conditions"][conds[0]]]
        dot_plot(conds, {AUDIO_NAMES[k]: [100 * rob["conditions"][c][k]["eer"] for c in conds] for k in systems},
                 "EER (%) on a stratified eval subset", "Robustness of the audio CNNs",
                 figs / "asvspoof_robustness.png", ref=50, ref_label="chance")  # fmt: skip
        lines += ["", f"Robustness (stratified eval subset, n = {rob['subset']['n']}):", "",
                  "| Condition | " + " | ".join(f"{AUDIO_NAMES[k]} EER" for k in systems)
                  + " | " + " | ".join(f"{AUDIO_NAMES[k]} FPR @ dev thr." for k in systems) + " |",
                  "|---|" + "---|" * (2 * len(systems))]  # fmt: skip
        for c in conds:
            r = rob["conditions"][c]
            lines.append(f"| {c} | " + " | ".join(pct(r[k]["eer"]) for k in systems) + " | "
                         + " | ".join(pct(r[k]["fpr_at_dev_threshold"]) for k in systems) + " |")  # fmt: skip
        site["robustness"] = {c: {k: rob["conditions"][c][k]["eer"] for k in systems} for c in conds}
    if itw:
        head = f"Cross-dataset: In-the-Wild ({itw['n']:,} clips, {itw['n_speakers']} speakers; no tuning on it):"
        lines += ["", head,
                  "", "| System | EER | ROC-AUC | FPR @ ASVspoof-dev thr. | FNR @ ASVspoof-dev thr. |",
                  "|---|---|---|---|---|"]  # fmt: skip
        for name, s in itw["systems"].items():
            t = s["at_asvspoof_dev_threshold"]
            lines.append(f"| {AUDIO_NAMES.get(name, name)} | {pct(s['eer'])} | {num(s['roc_auc'])} | {pct(t['fpr'])} "
                         f"| {pct(t['fnr'])} |")  # fmt: skip
        site["in_the_wild"] = {"n": itw["n"], "systems": {k: v["eer"] for k, v in itw["systems"].items()}}
    if lat:
        site["latency"] = {k: lat[k] for k in ("stages", "peak_rss_mb") if k in lat}
        for k in ("cnn", "cnn_trim"):
            if k in lat:
                site["latency"][k] = {
                    "params": lat[k]["params"],
                    "keras_batch1_ms": lat[k]["keras_batch1"]["mean_ms"],
                    "tflite_fp32_ms": lat[k]["tflite_fp32"]["batch1"]["mean_ms"],
                    "tflite_int8_ms": lat[k]["tflite_int8_dynamic"]["batch1"]["mean_ms"],
                    "tflite_int8_bytes": lat[k]["tflite_int8_dynamic"]["file_bytes"],
                    "parity": lat[k].get("parity_subset"),
                }
    if audit:
        site["audit"] = {s: {k: v[k] for k in ("n", "n_bonafide", "n_spoof", "n_speakers")}
                         for s, v in audit["splits"].items()}  # fmt: skip
        site["audit"]["speaker_overlap"] = audit["speaker_overlap"]
    return "\n".join(lines) + "\n", site


# --------------------------------------------------------------------------- visual
def visual_section(res: Path, figs: Path) -> tuple[str, dict]:
    m = _load(res / "metrics.json")
    if m is None:
        return "", {}
    rob, lat, audit = _load(res / "robustness.json"), _load(res / "latency.json"), _load(res / "data_audit.json")
    S = m["systems"]
    meta = m["meta"]
    types = ["inpainting", "insight", "text2img"]
    lines = [
        f"_Run {meta['run_id']} at commit `{meta['git_sha'][:7]}` on {meta['date_utc'][:10]} "
        f"({meta['runner']}); scores in `results/visual/dff/test_scores.csv.gz`._",
        "",
        "| System | Val AUC | Test ROC-AUC | Test EER | PR-AUC (fake) | FPR @ val thr. | FNR @ val thr. "
        "| AUC inpainting | AUC face swap | AUC text2img |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in S.items():
        t = s["test"]
        pt = t["per_type"]

        def cell(k: str, pt=pt) -> str:
            mark = "" if pt[k]["seen_in_training"] else " (unseen)"
            return f"{num(pt[k]['roc_auc'])}{mark}"

        label = VISUAL_NAMES.get(name, name)
        lines.append(
            f"| {label} | {num(s['val']['roc_auc'])} | **{num(t['roc_auc'])}** | {pct(t['eer'])} "
            f"| {num(t['pr_auc_fake'])} | {pct(t['at_val_threshold']['fpr'])} | {pct(t['at_val_threshold']['fnr'])} "
            f"| {cell('inpainting')} | {cell('insight')} | {cell('text2img')} |"
        )
    shown = [k for k in ("meta_lr", "effnet_probe", "effnet_ft", "effnet_ft_no_insight") if k in S]
    dot_plot([{"insight": "insight (face swap)"}.get(t, t) for t in types],
             {VISUAL_NAMES[k]: [S[k]["test"]["per_type"][t]["roc_auc"] for t in types] for k in shown},
             "Test ROC-AUC, real vs. one manipulation type (higher is better)", "DeepFakeFace: per-manipulation AUC",
             figs / "dff_per_type.png", xlim=(0.4, 1.0), ref=0.5, ref_label="chance")  # fmt: skip
    site: dict = {"meta": meta, "systems": {}}
    for name, s in S.items():
        site["systems"][name] = {
            "label": VISUAL_NAMES.get(name, name),
            "held_out": s.get("held_out_type"),
            "test_auc": s["test"]["roc_auc"],
            "test_eer": s["test"]["eer"],
            "per_type_auc": {t: s["test"]["per_type"][t]["roc_auc"] for t in types},
        }
    if rob:
        conds = list(rob["conditions"])
        systems = [k for k in ("effnet_probe", "effnet_ft") if k in rob["conditions"][conds[0]]]
        dot_plot(conds, {VISUAL_NAMES[k]: [rob["conditions"][c][k]["roc_auc"] for c in conds] for k in systems},
                 "ROC-AUC on a stratified test subset", "Robustness of the visual models",
                 figs / "dff_robustness.png", xlim=(0.4, 1.0), ref=0.5, ref_label="chance")  # fmt: skip
        lines += ["", f"Robustness (test subset, n = {rob['subset']['n']}):", "",
                  "| Condition | " + " | ".join(f"{VISUAL_NAMES[k]} AUC" for k in systems) + " | "
                  + " | ".join(f"{VISUAL_NAMES[k]} FPR @ val thr." for k in systems) + " |",
                  "|---|" + "---|" * (2 * len(systems))]  # fmt: skip
        for c in conds:
            r = rob["conditions"][c]
            lines.append(f"| {c} | " + " | ".join(num(r[k]["roc_auc"]) for k in systems) + " | "
                         + " | ".join(pct(r[k]["fpr_at_val_threshold"]) for k in systems) + " |")  # fmt: skip
        site["robustness"] = {c: {k: rob["conditions"][c][k]["roc_auc"] for k in systems} for c in conds}
    if lat:
        site["latency"] = {
            "face_detect_align_crop_ms": lat["face_detect_align_crop"]["mean_ms"],
            "keras_batch1_ms": lat["keras_batch1"]["mean_ms"],
            "keras_batch16_per_image_ms": lat["keras_batch16_per_image_ms"],
            "gradcam_ms": lat["gradcam_per_image"]["mean_ms"],
            "exports": {k: {"ms": v["batch1"]["mean_ms"], "bytes": v["file_bytes"],
                            "max_abs_prob_diff": v["max_abs_prob_diff_vs_keras"]} for k, v in lat["exports"].items()},
        }  # fmt: skip
    if audit:
        site["audit"] = {s: {k: v[k] for k in ("n", "n_real", "n_fake", "n_groups")}
                         for s, v in audit["selected"].items()}  # fmt: skip
        site["audit"]["group_overlap"] = audit["group_overlap"]
    return "\n".join(lines) + "\n", site


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(ROOT / "results"))
    parser.add_argument("--figures", default=str(ROOT / "docs/figures"))
    parser.add_argument("--site", default=str(ROOT / "frontend/src/data/benchmarks.json"))
    parser.add_argument("--tables", default=str(ROOT / "docs/benchmark_tables.md"))
    args = parser.parse_args()
    res, figs = Path(args.results), Path(args.figures)
    a_md, a_site = audio_section(res / "audio/asvspoof2019_la", figs)
    v_md, v_site = visual_section(res / "visual/dff", figs)
    tables = "<!-- generated by backend/scripts/make_report.py from results/ - do not edit -->\n\n"
    if a_md:
        tables += "## Audio: ASVspoof 2019 LA\n\n" + a_md + "\n"
    if v_md:
        tables += "## Visual: DeepFakeFace\n\n" + v_md
    Path(args.tables).write_text(tables)
    site = {"audio": a_site or None, "visual": v_site or None}
    Path(args.site).parent.mkdir(parents=True, exist_ok=True)
    Path(args.site).write_text(json.dumps(site, indent=1, default=float) + "\n")
    print(tables)


if __name__ == "__main__":
    main()
