"""Evaluate trained frame models, their ensemble and the temporal head.

    python -m deepfake_detector.evaluation.evaluate --split test --output reports/

Reports frame-level and video-level AUC, accuracy, precision/recall/F1, EER and the
confusion matrix for every model plus the weighted ensemble, and writes
``metrics.json``, ``metrics.md`` and (if matplotlib is installed) ``roc.png``.
``--tune-ensemble`` grid-searches ensemble weights on the validation split.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from ..config import Config, load_config
from ..inference.fusion import aggregate_frames
from ..utils.logging import configure_logging
from .metrics import binary_report, roc_curve

logger = logging.getLogger(__name__)


def predict_frames(model, paths: list[str], image_size: int, batch_size: int) -> np.ndarray:
    import tensorflow as tf

    from ..data.datasets import _load_png

    ds = (
        tf.data.Dataset.from_tensor_slices(paths)
        .map(lambda p: _load_png(p, image_size), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return np.asarray(model.predict(ds, verbose=0), dtype=np.float64).reshape(-1)


def collect_scores(cfg: Config, split: str) -> dict:
    """Return per-video labels and per-model frame score arrays."""
    from ..data.datasets import video_frame_groups
    from ..inference.scorers import load_keras_model

    groups = video_frame_groups(cfg, split)
    if not groups:
        raise SystemExit(f"No videos in split '{split}'")
    video_ids = sorted(groups)
    flat = [p for vid in video_ids for p in groups[vid][0]]
    offsets = np.cumsum([0] + [len(groups[v][0]) for v in video_ids])
    labels = np.array([groups[v][1] for v in video_ids])

    scores: dict[str, list[np.ndarray]] = {}
    for name in cfg.model.backbones:
        path = Path(cfg.api.weights_dir) / f"{name}.keras"
        if not path.exists():
            logger.warning("weights missing, skipping", extra={"model": name, "path": str(path)})
            continue
        frame_scores = predict_frames(load_keras_model(path), flat, cfg.data.image_size, cfg.data.batch_size)
        scores[name] = [frame_scores[offsets[i] : offsets[i + 1]] for i in range(len(video_ids))]
    if not scores:
        raise SystemExit(f"No trained weights found in {cfg.api.weights_dir}. Train models first.")

    temporal = None
    tpath = Path(cfg.api.weights_dir) / "temporal.keras"
    if tpath.exists():
        from ..data.datasets import sequence_dataset

        seq = sequence_dataset(cfg, split, training=False)
        tmodel = load_keras_model(tpath)
        temporal = np.asarray(tmodel.predict(seq.dataset, verbose=0)).reshape(-1)
        # sequence_dataset iterates videos in manifest order; re-align by sorted id
        seq_ids = list(groups)
        order = {vid: i for i, vid in enumerate(seq_ids)}
        temporal = np.array([temporal[order[v]] for v in video_ids])
    return {"video_ids": video_ids, "labels": labels, "scores": scores, "temporal": temporal}


def _ensemble(scores: dict[str, list[np.ndarray]], weights: dict[str, float]) -> list[np.ndarray]:
    names = list(scores)
    total = sum(weights.get(n, 0.0) for n in names) or float(len(names))
    n_videos = len(scores[names[0]])
    return [sum(scores[n][i] * weights.get(n, 1.0) for n in names) / total for i in range(n_videos)]


def evaluate(cfg: Config, split: str, aggregation: str, threshold: float) -> tuple[dict, dict]:
    data = collect_scores(cfg, split)
    labels = data["labels"]
    weights = dict(zip(cfg.model.backbones, cfg.model.ensemble_weights, strict=True))
    per_model = dict(data["scores"])
    if len(per_model) > 1:
        per_model["ensemble"] = _ensemble(data["scores"], weights)

    report: dict = {"split": split, "aggregation": aggregation, "n_videos": len(labels), "models": {}}
    curves: dict = {}
    for name, vids in per_model.items():
        frame_y = np.concatenate([np.full(len(s), y) for s, y in zip(vids, labels, strict=True)])
        frame_s = np.concatenate(vids)
        video_s = np.array([aggregate_frames(s, aggregation) for s in vids])
        report["models"][name] = {
            "frame_level": binary_report(frame_y, frame_s, threshold).to_dict(),
            "video_level": binary_report(labels, video_s, threshold).to_dict(),
        }
        curves[name] = roc_curve(labels, video_s)
    if data["temporal"] is not None:
        report["models"]["temporal"] = {"video_level": binary_report(labels, data["temporal"], threshold).to_dict()}
        curves["temporal"] = roc_curve(labels, data["temporal"])
    return report, curves


def tune_ensemble(cfg: Config, aggregation: str, steps: int = 11) -> dict:
    data = collect_scores(cfg, "val")
    names = list(data["scores"])
    if len(names) != 2:
        raise SystemExit("--tune-ensemble currently supports exactly two backbones")
    from .metrics import roc_auc

    results = []
    for w in np.linspace(0.0, 1.0, steps):
        ens = _ensemble(data["scores"], {names[0]: w, names[1]: 1.0 - w})
        auc = roc_auc(data["labels"], [aggregate_frames(s, aggregation) for s in ens])
        results.append({"weights": {names[0]: round(float(w), 3), names[1]: round(float(1 - w), 3)}, "val_auc": auc})
    best = max(results, key=lambda r: r["val_auc"])
    return {"grid": results, "best": best}


def to_markdown(report: dict) -> str:
    lines = [
        f"### Evaluation on `{report['split']}` ({report['n_videos']} videos, aggregation: {report['aggregation']})",
        "",
        "| Model | Level | AUC | Acc | F1 | EER |",
        "|---|---|---|---|---|---|",
    ]
    for name, levels in report["models"].items():
        for level, m in levels.items():
            lines.append(
                f"| {name} | {level.replace('_level', '')} | {m['auc']:.4f} | {m['accuracy']:.4f} "
                f"| {m['f1']:.4f} | {m['eer']:.4f} |"
            )
    return "\n".join(lines) + "\n"


def plot_roc(curves: dict, path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, (fpr, tpr, _) in curves.items():
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Video-level ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--aggregation", default=None, help="frame->video aggregation (default: api.frame_aggregation)")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--tune-ensemble", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(json_logs=False)
    cfg = load_config(args.config)
    aggregation = args.aggregation or cfg.api.frame_aggregation
    threshold = args.threshold if args.threshold is not None else cfg.fusion.threshold
    args.output.mkdir(parents=True, exist_ok=True)

    if args.tune_ensemble:
        tuned = tune_ensemble(cfg, aggregation)
        (args.output / "ensemble_tuning.json").write_text(json.dumps(tuned, indent=2))
        print(json.dumps(tuned["best"], indent=2))
        return

    report, curves = evaluate(cfg, args.split, aggregation, threshold)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2))
    md = to_markdown(report)
    (args.output / "metrics.md").write_text(md)
    if plot_roc(curves, args.output / "roc.png"):
        logger.info("saved ROC curve to %s", args.output / "roc.png")
    print(md)


if __name__ == "__main__":
    main()
