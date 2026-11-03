"""Command-line inference.

python -m deepfake_detector.inference.cli path/to/video.mp4 --output result.json --gradcam cam.png
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

from ..config import load_config
from ..utils.logging import configure_logging
from .predictor import DeepfakeDetector


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deepfake detection on an image, video or audio file.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--config", default=None)
    parser.add_argument("--weights-dir", default=None, help="override api.weights_dir")
    parser.add_argument("--output", type=Path, default=None, help="write the full JSON result here")
    parser.add_argument("--gradcam", type=Path, default=None, help="save the Grad-CAM overlay PNG here")
    args = parser.parse_args(argv)

    configure_logging(json_logs=False)
    cfg = load_config(args.config)
    if args.weights_dir:
        cfg.api.weights_dir = args.weights_dir
    detector = DeepfakeDetector(cfg)
    result = detector.predict_file(args.path)

    if args.gradcam and result.get("gradcam"):
        data = result["gradcam"]["overlay"].split(",", 1)[1]
        args.gradcam.write_bytes(base64.b64decode(data))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2))

    summary = {k: result[k] for k in ("media_type", "mode", "label", "probability_fake", "processing_ms")}
    visual = result["modalities"]["visual"]
    if visual:
        summary["per_model"] = {k: v["probability"] for k, v in visual["per_model"].items()}
    audio = result["modalities"]["audio"]
    if audio:
        summary["audio"] = audio.get("probability") if audio.get("available") else audio.get("reason")
    if result["warning"]:
        print(f"WARNING: {result['warning']}", file=sys.stderr)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
