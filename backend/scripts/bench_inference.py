"""End-to-end CPU latency of the served models (the API's DeepfakeDetector, not a micro-benchmark).

    python scripts/bench_inference.py --config configs/serve.yaml --out ../results/inference/latency.json

Builds a 10 s 1280x720 test video from the public-domain fixture portrait (slow zoom, 25 fps)
with a tone as its audio track, then times ``predict_file`` for 8 / 16 / 32 sampled frames with
and without Grad-CAM, a single image, and 10 s / 60 s audio files. Reports the median of the
repeated runs, the per-stage breakdown of the median run and the process's peak RSS. The scores
on these inputs are meaningless; only the timings are reported.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from deepfake_detector.config import load_config
from deepfake_detector.inference.predictor import DeepfakeDetector
from deepfake_detector.preprocessing.audio import write_wav

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "astronaut.jpg"


def make_video(path: Path, seconds: int = 10) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg is required to build the test video")
    zoom = f"zoompan=z='min(zoom+0.0008,1.3)':d={seconds * 25}:s=1280x720:fps=25"
    cmd = [ffmpeg, "-v", "error", "-y", "-loop", "1", "-i", str(FIXTURE), "-f", "lavfi", "-i",
           f"sine=frequency=220:sample_rate=16000:duration={seconds}", "-vf", zoom, "-t", str(seconds),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)]  # fmt: skip
    subprocess.run(cmd, check=True, timeout=300)


def run(det: DeepfakeDetector, path: Path, reps: int) -> dict:
    det.predict_file(path)  # warm-up (graph tracing, caches)
    results = []
    for _ in range(reps):
        t0 = time.perf_counter()
        res = det.predict_file(path)
        results.append(((time.perf_counter() - t0) * 1000.0, res))
    results.sort(key=lambda r: r[0])
    ms, median = results[len(results) // 2]
    return {
        "median_ms": round(ms, 1),
        "min_ms": round(results[0][0], 1),
        "max_ms": round(results[-1][0], 1),
        "reps": reps,
        "stages_ms_of_median_run": median["timings_ms"],
        "mode": median["mode"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()
    cfg = load_config(args.config)
    report: dict = {
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": os.environ.get("GITHUB_SHA", "local"),
        "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
        "runner": f"{platform.system()} {platform.machine()} cpus={os.cpu_count()}",
        "config": args.config,
        "backbones": cfg.model.backbones,
        "cases": {},
    }
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        video = tmp_path / "clip.mp4"
        make_video(video)
        wav10, wav60 = tmp_path / "a10.wav", tmp_path / "a60.wav"
        for p, sec in ((wav10, 10), (wav60, 60)):
            t = np.arange(16000 * sec) / 16000
            write_wav(p, 0.3 * np.sin(2 * np.pi * 180 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t)), 16000)
        t0 = time.perf_counter()
        det = DeepfakeDetector(cfg)
        report["model_load_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        report["model_info"] = {k: v for k, v in det.info().items() if k in ("mode", "models")}
        for frames in (8, 16, 32):
            for gradcam in (False, True):
                cfg.api.frames_per_video, cfg.api.gradcam = frames, gradcam
                name = f"video_10s_{frames}frames{'_gradcam' if gradcam else ''}"
                report["cases"][name] = run(det, video, args.reps)
                print(name, report["cases"][name]["median_ms"], flush=True)
        cfg.api.gradcam = True
        report["cases"]["image_gradcam"] = run(det, FIXTURE, args.reps)
        for name, p in (("audio_10s", wav10), ("audio_60s", wav60)):
            report["cases"][name] = run(det, p, args.reps)
            print(name, report["cases"][name]["median_ms"], flush=True)
    report["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
