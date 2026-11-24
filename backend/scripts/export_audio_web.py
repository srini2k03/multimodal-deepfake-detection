"""Export the audio CNN for the browser demo and write a numerical parity fixture.

    python scripts/export_audio_web.py --model weights/audio_cnn.keras --out ../frontend/public/models/audio \
        [--config configs/serve.yaml] [--meta results/audio/asvspoof2019_la/metrics.json --system cnn_trim]

Writes ``audio_cnn.json`` (manifest: audio front-end settings, mel filterbank and layer table),
``audio_cnn.bin`` (little-endian float32 weights, BatchNorm folded into the convolutions) and
``audio_fixture.json`` (a deterministic test signal with the Python log-mel and per-clip logits).
``frontend/scripts/audio-parity.mjs`` re-computes the fixture in JavaScript and fails if the
browser implementation drifts from Python. The folding itself is checked here against Keras.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepfake_detector.config import AudioConfig, load_config
from deepfake_detector.preprocessing.audio import (
    audio_to_mel_clips,
    mel_filterbank,
)

FORMAT = "dfd-audio-cnn-v1"
AUDIO_KEYS = (
    "sample_rate", "n_fft", "hop_length", "n_mels", "fmin", "fmax", "clip_seconds", "pad_mode", "trim_silence",
    "trim_top_db",
)  # fmt: skip


def fold_layers(model) -> list[dict]:
    """Turn the Keras audio CNN into a flat op list with BatchNorm folded into the convolutions."""
    import keras

    ops: list[dict] = []
    layers = [layer for layer in model.layers if not isinstance(layer, keras.layers.InputLayer)]
    i = 0
    while i < len(layers):
        layer = layers[i]
        if isinstance(layer, keras.layers.Conv2D):
            w = layer.get_weights()
            kernel = w[0].astype(np.float64)  # (kh, kw, cin, cout)
            bias = w[1].astype(np.float64) if layer.use_bias else np.zeros(kernel.shape[-1])
            relu = layer.get_config().get("activation") == "relu"
            nxt = layers[i + 1] if i + 1 < len(layers) else None
            if isinstance(nxt, keras.layers.BatchNormalization):
                gamma, beta, mean, var = (a.astype(np.float64) for a in nxt.get_weights())
                scale = gamma / np.sqrt(var + nxt.epsilon)
                kernel = kernel * scale
                bias = (bias - mean) * scale + beta
                i += 1
            ops.append({"type": "conv3x3", "name": layer.name, "kernel": kernel, "bias": bias, "relu": relu})
        elif isinstance(layer, keras.layers.Activation):
            act = layer.get_config()["activation"]
            if act == "relu":
                ops[-1]["relu"] = True
            elif act != "sigmoid":
                raise ValueError(f"unsupported activation {act}")
        elif isinstance(layer, keras.layers.MaxPooling2D):
            ops.append({"type": "maxpool2", "name": layer.name})
        elif isinstance(layer, keras.layers.GlobalAveragePooling2D):
            ops.append({"type": "gap", "name": layer.name})
        elif isinstance(layer, keras.layers.Dense):
            w, b = layer.get_weights()
            relu = layer.get_config().get("activation") == "relu"
            ops.append({"type": "dense", "name": layer.name, "kernel": w.astype(np.float64), "bias": b, "relu": relu})
        elif isinstance(layer, (keras.layers.Dropout, keras.layers.SpatialDropout2D)):
            pass  # identity at inference
        else:
            raise ValueError(f"unsupported layer {layer.__class__.__name__} ({layer.name})")
        i += 1
    return ops


def numpy_forward(ops: list[dict], x: np.ndarray) -> float:
    """Reference forward pass over ``(n_mels, frames)`` in float64 (channels-last)."""
    h = x[:, :, None].astype(np.float64)
    for op in ops:
        if op["type"] == "conv3x3":
            k = op["kernel"]
            p = np.pad(h, ((1, 1), (1, 1), (0, 0)))
            out = np.zeros((h.shape[0], h.shape[1], k.shape[3]))
            for dy in range(3):
                for dx in range(3):
                    out += p[dy : dy + h.shape[0], dx : dx + h.shape[1], :] @ k[dy, dx]
            h = out + op["bias"]
        elif op["type"] == "maxpool2":
            hh, ww = h.shape[0] // 2 * 2, h.shape[1] // 2 * 2
            h = h[:hh, :ww].reshape(hh // 2, 2, ww // 2, 2, -1).max(axis=(1, 3))
        elif op["type"] == "gap":
            h = h.mean(axis=(0, 1))
        elif op["type"] == "dense":
            h = h @ op["kernel"] + op["bias"]
        if op.get("relu"):
            h = np.maximum(h, 0.0)
    return float(np.asarray(h).reshape(-1)[0])


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.asarray(a, dtype="<f4").tobytes()).decode("ascii")


def fixture_signal(sr: int, seconds: float = 6.5, seed: int = 7) -> np.ndarray:
    """Deterministic test signal: leading silence, a voiced chirp, noise bursts and a short tail."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    f0 = 110 + 90 * t / seconds
    voiced = 0.25 * np.sin(2 * np.pi * np.cumsum(f0) / sr) * (0.6 + 0.4 * np.sin(2 * np.pi * 2.5 * t))
    voiced += 0.08 * np.sin(2 * np.pi * 3 * np.cumsum(f0) / sr)
    x = voiced + 0.01 * rng.standard_normal(t.size)
    x[: int(0.35 * sr)] = 0.0005 * rng.standard_normal(int(0.35 * sr))
    return x.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--trim-silence", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--meta", default=None, help="benchmark metrics.json to copy headline numbers from")
    parser.add_argument("--system", default=None, help="system name inside --meta")
    parser.add_argument("--release", default=None)
    args = parser.parse_args()

    from deepfake_detector.inference.scorers import load_keras_model

    cfg = load_config(args.config)
    audio: AudioConfig = cfg.audio
    if args.trim_silence != "auto":
        audio.trim_silence = args.trim_silence == "true"
    model = load_keras_model(Path(args.model))
    ops = fold_layers(model)
    n_mels, n_frames = int(model.input_shape[1]), int(model.input_shape[2])

    # 1) folding check against Keras on random log-mels
    import keras

    logit_model = keras.Model(model.input, model.get_layer("logit").output)
    rng = np.random.default_rng(0)
    probe = rng.standard_normal((3, n_mels, n_frames)).astype(np.float32)
    keras_logits = np.asarray(logit_model(probe[..., None], training=False)).reshape(-1)
    np_logits = np.array([numpy_forward(ops, p) for p in probe])
    fold_err = float(np.max(np.abs(keras_logits - np_logits)))
    if fold_err > 1e-3:
        raise SystemExit(f"BatchNorm folding mismatch: {fold_err}")

    # 2) weights + manifest
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    blobs: list[bytes] = []
    offset = 0

    def put(a: np.ndarray) -> dict:
        nonlocal offset
        data = np.ascontiguousarray(a, dtype="<f4").tobytes()
        blobs.append(data)
        ref = {"offset": offset, "length": int(np.asarray(a).size), "shape": list(np.asarray(a).shape)}
        offset += len(data)
        return ref

    fb = mel_filterbank(audio.sample_rate, audio.n_fft, audio.n_mels, audio.fmin, audio.fmax)
    fb_ref = put(fb)
    layers = []
    for op in ops:
        entry = {"type": op["type"], "name": op["name"], "relu": bool(op.get("relu", False))}
        if op["type"] == "conv3x3":
            entry["kernel"] = put(np.transpose(op["kernel"], (3, 2, 0, 1)))  # -> (cout, cin, 3, 3)
            entry["bias"] = put(op["bias"])
        elif op["type"] == "dense":
            entry["kernel"] = put(op["kernel"].T)  # -> (out, in)
            entry["bias"] = put(op["bias"])
        layers.append(entry)
    weights = b"".join(blobs)
    (out / "audio_cnn.bin").write_bytes(weights)
    keras_sha = hashlib.sha256(Path(args.model).read_bytes()).hexdigest()
    manifest = {
        "format": FORMAT,
        "weights_file": "audio_cnn.bin",
        "weights_sha256": hashlib.sha256(weights).hexdigest(),
        "source_model_sha256": keras_sha,
        "input": {"n_mels": n_mels, "n_frames": n_frames},
        "audio": {k: getattr(audio, k) for k in AUDIO_KEYS},
        "mel_filterbank": fb_ref,
        "layers": layers,
        "fold_check_max_abs_logit_diff": fold_err,
    }
    if args.meta and args.system:
        m = json.loads(Path(args.meta).read_text())
        s = m["systems"][args.system]
        manifest["benchmark"] = {
            "dataset": "ASVspoof 2019 LA",
            "system": args.system,
            "eval_eer": s["eval"]["eer"],
            "eval_min_tdcf": s["eval"].get("min_tdcf"),
            "dev_threshold_logit": s["dev_threshold_logit"],
            "platt": s["platt"],
            "run_id": m["meta"]["run_id"],
            "git_sha": m["meta"]["git_sha"],
            "release": args.release,
        }
    (out / "audio_cnn.json").write_text(json.dumps(manifest, indent=1) + "\n")

    # 3) parity fixture: Python preprocessing + Keras logits for every clip of a test signal
    signal = fixture_signal(audio.sample_rate)
    clips = audio_to_mel_clips(signal, audio)
    logits = np.asarray(logit_model(clips, training=False)).reshape(-1)
    fixture = {
        "sample_rate": audio.sample_rate,
        "signal_f32": _b64(signal),
        "n_clips": len(clips),
        "first_clip_log_mel_f32": _b64(clips[0, :, :, 0]),
        "clip_logits": [float(v) for v in logits],
        "tolerance": {"log_mel": 2e-3, "logit": 2e-3},
    }
    (out / "audio_fixture.json").write_text(json.dumps(fixture) + "\n")
    summary = {"fold_check": fold_err, "n_clips": len(clips), "logits": fixture["clip_logits"], "bin_bytes": offset}
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
