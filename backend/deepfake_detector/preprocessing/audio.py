"""Audio loading and log-mel spectrogram features implemented with NumPy only.

WAV files are read with the standard library. Other formats (and audio tracks inside
videos) are decoded with ``ffmpeg`` when it is available on ``PATH``.
"""

from __future__ import annotations

import logging
import subprocess
import wave
from pathlib import Path

import numpy as np

from ..config import AudioConfig
from ..utils.optional import ffmpeg_path

logger = logging.getLogger(__name__)


class AudioUnavailableError(RuntimeError):
    """Raised when a file has no decodable audio track."""


# --------------------------------------------------------------------------- loading
def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise AudioUnavailableError(f"Unsupported WAV sample width: {width}")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, sr


def _decode_ffmpeg(path: Path, sample_rate: int) -> np.ndarray:
    exe = ffmpeg_path()
    if exe is None:
        raise AudioUnavailableError("ffmpeg not found; only .wav audio can be decoded")
    cmd = [exe, "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "s16le", "-"]
    proc = subprocess.run(cmd, capture_output=True, timeout=120, check=False)
    if proc.returncode != 0 or not proc.stdout:
        raise AudioUnavailableError("no decodable audio stream found")
    return np.frombuffer(proc.stdout, dtype="<i2").astype(np.float32) / 32768.0


def resample(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Linear-interpolation resampling (adequate for spectrogram features)."""
    if orig_sr == target_sr or signal.size == 0:
        return signal.astype(np.float32)
    duration = signal.size / orig_sr
    n_out = max(1, round(duration * target_sr))
    x_old = np.linspace(0.0, duration, signal.size, endpoint=False)
    x_new = np.linspace(0.0, duration, n_out, endpoint=False)
    return np.interp(x_new, x_old, signal).astype(np.float32)


def load_audio(path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    """Return a mono float32 waveform in [-1, 1] at ``sample_rate``."""
    path = Path(path)
    if path.suffix.lower() == ".wav":
        try:
            data, sr = _read_wav(path)
            return resample(data, sr, sample_rate)
        except (wave.Error, EOFError) as exc:
            logger.debug("stdlib wav reader failed, trying ffmpeg: %s", exc)
    return _decode_ffmpeg(path, sample_rate)


def write_wav(path: str | Path, signal: np.ndarray, sample_rate: int) -> None:
    pcm = (np.clip(signal, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


# --------------------------------------------------------------------------- features
def hz_to_mel(hz: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Triangular HTK-style mel filterbank of shape ``(n_mels, n_fft // 2 + 1)``."""
    fmax = min(fmax, sample_rate / 2.0)
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)
    bank = np.zeros((n_mels, fft_freqs.size), dtype=np.float32)
    for m in range(n_mels):
        lo, centre, hi = hz[m], hz[m + 1], hz[m + 2]
        up = (fft_freqs - lo) / max(centre - lo, 1e-9)
        down = (hi - fft_freqs) / max(hi - centre, 1e-9)
        bank[m] = np.maximum(0.0, np.minimum(up, down))
    # Slaney-style area normalisation keeps band energies comparable.
    enorm = 2.0 / np.maximum(hz[2 : n_mels + 2] - hz[:n_mels], 1e-9)
    return bank * enorm[:, None].astype(np.float32)


def stft_power(signal: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    """Power spectrogram ``(n_fft // 2 + 1, frames)`` with a Hann window and centre padding."""
    pad = n_fft // 2
    padded = np.pad(signal.astype(np.float32), (pad, pad), mode="reflect" if signal.size > pad else "constant")
    n_frames = 1 + max(0, (padded.size - n_fft) // hop_length)
    idx = np.arange(n_fft)[None, :] + hop_length * np.arange(n_frames)[:, None]
    frames = padded[idx] * np.hanning(n_fft).astype(np.float32)[None, :]
    return (np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2).T.astype(np.float32)


def log_mel_spectrogram(signal: np.ndarray, cfg: AudioConfig) -> np.ndarray:
    """Per-utterance standardised log-mel spectrogram ``(n_mels, frames)``."""
    power = stft_power(signal, cfg.n_fft, cfg.hop_length)
    mel = mel_filterbank(cfg.sample_rate, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax) @ power
    log_mel = np.log(mel + 1e-6)
    return ((log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)).astype(np.float32)


def frames_per_clip(cfg: AudioConfig) -> int:
    return 1 + int(cfg.clip_seconds * cfg.sample_rate) // cfg.hop_length


def segment_waveform(signal: np.ndarray, cfg: AudioConfig, min_fraction: float = 0.5) -> list[np.ndarray]:
    """Split into fixed-length clips; the final partial clip is zero-padded if long enough."""
    clip_len = int(cfg.clip_seconds * cfg.sample_rate)
    if signal.size == 0:
        return []
    segments = []
    for start in range(0, signal.size, clip_len):
        seg = signal[start : start + clip_len]
        if seg.size < clip_len:
            if seg.size < clip_len * min_fraction and segments:
                break
            seg = np.pad(seg, (0, clip_len - seg.size))
        segments.append(seg)
    return segments


def audio_to_mel_clips(signal: np.ndarray, cfg: AudioConfig) -> np.ndarray:
    """Return ``(clips, n_mels, frames, 1)`` features ready for the audio CNN."""
    clips = [log_mel_spectrogram(seg, cfg) for seg in segment_waveform(signal, cfg)]
    if not clips:
        return np.zeros((0, cfg.n_mels, frames_per_clip(cfg), 1), dtype=np.float32)
    return np.stack(clips)[..., None]
