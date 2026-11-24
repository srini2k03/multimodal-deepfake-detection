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


# --------------------------------------------------------------------------- silence and duration
SILENCE_FRAME_S = 0.025
SILENCE_HOP_S = 0.010


def frame_energy_db(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """Per-frame RMS energy in dB (25 ms frames, 10 ms hop); empty input gives an empty array."""
    frame, hop = int(SILENCE_FRAME_S * sample_rate), int(SILENCE_HOP_S * sample_rate)
    if signal.size < frame:
        return np.zeros(0, dtype=np.float64) if signal.size == 0 else np.array([_rms_db(signal)])
    n = 1 + (signal.size - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(signal[idx].astype(np.float64) ** 2, axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-10))


def _rms_db(x: np.ndarray) -> float:
    return float(20.0 * np.log10(max(float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))), 1e-10)))


def silence_profile(signal: np.ndarray, sample_rate: int, top_db: float = 40.0) -> dict[str, float]:
    """Leading/trailing non-speech duration: frames more than ``top_db`` below the loudest frame.

    ASVspoof 2019 LA is known to leak the label through leading-silence length (Mueller et al.,
    Interspeech 2021), so this is recorded for every utterance in the dataset audit.
    """
    energy = frame_energy_db(signal, sample_rate)
    duration = signal.size / sample_rate
    if energy.size == 0:
        return {"duration_s": 0.0, "lead_silence_s": 0.0, "trail_silence_s": 0.0, "peak_db": -200.0}
    active = np.where(energy > energy.max() - top_db)[0]
    first, last = int(active[0]), int(active[-1])
    lead = first * SILENCE_HOP_S
    trail = max(0.0, duration - (last * SILENCE_HOP_S + SILENCE_FRAME_S))
    return {
        "duration_s": float(duration),
        "lead_silence_s": float(lead),
        "trail_silence_s": float(trail),
        "peak_db": float(energy.max()),
    }


def trim_silence(signal: np.ndarray, sample_rate: int, top_db: float = 40.0) -> np.ndarray:
    """Remove leading and trailing frames more than ``top_db`` below the loudest frame."""
    prof = silence_profile(signal, sample_rate, top_db)
    start = int(prof["lead_silence_s"] * sample_rate)
    end = signal.size - int(prof["trail_silence_s"] * sample_rate)
    trimmed = signal[start:end]
    return trimmed if trimmed.size >= int(0.1 * sample_rate) else signal


def fix_length(signal: np.ndarray, length: int, pad_mode: str = "repeat") -> np.ndarray:
    """Crop to ``length`` samples from the start, or pad by tiling (``repeat``) or with zeros."""
    if signal.size >= length:
        return signal[:length]
    if signal.size == 0:
        return np.zeros(length, dtype=np.float32)
    if pad_mode == "repeat":
        reps = int(np.ceil(length / signal.size))
        return np.tile(signal, reps)[:length]
    if pad_mode == "zero":
        return np.pad(signal, (0, length - signal.size))
    raise ValueError("pad_mode must be 'repeat' or 'zero'")


def prepare_waveform(signal: np.ndarray, cfg: AudioConfig) -> np.ndarray:
    """Apply the configured silence trimming (shared by training, evaluation and the API)."""
    if cfg.trim_silence:
        return trim_silence(signal, cfg.sample_rate, cfg.trim_top_db)
    return signal


def segment_waveform(signal: np.ndarray, cfg: AudioConfig, min_fraction: float = 0.5) -> list[np.ndarray]:
    """Split into fixed-length clips. A partial final clip is kept only if it is at least
    ``min_fraction`` of a clip (or is the only clip) and is padded with ``cfg.pad_mode``."""
    clip_len = int(cfg.clip_seconds * cfg.sample_rate)
    if signal.size == 0:
        return []
    segments = []
    for start in range(0, signal.size, clip_len):
        seg = signal[start : start + clip_len]
        if seg.size < clip_len:
            if seg.size < clip_len * min_fraction and segments:
                break
            seg = fix_length(seg, clip_len, cfg.pad_mode)
        segments.append(seg)
    return segments


def clip_features(signal: np.ndarray, cfg: AudioConfig) -> np.ndarray:
    """One ``(n_mels, frames)`` log-mel for the first clip of an utterance (training/evaluation)."""
    wav = fix_length(prepare_waveform(signal, cfg), int(cfg.clip_seconds * cfg.sample_rate), cfg.pad_mode)
    return log_mel_spectrogram(wav, cfg)


def audio_to_mel_clips(signal: np.ndarray, cfg: AudioConfig) -> np.ndarray:
    """Return ``(clips, n_mels, frames, 1)`` features ready for the audio CNN."""
    clips = [log_mel_spectrogram(seg, cfg) for seg in segment_waveform(prepare_waveform(signal, cfg), cfg)]
    if not clips:
        return np.zeros((0, cfg.n_mels, frames_per_clip(cfg), 1), dtype=np.float32)
    return np.stack(clips)[..., None]
