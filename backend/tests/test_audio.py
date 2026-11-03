import numpy as np
import pytest

from deepfake_detector.config import AudioConfig
from deepfake_detector.preprocessing.audio import (
    audio_to_mel_clips,
    frames_per_clip,
    hz_to_mel,
    load_audio,
    log_mel_spectrogram,
    mel_filterbank,
    mel_to_hz,
    resample,
    segment_waveform,
)


def test_mel_roundtrip():
    hz = np.array([0.0, 440.0, 4000.0])
    assert np.allclose(mel_to_hz(hz_to_mel(hz)), hz)


def test_filterbank_shape_and_nonneg():
    fb = mel_filterbank(16000, 512, 40, 20, 8000)
    assert fb.shape == (40, 257)
    assert (fb >= 0).all() and (fb.sum(axis=1) > 0).all()


def test_log_mel_shape_and_peak_band():
    cfg = AudioConfig(clip_seconds=1.0)
    t = np.arange(cfg.sample_rate) / cfg.sample_rate
    mel = log_mel_spectrogram(np.sin(2 * np.pi * 1000 * t).astype(np.float32), cfg)
    assert mel.shape == (cfg.n_mels, frames_per_clip(cfg))
    peak_band = int(mel.mean(axis=1).argmax())
    centre_hz = mel_to_hz(np.linspace(hz_to_mel(cfg.fmin), hz_to_mel(cfg.fmax), cfg.n_mels + 2)[peak_band + 1])
    assert 700 < centre_hz < 1400


def test_segmentation_and_clips():
    cfg = AudioConfig(clip_seconds=1.0)
    sig = np.zeros(int(2.6 * cfg.sample_rate), np.float32)
    segs = segment_waveform(sig, cfg)
    assert len(segs) == 3 and all(s.size == cfg.sample_rate for s in segs)
    assert len(segment_waveform(np.zeros(int(2.2 * cfg.sample_rate)), cfg)) == 2  # short tail dropped
    clips = audio_to_mel_clips(sig + 1e-3 * np.random.default_rng(0).standard_normal(sig.size), cfg)
    assert clips.shape == (3, cfg.n_mels, frames_per_clip(cfg), 1)
    assert audio_to_mel_clips(np.zeros(0, np.float32), cfg).shape[0] == 0


def test_resample_length():
    out = resample(np.ones(8000, np.float32), 8000, 16000)
    assert out.size == 16000


def test_wav_roundtrip(wav_file):
    sig = load_audio(wav_file, 16000)
    assert sig.dtype == np.float32
    assert sig.size == pytest.approx(2.5 * 16000, abs=1)
    assert 0.35 < np.abs(sig).max() <= 0.41
    assert load_audio(wav_file, 8000).size == pytest.approx(2.5 * 8000, abs=1)
