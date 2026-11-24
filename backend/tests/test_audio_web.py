"""The browser audio model (frontend/src/demo/audioModel.js) must match Python + Keras.

Exports a small random-weight audio CNN with scripts/export_audio_web.py, then runs the Node
parity script on it. Skipped when TensorFlow or Node.js is unavailable.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.tf
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_browser_audio_model_matches_python(tmp_path):
    pytest.importorskip("tensorflow")
    from deepfake_detector.models.audio import build_audio_cnn

    model = build_audio_cnn(64, 101, filters=(4, 8))
    model.save(tmp_path / "tiny.keras")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("audio:\n  clip_seconds: 1.0\n  trim_silence: true\n  cnn_filters: [4, 8]\n")
    out = tmp_path / "web"
    script = str(ROOT / "backend/scripts/export_audio_web.py")
    export = [sys.executable, script, "--model", str(tmp_path / "tiny.keras"), "--out", str(out), "--config", str(cfg)]
    subprocess.run(export, check=True, capture_output=True, timeout=300)
    manifest = json.loads((out / "audio_cnn.json").read_text())
    assert manifest["audio"]["trim_silence"] is True and manifest["fold_check_max_abs_logit_diff"] < 1e-4
    parity = subprocess.run(
        ["node", str(ROOT / "frontend/scripts/audio-parity.mjs"), str(out)],
        capture_output=True, text=True, timeout=300, check=False,
    )  # fmt: skip
    assert parity.returncode == 0, parity.stderr + parity.stdout
    result = json.loads(parity.stdout.strip().splitlines()[-1])
    assert result["n_clips"] == result["expected_clips"] and result["max_abs_logit_diff"] < 2e-3
