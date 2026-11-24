"""Download the released benchmark models into ``weights/`` and write ``weights/model_card.json``.

    python scripts/fetch_weights.py                       # defaults below
    python scripts/fetch_weights.py --audio-system cnn    # serve the untrimmed-audio CNN instead
    DFD_CONFIG=configs/serve.yaml gunicorn -c gunicorn.conf.py wsgi:app

Files are verified against each release's SHA256SUMS. The model card copies the calibration
(Platt scaling fitted on the validation split) and headline test numbers from the committed
benchmark results, so the API can report calibrated probabilities and their provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

REPO = "srini2k03/multimodal-deepfake-detection"
ROOT = Path(__file__).resolve().parents[2]
AUDIO_RESULTS = ROOT / "results/audio/asvspoof2019_la"
VISUAL_RESULTS = ROOT / "results/visual/dff"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "dfd-fetch-weights", "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def fetch_asset(tag: str, name: str, dest: Path) -> str:
    """Download one release asset, check it against SHA256SUMS and return its digest."""
    base = f"https://github.com/{REPO}/releases/download/{tag}"
    sums = {}
    for line in _get(f"{base}/SHA256SUMS").decode().splitlines():
        digest, _, fname = line.partition("  ")
        sums[fname.strip()] = digest
    data = _get(f"{base}/{name}")
    digest = hashlib.sha256(data).hexdigest()
    if sums.get(name) != digest:
        raise SystemExit(f"checksum mismatch for {name}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    print(f"{tag}/{name} -> {dest} ({len(data):,} bytes, sha256 {digest[:12]})")
    return digest


def audio_card(system: str, digest: str, tag: str) -> dict:
    m = json.loads((AUDIO_RESULTS / "metrics.json").read_text())
    s = m["systems"][system]
    dev = s["dev"]
    return {
        "file": "audio_cnn.keras",
        "release": tag,
        "sha256": digest,
        "system": system,
        "trim_silence": system == "cnn_trim",
        "trained_on": "ASVspoof 2019 LA train (speaker-disjoint official split)",
        "calibration": {
            **s["platt"],
            "prior_fake": dev["n_spoof"] / dev["n"],
            "source": "Platt scaling on ASVspoof 2019 LA dev, re-weighted to equal priors",
        },
        "benchmark": {
            "eval_eer": s["eval"]["eer"],
            "eval_min_tdcf": s["eval"].get("min_tdcf"),
            "eval_roc_auc": s["eval"]["roc_auc"],
            "run_id": m["meta"]["run_id"],
            "git_sha": m["meta"]["git_sha"],
            "results": "results/audio/asvspoof2019_la/metrics.json",
        },
    }


def visual_card(system: str, digest: str, tag: str) -> dict:
    m = json.loads((VISUAL_RESULTS / "metrics.json").read_text())
    audit = json.loads((VISUAL_RESULTS / "data_audit.json").read_text())
    s = m["systems"][system]
    val = audit["selected"]["val"]
    return {
        "file": "efficientnetv2_b0.keras",
        "release": tag,
        "sha256": digest,
        "system": system,
        "backbone": "efficientnetv2_b0",
        "image_size": 224,
        "trained_on": "DeepFakeFace train split (identity-disjoint), image level",
        "calibration": {
            **s["platt"],
            "prior_fake": val["n_fake"] / val["n"],
            "source": "Platt scaling on the DeepFakeFace validation split, re-weighted to equal priors",
        },
        "benchmark": {
            "test_roc_auc": s["test"]["roc_auc"],
            "test_eer": s["test"]["eer"],
            "run_id": m["meta"]["run_id"],
            "git_sha": m["meta"]["git_sha"],
            "results": "results/visual/dff/metrics.json",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights-dir", default=str(ROOT / "backend/weights"))
    parser.add_argument("--audio-tag", default="audio-asvspoof2019-v1")
    parser.add_argument("--audio-system", default="cnn_trim", choices=["cnn", "cnn_trim"])
    parser.add_argument("--visual-tag", default="visual-dff-v1")
    parser.add_argument("--visual-system", default="effnet_ft")
    parser.add_argument("--skip-visual", action="store_true")
    parser.add_argument("--skip-audio", action="store_true")
    args = parser.parse_args()
    weights = Path(args.weights_dir)
    card: dict = {}
    if not args.skip_audio:
        digest = fetch_asset(args.audio_tag, f"{args.audio_system}.keras", weights / "audio_cnn.keras")
        card["audio"] = audio_card(args.audio_system, digest, args.audio_tag)
    if not args.skip_visual:
        name = f"{args.visual_system}_model.keras"
        digest = fetch_asset(args.visual_tag, name, weights / "efficientnetv2_b0.keras")
        card["visual"] = visual_card(args.visual_system, digest, args.visual_tag)
    (weights / "model_card.json").write_text(json.dumps(card, indent=1) + "\n")
    print(f"wrote {weights / 'model_card.json'}")


if __name__ == "__main__":
    main()
