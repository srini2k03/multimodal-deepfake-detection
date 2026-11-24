# Model card

Two independently trained detectors are released as GitHub release assets (not stored in git).
`backend/scripts/fetch_weights.py` downloads them, checks their SHA-256 and writes
`backend/weights/model_card.json`, which the API reads for calibration and provenance. Numbers
here come from the committed results; see [BENCHMARKS.md](BENCHMARKS.md) for the full tables.

<!-- card:start -->
_Filled in from the committed benchmark results._
<!-- card:end -->

## Intended use

- Research, teaching and triage: a **probabilistic signal** that a face image or a speech recording
  may be synthetic, shown together with its uncertainty and explanation maps.
- Not intended for automated decisions about people, for legal or forensic proof, or for content
  moderation without human review.

## Out of scope / known failure modes

- **Out-of-distribution media.** The audio model knows 2019-era TTS/VC on clean English read speech.
  The visual model knows diffusion inpainting, text-to-image and InsightFace swaps of celebrity
  photos. Modern voice cloning, phone-quality speech, other languages, talking-head video, lip-sync
  manipulation, heavy compression and adversarial edits are out of distribution. The cross-dataset
  results quantify how much worse it gets.
- **Shortcuts.** ASVspoof 2019 LA labels correlate with silence and DeepFakeFace labels with
  resolution. The trimmed-audio and resolution-capped models reduce, but do not remove, that
  reliance.
- **Faces.** Haar detection misses small, profile and occluded faces, which then fall back to a
  centre crop.
- **No fusion validation.** The fused score's weights are not learned or evaluated.

## Ethical considerations

- The visual training data shows real public figures (IMDB-WIKI). No dataset image is redistributed
  in this repository, and the committed figures are aggregate heatmaps only.
- A false "manipulated" score can harm a real person. Every interface shows probabilities with
  "likely" wording, separate per-modality scores and the evaluation caveats, never a verdict.
