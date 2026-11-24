# Model card

Two independently trained detectors are released as GitHub release assets (not stored in git).
`backend/scripts/fetch_weights.py` downloads them, checks their SHA-256 and writes
`backend/weights/model_card.json`, which the API reads for calibration and provenance. Numbers
here come from the committed results; see [BENCHMARKS.md](BENCHMARKS.md) for the full tables.

<!-- card:start -->
### Audio countermeasure (`cnn_trim`)

- Architecture: 3-block log-mel CNN, 154,641 parameters
- Training data: ASVspoof 2019 LA train (speaker-disjoint official protocol), leading/trailing silence trimmed
- ASVspoof 2019 LA eval: EER 19.01%, min t-DCF 0.449, ROC-AUC 0.8934
- In-the-Wild (cross-dataset): EER 45.46%
- CPU latency per 4 s clip: Keras 15.9 ms, TFLite int8 1.4 ms (164 KB)
- Provenance: [run 45371433877](https://github.com/srini2k03/multimodal-deepfake-detection/actions/runs/45371433877), commit `c9b2215`, release `audio-asvspoof2019-v1`

### Visual classifier (`effnet_ft`)

- Architecture: EfficientNetV2-B0 (ImageNet) + dropout + logit head, 224x224 face crops
- Training data: DeepFakeFace train split (identity-disjoint sample)
- DeepFakeFace test: ROC-AUC 0.914, EER 16.92%, face-size-matched AUC 0.910
- CPU latency per face crop: Keras batch 1 181 ms, batch 16 25 ms/image; face detection + alignment 33 ms
- Provenance: [run 45373710322](https://github.com/srini2k03/multimodal-deepfake-detection/actions/runs/45373710322), commit `8724ec3`, release `visual-dff-v1`
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
