# Screenshots

All taken with Playwright from real runs on 2026-11-24:

- `benchmark.png`: the site's Benchmark section, built from the committed results in `results/`.
- `demo-video.jpg`: the GitHub Pages browser demo after uploading a 6 s clip generated from the
  public-domain NASA portrait in `backend/tests/fixtures/` with a 220 Hz tone as its audio track.
  Face detection, crops and the audio CNN score are real; the visual scores are labelled placeholders.
- `demo-audio.jpg`: the browser demo on a generated 5 s tone. The score is real output of the trained
  audio CNN running in the browser; a tone is out of distribution, so the number says nothing useful.
- `api-gradcam.jpg`: the API dashboard with the released models (`configs/serve.yaml`) on the same
  authentic NASA portrait: calibrated score, per-stage latency and Grad-CAM. One input, not an accuracy claim.
