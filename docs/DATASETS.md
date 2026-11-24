# Datasets: what is used, what is not, and why

This project only uses data whose licence and access conditions allow it: openly downloadable,
with no request form, EULA signature, Kaggle login or Hugging Face gating. Every access claim below
comes from [`results/datasets/access_check.json`](../results/datasets/access_check.json). That file is
written by the [`Dataset access check`](../.github/workflows/dataset-access.yml) workflow, which queries
each host anonymously from a GitHub-hosted runner (`backend/scripts/check_dataset_access.py`). The
statistics in the tables come from the data audits that the benchmark runs commit
(`results/*/*/data_audit.json`).

## Decision summary

| Dataset | Modality | Licence (as published by the host) | Access (checked 2026-11-24) | Size | Role here |
|---|---|---|---|---|---|
| [ASVspoof 2019 LA](https://datashare.ed.ac.uk/handle/10283/3336) | audio | Open Data Commons Attribution (ODC-By 1.0) | anonymous download, HTTP range requests work | `LA.zip` 7,640,952,520 B | **train / dev / eval of the audio branch** |
| [In-the-Wild](https://huggingface.co/datasets/mueller91/In-The-Wild) (Müller et al. 2022) | audio | HF card: `cc-by-sa-4.0`; project page: CC BY-NC 4.0 | HF `gated: false` | 8,161,489,023 B | **cross-dataset test of the audio branch only** (non-commercial use assumed) |
| [DeepFakeFace (DFF)](https://huggingface.co/datasets/OpenRL/DeepFakeFace) (Song et al. 2023) | image | HF card: `openrail` (README body also shows `apache-2.0`); real photos come from IMDB-WIKI (academic research use) | HF `gated: false` | 4 zips, 5,125,305,860 B | **train / val / test of the visual branch** (image level) |
| [WildDeepfake](https://huggingface.co/datasets/xingjunm/WildDeepfake) (Zi et al. 2020, author upload) | video face sequences | `apache-2.0` | HF `gated: false` | 82.7 GB (test split 9.9 GB) | candidate cross-dataset visual test (see [Status](#status-of-the-candidates)) |
| [OpenForensics](https://zenodo.org/records/5528418) | image (multi-face) | CC BY 4.0 | Zenodo, open | 56.4 GB | not used: too large for the CPU budget; lawful candidate |
| [ASVspoof 2021 LA/DF eval](https://zenodo.org/records/4837263) | audio | ODC-By (LA) | Zenodo, open; keys on asvspoof.org | 7.8 GB (LA), 34.5 GB (DF) | not used yet; lawful candidate for more cross-dataset tests |
| [WaveFake](https://zenodo.org/records/5642694) | audio | CC BY-SA 4.0 | Zenodo API returned HTTP 500 during the check | 28.9 GB | not used |

### Excluded because access requires a request, agreement or login

| Dataset | Why it is excluded | How the owner could use it |
|---|---|---|
| FaceForensics++ | Google Form + terms of use before the download script is sent | apply at <https://github.com/ondyari/FaceForensics> |
| DFDC (Deepfake Detection Challenge) | Meta's dataset licence must be accepted | <https://ai.meta.com/datasets/dfdc/> |
| Celeb-DF v1/v2 | request form | <https://github.com/yuezunli/celeb-deepfakeforensics> |
| DeeperForensics-1.0 | agreement form | <https://github.com/EndlessSora/DeeperForensics-1.0> |
| FakeAVCeleb | request form | <https://github.com/DASH-Lab/FakeAVCeleb> |
| DeepSpeak, AV-Deepfake1M | gated / EULA | dataset pages |
| LAV-DF | Hugging Face `gated: auto` (README returns 401 without accepting terms) | <https://huggingface.co/datasets/ControlNet/LAV-DF> |
| MLAAD | Hugging Face `gated: auto` | <https://huggingface.co/datasets/mueller91/MLAAD> |
| Kaggle "real vs fake faces" sets and their Hugging Face re-uploads | Kaggle login, and the re-uploads have unclear provenance or licence | not recommended |

The repository still supports the FaceForensics++, Celeb-DF and generic layouts
(`deepfake_detector/data/layouts.py`). Anyone who obtains access can train on those datasets with the
existing commands.

## Consequence for fusion

None of the lawfully accessible datasets has **paired, labelled audio and video of the same
clip**. The paired sets (FakeAVCeleb, LAV-DF, AV-Deepfake1M, DeepSpeak) are all behind forms or
gates. Audio-visual fusion is therefore implemented and unit-tested, but it is **not evaluated**:
no fusion accuracy is reported anywhere in this repository. The audio and visual models are
trained and evaluated separately, on different datasets.

## ASVspoof 2019 LA (audio)

- **Content:** bona fide English read speech (VCTK speakers) and spoofed speech from 19 TTS / VC
  systems. 16 kHz FLAC.
- **Splits (official, speaker-disjoint):** train and dev use attacks A01–A06. Eval uses A07–A19.
  A16 and A19 reuse the A04 and A06 systems, so **11 of the 13 eval attacks are never seen in
  training**. Speaker overlap between splits is measured in the audit (expected 0).
- **Measured (`results/audio/asvspoof2019_la/data_audit.json`):**

  | Split | Utterances | Bona fide | Spoof | Speakers | Attacks | Median leading silence, bona fide / spoof |
  |---|---|---|---|---|---|---|
  | train | 25,380 | 2,580 | 22,800 | 20 | A01–A06 | 0.71 s / 0.33 s |
  | dev | 24,844 | 2,548 | 22,296 | 20 | A01–A06 | 0.88 s / 0.33 s |
  | eval | 71,237 | 7,355 | 63,882 | 67 | A07–A19 | 0.75 s / 0.04 s |

  Speaker overlap between every pair of splits: 0. Archive SHA-256 `208a7e4e…e4ec`.
- **Leakage control:** the official speaker-disjoint protocol is used unchanged. Model selection
  uses dev only. Eval is scored once, at the dev-chosen operating point.
- **Known bias:** the duration of leading and trailing non-speech differs between bona fide and
  spoofed files (Müller et al., *Speech is Silver, Silence is Golden*, Interspeech 2021). The
  benchmark measures this directly. One baseline sees only duration and silence lengths, and a
  second CNN is trained on silence-trimmed audio. See [BENCHMARKS.md](BENCHMARKS.md).
- **Obligations:** attribution (ODC-By). The archive is never redistributed. Only metrics and
  per-utterance scores are committed.

## In-the-Wild (audio, cross-dataset test only)

Real and deepfaked speech of public figures, collected from the internet: 31,779 clips
(19,963 bona fide, 11,816 spoofed) from 54 speakers. It is used
only as an out-of-distribution test set for the ASVspoof-trained systems, with no training or
threshold tuning on it. The operating threshold and the Platt calibration come from ASVspoof dev.

## DeepFakeFace (visual)

- **Content:** four subsets of 30,000 images each. `wiki` holds real celebrity photos selected
  from IMDB-WIKI. `inpainting` (Stable Diffusion inpainting), `insight` (InsightFace face swap) and
  `text2img` (Stable Diffusion v1.5) are fakes derived from those people and photos.
- **This is image-level data.** There is no video, audio, frame rate or temporal information, so
  the visual results say nothing about video-level aggregation or audio-visual consistency.
- **Leakage-free split:** each fake keeps the IMDB-WIKI path of its source photo. Each image is
  grouped by the IMDB-WIKI person id parsed from the file name, or by the shared path when no id
  parses, and whole groups are assigned to train / val / test by a seeded hash. A person, and
  every fake made from their photo, appears in exactly one split. The audit reports the group
  overlap between splits (expected 0) and how many fake keys match a real key.
- **Selected sample (`results/visual/dff/data_audit.json`):** train 20,000 images (5,000 identities:
  5,000 real, 15,000 fake), val 3,200 (800 identities), test 10,000 (2,500 identities). The
  identity overlap between splits is 0. Every one of the 30,000 fake keys per type matches a real
  key, and 99.99% of the file names yield an IMDB-WIKI person id.
- **Manipulation-type generalisation:** leave-one-generator-out models never see one fake type
  during training. They measure how detection degrades on an unseen manipulation.
- **Known biases (measured, not assumed):** in the test sample, every fake (all three types) is a
  512×512 JPEG, while the real photos are mostly JPEG (2,404 of 2,500) with varied size (median
  440×532, 5th percentile 234 px wide). The face detector fires on 72% of real photos and on 67–83%
  of fakes. A classifier on these file properties alone reaches test ROC-AUC 0.967 (BENCHMARKS.md). The benchmark
  re-encodes every face crop as JPEG at the same quality and adds random JPEG and rescale
  augmentation to both classes. It also reports a *metadata-only* classifier (file format,
  resolution, bytes per pixel, face-box size) that quantifies how much the files leak without
  looking at pixels. IMDB-WIKI is a celebrity dataset with known demographic skew. No
  per-demographic analysis is possible because no demographic labels are used.
- **Obligations:** cite the DFF paper. The IMDB-WIKI real photos are for academic research. No
  dataset image is committed to this repository. Figures show only aggregated Grad-CAM heatmaps,
  never faces.

## Status of the candidates

- **WildDeepfake:** licensed and ungated, with a 9.9 GB test split of video face sequences. It is
  the natural cross-dataset test for the visual model and for frame-to-video aggregation. Whether
  it was run is stated in [BENCHMARKS.md](BENCHMARKS.md).
- **OpenForensics, ASVspoof 2021, WaveFake:** lawful but not used, because of size or availability
  within the CPU budget.
