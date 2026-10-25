# Third-party notices

The MIT license in `LICENSE` covers the EvoFig code and the files authored for this
release. It does **not** cover third-party benchmarks, datasets, models or papers.
Those remain under their own licenses and terms. This file records what is named,
what is included and what a user must obtain separately.

## Benchmarks and their scorers

| Name | Used for | Included here? |
| --- | --- | --- |
| **FigureBench** | FigureBench test split and the four official measures (nine-item, Design, Communication, Fidelity) | The released **scores**, paper IDs and aggregate tables are included. The official dataset, source figures and scorer code are not redistributed. Obtain them from the FigureBench release and follow its license. |
| **SciFlow-Bench** | SciFlow node-F1, edge-F1 and S_graph | The released **scores** and per-paper rows are included. The benchmark PDFs, inputs and scorer code are not redistributed. |
| **Paper2Poster / PaperQuiz** | The 50 detail + 50 understanding multiple-choice protocol | The prompt text is included because it is part of the scoring recipe. The Paper2Poster code and posters are not redistributed. Cite Pang et al., NeurIPS 2025. |
| **LiveFigure** | External baseline and the nine-item rubric names | The released baseline scores are included. The LiveFigure application is not included; `analysis/cost_experiment/livefigure_orig/` is our own runner and calls a separately installed LiveFigure app. |
| **AutoFigure** | External baseline | Baseline scores only. The AutoFigure implementation is not included. |
| **Paper2SysArch** | External baseline | Baseline scores only. The implementation is not included. |
| **AutomaTikZ / TikZero** | External baselines | Baseline scores only. The implementations are not included. |
| **OpenEvolve / ShinkaEvolve** | Self-evolution baselines (layout-only adaptation) | Baseline scores only. The upstream projects are not included. |

## Models and weights

| Name | Role | Included here? |
| --- | --- | --- |
| **Qwen3.5-9B** | Base model for the trained Judge seats | No weights. The main run used a local copy and a checkpoint at `step-003174`; both are third-party model artifacts under their own license. |
| **Gemini 3.1 Flash Lite / Gemini 3.1 Flash Lite Image** | Generation-side text/vision and image model; nine-item and PaperQuiz ruler | No weights and no service. The code calls an OpenAI-compatible endpoint configured by the user. |
| **AltCLIP (`BAAI/AltCLIP`)** | Paper-figure CLIP column | No weights. The code loads a local AltCLIP checkpoint when the CLIP column is enabled. |
| **CLIP (`openai/clip-vit-base-patch32`)** | SciFlow CLIP-Consistency baseline | No weights. |
| **z-image-turbo** | Cheaper icon backend used in the cost experiment | No weights or service. |

Model names are recorded because a score is only readable when the ruler is named.
Naming a model here does not grant a license to it.

## Paper corpus

The 500-paper generation pool and the 1,500-paper source pool are lists of paper IDs
and metadata. The papers themselves come from ICLR, NeurIPS, ICML, ACL, CVPR, ICCV,
AAAI, Nature Portfolio, Elsevier and other publishers. **No paper PDF is included.**
Users must obtain each paper under the publisher's terms. The manifests are
metadata, not a redistribution of the articles.

The 115 frozen PaperQuiz banks in `data/quiz/` contain questions generated from the
papers. They do not contain the paper text. If you redistribute them further, check
the terms that apply to the underlying papers.

## Human evaluation

The human-evaluation responses are included with rater codes only. Names, contact
details, compensation notes and the original survey-platform export were removed.
The survey platform's terms are not redistributed. If you reuse the responses,
follow the consent terms under which they were collected.

## Software libraries

The released code imports, but does not vendor, the following libraries. Each is
under its own license:

- `python-pptx` (MIT)
- `Pillow` (HPND)
- `PyMuPDF` / `pymupdf` (AGPL or commercial; check your use)
- `pypdfium2` (Apache-2.0 / BSD-3-Clause components)
- `pypdf` (BSD-3-Clause)
- `pdfplumber` (MIT)
- `requests` (Apache-2.0)
- `opencv-python-headless` (Apache-2.0)
- `numpy` (BSD-3-Clause)
- `wordninja` (MIT)
- `pywin32` (PSF-style)

`PyMuPDF` is the one to check before commercial redistribution: its default license
is AGPL. The release does not ship PyMuPDF binaries; it declares it as a dependency.

## Prompts and rubrics

The following prompt files are original to this project and are covered by the MIT
license, except where a file says it adapts a published protocol:

- `code/benchmarks/editable-figure-bench/prompts/nine-v001.md`
- `code/benchmarks/editable-figure-bench/prompts/nine-v002.md`
- `code/benchmarks/editable-figure-bench/prompts/nine-v003.md`
- `code/benchmarks/editable-figure-bench/prompts/paperquiz-*.md`
- `code/benchmarks/editable-figure-bench/prompts/aesthetic-v001.md`

`aesthetic-v001.md` says it adapts Paper2Poster's three aesthetic criteria (Pang et
al., NeurIPS 2025); cite that work when you use it. The PaperQuiz prompts follow
Paper2Poster's released protocol; cite it as well.

## No warranty

The released scores, tables and prompts are provided as research artifacts. They
are not a legal opinion about the underlying datasets, models or papers. If you
plan to redistribute a derived dataset, check each upstream license first.
