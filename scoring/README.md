# Scoring overlay: `nine-v003`

The generation tree in `code/` is the `se` snapshot (`fdcfb6d5e`). That tree accepts
only `nine-v001` and `nine-v002`. The paper scored with `nine-v003`, defined on the
evaluation branch `e68fc3f4`.

This directory carries the **two files that differ** for scoring. It is not a second
product version: the generation chain stays at `se`.

## Files

| File | Source | Why it is needed |
| --- | --- | --- |
| `overlay/rfs/evaluation/livefigure_nine.py` | `e68fc3f4` | Adds `nine-v003`, retires v001/v002, and refuses to grade fidelity items without method text. |
| `overlay/rfs/evaluation/own_set_score.py` | `e68fc3f4` | Adds the method-text fallback (`nine_grounding_text`) used when the method section cannot be located. |
| `../code/benchmarks/editable-figure-bench/prompts/nine-v003.md` | run environment | The rubric prompt itself. |

The PaperQuiz prompts and the older nine-item prompts are already in
`code/benchmarks/editable-figure-bench/prompts/`; the `se` scorer reads them from
that directory.

## Apply

From the repository root:

```bash
cp scoring/overlay/rfs/evaluation/livefigure_nine.py code/rfs/evaluation/
cp scoring/overlay/rfs/evaluation/own_set_score.py code/rfs/evaluation/
```

Check that the ruler is active:

```bash
cd code
PYTHONPATH=. python -c "from rfs.evaluation.livefigure_nine import PROMPT_VERSION; print(PROMPT_VERSION)"
# nine-v003
```

## Environment

Scoring is off unless the two switches are set to `api`:

```bash
export API_BASE="<your OpenAI-compatible base URL>"
export API_KEY="<your key>"
export RFS_SCORE_MODEL="gemini-3.1-flash-lite"
export RFS_NINE_JUDGE=api
export RFS_PAPER_QUIZ=api
```

`API_KEY` or `GEMINI_API_KEY` is accepted. Do not commit either. `RFS_SCORE_MODEL`
is the ruler model; keep it separate from `MODEL_VLM`, which is a generation-side
knob.

## Why the fallback file matters

`nine-v003` grades accuracy / completeness / appropriateness against the paper's own
method text. If a paper has no extractable method section, the scorer would return
blank fidelity items. `own_set_score.py` falls back to the abstract and then to the
body (up to `NINE_GROUNDING_LIMIT = 12000` characters) and records
`nine_grounding_fallback` beside the scores. A blank item is never imputed or scored
as zero.

## What this does not reproduce

The overlay reproduces the **ruler**, not the 297 source PNGs. The per-paper scores
are in `data/tables_main/repeats_20261024/`; re-scoring requires the locked images
whose SHA-256 hashes are in `round1_manifest.jsonl`. See `docs/KNOWN_GAPS.md`.

