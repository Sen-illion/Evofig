# EvoFig

Source code and trained Judge weights for **EvoFig**, a PPTX-first pipeline that turns a
paper into an editable framework figure and improves it through a bounded self-evolution
loop.

This repository carries the two things needed to run the system: the self-evolution chain
and the Judge adapter. Benchmarks, scoring tables and evaluation artifacts are not
distributed here.

## Layout

| Path | Contents |
| --- | --- |
| `code/` | The self-evolution chain at commit `fdcfb6d5e` (the `se` line): generation, the two Judge seats, bounded repair, rollback and experience capture. |
| `weights/` | The Judge LoRA adapter (step 3174), stored with Git LFS. One adapter serves both Judge seats. |

## Install

```bash
git lfs install && git lfs pull      # weights/ is stored with LFS
python -m pip install -e code/
rfs doctor --json
```

Python 3.10+. Generation needs provider credentials; none are included. See
`code/configs/experiment_protocol.release.json` for the models, seeds and scoring switches
the protocol names.

## Running a paper

```bash
rfs paper-evolve-editable --paper paper.pdf --out out/run1
```

A single command does the whole chain: it generates the figure, then evolves it. Evolution
is on by default (`--evolution on`) and takes **5 rounds** (`--max-rounds`, 1-12). The run
keeps chaining from the last accepted state, and the figure it delivers is the accepted
round with the **highest measured score**; `evolution_state.json` records every accepted
round and which one was delivered.

To reproduce the settings behind the reported numbers, pin the flags the runs used:

```bash
rfs paper-evolve-editable --paper paper.pdf --out out/repro \
  --max-rounds 3 --candidates-per-round 2
```

Wave-after-wave evolution across a fixed paper set is a separate command,
`rfs paper-evolve-continuous`, which is not started automatically by the single-paper run.

## Notes

- Default typeface in this tree is **Arial** (theme font Calibri). Comic Sans MS exists only
  in later development builds and is not what produced the reported figures.
- Fonts, arrows, panels and labels are compiled as editable PowerPoint objects; raster
  assets carry only local icons.
- The `Qwen3.5-9B` base model is not redistributed. Download it separately and load
  `weights/judge-choice9b-sft-aesthetic-step-003174/adapter` on top.

## License

MIT. See `LICENSE`. Third-party models and benchmarks are not redistributed; see
`THIRD_PARTY_NOTICES.md`.
