# `code/` - the `se` product tree

This is the self-evolution chain at `fdcfb6d5e`. It is the code that produced the
main EvoFig set, the round sweep and the post-evolution numbers. It is not the
development `HEAD`.

## Install and check

```bash
python -m pip install -e .
rfs doctor --json
```

The package name is `research-figure-studio`; the CLI entry point is `rfs`.
Python >= 3.10.

## What is here

- `rfs/` - product package, including `workflows/continuous_evolution.py`.
- `scripts/` - run drivers, scoring drivers, benchmark list builders and the
  isolated-checkout deploy script.
- `configs/experiment_protocol.json` - the development protocol (contains
  host-local placeholders).
- `configs/experiment_protocol.release.json` - the path-free release copy.
- `benchmarks/editable-figure-bench/prompts/` - rubric and PaperQuiz prompts the
  code reads at runtime.
- `tests/` - unit, integration and e2e tests.

## Scoring

This tree's `livefigure_nine.py` accepts `nine-v001`/`nine-v002`. The paper used
`nine-v003`. Apply the two-file overlay in `../scoring/` before re-scoring; see
`../scoring/README.md`.

## Evolution rounds

`paper-evolve-editable` evolves a single paper by default: `--evolution on` and
`--max-rounds 5`. The run chains from the last accepted state and delivers the
accepted round with the highest measured score; `evolution_state.json` lists the
accepted rounds and the delivered one. Pass `--max-rounds 3 --candidates-per-round 2`
to match the reported runs. `paper-evolve-continuous` is the separate
wave-after-wave loop and is not started automatically.

## Default typeface

This tree compiles with Arial (theme font Calibri). Comic Sans MS appears only in
later development builds.

## License

MIT; see `../LICENSE`. Third-party benchmarks and models are not redistributed;
see `../THIRD_PARTY_NOTICES.md`.
