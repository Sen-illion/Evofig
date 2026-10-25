# Released weights

The only trained parameters EvoFig depends on are the two Judge seats. Both seats
load the same adapter, so one checkpoint covers both.

| Path | What it is | Size |
| --- | --- | ---: |
| `judge-choice9b-sft-aesthetic-step-003174/adapter/` | LoRA adapter (rank 16, alpha 32) for `Qwen3.5-9B` | 111 MB |
| `judge-choice9b-sft-aesthetic-step-003174/processor/` | tokenizer, chat template, processor config | 19 MB |

`adapter_model.safetensors` is tracked with Git LFS. Clone with `git lfs install`
first, or run `git lfs pull` after cloning; otherwise the file is a 130-byte
pointer.

## Not included

- The `Qwen3.5-9B` base weights. They are 19 GB and are not redistributable here;
  download them separately and load the adapter on top.
- Earlier training steps (`step-001000` ... `step-003000`) and the other
  `choice9b-*` runs. Only the checkpoint named by the protocol is released.
- Generation models. The pipeline calls hosted models named in
  `code/configs/experiment_protocol.release.json`; no image or VLM weights are
  distributed with this repository.

## How the two seats are wired

`configs/experiment_protocol.release.json` points both `advisory_checkpoint` and
`acceptance_checkpoint` at this adapter. The advisory seat proposes a repair; the
acceptance seat decides accept or rollback. Blanking either path makes the driver
fall back to the rule-based seat and the row is stamped `judge_seats=heuristic`,
which is why a row measured with rules is not directly comparable to one measured
with the trained Judge.

## Provenance

- Training data: 873 multiple-choice comparisons over figure candidates
  (749 positive, 124 negative). The corpus is not distributed with this
  repository.
- The held-out manifest used for the internal check is not redistributed.
- Step: 3174. `training_state.json` records the step, epoch and reference key
  (`sft-no-reference`).
