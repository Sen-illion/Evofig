---
base_model: Qwen3.5-9B
library_name: peft
pipeline_tag: image-text-to-text
tags:
- lora
- peft
- judge
- scientific-figure
---

# EvoFig Judge - choice9b-sft-aesthetic, step 3174

LoRA adapter for the EvoFig Judge seats. One adapter is loaded twice: the
advisory seat proposes repairs and the acceptance seat decides accept/rollback.
The weights are identical for both seats, matching the configuration used for
the reported runs.

## Files

- `adapter/adapter_model.safetensors` - LoRA weights (rank 16, alpha 32),
  tracked with Git LFS.
- `adapter/adapter_config.json` - PEFT configuration; target modules are the
  attention and MLP projections.
- `processor/` - chat template, tokenizer and processor config copied from the
  base model at training time.
- `training_state.json` - training step, epoch and reference key.

## Base model

`Qwen3.5-9B`. The base weights are **not** redistributed here and must be
obtained separately; the adapter is applied on top of them.

## Training

- Training examples: 873 (749 positive, 124 negative) multiple-choice comparisons over figure candidates.
- Objective: multiple-choice preference learning over figure candidates,
  scored with a reference key (`sft-no-reference`).
- Checkpoint: step 3174 of the SFT schedule. Earlier steps are not released.

## Usage

```python
from peft import PeftModel

base = load_base_model("Qwen3.5-9B")
judge = PeftModel.from_pretrained(base, "weights/judge-choice9b-sft-aesthetic-step-003174/adapter")
```

See `../README.md` for what the two seats do and how the protocol names them.

## Limitations

The Judge scores layout and figure quality within this pipeline. It is not a
general-purpose reward model and was not evaluated outside scientific figure
generation.

