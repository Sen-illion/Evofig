"""Light DPO trainer for the Judge preference pairs.

The loop is ours. Both images of a pair enter one sample, the reference log
probabilities are precomputed once and cached by fingerprint, the loss is the DPO
objective against the frozen reference, and every round writes a checkpoint, a metrics
line and a version lock so a run can be reproduced and resumed.

Two backends are offered and always chosen explicitly:

* ``transformers`` (default) - a 4-bit LoRA VLM through ``transformers``, ``peft`` and
  ``bitsandbytes``. It raises :class:`TrainingUnavailable` when the stack or the base
  model is missing instead of quietly falling back.
* ``synthetic`` - a deterministic hashed-feature scorer that still takes a real gradient
  step. It exists so the pipeline, the reference cache, the metrics and the resume path
  can be verified end to end on a machine without the training stack, including this
  one. It is never selected implicitly and it is never a substitute for a real run.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

from rfs.providers.embeddings import HashingNgramEmbedder

from .generation_smoke import GENERATION_SMOKE_FILE, run_generation_smoke
from .generate_eval import GENERATION_VAL_FILE, evaluate_generated_judgements
from .judge_schema import JUDGE_CONTRACT_VERSION
from .preference_pairs import PAIR_SCHEMA_VERSION
from .prompts import PROMPT_VERSION


TRAINING_SCHEMA_VERSION = "judge-dpo-training-v1"
TRAINING_BACKENDS = ("transformers", "synthetic")
SYNTHETIC_BACKEND = "synthetic"
# HuggingFace ``.loss`` is the mean NLL over tokens whose label is not -100. DPO is defined
# on the sequence log probability, which is that mean multiplied by the number of scored
# tokens. The string is part of the reference-cache fingerprint so a resumed run cannot
# reuse scores taken under mean-token reduction.
COMPLETION_LOGPROB_REDUCTION = "sum_completion_tokens"
LABEL_IGNORE_INDEX = -100
REFERENCE_CACHE = "reference_logprobs.jsonl"
STATE_FILE = "training_state.json"
METRICS_FILE = "metrics.jsonl"
REPORT_FILE = "training_report.json"
LOCK_FILE = "version_lock.json"
DATA_REPORT_FILE = "data_report.json"
CHECKPOINT_DIR = "checkpoints"
VALIDATION_METRICS_FILE = "validation_metrics.jsonl"
BEST_CHECKPOINT_FILE = "best_checkpoint.json"
# Held-out pairwise this high with a median margin past ``collapse_margin`` is the
# first-round collapse: the policy is forcing the rejected judgement to zero
# probability rather than judging better, and generation is about to go blank.
SATURATED_PAIRWISE = 0.99
BEST_CHECKPOINT_SELECTION = "smoke-passing checkpoint with the highest held-out pairwise accuracy; the earlier step wins a tie"
BEST_GENERATION_SELECTION = "smoke-passing checkpoint with the highest generated held-out pairwise accuracy; the earlier step wins a tie"
TRAINING_OBJECTIVES = ("dpo", "sft")
SFT_OBJECTIVE = "sft"
SELECTION_MODES = ("", "ranking", "generate", "choice")
BEST_CHOICE_SELECTION = "checkpoint whose held-out figure pick and defect pick neither fall and at least one rises; the earlier step wins a tie; a collapsed legal rate cannot win"
REQUIRED_RECORD_FIELDS = ("example_id", "messages", "chosen", "rejected", "images")
# The projection names every Qwen-style decoder uses. They are named explicitly instead of
# letting peft infer them: inference needs the module tree of a model family it knows, and
# a MoE vision-language backbone is not one of them.
DEFAULT_LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class TrainingUnavailable(RuntimeError):
    """Raised when the requested training stack or base model cannot be loaded."""


def dpo_loss(margin: float, beta: float) -> float:
    """Return ``-log sigmoid(beta * margin)`` without overflowing."""

    z = float(beta) * float(margin)
    return math.log1p(math.exp(-z)) if z >= 0 else (-z + math.log1p(math.exp(z)))


def dpo_gradient(margin: float, beta: float) -> float:
    """Return how far one step moves the margin: the ascent direction of the DPO loss.

    It is the negated derivative of :func:`dpo_loss`, so a caller adds it to the weights
    (``w += lr * dpo_gradient * (features_chosen - features_rejected)``) and the margin grows.
    The name is kept because callers and tests already read it this way.
    """

    z = float(beta) * float(margin)
    sigmoid = 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))
    return float(beta) * (1.0 - sigmoid)


def completion_token_count(labels: Any, *, ignore_index: int = LABEL_IGNORE_INDEX) -> Any:
    """Return how many tokens the HuggingFace mean NLL was taken over.

    Prompt tokens are already masked to ``ignore_index``. A missing labels tensor is treated
    as one token so stub models and the synthetic backend can keep feeding a sequence-level
    score without inventing a fake length.
    """

    if labels is None:
        return 1
    if hasattr(labels, "ne"):
        counted = labels.ne(ignore_index).sum()
        if hasattr(counted, "clamp"):
            return counted.clamp(min=1)
        return max(int(counted), 1)
    values = labels.tolist() if hasattr(labels, "tolist") else labels
    flat: list[Any] = []
    stack = [values]
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple)):
            stack.extend(item)
        else:
            flat.append(item)
    counted = sum(1 for item in flat if int(item) != int(ignore_index))
    return max(counted, 1)


def completion_logprob(output: Any, batch: dict[str, Any] | None = None) -> Any:
    """Return the completion log probability DPO uses: the sum over unmasked tokens.

    ``output.loss`` is mean NLL. Multiplying by the unmasked count recovers the sum of
    per-token log-softmax values without a second gather over logits. Chosen and rejected
    go through this function, so a 20-token empty ``diagnosis`` and a 200-token audit are
    no longer compared as if they were one token each.
    """

    mean_nll = output.loss
    labels = batch.get("labels") if isinstance(batch, dict) else None
    count = completion_token_count(labels)
    scale = count.to(dtype=mean_nll.dtype) if hasattr(count, "to") and hasattr(mean_nll, "dtype") else count
    return -mean_nll * scale


def preference_logprob_term(chosen_logprob: Any, rejected_logprob: Any, reference_margin: Any) -> Any:
    """Return the reference-relative log ratio the DPO loss is applied to.

    ``reference_margin`` is ``log p_ref(chosen) - log p_ref(rejected)``. The arguments are used
    as they are so the same formula serves both the reported float metric and the differentiable
    tensor path; a float() here would cut the backward graph of the real backend.
    """

    return (chosen_logprob - rejected_logprob) - reference_margin


def parameters_by_device(parameters: Iterable[Any]) -> dict[Any, list[Any]]:
    """Group the parameters that carry a gradient by the device each one lives on.

    Gradient clipping takes a list of tensors and a single device; with ``device_map="auto"`` the
    layers are spread over several GPUs, so the groups have to be clipped one device at a time.
    """

    groups: dict[Any, list[Any]] = {}
    for parameter in parameters:
        if getattr(parameter, "requires_grad", False) and getattr(parameter, "grad", None) is not None:
            groups.setdefault(getattr(parameter, "device", None), []).append(parameter)
    return groups


@dataclass(frozen=True)
class TrainingConfig:
    """Everything a run needs, so a config file alone can reproduce it."""

    dataset: str
    out_dir: str
    backend: str = "transformers"
    base_model: str = ""
    beta: float = 0.1
    learning_rate: float = 5e-06
    epochs: int = 1
    gradient_accumulation: int = 8
    max_length: int = 4096
    image_max_pixels: int = 0
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = DEFAULT_LORA_TARGET_MODULES
    load_in_4bit: bool = True
    gradient_checkpointing: bool = False
    shard_across_gpus: bool = False
    seed: int = 0
    max_steps: int = 0
    save_every: int = 50
    resume_from: str = ""
    init_from: str = ""
    validation_dataset: str = ""
    eval_limit: int = 0
    early_stop_margin: float = 0.0
    smoke_limit: int = 0
    smoke_max_new_tokens: int = 1024
    smoke_pass_rate: float = 0.5
    pairwise_patience: int = 0
    collapse_margin: float = 0.0
    objective: str = "dpo"
    generate_eval_limit: int = 0
    selection: str = ""

    def __post_init__(self) -> None:
        if int(self.image_max_pixels) < 0:
            raise ValueError("image_max_pixels cannot be negative; use 0 to leave the vision budget untouched")
        if int(self.eval_limit) < 0:
            raise ValueError("eval_limit cannot be negative; use 0 to score every held-out pair")
        if float(self.early_stop_margin) < 0:
            raise ValueError("early_stop_margin cannot be negative; use 0 to train the whole schedule")
        if int(self.smoke_limit) < 0:
            raise ValueError("smoke_limit cannot be negative; use 0 to skip generation smoke")
        if int(self.smoke_max_new_tokens) <= 0:
            raise ValueError("smoke_max_new_tokens must be a positive token budget")
        if not 0.0 < float(self.smoke_pass_rate) <= 1.0:
            raise ValueError("smoke_pass_rate must be in (0, 1]")
        if int(self.pairwise_patience) < 0:
            raise ValueError("pairwise_patience cannot be negative; use 0 to ignore held-out stalls")
        if float(self.collapse_margin) < 0:
            raise ValueError("collapse_margin cannot be negative; use 0 to ignore a saturated margin")
        if str(self.objective) not in TRAINING_OBJECTIVES:
            raise ValueError("objective must be dpo or sft")
        if str(self.selection) not in SELECTION_MODES:
            raise ValueError("selection must be ranking, generate, choice, or empty")
        if str(self.resume_from) and str(self.init_from):
            raise ValueError("resume_from continues an old step count; init_from starts a new run from loaded weights. Pass only one.")
        if int(self.generate_eval_limit) < 0:
            raise ValueError("generate_eval_limit cannot be negative; use 0 to skip generation eval")
        if (str(self.validation_dataset) or int(self.smoke_limit) > 0 or int(self.generate_eval_limit) > 0) and int(self.save_every) <= 0:
            raise ValueError("validation_dataset, smoke_limit and generate_eval_limit need save_every > 0: they run at every checkpoint")

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "TrainingConfig":
        known = {item.name for item in cls.__dataclass_fields__.values()}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError("unsupported training config keys: " + ", ".join(unknown))
        return cls(**{key: value for key, value in values.items() if value is not None})

    @classmethod
    def from_file(cls, path: str | Path, **overrides: Any) -> "TrainingConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"training config must be a JSON object: {path}")
        return cls.from_mapping({**payload, **{key: value for key, value in overrides.items() if value not in (None, "")}})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        body = json.dumps({**self.to_dict(), "dataset": "", "out_dir": ""}, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(body.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StepOutcome:
    """What one optimization step cost and how far the judgement moved."""

    loss: float
    margin: float

    @property
    def preferred(self) -> bool:
        """Whether the policy now puts the chosen judgement ahead of the rejected one."""

        return self.margin > 0.0


class PreferenceBackend(Protocol):
    """Score one completion and take one optimization step for one pair."""

    name: str

    def score(self, record: dict[str, Any], completion: str) -> float:
        """Return the log probability the policy assigns to the completion."""

    def step(self, record: dict[str, Any], chosen: str, rejected: str, *, beta: float, reference_margin: float) -> "StepOutcome":
        """Apply one gradient step and report the loss and the margin it optimised.

        ``reference_margin`` is ``log p_ref(chosen) - log p_ref(rejected)``. The backend rebuilds
        the margin from its own scores, because a margin the caller computed under ``no_grad``
        carries no gradient; passing the reference offset keeps the objective reference-relative.
        Returning the margin as well is what lets the loop drop its own scoring passes: the step
        already computed both log probabilities, and scoring the pair again would cost one extra
        full forward per completion while reporting a dropout-free number training never saw.
        """

    def flush(self) -> None:
        """Apply any pending accumulated gradient."""

    def save(self, directory: str | Path) -> None:
        """Write the trainable state."""

    def load(self, directory: str | Path) -> None:
        """Restore the trainable state."""

    def state(self) -> dict[str, Any]:
        """Return a small audit block describing the backend."""


@dataclass
class SyntheticPolicyBackend:
    """Deterministic hashed-feature policy with a real gradient step.

    The score is ``w . f(pair, completion)`` over signed hashing features of the images,
    the prompt and the completion. That is already a sequence-level score, not a mean
    token NLL, so the synthetic backend does not apply a second length reduction. One DPO
    gradient step moves the weights towards the certified judgement, so loss, margin and
    pairwise accuracy all respond to training exactly as they do in a real run; only the
    model is not a language model.
    """

    dimension: int = 128
    learning_rate: float = 0.05
    seed: int = 0
    weights: list[float] = field(default_factory=list)
    steps: int = 0
    name: str = "synthetic-hashed-preference-v1"

    def __post_init__(self) -> None:
        if not self.weights:
            self.weights = [0.0] * max(1, int(self.dimension))

    def features(self, record: dict[str, Any], completion: str) -> list[float]:
        images = " ".join(str(item) for item in (record.get("images") or []))
        prompt = " ".join(str(item.get("text") or "") for item in (record.get("messages") or [{}])[0].get("content") or [] if isinstance(item, dict))
        text = " ".join([images, prompt, str(completion)])
        return HashingNgramEmbedder(dimension=len(self.weights)).encode_text(text)

    def score(self, record: dict[str, Any], completion: str) -> float:
        return float(sum(weight * value for weight, value in zip(self.weights, self.features(record, completion))))

    def step(self, record: dict[str, Any], chosen: str, rejected: str, *, beta: float, reference_margin: float) -> float:
        chosen_features = self.features(record, chosen)
        rejected_features = self.features(record, rejected)
        margin = (self.score(record, chosen) - self.score(record, rejected)) - float(reference_margin)
        coefficient = self.learning_rate * dpo_gradient(margin, beta)
        for index in range(len(self.weights)):
            self.weights[index] += coefficient * (chosen_features[index] - rejected_features[index])
        self.steps += 1
        return StepOutcome(loss=dpo_loss(margin, beta), margin=margin)

    def sft_step(self, record: dict[str, Any], chosen: str) -> StepOutcome:
        """Move the synthetic weights toward the certified completion only."""

        chosen_features = self.features(record, chosen)
        score = self.score(record, chosen)
        for index in range(len(self.weights)):
            self.weights[index] += self.learning_rate * chosen_features[index]
        self.steps += 1
        return StepOutcome(loss=-float(score), margin=0.0)

    def flush(self) -> None:
        return None

    def save(self, directory: str | Path) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        (target / "synthetic_policy.json").write_text(json.dumps({"name": self.name, "weights": self.weights, "steps": self.steps}, ensure_ascii=False, separators=(",", ":")) + chr(10), encoding="utf-8")

    def load(self, directory: str | Path) -> None:
        payload = json.loads((Path(directory) / "synthetic_policy.json").read_text(encoding="utf-8"))
        weights = payload.get("weights")
        if not isinstance(weights, list) or not weights:
            raise ValueError(f"synthetic checkpoint has no weights: {directory}")
        self.weights = [float(item) for item in weights]
        self.steps = int(payload.get("steps") or 0)

    def state(self) -> dict[str, Any]:
        return {"name": self.name, "dimension": len(self.weights), "steps": self.steps, "weights_l1": round(sum(abs(item) for item in self.weights), 6)}


# transformers 5 renamed ``AutoModelForVision2Seq`` to ``AutoModelForImageTextToText``; a
# run accepts whichever name the installed stack exposes so the trainer is not pinned to one
# side of that rename.
VISION_AUTO_CLASSES = ("AutoModelForImageTextToText", "AutoModelForVision2Seq")


def resolve_vision_auto_class(transformers_module: Any) -> Any:
    """Return the auto class that loads a vision-language model."""

    for name in VISION_AUTO_CLASSES:
        candidate = getattr(transformers_module, name, None)
        if candidate is not None:
            return candidate
    raise TrainingUnavailable("the transformers backend needs one of: " + ", ".join(VISION_AUTO_CLASSES))


# A vision processor resizes every image to at most ``longest_edge`` *pixels*; the Qwen3-VL
# default is 16,777,216 (16.7 megapixels). That ceiling, not the weights, is what dominates
# activation memory during a step, so a run over real figures has to be able to lower it.
# Only the ceiling moves: the floor stays put so already small icons are not rescaled.
def apply_vision_budget(processor: Any, max_pixels: int) -> dict[str, Any]:
    """Cap the vision encoder's longest edge and report the budget actually applied.

    The processor describes the budget either as a plain mapping (``preprocessor_config.json``
    writes ``{"longest_edge": ..., "shortest_edge": ...}``) or as an attribute object
    (``transformers`` loads it back as a ``SizeDict``), so both shapes are handled.
    """

    budget = int(max_pixels)
    if budget <= 0:
        return {}
    image_processor = getattr(processor, "image_processor", None)
    size = getattr(image_processor, "size", None)
    read = size.get if isinstance(size, dict) else lambda name: getattr(size, name, None)
    longest, shortest = read("longest_edge"), read("shortest_edge")
    if longest is not None:
        if shortest is not None and int(shortest) > budget:
            raise TrainingUnavailable(f"image_max_pixels={budget} is below the processor shortest_edge={shortest}")
        if isinstance(size, dict):
            image_processor.size = {**size, "longest_edge": budget}
        else:
            try:
                size.longest_edge = budget
            except Exception as exc:  # a frozen budget object cannot be resized in place
                raise TrainingUnavailable(f"the processor budget {type(size).__name__} cannot be changed: {exc}") from exc
        return {"longest_edge": budget, "shortest_edge": shortest}
    if image_processor is not None and getattr(image_processor, "max_pixels", None) is not None:
        image_processor.max_pixels = budget
        return {"longest_edge": budget, "shortest_edge": None}
    raise TrainingUnavailable("image_max_pixels is set but the processor exposes no vision budget to cap")


def _processor_call(processor: Any, *, text: str, images: Sequence[str]) -> Any:
    """Run the VLM processor over one prompt, with or without a figure.

    ``images=[]`` is not the same request as leaving the argument out: the image processor
    indexes the list it is handed, so an empty one raises ``IndexError`` instead of
    processing the text alone. Every call that can carry no figure -- a reflection, or an
    answer whose preview was never rendered -- failed on exactly this, and the failure was
    recorded as an unreadable answer rather than as a missing figure.
    """

    if images:
        return processor(text=[text], images=list(images), return_tensors="pt")
    return processor(text=[text], return_tensors="pt")


def freest_visible_gpu(torch_module: Any) -> int:
    """The visible card with the most free memory, for a load that names exactly one device.

    ``device_map={"": 0}`` is a coin flip on a shared machine. The advisory seat used to take
    GPU 0 whichever card was free, so on a box where another session already held that card
    its answer came back as ``OutOfMemoryError``; the round recorded a failed judge answer and
    escalated to a structural re-plan, which redrew every icon. The seat now takes the
    emptiest visible card, and falls back to 0 whenever the question cannot be asked: no CUDA,
    a single card, or an older torch without ``mem_get_info``.
    """

    try:
        if not torch_module.cuda.is_available():
            return 0
        count = int(torch_module.cuda.device_count())
        if count <= 1:
            return 0
        best_index, best_free = 0, -1
        for index in range(count):
            try:
                free, _total = torch_module.cuda.mem_get_info(index)
            except Exception:
                continue
            if int(free) > best_free:
                best_index, best_free = index, int(free)
        return best_index
    except Exception:
        return 0


class TransformersVlmBackend:
    """4-bit LoRA vision-language policy scored on completion tokens only."""

    name = "transformers-4bit-lora"

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.torch = self._require("torch")
        self.transformers = self._require("transformers")
        self.peft = self._require("peft")
        if not config.base_model:
            raise TrainingUnavailable("the transformers backend needs --base-model")
        self.processor = self.transformers.AutoProcessor.from_pretrained(config.base_model, trust_remote_code=True)
        self.vision_budget = apply_vision_budget(self.processor, config.image_max_pixels)
        self.model = self._load_model()
        self.optimizer = self.torch.optim.AdamW([item for item in self.model.parameters() if item.requires_grad], lr=config.learning_rate)
        self.pending = 0

    def _require(self, name: str) -> Any:
        try:
            return __import__(name)
        except ImportError as exc:
            raise TrainingUnavailable(f"the transformers backend needs {name}; it is not installed") from exc

    def _load_model(self) -> Any:
        kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": getattr(self.torch, "bfloat16", None)}
        if self.config.shard_across_gpus:
            # ``auto`` spreads the layers over every visible GPU. It matters for the 35B MoE: the
            # 33 B expert parameters cannot be 4-bit quantised, so one 80 GB card sits at 77 GiB
            # per step while two cards hold 27 + 36 GiB and run a step slightly faster.
            kwargs["device_map"] = "auto"
        elif self.config.gradient_checkpointing:
            # Gradient checkpointing is what puts this 4-bit model on one card; which card is
            # decided by which one is actually free (see ``freest_visible_gpu``).
            kwargs["device_map"] = {"": freest_visible_gpu(self.torch)}
        self.device_map = str(kwargs.get("device_map") or "cpu")
        if self.config.load_in_4bit:
            try:
                bitsandbytes = self._require("bitsandbytes")
            except TrainingUnavailable:
                raise
            kwargs["quantization_config"] = self.transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=getattr(self.torch, "bfloat16", None),
                bnb_4bit_use_double_quant=True,
            )
            del bitsandbytes
        model = resolve_vision_auto_class(self.transformers).from_pretrained(self.config.base_model, **kwargs)
        lora = self.peft.LoraConfig(**self._lora_kwargs())
        model = self.peft.get_peft_model(model, lora)
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            enable_input_require_grads = getattr(model, "enable_input_require_grads", None)
            if callable(enable_input_require_grads):
                enable_input_require_grads()
        return model

    def _lora_kwargs(self) -> dict[str, Any]:
        """Return the LoRA settings, naming the projections peft must attach to."""

        kwargs: dict[str, Any] = {"r": self.config.lora_r, "lora_alpha": self.config.lora_alpha,
                                  "lora_dropout": self.config.lora_dropout, "task_type": "CAUSAL_LM"}
        targets = [str(item) for item in self.config.lora_target_modules if str(item)]
        if targets:
            kwargs["target_modules"] = targets
        return kwargs

    def _batch(self, record: dict[str, Any], completion: str) -> dict[str, Any]:
        messages = [dict(item) for item in (record.get("messages") or []) if isinstance(item, dict)]
        content = [{"type": "text", "text": str(completion)}]
        messages.append({"role": "assistant", "content": content})
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        images = [str(item) for item in (record.get("images") or [])]
        batch = _processor_call(self.processor, text=text, images=images)
        labels = batch["input_ids"].clone()
        prompt_length = self._prompt_length(record)
        labels[:, :prompt_length] = -100
        batch["labels"] = labels
        return {key: (value.to(self.model.device) if hasattr(value, "to") else value) for key, value in batch.items()}

    def _prompt_length(self, record: dict[str, Any]) -> int:
        messages = [dict(item) for item in (record.get("messages") or []) if isinstance(item, dict)]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images = [str(item) for item in (record.get("images") or [])]
        return int(_processor_call(self.processor, text=text, images=images)["input_ids"].shape[1])

    def score(self, record: dict[str, Any], completion: str) -> float:
        self.model.eval()
        batch = self._batch(record, completion)
        with self.torch.no_grad():
            output = self.model(**batch, use_cache=False)
        return float(completion_logprob(output, batch))

    def _loss(self, record: dict[str, Any], chosen: str, rejected: str, *, beta: float, reference_margin: float) -> tuple[Any, Any]:
        # ``.loss`` is the mean NEGATIVE log likelihood over the completion tokens (the prompt is
        # masked to -100). Negating it is the mean log probability, which lets a short JSON beat a
        # long audit whenever the few remaining tokens are typical. DPO wants the sequence sum,
        # recovered by :func:`completion_logprob`. Feeding ``.loss`` in as if it were a log
        # probability would still flip the objective. The reference margin is subtracted inside
        # the graph, not only in the reported metric, because DPO is defined against the frozen
        # reference policy.
        chosen_batch = self._batch(record, chosen)
        rejected_batch = self._batch(record, rejected)
        chosen_logprob = completion_logprob(self.model(**chosen_batch, use_cache=False), chosen_batch)
        rejected_logprob = completion_logprob(self.model(**rejected_batch, use_cache=False), rejected_batch)
        margin = preference_logprob_term(chosen_logprob, rejected_logprob, float(reference_margin))
        return -self.torch.nn.functional.logsigmoid(float(beta) * margin), margin

    def generate(self, record: dict[str, Any], *, max_new_tokens: int = 1024) -> str:
        """Decode a Judge answer with thinking turned off, matching local serving.

        Training only ever sees the JSON completion. Leaving the template's thinking block
        open makes the base model write analysis until the token cap, which is exactly the
        empty-exam failure the smoke has to catch.
        """

        self.model.eval()
        messages = [dict(item) for item in (record.get("messages") or []) if isinstance(item, dict)]
        try:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images = [str(item) for item in (record.get("images") or [])]
        batch = _processor_call(self.processor, text=text, images=images)
        batch = {key: (value.to(self.model.device) if hasattr(value, "to") else value) for key, value in batch.items()}
        prompt_tokens = int(batch["input_ids"].shape[1])
        with self.torch.no_grad():
            generated = self.model.generate(**batch, max_new_tokens=int(max_new_tokens), do_sample=False)
        answer_ids = generated[:, prompt_tokens:] if prompt_tokens else generated
        return str(self.processor.batch_decode(answer_ids, skip_special_tokens=True)[0]).strip()

    def step(self, record: dict[str, Any], chosen: str, rejected: str, *, beta: float, reference_margin: float) -> StepOutcome:
        self.model.train()
        loss, margin = self._loss(record, chosen, rejected, beta=beta, reference_margin=reference_margin)
        (loss / max(1, int(self.config.gradient_accumulation))).backward()
        self.pending += 1
        if self.pending >= max(1, int(self.config.gradient_accumulation)):
            self.flush()
        return StepOutcome(loss=float(loss.detach()), margin=float(margin.detach()))

    def sft_step(self, record: dict[str, Any], chosen: str) -> StepOutcome:
        """Fit the certified JSON completion, the string serving will have to write."""

        self.model.train()
        batch = self._batch(record, chosen)
        output = self.model(**batch, use_cache=False)
        loss = output.loss
        (loss / max(1, int(self.config.gradient_accumulation))).backward()
        self.pending += 1
        if self.pending >= max(1, int(self.config.gradient_accumulation)):
            self.flush()
        return StepOutcome(loss=float(loss.detach()), margin=0.0)

    def flush(self) -> None:
        if not self.pending:
            return
        for parameters in parameters_by_device(self.model.parameters()).values():
            self.torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.pending = 0

    def save(self, directory: str | Path) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(target / "adapter"))
        self.processor.save_pretrained(str(target / "processor"))

    def load(self, directory: str | Path) -> None:
        adapter = Path(directory) / "adapter"
        if not adapter.is_dir():
            raise ValueError(f"transformers checkpoint has no adapter: {adapter}")
        loader = getattr(self.peft, "load_peft_weights", None)
        applier = getattr(self.peft, "set_peft_model_state_dict", None)
        if loader is None or applier is None:
            raise TrainingUnavailable("this peft build cannot load an adapter into an existing model")
        # ``PeftModel.from_pretrained`` on a model that is already a PeftModel wraps it a second
        # time; the doubled ``base_model.model`` prefix then stops every key from matching, and
        # peft only warns about it. A resumed run would quietly keep training the untrained policy,
        # so the weights are applied in place and any key that does not land is an error.
        weights = loader(str(adapter))
        expected = len([name for name, _ in self.model.named_parameters() if "lora_" in str(name)])
        loaded = len([key for key in weights if "lora_" in str(key)])
        result = applier(self.model, weights)
        # ``missing_keys`` is not a signal here: it always lists the frozen base weights, which no
        # adapter file carries. A key the model did not accept, or fewer adapter keys than the model
        # has, is the signal that the checkpoint did not land.
        stray = [str(item) for item in (getattr(result, "unexpected_keys", None) or [])]
        if stray or loaded != expected or not loaded:
            detail = ", ".join(sorted(stray)[:3]) or "adapter key count mismatch"
            raise ValueError(
                "the checkpoint adapter does not fit this model; refusing to resume from a checkpoint "
                f"that would silently train from scratch (unexpected={len(stray)}, loaded={loaded}, "
                f"expected={expected}): {detail}"
            )

    def state(self) -> dict[str, Any]:
        trainable = sum(int(item.numel()) for item in self.model.parameters() if item.requires_grad)
        return {"name": self.name, "base_model": self.config.base_model, "trainable_parameters": trainable, "load_in_4bit": bool(self.config.load_in_4bit), "gradient_checkpointing": bool(self.config.gradient_checkpointing)}


def build_backend(config: TrainingConfig) -> PreferenceBackend:
    """Build the backend the config names, never a substitute for it."""

    backend = str(config.backend or "").strip().casefold()
    if backend == SYNTHETIC_BACKEND:
        return SyntheticPolicyBackend(learning_rate=config.learning_rate, seed=config.seed)
    if backend == "transformers":
        return TransformersVlmBackend(config)
    raise TrainingUnavailable(f"unknown training backend: {config.backend!r}; choose one of {', '.join(TRAINING_BACKENDS)}")


def load_training_records(dataset: str | Path) -> list[dict[str, Any]]:
    """Read the DPO records of one export, failing when the export is unusable."""

    root = Path(dataset)
    path = root / "dpo.jsonl" if root.is_dir() else root
    if not path.is_file():
        raise FileNotFoundError(f"no DPO export at {path}")
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        missing = [field for field in REQUIRED_RECORD_FIELDS if not row.get(field)]
        if missing:
            raise ValueError(f"dpo record {row.get('example_id')!r} misses {', '.join(missing)}")
        records.append(row)
    if not records:
        raise ValueError(f"the DPO export has no records: {path}")
    return records


def reference_scoring(config: TrainingConfig) -> dict[str, Any]:
    """The fields that decide whether a cached reference log-prob may be reused."""

    # ``image_max_pixels`` decides how much of each figure the model sees. ``logprob_reduction``
    # decides whether those scores are mean-token or sequence sums. Leaving either out would let
    # a resumed run quietly reuse numbers taken under a different scoring contract.
    return {
        "backend": config.backend,
        "base_model": config.base_model,
        "load_in_4bit": bool(config.load_in_4bit),
        "max_length": int(config.max_length),
        "image_max_pixels": int(config.image_max_pixels),
        "logprob_reduction": COMPLETION_LOGPROB_REDUCTION,
    }


def reference_key(config: TrainingConfig, records: Sequence[dict[str, Any]]) -> str:
    """Fingerprint the reference pass: same config and same pairs, same cache."""

    scoring = reference_scoring(config)
    digest = hashlib.sha256()
    digest.update(json.dumps(scoring, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(str(len(records)).encode("utf-8"))
    for record in records:
        digest.update(str(record.get("example_id")).encode("utf-8"))
        digest.update(hashlib.sha256(str(record.get("chosen")).encode("utf-8")).hexdigest().encode("utf-8"))
        digest.update(hashlib.sha256(str(record.get("rejected")).encode("utf-8")).hexdigest().encode("utf-8"))
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def precompute_reference(records: Sequence[dict[str, Any]], backend: PreferenceBackend, config: TrainingConfig, *, out_dir: str | Path, cache_dir: str | Path | None = None, allow_compute: bool = True) -> dict[str, Any]:
    """Score the pairs with the frozen reference once and cache them by fingerprint.

    The reference is the policy *before* any step is taken. A resumed run must find its
    reference in the cache: scoring a trained policy would silently move the reference,
    so a missing entry fails instead.
    """

    target = Path(out_dir) / REFERENCE_CACHE
    sources = [Path(cache_dir) / REFERENCE_CACHE, target] if cache_dir is not None else [target]
    key = reference_key(config, records)
    cached: dict[str, tuple[float, float]] = {}
    for source in sources:
        for row in _read_jsonl(source):
            if str(row.get("key")) == key:
                cached.setdefault(str(row.get("example_id")), (float(row.get("chosen")), float(row.get("rejected"))))
    if not allow_compute:
        missing = [str(record.get("example_id")) for record in records if str(record.get("example_id")) not in cached]
        if missing:
            raise ValueError("the reference cache does not cover this run; recompute it from the base model: " + ", ".join(sorted(missing)[:5]))
    computed = 0
    rows = []
    for record in records:
        example_id = str(record.get("example_id"))
        if example_id in cached:
            chosen, rejected = cached[example_id]
        else:
            chosen = backend.score(record, str(record["chosen"]))
            rejected = backend.score(record, str(record["rejected"]))
            computed += 1
        rows.append({"key": key, "example_id": example_id, "chosen": chosen, "rejected": rejected})
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + chr(10) for row in rows), encoding="utf-8")
    return {"key": key, "reused": len(rows) - computed, "computed": computed, "path": str(target), "reference": {str(row["example_id"]): (float(row["chosen"]), float(row["rejected"])) for row in rows}}


def _epoch_order(records: Sequence[dict[str, Any]], seed: int, epoch: int) -> list[int]:
    return sorted(range(len(records)), key=lambda index: hashlib.sha256(f"{seed}|{epoch}|{records[index].get('example_id')}".encode("utf-8")).hexdigest())


def _defect_types(record: dict[str, Any]) -> list[str]:
    basis = record.get("preference_basis") if isinstance(record.get("preference_basis"), dict) else {}
    return sorted({str(item.get("diagnosis_type")) for item in (basis.get("known_defects") or []) if isinstance(item, dict) and item.get("diagnosis_type")})


def _completion_stats(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize the reference completions the trainer is asked to reproduce."""

    lengths = [len(str(record.get("chosen") or "")) for record in records]
    valid = 0
    for record in records:
        try:
            value = json.loads(str(record.get("chosen")))
        except ValueError:
            continue
        valid += 1 if isinstance(value, dict) else 0
    seen: dict[str, int] = {}
    for record in records:
        text = str(record.get("chosen") or "")
        seen[text] = seen.get(text, 0) + 1
    repeated = sum(count for count in seen.values() if count > 1)
    ordered = sorted(lengths)
    median = ordered[len(ordered) // 2] if ordered else 0
    return {
        "records": len(records),
        "json_valid_rate": round(valid / len(records), 6) if records else 0.0,
        "mean_output_chars": round(sum(lengths) / len(lengths), 3) if records else 0.0,
        "median_output_chars": median,
        "templated_ratio": round(repeated / len(records), 6) if records else 0.0,
        "distinct_completions": len(seen),
    }


def version_lock(config: TrainingConfig, backend: PreferenceBackend, dataset_path: str | Path | None = None) -> dict[str, Any]:
    """The pinned versions a run has to be reproduced with."""

    def version(name: str) -> str:
        try:
            module = __import__(name)
        except ImportError:
            return "not_installed"
        return str(getattr(module, "__version__", "unknown"))

    digest = ""
    if dataset_path and Path(dataset_path).is_file():
        digest = hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()
    return {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "python": platform.python_version(),
        "torch": version("torch"),
        "transformers": version("transformers"),
        "peft": version("peft"),
        "bitsandbytes": version("bitsandbytes"),
        "accelerate": version("accelerate"),
        "pillow": version("PIL"),
        "backend": backend.name,
        "base_model": config.base_model,
        "load_in_4bit": bool(config.load_in_4bit),
        "image_max_pixels": int(config.image_max_pixels),
        "vision_budget": dict(getattr(backend, "vision_budget", {})),
        "device_map": str(getattr(backend, "device_map", "single-device")),
        "lora": {"r": config.lora_r, "alpha": config.lora_alpha, "dropout": config.lora_dropout,
                 "target_modules": [str(item) for item in config.lora_target_modules]},
        "judge_contract": JUDGE_CONTRACT_VERSION,
        "prompt_version": PROMPT_VERSION,
        "pair_schema_version": PAIR_SCHEMA_VERSION,
        "logprob_reduction": COMPLETION_LOGPROB_REDUCTION,
        "config_fingerprint": config.fingerprint(),
        "dataset_sha256": digest,
    }


def _checkpoint(directory: Path, backend: PreferenceBackend, *, step: int, epoch: int, state: dict[str, Any], reference_path: str | Path | None = None) -> str:
    """Write a self contained checkpoint: trainable state, run state and the reference."""

    backend.save(directory)
    (directory / STATE_FILE).write_text(json.dumps({**state, "step": step, "epoch": epoch}, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    if reference_path and Path(reference_path).is_file():
        shutil.copyfile(reference_path, directory / REFERENCE_CACHE)
    return str(directory)


def evaluate_preference_pairs(policy: PreferenceBackend, records: Sequence[dict[str, Any]], *, limit: int = 0) -> dict[str, Any]:
    """Score a held-out split with the policy that is training and report how well it ranks pairs.

    The margin is the quantity DPO optimises, so a median that keeps climbing after the accuracy
    has saturated is the collapse signal: the policy is separating the pairs by driving the
    rejected judgement towards zero probability rather than by judging better, and on a 35B
    backbone past roughly twenty the model stops emitting usable text at all. Scoring the pairs
    costs two forwards each, which is why a checkpoint can be judged at every save.
    """

    sample = list(records)
    if int(limit) > 0:
        sample = sample[: int(limit)]
    if not sample:
        raise ValueError("a held-out evaluation needs at least one preference pair")
    margins = [policy.score(record, str(record["chosen"])) - policy.score(record, str(record["rejected"])) for record in sample]
    ordered = sorted(margins)
    return {
        "pairs": len(sample),
        "pairwise_accuracy": round(sum(1 for margin in margins if margin > 0) / len(margins), 6),
        "mean_margin": round(sum(margins) / len(margins), 6),
        "median_margin": round(ordered[len(ordered) // 2], 6),
        "min_margin": round(ordered[0], 6),
        "max_margin": round(ordered[-1], 6),
    }


def checkpoint_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Rank a checkpoint: smoke pass, then held-out pairwise, then earlier step."""

    smoke = 1 if row.get("smoke_passed", True) else 0
    return (smoke, float(row.get("pairwise_accuracy") or 0.0), -int(row.get("step") or 0))


def pairwise_stale_count(evaluations: Sequence[dict[str, Any]]) -> int:
    """How many consecutive smoke-passing checkpoints failed to raise held-out pairwise."""

    best_accuracy = -1.0
    stale = 0
    for row in evaluations:
        if not row.get("smoke_passed", True):
            continue
        accuracy = float(row.get("pairwise_accuracy") or 0.0)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            stale = 0
        else:
            stale += 1
    return stale


def train_dpo(config: TrainingConfig, *, backend: PreferenceBackend | None = None) -> dict[str, Any]:
    """Run the DPO loop, writing checkpoints, metrics and the version lock."""

    run_dir = Path(config.out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    records = load_training_records(config.dataset)
    validation_records = load_training_records(config.validation_dataset) if config.validation_dataset else []
    policy = backend if backend is not None else build_backend(config)
    checkpoint_root = run_dir / CHECKPOINT_DIR
    resume_dir = Path(config.resume_from) if config.resume_from else None
    init_dir = Path(config.init_from) if config.init_from else None
    state: dict[str, Any] = {"step": 0, "epoch": 0, "metrics_rows": 0}
    if resume_dir is not None:
        stored = json.loads((resume_dir / STATE_FILE).read_text(encoding="utf-8"))
        state = {"step": int(stored.get("step") or 0), "epoch": int(stored.get("epoch") or 0), "metrics_rows": int(stored.get("metrics_rows") or 0)}
        policy.load(resume_dir)
    elif init_dir is not None:
        if not init_dir.is_dir():
            raise ValueError(f"init_from is not a checkpoint directory: {init_dir}")
        policy.load(init_dir)
    sft = str(config.objective) == SFT_OBJECTIVE
    if sft:
        if not callable(getattr(policy, "sft_step", None)):
            raise TrainingUnavailable("this backend cannot run SFT")
        reference = {"key": "sft-no-reference", "reused": 0, "computed": 0, "path": "", "reference": {}}
        reference_pairs: dict[str, tuple[float, float]] = {}
    else:
        reference = precompute_reference(records, policy, config, out_dir=run_dir, cache_dir=resume_dir, allow_compute=resume_dir is None)
        reference_pairs = reference["reference"]
    metrics_path = run_dir / METRICS_FILE
    validation_path = run_dir / VALIDATION_METRICS_FILE
    smoke_path = run_dir / GENERATION_SMOKE_FILE
    generation_path = run_dir / GENERATION_VAL_FILE
    buffer: list[dict[str, Any]] = []

    def flush_metrics() -> None:
        if not buffer:
            return
        with metrics_path.open("a", encoding="utf-8") as handle:
            for row in buffer:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + chr(10))
        buffer.clear()

    if not resume_dir:
        metrics_path.write_text("", encoding="utf-8")
        if validation_records:
            validation_path.write_text("", encoding="utf-8")
        if int(config.smoke_limit) > 0:
            smoke_path.write_text("", encoding="utf-8")
        if int(config.generate_eval_limit) > 0:
            generation_path.write_text("", encoding="utf-8")
    evaluations = _read_jsonl(validation_path) if validation_path.is_file() else []
    smokes = _read_jsonl(smoke_path) if smoke_path.is_file() else []
    best = max((row for row in evaluations if row.get("smoke_passed", True)), key=checkpoint_sort_key, default=None)
    stopped_early = False
    stop_reason = ""
    total_steps = len(records) * max(1, int(config.epochs))
    if config.max_steps:
        total_steps = min(total_steps, int(config.max_steps))
    counters: dict[str, Any] = {"loss": 0.0, "margin": 0.0, "preferred": 0, "steps": 0, "defect": {}, "domain": {}}
    checkpoints: list[str] = []
    step = int(state["step"])
    epoch = int(state["epoch"])
    while step < total_steps:
        order = _epoch_order(records, config.seed, epoch)
        for index in order:
            if step >= total_steps:
                break
            record = records[index]
            chosen, rejected = str(record["chosen"]), str(record["rejected"])
            if sft:
                outcome = policy.sft_step(record, chosen)
            else:
                reference_chosen, reference_rejected = reference_pairs[str(record["example_id"])]
                # The step computes both log probabilities to build its gradient, so the margin it
                # reports is the one it optimised: scoring the pair again here would double the number
                # of forwards per record and print a dropout-free number the optimiser never saw.
                outcome = policy.step(record, chosen, rejected, beta=config.beta,
                                      reference_margin=reference_chosen - reference_rejected)
            loss, margin, preferred = outcome.loss, outcome.margin, outcome.preferred
            step += 1
            counters["loss"] += loss
            counters["margin"] += margin
            counters["preferred"] += 1 if preferred else 0
            counters["steps"] += 1
            domain = str(record.get("domain") or "unknown")
            counters["domain"].setdefault(domain, [0, 0])
            counters["domain"][domain][0] += 1 if preferred else 0
            counters["domain"][domain][1] += 1
            for defect in _defect_types(record) or ["none"]:
                counters["defect"].setdefault(defect, [0, 0])
                counters["defect"][defect][0] += 1 if preferred else 0
                counters["defect"][defect][1] += 1
            buffer.append({"step": step, "epoch": epoch, "example_id": str(record.get("example_id")), "loss": round(loss, 6), "margin": round(margin, 6), "preferred": preferred, "domain": domain, "defects": _defect_types(record)})
            choice_due = str(config.selection) == "choice" and step == total_steps
            if (config.save_every and step % int(config.save_every) == 0) or choice_due:
                flush_metrics()
                target = checkpoint_root / f"step-{step:06d}"
                checkpoint_root.mkdir(parents=True, exist_ok=True)
                checkpoints.append(_checkpoint(target, policy, step=step, epoch=epoch, state={"metrics_rows": counters["steps"], "reference_key": reference["key"]}, reference_path=reference["path"]))
                smoke = None
                if int(config.smoke_limit) > 0:
                    smoke = {"step": step, "epoch": epoch, "checkpoint": str(target),
                             **run_generation_smoke(policy, validation_records or records, limit=int(config.smoke_limit),
                                                    max_new_tokens=int(config.smoke_max_new_tokens), pass_rate=float(config.smoke_pass_rate))}
                    smokes.append(smoke)
                    with smoke_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(smoke, ensure_ascii=False, separators=(",", ":")) + chr(10))
                    if not smoke["passed"]:
                        stopped_early = True
                        stop_reason = "generation_smoke"
                if str(config.selection) == "choice" and validation_records:
                    from .choice_items import choice_is_better, evaluate_choice_records

                    scored = evaluate_choice_records(policy, validation_records, limit=int(config.eval_limit), max_new_tokens=int(config.smoke_max_new_tokens))
                    if not scored.get("skipped"):
                        row = {"step": step, "epoch": epoch, "checkpoint": str(target),
                               "smoke_passed": True if smoke is None else bool(smoke["passed"]),
                               "selection": "choice",
                               "pairs": scored.get("pairs"),
                               "legal_rate": scored.get("legal_rate"),
                               "figure_accuracy": scored.get("figure_accuracy"),
                               "defect_accuracy": scored.get("defect_accuracy"),
                               "hard_recall": scored.get("hard_recall"),
                               "pairwise_accuracy": scored.get("figure_accuracy") or 0.0}
                        evaluations.append(row)
                        with validation_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + chr(10))
                        if row.get("smoke_passed", True) and choice_is_better(row, best):
                            best = row
                else:
                    generated = None
                    if int(config.generate_eval_limit) > 0 and validation_records:
                        generated = {"step": step, "epoch": epoch, "checkpoint": str(target),
                                     **evaluate_generated_judgements(policy, validation_records, limit=int(config.generate_eval_limit),
                                                                    max_new_tokens=int(config.smoke_max_new_tokens))}
                        if not generated.get("skipped"):
                            with generation_path.open("a", encoding="utf-8") as handle:
                                handle.write(json.dumps(generated, ensure_ascii=False, separators=(",", ":")) + chr(10))
                    if generated and not generated.get("skipped"):
                        row = {"step": step, "epoch": epoch, "checkpoint": str(target),
                               "smoke_passed": True if smoke is None else bool(smoke["passed"]),
                               "smoke_pass_rate": None if smoke is None else smoke.get("pass_rate"),
                               "selection": "generate",
                               "pairs": generated.get("pairs"),
                               "pairwise_accuracy": generated.get("pairwise_accuracy") or 0.0,
                               "submit_rate": generated.get("submit_rate"),
                               "diagnosis_recall": generated.get("diagnosis_recall"),
                               "median_margin": 0.0,
                               "mean_margin": 0.0,
                               "min_margin": 0.0,
                               "max_margin": 0.0}
                        evaluations.append(row)
                        with validation_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + chr(10))
                        if row.get("smoke_passed", True) and (best is None or checkpoint_sort_key(row) > checkpoint_sort_key(best)):
                            best = row
                        if not stopped_early and int(config.pairwise_patience) > 0 and pairwise_stale_count(evaluations) >= int(config.pairwise_patience):
                            stopped_early = True
                            stop_reason = "pairwise_patience"
                    elif validation_records:
                        row = {"step": step, "epoch": epoch, "checkpoint": str(target),
                               "smoke_passed": True if smoke is None else bool(smoke["passed"]),
                               "smoke_pass_rate": None if smoke is None else smoke.get("pass_rate"),
                               "selection": "ranking",
                               **evaluate_preference_pairs(policy, validation_records, limit=config.eval_limit)}
                        evaluations.append(row)
                        with validation_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + chr(10))
                        if row.get("smoke_passed", True) and (best is None or checkpoint_sort_key(row) > checkpoint_sort_key(best)):
                            best = row
                        if not sft and not stopped_early and config.early_stop_margin and row["median_margin"] >= float(config.early_stop_margin):
                            stopped_early = True
                            stop_reason = "early_stop_margin"
                        if not sft and not stopped_early and config.collapse_margin and row["median_margin"] >= float(config.collapse_margin) and row["pairwise_accuracy"] >= SATURATED_PAIRWISE:
                            stopped_early = True
                            stop_reason = "collapse_margin"
                        if not stopped_early and int(config.pairwise_patience) > 0 and pairwise_stale_count(evaluations) >= int(config.pairwise_patience):
                            stopped_early = True
                            stop_reason = "pairwise_patience"
                if stopped_early:
                    break
        policy.flush()
        epoch += 1
        if stopped_early:
            break
    flush_metrics()
    totals = {
        "steps": counters["steps"],
        "mean_loss": round(counters["loss"] / counters["steps"], 6) if counters["steps"] else None,
        "mean_margin": round(counters["margin"] / counters["steps"], 6) if counters["steps"] else None,
        "pairwise_accuracy": round(counters["preferred"] / counters["steps"], 6) if counters["steps"] else None,
        "train_memorization": round(counters["preferred"] / counters["steps"], 6) if counters["steps"] else None,
        "per_domain_accuracy": {key: round(value[0] / value[1], 6) for key, value in sorted(counters["domain"].items())},
        "per_defect_accuracy": {key: round(value[0] / value[1], 6) for key, value in sorted(counters["defect"].items())},
    }
    from .judge_schema import GATE_DIAGNOSIS_TYPES

    totals["gated_recall"] = {key: totals["per_defect_accuracy"].get(key) for key in GATE_DIAGNOSIS_TYPES}
    data_report = _completion_stats(records)
    (run_dir / DATA_REPORT_FILE).write_text(json.dumps(data_report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    lock = version_lock(config, policy, config.dataset if Path(config.dataset).is_file() else Path(config.dataset) / "dpo.jsonl")
    (run_dir / LOCK_FILE).write_text(json.dumps(lock, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    if best is not None:
        if best.get("selection") == "choice":
            selection = BEST_CHOICE_SELECTION
        elif best.get("selection") == "generate":
            selection = BEST_GENERATION_SELECTION
        else:
            selection = BEST_CHECKPOINT_SELECTION
        (run_dir / BEST_CHECKPOINT_FILE).write_text(json.dumps({
            "schema_version": TRAINING_SCHEMA_VERSION,
            "selection": selection,
            "step": int(best["step"]),
            "checkpoint": str(best["checkpoint"]),
            "metrics": {key: value for key, value in best.items() if key not in {"step", "checkpoint"}},
        }, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    report = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "config": config.to_dict(),
        "backend": policy.state(),
        "records": len(records),
        "target_steps": total_steps,
        "started_at_step": int(state["step"]),
        "resumed_from": str(resume_dir) if resume_dir else "",
        "initialized_from": str(init_dir) if init_dir else "",
        "validation": {
            "dataset": str(config.validation_dataset),
            "metrics_path": str(validation_path) if validation_records else "",
            "evaluations": evaluations,
            "best": best,
            "best_checkpoint": str(best["checkpoint"]) if best else "",
        },
        "generation_smoke": {
            "path": str(smoke_path) if int(config.smoke_limit) > 0 else "",
            "rows": smokes,
        },
        "stopped_early": stopped_early,
        "stopped_early_reason": stop_reason,
        "reference_cache": {"key": reference["key"], "reused": reference["reused"], "computed": reference["computed"], "path": reference["path"]},
        "metrics": totals,
        "data_report": data_report,
        "checkpoints": checkpoints,
        "metrics_path": str(metrics_path),
        "version_lock": lock,
        "ok": True,
    }
    (run_dir / REPORT_FILE).write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    return report
