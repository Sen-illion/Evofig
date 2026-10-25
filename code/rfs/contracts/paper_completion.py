from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


def _normalized(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _label(item: dict[str, Any]) -> str:
    return str(item.get("visible_label") or item.get("name") or item.get("text") or item.get("statement") or item.get("label") or item.get("id") or "").strip()


def _stable_id(prefix: str, label: str) -> str:
    slug = "_".join(part for part in re.sub(r"[^a-z0-9]+", " ", label.casefold()).split() if part)[:48]
    return f"{prefix}_{slug or 'item'}"


@dataclass(frozen=True)
class ConceptRule:
    key: str
    label: str
    field: str
    role: str
    pattern: str
    aliases: tuple[str, ...] = ()
    requires_overview: bool = False
    context_pattern: str | None = None


EMBEDDING_SUM_CONTEXT = r"\btoken(?: embeddings?)?\s*,\s*(?:the\s+)?(?:segment|segmentation)(?: embeddings?)?\s*,?\s+and\s+(?:the\s+)?position embeddings?\b"


CONCEPT_RULES = (
    ConceptRule("input_image", "Input Image", "inputs", "input", r"\binput images?\b|\bsplit an image\b|\braw image\b|\btest images?\b|\bimages? are shown\b", ("image", "2d image")),
    ConceptRule("images", "Images", "inputs", "input modality", r"\b(?:batch of\s+)?\(?images?\s*[,)]", ("training images", "image batch"), True),
    ConceptRule("image", "Image", "inputs", "input modality", r"\bimages?\s*\+\s*text\b|\bimage encoder\b", ("images", "input image"), True),
    ConceptRule("text", "Text", "inputs", "input modality", r"\b(?:batch of\s+)?\(?image\s*,\s*text\)?|\bimages?\s*\+\s*text\b|\btext training examples?\b", ("captions", "training text", "text batch")),
    ConceptRule("audio", "Audio", "inputs", "input modality", r"\baudio\b", (), True),
    ConceptRule("depth", "Depth", "inputs", "input modality", r"\bdepth\b", (), True),
    ConceptRule("thermal", "Thermal", "inputs", "input modality", r"\bthermal\b", (), True),
    ConceptRule("imu", "IMU", "inputs", "input modality", r"\bimu\b", (), True),
    ConceptRule("prompt", "Prompt", "inputs", "input", r"\b(?:input|segmentation|point|box|mask) prompts?\b|\bprompt encoder\b", ("sparse prompts", "dense prompts", "segmentation prompt"), True),
    ConceptRule("task_input", "Input", "inputs", "input", r"\bgiven an input\b|\btask (?:prompt|instruction)\b", ("task prompt", "task instruction"), True),
    ConceptRule("camera_ray", "Camera Rays", "inputs", "input", r"\bcamera rays?\b", ("camera ray", "ray", "rays")),
    ConceptRule("viewing_direction", "Viewing Direction", "inputs", "conditioning", r"\bview(?:ing)? directions?\b", ("view direction",)),
    ConceptRule("object_queries", "Object Queries", "inputs", "conditioning", r"\bobject quer(?:y|ies)\b", ("learned object queries",)),
    ConceptRule("class_descriptions", "Class Descriptions", "inputs", "conditioning", r"\b(?:names or descriptions|class names|class descriptions|text prompts)\b", ("class names", "text prompts")),
    ConceptRule("cnn_backbone", "CNN Backbone", "modules", "feature extractor", r"\b(?:conventional\s+|common\s+)?cnn backbone\b|\bconvolutional backbone\b", ("backbone", "convolutional backbone")),
    ConceptRule("backbone", "Backbone", "modules", "feature extractor", r"\bbackbones?\b", ("backbone architecture",), True),
    ConceptRule("image_encoder", "Image Encoder", "modules", "encoder", r"\bimage encoder\b"),
    ConceptRule("text_encoder", "Text Encoder", "modules", "encoder", r"\btext encoder\b"),
    ConceptRule("transformer_encoder", "Transformer Encoder", "modules", "encoder", r"\btransformer encoder\b|^encoder$", (), True),
    ConceptRule("transformer_decoder", "Transformer Decoder", "modules", "decoder", r"\btransformer decoder\b|\b(?:the\s+)?decoder receives\b|^decoder$", (), True),
    ConceptRule("position_embedding", "Position Embedding", "modules", "conditioning", r"\bposition embeddings?\b", ("positional embedding",), True),
    ConceptRule("positional_encoding", "Positional Encoding", "modules", "conditioning", r"\bposition(?:al)? encod(?:ing|ings)\b"),
    ConceptRule("image_patches", "Image Patches", "modules", "intermediate", r"\bimage patches?\b|\bfixed-size patches\b", ("patches",), True),
    ConceptRule("linear_projection", "Linear Projection of Flattened Patches", "modules", "projection", r"\blinear projection(?: of flattened patches)?\b|\blinearly embed\b", ("linear projection",), True),
    ConceptRule("class_token", "Class Token", "modules", "conditioning", r"\bclass(?:ification)? token\b|\bclass embedding\b", ("classification token", "class embedding"), True),
    ConceptRule("mlp_head", "MLP Head", "modules", "prediction head", r"\bmlp head\b|\bclassification head[^.]{0,100}\bmlp\b", ("mlp",)),
    ConceptRule("sampled_points", "Sampled 3D Points", "modules", "intermediate", r"\bsampl(?:e|ed|ing)\s+(?:the\s+)?(?:5d coordinates|3d points)\b|\b5d coordinates\b", ("5d coordinates", "spatial locations")),
    ConceptRule("prediction_ffn", "Feed Forward Network", "modules", "prediction head", r"\bfeed[ -]?forward network\b|\bprediction ffn\b|\bffn\b", ("ffn", "prediction feed-forward network")),
    ConceptRule("region_proposals", "Region Proposals", "modules", "proposal artifact", r"\bregion proposals?\b|\bregion proposal network\b|\brpn\b", ("proposals", "region proposal network (rpn)")),
    ConceptRule("roi_align", "RoIAlign", "modules", "alignment", r"\broialign\b"),
    ConceptRule("generate", "Generate", "modules", "generator", r"\bstarts? by generating\b|\bgenerate(?:s|d| an output)?\b", ("init", "generator"), True),
    ConceptRule("initial_output", "Initial Output", "modules", "artifact", r"\binitial output\b|\bpreviously generated output\b", (), True),
    ConceptRule("feedback", "Feedback", "modules", "feedback provider", r"\bget feedback\b|\breceive feedback\b|\bfeedback provider\b", ("feedback provider",), True),
    ConceptRule("self_feedback", "Self-Feedback", "modules", "feedback artifact", r"\bself[ -]?feedback\b|\bfeedback is passed back\b", ("self-feedback (nl)",), True),
    ConceptRule("refine", "Refine", "modules", "refiner", r"\brefines? the previously generated output\b|\biterate\s*/\s*refine\b", ("refiner",), True),
    ConceptRule("modality_encoders", "Modality Encoders", "modules", "module group", r"\bmodality encoders?\b|\bencoders for each modality\b", ("transformer-based encoders for each modality",)),
    ConceptRule("joint_embedding", "Joint Embedding Space", "modules", "shared representation", r"\b(?:single shared )?(?:joint|common) embedding space\b|\bshared representation space\b", ("joint embedding", "single shared joint embedding space")),
    ConceptRule("prompt_encoder", "Prompt Encoder", "modules", "encoder", r"\bprompt encoder\b"),
    ConceptRule("mask_decoder", "Mask Decoder", "modules", "decoder", r"\bmask decoder\b"),
    ConceptRule("mlp", "MLP", "modules", "neural field", r"\bneural radiance field\b|\bfeeding[^.]{0,100}\bmlp\b|\bmlp to produce\b", ("neural radiance field", "neural network")),
    ConceptRule("image_embeddings", "Image Embeddings", "modules", "representation", r"\bimage embeddings?\b|\bimage features?\b", ("image embedding", "image features"), True),
    ConceptRule("text_embeddings", "Text Embeddings", "modules", "representation", r"\btext embeddings?\b|\btext features?\b", ("text embedding", "text features")),
    ConceptRule("contrastive_objective", "Contrastive Learning", "modules", "training objective", r"\bcontrastive (?:pre-?training|learning|objective|loss)\b|\bcorrect pairings?\b", ("contrastive pre-training", "image-text contrastive loss", "correct pairings")),
    ConceptRule("bipartite_matching", "Bipartite Matching", "modules", "training objective", r"\bbipartite matching\b|\bhungarian matching\b", ("hungarian matching", "set prediction loss")),
    ConceptRule("volume_density", "Volume Density", "modules", "field output", r"\bvolume density\b", ("density", "sigma")),
    ConceptRule("color", "Color", "modules", "field output", r"\b(?:rgb|emitted) colou?r\b|\bradiance\b|\bcolou?r and volume density\b", ("rgb color", "radiance"), True),
    ConceptRule("volume_rendering", "Volume Rendering", "modules", "renderer", r"\b(?:differentiable\s+)?volume rendering\b", ("differentiable volume rendering",)),
    ConceptRule("class_predictions", "Class Predictions", "outputs", "output", r"\bclass (?:prediction|predictions|labels?)\b|\bpredicts?[^.]{0,80}\bclass\b", ("classes", "class labels")),
    ConceptRule("class_prediction", "Class Prediction", "outputs", "output", r"\bimage classification predictions?\b|\bclassification predictions?\b|\bclass label\b", ("image classification predictions", "classification predictions", "class label", "class")),
    ConceptRule("box_predictions", "Bounding Box Predictions", "outputs", "output", r"\bbounding box(?:es| predictions?)?\b|\bbox predictions?\b", ("bounding boxes", "box predictions")),
    ConceptRule("classification", "Classification", "outputs", "output head", r"\bclassification head\b|\bpredicts? the class label\b", ("class label",)),
    ConceptRule("box_regression", "Bounding-box Regression", "outputs", "output head", r"\bbounding[ -]?box regression\b|\bbox regression\b", ("bounding box", "box branch")),
    ConceptRule("mask_branch", "Mask Branch", "outputs", "output head", r"\bmask branch\b"),
    ConceptRule("refined_output", "Refined Output", "outputs", "output", r"\brefined output\b|\brefines? (?:the )?(?:previously generated )?output\b|\bquality of the output improves\b"),
    ConceptRule("emergent_alignment", "Emergent Cross-modal Alignment", "outputs", "output", r"\bemergent (?:cross-modal )?alignments?\b|\bbinding (?:agent|property)\b", ("emergent alignment", "binding agent", "binding property")),
    ConceptRule("valid_mask", "Valid Segmentation Mask", "outputs", "output", r"\bvalid masks?\b|\bobject masks?\b", ("valid masks", "valid mask"), True),
    ConceptRule("zero_shot_classifier", "Zero-Shot Classifier", "outputs", "inference output", r"\bzero[ -]?shot (?:linear )?classifier\b", ("zero-shot linear classifier",)),
    ConceptRule("zero_shot_prediction", "Zero-shot Prediction", "outputs", "inference output", r"\bzero[ -]?shot prediction\b", ("zero-shot prediction",)),
    ConceptRule("rendered_image", "Rendered Image", "outputs", "output", r"\b(?:rendered|synthesi[sz]ed) (?:images?|views?)\b|\bsynthesi[sz]e images?\b|\bnovel views?\b", ("synthesized image", "synthesized view", "novel view")),
    ConceptRule("source_tokens", "Inputs", "inputs", "source sequence", r"\bencoder maps an input sequence(?: of symbol representations)?\b|\blearned embeddings to convert the input\s+tokens\b", ("input sequence", "source tokens")),
    ConceptRule("input_embedding", "Input Embedding", "modules", "embedding", r"\blearned embeddings to convert the input\s+tokens\b|\binput embeddings?\b", ("input embeddings",), context_pattern=r"\bencoder\b.*\bdecoder\b|\bencoder and decoder stacks\b"),
    ConceptRule("encoder_stack", "Encoder Stack", "modules", "encoder", r"\bencoder is composed of a stack\b|\bencoder stack\b", ("encoder", "transformer encoder")),
    ConceptRule("target_tokens", "Outputs (shifted right)", "inputs", "target sequence", r"\boutput embeddings? (?:are )?offset by one position\b|\binput\s+tokens and output tokens\b", ("target tokens", "target sequence", "output sequence", "outputs shifted right")),
    ConceptRule("output_embedding", "Output Embedding", "modules", "embedding", r"\boutput embeddings?\b|\binput\s+tokens and output tokens\b", ("output embeddings",), context_pattern=r"\bencoder\b.*\bdecoder\b|\bencoder and decoder stacks\b"),
    ConceptRule("decoder_stack", "Decoder Stack", "modules", "decoder", r"\bdecoder is also composed of a stack\b|\bdecoder stack\b", ("decoder", "transformer decoder")),
    ConceptRule("output_linear", "Linear", "modules", "output projection", r"\blearned linear transfor(?:mation|\-\s*mation) and softmax function to convert the decoder output\b", ("linear layer", "output projection")),
    ConceptRule("output_softmax", "Softmax", "modules", "output normalization", r"\blinear transfor(?:mation|\-\s*mation) and softmax function to convert the decoder output\b", ("softmax layer",)),
    ConceptRule("output_probabilities", "Output Probabilities", "outputs", "output", r"\bpredicted next-token probabilities\b", ("predicted next-token probabilities", "next-token probabilities", "outputs")),
    ConceptRule("input_tokens", "Input Sequence", "inputs", "input sequence", r"\brepresent the input\s+sequence\b|\bfor a given token, its input representation\b", ("input tokens", "token sequence"), context_pattern=EMBEDDING_SUM_CONTEXT),
    ConceptRule("token_embeddings", "Token Embeddings", "modules", "embedding", rf"\btoken embeddings?\b|{EMBEDDING_SUM_CONTEXT}", (), context_pattern=EMBEDDING_SUM_CONTEXT),
    ConceptRule("segment_embeddings", "Segment Embeddings", "modules", "embedding", rf"\b(?:segment|segmentation) embeddings?\b|{EMBEDDING_SUM_CONTEXT}", ("segmentation embeddings",), context_pattern=EMBEDDING_SUM_CONTEXT),
    ConceptRule("bert_position_embeddings", "Position Embeddings", "modules", "embedding", r"\bposition embeddings?\b", (), context_pattern=EMBEDDING_SUM_CONTEXT),
    ConceptRule("input_representation", "Input Representation", "modules", "representation", r"\binput representation\b|\binput embeddings are the sum of\b", ("bert input representation",), context_pattern=EMBEDDING_SUM_CONTEXT),
    ConceptRule("bidirectional_encoder", "Bidirectional Transformer Encoder", "modules", "encoder", r"\bmulti-layer bidirectional transformer en-?\s*coder\b|\bbidirectional transformer encoder\b", ("bert", "bert encoder")),
    ConceptRule("masked_lm", "Masked LM", "modules", "training objective", r"\bmasked (?:language model(?:ing)?|lm)\b", ("masked language model", "masked language modeling", "mlm objective"), context_pattern=r"\bnext sentence prediction\b|\bNSP\b"),
    ConceptRule("next_sentence", "Next Sentence Prediction", "modules", "training objective", r"\bnext sentence prediction\b|\bNSP\b", ("nsp",), context_pattern=r"\bmasked (?:language model(?:ing)?|lm)\b"),
    ConceptRule("fine_tuning", "Fine-tuning", "modules", "adaptation", r"\bframework:\s*pre-training and fine-tuning\b|\bpre-trained model parameters are used to initialize models for different down-stream tasks\b", ("fine-tune", "fine tuning")),
    ConceptRule("downstream_tasks", "Downstream Tasks", "outputs", "task outputs", r"\bmodels for different down-stream tasks\b|\blabeled data from the downstream tasks\b|\bmodel many downstream tasks\b", ("task-specific outputs", "fine-tuning tasks")),
    ConceptRule("input_query", "Input Query", "inputs", "query", r"\bfor query x\b|\bgiven a query x\b|\bquery representation produced by a query encoder\b", ("query", "input x")),
    ConceptRule("query_encoder", "Query Encoder", "modules", "encoder", r"\bquery encoder\b"),
    ConceptRule("document_index", "Document Index", "modules", "non-parametric memory", r"\bdocument index\b|\bdense vector index\b", ("non-parametric memory", "wikipedia index")),
    ConceptRule("retriever", "Retriever", "modules", "retriever", r"\b(?:pre-trained |neural )?retriever\b"),
    ConceptRule("top_k_documents", "Top-K Documents", "modules", "retrieved context", r"\btop-?k documents\b|\btop k documents are retrieved\b", ("retrieved documents", "retrieved passages")),
    ConceptRule("generator_module", "Generator", "modules", "generator", r"\bpre-trained seq2seq model\s*\(\s*generator\s*\)|\bgenerator p|\bgenerator component\b", ("generate", "seq2seq generator", "pre-trained generator")),
    ConceptRule("output_sequence", "Output Sequence", "outputs", "generated output", r"\bgenerator produces the output sequence\b|\bgenerating the target sequence y\b|\bfinal prediction y\b", ("prediction y", "final prediction", "generated output", "output y")),
)


RELATION_RULES = (
    ("input_image", "image_patches", "data_flow"),
    ("image_patches", "linear_projection", "data_flow"),
    ("linear_projection", "transformer_encoder", "data_flow"),
    ("position_embedding", "transformer_encoder", "conditioning"),
    ("class_token", "transformer_encoder", "conditioning"),
    ("transformer_encoder", "mlp_head", "data_flow"),
    ("mlp_head", "class_prediction", "data_flow"),
    ("mlp_head", "classification", "data_flow"),
    ("input_image", "cnn_backbone", "data_flow"),
    ("input_image", "backbone", "data_flow"),
    ("cnn_backbone", "transformer_encoder", "data_flow"),
    ("positional_encoding", "transformer_encoder", "conditioning"),
    ("transformer_encoder", "transformer_decoder", "data_flow"),
    ("object_queries", "transformer_decoder", "conditioning"),
    ("transformer_decoder", "prediction_ffn", "data_flow"),
    ("prediction_ffn", "class_predictions", "branch"),
    ("prediction_ffn", "box_predictions", "branch"),
    ("class_predictions", "bipartite_matching", "training_objective"),
    ("box_predictions", "bipartite_matching", "training_objective"),
    ("backbone", "region_proposals", "data_flow"),
    ("region_proposals", "roi_align", "data_flow"),
    ("backbone", "roi_align", "feature_flow"),
    ("roi_align", "classification", "branch"),
    ("roi_align", "class_predictions", "branch"),
    ("roi_align", "box_regression", "branch"),
    ("roi_align", "box_predictions", "branch"),
    ("roi_align", "mask_branch", "branch"),
    ("task_input", "generate", "data_flow"),
    ("generate", "initial_output", "data_flow"),
    ("initial_output", "feedback", "evaluation"),
    ("feedback", "self_feedback", "feedback"),
    ("initial_output", "refine", "revision_input"),
    ("self_feedback", "refine", "feedback"),
    ("refine", "refined_output", "data_flow"),
    ("refined_output", "feedback", "feedback_loop"),
    ("image", "modality_encoders", "encoding"),
    ("text", "modality_encoders", "encoding"),
    ("audio", "modality_encoders", "encoding"),
    ("depth", "modality_encoders", "encoding"),
    ("thermal", "modality_encoders", "encoding"),
    ("imu", "modality_encoders", "encoding"),
    ("modality_encoders", "joint_embedding", "alignment"),
    ("joint_embedding", "emergent_alignment", "enables"),
    ("image", "image_encoder", "data_flow"),
    ("prompt", "prompt_encoder", "data_flow"),
    ("image_encoder", "mask_decoder", "feature_flow"),
    ("prompt_encoder", "mask_decoder", "conditioning"),
    ("mask_decoder", "valid_mask", "data_flow"),
    ("images", "image_encoder", "encoding"),
    ("text", "text_encoder", "encoding"),
    ("image_encoder", "image_embeddings", "data_flow"),
    ("text_encoder", "text_embeddings", "data_flow"),
    ("image_embeddings", "contrastive_objective", "alignment"),
    ("text_embeddings", "contrastive_objective", "alignment"),
    ("class_descriptions", "text_encoder", "encoding"),
    ("image_embeddings", "zero_shot_classifier", "classification"),
    ("text_embeddings", "zero_shot_classifier", "classification"),
    ("camera_ray", "sampled_points", "sampling"),
    ("sampled_points", "positional_encoding", "encoding"),
    ("positional_encoding", "mlp", "data_flow"),
    ("viewing_direction", "mlp", "conditioning"),
    ("mlp", "volume_density", "prediction"),
    ("mlp", "color", "prediction"),
    ("volume_density", "volume_rendering", "rendering_input"),
    ("color", "volume_rendering", "rendering_input"),
    ("volume_rendering", "rendered_image", "data_flow"),
    ("source_tokens", "input_embedding", "embedding"),
    ("input_embedding", "encoder_stack", "data_flow"),
    ("positional_encoding", "encoder_stack", "conditioning"),
    ("target_tokens", "output_embedding", "embedding"),
    ("output_embedding", "decoder_stack", "data_flow"),
    ("positional_encoding", "decoder_stack", "conditioning"),
    ("encoder_stack", "decoder_stack", "cross_attention"),
    ("decoder_stack", "output_linear", "data_flow"),
    ("output_linear", "output_softmax", "data_flow"),
    ("output_softmax", "output_probabilities", "data_flow"),
    ("input_tokens", "token_embeddings", "embedding"),
    ("token_embeddings", "input_representation", "sum"),
    ("segment_embeddings", "input_representation", "sum"),
    ("bert_position_embeddings", "input_representation", "sum"),
    ("input_representation", "bidirectional_encoder", "data_flow"),
    ("bidirectional_encoder", "masked_lm", "training_objective"),
    ("bidirectional_encoder", "next_sentence", "training_objective"),
    ("bidirectional_encoder", "fine_tuning", "initialization"),
    ("fine_tuning", "downstream_tasks", "data_flow"),
    ("input_query", "query_encoder", "encoding"),
    ("query_encoder", "retriever", "retrieval_query"),
    ("document_index", "retriever", "memory_lookup"),
    ("retriever", "top_k_documents", "retrieval"),
    ("input_query", "generator_module", "generation_input"),
    ("top_k_documents", "generator_module", "conditioning"),
    ("generator_module", "output_sequence", "generation"),
)


def _overview_candidates(parsed: dict[str, Any], limit: int = 2) -> list[dict[str, Any]]:
    positive = re.compile(r"\b(overview|framework|architecture|pipeline|procedure|approach|model|system|workflow|representation|encoder|decoder|directly predicts)\b", re.IGNORECASE)
    negative = re.compile(r"\b(comparison|differences?|versus|performance|result|ablation|visualization|qualitative|attention|distribution|interface|scaling|dimensions?|timings?|compute|memory)\b", re.IGNORECASE)
    ranked = []
    for index, figure in enumerate(parsed.get("document_index", {}).get("figures", [])):
        caption = str(figure.get("caption") or "")
        positive_match = positive.search(caption)
        if not positive_match:
            continue
        score = 8 + (4 if re.match(r"^(figure|fig\.)\s*[12]\b", caption, re.IGNORECASE) else 0)
        score += 3 if 160 <= len(caption) <= 1600 else 0
        score -= 14 if negative.search(caption) else 0
        ranked.append((score, -index, figure))
    ranked.sort(reverse=True)
    if not ranked:
        return []
    best_score = ranked[0][0]
    # Keep a second, explicitly architectural figure when it is not a result or
    # analysis figure.  Early Figure 1/2 overviews still rank first, while the
    # negative-caption penalty excludes late scaling, timing, attention, and
    # visualization figures.  The second page often contains the readable
    # subsystem labels that a short overview caption omits.
    minimum_score = max(8, best_score - 4)
    return [item for score, _, item in ranked if score >= minimum_score][:limit]


def _evidence_for_figure(parsed: dict[str, Any], figure: dict[str, Any]) -> dict[str, Any] | None:
    caption = str(figure.get("caption") or "")
    return next(
        (
            item
            for item in parsed.get("evidence", [])
            if item.get("kind") == "caption"
            and item.get("page") == figure.get("page")
            and str(item.get("text") or "").startswith(caption[:60])
        ),
        None,
    )


def _relevant_evidence(parsed: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = _overview_candidates(parsed)
    caption_evidence = [item for item in (_evidence_for_figure(parsed, figure) for figure in candidates) if item]
    selected_pages = {int(item.get("page") or 0) for item in caption_evidence if item.get("page")}
    selected = list(caption_evidence)
    for item in parsed.get("evidence", []):
        if int(item.get("page") or 0) not in selected_pages or item in selected:
            continue
        section = str(item.get("section_hint") or "").casefold()
        if any(term in section for term in ("reference", "acknowledg")):
            continue
        item_bbox = item.get("bbox") if isinstance(item.get("bbox"), list) and len(item.get("bbox")) == 4 else None
        captions = [value for value in caption_evidence if value.get("page") == item.get("page")]
        near_caption = False
        for caption in captions:
            caption_bbox = caption.get("bbox") if isinstance(caption.get("bbox"), list) and len(caption.get("bbox")) == 4 else None
            if item_bbox is None or caption_bbox is None:
                near_caption = True
                break
            if float(item_bbox[1]) <= float(caption_bbox[3]) + 72.0 and float(item_bbox[3]) >= float(caption_bbox[1]) - 540.0:
                near_caption = True
                break
        if near_caption:
            selected.append(item)
    selected_ids = {item["id"] for item in selected}
    page_count = max(1, int(parsed.get("page_count") or 1))
    early_page_limit = max(8, int(page_count * 0.45))
    useful_sections = ("abstract", "introduction", "method", "approach", "architecture", "framework", "model", "system", "figure captions")
    relevant = list(selected)
    for item in parsed.get("evidence", []):
        if item.get("id") in selected_ids:
            continue
        section = str(item.get("section_hint") or "").casefold()
        if any(term in section for term in ("reference", "acknowledg")):
            continue
        if int(item.get("page") or page_count + 1) <= early_page_limit or any(term in section for term in useful_sections):
            relevant.append(item)
    return selected, relevant


def _window_matches(pattern: re.Pattern[str], items: list[dict[str, Any]], max_window: int = 3) -> list[dict[str, Any]]:
    for start in range(len(items)):
        page = items[start].get("page")
        for size in range(2, max_window + 1):
            window = items[start:start + size]
            if len(window) != size or any(item.get("page") != page for item in window):
                break
            if pattern.search(" ".join(str(item.get("text") or "") for item in window)):
                return window
    return []


def _find_matches(rule: ConceptRule, selected: list[dict[str, Any]], relevant: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pattern = re.compile(rule.pattern, re.IGNORECASE)
    if rule.context_pattern:
        context = re.compile(rule.context_pattern, re.IGNORECASE | re.DOTALL)
        context_text = " ".join(str(item.get("text") or "") for item in [*selected, *relevant])
        if not context.search(context_text):
            return []
    primary = [item for item in selected if pattern.search(str(item.get("text") or ""))]
    if primary:
        return primary[:2]
    primary_window = _window_matches(pattern, selected)
    if primary_window:
        return primary_window
    if rule.requires_overview:
        return []
    secondary = [item for item in relevant if pattern.search(str(item.get("text") or ""))]
    secondary_window = _window_matches(pattern, relevant)
    if rule.context_pattern and secondary_window:
        return secondary_window
    if secondary:
        return secondary[:2]
    return secondary_window


def _entity_with_id(spec: dict[str, Any], item_id: str) -> dict[str, Any] | None:
    for field in ("inputs", "modules", "outputs", "innovations"):
        for item in spec.get(field, []) if isinstance(spec.get(field), list) else []:
            if isinstance(item, dict) and str(item.get("id") or "") == item_id:
                return item
    return None


def _reuse_or_add_concept(
    spec: dict[str, Any],
    rule: ConceptRule,
    evidence_ids: list[str],
    found: dict[str, dict[str, Any]],
    added_entities: list[str],
) -> None:
    """Record one concept on the contract without giving two modules the same id.

    A sealed module may already be called ``module_retriever`` while its label
    is the paper's own name. Matching only the label then appends a second
    ``Retriever`` with that same id, and the contract is refused as a duplicate.
    """
    item_id = _stable_id(rule.field.rstrip("s"), rule.label)
    existing = _entity_with_id(spec, item_id)
    if existing is not None:
        existing["evidence_ids"] = list(dict.fromkeys([*existing.get("evidence_ids", []), *evidence_ids]))
        found[rule.key] = existing
        return
    item = {"id": item_id, "name": rule.label, "role": rule.role, "evidence_ids": evidence_ids}
    spec.setdefault(rule.field, []).append(item)
    found[rule.key] = item
    added_entities.append(item_id)


def _find_existing(spec: dict[str, Any], rule: ConceptRule) -> tuple[str, dict[str, Any]] | None:
    accepted = {_normalized(rule.key), _normalized(rule.label), *(_normalized(value) for value in rule.aliases)}
    for field in ("inputs", "modules", "outputs", "innovations"):
        for item in spec.get(field, []) if isinstance(spec.get(field), list) else []:
            if not isinstance(item, dict):
                continue
            raw_value = _label(item)
            value = _normalized(raw_value)
            item_id = _normalized(item.get("id"))
            if item_id in accepted or value in accepted:
                return field, item
            if rule.label.isupper() and re.match(rf"^\s*{re.escape(rule.label)}\s*(?:\(|$)", raw_value, re.IGNORECASE):
                return field, item
            if rule.key != "self_feedback" and field == rule.field and len(value) >= 8 and any(len(candidate) >= 8 and (value in candidate or candidate in value) for candidate in accepted if candidate):
                return field, item
    return None


def _relation_bridge(source: dict[str, Any], target: dict[str, Any], relevant: list[dict[str, Any]]) -> dict[str, Any] | None:
    def terms(item: dict[str, Any]) -> set[str]:
        values = set()
        for word in re.findall(r"[a-z0-9]+", _label(item).casefold()):
            if len(word) < 4 or word in {"shifted", "right", "stack", "module", "layer"}:
                continue
            values.add(word[:-1] if word.endswith("s") and len(word) > 4 else word)
        return values

    source_terms = terms(source)
    target_terms = terms(target)
    distinct_target_terms = target_terms - source_terms or target_terms
    if not source_terms or not distinct_target_terms:
        return None
    return next(
        (
            item
            for item in relevant
            if any(term in _normalized(item.get("text")) for term in source_terms)
            and any(term in _normalized(item.get("text")) for term in distinct_target_terms)
        ),
        None,
    )


def _ground_statement(item: object, relevant: list[dict[str, Any]]) -> list[str]:
    if not isinstance(item, dict) or item.get("evidence_ids"):
        return []
    statement = str(item.get("text") or item.get("statement") or "").strip()
    if not statement or statement.casefold() == "unknown":
        return []
    stop_words = {"a", "an", "and", "are", "as", "for", "in", "is", "of", "on", "or", "the", "to", "we", "with"}
    target_terms = {word for word in re.findall(r"[a-z0-9]+", statement.casefold()) if len(word) >= 4 and word not in stop_words}
    if not target_terms:
        return []
    best_score = 0.0
    best_window: list[dict[str, Any]] = []
    for start in range(len(relevant)):
        page = relevant[start].get("page")
        for size in range(1, 4):
            window = relevant[start:start + size]
            if len(window) != size or any(value.get("page") != page for value in window):
                break
            text_terms = set(re.findall(r"[a-z0-9]+", " ".join(str(value.get("text") or "") for value in window).casefold()))
            score = len(target_terms & text_terms) / len(target_terms)
            if score > best_score:
                best_score = score
                best_window = window
    if best_score < 0.6:
        item["text"] = "unknown"
        item["status"] = "unknown"
        return []
    evidence_ids = [str(value.get("id")) for value in best_window if value.get("id")]
    item["evidence_ids"] = evidence_ids
    return evidence_ids


def _ground_declared_entities(spec: dict[str, Any], relevant: list[dict[str, Any]]) -> list[str]:
    grounded: list[str] = []
    for field in ("inputs", "modules", "outputs", "innovations"):
        for item in spec.get(field, []) if isinstance(spec.get(field), list) else []:
            if not isinstance(item, dict) or item.get("evidence_ids"):
                continue
            label = _label(item)
            normalized_label = _normalized(label)
            acronym = bool(label.isupper() and 3 <= len(normalized_label) <= 12)
            if not acronym and (len(normalized_label) < 8 or len(re.findall(r"[a-z0-9]+", label.casefold())) < 2):
                continue
            match: list[dict[str, Any]] = []
            for start in range(len(relevant)):
                page = relevant[start].get("page")
                for size in range(1, 4):
                    window = relevant[start:start + size]
                    if len(window) != size or any(value.get("page") != page for value in window):
                        break
                    combined_text = " ".join(str(value.get("text") or "") for value in window)
                    combined = _normalized(combined_text)
                    exact_acronym = acronym and re.search(rf"(?<![A-Za-z0-9]){re.escape(label)}(?![A-Za-z0-9])", combined_text, re.IGNORECASE)
                    if exact_acronym or normalized_label in combined:
                        match = window
                        break
                if match:
                    break
            evidence_ids = [str(value.get("id")) for value in match if value.get("id")]
            if evidence_ids:
                item["evidence_ids"] = evidence_ids
                grounded.append(str(item.get("id") or label))
    return grounded


def _correct_unsupported_entity_names(spec: dict[str, Any], parsed: dict[str, Any]) -> list[str]:
    evidence_by_id = {
        str(item.get("id")): item
        for item in parsed.get("evidence", [])
        if isinstance(item, dict) and item.get("id")
    }
    corrected: list[str] = []
    for item in spec.get("outputs", []) if isinstance(spec.get("outputs"), list) else []:
        if not isinstance(item, dict) or _normalized(_label(item)) != "zeroshotclassifier":
            continue
        evidence_text = " ".join(
            str(evidence_by_id[evidence_id].get("text") or "")
            for evidence_id in item.get("evidence_ids", [])
            if evidence_id in evidence_by_id
        )
        if re.search(r"\bzero[ -]?shot (?:linear )?classifier\b", evidence_text, re.IGNORECASE):
            continue
        if re.search(r"\bzero[ -]?shot prediction\b", evidence_text, re.IGNORECASE):
            item["name"] = "Zero-shot Prediction"
            corrected.append(str(item.get("id") or "zero_shot_prediction"))
    return corrected


def _deduplicate_entities(spec: dict[str, Any]) -> list[str]:
    terminology = spec.get("terminology") if isinstance(spec.get("terminology"), dict) else {}
    visible_aliases = {
        _normalized(source): str(visible).strip()
        for source, visible in terminology.items()
        if _normalized(source) and str(visible).strip()
    }
    id_remap: dict[str, str] = {}
    removed: list[str] = []
    seen: dict[str, dict[str, Any]] = {}
    for field in ("inputs", "modules", "outputs", "innovations"):
        items = spec.get(field, []) if isinstance(spec.get(field), list) else []
        field_seen = {} if field == "innovations" else seen
        unique: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            raw_key = _normalized(_label(item))
            if raw_key in visible_aliases:
                item["name"] = visible_aliases[raw_key]
            key = _normalized(_label(item))
            if field == "inputs" and (key == "input" or re.fullmatch(r"input[a-z0-9]{1,4}", key)):
                key = "input"
            if not key or key not in field_seen:
                if key:
                    field_seen[key] = item
                unique.append(item)
                continue
            kept = field_seen[key]
            kept["evidence_ids"] = list(dict.fromkeys([*kept.get("evidence_ids", []), *item.get("evidence_ids", [])]))
            duplicate_id = str(item.get("id") or "")
            kept_id = str(kept.get("id") or "")
            if duplicate_id and kept_id and duplicate_id != kept_id:
                id_remap[duplicate_id] = kept_id
            removed.append(duplicate_id or _label(item))
        spec[field] = unique
    if id_remap:
        for relation in spec.get("relations", []) if isinstance(spec.get("relations"), list) else []:
            if not isinstance(relation, dict):
                continue
            relation["source"] = id_remap.get(str(relation.get("source")), relation.get("source"))
            relation["target"] = id_remap.get(str(relation.get("target")), relation.get("target"))
    return removed


def _heuristic_rule_scope(
    selected: list[dict[str, Any]],
    relevant: list[dict[str, Any]],
    declared_rule_keys: set[str],
) -> tuple[set[str], dict[str, list[dict[str, Any]]]]:
    selected_ids = {str(item.get("id")) for item in selected if item.get("id")}
    scoped_relevant = relevant
    matches = {
        rule.key: found
        for rule in CONCEPT_RULES
        if (found := _find_matches(rule, selected, scoped_relevant))
    }
    candidates = set(matches)
    if not candidates:
        return set(), matches
    seeds = declared_rule_keys | {
        rule.key
        for rule in CONCEPT_RULES
        if rule.key in candidates and _find_matches(rule, selected, [])
    }
    adjacency = {key: set() for key in candidates}

    def locally_connected(left: str, right: str) -> bool:
        left_items, right_items = matches.get(left, []), matches.get(right, [])
        if not left_items or not right_items:
            return True
        for left_item in left_items:
            for right_item in right_items:
                left_page, right_page = int(left_item.get("page") or 0), int(right_item.get("page") or 0)
                if not left_page or not right_page:
                    return True
                page_gap = abs(left_page - right_page)
                if 0 < page_gap <= 3:
                    return True
                if page_gap != 0:
                    continue
                left_bbox, right_bbox = left_item.get("bbox"), right_item.get("bbox")
                if not (isinstance(left_bbox, list) and len(left_bbox) == 4 and isinstance(right_bbox, list) and len(right_bbox) == 4):
                    return True
                left_center = (float(left_bbox[1]) + float(left_bbox[3])) / 2.0
                right_center = (float(right_bbox[1]) + float(right_bbox[3])) / 2.0
                if abs(left_center - right_center) <= 360.0:
                    return True
        left_selected = any(str(item.get("id")) in selected_ids for item in left_items)
        right_selected = any(str(item.get("id")) in selected_ids for item in right_items)

        def method_grounded(items: list[dict[str, Any]]) -> bool:
            blocked = ("introduction", "background", "related work", "reference", "acknowledg", "experiment", "result")
            return any(not any(term in str(item.get("section_hint") or "").casefold() for term in blocked) for item in items)

        if left_selected and method_grounded(right_items):
            return True
        if right_selected and method_grounded(left_items):
            return True
        return False

    for source_key, target_key, _ in RELATION_RULES:
        if source_key in candidates and target_key in candidates and locally_connected(source_key, target_key):
            adjacency[source_key].add(target_key)
            adjacency[target_key].add(source_key)

    def component(start: str) -> set[str]:
        reached: set[str] = set()
        pending = [start]
        while pending:
            current = pending.pop()
            if current in reached:
                continue
            reached.add(current)
            pending.extend(adjacency.get(current, set()) - reached)
        return reached

    if seeds & candidates:
        allowed: set[str] = set()
        for seed in seeds & candidates:
            allowed.update(component(seed))
        return allowed, matches

    components: list[set[str]] = []
    remaining = set(candidates)
    while remaining:
        group = component(next(iter(remaining)))
        components.append(group)
        remaining -= group
    components.sort(key=lambda group: (len(group), sum(len(matches[key]) for key in group)), reverse=True)
    return components[0], matches


def augment_contract_from_evidence(spec: dict[str, Any], parsed: dict[str, Any]) -> dict[str, Any]:
    selected, relevant = _relevant_evidence(parsed)
    corrected_entities = _correct_unsupported_entity_names(spec, parsed)
    deduplicated_entities = _deduplicate_entities(spec)
    declared_entities = [
        item
        for field in ("inputs", "modules", "outputs", "innovations")
        for item in (spec.get(field, []) if isinstance(spec.get(field), list) else [])
        if isinstance(item, dict)
    ]
    fallback_declared = any(str(item.get("role") or "") == "paper-derived stage requiring VLM verification" for item in declared_entities)
    conservative_expansion = len(declared_entities) >= 3 and len([item for item in spec.get("relations", []) if isinstance(item, dict)]) >= 1 and not fallback_declared
    selected_pages = {int(item.get("page") or 0) for item in selected if item.get("page")}
    relevant_non_background = [
        item
        for item in relevant
        if not any(term in str(item.get("section_hint") or "").casefold() for term in ("related work", "reference", "acknowledg"))
        and not (
            any(term in str(item.get("section_hint") or "").casefold() for term in ("introduction", "background"))
            and re.search(r"\b(?:for example|e\.g\.|such as)\b[^\n]{0,160}\[\s*\d+", str(item.get("text") or ""), re.IGNORECASE)
        )
    ]
    initially_declared_rule_keys = {
        rule.key
        for rule in CONCEPT_RULES
        if _find_existing(spec, rule)
    }
    heuristic_rule_keys: set[str] | None = None
    heuristic_matches: dict[str, list[dict[str, Any]]] = {}
    if not conservative_expansion:
        heuristic_rule_keys, heuristic_matches = _heuristic_rule_scope(selected, relevant_non_background, initially_declared_rule_keys)
    grounded_statements = {
        field: _ground_statement(spec.get(field), relevant)
        for field in ("research_problem", "central_claim")
    }
    grounded_entities = _ground_declared_entities(spec, relevant)
    found: dict[str, dict[str, Any]] = {}
    added_entities: list[str] = []
    upgraded_entities: list[str] = []
    adopted_entities: list[str] = []
    for rule in CONCEPT_RULES:
        existing = _find_existing(spec, rule)
        if existing:
            matches = _find_matches(rule, selected, relevant)
        elif not conservative_expansion:
            matches = heuristic_matches.get(rule.key, []) if rule.key in (heuristic_rule_keys or set()) else []
        else:
            matches = _find_matches(rule, selected, [])
            if not matches:
                connected_to_declared = any(
                    (source_key == rule.key and target_key in initially_declared_rule_keys)
                    or (target_key == rule.key and source_key in initially_declared_rule_keys)
                    for source_key, target_key, _ in RELATION_RULES
                )
                if connected_to_declared:
                    matches = _find_matches(rule, [], relevant_non_background)
        if not matches:
            continue
        evidence_ids = list(dict.fromkeys(str(item.get("id")) for item in matches if item.get("id")))
        if existing:
            _, item = existing
            current = _label(item)
            if str(item.get("role") or "") == "paper-derived stage requiring VLM verification":
                item["role"] = rule.role
                adopted_entities.append(str(item.get("id") or rule.key))
            accepted_names = {_normalized(rule.label), *(_normalized(value) for value in rule.aliases)}
            exact_rule_id = _normalized(item.get("id")) == _normalized(rule.key)
            if _normalized(current) != _normalized(rule.label) and (exact_rule_id or _normalized(current) in accepted_names or _normalized(current) in _normalized(rule.label) or current.casefold().startswith(("neural network module", "learned "))):
                item["name"] = rule.label
                upgraded_entities.append(str(item.get("id") or rule.key))
            item["evidence_ids"] = list(dict.fromkeys(list(item.get("evidence_ids", [])) + evidence_ids))
            found[rule.key] = item
            continue
        _reuse_or_add_concept(spec, rule, evidence_ids, found, added_entities)

    if conservative_expansion:
        for _pass in range(3):
            progress = False
            connected_keys = set(found)
            for rule in CONCEPT_RULES:
                if rule.key in found or _find_existing(spec, rule):
                    continue
                connected_to_found = any(
                    (source_key == rule.key and target_key in connected_keys)
                    or (target_key == rule.key and source_key in connected_keys)
                    for source_key, target_key, _ in RELATION_RULES
                )
                if not connected_to_found:
                    continue
                matches = _find_matches(rule, selected, []) or _find_matches(rule, [], relevant_non_background)
                if not matches:
                    continue
                evidence_ids = list(dict.fromkeys(str(item.get("id")) for item in matches if item.get("id")))
                _reuse_or_add_concept(spec, rule, evidence_ids, found, added_entities)
                progress = True
            if not progress:
                break

    deduplicated_entities.extend(_deduplicate_entities(spec))
    found = {}
    for rule in CONCEPT_RULES:
        existing = _find_existing(spec, rule)
        if existing:
            found[rule.key] = existing[1]

    relations = spec.setdefault("relations", [])
    existing_pairs = {(str(item.get("source")), str(item.get("target"))) for item in relations if isinstance(item, dict)}
    selected_ids = [str(item.get("id")) for item in selected if item.get("id")]
    evidence_by_id = {str(item.get("id")): item for item in relevant if item.get("id")}
    added_relations: list[str] = []
    repaired_relations: list[str] = []
    for source_key, target_key, relation_type in RELATION_RULES:
        source = found.get(source_key)
        target = found.get(target_key)
        if not source or not target:
            continue
        pair = (str(source.get("id")), str(target.get("id")))
        if pair in existing_pairs:
            continue
        evidence_ids = list(dict.fromkeys(list(source.get("evidence_ids", [])) + list(target.get("evidence_ids", []))))
        if selected_ids and not any(value in selected_ids for value in evidence_ids):
            bridge = _relation_bridge(source, target, relevant)
            source_pages = [int(evidence_by_id[value].get("page") or 0) for value in source.get("evidence_ids", []) if value in evidence_by_id]
            target_pages = [int(evidence_by_id[value].get("page") or 0) for value in target.get("evidence_ids", []) if value in evidence_by_id]
            nearby = bool(source_pages and target_pages and min(abs(left - right) for left in source_pages for right in target_pages) <= 3)
            if not bridge and not nearby:
                continue
            if bridge:
                evidence_ids.append(str(bridge.get("id")))
        relations.append({"source": pair[0], "target": pair[1], "type": relation_type, "label": "", "evidence_ids": list(dict.fromkeys(evidence_ids))})
        existing_pairs.add(pair)
        added_relations.append(f"{pair[0]}->{pair[1]}")

    endpoints = {
        str(item.get("id")): item
        for field in ("inputs", "modules", "outputs", "innovations")
        for item in spec.get(field, []) if isinstance(item, dict) and item.get("id")
    }
    for relation in relations:
        if not isinstance(relation, dict) or relation.get("evidence_ids"):
            continue
        source = endpoints.get(str(relation.get("source")))
        target = endpoints.get(str(relation.get("target")))
        if not source or not target:
            continue
        candidate_ids = list(dict.fromkeys(list(source.get("evidence_ids", [])) + list(target.get("evidence_ids", []))))
        source_pages = [int(evidence_by_id[value].get("page") or 0) for value in source.get("evidence_ids", []) if value in evidence_by_id]
        target_pages = [int(evidence_by_id[value].get("page") or 0) for value in target.get("evidence_ids", []) if value in evidence_by_id]
        nearby = bool(source_pages and target_pages and min(abs(left - right) for left in source_pages for right in target_pages) <= 3)
        bridge = _relation_bridge(source, target, relevant)
        if not candidate_ids or not (any(value in selected_ids for value in candidate_ids) or nearby or bridge):
            continue
        if bridge and bridge.get("id"):
            candidate_ids.append(str(bridge["id"]))
        relation["evidence_ids"] = list(dict.fromkeys(candidate_ids))
        repaired_relations.append(f"{relation.get('source')}->{relation.get('target')}")

    overview_terms = [rule.key for rule in CONCEPT_RULES if _find_matches(rule, selected, [])]
    covered_terms = [key for key in overview_terms if key in found]
    return {
        "summary": "Conservative evidence-driven contract completion report.",
        "selected_overview_evidence_ids": selected_ids,
        "overview_term_count": len(overview_terms),
        "covered_overview_term_count": len(covered_terms),
        "overview_term_coverage": round(len(covered_terms) / max(1, len(overview_terms)), 4),
        "added_entities": added_entities,
        "upgraded_entities": upgraded_entities,
        "adopted_entities": adopted_entities,
        "added_relations": added_relations,
        "repaired_relations": repaired_relations,
        "grounded_statements": {field: ids for field, ids in grounded_statements.items() if ids},
        "grounded_entities": grounded_entities,
        "corrected_entities": corrected_entities,
        "deduplicated_entities": deduplicated_entities,
        "conservative_expansion": conservative_expansion,
        "expansion_page_scope": sorted(selected_pages),
        "new_entities_require_declared_neighbor": conservative_expansion,
        "rules_are_evidence_gated": True,
    }
