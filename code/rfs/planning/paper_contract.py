from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

from ..providers import call_vlm_json, resolve_vlm_model, vlm_credentials_available
from ..analysis.paper.document import evidence_excerpt, overview_figure_panel_excerpt, overview_figure_spatial_excerpt
from ..contracts.vlm_context import compact_evidence, compact_review, context_stats


DEFAULT_PREFERENCES = {
    "summary": "Default direct paper-figure preferences.",
    "figure_type": "method framework figure",
    "aspect_ratio": "16:9",
    "language": "English",
    "style_description": "clean academic illustration, restrained 2.5D, dense but readable",
    "preferred_flow": "left_to_right",
    "visual_density": "high",
    "preferred_palette": ["blue", "teal", "warm orange accent"],
    "avoid": ["cyberpunk", "commercial poster", "generic dashboard cards", "unrelated robots", "fake charts"],
    "must_show": [],
    "must_not_show": [],
}


def merge_preferences(raw: dict | None, aspect_ratio: str | None = None, language: str | None = None) -> dict:
    merged = dict(DEFAULT_PREFERENCES)
    if isinstance(raw, dict):
        merged.update(raw)
    if aspect_ratio:
        merged["aspect_ratio"] = aspect_ratio
    if language:
        merged["language"] = language
    merged["summary"] = "User and default preferences compiled for image generation."
    return merged


def _planner_prompt(parsed: dict, preferences: dict, paper_review: dict | None = None, evidence_max_chars: int = 58000) -> str:
    return f"""
# Summary

You are a paper-to-scientific-figure planner. Return JSON only.

Build a scientifically faithful plan for generating ONE complete raster framework figure from the supplied paper evidence. Do not create PPTX instructions. Do not invent stock modules such as encoder, retriever, memory, agent, reinforcement learning, knowledge base, or decoder unless supported by evidence.

Every important factual item must include one or more evidence_ids copied exactly from the supplied evidence labels. Mark uncertain fields as unknown. Use the paper's exact terminology.

Contract completeness rules:
- Treat Figure 1 and captions containing "overview", "framework", "architecture", or "pipeline" as high-priority evidence for the intended paper overview figure.
- Identify every major contribution pillar from the abstract and introduction. For system, dataset, or foundation-model papers, do not omit a data engine, dataset construction process, training loop, or deployment stage when it is a central contribution.
- Represent each separately named scientific component as a separate entity. Do not merge conditioning signals, special tokens, encoders, heads, inputs, or outputs into composite nodes when the paper names them independently.
- Give every input and output a stable id and include the boundary relations from input to the first processing entity and from the final processing entity to the output.
- Every relation endpoint must be an id declared in inputs, modules, outputs, or innovations.
- Include feedback and control edges explicitly instead of describing them only in prose.
- Before returning, check that every item in must_show is represented by a distinct entity or an explicit relation label.

Return this schema:
{{
  "summary": "Direct paper-figure planning result.",
  "paper_summary": {{
    "summary": "Structured paper summary.",
    "title": "...",
    "paper_type": "method|system|dataset|benchmark|analysis|survey|application|unknown",
    "research_problem": {{"text": "...", "evidence_ids": ["E0001"]}},
    "central_claim": {{"text": "...", "evidence_ids": ["E0001"]}},
    "inputs": [{{"name": "...", "evidence_ids": ["E0001"]}}],
    "outputs": [{{"name": "...", "evidence_ids": ["E0001"]}}],
    "core_modules": [{{"id": "stable_id", "name": "exact paper term", "role": "...", "evidence_ids": ["E0001"]}}],
    "innovations": [{{"text": "...", "evidence_ids": ["E0001"]}}],
    "training_flow": [{{"step": "...", "evidence_ids": ["E0001"]}}],
    "inference_flow": [{{"step": "...", "evidence_ids": ["E0001"]}}],
    "terminology": {{"Exact Term": "required visible label"}},
    "unknowns": []
  }},
  "figure_specification": {{
    "summary": "Scientific contract for the figure.",
    "figure_goal": "...",
    "research_problem": {{"text": "...", "evidence_ids": ["E0001"]}},
    "central_claim": {{"text": "...", "evidence_ids": ["E0001"]}},
    "storyline": [],
    "must_show": [{{"text": "...", "evidence_ids": ["E0001"]}}],
    "modules": [{{"id": "...", "name": "...", "role": "...", "evidence_ids": ["E0001"]}}],
    "relations": [{{"source": "module_id", "target": "module_id", "type": "data_flow|control_flow|training_only|inference_only|comparison|feedback", "label": "", "evidence_ids": ["E0001"]}}],
    "inputs": [{{"id": "stable_input_id", "name": "exact paper term", "evidence_ids": ["E0001"]}}],
    "outputs": [{{"id": "stable_output_id", "name": "exact paper term", "evidence_ids": ["E0001"]}}],
    "innovations": [],
    "feedback_loops": [],
    "panel_graphs": [{{
      "id": "panel_a",
      "panel_label": "a",
      "label": "exact panel meaning",
      "evidence_ids": [],
      "nodes": [{{"instance_id": "panel_a_unique_instance", "name": "exact visible term", "role": "input|module|objective|output|group", "evidence_ids": []}}],
      "relations": [{{"source": "panel_a_unique_instance", "target": "panel_a_other_instance", "type": "data_flow|conditioning|prediction|training_objective|feedback", "label": "", "evidence_ids": []}}]
    }}],
    "training_flow": [],
    "inference_flow": [],
    "topology": "linear|branch|feedback|multimodal|dense_multiframe|unknown",
    "overview_panels": [{{"id": "panel_a", "panel_label": "a", "label": "exact evidence-backed panel title", "description": "...", "evidence_ids": ["E0001"], "entity_ids": []}}],
    "required_labels": [],
    "optional_visible_labels": [],
    "visual_priorities": [],
    "terminology": {{}},
    "forbidden_inventions": [],
    "uncertainties": []
  }},
  "design_plan": {{
    "summary": "Information narrative for one complete image.",
    "reading_order": [],
    "groups": [],
    "innovation_emphasis": [],
    "preserve": [],
    "remove": []
  }},
  "layout_intent": {{
    "summary": "Layout intent for the image model.",
    "pattern": "left_to_right|two_stage|hub_and_spoke|loop|stacked|before_after",
    "canvas_ratio": "16:9",
    "regions": [{{"id": "...", "position": "left|center|right|top|bottom", "relative_size": "small|medium|large", "contains": []}}],
    "flow_description": "...",
    "whitespace": "..."
  }},
  "visual_metaphors": {{
    "summary": "Concrete visual objects for paper concepts.",
    "items": [{{"module_id": "...", "metaphor": "concrete visible objects", "must_show": [], "avoid_showing": []}}]
  }},
  "style_plan": {{
    "summary": "Image style contract.",
    "medium": "...",
    "palette": [],
    "viewpoint": "...",
    "line_and_shadow": "...",
    "visual_density": "...",
    "background": "...",
    "text_policy": "short exact labels only; no paragraphs, fake equations, fake axes, or invented numbers",
    "positive_reference_rules": [],
    "negative_reference_rules": []
  }}
}}

User preferences:
{json.dumps(preferences, ensure_ascii=False, indent=2)}

Universal structured paper review:
{json.dumps(paper_review or {}, ensure_ascii=False, indent=2)}

Paper evidence:
{evidence_excerpt(parsed, max_chars=evidence_max_chars)}
""".strip()


def _overview_figure_candidates(parsed: dict, limit: int = 4) -> list[dict[str, Any]]:
    positive = re.compile(r"\b(overview|framework|architecture|pipeline|procedure|approach|method|model|system|workflow)\b", re.IGNORECASE)
    negative = re.compile(r"\b(comparison|performance|result|ablation|visualization|qualitative|attention map|distribution)\b", re.IGNORECASE)
    ranked = []
    for index, item in enumerate(parsed.get("document_index", {}).get("figures", [])):
        caption = str(item.get("caption") or "").strip()
        if not caption:
            continue
        score = 0
        if positive.search(caption):
            score += 8
        if re.match(r"^(figure|fig\.)\s*[12]\b", caption, re.IGNORECASE):
            score += 5
        if 180 <= len(caption) <= 1400:
            score += 4
        score += min(5, len(re.findall(r"\b(?:uses?|takes?|feeds?|passes?|predicts?|produces?|outputs?|trains?|samples?|renders?|synthesizes?|optimizes?)\b", caption, re.IGNORECASE)))
        if negative.search(caption):
            score -= 7
        ranked.append((score, -index, item))
    return [item for _, _, item in sorted(ranked, reverse=True)[: max(1, int(limit))]]


def _fast_planner_prompt(parsed: dict, preferences: dict, evidence_max_chars: int = 36000) -> str:
    candidates = _overview_figure_candidates(parsed)
    return f"""
# Summary

You are compiling a scientific paper into a directed semantic graph for ONE framework figure. Return JSON only. Focus exclusively on scientific meaning; layout, illustration style, and image-generation wording will be compiled later by code.

Primary task:
1. Select the most information-rich method/architecture/framework figure caption from the candidates below. Figure 1 is not automatically best; prefer a caption that explicitly names components and directed operations. However, when an overview caption explicitly declares panels such as "a, high-level pre-training and usage" and "b, architecture", treat all declared panels as one indivisible target figure. Use detailed architecture or method evidence to enrich those panels, never to replace or omit one of them.
2. Extract every separately named input, component, intermediate representation, conditioning signal, training-only objective, inference-only step, and output needed to reproduce that overview.
3. Use exact short paper terms as visible names. Do not replace "image encoder" with a prose description such as "network that extracts image features".
   A visible name must be copied verbatim from one of its cited evidence blocks. Preserve the paper's original language and writing system; never translate a Chinese, Japanese, Korean, or other non-English term because the user preference says English. If no concise source phrase exists, use the shortest verbatim source-language phrase. Relation labels must follow the same rule or be left blank.
4. Split compound outputs and branches into separate entities. If a component predicts class and box, create separate class and box outputs. If two encoders produce two embeddings, create both embeddings.
5. Preserve direction. Every relation endpoint must be a declared entity id. Include input boundary edges and final output edges.
6. Separate training and inference. Matching, losses, supervision, optimization, and feedback belong in training_flow or training-only entities; test-time classification, decoding, retrieval, rendering, or deployment belong in inference_flow.
7. Every factual item must cite evidence_ids copied exactly from the evidence. Unsupported details must be omitted and recorded under uncertainties.
8. Ignore baseline/comparison methods mentioned only to contrast with the proposed method. Never import terms from references, acknowledgements, bibliography, or result-only figures.
9. For a multi-panel overview, set topology to dense_multiframe and include evidence-backed high-level training/usage stages as well as internal architecture modules. Do not collapse a training-versus-usage overview into only the internal neural-network layers.
10. Treat the bbox-aware figure-local blocks below as direct visual evidence. Short labels inside the target figure outrank generic architecture-family prose. A phrase such as "adaptation of the standard transformer encoder" is a description, not a separate visible module, unless "Transformer Encoder" itself appears as a short figure label or is required as an explicit operational stage.
11. Reconstruct each declared panel as a separate local graph. Do not connect a high-level workflow panel directly into an internal architecture panel merely because both describe the same model. Add cross-panel relations only when the evidence explicitly states that edge.
12. Preserve exact operational order. When a panel contains exact component labels and the paper describes them as followed-by or transformed-by stages, include every intermediate relation instead of jumping from a container title directly to the output.
13. Audit every panel against the concise panel-local inventory. A high-level workflow panel is incomplete if it contains only tensor variables while the inventory also shows a named model/process box, prediction, objective/loss, dataset, or deployment stage. Architecture-family prose must not substitute for those exact panel-local objects.
14. For every declared multi-panel overview, populate panel_graphs. Each panel-local visual occurrence gets a unique instance_id, so repeated labels such as the same model used in pre-training and inference remain separate instances. panel_graph relations must stay within their panel; use the global graph only as a compatibility summary.
15. In the concise panel-local inventory, every entry marked required_candidate=true must map to a panel_graph node using its evidence_id. Use its role_hint when one is given and infer the role from the surrounding evidence otherwise. Do not merge two repeated occurrences with different evidence IDs into one instance. Group/container labels may be isolated groups, but operational inputs, modules, objectives, and outputs must have their directed panel-local relations.
16. Drill one level below every named container. When the paper names a container and separately enumerates its parts - for example a memory with three layers (ontology, fact, passage), a group of three named agents (extraction, detection, resolution), or a graph with several described views - emit one entity per enumerated part, with its own exact name, and connect them inside the container. An enumeration of three parts must produce three entities, never one summary box.
17. Every entity, panel node, and visible label must carry a non-empty exact short term copied from the paper. Never emit an entity whose name is empty or whose name is only an identifier; if no concise source phrase exists, omit that entity and record the gap under uncertainties.

Return exactly this compact schema:
{{
  "summary": "Fast evidence-grounded semantic plan.",
  "paper_summary": {{
    "summary": "Structured paper summary.",
    "title": "...",
    "paper_type": "method|system|dataset|benchmark|analysis|survey|application|unknown",
    "research_problem": {{"text": "...", "evidence_ids": []}},
    "central_claim": {{"text": "...", "evidence_ids": []}},
    "inputs": [],
    "outputs": [],
    "core_modules": [],
    "innovations": [],
    "training_flow": [],
    "inference_flow": [],
    "terminology": {{}},
    "unknowns": []
  }},
  "figure_specification": {{
    "summary": "Scientific contract for the figure.",
    "figure_goal": "...",
    "research_problem": {{"text": "...", "evidence_ids": []}},
    "central_claim": {{"text": "...", "evidence_ids": []}},
    "storyline": [],
    "must_show": [],
    "inputs": [{{"id": "...", "name": "exact short term", "role": "input", "evidence_ids": []}}],
    "modules": [{{"id": "...", "name": "exact short term", "role": "module|intermediate|training_objective|conditioning", "evidence_ids": []}}],
    "outputs": [{{"id": "...", "name": "exact short term", "role": "output", "evidence_ids": []}}],
    "relations": [{{"source": "...", "target": "...", "type": "data_flow|encoding|conditioning|branch|alignment|prediction|training_objective|rendering_input|feedback", "label": "", "evidence_ids": []}}],
    "innovations": [],
    "feedback_loops": [],
    "panel_graphs": [{{
      "id": "panel_a",
      "panel_label": "a",
      "label": "exact panel meaning",
      "evidence_ids": [],
      "nodes": [{{"instance_id": "panel_a_unique_instance", "name": "exact visible term", "role": "input|module|objective|output|group", "evidence_ids": []}}],
      "relations": [{{"source": "panel_a_unique_instance", "target": "panel_a_other_instance", "type": "data_flow|conditioning|prediction|training_objective|feedback", "label": "", "evidence_ids": []}}]
    }}],
    "training_flow": [],
    "inference_flow": [],
    "topology": "linear|branch|feedback|multimodal|dense_multiframe|unknown",
    "required_labels": [],
    "terminology": {{}},
    "forbidden_inventions": [],
    "uncertainties": []
  }}
}}

Candidate overview captions:
{json.dumps(candidates, ensure_ascii=False, indent=2)}

Extraction scope (do not claim coverage beyond this scope):
{json.dumps({key: parsed.get('extraction_report', {}).get(key) for key in ('pdf_type', 'semantic_scope', 'readable_page_ratio', 'ocr_pages', 'warnings')}, ensure_ascii=False, indent=2)}

User preferences relevant to explanatory wording only (never translate scientific labels away from the paper's source language):
{json.dumps({key: preferences.get(key) for key in ('language', 'must_show', 'must_not_show')}, ensure_ascii=False, indent=2)}

Prioritized paper evidence:
{compact_evidence(parsed, max_chars=min(max(4000, int(evidence_max_chars)), 14000))}

BBox-aware target overview figure blocks (normalized [x0,y0,x1,y1], page-local; use for panel membership and visible-label selection, not as permission to invent arrows):
{overview_figure_spatial_excerpt(parsed, max_items=48)[:10000]}

Concise panel-local label inventory, recovered verbatim from the target figure region. Treat it as a completeness checklist: every entry marked required_candidate=true must appear as a panel node under its declared panel, and every container title listed here whose parts the paper enumerates separately must be drilled one level down:
{overview_figure_panel_excerpt(parsed, max_items_per_panel=24)[:12000]}
""".strip()


def _enforce_named_entities(spec: dict[str, Any]) -> list[str]:
    """Drop visible nodes whose name is empty, so no unlabelled box is shipped.

    The fast planner is instructed to name every entity, but a small model still
    returns the occasional id-only stub.  Such an entity compiles into a blank box
    with no readable label, so it is removed together with every relation and
    panel-local edge that points at it, and the dropped ids are recorded for
    review instead of being silently repaired with an invented name.
    """
    dropped: list[str] = []

    def keep_named(items: Any) -> list[Any]:
        kept: list[Any] = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("text") or item.get("statement") or "").strip()
            if not name:
                dropped.append(str(item.get("id") or item.get("instance_id") or ""))
                continue
            kept.append(item)
        return kept

    for field in ("inputs", "modules", "outputs", "innovations"):
        if isinstance(spec.get(field), list):
            spec[field] = keep_named(spec[field])
    dropped_ids = {value for value in dropped if value}
    if dropped_ids:
        spec["relations"] = [
            relation for relation in (spec.get("relations") or [])
            if isinstance(relation, dict)
            and str(relation.get("source")) not in dropped_ids
            and str(relation.get("target")) not in dropped_ids
        ]
        for panel in spec.get("panel_graphs") or []:
            if not isinstance(panel, dict):
                continue
            panel["nodes"] = keep_named(panel.get("nodes"))
            instance_ids = {str(node.get("instance_id")) for node in panel.get("nodes") or []}
            panel["relations"] = [
                relation for relation in (panel.get("relations") or [])
                if isinstance(relation, dict)
                and str(relation.get("source")) in instance_ids
                and str(relation.get("target")) in instance_ids
            ]
    return dropped


def _compile_fast_plan(raw: dict, preferences: dict) -> dict:
    result = normalize_plan(raw, preferences)
    spec = result["figure_specification"]
    dropped = _enforce_named_entities(spec)
    if dropped:
        uncertainties = spec.get("uncertainties")
        if not isinstance(uncertainties, list):
            uncertainties = []
            spec["uncertainties"] = uncertainties
        uncertainties.append(
            "Dropped %d unnamed entities so no unlabelled box ships: %s"
            % (len(dropped), ", ".join(sorted(set(dropped))))
        )
    entities = [
        item
        for field in ("inputs", "modules", "outputs", "innovations")
        for item in (spec.get(field, []) if isinstance(spec.get(field), list) else [])
        if isinstance(item, dict)
    ]
    entity_ids = [str(item.get("id") or "") for item in entities if str(item.get("id") or "")]
    labels = [str(item.get("name") or item.get("text") or item.get("statement") or "").strip() for item in entities]
    labels = [value for value in labels if value]
    topology = str(spec.get("topology") or "unknown")
    pattern = {"feedback": "loop", "multimodal": "hub_and_spoke", "dense_multiframe": "stacked", "branch": "two_stage"}.get(topology, "left_to_right")
    result["design_plan"] = {
        "summary": "Deterministic information narrative compiled from the fast semantic graph.",
        "reading_order": entity_ids,
        "groups": [],
        "innovation_emphasis": [str(item.get("name") or item.get("text") or item.get("statement") or "") for item in spec.get("innovations", []) if isinstance(item, dict)],
        "preserve": labels,
        "remove": list(spec.get("forbidden_inventions", []) if isinstance(spec.get("forbidden_inventions"), list) else []),
    }
    result["layout_intent"] = {
        "summary": "Deterministic layout intent compiled from semantic topology.",
        "pattern": pattern,
        "canvas_ratio": preferences.get("aspect_ratio", "16:9"),
        "regions": [],
        "flow_description": "Render all declared directed relations without reversing arrows.",
        "whitespace": "moderate",
    }
    result["visual_metaphors"] = {
        "summary": "Minimal paper-grounded visual objects; exact labels and arrows are added as an editable overlay.",
        "items": [{"module_id": str(item.get("id") or ""), "metaphor": str(item.get("name") or "scientific module"), "must_show": [], "avoid_showing": []} for item in entities],
    }
    result["style_plan"] = {
        "summary": "Fast-path style contract.",
        "medium": preferences.get("style_description"),
        "palette": preferences.get("preferred_palette", []),
        "viewpoint": "front-facing academic diagram",
        "line_and_shadow": "crisp thin lines with restrained shadows",
        "visual_density": preferences.get("visual_density", "high"),
        "background": "clean light background",
        "text_policy": "background regions only; exact labels and arrows are supplied by overlay_spec.json",
        "positive_reference_rules": [],
        "negative_reference_rules": preferences.get("avoid", []),
    }
    return result


def _first_sentence(text: str) -> str:
    sentences = [item.strip() for item in re.split(r"(?<=[.!?。！？])\s+", text) if len(item.strip()) > 20]
    return sentences[0][:500] if sentences else text[:500].strip()


def _plausible_component_heading(value: str) -> bool:
    text = re.sub(r"^\d+(?:\.\d+)*\s*", "", str(value or "")).strip()
    low = text.casefold()
    if not 3 <= len(text) <= 72 or len(text.split()) > 9:
        return False
    if any(term in low for term in ("http", "www.", "acknowledg", "reference", "appendix", "picture credit", "et al.", "university", "institute")):
        return False
    if re.search(r"[π∑=·]|\([^)]*\d{4}[^)]*\)|\b(?:fig(?:ure)?|table)\s*\d", text, re.IGNORECASE):
        return False
    if text.count(",") >= 2 or text.count(":") >= 1:
        return False
    component_terms = ("architecture", "pipeline", "framework", "module", "encoder", "decoder", "backbone", "parser", "reasoner", "renderer", "retrieval", "generation", "refinement", "training", "inference", "data engine", "model")
    return any(term in low for term in component_terms)


def _heuristic_plan(parsed: dict, preferences: dict, paper_review: dict | None = None) -> dict:
    if paper_review and paper_review.get("modules"):
        review_modules = [item for item in (list(paper_review.get("modules", [])) + list(paper_review.get("research_objects", [])) + list(paper_review.get("concepts", []))) if item.get("evidence_ids")][:16]
        modules = [{"id": str(item.get("id")), "name": str(item.get("visible_label") or item.get("statement") or item.get("id")), "role": str(item.get("visual_role") or "module"), "evidence_ids": list(item.get("evidence_ids", []))} for item in review_modules]
        relations = [{"source": str(item.get("source_id") or item.get("source") or ""), "target": str(item.get("target_id") or item.get("target") or ""), "type": str(item.get("relation_type") or item.get("type") or "data_flow"), "label": str(item.get("statement") or "")[:48], "evidence_ids": list(item.get("evidence_ids", []))} for item in paper_review.get("relations", []) if (item.get("source_id") or item.get("source")) and (item.get("target_id") or item.get("target")) and item.get("evidence_ids")]
        terminology = {str(item.get("statement")): str(item.get("visible_label") or item.get("statement")) for item in paper_review.get("terminology", []) if str(item.get("statement") or "").strip()}
        title = str(paper_review.get("paper_identity", {}).get("title") or parsed.get("source_name"))
        research_questions = paper_review.get("research_questions", [])
        claims = paper_review.get("central_claims", [])
        figure_goal = str((claims or research_questions or [{"statement": "Explain the paper method faithfully."}])[0].get("statement"))
        innovations = [item for item in paper_review.get("innovations", []) if item.get("evidence_ids")]
        forbidden = [str(item.get("statement")) for item in paper_review.get("forbidden_inventions", [])]
        inputs = [item for item in paper_review.get("inputs", []) if item.get("evidence_ids")]
        outputs = [item for item in paper_review.get("outputs", []) if item.get("evidence_ids")]
        known_modalities = ["image", "text", "audio", "depth", "thermal", "imu", "video"]
        expanded_inputs = []
        for item in inputs:
            statement = str(item.get("visible_label") or item.get("name") or item.get("statement") or "")
            present = [term for term in known_modalities if re.search(rf"\b{term}(?:s)?\b", statement, re.IGNORECASE)]
            if len(present) >= 3:
                for term in present:
                    expanded_inputs.append({"id": f"input_{term}", "name": term.upper() if term == "imu" else term.title(), "role": "input modality", "evidence_ids": list(item.get("evidence_ids", []))})
            else:
                expanded_inputs.append(item)
        inputs = expanded_inputs
        if modules:
            relation_pairs = {(item["source"], item["target"]) for item in relations}
            for item in inputs:
                input_id = str(item.get("id") or "")
                if input_id and (input_id, modules[0]["id"]) not in relation_pairs:
                    relations.append({"source": input_id, "target": modules[0]["id"], "type": "data_flow", "label": "encoding", "evidence_ids": list(item.get("evidence_ids", []))})
            if outputs and innovations:
                output_id = str(outputs[0].get("id") or "")
                innovation_id = str(innovations[0].get("id") or "")
                if output_id and innovation_id and (output_id, innovation_id) not in relation_pairs:
                    relations.append({"source": output_id, "target": innovation_id, "type": "enables", "label": "emergent alignment", "evidence_ids": list(innovations[0].get("evidence_ids", []))})
        return {
            "summary": "Paper-review-grounded fallback planning result.",
            "paper_summary": {"summary": "Structured paper summary derived from paper_review.json.", "title": title, "paper_type": paper_review.get("paper_identity", {}).get("paper_type", "unknown"), "research_problem": (research_questions or [{}])[0], "central_claim": (claims or [{}])[0], "inputs": inputs, "outputs": outputs, "core_modules": modules, "innovations": innovations, "training_flow": paper_review.get("workflows", {}).get("training", []), "inference_flow": paper_review.get("workflows", {}).get("inference", []), "terminology": terminology, "unknowns": paper_review.get("unknowns", [])},
            "figure_specification": {"summary": "Scientific contract derived from universal paper review.", "figure_goal": figure_goal, "storyline": [item["id"] for item in modules], "must_show": [{"text": item["name"], "evidence_ids": item["evidence_ids"]} for item in modules], "modules": modules, "relations": relations, "inputs": inputs, "outputs": outputs, "innovations": innovations, "visual_priorities": [item["name"] for item in modules if item["role"] in {"module", "innovation"}], "terminology": terminology, "forbidden_inventions": forbidden},
            "design_plan": {"summary": "Paper-review-grounded information narrative.", "reading_order": [item["id"] for item in modules], "groups": [], "innovation_emphasis": [item.get("statement") for item in innovations], "preserve": [item["name"] for item in modules], "remove": forbidden},
            "layout_intent": {"summary": "Layout intent awaiting template selection.", "pattern": "left_to_right", "canvas_ratio": preferences.get("aspect_ratio", "auto"), "regions": [], "flow_description": "Follow the selected reference template while preserving paper relations.", "whitespace": "reference-matched"},
            "visual_metaphors": {"summary": "Concrete visual metaphors grounded in review roles.", "items": [{"module_id": item["id"], "metaphor": f"dense academic visual scene for {item['name']}", "must_show": [item["name"]], "avoid_showing": forbidden} for item in modules]},
            "style_plan": {"summary": "Style plan will inherit the selected reference template.", "medium": preferences.get("style_description"), "palette": preferences.get("preferred_palette", []), "viewpoint": "reference-matched academic diagram", "line_and_shadow": "reference-derived", "visual_density": preferences.get("visual_density", "high"), "background": "clean light background", "text_policy": "Image2 renders only exact short labels from the whitelist.", "positive_reference_rules": [], "negative_reference_rules": preferences.get("avoid", [])},
        }
    evidence = parsed["evidence"]
    headings = parsed.get("headings", [])
    first = evidence[0]
    title = next((line.strip() for line in first["text"].splitlines() if 10 <= len(line.strip()) <= 180), parsed["source_name"])
    generic = {"abstract", "introduction", "related work", "method", "methods", "experiments", "results", "conclusion", "references"}
    names = [item for item in headings if item.lower() not in generic and _plausible_component_heading(item)][:6]
    if len(names) < 3:
        names = ["Research Input", "Core Method", "Research Output"]
    modules = []
    for index, name in enumerate(names):
        evidence_id = evidence[min(index, len(evidence) - 1)]["id"]
        modules.append({"id": f"module_{index + 1:02d}", "name": name, "role": "paper-derived stage requiring VLM verification", "evidence_ids": [evidence_id]})
    relations = [
        {"source": modules[index]["id"], "target": modules[index + 1]["id"], "type": "data_flow", "label": "", "evidence_ids": list(dict.fromkeys(modules[index]["evidence_ids"] + modules[index + 1]["evidence_ids"]))}
        for index in range(len(modules) - 1)
    ]
    central = _first_sentence(first["text"])
    return {
        "summary": "Heuristic paper-figure planning result; scientific details require review.",
        "paper_summary": {
            "summary": "Heuristic structured paper summary.",
            "title": title,
            "paper_type": "unknown",
            "research_problem": {"text": central, "evidence_ids": [first["id"]]},
            "central_claim": {"text": "unknown", "evidence_ids": []},
            "inputs": [],
            "outputs": [],
            "core_modules": modules,
            "innovations": [],
            "training_flow": [],
            "inference_flow": [],
            "terminology": {item["name"]: item["name"] for item in modules},
            "unknowns": ["central claim", "training flow", "inference flow", "innovations"],
        },
        "figure_specification": {
            "summary": "Heuristic scientific contract for the figure.",
            "figure_goal": "Summarize the paper's main flow without inventing unsupported details.",
            "storyline": [item["name"] for item in modules],
            "must_show": [{"text": item["name"], "evidence_ids": item["evidence_ids"]} for item in modules],
            "modules": modules,
            "relations": relations,
            "inputs": [],
            "outputs": [],
            "innovations": [],
            "visual_priorities": [item["name"] for item in modules],
            "terminology": {item["name"]: item["name"] for item in modules},
            "forbidden_inventions": ["unsupported external knowledge base", "unsupported reinforcement learning", "unsupported agent or memory module"],
        },
        "design_plan": {
            "summary": "Heuristic left-to-right information narrative.",
            "reading_order": [item["id"] for item in modules],
            "groups": [],
            "innovation_emphasis": [],
            "preserve": [item["name"] for item in modules],
            "remove": ["unsupported generic AI decorations"],
        },
        "layout_intent": {
            "summary": "Heuristic layout intent.",
            "pattern": preferences.get("preferred_flow") or "left_to_right",
            "canvas_ratio": preferences.get("aspect_ratio", "16:9"),
            "regions": [{"id": item["id"], "position": "left" if index == 0 else "right" if index == len(modules) - 1 else "center", "relative_size": "medium", "contains": [item["id"]]} for index, item in enumerate(modules)],
            "flow_description": "A clear left-to-right scientific pipeline.",
            "whitespace": "moderate",
        },
        "visual_metaphors": {
            "summary": "Concrete fallback visual metaphors.",
            "items": [{"module_id": item["id"], "metaphor": f"dense academic mini-scene representing {item['name']}", "must_show": ["paper-specific structure", "visible process cue"], "avoid_showing": ["generic robot", "fake chart"]} for item in modules],
        },
        "style_plan": {
            "summary": "Style plan compiled from user preferences.",
            "medium": preferences.get("style_description"),
            "palette": preferences.get("preferred_palette", []),
            "viewpoint": "front-facing academic diagram",
            "line_and_shadow": "crisp thin lines with restrained soft shadows",
            "visual_density": preferences.get("visual_density", "high"),
            "background": "clean light background",
            "text_policy": "short exact labels only; no paragraphs, fake equations, fake axes, or invented numbers",
            "positive_reference_rules": [],
            "negative_reference_rules": preferences.get("avoid", []),
        },
    }


def build_review_grounded_plan(parsed: dict, preferences: dict, paper_review: dict) -> dict:
    """Compile a deterministic figure plan from an evidence-validated paper review."""
    return normalize_plan(_heuristic_plan(parsed, preferences, paper_review=paper_review), preferences)


def _ensure_summary(value: Any, fallback: str) -> dict:
    data = value if isinstance(value, dict) else {}
    data.setdefault("summary", fallback)
    return data


#: Plan keys whose value names an object instead of describing it.  Whitespace a
#: model wrapped across lines has to be removed from these rather than collapsed
#: into a space, or the name stops matching the paper and no later step can ever
#: ground it again.
IDENTIFIER_KEYS = frozenset(
    {
        "id",
        "instance_id",
        "source",
        "target",
        "key",
        "role",
        "kind",
        "type",
        "status",
        "detail_level",
        "topology",
        "evidence_ids",
        "node_id",
        "panel_id",
        "parent_id",
        "source_node_id",
        "source_relation_id",
    }
)


def _without_whitespace(value: str) -> str:
    return re.sub(r"\s+", "", value)


def repair_plan_whitespace(node: Any, *, parent_key: str = "") -> Any:
    """Rebuild a plan without the whitespace the model wrapped inside its names.

    A planner answer that escapes a line break in the middle of a key or an
    identifier (``I_\\npool_A``, ``central\\n_claim``) is still valid JSON, so it
    parses cleanly and then fails every grounding check for a reason no later
    step can fix.  Keys and identifier values lose their whitespace entirely;
    prose keeps single spaces so sentences stay readable.
    """

    if isinstance(node, dict):
        rebuilt: dict[Any, Any] = {}
        for key, value in node.items():
            name = _without_whitespace(key) if isinstance(key, str) else key
            if not name:
                name = key
            rebuilt[name] = repair_plan_whitespace(value, parent_key=str(name))
        return rebuilt
    if isinstance(node, list):
        return [repair_plan_whitespace(item, parent_key=parent_key) for item in node]
    if isinstance(node, str):
        if parent_key in IDENTIFIER_KEYS:
            return _without_whitespace(node)
        return re.sub(r"\s+", " ", node).strip()
    return node


def normalize_plan(raw: dict, preferences: dict) -> dict:
    result = repair_plan_whitespace(raw if isinstance(raw, dict) else {})
    result.setdefault("summary", "Direct paper-figure planning result.")
    result["paper_summary"] = _ensure_summary(result.get("paper_summary"), "Structured paper summary.")
    result["figure_specification"] = _ensure_summary(result.get("figure_specification"), "Scientific contract for the figure.")
    result["design_plan"] = _ensure_summary(result.get("design_plan"), "Information narrative for the figure.")
    result["layout_intent"] = _ensure_summary(result.get("layout_intent"), "Layout intent for the image model.")
    result["visual_metaphors"] = _ensure_summary(result.get("visual_metaphors"), "Concrete visual metaphors.")
    result["style_plan"] = _ensure_summary(result.get("style_plan"), "Image style contract.")
    result["layout_intent"].setdefault("canvas_ratio", preferences.get("aspect_ratio", "16:9"))
    return result


def plan_fast_paper_contract(
    parsed: dict,
    preferences: dict,
    mode: str = "vlm",
    model: str | None = None,
    timeout_seconds: int = 45,
    retries: int = 2,
    evidence_max_chars: int = 36000,
    deadline_at: float | None = None,
) -> tuple[dict, dict]:
    prompt = _fast_planner_prompt(parsed, preferences, evidence_max_chars=evidence_max_chars)
    metadata: dict[str, Any] = {"requested_mode": mode, "mode": mode, "model": None, "warning": None, "prompt": prompt, "provider": {}, "context": context_stats(prompt, field_names=["candidate_captions", "extraction_scope", "preferences", "prioritized_evidence", "figure_spatial_excerpt", "panel_inventory"])}
    if mode == "vlm" and vlm_credentials_available():
        resolved = resolve_vlm_model("RFS_FAST_FRAMEWORK_MODEL", "RFS_PAPER_TO_IMAGE_MODEL", "RFS_PAPER_PLANNER_MODEL", explicit_model=model)
        try:
            raw = call_vlm_json(
                prompt,
                [],
                model=resolved,
                timeout=max(10, int(timeout_seconds)),
                retries=max(0, int(retries)),
                call_metadata=metadata["provider"],
                deadline_at=deadline_at,
            )
            metadata["model"] = resolved
            return _compile_fast_plan(raw, preferences), metadata
        except Exception as exc:
            metadata.update({"mode": "heuristic_after_vlm_failure", "warning": str(exc), "model": resolved})
    elif mode == "vlm":
        metadata.update({"mode": "heuristic_without_credentials", "warning": "API_BASE and API_KEY/GEMINI_API_KEY are required for VLM planning"})
    return normalize_plan(_heuristic_plan(parsed, preferences), preferences), metadata


def plan_paper_image(parsed: dict, preferences: dict, mode: str = "vlm", model: str | None = None, reference_images: list[str] | None = None, paper_review: dict | None = None, timeout_seconds: int = 240, retries: int = 1, evidence_max_chars: int = 58000, deadline_at: float | None = None) -> tuple[dict, dict]:
    prompt = _planner_prompt(parsed, preferences, paper_review=compact_review(paper_review), evidence_max_chars=min(max(4000, int(evidence_max_chars)), 14000))
    metadata = {"requested_mode": mode, "mode": mode, "model": None, "warning": None, "prompt": prompt, "provider": {}, "context": context_stats(prompt, image_count=len(reference_images or []), field_names=["preferences", "compact_paper_review", "prioritized_evidence", "reference_images"])}
    if mode == "vlm" and vlm_credentials_available():
        resolved = resolve_vlm_model("RFS_PAPER_TO_IMAGE_MODEL", "RFS_PAPER_PLANNER_MODEL", explicit_model=model)
        try:
            raw = call_vlm_json(prompt, reference_images or [], model=resolved, timeout=max(10, int(timeout_seconds)), retries=max(0, int(retries)), call_metadata=metadata["provider"], deadline_at=deadline_at)
            metadata["model"] = resolved
            return normalize_plan(raw, preferences), metadata
        except Exception as exc:
            metadata.update({"mode": "heuristic_after_vlm_failure", "warning": str(exc), "model": resolved})
    elif mode == "vlm":
        metadata.update({"mode": "heuristic_without_credentials", "warning": "API_BASE and API_KEY/GEMINI_API_KEY are required for VLM planning"})
    return normalize_plan(_heuristic_plan(parsed, preferences, paper_review=paper_review), preferences), metadata


def _item_label_for_prompt(item: dict[str, Any]) -> str:
    return str(item.get("visible_label") or item.get("name") or item.get("text") or item.get("statement") or item.get("label") or item.get("id") or "").strip()


def collect_visible_labels(spec: dict, limit: int = 40) -> list[str]:
    labels: list[str] = []
    normalized_labels: set[str] = set()
    panel_graphs = [item for item in spec.get("panel_graphs", []) if isinstance(item, dict) and item.get("nodes")] if isinstance(spec.get("panel_graphs"), list) else []

    def add(value: object) -> None:
        if isinstance(value, dict):
            value = value.get("visible_label") or value.get("name") or value.get("text") or value.get("statement")
        label = str(value or "").strip()
        normalized = re.sub(r"[^\w\u4e00-\u9fff]+", "", label.casefold())
        if label and len(label) <= 64 and normalized and normalized not in normalized_labels:
            labels.append(label)
            normalized_labels.add(normalized)

    explicit_required = spec.get("required_labels", []) if isinstance(spec.get("required_labels"), list) else []
    for item in explicit_required:
        add(item)
    if not explicit_required:
        terminology = spec.get("terminology", {})
        if isinstance(terminology, dict):
            for value in terminology.values():
                add(value)
        elif isinstance(terminology, list):
            for item in terminology:
                add(item)
        for field in ("inputs", "modules", "outputs"):
            for item in spec.get(field, []) if isinstance(spec.get(field), list) else []:
                add(item)
        for item in spec.get("innovations", []) if isinstance(spec.get("innovations"), list) else []:
            if isinstance(item, dict) and (item.get("visible_label_required") or item.get("must_appear_in_figure")):
                add(item)
        for relation in spec.get("relations", []) if isinstance(spec.get("relations"), list) else []:
            if isinstance(relation, dict):
                add(relation.get("label"))
        for panel in panel_graphs:
            if not isinstance(panel, dict):
                continue
            add(panel.get("label"))
            for node in panel.get("nodes", []) if isinstance(panel.get("nodes"), list) else []:
                add(node)
            for relation in panel.get("relations", []) if isinstance(panel.get("relations"), list) else []:
                if isinstance(relation, dict):
                    add(relation.get("label"))
    return labels[: max(1, int(limit))]


def collect_visual_relations(spec: dict) -> list[dict[str, str]]:
    panel_graphs = [item for item in spec.get("panel_graphs", []) if isinstance(item, dict) and item.get("nodes")] if isinstance(spec.get("panel_graphs"), list) else []
    if len(panel_graphs) >= 2:
        result: list[dict[str, str]] = []
        for panel in panel_graphs:
            panel_id = str(panel.get("id") or "")
            panel_label = str(panel.get("panel_label") or panel_id)
            nodes = {
                str(item.get("instance_id") or item.get("id") or ""): item
                for item in panel.get("nodes", []) if isinstance(item, dict)
            }
            for relation in panel.get("relations", []) if isinstance(panel.get("relations"), list) else []:
                if not isinstance(relation, dict):
                    continue
                source = str(relation.get("source") or "")
                target = str(relation.get("target") or "")
                if source not in nodes or target not in nodes or source == target:
                    continue
                result.append({
                    "source": source,
                    "source_label": _item_label_for_prompt(nodes[source]),
                    "target": target,
                    "target_label": _item_label_for_prompt(nodes[target]),
                    "type": str(relation.get("type") or "data_flow"),
                    "label": str(relation.get("label") or "").strip(),
                    "panel_id": panel_id,
                    "panel_label": panel_label,
                })
        return result
    entities = {
        str(item.get("id")): str(item.get("visible_label") or item.get("name") or item.get("text") or item.get("statement") or item.get("id"))
        for field in ("inputs", "modules", "outputs", "innovations")
        for item in (spec.get(field, []) if isinstance(spec.get(field), list) else [])
        if isinstance(item, dict) and item.get("id")
    }
    repeatable = {re.sub(r"[^\w\u4e00-\u9fff]+", "", str(value).casefold()) for value in spec.get("repeatable_labels", []) if str(value).strip()} if isinstance(spec.get("repeatable_labels"), list) else set()
    shared_ids = {
        item_id for item_id, label in entities.items()
        if re.sub(r"[^\w\u4e00-\u9fff]+", "", label.casefold()) in repeatable
    }
    grouped: dict[tuple[str, str], dict[str, object]] = {}
    for relation in spec.get("relations", []) if isinstance(spec.get("relations"), list) else []:
        if not isinstance(relation, dict):
            continue
        source, target = str(relation.get("source") or ""), str(relation.get("target") or "")
        if not source or not target or source == target or source in shared_ids or target in shared_ids:
            continue
        entry = grouped.setdefault((source, target), {"types": [], "labels": []})
        relation_type = str(relation.get("type") or "data_flow")
        if relation_type not in entry["types"]:
            entry["types"].append(relation_type)
        label = str(relation.get("label") or "").strip()
        if label and label not in entry["labels"]:
            entry["labels"].append(label)
    priority = ("feedback_loop", "evaluation", "revision_input", "feedback", "conditioning", "data_flow", "prediction", "generation_input")
    result: list[dict[str, str]] = []
    for (source, target), entry in grouped.items():
        types = list(entry["types"])
        relation_type = next((value for value in priority if value in types), types[0] if types else "data_flow")
        result.append({
            "source": source,
            "source_label": entities.get(source, source),
            "target": target,
            "target_label": entities.get(target, target),
            "type": relation_type,
            "label": str(entry["labels"][0]) if entry["labels"] else "",
        })
    return result


def collect_visual_roles(spec: dict) -> list[dict[str, str]]:
    panel_graphs = [item for item in spec.get("panel_graphs", []) if isinstance(item, dict) and item.get("nodes")] if isinstance(spec.get("panel_graphs"), list) else []
    if len(panel_graphs) >= 2:
        return [
            {
                "id": str(item.get("instance_id") or item.get("id") or ""),
                "label": _item_label_for_prompt(item),
                "field": "panel_instance",
                "role": str(item.get("role") or "module"),
                "visual_constraint": "Keep this occurrence inside its declared panel and connect it only through panel-local relations.",
                "visual_cue": "A compact paper-grounded visual object matching this panel-local role.",
                "panel_id": str(panel.get("id") or ""),
            }
            for panel in panel_graphs
            for item in panel.get("nodes", []) if isinstance(item, dict)
        ]
    result: list[dict[str, str]] = []
    label_to_index: dict[str, int] = {}
    for field in ("inputs", "modules", "outputs", "innovations"):
        for item in spec.get(field, []) if isinstance(spec.get(field), list) else []:
            if not isinstance(item, dict) or not item.get("id"):
                continue
            label = str(item.get("visible_label") or item.get("name") or item.get("text") or item.get("statement") or item.get("id"))
            normalized_label = re.sub(r"[^\w\u4e00-\u9fff]+", "", label.casefold())
            if field == "innovations" and normalized_label in label_to_index:
                existing = result[label_to_index[normalized_label]]
                existing["visual_constraint"] += " Also highlight this entity as an evidence-backed paper innovation without duplicating the node."
                existing["visual_cue"] += " Add a compact innovation emphasis or mechanism inset inside the same entity."
                existing["innovation"] = "true"
                continue
            role = str(item.get("role") or field.rstrip("s"))
            normalized_role = re.sub(r"[^a-z0-9]+", "", role.casefold())
            if "datasource" in normalized_role or role.casefold() in {"dataset", "source dataset"}:
                constraint = "Show as an upstream data source that provides declared modalities; never show it as a peer modality or processing module."
                visual_cue = "Layered dataset cards, database/archive symbols, or source-record thumbnails grounded in the named source."
            elif "modality" in normalized_role:
                constraint = "Show as an observed input modality downstream of any declared data sources and upstream of the processing method."
                visual_cue = "A representative sample or measurement thumbnail for the named modality, not an empty text card."
            elif field == "inputs":
                constraint = "Show at the input boundary and connect only to its declared downstream target."
                visual_cue = "A compact paper-grounded input example or input-object icon."
            elif field == "outputs" or any(term in normalized_role for term in ("output", "prediction", "evaluation", "classification", "regressionhead")):
                constraint = "Show at the output/evaluation boundary, not as an upstream input or internal method module."
                visual_cue = "A compact outcome, prediction, or evaluation visual that does not introduce new metrics or numbers."
            elif field == "innovations":
                constraint = "Show as a highlighted paper innovation attached to its supported method context, not as an unrelated pipeline stage."
                visual_cue = "A small mechanism inset that visually explains the named innovation."
            else:
                constraint = "Show as an internal method operation or component between its declared incoming and outgoing relations."
                visual_cue = "A mechanism-level mini diagram or scientific icon showing what the named operation does."
            result.append({
                "id": str(item.get("id")),
                "label": label,
                "field": field,
                "role": role,
                "visual_constraint": constraint,
                "visual_cue": visual_cue,
            })
            if normalized_label:
                label_to_index[normalized_label] = len(result) - 1
    return result


def _topology_specific_prompt_rules(spec: dict) -> str:
    topology = str(spec.get("topology") or "unknown")
    normalized_labels = {
        re.sub(r"[^\w\u4e00-\u9fff]+", "", str(item.get("name") or item.get("label") or item.get("title") or "").casefold())
        for field in ("inputs", "modules", "outputs", "innovations")
        for item in (spec.get(field, []) if isinstance(spec.get(field), list) else [])
        if isinstance(item, dict)
    }
    feedback_signature = {
        "inputx", "generate", "initialoutput", "feedback", "selffeedback", "refine", "refinedoutput", "modelm",
    }
    dense_signature = {
        "promptablesegmentationtask", "segmentanythingmodel", "dataengine", "image", "prompt",
        "imageencoder", "promptencoder", "maskdecoder", "validsegmentationmask",
        "assistedmanual", "semiautomatic", "fullyautomatic", "sa1b",
    }
    if topology == "feedback" and feedback_signature.issubset(normalized_labels):
        return """Feedback-loop topology hard rules:
- Keep one top-row forward chain: input x -> Generate -> Initial Output -> FEEDBACK.
- Initial Output must have exactly two distinct outgoing arrows: one to FEEDBACK and one directly to REFINE.
- FEEDBACK must produce Self-Feedback, and Self-Feedback must have its own distinct outgoing arrow into REFINE.
- REFINE must therefore receive exactly two visually separate incoming arrows: Initial Output -> REFINE and Self-Feedback -> REFINE. Do not merge them upstream and do not redirect either arrow to the other source node.
- The connector labeled iterate must start at Refined output and terminate only at FEEDBACK. Its arrowhead must visibly touch the FEEDBACK boundary.
- Model M is a reusable badge inside Generate, FEEDBACK, and/or REFINE; it is not a separate flow node and must not receive extra arrows.
- Forbidden connectors: Initial Output -> Self-Feedback; Refined output -> Generate; REFINE -> Self-Feedback; Self-Feedback -> FEEDBACK; FEEDBACK -> REFINE as a shortcut that omits the visible Self-Feedback node.
- Preserve the supplied blueprint's two-row geometry and right-side return loop. Do not rearrange the loop into a different circular or bottom-return layout."""
    if topology == "feedback":
        return """Feedback-loop topology hard rules:
- Follow only the declared nodes and directed relations in the scientific contract.
- Keep the forward path and return path visually separate, with the return arrowhead touching its declared target.
- Preserve every declared intermediate feedback artifact; do not shortcut across it.
- If two declared sources enter one refinement node, draw two visibly distinct incoming arrows.
- Do not introduce Self-Refine-specific labels, shared-model badges, or connector names unless they are present in the contract."""
    if topology == "dense_multiframe" and dense_signature.issubset(normalized_labels):
        return """Dense multi-frame topology hard rules:
- Preserve exactly three macro panels: Promptable Segmentation Task, Segment Anything Model, and Data Engine.
- Image must connect only to Image Encoder; Prompt must connect only to Prompt Encoder.
- Image Encoder and Prompt Encoder must each have a separate directed arrow into Mask Decoder. Do not chain one encoder through the other.
- Mask Decoder must connect directly to Valid Segmentation Mask inside the model panel.
- Data Engine must be a separate vertical chain: Assisted-manual -> Semi-automatic -> Fully Automatic -> SA-1B.
- Draw the Segment Anything Model -> Data Engine annotation-support relation as a distinct high-level container-to-container arrow, visually separate from the mask-prediction path.
- Forbidden connectors: Prompt -> Image Encoder; Image -> Prompt Encoder; Image Encoder -> Prompt Encoder; Prompt Encoder -> Image Encoder; Fully Automatic -> Valid Segmentation Mask; Mask Decoder -> Data Engine as a replacement for the required container-level support arrow.
- Do not let any Data Engine stage enter Valid Segmentation Mask, and do not let the model inference path enter SA-1B."""
    if topology == "dense_multiframe":
        panels = [item for item in spec.get("overview_panels", []) if isinstance(item, dict) and str(item.get("label") or "").strip()]
        panel_lines = "\n".join(
            f"  {str(item.get('panel_label') or index + 1)}. {str(item.get('label') or '').strip()} — {str(item.get('description') or '').strip()}"
            for index, item in enumerate(panels)
        )
        panel_rule = (
            f"- Preserve exactly {len(panels)} macro panels with these evidence-backed meanings:\n{panel_lines}"
            if panels else
            "- Preserve the declared macro-panel grouping and every directed cross-panel relation."
        )
        return f"""Dense multi-frame topology hard rules:
{panel_rule}
- Preserve the declared macro-panel grouping and every directed cross-panel relation.
- Keep independent input branches separate until their declared convergence node.
- Keep panel-local output paths separate from dataset, training, or data-engine paths.
- Draw high-level container-to-container relations separately from internal module arrows.
- Do not replace evidence-backed pretraining, data, or evaluation panels with an invented architecture-detail inset.
- Do not add a titled internal inset, legend, or architecture breakdown unless its title is in the exact visible label whitelist.
- Do not introduce SAM-specific encoders, masks, data-engine stages, or dataset labels unless they are present in the contract."""
    return "No additional topology-specific rules. Follow the directed connector checklist exactly."


def compile_image_prompt(
    plan: dict,
    preferences: dict,
    candidate_variant: int = 1,
    selected_template: dict | None = None,
    *,
    deterministic_overlay: bool = False,
    exemplar_style_brief: dict | None = None,
) -> str:
    spec = plan["figure_specification"]
    design = plan["design_plan"]
    layout = plan["layout_intent"]
    metaphors = plan["visual_metaphors"]
    style = plan["style_plan"]
    template = selected_template or {}
    semantic_plan = template.get("semantic_plan", {}) if isinstance(template.get("semantic_plan"), dict) else {}
    labels = collect_visible_labels(spec)
    repeatable_labels = [str(value).strip() for value in spec.get("repeatable_labels", []) if str(value).strip()] if isinstance(spec.get("repeatable_labels"), list) else []
    optional_visible_labels = [str(value).strip() for value in spec.get("optional_visible_labels", []) if str(value).strip()] if isinstance(spec.get("optional_visible_labels"), list) else []
    repeatable_normalized = {re.sub(r"[^\w\u4e00-\u9fff]+", "", value.casefold()) for value in repeatable_labels}
    numbered_labels = "\n".join(
        f"{index}. {label}" + (" (may repeat to show evidence-supported component reuse)" if re.sub(r"[^\w\u4e00-\u9fff]+", "", label.casefold()) in repeatable_normalized else "")
        for index, label in enumerate(labels, start=1)
    )
    panel_graphs = [item for item in spec.get("panel_graphs", []) if isinstance(item, dict) and item.get("nodes")] if isinstance(spec.get("panel_graphs"), list) else []
    panel_graph_active = len(panel_graphs) >= 2
    visual_relations = [] if panel_graph_active else collect_visual_relations(spec)
    visual_roles = collect_visual_roles(spec)
    panel_instance_lines: list[str] = []
    panel_connector_lines: list[str] = []
    for panel in panel_graphs:
        panel_id = str(panel.get("id") or "panel")
        panel_label = str(panel.get("panel_label") or panel_id)
        node_by_id = {
            str(item.get("instance_id") or item.get("id") or ""): item
            for item in panel.get("nodes", []) if isinstance(item, dict)
        }
        for item in node_by_id.values():
            instance_id = str(item.get("instance_id") or item.get("id") or "")
            panel_instance_lines.append(f"- Panel {panel_label}: {instance_id} = {_item_label_for_prompt(item)} [{str(item.get('role') or 'module')}]" )
        for relation in panel.get("relations", []) if isinstance(panel.get("relations"), list) else []:
            if not isinstance(relation, dict):
                continue
            source = str(relation.get("source") or "")
            target = str(relation.get("target") or "")
            if source not in node_by_id or target not in node_by_id:
                continue
            source_label = _item_label_for_prompt(node_by_id[source])
            target_label = _item_label_for_prompt(node_by_id[target])
            relation_label = str(relation.get("label") or "").strip()
            panel_connector_lines.append(
                f"{len(panel_connector_lines) + 1}. Panel {panel_label}: {source_label} ({source}) -> {target_label} ({target}) [{str(relation.get('type') or 'data_flow')}]"
                + (f" label: {relation_label}" if relation_label else "")
            )
    connector_checklist = "\n".join(panel_connector_lines) if panel_graph_active else "\n".join(
        f"{index}. {item['source_label']} -> {item['target_label']} [{item['type']}]" + (f" label: {item['label']}" if item["label"] else "")
        for index, item in enumerate(visual_relations, start=1)
    )
    input_roles = [item for item in visual_roles if item.get("role") == "input"]
    output_roles = [item for item in visual_roles if item.get("role") == "output"]
    relations_by_target: dict[str, list[dict]] = {}
    for relation in visual_relations:
        relations_by_target.setdefault(str(relation.get("target") or ""), []).append(relation)
    shared_input_rules = []
    for target_id, incoming_relations in relations_by_target.items():
        peer_inputs = [item for item in incoming_relations if any(role.get("id") == item.get("source") for role in input_roles)]
        if len(peer_inputs) < 2:
            continue
        target_label = str(peer_inputs[0].get("target_label") or target_id)
        source_labels = [str(item.get("source_label") or item.get("source")) for item in peer_inputs]
        shared_input_rules.append(
            f"- Treat {', '.join(source_labels)} as peer input sources. Never draw arrows between these peer cards. Draw one separate directed arrow from each peer input to {target_label}."
        )
    output_labels = [str(item.get("label") or item.get("id")) for item in output_roles]
    boundary_role_rules = ("- Panel-local instance roles and panel-local connector checklists are authoritative. Never connect instances across panels unless a separate explicit cross-panel relation is listed." if panel_graph_active else "\n".join([
        *shared_input_rules,
        *( [f"- Treat {', '.join(output_labels)} as output-only nodes. Never draw an arrow from an output back into an input or method module unless an explicit feedback relation appears in the connector checklist."] if output_labels else [] ),
    ]) or "- Keep all declared inputs as source roles and all declared outputs as sink roles unless an explicit feedback relation says otherwise.")
    topology_specific_rules = _topology_specific_prompt_rules(spec)
    compact_roles = (
        [
            {
                "id": str(item.get("instance_id") or item.get("id") or ""),
                "label": _item_label_for_prompt(item),
                "role": str(item.get("role") or "module"),
                "panel_id": str(panel.get("id") or ""),
                "visual_cue": "A compact paper-grounded visual object matching this panel-local role.",
            }
            for panel in panel_graphs
            for item in panel.get("nodes", []) if isinstance(item, dict)
        ]
        if panel_graph_active else
        [
            {"id": item["id"], "label": item["label"], "role": item["role"], "visual_cue": item["visual_cue"]}
            for item in visual_roles
        ]
    )
    compact_contract = {
        "figure_goal": spec.get("figure_goal"),
        "central_claim": spec.get("central_claim", {}).get("text") if isinstance(spec.get("central_claim"), dict) else spec.get("central_claim"),
        "topology": spec.get("topology"),
        "training_flow": spec.get("training_flow", []),
        "inference_flow": spec.get("inference_flow", []),
        "forbidden_inventions": spec.get("forbidden_inventions", []),
        "uncertainties": spec.get("uncertainties", []),
        "overview_panels": ([
            {
                "id": item.get("id"),
                "panel_label": item.get("panel_label"),
                "label": item.get("label"),
                "description": item.get("description"),
                "entity_ids": [str(node.get("instance_id") or node.get("id") or "") for node in item.get("nodes", []) if isinstance(node, dict)],
            }
            for item in panel_graphs
        ] if panel_graph_active else [
            {
                "id": item.get("id"),
                "panel_label": item.get("panel_label"),
                "label": item.get("label"),
                "description": item.get("description"),
                "entity_ids": item.get("entity_ids", []),
            }
            for item in spec.get("overview_panels", []) if isinstance(item, dict)
        ]),
        "panel_graphs": [
            {
                "id": panel.get("id"),
                "panel_label": panel.get("panel_label"),
                "label": panel.get("label"),
                "nodes": [
                    {
                        "instance_id": item.get("instance_id") or item.get("id"),
                        "name": _item_label_for_prompt(item),
                        "role": item.get("role"),
                    }
                    for item in panel.get("nodes", []) if isinstance(item, dict)
                ],
                "relations": [
                    {
                        "source": item.get("source"),
                        "target": item.get("target"),
                        "type": item.get("type"),
                        "label": item.get("label"),
                    }
                    for item in panel.get("relations", []) if isinstance(item, dict)
                ],
            }
            for panel in panel_graphs
        ],
        "supporting_visual_cues": [
            {
                "visual_cue": item.get("visual_cue") or "one unlabeled panel-local scientific mechanism detail",
                "must_be_unlabeled": True,
            }
            for item in spec.get("supporting_details", []) if isinstance(item, dict)
        ],
    }
    compact_style = {
        "medium": style.get("medium"),
        "palette": style.get("palette", []),
        "viewpoint": style.get("viewpoint"),
        "line_and_shadow": style.get("line_and_shadow"),
        "visual_density": style.get("visual_density"),
        "background": style.get("background"),
        "negative_reference_rules": style.get("negative_reference_rules", []),
        "template_style": style.get("template_style", {}),
    }
    compact_template = {
        "template_id": template.get("template_id"),
        "visual_density": template.get("visual_density"),
        "style": template.get("style", {}),
        # Semantic planning owns scientific labels and node membership, while the
        # curated template still contributes content-free macro geometry and flow
        # rhythm. Dropping these panels made the generated canvas unnecessarily
        # sparse and discarded the project-level exemplar experience.
        "panels": template.get("panels", []),
        "connectors": template.get("connectors", []),
    }
    compact_semantic_plan = {
        "applied": bool(semantic_plan.get("applied")),
        "topology": semantic_plan.get("topology"),
        "panels": [
            {
                "id": item.get("id"),
                "panel_label": item.get("panel_label"),
                "label": item.get("label"),
                "bbox_percent": item.get("bbox_percent"),
                "groups": item.get("groups", []),
            }
            for item in semantic_plan.get("panels", []) if isinstance(item, dict)
        ],
        "nodes": [
            {
                "id": item.get("id"),
                "label": item.get("label"),
                "rank": item.get("rank"),
                "bbox_percent": item.get("bbox_percent"),
            }
            for item in semantic_plan.get("nodes", []) if isinstance(item, dict)
        ],
        "connectors": [
            {
                "source": item.get("source"),
                "target": item.get("target"),
                "type": item.get("type"),
                "label": item.get("label"),
                "path_percent": item.get("path_percent"),
                "route_style": item.get("route_style"),
            }
            for item in semantic_plan.get("connectors", []) if isinstance(item, dict)
        ],
    }
    global_structure = plan.get("global_structure_plan") if isinstance(plan.get("global_structure_plan"), dict) else {}
    layout_candidates = [item for item in plan.get("layout_candidates", []) if isinstance(item, dict) and isinstance(item.get("plan"), dict)]
    if layout_candidates:
        global_structure = layout_candidates[(max(1, int(candidate_variant)) - 1) % len(layout_candidates)]["plan"]
    segmentation = plan.get("segmentation_plan") if isinstance(plan.get("segmentation_plan"), dict) else {}
    structure_section = f"""
Pre-generation global structure and segmentation contract (hard structural guidance):
{json.dumps({
    'topology': global_structure.get('topology') or spec.get('topology'),
    'reading_order': global_structure.get('reading_order', []),
    'nodes': [
        {'id': item.get('id'), 'label': item.get('label'), 'bbox_percent': item.get('bbox_percent'), 'role': item.get('role')}
        for item in global_structure.get('nodes', []) if isinstance(item, dict)
    ],
    'relations': global_structure.get('relations', []),
    'slots': [
        {'slot_id': item.get('slot_id'), 'semantic_node_ids': item.get('semantic_node_ids'), 'bbox_percent': item.get('bbox_percent'), 'input_ports': item.get('input_ports'), 'output_ports': item.get('output_ports')}
        for item in segmentation.get('slots', []) if isinstance(item, dict)
    ],
}, ensure_ascii=False, separators=(',', ':'))}

Treat this contract as the shared source of truth for the future editable PPT stage:
- preserve the planned node count, relative geometry, reading order, and topology;
- keep each planned slot visually separable and leave its declared input/output corridors clear;
- do not merge or split planned semantic nodes;
- reserve clean space for native PPT labels, formulas, relation labels, and connectors;
- if a local visual detail conflicts with the contract, preserve the contract and simplify the detail.
"""
    if deterministic_overlay:
        visible_label_section = f"""Scientific concept-to-node mapping. These terms describe node meaning only; do not render any of them as visible text in the visual substrate:
{json.dumps(labels, ensure_ascii=False)}

Reserve a clean translucent or light header band inside every node for later deterministic text composition. Do not render letters, words, numbers, formulas, axis labels, panel IDs, legends, headings, watermarks, or pseudo-text anywhere in the image."""
        connector_section = f"""Post-composition directed connector contract. These relations will be drawn later by the deterministic overlay compiler; do not render any arrow, line, connector, arrowhead, loop, or flow path in the visual substrate:
{connector_checklist or '- No visible connector is rendered by Image2.'}

Keep the whitespace corridors around the supplied semantic connector paths clear enough for the later overlay. Preserve node bounding boxes, but leave all relationships visually unconnected."""
        overlay_requirements = """- Generate only the image-rich visual substrate inside the supplied node geometry.
- Do not render any scientific text or any connector geometry; exact labels and arrows are owned by overlay_spec.json and semantic_plan.path_percent.
- Reserve a clean header band in each node and clear connector corridors between nodes.
- Enrich every input, core method, innovation, and output node with a paper-grounded visual cue from the role checklist.
- Use roughly 55-85% of each node interior below the reserved header band for a representative sample, mechanism inset, scientific icon, or outcome cue.
- For modality nodes, show a representative sample; for data sources, show archive cues; for method nodes, show the internal operation; for outputs, show a compact outcome cue.
- Preserve every node bounding box and layer assignment. Do not merge, split, move, rename, or add nodes.
- Do not add arrows, lines, loops, legends, headings, architecture submodule names, letters, numbers, fake UI text, formulas, axes, charts, citations, or watermarks.
- Keep all major objects complete and inside the canvas; avoid large blank margins.
- Do not invent datasets, mechanisms, metrics, modules, or results.
- Treat the supplied visual-only blueprint as a hard macro-layout guide and replace only the empty node interiors with paper-grounded visual explanation."""
        output_mode_summary = "Create one publication-quality visual substrate for a scientific framework figure. Exact labels and connectors will be composed deterministically after generation."
    else:
        visible_label_section = f"""Exact visible label whitelist (render these exactly and do not invent alternatives):
{json.dumps(labels, ensure_ascii=False)}

Evidence-backed optional visible labels (allowed if useful, but not required):
{json.dumps(optional_visible_labels, ensure_ascii=False)}

Mandatory visible label checklist:
{numbered_labels}

Every checklist item must appear at least once as clearly readable text in the completed figure. Labels explicitly marked as repeatable may appear more than once only to show reuse of the same paper-supported component; every other label must appear exactly once. An icon, symbol, equation variable, or unlabeled output card does not satisfy a checklist item. Reserve enough width for long labels and place each output label beside or inside its output node."""
        connector_section = f"""Mandatory directed connector checklist:
{connector_checklist}

Every connector above must be visible with the stated direction. Do not bypass an intermediate module, do not replace two required inputs with one shortcut arrow, and do not add direct source-to-output paths that are absent from this checklist. Shared component labels may repeat as visual badges without adding extra implementation-level arrows."""
        overlay_requirements = """- Preserve every paper-supported core module and relation.
- Use the exact short labels listed in terminology when labels are necessary.
- Make arrows unambiguous and follow the specified source-to-target direction.
- Emphasize the paper's actual innovation rather than generic AI decoration.
- Use a clean academic composition with dense but readable information hierarchy.
- Enrich every input, core method, innovation, and output node with a paper-grounded visual cue from the role checklist. Do not return a bare text-only flowchart.
- Use roughly 35-60% of each node interior for a representative example, mechanism inset, scientific icon, or outcome cue while keeping the exact label clearly readable.
- For modality nodes, show a representative sample of that modality; for data sources, show dataset/archive cues; for method nodes, show the internal operation; for outputs, show a compact outcome/evaluation cue.
- Preserve the supplied geometry and connectors while adding visual explanation inside the nodes; do not merely add gradients or shadows to the blueprint boxes.
- Keep all major objects complete and inside the canvas; avoid large blank margins.
- Do not invent datasets, mechanisms, formulas, metrics, modules, or results.
- Do not render paragraphs, citations, fake equations, fake axes, fake charts, or invented numbers.
- Avoid commercial-poster styling, cyberpunk effects, unrelated robots, generic brains, and repeated dashboard cards.
- Treat the supplied blueprint as a hard macro-layout guide, but replace all reference content with this paper's content.
- When semantic blueprint geometry is supplied, preserve every node bounding box, layer assignment, connector path, and arrow direction; visual decoration may change but graph geometry may not.
- Template panels are macro spatial regions, not an exact limit on the number of scientific content nodes inside those regions.
- Every visible scientific word must come from the exact label whitelist; do not paraphrase labels.
- The exact visible label whitelist is the only permitted visible text. Do not add a legend, inset title, process verb, architecture submodule, or explanatory heading outside that whitelist.
- Render every mandatory visible label at least once; repeat only labels explicitly marked as evidence-supported shared components, and never replace an output label with an icon-only node.
- Keep separately named scientific components as separately editable nodes; do not collapse special tokens, embeddings, heads, inputs, or outputs into composite labels.
- Include explicit input-boundary and output-boundary connectors."""
        output_mode_summary = "Create one complete publication-quality scientific framework figure as a raster image."
    if exemplar_style_brief:
        exemplar_keys = [
            "layout_rules",
            "structure_rules",
            "density_target",
            "hierarchy_rules",
            "color_rules",
            "text_rules",
            "connector_rules",
            "avoid_rules",
            "anti_copy_rules",
            "project_exemplar_rules",
            "user_style_rules",
            "selected_positive_ids",
            "selected_negative_ids",
        ]
        exemplar_compact = {key: exemplar_style_brief.get(key) for key in exemplar_keys if key in exemplar_style_brief}
        exemplar_section = f"""
Reference exemplar style brief (soft visual guidance only; never overrides the scientific contract or exact labels):
{json.dumps(exemplar_compact, ensure_ascii=False, separators=(',', ':'))}

Exemplar alignment instructions:
- Apply only reusable design/style rules explicitly present in the brief (layout, structure, density, hierarchy, color, text, connector grammar, panel surfaces, overall mood).
- Do not copy reference-exemplar paper terms, logos, captions, unique objects, datasets, or scientific content; the current paper contract is the only content authority.
- Do not let style rules break the paper's layout, structure, scientific relations, or exact labels.
"""
    else:
        exemplar_section = ""
    return f"""
# Summary

{output_mode_summary}

Candidate variant: {candidate_variant}.
Scientific goal: {spec.get('figure_goal')}.
Target canvas ratio: {preferences.get('aspect_ratio', '16:9')}.
Text language: {preferences.get('language', 'English')}.

Compact scientific contract:
{json.dumps(compact_contract, ensure_ascii=False, separators=(',', ':'))}

Style contract:
{json.dumps(compact_style, ensure_ascii=False, separators=(',', ':'))}

{exemplar_section}
Reference-derived content-free template contract:
{json.dumps(compact_template, ensure_ascii=False, separators=(',', ':'))}

Paper-grounded semantic blueprint geometry:
{json.dumps(compact_semantic_plan, ensure_ascii=False, separators=(',', ':'))}

{structure_section}

{visible_label_section}

{connector_section}

Boundary-role anti-hallucination rules:
{boundary_role_rules}

Mandatory panel-local instance checklist:
{chr(10).join(panel_instance_lines) if panel_instance_lines else '- No separate panel-local instance graph; use the global role checklist.'}

When the panel-local instance checklist is present, it is the authoritative visible occurrence model. Repeated labels are separate visible instances with different instance IDs. The global entity graph is only a compatibility summary and must not create extra cross-panel arrows or replace panel-local nodes.

Mandatory entity role and containment checklist:
{json.dumps(compact_roles, ensure_ascii=False, separators=(',', ':'))}

Panel-local supporting visual cues in the compact contract are evidence-backed planning metadata only. Use them as unlabeled internal texture or mechanism detail when helpful. Never render the cue wording itself, never turn the cues into a legend, never give them titles, never promote them to peer top-level nodes, and never invent connectors for them.

Preserve these semantic roles in the visible composition. A data source, modality, processing module, innovation, and output are not interchangeable even when all labels are present. In particular, do not flatten a source -> modality -> method chain into peer inputs or place a dataset beside modalities as though it were another sensor type.

Topology-specific rendering rules:
{topology_specific_rules}

Forbidden copied reference terms:
{json.dumps(template.get('forbidden_copy_terms', []), ensure_ascii=False)}

Hard requirements:
{overlay_requirements}
""".strip()


def _is_grounding_gap(text: str) -> bool:
    """Is this complaint about a missing citation rather than broken content?

    A plan whose modules exist but whose evidence is thin still describes the
    paper: it can be drawn, and the run can say exactly which objects went
    unsourced. Treating that like a missing module threw the whole figure away
    after a paid extraction, and the bulk corpus cannot afford to lose a paper
    it already understood.
    """
    lowered = str(text).casefold()
    return "evidence_ids" in lowered or "not verbatim evidence" in lowered


def validate_plan_grounding(plan: dict, parsed: dict) -> dict:
    evidence_items = [item for item in parsed.get("evidence", []) if isinstance(item, dict) and item.get("id")]
    valid_evidence = {item["id"] for item in evidence_items}
    evidence_text = {str(item["id"]): str(item.get("text") or "") for item in evidence_items}
    all_evidence_text = " ".join(evidence_text.values())
    errors: list[str] = []
    warnings: list[str] = []
    spec = plan.get("figure_specification", {})

    def normalized_term(value: str) -> str:
        return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).casefold()).strip()

    def script_counts(value: str) -> tuple[int, int]:
        cjk = sum(
            "\u3400" <= char <= "\u9fff"
            or "\u3040" <= char <= "\u30ff"
            or "\uac00" <= char <= "\ud7af"
            for char in value
        )
        latin = sum(("a" <= char.casefold() <= "z") for char in value)
        return cjk, latin

    def latin_language(value: str) -> str | None:
        folded = "".join(char for char in unicodedata.normalize("NFKD", value.casefold()) if not unicodedata.combining(char))
        words = re.findall(r"[a-z]{2,}", folded)
        profiles = {
            "english": {"the", "and", "that", "with", "from", "this", "we", "our", "for", "into", "method", "results"},
            "spanish": {"el", "la", "los", "las", "un", "una", "que", "con", "para", "por", "del", "este", "esta", "metodo", "resultados", "relaciones"},
            "french": {"le", "la", "les", "un", "une", "des", "que", "avec", "pour", "dans", "cette", "methode", "resultats", "relations"},
            "german": {"der", "die", "das", "den", "dem", "ein", "eine", "und", "mit", "fur", "von", "diese", "methode", "ergebnisse"},
            "portuguese": {"o", "os", "as", "um", "uma", "que", "com", "para", "por", "este", "esta", "metodo", "resultados", "relacoes"},
        }
        scores = {name: sum(word in markers for word in words) for name, markers in profiles.items()}
        language, score = max(scores.items(), key=lambda item: item[1])
        if language == "english" or score < 5 or score < scores["english"] + 2:
            return None
        return language

    detected_latin_language = latin_language(all_evidence_text)

    def english_label_ratio(value: str) -> float | None:
        tokens = re.findall(r"[A-Za-z]{3,}", value)
        if not tokens:
            return None
        try:
            import wordninja

            known_words = wordninja.DEFAULT_LANGUAGE_MODEL._wordcost
        except Exception:
            return None
        known = sum(token.casefold() in known_words or token.casefold() in {"encoder", "decoder", "embedding", "transformer"} for token in tokens)
        return known / len(tokens)

    def validate_visible_label(location: str, label: str, evidence_ids: list[str], *, fallback_text: str = "") -> None:
        value = str(label or "").strip()
        if not value:
            return
        source = " ".join(evidence_text.get(str(evidence_id), "") for evidence_id in evidence_ids).strip() or fallback_text
        if not source or normalized_term(value) in normalized_term(source):
            return
        label_cjk, label_latin = script_counts(value)
        source_cjk, source_latin = script_counts(source)
        # Only judge the writing system when the cited evidence is actually
        # dominated by the other script. A figure export can leave a couple of
        # mangled glyphs inside an otherwise English span, and those must not
        # read as a CJK source that an English label has overridden.
        source_script_total = max(1, source_cjk + source_latin)
        changed_from_cjk = source_cjk >= 2 and label_cjk == 0 and label_latin >= 2 and source_cjk / source_script_total >= 0.5
        changed_from_latin = source_latin >= 4 and source_cjk == 0 and label_cjk >= 1 and source_latin / source_script_total >= 0.5
        if changed_from_cjk or changed_from_latin:
            errors.append(f"{location} visible label changes the source writing system and is not verbatim evidence: {value}")
            return
        if detected_latin_language and normalized_term(value) not in normalized_term(all_evidence_text):
            ratio = english_label_ratio(value)
            if ratio is not None and ratio >= 0.8:
                errors.append(f"{location} visible label appears translated from {detected_latin_language} and is not verbatim evidence: {value}")

    modules = spec.get("modules") if isinstance(spec.get("modules"), list) else []
    if not modules:
        errors.append("figure_specification.modules is empty")
    module_ids: set[str] = set()
    for index, module in enumerate(modules):
        if not isinstance(module, dict):
            errors.append(f"module {index + 1} is not an object")
            continue
        module_id = str(module.get("id") or "").strip()
        if not module_id:
            errors.append(f"module {index + 1} has no id")
        elif module_id in module_ids:
            errors.append(f"duplicate module id: {module_id}")
        else:
            module_ids.add(module_id)
        evidence_ids = module.get("evidence_ids") if isinstance(module.get("evidence_ids"), list) else []
        if not evidence_ids:
            errors.append(f"module {module_id or index + 1} has no evidence_ids")
        invalid = [item for item in evidence_ids if item not in valid_evidence]
        if invalid:
            errors.append(f"module {module_id or index + 1} references unknown evidence ids: {invalid}")
        validate_visible_label(f"module {module_id or index + 1}", str(module.get("name") or module.get("visible_label") or ""), evidence_ids)

    endpoint_ids = set(module_ids)
    for field in ("inputs", "outputs", "innovations"):
        for index, item in enumerate(spec.get(field) if isinstance(spec.get(field), list) else []):
            if not isinstance(item, dict):
                continue
            endpoint_id = str(item.get("id") or item.get("name") or item.get("visible_label") or item.get("text") or "").strip()
            if endpoint_id:
                endpoint_ids.add(endpoint_id)
            evidence_ids = item.get("evidence_ids") if isinstance(item.get("evidence_ids"), list) else []
            if not evidence_ids:
                errors.append(f"{field}[{index}] has no evidence_ids")
            invalid = [value for value in evidence_ids if value not in valid_evidence]
            if invalid:
                errors.append(f"{field}[{index}] references unknown evidence ids: {invalid}")
            validate_visible_label(f"{field}[{index}]", str(item.get("name") or item.get("visible_label") or item.get("text") or ""), evidence_ids)

    for index, panel in enumerate(spec.get("overview_panels") if isinstance(spec.get("overview_panels"), list) else []):
        if not isinstance(panel, dict):
            continue
        evidence_ids = panel.get("evidence_ids") if isinstance(panel.get("evidence_ids"), list) else []
        if not evidence_ids:
            errors.append(f"overview_panels[{index}] has no evidence_ids")
        invalid = [value for value in evidence_ids if value not in valid_evidence]
        if invalid:
            errors.append(f"overview_panels[{index}] references unknown evidence ids: {invalid}")
        validate_visible_label(f"overview_panels[{index}]", str(panel.get("label") or ""), evidence_ids)

    relations = spec.get("relations") if isinstance(spec.get("relations"), list) else []
    for index, relation in enumerate(relations):
        if not isinstance(relation, dict):
            errors.append(f"relation {index + 1} is not an object")
            continue
        source = str(relation.get("source") or "").strip()
        target = str(relation.get("target") or "").strip()
        if source not in endpoint_ids or target not in endpoint_ids:
            errors.append(f"relation {index + 1} has unknown endpoint: {source} -> {target}")
        evidence_ids = relation.get("evidence_ids") if isinstance(relation.get("evidence_ids"), list) else []
        if not evidence_ids:
            errors.append(f"relation {source} -> {target} has no evidence_ids")
        invalid = [item for item in evidence_ids if item not in valid_evidence]
        if invalid:
            errors.append(f"relation {source} -> {target} references unknown evidence ids: {invalid}")
        validate_visible_label(f"relation {source} -> {target}", str(relation.get("label") or ""), evidence_ids)

    panel_graph_relation_count = 0
    panel_instance_ids: set[str] = set()
    for panel_index, panel in enumerate(spec.get("panel_graphs") if isinstance(spec.get("panel_graphs"), list) else []):
        if not isinstance(panel, dict):
            errors.append(f"panel_graphs[{panel_index}] is not an object")
            continue
        panel_evidence_ids = panel.get("evidence_ids") if isinstance(panel.get("evidence_ids"), list) else []
        invalid_panel_evidence = [value for value in panel_evidence_ids if value not in valid_evidence]
        if invalid_panel_evidence:
            errors.append(f"panel_graphs[{panel_index}] references unknown evidence ids: {invalid_panel_evidence}")
        local_ids: set[str] = set()
        for node_index, node in enumerate(panel.get("nodes") if isinstance(panel.get("nodes"), list) else []):
            if not isinstance(node, dict):
                errors.append(f"panel_graphs[{panel_index}].nodes[{node_index}] is not an object")
                continue
            instance_id = str(node.get("instance_id") or node.get("id") or "").strip()
            if not instance_id:
                errors.append(f"panel_graphs[{panel_index}].nodes[{node_index}] has no instance_id")
            elif instance_id in panel_instance_ids:
                errors.append(f"duplicate panel instance id: {instance_id}")
            else:
                panel_instance_ids.add(instance_id)
                local_ids.add(instance_id)
            node_evidence_ids = node.get("evidence_ids") if isinstance(node.get("evidence_ids"), list) else []
            if not node_evidence_ids:
                errors.append(f"panel instance {instance_id or node_index} has no evidence_ids")
            invalid_node_evidence = [value for value in node_evidence_ids if value not in valid_evidence]
            if invalid_node_evidence:
                errors.append(f"panel instance {instance_id or node_index} references unknown evidence ids: {invalid_node_evidence}")
            validate_visible_label(f"panel instance {instance_id or node_index}", _item_label_for_prompt(node), node_evidence_ids)
        for relation_index, panel_relation in enumerate(panel.get("relations") if isinstance(panel.get("relations"), list) else []):
            if not isinstance(panel_relation, dict):
                errors.append(f"panel_graphs[{panel_index}].relations[{relation_index}] is not an object")
                continue
            source = str(panel_relation.get("source") or "").strip()
            target = str(panel_relation.get("target") or "").strip()
            if source not in local_ids or target not in local_ids:
                errors.append(f"panel_graphs[{panel_index}] relation has non-local endpoint: {source} -> {target}")
            relation_evidence_ids = panel_relation.get("evidence_ids") if isinstance(panel_relation.get("evidence_ids"), list) else []
            if not relation_evidence_ids:
                errors.append(f"panel relation {source} -> {target} has no evidence_ids")
            invalid_relation_evidence = [value for value in relation_evidence_ids if value not in valid_evidence]
            if invalid_relation_evidence:
                errors.append(f"panel relation {source} -> {target} references unknown evidence ids: {invalid_relation_evidence}")
            panel_graph_relation_count += 1

    terminology = spec.get("terminology") if isinstance(spec.get("terminology"), dict) else {}
    for source_term, visible_label in terminology.items():
        validate_visible_label(f"terminology[{source_term}]", str(visible_label or ""), [], fallback_text=all_evidence_text)

    for field in ("must_show", "innovations"):
        items = spec.get(field) if isinstance(spec.get(field), list) else []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            evidence_ids = item.get("evidence_ids") if isinstance(item.get("evidence_ids"), list) else []
            invalid = [value for value in evidence_ids if value not in valid_evidence]
            if invalid:
                errors.append(f"{field}[{index}] references unknown evidence ids: {invalid}")

    for field in ("research_problem", "central_claim"):
        item = spec.get(field)
        if not isinstance(item, dict):
            errors.append(f"figure_specification.{field} is missing")
            continue
        text = str(item.get("text") or item.get("statement") or "").strip()
        evidence_ids = item.get("evidence_ids") if isinstance(item.get("evidence_ids"), list) else []
        explicitly_unknown = item.get("status") == "unknown" or text.casefold() == "unknown" or "require" in text.casefold() and "review" in text.casefold()
        if not explicitly_unknown and not evidence_ids:
            errors.append(f"figure_specification.{field} has no evidence_ids")
        invalid = [value for value in evidence_ids if value not in valid_evidence]
        if invalid:
            errors.append(f"figure_specification.{field} references unknown evidence ids: {invalid}")

    grounding_gaps = [item for item in errors if _is_grounding_gap(item)]
    errors = [item for item in errors if not _is_grounding_gap(item)]
    return {
        "summary": "Scientific grounding and relation-contract validation.",
        "ok": not errors,
        "module_count": len(modules),
        "relation_count": len(relations),
        "panel_graph_relation_count": panel_graph_relation_count,
        "evidence_count": len(valid_evidence),
        "errors": errors,
        "grounding_gaps": grounding_gaps,
        "grounding_gap_count": len(grounding_gaps),
        "warnings": warnings,
        "detected_source_language": detected_latin_language or "unknown_or_english",
    }


EVIDENCE_REPAIR_FIELDS = ("inputs", "outputs", "innovations")


def _slot_label(item: dict[str, Any]) -> str:
    return str(item.get("visible_label") or item.get("name") or item.get("text") or item.get("statement") or item.get("label") or item.get("id") or "").strip()


def _slot_name(item: dict[str, Any], index: int) -> str:
    return str(item.get("id") or item.get("name") or item.get("visible_label") or item.get("text") or (index + 1)).strip()


def collect_evidence_slots(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index every evidence-bearing plan object under a key a repair prompt can name.

    The keys are derived from the plan alone, so a repair answer that only names
    keys and evidence ids can be applied back onto the very objects it cites.
    """
    slots: dict[str, dict[str, Any]] = {}
    if not isinstance(spec, dict):
        return slots

    def register(key: str, item: dict[str, Any], kind: str) -> None:
        if key in slots:
            return
        existing = item.get("evidence_ids") if isinstance(item.get("evidence_ids"), list) else []
        slots[key] = {"kind": kind, "label": _slot_label(item), "object": item, "evidence_ids": [str(value) for value in existing]}

    modules = spec.get("modules") if isinstance(spec.get("modules"), list) else []
    for index, module in enumerate(modules):
        if isinstance(module, dict):
            register("module:" + (str(module.get("id") or "").strip() or str(index + 1)), module, "module")
    for field in EVIDENCE_REPAIR_FIELDS:
        items = spec.get(field) if isinstance(spec.get(field), list) else []
        for index, item in enumerate(items):
            if isinstance(item, dict):
                register(field + ":" + _slot_name(item, index), item, field)
    panels = spec.get("overview_panels") if isinstance(spec.get("overview_panels"), list) else []
    for index, panel in enumerate(panels):
        if isinstance(panel, dict):
            register("overview_panel:" + str(index), panel, "overview_panel")
    relations = spec.get("relations") if isinstance(spec.get("relations"), list) else []
    for relation in relations:
        if isinstance(relation, dict):
            register("relation:" + str(relation.get("source") or "") + "->" + str(relation.get("target") or ""), relation, "relation")
    graphs = spec.get("panel_graphs") if isinstance(spec.get("panel_graphs"), list) else []
    for panel_index, panel in enumerate(graphs):
        if not isinstance(panel, dict):
            continue
        panel_id = str(panel.get("id") or "panel_" + str(panel_index))
        nodes = panel.get("nodes") if isinstance(panel.get("nodes"), list) else []
        for node in nodes:
            if isinstance(node, dict):
                instance_id = str(node.get("instance_id") or node.get("id") or "").strip()
                if instance_id:
                    register("panel_instance:" + instance_id, node, "panel_instance")
        panel_relations = panel.get("relations") if isinstance(panel.get("relations"), list) else []
        for relation in panel_relations:
            if isinstance(relation, dict):
                register("panel_relation:" + panel_id + ":" + str(relation.get("source") or "") + "->" + str(relation.get("target") or ""), relation, "panel_relation")
    return slots


def ungrounded_evidence_slots(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """List the plan objects that still owe an evidence citation, in plan order."""
    slots = collect_evidence_slots(spec)
    return [{name: value for name, value in slot.items() if name != "object"} | {"key": key} for key, slot in slots.items() if not slot["evidence_ids"]]


def apply_evidence_fixes(spec: dict[str, Any], fixes: Any, *, valid_evidence: set[str]) -> dict[str, Any]:
    """Fill still-empty citations from a repair answer, and nothing else.

    A repair may only name an existing key and cite evidence ids the paper
    really contains. It cannot rename, reorder, drop or restructure anything,
    because the plan objects themselves are already in place.
    """
    slots = collect_evidence_slots(spec)
    # A repair answer may quote a key with the same stray whitespace the plan
    # arrived with, so match on the whitespace-free form of the key.
    lookup = {_without_whitespace(key): key for key in slots}
    applied: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for item in fixes if isinstance(fixes, list) else []:
        if not isinstance(item, dict):
            continue
        key = _without_whitespace(str(item.get("key") or ""))
        key = lookup.get(key, key)
        slot = slots.get(key)
        if slot is None:
            rejected.append({"key": key, "reason": "unknown slot"})
            continue
        if slot["evidence_ids"]:
            rejected.append({"key": key, "reason": "already cited"})
            continue
        raw = item.get("evidence_ids") if isinstance(item.get("evidence_ids"), list) else []
        cited = list(dict.fromkeys(_without_whitespace(str(value)) for value in raw if str(value) in valid_evidence))
        if not cited:
            rejected.append({"key": key, "reason": "no known evidence id"})
            continue
        slot["object"]["evidence_ids"] = cited
        applied.append({"key": key, "evidence_ids": cited})
    return {"applied": applied, "rejected": rejected}


def _evidence_repair_prompt(slots: list[dict[str, Any]], errors: list[str], parsed: dict, evidence_max_chars: int) -> str:
    slot_lines = "\n".join("- " + str(slot["key"]) + " | " + str(slot["kind"]) + " | " + (str(slot["label"]) or "(no visible label)") for slot in slots)
    error_lines = "\n".join("- " + str(value) for value in errors) or "- none reported"
    return (
        "The figure plan returned for this paper is missing evidence citations on the objects listed below. Fix only the citations.\n\n"
        "Validator errors:\n" + error_lines + "\n\n"
        "Objects that still owe a citation:\n" + slot_lines + "\n\n"
        "Rules:\n"
        '1. Return JSON only, exactly this shape: {"evidence_fixes":[{"key":"...","evidence_ids":["E0001"]}]}.' + "\n"
        "2. Cite only evidence ids that appear in the evidence blocks below. Never invent, renumber or guess an id.\n"
        "3. Cite the evidence that actually states that exact visible label and its role. Leave a key out when nothing supports it.\n"
        "4. You cannot rename, merge, split, reorder or drop anything. Only these missing citations may change.\n\n"
        "Evidence blocks:\n" + compact_evidence(parsed, max_chars=evidence_max_chars)
    )


def repair_plan_evidence(plan: dict[str, Any], parsed: dict, errors: list[str], *, model: str | None = None, timeout_seconds: int = 90, retries: int = 0, evidence_max_chars: int = 20000, deadline_at: float | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Ask the planner once to cite the objects its own plan left ungrounded.

    The repair answer is applied onto the existing plan objects, so the science
    the model already committed to cannot change; only missing citations are
    added, and only with evidence ids the paper really contains.
    """
    spec = plan.get("figure_specification") if isinstance(plan.get("figure_specification"), dict) else {}
    slots = ungrounded_evidence_slots(spec)
    metadata: dict[str, Any] = {"requested_mode": "vlm", "mode": "skipped", "model": None, "warning": None, "prompt": "", "provider": {}, "ungrounded_slots": len(slots), "applied": [], "rejected": []}
    if not slots:
        return plan, metadata
    if not vlm_credentials_available():
        metadata.update({"mode": "no_credentials", "warning": "API_BASE and API_KEY/GEMINI_API_KEY are required for evidence repair"})
        return plan, metadata
    valid_evidence = {str(item.get("id")) for item in parsed.get("evidence", []) if isinstance(item, dict) and item.get("id")}
    prompt = _evidence_repair_prompt(slots, list(errors or []), parsed, evidence_max_chars)
    metadata["prompt"] = prompt
    resolved = resolve_vlm_model("RFS_FAST_FRAMEWORK_MODEL", "RFS_PAPER_TO_IMAGE_MODEL", "RFS_PAPER_PLANNER_MODEL", explicit_model=model)
    try:
        raw = call_vlm_json(prompt, [], model=resolved, timeout=max(10, int(timeout_seconds)), retries=max(0, int(retries)), call_metadata=metadata["provider"], deadline_at=deadline_at)
    except Exception as exc:
        metadata.update({"mode": "failed", "model": resolved, "warning": str(exc)})
        return plan, metadata
    fixes = raw.get("evidence_fixes") if isinstance(raw, dict) else None
    outcome = apply_evidence_fixes(spec, fixes, valid_evidence=valid_evidence)
    # Kept so a run that cited nothing can be diagnosed afterwards instead of
    # being guessed at: without the answer nobody can tell an empty reply from
    # a reply we failed to read.
    metadata.update(
        {
            "mode": "vlm",
            "model": resolved,
            "answer_keys": sorted(str(key) for key in raw) if isinstance(raw, dict) else [],
            "fixes_returned": len(fixes) if isinstance(fixes, list) else 0,
            "raw_response": raw if isinstance(raw, (dict, list)) else str(raw)[:2000],
            "applied": outcome["applied"],
            "rejected": outcome["rejected"],
        }
    )
    return plan, metadata
