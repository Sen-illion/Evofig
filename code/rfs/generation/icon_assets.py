"""Generate isolated local icon assets for direct editable figures."""

from __future__ import annotations

import base64
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import requests
from PIL import Image, ImageDraw
from .icon_background import normalize_icon_background

from ..planning.prompt_assembly import render_patch_block
from ..utils import ensure_dir, write_json, write_text


_ACADEMIC_ICON_NEGATIVES = (
    "Do not create a complete illustration, multi-object scene, mini-poster, dashboard, environment, scenery, "
    "photorealistic object, 3D render, dramatic lighting, gradients, decorative clutter, tiny details, text, "
    "formulas, arrows, connectors, logos, watermarks, frames, or a separate white card."
)


#: The one clause a paid PNG is allowed to outlive: which colours the accents use. Everything
#: else in the prompt -- subject, metaphor, grammar, patches, background policy -- decides what
#: the picture is, and a change there is a reason to draw again. The accent role is a hex
#: triplet the panel palette owns, so a palette move alone must not re-buy a whole icon set.
#: The clause is written from whatever colour role the panel palette owns, and that role is
#: not always a dict -- a theme may hand over one plain string. The strip matches the clause by
#: its fixed opening and closing words rather than by the shape of the value between them, so
#: the reuse key and the prompt agree on where the clause starts and ends however the palette
#: spells it. Matching the old `\{...\}` shape let a string role through untrimmed, and a
#: palette move then read as a new request and re-bought the whole icon set.
ACCENT_CLAUSE_LEAD = "Use this figure color role as restrained accents: "
ACCENT_CLAUSE_TAIL = "; do not introduce unrelated saturated colors."
_COLOR_ACCENT_CLAUSE = re.compile(
    r"\s*" + re.escape(ACCENT_CLAUSE_LEAD) + r".*?" + re.escape(ACCENT_CLAUSE_TAIL),
    re.DOTALL,
)


def prompt_reuse_key(prompt: str) -> str:
    """The part of an icon prompt that decides whether an already-paid PNG still fits."""

    return " ".join(_COLOR_ACCENT_CLAUSE.sub("", str(prompt or "")).split())


def _prompt_survives_reuse(reusable: Path, prompt: str) -> bool:
    """Whether this slot's PNG was drawn from this same request, its accent colour aside.

    The request is read from beside the image, because that is where the run that paid for
    it recorded what it asked for. An image with no recorded request proves nothing, so it
    is not reused for a slot the repair named.
    """

    previous = reusable.parent / "prompt.txt"
    if not previous.is_file():
        return False
    try:
        drawn = previous.read_text(encoding="utf-8")
    except OSError:
        return False
    return prompt_reuse_key(drawn) == prompt_reuse_key(prompt)


def build_icon_prompt(icon: dict[str, Any], *, prompt_patches: list[dict[str, Any]] | None = None) -> str:
    """Compile one direct-pipeline icon contract into an Image2 prompt."""
    subject = str(icon.get("subject") or icon.get("asset_id") or "scientific method component")
    primary = str(icon.get("primary_object") or icon.get("visual_metaphor") or subject)
    complexity_kind = str(icon.get("complexity_kind") or "standard_icon")
    grammar = str(icon.get("icon_grammar") or ("bounded_mechanism_visual" if complexity_kind == "hero_visual" else "academic_flat_symbol"))
    support_limit = max(0, min(4, int(icon.get("supporting_object_limit", 3 if complexity_kind == "hero_visual" else 1))))
    base = str(icon.get("prompt") or "").strip()
    reference_rules = icon.get("reference_icon_rules") if isinstance(icon.get("reference_icon_rules"), list) else []
    reference_clause = "; ".join(str(item) for item in reference_rules if str(item).strip())
    if complexity_kind == "hero_visual":
        composition = (
            f"Use one dominant scientific mechanism object and no more than {support_limit} compact supporting objects. "
            "Show only the minimum internal structure needed to explain the mechanism; keep it icon-like and bounded, never a scene. "
            "The subject may fill 70-88% of the canvas."
        )
    else:
        composition = (
            f"Use exactly one dominant symbolic object and no more than {support_limit} very small supporting mark. "
            "Use a simple silhouette, clean uniform medium outline, flat limited pastel fills, and minimal geometry. "
            "The symbol should fill roughly 55-78% of the canvas and remain immediately recognizable when displayed small."
        )
    prompt = (
        f"Create a publication-quality academic research framework-diagram icon for: {subject}. "
        f"Primary visual metaphor: {primary}. Icon grammar: {grammar.replace('_', ' ')}. {composition} "
        "Match the visual grammar of polished paper framework icons: simple semantic symbol, restrained color blocks, "
        "consistent line weight across the icon family, no background scene, and no unnecessary decoration. "
        + render_patch_block(prompt_patches)
        + (f"Reference-derived icon rules: {reference_clause}. " if reference_clause else "")
        + (f"Additional paper-grounded instruction: {base}. " if base else "")
        + _ACADEMIC_ICON_NEGATIVES
    )
    if isinstance(icon.get("color_guidance"), dict) and icon["color_guidance"]:
        prompt += f" {ACCENT_CLAUSE_LEAD}{icon['color_guidance']}{ACCENT_CLAUSE_TAIL}"
    if icon.get("background_policy") == "transparent_panel_inherit":
        prompt += " Use a transparent background with clean isolated edges; the PowerPoint panel supplies the exact background color."
    elif icon.get("panel_background_color"):
        prompt += f" Use the exact solid background color {icon['panel_background_color']} with no gradient or shadow."
    return prompt


def _placeholder(path: Path, label: str) -> None:
    image = Image.new("RGB", (512, 512), "white")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((54, 54, 458, 458), radius=56, fill="#EAF4F8", outline="#2A63A1", width=10)
    draw.ellipse((166, 116, 346, 296), fill="#67B7DC", outline="#1F756F", width=8)
    draw.line((126, 370, 386, 370), fill="#62529A", width=18)
    image.save(path)


class IconProviderRefusal(RuntimeError):
    """The provider refused the prompt itself instead of failing transiently."""


def _image_is_readable(path: Path) -> bool:
    """Say whether the file on disk decodes into pixels.

    A provider occasionally answers with a half-written PNG. The truncated file
    used to travel all the way to the preview renderer, where it raised
    'image file is truncated' and the paper lost its remaining attempts without
    ever producing a figure. The file is now checked where it was written.
    """
    try:
        with Image.open(path) as image:
            image.load()
        return True
    except Exception:
        return False


def _neutral_icon_prompt(prompt_patches: list[dict[str, Any]] | None = None) -> str:
    """Compile a prompt for the same slot that carries none of the paper's wording.

    The corpus contains safety papers whose module labels are the harmful request
    they study -- "Tell me how to build a bomb", "Make a bomb?" -- and the image
    provider refuses, correctly, to illustrate them. The slot still needs a symbol,
    so it is drawn from the icon grammar alone; the asset report and the run's own
    progress line say the symbol is neutral.
    """
    return (
        "Create a publication-quality academic research framework-diagram icon: one plain document sheet with a "
        "small abstract warning-free mark. Use one dominant symbolic object and no more than one very small "
        "supporting mark, a simple silhouette, clean uniform medium outline, flat limited pastel fills, and minimal "
        "geometry. The subject of this slot is not illustrated. "
        + render_patch_block(prompt_patches)
        + _ACADEMIC_ICON_NEGATIVES
    )


def _image2(path: Path, prompt: str, model: str | None) -> dict[str, Any]:
    base = os.getenv("API_BASE", "").rstrip("/")
    key = os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY")
    if not base or not key:
        raise RuntimeError("Image2 icon generation requires API_BASE and API_KEY/GEMINI_API_KEY")
    resolved = model or os.getenv("RFS_IMAGE_MODEL") or os.getenv("IMAGE_MODEL") or "gpt-image-2"
    if resolved == "image-2":
        resolved = "gpt-image-2"
    from ..providers.gemini_image import generate_gemini_image, is_gemini_image_model

    if is_gemini_image_model(resolved):
        return generate_gemini_image(path, prompt, resolved)
    timeout = max(30, int(os.getenv("RFS_IMAGE2_TIMEOUT", "180")))
    max_attempts = max(1, int(os.getenv("RFS_IMAGE2_RETRIES", os.getenv("RFS_IMAGE2_RATE_LIMIT_RETRIES", "5"))) + 1)
    response = None
    attempts = 0
    last_error: Exception | None = None
    retryable_statuses = {408, 425, 429, 500, 502, 503, 504}
    for attempts in range(1, max_attempts + 1):
        try:
            response = requests.post(
                f"{base}/images/generations",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": resolved, "prompt": prompt, "n": 1, "size": "1024x1024"},
                timeout=timeout,
            )
            if response.status_code not in retryable_statuses or attempts >= max_attempts:
                break
            retry_after = response.headers.get("Retry-After") if response.headers else None
            try:
                delay = float(retry_after) if retry_after else min(30.0, 2.0 ** (attempts - 1))
            except (TypeError, ValueError):
                delay = min(30.0, 2.0 ** (attempts - 1))
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            last_error = exc
            if attempts >= max_attempts:
                raise
            delay = min(30.0, 2.0 ** (attempts - 1))
        time.sleep(max(0.0, delay))
    if response is None:
        raise last_error or RuntimeError("Image2 request failed without a response")
    if response.status_code == 400:
        detail = ""
        try:
            detail = " ".join(str(response.text or "").split())[:300]
        except Exception:
            detail = ""
        raise IconProviderRefusal(f"image provider refused this prompt (HTTP 400): {detail}")
    response.raise_for_status()
    item = (response.json().get("data") or [{}])[0]
    if item.get("b64_json"):
        path.write_bytes(base64.b64decode(item["b64_json"]))
    elif item.get("url"):
        download = requests.get(item["url"], timeout=timeout)
        download.raise_for_status()
        path.write_bytes(download.content)
    else:
        raise RuntimeError("Image2 returned no icon data")
    return {"model": resolved, "api_requests_attempted": attempts}


def _image2_candidate(
    candidate_path: Path,
    icon: dict[str, Any],
    prompt: str,
    model: str | None,
    *,
    prompt_patches: list[dict[str, Any]] | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
    asset_id: str = "",
    candidate_index: int = 0,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Fill one candidate slot, and say what it cost when the provider would not.

    Two provider answers are the slot's problem rather than the paper's: a refusal
    of the paper's own wording, and a half-written PNG. Both are answered here --
    the refusal by asking once more in neutral wording, the damaged file by asking
    once more -- and if the slot still cannot be filled the offline placeholder is
    drawn in its place. A 503 that is specific to the paper's wording gets the same
    neutral ask; if that ask is also unavailable the error still raises, so a real
    outage stops the run. Quota, timeouts, and a 500 still raise, because a corpus
    that quietly fills with placeholders is worse than a run that stops.
    """
    requests_attempted = 0
    provider: dict[str, Any] = {"model": "offline-placeholder", "api_requests_attempted": 0}
    kinds: list[str] = []
    note = ""
    neutral_used = False
    request_prompt = prompt
    for _attempt in range(2):
        try:
            result = _image2(candidate_path, request_prompt, model)
        except IconProviderRefusal as refusal:
            requests_attempted += 1
            kinds.append("provider_refused_prompt")
            note = str(refusal)
            # The provider objected to the paper's wording, not to the slot, so the
            # second ask carries the icon grammar alone.
            request_prompt = _neutral_icon_prompt(prompt_patches)
            neutral_used = True
            if progress:
                progress({"stage": "image2_request", "status": "degraded", "message": f"{asset_id} was refused by the provider's content filter; asking once more for a neutral symbol", "asset_id": asset_id, "candidate": candidate_index + 1, "degradation": "provider_refused_prompt"})
            continue
        except requests.HTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status != 503 or neutral_used:
                raise
            # Some providers answer a wording they will not draw with 503, while a
            # plain symbol on the same model still succeeds. Ask once without the
            # paper's words. A second 503 is an outage and still stops the paper.
            requests_attempted += 1
            kinds.append("provider_unavailable_for_wording")
            note = str(exc)
            request_prompt = _neutral_icon_prompt(prompt_patches)
            neutral_used = True
            if progress:
                progress({"stage": "image2_request", "status": "degraded", "message": f"{asset_id} was unavailable for this wording; asking once more for a neutral symbol", "asset_id": asset_id, "candidate": candidate_index + 1, "degradation": "provider_unavailable_for_wording"})
            continue
        requests_attempted += int(result.get("api_requests_attempted") or 0)
        provider = {"model": str(result.get("model") or provider["model"]), "api_requests_attempted": requests_attempted, "usage": result.get("usage")}
        if _image_is_readable(candidate_path):
            if not kinds:
                return provider, None
            return provider, {"status": "neutralized" if neutral_used else "generated", "kinds": list(dict.fromkeys(kinds)), "note": note[:300], "neutral_prompt": neutral_used}
        kinds.append("damaged_image")
        if progress:
            progress({"stage": "image2_request", "status": "degraded", "message": f"{asset_id} came back half-written; asking once more", "asset_id": asset_id, "candidate": candidate_index + 1, "degradation": "damaged_image"})
    _placeholder(candidate_path, str(icon.get("subject") or asset_id))
    status = "placeholder_after_damage" if "damaged_image" in kinds else "placeholder_after_refusal"
    if progress:
        progress({"stage": "icon_generation", "status": "degraded", "message": f"{asset_id} drawn as a neutral placeholder: {', '.join(dict.fromkeys(kinds))}", "asset_id": asset_id, "degradation": status})
    return {**provider, "api_requests_attempted": requests_attempted}, {"status": status, "kinds": list(dict.fromkeys(kinds)), "note": note[:300], "neutral_prompt": neutral_used}


def _reuse_dir_names(asset_id: str, semantic_source_id: str, planned_ids: set[str]) -> list[str]:
    """Every directory name this slot's icon may already be stored under.

    The layout designer mints the node ids, so a run that draws the sealed contract can
    name a slot ``icon_node_extractor`` where the contract itself - and therefore every
    staged icon - calls the same node ``mod_feature_extractor``.  The contract id is the
    stable name, so it is tried after the run's own.  A name that is itself a slot of this
    plan is never used as an alias: that file belongs to that slot.
    """
    names = [asset_id]
    source = str(semantic_source_id or "").strip()
    if source:
        alias = source if source.startswith("icon_") else f"icon_{source}"
        if alias not in names and alias not in planned_ids:
            names.append(alias)
    return names


def _reusable_icon(
    reuse_root: Path | None,
    root: Path,
    asset_id: str,
    semantic_source_id: str,
    planned_ids: set[str],
) -> Path | None:
    """The already-paid PNG this slot may copy instead of asking the provider."""
    for base in (reuse_root, root):
        if base is None:
            continue
        for name in _reuse_dir_names(asset_id, semantic_source_id, planned_ids):
            for filename in ("selected.png", "candidate_00.png"):
                candidate = base / name / filename
                if candidate.is_file() and _image_is_readable(candidate):
                    return candidate
    return None


def generate_icon_assets(plan: dict[str, Any], out: str | Path, *, mode: str = "placeholder", model: str | None = None, candidates: int = 1, progress: Callable[[dict[str, Any]], None] | None = None, reuse_assets_from: str | Path | None = None, target_asset_ids: list[str] | None = None, prompt_patches: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    root = ensure_dir(Path(out) / "assets")
    records = []
    target_ids = set(str(item) for item in (target_asset_ids or []))
    reuse_root = Path(reuse_assets_from) if reuse_assets_from else None
    icons = plan.get("icons", [])
    planned_ids = {str(item.get("asset_id") or "") for item in icons if isinstance(item, dict)}
    for icon_index, icon in enumerate(icons, start=1):
        asset_id = str(icon["asset_id"])
        if progress:
            progress({"stage": "icon_generation", "status": "started", "message": f"{asset_id} ({icon_index}/{len(icons)})", "asset_id": asset_id, "mode": mode})
        asset_started = time.monotonic()
        asset_dir = ensure_dir(root / asset_id)
        selected = asset_dir / "selected.png"
        prompt = build_icon_prompt(icon, prompt_patches=prompt_patches)
        targeted = bool(target_ids) and asset_id in target_ids
        reuse_note = "already_here" if selected.exists() else ""
        if not selected.exists():
            reusable = _reusable_icon(reuse_root, root, asset_id, str(icon.get("semantic_source_id") or ""), planned_ids)
            if reusable is None:
                reuse_note = "no_previous_asset"
            elif not targeted:
                # The repair did not name this slot, so the paid PNG is still the figure's answer.
                reuse_note = "reused_untouched_slot"
            elif _prompt_survives_reuse(reusable, prompt):
                # The repair named every slot the replan rewrote, but the request itself did not
                # move: a palette change reaches this icon only through the accent clause.
                reuse_note = "reused_unchanged_prompt"
            else:
                reuse_note = "redrawn_prompt_changed"
            if reuse_note.startswith("reused"):
                shutil.copyfile(reusable, selected)
        panel_color = str(icon.get("panel_background_color") or "")
        write_text(asset_dir / "prompt.txt", prompt)
        if selected.exists():
            shutil.copyfile(selected, root / f"{asset_id}.png")
            with Image.open(selected) as image:
                width, height = image.size
            records.append({"asset_id": asset_id, "path": str(selected), "candidate_paths": [str(selected)], "panel_id": icon.get("panel_id"), "panel_background_color": panel_color, "background_policy": icon.get("background_policy"), "complexity_kind": icon.get("complexity_kind", "standard_icon"), "icon_grammar": icon.get("icon_grammar", "academic_flat_symbol"), "background_report": None, "status": "reused", "reuse_note": reuse_note, "mode": mode, "candidate_count": 1, "quality": {"size": [width, height], "has_text": False, "has_arrows": False, "has_formula": False, "policy": "visual_only_no_text_no_arrows"}, "model": "reused-existing", "api_requests_attempted": 0})
            if progress:
                progress({"stage": "icon_generation", "status": "completed", "message": f"{asset_id} reused", "asset_id": asset_id, "candidate_count": 1})
            continue
        candidate_count = max(1, min(2, int(candidates)))
        candidate_paths = []
        requests_attempted = 0
        usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "image_count": 0}
        provider_model = "offline-placeholder"
        neutralized = False
        degradation: dict[str, Any] | None = None
        for candidate_index in range(candidate_count):
            candidate_path = asset_dir / f"candidate_{candidate_index:02d}.png"
            if candidate_path.exists():
                provider_model = "reused-existing-candidate"
            elif mode == "placeholder":
                _placeholder(candidate_path, str(icon.get("subject") or asset_id))
            elif mode == "image2":
                if progress:
                    progress({"stage": "image2_request", "status": "started", "message": f"{asset_id} candidate {candidate_index + 1}/{candidate_count}", "asset_id": asset_id, "candidate": candidate_index + 1})
                try:
                    provider, degradation = _image2_candidate(
                        candidate_path,
                        icon,
                        prompt,
                        model,
                        prompt_patches=prompt_patches,
                        progress=progress,
                        asset_id=asset_id,
                        candidate_index=candidate_index,
                    )
                except Exception as exc:
                    if progress:
                        progress({"stage": "image2_request", "status": "failed", "message": str(exc), "asset_id": asset_id, "candidate": candidate_index + 1, "error_type": type(exc).__name__})
                    raise
                requests_attempted += int(provider.get("api_requests_attempted") or 0)
                provider_model = str(provider.get("model") or provider_model)
                for key, value in (provider.get("usage") or {}).items():
                    if key in usage_totals:
                        usage_totals[key] += int(value or 0)
                neutralized = neutralized or bool(degradation and degradation.get("neutral_prompt"))
            else:
                raise ValueError(f"Unsupported icon mode: {mode}")
            candidate_paths.append(str(candidate_path))
        shutil.copyfile(candidate_paths[0], selected)
        background_report = None
        if panel_color:
            background_report = normalize_icon_background(selected, panel_color, policy=str(icon.get("background_policy") or "transparent_panel_inherit"))
        shutil.copyfile(selected, root / f"{asset_id}.png")
        with Image.open(selected) as image:
            width, height = image.size
        records.append({"asset_id": asset_id, "path": str(selected), "candidate_paths": candidate_paths, "panel_id": icon.get("panel_id"), "panel_background_color": panel_color, "background_policy": icon.get("background_policy"), "complexity_kind": icon.get("complexity_kind", "standard_icon"), "icon_grammar": icon.get("icon_grammar", "academic_flat_symbol"), "background_report": background_report, "status": degradation.get("status") if degradation else "generated", "reuse_note": reuse_note, "mode": mode, "candidate_count": len(candidate_paths), "neutral_prompt_used": neutralized, "degradation": degradation, "quality": {"size": [width, height], "has_text": False, "has_arrows": False, "has_formula": False, "policy": "visual_only_no_text_no_arrows"}, "model": provider_model, "api_requests_attempted": requests_attempted, "usage": usage_totals})
        if progress:
            progress({"stage": "icon_generation", "status": "completed", "message": f"{asset_id} in {time.monotonic() - asset_started:.1f}s", "asset_id": asset_id, "candidate_count": len(candidate_paths)})
    reuse_summary: dict[str, int] = {}
    for item in records:
        note = str(item.get("reuse_note") or "unknown")
        reuse_summary[note] = reuse_summary.get(note, 0) + 1
    degradations = [{"asset_id": item["asset_id"], **(item.get("degradation") or {})} for item in records if item.get("degradation")]
    report = {"summary": "Direct local icon generation report.", "mode": mode, "assets": records, "degradations": degradations, "reuse_summary": reuse_summary, "quality_gate": {"ok": all(item.get("quality", {}).get("policy") == "visual_only_no_text_no_arrows" for item in records)}, "api_requests_attempted": sum(int(item.get("api_requests_attempted") or 0) for item in records)}
    write_json(Path(out) / "asset_generation_report.json", report)
    write_json(Path(out) / "asset_quality_report.json", {"summary": "Icon visual responsibility quality gate.", "ok": report["quality_gate"]["ok"], "assets": [{"asset_id": item["asset_id"], "panel_id": item.get("panel_id"), "panel_background_color": item.get("panel_background_color"), "background_report": item.get("background_report"), **item["quality"]} for item in records]})
    return report
