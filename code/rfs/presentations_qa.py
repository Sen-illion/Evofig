from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageStat

from .utils import ensure_dir, write_json, write_text


def _default_node_path() -> str:
    explicit = os.getenv("RFS_PRESENTATIONS_NODE", "").strip()
    if explicit:
        return explicit
    home = Path(os.getenv("USERPROFILE") or os.getenv("HOME") or str(Path.home()))
    bundled = home / ".cache" / "codex-runtimes" / "codex-primary-runtime" / "dependencies" / "node" / "bin" / "node.exe"
    if bundled.exists():
        return str(bundled)
    return shutil.which("node") or "node"


def _default_presentations_script() -> str | None:
    explicit = os.getenv("RFS_PRESENTATIONS_INSPECT_SCRIPT", "").strip()
    if explicit:
        return explicit
    home = Path(os.getenv("USERPROFILE") or os.getenv("HOME") or str(Path.home()))
    root = home / ".codex" / "plugins" / "cache" / "openai-primary-runtime" / "presentations"
    if not root.exists():
        return None
    candidates = sorted(root.glob("*/skills/presentations/scripts/inspect_template_deck.mjs"), key=lambda p: str(p), reverse=True)
    return str(candidates[0]) if candidates else None


def _read_json_lenient(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="ignore")
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        result: dict[str, Any] = {"parse_status": "failed", "raw_length": len(text)}
        slide_match = re.search(r'"slideCount"\s*:\s*(\d+)', text)
        media_match = re.search(r'"mediaCount"\s*:\s*(\d+)', text)
        if slide_match:
            result["slideCount"] = int(slide_match.group(1))
        if media_match:
            result["packageParts"] = {"mediaCount": int(media_match.group(1))}
        return result


def _pptx_object_counts(pptx_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "pptx": str(pptx_path),
        "exists": pptx_path.exists(),
        "size": pptx_path.stat().st_size if pptx_path.exists() else 0,
    }
    if not pptx_path.exists():
        result["error"] = "pptx_missing"
        return result
    try:
        with zipfile.ZipFile(pptx_path) as zf:
            names = zf.namelist()
            slide_names = [n for n in names if n.startswith("ppt/slides/slide") and n.endswith(".xml")]
            result["media_files"] = sum(1 for n in names if n.startswith("ppt/media/"))
            result["slides"] = len(slide_names)
            if slide_names:
                slide_xml = zf.read(slide_names[0]).decode("utf-8", errors="ignore")
                result["pictures"] = slide_xml.count("<p:pic")
                result["connectors"] = slide_xml.count("<p:cxnSp")
                result["text_bodies"] = slide_xml.count("<p:txBody")
                result["shapes"] = slide_xml.count("<p:sp")
    except Exception as exc:
        result["error"] = str(exc)
    return result


def _find_final_png(out_dir: Path) -> Path | None:
    for name in ["final_600dpi.png", "final_600dpi_reference_text_v3.png"]:
        path = out_dir / name
        if path.exists():
            return path
    candidates = sorted(out_dir.glob("final_*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _image_diff(rfs_png: Path | None, presentations_preview: Path | None) -> dict[str, Any] | None:
    if not rfs_png or not presentations_preview or not rfs_png.exists() or not presentations_preview.exists():
        return None
    try:
        img1 = Image.open(rfs_png).convert("RGB")
        img2 = Image.open(presentations_preview).convert("RGB")
        img2_resized = img2.resize(img1.size) if img1.size != img2.size else img2
        diff = ImageChops.difference(img1, img2_resized)
        stat = ImageStat.Stat(diff)
        rms = (sum(v * v for v in stat.rms) / 3) ** 0.5
        pix = diff.load()
        width, height = diff.size
        changed = 0
        for y in range(height):
            for x in range(width):
                if max(pix[x, y]) > 16:
                    changed += 1
        return {
            "rfs_export": str(rfs_png),
            "presentations_preview": str(presentations_preview),
            "rfs_size": list(img1.size),
            "presentations_size": list(img2.size),
            "rms_0_255": round(float(rms), 3),
            "changed_pixel_percent_threshold16": round(changed * 100 / (width * height), 3),
            "likely_causes": ["renderer difference", "font substitution", "connector fallback path rendering"],
        }
    except Exception as exc:
        return {"error": str(exc), "rfs_export": str(rfs_png), "presentations_preview": str(presentations_preview)}


def _write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    outputs = report.get("deliverables", {})
    plugin = report.get("presentations_plugin_qa", {})
    counts = report.get("editable_object_counts", {})
    diff = report.get("image_diff") or {}
    lines = [
        "# Summary",
        "RFS remains the primary PPTX compiler. The Presentations plugin is used only for optional import/render/layout QA.",
        "",
        "## Outputs",
        f"- Editable PPTX: `{outputs.get('editable_pptx')}`",
        f"- PNG: `{outputs.get('png')}`",
        f"- PDF: `{outputs.get('pdf')}`",
        f"- Presentations preview: `{plugin.get('preview')}`",
        "",
        "## Editable Structure",
        f"- Pictures: {counts.get('pictures')}",
        f"- Connectors: {counts.get('connectors')}",
        f"- Text bodies: {counts.get('text_bodies')}",
        f"- Media files: {counts.get('media_files')}",
        "",
        "## Presentations QA",
        f"- Status: {plugin.get('status')}",
        f"- Workspace: `{plugin.get('workspace')}`",
        f"- Connector fallback warnings: {plugin.get('autoRouteConnectorPx_failed_count')}",
        "",
        "## Interpretation",
        str(plugin.get("interpretation") or "Use Presentations for QA only; keep RFS as the source of truth for connector-heavy research figures."),
    ]
    if diff:
        lines.extend([
            "",
            "## Render Difference",
            f"- RFS export size: {diff.get('rfs_size')}",
            f"- Presentations preview size: {diff.get('presentations_size')}",
            f"- RMS pixel difference: {diff.get('rms_0_255')}",
            f"- Changed pixel percent: {diff.get('changed_pixel_percent_threshold16')}",
        ])
    write_text(path, "\n".join(lines) + "\n")


def run_presentations_qa(
    out_dir: str | Path,
    pptx: str | Path | None = None,
    workspace: str | Path | None = None,
    scale: int = 2,
    run_inspect: bool = True,
    node_path: str | None = None,
    script_path: str | None = None,
) -> dict[str, Any]:
    out = ensure_dir(out_dir)
    pptx_path = Path(pptx) if pptx else out / "editable_composition.pptx"
    workspace_path = ensure_dir(workspace or out / "presentations_plugin_qa_workspace")
    inspect_dir = workspace_path / "template-inspect"
    manifest_path = inspect_dir / "template-manifest.json"
    preview_path = inspect_dir / "source-slides" / "source-slide-01.png"
    layout_path = inspect_dir / "layouts" / "source-slide-01.layout.json"
    log_path = workspace_path / "presentations_inspect.log"
    node = node_path or _default_node_path()
    script = script_path or _default_presentations_script()

    inspect_status = "skipped"
    return_code: int | None = None
    combined_output = ""
    if run_inspect:
        if not script or not Path(script).exists():
            inspect_status = "skipped_missing_presentations_plugin"
            combined_output = "Presentations inspect script not found.\n"
        else:
            env = os.environ.copy()
            if os.getenv("USERPROFILE"):
                env["HOME"] = os.getenv("USERPROFILE", "")
            cmd = [node, script, "--workspace", str(workspace_path), "--pptx", str(pptx_path), "--scale", str(scale)]
            started = time.time()
            proc = subprocess.run(cmd, cwd=str(out), capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=180)
            return_code = proc.returncode
            combined_output = (proc.stdout or "") + (proc.stderr or "")
            inspect_status = "completed" if proc.returncode == 0 else "failed"
            combined_output += f"\nRFS_PRESENTATIONS_QA_ELAPSED_SECONDS={time.time() - started:.3f}\n"
    else:
        combined_output = "Presentations inspect execution skipped by run_inspect=false.\n"
    write_text(log_path, combined_output)

    manifest = _read_json_lenient(manifest_path)
    object_counts = _pptx_object_counts(pptx_path)
    fallback_count = combined_output.count("autoRouteConnectorPx failed")
    preview_exists = preview_path.exists()
    final_png = _find_final_png(out)
    diff = _image_diff(final_png, preview_path if preview_exists else None)
    if inspect_status == "completed" and fallback_count:
        plugin_status = "import_render_completed_with_connector_fallback_warnings"
    elif inspect_status == "completed":
        plugin_status = "import_render_completed"
    else:
        plugin_status = inspect_status

    report = {
        "summary": "Presentations plugin QA report. RFS remains the authoritative PPTX compiler; Presentations is inspection-only.",
        "policy": "pptx_first_rfs_primary; presentations_plugin_qa_only; no_pptx_mutation",
        "deliverables": {
            "editable_pptx": str(pptx_path),
            "png": str(final_png) if final_png else None,
            "pdf": str(out / "review.pdf") if (out / "review.pdf").exists() else None,
        },
        "editable_object_counts": object_counts,
        "presentations_plugin_qa": {
            "workspace": str(workspace_path),
            "manifest": str(manifest_path),
            "preview": str(preview_path) if preview_exists else None,
            "layout_json": str(layout_path) if layout_path.exists() else None,
            "log": str(log_path),
            "node": node,
            "script": script,
            "scale": scale,
            "return_code": return_code,
            "status": plugin_status,
            "autoRouteConnectorPx_failed_count": fallback_count,
            "manifest_slide_count": manifest.get("slideCount"),
            "manifest_media_count": (manifest.get("packageParts") or {}).get("mediaCount") if isinstance(manifest.get("packageParts"), dict) else None,
            "interpretation": "Use Presentations for QA/inspection. Do not use it as the primary compiler for reference-locked connector-heavy research figures unless connector fallback warnings are resolved.",
        },
        "image_diff": diff,
    }
    report_json = out / "presentations_plugin_qa_report.json"
    report_md = out / "presentations_plugin_qa_report.md"
    report["report_json"] = str(report_json)
    report["report_md"] = str(report_md)
    write_json(report_json, report)
    _write_markdown_report(report_md, report)
    return report
