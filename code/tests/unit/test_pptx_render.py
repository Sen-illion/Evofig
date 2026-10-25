"""The preview must be the compiled deck, and a fallback must never be silent."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rfs.composition import pptx_render
from rfs.composition.pptx import compile_ppt
from rfs.composition.pptx_render import (
    PptxRenderError,
    available_backends,
    render_deck_preview,
    render_mode,
    render_pptx_page,
)


def _program(width: int = 1600, height: int = 900) -> dict:
    return {
        "canvas": {"width_px": width, "height_px": height, "width_in": 13.333, "height_in": 7.5, "background": "#FFFFFF"},
        "style": {"palette": ["#FFFFFF", "#4A90C2"]},
        "title_block": {"title": "Paper Method", "subtitle": "Editable paper framework", "bbox_percent": {"x": .04, "y": .015, "w": .92, "h": .08}},
        "panels": [{"id": "panel_a", "title": "Pipeline Stage", "bbox_percent": {"x": .05, "y": .2, "w": .9, "h": .6}}],
        "cards": [],
        "semantic_nodes": [{"id": "node_a", "label": "Encoder", "panel_id": "panel_a", "bbox_percent": {"x": .2, "y": .4, "w": .2, "h": .15}}],
        "slots": [],
        "assets": [],
        "arrows": [],
    }


class RenderModeTests(unittest.TestCase):
    def test_mode_defaults_to_the_compiled_deck(self) -> None:
        with patch.dict("os.environ", {}, clear=False):
            for name in ("RFS_PREVIEW_RENDER",):
                with patch.dict("os.environ", {name: ""}, clear=False):
                    self.assertEqual(render_mode(), "auto")

    def test_unknown_mode_is_ignored_rather_than_breaking_the_run(self) -> None:
        with patch.dict("os.environ", {"RFS_PREVIEW_RENDER": "banana"}, clear=False):
            self.assertEqual(render_mode(), "auto")

    def test_program_mode_is_honoured(self) -> None:
        with patch.dict("os.environ", {"RFS_PREVIEW_RENDER": "program"}, clear=False):
            self.assertEqual(render_mode(), "program")


class RenderDeckPreviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.program_path = self.root / "figure_program.json"
        self.program_path.write_text(json.dumps(_program()), encoding="utf-8")
        self.pptx = self.root / "editable_composition.pptx"
        compile_ppt(_program(), self.root)

    def test_the_canvas_the_program_declares_drives_the_render_size(self) -> None:
        program = _program(width=1280, height=720)
        seen: dict = {}

        def fake_render(pptx, out, **kwargs):
            seen.update(kwargs)
            Path(out).write_bytes(b"stub")
            return {"backend": "libreoffice", "preview": str(out)}

        with patch.object(pptx_render, "render_pptx_page", fake_render):
            preview, info = render_deck_preview(program, self.pptx, self.root, self.root / "preview.png")
        self.assertEqual((seen["width_px"], seen["height_px"]), (1280, 720))
        self.assertEqual(info["backend"], "libreoffice")
        self.assertTrue(preview.exists())

    def test_a_missing_engine_falls_back_and_says_why(self) -> None:
        with patch.object(pptx_render, "available_backends", lambda: []):
            preview, info = render_deck_preview(_program(), self.pptx, self.root, self.root / "preview.png")
        self.assertTrue(preview.exists(), "the program renderer must still produce a preview")
        self.assertEqual(info["backend"], "program")
        self.assertIn("no Office engine", info["fallback_reason"])

    def test_an_engine_failure_falls_back_and_repeats_the_reason(self) -> None:
        with patch.object(pptx_render, "available_backends", lambda: ["libreoffice"]), \
                patch.object(pptx_render, "_convert_with_libreoffice", side_effect=PptxRenderError("engine exploded")):
            preview, info = render_deck_preview(_program(), self.pptx, self.root, self.root / "preview.png")
        self.assertTrue(preview.exists())
        self.assertEqual(info["backend"], "program")
        self.assertIn("engine exploded", info["fallback_reason"])

    def test_program_mode_is_recorded_as_a_choice_not_a_fallback(self) -> None:
        with patch.dict("os.environ", {"RFS_PREVIEW_RENDER": "program"}, clear=False):
            preview, info = render_deck_preview(_program(), self.pptx, self.root, self.root / "preview.png")
        self.assertTrue(preview.exists())
        self.assertEqual(info["backend"], "program")
        self.assertEqual(info.get("forced"), "RFS_PREVIEW_RENDER=program")
        self.assertNotIn("fallback_reason", info)


class RenderPptxPageTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.pptx = self.root / "editable_composition.pptx"
        compile_ppt(_program(), self.root)

    def test_a_missing_deck_is_an_error_not_a_blank_preview(self) -> None:
        with self.assertRaises(PptxRenderError) as caught:
            render_pptx_page(self.root / "nope.pptx", self.root / "out.png", backend="libreoffice")
        self.assertIn("missing", str(caught.exception))

    def test_no_available_backend_is_an_error(self) -> None:
        with patch.object(pptx_render, "available_backends", lambda: []):
            with self.assertRaises(PptxRenderError) as caught:
                render_pptx_page(self.pptx, self.root / "out.png")
        self.assertIn("no Office engine", str(caught.exception))

    def test_an_unknown_backend_is_rejected(self) -> None:
        with self.assertRaises(PptxRenderError) as caught:
            render_pptx_page(self.pptx, self.root / "out.png", backend="wordperfect")
        self.assertIn("unknown render backend", str(caught.exception))

    def test_a_relative_file_argument_is_resolved_before_the_engine_sees_it(self) -> None:
        # PowerPoint COM rejects a relative path outright (0x80070003).
        seen: dict = {}

        def fake_convert(pptx, pdf, timeout):
            seen["pptx"] = pptx
            Path(pdf).write_bytes(b"%PDF-1.4 stub")

        previous = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, previous)
        with patch.object(pptx_render, "_convert_with_libreoffice", fake_convert), \
                patch.object(pptx_render, "_rasterise_pdf", lambda *a, **k: seen.setdefault("rasterised", True)):
            render_pptx_page("editable_composition.pptx", self.root / "out.png", backend="libreoffice")
        self.assertNotEqual(seen["pptx"], Path("editable_composition.pptx"))
        self.assertTrue(seen["pptx"].is_absolute())


class LibreOfficeConversionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.pptx = self.root / "editable_composition.pptx"
        compile_ppt(_program(), self.root)

    def test_every_conversion_gets_its_own_profile_and_output_directory(self) -> None:
        # Two decks in one directory are both called editable_composition.pptx,
        # so a shared output directory makes LibreOffice abort the second write.
        commands: list[list[str]] = []

        def fake_run(command, **kwargs):
            commands.append(command)
            return subprocess.CompletedProcess(command, 1, "", "boom: no pdf written")

        with patch.object(pptx_render, "_libreoffice_binary", lambda: "/usr/bin/soffice"), \
                patch.object(pptx_render.subprocess, "run", fake_run):
            for _ in range(2):
                with self.assertRaises(PptxRenderError):
                    pptx_render._convert_with_libreoffice(self.pptx, self.root / "page.pdf", 30)

        self.assertEqual(len(commands), 2)
        profiles = [next(part for part in cmd if part.startswith("-env:UserInstallation=")) for cmd in commands]
        outdirs = [cmd[cmd.index("--outdir") + 1] for cmd in commands]
        self.assertNotEqual(profiles[0], profiles[1])
        self.assertNotEqual(outdirs[0], outdirs[1])

    def test_a_hung_engine_is_reported_as_a_timeout(self) -> None:
        def fake_run(command, **kwargs):
            raise subprocess.TimeoutExpired(command, 30)

        with patch.object(pptx_render, "_libreoffice_binary", lambda: "/usr/bin/soffice"), \
                patch.object(pptx_render.subprocess, "run", fake_run):
            with self.assertRaises(PptxRenderError) as caught:
                pptx_render._convert_with_libreoffice(self.pptx, self.root / "page.pdf", 30)
        self.assertIn("timed out", str(caught.exception))

    def test_a_missing_engine_is_reported_before_anything_runs(self) -> None:
        with patch.object(pptx_render, "_libreoffice_binary", lambda: None):
            with self.assertRaises(PptxRenderError) as caught:
                pptx_render._convert_with_libreoffice(self.pptx, self.root / "page.pdf", 30)
        self.assertIn("not installed", str(caught.exception))

    def test_the_temporary_profile_and_output_directory_are_cleaned_up(self) -> None:
        made: list[str] = []
        real_mkdtemp = tempfile.mkdtemp

        def tracking_mkdtemp(*args, **kwargs):
            path = real_mkdtemp(*args, **kwargs)
            made.append(path)
            return path

        def fake_run(command, **kwargs):
            return subprocess.CompletedProcess(command, 1, "", "no pdf")

        with patch.object(pptx_render, "_libreoffice_binary", lambda: "/usr/bin/soffice"), \
                patch.object(pptx_render.subprocess, "run", fake_run), \
                patch.object(pptx_render.tempfile, "mkdtemp", tracking_mkdtemp):
            with self.assertRaises(PptxRenderError):
                pptx_render._convert_with_libreoffice(self.pptx, self.root / "page.pdf", 30)
        self.assertTrue(made, "the conversion must have made its own scratch directories")
        self.assertTrue(all(not Path(path).exists() for path in made))


class WorkflowPreviewReportTests(unittest.TestCase):
    def test_the_workflow_records_which_drawing_produced_each_preview(self) -> None:
        from rfs.workflows.paper_evolve_editable import _preview_from_deck

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            program_path = root / "figure_program.json"
            program_path.write_text(json.dumps(_program()), encoding="utf-8")
            compile_ppt(_program(), root)
            with patch.dict("os.environ", {"RFS_PREVIEW_RENDER": "program"}, clear=False):
                preview = _preview_from_deck(program_path, root, root / "draft_preview.png")
                _preview_from_deck(program_path, root, root / "final_draft_preview.png")
            self.assertTrue(preview.exists())
            report = json.loads((root / "preview_render_report.json").read_text(encoding="utf-8"))
            renders = report["renders"]
            self.assertEqual(len(renders), 2, "every preview of the attempt is listed")
            self.assertTrue(all(entry["backend"] in {"libreoffice", "powerpoint", "program"} for entry in renders))
            self.assertTrue(all(entry["program"] == str(program_path) for entry in renders))


class AvailableBackendsTests(unittest.TestCase):
    def test_backends_are_reported_for_the_host(self) -> None:
        for backend in available_backends():
            self.assertIn(backend, pptx_render.BACKENDS)

    def test_a_host_without_libreoffice_or_powerpoint_reports_nothing(self) -> None:
        with patch.object(pptx_render, "_libreoffice_binary", lambda: None), \
                patch.object(pptx_render, "_powerpoint_available", lambda: False):
            self.assertEqual(available_backends(), [])


if __name__ == "__main__":
    unittest.main()