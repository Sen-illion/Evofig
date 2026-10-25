"""An icon the repair named is only drawn again when its request really moved.

The replan rewrites every icon record, so ``changed_asset_ids`` names every slot even when the
only difference is the accent colour the panel palette owns. Those redraws cost $0.0336 each
and 18 of the 19 repair rounds in the SciFlow probe were rolled back afterwards, so the icon
they paid for never reached the deliverable.
"""

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from rfs.generation.icon_assets import build_icon_prompt, generate_icon_assets, prompt_reuse_key


def _icon(subject: str = "A", guidance: dict | None = None, background: str = "transparent_panel_inherit") -> dict:
    icon = {"asset_id": "icon_a", "subject": subject, "prompt": subject, "background_policy": background}
    if guidance is not None:
        icon["color_guidance"] = guidance
    return icon


class PromptReuseKeyTest(unittest.TestCase):
    def test_accent_colour_alone_is_not_a_reason_to_draw_again(self):
        green = _icon(guidance={"fill": "#ECFDF5", "border": "#10B981", "text": "#064E3B"})
        slate = _icon(guidance={"fill": "#F8FAFC", "border": "#193247", "text": "#111827"})
        self.assertEqual(prompt_reuse_key(build_icon_prompt(green)), prompt_reuse_key(build_icon_prompt(slate)))

    def test_a_nested_colour_role_is_still_only_a_colour(self):
        # The clause used to be trimmed by matching the braces of its value, so a role that
        # carried a nested mapping closed the match early and the whole clause stayed in the
        # key -- which made a palette move look like a new request and re-bought the set.
        role = _icon(guidance={"role": "accent", "colors": {"fill": "#ECFDF5", "border": "#10B981"}})
        other = _icon(guidance={"role": "accent", "colors": {"fill": "#F8FAFC", "border": "#193247"}})

        self.assertNotIn("restrained accents", prompt_reuse_key(build_icon_prompt(role)))
        self.assertEqual(prompt_reuse_key(build_icon_prompt(role)), prompt_reuse_key(build_icon_prompt(other)))

    def test_the_subject_still_is(self):
        self.assertNotEqual(prompt_reuse_key(build_icon_prompt(_icon("A"))), prompt_reuse_key(build_icon_prompt(_icon("B"))))

    def test_the_background_policy_still_is(self):
        transparent = _icon(background="transparent_panel_inherit")
        solid = {**_icon(background="solid"), "panel_background_color": "#FFFFFF"}
        self.assertNotEqual(prompt_reuse_key(build_icon_prompt(transparent)), prompt_reuse_key(build_icon_prompt(solid)))


class TargetedIconReuseTest(unittest.TestCase):
    def _frozen(self, root: Path, icon: dict) -> Path:
        frozen = root / "frozen"
        (frozen / "icon_a").mkdir(parents=True)
        Image.new("RGB", (16, 16), "white").save(frozen / "icon_a" / "selected.png")
        (frozen / "icon_a" / "prompt.txt").write_text(build_icon_prompt(icon), encoding="utf-8")
        return frozen

    def test_a_palette_move_keeps_the_paid_icon(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frozen = self._frozen(root, _icon(guidance={"fill": "#ECFDF5", "border": "#10B981", "text": "#064E3B"}))
            icon = _icon(guidance={"fill": "#F8FAFC", "border": "#193247", "text": "#111827"})
            report = generate_icon_assets(
                {"icons": [icon]},
                root / "round_01",
                mode="placeholder",
                reuse_assets_from=frozen,
                target_asset_ids=["icon_a"],
            )
        self.assertEqual(report["api_requests_attempted"], 0)
        self.assertEqual(report["assets"][0]["status"], "reused")
        self.assertEqual(report["assets"][0]["reuse_note"], "reused_unchanged_prompt")

    def test_a_moved_request_draws_again(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frozen = self._frozen(root, _icon("A"))
            report = generate_icon_assets(
                {"icons": [_icon("B")]},
                root / "round_01",
                mode="placeholder",
                reuse_assets_from=frozen,
                target_asset_ids=["icon_a"],
            )
        self.assertEqual(report["assets"][0]["status"], "generated")
        self.assertEqual(report["assets"][0]["reuse_note"], "redrawn_prompt_changed")

    def test_an_unnamed_slot_keeps_reusing_without_a_recorded_request(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frozen = root / "frozen"
            (frozen / "icon_a").mkdir(parents=True)
            Image.new("RGB", (16, 16), "white").save(frozen / "icon_a" / "selected.png")
            report = generate_icon_assets({"icons": [_icon("B")]}, root / "round_01", mode="placeholder", reuse_assets_from=frozen)
        self.assertEqual(report["assets"][0]["status"], "reused")
        self.assertEqual(report["assets"][0]["reuse_note"], "reused_untouched_slot")

    def test_a_named_slot_without_a_recorded_request_is_drawn_again(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frozen = root / "frozen"
            (frozen / "icon_a").mkdir(parents=True)
            Image.new("RGB", (16, 16), "white").save(frozen / "icon_a" / "selected.png")
            report = generate_icon_assets(
                {"icons": [_icon("A")]},
                root / "round_01",
                mode="placeholder",
                reuse_assets_from=frozen,
                target_asset_ids=["icon_a"],
            )
        self.assertEqual(report["assets"][0]["status"], "generated")
        self.assertEqual(report["assets"][0]["reuse_note"], "redrawn_prompt_changed")

    def test_the_report_counts_what_it_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frozen = self._frozen(root, _icon("A"))
            report = generate_icon_assets(
                {"icons": [_icon("A")]},
                root / "round_01",
                mode="placeholder",
                reuse_assets_from=frozen,
                target_asset_ids=["icon_a"],
            )
        self.assertEqual(report["reuse_summary"], {"reused_unchanged_prompt": 1})


if __name__ == "__main__":
    unittest.main()
