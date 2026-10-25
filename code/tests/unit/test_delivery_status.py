import tempfile
import unittest
from pathlib import Path
import zipfile

from rfs.evaluation.delivery_status import classify_delivery_status


class DeliveryStatusTests(unittest.TestCase):
    def test_visual_issue_does_not_block_delivery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with zipfile.ZipFile(root / "editable_composition.pptx", "w") as archive:
                archive.writestr("[Content_Types].xml", "<Types/>")
            result = classify_delivery_status(
                compiled={"pptx": str(root / "editable_composition.pptx")},
                root=root,
                quality_reports={"geometry": {"status": "blocked"}},
                visual_review={"status": "needs_repair"},
            )
            self.assertEqual(result["generation_status"], "completed")
            self.assertEqual(result["delivery_status"], "available")
            self.assertEqual(result["quality_status"], "needs_repair")
            self.assertTrue(result["ok"])

    def test_a_finished_deck_survives_an_invalid_method_contract(self):
        """A deck that exists is delivered; the contract's errors degrade it.

        A method contract that reported itself invalid used to answer the
        delivery question "no" and the compiled PowerPoint was discarded with
        it.  The deck is what delivery answers for, so its figure stays and the
        unresolved errors are named instead.
        """

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with zipfile.ZipFile(root / "editable_composition.pptx", "w") as archive:
                archive.writestr("[Content_Types].xml", "<Types/>")
            result = classify_delivery_status(
                compiled={"pptx": str(root / "editable_composition.pptx")},
                root=root,
                method_validation={"ok": False, "errors": ["method object[1] references unknown output_ids: solver"]},
            )
            self.assertEqual(result["generation_status"], "completed")
            self.assertEqual(result["delivery_status"], "available")
            self.assertEqual(result["quality_status"], "warning")
            self.assertFalse(result["quality_passed"])
            self.assertEqual(result["blocking_generation_errors"], [])
            self.assertEqual(result["contract_degradations"], ["method_contract_invalid"])
            self.assertTrue(any("solver" in str(issue.get("detail")) for issue in result["quality_issues"]))
            self.assertTrue(result["ok"])

    def test_missing_ppt_blocks_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = classify_delivery_status(compiled=None, root=tmp)
            self.assertEqual(result["generation_status"], "failed")
            self.assertEqual(result["delivery_status"], "unavailable")
            self.assertFalse(result["ok"])


if __name__ == "__main__":
    unittest.main()
