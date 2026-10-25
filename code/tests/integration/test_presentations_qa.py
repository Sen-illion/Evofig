import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from rfs.presentations_qa import run_presentations_qa


def _make_minimal_pptx(path: Path) -> None:
    slide_xml = """
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:pic></p:pic>
      <p:pic></p:pic>
      <p:cxnSp></p:cxnSp>
      <p:sp><p:txBody></p:txBody></p:sp>
    </p:spTree>
  </p:cSld>
</p:sld>
""".strip()
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("ppt/slides/slide1.xml", slide_xml)
        zf.writestr("ppt/media/image1.png", b"not-a-real-image")


class PresentationsQaTests(unittest.TestCase):
    def test_report_only_qa_does_not_require_plugin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pptx = root / "editable_composition.pptx"
            _make_minimal_pptx(pptx)
            Image.new("RGB", (64, 64), "white").save(root / "final_600dpi.png")

            report = run_presentations_qa(root, run_inspect=False)

            self.assertEqual(report["policy"], "pptx_first_rfs_primary; presentations_plugin_qa_only; no_pptx_mutation")
            self.assertEqual(report["editable_object_counts"]["pictures"], 2)
            self.assertEqual(report["editable_object_counts"]["connectors"], 1)
            self.assertEqual(report["editable_object_counts"]["text_bodies"], 1)
            self.assertEqual(report["presentations_plugin_qa"]["status"], "skipped")
            self.assertTrue((root / "presentations_plugin_qa_report.json").exists())
            self.assertTrue((root / "presentations_plugin_qa_report.md").exists())
            saved = json.loads((root / "presentations_plugin_qa_report.json").read_text(encoding="utf-8"))
            self.assertTrue(saved["summary"].startswith("Presentations plugin QA report"))


if __name__ == "__main__":
    unittest.main()
