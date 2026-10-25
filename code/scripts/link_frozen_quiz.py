#!/usr/bin/env python3
"""Link the frozen PaperQuiz bank into a staged scoring tree.

``benchmark score-own --paper-quiz api`` reads ``questions.json`` from each run directory. When it
finds one it grades against those questions; when it does not it writes a fresh bank from the paper
first. The external systems in Table 1 were all graded against one shared frozen bank, so linking
that same bank is what makes this project's PaperQuiz column a comparison instead of a column of its
own -- and it saves the extra question-generation call per paper.

The staged tree already names each paper with ``safe_paper_id``, which is the spelling the bank
uses, so matching is by directory name and the manifest is not needed.

    python scripts/link_frozen_quiz.py \
        --staging <DATA_ROOT>/abl/staging_main/ours_own500 \
        --questions-root <DATA_ROOT>/efb_main/own/questions
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rfs.utils import write_json  # noqa: E402 - path first

BANK_NAME = "questions.json"
STAGED_MARKERS = ("delivery.pptx", "preview.png")


def link(origin: Path, dest: Path) -> bool:
    """Point ``dest`` at ``origin``, replacing whatever was there. False if the source is absent.

    A symbolic link is what we want, but Windows refuses one without developer mode, and the bank
    is small, so a hard link and then a copy stand in rather than failing the staging step.
    """

    if not origin.is_file():
        return False
    if dest.is_symlink() or dest.exists():
        dest.unlink()
    try:
        os.symlink(origin, dest)
    except OSError:
        try:
            os.link(origin, dest)
        except OSError:
            shutil.copyfile(origin, dest)
    return True


def staged_papers(staging: Path) -> list:
    """The papers a scoring tree actually holds, in name order."""

    return sorted(
        child for child in staging.iterdir()
        if child.is_dir() and any((child / marker).is_file() for marker in STAGED_MARKERS)
    )


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--staging", required=True, help="The staged tree the scorer will read.")
    parser.add_argument("--questions-root", default="<DATA_ROOT>/efb_main/own/questions",
                        help="The frozen bank, one directory per safe paper id.")
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero if any staged paper has no bank, instead of reporting it.")
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    args = parser.parse_args(argv)

    staging = Path(args.staging)
    questions_root = Path(args.questions_root)
    if not staging.is_dir():
        raise SystemExit("no such staging tree: %s" % staging)

    considered = staged_papers(staging)
    linked = 0
    absent = []
    for paper in considered:
        if link(questions_root / paper.name / BANK_NAME, paper / BANK_NAME):
            linked += 1
        else:
            absent.append(paper.name)

    report = {
        "summary": "Link the frozen PaperQuiz bank into a staged scoring tree.",
        "staging": str(staging),
        "questions_root": str(questions_root),
        "questions_root_exists": questions_root.is_dir(),
        "considered": len(considered),
        "linked": linked,
        "absent": absent,
    }
    write_json(staging / "_quiz_link_report.json", report)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("linked=%d absent=%d of %d staged papers" % (linked, len(absent), len(considered)))
        for name in absent[:20]:
            print("  no frozen bank: %s" % name)
    if args.strict and absent:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())