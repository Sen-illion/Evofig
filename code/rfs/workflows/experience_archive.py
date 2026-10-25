"""Move the Memory rows no later paper could act on into an archive, and keep the rest.

The library is read as a set of lessons, not as a log: a row is worth keeping when another paper
can be drawn by it. A directory move cannot tell the two apart, and one that was meant to clear
out the shells also carried away the rows that had just been measured and improved a figure --
which is why the archive and the library both have to be read as rows, not as directories.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..evolution.direct import ExperienceStore
from ..utils import write_json


ARCHIVE_CONTRACT = "rfs.experience_archive"
ARCHIVE_REPORT_FILE = "archive_report.json"


def archive_untransferable_experiences(library: str | Path, *, out: str | Path, dry_run: bool = False) -> dict[str, Any]:
    """Split a library into the rows that travel and the rows that do not, and report each one.

    ``dry_run`` decides every row and writes nothing at all, so a caller can read the verdict
    before anything moves. The archive is append-only: an archive directory that already holds
    an earlier move keeps its rows, and the report beside it is replaced.
    """

    root = Path(library).resolve()
    destination = Path(out).resolve()
    report = ExperienceStore(root).archive_untransferable(destination, dry_run=dry_run)
    report["summary"] = (
        f"{report['archived']} of {report['declared_records']} row(s) state no lesson another paper "
        f"can act on; {report['kept']} stayed in the library"
        + (" (dry run: nothing was moved)." if dry_run else f", {report['archived']} moved to the archive.")
    )
    report["contract"] = ARCHIVE_CONTRACT
    report["ok"] = True
    write_json(destination / ARCHIVE_REPORT_FILE, report)
    return report


__all__ = ["ARCHIVE_CONTRACT", "ARCHIVE_REPORT_FILE", "archive_untransferable_experiences"]