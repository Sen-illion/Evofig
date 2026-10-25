"""Verify exact A-source member bindings without extracting archive paths."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import tempfile
import tarfile
import zipfile

from .source_snapshot_io import copy_verified, sha256_file


def source_member_name(value) -> str:
    """Archive member names are case sensitive, including on Windows hosts."""
    return str(PurePosixPath(str(value).replace("\\", "/")))


def verify_source_members(archive: Path | None, original_files: dict[str, Path],
                          source_names: list[str], figure_hashes: set[str]) -> dict:
    """Require copied figure bytes to occur among their declared source members."""
    wanted = {source_member_name(name) for name in source_names if name}
    found = {}
    def add(name, stream):
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        normalized = source_member_name(name)
        value = digest.hexdigest()
        if normalized in found and found[normalized] != value:
            raise ValueError(f"Conflicting duplicate source archive member: {name}")
        found[normalized] = value
    if archive is not None:
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as source:
                for member in source.infolist():
                    if source_member_name(member.filename) in wanted and not member.is_dir():
                        with source.open(member) as stream:
                            add(member.filename, stream)
        else:
            with tarfile.open(archive, "r:*") as source:
                for member in source:
                    if source_member_name(member.name) in wanted and member.isfile():
                        with source.extractfile(member) as stream:
                            add(member.name, stream)
    else:
        found = {source_member_name(name): sha256_file(path) for name, path in original_files.items() if source_member_name(name) in wanted}
    if wanted - found.keys():
        raise ValueError("Declared source members are missing: " + ", ".join(sorted(wanted - found.keys())))
    if not figure_hashes <= set(found.values()):
        raise ValueError("Selected figure bytes do not match declared original source members")
    return {"method": "exact_source_member_sha256", "members": found, "figure_hashes": sorted(figure_hashes)}


def recover_source_member(archive: Path, name: str, target: Path, expected_sha256: str) -> None:
    """Copy declared archive bytes to a new asset; never extract archive paths."""
    target.parent.mkdir(parents=True, exist_ok=True)
    def copy_stream(stream):
        with tempfile.TemporaryDirectory(prefix="source-member-", dir=target.parent) as temporary:
            candidate = Path(temporary) / "member"
            with candidate.open("wb") as output:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    output.write(chunk)
            copy_verified(candidate, target, expected_sha256)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                if source_member_name(member.filename) == source_member_name(name) and not member.is_dir():
                    with source.open(member) as stream:
                        copy_stream(stream)
                    return
    else:
        with tarfile.open(archive, "r:*") as source:
            for member in source:
                if source_member_name(member.name) == source_member_name(name) and member.isfile():
                    with source.extractfile(member) as stream:
                        copy_stream(stream)
                    return
    raise ValueError(f"Declared source member is no longer available: {name}")
