#!/usr/bin/env python3
"""Audit the anonymous code package for private paths and generated artifacts."""

from __future__ import annotations

import argparse
import gzip
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]

TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

IGNORED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
}

GENERATED_PARTS = {
    "benchmark_results",
    "docs",
    "local_artifacts",
    "local_data",
    "outputs",
    "results",
}

FORBIDDEN_NAMES = {
    ".DS_Store": "macOS metadata file",
    ".env": "environment file",
}

ALLOWED_USER_SEGMENTS = {
    "anonymous",
    "example",
    "reviewer",
    "runner",
    "user",
    "username",
    "your-user",
    "your_username",
}

PRIVATE_PATH_RE = re.compile(
    r"(?P<path>/(?:Users|users|home)/(?P<user>[A-Za-z0-9._-]+)(?:/[^\s'\"`<>)]*)?)"
)
SECRET_TOKEN_RE = re.compile(r"\b(?:sk|xox[baprs]?)-[A-Za-z0-9_-]{20,}\b")
SECRET_ASSIGNMENT_RE = re.compile(
    r"\b(?:api[_-]?key|access[_-]?token|secret|password)\b\s*[:=]\s*['\"][^'\"<>{}\s]{8,}['\"]",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Finding:
    path: Path
    reason: str
    line_number: int | None = None
    line: str | None = None

    def format(self, root: Path) -> str:
        try:
            display = self.path.relative_to(root)
        except ValueError:
            display = self.path
        if self.line_number is None:
            return f"{display}: {self.reason}"
        return f"{display}:{self.line_number}: {self.reason}: {self.line}"


def _git_paths(root: Path) -> list[Path]:
    proc = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return [path for path in root.rglob("*") if path.is_file()]
    return [root / line for line in proc.stdout.splitlines() if line.strip()]


def _is_ignored(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    return any(part in IGNORED_PARTS for part in rel.parts)


def _is_generated_path(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    return any(part in GENERATED_PARTS for part in rel.parts)


def _is_text_candidate(path: Path) -> bool:
    if path.name.endswith(".jsonl.gz"):
        return True
    if path.suffix in {".gz", ".jpg", ".jpeg", ".png", ".npz", ".pt", ".pth", ".zip"}:
        return False
    return path.suffix.lower() in TEXT_SUFFIXES


def _read_text(path: Path) -> str | None:
    if not _is_text_candidate(path):
        return None
    try:
        if path.name.endswith(".jsonl.gz"):
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                return handle.read()
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None
    except OSError:
        return None


def _allowed_private_path(match: re.Match[str]) -> bool:
    user = match.group("user")
    return user in ALLOWED_USER_SEGMENTS or user.startswith("user")


def scan_paths(paths: Iterable[Path], *, root: Path) -> list[Finding]:
    """Scan explicit paths and return anonymity findings."""

    findings: list[Finding] = []
    root = root.resolve()

    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        if not path.exists() or path.is_dir() or _is_ignored(path, root):
            continue

        if path.name in FORBIDDEN_NAMES:
            findings.append(Finding(path=path, reason=FORBIDDEN_NAMES[path.name]))

        if _is_generated_path(path, root):
            findings.append(Finding(path=path, reason="generated or local artifact path"))

        text = _read_text(path)
        if text is None:
            continue

        for line_number, line in enumerate(text.splitlines(), start=1):
            for match in PRIVATE_PATH_RE.finditer(line):
                if not _allowed_private_path(match):
                    findings.append(
                        Finding(
                            path=path,
                            line_number=line_number,
                            line=line.strip(),
                            reason="absolute private user path",
                        )
                    )
            if SECRET_TOKEN_RE.search(line) or SECRET_ASSIGNMENT_RE.search(line):
                findings.append(
                    Finding(
                        path=path,
                        line_number=line_number,
                        line=line.strip(),
                        reason="possible secret or API token",
                    )
                )

    return findings


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--strict", action="store_true", help="Return non-zero when findings exist.")
    parser.add_argument("paths", nargs="*", type=Path, help="Optional explicit paths to scan.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    paths = args.paths or _git_paths(root)
    findings = scan_paths(paths, root=root)

    if findings:
        for finding in findings:
            print(finding.format(root), file=sys.stderr)
        return 1 if args.strict else 0

    print("No anonymous-package audit findings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
