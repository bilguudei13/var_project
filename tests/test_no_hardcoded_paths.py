"""Guard that productive files contain no hardcoded user-machine paths.

Scans Python sources, Jupyter notebooks and R scripts for absolute
paths that only exist on a single contributor's machine. Audit
artefacts under output/ and report/ are excluded; the report mentions
some of these paths as historical context.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Patterns that indicate a machine-specific path. The Windows patterns
# are written as raw strings; the WSL pattern matches /c/Users style.
BAD_PATH_PATTERNS = [
    re.compile(r"C:\\Users", re.IGNORECASE),
    re.compile(r"c:/Users", re.IGNORECASE),
    re.compile(r"G:/My Drive", re.IGNORECASE),
    re.compile(r"/c/Users/"),
]

# Directories that are not considered productive code and may
# legitimately contain such strings (audit PDFs, this very test,
# build caches).
EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "renv",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "node_modules",
    # audit artefacts and historical notes — not productive code
    "output",
    "audit_reports",
    "tmp",
}

# File extensions to scan.
SCANNED_SUFFIXES = {".py", ".ipynb", ".R", ".r"}


def _iter_productive_files() -> list[Path]:
    files: list[Path] = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in SCANNED_SUFFIXES:
            continue
        if any(part in EXCLUDED_DIRS for part in path.relative_to(REPO_ROOT).parts):
            continue
        if path.resolve() == Path(__file__).resolve():
            # Never scan this test file itself — it has to contain the patterns.
            continue
        files.append(path)
    return files


def _notebook_text(path: Path) -> str:
    """Concatenate the source of every code/markdown cell in a notebook."""
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    parts: list[str] = []
    for cell in nb.get("cells", []):
        src = cell.get("source", "")
        if isinstance(src, list):
            parts.append("".join(src))
        else:
            parts.append(str(src))
    return "\n".join(parts)


def _file_text(path: Path) -> str:
    if path.suffix == ".ipynb":
        return _notebook_text(path)
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


@pytest.mark.parametrize("path", _iter_productive_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_file_has_no_hardcoded_user_paths(path: Path) -> None:
    text = _file_text(path)
    offenders = [pat.pattern for pat in BAD_PATH_PATTERNS if pat.search(text)]
    assert not offenders, (
        f"{path.relative_to(REPO_ROOT)} contains hardcoded user path(s): {offenders}. "
        "Replace with a repo-relative path or an environment variable."
    )
