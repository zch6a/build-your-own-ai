"""The release notes for a version: its section of CHANGELOG.md, printed for `gh release create --notes-file`.

    python scripts/release_notes.py 0.2.0
"""

from __future__ import annotations

import sys
from pathlib import Path

CHANGELOG = Path(__file__).resolve().parent.parent / "CHANGELOG.md"


def notes(changelog: str, version: str) -> str:
    """The text between the version's `## ` heading and the next one.

    >>> notes("# C\\n\\n## 0.2.0 — 2026-11-13\\n\\n- a\\n\\n## 0.1.0\\n- b\\n", "0.2.0")
    '- a'
    """
    lines = changelog.replace("\r\n", "\n").split("\n")
    heads = (f"## {version} ", f"## [{version}]")
    start = next((i for i, line in enumerate(lines) if line.startswith(heads) or line == f"## {version}"), None)
    if start is None:
        raise SystemExit(f"CHANGELOG.md has no section for {version}")
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    body = "\n".join(lines[start + 1 : end]).strip()
    if not body:
        raise SystemExit(f"CHANGELOG.md's section for {version} is empty")
    return body


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/release_notes.py <version>")
    sys.stdout.write(notes(CHANGELOG.read_text(encoding="utf-8"), sys.argv[1].removeprefix("v")) + "\n")
