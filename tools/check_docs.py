#!/usr/bin/env python3
"""Warn-only documentation consistency check.

Scans live-method docs for the status-header convention and surfaces STALE docs.
Prints a punch-list; it does NOT block commits (the pre-commit hook ignores the exit
code). Run `python3 tools/check_docs.py --strict` to get a non-zero exit for CI.

Convention (see CLAUDE.md): a live doc's first lines carry
    Status: LIVE | STALE | ARCHIVE  ·  Updated: YYYY-MM-DD  ·  Maintainer: ...
    Scope: ...  ·  Describes: ...
Stdlib only, Python 3.6+ (must run under the system python the git hook uses).
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Live docs that should carry a status header. Globs are relative to repo root.
LIVE_GLOBS = [
    "manual/README.md",
    "manual/docs/**/*.md",
    "manual/cgas/*.md",
    "manual/cgas/research/*.md",
]
# Never warn about these (archived / dead / vendored).
EXCLUDE_PARTS = {"archive", "ARCHIVE", "scatter", ".ipynb_checkpoints"}

HEADER_SCAN_LINES = 10

# Lenient matchers: allow **bold** markers; accept Date: as a legacy alias for Updated:.
RE_STATUS = re.compile(r"\*{0,2}Status\*{0,2}\s*:", re.M)
RE_DATE = re.compile(r"\*{0,2}(?:Updated|Date)\*{0,2}\s*:", re.M)
RE_STALE = re.compile(r"STALE", re.I)


def is_excluded(path: Path) -> bool:
    return any(part in EXCLUDE_PARTS for part in path.parts)


def collect() -> list:
    seen, out = set(), []
    for pattern in LIVE_GLOBS:
        for p in ROOT.glob(pattern):
            if p.is_file() and not is_excluded(p) and p not in seen:
                seen.add(p)
                out.append(p)
    return sorted(out)


def main() -> int:
    strict = "--strict" in sys.argv
    missing, stale = [], []
    for p in collect():
        head = "\n".join(p.read_text(errors="replace").splitlines()[:HEADER_SCAN_LINES])
        if RE_STALE.search(head):
            stale.append(p)  # explicit STALE marker takes priority over header shape
        elif not (RE_STATUS.search(head) and RE_DATE.search(head)):
            missing.append(p)

    rel = lambda p: p.relative_to(ROOT)
    if missing:
        print("⚠  docs missing a status header (Status: / Updated:):")
        for p in missing:
            print("     " + str(rel(p)))
    if stale:
        print("⚠  docs marked STALE (fix or remove):")
        for p in stale:
            print("     " + str(rel(p)))
    if not missing and not stale:
        print("✓ doc headers OK ({} live docs checked)".format(len(collect())))

    return 1 if (strict and (missing or stale)) else 0


if __name__ == "__main__":
    sys.exit(main())
