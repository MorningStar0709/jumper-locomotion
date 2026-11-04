#!/usr/bin/env python3
"""Check that every relative link and image reference in the markdown resolves.

This repository is a curated slice of a larger checkout, so a link that was fine there can
point at a file that is deliberately absent here. CI runs this so the README cannot drift
into a set of dead ends.

    python tools/check_links.py

Exits 0 when everything resolves; prints each miss and exits 1 otherwise.
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SKIP = {".git", "node_modules", "__pycache__"}

# Every link target is what follows "](…". Matching the target rather than the whole
# [text](target) pair is what makes nested forms work: [![alt](img)](outer) yields both
# "img" and "outer", where a [..](..) pattern would see only the inner one.
LINK = re.compile(r"\]\(\s*([^)]+?)\s*\)")
FENCE = re.compile(r"```.*?```", re.S)
INLINE = re.compile(r"`[^`\n]*`")
SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


def markdown_files() -> list[pathlib.Path]:
    return [p for p in ROOT.rglob("*.md") if not SKIP & set(p.relative_to(ROOT).parts)]


def prose(text: str) -> str:
    """Drop fenced blocks and inline code: a link written inside code is not a link."""
    return INLINE.sub(" ", FENCE.sub(" ", text))


def targets(text: str) -> list[str]:
    return [match.group(1) for match in LINK.finditer(text)]


def resolve(source: pathlib.Path, target: str) -> pathlib.Path | None:
    """The path a target points at, or None when it does not point inside this tree."""
    if not target or target.startswith("#") or SCHEME.match(target) or target.startswith("/"):
        return None
    target = target.split("#", 1)[0]
    if not target:
        return None
    return (source.parent / target).resolve()


def main() -> int:
    sources = markdown_files()
    misses = []
    checked = 0
    for source in sources:
        for target in targets(prose(source.read_text(encoding="utf-8"))):
            path = resolve(source, target)
            if path is None:
                continue
            checked += 1
            if not path.exists():
                misses.append((source.relative_to(ROOT), target))
    for source, target in misses:
        print(f"{source}: {target} does not exist")
    print(f"checked {checked} relative links in {len(sources)} files, {len(misses)} missing")
    return 1 if misses else 0


if __name__ == "__main__":
    sys.exit(main())