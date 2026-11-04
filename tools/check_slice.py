#!/usr/bin/env python3
"""Assert what this repository deliberately does not contain.

The portfolio is a curated slice of a larger checkout: the simulation framework, the robot
assets and every trained checkpoint stay upstream. That is a deliberate publication
decision, so CI guards it — a future commit that drops a checkpoint or an upstream asset
into this tree should fail here rather than in a licence review.

Everything is decided from `git ls-files`, never from the working tree: a file that exists
locally but is not tracked is not published, and a required file that was deleted from the
index is not published either, whatever is still lying around on disk.

    python tools/check_slice.py
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

FORBIDDEN_DIRS = ("rl", "assets", "logs", "out", "target", "vendor", "node_modules", ".openchamber")
FORBIDDEN_SUFFIXES = (".pt", ".onnx", ".rknn", ".npz", ".mp4", ".so", ".dylib", ".dll")
FORBIDDEN_PARTS = (".egg-info", "__pycache__", ".pytest_cache", ".ruff_cache")
REQUIRED_FILES = ("LICENSE", "NOTICE", "README.md", "README.zh.md", "pyproject.toml")
REQUIRED_TREES = ("licenses",)


def tracked() -> list[pathlib.Path]:
    try:
        listed = subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
        ).stdout
    except subprocess.CalledProcessError as error:
        # git's own words are the only thing that makes this triageable from a CI log.
        sys.stderr.write(error.stderr.decode(errors="replace"))
        raise
    return [pathlib.Path(name) for name in listed.decode().split("\0") if name]


def main() -> int:
    paths = tracked()
    problems: list[str] = []

    for name in FORBIDDEN_DIRS:
        if any(p.parts and p.parts[0] == name for p in paths):
            problems.append(f"tracked files under {name}/ — that tree stays upstream")

    for path in paths:
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"{path} — binaries and checkpoints stay upstream")
        if set(FORBIDDEN_PARTS) & set(path.parts):
            problems.append(f"{path} — build and cache directories are not published")

    for name in REQUIRED_FILES:
        if pathlib.Path(name) not in paths:
            problems.append(f"{name} is not tracked")
    for name in REQUIRED_TREES:
        if not any(p.parts and p.parts[0] == name for p in paths):
            problems.append(f"{name}/ is not tracked")

    for problem in problems:
        print(problem)
    print(f"checked {len(paths)} tracked paths, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())