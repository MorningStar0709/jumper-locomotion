#!/usr/bin/env python3
"""Assert what this repository deliberately does not contain.

The portfolio is a curated slice of a larger checkout: the simulation framework, the robot
assets and every trained checkpoint stay upstream. That is a deliberate publication
decision, so CI guards it — a future commit that drops a checkpoint or an upstream asset
into this tree should fail here rather than in a licence review.

    python tools/check_slice.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

FORBIDDEN_DIRS = ("rl", "assets", "logs", "out", "target", "vendor", "node_modules", ".openchamber")
FORBIDDEN_SUFFIXES = (".pt", ".onnx", ".rknn", ".npz", ".mp4", ".so", ".dylib", ".dll", ".egg-info")
REQUIRED = ("LICENSE", "NOTICE", "licenses", "README.md", "README.zh.md", "pyproject.toml")


def tracked() -> list[pathlib.Path]:
    """Every path git tracks, so an untracked build artefact does not fail the guard."""
    import subprocess

    listed = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
    ).stdout
    return [pathlib.Path(name) for name in listed.decode().split("\0") if name]


def main() -> int:
    paths = tracked()
    problems = []

    for name in FORBIDDEN_DIRS:
        if any(p.parts and p.parts[0] == name for p in paths):
            problems.append(f"tracked files under {name}/ — that tree stays upstream")

    for path in paths:
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"{path} — binaries and checkpoints stay upstream")

    for name in REQUIRED:
        if not (ROOT / name).exists():
            problems.append(f"{name} is missing")

    for problem in problems:
        print(problem)
    print(f"checked {len(paths)} tracked paths, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())