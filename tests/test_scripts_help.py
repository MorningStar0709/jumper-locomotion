"""Every entry script answers `--help` with a usage line and exit code 0.

`play.py` did not. Its `--physics-hz` help carried a literal `7-30%`, and
argparse renders help text by `%`-interpolation, so

    python scripts/play.py --help

died with `ValueError: unsupported format character 'b'` -- argparse reads the
space after the percent as a printf flag and then looks for the conversion
character at the next word's first letter. Found while walking a freshly built
bundle the way a user would, not by any test of the bundle's contents.

The check is a subprocess rather than an in-process `parse_args` because that is
how the scripts get used, and because the crash happened before any argument was
parsed: a script whose help cannot be printed is a script nobody can start
working with.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: The four entry points a person types first.  Ad-hoc tools under `tools/` have
#: their own tests and their own parsers.
SCRIPTS = ("train.py", "play.py", "export.py", "deploy.py")


@pytest.mark.parametrize("script", SCRIPTS)
def test_the_help_of_a_script_is_printable(script: str) -> None:
    finished = subprocess.run(
        [sys.executable, str(REPO / "scripts" / script), "--help"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert finished.returncode == 0, finished.stderr[-2000:]
    assert "usage:" in finished.stdout.lower()
