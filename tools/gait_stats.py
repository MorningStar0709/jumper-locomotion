"""Gait statistics from a `measure.csv`: who is on the ground, when, and how often.

`play.py --measure` writes joint positions, velocities, torques and the six foot
contact forces at 50 Hz. The question people actually ask of that file is "is it
walking the gait we think it is", and answering it by hand has produced two wrong
numbers already -- a per-foot threshold that turned normal contacts into flight
phases, and a duty factor read off the wrong column.

    .venv/bin/python tools/gait_stats.py logs/.../measure/measure.csv

**The threshold is absolute, in newtons, and applies to every foot alike.** A
relative threshold (a fraction of each foot's own peak) silently fails on this
robot: the front palms carry about twice the load of the middle and rear tips,
so a per-foot threshold lands above the load of a normally standing leg and
reports it as airborne.
"""

from __future__ import annotations

import csv
import sys
import math
from pathlib import Path


DEFAULT_THRESHOLD_N = 2.0
FEET = ("LF", "RF", "LM", "RM", "LR", "RR")


def read_forces(path: Path) -> tuple[list[float], list[list[float]]]:
    """The six foot force columns of a measure file, in `FEET` order, plus time.

    Returns `(times, rows)` —— 名字里不提 time，是因为它以前注解成 `list[str]`，
    而实际返回的第二个元素是每一帧的秒数。
    """
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        columns = {}
        for name in FEET:
            matches = [i for i, column in enumerate(header) if column.startswith(f"force/{name}_")]
            if len(matches) != 1:
                raise ValueError(f"{path}: expected one force column for {name}, found {len(matches)}")
            columns[name] = matches[0]
        times = []
        rows = []
        for row in reader:
            times.append(float(row[0]))
            rows.append([float(row[columns[name]]) for name in FEET])
    return times, rows


def contact_states(rows: list[list[float]], threshold: float = DEFAULT_THRESHOLD_N) -> list[tuple[str, ...]]:
    """Which feet are loaded on each frame, as a tuple of names."""
    return [
        tuple(name for name, force in zip(FEET, row) if force >= threshold)
        for row in rows
    ]


def duty(rows: list[list[float]], threshold: float = DEFAULT_THRESHOLD_N) -> dict[str, float]:
    """Fraction of frames each foot spends loaded."""
    if not rows:
        return {name: 0.0 for name in FEET}
    return {
        name: sum(1 for row in rows if row[index] >= threshold) / len(rows)
        for index, name in enumerate(FEET)
    }


def support_shares(rows: list[list[float]], threshold: float = DEFAULT_THRESHOLD_N) -> dict[int, float]:
    """Fraction of frames with 0..6 feet loaded."""
    states = contact_states(rows, threshold)
    total = len(states) or 1
    return {count: sum(1 for s in states if len(s) == count) / total for count in range(7)}


def tripod_shares(rows: list[list[float]], threshold: float = DEFAULT_THRESHOLD_N) -> dict[str, float]:
    """The two alternating tripods, and every other three-foot combination."""
    first = {"LF", "RM", "LR"}
    second = {"RF", "LM", "RR"}
    states = contact_states(rows, threshold)
    shares = {"LF+RM+LR": 0.0, "RF+LM+RR": 0.0, "other three-foot": 0.0, "both tripods down": 0.0}
    total = len(states) or 1
    for state in states:
        present = set(state)
        if len(present) == 3 and present == first:
            shares["LF+RM+LR"] += 1
        elif len(present) == 3 and present == second:
            shares["RF+LM+RR"] += 1
        elif len(present) == 3:
            shares["other three-foot"] += 1
        elif len(present) > 3 and first <= present and second <= present:
            shares["both tripods down"] += 1
    return {name: count / total for name, count in shares.items()}


def step_frequency(times: list[float], rows: list[list[float]], threshold: float = DEFAULT_THRESHOLD_N) -> float:
    """Walking frequency in Hz, counted from the tripod alternation.

    One full cycle is one alternation from tripod A to tripod B and back, so the
    number of completed cycles is half the number of transitions.
    """
    if len(times) < 3:
        return 0.0
    first = {"LF", "RM", "LR"}
    swaps = []
    previous = None
    for index, state in enumerate(contact_states(rows, threshold)):
        present = set(state)
        label = "A" if present == first else ("B" if present == {"RF", "LM", "RR"} else None)
        if label is None:
            continue                      # a two-foot transition frame says nothing about which
        if previous is not None and label != previous:
            swaps.append(index)           # tripod changed: count the first frame of the new one
        previous = label
    if len(swaps) < 2:
        return 0.0
    # One cycle is two alternations, and the cycle count is measured between the
    # first and last alternation -- measuring it to the end of the recording adds
    # whatever fraction of a cycle the file happened to stop in (1.98 s of a
    # 0.32 s cycle reads as 3.03 Hz instead of 3.13).
    cycles = (len(swaps) - 1) / 2
    span = times[swaps[-1]] - times[swaps[0]]
    if span <= 0:
        return 0.0
    return cycles / span


def report(path: Path, threshold: float = DEFAULT_THRESHOLD_N) -> dict:
    times, rows = read_forces(path)
    return {
        "file": str(path),
        "frames": len(rows),
        "seconds": round(times[-1] - times[0], 3) if times else 0.0,
        "threshold_n": threshold,
        "duty": {name: round(value, 3) for name, value in duty(rows, threshold).items()},
        "support": {count: round(value, 3) for count, value in support_shares(rows, threshold).items()},
        "tripods": {name: round(value, 3) for name, value in tripod_shares(rows, threshold).items()},
        "step_hz": round(step_frequency(times, rows, threshold), 2),
    }


def main(argv: list[str] | None = None) -> int:
    import json

    argv = list(sys.argv[1:] if argv is None else argv)
    threshold = DEFAULT_THRESHOLD_N
    if "--threshold" in argv:
        index = argv.index("--threshold")
        if index + 1 >= len(argv):
            print("usage: gait_stats.py MEASURE_CSV [--threshold N]", file=sys.stderr)
            return 2
        try:
            threshold = float(argv[index + 1])
        except ValueError:
            print(f"gait_stats.py: --threshold wants a number of newtons, "
                  f"not {argv[index + 1]!r}", file=sys.stderr)
            return 2
        # `nan` 让每个 `force >= threshold` 都为假（占空比全 0、步频 0.0）；`0` 和负数
        # 让每只脚永远着地（占空比 100%、单足支撑 100%）——都会打印出一份
        # 看着像结果的报告。
        if not math.isfinite(threshold) or threshold <= 0:
            print(f"gait_stats.py: --threshold must be a finite number of newtons "
                  f"greater than zero, not {argv[index + 1]!r}", file=sys.stderr)
            return 2
        del argv[index:index + 2]
    if len(argv) != 1:
        print("usage: gait_stats.py MEASURE_CSV [--threshold N]", file=sys.stderr)
        return 2
    print(json.dumps(report(Path(argv[0]), threshold), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
