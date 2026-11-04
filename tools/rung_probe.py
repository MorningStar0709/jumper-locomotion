"""The rung probe reports whether a checkpoint can walk a ground before a run is spent on it.

**What it says, and what it does not.** It answers "does the incumbent policy still
walk here", using the repository's own ceiling rule (`docs/EXPLORATION.md`): median
actual speed under 0.85x the command *and* vx error above twice the same policy's
error on flat ground. It does **not** predict whether training on that ground will
help. The 4 cm stair rung disproved that: the 3 cm arm walks it at 0.37 m/s and
zero stalls, and 500 iterations on it still destroyed the gait. A replay-only probe
cannot see training dynamics, so its answer is a floor, not a verdict on worth.

**The stall count rides along** because it is the discrete signal §二十 asked for:
`stalled_fraction` is the share of a 120 s replay spent under a travel command
without leaving a 10 cm radius (`mjrl.stall`). It separates policies the continuous
speed cannot resolve at 7 or 14 seeds on 5 mm ground, where two policies both read
"about 0.04 m/s" and differ only in whether they move at all.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

#: Ceiling from `docs/EXPLORATION.md`: speed under this share of the command.
CEILING_RATIO = 0.85

#: Ratio below which the policy is not walking the ground at all.
#:
#: Anchors on the ladder's own numbers: mix at 100 mm reads 0.06, base at 80 mm 0.23,
#: base at 50 mm 0.63. The rungs that were *marginal but paid off* -- 3 cm at 0.83 --
#: sit above this, and so does 4 cm at 0.73, which destroyed the policy. That gap is
#: why this threshold refuses collapsed ground only: no replay-only number separated
#: the rung that paid from the rung that did not, so the probe does not pretend to.
COLLAPSE_RATIO = 0.60

#: Ceiling from the same rule: vx error above this multiple of the policy's own flat error.
CEILING_MULTIPLE = 2.0

#: Seeds on flat ground. Flat replays are tight (MAD 4-8 % on 2 cm, less on paving),
#: so three are enough to place the reference; the candidate ground gets more.
FLAT_SEEDS = 3


def plan(*, ground: str, value: float, checkpoint: Path, seeds: int, steps: int,
         command: tuple[float, float, float], out: Path,
         task: str = "jumper.tripod") -> list[dict]:
    """One replay per seed, each writing its own evaluation JSON.

    The last `FLAT_SEEDS` entries are the flat reference: `--terrain-roughness-mm 0`
    with the same switches, which builds the 1 m paving grid every report calls
    "flat". The label, the `seed` field and `--seed` all come from the same number,
    so a line in the log can be replayed by hand.
    """
    switch = ("--terrain-steps-m" if ground == "steps" else "--terrain-roughness-mm")
    runs = []
    for index in range(seeds + FLAT_SEEDS):
        flat = index >= seeds
        number = index - seeds if flat else index
        label = f"flat-s{number}" if flat else f"ground-s{number}"
        target = out.with_name(f"{out.stem}-{label}{out.suffix}")
        surface = ["--terrain-roughness-mm", "0"] if flat else [switch, str(value)]
        runs.append({
            "flat": flat,
            "seed": number,
            "evaluation": target,
            "argv": [
                "--task", task, "--checkpoint", str(checkpoint), "--seed", str(number),
                "--steps", str(steps), "--command", *[str(x) for x in command],
                "--terrain", *surface, "--terrain-seed", "0",
                "--evaluation-out", str(target),
            ],
        })
    return runs


def _median_stalls(reports: list[dict]) -> float | None:
    """Share of the replay spent stalled, or None when the rule could not fire.

    A turn-in-place replay commands no travel, so the rule never applies and the
    replay reports None. Averaging that in as 0.0 would put a number in a table
    next to real duty cycles.
    """
    values = [r["stalled_fraction"] for r in reports if r["stalled_fraction"] is not None]
    return statistics.median(values) if values else None


def verdict(reports: list[dict], flat: list[dict], command: float) -> dict:
    """Apply the ceiling rule to medians; `collapsed` means the policy cannot hold this ground."""
    ratio = statistics.median(r["actual_velocity_mean"][0] for r in reports) / command
    error = statistics.median(r["mean_absolute_error"][0] for r in reports)
    flat_error = statistics.median(r["mean_absolute_error"][0] for r in flat)
    multiple = error / flat_error if flat_error else float("inf")
    if ratio < COLLAPSE_RATIO:
        label = "collapsed"
    elif ratio < CEILING_RATIO and multiple > CEILING_MULTIPLE:
        label = "marginal"
    else:
        label = "holds"
    return {
        "ratio": ratio,
        "error": error,
        "flat_error": flat_error,
        "multiple": multiple,
        "stalled": _median_stalls(reports),
        "verdict": label,
    }


def main(argv: list[str] | None = None, run=subprocess.run) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="a policy that walks the neighbourhood of this rung")
    parser.add_argument("--ground", choices=("steps", "rough"), required=True)
    parser.add_argument("--value", type=float, required=True,
                        help="riser height in metres (steps) or amplitude in millimetres (rough)")
    parser.add_argument("--seeds", type=int, default=7)
    parser.add_argument("--task", default="jumper.tripod",
                        help="task that owns the --terrain switches (default jumper.tripod)")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--command", type=float, nargs=3, default=(0.5, 0.0, 0.0))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.seeds < 1:
        parser.error("--seeds must be at least 1: the ground sample is what the verdict is computed from")
    if args.command[0] <= 0.0:
        parser.error("--command VX must be positive: the ratio is measured against it")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    reports, flat = [], []
    for item in plan(ground=args.ground, value=args.value, checkpoint=args.checkpoint,
                     seeds=args.seeds, steps=args.steps, command=tuple(args.command),
                     out=args.out, task=args.task):
        completed = run([sys.executable, "scripts/play.py", "--headless", *item["argv"]],
                        capture_output=True, text=True)
        if completed.returncode != 0:
            print(f"[{'flat' if item['flat'] else 'ground'} {item['seed']}] "
                  f"rc={completed.returncode}\n{completed.stderr[-800:]}", file=sys.stderr)
            # `3` is this tool's own "collapsed" code; a replay that died with it
            # must not be read as a verdict.
            return 1 if completed.returncode == 3 else completed.returncode
        report = json.loads(item["evaluation"].read_text())
        (flat if item["flat"] else reports).append(report)
        stalled = report["stalled_fraction"]
        share = "n/a (no travel commanded)" if stalled is None else f"{stalled:.1%}"
        print(f"[{'flat' if item['flat'] else 'ground'} {item['seed']}] "
              f"vx {report['actual_velocity_mean'][0]:.3f} m/s "
              f"MAE {report['mean_absolute_error'][0]:.4f} stalled {share}", flush=True)

    answer = verdict(reports, flat, args.command[0])
    summary = {"ground": args.ground, "value": args.value, "seeds": args.seeds,
               "task": args.task, "checkpoint": str(args.checkpoint),
               "ceiling_ratio": CEILING_RATIO,
               "ceiling_multiple": CEILING_MULTIPLE, **answer}
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    share = "n/a" if answer["stalled"] is None else f"{answer['stalled']:.1%}"
    print(f"[{answer['verdict']}] {args.ground} {args.value}: "
          f"ratio {answer['ratio']:.2f} ({CEILING_RATIO:.2f} needed), "
          f"MAE {answer['multiple']:.1f}x own flat, stalled {share}", flush=True)
    return 3 if answer["verdict"] == "collapsed" else 0


if __name__ == "__main__":
    raise SystemExit(main())