"""Draw the two figures the README uses.

Both read numbers straight from the tables in ``docs/reports/``: nothing here is
recomputed, so a figure cannot drift from the report it cites. Re-run after editing a
table below; the comment on each block names the table it copies.

    python docs/figures/draw.py            # writes docs/media/*.png
"""

from __future__ import annotations

import itertools
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent.parent / "media"

# research-terrain-ladder-20261101.md §二: median actual/commanded speed over 5 seeds,
# 6000 steps, straight command, and the same policy's error multiple of its own flat error.
# Pairs are (speed ratio, MAE multiple). A ground missing from a policy's dict is a rung that
# policy never reached — deliberate absence, not a zero. Every key must appear in GROUNDS,
# which is checked below, so a mis-keyed rung raises instead of quietly shortening a curve.
GROUNDS = ["1 m pave", "5 mm", "10 mm", "25 mm", "35 mm", "50 mm", "65 mm", "80 mm", "100 mm"]
POLICIES: dict[str, dict[str, tuple[float, float] | None]] = {
    "baseline": {
        "1 m pave": (0.94, 1.0), "5 mm": (0.92, 1.4), "10 mm": (0.90, 1.7),
        "25 mm": (0.87, 2.1), "35 mm": (0.81, 3.1), "50 mm": (0.63, 6.1),
    },
    "trained to 2 cm stairs": {
        "1 m pave": (0.97, 1.0), "5 mm": (0.94, 1.9), "10 mm": (0.96, 1.3),
        "25 mm": (0.94, 1.8), "35 mm": (0.94, 2.0), "50 mm": (0.90, 3.0),
        "65 mm": (0.86, 4.3), "80 mm": (0.77, 6.7), "100 mm": (0.60, 11.0),
    },
    "trained on 5 mm, 3000 it": {
        "1 m pave": (0.96, 1.0), "5 mm": (0.93, 1.6), "10 mm": (0.94, 1.6),
        "25 mm": (0.91, 2.1), "35 mm": (0.88, 3.0), "50 mm": (0.71, 6.7),
    },
    "trained on a mixture": {
        "1 m pave": (0.97, 1.0), "5 mm": (0.94, 1.7), "10 mm": (0.94, 1.6),
        "25 mm": (0.94, 1.7), "35 mm": (0.91, 2.5), "50 mm": (0.85, 3.8),
        "65 mm": (0.74, 6.6), "80 mm": (0.55, 10.9), "100 mm": (0.06, 22.6),
    },
}
# The gate: usable above 0.85 of the commanded speed and no worse than 3x the policy's own
# flat-ground error (research-terrain-ladder-20261101.md §二, the footnote on the fixed 3x).
HOLDS = 0.85
MAE_LIMIT = 3.0

# A misspelled ground would drop a point and shorten a curve with no error, so the table is
# checked once at import: unknown keys fail loudly, and a policy cannot be added without a
# marker (the marks list is cycled rather than zipped, which would truncate silently).
for _name, _rows in POLICIES.items():
    _unknown = sorted(set(_rows) - set(GROUNDS))
    if _unknown:
        raise ValueError(f"{_name} names grounds that are not in GROUNDS: {_unknown}")


def ceiling() -> None:
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8.2, 6.6), sharex=True,
                                      gridspec_kw={"height_ratios": [1.35, 1]})
    markers = itertools.cycle(["o", "s", "^", "D", "v", "P"])
    for name, rows in POLICIES.items():
        marker = next(markers)
        xs, ys = [], []
        for i, ground in enumerate(GROUNDS):
            value = rows.get(ground)
            if value is None:
                continue
            xs.append(i)
            ys.append(value[0])
        top.plot(xs, ys, marker=marker, linewidth=1.8, markersize=6, label=name)
        xs, ys = [], []
        for i, ground in enumerate(GROUNDS):
            value = rows.get(ground)
            if value is None:
                continue
            xs.append(i)
            ys.append(value[1])
        bottom.plot(xs, ys, marker=marker, linewidth=1.4, markersize=5,
                    linestyle="--", label=name)

    top.axhline(HOLDS, color="#b45309", linewidth=1.2, linestyle=":")
    top.annotate("the gate: 0.85 of the commanded speed", xy=(0.08, HOLDS), xytext=(0.08, 0.44),
                 fontsize=9, color="#b45309")
    top.set_ylabel("actual / commanded speed")
    top.set_ylim(0, 1.05)
    top.legend(fontsize=8.5, frameon=False, loc="lower left")
    top.set_title("Where each policy stops walking", loc="left", fontsize=11)

    bottom.axhline(MAE_LIMIT, color="#b45309", linewidth=1.2, linestyle=":")
    bottom.annotate("3x its own flat error", xy=(0.08, MAE_LIMIT), xytext=(0.08, 3.4),
                    fontsize=9, color="#b45309")
    bottom.set_ylabel("velocity error,\nmultiple of flat")
    bottom.set_yscale("log")
    bottom.set_yticks([1, 2, 3, 5, 7, 11, 20])
    bottom.set_yticklabels(["1x", "2x", "3x", "5x", "7x", "11x", "20x"])
    bottom.set_xlabel("ground: bump amplitude (mm), and 1 m paving at the left")
    bottom.set_xticks(range(len(GROUNDS)))
    bottom.set_xticklabels(["pave", "5", "10", "25", "35", "50", "65", "80", "100"])
    bottom.legend(fontsize=8.5, frameon=False, loc="upper left")

    for axis in (top, bottom):
        axis.grid(alpha=0.18, linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "ceiling-curves.png", dpi=170)
    plt.close(fig)


# research-heading-drift-20261102.md and research-heading-loop-20261102.md: lateral
# deviation after 10 m on paving, median over 5 seeds and four policies, open loop then
# closed. Drawn to scale, not a recorded trajectory: the paths are straight lines whose
# spread is the measured median.
DRIFT_OPEN = 1.84
DRIFT_CLOSED = 0.096


def drift() -> None:
    fig, axis = plt.subplots(figsize=(8.2, 3.4))
    axis.plot([0, 10], [0, 0], color="#9ca3af", linewidth=1.0, linestyle=":")
    for value, colour, tail in (
        (DRIFT_OPEN, "#9a3412", f"open loop — {DRIFT_OPEN:.2f} m off"),
        (DRIFT_CLOSED, "#15803d", f"heading hold — {DRIFT_CLOSED:.3f} m off"),
    ):
        axis.plot([0, 10], [0, value], color=colour, linewidth=2.4, solid_capstyle="round")
        axis.annotate(tail, xy=(10.2, value), fontsize=9.5, color=colour, va="center")
    axis.annotate("median of 4 policies x 5 seeds, straight command at 0.5 m/s",
                  xy=(0, -0.72), fontsize=8.5, color="#6b7280")
    axis.annotate("intended line", xy=(1.2, 0), xytext=(1.2, -0.42), fontsize=9,
                  color="#6b7280")
    axis.set_xlim(0, 15.0)
    axis.set_ylim(-0.8, 2.1)
    axis.set_xlabel("distance walked (m)")
    axis.set_ylabel("lateral deviation (m)")
    axis.set_title("10 m straight, same command, same policy family", loc="left", fontsize=11)
    axis.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
    axis.grid(axis="x", alpha=0.18, linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "heading-drift.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    ceiling()
    drift()
    for name in ("ceiling-curves.png", "heading-drift.png"):
        print(name, (OUT / name).stat().st_size)
    # The table check above runs at import: an unknown ground name raises before anything is
    # drawn, so a mis-keyed rung cannot reach a figure.