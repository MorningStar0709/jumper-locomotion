"""jumper.tripod -- tripod gait: 3 legs swinging / 3 in support, the fastest.

Tripod gait: two groups of legs alternate in antiphase, three swinging and
three in support at a time. The fastest of the three gaits, with the smallest
and least stable support polygon.

The robot definition and the environment skeleton live in `tasks/jumper/common/`.
This directory holds only what belongs to this task: its fixed name (the register
call below), its environment config (`env_cfg.py`), its hyper-parameters
(`rl_cfg.py`), and the MDP terms only it uses (`mdp/`, here its gait reward).
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.tripod",
    assets=JUMPER_ASSETS,
    description="jumper hexapod flat-ground velocity tracking; tripod gait, 3 swinging / 3 in support, fastest",
    tags=("locomotion", "jumper", "flat", "tripod"),
)


def cli_args(group) -> tuple[str, ...]:
    """This task's own command-line arguments; see `tasks.load_cli_args`.

    `--delay` is trainable here rather than in `env_cfg.py` because it decides
    what this task trains, and the run that wants it should say so on its own
    command line. Declaring an argument needs `argparse` and nothing else, which
    is why it lives in this always-imported module rather than beside the config.
    """
    group.add_argument(
        "--delay",
        action="store_true",
        help="command latency on every joint: targets arrive 1-4 physics steps late",
    )
    group.add_argument(
        "--terrain",
        action="store_true",
        help="walk on generated ground instead of the flat plane every result so far was measured on",
    )
    group.add_argument(
        "--terrain-roughness-mm",
        type=float,
        default=0.0,
        metavar="MM",
        help="height noise of that ground, in millimetres; 0 keeps it flat and is the L0 rung of the ladder",
    )
    group.add_argument(
        "--terrain-steps-m",
        type=float,
        default=0.0,
        metavar="M",
        help="riser height of a staircase that fills the whole grid, in metres, climbing outward from the spawn platform; foot_target_height is 0.025, so 0.02 is already near the ceiling. Give both this and --terrain-roughness-mm to train on a mixture: two columns, with the split set by --terrain-stairs-share (1:1 by default); that mixture cannot be replayed",
    )
    group.add_argument(
        "--terrain-stairs-share",
        type=float,
        default=0.5,
        metavar="SHARE",
        help="share of the training envs spawned on the staircase when both surfaces are asked for; the grid gives each surface one full tile either way; 0.5 is the inherited split, and lowering it asks whether the cost the mixed arm paid on flat ground came from the staircase eating half the training budget",
    )
    group.add_argument(
        "--terrain-seed",
        type=int,
        default=0,
        metavar="N",
        help="which generated ground --terrain builds; fixed by default so two rungs of the ladder are the same field",
    )
    group.add_argument(
        "--yaw-weight",
        type=float,
        default=2.0,
        metavar="W",
        help="price of the yaw tracking term, which `velocity_env.py` already narrows to yaw alone; 2.0 is the inherited value and every result before 2026-11-01 was measured at it. Raising it is how an arm asks whether the speed-for-heading exchange rate is a price the reward set or a property of the robot",
    )
    group.add_argument(
        "--terrain-scan",
        action="store_true",
        help="feed the terrain scan to the critic (not the actor) -- constant on flat ground, and worth its width as soon as --terrain adds roughness",
    )
    return (
        "delay",
        "yaw_weight",
        "terrain",
        "terrain_roughness_mm",
        "terrain_seed",
        "terrain_steps_m",
        "terrain_stairs_share",
        "terrain_scan",
    )
