"""Terminations this repository adds to mjlab's.

Mjlab ships four: time out, bad orientation, root height below a minimum, and NaN
detection. None of them can express "the command asked it to walk and it went
nowhere" — and that is the failure this project keeps producing on ground it cannot
handle (see `tasks/jumper/common/velocity_env.py` for where the rule is installed).
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import torch

from mjrl.stall import STALL_MIN_COMMAND, STALL_SECONDS, STALL_TRAVEL, stall_window

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv



_WARNED: set[str] = set()


def warn_once(message: str) -> None:
    """Say something once per process: a term runs every step, at every env."""
    if message in _WARNED:
        return
    _WARNED.add(message)
    print(f"[mjrl] warning: {message}", file=sys.stderr)


def stalled_under_command(
    env: "ManagerBasedRlEnv",
    command_name: str = "twist",
    min_command: float = STALL_MIN_COMMAND,
    max_travel: float = STALL_TRAVEL,
    seconds: float = STALL_SECONDS,
) -> torch.Tensor:
    """Terminate environments that were told to travel and went nowhere.

        anchor = position when the current run of no-travel began
        stalled = (|cmd[:, :2]| > min_command) and (|pos - anchor| <= max_travel)
        fail once `stalled` has held for `seconds` without a break

    **Why this is a termination and not a reward penalty.** A penalty would have to
    outweigh everything standing still still earns — the tracking rewards are
    `exp(-err^2/std^2)` and stay positive, the posture terms never ask for
    displacement, and `standing_sway` is gated to *near-zero* commands, so under a
    walking command it is exactly zero. That total is a function of the ground and
    the command, so the weight would be a guess. Ending the episode removes the rest
    of the reward outright, and says what actually happened: this deployment failed,
    at the same level as falling over.

    **The window, and where a run starts.** The step that (re)anchors already counts
    as the first step of a run, so an unbroken non-travel of `stall_window` steps
    ends the episode on that step; `mjrl.replay` counts from the same step, which is
    what makes the termination and the reported `stalled_fraction` the same rule.

    **The state.** Two tensors per environment, kept on `env`: the anchor position
    and a step counter. Both are reset by motion past `max_travel`, by a command
    that asks for no travel, and by a fresh episode, so a stall cannot leak across a
    respawn. Nothing else in the repository keeps per-term state, and this is the
    smallest form that measures an unbroken run rather than an average.
    """
    asset = env.scene["robot"]
    position = asset.data.root_link_pos_w[:, :2]
    try:
        command = env.command_manager.get_command(command_name)
    except KeyError:
        # `CommandManager.get_command` indexes its terms directly, so a task that
        # keeps some other command and drops this one raises rather than returning
        # None (`NullCommandManager`, used when `cfg.commands` is empty, does return
        # None). Only a task that keeps a real command manager and drops *this*
        # command gets here, which is what a typo in `command_name` looks like, so
        # it says so once. A task that empties `cfg.commands` altogether gets
        # `NullCommandManager` instead: its `get_command` returns None and the term
        # below returns all-False without a word. `jumper.swing` empties it, which
        # is why it passes `stall_gate=False` rather than relying on a warning.
        warn_once(f"the stall rule found no {command_name!r} command: it can never fire")
        command = None
    if command is None:
        return torch.zeros(position.shape[0], dtype=torch.bool, device=position.device)

    window = stall_window(env.step_dt, seconds)
    if window < 2:
        # One step is not a window: every commanded episode would end on its first
        # step and train on nothing. Checked here, against the period the task
        # actually runs at, because a task can change it after the builder returns.
        raise ValueError(
            f"the stall window is {window} step(s) at {env.step_dt:.4f} s per control "
            f"step: it must cover more than one step, or every commanded episode ends "
            f"on its first one"
        )
    commanded = torch.norm(command[:, :2], dim=1) > min_command

    steps = getattr(env, "_stall_steps", None)
    anchor = getattr(env, "_stall_anchor", None)
    if (
        steps is None
        or anchor is None
        or steps.shape != commanded.shape
        or steps.device != commanded.device
        or anchor.shape != position.shape
    ):
        steps = torch.zeros_like(commanded, dtype=torch.long)
        anchor = position.clone()

    # `manager_based_rl_env.step` increments the age before terminations are
    # computed, and a reset sets it to 0 afterwards -- so the first step of a new
    # episode is seen here as 1, not 0.
    fresh = env.episode_length_buf <= 1
    steps = torch.where(fresh, torch.zeros_like(steps), steps)
    anchor = torch.where(fresh.unsqueeze(1), position, anchor)

    travelled = torch.norm(position - anchor, dim=1) > max_travel
    steps = torch.where(travelled, torch.zeros_like(steps), steps)
    anchor = torch.where(travelled.unsqueeze(1), position, anchor)

    steps = torch.where(commanded, steps + 1, torch.zeros_like(steps))
    anchor = torch.where((~commanded).unsqueeze(1), position, anchor)

    env._stall_steps = steps
    env._stall_anchor = anchor

    return commanded & (steps >= window)