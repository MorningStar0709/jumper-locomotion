"""The stall termination: a command to travel plus a body that goes nowhere.

**The failure this exists for.** On 3 cm stairs a flat-initialised policy's reward
climbed 44.3 -> 62.3 while its tracking error climbed 0.297 -> 0.433, and the
replay measured 0.010 m/s where the baseline managed 0.252. On 4 cm the same shape
(41.9 -> 59.6, error 0.254 -> 0.425) and it also overwrote the 3 cm gait the arm
had arrived with (0.415 -> 0.035). Both runs learned that the cheapest answer to
"walk" on ground it cannot walk is to stand, because nothing in the MDP called
that a failure: the tracking reward is `exp(-err^2/std^2)` and stays positive, the
posture terms do not require displacement, `standing_sway` is gated to *near-zero*
commands and is therefore invisible under a walking command, and the only
termination is a 50 degree tilt.

**What it measures, and the version this replaced.** The rule is net displacement
inside a window, not instantaneous speed. Speed was tried first and it never fired
on its own motivating case: the baseline on 4 cm averages 0.045 m/s but rocks
through 0-0.15, clearing a speed counter every few steps. A body that rocks in
place travels nowhere, and that is what the term now asks.

**What it does to evaluation.** `episode_resets` stops being a fixed property of
the ground and becomes a measurement: a stalled policy now ends its episodes early,
which is the discrete quantity §二十 asked for on ground where the continuous speed
cannot separate two policies at 7 or 14 seeds.
"""

from __future__ import annotations

import pytest
import torch

from tasks.jumper.common.mdp import terminations

#: Control period at 50 Hz, the cadence every run in this repository trains at.
STEP_DT = 0.02

#: The window the term judges travel over, in control steps.
WINDOW = int(round(terminations.STALL_SECONDS / STEP_DT))


class _FakeData:
    def __init__(self, position: torch.Tensor):
        self.root_link_pos_w = position


class _FakeAsset:
    def __init__(self, position: torch.Tensor):
        self.data = _FakeData(position)


class _FakeScene:
    def __init__(self, position: torch.Tensor):
        self._asset = _FakeAsset(position)

    def __getitem__(self, _name: str) -> _FakeAsset:
        return self._asset


class _FakeCommandManager:
    def __init__(self, command: torch.Tensor | None):
        self._command = command

    def get_command(self, _name: str) -> torch.Tensor | None:
        return self._command


class _FakeEnv:
    """Only what the term reads: position, command, episode age, control period."""

    def __init__(self, command: torch.Tensor | None, position: torch.Tensor, age: int = 1):
        self.scene = _FakeScene(position)
        self.command_manager = _FakeCommandManager(command)
        self.episode_length_buf = torch.full((position.shape[0],), age, dtype=torch.long)
        self.step_dt = STEP_DT


def _env(vx: float, wz: float = 0.0, x: float = 0.0, age: int = 1) -> _FakeEnv:
    return _FakeEnv(
        command=torch.tensor([[vx, 0.0, wz]]),
        position=torch.tensor([[x, 0.0, 0.25]]),
        age=age,
    )


def _move(env: _FakeEnv, dx: float) -> None:
    env.scene._asset.data.root_link_pos_w = torch.tensor(
        [[env.scene._asset.data.root_link_pos_w[0, 0] + dx, 0.0, 0.25]]
    )


def _terminates(env: _FakeEnv) -> bool:
    """Advance one control step the way the environment does, then ask the term.

    `manager_based_rl_env.step` increments the episode age *before* the termination
    manager runs, and a reset sets that age to 0 afterwards, so the term sees 1 on
    the first step of a new episode.
    """
    env.episode_length_buf += 1
    return bool(terminations.stalled_under_command(env).item())


def test_a_command_to_stand_is_not_a_stall():
    """No linear command means no expectation of travel — standing is the answer."""
    env = _env(vx=0.0)
    for _ in range(200):
        assert _terminates(env) is False


def test_a_pure_turn_in_place_is_not_a_stall():
    """A yaw command with no linear part asks for a pivot; the body may not move."""
    env = _env(vx=0.0, wz=0.75)
    for _ in range(200):
        assert _terminates(env) is False


def test_a_body_that_never_moves_fails_the_window():
    """74 still steps are a pause; the 75th is a stall."""
    env = _env(vx=0.5)
    for step in range(WINDOW - 1):
        assert _terminates(env) is False, f"terminated at step {step}"
    assert _terminates(env) is True


def test_travelling_the_threshold_inside_the_window_is_not_a_stall():
    """0.5 m/s covers 0.75 m in the window against a 0.10 m threshold."""
    env = _env(vx=0.5)
    for _ in range(WINDOW - 1):
        assert _terminates(env) is False
    _move(env, 0.15)
    assert _terminates(env) is False
    # The anchor moved with the body, so the next window starts clean: the step
    # that re-anchored already counted as the first of the new run.
    for _ in range(WINDOW - 2):
        assert _terminates(env) is False


def test_rocking_in_place_is_a_stall_however_fast_the_body_sways():
    """The 4 cm baseline averages 0.045 m/s by rocking ±0.03 m — and goes nowhere."""
    env = _env(vx=0.5)
    for step in range(WINDOW - 1):
        _move(env, 0.03 if step % 2 else -0.03)
        assert _terminates(env) is False, f"terminated at step {step}"
    assert _terminates(env) is True


def test_the_minimum_command_is_the_last_command_that_asks_for_nothing():
    """`> min_command`, in both layers: exactly 0.20 m/s is not a command to travel.

    The comparison is strict on purpose and is the same comparison `mjrl.replay`
    makes, so the termination and the reported `stalled_fraction` cannot disagree
    about which commands it applies to.
    """
    assert terminations.STALL_MIN_COMMAND == 0.20
    at = _env(vx=0.20)
    for _ in range(WINDOW * 3):
        assert _terminates(at) is False, "0.20 m/s is the boundary, not past it"
    past = _env(vx=0.201)
    for _ in range(WINDOW - 1):
        assert _terminates(past) is False
    assert _terminates(past) is True, "0.201 m/s asks for travel"


def test_staying_inside_the_radius_keeps_the_run_alive_on_both_sides_of_it():
    """The radius is `<= max_travel` for a stall: 0.10 m is still inside, 0.11 is out."""
    assert terminations.STALL_TRAVEL == 0.10
    inside = _env(vx=0.5)
    for _ in range(WINDOW - 1):
        assert _terminates(inside) is False
    _move(inside, terminations.STALL_TRAVEL)
    assert _terminates(inside) is True, "0.10 m is the edge of the radius, not outside it"
    outside = _env(vx=0.5)
    for _ in range(WINDOW - 1):
        assert _terminates(outside) is False
    _move(outside, terminations.STALL_TRAVEL + 0.01)
    assert _terminates(outside) is False, "0.11 m of travel re-anchored the window"


def test_a_new_episode_does_not_inherit_the_last_one_s_stall():
    """A respawned environment starts clean even if the previous one was stuck."""
    env = _env(vx=0.5)
    for _ in range(WINDOW - 1):
        assert _terminates(env) is False

    env.episode_length_buf = torch.zeros(1, dtype=torch.long)
    assert _terminates(env) is False

    for _ in range(WINDOW - 2):
        assert _terminates(env) is False
    assert _terminates(env) is True


def test_a_task_without_a_velocity_command_cannot_stall():
    """`jumper.swing` has no `twist` command; the rule must not break it."""
    env = _FakeEnv(command=None, position=torch.tensor([[0.0, 0.0, 0.25]]))
    for _ in range(200):
        assert _terminates(env) is False


def test_the_task_installs_the_stall_termination():
    """The rule has to be on the task, not only in the term's own file."""
    from tasks import load_env_cfg

    cfg = load_env_cfg("jumper.tripod", task_args={})
    assert "stalled" in cfg.terminations
    term = cfg.terminations["stalled"]
    assert term.func is terminations.stalled_under_command
    # Exactly these keys: the window is not a parameter any more (both layers derive
    # it from `step_dt` through `mjrl.stall.stall_window`), so a second key here
    # would be a second derivation waiting to disagree.
    assert term.params == {"command_name": "twist"}


def test_the_five_foot_task_opts_out():
    """Its carried-arm hold is exercised by stepping zero actions: a passive robot
    under a walking command is exactly what this rule would end."""
    from tasks import load_env_cfg

    cfg = load_env_cfg("jumper.five_foot", task_args={})
    assert "stalled" not in cfg.terminations

def test_the_stall_term_is_training_only():
    """A replay must not end an episode at a stall; it has to measure one.

    Truncating in replay redefines every archived number: `actual_velocity_mean`
    becomes "how it walked before the first stall" instead of "how it walked for
    120 s", and the 24 sections already written used the second meaning.
    """
    import tasks

    training = tasks.load_env_cfg("jumper.tripod", task_args={})
    replay = tasks.load_env_cfg("jumper.tripod", play=True, task_args={})
    assert "stalled" in training.terminations
    assert "stalled" not in replay.terminations
    assert "fell_over" in replay.terminations


def test_a_control_period_that_collapses_the_window_is_refused_where_it_runs():
    """The window is checked against the period the task runs at, not the builder's.

    `jumper.posture` changes `decimation` and `timestep` after `velocity_env_cfg`
    returns, so a build-time check validated a period that task never uses. The
    term therefore checks on the way in, against the period it really runs at: the
    window is `round(seconds / step_dt)`, so with the default 1.5 s it collapses to
    one step only past 1.0 s per control step (at exactly 1.0 s it is still 2). Two
    seconds is over that line.
    """
    env = _env(0.5)
    env.step_dt = 2.0

    with pytest.raises(ValueError, match="more than one step"):
        terminations.stalled_under_command(env)
