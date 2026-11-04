"""A replay starts at the rate the task trained at.

The default used to be a hard 1000 Hz, and every published baseline was measured
there: 7% optimistic on straight, 10% on turning, 30% on the mixed case, and it
made a delay-trained policy's turning deficit disappear. A default that silently
changes the world is not a default.
"""

from __future__ import annotations

from types import SimpleNamespace

import tasks

import play


def a_task_cfg():
    return tasks.load_env_cfg("jumper.tripod")


def test_no_flag_means_the_tasks_own_rate():
    """`None` is the whole point: the rate is the task's, and only the task knows it."""
    args = SimpleNamespace(physics_hz=None)

    assert play._physics_hz(args, a_task_cfg()) == 200.0


def test_the_flag_still_overrides():
    args = SimpleNamespace(physics_hz=1000.0)

    assert play._physics_hz(args, a_task_cfg()) == 1000.0


def test_the_default_is_the_rate_the_task_trained_at():
    cfg = a_task_cfg()

    assert play._default_physics_hz(cfg) == 200.0
    assert cfg.sim.mujoco.timestep == 0.005


def test_setting_the_tasks_own_rate_changes_nothing():
    cfg = a_task_cfg()

    play._set_physics_rate(cfg, play._default_physics_hz(cfg))

    assert cfg.sim.mujoco.timestep == 0.005
    assert cfg.decimation == 4


def test_an_explicit_rate_still_wins():
    """The finer grid is still reachable — as an experiment, not as the score."""
    cfg = a_task_cfg()

    play._set_physics_rate(cfg, 1000.0)

    assert cfg.sim.mujoco.timestep == 0.001
    assert cfg.decimation == 20
    assert cfg.sim.mujoco.timestep * cfg.decimation == 0.02
