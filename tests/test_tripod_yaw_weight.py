"""`--yaw-weight` on `jumper.tripod`: heading's price is a knob, and default is 2.0.

Every arm trained on rough ground this week traded heading for forward speed: on
pavement `S2cm-ext` gained 47% on `vx` and lost 31% on yaw MAE, and the three
continuous-roughness arms lost 39% on heading straight while gaining nothing on
their own ground. The open question is whether that exchange rate is a property
of the robot or a price the reward set -- `track_angular_velocity` scored 0.80 to
1.21 against `track_linear_velocity`'s 1.26 to 1.60 on the runs of 2026-10-31, so
the heading term pays, and paying more may buy a different point on the frontier.

`five_foot` already sets this weight to 5.0 outright. This task gets it as an
argument instead, so the arm that asks for a different price says so on its own
command line and a manifest can reproduce it. The default stays the value
inherited from `rl/mjlab/tasks/velocity/velocity_env_cfg.py:286` (2.0): a knob
whose default differs from the thing it replaces silently changes every run that
does not mention it.
"""

from __future__ import annotations

import pytest

import tasks


def yaw_weight(**task_args: object) -> float:
    cfg = tasks.load_env_cfg("jumper.tripod", task_args=task_args or None)
    return cfg.rewards["track_angular_velocity"].weight


def test_heading_costs_what_it_cost_before() -> None:
    """默认值必须等于上游的 2.0，否则所有没提这一项的 run 都被悄悄改了。"""
    assert yaw_weight() == pytest.approx(2.0)


def test_asking_for_a_different_price_moves_it() -> None:
    assert yaw_weight(yaw_weight=5.0) == pytest.approx(5.0)


def test_zero_is_allowed_and_negative_is_not() -> None:
    """0 是「航向不要钱」的合法实验；负数会奖励偏航误差，那不是实验。"""
    assert yaw_weight(yaw_weight=0.0) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="yaw"):
        yaw_weight(yaw_weight=-1.0)


def test_the_switch_is_this_tasks_own_argument() -> None:
    from tasks.jumper.tripod import cli_args

    class _Parser:
        def __init__(self) -> None:
            self.names: set[str] = set()

        def add_argument(self, name: str, *args: object, **kwargs: object) -> None:
            self.names.add(name)

    parser = _Parser()
    dests = cli_args(parser)
    assert "yaw_weight" in dests
    assert "--yaw-weight" in parser.names