"""`--set path=value`: what it means, and what it refuses to guess."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from mjrl.overrides import OverrideError, apply_assignments, parse_value


def _cfg(**overrides):
    """A stand-in for an environment config: attributes, dicts and lists."""
    base = dict(
        events={
            "foot_friction": SimpleNamespace(params={"ranges": (0.5, 1.5), "operation": "abs"}),
            "reset_base": SimpleNamespace(params={"pose_range": {"x": (-0.5, 0.5)}}),
        },
        rewards={"track_linear_velocity": SimpleNamespace(weight=2.0)},
        episode_length_s=20.0,
        measured_points=[0.0, 0.05, 0.1],
        decimation=10,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_a_literal_value_keeps_its_type():
    assert parse_value("0.4") == 0.4
    assert parse_value("(0.5, 1.5)") == (0.5, 1.5)
    assert parse_value("True") is True
    assert parse_value('"plane"') == "plane"


def test_a_bare_word_is_a_string():
    assert parse_value("plane") == "plane"


def test_it_writes_a_dataclass_field():
    cfg = _cfg()
    assert apply_assignments(cfg, ["episode_length_s=1e9"]) == ["episode_length_s"]
    assert cfg.episode_length_s == 1e9


def test_it_writes_through_a_dict_and_an_object():
    cfg = _cfg()
    apply_assignments(cfg, ["rewards.track_linear_velocity.weight=0.0"])
    assert cfg.rewards["track_linear_velocity"].weight == 0.0


def test_it_writes_a_value_inside_a_dict_of_params():
    cfg = _cfg()
    apply_assignments(cfg, ["events.foot_friction.params.ranges=(1.0, 1.0)"])
    assert cfg.events["foot_friction"].params["ranges"] == (1.0, 1.0)


def test_it_reaches_into_a_list_by_index():
    cfg = _cfg()
    apply_assignments(cfg, ["measured_points.1=0.08"])
    assert cfg.measured_points == [0.0, 0.08, 0.1]


def test_a_term_replay_dropped_is_lifted_from_the_training_config():
    cfg = _cfg()
    training = _cfg(
        events={
            "push_robot": SimpleNamespace(params={"velocity_range": {"x": (-0.2, 0.2)}}),
        }
    )
    apply_assignments(
        cfg,
        ["events.push_robot.params.velocity_range.x=(-0.5, 0.5)"],
        fallback_factory=lambda: training,
    )
    assert cfg.events["push_robot"].params["velocity_range"]["x"] == (-0.5, 0.5)


def test_the_training_config_is_built_only_when_it_is_needed():
    calls = []

    def factory():
        calls.append(1)
        return _cfg()

    cfg = _cfg()
    apply_assignments(cfg, ["decimation=5", "episode_length_s=1.0"], fallback_factory=factory)
    assert calls == []


def test_an_unknown_field_names_the_path():
    with pytest.raises(OverrideError, match="events.nope"):
        apply_assignments(_cfg(), ["events.nope.params.x=1.0"])

    @dataclass(slots=True)
    class Locked:
        kept: float = 1.0

    with pytest.raises(OverrideError, match="no settable field 'nope'"):
        apply_assignments(Locked(), ["nope=1.0"])


def test_an_assignment_without_a_value_is_refused():
    with pytest.raises(OverrideError, match="PATH=VALUE"):
        apply_assignments(_cfg(), ["episode_length_s"])


def test_an_empty_path_is_refused():
    with pytest.raises(OverrideError, match="PATH=VALUE"):
        apply_assignments(_cfg(), ["=1.0"])


def test_nothing_to_apply_changes_nothing():
    cfg = _cfg()
    assert apply_assignments(cfg, []) == []
    assert cfg.episode_length_s == 20.0


def test_a_misspelt_field_is_an_error_not_a_new_attribute():
    """`--set episode_lenght_s=1` 曾经静默造一个新属性，打印成功、什么都没改。"""
    cfg = _cfg()
    with pytest.raises(OverrideError, match="no settable field 'episode_lenght_s'"):
        apply_assignments(cfg, ["episode_lenght_s=1.0"])
    assert not hasattr(cfg, "episode_lenght_s")


def test_a_misspelt_params_key_is_an_error_not_a_new_key():
    cfg = _cfg()
    with pytest.raises(OverrideError, match="events.foot_friction.params.range: no key 'range'"):
        apply_assignments(cfg, ["events.foot_friction.params.range=(1.0, 1.0)"])
    assert "range" not in cfg.events["foot_friction"].params


def test_a_tuple_target_says_so_instead_of_naming_an_attribute():
    cfg = _cfg(points=(1.0, 2.0))
    with pytest.raises(OverrideError, match="sits in a tuple"):
        apply_assignments(cfg, ["points.0=5.0"])
