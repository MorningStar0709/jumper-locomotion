"""The scripted pad, and the flag that reaches it."""

from __future__ import annotations

import pytest

from mjrl.pad_script import ScriptedPad, parse_axes


def test_a_scripted_pad_publishes_the_sticks_it_was_given():
    pad = ScriptedPad({"Ly": 1.0, "Rx": -0.5})

    frame = pad.state()

    assert pad.connected
    assert (frame.ly, frame.rx) == (1.0, -0.5)
    assert (frame.lx, frame.ry, frame.lt) == (0.0, 0.0, 0.0)
    assert frame.buttons == ()


def test_an_unknown_axis_is_refused_rather_than_ignored():
    with pytest.raises(ValueError, match="no axis named 'LY'"):
        ScriptedPad({"LY": 1.0})

    with pytest.raises(ValueError, match="no axis named 'LY'"):
        parse_axes(["LY=1.0"])


def test_a_stick_out_of_range_or_not_a_number_is_refused():
    for bad in ("nan", "inf", "2.0"):
        with pytest.raises(ValueError, match="out of range|wants a number"):
            ScriptedPad(parse_axes([f"Ly={bad}"]))


def test_the_flag_parses_axis_value_pairs():
    assert parse_axes(["Ly=1.0", "hat_x=-1"]) == {"Ly": 1.0, "hat_x": -1.0}
    assert parse_axes(None) == {}

    with pytest.raises(ValueError, match="AXIS=VALUE"):
        parse_axes(["Ly"])


def test_the_pad_can_be_closed_without_a_device():
    pad = ScriptedPad({"Ly": 1.0})

    pad.close()

    assert pad.state().ly == 1.0