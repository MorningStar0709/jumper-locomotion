"""The scripted pad reaches the app path, and only there."""

from __future__ import annotations

import argparse

import pytest

from scripts import play


def test_the_pad_flag_needs_an_app(capsys):
    parser = argparse.ArgumentParser(prog="play")
    args = argparse.Namespace(app=None, app_pad=["Ly=1.0"])

    with pytest.raises(SystemExit):
        play._validate_app_pad(parser, args)

    assert "needs --app" in capsys.readouterr().err


def test_the_pad_flag_is_accepted_with_an_app():
    parser = argparse.ArgumentParser(prog="play")
    args = argparse.Namespace(app="x.app", app_pad=["Ly=1.0"])

    play._validate_app_pad(parser, args)


def test_an_unknown_axis_stops_the_run_before_the_world_is_built():
    from mjrl.pad_script import parse_axes

    with pytest.raises(ValueError, match="no axis named 'LY'"):
        parse_axes(["LY=1.0"])