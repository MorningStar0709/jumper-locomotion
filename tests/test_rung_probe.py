"""The rung probe applies the repository's ceiling rule before a training run.

It exists because two ladders climbed a rung that could not be walked and each cost
a training run plus an evaluation to find out. What it can and cannot say was
measured: the 4 cm stair rung is *walkable* by the 3 cm arm (0.37 m/s, zero stalls)
and training on it still destroyed the policy, so a replay-only probe is a floor --
"this policy cannot even walk here" -- and never a promise that training will help.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import rung_probe  # `pythonpath = ["tools", "scripts"]`


def _report(vx, mae, stalled=0.0):
    return {"actual_velocity_mean": [vx, 0.0, 0.0], "mean_absolute_error": [mae, 0.0, 0.0],
            "stalled_fraction": stalled}


def test_a_ground_the_policy_cannot_walk_is_refused():
    answer = rung_probe.verdict([_report(0.12, 0.36, 0.42)] * 7, [_report(0.47, 0.02)] * 3, 0.5)
    assert answer["verdict"] == "collapsed"
    assert answer["ratio"] == pytest.approx(0.24)
    assert answer["multiple"] == pytest.approx(18.0)
    assert answer["stalled"] == pytest.approx(0.42)


def test_a_slow_ground_is_marginal_not_refused():
    """The measured 4 cm case: ratio 0.73, MAE 4.7x its flat 0.0287.

    It is labelled, not refused. Two rungs inside the ceiling region behaved
    oppositely -- 3 cm at 0.83 produced the arm that walks 4 cm, 4 cm at 0.73
    destroyed the gait -- and no replay-only number separated them, so refusing
    here would refuse the rung that paid.
    """
    answer = rung_probe.verdict([_report(0.367, 0.133)] * 7, [_report(0.473, 0.0287)] * 3, 0.5)
    assert answer["verdict"] == "marginal"
    assert answer["ratio"] == pytest.approx(0.734)
    assert answer["multiple"] == pytest.approx(4.63, abs=0.01)
    assert rung_probe.verdict([_report(0.415, 0.0862)] * 7, [_report(0.473, 0.0287)] * 3,
                              0.5)["verdict"] == "marginal"


def test_a_ground_it_walks_well_holds():
    answer = rung_probe.verdict([_report(0.47, 0.03)] * 7, [_report(0.473, 0.0287)] * 3, 0.5)
    assert answer["verdict"] == "holds"


def test_an_error_jump_alone_does_not_make_a_ground_marginal():
    answer = rung_probe.verdict([_report(0.49, 0.09)] * 7, [_report(0.47, 0.02)] * 3, 0.5)
    assert answer["verdict"] == "holds"


def test_medians_are_used_so_one_bad_seed_does_not_decide():
    reports = [_report(0.45, 0.02)] * 6 + [_report(0.05, 0.45)]
    answer = rung_probe.verdict(reports, [_report(0.47, 0.02)] * 3, 0.5)
    assert answer["verdict"] == "holds"


def test_the_plan_ends_with_flat_reference_cells(tmp_path: Path):
    runs = rung_probe.plan(ground="steps", value=0.04, checkpoint=Path("ckpt.pt"),
                           seeds=3, steps=100, command=(0.5, 0.0, 0.0),
                           out=tmp_path / "probe.json")
    assert [item["flat"] for item in runs] == [False] * 3 + [True] * rung_probe.FLAT_SEEDS
    assert len({item["evaluation"] for item in runs}) == len(runs)
    ground = runs[0]["argv"]
    assert ground[ground.index("--checkpoint") + 1] == "ckpt.pt"
    assert ground[ground.index("--terrain-steps-m") + 1] == "0.04"
    flat = runs[-1]["argv"]
    assert flat[flat.index("--terrain-roughness-mm") + 1] == "0"
    assert "--terrain-steps-m" not in flat


def _fake_run(ground_vx, ground_mae, flat_mae):
    """A runner that writes what play.py would have written, per cell."""
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        target = Path(command[command.index("--evaluation-out") + 1])
        target.parent.mkdir(parents=True, exist_ok=True)
        flat = "flat-s" in target.name
        target.write_text(json.dumps({
            "actual_velocity_mean": [0.47 if flat else ground_vx, 0.0, 0.0],
            "mean_absolute_error": [flat_mae if flat else ground_mae, 0.0, 0.0],
            "stalled_fraction": 0.0,
        }))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    run.calls = calls
    return run


def test_the_cli_exits_three_only_when_the_ground_is_collapsed(tmp_path: Path):
    run = _fake_run(0.12, 0.36, 0.02)
    rc = rung_probe.main([
        "--checkpoint", "ckpt.pt", "--ground", "steps", "--value", "0.04",
        "--out", str(tmp_path / "probe.json"),
    ], run=run)
    assert rc == 3
    summary = json.loads((tmp_path / "probe.json").read_text())
    assert summary["verdict"] == "collapsed"
    assert len(run.calls) == 7 + rung_probe.FLAT_SEEDS


def test_the_cli_exits_zero_on_a_marginal_rung_too(tmp_path: Path):
    run = _fake_run(0.367, 0.133, 0.0287)
    rc = rung_probe.main([
        "--checkpoint", "ckpt.pt", "--ground", "rough", "--value", "35",
        "--out", str(tmp_path / "probe.json"),
    ], run=run)
    assert rc == 0
    assert json.loads((tmp_path / "probe.json").read_text())["verdict"] == "marginal"

def test_a_turn_in_place_reports_no_stall_share_rather_than_zero():
    """The rule never fires below `STALL_MIN_COMMAND`, so a 0.0 would be a lie."""
    reports = [dict(_report(0.01, 0.4), stalled_fraction=None)] * 7
    answer = rung_probe.verdict(reports, [_report(0.47, 0.02)] * 3, 0.5)
    assert answer["stalled"] is None
    assert answer["verdict"] == "collapsed"


def test_the_flat_seed_written_to_the_summary_is_the_one_in_the_command(tmp_path: Path):
    runs = rung_probe.plan(ground="rough", value=5.0, checkpoint=Path("ckpt.pt"),
                           seeds=7, steps=100, command=(0.5, 0.0, 0.0),
                           out=tmp_path / "probe.json")
    first_flat = runs[7]
    assert first_flat["seed"] == 0
    assert first_flat["evaluation"].name == "probe-flat-s0.json"
    assert first_flat["argv"][first_flat["argv"].index("--seed") + 1] == "0"
    assert runs[0]["argv"][runs[0]["argv"].index("--task") + 1] == "jumper.tripod"


def test_a_seed_count_that_leaves_no_ground_sample_is_a_usage_error(tmp_path: Path, capsys):
    with pytest.raises(SystemExit) as raised:
        rung_probe.main(["--checkpoint", "ckpt.pt", "--ground", "steps", "--value", "0.04",
                         "--seeds", "0", "--out", str(tmp_path / "probe.json")],
                        run=_fake_run(0.4, 0.03, 0.02))
    assert raised.value.code == 2
    assert "--seeds" in capsys.readouterr().err


def test_a_command_with_no_forward_speed_is_a_usage_error(tmp_path: Path, capsys):
    """`--command 0 0.5 0` divides by zero in `verdict` after every cell is paid for."""
    with pytest.raises(SystemExit) as raised:
        rung_probe.main(["--checkpoint", "ckpt.pt", "--ground", "steps", "--value", "0.04",
                         "--command", "0", "0.5", "0",
                         "--out", str(tmp_path / "probe.json")],
                        run=_fake_run(0.4, 0.03, 0.02))
    assert raised.value.code == 2
    assert "VX" in capsys.readouterr().err


def test_a_dead_replay_aborts_without_writing_a_verdict(tmp_path: Path):
    def broken(command, **_kwargs):
        return SimpleNamespace(returncode=3, stdout="", stderr="nconmax overflow")

    out = tmp_path / "probe.json"
    rc = rung_probe.main(["--checkpoint", "ckpt.pt", "--ground", "steps", "--value", "0.04",
                          "--out", str(out)], run=broken)
    assert rc == 1, "a replay that exited 3 must not be read as this tool's collapsed verdict"
    assert not out.exists()


def test_the_argv_it_writes_parses_with_the_play_parser(tmp_path: Path, monkeypatch):
    """Renaming a switch in `play.py` or in the tripod task must break this, not a real probe."""
    import _cli  # `pythonpath` in pyproject puts `scripts/` on the path
    import play

    runs = rung_probe.plan(ground="steps", value=0.04, checkpoint=Path("ckpt.pt"),
                           seeds=1, steps=100, command=(0.5, 0.0, 0.0),
                           out=tmp_path / "probe.json")
    monkeypatch.setattr(sys, "argv", ["play.py", *runs[0]["argv"]])
    parser = _cli.build_parser("play.py", "", replay=True)
    play.add_replay_args(parser)
    args = _cli.parse_with_task_args(parser)
    assert args.task == "jumper.tripod"
    assert args.terrain is True
    assert args.task_args["terrain_steps_m"] == 0.04
    assert args.task_args["terrain_seed"] == 0
