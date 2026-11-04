"""The contract of `tools/benchmark.py`: a manifest in, one score file out.

The point of the tool is a fixed shape, not a pretty table: every evaluation
runs the same command and lands in the same JSON, so that two results a week
apart can still be compared. These tests pin the shape, not the numbers.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import benchmark
import gpu_lock


@pytest.fixture(autouse=True)
def a_lock_of_its_own(tmp_path, monkeypatch):
    """每台机器上都可能正好有训练在跑——测试不能因此变红，也不能因此变绿。"""
    monkeypatch.setattr(gpu_lock, "DEFAULT_LOCK", tmp_path / "gpu.lock")


def a_case(**over):
    base = {
        "name": "straight",
        "command": [0.5, 0.0, 0.0],
        "checkpoints": [{"seed": 2, "path": "logs/x/model_3000.pt"}],
    }
    base.update(over)
    return base


def a_manifest(*cases, **extra):
    body = {"task": "jumper.tripod", "steps": 1000, "cases": list(cases) or [a_case()]}
    body.update(extra)
    return body


def written(tmp_path, body, name="bench.json"):
    path = tmp_path / name
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def an_evaluation(mae=(0.02, 0.01, 0.05), actual=(0.48, 0.0, 0.01), resets=0):
    return {
        "schema": "fixed_body_command_evaluation/1",
        "axes": ["vx", "vy", "yaw_rate"],
        "mean_absolute_error": list(mae),
        "actual_velocity_mean": list(actual),
        "episode_resets": resets,
        "observation_noise": True,
    }


def fake_run(payload, seen=None):
    def run(command):
        if seen is not None:
            seen.append(list(command))
        out = Path(command[command.index("--evaluation-out") + 1])
        out.write_text(json.dumps(payload), encoding="utf-8")
        return 0

    return run


def test_a_manifest_without_cases_is_rejected(tmp_path):
    path = written(tmp_path, {"task": "jumper.tripod", "steps": 1000})

    with pytest.raises(ValueError) as excinfo:
        benchmark.load_manifest(path)

    assert "cases" in str(excinfo.value)


def test_a_case_without_a_command_is_rejected(tmp_path):
    body = a_manifest()
    del body["cases"][0]["command"]

    with pytest.raises(ValueError) as excinfo:
        benchmark.load_manifest(written(tmp_path, body))

    assert "command" in str(excinfo.value)


def test_each_case_carries_the_task_and_the_step_budget(tmp_path):
    loaded = benchmark.load_manifest(written(tmp_path, a_manifest(steps=6000)))

    assert loaded[0]["task"] == "jumper.tripod"
    assert loaded[0]["steps"] == 6000


def test_the_play_command_points_at_the_checkpoint_and_the_output(tmp_path):
    loaded = benchmark.load_manifest(written(tmp_path, a_manifest()))
    case, checkpoint = loaded[0], loaded[0]["checkpoints"][0]
    out = tmp_path / "out.json"

    command = benchmark.build_play_command(case, checkpoint, out=out)

    assert command[:2] == [sys.executable, "scripts/play.py"]
    assert command[command.index("--task") + 1] == "jumper.tripod"
    assert command[command.index("--steps") + 1] == "1000"
    assert command[command.index("--command") + 1 : command.index("--command") + 4] == [
        "0.5",
        "0.0",
        "0.0",
    ]
    assert command[command.index("--checkpoint") + 1] == "logs/x/model_3000.pt"
    assert command[command.index("--evaluation-out") + 1] == str(out)
    assert "--headless" in command


def test_the_summary_reports_both_the_worst_and_the_average_per_axis():
    rows = [
        {"case": "a", "mae": {"vx": 0.02, "vy": 0.01, "yaw_rate": 0.05}},
        {"case": "b", "mae": {"vx": 0.04, "vy": 0.00, "yaw_rate": 0.03}},
    ]

    summary = benchmark.summarise(rows)

    assert summary["worst_mae"]["vx"] == pytest.approx(0.04)
    assert summary["mean_mae"]["vx"] == pytest.approx(0.03)
    assert summary["worst_mae"]["yaw_rate"] == pytest.approx(0.05)
    assert len(summary["cases"]) == 2


def test_a_run_writes_one_score_file_with_a_row_per_checkpoint(tmp_path):
    path = written(tmp_path, a_manifest())
    seen: list[list[str]] = []

    rc = benchmark.main([str(path)], run=fake_run(an_evaluation(), seen))

    assert rc == 0
    assert len(seen) == 1
    score = json.loads((tmp_path / "bench.score.json").read_text(encoding="utf-8"))
    row = score["cases"][0]
    assert row["case"] == "straight"
    assert row["seed"] == 2
    assert row["steps"] == 1000
    assert row["mae"]["vx"] == pytest.approx(0.02)
    assert row["mae"]["yaw_rate"] == pytest.approx(0.05)
    assert row["resets"] == 0


def test_each_checkpoint_gets_its_own_raw_evaluation_file(tmp_path):
    body = a_manifest(
        a_case(
            checkpoints=[
                {"seed": 2, "path": "logs/x/model_3000.pt"},
                {"seed": 2, "path": "logs/x/model_9999.pt"},
            ]
        )
    )
    seen: list[list[str]] = []

    benchmark.main([str(written(tmp_path, body))], run=fake_run(an_evaluation(), seen))

    outs = [command[command.index("--evaluation-out") + 1] for command in seen]
    assert len(outs) == 2
    assert len(set(outs)) == 2


def test_the_script_says_how_to_use_it_when_called_without_arguments():
    result = subprocess.run(
        [sys.executable, str(Path(benchmark.__file__))],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "usage" in result.stderr


def test_a_failing_run_is_reported_in_the_score_file(tmp_path):
    path = written(tmp_path, a_manifest())

    rc = benchmark.main([str(path)], run=lambda command: 3)

    score = json.loads((tmp_path / "bench.score.json").read_text(encoding="utf-8"))
    assert rc != 0
    assert "3" in score["cases"][0]["error"]


def test_a_case_may_carry_set_assignments(tmp_path):
    case = a_case(set=["episodes.push=1", "events.push_robot.params.velocity_range.x=(-0.5,0.5)"])
    cases = benchmark.load_manifest(written(tmp_path, a_manifest(case)))
    command = benchmark.build_play_command(cases[0], case["checkpoints"][0], out=tmp_path / "o.json")
    pairs = [command[i + 1] for i, token in enumerate(command) if token == "--set"]
    assert pairs == case["set"]


def test_a_case_without_set_assignments_adds_no_flag(tmp_path):
    cases = benchmark.load_manifest(written(tmp_path, a_manifest(a_case())))
    command = benchmark.build_play_command(cases[0], a_case()["checkpoints"][0], out=tmp_path / "o.json")
    assert "--set" not in command


def test_set_must_be_a_list_of_strings(tmp_path):
    with pytest.raises(ValueError, match="set"):
        benchmark.load_manifest(written(tmp_path, a_manifest(a_case(set="a=1"))))


def test_a_manifest_may_fix_the_seed(tmp_path):
    cases = benchmark.load_manifest(written(tmp_path, a_manifest(a_case(), seed=0)))
    command = benchmark.build_play_command(cases[0], a_case()["checkpoints"][0], out=tmp_path / "o.json")
    assert command[command.index("--seed") + 1] == "0"


def test_a_manifest_without_a_seed_passes_none(tmp_path):
    cases = benchmark.load_manifest(written(tmp_path, a_manifest(a_case())))
    command = benchmark.build_play_command(cases[0], a_case()["checkpoints"][0], out=tmp_path / "o.json")
    assert "--seed" not in command


def test_a_nonsense_seed_is_refused(tmp_path):
    for bad in (-1, 4294967296, "0", True, 1.5):
        with pytest.raises(ValueError, match="seed"):
            benchmark.load_manifest(written(tmp_path, a_manifest(a_case(), seed=bad)))


# --- 任务自己的开关 ---------------------------------------------------------
# 延迟训练的模型要在**带延迟**的条件下评测才有意义，而 `--delay` 是任务自己
# 的 flag（`tasks/jumper/tripod/__init__.py` 的 `cli_args`），不是 `--set`
# 那种配置覆盖。清单得能把这类 flag 原样交给 `play.py`。

def test_a_case_may_carry_task_flags(tmp_path):
    case = a_case(args=["--delay"])
    cases = benchmark.load_manifest(written(tmp_path, a_manifest(case)))

    command = benchmark.build_play_command(
        cases[0], case["checkpoints"][0], out=tmp_path / "o.json")

    assert "--delay" in command


def test_a_case_without_flags_adds_none(tmp_path):
    cases = benchmark.load_manifest(written(tmp_path, a_manifest(a_case())))

    command = benchmark.build_play_command(
        cases[0], a_case()["checkpoints"][0], out=tmp_path / "o.json")

    assert "--delay" not in command


def test_flags_that_are_not_long_options_are_refused(tmp_path):
    with pytest.raises(ValueError, match="args"):
        benchmark.load_manifest(written(tmp_path, a_manifest(a_case(args=["-d"]))))


def test_a_measurement_that_measured_nothing_is_a_failed_row(tmp_path):
    """`play.py` 在没采到有效样本时写 `null`；那不是 0 误差，是没测。"""
    payload = an_evaluation()
    payload["mean_absolute_error"] = None
    manifest = written(tmp_path, a_manifest())
    run = fake_run(payload)

    assert benchmark.main([str(manifest)], run=run) == 1
    score = json.loads(manifest.with_suffix(".score.json").read_text(encoding="utf-8"))
    row = score["cases"][0]
    assert "measured nothing" in row["error"]
    assert "mae" not in row, "a row that measured nothing must not carry a score"


def test_a_broken_evaluation_file_does_not_stop_the_run(tmp_path):
    good = a_case(name="second")
    manifest = written(tmp_path, a_manifest(a_case(name="first"), good))

    def run(command):
        out = Path(command[command.index("--evaluation-out") + 1])
        if "first" in out.name:
            out.write_text("{not json", encoding="utf-8")
        else:
            out.write_text(json.dumps(an_evaluation()), encoding="utf-8")
        return 0

    assert benchmark.main([str(manifest)], run=run) == 1
    score = json.loads(manifest.with_suffix(".score.json").read_text(encoding="utf-8"))
    assert [row["case"] for row in score["cases"]] == ["first", "second"]
    assert score["cases"][0]["error"], score["cases"][0]
    assert score["cases"][1]["mae"]["vx"] == pytest.approx(0.02)


def test_two_checkpoints_with_the_same_stem_keep_their_own_evidence(tmp_path):
    case = a_case(checkpoints=[{"seed": 2, "path": "logs/runA/model_3000.pt"},
                               {"seed": 2, "path": "logs/runB/model_3000.pt"}])
    manifest = written(tmp_path, a_manifest(case))

    assert benchmark.main([str(manifest)], run=fake_run(an_evaluation())) == 0
    raws = sorted(p.name for p in manifest.with_suffix(".score.d").iterdir())
    assert len(raws) == 2, raws


def test_the_row_records_the_conditions_it_was_measured_under(tmp_path):
    case = a_case(set=["events.foot_friction.params.ranges=(1.2,1.2)"], args=["--delay"])
    manifest = written(tmp_path, a_manifest(case), name="sweep.json")

    assert benchmark.main([str(manifest)], run=fake_run(an_evaluation())) == 0
    row = json.loads(manifest.with_suffix(".score.json").read_text(encoding="utf-8"))["cases"][0]
    assert row["set"] == ["events.foot_friction.params.ranges=(1.2,1.2)"]
    assert row["args"] == ["--delay"]


def test_a_checkpoint_path_that_is_not_text_is_refused_cleanly(tmp_path):
    manifest = written(tmp_path, a_manifest(a_case(checkpoints=[{"seed": 2, "path": None}])))
    assert benchmark.main([str(manifest)], run=fake_run(an_evaluation())) == 2


def test_a_command_with_a_null_is_refused_cleanly(tmp_path):
    manifest = written(tmp_path, a_manifest(a_case(command=[None, 0, 0])))
    assert benchmark.main([str(manifest)], run=fake_run(an_evaluation())) == 2


def test_a_misspelt_case_key_is_refused(tmp_path):
    """`"arg"` 少一个 s，静默忽略的话出来的是一份看着正常的「无延迟」数字。"""
    manifest = written(tmp_path, a_manifest(a_case(arg=["--delay"])))
    assert benchmark.main([str(manifest)], run=fake_run(an_evaluation())) == 2


# --- 评测要取 GPU 锁 ------------------------------------------------------------
# 训练已经按 run 取锁；评测不取锁，就是同一张卡上的第二个写入者——两边的读数
# 都不可信，而 DDS 与 CUDA 的争抢不会报错。75 是「现在不行，等下再来」。


def test_a_busy_card_stops_the_benchmark_before_it_starts(tmp_path):
    path = written(tmp_path, a_manifest())
    seen = []

    with gpu_lock.hold(tmp_path / "gpu.lock", role="training"):
        rc = benchmark.main([str(path)], run=fake_run(an_evaluation(), seen))

    assert rc == gpu_lock.BUSY_RC
    assert seen == []


def test_a_lock_that_cannot_be_opened_is_not_a_busy_card(tmp_path):
    """锁坏了等多久都不会好，卡被占着等一会儿就好——两种要分开报。

    `O_NOFOLLOW` 让路径上的软链变成 `OSError`；这条路径过去没有任何测试，
    回退成「当成卡被占着」或「抛 traceback」都不会有人发现。
    """
    (tmp_path / "gpu.lock").symlink_to(tmp_path / "elsewhere")
    path = written(tmp_path, a_manifest())

    rc = benchmark.main([str(path)], run=fake_run(an_evaluation()))

    assert rc == gpu_lock.UNAVAILABLE_RC
    assert not path.with_suffix(".score.json").exists()


def test_the_card_is_marked_as_taken_while_it_evaluates(tmp_path):
    path = written(tmp_path, a_manifest())
    seen = {}
    inner = fake_run(an_evaluation())

    def run(command):
        seen["holder"] = gpu_lock.status(tmp_path / "gpu.lock")
        return inner(command)

    assert benchmark.main([str(path)], run=run) == 0
    assert seen["holder"]["role"].startswith("benchmark")
    assert gpu_lock.status(tmp_path / "gpu.lock") is None


def test_a_case_cannot_set_the_flags_the_tool_decides(tmp_path):
    """case 的 `args` 再写一遍 `--steps` 不报错，只是 argparse 取最后一次：
    命令行里跑的是 case 的值，清单里记的是工具的值。"""
    path = tmp_path / "b.json"
    path.write_text(json.dumps({
        "task": "jumper.tripod",
        "cases": [{"name": "a", "command": [0.5, 0, 0], "args": ["--steps=50"],
                   "checkpoints": [{"path": "x/model_1.pt"}]}],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="which this tool decides"):
        benchmark.load_manifest(path)


def test_a_case_name_that_is_not_a_string_is_refused(tmp_path):
    """`["straight"]` 会在 `name in seen_names` 上抛 `TypeError`，而不是一条人话。"""
    path = tmp_path / "b.json"
    path.write_text(json.dumps({
        "task": "jumper.tripod",
        "cases": [{"name": ["straight"], "command": [0.5, 0, 0],
                   "checkpoints": [{"path": "x/model_1.pt"}]}],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="not a string"):
        benchmark.load_manifest(path)


def test_an_actual_velocity_of_the_wrong_length_is_refused(tmp_path):
    """`null` 是「没有有效样本」，但三个数写成两个是坏文件，不能静默丢掉。"""
    path = tmp_path / "e.json"
    path.write_text(json.dumps({
        "axes": ["vx", "vy", "yaw_rate"], "mean_absolute_error": [0.1, 0.2, 0.3],
        "actual_velocity_mean": [0.5, 0.0], "episode_resets": 0,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="which is not 3 numbers"):
        benchmark.read_result(path)


def test_the_score_files_sit_beside_the_manifest_whatever_the_cwd(tmp_path, monkeypatch):
    """子进程用 `cwd=REPO`，父进程的 cwd 可能在任何地方——路径必须是绝对的。"""
    manifest = tmp_path / "b.json"
    manifest.write_text(json.dumps({
        "task": "jumper.tripod",
        "cases": [{"name": "a", "command": [0.5, 0, 0],
                   "checkpoints": [{"path": "x/model_1.pt"}]}],
    }), encoding="utf-8")
    seen = []

    def run(command):
        out = Path(command[command.index("--evaluation-out") + 1])
        seen.append(out)
        out.write_text(json.dumps({
            "axes": ["vx", "vy", "yaw_rate"], "mean_absolute_error": [0.1, 0.2, 0.3],
            "actual_velocity_mean": [0.5, 0.0, 0.6], "episode_resets": 0,
        }), encoding="utf-8")
        return 0

    monkeypatch.chdir(tmp_path)
    # 用相对路径调用：`Path("b.json").parent` 是 `.`，路径若不按仓库根解析，
    # 子进程（`cwd=REPO`）就会把评分文件写进仓库根——而父进程在这里找它。
    assert benchmark.main(["b.json"], run=run) == 0
    assert seen and seen[0].is_absolute()
    assert tmp_path in seen[0].parents
    assert (tmp_path / "b.score.json").exists()


def test_an_abbreviated_reserved_flag_is_still_reserved(tmp_path):
    """`argparse` 没关 `allow_abbrev`：`--ste=50` 会被展开成 `--steps`。

    完整拼写比对看不见它，于是清单能悄悄改掉本工具自己写的那几个参数，而行里
    记的仍然是被改之前的意图。
    """
    with pytest.raises(ValueError, match="which this tool decides"):
        benchmark.load_manifest(written(tmp_path, a_manifest(a_case(args=["--ste=50"]))))


def test_a_single_letter_abbreviation_reaches_a_reserved_flag_too(tmp_path):
    """`--t` 只有 `--task` 一个候选，argparse 不会报 ambiguous，会直接展开。

    而 case 的参数拼在工具自己写的 `--task` **之后**，argparse 取最后一个——评分
    文件记的是清单里的任务名，实际评的是另一个。
    """
    with pytest.raises(ValueError, match="which this tool decides"):
        benchmark.load_manifest(written(tmp_path,
                                        a_manifest(a_case(args=["--t=jumper.other"]))))
