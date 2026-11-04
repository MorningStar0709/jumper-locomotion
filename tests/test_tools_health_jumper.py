"""`tools/health_jumper.py` 的判据。

这个脚本是巡检线唯一的大脑：它报什么，另一个 agent 就按什么分支处置。它自己
没有测试，等于「判据由巡检时的临场表现决定」。这里钉住四件事：四个（现在是
五个）verdict 各自的触发条件、读不到状态时的分流、磁盘水位如何升级结论、
以及「没有训练在跑」不能被报成「有东西坏了」。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import health_jumper

PLENTY = {"free_gib": 900.0, "total_gib": 1000.0, "used_fraction": 0.10}
FULLISH = {"free_gib": 120.0, "total_gib": 1000.0, "used_fraction": 0.88}
CRITICAL = {"free_gib": 20.0, "total_gib": 1000.0, "used_fraction": 0.98}
NO_GPU = {"limit_w": None, "mem_mib": None, "util": None}
GPU = {"limit_w": 175.0, "mem_mib": 9437.0, "util": 91.0}


def _quiet(monkeypatch: pytest.MonkeyPatch, *, disk=PLENTY, gpu=GPU, wrapper=False, alive=True):
    monkeypatch.setattr(health_jumper, "gpu_readings", lambda: gpu)
    monkeypatch.setattr(health_jumper, "disk_readings", lambda root: disk)
    monkeypatch.setattr(health_jumper, "wrapper_running", lambda: wrapper)
    monkeypatch.setattr(health_jumper, "pid_alive", lambda pid: alive)


def _status(root: Path, **fields) -> Path:
    run = root / "jumper" / "jumper.tripod" / "2026-10-30_00-00-00"
    run.mkdir(parents=True, exist_ok=True)
    path = run / "status.json"
    path.write_text(json.dumps({"v": 1, "status": "running", "pid": 4242, "ts": 1.0, **fields}))
    return path


def _names(result: dict) -> list[str]:
    return [check["name"] for check in result["checks"]]


def test_nothing_has_been_started_is_idle_not_a_problem(tmp_path, monkeypatch):
    """空 logs + 没有训练进程 = 没什么可看，不是「状态缺失要人查」。"""
    _quiet(monkeypatch)
    result = health_jumper.evaluate(tmp_path)
    assert result["verdict"] == "idle"
    assert result["ok"] is True


def test_a_wrapper_without_a_status_file_is_worth_investigating(tmp_path, monkeypatch):
    """有 long_run 在跑却没写出状态文件，是真问题：要么没包 wrapper，要么刚启动。"""
    _quiet(monkeypatch, wrapper=True)
    result = health_jumper.evaluate(tmp_path)
    assert result["verdict"] == "investigate"
    assert result["ok"] is False


def test_a_running_and_alive_run_waits(tmp_path, monkeypatch):
    _quiet(monkeypatch)
    _status(tmp_path, status="running", interval_s=30.0)
    result = health_jumper.evaluate(tmp_path)
    assert result["verdict"] == "wait"
    assert result["ok"] is True


def test_a_running_run_whose_process_is_gone_is_an_alarm(tmp_path, monkeypatch):
    _quiet(monkeypatch, alive=False)
    _status(tmp_path, status="running")
    result = health_jumper.evaluate(tmp_path)
    assert result["verdict"] == "alarm"
    assert result["ok"] is False


def test_a_completed_run_is_done(tmp_path, monkeypatch):
    _quiet(monkeypatch)
    _status(tmp_path, status="completed", rc=0)
    assert health_jumper.evaluate(tmp_path)["verdict"] == "done"


def test_a_crashed_run_is_an_alarm(tmp_path, monkeypatch):
    _quiet(monkeypatch)
    _status(tmp_path, status="crashed", rc=143)
    assert health_jumper.evaluate(tmp_path)["verdict"] == "alarm"


def test_an_interrupted_run_is_not_an_alarm(tmp_path, monkeypatch):
    """人叫停的一轮不该升级成 alarm——否则每次主动收工都会被当成故障，告警就废了。

    它还必须是终态：`running` 会让巡检一直报「进程没了」。
    """
    _quiet(monkeypatch, alive=False)
    _status(tmp_path, status="interrupted", rc=143, signal=15)
    assert health_jumper.evaluate(tmp_path)["verdict"] == "done"


def test_a_status_file_that_cannot_be_read_is_investigated(tmp_path, monkeypatch):
    _quiet(monkeypatch, wrapper=True)
    run = tmp_path / "jumper" / "jumper.tripod" / "2026-10-30_00-00-00"
    run.mkdir(parents=True)
    (run / "status.json").write_text("{not json")
    result = health_jumper.evaluate(tmp_path)
    assert result["verdict"] == "investigate"
    assert "corrupt" in json.dumps(result, ensure_ascii=False) or result["checks"][0]["note"]


def test_a_parse_error_while_running_is_investigated(tmp_path, monkeypatch):
    _quiet(monkeypatch)
    _status(tmp_path, status="running", parse_error=True)
    assert health_jumper.evaluate(tmp_path)["verdict"] == "investigate"


def test_a_parse_error_left_behind_by_a_finished_run_does_not_change_the_verdict(tmp_path, monkeypatch):
    """`completed` 里残留的 parse_error 不该把结论改成 investigate。"""
    _quiet(monkeypatch)
    _status(tmp_path, status="completed", parse_error=True)
    assert health_jumper.evaluate(tmp_path)["verdict"] == "done"


def test_the_disk_is_a_check_of_its_own(tmp_path, monkeypatch):
    _quiet(monkeypatch)
    _status(tmp_path, status="running")
    result = health_jumper.evaluate(tmp_path)
    disk = next(check for check in result["checks"] if check["name"] == "disk")
    assert disk["ok"] is True
    assert disk["value"]["used_fraction"] == pytest.approx(0.10)


def test_a_nearly_full_disk_turns_a_good_verdict_into_an_investigation(tmp_path, monkeypatch):
    """磁盘 85% 以上是知识库里「在线健康四件套」的一条；它单独不足以报警。"""
    _quiet(monkeypatch, disk=FULLISH)
    _status(tmp_path, status="running")
    result = health_jumper.evaluate(tmp_path)
    assert result["verdict"] == "investigate"
    assert result["ok"] is False


def test_a_full_disk_is_an_alarm_even_when_the_training_looks_fine(tmp_path, monkeypatch):
    _quiet(monkeypatch, disk=CRITICAL)
    _status(tmp_path, status="running")
    assert health_jumper.evaluate(tmp_path)["verdict"] == "alarm"


def test_a_full_disk_does_not_downgrade_an_existing_alarm(tmp_path, monkeypatch):
    _quiet(monkeypatch, disk=FULLISH, alive=False)
    _status(tmp_path, status="running")
    assert health_jumper.evaluate(tmp_path)["verdict"] == "alarm"


def test_a_full_disk_does_not_turn_done_into_something_else_by_luck(tmp_path, monkeypatch):
    """跑完了但磁盘满：该报 investigate（要人清理），不能报 done。"""
    _quiet(monkeypatch, disk=FULLISH)
    _status(tmp_path, status="completed")
    assert health_jumper.evaluate(tmp_path)["verdict"] == "investigate"


def test_an_unreadable_disk_does_not_invent_a_number(tmp_path, monkeypatch):
    _quiet(monkeypatch, disk={"free_gib": None, "total_gib": None, "used_fraction": None})
    _status(tmp_path, status="running")
    result = health_jumper.evaluate(tmp_path)
    disk = next(check for check in result["checks"] if check["name"] == "disk")
    assert disk["ok"] is False
    assert result["verdict"] == "wait"


def test_the_checks_cover_status_process_parse_gpu_and_disk(tmp_path, monkeypatch):
    _quiet(monkeypatch)
    _status(tmp_path, status="running")
    assert _names(health_jumper.evaluate(tmp_path)) == [
        "status_file",
        "process",
        "parse",
        "gpu",
        "disk",
    ]


def test_a_machine_without_nvidia_smi_still_reports_the_rest(tmp_path, monkeypatch):
    _quiet(monkeypatch, gpu=NO_GPU)
    _status(tmp_path, status="running")
    result = health_jumper.evaluate(tmp_path)
    assert result["verdict"] == "wait"
    assert result["ok"] is True


def test_the_exit_code_follows_the_verdict(tmp_path, monkeypatch, capsys):
    _quiet(monkeypatch)
    _status(tmp_path, status="completed")
    rc = health_jumper.main(["--json", "--logs-root", str(tmp_path)])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "done"


def test_an_idle_machine_exits_zero(tmp_path, monkeypatch, capsys):
    """`idle` 是「没什么可看」，不是失败；巡检不该因为它连报三轮升级成 alarm。"""
    _quiet(monkeypatch)
    rc = health_jumper.main(["--json", "--logs-root", str(tmp_path)])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "idle"


def test_a_pid_beyond_the_pid_range_is_not_alive():
    """`status.json` 被写坏时可能是 `5000000000`：`os.kill` 抛 `OverflowError`，
    而那不是 `OSError`——漏出去，这个只读检查就在它专门报告的场景里 traceback 了。"""
    assert health_jumper.pid_alive(5_000_000_000) is False


def test_a_status_file_that_vanishes_between_glob_and_stat_is_not_a_crash(tmp_path, monkeypatch):
    """巡检快跑完时有人清理 logs：`glob` 与 `stat` 之间目录被删不该 traceback。"""
    ghost = tmp_path / "logs" / "jumper" / "jumper.tripod" / "gone" / "status.json"
    monkeypatch.setattr(type(tmp_path), "glob", lambda self, pattern: iter([ghost]), raising=False)

    data, age, reason = health_jumper.latest_status(tmp_path / "logs")

    assert data is None
    assert age is None
    assert reason == "corrupt"


def test_a_status_file_that_vanishes_between_the_read_and_the_stat_is_not_a_crash(
        tmp_path, monkeypatch):
    """`latest_status` 的 `(data, None, "ok")` 分支是真实的。

    目录在 `glob`、`read`、`stat` 之间被清理时就会走到它。调用方拿 `None` 去
    `round()` 或比大小，得到的是一段 traceback——而这个工具是巡检唯一的读数来源。
    """
    run_dir = tmp_path / "logs" / "jumper" / "jumper.tripod" / "2026-10-31_00-00-00"
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text(json.dumps({
        "status": "running", "pid": 1, "iter": 10, "ts": 1793400000.0,
        "interval_s": 30}), encoding="utf-8")

    real_stat = Path.stat

    def gone(path, *args, **kwargs):
        # 只让这一个文件「消失」：全局打断 `stat` 会把 `glob` 自己的 `is_dir()`
        # 也打掉，测的就不是这条分支了。
        if path.name == "status.json":
            raise OSError("directory vanished")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", gone, raising=False)
    _quiet(monkeypatch)

    verdict = health_jumper.evaluate(tmp_path / "logs")

    # 钉住**确切的**结论：读不到年龄就等于「无法证明它是新的」，`running` 的 run
    # 会被当成卡住的写者。写成 `in {...}` 的话，把 `age is None` 从
    # `stale = running and (age is None or ...)` 里拿掉（结论变成 wait）它仍然绿。
    assert verdict["verdict"] == "alarm"
    status_check = next(c for c in verdict["checks"] if c["name"] == "status_file")
    assert status_check["value"]["age_s"] is None
