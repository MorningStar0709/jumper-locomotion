"""`tools/run_state.py` 的契约：`status.json` 的字段、读法，与运行目录的发现。

这三件事原本散在三个文件里——`long_run` 写、`health_jumper` 读、`run_queue` 核验，
谁也不知道对方的字段名。这里钉住一处定义，三方都必须从这里取。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

import run_state


def a_finished_run() -> dict:
    """`long_run` 跑完一条之后真正写到盘上的那种字典。"""
    return {
        "v": 1,
        "pid": 4242,
        "status": "completed",
        "run_dir": "/tmp/logs/jumper.tripod/2026-10-30_00-18-00",
        "iter": 2999,
        "percent": 99.0,
        "eta_s": 0.0,
        "step_time": 1.07,
        "elapsed_s": 3273.2,
        "limit_w": 175.0,
        "mem_mib": 9437,
        "util": 91,
        "power_w": 174.2,
        "ts": time.time(),
        "parse_error": False,
        "rc": 0,
        "interval_s": 30.0,
        "gpu": {"name": "NVIDIA GeForce RTX 5090 Laptop GPU", "total_mib": 24463},
    }


def test_validate_accepts_what_long_run_writes():
    assert run_state.validate(a_finished_run()) == []


def test_validate_names_the_missing_field():
    status = a_finished_run()
    del status["pid"]

    problems = run_state.validate(status)

    assert problems == ["missing pid"]


def test_validate_rejects_a_status_value_nobody_writes():
    status = a_finished_run()
    status["status"] = "exploded"

    problems = run_state.validate(status)

    assert any("exploded" in problem for problem in problems), problems


def test_validate_accepts_a_run_stopped_on_purpose():
    """被 `pkill` 收工的一轮是终态，和崩溃分开记：前者不该升级成告警。"""
    status = a_finished_run()
    status["status"] = "interrupted"
    status["signal"] = 15

    assert run_state.validate(status) == []


def test_validate_rejects_a_schema_it_does_not_know():
    status = a_finished_run()
    status["v"] = 99

    problems = run_state.validate(status)

    assert any("99" in problem for problem in problems), problems


def test_validate_says_so_when_the_status_is_not_an_object():
    assert run_state.validate([]) == ["status is a list, not an object"]


def test_read_returns_none_when_the_file_is_not_there(tmp_path: Path):
    assert run_state.read(tmp_path / "status.json") is None


def test_read_returns_none_when_the_file_is_half_written(tmp_path: Path):
    path = tmp_path / "status.json"
    path.write_text('{"v": 1, "sta', encoding="utf-8")

    assert run_state.read(path) is None


def test_read_returns_none_when_the_payload_is_not_an_object(tmp_path: Path):
    path = tmp_path / "status.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    assert run_state.read(path) is None


def test_read_gives_back_what_write_put_there(tmp_path: Path):
    path = tmp_path / "status.json"
    run_state.write(path, a_finished_run())

    assert run_state.read(path)["iter"] == 2999


def test_write_leaves_no_half_finished_file_behind(tmp_path: Path):
    path = tmp_path / "status.json"
    run_state.write(path, a_finished_run())

    assert [p.name for p in tmp_path.iterdir()] == ["status.json"]


def test_write_replaces_an_older_status_in_one_step(tmp_path: Path):
    path = tmp_path / "status.json"
    run_state.write(path, a_finished_run())
    status = a_finished_run()
    status["iter"] = 3000
    run_state.write(path, status)

    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["iter"] == 3000


def a_run_dir(root: Path, name: str, *, age_s: float) -> Path:
    run_dir = root / "jumper" / "jumper.tripod" / name
    run_dir.mkdir(parents=True)
    stamp = time.time() - age_s
    os.utime(run_dir, (stamp, stamp))
    return run_dir


def test_run_dirs_lists_the_run_directories_under_a_root(tmp_path: Path):
    first = a_run_dir(tmp_path, "2026-10-30_00-18-00", age_s=600)
    second = a_run_dir(tmp_path, "2026-10-30_03-18-13", age_s=300)

    assert sorted(run_state.run_dirs(tmp_path)) == sorted([first, second])


def test_run_dirs_ignores_files(tmp_path: Path):
    a_run_dir(tmp_path, "2026-10-30_00-18-00", age_s=600)
    (tmp_path / "jumper" / "jumper.tripod" / "notes.txt").write_text("x", encoding="utf-8")

    assert len(run_state.run_dirs(tmp_path)) == 1


def test_newest_run_dir_picks_the_one_just_written(tmp_path: Path):
    a_run_dir(tmp_path, "2026-10-30_00-18-00", age_s=600)
    newest = a_run_dir(tmp_path, "2026-10-30_03-18-13", age_s=1)

    assert run_state.newest_run_dir(time.time() - 60, root=tmp_path) == newest


def test_newest_run_dir_can_be_told_what_to_ignore(tmp_path: Path):
    old = a_run_dir(tmp_path, "2026-10-30_00-18-00", age_s=1)
    newest = a_run_dir(tmp_path, "2026-10-30_03-18-13", age_s=1)

    got = run_state.newest_run_dir(time.time() - 60, exclude={old}, root=tmp_path)

    assert got == newest


def test_newest_run_dir_returns_nothing_when_everything_is_older(tmp_path: Path):
    a_run_dir(tmp_path, "2026-10-30_00-18-00", age_s=600)

    assert run_state.newest_run_dir(time.time() - 60, root=tmp_path) is None


def test_newest_run_dir_returns_nothing_when_there_are_no_runs(tmp_path: Path):
    assert run_state.newest_run_dir(time.time() - 60, root=tmp_path) is None


@pytest.mark.parametrize("field", run_state.REQUIRED)
def test_every_required_field_is_named_by_a_constant(field: str):
    assert field in run_state.FIELDS


def test_the_fields_of_an_interrupted_run_are_in_the_contract():
    """`signal` 与 `sample_error` 也是 `long_run` 写进 `status.json` 的东西。

    `FIELDS` 自称是这个文件的全部字段；少写两个，照它取字段的读者就取不到
    「被什么叫停的」——而这是 `interrupted` 唯一的解释。
    """
    assert {"signal", "sample_error", "error"} <= set(run_state.FIELDS)

    status = {"v": run_state.SCHEMA_VERSION, "status": "interrupted", "pid": 1,
              "ts": 1793400000.0, "signal": 15}
    assert run_state.validate(status) == []
