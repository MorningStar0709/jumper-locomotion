"""每任务一份显存标定。

`960 MiB + 2.08 MiB/env` 是在 `jumper.tripod` + warp 上量出来的。把它当全局常量，
换任务就会算得又准又没依据——准到不会触发警告，错到会挡掉本来跑得下的规模。
这里钉住：有标定就用标定，没有就用最保守的那一份，并且**说出来**。
"""

from __future__ import annotations

import json
from pathlib import Path

import long_run


def test_the_shipped_calibration_covers_our_task():
    table = long_run.load_calibrations()

    tripod = table["jumper.tripod"]
    assert tripod["overhead_mib"] > 0
    assert tripod["per_env_mib"] > 0


def test_a_known_task_gets_the_numbers_that_were_measured_for_it():
    """钉住的是「查表查对了」，不是某个数。

    显存标定随实测重写：2026-10-29 那版是 `960 + 2.08/env`（nconmax 128 时代），
    2026-11-01 按 256 的实测改写成 `960 + 3.01/env`。把旧数写死在这里，改标定
    就得改测试——那是让测试替测量签字。
    """
    overhead, per_env, known = long_run.calibration_for("jumper.tripod")
    entry = long_run.load_calibrations()["jumper.tripod"]

    assert (overhead, per_env) == (entry["overhead_mib"], entry["per_env_mib"])
    assert known is True


def test_an_unknown_task_gets_the_largest_numbers_on_record():
    """保守在这里的意思是「最大的那个」：宁可拒绝一条跑得下的 run，不要 OOM 一条跑着的。"""
    overhead, per_env, known = long_run.calibration_for("jumper.ripple")

    assert known is False
    assert overhead == max(entry["overhead_mib"] for entry in long_run.load_calibrations().values())
    assert per_env == max(entry["per_env_mib"] for entry in long_run.load_calibrations().values())


def test_no_task_at_all_is_not_known():
    """不知道是哪个任务时，退回表里最大的那一组，而不是模块常量。

    这两个数曾经相等，于是「取最大」与「取常量」在测试里分不出来；标定一改，
    差别就露出来了。
    """
    overhead, per_env, known = long_run.calibration_for(None)
    table = long_run.load_calibrations().values()

    assert known is False
    assert (overhead, per_env) == (
        max(entry["overhead_mib"] for entry in table),
        max(entry["per_env_mib"] for entry in table),
    )


def test_a_broken_calibration_file_falls_back_to_the_module_constants(tmp_path: Path, monkeypatch):
    broken = tmp_path / "calibration.json"
    broken.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(long_run, "CALIBRATION_PATH", broken)

    assert long_run.calibration_for("jumper.tripod") == (
        long_run.VRAM_OVERHEAD_MIB,
        long_run.VRAM_PER_ENV_MIB,
        False,
    )


def test_a_calibration_entry_without_numbers_is_ignored(tmp_path: Path, monkeypatch):
    table = tmp_path / "calibration.json"
    table.write_text(json.dumps({"jumper.tripod": {"note": "half written"}}), encoding="utf-8")
    monkeypatch.setattr(long_run, "CALIBRATION_PATH", table)

    assert long_run.load_calibrations() == {}


def test_estimate_uses_the_numbers_it_is_given():
    assert long_run.estimate_vram_mib(1000, overhead=1000.0, per_env=2.0) == 3000


def test_the_budget_can_be_judged_with_a_tasks_own_numbers():
    cheap = long_run.fit_verdict(4096, free_mib=20000, reserve_mib=1024,
                                 allow_over_budget=False, overhead=960.0, per_env=2.08)
    dear = long_run.fit_verdict(4096, free_mib=20000, reserve_mib=1024,
                                allow_over_budget=False, overhead=4000.0, per_env=6.0)

    assert cheap["ok"] is True
    assert dear["ok"] is False
