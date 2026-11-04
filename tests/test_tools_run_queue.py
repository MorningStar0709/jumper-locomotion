"""`tools/run_queue.py` 的契约测试。

四件事被钉住，都是「一夜无人值守之后才知道出问题」的那一类：

1. 清单到命令的映射——参数顺序错了，训练照样跑，只是跑的是别的东西。
2. 清单本身被检查——空清单跑完 rc=0 会看起来像成功。
3. **完成度核验**：`long_run` 退出码 0 只说明进程没崩，不说明它跑到了第 10000 次迭代。
   这条是知识库 `reports/Voxline/long-task-online-health-checks.md` 的「退出 0 ≠ 完成」。
4. 失败即停与汇总——第一条崩了还继续跑，等于拿三倍时间换三个同样的失败；
   而「停在哪一条、为什么」必须留下记录，否则明早只有一堆目录。

放在 `tests/` 而不是 `tools/`：`pyproject.toml` 的 `testpaths` 只认这个目录。
"""

import itertools
import json
import sys
from pathlib import Path

import pytest

import run_queue
import run_state

REPO = Path(__file__).resolve().parent.parent
SAVE_INTERVAL = 500

_stamps = itertools.count()


def manifest(tmp_path, runs, **extra):
    path = tmp_path / "queue.json"
    path.write_text(json.dumps({"runs": runs, **extra}), encoding="utf-8")
    return path


def finished_run(root, *, iteration=9950, status="completed", rc=0,
                 checkpoints=(0, 5000, 9950), name="jumper.tripod"):
    """在 `root` 下造一个 `logs/<model>/<task>/<stamp>/` 运行目录。

    每次调用给一个不同的时间戳：两个 run 落在同一个目录里时，
    `newest_run_dir` 无法区分是谁写的，测试也就看不出「第二条没有新建目录」。
    """
    stamp = f"2026-10-30_00-00-{next(_stamps):02d}"
    run_dir = root / "logs" / "jumper" / name / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "status.json").write_text(
        json.dumps({"status": status, "rc": rc, "iter": iteration}), encoding="utf-8")
    for number in checkpoints:
        (run_dir / f"model_{number}.pt").write_bytes(b"")
    return run_dir


def test_the_command_hands_every_manifest_value_to_long_run():
    command = run_queue.build_command(
        {"task": "jumper.tripod", "seed": 3, "iterations": 10000, "num_envs": 4096})

    assert command[0] == sys.executable
    assert command[1:] == [
        "tools/long_run.py",
        "--task", "jumper.tripod",
        "--seed", "3",
        "--max-iterations", "10000",
        "--num_envs", "4096",
        "--headless",
    ]


def test_an_empty_manifest_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no runs"):
        run_queue.load_manifest(manifest(tmp_path, []))


def test_a_run_missing_a_field_names_the_field(tmp_path):
    path = manifest(tmp_path, [{"task": "jumper.tripod", "seed": 1}])

    with pytest.raises(ValueError, match="iterations"):
        run_queue.load_manifest(path)


def test_a_run_that_is_not_an_object_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not an object"):
        run_queue.load_manifest(manifest(tmp_path, ["jumper.tripod"]))


def test_a_field_the_queue_cannot_honour_is_refused_not_ignored(tmp_path):
    path = manifest(tmp_path, [{
        "task": "jumper.tripod", "seed": 1, "iterations": 10000, "num_envs": 4096,
        "save_interval": 5000,
    }])

    with pytest.raises(ValueError, match="save_interval"):
        run_queue.load_manifest(path)


def test_a_budget_that_is_not_an_integer_is_refused(tmp_path):
    path = manifest(tmp_path, [{
        "task": "jumper.tripod", "seed": 1, "iterations": "10000", "num_envs": 4096,
    }])

    with pytest.raises(ValueError, match="iterations"):
        run_queue.load_manifest(path)


@pytest.mark.parametrize("iterations", [
    run_queue.DEFAULT_SAVE_INTERVAL - 1,
    run_queue.DEFAULT_SAVE_INTERVAL,
])
def test_a_budget_at_or_below_the_save_interval_is_refused(tmp_path, iterations):
    """两侧都要钉住：低于边界时阈值为负，等于边界时阈值为 0，`last < expected` 都永不成立。"""
    path = manifest(tmp_path, [{
        "task": "jumper.tripod", "seed": 1,
        "iterations": iterations, "num_envs": 4096,
    }])

    with pytest.raises(ValueError, match="save interval"):
        run_queue.load_manifest(path)


def test_a_key_at_the_top_level_is_refused_not_ignored(tmp_path):
    """写在清单顶层的键同样会让人以为它生效了。"""
    path = manifest(tmp_path, [{
        "task": "jumper.tripod", "seed": 1, "iterations": 10000, "num_envs": 4096,
    }], save_interval=500)

    with pytest.raises(ValueError, match="save_interval"):
        run_queue.load_manifest(path)


def test_a_finished_run_verifies(tmp_path, monkeypatch):
    monkeypatch.setattr(run_state, "REPO", tmp_path)
    run_dir = finished_run(tmp_path)

    verdict = run_queue.verify_run(run_dir, 10000, SAVE_INTERVAL, 0)

    assert verdict["ok"] is True
    assert verdict["problems"] == []
    assert verdict["last_checkpoint"] == 9950


def test_a_run_that_stopped_early_is_caught(tmp_path, monkeypatch):
    monkeypatch.setattr(run_state, "REPO", tmp_path)
    run_dir = finished_run(tmp_path, iteration=5000, checkpoints=(0, 5000))

    verdict = run_queue.verify_run(run_dir, 10000, SAVE_INTERVAL, 0)

    assert verdict["ok"] is False
    assert any("9500" in problem for problem in verdict["problems"])


def test_a_run_stopping_exactly_at_the_budget_verifies(tmp_path, monkeypatch):
    """阈值是「最后一个 checkpoint ≥ iterations − save_interval」，边界本身必须通过。"""
    monkeypatch.setattr(run_state, "REPO", tmp_path)
    run_dir = finished_run(tmp_path, iteration=9500, checkpoints=(0, 5000, 9500))

    verdict = run_queue.verify_run(run_dir, 10000, SAVE_INTERVAL, 0)

    assert verdict["ok"] is True


def test_a_run_one_iteration_short_is_caught(tmp_path, monkeypatch):
    """拒绝的一侧也要钉在边界上：9499 必须失败，否则 `last < expected - 1` 能蒙混过关。"""
    monkeypatch.setattr(run_state, "REPO", tmp_path)
    run_dir = finished_run(tmp_path, iteration=9499, checkpoints=(0, 5000, 9499))

    verdict = run_queue.verify_run(run_dir, 10000, SAVE_INTERVAL, 0)

    assert verdict["ok"] is False
    assert any("9500" in problem for problem in verdict["problems"])


def test_a_crashed_run_is_caught_even_with_an_exit_code_of_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(run_state, "REPO", tmp_path)
    run_dir = finished_run(tmp_path, status="crashed", rc=1)

    verdict = run_queue.verify_run(run_dir, 10000, SAVE_INTERVAL, 0)

    assert verdict["ok"] is False
    assert any("crashed" in problem for problem in verdict["problems"])


def test_a_run_directory_without_a_status_file_is_caught(tmp_path, monkeypatch):
    monkeypatch.setattr(run_state, "REPO", tmp_path)
    run_dir = finished_run(tmp_path)
    (run_dir / "status.json").unlink()

    verdict = run_queue.verify_run(run_dir, 10000, SAVE_INTERVAL, 0)

    assert verdict["ok"] is False
    assert any("status.json" in problem for problem in verdict["problems"])


def test_a_status_file_that_is_not_json_is_caught(tmp_path, monkeypatch):
    monkeypatch.setattr(run_state, "REPO", tmp_path)
    run_dir = finished_run(tmp_path)
    (run_dir / "status.json").write_text("not json at all", encoding="utf-8")

    verdict = run_queue.verify_run(run_dir, 10000, SAVE_INTERVAL, 0)

    assert verdict["ok"] is False
    assert any("cannot read" in problem for problem in verdict["problems"])


def test_a_status_file_that_is_not_utf8_is_a_problem_not_a_crash(tmp_path, monkeypatch):
    """不是 UTF-8 的字节也要给判定；让它抛出去，汇总就永远不写。"""
    monkeypatch.setattr(run_state, "REPO", tmp_path)
    run_dir = finished_run(tmp_path)
    (run_dir / "status.json").write_bytes(b'{"status": "\xff\xfe completed"}')

    verdict = run_queue.verify_run(run_dir, 10000, SAVE_INTERVAL, 0)

    assert verdict["ok"] is False
    assert any("cannot read" in problem for problem in verdict["problems"])


def test_a_null_status_is_not_evidence_of_anything(tmp_path, monkeypatch):
    """`null` 是合法 JSON；它不能和「读失败」共用一个哨兵值。"""
    monkeypatch.setattr(run_state, "REPO", tmp_path)
    run_dir = finished_run(tmp_path)
    (run_dir / "status.json").write_text("null", encoding="utf-8")

    verdict = run_queue.verify_run(run_dir, 10000, SAVE_INTERVAL, 0)

    assert verdict["ok"] is False
    assert any("status record" in problem for problem in verdict["problems"])


def test_an_empty_status_object_is_not_evidence_of_anything(tmp_path, monkeypatch):
    """`{}` 是合法 JSON，却什么也没说——它不能当成功处理，也不能被静默跳过。"""
    monkeypatch.setattr(run_state, "REPO", tmp_path)
    run_dir = finished_run(tmp_path)
    (run_dir / "status.json").write_text("{}", encoding="utf-8")

    verdict = run_queue.verify_run(run_dir, 10000, SAVE_INTERVAL, 0)

    assert verdict["ok"] is False
    assert verdict["problems"]


def test_a_status_file_that_is_not_an_object_is_caught(tmp_path, monkeypatch):
    """`[]` / `"x"` 也是合法 JSON；取值前必须先确认它是一个对象。"""
    monkeypatch.setattr(run_state, "REPO", tmp_path)
    run_dir = finished_run(tmp_path)
    (run_dir / "status.json").write_text("[1, 2, 3]", encoding="utf-8")

    verdict = run_queue.verify_run(run_dir, 10000, SAVE_INTERVAL, 0)

    assert verdict["ok"] is False
    assert any("status record" in problem for problem in verdict["problems"])


def test_a_missing_run_directory_is_caught(tmp_path, monkeypatch):
    monkeypatch.setattr(run_state, "REPO", tmp_path)

    verdict = run_queue.verify_run(None, 10000, SAVE_INTERVAL, 1)

    assert verdict["ok"] is False
    assert any("no run directory" in problem for problem in verdict["problems"])


def test_a_run_that_cannot_be_started_is_recorded(tmp_path, monkeypatch):
    """子进程起不来时（解释器坏了、`tools/long_run.py` 不见了），汇总仍要写。"""
    monkeypatch.setattr(run_state, "REPO", tmp_path)

    def spawn(command):
        raise OSError("cannot execute")

    path = manifest(tmp_path, [
        {"task": "jumper.tripod", "seed": 1, "iterations": 10000, "num_envs": 4096},
    ])

    rc = run_queue.main([str(path)], spawn=spawn)

    assert rc != 0
    result = json.loads(path.with_suffix(".result.json").read_text(encoding="utf-8"))
    assert result["runs"][0]["ok"] is False
    assert any("cannot start" in problem for problem in result["runs"][0]["problems"])


def test_a_directory_left_by_an_earlier_run_is_not_evidence(tmp_path, monkeypatch):
    """上一条 run 刚写过盘的目录，不能被当成这一条的证据。"""
    monkeypatch.setattr(run_state, "REPO", tmp_path)
    finished_run(tmp_path, name="jumper.tripod")

    def spawn(command):
        return 0  # 什么目录也不建

    path = manifest(tmp_path, [
        {"task": "jumper.tripod", "seed": 1, "iterations": 10000, "num_envs": 4096},
    ])

    rc = run_queue.main([str(path)], spawn=spawn)

    assert rc != 0
    result = json.loads(path.with_suffix(".result.json").read_text(encoding="utf-8"))
    assert result["runs"][0]["run_dir"] is None


def test_the_queue_runs_in_order_and_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(run_state, "REPO", tmp_path)
    calls = []

    def spawn(command):
        calls.append(command)
        finished_run(tmp_path)
        return 0

    path = manifest(tmp_path, [
        {"name": "first", "task": "jumper.tripod", "seed": 1,
         "iterations": 10000, "num_envs": 4096},
        {"name": "second", "task": "jumper.tripod", "seed": 2,
         "iterations": 10000, "num_envs": 4096},
    ])

    rc = run_queue.main([str(path)], spawn=spawn)

    assert rc == 0
    assert [c[c.index("--seed") + 1] for c in calls] == ["1", "2"]
    result = json.loads(path.with_suffix(".result.json").read_text(encoding="utf-8"))
    assert [run["name"] for run in result["runs"]] == ["first", "second"]
    assert [run["ok"] for run in result["runs"]] == [True, True]
    assert [run["run_dir"] for run in result["runs"]].count(None) == 0


def test_a_failed_run_stops_the_queue_and_says_which(tmp_path, monkeypatch):
    monkeypatch.setattr(run_state, "REPO", tmp_path)
    calls = []

    def spawn(command):
        calls.append(command)
        return 1

    path = manifest(tmp_path, [
        {"name": "first", "task": "jumper.tripod", "seed": 1,
         "iterations": 10000, "num_envs": 4096},
        {"name": "second", "task": "jumper.tripod", "seed": 2,
         "iterations": 10000, "num_envs": 4096},
    ])

    rc = run_queue.main([str(path)], spawn=spawn)

    assert rc != 0
    assert len(calls) == 1
    result = json.loads(path.with_suffix(".result.json").read_text(encoding="utf-8"))
    assert len(result["runs"]) == 1
    assert result["runs"][0]["ok"] is False
    assert result["runs"][0]["name"] == "first"
    assert result["runs"][0]["problems"]


def test_the_shipped_manifest_is_three_seeds_of_ten_thousand():
    runs = run_queue.load_manifest(REPO / "manifests" / "tripod-3seeds.json")

    assert [run["seed"] for run in runs] == [0, 1, 2]
    assert {run["iterations"] for run in runs} == {10000}
    assert {run["num_envs"] for run in runs} == {4096}


def test_a_run_can_carry_its_own_flags():
    command = run_queue.build_command({
        "task": "jumper.tripod", "seed": 0, "iterations": 3000, "num_envs": 4096,
        "args": ["--delay"],
    })

    assert command[-1] == "--delay"


def test_flags_that_are_not_flags_are_refused(tmp_path):
    path = manifest(tmp_path, [{
        "task": "jumper.tripod", "seed": 0, "iterations": 3000, "num_envs": 4096,
        "args": ["--delay", "oops"],
    }])

    with pytest.raises(ValueError, match="args"):
        run_queue.load_manifest(path)


def test_a_manifest_without_flags_still_builds_the_same_command():
    plain = run_queue.build_command(
        {"task": "jumper.tripod", "seed": 0, "iterations": 3000, "num_envs": 4096})

    assert plain[-1] == "--headless"


# --- 断点续跑 -----------------------------------------------------------------
# 一夜被中断之后，重跑整份清单会把已经跑完的种子从头再来一小时；
# 而只给 `--resume` 不给目录，训练会按 mtime 续到别的种子上。
# 所以队列要自己记账：谁跑完了、谁停在哪个 checkpoint。

def result_with(path, records):
    path.with_suffix(".result.json").write_text(
        json.dumps({"manifest": str(path), "runs": records}, ensure_ascii=False),
        encoding="utf-8")


def one_run(name="one", iterations=3000):
    return {"name": name, "task": "jumper.tripod", "seed": 2,
            "iterations": iterations, "num_envs": 4096}


def test_a_run_that_already_verified_is_not_started_again(tmp_path, monkeypatch):
    monkeypatch.setattr(run_state, "REPO", tmp_path)
    done = finished_run(tmp_path, iteration=2950, checkpoints=(0, 1500, 2950))
    path = manifest(tmp_path, [one_run()])
    result_with(path, [{"name": "one", "run_dir": str(done), "rc": 0, "ok": True,
                        "problems": [], "last_checkpoint": 2950, "checkpoints": 3}])
    calls = []

    rc = run_queue.main([str(path)], spawn=lambda command: calls.append(command) or 0)

    assert rc == 0
    assert calls == []
    result = json.loads(path.with_suffix(".result.json").read_text(encoding="utf-8"))
    assert result["runs"][0]["ok"] is True
    assert result["runs"][0]["run_dir"] == str(done)


def test_a_partial_run_resumes_from_its_last_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(run_state, "REPO", tmp_path)
    partial = finished_run(tmp_path, iteration=900, checkpoints=(0, 900),
                           status="crashed", rc=143)
    path = manifest(tmp_path, [one_run()])
    result_with(path, [{"name": "one", "run_dir": str(partial), "rc": 143, "ok": False,
                        "problems": ["status is 'crashed', not 'completed'"],
                        "last_checkpoint": 900, "checkpoints": 2}])
    calls = []

    def spawn(command):
        calls.append(command)
        finished_run(tmp_path, iteration=2950, checkpoints=(2950,))
        return 0

    rc = run_queue.main([str(path)], spawn=spawn)

    assert rc == 0
    command = calls[0]
    assert command[command.index("--max-iterations") + 1] == "2100"
    assert command[command.index("--checkpoint") + 1] == str(partial)
    result = json.loads(path.with_suffix(".result.json").read_text(encoding="utf-8"))
    assert result["runs"][0]["ok"] is True
    assert result["runs"][0]["resumed_from"] == str(partial)


def test_a_resumed_run_is_verified_against_the_full_budget(tmp_path, monkeypatch):
    """续跑的 checkpoint 编号继续往上走，所以阈值仍是清单里的总数，不是剩余数。"""
    monkeypatch.setattr(run_state, "REPO", tmp_path)
    partial = finished_run(tmp_path, iteration=900, checkpoints=(0, 900))
    path = manifest(tmp_path, [one_run()])
    result_with(path, [{"name": "one", "run_dir": str(partial), "rc": 0, "ok": False,
                        "problems": ["stopped early"], "last_checkpoint": 900,
                        "checkpoints": 2}])

    calls = []

    def spawn(command):
        calls.append(command)
        finished_run(tmp_path, iteration=2900, checkpoints=(2900,))
        return 0

    rc = run_queue.main([str(path)], spawn=spawn)

    assert calls[0][calls[0].index("--checkpoint") + 1] == str(partial)
    assert rc != 0
    result = json.loads(path.with_suffix(".result.json").read_text(encoding="utf-8"))
    assert "2900" in " ".join(result["runs"][0]["problems"])


def test_a_result_file_is_written_before_the_next_run_starts(tmp_path, monkeypatch):
    """中断随时可能来；第一条的成绩必须在第二条开始之前就落盘。"""
    monkeypatch.setattr(run_state, "REPO", tmp_path)
    path = manifest(tmp_path, [one_run("first"), one_run("second")])
    seen = {}

    def spawn(command):
        if "first" not in seen:
            seen["first"] = command
            finished_run(tmp_path, iteration=2950, checkpoints=(2950,))
            return 0
        seen["second"] = json.loads(
            path.with_suffix(".result.json").read_text(encoding="utf-8"))
        return 1

    run_queue.main([str(path)], spawn=spawn)

    assert [run["name"] for run in seen["second"]["runs"]] == ["first"]
    assert seen["second"]["runs"][0]["ok"] is True


def test_a_run_with_nothing_left_to_do_is_not_started(tmp_path, monkeypatch):
    monkeypatch.setattr(run_state, "REPO", tmp_path)
    partial = finished_run(tmp_path, iteration=3000, checkpoints=(0, 3000))
    path = manifest(tmp_path, [one_run()])
    result_with(path, [{"name": "one", "run_dir": str(partial), "rc": 143, "ok": False,
                        "problems": ["crashed"], "last_checkpoint": 3000,
                        "checkpoints": 2}])
    calls = []

    rc = run_queue.main([str(path)], spawn=lambda command: calls.append(command) or 0)

    assert rc == 0
    assert calls == []
    result = json.loads(path.with_suffix(".result.json").read_text(encoding="utf-8"))
    assert result["runs"][0]["ok"] is True


def test_a_broken_result_file_does_not_stop_the_queue(tmp_path, monkeypatch):
    monkeypatch.setattr(run_state, "REPO", tmp_path)
    path = manifest(tmp_path, [one_run()])
    path.with_suffix(".result.json").write_text("{not json", encoding="utf-8")
    calls = []

    def spawn(command):
        calls.append(command)
        finished_run(tmp_path, iteration=2950, checkpoints=(2950,))
        return 0

    rc = run_queue.main([str(path)], spawn=spawn)

    assert rc == 0
    assert len(calls) == 1


def test_the_real_spawn_runs_from_the_repository_root(monkeypatch):
    """不带注入 `spawn` 的那条路：`main()` 用 `_spawn`，它必须能跑起来。

    这条曾经在改名后指向一个不存在的模块名，而所有测试都注入了 `spawn`，所以整套
    测试绿着、真实入口在第一条就 `NameError`。
    """
    recorded = {}

    def fake_call(command, cwd=None, **kwargs):
        recorded["command"] = command
        recorded["cwd"] = cwd
        return 0

    monkeypatch.setattr(run_queue.subprocess, "call", fake_call)

    assert run_queue._spawn(["tools/long_run.py", "--task", "x"]) == 0
    assert recorded["cwd"] == run_state.REPO


def test_the_ledger_is_written_in_one_piece(tmp_path, monkeypatch):
    """账本是「被打断」时要靠的那一份，不能写到一半。"""
    path = tmp_path / "night.result.json"
    replaced = {}
    real = run_queue.os.replace

    def spy(src, dst):
        replaced["src"] = Path(src)
        replaced["dst"] = Path(dst)
        return real(src, dst)

    monkeypatch.setattr(run_queue.os, "replace", spy)
    run_queue._write_result(path, tmp_path / "night.json", "2026-10-30T00:00:00",
                            [{"name": "a", "ok": True}])

    assert replaced["dst"] == path
    assert replaced["src"].name.endswith(".tmp")
    assert not replaced["src"].exists()
    assert json.loads(path.read_text(encoding="utf-8"))["runs"][0]["name"] == "a"


def test_two_runs_that_differ_only_by_flags_have_different_labels():
    plain = {"task": "jumper.tripod", "seed": 2, "iterations": 3000, "num_envs": 4096}
    delayed = {**plain, "args": ["--delay"]}
    assert run_queue.label_of(plain) != run_queue.label_of(delayed)


def a_run(**extra):
    base = {"task": "jumper.tripod", "seed": 0, "iterations": 3000, "num_envs": 4096}
    base.update(extra)
    return base


def test_the_manifest_cannot_set_the_flags_the_queue_owns(tmp_path):
    for flag in ("--resume", "--max-iterations=5000", "--seed"):
        path = manifest(tmp_path, [a_run(args=[flag])])
        assert run_queue.main([str(path)]) == 2, flag


def test_a_stale_ledger_entry_does_not_stop_the_queue(tmp_path, monkeypatch):
    """账本说跑完、盘上没证据：报出来，但下一条照跑。"""
    # `run_state.REPO` 指向这台机器的 `logs/`：不换掉，这条测试的结果取决于
    # 跑它的那台机器上碰巧有什么目录。
    monkeypatch.setattr(run_state, "REPO", tmp_path)
    path = manifest(tmp_path, [a_run(name="first"), a_run(name="second", seed=2)])
    path.with_suffix(".result.json").write_text(json.dumps({"runs": [
        {"name": "first", "ok": True, "run_dir": str(tmp_path / "gone"), "rc": 0},
    ]}), encoding="utf-8")
    seen = []

    rc = run_queue.main([str(path)], spawn=lambda command: seen.append(command) or 0)

    assert rc == 2
    assert len(seen) == 1, "the second run still ran"


# --- 从某一条更早的 run 续跑 ---------------------------------------------------
# 阶梯的每一级都是「上一级的最后一步接着往下跑」，而清单里写不了 `--checkpoint`
# （队列把它当自有参数拒收）。于是续跑只能靠人手一条一条发——越级正是从这里来的。


def a_directory_with_checkpoints(root, numbers, name="jumper.tripod"):
    run_dir = root / "logs" / "jumper" / name / f"2026-10-30_23-00-{next(_stamps):02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for number in numbers:
        (run_dir / f"model_{number}.pt").write_bytes(b"")
    return run_dir


def test_a_run_can_continue_from_an_earlier_directory(tmp_path):
    source = a_directory_with_checkpoints(tmp_path, (9999,))

    command = run_queue.build_command(a_run(start_from=str(source)))

    assert command[command.index("--checkpoint") + 1] == str(source)
    assert command[command.index("--max-iterations") + 1] == "3000"


def test_a_start_that_is_not_there_is_refused_at_load(tmp_path):
    """路径打错一个字符就当从零开始跑，是这条清单能犯的最贵的错。"""
    missing = tmp_path / "logs" / "jumper" / "jumper.tripod" / "2026-10-30_00-00-00"

    with pytest.raises(ValueError, match="start_from"):
        run_queue.load_manifest(manifest(tmp_path, [a_run(start_from=str(missing))]))


def test_a_resumed_run_stopping_early_is_caught(tmp_path, monkeypatch):
    """编号接着起点往上走，阈值也必须跟着起点走，否则 10200 会被当成「超过 2500」。"""
    monkeypatch.setattr(run_state, "REPO", tmp_path)
    run_dir = finished_run(tmp_path, iteration=10200, checkpoints=(10000, 10200))

    verdict = run_queue.verify_run(run_dir, 3000, SAVE_INTERVAL, 0, start_iteration=9999)

    assert verdict["ok"] is False
    assert any("12499" in problem for problem in verdict["problems"])


def test_a_resumed_run_that_reached_its_budget_verifies(tmp_path, monkeypatch):
    monkeypatch.setattr(run_state, "REPO", tmp_path)
    run_dir = finished_run(tmp_path, iteration=12998, checkpoints=(10000, 12499, 12998))

    verdict = run_queue.verify_run(run_dir, 3000, SAVE_INTERVAL, 0, start_iteration=9999)

    assert verdict["ok"] is True
    assert verdict["last_checkpoint"] == 12998


def test_a_resumed_run_is_recorded_with_where_it_started(tmp_path, monkeypatch):
    monkeypatch.setattr(run_state, "REPO", tmp_path)
    source = finished_run(tmp_path, iteration=9999, checkpoints=(9999,))
    path = manifest(tmp_path, [a_run(name="arm-d", start_from=str(source))])
    calls = []

    def spawn(command):
        calls.append(command)
        finished_run(tmp_path, iteration=12998, checkpoints=(12950, 12998))
        return 0

    rc = run_queue.main([str(path)], spawn=spawn)

    assert rc == 0
    assert calls[0][calls[0].index("--checkpoint") + 1] == str(source)
    result = json.loads(path.with_suffix(".result.json").read_text(encoding="utf-8"))
    assert result["runs"][0]["started_from"] == str(source)


def test_the_resume_baseline_is_the_checkpoint_training_will_load(tmp_path):
    """`train.py` 按 **mtime** 取最新的那个文件，不是按名字里的编号。

    `mjrl.checkpoint._sort_key` 记着理由：续跑写出的新 run 编号可能比盘上的旧文件
    小，所以「刚训的那个」是 mtime 最新的，不是编号最大的。账本若按编号算基线，
    它算的是另一个文件——阈值从此与训练实际加载的东西无关。
    """
    import os

    source = tmp_path / "run"
    source.mkdir()
    older = source / "model_500.pt"
    newer = source / "model_100.pt"
    older.write_bytes(b"")
    newer.write_bytes(b"")
    os.utime(older, (1_600_000_000, 1_600_000_000))
    os.utime(newer, (1_700_000_000, 1_700_000_000))

    assert run_queue.start_iteration_of({"start_from": str(source)}) == 100


def test_a_relative_start_from_is_read_from_the_repository_root(tmp_path, monkeypatch):
    """清单里写的是 `logs/...`，而队列子进程的 cwd 是仓库根。

    在别处启动队列时，`Path(start_from)` 会按队列进程的 cwd 解析，校验失败（或校验
    通过而子进程找不到），两种都比直接说「路径不存在」更难查。
    """
    repo = tmp_path / "repo"
    (repo / "logs" / "jumper" / "tripod" / "run").mkdir(parents=True)
    (repo / "logs" / "jumper" / "tripod" / "run" / "model_800.pt").write_bytes(b"")
    monkeypatch.setattr(run_state, "REPO", repo, raising=False)
    monkeypatch.setattr(run_queue.run_state, "REPO", repo, raising=False)

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert run_queue.start_iteration_of(
        {"start_from": "logs/jumper/tripod/run"}) == 800


def test_the_ledger_key_tells_two_starting_points_apart(tmp_path):
    """同一个 task/seed/预算，从 3000 续和从 10000 续是两条不同的训练。"""
    base = {"task": "jumper.tripod", "seed": 2, "iterations": 7000, "num_envs": 4096}

    a = run_queue.label_of({**base, "start_from": "logs/a"})
    b = run_queue.label_of({**base, "start_from": "logs/b"})

    assert a != b


def test_the_remaining_budget_is_measured_from_the_resume_point(tmp_path):
    """续跑条的 `iterations` 是「再跑多少」，而 checkpoint 的编号是绝对的。

    7000 预算从 10000 起步，跑到 10600 时剩下的应该是 6400，不是 7000-10600。
    """
    source = tmp_path / "source"
    source.mkdir()
    for number in (9950, 10000):
        (source / f"model_{number}.pt").write_bytes(b"")

    prior_dir = tmp_path / "prior"
    prior_dir.mkdir()
    for number in (10400, 10600):
        (prior_dir / f"model_{number}.pt").write_bytes(b"")

    run = {"name": "B-ext", "task": "jumper.tripod", "seed": 2, "iterations": 7000,
           "num_envs": 4096, "start_from": str(source)}
    prior = {"ok": False, "run_dir": str(prior_dir)}

    plan = run_queue.plan_run(run, prior)

    assert plan["action"] == "resume"
    assert plan["command"][plan["command"].index("--max-iterations") + 1] == "6400"


def test_a_resumed_entry_is_not_given_a_second_checkpoint(tmp_path):
    """清单带 `start_from`、账本又留了上次的 `run_dir` 时，只能有一个 `--checkpoint`。

    两个都给，argparse 取最后一个，训练续到的地方就与账本算阈值的地方不是同一个。
    """
    source = tmp_path / "source"
    source.mkdir()
    for number in (9950, 10000):
        (source / f"model_{number}.pt").write_bytes(b"")

    prior_dir = tmp_path / "prior"
    prior_dir.mkdir()
    for number in (10200, 10400):
        (prior_dir / f"model_{number}.pt").write_bytes(b"")

    run = {"name": "B-ext", "task": "jumper.tripod", "seed": 2, "iterations": 7000,
           "num_envs": 4096, "start_from": str(source)}
    plan = run_queue.plan_run(run, {"ok": False, "run_dir": str(prior_dir)})

    assert plan["command"].count("--checkpoint") == 1
    assert str(prior_dir) in plan["command"]


def test_the_summary_names_the_manifest_that_was_run(tmp_path):
    """汇总里的 `manifest` 是跑的那份清单，不是汇总自己。

    写成同一个值，归档字段就永远指向结果文件——而它存在的理由正是「被打断之后还能
    看出跑的是哪份清单」。
    """
    manifest = tmp_path / "night.json"
    manifest.write_text('{"runs": [{"name": "a", "task": "jumper.tripod", "seed": 0, '
                        '"iterations": 60, "num_envs": 8}]}', encoding="utf-8")
    run_queue.main([str(manifest)], spawn=lambda command: 0)

    summary = json.loads((tmp_path / "night.result.json").read_text(encoding="utf-8"))
    assert summary["manifest"] == str(manifest)


def test_a_manifest_cannot_reach_the_supervisors_own_switches(tmp_path):
    """`long_run` 的四个开关会被它自己摘掉，写进清单只会悄悄改掉包装层策略。"""
    for flag in ("--allow-over-budget", "--interval=0.001", "--reserve-mib=0",
                 "--print-command"):
        manifest = tmp_path / "night.json"
        manifest.write_text(json.dumps({"runs": [{
            "name": "a", "task": "jumper.tripod", "seed": 0,
            "iterations": 60, "num_envs": 8, "args": [flag]}]}), encoding="utf-8")

        with pytest.raises(ValueError, match="the queue decides"):
            run_queue.load_manifest(manifest)


def test_a_name_that_is_not_a_string_is_refused(tmp_path):
    """账本的键就是这个值；非字符串写进结果文件后两个读取方都会把它丢掉。"""
    manifest = tmp_path / "night.json"
    manifest.write_text(json.dumps({"runs": [{
        "name": 5, "task": "jumper.tripod", "seed": 0,
        "iterations": 60, "num_envs": 8}]}), encoding="utf-8")

    with pytest.raises(ValueError, match="name must be a string"):
        run_queue.load_manifest(manifest)


def test_a_run_that_cannot_be_planned_still_leaves_a_summary(tmp_path, monkeypatch):
    """规划抛异常时汇总也得写下去——它是「被打断」时要靠的那一份。"""
    manifest = tmp_path / "night.json"
    manifest.write_text(json.dumps({"runs": [{
        "name": "a", "task": "jumper.tripod", "seed": 0,
        "iterations": 60, "num_envs": 8}]}), encoding="utf-8")

    def explode(run, prior):
        raise ValueError("model_500.pt is not a checkpoint this can resume from")

    monkeypatch.setattr(run_queue, "plan_run", explode)
    # 这条用例走不到 spawn，但 `main()` 里任何一次对真实 `logs/` 的 glob 都会让结果
    # 取决于这台机器上跑过什么。
    monkeypatch.setattr(run_queue.run_state, "REPO", tmp_path)

    assert run_queue.main([str(manifest)], spawn=lambda command: 0) == 2
    summary = json.loads((tmp_path / "night.result.json").read_text(encoding="utf-8"))
    assert summary["runs"][0]["ok"] is False
    assert "cannot plan this run" in summary["runs"][0]["problems"][0]


def test_a_child_killed_by_a_signal_does_not_become_rc_247(tmp_path):
    """`subprocess.call` 用负值表示「被信号杀掉」。

    原样交给 `SystemExit` 时 shell 看到的是 256-9=247；`long_run` 自己已经把这种
    情况换成 `128+n`，队列不能在最后一步把它变回一个谁也不认识的数。
    """
    manifest = tmp_path / "night.json"
    manifest.write_text(json.dumps({"runs": [{
        "name": "a", "task": "jumper.tripod", "seed": 0,
        "iterations": 60, "num_envs": 8}]}), encoding="utf-8")

    assert run_queue.main([str(manifest)], spawn=lambda command: -9) == 128 + 9
    summary = json.loads((tmp_path / "night.result.json").read_text(encoding="utf-8"))
    assert summary["runs"][0]["rc"] == 128 + 9


def test_evidence_that_cannot_be_read_still_leaves_a_summary(tmp_path, monkeypatch):
    """`newest_run_dir` 也是磁盘操作。

    它 `stat` 的目录可能正好被人清理掉（磁盘水位那条规矩就是叫人去清旧 run）。
    不包住这一步，循环会带着 traceback 退出，末尾那次 `_write_result` 一次都不跑。
    """
    manifest = tmp_path / "night.json"
    manifest.write_text(json.dumps({"runs": [{
        "name": "a", "task": "jumper.tripod", "seed": 0,
        "iterations": 60, "num_envs": 8}]}), encoding="utf-8")

    def gone(*args, **kwargs):
        raise OSError("directory vanished")

    monkeypatch.setattr(run_queue.run_state, "newest_run_dir", gone)
    # `main()` 里还有 `run_dirs()` 也在 glob 真实的 `logs/`：正在跑的训练会不断
    # 刷新 run 目录的 mtime，2 秒容差可能把别的 run 当成这一条的。
    monkeypatch.setattr(run_queue.run_state, "REPO", tmp_path)

    assert run_queue.main([str(manifest)], spawn=lambda command: 0) == 2
    summary = json.loads((tmp_path / "night.result.json").read_text(encoding="utf-8"))
    assert summary["runs"][0]["ok"] is False
    assert "cannot read the evidence" in summary["runs"][0]["problems"][0]


def test_a_logs_directory_that_cannot_be_listed_still_leaves_a_summary(tmp_path, monkeypatch):
    """`run_dirs()` 是 `main()` 里四条「规划之后」失败路径中唯一没被钉住的。

    它的契约与另外三条不同：`rc` 是 `None`（还没启动过任何东西），而且汇总要靠
    循环外面那次 `_write_result` 落下。去掉 `break` 的话 `known` 未绑定，循环会
    带 traceback 走掉——正是这一段存在的理由。
    """
    manifest = tmp_path / "night.json"
    manifest.write_text(json.dumps({"runs": [{
        "name": "a", "task": "jumper.tripod", "seed": 0,
        "iterations": 60, "num_envs": 8}]}), encoding="utf-8")
    monkeypatch.setattr(run_queue.run_state, "REPO", tmp_path)

    def unlistable(root=None):
        raise OSError("logs is gone")

    monkeypatch.setattr(run_queue.run_state, "run_dirs", unlistable)
    started = []

    assert run_queue.main([str(manifest)], spawn=lambda command: started.append(command) or 0) == 2
    assert started == []          # 没启动过任何东西，`rc` 就不是一个数
    summary = json.loads((tmp_path / "night.result.json").read_text(encoding="utf-8"))
    assert summary["runs"][0]["rc"] is None
    assert "cannot read the logs directory" in summary["runs"][0]["problems"][0]
