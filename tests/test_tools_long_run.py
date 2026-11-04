"""`tools/long_run.py` 的 stdout 解析测试。

判据来自真实冒烟输出，不是构造的样例——这三行的原始形态（含前导空格）：

                             Iteration time: 0.73s
                               Time elapsed: 0:00:02
                                        ETA: 0:00:00

`parse_line` 的契约：能认的行返回 `(key, value)`，认不出的返回 `(None, None)`,
**任何输入都不许抛异常**——日志格式变了只该让字段变空，不该让 wrapper 崩掉。

放在 `tests/` 而不是 `tools/` 的原因：`pyproject.toml` 的 `testpaths = ["tests"]`
与 CI 的 `pytest tests/` 都只认这个目录，放 `tools/` 等于让契约无人看守。

时间行的来源是 `rl/rsl_rl/utils/logger.py:277-278` 的 `datetime.timedelta`，
越过 24 小时会多出 `N day, ` 前缀——那条路径同样要能解析。
"""

import errno
import json
import signal
import sys

import pytest

import long_run
import run_state
from long_run import _own_args, parse_line


def test_peak_memory_is_the_high_water_mark_of_the_run(monkeypatch):
    """`mem_mib` 是采样那一刻的数，`peak_mem_mib` 才是这一跑要回答的问题。"""
    readings = iter([{"mem_mib": 12000.0}, {"mem_mib": 13728.0}, {"mem_mib": 9000.0}])
    monkeypatch.setattr(long_run, "sample_gpu", lambda: next(readings))
    status = {}

    for _ in range(3):
        long_run._sample(status)

    assert status["mem_mib"] == 9000.0
    assert status["peak_mem_mib"] == 13728.0


def test_a_post_exit_sample_cannot_lower_the_memory_a_run_used(monkeypatch):
    """2026-10-31 11-12-58 的现场：终态那次采样取在 `proc.wait()` 之后。

    子进程已经退出，显存回到 66 MiB，于是「这次训练用了多少」被写成 66。
    """
    monkeypatch.setattr(long_run, "sample_gpu", lambda: {"mem_mib": 13728.0})
    status = {}
    long_run._sample(status)

    monkeypatch.setattr(long_run, "sample_gpu", lambda: {"mem_mib": 66.0})
    long_run._sample(status, alive=False)

    assert status["mem_mib"] == 13728.0
    assert status["peak_mem_mib"] == 13728.0


def test_memory_stays_unknown_when_nothing_was_sampled(monkeypatch):
    """没采到就是没采到——0 MiB 会被当成一次真的测量。"""
    monkeypatch.setattr(long_run, "sample_gpu", lambda: {"mem_mib": None})
    status = {}

    long_run._sample(status)

    assert status.get("peak_mem_mib") is None


def test_iteration_time_with_padding():
    assert parse_line("                         Iteration time: 0.73s") == ("iter_time", 0.73)
    assert parse_line("Iteration time: 2.25s") == ("iter_time", 2.25)


def test_time_elapsed_to_seconds():
    assert parse_line("                           Time elapsed: 0:00:09") == ("elapsed", 9)
    assert parse_line("Time elapsed: 1:02:03") == ("elapsed", 3723)


def test_timedelta_past_one_day():
    # `timedelta` 的渲染：满 24 小时候前缀变成 `N day, `，不是 `24:xx:xx`。
    assert parse_line("Time elapsed: 1 day, 3:04:05") == ("elapsed", 97445)
    assert parse_line("ETA: 2 days, 0:00:01") == ("eta", 172801)


def test_eta_to_seconds():
    assert parse_line("                                    ETA: 0:00:00") == ("eta", 0)
    assert parse_line("ETA: 0:35:10") == ("eta", 2110)


def test_unknown_line_is_silent():
    assert parse_line("") == (None, None)
    assert parse_line("[mjrl] backend=warp device=cuda:0 num_envs=256 (auto)") == (None, None)
    assert parse_line("Episode_Reward/track_linear_velocity: 0.0486") == (None, None)


def test_never_raises_on_junk():
    for junk in ("\x00\x01", "a" * 10000, "ETA:", "Iteration time: abcs"):
        assert parse_line(junk) == (None, None)


def test_dry_run_is_forwarded_to_train_py():
    # `--dry-run` 是 `scripts/train.py` 自己的开关（`_cli.py`:157）。wrapper 曾经
    # 把它截走，于是「预演」变成「真跑」——这条测试守的就是那个回归。
    own, passthrough = _own_args(["--task", "jumper.flat", "--dry-run"])
    assert "--dry-run" in passthrough
    assert own["print_command"] is False


def test_a_dry_run_does_not_need_the_card():
    # `--dry-run` at `scripts/train.py:128` returns before any backend work.
    assert long_run._needs_the_card(["--task", "x", "--dry-run"]) is False
    assert long_run._needs_the_card(["--task", "x"]) is True


def test_the_query_commands_are_card_free_too():
    """`--list` and `--help` never build an environment either.

    `--json` is not one of them: `scripts/train.py:110-124` reads it only inside
    `if args.list`, and a bare `--json` runs a full training. It must stay behind
    the lock and the budget.
    """
    for flag in ("--list", "--help", "-h"):
        assert long_run._needs_the_card([flag]) is False, flag


def test_a_dry_run_is_not_queued_behind_another_training(monkeypatch):
    """预演不该因为卡被占而 exit 75——它一个字节的 GPU 工作都不做。"""

    def busy(*args, **kwargs):
        raise long_run.gpu_lock.GpuBusy({"pid": 1, "role": "a real training"}, 0.0)

    seen = []

    def fake_supervise(own, passthrough, command, total, **kwargs):
        seen.append(kwargs)
        return 0

    monkeypatch.setattr(long_run.gpu_lock, "hold", busy)
    monkeypatch.setattr(long_run, "_supervise", fake_supervise)
    assert long_run.main(["--task", "jumper.flat", "--dry-run"]) == 0
    assert seen == [{"preview": True}]


def test_a_run_that_fails_is_not_blamed_on_the_lock(monkeypatch):
    """`Popen` can fail with `ENOSPC`; that is not "the lock path is unusable" (71).

    The handler used to wrap the whole run, so a crashed training was reported as a
    lock problem and exited with a code that says "retrying is pointless".
    """
    import contextlib

    def free_lock(*args, **kwargs):
        return contextlib.nullcontext()

    def boom(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(long_run.gpu_lock, "hold", free_lock)
    monkeypatch.setattr(long_run, "_supervise", boom)
    with pytest.raises(OSError):
        long_run.main(["--task", "jumper.flat"])


def test_the_card_is_measured_before_the_backend_is_asked(monkeypatch):
    """`resolve()` may `import torch` and `wp.init()`: measuring after it counts ourselves."""
    order = []

    def card():
        order.append("card")
        return 23000

    def envs(passthrough):
        order.append("envs")
        return 4096

    monkeypatch.setattr(long_run, "free_vram_mib", card)
    monkeypatch.setattr(long_run, "_num_envs", envs)
    monkeypatch.setattr(long_run, "fit_verdict",
                        lambda *a, **k: {"ok": False, "suggested": 4096, "reason": "too big"})
    own = {"reserve_mib": 1024, "allow_over_budget": False, "interval": 5.0}
    long_run._supervise(own, ["--task", "x"], ["true"], None)
    assert order == ["card", "envs"]


def test_a_dry_run_is_not_refused_by_the_vram_budget(monkeypatch):
    """A dry run exists to check a (possibly oversized) config before paying for it."""

    def refuse_to_guess(*args, **kwargs):
        raise AssertionError("a dry run must not be measured against the VRAM budget")

    monkeypatch.setattr(long_run, "fit_verdict", refuse_to_guess)
    monkeypatch.setattr(long_run.subprocess, "call", lambda *a, **k: 0)
    own = {"reserve_mib": 1024, "allow_over_budget": False, "interval": 5.0}
    assert long_run._supervise(own, ["--task", "x", "--dry-run"], ["true"], None,
                               preview=True) == 0


def test_a_dry_run_does_not_overwrite_another_runs_status(monkeypatch):
    """A preview has no run directory of its own, so it must not adopt someone else's."""

    def never(*args, **kwargs):
        raise AssertionError("a preview must not write a status file")

    monkeypatch.setattr(long_run.subprocess, "call", lambda *a, **k: 0)
    monkeypatch.setattr(run_state, "newest_run_dir", never)
    monkeypatch.setattr(long_run, "write_status", never)
    own = {"reserve_mib": 1024, "allow_over_budget": False, "interval": 5.0}
    assert long_run._supervise(own, ["--task", "x", "--dry-run"], ["true"], None,
                               preview=True) == 0


def test_local_flags_are_consumed():
    own, passthrough = _own_args(["--task", "x", "--interval", "5", "--print-command"])
    assert own["interval"] == 5.0
    assert own["print_command"] is True
    assert passthrough == ["--task", "x"]


@pytest.mark.parametrize("bad", [["--interval"], ["--interval", "abc"], ["--interval", "0"], ["--interval", "-3"], ["--interval", "inf"], ["--interval", "nan"]])
def test_bad_interval_is_a_usage_error(bad):
    # inf/nan 曾经漏过 `<= 0` 的检查：`next_sample = now + inf` 之后再也不会触发采样，
    # 状态文件停在第一次写入，看起来像训练卡死。
    with pytest.raises(SystemExit):
        _own_args(bad)


class _FakeProc:
    """够 `_supervise` 用：空 stdout、可控的 `poll()`、固定的 pid。"""

    def __init__(self, alive_reads: int = 0) -> None:
        import io

        self.pid = 4242
        self.stdout = io.StringIO("")
        self._calls = 0
        self._alive = alive_reads

    def poll(self):
        self._calls += 1
        return None if self._calls <= self._alive else 0

    def wait(self, *args, **kwargs):
        return 0

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        pass


def _supervised(monkeypatch, *, dirs_over_time, alive_reads=3, sample=None, statuses=None,
                expect_rc=0, proc=None):
    """跑一次 `_supervise`，返回 `write_status` 收到的运行目录序列。

    `sample` 替换采样函数（用它把循环打断），`statuses` 收到每一份写出的状态副本。
    """
    written = []
    calls = {"n": 0}

    def fake_run_dirs(root=None):
        # 第一次调用是 spawn 前的快照，之后是循环里的解析。
        index = min(calls["n"], len(dirs_over_time) - 1)
        calls["n"] += 1
        return list(dirs_over_time[index])

    monkeypatch.setattr(run_state, "run_dirs", fake_run_dirs, raising=False)
    monkeypatch.setattr(long_run.subprocess, "Popen",
                        lambda *a, **k: proc or _FakeProc(alive_reads), raising=False)
    monkeypatch.setattr(long_run, "_sample", sample or (lambda status: None), raising=False)

    def record(run_dir, status):
        written.append(run_dir)
        if statuses is not None:
            statuses.append(dict(status))

    monkeypatch.setattr(long_run, "write_status", record, raising=False)
    monkeypatch.setattr(long_run, "gpu_identity",
                        lambda: {"name": "fake", "total_mib": 1000,
                                 "limit_w_start": 175.0}, raising=False)
    monkeypatch.setattr(long_run, "free_vram_mib", lambda: 20000, raising=False)
    monkeypatch.setattr(long_run, "_num_envs", lambda passthrough: 16, raising=False)
    monkeypatch.setattr(long_run, "fit_verdict",
                        lambda *a, **k: {"ok": True, "over_budget": False, "reason": "",
                                        "estimate_mib": 100, "budget_mib": 20000},
                        raising=False)
    own = {"reserve_mib": 1024, "allow_over_budget": False, "interval": 0.0}
    rc = long_run._supervise(own, ["--task", "x"], ["true"], 10)
    assert rc == expect_rc
    return written


def test_a_new_run_never_writes_into_the_previous_runs_directory(monkeypatch, tmp_path):
    """`since` 有 2 秒容差：上一条刚写完盘的目录会被认成这一条的证据。

    2026-10-30 的队列实测过——seed1 的 status.json 写进了 seed0 的目录，把两份记录
    都毁了。修复是 spawn 前先取快照，解析时把已有目录排除掉。
    """
    old = tmp_path / "old"
    old.mkdir()
    written = _supervised(monkeypatch, dirs_over_time=[[old], [old]])
    assert old not in written


def test_the_run_directory_is_taken_as_soon_as_it_appears(monkeypatch, tmp_path):
    """子进程要十几秒才建目录；解析必须重试，不能把 `None` 固化下来。"""
    old = tmp_path / "old"
    old.mkdir()
    new = tmp_path / "new"
    new.mkdir()
    written = _supervised(monkeypatch, dirs_over_time=[[old], [old], [old, new]])
    assert written[-1] == new


def test_a_status_off_contract_is_still_written_and_says_so(tmp_path, capsys):
    """缺字段不该杀掉训练，但也不能一声不响——读的人会以为那份记录是完整的。"""
    status = {"status": "running"}

    long_run.write_status(tmp_path, status)

    assert json.loads((tmp_path / "status.json").read_text(encoding="utf-8")) == status
    assert "off contract" in capsys.readouterr().err


def test_a_sigterm_becomes_an_exception_the_cleanup_can_catch():
    """`pkill` 发的是 SIGTERM，默认动作是立刻死——`finally` 不执行，状态永远停在 `running`。

    2026-10-30 22:00 那次启动被 kill 后，目录里留着一份 `status: running / pid 339735`，
    巡检每轮都读到「进程没了但状态还写着 running」，只能报 `investigate`。

    这里直接调已经装上的处理器，不发真信号：要钉住的是「装了什么、它做什么」，
    信号的送达是内核的事。
    """
    restore = long_run._install_termination_handlers()
    try:
        installed = signal.getsignal(signal.SIGTERM)
        with pytest.raises(long_run.Terminated):
            installed(signal.SIGTERM, None)
    finally:
        restore()


def test_the_handlers_are_restored_when_the_run_ends():
    """装过的处理器必须收回去，否则同一进程里的下一次运行继承一个已经退出的世界。"""
    before = signal.getsignal(signal.SIGTERM)

    restore = long_run._install_termination_handlers()
    assert signal.getsignal(signal.SIGTERM) is not before
    restore()

    assert signal.getsignal(signal.SIGTERM) is before


def test_a_run_stopped_by_signal_is_recorded_as_interrupted(monkeypatch, tmp_path):
    """被打断要写 `interrupted` 而不是 `crashed`：前者是「人叫停的」，不该升级成 alarm。"""
    run = tmp_path / "run"
    run.mkdir()
    statuses: list[dict] = []

    def stop(status):
        raise long_run.Terminated(signal.SIGTERM)

    _supervised(monkeypatch, dirs_over_time=[[run]], sample=stop, statuses=statuses,
                expect_rc=128 + signal.SIGTERM)

    assert statuses[-1]["status"] == "interrupted"
    assert statuses[-1]["signal"] == signal.SIGTERM


def test_a_ctrl_c_is_a_stop_not_a_crash(monkeypatch, tmp_path):
    """Ctrl-C 也是人叫停。记成 `crashed` 会让巡检为一次正常收工拉响警报。

    这是唯一走 `KeyboardInterrupt` 而不是 `Terminated` 的停机路径：前台跑训练时
    按的那一下。它过去与「循环体崩了」共用同一个 `else` 分支。
    """
    run = tmp_path / "run"
    run.mkdir()
    statuses: list[dict] = []
    calls = {"n": 0}

    def stop(status):
        # 只在循环里那一次抛；收尾那次采样要正常返回，否则 Ctrl-C 会从
        # `finally` 里逃出去，把 pytest 整场中止。
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyboardInterrupt

    _supervised(monkeypatch, dirs_over_time=[[run]], sample=stop, statuses=statuses,
                expect_rc=128 + signal.SIGINT)

    assert statuses[-1]["status"] == "interrupted"
    assert statuses[-1]["signal"] == signal.SIGINT


def test_the_terminal_state_is_set_before_the_child_is_reaped(monkeypatch, tmp_path):
    """`terminate` + `wait(timeout=15)` 最长十五秒，处理器还装着。

    这期间再来一个信号（`pkill` 连发、`pkill` 之后 Ctrl-C），控制直接落到
    `finally`，写下的是那一刻的 `status`。终态若还排在 `wait()` 之后，写出去的
    就是 `running`——「进程没了、状态写着 running」，正是这次改动要消灭的状态。
    """

    class _DiesWhileWaiting(_FakeProc):
        def wait(self, *args, **kwargs):
            raise long_run.Terminated(signal.SIGTERM)

    run = tmp_path / "run"
    run.mkdir()
    statuses: list[dict] = []

    def stop(status):
        raise long_run.Terminated(signal.SIGTERM)

    _supervised(monkeypatch, dirs_over_time=[[run]], sample=stop, statuses=statuses,
                proc=_DiesWhileWaiting(), expect_rc=128 + signal.SIGTERM)

    assert statuses[-1]["status"] == "interrupted"


def test_the_final_sample_still_runs_with_its_handlers_installed(monkeypatch, tmp_path):
    """收尾那次采样要几秒（`nvidia-smi` 有超时）。

    处理器若在它之前就还原，这个窗口里的 SIGTERM 走默认动作——进程立刻死，
    终态一条都没写。还原必须排在写盘之后。
    """
    run = tmp_path / "run"
    run.mkdir()
    seen: list[object] = []

    def sample(status):
        seen.append(signal.getsignal(signal.SIGTERM))

    _supervised(monkeypatch, dirs_over_time=[[run]], sample=sample)

    assert seen
    assert all(handler is long_run._raise_terminated for handler in seen)


class _Killed(_FakeProc):
    """子进程被外面的信号杀了——本进程一条信号都没收到。"""

    def poll(self):
        return -9

    def wait(self, *args, **kwargs):
        return -9


def test_a_child_killed_from_outside_exits_with_a_shell_code(monkeypatch, tmp_path):
    """OOM 的 SIGKILL 让 `Popen.returncode` 是 `-9`。

    直接把负值返回给 shell，调用方看到的是 247，而 `status.json` 里写的是 `-9`——
    两个都读不出「被谁杀的」。按 shell 惯例换成 128+n，并把信号单独记下来。
    """
    new = tmp_path / "new"
    new.mkdir()
    statuses = []
    written = _supervised(monkeypatch, dirs_over_time=[[], [new]],
                          proc=_Killed(), statuses=statuses, expect_rc=128 + 9)
    assert written[-1] == new
    final = [item for item in statuses if item.get("status") != "running"][-1]
    assert final["status"] == "crashed"
    assert final["child_signal"] == 9
    assert final["rc"] == -9


def test_the_interval_flag_is_read_in_both_spellings():
    """`--interval=30` 与 `--interval 30` 必须是同一个意思。

    只认空格那一版会让 `--interval=30` 原样透传给 `train.py`——它不认识，于是
    训练在 argparse 上退 2，而 `long_run` 记的是「wrapper 参数没问题」。
    """
    for argv in (["--interval", "30"], ["--interval=30"]):
        own, passthrough = long_run._own_args(list(argv))
        assert own["interval"] == 30.0
        assert passthrough == []


def test_the_signal_that_killed_the_child_is_recorded_on_the_crashed_path(tmp_path, monkeypatch):
    """`rc` 为负说明子进程是**被信号杀的**（OOM 的 `SIGKILL` 最常见）。

    `status.json` 只写一个负数，读的人只会看到一个陌生的 rc；记下信号号，
    `health_jumper` 与巡检才能把它和「训练自己崩了」分开。
    """
    statuses = []

    def boom(status):
        raise RuntimeError("child died")

    monkeypatch.setattr(long_run, "_reap", lambda proc: -9, raising=False)
    with pytest.raises(RuntimeError):
        _supervised(monkeypatch, dirs_over_time=[[tmp_path / "new"]],
                    sample=boom, statuses=statuses)

    assert statuses[-1]["status"] == "crashed"
    assert statuses[-1]["rc"] == -9
    assert statuses[-1]["child_signal"] == 9


def test_print_command_still_works_when_mjrl_is_broken(monkeypatch, capsys):
    """`_load_env()` 是 `main()` 的第一句。

    局部导入只是把失败推后：`mjrl` 一坏，`--help`/`--list`/`--print-command` 仍然
    会 `ImportError`。这几条命令的价值恰恰在于「环境坏了的时候还能用」——巡检的
    判决命令也依赖同一个前提。
    """
    import builtins

    real_import = builtins.__import__

    def broken(name, *args, **kwargs):
        if name.startswith("mjrl"):
            raise ImportError("mjrl is broken")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken)
    # 走真的入口：只调 `_load_env()` 的话，未来的改动把 `mjrl` 的 import 挪到
    # `main()` 更早的地方（或 `print_command` 分支之前）仍然全绿。
    assert long_run.main(["--print-command", "--task", "jumper.tripod"]) == 0

    # 但要说一句：`.env` 里的 `CUDA_VISIBLE_DEVICES` 会静默退回默认值，多卡机器上
    # 那就是「跑在另一张卡上」。**stdout 必须是干净的命令行本身**——调用方直接执行
    # 它，多一行说明就不再是可执行的命令。所以两个流一起读，两边都断言。
    out, err = capsys.readouterr()
    assert ".env was not read" in err
    assert ".env was not read" not in out
    assert out.strip().startswith(sys.executable)
