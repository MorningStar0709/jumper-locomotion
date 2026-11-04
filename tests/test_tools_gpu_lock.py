"""`tools/gpu_lock.py` 的契约测试。

这个模块存在只为一件事：**一张卡上同时只该有一个训练者在跑**。它不是锁的
实现秀，所以测的不是 `flock` 本身，而是四件别人会依赖的行为：

1. 拿着锁的时候，锁文件里写得清是谁、从什么时候开始、在跑什么；
2. 拿不到锁时**拒绝并说出持有者**，而不是安静地挤上去；
3. 释放之后文件里不再有持有者；
4. 持锁的进程已经死了的话，状态里要标出来——内核早就把 `flock` 收了，
   但文件内容会留着，照着内容判断「有人在用」会把卡误判成被占。

还有一条不属于锁本身但同样重要：`run` 子命令在拿不到锁时**一个字节都不执行**
传给它的命令。

放在 `tests/` 而不是 `tools/` 的原因：`pyproject.toml` 的 `testpaths = ["tests"]`
与 CI 都只跑这个目录，放别处等于这份契约无人看守。
"""

import errno
from pathlib import Path
import json
import os
import subprocess
import sys
import time

import pytest

import gpu_lock

#: 被 `run` 子命令执行的那个脚本——用模块自己的路径，测的是真实入口。
SCRIPT = Path(gpu_lock.__file__).resolve()


def test_a_lock_that_was_never_taken_has_no_holder(tmp_path):
    assert gpu_lock.read_holder(tmp_path / "gpu.lock") is None


def test_holding_records_who_holds_it(tmp_path):
    lock = tmp_path / "gpu.lock"
    with gpu_lock.hold(lock, "train jumper.tripod"):
        holder = gpu_lock.read_holder(lock)
    assert holder["role"] == "train jumper.tripod"
    assert holder["pid"] == os.getpid()
    assert holder["started"] > 0


def test_releasing_leaves_it_free(tmp_path):
    lock = tmp_path / "gpu.lock"
    with gpu_lock.hold(lock, "train jumper.tripod"):
        pass
    assert gpu_lock.read_holder(lock) is None


def test_a_second_holder_is_refused_and_told_who_has_it(tmp_path):
    lock = tmp_path / "gpu.lock"
    with gpu_lock.hold(lock, "the training run"):
        with pytest.raises(gpu_lock.GpuBusy) as refused:
            with gpu_lock.hold(lock, "a replay"):
                pass
    assert "the training run" in str(refused.value)
    assert refused.value.holder["role"] == "the training run"


def test_a_holder_that_died_is_not_reported_as_alive(tmp_path):
    lock = tmp_path / "gpu.lock"
    lock.write_text(json.dumps({"pid": 2**22 + 12345, "role": "ghost", "started": 1.0, "cmd": []}))

    state = gpu_lock.status(lock)
    assert state["role"] == "ghost"
    assert state["alive"] is False


def test_a_holder_that_is_running_is_reported_as_alive(tmp_path):
    lock = tmp_path / "gpu.lock"
    with gpu_lock.hold(lock, "me"):
        assert gpu_lock.status(lock)["alive"] is True


def test_status_of_a_free_lock_is_none(tmp_path):
    assert gpu_lock.status(tmp_path / "gpu.lock") is None


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=60, check=False
    )


def test_run_refuses_and_executes_nothing_while_the_card_is_taken(tmp_path):
    lock = tmp_path / "gpu.lock"
    marker = tmp_path / "the-command-ran"
    payload = f"open({str(marker)!r}, 'w').close()"

    with gpu_lock.hold(lock, "someone else's training"):
        done = _run("run", "--lock", str(lock), "--role", "mine", "--",
                    sys.executable, "-c", payload)

    assert done.returncode == 75, done.stdout + done.stderr
    assert not marker.exists()
    assert "someone else's training" in done.stdout + done.stderr


def test_run_hands_the_child_status_back(tmp_path):
    lock = tmp_path / "gpu.lock"
    done = _run("run", "--lock", str(lock), "--role", "mine", "--",
                sys.executable, "-c", "raise SystemExit(7)")
    assert done.returncode == 7


def test_run_releases_the_lock_when_the_child_is_done(tmp_path):
    lock = tmp_path / "gpu.lock"
    _run("run", "--lock", str(lock), "--role", "mine", "--", sys.executable, "-c", "pass")
    assert gpu_lock.read_holder(lock) is None


def test_a_broken_flock_is_not_reported_as_a_busy_gpu(tmp_path, monkeypatch):
    """只有「别人持锁」才算忙。`ENOLCK`（文件系统不支持 flock）是锁坏了。

    把「锁坏了」说成「GPU 有人在用」，会让这台机器永远开不了任何训练——
    正好是这个模块要避免的那件事。
    """

    def cannot_lock(fd, operation):
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(gpu_lock.fcntl, "flock", cannot_lock)
    with pytest.raises(OSError):
        with gpu_lock.hold(tmp_path / "gpu.lock", "mine"):
            pass


def test_a_symlink_at_the_lock_path_is_not_written_through(tmp_path):
    """The lock lives in a shared directory: a planted symlink must not become "truncate"."""
    victim = tmp_path / "victim"
    victim.write_text("precious")
    lock = tmp_path / "gpu.lock"
    lock.symlink_to(victim)

    with pytest.raises(OSError):
        with gpu_lock.hold(lock, "mine"):
            pass
    assert victim.read_text() == "precious"


def test_an_unusable_lock_path_is_a_refusal_not_a_traceback(tmp_path):
    """A mount that cannot `flock`, or a planted symlink, is not "someone else is training"."""
    victim = tmp_path / "victim"
    victim.write_text("precious")
    lock = tmp_path / "gpu.lock"
    lock.symlink_to(victim)
    done = _run("run", "--lock", str(lock), "--role", "mine", "--", sys.executable, "-c", "pass")
    assert done.returncode == gpu_lock.UNAVAILABLE_RC
    assert "cannot use the lock" in done.stdout + done.stderr
    assert "Traceback" not in done.stdout + done.stderr


def test_a_nonsense_wait_is_refused_by_the_library_too(tmp_path):
    """`nan` makes `now >= deadline` never true: the loop would spin with no message."""
    for bad in (float("nan"), float("inf"), -1.0):
        with pytest.raises(ValueError):
            with gpu_lock.hold(tmp_path / "gpu.lock", "mine", wait_s=bad):
                pass


def test_a_wait_of_infinity_is_rejected(tmp_path):
    """`1e309` 解析成 `inf`，放过去就是永久等待且一声不响。"""
    done = _run("run", "--wait", "1e309", "--", sys.executable, "-c", "pass")
    assert done.returncode != 0
    assert "--wait must be" in done.stdout + done.stderr


def test_the_requested_sleeps_stay_inside_the_wait_budget(tmp_path, monkeypatch):
    """The deadline is read from the clock; what we ask to sleep must fit inside it."""
    calls: list[float] = []
    real_sleep = time.sleep  # `gpu_lock.time` is the real module: do not call it inside the stub

    def slow_sleep(seconds):
        calls.append(seconds)
        real_sleep(0.3)

    monkeypatch.setattr(gpu_lock.time, "sleep", slow_sleep)
    lock = tmp_path / "gpu.lock"
    with gpu_lock.hold(lock, "someone else's training"):
        with pytest.raises(gpu_lock.GpuBusy):
            with gpu_lock.hold(lock, "mine", wait_s=0.5):
                pass
    # Two retries at 0.1 s each fit a 0.5 s budget; asserting on the *requested*
    # sleeps keeps this independent of how long the machine actually took.
    assert calls, "no retry was attempted"
    # 每次 stub sleep 真的睡 0.3 s，所以 0.5 s 的预算里最多容得下两次请求；
    # 第三次就说明循环没在看钟，而是按固定次数重试（那种实现的总和也恰好是 0.5）。
    assert len(calls) <= 2, f"retried {len(calls)} times on a 0.5s budget"
    assert sum(calls) <= 0.5, f"asked to sleep {sum(calls):.2f}s on a 0.5s budget"


def test_a_status_with_an_absurd_pid_does_not_crash(tmp_path):
    """写坏的记录要被当成「没活着」，不是让 `status` 甩一个 `OverflowError`。"""
    lock = tmp_path / "gpu.lock"
    lock.write_text(json.dumps({"pid": 10**30, "role": "corrupt", "started": 1.0, "cmd": []}))
    state = gpu_lock.status(lock)
    assert state["alive"] is False


def test_a_lock_file_that_is_not_utf8_reads_as_free(tmp_path):
    lock = tmp_path / "gpu.lock"
    lock.write_bytes(b"\xff\xfe not utf-8")
    assert gpu_lock.read_holder(lock) is None
    assert gpu_lock.status(lock) is None


def test_run_of_a_file_without_execute_permission_fails_cleanly(tmp_path):
    """这个工具的价值之一是在干活之前给出人能读的拒绝，而不是 traceback。"""
    plain = tmp_path / "not-executable"
    plain.write_text("#!/bin/sh\ntrue\n")
    done = _run("run", "--lock", str(tmp_path / "gpu.lock"), "--role", "mine", "--", str(plain))
    assert done.returncode != 0
    assert "cannot run" in done.stdout + done.stderr
    assert "Traceback" not in done.stdout + done.stderr


def test_a_child_killed_by_a_signal_reports_the_shell_convention(tmp_path):
    """`subprocess.call` 给负值，退出码说 241 就会让调用方读错死因。"""
    done = _run("run", "--lock", str(tmp_path / "gpu.lock"), "--role", "mine", "--",
                "sh", "-c", "kill -TERM $$")
    assert done.returncode == 143


def test_run_gives_up_after_its_wait_when_the_card_stays_taken(tmp_path):
    lock = tmp_path / "gpu.lock"
    with gpu_lock.hold(lock, "someone else's training"):
        done = _run("run", "--lock", str(lock), "--role", "mine", "--wait", "0.3", "--",
                    sys.executable, "-c", "pass")
    assert done.returncode == 75


def test_a_corrupt_start_time_does_not_break_the_report():
    """`started` 是磁盘上的数字，被写坏成 `inf` 时 `time.localtime` 抛 `OverflowError`，
    而 `_describe` 专门用来报告坏掉的锁文件。"""
    assert "pid 42" in gpu_lock._describe({"pid": 42, "role": "train", "started": float("inf")})
    assert "since" not in gpu_lock._describe({"pid": 42, "role": "train", "started": float("nan")})


def test_an_epoch_that_cannot_even_be_converted_is_described_without_raising():
    """`math.isfinite(10**400)` 自己就抛 `OverflowError`。

    这段代码专门用来**报告**一个写坏的锁文件：它在坏输入上崩掉，等于把唯一的
    诊断通道也一起关掉了。`1e30` 能过 `isfinite`，却让 `localtime` 抛。
    """
    for bad in (10**400, 1e30, float("inf"), float("nan"), -1, 0, "1793400000", True):
        text = gpu_lock._describe({"pid": 7, "role": "train", "started": bad})
        assert text.startswith("pid 7 (train")
        assert "since" not in text

    # 正向也要钉住：把 `_sane_epoch` 改成恒假，上面这一整圈仍然全绿，而每一句
    # 占用说明都会悄悄少掉「从什么时候开始」。
    assert "since 2026-" in gpu_lock._describe(
        {"pid": 7, "role": "train", "started": 1793400000})
