"""一张卡上同时只该有一个训练者在跑。

    python tools/gpu_lock.py status
    python tools/gpu_lock.py run --role "train jumper.tripod" -- \\
        python scripts/train.py --task jumper.tripod --backend warp --num_envs 4096

**它只是约定，不是强制。** `flock` 只约束愿意先来抢锁的进程；Docker 容器里
的另一个服务、别人手敲的 `nvidia-smi`, 都不会看一眼这个文件。它买到的是：
**我们自己的训练、回放、性能测试之间不再互相踩**，以及「这张卡现在归谁」
有一个能查的答案。

两个设计选择值得写下来：

- **锁的存活性由内核保证，文件内容只管描述。** 持锁进程一死，`flock` 立刻
  放开；但文件里那行 JSON 还留着。所以 `status` 会去探那个 pid，把「有人在用」
  和「上次的人没擦干净」分开——照着文件内容判断会把空闲的卡当成占用。
- **释放时把文件清空**，不留上一轮的记录。要历史就去看 `status.json` 和日志，
  锁文件只回答「此刻」。

为什么不是 `nvidia-smi` 的显存判断：显存是**事后**才知道的（分配失败才报错），
而锁是**事前**的。两者互补，先有锁再谈预算。
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

#: 抢不到锁的退出码。`EX_TEMPFAIL`：不是用法错误（2），也不是命令自己失败——
#: 是「现在不行，等下再来」，调用方据此区分「重试」和「别重试」。
BUSY_RC = 75

#: 锁本身用不了（NFS/FUSE 上不支持 `flock`、路径是软链、是目录……）。
#: 和 75 分开：`EX_OSERR` 的意思是「这台机器现在做不了这件事」，重试也没用，
#: 除非换个锁路径——「有人占着」和「这条路根本不通」不该混成一个数。
UNAVAILABLE_RC = 71

#: 默认锁文件。放 `XDG_RUNTIME_DIR`（登录会话私有、重启即清），没有就退到 `/tmp`。
DEFAULT_LOCK = Path(os.environ.get("XDG_RUNTIME_DIR") or "/tmp") / "jumper-gpu.lock"


class GpuBusy(RuntimeError):
    """卡被别的进程占着。`holder` 是对方的记录（可能为 None：文件读不懂）。"""

    def __init__(self, holder: dict | None, waited: float = 0.0) -> None:
        self.holder = holder
        who = _describe(holder) if holder else "another process"
        # 0.09 s 不是「等过」——那是浮点里剩下的零头，写出来只会让人以为发生过排队。
        waited_note = f" after waiting {waited:.1f} s" if waited >= 0.1 else ""
        super().__init__(f"the GPU is in use by {who}{waited_note}")


#: 2100-01-01。一个 epoch 秒值超过它就一定是坏数据（纳秒戳、位数写错）。
_EPOCH_CEILING = 4_102_444_800


def _sane_epoch(value: object) -> bool:
    """能不能安全地交给 `time.localtime`。

    三步都得做：`isfinite(10**400)` 自己抛 `OverflowError`（int 转 float 就溢出），
    所以先 `float()` 并接住；`1e30` 能过 `isfinite` 却让 `localtime` 抛
    `OverflowError`，所以要有上限；负数与 `0` 不是时间戳。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        seconds = float(value)
    except OverflowError:
        return False
    return math.isfinite(seconds) and 0 < seconds <= _EPOCH_CEILING


def _describe(holder: dict) -> str:
    pid = holder.get("pid")
    role = holder.get("role") or "?"
    started = holder.get("started")
    when = ""
    if _sane_epoch(started):
        when = f", since {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(started))}"
    return f"pid {pid} ({role}{when})"


def _pid_alive(pid: object) -> bool:
    """`PermissionError` 表示这个 pid 存在、只是不归我——所以它活着。"""
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, OverflowError):
        # `OverflowError`：文件里写着 `10**30` 这种 pid，`os.kill` 不收。
        return False
    return True


def read_holder(path: Path) -> dict | None:
    """锁文件里写的持有者。空文件、不存在、写坏了都返回 None。"""
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        # 锁文件在共享目录里，别人写进去的字节可能是任意编码。
        return None
    if not raw:
        return None
    try:
        holder = json.loads(raw)
    except ValueError:
        return None
    return holder if isinstance(holder, dict) else None


def status(path: Path = DEFAULT_LOCK) -> dict | None:
    """此刻谁在用。`alive` 说明那个 pid 还在不在——文件可能只是没擦干净。"""
    holder = read_holder(path)
    if holder is None:
        return None
    return {**holder, "alive": _pid_alive(holder.get("pid"))}


def _is_contention(exc: OSError) -> bool:
    """「别人持锁」和「锁坏了」都必须分得清：后者不能被说成前者。"""
    return exc.errno in (errno.EACCES, errno.EAGAIN)


@contextlib.contextmanager
def hold(path: Path = DEFAULT_LOCK, role: str = "", wait_s: float = 0.0) -> Iterator[dict]:
    """拿着锁执行 `with` 里的东西；拿不到抛 `GpuBusy`。"""
    # 值域在这里再验一次：CLI 挡住了 `nan`/`inf`，但库的调用方绕过 CLI 时
    # `nan` 会让 `now >= deadline` 永远为假——无声地空转下去。
    if not math.isfinite(wait_s) or wait_s < 0:
        raise ValueError(f"wait_s must be a finite number of seconds, 0 or more, got {wait_s!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    # `O_NOFOLLOW`：这个路径在共享目录里，预先放一个软链不能让它变成「清空别人文件」。
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o644)
    try:
        deadline = time.monotonic() + wait_s
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if not _is_contention(exc):
                    raise
                now = time.monotonic()
                if now >= deadline:
                    raise GpuBusy(read_holder(path), max(0.0, now - (deadline - wait_s))) from None
                # 用时钟算还剩多少，而不是假设每次 `sleep(0.1)` 正好睡 0.1 s。
                time.sleep(min(0.1, deadline - now))
        # 内容是给人看的：谁、什么时候开始、在跑什么。锁本身由内核持有。
        holder = {
            "pid": os.getpid(),
            "role": role or " ".join(sys.argv),
            "started": time.time(),
            "cmd": sys.argv,
        }
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps(holder, ensure_ascii=False).encode("utf-8"))
        os.fsync(fd)
        try:
            yield holder
        finally:
            # 清空 = 释放之后 `status` 说「空闲」，不留下一条会误导人的记录。
            try:
                os.ftruncate(fd, 0)
            except OSError:
                pass
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _parse(argv: list[str]) -> tuple[dict, list[str]]:
    """`run` 的 `--` 之前归自己，之后原样交给子进程。"""
    own: dict = {"command": "status", "lock": DEFAULT_LOCK, "role": "", "wait": 0.0}
    if argv and argv[0] in ("status", "run"):
        own["command"] = argv[0]
        argv = argv[1:]
    rest: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--":
            rest = argv[i + 1:]
            break
        if a in ("--lock", "--role", "--wait") and i + 1 < len(argv):
            value = argv[i + 1]
            i += 2
        elif a.startswith(("--lock=", "--role=", "--wait=")):
            a, value = a.split("=", 1)
            i += 1
        else:
            raise SystemExit(f"[gpu_lock] unexpected argument {a!r}")
        if a == "--lock":
            own["lock"] = Path(value)
        elif a == "--role":
            own["role"] = value
        else:
            try:
                own["wait"] = float(value)
            except ValueError:
                raise SystemExit(f"[gpu_lock] --wait expects seconds, got {value!r}") from None
            if not math.isfinite(own["wait"]) or own["wait"] < 0:
                # `1e309` 会浮成 `inf`：放过去就是永久等待，且一声不响。
                raise SystemExit("[gpu_lock] --wait must be a number of seconds, 0 or more")
    return own, rest


def main(argv: list[str]) -> int:
    own, rest = _parse(argv)

    if own["command"] == "status":
        state = status(own["lock"])
        print(json.dumps(state, ensure_ascii=False) if state else "the GPU lock is free")
        return 0

    if not rest:
        raise SystemExit("[gpu_lock] run needs a command after --")
    try:
        with hold(own["lock"], own["role"], own["wait"]):
            try:
                rc = subprocess.call(rest)
            except OSError as exc:
                # 不存在、没有执行权限、不是文件——都该是这句话，不是一个 traceback。
                raise SystemExit(
                    f"[gpu_lock] cannot run {rest[0]!r}: {exc.strerror or exc}"
                ) from None
            # 子进程被信号杀死时 `call` 给的是负值，按 shell 惯例报 128+signal。
            return 128 - rc if rc < 0 else rc
    except GpuBusy as busy:
        print(f"[gpu_lock] {busy}", file=sys.stderr)
        return BUSY_RC
    except OSError as exc:
        # 不是争用也不是命令失败：锁文件这条路本身不通（`ENOLCK` 的挂载、
        # 预先摆好的软链、被换成了目录）。说清楚，别甩 traceback。
        print(f"[gpu_lock] cannot use the lock at {own['lock']}: "
              f"{exc.strerror or exc}", file=sys.stderr)
        return UNAVAILABLE_RC


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
