"""`status.json` 的契约，与运行目录的发现。

同一个文件有三个读者：`long_run` 写它，`health_jumper` 判断它，`run_queue` 核验它。
在这之前，字段名是三方各自记着的——2026-10-30 的事故里，一条 run 把状态写进了
上一条的目录，而两个读者都不会因此报错。字段定义收在这里，改一处就改三方。

一条运行的状态只有四种值：`running` 写于启动之后，`completed`、`crashed` 与
`interrupted` 是终态。`crashed` 是非零退出或自己崩了；`interrupted` 是**被信号叫停**
（`pkill`、`Ctrl-C`），运维上不是故障，不该升级成告警。
"""

from __future__ import annotations

import json
import os
from collections.abc import Collection
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent

#: 契约版本。字段含义变化时加一，读者据此拒绝看不懂的文件，而不是猜。
SCHEMA_VERSION = 1

#: `long_run` 写进 `status.json` 的全部字段。读者可以用 `.get()` 取自己关心的那些，
#: 但写入方不许悄悄少写——`validate` 会点名。
FIELDS: tuple[str, ...] = (
    "v",
    "pid",
    "status",
    "run_dir",
    "iter",
    "percent",
    "eta_s",
    "step_time",
    "elapsed_s",
    "limit_w",
    "mem_mib",
    "util",
    "power_w",
    "ts",
    "parse_error",
    "rc",
    "interval_s",
    "gpu",
    # 只有被信号叫停的那条才有；`interrupted` 没有它就说不清是被什么叫停的。
    "signal",
    # 收尾采样失败时写在这里。它自己出现，恰恰说明前面那次采样没写出字段。
    "sample_error",
    # 崩溃或被叫停时的异常文本（`long_run.py` 每条非正常结束都写）。读契约的人
    # 只靠 `status` 知道「坏了」，靠它才知道「坏在哪」。
    "error",
    # 子进程自己被信号杀掉时的信号号（OOM 的 `SIGKILL` 是 9）。与 `signal` 分开：
    # 后者是**本进程**收到的信号（人叫停），前者是本进程一条都没收到的那一种。
    "child_signal",
)

#: 一份 `status.json` 必须带的字段。缺了它就没法判断这条 run 是死是活。
REQUIRED: tuple[str, ...] = ("v", "status", "pid", "ts")

#: 一条运行可能处于的状态。`running` 之外三个都是终态。
STATUS_VALUES: tuple[str, ...] = ("running", "completed", "crashed", "interrupted")


def validate(status: Any) -> list[str]:
    """返回字段层面的问题，空列表表示这份状态是可读的。

    只查契约，不查语义：`iter=0` 与 `iter=9999` 一样合法，读的人自己判断够不够。
    """
    if not isinstance(status, dict):
        return [f"status is a {type(status).__name__}, not an object"]
    missing = [f"missing {field}" for field in REQUIRED if field not in status]
    if missing:
        return missing

    problems: list[str] = []
    if status["v"] != SCHEMA_VERSION:
        problems.append(f"schema version {status['v']}, expected {SCHEMA_VERSION}")
    if status["status"] not in STATUS_VALUES:
        problems.append(
            f"status {status['status']!r} is not one of {', '.join(STATUS_VALUES)}"
        )
    if not isinstance(status["ts"], (int, float)):
        problems.append("ts is not a number")
    return problems


def read(path: Path) -> dict | None:
    """读一份 `status.json`；读不出来就返回 `None`，不抛。

    半个文件、非 UTF-8、JSON 不是对象，都归为「读不出来」——调用方要判断的是
    「有没有一条可信的状态」，不是「JSON 错在哪」。
    """
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def write(path: Path, status: dict) -> None:
    """原子写：先写 `.json.tmp` 再 `os.replace`，读的人永远看不到半个文件。"""
    target = Path(path)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, target)


def run_dirs(root: Path | None = None) -> list[Path]:
    """`logs/*/*/<date-time>/` 里的所有运行目录。"""
    base = REPO / "logs" if root is None else Path(root)
    return [p for p in base.glob("*/*/*") if p.is_dir()]


def newest_run_dir(
    since: float, *, exclude: Collection[Path] = (), root: Path | None = None
) -> Path | None:
    """`logs/*/*/<date-time>/` 里 mtime 不早于 `since` 的最新一个。

    `exclude` 排除**调用之前就存在**的目录：`since` 有 2 秒容差，一条刚写完盘的旧
    run 会落在容差里，被认成这一条留下的证据。
    """
    skipped = set(exclude)
    candidates = [p for p in run_dirs(root) if p not in skipped]
    fresh = [p for p in candidates if p.stat().st_mtime >= since - 2]
    return max(fresh, key=lambda p: p.stat().st_mtime) if fresh else None
