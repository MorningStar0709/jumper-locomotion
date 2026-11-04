"""只读健康检查：一次调用回答「训练现在怎么样」，只输出，不改任何东西。

    python tools/health_jumper.py          # 人读
    python tools/health_jumper.py --json   # 机读，巡检用

五项检查，各自独立失败：
  1. status.json —— 最新一轮 run 的状态与年龄
  2. process    —— 那个 pid 还活着吗（`status: running` 却没人 = 死了）
  3. parse      —— stdout 里还能不能解析出迭代号与 ETA
  4. gpu        —— 当前功耗限额、显存、利用率
  5. disk       —— 训练所在盘还剩多少（知识库「在线健康四件套」之一）

`verdict` 有五个值，和巡检 SOP 的分支一一对应：

    idle        → 没有任何训练在跑，也没什么可看（不是故障）
    wait        → 正常推进，继续等
    investigate → 状态缺失或读不懂，去查证据
    done        → 这一轮跑完了
    alarm       → 进程没了但状态还写着 running，或者非零退出却不是人为叫停，
                  或者盘要满了

`idle` 是后加的：以前「没有 status.json」一律报 `investigate`，那是把「还没开始」
和「包错了 wrapper」混成一种。巡检每 30 分钟看一次，空转的机器会连报三轮，把告警
本身废掉——这条线之前就被停用过一次。

**没有写操作。** kill / 重启 / 删除 / 改配置都不在这里——那是主会话的事。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import run_state

REPO = Path(__file__).resolve().parent.parent


def _num(text: str) -> float | None:
    try:
        return float(text.strip().split()[0])
    except (ValueError, IndexError):
        return None


def latest_status(logs_root: Path) -> tuple[dict | None, float | None, str]:
    """最新的 status.json、它的年龄（秒）、以及读不到的原因。

    三种「读不到」必须分开，否则一个被写坏的文件会被诊断成「wrapper 没启用」，
    把排查引向完全错误的方向：`missing`（目录里没有）与 `corrupt`（有但读不懂）。
    """
    def _mtime(path: Path) -> float:
        # 目录可能在 `glob` 与 `stat` 之间被删（巡检快跑完时正好有人清理 logs）。
        # 这个工具是只读诊断，缺一个文件不是故障——按「没有」处理，别 traceback。
        try:
            return path.stat().st_mtime
        except OSError:
            return float("-inf")

    paths = sorted(logs_root.glob("*/*/*/status.json"), key=_mtime)
    if not paths:
        return None, None, "missing"
    newest = paths[-1]
    # 读法由 `run_state` 定义：半写、非法 UTF-8、非对象，一律是「写坏了」。
    data = run_state.read(newest)
    if data is None:
        return None, None, "corrupt"
    try:
        age = time.time() - newest.stat().st_mtime
    except OSError:
        return data, None, "ok"
    return data, age, "ok"


#: `running` 的状态文件超过「采样周期的这个倍数」就认为写它的进程已经不在了。
#: 周期由 wrapper 自己写进 `status.json` 的 `interval_s`；缺省按 30 s 算。
STALE_FACTOR = 4.0
DEFAULT_INTERVAL_S = 30.0


def pid_alive(pid: object) -> bool:
    # `status.json` 是磁盘上的文件，可能被改坏：字符串 pid 会让 `os.kill` 抛
    # TypeError，负数 pid 会探测整个进程组（几乎总是「活着」），而超出 pid 范围的
    # 整数（`5000000000`）会让它抛 `OverflowError`——那不是 `OSError`，漏出去这个只读
    # 检查就在「状态文件被写坏」这个**它专门用来报告的场景**里 traceback 了。
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # 进程存在，只是不归我们管。这是「活着」的证据，判成死会产生假 alarm。
        return True
    except (OSError, OverflowError):
        return False


def gpu_readings() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={_gpu_index()}",
             "--query-gpu=enforced.power.limit,memory.used,utilization.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()[0]
        limit, mem, util = (p.strip() for p in out.split(","))
        return {"limit_w": _num(limit), "mem_mib": _num(mem), "util": _num(util)}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {"limit_w": None, "mem_mib": None, "util": None}


def _gpu_index() -> str:
    """与 `long_run.py` 同一套：先看 `CUDA_VISIBLE_DEVICES`，否则物理 0 号。"""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    return visible if visible.isdigit() else "0"


#: 盘用到这个比例就自己查（清理旧 run、挪 checkpoint），但不惊动 alarm。
DISK_WARN_FRACTION = 0.85
#: 用到这里训练随时会因为写不出 checkpoint 而崩，属于 alarm。
DISK_ALARM_FRACTION = 0.95


def disk_readings(root: Path) -> dict:
    """训练写在哪个盘上还剩多少。读不到就报 null，不猜。"""
    try:
        usage = shutil.disk_usage(root)
    except OSError:
        return {"free_gib": None, "total_gib": None, "used_fraction": None}
    total = float(usage.total)
    return {
        "free_gib": round(usage.free / 1024**3, 1),
        "total_gib": round(total / 1024**3, 1),
        "used_fraction": round(1.0 - usage.free / total, 3) if total else None,
    }


#: 认得出的训练入口。`run_queue` 自己不开训练，但它活着意味着队列还会再起一条。
WRAPPER_PATTERNS = ("tools/long_run.py", "tools/run_queue.py")


def wrapper_running() -> bool:
    """有没有训练入口在跑——用来把「还没开始」和「包错了 wrapper」分开。

    `pgrep` 失败（没装、被禁）时返回 False：宁可把空转报成 idle，也不要在没有
    证据的情况下把巡检引向排查。真正的证据仍然只有 `status.json`。
    """
    for pattern in WRAPPER_PATTERNS:
        try:
            done = subprocess.run(
                ["pgrep", "-f", pattern], capture_output=True, text=True, timeout=10, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return False
        if done.returncode == 0 and done.stdout.strip():
            return True
    return False


def evaluate(logs_root: Path) -> dict:
    status, age, why = latest_status(logs_root)
    gpu = gpu_readings()
    checks = []

    if status is None:
        note = {
            "missing": "没有 status.json；训练没用 long_run.py 包，或还没开始",
            "corrupt": "status.json 存在但读不出来（写了一半 / 被改坏 / 版本不符）",
        }[why]
        checks.append({"name": "status_file", "ok": False, "value": None, "note": note})
        if why == "missing" and not wrapper_running():
            # 没有任何训练入口在跑，也没有状态文件：这是空转，不是故障。
            verdict, ok = "idle", True
        else:
            verdict, ok = "investigate", False
    else:
        state = status.get("status")
        # `age is None` 是「读到了内容，但文件的 mtime 在两次读之间没了」（目录被
        # 清理）。这个工具是只读诊断：它必须把这个状态报出来，而不是在 `round()` 上
        # 抛 `TypeError`——那正是它存在的理由被绕过的方式。
        age_s = None if age is None else round(age, 1)
        checks.append({"name": "status_file", "ok": True,
                       "value": {"status": state, "age_s": age_s},
                       "note": status.get("run_dir")})

        running = state == "running"
        alive = pid_alive(status.get("pid"))
        interval = status.get("interval_s")
        stale_after = STALE_FACTOR * (interval if isinstance(interval, (int, float)) and interval > 0
                                      else DEFAULT_INTERVAL_S)
        # `running` 且进程在，但状态文件已经很久没刷新：写它的人卡住了（或者进程表
        # 撒谎）。只看 pid 会让这种情况永远报 wait。
        # 拿不到年龄就等于「无法证明它是新的」：`running` 的 run 在这种状态下应当
        # 被当成过旧（turn 到 investigate），而不是被当成刚写过。
        stale = running and (age is None or age > stale_after)
        checks.append({"name": "process", "ok": (not running) or (alive and not stale),
                       "value": {"alive": alive, "age_s": age_s, "stale_after_s": stale_after},
                       "note": f"pid={status.get('pid')}" + (" 状态文件过旧" if stale else "")})

        # 解析失败时 iter/eta 会是 null，但进程活得好好的——不把这条单独拎出来，
        # 巡检会一直 wait，而日志格式可能早就变了。只在 running 时升级：
        # 已经 completed 的 run 里残留的 parse_error 不该把结论改掉，更不该把
        # crashed 降级成 investigate（那会把真正的 alarm 吃掉）。
        parse_error = bool(status.get("parse_error"))
        checks.append({"name": "parse", "ok": not (parse_error and running), "value": parse_error,
                       "note": "stdout 里认得出 Iteration time/ETA，但解析不出数字" if parse_error else ""})

        if running and parse_error:
            verdict = "investigate"
        elif running and alive and not stale:
            verdict = "wait"
        elif running and (not alive or stale):
            verdict = "alarm"
        elif state == "crashed":
            verdict = "alarm"
        elif state in ("completed", "interrupted"):
            # `interrupted` 是人叫停的（`pkill`、Ctrl-C），不是故障：把它报成 alarm
            # 会让每次主动收工都拉一次警报，最后没人看警报。阶段推进由 `run_queue`
            # 的核验管（它要求 `completed`），不归这里判。
            verdict = "done"
        else:
            # 未知状态（未来的 v:2、被手工改坏的文件）不能当成「跑完了」。
            verdict = "investigate"
        ok = verdict in ("wait", "done")

    checks.append({"name": "gpu", "ok": gpu["limit_w"] is not None, "value": gpu,
                   "note": "175 W 为满血；135 W 约七成"})

    disk = disk_readings(logs_root)
    disk_ok = disk["used_fraction"] is not None and disk["used_fraction"] < DISK_WARN_FRACTION
    checks.append({"name": "disk", "ok": disk_ok, "value": disk,
                   "note": f"用到 {DISK_WARN_FRACTION:.0%} 要自己清，{DISK_ALARM_FRACTION:.0%} 算 alarm"})
    # 盘满会先让「跑完了」变成「写不出下一个 checkpoint」，所以它能把 wait/done/idle
    # 降级；但不能把已有的 alarm 降回 investigate——那会把真正的故障藏起来。
    used = disk["used_fraction"]
    if used is not None and used >= DISK_ALARM_FRACTION:
        verdict, ok = "alarm", False
    elif used is not None and used >= DISK_WARN_FRACTION and verdict in ("wait", "done", "idle"):
        verdict, ok = "investigate", False

    return {
        "ok": ok,
        "verdict": verdict,
        "checks": checks,
        "ts": time.time(),
    }


def main(argv: list[str]) -> int:
    logs_root = REPO / "logs"
    as_json = "--json" in argv
    for i, a in enumerate(argv):
        if a == "--logs-root" and i + 1 < len(argv):
            logs_root = Path(argv[i + 1])

    result = evaluate(logs_root)

    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"verdict: {result['verdict']}  (ok={result['ok']})")
        for c in result["checks"]:
            mark = "OK  " if c["ok"] else "FAIL"
            print(f"  [{mark}] {c['name']:12} {c['value']}  {c['note'] or ''}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
