"""包住 `scripts/train.py`，把「它现在怎么样」写成一个 JSON 文件。

    python tools/long_run.py --task jumper.tripod --num_envs 1024 --max-iterations 2000

训练本身一行不改：这个脚本只读它的 stdout、只采 `nvidia-smi`、只写
`<run_dir>/status.json`。想停就 Ctrl-C——**那一刻它会先写一份终态再退出**，
不会留下一个永远 `running` 的状态文件。

**为什么不写进 train.py**：训练入口是科学计算语义，看门狗是运维语义。混在
一起之后每次冒烟、每次 play 都要背这个成本，而真正需要监控的只有长跑。

`status.json` 的契约（字段只增不改语义，`v` 变了就是另一套）：

    v / pid / status / run_dir / iter / percent / eta_s / step_time / elapsed_s
    limit_w / mem_mib / util / power_w / ts / parse_error / rc / gpu

`gpu` 是启动时那一份硬件账：卡名、总显存、功耗限额起点，加上本次申请的
显存估算与预算（`estimate_mib` / `budget_mib`），以及限额在整轮里的摆动区间
（`limit_min_w` / `limit_max_w`）。限额不是常数——这台机器上实测在
135/150/175 W 之间跳，只看某一刻的数会把「这一轮为什么慢」归错因。

`status` 取 `running` | `completed` | `crashed` | `interrupted`。`crashed` 是非零退出
或自己崩了，`interrupted` 是**被信号叫停**（`pkill`、Ctrl-C）——两种都带 `rc`，
后者还带 `signal`。分开记是因为运维含义不同：一个要查，一个不用。

**采样不依赖 stdout**：早先的实现把采样挂在读到的每一行上，于是模型编译那几十秒
（一个字都不输出）里状态文件根本不刷新，看上去像死了。现在 stdout 由一条读线程
推进队列，主循环按 `time.monotonic()` 的节拍采样，与训练说不说话无关。

**子进程强制不缓冲**：Python 的 stdout 接管道时是 8 KiB 块缓冲，训练侧没有
`flush=True`，所以逐行读到的其实是每 8 KiB 一批。子进程环境里放
`PYTHONUNBUFFERED=1` 才对得上「逐行」这个假设。
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import gpu_lock
import run_state

REPO = Path(__file__).resolve().parent.parent
TRAIN = REPO / "scripts" / "train.py"

#: 认得的 stdout 行 -> 字段名。认不出返回 (None, None)，调用方只记一次 parse_error。
_ITER = re.compile(r"Iteration time:\s*([\d.]+)s")
#: `datetime.timedelta` 的两种渲染：`H:MM:SS`，以及满 24 小时后多出的 `N day, H:MM:SS`。
#: 只写 `H:MM:SS` 会在长跑越过一天时静默丢字段——那正是最需要 ETA 的时候。
_HMS = re.compile(r"(?:(\d+)\s+days?,\s*)?(\d+):(\d+):(\d+)")
_ELAPSED = re.compile(r"Time elapsed:\s*" + _HMS.pattern)
_ETA = re.compile(r"ETA:\s*" + _HMS.pattern)
#: 只问第一张卡：多卡机器上 `nvidia-smi` 每卡一行，整段 split 会得到 3N 个字段，
#: 解包失败后被 except 吞掉，于是「硬件正常」和「没有显卡」看起来一模一样。
_SAMPLE_FIELDS = "enforced.power.limit,memory.used,utilization.gpu,power.draw"

#: 显存标定（2026-10-29 实测，`jumper.tripod` + warp/cuda，20 iterations 的峰值）：
#: 512 → 1765 MiB、2048 → 5195 MiB、4096 → 9437 MiB。
#: 拟合取保守的一侧，在 2048/4096 上高估 18/44 MiB、在 512 上高估 260 MiB。
#: 宁愿早一点喊停，也不要跑到第 2000 iter 才 OOM——那时 checkpoint 已经写过几十个。
VRAM_OVERHEAD_MIB = 960
VRAM_PER_ENV_MIB = 2.08

#: 每任务一份实测标定。上面两个常量是 `jumper.tripod` + warp 的量法，也是表坏掉时
#: 的兜底——它们只有在这一个任务上才是「已知」，在别的任务上只是「猜得比较像」。
CALIBRATION_PATH = Path(__file__).resolve().parent / "calibration.json"
#: 预算总留一段：显存碎片、别的进程的波动、驱动自己的缓存，都不是线性模型里的项。
DEFAULT_RESERVE_MIB = 1024
#: 没写 `--num_envs`、环境里也没有 `MJRL_NUM_ENVS` 时，各后端自己用的默认值
#: （`scripts/_cli.py:132` 的顺序是 flag → `MJRL_NUM_ENVS` → 后端默认）。
#: 预算按后端真正会用的那个数算，否则 warp 和 native 会差出 64 倍。
BACKEND_DEFAULT_NUM_ENVS = {"warp": 4096, "native": 64}


def _seconds(match: re.Match[str]) -> int:
    days, h, mi, se = (int(g) if g is not None else 0 for g in match.groups())
    return ((days * 24) + h) * 3600 + mi * 60 + se


def parse_line(line: str) -> tuple[str | None, float | int | None]:
    """一行 stdout -> (字段, 值)。永不抛异常。"""
    try:
        s = line.strip()
        if m := _ITER.search(s):
            return ("iter_time", float(m.group(1)))
        if m := _ELAPSED.search(s):
            return ("elapsed", _seconds(m))
        if m := _ETA.search(s):
            return ("eta", _seconds(m))
    except (ValueError, AttributeError):
        pass
    return (None, None)


def _num(text: str) -> float | None:
    """`135.00 W` / `13774 MiB` / `[N/A]` -> float 或 None。"""
    try:
        return float(text.strip().split()[0])
    except (ValueError, IndexError):
        return None


def _gpu_index() -> str:
    """要采样哪张卡：优先看 `CUDA_VISIBLE_DEVICES`，否则物理 0 号。

    包装器把 `--device cuda:1` 之类的参数原样转发给训练，而 `nvidia-smi --id=0`
    永远指物理 0 号。只看第一张卡会在多卡机器上报错卡——于是「限额 175」说的
    根本不是正在跑训练的那张。
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    return visible if visible.isdigit() else "0"


def sample_gpu() -> dict:
    """一次 `nvidia-smi`，读不到就全是 None——不许因为采样失败就中断训练。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={_gpu_index()}",
             f"--query-gpu={_SAMPLE_FIELDS}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip().splitlines()[0]
        limit, mem, util, power = (p.strip() for p in out.split(","))
        return {
            "limit_w": _num(limit),
            "mem_mib": _num(mem),
            "util": _num(util),
            "power_w": _num(power),
        }
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        # 注意：多卡机器上 `nvidia-smi` 每卡一行，整段 split 会得到 3N 个字段并
        # 在解包里抛 ValueError。取 `splitlines()[0]` 就是对这种情况的处理。
        return {"limit_w": None, "mem_mib": None, "util": None, "power_w": None}


def _a_measurement(value: object) -> bool:
    """标定表里的一个数能不能用：有限、为正、不是布尔。

    `per_env_mib=0` 会让 `fit_verdict` 的 `// per_env` 抛 `ZeroDivisionError`，
    `NaN` 会让 `math.ceil` 抛 `ValueError`，负数和 0 则会静默低估显存。
    这三种都不该出现在「一个坏文件不挡启动」的路径上。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        # `math.isfinite` 会把整数转成 double：一个 309 位的整数会在这里抛
        # `OverflowError`，而这条路径的契约是「坏文件不挡启动」。
        return math.isfinite(value) and value > 0
    except OverflowError:
        return False


def load_calibrations(path: Path | None = None) -> dict[str, dict]:
    """读标定表；读不出来或条目不成形就当没有，绝不因为一个坏文件拒绝启动。"""
    target = CALIBRATION_PATH if path is None else Path(path)
    try:
        table = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(table, dict):
        return {}
    good = {}
    for task, entry in table.items():
        if not isinstance(entry, dict):
            continue
        overhead, per_env = entry.get("overhead_mib"), entry.get("per_env_mib")
        if _a_measurement(overhead) and _a_measurement(per_env):
            good[task] = {"overhead_mib": float(overhead), "per_env_mib": float(per_env)}
    return good


def calibration_for(task: str | None) -> tuple[float, float, bool]:
    """这个任务的 (overhead, per_env, 是否真的量过)。

    没量过的任务取表里**最大**的那一组：宁可拒绝一条本来跑得下的 run，也不要
    OOM 一条正在跑的。表为空（坏文件）时退回模块常量，并同样报「没量过」。
    """
    table = load_calibrations()
    if task is not None and task in table:
        entry = table[task]
        return entry["overhead_mib"], entry["per_env_mib"], True
    if table:
        return (max(e["overhead_mib"] for e in table.values()),
                max(e["per_env_mib"] for e in table.values()),
                False)
    return float(VRAM_OVERHEAD_MIB), float(VRAM_PER_ENV_MIB), False


def estimate_vram_mib(
    num_envs: int,
    *,
    overhead: float = VRAM_OVERHEAD_MIB,
    per_env: float = VRAM_PER_ENV_MIB,
) -> int:
    """这一轮大致要吃多少显存。线性外推，不是物理模型。"""
    if num_envs < 0:
        raise ValueError(f"num_envs must be 0 or more, got {num_envs}")
    return math.ceil(overhead + per_env * num_envs)


def free_vram_mib() -> int | None:
    """这张卡现在还剩多少显存。读不到返回 None——不猜。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={_gpu_index()}", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip().splitlines()[0]
        return int(float(out.strip()))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def gpu_identity() -> dict:
    """卡名、总显存、此刻的功耗限额。跟 `sample_gpu` 分开：这个是启动时一次。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={_gpu_index()}",
             "--query-gpu=name,memory.total,enforced.power.limit",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip().splitlines()[0]
        name, total, limit = (p.strip() for p in out.split(",", 2))
        return {"name": name, "total_mib": int(float(total)), "limit_w_start": _num(limit)}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {"name": None, "total_mib": None, "limit_w_start": None}


def fit_verdict(
    num_envs: int,
    free_mib: int | None,
    reserve_mib: int = DEFAULT_RESERVE_MIB,
    allow_over_budget: bool = False,
    *,
    overhead: float = VRAM_OVERHEAD_MIB,
    per_env: float = VRAM_PER_ENV_MIB,
) -> dict:
    """装得下吗。`ok=False` 时 `suggested` 是一个**确实装得下**的 num_envs。"""
    estimate = estimate_vram_mib(num_envs, overhead=overhead, per_env=per_env)
    base = {"estimate_mib": estimate, "budget_mib": None, "suggested": None,
            "over_budget": False, "reason": None}
    if free_mib is None:
        # 读不到数不是拒绝的理由：拒绝要基于一个数，不是基于「我没读到数」。
        return {**base, "ok": True}
    budget = max(0, free_mib - reserve_mib)
    if estimate <= budget:
        return {**base, "ok": True, "budget_mib": budget}
    suggested = int((budget - overhead) // per_env)
    return {
        **base,
        "ok": allow_over_budget,
        "budget_mib": budget,
        "suggested": suggested if suggested >= 1 else None,
        "over_budget": True,
        "reason": (
            f"num_envs={num_envs} needs about {estimate} MiB, but only {budget} MiB "
            f"is usable ({free_mib} MiB free minus {reserve_mib} MiB reserve)"
        ),
    }


def write_status(run_dir: Path | None, status: dict) -> None:
    """原子写：先写 `.json.tmp` 再 `os.replace`，读的人永远看不到半个文件。"""
    if run_dir is None:
        return
    target = run_dir / "status.json"
    problems = run_state.validate(status)
    if problems:
        # 契约（`run_state.FIELDS`）不是装饰：读的人靠它。缺字段不该拦住训练，
        # 但必须留痕，否则 `health_jumper` 读到一份残缺记录会给出「正常」。
        print(f"[long_run] warning: {target.name} is off contract: {'; '.join(problems)}",
              file=sys.stderr)
    try:
        run_state.write(target, status)
    except OSError as e:
        # 写不进去不该杀掉训练，但也不能一声不响——否则磁盘满和一切正常看起来一样。
        print(f"[long_run] warning: cannot write {target}: {e}", file=sys.stderr)


def _own_args(argv: list[str]) -> tuple[dict, list[str]]:
    """摘掉本脚本自己的开关，其余原样透传给 train.py。

    只认四个：`--interval`、`--print-command`、`--reserve-mib`、`--allow-over-budget`。
    **不认 `--dry-run`** —— 那是 `scripts/train.py` 自己的开关（`_cli.py`:157），
    在这里截走会让「预演」变成「真跑」。
    """
    own = {"interval": 30.0, "print_command": False,
           "reserve_mib": DEFAULT_RESERVE_MIB, "allow_over_budget": False}
    passthrough: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        value: str | None = None
        if a in ("--interval", "--reserve-mib"):
            if i + 1 >= len(argv):
                flag = "seconds" if a == "--interval" else "MiB"
                raise SystemExit(f"[long_run] {a} needs a value, in {flag}")
            value = argv[i + 1]
            i += 2
        elif a.startswith(("--interval=", "--reserve-mib=")):
            a, value = a.split("=", 1)
            i += 1
        elif a == "--print-command":
            own["print_command"] = True
            i += 1
            continue
        elif a == "--allow-over-budget":
            own["allow_over_budget"] = True
            i += 1
            continue
        else:
            passthrough.append(a)
            i += 1
            continue

        assert value is not None
        if a == "--interval":
            try:
                own["interval"] = float(value)
            except ValueError:
                raise SystemExit(f"[long_run] --interval expects a number, got {value!r}") from None
            # `inf`/`nan` 能通过 `> 0` 的检查，但前者让采样再也不触发，后者让
            # 节拍变成未定义——两种都是「看起来在跑，其实状态不再更新」。
            if not math.isfinite(own["interval"]) or own["interval"] <= 0:
                raise SystemExit("[long_run] --interval must be a finite number greater than 0")
        else:
            try:
                reserve = float(value)
            except ValueError:
                raise SystemExit(f"[long_run] --reserve-mib expects a number, got {value!r}") from None
            if not math.isfinite(reserve) or reserve < 0:
                raise SystemExit("[long_run] --reserve-mib must be a finite number of MiB, 0 or more")
            own["reserve_mib"] = int(reserve)
    return own, passthrough


def _max_iterations(args: list[str]) -> int | None:
    for i, a in enumerate(args):
        if a == "--max-iterations" and i + 1 < len(args):
            try:
                return int(args[i + 1])
            except ValueError:
                return None
        if a.startswith("--max-iterations="):
            try:
                return int(a.split("=", 1)[1])
            except ValueError:
                return None
    return None


def _load_env() -> None:
    """把 `.env` / `.env.local` 填进自己的环境（不覆盖已设的值）。

    在 `main` 里无条件调用一次：`MJRL_BACKEND`、`CUDA_VISIBLE_DEVICES`、
    `MJRL_NUM_ENVS` 都是同一次导入的一部分，只让其中一个按「有没有传
    `--num_envs`」决定读不读，等于让同一份配置在两个入口下指向不同的卡。
    """
    # 局部导入只是把失败推后：`_load_env()` 是 `main()` 的第一句，`--help`/`--list`/
    # `--print-command` 都要先过它。`mjrl` 坏掉时这几条命令必须照样能用（C1 的正题；
    # 巡检的判决命令也依赖同一个前提），所以导入失败等于「没有 `.env` 可读」，不是故障。
    try:
        from mjrl.dotenv import load_dotenv
    except ImportError as exc:
        # 说一句，但不是故障：`.env` 里的 `MJRL_BACKEND`/`CUDA_VISIBLE_DEVICES`
        # 会**静默**退回默认值，多卡机器上那就是「跑在另一张卡上」。stdout 保持
        # 干净（`--print-command` 只打印命令），所以走 stderr。
        print(f"[long_run] .env was not read: {exc}", file=sys.stderr)
        return

    load_dotenv(REPO / ".env", REPO / ".env.local")


def _flag(args: list[str], name: str) -> str | None:
    for i, a in enumerate(args):
        if a == name and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def _num_envs(args: list[str], env: dict[str, str] | None = None) -> int:
    """后端真正会用的环境数，用后端自己的解析器算。

    flag → `MJRL_NUM_ENVS` → 后端默认值，而「后端默认值」取决于 `--backend`
    与 `MJRL_BACKEND` 解析后的结果（`auto` 在这台机器上是 warp，在只有 CPU 的
    机器上是 native，默认值 64）。自己重写这张表就会在 `auto`/`MJRL_BACKEND`
    上和训练侧差出 64 倍——预算算大的后果是拦掉一次本该跑得起来的训练。
    """
    if "--num_envs" in args and args.index("--num_envs") == len(args) - 1:
        # 先分清「没给值」和「给了一个太小/不是数的值」：两者的提示不一样。
        raise SystemExit("[long_run] --num_envs needs a value") from None
    raw = _flag(args, "--num_envs")
    if raw is not None:
        try:
            n = int(raw)
        except ValueError:
            raise SystemExit(f"[long_run] --num_envs expects an integer, got {raw!r}") from None
        if n < 1:
            raise SystemExit(f"[long_run] --num_envs must be 1 or more, got {n}")
        return n
    raw = (env if env is not None else os.environ).get("MJRL_NUM_ENVS") or None
    if raw is not None:
        try:
            n = int(raw)
        except ValueError:
            raise SystemExit(f"[long_run] MJRL_NUM_ENVS is not an integer: {raw!r}") from None
        if n < 1:
            raise SystemExit(f"[long_run] MJRL_NUM_ENVS must be 1 or more, got {n}")
        return n
    return _backend_default_envs(_flag(args, "--backend") or "auto")


def _backend_default_envs(backend: str) -> int:
    """后端在没给 `num_envs` 时会用多少。问训练侧自己的解析器，不重写它的规则。"""
    try:
        from mjrl.backend.resolve import BackendUnavailable, resolve

        return int(resolve(backend=backend, device="auto", num_envs=None).num_envs)
    except (BackendUnavailable, ImportError, AttributeError, TypeError,
            ValueError, OSError) as exc:
        # 问不出来就按最大的一档算：预算偏大最坏是拦一次能跑的训练（有个明确的
        # 开关能压过去），偏小则会把一次 OOM 放进去。但要说出来——一个坏掉的
        # import 和「文档里写的回退」在这里长得一模一样，不说就没人能分开。
        widest = max(BACKEND_DEFAULT_NUM_ENVS.values())
        print(f"[long_run] cannot ask the backend for its default environment count "
              f"({exc.__class__.__name__}: {exc}); assuming the largest, {widest}",
              file=sys.stderr)
        return widest


def _sample(status: dict, *, alive: bool = True) -> None:
    """采一次卡，并把功耗限额的摆动区间记进 `gpu`。

    `alive=False` 是收尾那次：子进程已经退出，显存早就还回去了，这一采只用来
    补功耗与用量，不能覆盖 `mem_mib`——否则「这一跑用了多少」会被写成退出后的
    空闲值（2026-10-31 11-12-58 就是这么记下 66 MiB 的）。

    峰值单独记在 `peak_mem_mib` 里。它会话里唯一的常驻口径：`mem_mib` 是采样那一刻，
    峰值才是「这次训练要占多少」。
    """
    was = status.get("mem_mib")
    status.update(sample_gpu())
    if not alive and was is not None:
        status["mem_mib"] = was
    mem = status.get("mem_mib")
    if mem is not None:
        peak = status.get("peak_mem_mib")
        status["peak_mem_mib"] = mem if peak is None else max(peak, mem)
    limit, gpu = status.get("limit_w"), status.get("gpu") or {}
    if limit is None:
        return
    lo, hi = gpu.get("limit_min_w"), gpu.get("limit_max_w")
    gpu["limit_min_w"] = limit if lo is None else min(lo, limit)
    gpu["limit_max_w"] = limit if hi is None else max(hi, limit)


def _reader(stream, sink: queue.Queue) -> None:
    """把子进程的每一行塞进队列，EOF 时塞一个 `None` 当哨兵。"""
    try:
        for line in stream:
            sink.put(line)
    finally:
        sink.put(None)


#: `scripts/train.py` 在这些开关上于 `_run` 之前就返回（`--list` 在 :119、
#: `--dry-run` 在 :128），`--help` 更早在 argparse 里退。它们都不建环境、
#: 不申请显存，所以既不该排在别人的锁后面，也不该被显存预算拦下——否则
#: 「先把一份配置看一眼」这件事，在没有空卡的时候永远做不到。
CARD_FREE_FLAGS = ("--dry-run", "--list", "--help", "-h")


def _needs_the_card(passthrough: list[str]) -> bool:
    """这条命令会不会真的碰卡。"""
    return not any(f in passthrough for f in CARD_FREE_FLAGS)


def main(argv: list[str]) -> int:
    _load_env()
    own, passthrough = _own_args(argv)
    command = [sys.executable, str(TRAIN), *passthrough]
    total = _max_iterations(passthrough)

    if own["print_command"]:
        print(" ".join(command))
        return 0

    if not _needs_the_card(passthrough):
        return _supervise(own, passthrough, command, total, preview=True)

    task = next((passthrough[i + 1] for i, a in enumerate(passthrough)
                 if a == "--task" and i + 1 < len(passthrough)), None)
    # 先拿锁再算账：中间那段空档足够另一个训练挤进来，把刚量到的余量吃掉。
    # `ExitStack` 把「拿锁」和「跑训练」分开：只有拿锁这一步的 `OSError` 才是
    # 锁的问题。整个 `with` 包起来的话，`Popen` 的 `EMFILE`、运行目录被删掉的
    # `EACCES` 都会被报成「锁不可用」并退出 71。
    with contextlib.ExitStack() as stack:
        try:
            stack.enter_context(gpu_lock.hold(gpu_lock.DEFAULT_LOCK,
                                              role=f"long_run {task or 'train'}"))
        except gpu_lock.GpuBusy as busy:
            print(f"[long_run] {busy}; nothing was started", file=sys.stderr)
            return gpu_lock.BUSY_RC
        except OSError as exc:
            # 锁那条路本身不通（`ENOLCK` 的挂载、软链、目录……）。和「有人在用」
            # 分开报，也分开退出：这个数重试没有用，换锁路径才有用。
            print(f"[long_run] cannot use the GPU lock at {gpu_lock.DEFAULT_LOCK}: "
                  f"{exc.strerror or exc}", file=sys.stderr)
            return gpu_lock.UNAVAILABLE_RC
        return _supervise(own, passthrough, command, total)


class Terminated(Exception):
    """收到 SIGTERM/SIGHUP。做成异常，让收尾路径跑完再退出。

    不继承 `KeyboardInterrupt`：那是 `BaseException`，pytest 之类的框架会把它当成
    「用户按了 Ctrl-C」并中止整场测试——一条断言要能抓住它。

    SIGTERM 的默认动作是立刻终止进程，`finally` 不执行——2026-10-30 22:00 那次被
    `pkill` 之后，目录里留着一份 `status: running / pid 339735`，而进程早就没了，
    巡检每轮读到「进程不在但状态写着 running」只能报 `investigate`。
    """

    def __init__(self, signum: int) -> None:
        super().__init__(f"signal {signum}")
        self.signum = signum


def _raise_terminated(signum: int, frame) -> None:  # noqa: ARG001 - 信号处理器的签名
    raise Terminated(signum)


def _install_termination_handlers() -> Callable[[], None]:
    """把 SIGTERM/SIGHUP 换成会抛异常的处理器，返回还原函数。"""
    previous: dict[int, object] = {}
    for name in ("SIGTERM", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            previous[sig] = signal.signal(sig, _raise_terminated)
        except (ValueError, OSError):
            # 不在主线程、或平台不支持：留默认行为，比假装装上了好。
            continue

    def restore() -> None:
        for sig, handler in previous.items():
            signal.signal(sig, handler)

    return restore


def _reap(proc) -> int | None:
    """收掉子进程，返回它的退出码。

    收尾期间再来一个信号是预期的，不是错误：终态在进入这里之前就已经定下来了，
    这里只负责把它杀干净。让第二个信号从 `except` 里逃出去，等于把一次正常停机
    变成一段 traceback。
    """
    try:
        proc.terminate()
        try:
            return proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            return proc.wait()
    except (KeyboardInterrupt, Terminated):
        try:
            proc.kill()
            return proc.wait()
        except (KeyboardInterrupt, Terminated, OSError):
            return None


def _supervise(own: dict, passthrough: list[str], command: list[str], total: int | None,
               *, preview: bool = False) -> int:
    if preview:
        # 预演不写 status.json：它没有自己的运行目录，`newest_run_dir` 会把
        # 别人刚写的那份认成自己的，然后拿「completed + 预演的 pid」盖掉它。
        print(f"[long_run] preview: {' '.join(command)}", file=sys.stderr)
        return subprocess.call(command, cwd=REPO, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    # 先量卡再问后端：`resolve()` 在 `auto` 下会 `import torch` 并 `wp.init()`，
    # 那是本进程自己的 CUDA 上下文——量在它后面，等于把自己的开销算成别人占的。
    free_mib = free_vram_mib()
    task = next((passthrough[i + 1] for i, a in enumerate(passthrough)
                 if a == "--task" and i + 1 < len(passthrough)), None)
    overhead, per_env, calibrated = calibration_for(task)
    if not calibrated:
        print(f"[long_run] no calibration for {task or 'this task'}; assuming the largest "
              f"on record ({overhead:.0f} MiB + {per_env:.2f} MiB/env)", file=sys.stderr)
    verdict = fit_verdict(
        _num_envs(passthrough), free_mib,
        own["reserve_mib"], own["allow_over_budget"],
        overhead=overhead, per_env=per_env)
    if not verdict["ok"]:
        hint = (f"try --num_envs {verdict['suggested']}" if verdict["suggested"]
                else "free some memory first, or pick a smaller --num_envs")
        print(f"[long_run] refusing to start: {verdict['reason']} — {hint}", file=sys.stderr)
        print("[long_run] --allow-over-budget overrides this", file=sys.stderr)
        return 2
    if verdict["over_budget"]:
        print(f"[long_run] warning: {verdict['reason']}; going ahead anyway "
              "(--allow-over-budget)", file=sys.stderr)

    started = time.monotonic()
    status: dict = {
        "v": run_state.SCHEMA_VERSION,
        "pid": None,
        "status": "running",
        "run_dir": None,
        "iter": None,
        "percent": None,
        "eta_s": None,
        "step_time": None,
        "elapsed_s": None,
        "limit_w": None,
        "mem_mib": None,
        "peak_mem_mib": None,
        "util": None,
        "power_w": None,
        "ts": time.time(),
        "parse_error": False,
        "rc": None,
        # 采样周期写进契约，健康检查据此判断「状态文件多久没刷新才算卡住」。
        "interval_s": own["interval"],
    }
    status["gpu"] = {
        **gpu_identity(),
        "estimate_mib": verdict["estimate_mib"],
        "budget_mib": verdict["budget_mib"],
        "limit_min_w": None,
        "limit_max_w": None,
    }

    # 快照必须取在 `Popen` 之前：`newest_run_dir` 的 `since` 有 2 秒容差，上一条刚
    # 写完盘的运行目录会落进容差里。2026-10-30 的队列实测过——seed1 全程把 status.json
    # 写进了 seed0 的目录，两份记录一起毁掉，而状态内容看上去完全正常。
    known = set(run_state.run_dirs())
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        command, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors="replace", bufsize=1, env=env,
    )
    status["pid"] = proc.pid
    assert proc.stdout is not None
    sink: queue.Queue = queue.Queue()
    threading.Thread(target=_reader, args=(proc.stdout, sink), daemon=True).start()

    run_dir: Path | None = None
    seen = 0
    next_sample = 0.0
    eof = False
    restore_handlers = _install_termination_handlers()
    try:
        while not eof or proc.poll() is None:
            try:
                line = sink.get(timeout=0.2)
            except queue.Empty:
                line = ""
            if line is None:
                eof = True
            elif line:
                sys.stdout.write(line)
                key, value = parse_line(line)
                if key is None:
                    if "Iteration time" in line or "ETA" in line:
                        status["parse_error"] = True
                elif key == "iter_time":
                    seen += 1
                    status["step_time"] = value
                    status["iter"] = seen - 1
                    if total:
                        status["percent"] = round(100.0 * seen / total, 1)
                elif key == "eta":
                    status["eta_s"] = value
                elif key == "elapsed":
                    status["elapsed_s"] = value

            now = time.monotonic()
            if now >= next_sample:
                next_sample = now + own["interval"]
                if run_dir is None:
                    run_dir = run_state.newest_run_dir(time.time() - (now - started),
                                             exclude=known)
                    status["run_dir"] = str(run_dir) if run_dir else None
                _sample(status)
                status["ts"] = time.time()
                write_status(run_dir, status)

        rc = proc.wait()
        status["status"] = "completed" if rc == 0 else "crashed"
        status["rc"] = rc
        if rc < 0:
            # `Popen.returncode` 是负的那说明**子进程自己**被信号杀了（OOM 的
            # SIGKILL、外面的 `kill -9`）；本进程一条信号都没收到，`status["signal"]`
            # 不会填。不记下来，盘上就只有 `rc: -9`——读的人看不出是被谁杀的。
            status["child_signal"] = -rc
    except (KeyboardInterrupt, Exception) as e:  # noqa: BLE001 - 写终态，再原样抛出
        # 先把终态定下来，再收子进程。`terminate` + `wait(timeout=15)` 最长十五秒，
        # 而处理器这时还装着：这期间再来一个信号（`pkill` 连发、`pkill` 之后 Ctrl-C）
        # 会直接跳到 `finally`，写下的是那一刻的 `status`。终态若还排在 `wait()`
        # 之后，写出去的就是 `running`——「进程没了、状态写着 running」。
        if isinstance(e, Terminated):
            # 人叫停的。和崩溃分开记：前者不该升级成告警。
            status["status"] = "interrupted"
            status["signal"] = e.signum
        elif isinstance(e, KeyboardInterrupt):
            # Ctrl-C 也是人叫停，只是走的是另一条异常路径：前台按下的那一下没有
            # 我们的处理器，是解释器自己抛的。
            status["status"] = "interrupted"
            status["signal"] = int(signal.SIGINT)
        else:
            status["status"] = "crashed"
        status["error"] = f"{type(e).__name__}: {e}"
        rc = _reap(proc)
        status["rc"] = rc
        if rc is not None and rc < 0:
            # 这条路径也会拿到负值：`terminate()`/`kill()` 作用在一个已经被 OOM
            # 收走的子进程上，`wait()` 回的是它真正的死因（-9/-15），不是我们发的
            # 那个信号。不记下来，`status.json` 在这一路上仍然只有裸的负 rc。
            status["child_signal"] = -rc
        print(f"\n[long_run] aborted ({type(e).__name__}); child rc={rc}", file=sys.stderr)
        if not isinstance(e, (KeyboardInterrupt, Terminated)):
            raise
    finally:
        # 处理器要等终态写完再还回去。收尾那次 `_sample` 走 `nvidia-smi`，有几秒
        # 窗口；还原早了，那几秒里来的 SIGTERM 就是默认动作——进程立刻死，
        # 一条终态都没写。
        try:
            if run_dir is None:
                run_dir = run_state.newest_run_dir(time.time() - (time.monotonic() - started),
                                         exclude=known)
                status["run_dir"] = str(run_dir) if run_dir else None
            try:
                _sample(status, alive=False)
            except BaseException as exc:  # noqa: BLE001 - 采样坏了也不能不写终态
                # 收尾时的采样失败（nvidia-smi 超时、第二次 SIGTERM）过去会把异常从
                # `finally` 抛出去，于是终态一条都没写——正是这个脚本存在的理由被绕过。
                # `BaseException` 而不是 `Exception`：收尾这几秒里来的**第二次 Ctrl-C**
                # 是 `KeyboardInterrupt`，它照样会把 `write_status` 跳过，把 `running`
                # 留在盘上——正是这次改动要消掉的那个状态。
                status["sample_error"] = f"{type(exc).__name__}: {exc}"
            status["ts"] = time.time()
            write_status(run_dir, status)
        finally:
            restore_handlers()

    # 预演没有运行目录（`train.py --dry-run` 在写任何东西之前就返回了）。
    where = f" status={run_dir}/status.json" if run_dir is not None else ""
    if status["status"] == "interrupted":
        # 被信号叫停按 shell 惯例退 128+signum（143 是 SIGTERM），让调用方看得懂
        # 「有人叫停」和「自己崩了」的区别。`status["rc"]` 留的是子进程的码。
        code = 128 + int(status.get("signal") or signal.SIGTERM)
    else:
        code = status["rc"] if isinstance(status["rc"], int) else 1
        if code < 0:
            # 负 rc 直接返回，调用方看到的是 247，`status.json` 里是 `-9`，两个都读不出
            # 「被谁杀的」。按 shell 惯例换成 128+n（子进程 SIGKILL → 137）。
            code = 128 - code
    print(f"[long_run] {status['status']} rc={code}{where}")
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
