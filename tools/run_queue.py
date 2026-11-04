"""跑一串训练：清单 → 串行 → 每条核验 → 汇总。

一个 JSON 清单描述今晚要跑的几条训练，这个脚本一条一条交给 `tools/long_run.py`，
每条跑完**先核验再往下走**，最后写一份汇总。

为什么需要它，知识库 `reports/Voxline/long-task-online-health-checks.md` 已经写过：

- **退出 0 ≠ 完成**。`long_run` 的 rc=0 只说明进程没崩；跑到第 3000 次迭代就退出、
  或者策略塌缩后自己收掉，rc 都是 0。所以每条跑完要看 `status.json` 的 `status`
  和最后一个 checkpoint 的编号，不是看退出码。
- **串行要独立子进程**（`retrospectives/SciMonitor/probe-matrix-full-saga.md` §4.1）。
  显存和 CUDA 上下文不会因为一个 run 结束就还回来，所以下面每条都起一个新进程，
  而不是在同一个进程里反复调用训练入口。

失败即停：第一条没跑通，后面几条大概率犯同一个错误，继续跑只是拿三倍时间换三个
同样的失败。汇总文件无论如何都会写，因为「停在哪一条、为什么」比「全部通过」更需要留档。

汇总写在清单旁边：`queue.json` → `queue.result.json`。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import run_state

#: `tasks/jumper/common/ppo.py:112` 的 `save_interval`。它不是 `long_run` 的参数，
#: 清单里也不接受这个键——核验用的节奏必须与训练实际用的那一个相同，不能让清单改。
DEFAULT_SAVE_INTERVAL = 50

REQUIRED_FIELDS = ("task", "seed", "iterations", "num_envs")
INT_FIELDS = ("seed", "iterations", "num_envs")
ALLOWED_FIELDS = frozenset({"name", "args", "start_from", *REQUIRED_FIELDS})


def _resolve(path: str | Path) -> Path:
    """清单里的相对路径按**仓库根**解析。

    队列子进程的 cwd 就是仓库根（`_spawn`），清单里写的也是 `logs/...` 这种相对
    仓库根的写法；按队列自己的 cwd 解析，在别处启动就会校验失败或指向别处。
    """
    candidate = Path(path)
    return candidate if candidate.is_absolute() else run_state.REPO / candidate


def load_manifest(path: Path) -> list[dict]:
    """读清单，缺什么就点名说什么，不猜默认值。

    不认识的键也拒绝：静默丢掉一个字段，等于让写清单的人以为它生效了。
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        unknown_top = sorted(set(data) - {"runs"})
        if unknown_top:
            raise ValueError(
                f"{path}: cannot honour {', '.join(unknown_top)} at the top level — "
                f"the manifest only knows 'runs'")
    runs = data.get("runs") if isinstance(data, dict) else None
    if not isinstance(runs, list) or not runs:
        raise ValueError(f"{path}: no runs")
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ValueError(f"run {index}: not an object")
        missing = [field for field in REQUIRED_FIELDS if field not in run]
        if missing:
            raise ValueError(f"run {index}: missing {', '.join(missing)}")
        unknown = sorted(set(run) - ALLOWED_FIELDS)
        if unknown:
            raise ValueError(
                f"run {index}: cannot honour {', '.join(unknown)} — "
                f"the queue only knows {', '.join(sorted(ALLOWED_FIELDS))}")
        # `name` 会原样变成账本的键（`label_of` 直接返回它）。非字符串写进结果文件后，
        # 两个读取方都只留字符串键，那条记录就永远丢了：下一次从头再跑，`fire_next`
        # 还会在 `', '.join(missing)` 上抛 `TypeError` 而不是给判决。
        name = run.get("name")
        if name is not None and not isinstance(name, str):
            raise ValueError(f"run {index}: name must be a string, not {name!r}")
        for field in INT_FIELDS:
            value = run[field]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"run {index}: {field} must be an integer, not {value!r}")
        # `iterations <= save_interval` 会让阈值降到 0 以下，预算检查就永远成立。
        if run["iterations"] <= DEFAULT_SAVE_INTERVAL:
            raise ValueError(
                f"run {index}: iterations must exceed the {DEFAULT_SAVE_INTERVAL}-iteration "
                f"save interval, not {run['iterations']}")
        # 任务自己的开关（目前只有 tripod 的 `--delay`）。只收长选项：短选项和裸值
        # 会和队列自己摆的参数混在一起，谁也说不清哪一段是谁放的。
        args = run.get("args", [])
        if not isinstance(args, list) or any(
                not isinstance(item, str) or not item.startswith("--") for item in args):
            raise ValueError(
                f"run {index}: args must be a list of long flags such as --delay, "
                f"not {args!r}")
        # 长选项也会撞车：`--resume` 会让 long_run 自己找最新目录（可能是别的种子），
        # `--max-iterations=5000` 会让实际预算与核验预算不一致。
        owned = sorted({item.split("=", 1)[0] for item in args} & QUEUE_OWNED_FLAGS)
        if owned:
            raise ValueError(
                f"run {index}: the queue decides {', '.join(owned)} itself; "
                f"setting it here would make the ledger disagree with the run")
        # 续跑的起点是一个**已存在的目录**。路径打错一个字符就当从零开始跑：
        # 训练会照常完成、核验会照常通过，只是跑的是另一条训练。
        start_from = run.get("start_from")
        if start_from is not None:
            if not isinstance(start_from, str) or not start_from:
                raise ValueError(
                    f"run {index}: start_from must be a path, not {start_from!r}")
            if not _resolve(start_from).is_dir():
                raise ValueError(
                    f"run {index}: start_from {start_from!r} is not a directory "
                    f"(resolved against the repository root)")
    return runs


#: 队列自己摆的参数。清单的 `args` 只能带任务自己的开关（`--delay` 这类）。
QUEUE_OWNED_FLAGS = frozenset({
    "--task", "--seed", "--max-iterations", "--num-envs", "--num_envs", "--headless",
    "--checkpoint", "--resume", "--dry-run", "--list", "--help", "-h",
    # `long_run` 自己的四个开关。`long_run._own_args` 会把它们从透传里摘掉，所以在清单
    # 里写下去不报错，只是**悄悄改掉包装层的策略**：`--allow-over-budget` 让一次被显存
    # 预算拒绝的运行照跑（账本一个字都不记）、`--interval=0.001` 把采样打成风暴、
    # `--reserve-mib=0` 抹掉安全余量。与 `--resume` 同一类：账本会和实际跑的不一致。
    "--interval", "--reserve-mib", "--allow-over-budget", "--print-command",
})


def build_command(run: dict) -> list[str]:
    """清单里的一条 → `tools/long_run.py` 的一条命令行。"""
    command = [
        sys.executable, "tools/long_run.py",
        "--task", str(run["task"]),
        "--seed", str(run["seed"]),
        "--max-iterations", str(run["iterations"]),
        "--num_envs", str(run["num_envs"]),
        "--headless",
        *[str(flag) for flag in run.get("args", [])],
    ]
    # 续跑的起点只能由清单指定：`--checkpoint` 是队列自有参数，`args` 里写会被拒收。
    if run.get("start_from"):
        command += ["--checkpoint", str(run["start_from"])]
    return command


def verify_run(run_dir: Path | None, iterations: int, save_interval: int, rc: int,
               *, start_iteration: int = 0) -> dict:
    """一条 run 是否真的跑完了。

    判据是最后一个 checkpoint 的编号，不是退出码：训练在 `max_iterations` 上停住时，
    最后一次保存的编号离 `iterations` 至多远一个 `save_interval`。

    续跑的编号从起点接着往上走，所以阈值必须以 `start_iteration` 为基线；否则一条
    只跑了 200 步的续跑会凭空通过（200 ≥ 3000 − 50 当然不成立，可 10200 会）。
    """
    if run_dir is None:
        return {
            "ok": False,
            "problems": [f"no run directory: long_run left nothing behind (exit code {rc})"],
            "last_checkpoint": None,
            "checkpoints": 0,
        }

    problems: list[str] = []
    status_path = Path(run_dir) / "status.json"
    # 读法由 `run_state` 定义：读不出来一律是没有可信状态，不是「JSON 错在哪」。
    status = run_state.read(status_path)
    if status is None:
        problems.append(f"cannot read {status_path.name}: no status record")
    elif not status:
        # `{}` 是合法 JSON，却什么都没说——不能当成功，也不能被跳过。
        problems.append(f"{status_path.name} carries no status record")
    elif status.get("status") != "completed":
        problems.append(f"status is {status.get('status')!r}, not 'completed'")

    numbers = sorted(
        int(path.stem.split("_", 1)[1])
        for path in Path(run_dir).glob("model_*.pt")
        if path.stem.split("_", 1)[1].isdigit()
    )
    last = numbers[-1] if numbers else None
    expected = start_iteration + iterations - save_interval
    if last is None:
        problems.append("no checkpoints were written")
    elif last < expected:
        problems.append(
            f"stopped early: the last checkpoint is model_{last}.pt, "
            f"expected at least model_{expected}.pt"
        )
    if rc != 0:
        problems.append(f"long_run exited {rc}")

    return {
        "ok": not problems,
        "problems": problems,
        "last_checkpoint": last,
        "checkpoints": len(numbers),
    }


def _checkpoint_numbers(run_dir: Path) -> list[int]:
    numbers = []
    for path in Path(run_dir).glob("model_*.pt"):
        suffix = path.stem.split("_", 1)[1]
        if suffix.isdigit():
            numbers.append(int(suffix))
    return sorted(numbers)


def label_of(run: dict) -> str:
    """一条 run 在汇总里的名字。清单没给就自己拼一个，别让两条重名。

    名字里必须带上 `args`：只差 `--delay` 的两条是两条完全不同的训练，共用一个
    账本键会让后一条被当成「已跑完」跳过，或续到前一条的目录里。
    """
    if run.get("name"):
        return run["name"]
    # 预算和环境数也进名字：`iterations` 从 3000 改成 6000 是同名的另一条训练，
    # 共用账本键会让第二条被当成「已跑完」跳过。
    label = (f"{run['task']} seed {run['seed']} "
             f"{run['iterations']}it {run['num_envs']}env")
    args = run.get("args") or ()
    if args:
        label = f"{label} {' '.join(str(flag) for flag in args)}"
    # 起点也进名字：同一个 task/seed/预算，从 3000 续和从 10000 续是两条训练。
    if run.get("start_from"):
        label = f"{label} from {run['start_from']}"
    return label


def last_checkpoint(run_dir: Path) -> int | None:
    numbers = _checkpoint_numbers(run_dir)
    return numbers[-1] if numbers else None


def start_iteration_of(run: dict) -> int:
    """续跑起点目录里**训练会加载的**那个 checkpoint 的编号；从零开始就是 0。

    必须走 `mjrl.checkpoint.newest_checkpoint`，与 `scripts/train.py` 同一个解析器：
    它按 mtime 取最新文件，并且会向下找一层。按编号取最大会算出另一个文件——续跑
    写出的新 run 编号可能比盘上的旧文件小，那时编号最大的不是最新的。
    """
    source = run.get("start_from")
    if not source:
        return 0
    # 局部导入：`tools/` 顶层不许碰 `mjl*`（C1）。`fire_next` 与 `next_stage` 都
    # import 本模块，模块级一挂，巡检的判决命令连一句结论都打不出来——而那正是
    # 环境坏了、最需要看门狗开口的时候。
    from mjrl.checkpoint import newest_checkpoint

    found = newest_checkpoint(_resolve(source))
    if found is None:
        return 0
    match = re.match(r"^model_(\d+)\.pt$", found.name)
    if match is None:
        # `newest_checkpoint` 接受任何 `model_*.pt`（`_sort_key` 给非规格名字排代价
        # -1）。返回 0 会把「起点在这」变成「从零开始」：阈值退化成整段预算，那条
        # 条目于是能被判成 `done` 而一次都没跑。宁可拒绝。
        raise ValueError(
            f"{found} is the newest checkpoint but not model_<iteration>.pt, so the "
            f"iteration it resumes from cannot be read")
    return int(match.group(1))


def previous_runs(result_path: Path) -> dict[str, dict]:
    """上一次的汇总，按名字索引。读不出来就当没有——旧账本坏了不该挡住新的一夜。"""
    try:
        payload = json.loads(Path(result_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    runs = payload.get("runs") if isinstance(payload, dict) else None
    if not isinstance(runs, list):
        return {}
    return {
        record["name"]: record
        for record in runs
        if isinstance(record, dict) and isinstance(record.get("name"), str)
    }


def plan_run(run: dict, prior: dict | None) -> dict:
    """这一条该跳过、该续跑，还是从零开始。

    跳过和续跑都只信上一次的账本，不信 mtime：`logs/<model>/<task>/` 下可能躺着
    别的种子的目录，按最新一个续跑会续错对象（2026-10-30 seed1 的事故）。
    """
    label = label_of(run)
    budget = run["iterations"]
    done = {"name": label, "action": "done", "command": None,
            "resume_from": None, "budget": budget, "record": prior}
    if prior and prior.get("ok"):
        return done

    command = build_command(run)
    if not (prior and prior.get("run_dir")):
        return {"name": label, "action": "start", "command": command,
                "resume_from": None, "budget": budget, "record": None}

    run_dir = Path(prior["run_dir"])
    last = last_checkpoint(run_dir) if run_dir.is_dir() else None
    if last is None:
        return {"name": label, "action": "start", "command": command,
                "resume_from": None, "budget": budget, "record": None}
    # 预算与 checkpoint 编号不是同一种计数：续跑条的 `iterations` 是「再跑多少」，
    # 而编号是绝对的。从 10000 起步、跑到 10600 时，7000 的预算只剩 6400。
    start = start_iteration_of(run)
    remaining = budget - (last - start)
    if remaining <= 0:
        return {**done, "resume_from": str(run_dir)}
    command[command.index("--max-iterations") + 1] = str(remaining)
    # 清单给的起点与账本记的上次目录只能留一个：两个都给时 argparse 取最后一个，
    # 训练续到的地方就与账本算阈值的地方不是同一个。
    while "--checkpoint" in command:
        index = command.index("--checkpoint")
        del command[index:index + 2]
    command += ["--checkpoint", str(run_dir)]
    return {"name": label, "action": "resume", "command": command,
            "resume_from": str(run_dir), "budget": budget, "record": None}


def _write_result(path: Path, manifest: Path, started: str,
                  records: list[dict]) -> None:
    """整份写完再换名：这份文件正是「被打断」时要靠的那一份。

    `manifest` 是**跑的那份清单**，不是这份汇总自己。两者写成同一个值，归档字段就
    永远指向结果文件，而它存在的理由正是「被打断后还能看出跑的是哪份清单」。
    """
    summary = {"manifest": str(manifest), "started": started,
               "finished": _now(), "runs": records}
    tmp = Path(f"{path}.tmp")
    tmp.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _spawn(command: list[str]) -> int:
    return subprocess.call(command, cwd=run_state.REPO)


def main(argv: list[str] | None = None, *,
         spawn: Callable[[list[str]], int] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    spawn = spawn or _spawn
    if not argv:
        print("usage: run_queue.py <manifest.json>", file=sys.stderr)
        return 2

    path = Path(argv[0])
    try:
        runs = load_manifest(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[run_queue] {exc}", file=sys.stderr)
        return 2

    result = path.with_suffix(".result.json")
    started = _now()
    prior = previous_runs(result)
    records: list[dict] = []
    rc_all = 0
    stopped = False
    for run in runs:
        label = label_of(run)
        # 只有「规划」受这个守卫保护：`start_iteration_of` 对非 `model_<n>.pt` 的名字
        # 故意抛 `ValueError`。把 spawn、`_write_result`、`verify_run` 一起包进来，
        # 一次磁盘写失败就会被记成「规划失败」，在同一份汇总里多留一条假记录。
        try:
            plan = plan_run(run, prior.get(label_of(run)))
            label = plan["name"]
            start_iteration = start_iteration_of(run)
        except (ValueError, OSError) as exc:
            records.append({
                "name": label,
                "run_dir": None,
                "rc": None,
                "resumed_from": None,
                "ok": False,
                "problems": [f"cannot plan this run: {exc}"],
                "last_checkpoint": None,
                "checkpoints": 0,
            })
            print(f"[run_queue] {label} cannot plan: {exc}", file=sys.stderr)
            rc_all = 2
            stopped = True
            break

        if plan["action"] == "done":
            known_dir = (plan["record"] or {}).get("run_dir")
            run_dir = Path(known_dir) if known_dir else None
            verdict = verify_run(run_dir, plan["budget"], DEFAULT_SAVE_INTERVAL, 0,
                                 start_iteration=start_iteration)
            records.append({
                "name": label,
                "run_dir": str(run_dir) if run_dir is not None else None,
                "rc": 0,
                "resumed_from": plan["resume_from"],
                "started_from": run.get("start_from"),
                **verdict,
            })
            print(f"[run_queue] {label}: already done, skipped", file=sys.stderr)
            _write_result(result, path, started, records)
            if not verdict["ok"]:
                # 账本说跑完了、盘上的证据却说没有：下一条该照跑，让整晚停在
                # 一条陈旧记录上没有任何好处。
                rc_all = 2
                print(f"[run_queue] {label}: the ledger says done, the disk disagrees",
                      file=sys.stderr)
            continue

        command = plan["command"]
        assert command is not None
        print(f"[run_queue] {label}{' (resuming)' if plan['action'] == 'resume' else ''}: "
              f"{' '.join(command[1:])}", file=sys.stderr)
        since = time.time()
        try:
            known = set(run_state.run_dirs())
        except OSError as exc:
            records.append({
                "name": label,
                "run_dir": None,
                "rc": None,
                "resumed_from": plan["resume_from"],
                "ok": False,
                "problems": [f"cannot read the logs directory: {exc}"],
                "last_checkpoint": None,
                "checkpoints": 0,
            })
            print(f"[run_queue] {label} cannot read the logs directory: {exc}",
                  file=sys.stderr)
            rc_all = 2
            stopped = True
            break
        try:
            rc = spawn(command)
        except OSError as exc:
            records.append({
                "name": label,
                "run_dir": None,
                "rc": None,
                "resumed_from": plan["resume_from"],
                "ok": False,
                "problems": [f"cannot start long_run: {exc}"],
                "last_checkpoint": None,
                "checkpoints": 0,
            })
            print(f"[run_queue] {label} cannot start: {exc}", file=sys.stderr)
            rc_all = 2
            stopped = True
            break
        if rc < 0:
            # `subprocess.call` 在子进程被信号杀掉时返回负值。原样交给
            # `SystemExit` 会让 shell 看到 247，而 `long_run` 已经把这种情况换成了
            # 128+n——队列不能在这里把它变回不可读。
            rc = 128 - rc
        # 排除这一条之前就存在的目录：`newest_run_dir` 有 2 秒容差，
        # 上一条刚写完盘的目录会被认成这一条留下的证据。
        #
        # 这一步也是磁盘操作：`run_dirs()` 的结果在 `stat` 之间可能被人删掉（
        # `health_jumper` 的磁盘水位就是让操作者去清理旧 run 的）。不包住它，
        # 整个循环会带着 traceback 退出，末尾那次 `_write_result` 一次都不跑——
        # 正是「被打断时要靠一份汇总」的反面。
        try:
            run_dir = run_state.newest_run_dir(since, exclude=known)
            verdict = verify_run(run_dir, run["iterations"], DEFAULT_SAVE_INTERVAL, rc,
                                 start_iteration=start_iteration)
        except OSError as exc:
            records.append({
                "name": label,
                "run_dir": None,
                "rc": rc,
                "resumed_from": plan["resume_from"],
                "ok": False,
                "problems": [f"cannot read the evidence for this run: {exc}"],
                "last_checkpoint": None,
                "checkpoints": 0,
            })
            print(f"[run_queue] {label} cannot read the evidence: {exc}", file=sys.stderr)
            rc_all = rc or 2
            stopped = True
            break
        records.append({
            "name": label,
            "run_dir": str(run_dir) if run_dir is not None else None,
            "rc": rc,
            "resumed_from": plan["resume_from"],
            "started_from": run.get("start_from"),
            **verdict,
        })
        # 每跑完一条就落盘：中断随时可能来，第一条的成绩不能在第二条开始后才存在。
        _write_result(result, path, started, records)
        if not verdict["ok"]:
            print(f"[run_queue] {label} did not verify: {'; '.join(verdict['problems'])}",
                  file=sys.stderr)
            rc_all = rc or 2
            break
    _write_result(result, path, started, records)
    outcome = ("all runs verified" if rc_all == 0
               else "stopped early" if stopped
               else "finished with problems")
    print(f"[run_queue] {outcome} — {len(records)}/{len(runs)} — {result}", file=sys.stderr)
    return rc_all


if __name__ == "__main__":
    raise SystemExit(main())
