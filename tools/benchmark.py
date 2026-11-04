"""Run one manifest of evaluations through `scripts/play.py` and write one score file.

The tool exists so that two measurements taken weeks apart are comparable: the
same command, the same fields, one file. It does not evaluate anything itself —
`--evaluation-out` already does that — it only drives and collects.

    .venv/bin/python tools/benchmark.py benchmarks/tripod.json

Writes `<manifest>.score.json` next to the manifest, and one raw evaluation per
checkpoint into `<manifest>.score.d/`.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

import gpu_lock

REPO = Path(__file__).resolve().parent.parent
DEFAULT_STEPS = 1000
AXES = ("vx", "vy", "yaw_rate")
#: 一个 case 能带的东西。拼错的键必须报错：`"arg": ["--delay"]` 静默忽略，
#: 出来的是一份看起来正常的「无延迟」数字。
ALLOWED_CASE_KEYS = frozenset({"name", "command", "checkpoints", "set", "args", "steps"})
#: `build_play_command` 自己摆的旗标。case 的 `args` 再写一遍不报错，只是 argparse 取
#: 最后一次——命令行里真正跑的是 case 的值，而清单里记的是工具的值，两边对不上。
RESERVED_FLAGS = frozenset({
    "--task", "--headless", "--steps", "--command", "--set", "--seed",
    "--evaluation-out", "--checkpoint",
})


def load_manifest(path: Path) -> list[dict]:
    """Read a manifest and return its cases, each carrying the task and the budget."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: the manifest must be an object")

    task = payload.get("task")
    if not isinstance(task, str) or not task:
        raise ValueError(f"{path}: names no task")

    steps = payload.get("steps", DEFAULT_STEPS)
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError(f"{path}: steps must be a positive integer")

    seed = payload.get("seed")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)
                             or not 0 <= seed <= 0xFFFF_FFFF):
        raise ValueError(f"{path}: seed must be between 0 and 4294967295")

    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{path}: has no cases")

    loaded = []
    seen_names: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"{path}: case {index} is not an object")
        name = case.get("name")
        if name is None:
            name = f"case{index}"
        if not isinstance(name, str) or not name:
            # 会被 `name in seen_names` 撞成 `TypeError`（列表不可哈希），而它同时也是
            # 证据文件名的一部分。
            raise ValueError(f"{path}: case {index} has a name that is not a string")
        if name in seen_names:
            # 两个同名 case 会写同一个 `.score.d/` 文件，后一条盖掉前一条的证据。
            raise ValueError(f"{path}: case name {name!r} is used twice")
        seen_names.add(name)
        case_steps = case.get("steps", steps)
        if (isinstance(case_steps, bool) or not isinstance(case_steps, int)
                or case_steps <= 0):
            raise ValueError(f"{path}: case {name} has steps that are not a positive integer")
        unknown = set(case) - ALLOWED_CASE_KEYS
        if unknown:
            raise ValueError(
                f"{path}: case {name} has keys this queue cannot honour: "
                f"{', '.join(sorted(unknown))}")
        command = case.get("command")
        if (not isinstance(command, list) or len(command) != 3
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       for value in command)):
            raise ValueError(f"{path}: case {name} has no command of three numbers")
        checkpoints = case.get("checkpoints")
        if not isinstance(checkpoints, list) or not checkpoints:
            raise ValueError(f"{path}: case {name} has no checkpoints")
        for checkpoint in checkpoints:
            if not isinstance(checkpoint, dict) or "path" not in checkpoint:
                raise ValueError(f"{path}: case {name} has a checkpoint without a path")
            unknown_checkpoint = set(checkpoint) - {"path", "seed"}
            if unknown_checkpoint:
                raise ValueError(
                    f"{path}: case {name} has a checkpoint with keys this tool cannot "
                    f"honour: {', '.join(sorted(unknown_checkpoint))}")
            if not isinstance(checkpoint["path"], str) or not checkpoint["path"]:
                raise ValueError(
                    f"{path}: case {name} has a checkpoint whose path is not a path")
            checkpoint_seed = checkpoint.get("seed", 0)
            if (isinstance(checkpoint_seed, bool) or not isinstance(checkpoint_seed, int)
                    or not 0 <= checkpoint_seed <= 0xFFFF_FFFF):
                raise ValueError(
                    f"{path}: case {name} has a checkpoint seed that is not 0..4294967295")
        assignments = case.get("set", [])
        if not isinstance(assignments, list) or not all(
            isinstance(item, str) and "=" in item for item in assignments
        ):
            raise ValueError(f"{path}: case {name} has a set that is not a list of PATH=VALUE")
        # 任务自己的开关（`--delay` 这类），原样交给 play.py。只收长选项：
        # 短选项和裸值会和 play.py 自己的参数混在一起，说不清是谁放的。
        flags = case.get("args", [])
        if not isinstance(flags, list) or any(
                not isinstance(item, str) or not item.startswith("--") for item in flags):
            raise ValueError(
                f"{path}: case {name} has args that are not long flags such as --delay")
        # 按前缀比对：`scripts/play.py` 的 `argparse` 没关 `allow_abbrev`，`--ste=50`
        # 会被展开成 `--steps`，而完整拼写比对看不见它。
        #
        # 只跳过一个多余的连字符：`--t` 已经能展开成 `--task`（`play.py` 里没有第二个
        # `--t*`，`--tensorboard`/`--tb-*` 是训练侧的），`--t=jumper.five_foot` 会
        # 悄悄替换掉本工具写在命令前面的 `--task`，而评分文件记的是清单里的任务名。
        written = {item.split("=", 1)[0] for item in flags}
        clashing = sorted(
            name for name in written
            if name in RESERVED_FLAGS or (
                len(name) > 2 and any(reserved.startswith(name) for reserved in RESERVED_FLAGS)))
        if clashing:
            raise ValueError(
                f"{path}: case {name} sets {', '.join(clashing)}, which this tool decides; "
                f"the ledger would disagree with what actually ran")
        loaded.append(
            {
                "name": name,
                "task": task,
                "steps": case_steps,
                "command": [float(value) for value in command],
                "checkpoints": checkpoints,
                "set": list(assignments),
                "args": list(flags),
                "seed": seed,
            }
        )
    return loaded


def build_play_command(case: dict, checkpoint: dict, *, out: Path) -> list[str]:
    """The one command every evaluation runs, whatever the case."""
    command = [sys.executable, "scripts/play.py", "--task", case["task"], "--headless"]
    command += ["--steps", str(case["steps"])]
    command += ["--command", *(str(value) for value in case["command"])]
    for assignment in case.get("set", ()):
        command += ["--set", assignment]
    command += [str(flag) for flag in case.get("args", ())]
    if case.get("seed") is not None:
        command += ["--seed", str(case["seed"])]
    command += ["--evaluation-out", str(out), "--checkpoint", str(checkpoint["path"])]
    return command


def read_result(path: Path) -> dict:
    """Turn one `--evaluation-out` file into the fields the score file keeps."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{Path(path).name} is not an evaluation object")
    axes = payload.get("axes") or list(AXES)
    if not isinstance(axes, list) or list(axes) != list(AXES):
        # 轴对不上就写不出可比的分；`{"axes": ["vx","vy","wz"]}` 会让 `summarise`
        # 产出空的 worst/mean，而 `main()` 照样返回 0。
        raise ValueError(f"{Path(path).name} reports axes {axes!r}, not {list(AXES)!r}")
    mae = payload.get("mean_absolute_error")
    actual = payload.get("actual_velocity_mean")
    if not mae or not isinstance(mae, list) or len(mae) != len(axes):
        # `play.py` 写 `null` 的意思是「一个有效样本都没有」——那不是 0 误差。
        raise ValueError(f"{Path(path).name} measured nothing (mean_absolute_error is {mae!r})")
    if actual is None or actual == []:
        # `play.py` 在没有有效样本时写 `null`：那确实是「没有实测值」。
        actual = []
    elif not isinstance(actual, list) or len(actual) != len(axes):
        # 但写得出东西、长度却对不上，是另一回事——静默丢掉会让一份坏掉的评测
        # 看起来只是「没测到」。
        raise ValueError(
            f"{Path(path).name} reports {actual!r} as actual_velocity_mean, "
            f"which is not {len(axes)} numbers")
    return {
        "mae": dict(zip(axes, mae)),
        "actual": dict(zip(axes, actual)),
        "resets": payload.get("episode_resets"),
    }


def summarise(rows: Iterable[dict]) -> dict:
    """Worst and average per axis — the worst case is what a policy is judged on."""
    rows = list(rows)
    worst: dict[str, float] = {}
    mean: dict[str, float] = {}
    for axis in AXES:
        values = [row["mae"][axis] for row in rows if axis in row.get("mae", {})]
        if values:
            worst[axis] = max(values)
            mean[axis] = sum(values) / len(values)
    return {"cases": rows, "worst_mae": worst, "mean_mae": mean}


def _default_run(command: list[str]) -> int:
    return subprocess.call(command, cwd=str(REPO))


def main(argv: list[str] | None = None, *, run: Callable[[list[str]], int] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("[benchmark] usage: benchmark.py <manifest.json>", file=sys.stderr)
        return 2

    manifest = Path(argv[0])
    try:
        cases = load_manifest(manifest)
    except (OSError, ValueError) as exc:
        print(f"[benchmark] {exc}", file=sys.stderr)
        return 2

    run = _default_run if run is None else run
    # 子进程用 `cwd=REPO`（`_default_run`），父进程的 cwd 可能在任何地方。相对路径
    # 传下去，`play.py` 会写到自己那侧的 `benchmarks/…`，父进程于是在每一行上看到
    # `play wrote no evaluation file`。
    out_dir = manifest.with_suffix(".score.d").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 评测与训练抢同一张卡。训练已经按 run 取锁；评测不取锁就是同一个卡上的
    # 第二个写入者——两边的数都不可信，而且不会报错，只会变慢、变飘。
    # 取锁与评测分开：`except OSError` 只包住取锁这一步。包住整个 `with` 会把
    # `_evaluate` 里的 `OSError`（play.py 不见了、fd 用尽、写 stderr 失败）报成
    # 「显卡锁不可用」，把一次评测失败变成一条运维故障。
    lock = gpu_lock.hold(gpu_lock.DEFAULT_LOCK, role=f"benchmark {manifest.name}")
    try:
        lock.__enter__()
    except gpu_lock.GpuBusy as busy:
        print(f"[benchmark] {busy}", file=sys.stderr)
        return gpu_lock.BUSY_RC
    except OSError as exc:
        # 71 是锁文件本身不可用（目录不存在、权限、文件系统只读、路径上是软链）。
        # 把它和 75 分开：「卡被占着」等一会儿就好，「锁坏了」等多久都不会好。
        print(f"[benchmark] the GPU lock is unavailable: {exc}", file=sys.stderr)
        return gpu_lock.UNAVAILABLE_RC
    try:
        rows = _evaluate(cases, run, out_dir)
    finally:
        lock.__exit__(None, None, None)

    summary = summarise(rows)
    score_path = manifest.with_suffix(".score.json").resolve()
    score_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[benchmark] {score_path}", file=sys.stderr)
    return 0 if all("error" not in row for row in rows) else 1


def _evaluate(cases: list[dict], run: Callable[[list[str]], int], out_dir: Path) -> list[dict]:
    rows = []
    for case in cases:
        for checkpoint in case["checkpoints"]:
            seed = checkpoint.get("seed", 0)
            checkpoint_path = Path(checkpoint["path"])
            # 文件名带路径摘要：`runA/ckpt/model_3000.pt` 与 `runB/ckpt/model_3000.pt`
            # 同名同种子，只用目录名或只用 stem 都会让后一条覆盖前一条的证据。
            digest = hashlib.sha1(str(checkpoint_path).encode("utf-8")).hexdigest()[:8]
            out = out_dir / (
                f"{case['name']}-seed{seed}-{checkpoint_path.stem}-{digest}.json")
            row = {
                "case": case["name"],
                "checkpoint": str(checkpoint["path"]),
                "seed": seed,
                "replay_seed": case.get("seed"),
                "steps": case["steps"],
                "command": list(case["command"]),
                "set": list(case.get("set", ())),
                "args": list(case.get("args", ())),
            }
            rc = run(build_play_command(case, checkpoint, out=out))
            if rc != 0:
                row["error"] = f"play exited {rc}"
            elif not out.exists():
                row["error"] = "play wrote no evaluation file"
            else:
                try:
                    row.update(read_result(out))
                except (OSError, ValueError) as exc:
                    row["error"] = str(exc)
            rows.append(row)
            print(f"[benchmark] {row['case']} seed{seed} {row.get('error') or row['mae']}", file=sys.stderr)
    return rows


if __name__ == "__main__":
    sys.exit(main())
