"""从上一轮的账本里接出下一段：只续那些确实跑完的条目。

夜间挂机时没人看曲线，所以「要不要续」必须由产物回答，不能由人回答。判据只有
一条：这一条的账本记录 `ok` 为真、且它自己的 run 目录里确实有 checkpoint。缺一
项就拒绝——从一条没跑完的 run 上续训，得到的是一段没人知道起点的训练。

新的一段用新名字（`<原名>-ext`），因为 `run_queue` 的账本按名字索引，同名会被
当成「已经跑完」直接跳过，于是续训永远发不出去。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import run_queue


class NotDone(Exception):
    """这一条不能续：没跑完，或它的产物不在盘上。"""


class Unknown(Exception):
    """清单里没有这个名字——打错一个字，就是一次从零开始的训练。"""


def load_ledger(path: Path) -> dict[str, dict]:
    """上一次的汇总，按名字索引。"""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        # 带上文件名：这条消息会被巡检原样抄进日志行，没有路径的
        # "Expecting value: line 1 column 1" 什么也说明不了。
        raise ValueError(f"{path}: {exc}") from exc
    runs = payload.get("runs") if isinstance(payload, dict) else None
    if not isinstance(runs, list):
        raise ValueError(f"{path}: no runs list")
    return {
        record["name"]: record
        for record in runs
        if isinstance(record, dict) and isinstance(record.get("name"), str)
    }


def _entry(record: dict, run: dict, extra: int) -> dict:
    name = record["name"]
    if not record.get("ok"):
        raise NotDone(f"{name}: the ledger says it did not finish ({record.get('problems')})")
    run_dir = record.get("run_dir")
    if not run_dir or not Path(run_dir).is_dir():
        raise NotDone(f"{name}: no run directory at {run_dir}")
    if not list(Path(run_dir).glob("model_*.pt")):
        raise NotDone(f"{name}: {run_dir} carries no checkpoint")

    entry = {
        "name": f"{name}-ext",
        "task": run["task"],
        "seed": run["seed"],
        "iterations": extra,
        "num_envs": run["num_envs"],
        "start_from": str(run_dir),
    }
    if run.get("args"):
        entry["args"] = list(run["args"])
    return entry


def extend_manifest(ledger: dict[str, dict], manifest: Path, *,
                    only: list[str], extra: int) -> dict:
    """把 `only` 里点名的条目接成下一段，顺序按 `only` 给的顺序。"""
    # 用队列自己的校验与命名：`run_queue` 不要求 `name`，无名条目的账本键是它拼
    # 出来的标签。按 `run["name"]` 索引，无名条目会挂在 `None` 上，于是「清单里
    # 明明有它」变成「清单里没有它」。
    # 队列会拒收 `iterations <= save_interval` 的条目，这一层不收的话，生成的清单
    # 要到队列启动时才被拒——而那时 `fire_next` 已经把「第二段点过火」记下去了。
    if extra <= run_queue.DEFAULT_SAVE_INTERVAL:
        raise ValueError(
            f"extra must exceed the {run_queue.DEFAULT_SAVE_INTERVAL}-iteration save "
            f"interval, not {extra}")
    runs = {run_queue.label_of(run): run for run in run_queue.load_manifest(manifest)}
    entries = []
    for name in only:
        if name not in runs:
            raise Unknown(f"{name}: not in {manifest}")
        record = ledger.get(name)
        if record is None:
            raise NotDone(f"{name}: no ledger record — it never ran")
        entries.append(_entry(record, runs[name], extra))
    return {"runs": entries}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--only", required=True, help="逗号分隔的条目名，按给的顺序续")
    parser.add_argument("--extra", type=int, required=True,
                        help="每条再跑多少 iteration（必须大于保存间隔 50）")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    only = [name.strip() for name in args.only.split(",") if name.strip()]
    kept, refused = [], []
    try:
        ledger = load_ledger(args.ledger) if args.ledger.exists() else {}
        for name in only:
            try:
                kept += extend_manifest(ledger, args.manifest, only=[name], extra=args.extra)["runs"]
            except NotDone as exc:
                refused.append(str(exc))
    except Unknown as exc:
        print(f"[next_stage] {exc}", file=sys.stderr)
        return 3
    except (OSError, ValueError) as exc:
        print(f"[next_stage] {exc}", file=sys.stderr)
        return 3

    for reason in refused:
        print(f"[next_stage] refused: {reason}", file=sys.stderr)
    if not kept:
        print("[next_stage] nothing to extend", file=sys.stderr)
        return 3

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"runs": kept}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for run in kept:
        print(f"[next_stage] {run['name']} from {run['start_from']} for {run['iterations']} iterations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
