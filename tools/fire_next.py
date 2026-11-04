"""第一段跑完了没有、该不该点火、点火做什么——一个命令回答。

巡检每 30 分钟醒一次，它不该重新推导这件事，也不该自己算阈值。所以判据全部
在这里，agent 只负责在它说 `ready` 的时候调用它、把它的输出抄进日志行。

判据（全部必须成立）：
  * 第一段账本存在，且条目数与清单一致——账本比清单短意味着还有条目没跑
  * 每一条 `ok` 为真——没跑完的那条不续，也不许替它做主
  * 没有 `run_queue.py` 进程在跑——否则这一段还没结束
  * 第二段还没点过火——清单或账本已在，就是已经点过，重复点火只会得到 rc=75

`--fire` 才真的生成清单并启动队列；不带它是干跑，只打印判断。两种模式的输出
是同一份 JSON（`fired` 在两种模式下都在，值不同），所以日志行里抄哪一次都成立。

点火是**启动**，不是等待：队列要跑几小时，而这是一次决定。子进程进自己的会话，
调用它的巡检进程退出或被超时 kill 都不会把队列带走。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import next_stage
import run_queue
import run_state


# 会攥住 GPU 与锁的作业。`run_queue.py` 是队列本身；`long_run.py` 是单臂的外壳，
# `train.py` 是它的子进程；`play.py` 是评测。只查队列等于只看门卫不看房间——
# 2026-11-01 23:44 就是这样：没有队列，`train.py` 正攥着 16 GB 在跑。
LONG_JOB_PATTERN = "run_queue.py|long_run.py|train.py|play.py"


def work_is_running(*, run=subprocess.run) -> bool:
    """有没有会抢卡或抢锁的作业在跑。查的是进程表，不是账本——账本是跑完才写的。"""
    try:
        finished = run(["pgrep", "-af", LONG_JOB_PATTERN], capture_output=True, text=True)
    except OSError:
        # 查不出来就当它在跑：宁可这一轮不动，也不要和正在跑的那一段撞车。
        return True
    # pgrep 的 1 是「一条都没匹配到」，2 及以上是它自己出错。出错时 stdout 也是
    # 空的，按「没有队列」读就是和正在跑的那一段撞车。
    if getattr(finished, "returncode", 0) not in (0, 1):
        return True
    return bool(finished.stdout.strip())


def judge(*, ledger: Path, manifest: Path, extend_manifest: Path, extend_ledger: Path,
          running: bool) -> dict:
    """该不该点火。返回的 `reasons` 是给日志行抄的原话。"""
    reasons: list[str] = []

    if running:
        reasons.append("a training, evaluation or queue process is still alive")
    if not ledger.exists():
        reasons.append(f"no ledger at {ledger} — the first stage has not finished")
    else:
        try:
            recorded = next_stage.load_ledger(ledger)
            source = run_queue.load_manifest(manifest)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            # 读不了是「不知道」，不是「跑完了」。两种都不点，但原因要说出来原样。
            reasons.append(f"cannot read the stage: {exc}")
        else:
            # 账本键就是 `run_queue` 给这条 run 的名字；无名条目的名字是拼出来的，
            # 按 `run["name"]`（None）比对会谎报「清单里没有它」。
            labels = [run_queue.label_of(run) for run in source]
            missing = [label for label in labels if label not in recorded]
            unfinished = [name for name, record in recorded.items() if not record.get("ok")]
            if missing:
                reasons.append(f"the ledger has no record for {', '.join(missing)}")
            if unfinished:
                reasons.append(f"did not finish: {', '.join(sorted(unfinished))}")
    if extend_manifest.exists() or extend_ledger.exists():
        hit = extend_manifest if extend_manifest.exists() else extend_ledger
        reasons.append(f"already fired: {hit.name} exists")

    return {"ready": not reasons, "reasons": reasons}


def _spawn(command: list[str]):
    """启动队列就返回。独立会话：父进程被超时 kill 时用它当挡箭牌。

    三条流一律接 `DEVNULL`。继承的话队列会攥着调用者的 stdout 活上几个小时，
    `subprocess.run(capture_output=True)`、`$(...)` 和 ssh 会话都会一直读到它退出
    ——那正是这次修复要消掉的「等」。队列自己的记录在账本与 `status.json` 里。
    """
    return subprocess.Popen(
        command, cwd=run_state.REPO, start_new_session=True,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _path(value: Path) -> Path:
    """相对路径按仓库根解析：巡检在别处启动时，cwd 不是仓库根。

    静默读一个不存在的账本会被判成「第一段还没跑完」——结论没错，原因错了。
    """
    return value if value.is_absolute() else run_state.REPO / value


def _refuse(verdict: dict, reason: str) -> int:
    verdict["ready"] = False
    verdict["reasons"] = [*verdict["reasons"], reason]
    print(json.dumps(verdict, ensure_ascii=False))
    return 3


def main(argv: list[str] | None = None, *, run=subprocess.run, spawn=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--extend-manifest", type=Path, required=True)
    parser.add_argument("--extend-ledger", type=Path, required=True)
    parser.add_argument("--only", required=True)
    parser.add_argument("--extra", type=int, required=True)
    parser.add_argument("--fire", action="store_true", help="真的生成清单并点火")
    args = parser.parse_args(argv)
    args.ledger = _path(args.ledger)
    args.manifest = _path(args.manifest)
    args.extend_manifest = _path(args.extend_manifest)
    args.extend_ledger = _path(args.extend_ledger)
    spawn = spawn or _spawn

    verdict = judge(
        ledger=args.ledger, manifest=args.manifest,
        extend_manifest=args.extend_manifest, extend_ledger=args.extend_ledger,
        running=work_is_running(run=run))

    if not verdict["ready"]:
        print(json.dumps(verdict, ensure_ascii=False))
        return 3

    only = [name.strip() for name in args.only.split(",") if name.strip()]
    try:
        source = run_queue.load_manifest(args.manifest)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _refuse(verdict, f"cannot read {args.manifest}: {exc}")
    known = [run_queue.label_of(entry) for entry in source]
    unknown = [name for name in only if name not in known]
    if not only or unknown:
        # 名单里有一个打错的字，生成的就是一份少了那一条的清单——静默少跑一条，
        # 比什么都不跑更贵。
        named = ", ".join(unknown) if unknown else "nothing"
        return _refuse(verdict, f"--only names {named} — not in {args.manifest}")
    # 与队列的**真实**下限一致：`run_queue.load_manifest` 拒收 `iterations <= 50`，
    # 放 `--extra 20` 过去只会让清单生成成功、队列启动即退 2，而那时「第二段点过火」
    # 已经记下，下一段永远被 `already fired` 挡住。
    if args.extra <= run_queue.DEFAULT_SAVE_INTERVAL:
        return _refuse(
            verdict,
            f"--extra {args.extra} — below the {run_queue.DEFAULT_SAVE_INTERVAL}-iteration "
            f"save interval, so the queue would refuse the manifest it makes")

    # 生成、写盘、点火全部包住：`NotDone`/`ValueError`/`OSError` 在这里都是「拒绝」
    # 而不是 traceback，判决的契约是一个 JSON。
    try:
        extended = next_stage.extend_manifest(
            next_stage.load_ledger(args.ledger), args.manifest, only=only,
            extra=args.extra)
    except (next_stage.NotDone, next_stage.Unknown, ValueError, OSError) as exc:
        return _refuse(verdict, f"cannot build the next stage: {exc}")
    verdict["runs"] = [run["name"] for run in extended["runs"]]
    # 干跑与点火带同一组键。点火那条要等 `spawn` 真的返回才算数（见下）。
    verdict["fired"] = False

    if not args.fire:
        print(json.dumps(verdict, ensure_ascii=False))
        return 0

    # 顺序是「写 → 点火 → 失败则删」：
    # * 先写：`spawn` 是非阻塞 `Popen`，子进程一起来就读这份清单，晚一步它读到的是
    #   「文件不存在」，退 2，而清单随后才出现——一次失败的启动会永久锁死后续每一轮。
    # * 失败则删：这个文件名正是下一段「已经点过火」的守卫，留着它就是留着证据。
    # 路径算术，不会抛 `OSError`：放在 `try` 外面，清理分支才不需要靠
    # `UnboundLocalError` 去猜「它绑定过没有」，也才不会顺手吞掉真正的拼写错误。
    staging = args.extend_manifest.with_suffix(args.extend_manifest.suffix + ".part")
    try:
        args.extend_manifest.parent.mkdir(parents=True, exist_ok=True)
        # 先写临时文件再 `os.replace`：`write_text` 是「截断 → 写」，`ENOSPC`/`EIO`
        # 会留下一个**存在但内容不全**的清单，而它的文件名正是下一段「已经点过火」
        # 的守卫——写失败和点火失败会以同一个死锁收场。
        staging.write_text(json.dumps(extended, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
        os.replace(staging, args.extend_manifest)
    except OSError as exc:
        leftover = ""
        try:
            staging.unlink()
        except FileNotFoundError:
            # 最常见的一种：`mkdir` 自己就失败了，临时文件从来没写出去过。
            pass
        except OSError as cleanup:
            # 与下面 spawn 的失败分支同一条规矩：删不掉就说出来。
            leftover = f"; and {staging.name} is still on disk ({cleanup})"
        return _refuse(verdict, f"cannot write the next stage: {exc}{leftover}")

    try:
        spawn([sys.executable, "tools/run_queue.py", str(args.extend_manifest)])
    except OSError as exc:
        # `fired` 这时还是 false：判决里说「点过火」而实际没起来，是这一批修复
        # 专门要消掉的那种谎。
        verdict["fired"] = False
        leftover = ""
        try:
            args.extend_manifest.unlink()
        except OSError as cleanup:
            # 删不掉就得说出来：守卫文件还在，之后每一轮都会答 `already fired`，
            # 而操作者只看到「启动失败」——那正是这次修复要消掉的那种沉默。
            leftover = (f"; and {args.extend_manifest.name} is still on disk "
                        f"({cleanup}), so the next check will say `already fired`")
        return _refuse(verdict, f"cannot start the queue: {exc}{leftover}")
    verdict["fired"] = True
    print(json.dumps(verdict, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
