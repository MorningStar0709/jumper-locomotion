import json
from pathlib import Path

import fire_next

NAMES = ["B-scratch-terrain5", "A-scratch-plane", "C-scratch-terrain10", "D-resume-plane"]


def a_run_dir(tmp_path, name):
    path = tmp_path / name
    path.mkdir(parents=True, exist_ok=True)
    for number in (0, 2999):
        (path / f"model_{number}.pt").write_bytes(b"")
    return path


def a_manifest(tmp_path, names=NAMES):
    path = tmp_path / "queue.json"
    path.write_text(json.dumps({"runs": [
        {"name": name, "task": "jumper.tripod", "seed": 2,
         "iterations": 3000, "num_envs": 4096}
        for name in names
    ]}), encoding="utf-8")
    return path


def a_ledger(tmp_path, names=NAMES, *, ok=True, drop=()):
    path = tmp_path / "queue.result.json"
    runs = [
        {"name": name, "ok": ok, "run_dir": str(a_run_dir(tmp_path, name)),
         "last_checkpoint": 2999, "rc": 0 if ok else 1}
        for name in names if name not in drop
    ]
    path.write_text(json.dumps({"runs": runs}), encoding="utf-8")
    return path


def judge(tmp_path, *, running=False, **kwargs):
    # 默认账本必须懒算：写成 `kwargs.pop("ledger", a_ledger(tmp_path))` 会在调用前
    # 就把文件写一遍，把用例自己准备的账本覆盖掉——那不是被测对象的错。
    ledger = kwargs.pop("ledger", None) or a_ledger(tmp_path)
    return fire_next.judge(
        ledger=ledger,
        manifest=a_manifest(tmp_path),
        extend_manifest=tmp_path / "extend.json",
        extend_ledger=tmp_path / "extend.result.json",
        running=running)


def test_a_finished_stage_is_ready(tmp_path):
    verdict = judge(tmp_path)

    assert verdict["ready"] is True
    assert verdict["reasons"] == []


def test_a_living_queue_is_not_done(tmp_path):
    verdict = judge(tmp_path, running=True)

    assert verdict["ready"] is False
    assert "still alive" in verdict["reasons"][0]


def test_no_ledger_means_the_stage_did_not_finish(tmp_path):
    verdict = judge(tmp_path, ledger=tmp_path / "nope.json")

    assert verdict["ready"] is False
    assert "has not finished" in verdict["reasons"][0]


def test_a_run_that_did_not_finish_holds_the_whole_stage(tmp_path):
    verdict = judge(tmp_path, ledger=a_ledger(tmp_path, ok=False))

    assert verdict["ready"] is False
    assert any("did not finish" in reason for reason in verdict["reasons"])


def test_a_short_ledger_is_not_a_finished_stage(tmp_path):
    verdict = judge(tmp_path, ledger=a_ledger(tmp_path, drop=("D-resume-plane",)))

    assert verdict["ready"] is False
    assert any("no record for D-resume-plane" in reason for reason in verdict["reasons"])


def test_firing_twice_is_refused(tmp_path):
    (tmp_path / "extend.json").write_text("{}", encoding="utf-8")

    verdict = judge(tmp_path)

    assert verdict["ready"] is False
    assert any("already fired" in reason for reason in verdict["reasons"])


def test_a_dry_run_says_ready_and_writes_nothing(tmp_path):
    calls = []

    rc = fire_next.main([
        "--ledger", str(a_ledger(tmp_path)), "--manifest", str(a_manifest(tmp_path)),
        "--extend-manifest", str(tmp_path / "extend.json"),
        "--extend-ledger", str(tmp_path / "extend.result.json"),
        "--only", "B-scratch-terrain5", "--extra", "7000"],
        run=lambda *a, **k: type("R", (), {"stdout": ""})(), spawn=calls.append)

    assert rc == 0
    assert not (tmp_path / "extend.json").exists()
    assert calls == []


def test_firing_writes_the_manifest_and_starts_one_queue(tmp_path):
    calls = []

    rc = fire_next.main([
        "--ledger", str(a_ledger(tmp_path)), "--manifest", str(a_manifest(tmp_path)),
        "--extend-manifest", str(tmp_path / "extend.json"),
        "--extend-ledger", str(tmp_path / "extend.result.json"),
        "--only", "B-scratch-terrain5", "--extra", "7000", "--fire"],
        run=lambda *a, **k: type("R", (), {"stdout": ""})(),
        spawn=lambda command: calls.append(command) or 0)

    assert rc == 0
    assert len(calls) == 1
    assert calls[0][-1].endswith("extend.json")
    written = json.loads((tmp_path / "extend.json").read_text(encoding="utf-8"))
    assert written["runs"][0]["name"] == "B-scratch-terrain5-ext"
    assert written["runs"][0]["iterations"] == 7000


def test_a_refusal_fires_nothing(tmp_path):
    calls = []

    rc = fire_next.main([
        "--ledger", str(a_ledger(tmp_path, ok=False)), "--manifest", str(a_manifest(tmp_path)),
        "--extend-manifest", str(tmp_path / "extend.json"),
        "--extend-ledger", str(tmp_path / "extend.result.json"),
        "--only", "B-scratch-terrain5", "--extra", "7000", "--fire"],
        run=lambda *a, **k: type("R", (), {"stdout": ""})(),
        spawn=lambda command: calls.append(command) or 0)

    assert rc == 3
    assert calls == []
    assert not (tmp_path / "extend.json").exists()


def test_a_dry_run_and_a_fire_carry_the_same_keys(tmp_path, capsys):
    """日志行里抄哪一次的都成立——多一个 `fired` 键就破坏了这一点。"""
    _main(tmp_path, fired=False)
    dry = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    _main(tmp_path, fired=True, spawn=lambda command: 0)
    real = json.loads(capsys.readouterr().out.strip().splitlines()[-1])

    assert set(dry) == set(real)


def test_the_already_fired_reason_names_the_file_that_exists(tmp_path):
    """清单不在盘上、只有账本时，日志里不该出现一个不存在的文件名。"""
    (tmp_path / "extend.result.json").write_text(json.dumps({"runs": []}), encoding="utf-8")

    verdict = judge(tmp_path)

    assert not verdict["ready"]
    assert any("extend.result.json" in reason for reason in verdict["reasons"])


def test_a_pgrep_that_errors_counts_as_a_living_queue():
    """pgrep 自己出错（rc 2）时 stdout 也是空的，读成「没有队列」就会撞车。"""
    class Finished:
        returncode = 2
        stdout = ""

    assert fire_next.work_is_running(run=lambda *a, **k: Finished())


class Alive:
    """一次 pgrep 的结果：rc 与匹配行。"""

    def __init__(self, stdout="", returncode=0):
        self.stdout = stdout
        self.returncode = returncode


def test_a_training_run_blocks_the_ignition_even_with_no_queue():
    """2026-11-01 23:44 的现场：没有 run_queue，可 train.py 攥着 16 GB 在跑。

    只看 run_queue 的判定会在这时回一句 ready，点火后撞上 `flock … rc=75`，
    半小时后再来一遍，永远如此。
    """
    alive = Alive("2121089 .venv/bin/python scripts/train.py --task jumper.tripod\n")

    assert fire_next.work_is_running(run=lambda *a, **k: alive)


def test_an_evaluation_blocks_the_ignition_too():
    """评测与训练抢同一块卡，`play.py` 在跑时点火同样是撞车。"""
    alive = Alive("2080687 .venv/bin/python scripts/play.py --headless --steps 6000\n")

    assert fire_next.work_is_running(run=lambda *a, **k: alive)


def test_an_idle_machine_is_not_running():
    """pgrep 的 1 是「一条都没匹配到」——这才是空场，可以点火。"""
    idle = Alive("", returncode=1)

    assert not fire_next.work_is_running(run=lambda *a, **k: idle)


def test_the_pattern_names_every_long_job_that_holds_the_gpu():
    """漏掉一个进程名，就等于对那一类作业失明；这里把名单钉死。"""
    seen = []

    def spy(command, **kwargs):
        seen.append(command)
        return Alive("", returncode=1)

    fire_next.work_is_running(run=spy)

    pattern = " ".join(seen[0])
    for name in ("run_queue.py", "train.py", "play.py", "long_run.py"):
        assert name in pattern


def test_a_broken_ledger_is_a_refusal_not_a_traceback(tmp_path):
    broken = tmp_path / "queue.result.json"
    broken.write_text("{not json", encoding="utf-8")

    verdict = judge(tmp_path, ledger=broken)

    assert not verdict["ready"]
    assert any("queue.result.json" in reason for reason in verdict["reasons"])


def test_firing_does_not_wait_for_the_queue(tmp_path):
    """点火是「启动」，不是「跑完」：阻塞在这里等于把整晚押在一次调用里。

    而且同进程组时，父进程被超时 kill 会把队列一起带走。
    """
    waited = []

    class Proc:
        pid = 4242

        def wait(self, *args, **kwargs):
            waited.append(1)

    def spawn(command):
        return Proc()

    code = _main(tmp_path, fired=True, spawn=spawn)

    assert code == 0
    assert not waited


def test_the_detached_queue_does_not_hold_the_callers_output(monkeypatch):
    """继承 stdout 的队列会让 `capture_output=True` 一直读到它退出。"""
    seen = {}

    class Spawned:
        pass

    def fake_popen(command, **kwargs):
        seen["command"] = command
        seen.update(kwargs)
        return Spawned()

    monkeypatch.setattr(fire_next.subprocess, "Popen", fake_popen)

    fire_next._spawn(["python", "tools/run_queue.py", "x.json"])

    assert seen["command"] == ["python", "tools/run_queue.py", "x.json"]
    assert seen["stdin"] is fire_next.subprocess.DEVNULL
    assert seen["stdout"] is fire_next.subprocess.DEVNULL
    assert seen["stderr"] is fire_next.subprocess.DEVNULL
    assert seen["start_new_session"] is True


def test_an_only_name_that_is_not_in_the_manifest_is_refused(tmp_path):
    out = tmp_path / "extend.json"

    code = _main(tmp_path, fired=False, only="Nope")

    assert code == 3
    assert not out.exists()


def test_an_extra_budget_below_one_iteration_is_refused(tmp_path):
    code = _main(tmp_path, fired=False, extra=0)

    assert code == 3


def test_an_unnamed_entry_is_matched_by_the_label_the_queue_would_give_it(tmp_path):
    """`run_queue` 不要求 `name`：无名条目的账本键是 `label_of()` 拼出来的。

    按 `run["name"]`（None）去比对，会报「清单里没有它」——而它明明在清单里。
    """
    import run_queue

    unnamed = {"task": "jumper.tripod", "seed": 2, "iterations": 3000, "num_envs": 4096}
    label = run_queue.label_of(unnamed)
    manifest = tmp_path / "queue.json"
    manifest.write_text(json.dumps({"runs": [unnamed]}), encoding="utf-8")
    ledger = tmp_path / "queue.result.json"
    ledger.write_text(json.dumps({"runs": [
        {"name": label, "ok": True, "run_dir": str(a_run_dir(tmp_path, "unnamed")),
         "last_checkpoint": 2999, "rc": 0}]}), encoding="utf-8")

    verdict = fire_next.judge(ledger=ledger, manifest=manifest,
                              extend_manifest=tmp_path / "extend.json",
                              extend_ledger=tmp_path / "extend.result.json",
                              running=False)

    assert verdict["ready"], verdict["reasons"]


def _main(tmp_path, *, fired, spawn=lambda command: 0, only="B-scratch-terrain5", extra=7000):
    """`spawn` 默认是记录器，不是真的 `_spawn`：这个文件里的任何用例都不该点火。

    这条默认值是修复一次真实事故来的——早先 `fired=True` 的用例没注入，`_spawn`
    真的把 `run_queue` 拉起来了，队列又拉起 `long_run`，测试跑一次就在 GPU 上
    留一个崩掉的训练目录。
    """
    argv = ["--ledger", str(a_ledger(tmp_path)), "--manifest", str(a_manifest(tmp_path)),
            "--extend-manifest", str(tmp_path / "extend.json"),
            "--extend-ledger", str(tmp_path / "extend.result.json"),
            "--only", only, "--extra", str(extra)]
    if fired:
        argv.append("--fire")
    return fire_next.main(argv, run=lambda *a, **k: type("F", (), {"returncode": 1, "stdout": ""})(),
                          spawn=spawn)


def test_an_extra_below_the_save_interval_is_refused(tmp_path, capsys):
    """`--extra 20` 会生成一份队列必然拒收的清单。

    `run_queue` 要求 `iterations > save_interval(50)`。放过去的话：清单生成成功、
    队列启动即退 2，而「第二段点过火」已经记下，下一段永远被 `already fired` 挡住。
    """
    assert _main(tmp_path, fired=True, extra=20) == 3
    verdict = json.loads(capsys.readouterr().out)
    assert verdict["ready"] is False
    # 拒绝必须来自这一层：`next_stage` 也有一条同样的下限，两边文案一样的话
    # 这条用例在下一层被删掉之后仍然会绿。
    assert "below the 50-iteration save interval" in " ".join(verdict["reasons"])
    assert not (tmp_path / "extend.json").exists()


def test_a_spawn_that_fails_leaves_no_evidence_behind(tmp_path, capsys):
    """点火失败时清单不能留在盘上——它就是「已经点过火」的证据。"""
    def refuse(command):
        raise OSError("no such file or directory")

    assert _main(tmp_path, fired=True, spawn=refuse) == 3
    verdict = json.loads(capsys.readouterr().out)
    assert verdict["ready"] is False
    assert "cannot start the queue" in " ".join(verdict["reasons"])
    assert not (tmp_path / "extend.json").exists()


def test_the_manifest_is_on_disk_before_the_queue_is_started(tmp_path, capsys):
    """子进程一起来就读清单。

    先点火后落盘的顺序下，`load_manifest` 读到的是「文件不存在」，退 2，而清单随后
    才出现——一次失败的启动会永久锁死后续每一轮。
    """
    seen = {}

    def look(command):
        path = Path(command[-1])
        seen["exists"] = path.exists()
        seen["runs"] = len(json.loads(path.read_text(encoding="utf-8"))["runs"])
        return 0

    assert _main(tmp_path, fired=True, spawn=look) == 0
    assert seen == {"exists": True, "runs": 1}
    assert json.loads(capsys.readouterr().out)["fired"] is True


def test_a_write_that_fails_leaves_no_guard_behind(tmp_path, monkeypatch, capsys):
    """`write_text` 是「截断 → 写」。

    `ENOSPC`/`EIO` 会留下一个存在但内容不全的清单，而它正是下一段「已经点过火」的
    守卫——写失败与点火失败会以同一个死锁收场。所以要么整份出现，要么不出现。
    """
    def boom(src, dst):
        raise OSError("no space left on device")

    monkeypatch.setattr(fire_next.os, "replace", boom)
    assert _main(tmp_path, fired=True) == 3
    verdict = json.loads(capsys.readouterr().out)
    assert verdict["ready"] is False
    assert "cannot write the next stage" in " ".join(verdict["reasons"])
    assert not (tmp_path / "extend.json").exists()
    assert not list(tmp_path.glob("extend.json.part"))


def test_a_rollback_that_fails_says_the_guard_is_still_there(tmp_path, monkeypatch, capsys):
    """删不掉守卫文件时不能说「什么都没有」——之后每一轮都会答 `already fired`。"""
    def refuse(command):
        raise OSError("no such file or directory")

    def stuck(self):
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "unlink", stuck, raising=False)
    assert _main(tmp_path, fired=True, spawn=refuse) == 3
    verdict = json.loads(capsys.readouterr().out)
    reason = " ".join(verdict["reasons"])
    assert "cannot start the queue" in reason
    assert "still on disk" in reason
    assert verdict["fired"] is False


def test_a_directory_that_cannot_be_made_is_refused_without_claiming_residue(
        tmp_path, monkeypatch, capsys):
    """`mkdir` 失败时临时文件从来没存在过。

    清理分支若一律说「还留在盘上」，操作者会去找一个不存在的文件；而当初的写法
    靠 `UnboundLocalError` 兜住这条路径，顺带会吞掉真正的拼写错误。
    """
    real_mkdir = Path.mkdir

    def no_mkdir(self, *args, **kwargs):
        # 只打断首页清单的那一次 `mkdir`：全局打断会连运行目录的创建一起挡住，
        # 测到的就不是这个分支了。
        if self == tmp_path:
            raise OSError("read-only file system")
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", no_mkdir, raising=False)
    assert _main(tmp_path, fired=True) == 3
    verdict = json.loads(capsys.readouterr().out)
    reason = " ".join(verdict["reasons"])
    assert "cannot write the next stage" in reason
    assert "still on disk" not in reason
