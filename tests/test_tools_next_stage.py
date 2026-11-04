import json

import pytest

import next_stage


def a_finished_run(tmp_path, *, name="B-scratch-terrain5", ok=True,
                   checkpoints=(0, 2999), run_dir=None):
    run_dir = run_dir or tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    for number in checkpoints:
        (run_dir / f"model_{number}.pt").write_bytes(b"")
    return {
        "name": name,
        "ok": ok,
        "run_dir": str(run_dir),
        "last_checkpoint": checkpoints[-1] if checkpoints else None,
        "rc": 0 if ok else 1,
    }


def a_ledger(tmp_path, *runs):
    path = tmp_path / "queue.result.json"
    path.write_text(json.dumps({"manifest": "queue.json", "runs": list(runs)}), encoding="utf-8")
    return path


def a_manifest(tmp_path, *runs, name="queue.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"runs": list(runs)}), encoding="utf-8")
    return path


def test_a_finished_run_becomes_a_continuation(tmp_path):
    ledger = a_ledger(tmp_path, a_finished_run(tmp_path))
    manifest = a_manifest(tmp_path, {
        "name": "B-scratch-terrain5", "task": "jumper.tripod", "seed": 2,
        "iterations": 3000, "num_envs": 4096,
        "args": ["--terrain", "--terrain-roughness-mm=5"],
    })

    extended = next_stage.extend_manifest(
        next_stage.load_ledger(ledger), manifest, only=["B-scratch-terrain5"], extra=7000)

    assert len(extended["runs"]) == 1
    run = extended["runs"][0]
    assert run["name"] == "B-scratch-terrain5-ext"
    assert run["iterations"] == 7000
    assert run["start_from"] == str(tmp_path / "run")
    assert run["args"] == ["--terrain", "--terrain-roughness-mm=5"]
    assert run["seed"] == 2


def test_a_run_that_did_not_finish_is_refused(tmp_path):
    ledger = a_ledger(tmp_path, a_finished_run(tmp_path, ok=False))
    manifest = a_manifest(tmp_path, {
        "name": "B-scratch-terrain5", "task": "jumper.tripod", "seed": 2,
        "iterations": 3000, "num_envs": 4096,
    })

    try:
        next_stage.extend_manifest(
            next_stage.load_ledger(ledger), manifest, only=["B-scratch-terrain5"], extra=7000)
    except next_stage.NotDone as exc:
        assert "B-scratch-terrain5" in str(exc)
    else:
        raise AssertionError("a run that did not finish must not be extended")


def test_a_name_outside_the_manifest_is_refused(tmp_path):
    ledger = a_ledger(tmp_path, a_finished_run(tmp_path))
    manifest = a_manifest(tmp_path, {
        "name": "A-scratch-plane", "task": "jumper.tripod", "seed": 2,
        "iterations": 3000, "num_envs": 4096,
    })

    try:
        next_stage.extend_manifest(
            next_stage.load_ledger(ledger), manifest, only=["B-scratch-terrain5"], extra=7000)
    except next_stage.Unknown as exc:
        assert "B-scratch-terrain5" in str(exc)
    else:
        raise AssertionError("a name the manifest does not carry is a typo, not an extension")


def test_a_run_without_a_directory_is_refused(tmp_path):
    run = a_finished_run(tmp_path)
    run["run_dir"] = str(tmp_path / "gone")
    ledger = a_ledger(tmp_path, run)
    manifest = a_manifest(tmp_path, {
        "name": "B-scratch-terrain5", "task": "jumper.tripod", "seed": 2,
        "iterations": 3000, "num_envs": 4096,
    })

    try:
        next_stage.extend_manifest(
            next_stage.load_ledger(ledger), manifest, only=["B-scratch-terrain5"], extra=7000)
    except next_stage.NotDone as exc:
        assert "gone" in str(exc)
    else:
        raise AssertionError("a run directory that is not there cannot be a starting point")


def test_the_extension_keeps_the_order_asked_for(tmp_path):
    first = a_finished_run(tmp_path, name="B", run_dir=tmp_path / "b")
    second = a_finished_run(tmp_path, name="C", run_dir=tmp_path / "c")
    ledger = a_ledger(tmp_path, first, second)
    manifest = a_manifest(
        tmp_path,
        {"name": "C", "task": "jumper.tripod", "seed": 2, "iterations": 3000, "num_envs": 4096},
        {"name": "B", "task": "jumper.tripod", "seed": 2, "iterations": 3000, "num_envs": 4096},
    )

    extended = next_stage.extend_manifest(
        next_stage.load_ledger(ledger), manifest, only=["B", "C"], extra=7000)

    assert [run["name"] for run in extended["runs"]] == ["B-ext", "C-ext"]


def test_the_manifest_it_writes_is_one_the_queue_accepts(tmp_path):
    ledger = a_ledger(tmp_path, a_finished_run(tmp_path))
    manifest = a_manifest(tmp_path, {
        "name": "B-scratch-terrain5", "task": "jumper.tripod", "seed": 2,
        "iterations": 3000, "num_envs": 4096,
    })
    out = tmp_path / "extend.json"

    rc = next_stage.main([
        str(ledger), "--manifest", str(manifest),
        "--only", "B-scratch-terrain5", "--extra", "7000", "--output", str(out)])

    assert rc == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["runs"][0]["start_from"] == str(tmp_path / "run")
    assert written["runs"][0]["name"] == "B-scratch-terrain5-ext"


def test_one_that_did_not_finish_does_not_block_the_others(tmp_path):
    good = a_finished_run(tmp_path, name="B", run_dir=tmp_path / "b")
    bad = a_finished_run(tmp_path, name="C", ok=False, run_dir=tmp_path / "c")
    ledger = a_ledger(tmp_path, good, bad)
    manifest = a_manifest(
        tmp_path,
        {"name": "B", "task": "jumper.tripod", "seed": 2, "iterations": 3000, "num_envs": 4096},
        {"name": "C", "task": "jumper.tripod", "seed": 2, "iterations": 3000, "num_envs": 4096},
    )
    out = tmp_path / "extend.json"

    rc = next_stage.main([
        str(ledger), "--manifest", str(manifest),
        "--only", "B,C", "--extra", "7000", "--output", str(out)])

    assert rc == 0
    assert [run["name"] for run in json.loads(out.read_text(encoding="utf-8"))["runs"]] == ["B-ext"]


def test_nothing_extendable_is_a_refusal_not_an_empty_manifest(tmp_path):
    ledger = a_ledger(tmp_path, a_finished_run(tmp_path, ok=False))
    manifest = a_manifest(tmp_path, {
        "name": "B-scratch-terrain5", "task": "jumper.tripod", "seed": 2,
        "iterations": 3000, "num_envs": 4096,
    })
    out = tmp_path / "extend.json"

    rc = next_stage.main([
        str(ledger), "--manifest", str(manifest),
        "--only", "B-scratch-terrain5", "--extra", "7000", "--output", str(out)])

    assert rc == 3
    assert not out.exists()


def test_a_run_directory_without_a_checkpoint_is_refused(tmp_path):
    """目录在、但里面没有 `model_*.pt`：训练从那里续不起来。"""
    run_dir = tmp_path / "empty"
    run_dir.mkdir()
    ledger = tmp_path / "l.json"
    ledger.write_text(json.dumps({"runs": [
        {"name": "A", "ok": True, "run_dir": str(run_dir), "rc": 0}]}), encoding="utf-8")
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"runs": [
        {"name": "A", "task": "t", "seed": 0, "iterations": 3000, "num_envs": 8}]}),
        encoding="utf-8")

    with pytest.raises(next_stage.NotDone) as caught:
        next_stage.extend_manifest(next_stage.load_ledger(ledger), manifest,
                                   only=["A"], extra=100)

    assert str(run_dir) in str(caught.value)


def test_the_written_entry_passes_the_queues_own_validation(tmp_path):
    """不只看两个字段：这份文件是喂给 `run_queue` 的，就让队列的校验器判它。"""
    import run_queue

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "model_2999.pt").write_bytes(b"")
    ledger = tmp_path / "l.json"
    ledger.write_text(json.dumps({"runs": [
        {"name": "A", "ok": True, "run_dir": str(run_dir), "rc": 0}]}), encoding="utf-8")
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"runs": [
        {"name": "A", "task": "t", "seed": 0, "iterations": 3000, "num_envs": 8,
         "args": ["--terrain"]}]}), encoding="utf-8")

    written = next_stage.extend_manifest(
        next_stage.load_ledger(ledger), manifest, only=["A"], extra=7000)

    out = tmp_path / "out.json"
    out.write_text(json.dumps(written), encoding="utf-8")
    assert run_queue.load_manifest(out) == written["runs"]


def test_an_extension_shorter_than_the_save_interval_is_refused(tmp_path):
    """队列要求 `iterations > 50`；这一层不收，清单要到队列启动时才被拒。"""
    with pytest.raises(ValueError, match="save interval"):
        next_stage.extend_manifest(
            next_stage.load_ledger(a_ledger(tmp_path)), a_manifest(tmp_path),
            only=["B-scratch-terrain5"], extra=20)
