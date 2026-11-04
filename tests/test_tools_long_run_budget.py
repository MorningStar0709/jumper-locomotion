"""起跑前的显存预算：`tools/long_run.py` 该在训练启动**之前**说「装不下」。

标定数据（2026-10-29 实测，`jumper.tripod` + warp/cuda，`nvidia-smi memory.used`
峰值，每档 20 iterations）：

| num_envs | 峰值 MiB |
|---|---|
| 512  | 1765 |
| 2048 | 5195 |
| 4096 | 9437 |

拟合取**保守**的一侧：`960 + 2.08 × num_envs`，在 2048/4096 上高估 18/44 MiB，
在 512 上高估 260 MiB。宁愿早一点喊停，也不要跑到第 2000 iter 才 OOM——
那时 checkpoint 已经写过几十个，人得从废墟里判断哪一个是好的。

估算是**线性外推**，不是物理模型：它只管「这张卡装不装得下」，不管显存碎片、
别的进程的波动。所以预算总留一段 reserve（默认 1024 MiB）。
"""

import json
import os

import pytest

import long_run


# 实测点。允许 20% 的保守余量，但不许低于实测——低估就是放它去 OOM。
@pytest.mark.parametrize("num_envs, measured", [(512, 1765), (2048, 5195), (4096, 9437)])
def test_the_estimate_covers_every_measured_point(num_envs, measured):
    estimate = long_run.estimate_vram_mib(num_envs)
    assert measured <= estimate <= measured * 1.20


def test_the_estimate_grows_with_the_environment_count():
    counts = [64, 512, 2048, 4096, 8192]
    estimates = [long_run.estimate_vram_mib(n) for n in counts]
    assert estimates == sorted(estimates)
    assert len(set(estimates)) == len(counts)


def test_no_environments_still_cost_the_overhead():
    # 进程本身、CUDA context、warp/mujoco 的模块：不随 num_envs 走，但不为 0。
    assert long_run.estimate_vram_mib(0) == long_run.VRAM_OVERHEAD_MIB
    assert long_run.VRAM_OVERHEAD_MIB > 0


def test_a_negative_count_is_a_usage_error():
    with pytest.raises(ValueError):
        long_run.estimate_vram_mib(-1)


def test_it_refuses_when_the_estimate_does_not_fit():
    verdict = long_run.fit_verdict(16384, free_mib=24463, reserve_mib=1024)
    assert verdict["ok"] is False
    assert verdict["reason"]


def test_the_suggestion_actually_fits():
    verdict = long_run.fit_verdict(16384, free_mib=24463, reserve_mib=1024)
    suggested = verdict["suggested"]
    assert suggested is not None and suggested < 16384
    assert long_run.estimate_vram_mib(suggested) <= verdict["budget_mib"]
    # 再往上加一档就该装不下——否则「建议值」是个偷懒的减半。
    assert long_run.estimate_vram_mib(suggested + 1) > verdict["budget_mib"]


def test_the_reserve_is_what_decides_it():
    # 同一个请求，reserve 为 0 时装得下，留出 8 GiB 就装不下。
    roomy = long_run.fit_verdict(2048, free_mib=10000, reserve_mib=0)
    tight = long_run.fit_verdict(2048, free_mib=10000, reserve_mib=8192)
    assert roomy["ok"] is True
    assert tight["ok"] is False


def test_over_budget_can_be_allowed_on_purpose():
    verdict = long_run.fit_verdict(16384, free_mib=24463, reserve_mib=1024, allow_over_budget=True)
    assert verdict["ok"] is True
    assert verdict["over_budget"] is True


def test_an_unreadable_gpu_does_not_block_the_run():
    # 采样失败不是拒绝的理由：这张卡上可能没有 nvidia-smi，或者换了个驱动版本。
    # 拒绝要基于一个数，不是基于「我没读到数」。
    verdict = long_run.fit_verdict(4096, free_mib=None, reserve_mib=1024)
    assert verdict["ok"] is True
    assert verdict["budget_mib"] is None


def test_the_wrapper_takes_its_budget_flags_out_of_the_passthrough():
    own, passthrough = long_run._own_args(
        ["--task", "jumper.tripod", "--reserve-mib", "2048", "--allow-over-budget",
         "--num_envs", "4096"]
    )
    assert own["reserve_mib"] == 2048
    assert own["allow_over_budget"] is True
    assert passthrough == ["--task", "jumper.tripod", "--num_envs", "4096"]


@pytest.mark.parametrize("bad", ["abc", "-1", "nan"])
def test_a_nonsense_reserve_is_a_usage_error(bad):
    with pytest.raises(SystemExit):
        long_run._own_args(["--reserve-mib", bad, "--task", "jumper.tripod"])


def test_the_environment_sets_the_environment_count(monkeypatch):
    """No `--num_envs`: the backend reads `MJRL_NUM_ENVS`, so the budget must too."""
    monkeypatch.setenv("MJRL_NUM_ENVS", "1024")
    assert long_run._num_envs([]) == 1024


def test_each_backend_has_its_own_default_size(monkeypatch):
    """With nothing in the environment, the answer comes from the backend's own resolver.

    `resolve()` reads `MJRL_BACKEND` and `MJRL_DEVICE` from `os.environ`, so "nothing
    in the environment" has to be arranged, not assumed: a shell that exports
    `MJRL_DEVICE=cuda:0` would otherwise decide what this asserts.
    """
    for name in ("MJRL_BACKEND", "MJRL_DEVICE"):
        monkeypatch.delenv(name, raising=False)
    assert long_run._backend_default_envs("warp") == 4096
    assert long_run._backend_default_envs("native") == 64


def test_a_backend_that_cannot_be_asked_says_so(monkeypatch, capsys):
    """The fallback is a documented behaviour, not a place to hide a broken import."""
    import sys

    import mjrl.backend.resolve  # noqa: F401 - `mjrl.backend.resolve` is shadowed by the function
    resolve_mod = sys.modules["mjrl.backend.resolve"]

    def boom(**kwargs):
        raise resolve_mod.BackendUnavailable("no card in this process")

    monkeypatch.setattr(resolve_mod, "resolve", boom)
    assert long_run._backend_default_envs("warp") == 4096
    err = capsys.readouterr().err
    assert "cannot ask the backend" in err
    assert "4096" in err


def test_an_unknown_backend_falls_back_to_the_largest_default():
    """A budget that is too small lets an OOM through; too large only refuses a run."""
    assert long_run._backend_default_envs("nonsense") == 4096


def test_the_repository_env_file_is_read_before_the_budget_is_computed(monkeypatch):
    """`.env` is what `scripts/train.py` will read, so the wrapper has to load it too.

    The expectation is read back from the environment *after* the loader ran, so
    `.env.local` (which wins over `.env`) cannot make this fail for a reason that
    has nothing to do with the code. The process environment is swapped for a copy
    because `_load_env` writes into `os.environ` and never restores it -- leaving
    that leak in place makes every later test in the session depend on file order.
    """
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.delenv("MJRL_NUM_ENVS", raising=False)
    long_run._load_env()
    if "MJRL_NUM_ENVS" not in os.environ:
        pytest.skip("neither .env nor .env.local sets MJRL_NUM_ENVS")
    assert long_run._num_envs(["--backend", "native"]) == int(os.environ["MJRL_NUM_ENVS"])


def test_a_dangling_num_envs_is_a_usage_error():
    """A flag with no value would silently fall back to the default and mis-budget."""
    with pytest.raises(SystemExit):
        long_run._num_envs(["--task", "jumper.tripod", "--num_envs"])


def test_a_negative_num_envs_says_it_is_too_small():
    """`-5` is a value, not a missing one: the message has to name the real problem."""
    with pytest.raises(SystemExit) as caught:
        long_run._num_envs(["--num_envs", "-5"], env={})
    assert "1 or more" in str(caught.value)


def test_the_refusal_hint_names_a_flag_that_exists(monkeypatch, capsys):
    """建议里写的开关必须真的存在：`--num-envs` 会被 argparse 拒掉。"""
    # 可用预算 476 MiB，连固定开销都不够，于是没有可建议的 num_envs。
    monkeypatch.setattr(long_run, "free_vram_mib", lambda: 1500)
    own = {"reserve_mib": 1024, "allow_over_budget": False, "interval": 5.0}
    rc = long_run._supervise(own, ["--num_envs", "65536"], ["true"], None)
    assert rc == 2
    err = capsys.readouterr().err
    assert "--num-envs" not in err


def _sample_of(**fields):
    return {"limit_w": None, "mem_mib": 1.0, "util": 2.0, "power_w": 3.0, **fields}


def test_the_power_limit_range_follows_the_samples(monkeypatch):
    """`status.json` 的 `limit_min_w`/`limit_max_w` 是判读「当时卡在什么档」的唯一依据。"""
    seq = iter([175.0, 135.0, 150.0])
    monkeypatch.setattr(long_run, "sample_gpu", lambda: _sample_of(limit_w=next(seq)))
    status = {"gpu": {"limit_min_w": None, "limit_max_w": None}}
    for _ in range(3):
        long_run._sample(status)
    assert status["gpu"]["limit_min_w"] == 135.0
    assert status["gpu"]["limit_max_w"] == 175.0


def test_an_unreadable_sample_leaves_the_range_empty(monkeypatch):
    """读不到就是 `null`，不能写成 0——那会让平直的 0 W 看起来像一次真实的测量。"""
    monkeypatch.setattr(long_run, "sample_gpu", lambda: _sample_of(limit_w=None))
    status = {"gpu": {"limit_min_w": None, "limit_max_w": None}}
    long_run._sample(status)
    assert status["gpu"]["limit_min_w"] is None
    assert status["gpu"]["limit_max_w"] is None


@pytest.mark.parametrize("bad", [0, -1.0, float("nan"), float("inf"), True, "2.08"])
def test_a_calibration_entry_that_is_not_a_positive_number_is_ignored(tmp_path, bad):
    """坏数字要么崩在 `// per_env`，要么静默低估；两种都不该发生。"""
    table = tmp_path / "calibration.json"
    table.write_text(json.dumps({"jumper.tripod": {
        "overhead_mib": 960, "per_env_mib": bad}}), encoding="utf-8")

    assert long_run.load_calibrations(table) == {}
