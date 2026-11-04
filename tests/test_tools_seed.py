"""`mjrl.seed.set_seed` 的契约测试。

只承诺一件事：**同一 seed 下 torch / numpy / random 的随机流一致**。
**不承诺** warp 的仿真轨迹逐位复现——它的 kernel 不保证这一点，写在这里是为了
让后来者不会拿两个 run 的曲线逐点比较。

放在 `tests/` 而不是 `tools/`：`pyproject.toml` 的 `testpaths = ["tests"]` 与 CI
都只跑这个目录，放别处等于这份契约无人看守。
"""

import random

import numpy as np
import pytest
import torch

from mjrl.seed import set_seed


def test_torch_stream_is_reproducible():
    set_seed(42)
    a = torch.rand(3)
    set_seed(42)
    b = torch.rand(3)
    assert torch.equal(a, b)


def test_numpy_stream_is_reproducible():
    set_seed(7)
    a = np.random.rand(3)
    set_seed(7)
    b = np.random.rand(3)
    assert np.array_equal(a, b)


def test_random_stream_is_reproducible():
    set_seed(7)
    a = [random.random() for _ in range(3)]
    set_seed(7)
    b = [random.random() for _ in range(3)]
    assert a == b


def test_different_seeds_differ():
    set_seed(42)
    a = torch.rand(3)
    set_seed(43)
    b = torch.rand(3)
    assert not torch.equal(a, b)


@pytest.mark.parametrize("bad", [-1, 2**32, 2**32 + 1])
def test_seed_out_of_numpy_range_fails_loudly(bad):
    # `np.random.seed` 的合法区间是 0..2**32-1；越界必须在进入 NumPy 之前就报错，
    # 否则训练会崩在一个看起来与 seed 无关的 traceback 里。
    with pytest.raises(ValueError, match="between 0 and 4294967295"):
        set_seed(bad)


def test_seed_accepts_the_range_edges():
    set_seed(0)
    set_seed(2**32 - 1)


def test_the_cli_argument_refuses_what_numpy_would_raise_on():
    import argparse
    import pytest
    from mjrl.seed import seed_argument

    assert seed_argument("0") == 0
    assert seed_argument("42") == 42
    for bad in ("-1", "4294967296", "abc", ""):
        with pytest.raises(argparse.ArgumentTypeError):
            seed_argument(bad)
