"""`jumper.tripod` on generated ground: four switches, all off by default.

The terrain ladder climbs in steps
of five millimetres, so the flag has to take millimetres and nothing else: a
config that reads "roughness 5" and means five centimetres would turn L1 into L4
without saying so.

**The default has to stay flat ground.** `tasks/jumper/common/velocity_env.py`
drops the generator on purpose -- a flat-trained policy resumed onto generated
terrain at a mean level of 2.47 fell within 15 control steps -- and every existing
result was measured on that flat ground. A default that quietly grew terrain
would invalidate them and look like a regression instead.
"""

from __future__ import annotations

import argparse

import pytest
import torch

tasks = pytest.importorskip("tasks", reason="the terrain switches live in the task registry")


def _cfg(**task_args):
    """Build the config the way `scripts/train.py` does -- flags as `task_args`.

    `load_env_cfg` forwards `task_args` into the task's own `env_cfg`, which is
    the contract this file exists to check: a flag the parser accepts but the
    registry drops would otherwise look like a working knob.
    """
    return tasks.load_env_cfg("jumper.tripod", task_args=task_args)


def _generator(cfg):
    terrain = cfg.scene.terrain
    assert terrain is not None
    return terrain.terrain_generator


def test_the_default_is_still_flat_ground() -> None:
    cfg = _cfg()

    assert cfg.scene.terrain.terrain_type == "plane"
    assert cfg.scene.terrain.terrain_generator is None
    # The curriculum still starts on the flattest row, which is what makes it a
    # curriculum rather than a random scatter.
    assert cfg.scene.terrain.max_init_terrain_level == 0


def test_terrain_with_no_roughness_is_a_generator_that_is_flat() -> None:
    """L0 of the ladder: prove the pipe works before adding a millimetre."""
    cfg = _cfg(terrain=True, terrain_roughness_mm=0.0)
    generator = _generator(cfg)

    assert cfg.scene.terrain.terrain_type == "generator"
    assert generator is not None
    assert generator.curriculum is True
    assert cfg.scene.terrain.max_init_terrain_level == 0
    # Zero roughness must be the flat preset, not a zero-amplitude rough field:
    # the latter divides by its quantisation step and raises from four frames away.
    assert "flat" in generator.sub_terrains
    assert "rough" not in generator.sub_terrains


def test_a_single_row_ladder_installs_no_terrain_curriculum() -> None:
    """One row has nothing to promote along, so the term should not be there.

    The ladder fixes the amplitude per run, which makes the grid a single row and
    leaves `TerrainLevels` with no level to move between: measured over 500
    iterations it sat at `level 0` with `level_max 0` and `promoted` 0.0254, so
    the only thing it could do was run. mjlab's distance rule must stay dropped
    either way -- see `velocity_env_cfg`, which explains why the two cannot be
    left to race.
    """
    cfg = _cfg(terrain=True, terrain_roughness_mm=5.0)

    assert "terrain" not in cfg.curriculum
    assert "terrain_levels" not in cfg.curriculum


@pytest.mark.parametrize(("millimetres", "metres"), [(5.0, 0.005), (10.0, 0.010)])
def test_roughness_is_read_as_millimetres(millimetres: float, metres: float) -> None:
    cfg = _cfg(terrain=True, terrain_roughness_mm=millimetres)
    rough = _generator(cfg).sub_terrains["rough"]

    # The *range* is metres, and it starts at the base plane: sampling from
    # `(a, a)` would ask for one height and get a raised plate.
    assert rough.noise_range == (0.0, metres)
    # One millimetre of vertical resolution, or "five millimetres" collapses
    # into two levels and the ladder measures nothing between L1 and L2.
    assert rough.vertical_scale == 0.001
    assert rough.noise_step == 0.001


def test_steps_are_read_as_metres() -> None:
    """`foot_target_height` is 25 mm, so 20 mm is already near the ceiling."""
    cfg = _cfg(terrain=True, terrain_steps_m=0.02)

    assert _generator(cfg).sub_terrains["stairs"].step_height_range == (0.02, 0.02)


def test_a_staircase_is_the_only_column_in_the_grid() -> None:
    """A grid with a flat column beside the stairs measures the flat column.

    `curriculum=True` gives each sub-terrain exactly one column, and
    `_proportional_counts` hands the single env of a replay column 0 -- the first
    inserted, which was `flat`. So `--terrain-steps-m 1/2/3 cm` was a flat-ground
    replay with a staircase somewhere else in the world: measured vx MAE
    0.0320 / 0.0318 / 0.0318 against a flat replay's 0.0317, three amplitudes in a
    row, because nothing about the robot changed. The staircase has to be the
    whole grid for the number to be about the staircase.
    """
    generator = _generator(_cfg(terrain=True, terrain_steps_m=0.02))

    assert set(generator.sub_terrains) == {"stairs"}


def test_the_staircase_climbs_outward() -> None:
    """`pyramid_stairs` is a descent, and the robot walks away from its spawn.

    `BoxPyramidStairsTerrainCfg.function` stacks its boxes so that the k-th ring's
    top is `(k + 1) * step_height`, with the highest ring at the centre, and it
    puts the spawn origin on that centre platform: `(num_steps + 1) *
    step_height` **above** grade. A robot spawned there and commanded forward
    walks off the platform and down 70 risers, so `--terrain-steps-m 0.02` was a
    one-and-a-half-metre drop test with a staircase painted on it.

    `BoxInvertedPyramidStairsTerrainCfg` is the same stair with the ring order
    reversed -- spawn at `-(num_steps + 1) * step_height`, every tread a riser
    higher as the radius grows -- which is the climb the ladder asks for. This
    test pins both halves of that: the class, and the geometry it produces.
    """
    import mujoco
    import numpy as np
    from mjlab.terrains.primitive_terrains import (
        BoxInvertedPyramidStairsTerrainCfg,
    )
    from mjlab.terrains.terrain_generator import TerrainGenerator

    riser = 0.02
    generator_cfg = _generator(_cfg(terrain=True, terrain_steps_m=riser))

    assert isinstance(
        generator_cfg.sub_terrains["stairs"], BoxInvertedPyramidStairsTerrainCfg
    )

    spec = mujoco.MjSpec()
    generator = TerrainGenerator(generator_cfg)
    generator.compile(spec)
    treads = sorted(
        (
            float(np.hypot(geom.pos[0], geom.pos[1])),
            float(geom.pos[2] + geom.size[2]),
        )
        for geom in spec.body("terrain").geoms
        if geom.type == mujoco.mjtGeom.mjGEOM_BOX
    )
    tops = [top for _, top in treads]

    # Walking outward gains height and never loses it, and every height is an
    # exact multiple of the riser apart -- one ring, one riser. The outermost box
    # is the flat border frame at grade, hence the top of the ladder being 0.
    assert all(later >= earlier - 1e-9 for earlier, later in zip(tops, tops[1:]))
    assert tops[-1] == pytest.approx(0.0, abs=1e-9)
    treads = sorted({round(top, 6) for top in tops})
    assert {round(b - a, 6) for a, b in zip(treads, treads[1:])} == {round(riser, 6)}

    # The robot spawns on the platform at the bottom of the stair, below grade,
    # which is the lowest surface in the grid.
    origin_z = float(generator.terrain_origins[0, 0, 2])
    assert origin_z <= -riser
    assert min(tops) == pytest.approx(origin_z, abs=1e-9)


def test_the_climbing_staircase_is_not_painted_the_robot_red() -> None:
    """爬的那半张网格不能和机器人同色。

    机器人本体是品牌红（`_MUJOCO_RED`，0.90/0.30/0.30），而
    `BoxInvertedPyramidStairsTerrainCfg` 原本也用同一个红做爬升坡道：1080p 特写里
    红腿压在红台阶上，脚踩在哪一级看不出来，而「能爬多高」正是这条线唯一的结论。

    `brand_ramp` 的蓝／红分工是为了在回放里一眼分清「下」和「上」，但上坡这一半的
    观众永远是这台红色机器人，所以地面该退到背景去。下行的那半张仍是蓝的——场景里
    没有别的东西是蓝的，它没有这个问题。
    """
    import mujoco
    import numpy as np
    from mjlab.terrains.terrain_generator import TerrainGenerator

    generator = TerrainGenerator(_generator(_cfg(terrain=True, terrain_steps_m=0.02)))
    spec = mujoco.MjSpec()
    generator.compile(spec)

    rgbs = np.array(
        [
            tuple(geom.rgba[:3])
            for geom in spec.body("terrain").geoms
            if geom.type == mujoco.mjtGeom.mjGEOM_BOX
        ]
    )
    assert len(rgbs) > 10
    # How far red stands above the brighter of the other two channels. The robot's
    # red scores 0.60; anything under 0.25 reads as a neutral surface.
    redness = rgbs[:, 0] - np.maximum(rgbs[:, 1], rgbs[:, 2])
    assert float(redness.max()) < 0.25


@pytest.mark.parametrize("bad_seed", [-1, 2**32])
def test_a_terrain_seed_the_rng_cannot_take_is_a_usage_error(bad_seed: int) -> None:
    """`np.random.default_rng` takes only a non-negative 32-bit seed.

    Left to it, `--terrain-seed -1` passes every check here and dies inside
    `TerrainGenerator.__init__` -- at env construction, one layer below the flag,
    where the message says nothing about the switch that caused it. The two
    neighbouring terrain numbers are checked at the flag; this one should be too.
    """
    with pytest.raises(ValueError, match="terrain_seed"):
        _cfg(terrain=True, terrain_seed=bad_seed)


def test_a_mixture_puts_the_staircase_first_and_the_bumps_second() -> None:
    """训练用混合地面：一半机器人站台阶，一半站起伏。

    这一格问的是「这个策略能不能同时站住两种地面」，而单表面的两条线已经各自答完
    了（2 cm 台阶赢，5 mm 连续起伏是渐近收益）。混合只是把两个分布放进同一张网格。

    次序是契约的一部分：`curriculum=True` 给每个子地形一列，`_proportional_counts`
    从第 0 列开始分配，所以列 0 是台阶。单环境回放因此只会遇到台阶——这也是下面
    `test_a_mixture_cannot_be_replayed` 拒绝回放的原因，不是顺手排的序。
    """
    from mjlab.terrains.heightfield_terrains import HfRandomUniformTerrainCfg
    from mjlab.terrains.primitive_terrains import (
        BoxInvertedPyramidStairsTerrainCfg,
    )

    generator = _generator(
        _cfg(terrain=True, terrain_roughness_mm=5.0, terrain_steps_m=0.02)
    )

    assert list(generator.sub_terrains) == ["stairs", "rough"]
    assert isinstance(generator.sub_terrains["stairs"], BoxInvertedPyramidStairsTerrainCfg)
    assert isinstance(generator.sub_terrains["rough"], HfRandomUniformTerrainCfg)
    # 两列等权：4096 个环境按 1:1 分，谁也不比谁多一个数量级。
    assert generator.sub_terrains["stairs"].proportion == pytest.approx(
        generator.sub_terrains["rough"].proportion
    )


def test_a_share_moves_the_split() -> None:
    """混合地面的比例要能给：1:1 只是默认，不是唯一。

    3 cm 台阶 + 5 mm 起伏那一臂的代价落在两块连续地面上（铺装 +19.3 %、5 mm +49 %），
    一个可能的来源是"台阶那一列吃掉了半个训练量"。把台阶的比例降到 1/4 能在不改其它
    任何东西的前提下检验这个解释——所以需要一个参数，而不是再写一套生成器。
    """
    generator = _generator(
        _cfg(
            terrain=True,
            terrain_roughness_mm=5.0,
            terrain_steps_m=0.02,
            terrain_stairs_share=0.25,
        )
    )

    assert generator.sub_terrains["stairs"].proportion == pytest.approx(0.25)
    assert generator.sub_terrains["rough"].proportion == pytest.approx(0.75)


def test_a_share_without_both_surfaces_is_a_usage_error() -> None:
    """只给一种地面时，比例没有意义，别让它悄悄当作没写。

    单表面网格整个都铺同一种地形，`proportion` 只在两列之间有含义。若允许把
    `--terrain-stairs-share 0.25` 和单个开关一起给，得到的仍是一整格台阶，而命令行
    读起来像"四分之一台阶"——这正是那种事后没人会回头查的静默错配。
    """
    with pytest.raises(ValueError, match="share"):
        _cfg(terrain=True, terrain_steps_m=0.02, terrain_stairs_share=0.25)


@pytest.mark.parametrize(
    "share", [-0.1, 0.0, 1.0, 1.5, float("nan"), float("inf")], ids=repr
)
def test_a_share_outside_the_open_interval_is_refused(share: float) -> None:
    """0 或 1 不是"几乎全给一边"，是"一边一个环境都没有"。

    `_proportional_counts` 保证每个桶至少一个环境，所以 0 和 1 不会崩，但训练分布会
    变成只有一个列、另一个列只挂着一个摆设环境——那和单表面网格的差别只有一个噪声源，
    而标签却写着混合。

    区间外的值和 nan/inf 一起测，因为命令行能吃下它们（`float("nan")` 解析成功）：
    写成 `share <= 0.0 or share >= 1.0` 能过 0 和 1，却把 nan 放进 `proportion=`。
    """
    with pytest.raises(ValueError, match="share"):
        _cfg(
            terrain=True,
            terrain_roughness_mm=5.0,
            terrain_steps_m=0.02,
            terrain_stairs_share=share,
        )


def test_a_share_without_the_switch_is_a_usage_error() -> None:
    """没有 --terrain 时比例没有落点，别让它当作没写。

    这是 `test_a_size_without_the_switch_is_a_usage_error` 的同一条：命令行写着
    "四分之一台阶"，建出来的却是一块平地，而且事后没有任何东西会指出这一点。
    """
    with pytest.raises(ValueError, match="terrain_stairs_share"):
        _cfg(terrain_stairs_share=0.25)


def test_a_mixture_cannot_be_replayed() -> None:
    """混合地面是训练配置，回放它只会量到台阶那一列。

    `play.py` 起一个环境，`_proportional_counts(1, [0.5, 0.5])` 返回 `[1, 0]`——它站
    在列 0，也就是台阶上。跑出来的数会写着「混合地面」，量的却是纯台阶。所以评测必须
    分开跑两种纯地面，各用各的开关；混合配置回放直接拒。
    """
    with pytest.raises(ValueError, match="mixture"):
        tasks.load_env_cfg(
            "jumper.tripod",
            play=True,
            task_args=dict(
                terrain=True, terrain_roughness_mm=5.0, terrain_steps_m=0.02
            ),
        )


def test_the_ground_is_the_same_for_the_same_terrain_seed() -> None:
    """A terrain number that cannot be reproduced is not a number.

    `TerrainGeneratorCfg.seed` defaults to `None`, and `rl/mjlab/terrains/
    terrain_generator.py` then draws `np.random.randint(0, 10000)`: every run
    generates a different 40 m surface. Measured cost of that on
    `jumper.tripod` (2026-10-31, 107 replays): the same checkpoint, the same
    amplitude, only the surface changed -- vx MAE spread 0.0244-0.0415 on a
    *fixed paved* floor and wider once roughness arrived, so a rung's number
    could not be compared with the rung above it.

    The default is a fixed number rather than `None` on purpose. The ladder
    reads a single row at a single amplitude, so a run already sees exactly one
    surface; pinning it costs no diversity and buys comparability between L1 and
    L2. `--terrain-seed` is how a caller asks for a different one deliberately.

    What pinning buys is a *field*, not a repeatable number -- and the field
    itself is asserted by `test_the_same_seed_builds_the_same_heightfield`, not
    by a digest quoted here. Replaying the same pinned field still is not
    bit-identical: three flat replays with `--seed 0` give vx MAE 0.0383 / 0.0377
    / 0.0381, and at 10 mm the same field gave 0.055 and 0.179, because the policy
    sits close enough to its stall edge that 1 percent of numerical difference
    decides whether it walks. A rung's number therefore still needs several
    replays; the seed only removes the surface as one of the things varying
    between them.
    """
    default = _generator(_cfg(terrain=True, terrain_roughness_mm=5.0))
    assert default.seed == 0

    pinned = _generator(_cfg(terrain=True, terrain_roughness_mm=5.0, terrain_seed=7))
    assert pinned.seed == 7

    # The staircase grid needs the seed too, not just the rough field.
    assert _generator(_cfg(terrain=True, terrain_steps_m=0.02, terrain_seed=9)).seed == 9


def test_the_same_seed_builds_the_same_heightfield() -> None:
    """The flag reaches the ground, not just the dataclass field.

    `test_the_ground_is_the_same_for_the_same_terrain_seed` reads `seed` back off
    the config, which only proves the flag was plumbed into `TerrainGeneratorCfg`.
    What the switch is for is the surface: `TerrainGenerator.__init__` turns
    `cfg.seed` into `np.random.default_rng(seed)` and `random_rough` fills a
    400x400 heightfield from it, so a regression between the two -- the field set
    but the generator seeded elsewhere -- would leave `seed == 7` true and the
    ground random. This builds both and compares the bytes.

    Latest measured digests, first 16 of 64 hex characters: seed 0 ->
    `5b84dac182a6c7e5`, seed 7 -> `f313daef2704fc11`. They are not asserted --
    a digest may move with the mujoco or numpy build -- only the equality above
    and the inequality below are. Cross-process determinism is not claimed; the
    same interpreter building twice from one seed is.
    """
    import hashlib

    import mujoco
    import numpy as np
    from mjlab.terrains.terrain_generator import TerrainGenerator

    def field(terrain_seed: int) -> str:
        cfg = _generator(
            _cfg(terrain=True, terrain_roughness_mm=10.0, terrain_seed=terrain_seed)
        )
        spec = mujoco.MjSpec()
        TerrainGenerator(cfg).compile(spec)
        assert spec.hfields, "a rough grid must build a heightfield"
        heights = spec.hfields[0].userdata
        heights_f32 = np.fromiter(heights, dtype=np.float32, count=len(heights))
        return hashlib.sha256(heights_f32.tobytes()).hexdigest()

    assert field(0) == field(0)
    assert field(7) != field(0)


def test_a_terrain_seed_without_the_switch_is_a_usage_error() -> None:
    with pytest.raises(ValueError, match="terrain"):
        _cfg(terrain_seed=7)


def test_a_size_without_the_switch_is_a_usage_error() -> None:
    """`--terrain-roughness-mm 5` without `--terrain` reads as if it applied."""
    with pytest.raises(ValueError, match="terrain"):
        _cfg(terrain_roughness_mm=5.0)


def test_a_negative_size_is_a_usage_error() -> None:
    with pytest.raises(ValueError, match="terrain_roughness_mm"):
        _cfg(terrain=True, terrain_roughness_mm=-1.0)
    with pytest.raises(ValueError, match="terrain_steps_m"):
        _cfg(terrain=True, terrain_steps_m=-0.02)


def test_the_switches_are_this_tasks_own_arguments() -> None:
    from tasks.jumper.tripod import cli_args

    parser = argparse.ArgumentParser()
    dests = cli_args(parser.add_argument_group("jumper.tripod"))

    assert set(dests) == {
        "delay",
        "terrain",
        "terrain_roughness_mm",
        "terrain_seed",
        "terrain_stairs_share",
        "terrain_steps_m",
        "terrain_scan",
        "yaw_weight",
    }


def test_generated_ground_actually_compiles_and_steps() -> None:
    """The config reading right is not the same as the surface existing.

    Every field this file asserts about is read back off a dataclass, and a
    dataclass accepts numbers the terrain compiler then rejects -- a row of one,
    a step wider than its tile, a heightfield the contact budget cannot carry.
    That failure arrives at `ManagerBasedRlEnv(...)`, which is exactly where a
    ladder run would meet it, so the build is the assertion that matters.
    """
    import importlib

    pytest.importorskip("mjlab", reason="building a scene needs a simulator")
    resolve = importlib.import_module("mjrl.backend.resolve")
    select = importlib.import_module("mjrl.backend.select")
    env_cls = importlib.import_module("mjlab.envs").ManagerBasedRlEnv

    select.use_backend(resolve.resolve(backend="native", device="cpu", num_envs=2))
    # The staircase on its own: rough ground has its own build test, and the two
    # cannot be combined (see `test_bumps_and_steps_together_are_refused`).
    cfg = _cfg(terrain=True, terrain_steps_m=0.02)
    cfg.scene.num_envs = 2
    env = env_cls(cfg, device="cpu")
    try:
        env.reset()
        action = torch.zeros(
            env.num_envs, env.action_manager.total_action_dim, device="cpu"
        )
        for _ in range(5):
            obs, reward, terminated, truncated, _ = env.step(action)
        # A surface that failed to generate shows up as a reward that is not a
        # number long before any fall does.
        assert torch.isfinite(reward).all()
    finally:
        env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the overflow lives in the warp backend")
@pytest.mark.parametrize("millimetres", [10.0, 25.0])
def test_rough_ground_compiles_on_the_gpu_backend_too(millimetres: float) -> None:
    """160 只够到 20 mm（25 mm 要 168），所以现在是 256。

    2026-10-31 夜间 C 臂（`--terrain-roughness-mm=10`）死在这里：`nconmax
    overflow (nconmax must be >= 142)`、随后 `njmax overflow (njmax must be
    >= 590)`。一个 iteration 都没跑，整夜的延长段因此没有点火。

    同一天下午的扫描又量了一遍**建场**接触数（`--set sim.nconmax=1`，从报错里
    读）：0 mm 44、5 mm 98、10 mm 142、15 mm 153、20 mm 158、25 mm 168、35 mm
    190。当年的 160 给了 20 mm 两个接触的余量，到 25 mm 直接建不出来——而天花板
    已经测到 35 mm 以上，往上走必须先抬这个数；256 盖到 50 mm。

    native 后端不查这两个上限——上面那个测试全绿它也照样崩，所以这条必须跑
    warp。跳过的机器只是没复现它，不是它不存在。
    """
    import importlib

    pytest.importorskip("mjlab", reason="building a scene needs a simulator")
    resolve = importlib.import_module("mjrl.backend.resolve")
    select = importlib.import_module("mjrl.backend.select")
    env_cls = importlib.import_module("mjlab.envs").ManagerBasedRlEnv

    # 后端是 mjlab 的全局状态；不收回，后面每条测试都会跑在这个后端上，
    # 于是「这条测试要 GPU」变成「整场测试要 GPU」。
    from mjlab.sim import get_simulation_cls, set_simulation_cls

    previous = get_simulation_cls()
    try:
        # 后端切换与环境构造也在 `try` 里：这两个上限溢出的地方**正是构造**，
        # 而那条路径过去会把 warp 泄漏给后面每一条测试。
        select.use_backend(resolve.resolve(backend="warp", device="cuda", num_envs=2))
        cfg = _cfg(terrain=True, terrain_roughness_mm=millimetres)
        cfg.scene.num_envs = 2
        env = env_cls(cfg, device="cuda")
        try:
            env.reset()
            action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device="cuda")
            for _ in range(5):
                obs, reward, terminated, truncated, _ = env.step(action)
            assert torch.isfinite(reward).all()
        finally:
            env.close()
    finally:
        set_simulation_cls(previous)


def test_a_plane_replay_compiles_on_the_gpu_backend_too() -> None:
    """评测比训练先撞上接触上限。

    2026-10-31 的 L0 闸门就是这样死的：训练开了地形（`nconmax=160`）跑得好好的，
    `benchmark.py` 调 `play.py` 在**平地**上评它，而平地上 `nconmax` 还是基类的
    35 —— `nconmax overflow (nconmax must be >= 109)`。建场测试全在 native 上，
    一条都没红。
    """
    import importlib

    pytest.importorskip("mjlab", reason="building a scene needs a simulator")
    resolve = importlib.import_module("mjrl.backend.resolve")
    select = importlib.import_module("mjrl.backend.select")
    env_cls = importlib.import_module("mjlab.envs").ManagerBasedRlEnv

    from mjlab.sim import get_simulation_cls, set_simulation_cls

    previous = get_simulation_cls()
    try:
        select.use_backend(resolve.resolve(backend="warp", device="cuda", num_envs=2))
        cfg = _cfg()          # 没有 `--terrain`：就是评测走的那条路
        cfg.scene.num_envs = 2
        env = env_cls(cfg, device="cuda")
        try:
            env.reset()
            action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device="cuda")
            for _ in range(5):
                obs, reward, terminated, truncated, _ = env.step(action)
            assert torch.isfinite(reward).all()
        finally:
            env.close()
    finally:
        set_simulation_cls(previous)


def test_the_contact_budget_is_the_robots_not_the_grounds() -> None:
    """`nconmax` 按机器人给，不按地面给。

    中间踩过两次：写死 128 让 10 mm 建场就死（要 142）；把抬高挪进 `if terrain:`
    之后，**平地回放**死在 `nconmax must be >= 109`。22 自由度的六足站在平地上
    也要一百多个接触，上游基类的 35 是给 Go1/G1 的。
    """
    from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

    plain = _cfg()
    # 未改过的上游基线，不是任务自己的 cfg——后者就是 `plain` 本身，比它等于
    # 拿一条断言跟自己比。
    base = make_velocity_env_cfg()

    # `njmax`（约束行）不动：基类给的 1500 已经盖住实测需要的 590。早先无条件写
    # 640 是**降低**它，没有测量支持。
    assert plain.sim.njmax == base.sim.njmax
    # `nconmax` 无条件给够：平地回放 109、10 mm 142、25 mm 168、35–50 mm 190；
    # 当年取 160 只够到 20 mm，现在取 256。
    assert plain.sim.nconmax == 256
    assert _cfg(terrain=True, terrain_roughness_mm=10.0).sim.nconmax == 256


def test_zero_roughness_is_paved_not_one_big_box() -> None:
    """`flat()` 是一整块箱子；足网格对箱子走 CCD，零穿透就丢接触。

    `scenes/rough.py` 实测（每足每秒的单步接触抖动）：一块 12 m 箱子 0.96，
    1.0 m 铺装 0.06，0.5 m 铺装 0.55。L0 站在这块箱子上，量到的是接触伪影。
    """
    sub = _generator(_cfg(terrain=True, terrain_roughness_mm=0.0)).sub_terrains

    flat = sub["flat"]
    assert flat.grid_width == 1.0
    assert flat.grid_height_range == (0.0, 0.0)
    assert flat.merge_similar_heights is False


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_a_non_finite_size_is_a_usage_error(value: float) -> None:
    """`nan < 0` 和 `nan > 0` 都是 False，`nan` 会静默走 flat 分支；
    `inf` 活过检查，然后在生成器里 `int(inf / vertical_scale)` 爆炸。"""
    with pytest.raises(ValueError, match="finite"):
        _cfg(terrain=True, terrain_roughness_mm=value)
    with pytest.raises(ValueError, match="finite"):
        _cfg(terrain=True, terrain_steps_m=value)


# ── The terrain scan: the critic's eyes, never the actor's ─────────────────


def _scan_sensor(cfg):
    sensors = [s for s in (cfg.scene.sensors or ()) if s.name == "terrain_scan"]
    return sensors[0] if sensors else None


def test_the_scan_reaches_the_critic_and_never_the_actor() -> None:
    """L3 的整个前提：价值函数能看见地形，策略不能。

    actor 看不见不是简化，是部署约束——机器人身上没有前视高度传感器，
    一个依赖它的 actor 导出后就是废的。所以这条一旦反转，L3 会训出一个
    回放很好看、上机必垮的策略，而且训练曲线不会报警。
    """
    cfg = _cfg(terrain=True, terrain_roughness_mm=5.0, terrain_scan=True)

    assert "height_scan" not in (cfg.observations["actor"].terms or {})
    assert "height_scan" in (cfg.observations["critic"].terms or {})
    assert _scan_sensor(cfg) is not None


def test_the_scan_is_off_unless_it_is_asked_for() -> None:
    """默认关，且关的时候传感器整个拿掉，不是留一个常数通道。

    平地上每一根射线打到同一个平面，117 个输入全是同一个数——和把夹爪两个
    执行器喂进去一样，占 critic 的宽度、不提供信息。
    """
    for task_args in (
        {},
        {"terrain": True, "terrain_roughness_mm": 5.0},
        {"terrain": True, "terrain_steps_m": 0.02},
    ):
        cfg = _cfg(**task_args)
        assert "height_scan" not in (cfg.observations["actor"].terms or {})
        assert "height_scan" not in (cfg.observations["critic"].terms or {})
        assert _scan_sensor(cfg) is None


def test_a_scan_without_terrain_is_a_usage_error() -> None:
    """`--terrain-scan` 单独给出来是个静默的空转。

    平地正是这个开关「默认关」的理由：没有起伏时扫描全是常数。允许它单独
    存在，训出来的 critic 会对着一条死通道学价值；错误在 1000 个 iteration
    之后才以「L3 没有效果」的形式出现，而真实原因是它从来没被打开过。
    """
    with pytest.raises(ValueError, match="terrain_scan"):
        _cfg(terrain_scan=True)
    # `--terrain` 但两个尺寸都是零：L0 就是这么走的（1 m 铺装），地面是平的，
    # 117 根射线依然落在一个高度上。
    with pytest.raises(ValueError, match="terrain_scan"):
        _cfg(terrain=True, terrain_scan=True)


def test_the_scan_covers_a_body_length_or_two() -> None:
    """格子的几何是契约：0.6 x 0.4 m、5 cm 一格、挂在 base_link 上。

    117 个采样点喂给 critic，宽度写死在 `SCAN_SIZE` 里；改它会让四个任务
    的 critic 输入宽度一起变，所以这里把当前值钉住——要改是决定，不是顺手。
    """
    cfg = _cfg(terrain=True, terrain_roughness_mm=5.0, terrain_scan=True)
    sensor = _scan_sensor(cfg)
    assert sensor is not None

    assert sensor.frame.name == "base_link"
    assert sensor.pattern.size == (0.6, 0.4)
    assert sensor.pattern.resolution == 0.05
    # The resolution has to divide the size on both axes, or the grid is ragged.
    for extent in sensor.pattern.size:
        steps = extent / sensor.pattern.resolution
        assert abs(steps - round(steps)) < 1e-9
    # Endpoints included: 13 rays across, 9 along. Derived from the sensor, not
    # from the literals above, or the assertion only proves arithmetic.
    across, along = sensor.pattern.size
    res = sensor.pattern.resolution
    assert (across / res + 1) * (along / res + 1) == pytest.approx(117.0)

    # The critic normalises by `SCAN_MAX_DIST`, so the scale is not a free number.
    assert cfg.observations["critic"].terms["height_scan"].scale == pytest.approx(0.5)
