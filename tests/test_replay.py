"""A replay does not evaluate the reward terms, and the policy must not be able to tell.

`mjrl/replay.py` takes the reward manager out of a replay step. Nothing in a replay
reads a reward, and evaluating them was the largest single cost of the step -- 4.7
of 11.3 ms on jumper.posture with one environment on warp.

**What could go wrong is silent.** A reward term that is the first thing in a step
to advance some state an observation reads -- a clock advanced lazily for whoever
asks first, a cache keyed on the step counter -- would, once skipped, leave that
advance to the observation. The observation runs after the reset and after the
command update, so the policy would be shown a phase one step off after every
reset, or a reference one step late, and every replay would look fine. Both kinds
of state exist in this repository today: jumper.posture's gait clock and
jumper.jump's reference cache.

So this runs every registered task twice from the same seed, with and without the
rewards, through time-outs and command resamples, and compares what the actor is
shown bit for bit. jumper.posture's actor and rewards do share one gait clock, and
a reward term is the first to advance it; it passes here because the clock
integrates the tempo the observation recorded and counts a reset as asking
(`CadenceClock` in `tasks/jumper/posture/mdp/cadence.py`). Made to integrate the
command live at whoever asks first, it fails at step 0; made to skip that catch-up
at a reset, at step 19, the first time-out.
"""

from __future__ import annotations

import pytest

import tasks

try:
    import torch
except ImportError:
    torch = None

#: Steps per rollout, an episode length and a command resampling interval, all in
#: control steps. The episode is short so that the run crosses time-out resets --
#: where a lazily advanced clock goes wrong -- and the resampling so that it
#: crosses command changes, where one integrating a command-dependent rate does.
STEPS = 48
EPISODE_STEPS = 20
RESAMPLE_STEPS = 7

#: **Two environments, so a replay is compared batched, the way training runs** --
#: rollout on two threads, every environment stepped in per-thread scratch and put
#: back into its own MjData -- not the one-environment special case of one thread
#: and one history. It ran one until native was deterministic across environments:
#: each one's warm start and control leaked through rollout's per-thread scratch,
#: two identical unskipped runs of jumper.posture diverged at step 2 and
#: jumper.swing at step 5, and this test failed for a reason that had nothing to do
#: with rewards. `tests/test_native_determinism.py` holds that now.
#:
#: Both environments still time out on the same step, in every task, so a reset
#: never lands on one alone.
NUM_ENVS = 2


class _Counted:
    """A reward term that counts its calls and is otherwise the term.

    Attribute access falls through, so a class-based term keeps its `reset` --
    which the manager calls through `term_cfg.func` on every reset.
    """

    def __init__(self, func) -> None:
        self._func = func
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self._func(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._func, name)


def _clone(obs):
    if isinstance(obs, dict):
        return {k: _clone(v) for k, v in obs.items()}
    return obs.clone()


def _equal(a, b) -> bool:
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    return torch.equal(a, b)


def _rollout(task_id: str, skip: bool):
    """The actor's observation after every step, and how many reward terms ran.

    **The training config, not `play=True`.** Replay opens a connected gamepad, and
    a pad touched between the two rollouts would make them differ for a reason that
    has nothing to do with rewards. What is under test -- the order the managers
    run in and whatever state a reward term leaves behind -- is the same in both.
    """
    from mjlab.envs import ManagerBasedRlEnv
    from mjrl.replay import skip_rewards

    cfg = tasks.load_env_cfg(task_id)
    cfg.scene.num_envs = NUM_ENVS
    cfg.seed = 0
    step_dt = cfg.sim.mujoco.timestep * cfg.decimation
    cfg.episode_length_s = EPISODE_STEPS * step_dt
    for term in cfg.commands.values():
        if hasattr(term, "resampling_time_range"):
            term.resampling_time_range = (RESAMPLE_STEPS * step_dt,) * 2

    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        counted = []
        for term_cfg in env.reward_manager._term_cfgs:
            term_cfg.func = _Counted(term_cfg.func)
            counted.append(term_cfg.func)
        if skip:
            skip_rewards(env)

        actions = torch.Generator().manual_seed(1)
        seen, resets = [], 0
        for _ in range(STEPS):
            action = torch.rand(env.action_space.shape, generator=actions) - 0.5
            obs, _, terminated, truncated, _ = env.step(action)
            assert "actor" in obs, f"{task_id} has no 'actor' observation group"
            seen.append(_clone(obs["actor"]))
            resets += int((terminated | truncated).sum())
        return seen, sum(c.calls for c in counted), resets
    finally:
        env.close()


@pytest.fixture
def native_backend():
    """Build on native cpu, and put back whatever backend was registered before."""
    from mjlab.sim import get_simulation_cls, set_simulation_cls
    from mjrl.backend import native_sim
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    saved_cls = get_simulation_cls()
    saved_nthread = native_sim._DEFAULT_NTHREAD
    use_backend(resolve(backend="native", device="cpu", num_envs=NUM_ENVS))
    try:
        yield
    finally:
        set_simulation_cls(saved_cls)
        native_sim._DEFAULT_NTHREAD = saved_nthread


@pytest.mark.parametrize("task_id", tasks.list_ids())
@pytest.mark.skipif(torch is None, reason="simulation parity requires torch")
def test_skipping_rewards_does_not_change_what_the_policy_sees(
    task_id: str, native_backend
) -> None:
    trained, ran, resets = _rollout(task_id, skip=False)
    replayed, ran_skipped, resets_skipped = _rollout(task_id, skip=True)

    # The controls: the counting works, and the run crossed the resets it exists
    # to cross. Without the first, "no reward term ran" below would pass vacuously.
    assert ran > 0, f"{task_id}: no reward term was ever evaluated, even unskipped"
    assert resets > 0, f"{task_id}: the rollout never reset; shorten EPISODE_STEPS"

    assert ran_skipped == 0, f"{task_id}: {ran_skipped} reward term calls in a replay"
    assert resets_skipped == resets, (
        f"{task_id}: {resets} resets with rewards, {resets_skipped} without"
    )
    for step, (a, b) in enumerate(zip(trained, replayed, strict=True)):
        if _equal(a, b):
            continue
        # Only now is it worth a third rollout: to tell a reward term that matters
        # from a simulation that is not deterministic, which would make any
        # difference here meaningless -- see `NUM_ENVS`.
        again, _, _ = _rollout(task_id, skip=False)
        if not all(_equal(x, y) for x, y in zip(trained, again, strict=True)):
            pytest.fail(
                f"{task_id}: two identical rollouts with the rewards on already "
                f"differ, so the difference at step {step} says nothing about "
                f"skipping them. See NUM_ENVS."
            )
        pytest.fail(
            f"{task_id}: the actor's observation differs at step {step} once rewards "
            f"are skipped -- a reward term is the first to advance something the "
            f"observation reads. See mjrl/replay.py."
        )


@pytest.fixture
def velocity_replay_configs(monkeypatch):
    """Use the real command schema with torch, a schema-shaped double otherwise."""
    import copy
    import sys
    from types import SimpleNamespace

    if torch is None:
        class UniformVelocityCommandCfg(SimpleNamespace):
            pass

        UniformVelocityCommandCfg.Ranges = SimpleNamespace
        monkeypatch.setitem(sys.modules, "mjlab.tasks.velocity.mdp.velocity_command",
                            SimpleNamespace(UniformVelocityCommandCfg=UniformVelocityCommandCfg))
    else:
        from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg

    command = UniformVelocityCommandCfg(
        entity_name="robot", resampling_time_range=(.02, .02), heading_command=True,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-.5, .5), lin_vel_y=(-.5, .5), ang_vel_z=(-.75, .75),
            heading=(-3.14, 3.14),
        ),
    )
    training = SimpleNamespace(commands={"twist": command}, curriculum={
        "command": SimpleNamespace(params={
            "levels": (.25, .35, .5, .5), "ang_levels": (.3, .5, .75, .75),
        }),
    })
    replay = SimpleNamespace(commands={"twist": copy.deepcopy(command)})
    checkpoint = {"infos": {"curriculum": {"command": {"level": 0}}}}
    return training, replay, checkpoint


def test_velocity_replay_matches_saved_rung_instead_of_final_task_ceiling(velocity_replay_configs):
    from mjrl.replay import configure_velocity_replay

    training, replay, checkpoint = velocity_replay_configs
    assert replay.commands["twist"].ranges.lin_vel_x == (-.5, .5)  # control
    info = configure_velocity_replay(replay, training, checkpoint)
    assert info["status"] == "matched" and info["curriculum_level"] == 0
    assert info["ranges"]["lin_vel_x"] == [-.25, .25]
    assert info["ranges"]["ang_vel_z"] == [-.3, .3]
    assert replay.commands["twist"].ranges.lin_vel_x == (-.25, .25)
    assert training.commands["twist"].ranges.lin_vel_x == (-.5, .5)


@pytest.mark.parametrize("level", [None, -1, 4, True, 0.5])
def test_velocity_replay_refuses_missing_or_invalid_curriculum_provenance(
    velocity_replay_configs, level,
):
    from mjrl.replay import configure_velocity_replay

    training, replay, checkpoint = velocity_replay_configs
    checkpoint["infos"]["curriculum"]["command"]["level"] = level
    with pytest.raises(ValueError, match="saved command curriculum level"):
        configure_velocity_replay(replay, training, checkpoint)


def test_velocity_replay_distinguishes_no_curriculum_from_unsupported_task(
    velocity_replay_configs,
):
    from mjrl.replay import configure_velocity_replay

    training, replay, checkpoint = velocity_replay_configs
    training.curriculum = {}
    assert configure_velocity_replay(replay, training, {})["source"] == "training_task_config"
    training.commands = {}
    info = configure_velocity_replay(replay, training, checkpoint)
    assert info["status"] == "not_applicable"
    with pytest.raises(ValueError, match="unsupported"):
        configure_velocity_replay(replay, training, checkpoint, (0, 0, 0))


@pytest.mark.parametrize("command", [(.3, 0, 0), (0, -.4, 0), (0, 0, .5)])
def test_fixed_velocity_replay_refuses_commands_above_saved_rung(
    velocity_replay_configs, command,
):
    from mjrl.replay import configure_velocity_replay

    training, replay, checkpoint = velocity_replay_configs
    with pytest.raises(ValueError, match="outside trained range"):
        configure_velocity_replay(replay, training, checkpoint, command)


@pytest.mark.parametrize("command", [(.2, 0, 0), (0, .2, 0), (0, 0, .2), (0, 0, 0)])
def test_fixed_velocity_replay_disables_sampling_overrides(velocity_replay_configs, command):
    from mjrl.replay import configure_velocity_replay

    training, replay, checkpoint = velocity_replay_configs
    info = configure_velocity_replay(replay, training, checkpoint, command)
    cfg = replay.commands["twist"]
    assert info["fixed_body_command"] == list(command)
    assert cfg.ranges.lin_vel_x == (command[0], command[0])
    assert cfg.ranges.lin_vel_y == (command[1], command[1])
    assert cfg.ranges.ang_vel_z == (command[2], command[2])
    assert not cfg.heading_command and cfg.ranges.heading is None
    assert cfg.rel_heading_envs == cfg.rel_world_envs == cfg.rel_standing_envs == 0
    assert cfg.rel_forward_envs == cfg.init_velocity_prob == 0


@pytest.mark.skipif(torch is None, reason="the real command sampler requires torch")
def test_fixed_velocity_replay_survives_every_real_resample_and_partial_reset(
    velocity_replay_configs,
):
    from types import SimpleNamespace

    from mjrl.replay import configure_velocity_replay

    training, replay, checkpoint = velocity_replay_configs
    robot = SimpleNamespace(data=SimpleNamespace(
        root_link_lin_vel_b=torch.zeros(2, 3), root_link_ang_vel_b=torch.zeros(2, 3),
        heading_w=torch.zeros(2),
    ))
    env = SimpleNamespace(num_envs=2, device="cpu", step_dt=.02, scene={"robot": robot})
    ids = torch.arange(2)
    sampled = training.commands["twist"].build(env)
    sampled.reset(ids)
    control = sampled.command.clone()
    sampled.reset(ids)
    assert not torch.equal(control, sampled.command)  # Random commands really can change.
    configure_velocity_replay(replay, training, checkpoint, (.2, -.1, .2))
    fixed = replay.commands["twist"].build(env)
    expected = torch.tensor([[.2, -.1, .2]] * 2)
    fixed.reset(ids)
    for step in range(12):
        fixed.is_standing_env[:] = True  # State overrides must be cleared too.
        fixed.is_world_env[:] = True
        fixed._joystick_enabled = SimpleNamespace(value=True)
        fixed._joystick_sliders = [SimpleNamespace(value=99.0)] * 3
        fixed._joystick_get_env_idx = lambda: 0
        fixed.compute(.02)
        if step % 3 == 0:
            fixed.reset(torch.tensor([step % 2]))
            fixed.compute(0)
        assert torch.equal(fixed.command, expected)


def test_fixed_velocity_evaluation_excludes_auto_reset_poses_and_counts_resets(tmp_path):
    import json
    from types import SimpleNamespace

    from mjrl.replay import FixedCommandEvaluation

    class Tensor:
        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return self.value

    data = SimpleNamespace(root_link_lin_vel_b=Tensor([[.1, 0, 0], [99, 99, 0]]),
                           root_link_ang_vel_b=Tensor([[0, 0, .1], [0, 0, 99]]),
                           root_link_pos_w=Tensor([[0, 0, .29], [9, 9, .29]]),
                           root_link_quat_w=Tensor([[1, 0, 0, 0], [1, 0, 0, 0]]))
    env = SimpleNamespace(num_envs=2, step_dt=.02, command_manager=SimpleNamespace(
        get_term=lambda name: SimpleNamespace(robot=SimpleNamespace(data=data)),
    ))
    measured = FixedCommandEvaluation(env, (.2, 0, .2), {"source": "checkpoint_curriculum"})
    measured.update(Tensor([0, 1]))
    result = measured.write(tmp_path / "evaluation.json")
    assert result["episode_resets"] == result["excluded_reset_samples"] == 1
    assert result["valid_samples"] == 1
    assert result["actual_velocity_mean"] == [.1, 0, .1]
    assert result["mean_absolute_error"] == pytest.approx([.1, 0, .1])
    assert result["root_mean_square_error"] == pytest.approx([.1, 0, .1])
    assert result["simulated_seconds"] == .02
    assert json.loads((tmp_path / "evaluation.json").read_text())["frame"] == "body"


@pytest.fixture
def velocity_play_entrypoint(monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    spec = importlib.util.spec_from_file_location("_cli", scripts / "_cli.py")
    cli = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "_cli", cli)
    spec.loader.exec_module(cli)
    spec = importlib.util.spec_from_file_location("_velocity_play_tests", scripts / "play.py")
    play = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(play)
    return play


@pytest.mark.parametrize("arguments,message", [
    (["--command", "nan", "0", "0"], "three finite values"),
    (["--command", "0", "inf", "0"], "three finite values"),
    (["--evaluation-out", "metrics.json"], "--command unless --app"),
    (["--command", "0", "0", "0", "--evaluation-out", "metrics.json", "--steps", "0"],
     "positive finite --steps"),
    (["--command", "0", "0", "0", "--agent", "zero"], "trained checkpoint"),
    (["--checkpoint-command-ranges", "--agent", "random"], "trained checkpoint"),
])
def test_velocity_play_rejects_invalid_evaluations_before_allocating(
    velocity_play_entrypoint, monkeypatch, capsys, arguments, message,
):
    import sys

    def should_not_resolve(args):
        pytest.fail("Invalid evaluation reached environment resolution")

    monkeypatch.setattr(velocity_play_entrypoint, "resolve_all", should_not_resolve)
    monkeypatch.setattr(sys, "argv", ["play.py", *arguments])
    with pytest.raises(SystemExit) as error:
        velocity_play_entrypoint.main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("arguments", [
    [],
    ["--replay-info-out", "info.json"],
    ["--checkpoint-command-ranges", "--command", ".2", "0", "0", "--steps", "50",
     "--evaluation-out", "metrics.json"],
])
def test_velocity_play_preserves_default_and_accepts_opt_in_cli(
    velocity_play_entrypoint, monkeypatch, capsys, arguments,
):
    import sys
    from types import SimpleNamespace

    monkeypatch.setattr(velocity_play_entrypoint, "resolve_all", lambda args: (
        SimpleNamespace(id="robot.motion"), SimpleNamespace(num_envs=1), None,
    ))
    monkeypatch.setattr(sys, "argv", ["play.py", "--dry-run", *arguments])
    velocity_play_entrypoint.main()
    assert "--dry-run, stopping here" in capsys.readouterr().out


class _Tensor:
    """The three torch methods the evaluation calls on anything it reads."""

    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.value


def _stall_env(position, velocity):
    """A one-environment fake with the fields the evaluation reads."""
    return _stall_envs([position], [velocity])


def _stall_envs(positions, velocities):
    """The same fake with one row per environment, in the order given."""
    from types import SimpleNamespace

    data = SimpleNamespace(
        root_link_lin_vel_b=_Tensor([list(v) for v in velocities]),
        root_link_ang_vel_b=_Tensor([[0.0, 0.0, 0.0] for _ in velocities]),
        root_link_pos_w=_Tensor([list(p) for p in positions]),
        root_link_quat_w=_Tensor([[1.0, 0.0, 0.0, 0.0] for _ in positions]),
    )
    return SimpleNamespace(num_envs=len(positions), step_dt=.02, command_manager=SimpleNamespace(
        get_term=lambda name: SimpleNamespace(robot=SimpleNamespace(data=data)),
    ))


def _drive(env, command, position, velocity, steps):
    """Step the evaluation with the body at `position` and moving at `velocity`."""
    from mjrl.replay import FixedCommandEvaluation

    measured = FixedCommandEvaluation(env, command, {})
    data = env.command_manager.get_term("twist").robot.data
    for index in range(steps):
        data.root_link_pos_w.value = [[position(index), 0.0, 0.29]]
        data.root_link_lin_vel_b.value = [[*velocity(index)]]
        measured.update(_Tensor([0] * env.num_envs))
    return measured


def test_a_standing_policy_on_a_walking_command_is_counted_not_truncated():
    """The discrete signal §二十 asked for: steps spent commanded-but-not-travelling.

    The episode is not ended here. `episode_resets` therefore says what it said
    before this counter existed, and `stalled_fraction` says how much of the run was
    a stall -- the number that separates two policies the continuous speed cannot
    resolve at 7 or 14 seeds.
    """
    steps = 200
    window = round(1.5 / .02)
    env = _stall_env((0.0, 0.0, 0.29), (0.01, 0.0, 0.0))
    measured = _drive(env, (.5, 0, 0), lambda _i: 0.0, lambda _i: (0.01, 0.0, 0.0), steps)
    result = measured.report()
    assert result["episode_resets"] == 0
    assert result["valid_samples"] == steps
    # The anchor step counts as the first step of the run, exactly as in the
    # training termination, so the window closes on step `window` and the run is
    # counted from step `window` onward: `steps - window + 1` steps.
    assert result["stalled_steps"] == steps - window + 1
    assert result["stalled_fraction"] == pytest.approx((steps - window + 1) / steps)


def test_the_counted_window_is_the_training_one_to_the_step():
    """The anchor step agrees with `stalled_under_command`, which fires on step 75."""
    from mjrl.stall import stall_window

    steps = 60
    window = stall_window(.02)
    assert window == 75
    env = _stall_env((0.0, 0.0, 0.29), (0.01, 0.0, 0.0))
    result = _drive(env, (.5, 0, 0), lambda _i: 0.0, lambda _i: (0.01, 0.0, 0.0), steps).report()
    assert result["stalled_steps"] == 0, "60 steps cannot close a 75-step window"
    assert result["stall_rule"] == {"min_command_mps": 0.20, "max_travel_m": 0.10, "seconds": 1.5}


def test_a_reset_starts_the_count_everywhere_in_the_fake():
    """A respawn clears the anchor and the run: the new episode is judged afresh."""
    steps = 200
    env = _stall_env((0.0, 0.0, 0.29), (0.01, 0.0, 0.0))
    from mjrl.replay import FixedCommandEvaluation

    measured = FixedCommandEvaluation(env, (.5, 0, 0), {})
    data = env.command_manager.get_term("twist").robot.data
    for index in range(steps):
        data.root_link_pos_w.value = [[0.0, 0.0, 0.29]]
        data.root_link_lin_vel_b.value = [[0.01, 0.0, 0.0]]
        # Respawn at step 100, the way the environment does at an episode end.
        measured.update(_Tensor([1 if index == 100 else 0]))
    result = measured.report()
    assert result["episode_resets"] == 1
    assert result["stalled_per_env"] == [steps - 2 * 75 + 1]
    assert measured.stall_anchor == [(0.0, 0.0)], "the anchor did not survive the respawn"


def test_each_environment_keeps_its_own_anchor_and_count():
    """One standing robot must not lend its anchor to the walking one beside it."""
    steps = 200
    env = _stall_envs([(0.0, 0.0, 0.29), (0.0, 0.0, 0.29)], [(0.01, 0, 0), (0.5, 0, 0)])
    from mjrl.replay import FixedCommandEvaluation

    measured = FixedCommandEvaluation(env, (.5, 0, 0), {})
    data = env.command_manager.get_term("twist").robot.data
    for index in range(steps):
        data.root_link_pos_w.value = [[0.0, 0.0, 0.29], [0.02 * index, 0.0, 0.29]]
        data.root_link_lin_vel_b.value = [[0.01, 0.0, 0.0], [0.5, 0.0, 0.0]]
        measured.update(_Tensor([0, 0]))
    result = measured.report()
    assert result["stalled_per_env"] == [steps - 75 + 1, 0]
    assert result["stalled_steps"] == steps - 75 + 1


def _facing(yaw: float):
    """An identity translation and a heading, as `[w, x, y, z]`."""
    import math

    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


def test_cross_track_is_measured_from_the_line_the_walk_started_on():
    """Sideways travel is what a `line` loop changes, and nothing else reports it.

    Latching is the first sampled step: a run that starts 0.4 m off its own line
    starts *on* it, because the line is defined by where the walk began.
    """
    from mjrl.replay import FixedCommandEvaluation

    env = _stall_env((0.0, 0.4, 0.29), (0.5, 0.0, 0.0))
    measured = FixedCommandEvaluation(env, (.5, 0, 0), {})
    data = env.command_manager.get_term("twist").robot.data
    for index in range(5):
        # Walks forward and drifts 0.05 m to the left per step.
        data.root_link_pos_w.value = [[0.1 * index, 0.4 + 0.05 * index, 0.29]]
        measured.update(_Tensor([0]))
    cross = measured.report()["cross_track"]
    assert cross["final_per_env_m"] == [pytest.approx(0.2)]
    assert cross["max_abs_m"] == pytest.approx(0.2)
    # Five samples, one of which only draws the line: the mean is over the four
    # that measured something, not over every step of the replay.
    assert cross["measured_samples"] == 4
    assert cross["mean_abs_m"] == pytest.approx(0.125), (
        "the first step latches and measures nothing, so it must not be counted"
    )


def test_a_respawn_forgets_the_offset_it_reported_before():
    """`final_per_env_m` must describe the episode in progress, not an older one."""
    from mjrl.replay import FixedCommandEvaluation

    env = _stall_env((0.0, 0.0, 0.29), (0.5, 0.0, 0.0))
    data = env.command_manager.get_term("twist").robot.data
    evaluation = FixedCommandEvaluation(env, (0.5, 0, 0), {})
    data.root_link_pos_w.value = [[0.0, 0.0, 0.29]]
    evaluation.update(_Tensor([0]))
    data.root_link_pos_w.value = [[0.0, 0.30, 0.29]]
    evaluation.update(_Tensor([0]))
    assert evaluation.report()["cross_track"]["final_per_env_m"] == [pytest.approx(0.30)]
    evaluation.update(_Tensor([1]))
    assert evaluation.report()["cross_track"]["final_per_env_m"] == [None], (
        "the previous episode's offset survived the respawn"
    )


def test_the_line_is_drawn_when_the_walk_starts_not_when_the_episode_does():
    """A standing start must not fix the line: the loop draws it at the first walking step.

    `scripts/play.py` ramps the command up, so the first samples are taken while
    the robot is still standing. A line latched there measures the settle, not the
    walk, and inflates every lateral number by the drift of a robot that has not
    started moving yet.

    A fixed-command run cannot cross the threshold, so the two halves are separate
    evaluations: one that stands for its whole run, one that walks from its first
    sample.
    """
    from mjrl.replay import FixedCommandEvaluation

    env = _stall_env((0.0, 0.0, 0.29), (0.5, 0.0, 0.0))
    data = env.command_manager.get_term("twist").robot.data
    still = FixedCommandEvaluation(env, (0, 0, 0), {})
    for index in range(3):
        data.root_link_pos_w.value = [[0.0, 0.2 * index, 0.29]]
        still.update(_Tensor([0]))
    assert still.report()["cross_track"]["final_per_env_m"] == [None], "a standing run drew a line"
    walking = FixedCommandEvaluation(env, (.5, 0, 0), {})
    data.root_link_pos_w.value = [[0.0, 0.5, 0.29]]
    walking.update(_Tensor([0]))
    data.root_link_pos_w.value = [[0.1, 0.5, 0.29]]
    walking.update(_Tensor([0]))
    assert walking.report()["cross_track"]["final_per_env_m"] == [pytest.approx(0.0)]


def test_cross_track_is_positive_to_the_left_of_the_start_heading():
    """The sign is the controller's sign: a `line` loop turns right on a positive one."""
    from mjrl.replay import FixedCommandEvaluation

    env = _stall_env((0.0, 0.0, 0.29), (0.5, 0.0, 0.0))
    measured = FixedCommandEvaluation(env, (.5, 0, 0), {})
    data = env.command_manager.get_term("twist").robot.data
    # Facing +y, so its left is -x: moving to -x is moving left of the line.
    data.root_link_quat_w.value = [_facing(3.141592653589793 / 2.0)]
    data.root_link_pos_w.value = [[0.0, 0.0, 0.29]]
    measured.update(_Tensor([0]))
    data.root_link_pos_w.value = [[-0.3, 0.0, 0.29]]
    measured.update(_Tensor([0]))
    assert measured.report()["cross_track"]["final_per_env_m"] == [pytest.approx(0.3)]


def test_a_respawn_starts_a_new_line():
    """A new episode is judged from its own start, not the one before it."""
    from mjrl.replay import FixedCommandEvaluation

    env = _stall_env((0.0, 0.0, 0.29), (0.5, 0.0, 0.0))
    measured = FixedCommandEvaluation(env, (.5, 0, 0), {})
    data = env.command_manager.get_term("twist").robot.data
    data.root_link_pos_w.value = [[0.0, 0.0, 0.29]]
    measured.update(_Tensor([0]))
    data.root_link_pos_w.value = [[0.2, 0.5, 0.29]]
    measured.update(_Tensor([0]))
    assert measured.report()["cross_track"]["final_per_env_m"] == [pytest.approx(0.5)]
    # Respawn at a new place: the post-reset pose is excluded, and the next step
    # latches a line through wherever the new episode begins.
    measured.update(_Tensor([1]))
    data.root_link_pos_w.value = [[5.0, 5.0, 0.29]]
    measured.update(_Tensor([0]))
    data.root_link_pos_w.value = [[5.1, 5.2, 0.29]]
    measured.update(_Tensor([0]))
    assert measured.report()["cross_track"]["final_per_env_m"] == [pytest.approx(0.2)]


def test_a_turn_in_place_reports_no_stall_fraction_at_all():
    """A turn command asks for no travel, so the fraction is unknown, not zero."""
    steps = 200
    env = _stall_env((0.0, 0.0, 0.29), (0.0, 0.0, 0.75))
    result = _drive(env, (0.0, 0, 0.75), lambda _i: 0.0, lambda _i: (0.0, 0.0, 0.75), steps).report()
    assert result["stalled_steps"] == 0
    assert result["stalled_fraction"] is None


def test_a_walking_policy_is_not_counted_as_stalled():
    steps = 200
    env = _stall_env((0.0, 0.0, 0.29), (0.5, 0.0, 0.0))
    measured = _drive(env, (.5, 0, 0), lambda i: 0.02 * i, lambda _i: (0.5, 0.0, 0.0), steps)
    result = measured.report()
    assert result["stalled_steps"] == 0
    assert result["stalled_fraction"] == 0.0


def test_a_standing_policy_under_a_standing_command_is_not_a_stall():
    steps = 200
    env = _stall_env((0.0, 0.0, 0.29), (0.0, 0.0, 0.0))
    measured = _drive(env, (0.0, 0, 0), lambda _i: 0.0, lambda _i: (0.0, 0.0, 0.0), steps)
    assert measured.report()["stalled_steps"] == 0


def test_an_app_replay_draws_the_line_when_the_body_starts_moving():
    """`--app` has no fixed command, so the body's own speed says when walking began.

    The person drives, so nothing here knows what was asked for. The line must
    still be drawn at the same instant the controller's `line` loop draws it --
    the first step the robot actually moves -- or the measurement would include
    the settle before the walk and report a loop that never steered.
    """
    from mjrl.replay import FixedCommandEvaluation

    env = _stall_env((0.0, 0.4, 0.29), (0.0, 0.0, 0.0))
    measured = FixedCommandEvaluation(env, None, {})
    data = env.command_manager.get_term("twist").robot.data
    # Two steps standing, then walking forward while drifting 0.05 m left each step.
    for index, (vx, y) in enumerate(((0.0, 0.4), (0.0, 0.4), (0.5, 0.45), (0.5, 0.50), (0.5, 0.55))):
        data.root_link_pos_w.value = [[0.1 * index, y, 0.29]]
        data.root_link_lin_vel_b.value = [[vx, 0.0, 0.0]]
        measured.update(_Tensor([0]))
    cross = measured.report()["cross_track"]
    # Line drawn at the first moving step (y = 0.45), then two samples at 0.50
    # and 0.55: 0.10 m off, and the two standing steps contribute nothing.
    assert cross["final_per_env_m"] == [pytest.approx(0.10)], (
        "the line was drawn before the robot started moving"
    )
    assert cross["measured_samples"] == 2


def test_an_app_replay_reports_no_speed_error_because_it_has_no_target():
    """Speed tracking needs a command; under `--app` there is none, and None says so.

    A zero would read as "tracked perfectly" in any table that mixes the two.
    """
    from mjrl.replay import FixedCommandEvaluation

    env = _stall_env((0.0, 0.0, 0.29), (0.5, 0.0, 0.0))
    measured = FixedCommandEvaluation(env, None, {})
    data = env.command_manager.get_term("twist").robot.data
    for index in range(3):
        data.root_link_pos_w.value = [[0.1 * index, 0.0, 0.29]]
        measured.update(_Tensor([0]))
    report = measured.report()
    assert report["command"] is None
    assert report["command_source"] == "motion"
    assert report["mean_absolute_error"] is None
    assert report["root_mean_square_error"] is None
    assert report["mean_xy_tracking_error"] is None
    assert report["actual_velocity_mean"][0] == pytest.approx(0.5), (
        "the speed itself is still measured even without a target"
    )
    assert report["stalled_fraction"] is None, (
        "the stall rule needs a commanded direction, and nobody commanded one"
    )


def test_an_app_replay_that_never_moves_measures_no_cross_track():
    from mjrl.replay import FixedCommandEvaluation

    env = _stall_env((0.0, 0.0, 0.29), (0.0, 0.0, 0.0))
    measured = FixedCommandEvaluation(env, None, {})
    for index in range(4):
        measured.update(_Tensor([0]))
    cross = measured.report()["cross_track"]
    assert cross["measured_samples"] == 0
    assert cross["mean_abs_m"] is None
    assert cross["max_abs_m"] is None


def test_an_app_evaluation_is_allowed_without_a_command_but_still_needs_steps(capsys):
    """The guard that used to forbid measuring under `--app` at all.

    `--app` runs the bundle's controller, whose commands come from the person;
    demanding `--command` there made every attempt to measure a loop under the
    controller impossible, which is how the `line` block went unmeasured.
    """
    import argparse
    from types import SimpleNamespace

    from scripts.play import _validate_command_args

    def verdict(**overrides):
        fields = dict(app="bundle.app", agent="trained", command=None,
                      command_script=None, checkpoint_command_ranges=None,
                      evaluation_out="metrics.json", steps=100)
        fields.update(overrides)
        capsys.readouterr()
        try:
            _validate_command_args(argparse.ArgumentParser(), SimpleNamespace(**fields))
        except SystemExit:
            return capsys.readouterr().err
        return ""

    assert verdict() == ""
    assert "positive finite --steps" in verdict(steps=None)
    assert "positive finite --steps" in verdict(steps=0)
    # A fixed command is still the trained path's own, not the app's.
    assert "trained checkpoint" in verdict(command=(0.5, 0.0, 0.0))
