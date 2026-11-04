"""Every registered task has to build, step, and keep the dimensions it ships.

`tests/test_registry.py` pins that a task id resolves to a config and that
`--list` works without a simulator; `tests/test_task_parity.py` pins that the four
locomotion tasks agree on their numbers. Neither of them ever builds a scene. So
a task could load its config, appear in `--list`, train, export and ship while
the thing that actually runs -- the scene, the action term, the observation
group -- was never constructed by anything but the training that discovered it.

The failures this catches all produce a run rather than an error, which is why
they need a test:

- a reward that is NaN from the first step. PPO accepts it, the loss goes NaN,
  the run keeps writing event files, and the only symptom is a policy that
  learned nothing;
- a policy observation group that grew or lost a term. Training still works and
  the checkpoint still loads, but the exported ONNX now wants a wider input than
  the deployed policy provides, and the mismatch surfaces on the robot;
- an action dimension that no longer matches the joints the controller drives.
  Same shape: a bundle that builds, converts, uploads, and moves the wrong
  number of motors.

**The dimension table below is a change detector, not a derivation.** Each entry
was read off a real build -- native backend, `cpu`, `num_envs=2`, on 2026-10-29,
the day the table was written -- and nothing recomputes it. That is deliberate:
these numbers are what every exported policy and every recorded run was made
against, so the test's job is to make a change to them loud and require the
person changing them to update this file and say why. It is not evidence that
the numbers are right -- only a build is that, and the build is the other half
of this file.
"""

from __future__ import annotations

import importlib

import pytest
import torch

pytest.importorskip("mjlab", reason="building a scene needs mjlab")

import tasks

#: `task id -> (action dim, actor obs dim, critic obs dim)`.
#:
#: Read off a build, not computed: see the module docstring. An entry is only
#: allowed to change together with the code that changes it, and the commit that
#: changes it should say which exported policy it invalidates.
CONTRACT: dict[str, tuple[int, int, int]] = {
    # Imitation tasks: all 22 joints driven, a clip phase plus the reference
    # joints in the policy's own group, privileged body state for the critic.
    "jumper.dance": (22, 209, 275),
    "jumper.dance_brazilian": (22, 209, 275),
    "jumper.dance_dream_wings": (22, 209, 275),
    "jumper.dance_maze": (22, 209, 275),
    "jumper.dance_waist": (22, 209, 275),
    "jumper.gesture_bow": (22, 209, 275),
    "jumper.gesture_hello": (22, 209, 275),
    "jumper.gesture_paw": (22, 209, 275),
    "jumper.gesture_salute": (22, 209, 275),
    # Five feet, the left-front arm carried: 16 driven joints, per-foot terms for
    # five contact points, and the pose command on top of the velocity one.
    "jumper.five_foot": (16, 407, 560),
    # The velocity skeleton and the three ladders on it. `flat` differs from the
    # three gaited tasks only in having no gait phase in the observation, which
    # is the whole 409 vs 411.
    "jumper.flat": (20, 409, 592),
    "jumper.ripple": (20, 411, 594),
    "jumper.tetrapod": (20, 411, 594),
    "jumper.tripod": (20, 411, 594),
    # A recorded jump with no command: one-shot phase and the reference future,
    # and a critic that is only barely privileged (170 against 167).
    "jumper.jump": (20, 167, 170),
    # The same jump with the recording as a fading prior: no reference joints in
    # the policy's group, so the critic carries the phase, the prior target and
    # the feedforward term instead.
    "jumper.ref_free_jump": (20, 406, 435),
    # Posture commanded on top of the tripod gait: a second command, its tracking
    # terms, and the posture state in the critic's group only.
    "jumper.posture": (20, 415, 602),
    # Stands on a swing: the deck's own terms in the critic's group and nothing
    # of the deck in the policy's, which is why 520 against 763.
    "jumper.swing": (20, 520, 763),
}

#: Two environments: one is enough to build, two is enough for a term that
#: accidentally assumes a single env to misbehave.
NUM_ENVS = 2


def _build(task: str):
    """Construct one task's scene on the CPU with the native backend."""
    resolve = importlib.import_module("mjrl.backend.resolve")
    select = importlib.import_module("mjrl.backend.select")
    from mjlab.envs import ManagerBasedRlEnv

    select.use_backend(
        resolve.resolve(backend="native", device="cpu", num_envs=NUM_ENVS)
    )
    cfg = tasks.load_env_cfg(task)
    cfg.scene.num_envs = NUM_ENVS
    return ManagerBasedRlEnv(cfg, device="cpu")


@pytest.fixture(scope="module")
def built():
    """Build each task at most once, however many assertions ask for it.

    Building a scene costs a second or two and compiling its models costs more;
    doing it per test would make this file the slowest thing in the suite for no
    extra coverage. The environments are closed at the end of the module because
    a mujoco model keeps its memory until it is.
    """
    cache = {}

    def get(task: str):
        if task not in cache:
            cache[task] = _build(task)
        return cache[task]

    try:
        yield get
    finally:
        for env in cache.values():
            env.close()


def _ids() -> list[str]:
    return sorted(tasks.list_ids())


def test_every_registered_task_builds_and_steps(built) -> None:
    """The gap this file exists for: a config is not a scene.

    A task whose scene cannot be constructed fails `scripts/train.py` in a minute
    and this test in a second, which is most of the value. A task whose scene
    builds but cannot take a step is worse, because `--num_envs` changes which
    one you get and `play` exercises a different path than training.
    """
    failures = []
    for task in _ids():
        try:
            env = built(task)
            env.reset()
            action = torch.zeros(
                env.num_envs, env.action_manager.total_action_dim, device="cpu"
            )
            for _ in range(5):
                env.step(action)
        except Exception as exc:  # noqa: BLE001 - the point is to report every one
            failures.append(f"{task}: {type(exc).__name__}: {exc}")
    assert not failures, "these tasks do not build or do not step:\n  " + "\n  ".join(
        failures
    )


def test_the_policy_and_action_dimensions_are_the_recorded_ones(built) -> None:
    """The numbers an exported policy is made against, pinned to the build.

    Compared group by group rather than as a tuple so that a failure says which
    dimension moved: a wider actor group is a policy that will not load against
    an existing ONNX, while a different action dim is a bundle driving the wrong
    number of motors.
    """
    offenders = []
    for task in _ids():
        env = built(task)
        dims = env.observation_manager.group_obs_dim
        actual = (
            env.action_manager.total_action_dim,
            dims["actor"][0],
            dims["critic"][0],
        )
        expected = CONTRACT.get(task)
        if expected is None:
            offenders.append(f"{task}: not in CONTRACT (built as {actual})")
        elif actual != expected:
            moved = ", ".join(
                f"{name} {was} -> {now}"
                for name, was, now in zip(
                    ("action", "actor obs", "critic obs"), expected, actual
                )
                if was != now
            )
            offenders.append(f"{task}: {moved}")
    assert not offenders, (
        "dims differ from CONTRACT:\n  " + "\n  ".join(offenders) + "\n\nIf the "
        "change is deliberate, update CONTRACT and say which exported policy it "
        "invalidates. If it is not, something changed a reward's observation or "
        "an action term without anyone deciding to."
    )


def test_the_contract_covers_exactly_the_registered_tasks() -> None:
    """Adding a task must add a row here, and renaming one must move it.

    Without this, a new task is covered only by the build test and its dimensions
    go unpinned -- silently, which is the whole problem. A stale row is the other
    half: it names a task that no longer exists, so the entry that used to pin it
    protects nothing while looking like it does.
    """
    registered = set(_ids())
    missing = registered - set(CONTRACT)
    stale = set(CONTRACT) - registered
    assert not missing, f"registered but not in CONTRACT: {sorted(missing)}"
    assert not stale, f"in CONTRACT but not registered: {sorted(stale)}"


def test_the_contract_table_is_not_vacuous() -> None:
    """A table of identical rows would pass the comparison above while pinning
    nothing, so the shape of the table is checked too."""
    rows = set(CONTRACT.values())
    assert len(CONTRACT) == len(_ids())
    assert len(rows) > 1, f"every task has the same dims {rows}, which proves nothing"
    assert all(a > 0 and b > 0 and c > 0 for a, b, c in CONTRACT.values())


def test_the_critic_sees_at_least_what_the_actor_sees(built) -> None:
    """Derivable, and true of every task for a reason rather than by coincidence.

    The critic group is the actor's group plus privileged state; a task where it
    came out *narrower* would mean the asymmetric split was inverted, and PPO
    would train a critic on less information than the policy it is criticising.
    """
    offenders = [
        f"{task}: actor {built(task).observation_manager.group_obs_dim['actor'][0]}"
        f" > critic {built(task).observation_manager.group_obs_dim['critic'][0]}"
        for task in _ids()
        if built(task).observation_manager.group_obs_dim["actor"][0]
        > built(task).observation_manager.group_obs_dim["critic"][0]
    ]
    assert not offenders, "the critic sees less than the actor:\n  " + "\n  ".join(
        offenders
    )


def test_a_freshly_reset_task_has_a_finite_reward_and_observation(built) -> None:
    """NaN here is the quietest failure in the file.

    A NaN reward does not raise: it propagates into the advantage, then into the
    loss, and the run finishes with a policy that never learned. Catching it at
    step zero, before a single gradient, is the difference between a test failure
    and an hour of training.
    """
    offenders = []
    for task in _ids():
        env = built(task)
        obs, _ = env.reset()
        action = torch.zeros(
            env.num_envs, env.action_manager.total_action_dim, device="cpu"
        )
        for _ in range(5):
            obs, reward, terminated, truncated, _ = env.step(action)
        if not torch.isfinite(reward).all():
            offenders.append(f"{task}: reward {reward}")
        for group in env.observation_manager.group_obs_dim:
            if not torch.isfinite(obs[group]).all():
                offenders.append(f"{task}: obs[{group}] has a non-finite value")
    assert not offenders, "non-finite values at reset:\n  " + "\n  ".join(offenders)
