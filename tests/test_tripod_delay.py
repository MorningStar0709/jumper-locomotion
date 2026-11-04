"""The delay switch on `jumper.tripod`: off unless asked for, and real when asked.

Command delay models the bus between the policy and the motor; without it a
policy is trained against an actuator that responds in the same physics step it
was commanded, which no servo does. It stays off by default because turning it
on changes what this task trains -- the three-seed baseline was measured without
it -- so the manifest that wants it says so.
"""

from __future__ import annotations

import tasks


def actuator(task_id: str, **task_args):
    cfg = tasks.load_env_cfg(task_id, task_args=task_args or None)
    actuators = cfg.scene.entities["robot"].articulation.actuators
    assert len(actuators) == 1, "this robot's actuators are one group"
    return actuators[0]


def test_delay_is_off_by_default():
    assert actuator("jumper.tripod").delay_max_lag == 0


def test_asking_for_delay_turns_it_on():
    delayed = actuator("jumper.tripod", delay=True)
    assert delayed.delay_min_lag == 1
    assert delayed.delay_max_lag == 4
    assert delayed.delay_min_lag <= delayed.delay_max_lag


def test_a_delayed_actuator_holds_its_lag_one_step_in_five():
    delayed = actuator("jumper.tripod", delay=True)
    # `hold_prob` 是「保持当前 lag 的概率」，不是「重新抽 lag 的概率」。
    assert delayed.delay_hold_prob == 0.2
    assert delayed.delay_update_period == 1


def test_delay_does_not_change_the_actuator_that_was_there_before():
    plain, delayed = actuator("jumper.tripod"), actuator("jumper.tripod", delay=True)
    assert plain.effort_limit == delayed.effort_limit
    assert plain.stiffness == delayed.stiffness
    assert plain.damping == delayed.damping
    assert plain.armature == delayed.armature


def test_another_task_is_unaffected():
    assert actuator("jumper.flat").delay_max_lag == 0
