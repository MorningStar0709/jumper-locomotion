---
name: train
description: Train a policy in this repository and report what was measured -- pick the backend and the environment count from the machine in front of you, start it with `tools/long_run.py` (or `tools/run_queue.py` for a list of runs) rather than bare `scripts/train.py`, read the numbers that say whether it is learning, pick the checkpoint with `tools/benchmark.py` instead of taking the newest, replay it with `scripts/play.py --measure`, and hand over a checkpoint path with measured tracking rather than an assumed success. Use this whenever someone asks for training or for a result from it: "训练一个三足步态", "train tripod", "把 jumper.jump 跑到收敛", "how is the run doing", "is this checkpoint any good", "continue the run from last night", or the README's one-sentence prompt. Reach for it before typing a `--num_envs` copied from a tutorial, because the failures are quiet -- a GPU number on a CPU machine exhausts memory an hour in, a run that "finished" with a flat tracking reward looks like one that learnt, a replay with observation noise off flatters every policy, an unseeded replay differs from itself by more than most findings, and a policy from the native backend is reported as if it were the warp one.
---

# Training a policy

`docs/USAGE.md` is the manual for the entry points, `docs/TUTORIAL.md` walks one policy from
training to a bundle, and `docs/DESIGN.md` §9.2 says why CPU and GPU are not the same run.
**This skill does not restate them.** It gives the order, the numbers that were measured
here, and the two places where a run goes wrong without saying so.

## The one thing to understand first

**A finished run is not a result; a measured replay is.** `train.py` stops when its
iteration count is up whether or not the policy learnt anything, and nothing in its last
screen says which. The result of this skill is three things stated together: the checkpoint
path, the backend and environment count it was trained with, and what `play --measure` read
off it. Leave one out and the next person has a file and a guess.

## Procedure

### 1. Measure the machine before choosing anything

```bash
python3 .claude/skills/setup-env/scripts/detect.py
```

It says which backend this machine has. **`warp:cuda` on an NVIDIA machine, `native:cpu`
everywhere else**, including all of macOS -- there is no third answer, and `--backend auto`
(the default) picks the same one. If the environment has never passed `gates.py`, that is
the `setup-env` skill, first.

### 2. Size the run for the backend, not from a page

On the native backend every environment holds its own `MjModel` and `MjData`, memory grows
with `--num_envs`, and the thread pool is capped at 8 (`--cpu_threads`). Measured on
`jumper.tripod`, 3 iterations, M3 Max (16 cores, 64 GB), 2026-10-08:

| `--num_envs` | env-steps / s | iteration | note |
|---|---|---|---|
| 16 | 454 | 0.84 s | enough to see it run, not to learn |
| 64 | 1777 | 0.86 s | the knee: four times the batch for the same wall time |
| 256 | 1850 | 3.3 s | the pool is saturated; more environments only lengthen each iteration |

So on a CPU machine start at 64, and raise it only when memory says there is room and the
step rate has not flattened. On a GPU, 4096 is the tutorial's number and memory is the only
ceiling -- measured here on `jumper.tripod` (warp, 4096 envs): 1.07 s/iter, 9.4 GB, 92k
env-steps/s. 8192 buys 13.8% more throughput for 8.8 GB more memory and pins the card at its
175 W limit, so 4096 is the knee, not a compromise. **A policy trained at 64 environments is
not the policy the GPU recipe produces**:
the effective batch is two orders of magnitude smaller, and `rl_cfg.py`'s
`num_steps_per_env` / learning rate were chosen for the large batch (DESIGN §9.2). Say which
backend the checkpoint came from every time it is mentioned.

Resolve first, then run. **Take the GPU through `tools/long_run.py`, not through `train.py`.** It
wraps the same command and adds the four things an unattended run needs: a `flock` on the card
(`tools/gpu_lock.py`, refused with rc 75 when someone else holds it), a VRAM budget checked
against the calibration table (`tools/calibration.json`, refused with rc 2 and a suggested
`--num_envs`), a `status.json` every 30 s in the run directory, and the checkpoint-honest stop
path. `scripts/train.py` is what it wraps -- run it directly and none of that exists, which is
how a 3-hour run once died with nothing on disk saying why.

```bash
.venv/bin/python tools/gpu_lock.py status                          # is the card free?
.venv/bin/python tools/long_run.py --task jumper.tripod --backend warp --device cuda \
    --num_envs 4096 --headless --max-iterations 10000 --seed 0      # one run
.venv/bin/python tools/run_queue.py manifests/tripod-3seeds.json    # a list of runs, serially
```

`--seed N` is not optional for anything you intend to compare: without it two replays of the
same checkpoint differ by ~0.008, which is larger than most of the differences this repo
argues about. `--dry-run`, `--list` and `--help` are exempt from the lock and the budget on
purpose -- a preview must not need the card.

`--headless` when nobody is watching; on macOS the live viewer needs `.venv/bin/mjpython`
and without it the run goes headless on its own and says so. TensorBoard starts with the
run unless `--no-tensorboard`; the log directory is printed as `[mjrl] log directory ...`
and is `logs/<model>/<task>/<timestamp>/`. The browser no longer opens by itself
(`--tb-open-browser`).

### 3. Read the run, not the clock

Every iteration prints a block. The lines that say whether it is learning:

- **`Mean reward`** rising, and for the locomotion tasks the **tracking** terms in
  `Episode_Reward/` -- `track_lin_vel_xy_exp` and `track_ang_vel_z_exp` -- climbing toward
  their weights. Flat tracking with a rising mean reward is a policy that found the
  stand-still or the regularisers and not the command.
- **`Mean action std`** falling from 1.0. A std that never moves is a learning rate or a
  batch that is too small for the signal.
- **`Metrics/`** -- the task's own: `slip_velocity_mean` down, `peak_height_mean` up for
  the gaits.
- **the curriculum level**, where the task has one (`Curriculum/`): a ladder that never
  climbs past level 0 in thousands of iterations is a command range the policy cannot reach.

To read a run that is not on your screen -- an unattended or overnight one -- do not tail the
log; ask the wrapper:

```bash
.venv/bin/python tools/health_jumper.py            # verdict + checks, human
.venv/bin/python tools/health_jumper.py --json     # the same for a script or a patrol round
```

`verdict` is `idle` (nothing is running), `wait`, `investigate`, `done`, or `alarm`; exit code
is 0 only for `idle`/`wait`/`done`. It reads `status.json`, the pid, the parse state, the GPU,
and the disk. `status.json` lives in the run directory the wrapper created -- if a run's
status is missing, check which directory it wrote before assuming the run failed: two
consecutive runs once shared one directory because the wrapper adopted the previous run's,
and the training itself was fine.

`--resume` continues the newest run, `--checkpoint <path>` a particular one, and
`--max-iterations N` under either is **N more**, not N in total (USAGE "Resuming training").
Continue a run with `--checkpoint <run directory>`, never by letting it pick the newest: the
newest is chosen by mtime and will happily continue a 2-iteration smoke test, or another
seed's run. `tools/run_queue.py` reads `<manifest>.result.json` and resumes what it already
finished; a queue that was interrupted needs no manual bookkeeping.

### 4. Replay with the measurement on

```bash
.venv/bin/python scripts/play.py --task jumper.tripod --headless --measure --speed 0 \
    --checkpoint logs/jumper/jumper.tripod/<timestamp> --seed 0
```

With no `--checkpoint` it takes the newest -- which is why you pass one. `--measure` writes
`measure.csv` and `measure.png` into `measure/` beside the checkpoint: joint positions,
velocities and actuator torques per control step, which is the evidence a report quotes.
`--headless` opens no window -- the live plot is for a human watching a replay, and
`--live-plot` is how to ask for it anyway; the csv and the png are written either way.
`measure.csv` is written when the replay **ends**, so a run killed halfway leaves nothing:
give `--steps` and let it finish. Observation noise stays **on** by default, as in training --
`--no-obs-noise` is the clean signal and flatters the policy, so compare the two rather than
reporting only the second. On a macOS desktop the same command without `--headless` under
`.venv/bin/mjpython` shows it.

`--physics-hz` defaults to the rate the task was trained at (`1 / sim.mujoco.timestep`, 200 Hz
for the walkers). Leave it there. 1000 Hz is a what-if, and a checkpoint measured at 1000 Hz
looked 70% better at straight-line tracking than the same checkpoint at the rate it trained on.

**Which checkpoint** is `tools/benchmark.py`'s answer, not the newest one's:

```bash
.venv/bin/python tools/benchmark.py benchmarks/tripod-3ckpt-200hz.json
```

It runs a manifest of `(cases x checkpoints)` through `--evaluation-out` with a fixed replay
seed and writes `<manifest>.score.json` plus one raw evaluation per run. 20 seconds of replay
(1000 steps) ranks checkpoints but cannot separate them -- checkpoint-to-checkpoint gaps are
smaller than seed spread; 120 seconds for anything you will report. Pick per case: on this
robot the straight-line best checkpoint is not the turning best one. `--evaluation-out` alone
(one command, one shape) is still the cheapest single number:

```bash
.venv/bin/python scripts/play.py --task jumper.tripod --headless --steps 1000 \
    --command 0.5 0 0 --evaluation-out /tmp/eval.json --checkpoint <run dir> --seed 0
```

A replay in a Design map (`--scene <file>.map`) is replay-only; `train.py` refuses maps,
and that is the documented boundary, not a bug to work around.

### 5. Report what was measured

The handover is one block, and every number in it came from a run:

```
checkpoint   logs/jumper/jumper.tripod/<timestamp>/model_<N>.pt   (per case: straight vs turning)
backend      warp:cuda, 4096 envs, seed <s>, <N> iterations, <wall time>   (or native:cpu, 64 envs)
training     mean reward <a> -> <b>; error_vel_xy <x>, error_vel_yaw <y>; std <s>; level <L>
replay       <case>: MAE <v> over <n> steps, obs noise on, --seed 0, <rate> Hz; measure/ at <path>
gaps         what is still unknown -- torque headroom, terrain, real hardware, untested ranges
```

"Training took time and compute; here is the checkpoint" is not a report. Neither is a
reward number with no backend beside it. When the replay was not run, say so; when only one
seed was run, say that too -- a single seed is a direction, not a ranking, because the
seed-to-seed spread on `jumper.tripod` is 0.017-0.028 in MAE, larger than most checkpoint gaps.

### 6. From here

Exporting the policy and packaging it is the `deploy` skill, and it begins with
`scripts/export.py` on the checkpoint this skill ends with. A new task is the `new-task`
skill, before any of the above.

## Failures worth recognising

| symptom | likely cause |
|---|---|
| `MemoryError`, or the machine swaps, an hour in | a GPU `--num_envs` on the native backend; each environment is a model copy |
| warp reports `['cpu']` alone on an NVIDIA machine | the environment, not the run: `setup-env`, gate A |
| the reward rises and the tracking terms do not | the policy found the regularisers; read the `Episode_Reward/` lines, not the total |
| `Mean action std` sits at 1.00 for hundreds of iterations | the batch is too small for the signal: more environments, or the GPU recipe |
| the replay looks better than the training curve | observation noise off (`--no-obs-noise`); compare with it on |
| two replays of one checkpoint disagree by ~0.008 | no `--seed`; pass one before believing any difference |
| a replay looks better than the training curve *and* noise was on | `--physics-hz` above the task's rate; leave it at the default |
| `--max-iterations 100` on a resume stops almost at once | it is 100 **more**; the run was already past it |
| a resumed run continues somebody else's run | `--checkpoint` was a file or omitted; give the **run directory** of the run you mean |
| `[long_run] the GPU is in use by ...` (rc 75) | another run holds the lock; it prints who and since when -- wait, or ask |
| `[long_run] refusing to start: ... try --num_envs N` (rc 2) | the VRAM budget, computed from `tools/calibration.json`; an uncalibrated task says so and assumes the largest |
| a finished run has no `status.json` | it was started with `scripts/train.py` instead of `tools/long_run.py`; the training may still be fine |
| `idle` from `health_jumper` | nothing is running; not a failure |
| `train.py` refuses a `--scene` | an imported `.map`; maps are replay-only by design |
| the run goes headless on macOS without being asked | `python` instead of `.venv/bin/mjpython`; the run is fine, only the window is missing |

## When something does not match

Do not change a reward weight or a command range to make a run look better. Those are the
task's decisions (`CLAUDE.md`, "Where a parameter lives"), and a number moved to fix one
run is a comparison broken for every run after it. Report the run as it was; a tuning
change is a separate, named change with a measurement beside it.
