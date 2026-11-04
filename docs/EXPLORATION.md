# Exploration principles

This file exists because the work here is **exploration of a ceiling, not pursuit of a score**. There
is no product, no customer and no board to ship to yet, so the thing worth producing is a trustworthy
account of how far this robot's policy gets and where it stops. Without that written down, exploration
drifts into maximising whichever number is easiest to move, and every result becomes a claim nobody
can check.

## What is being explored

One policy, blind, proprioceptive only: joint positions and velocities, base angular velocity, six
foot heights, foot contact and force, action history. It never sees the ground ahead, by design --
`tasks/jumper/common/velocity_env.py:545-566` gives the terrain scan to the critic and deletes it from
the actor, because the robot has no way to measure what is in front of it, and a policy that depends on
it could not be deployed.

So the ceiling being explored is **how far a blind policy gets, terrain by terrain**, not "all-terrain
adaptation" as a claim. Blind means it reacts when a foot lands; it does not anticipate. That is the
physics of this machine, not a limitation of the training.

## Five rules

1. **A number without its conditions does not exist.** Every figure is quoted with terrain family,
   amplitude or step height, command, seed, step count and physics rate. A bare "0.03" is not a result.
2. **One variable per experiment.** On 2026-10-30 a terrain switch and a resumed run went in together
   and three days went into separating them. A control arm is cheaper than an argument.
3. **The failure point is the result.** What matters is the setting where it starts to fall, not one
   setting that happens to look good. Report the cliff, and the setting just before it.
4. **No input the robot cannot have.** No actor-side height scan, no simulated ground-truth linear
   velocity, no external state a sensor on the machine could not produce. The critic may use them; the
   exported policy may not.
5. **A curve with a lucky cell is not a curve.** Any setting that came out surprisingly good is re-run
   with another seed before it enters a conclusion.
6. **A difference the measurement cannot resolve is not a difference.** Seven replay seeds resolve a
   speed difference of roughly 40% on 2 cm stairs and 215% on 5 mm rubble; a threshold written as
   "15%" is finer than the instrument. Compute the minimum detectable effect before judging a cell
   (see below), and write "below resolution" instead of "better" when the effect is smaller.

## Protocol: scan, train one level, scan again

Training first is the expensive order. The cheap order is:

1. **Scan with the checkpoint we already have.** The task's own switches do this: `--terrain`,
   `--terrain-roughness-mm`, `--terrain-steps-m`, plus `play.py --headless --steps 1000 --command ...`
   and `--evaluation-out`. Costs minutes per cell, no training.
2. **Name the ceiling.** Measured over a long window (6000 steps, not 1000 -- a stumble occupies
   more of a short window and reads as a collapse that is not there), the ceiling is the first rung
   where the median actual speed falls under 0.85x the command *and* vx error exceeds twice that
   policy's own flat error. `resets` is not the signal: it stayed zero across 107 cells, and a
   threshold of 0.6x the command never fired even on the worst rung measured. That sentence --
   "this policy holds to N mm of rubble" -- is the result of step 1.
3. **Probe the rung before training on it.** `tools/rung_probe.py --checkpoint <one that walks the
   neighbourhood> --ground steps|rough --value N` replays the candidate ground on seven seeds and
   a flat reference on three, and labels the rung `collapsed` (under 0.60x the command: do not
   train there), `marginal` (past the ceiling of step 2 but still walking) or `holds`. It refuses
   only the first: two rungs inside the marginal region behaved oppositely -- 3 cm at 0.83 produced
   the arm that walks 4 cm, 4 cm at 0.73 destroyed the gait -- and no replay-only number separates
   them. The probe is a floor, never a promise.
4. **Train one level, and only one.** The level just past the ceiling, or the level the ceiling sits
   at, one arm, one variable.
5. **Re-scan the same grid.** The unit of progress is **how far the ceiling moved**, not how much a
   training metric improved. A run that improves `error_vel_xy` and leaves the ceiling where it was has
   taught us nothing about capability.

Scanning and climbing are different experiments and both are wanted: climbing asks "can it pass this
level", scanning asks "where does it stop". Scanning comes first because it is cheaper and it makes the
climbing meaningful.

## What counts as a result

- A curve: setting -> resets, distance travelled, mean speed, each row carrying seed, command, steps
  and physics rate.
- A boundary in one sentence, stated with its family and amplitude.
- Raw measurement JSON kept (`--evaluation-out`, `measure/`), never summarised away.
- A report under `docs/reports/` when the result is worth someone else reading; the ledger
  `docs/patrol/patrol-issues-train-jumper.md` when it is a defect to fix rather than a finding.

A finding that only exists in a chat message has not been recorded.

## Reporting an effect: pairing, resolution, seeds

Measured on 2026-11-01 from already-recorded cells, no new runs (report §十六):

- **Pair by seed.** The same replay seed is the same initial condition, so subtract arm from baseline
  per seed and report the mean of the differences. Taking a difference of two medians mixes the two
  arms' seed compositions into the effect: on 2 cm stairs turn bias it reported −30.2 mrad where the
  paired estimate is −19.8 mrad, and pairing halved the spread on 5 mm (SD 77.9 -> 19.9 mrad/s).
- **Compute the minimum detectable effect first.** With 7 seeds the 80%-power MDE is about twice the
  paired standard deviation. Measured: 2 cm straight 0.017 m/s (39% of that arm), 2 cm turn
  0.0074 m/s, 5 mm straight **0.077 m/s (215%)**, 5 mm turn 0.0045 m/s. A cell whose whole spread is
  larger than the effect being discussed has an answer of "unordered", not "marginally better".
- **Seed count follows the effect wanted.** SE falls with the square root of the count: 7 -> 14 seeds
  shrinks the MDE to 0.71x. Wanting to resolve 20% needs at least 14; resolving 10% on rough ground
  is not reachable at replay cost.
- **A gate threshold is a statement about power, not a promise.** A gate written "worse than 15% and
  p < 0.05 rejects" can only reject effects two to three times that size at the seed counts in use.
  Say that in the report rather than quoting the threshold as if it were resolved.

## What is frozen, and why

- **Deployment and packaging** until a robot exists. The vendor has not built one. `out/tripod-app/`
  stays as a demonstration that the export-to-package chain works; it is not a deliverable.
- **Appearance** (`jumper-design` is a separate repository) -- it does not change any number here.
- **VLA / high-level policies** -- no camera, no board, no demonstration data.

## The engine already has more terrain than we use

`rl/mjlab/terrains/` ships fourteen generators: rubble (`HfRandomUniform`), Perlin noise, waves,
sloped pyramids, three kinds of stairs, stepping stones, narrow beams, tilted grids, nested rings,
random grids and spreads. The current ladder uses one family at one amplitude because it was built to
measure a single rung. Nothing in the engine limits us to that.

## Honesty clause

None of this is enforced by a test. It is enforced by the person reading the report, which is why every
report ends with its boundaries rather than its highlights.
