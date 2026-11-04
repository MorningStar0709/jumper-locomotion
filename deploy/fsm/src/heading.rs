//! Holding the heading a walk started on.
//!
//! A policy trained on velocity alone has no reference for which way it points:
//! measured over ten metres of paving it drifts 1.84 m across and turns 19.6°,
//! with nothing correcting it. The fix belongs in the controller rather than in
//! another training run, because the correction is a fact about the robot, not a
//! skill the network has to learn -- and because the channel it writes to,
//! `ang_vel_z`, is one the policy already observes and was trained on.
//!
//! The loop is proportional with feed-forward: the operator's own `yaw_rate`
//! passes through untouched, and the correction is added to it. A controller
//! that assigned the correction instead would discard the request and lag by
//! `yaw_rate / kp` radians -- 29° on a sustained 0.5 rad/s turn, which is the
//! turn itself being fought.
//!
//! What it does *not* do, and says so: a body twist over the feet (`base_twist`,
//! which `jumper.posture` drives) turns the robot without a `yaw_rate` behind
//! it, so while the operator twists, the reference follows the body instead of
//! fighting it.

use crate::layout::CommandRanges;
use crate::operator::OperatorSpec;
use crate::types::Command;

/// The commanded planar speed at or below which the robot counts as standing.
///
/// Standing is not a heading worth holding: a turn in place is the operator's
/// request, so the anchor is dropped and the next walk starts from wherever the
/// robot actually points.
pub const STANDING_BELOW: f32 = 0.02;

/// The body twist above which the operator is deliberately turning the body over
/// the feet, and the heading is theirs rather than the loop's.
pub const TWIST_ABOVE: f32 = 0.02;

/// One mode's heading hold, built from its contract's `heading` block.
#[derive(Debug, Clone)]
pub struct HeadingHold {
    /// Radians of corrected turn per radian of error, per second.
    kp: f32,
    /// The ceiling on the correction, in rad/s.
    max_rate: f32,
    /// Error smaller than this is left alone, in radians.
    deadband: f32,
    /// The policy's control period, in seconds.
    dt: f32,
    /// The `ang_vel_z` band the policy was trained on, `[lo, hi]`, held as the
    /// pair rather than as a magnitude: a range like `[-0.3, 0.9]` is not
    /// symmetric, and a symmetric clamp would hand the policy a rate below its
    /// own floor.
    yaw: [f32; 2],
    /// The heading being held, or `None` before the first walking step.
    anchor: Option<f32>,
}

impl HeadingHold {
    /// Panics when the numbers cannot make a hold. `f32::clamp` panics on
    /// `min > max` or a NaN bound, and without this that panic lands inside the
    /// walking loop on the first step; a contract that reaches here is checked,
    /// so a panic at construction is the caller's bug, heard once and in the
    /// right place.
    ///
    /// `yaw` must be finite **and straddle zero**. A one-sided band such as
    /// `[0.0, 0.5]` does not make `clamp` panic, so the panic contract is not
    /// the whole reason: it silently drops every correction in one direction,
    /// and the contract would report a hold the robot is not applying. A NaN
    /// `kp` is worse than a panic -- it reaches the policy as `ang_vel_z` -- and
    /// a `max_rate` of zero or less panics on the first correction, which is the
    /// same failure moved off the walking loop.
    pub fn new(kp: f32, max_rate: f32, deadband: f32, dt: f32, yaw: [f32; 2]) -> Self {
        // `check_heading` refuses such a contract, but `of` documents callers
        // that skip it -- and those would get a hold whose clamp silently drops
        // every correction in one direction. This is the choke point.
        assert!(
            kp.is_finite() && 0.0 < kp,
            "HeadingHold needs a positive kp, got {kp}"
        );
        assert!(
            max_rate.is_finite() && 0.0 < max_rate,
            "HeadingHold needs a positive max_rate, got {max_rate}"
        );
        assert!(
            dt.is_finite() && 0.0 < dt,
            "HeadingHold needs a positive dt, got {dt}"
        );
        assert!(
            yaw[0].is_finite()
                && yaw[1].is_finite()
                && yaw[0] <= yaw[1]
                && yaw[0] < 0.0
                && 0.0 < yaw[1],
            "HeadingHold needs a finite band with lo < 0 < hi, got {yaw:?}"
        );
        Self { kp, max_rate, deadband, dt, yaw, anchor: None }
    }

    /// The hold this contract asks for, or `None` when it asks for none.
    ///
    /// The command carrying the block is the one the loop corrects, and its own
    /// `ang_vel_z` range is the band the sum is held to. `OperatorSpec::check`
    /// refuses a block on a command without that axis, so the lookup cannot
    /// miss; it panics rather than substituting a ceiling, because a fallback
    /// would re-introduce exactly the out-of-band rate the clamp exists to
    /// prevent, and only for a caller that skipped `check`.
    pub fn of(spec: &OperatorSpec, ranges: &CommandRanges, dt: f32) -> Option<Self> {
        let c = spec.command.iter().find(|c| c.heading.is_some())?;
        let heading = c.heading.as_ref()?;
        let yaw = *ranges
            .term(&c.term)
            .and_then(|m| m.get("ang_vel_z"))
            .expect("OperatorSpec::check refuses a heading block without an ang_vel_z range");
        Some(Self::new(heading.kp, heading.max_rate, heading.deadband, dt, yaw))
    }

    /// Forget the heading. Called where the command shaper is reset: a mode that
    /// is not walking has no heading to keep, and a stale anchor would turn the
    /// first step of the next walk into a correction the operator never asked
    /// for.
    pub fn reset(&mut self) {
        self.anchor = None;
    }

    /// The `ang_vel_z` this step should carry, or `None` to leave the command
    /// exactly as the operator wrote it.
    ///
    /// `command` is the shaped command -- after the band and the ramp, before
    /// the observation -- so the correction is added to what the policy will
    /// actually see, and the ramp cannot smooth it away.
    pub fn step(&mut self, quat: [f32; 4], command: &Command) -> Option<f32> {
        // Only a walk has a heading to hold. A robot standing or turning in
        // place is doing what it was told, and an anchor kept across it would
        // turn the first step of the next walk into a correction nobody asked
        // for.
        //
        // The velocity's own length, not `lin_vel_x`: a walk at 45° asks for
        // less forward than one straight ahead, and both are walks.
        let walking =
            (command.lin_vel_x * command.lin_vel_x + command.lin_vel_y * command.lin_vel_y).sqrt();
        // A host can hand the controller a non-finite `yaw_rate` -- `py.rs::set_command`
        // and the web `setAxis` both take raw `f32`s, and `operator::scale` passes a NaN
        // stick straight through. `wrap` maps a non-finite argument to 0.0, so
        // integrating it would teleport the reference to heading zero and the loop
        // would drive the robot there at up to `max_rate`. A request that is not a
        // number is no request.
        let request = if command.yaw_rate.is_finite() { command.yaw_rate } else { 0.0 };
        if walking <= STANDING_BELOW {
            self.anchor = None;
            return None;
        }

        let yaw = yaw_of(quat);
        // `imu.valid` says the reading is there, not that it is a number. A NaN
        // would become the anchor on the first step and make every later error
        // zero -- a hold that never corrects while looking healthy. Skipping the
        // tick is enough: the anchor is a heading, not a reading, and dropping
        // it would take the drift up to this sample as the new reference, which
        // is the error this loop exists to remove.
        if !yaw.is_finite() {
            // Skip the tick, keep the anchor -- but still record the turn the
            // operator asked for. A skipped advance would make that turn read as
            // drift on recovery and be corrected away, so a burst of NaN samples
            // would undo a deliberate turn even though the request was held.
            if let Some(anchor) = self.anchor {
                self.anchor = Some(wrap(anchor + request * self.dt));
            }
            return None;
        }
        // A body twist over the feet moves the IMU with no `yaw_rate` behind it.
        // While the operator is twisting, the body is where they want it: the
        // reference follows rather than fighting the gesture.
        if command.base_twist.abs() > TWIST_ABOVE {
            self.anchor = Some(yaw);
            return Some(self.inside(request));
        }

        let Some(anchor) = self.anchor else {
            // First step of a walk: this heading is the one to hold, and there
            // is nothing to correct yet.
            self.anchor = Some(yaw);
            return Some(self.inside(request));
        };

        let error = wrap(anchor - yaw);
        // The operator's own turn moves the reference with it, so `wz` is a
        // request to turn rather than an error to cancel. One policy step of it,
        // which is the step the correction is applied over -- and it advances by
        // the *request*, never by the corrected value, or the loop would feed
        // itself.
        self.anchor = Some(wrap(anchor + request * self.dt));

        if error.abs() <= self.deadband {
            return Some(self.inside(request));
        }
        let correction = (self.kp * error).clamp(-self.max_rate, self.max_rate);
        Some(self.inside(request + correction))
    }

    /// The request plus the correction, held inside the band the policy was
    /// trained on -- both ends, not a symmetric magnitude around zero.
    fn inside(&self, rate: f32) -> f32 {
        rate.clamp(self.yaw[0], self.yaw[1])
    }
}

/// The robot's yaw from its attitude quaternion, `(w, x, y, z)`.
pub fn yaw_of(quat: [f32; 4]) -> f32 {
    let [w, x, y, z] = quat;
    let siny = 2.0 * (w * z + x * y);
    let cosy = 1.0 - 2.0 * (y * y + z * z);
    siny.atan2(cosy)
}

/// An angle in (−π, π]: the difference between two headings is the short way
/// round, so a robot either side of the seam is a small error and not a full
/// turn.
pub fn wrap(a: f32) -> f32 {
    use std::f32::consts::{PI, TAU};
    if !a.is_finite() {
        return 0.0;
    }
    let mut r = (a + PI) % TAU;
    if r <= 0.0 {
        r += TAU;
    }
    r - PI
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 0.02;
    const LIMIT: f32 = 0.75;

    fn quat(yaw: f32) -> [f32; 4] {
        let (s, c) = ((yaw / 2.0).sin(), (yaw / 2.0).cos());
        [c, 0.0, 0.0, s]
    }

    fn walking(vx: f32, wz: f32) -> Command {
        Command { lin_vel_x: vx, yaw_rate: wz, ..Default::default() }
    }

    /// A drift to the left of the heading being held asks for a right turn. The
    /// sign is the whole of this module: get it backwards and the loop turns a
    /// slow drift into a slow circle, which on a 25 kg robot looks like a gait
    /// problem rather than a controller one.
    #[test]
    fn a_drift_to_the_left_asks_for_a_right_turn() {
        let mut h = HeadingHold::new(1.0, 0.5, 0.0, DT, [-LIMIT, LIMIT]);
        // First step: the anchor is the heading the walk starts on.
        assert_eq!(h.step(quat(0.0), &walking(0.5, 0.0)), Some(0.0));
        // The robot is now 0.2 rad to the left (counter-clockwise, +yaw).
        let wz = h.step(quat(0.2), &walking(0.5, 0.0)).expect("walking");
        assert!(wz < 0.0, "drifted left but asked for {wz}");
        assert!((wz + 0.2).abs() < 1e-5, "kp 1 on a 0.2 rad error gave {wz}");
    }

    /// Standing is not a heading worth holding. The anchor is dropped so the next
    /// walk starts from where the robot actually points.
    #[test]
    fn standing_drops_the_anchor() {
        let mut h = HeadingHold::new(1.0, 0.5, 0.0, DT, [-LIMIT, LIMIT]);
        h.step(quat(0.0), &walking(0.5, 0.0));
        assert_eq!(h.step(quat(0.4), &walking(0.0, 0.0)), None);
        // Walk again without turning: the anchor is this new heading, so the
        // first correction is zero, not 0.4 rad of catch-up.
        let wz = h.step(quat(0.4), &walking(0.5, 0.0)).expect("walking again");
        assert!(wz.abs() < 1e-6, "the anchor survived standing: {wz}");
    }

    /// A turn that is not a number is no turn. A host can hand the controller a
    /// NaN `yaw_rate`; `wrap` maps a non-finite argument to zero, so integrating
    /// it would silently teleport the reference to heading 0.0 rad and the loop
    /// would then drive the robot there at up to `max_rate`.
    #[test]
    fn a_non_finite_turn_request_is_not_integrated() {
        let mut h = HeadingHold::new(1.0, 0.5, 0.0, DT, [-LIMIT, LIMIT]);
        assert_eq!(h.step(quat(0.0), &walking(0.5, 0.0)), Some(0.0));
        let wz = h.step(quat(0.0), &walking(0.5, f32::NAN)).expect("walking");
        assert!(wz.is_finite(), "a NaN request reached the policy: {wz}");
        assert!(wz.abs() < 1e-6, "a NaN request asked for {wz}");
        // The reference did not move with it: the next real request still lands
        // where it was asked, not on top of a teleported anchor.
        let wz = h.step(quat(0.0), &walking(0.5, 0.4)).expect("walking");
        assert!((wz - 0.4).abs() < 1e-6, "the anchor moved during the NaN tick: {wz}");
    }

    /// The constructor is the choke point for a caller that skipped `check`, so
    /// it has to cover the invariants that would panic or poison the loop later.
    #[test]
    #[should_panic(expected = "positive kp")]
    fn a_nan_gain_is_refused_at_construction() {
        HeadingHold::new(f32::NAN, 0.5, 0.0, DT, [-LIMIT, LIMIT]);
    }

    #[test]
    #[should_panic(expected = "positive max_rate")]
    fn a_negative_correction_ceiling_is_refused_at_construction() {
        HeadingHold::new(1.0, -0.5, 0.0, DT, [-LIMIT, LIMIT]);
    }

    /// A commanded turn passes through as it was asked: the loop corrects drift,
    /// never the request. Without this the correction replaces `yaw_rate` and a
    /// sustained turn lags by `yaw_rate / kp` -- 29° at 0.5 rad/s.
    #[test]
    fn a_commanded_turn_is_handed_on_not_cancelled() {
        let mut h = HeadingHold::new(1.0, 0.5, 0.0, DT, [-LIMIT, LIMIT]);
        h.step(quat(0.0), &walking(0.5, 0.0));
        let mut yaw = 0.0_f32;
        for _ in 0..100 {
            let wz = h.step(quat(yaw), &walking(0.5, 0.4)).expect("walking");
            assert!((wz - 0.4).abs() < 1e-4, "a 0.4 rad/s turn came out as {wz}");
            yaw = wrap(yaw + 0.4 * DT);
        }
    }

    /// A turn that is not carried out is corrected on top of the request: the
    /// correction is a delta, so a robot that turns half as fast as asked still
    /// gets the request.
    #[test]
    fn a_turn_the_robot_did_not_make_is_corrected_on_top_of_the_request() {
        let mut h = HeadingHold::new(1.0, 0.5, 0.0, DT, [-LIMIT, LIMIT]);
        h.step(quat(0.0), &walking(0.5, 0.4));
        // The anchor advances as the request is handed on, so the error appears
        // on the step after: first nothing to correct, then one step of 0.4 rad/s
        // at 20 ms is 0.008 rad, and that much is added on top of the request.
        assert_eq!(h.step(quat(0.0), &walking(0.5, 0.4)), Some(0.4));
        let wz = h.step(quat(0.0), &walking(0.5, 0.4)).expect("walking");
        assert!(wz > 0.4, "the request was lost: {wz}");
        assert!((wz - (0.4 + 0.008)).abs() < 1e-4, "{wz}");
    }

    /// Body over feet is the operator's gesture, not drift: while it is held the
    /// reference follows the body, so the robot does not steer to undo a twist
    /// the person asked for.
    #[test]
    fn a_body_twist_moves_the_reference_instead_of_fighting_it() {
        let mut h = HeadingHold::new(1.0, 0.5, 0.0, DT, [-LIMIT, LIMIT]);
        h.step(quat(0.0), &walking(0.5, 0.0));
        let mut twisted = walking(0.5, 0.0);
        twisted.base_twist = 0.2;
        // The body has turned 0.3 rad and the loop was told why.
        let wz = h.step(quat(0.3), &twisted).expect("walking");
        assert!(wz.abs() < 1e-6, "a body twist was corrected as drift: {wz}");
        // Walking on from the twisted heading asks for nothing either.
        let wz = h.step(quat(0.3), &walking(0.5, 0.0)).expect("walking");
        assert!(wz.abs() < 1e-6, "the twist left a stale anchor: {wz}");
    }

    /// The error is wrapped, so a heading just either side of the seam is a small
    /// correction and not a full turn.
    #[test]
    fn the_error_wraps_across_the_seam() {
        let mut h = HeadingHold::new(1.0, 10.0, 0.0, DT, [-10.0, 10.0]);
        h.step(quat(3.10), &walking(0.5, 0.0));
        let wz = h.step(quat(-3.10), &walking(0.5, 0.0)).expect("walking");
        assert!(wz.abs() < 0.1, "0.083 rad across the seam came out as {wz}");
    }

    /// A large error is capped by the contract's rate.
    #[test]
    fn the_correction_is_capped() {
        let mut h = HeadingHold::new(10.0, 0.5, 0.0, DT, [-LIMIT, LIMIT]);
        h.step(quat(0.0), &walking(0.5, 0.0));
        let wz = h.step(quat(1.0), &walking(0.5, 0.0)).expect("walking");
        assert!((wz + 0.5).abs() < 1e-6, "10 rad of error gave {wz}");
    }

    /// The request plus the correction stays inside the axis the policy was
    /// trained on, even when the operator is already at its edge.
    #[test]
    fn the_sum_stays_inside_the_trained_axis() {
        let mut h = HeadingHold::new(10.0, 0.5, 0.0, DT, [-0.75, 0.75]);
        h.step(quat(0.0), &walking(0.5, 0.7));
        let wz = h.step(quat(0.5), &walking(0.5, 0.7)).expect("walking");
        assert!(wz.abs() <= 0.75 + 1e-6, "the sum left the trained band: {wz}");

        // The direction that adds up: a full turn request plus a full
        // correction asks for 1.2 rad/s against a 0.75 ceiling. Only the clamp
        // makes that a rate the policy was trained at.
        let mut h = HeadingHold::new(10.0, 0.5, 0.0, DT, [-0.75, 0.75]);
        h.step(quat(0.0), &walking(0.5, 0.7));
        let wz = h.step(quat(-0.5), &walking(0.5, 0.7)).expect("walking");
        assert!(
            (wz - 0.75).abs() < 1e-6,
            "the sum should stop at the trained ceiling, got {wz}"
        );
    }

    /// A trained band is a pair, not a magnitude. `[-0.3, 0.9]` is what a policy
    /// that turns harder one way was trained on; a symmetric clamp at 0.9 would
    /// hand it −0.8, below its own floor, and call that inside the band.
    #[test]
    fn an_asymmetric_band_is_held_at_both_ends() {
        let mut h = HeadingHold::new(10.0, 0.5, 0.0, DT, [-0.3, 0.9]);
        h.step(quat(0.0), &walking(0.5, -0.3));
        // Requested −0.3, a further −0.5 of correction: −0.8 against the floor.
        let wz = h.step(quat(0.5), &walking(0.5, -0.3)).expect("walking");
        assert!(
            (wz - (-0.3)).abs() < 1e-6,
            "the sum went below the trained floor: {wz}"
        );
    }

    /// `imu.valid` reports presence, not sanity. A NaN attitude skips the tick
    /// and keeps the anchor: the reference is the heading the walk started on,
    /// so re-anchoring on the next good sample would adopt the drift.
    #[test]
    fn a_nan_attitude_skips_the_tick_and_keeps_the_anchor() {
        let mut h = HeadingHold::new(1.0, 0.5, 0.0, DT, [-LIMIT, LIMIT]);
        h.step(quat(0.0), &walking(0.5, 0.0));
        assert_eq!(h.step([f32::NAN, 0.0, 0.0, 1.0], &walking(0.5, 0.2)), None);
        // The robot comes back 0.4 rad off the heading it started on, and the
        // loop still corrects against that heading -- not against 0.4. The
        // skipped tick's own request (0.2 rad/s for one step) moved the
        // reference by 0.004, so the error is -0.396, not -0.4.
        let wz = h.step(quat(0.4), &walking(0.5, 0.0)).expect("walking again");
        assert!((-0.396 - wz).abs() < 1e-5, "the anchor was re-adopted: {wz}");
    }

    /// A band that is not a band is the caller's bug, and it has to be heard at
    /// construction rather than as a panic inside `clamp` on the first step.
    #[test]
    #[should_panic(expected = "finite band with lo < 0 < hi")]
    fn a_band_that_is_not_a_band_is_refused_at_construction() {
        HeadingHold::new(1.0, 0.5, 0.0, DT, [0.5, -0.5]);
    }

    /// One side of zero is not a band: `clamp(rate, 0.0, 0.5)` turns every
    /// negative correction into no turn, and the hold would report itself
    /// applied while the robot drifts one way.
    #[test]
    #[should_panic(expected = "finite band with lo < 0 < hi")]
    fn a_band_on_one_side_of_zero_is_refused_at_construction() {
        HeadingHold::new(1.0, 0.5, 0.0, DT, [0.0, 0.5]);
    }

    /// A tick with no attitude still records the commanded turn, so recovering
    /// from a burst of bad samples does not read a deliberate turn as drift.
    #[test]
    fn a_skipped_tick_still_records_the_commanded_turn() {
        let mut h = HeadingHold::new(1.0, 0.5, 0.0, DT, [-LIMIT, LIMIT]);
        h.step(quat(0.0), &walking(0.5, 0.0));
        assert_eq!(h.step([f32::NAN, 0.0, 0.0, 1.0], &walking(0.5, 0.4)), None);
        // The robot did not turn; the reference did, by exactly the request.
        let wz = h.step(quat(0.0), &walking(0.5, 0.0)).expect("walking again");
        assert!((wz - 0.008).abs() < 1e-5, "the commanded turn was lost: {wz}");
    }

    /// Inside the deadband nothing is corrected: a straight-walking policy must
    /// not be told to dither.
    #[test]
    fn inside_the_deadband_nothing_moves() {
        let mut h = HeadingHold::new(1.0, 0.5, 0.05, DT, [-LIMIT, LIMIT]);
        h.step(quat(0.0), &walking(0.5, 0.0));
        assert_eq!(h.step(quat(0.03), &walking(0.5, 0.0)), Some(0.0));
        assert!(h.step(quat(0.06), &walking(0.5, 0.0)).unwrap() < 0.0);
    }

    /// Closing on the heading from a drift converges, with a plant that only
    /// partly obeys the correction -- which is what a real one does. The residual
    /// is the steady-state error of a proportional loop against a constant
    /// disturbance: 0.02 rad/s of drift over 0.7 authority settles at 0.029 rad.
    #[test]
    fn a_closed_loop_converges_from_a_drift() {
        let mut h = HeadingHold::new(1.0, 0.5, 0.0, DT, [-LIMIT, LIMIT]);
        // Anchor at zero, then the robot is knocked 0.25 rad to the left.
        h.step(quat(0.0), &walking(0.5, 0.0));
        let mut yaw = 0.25_f32;
        for _ in 0..500 {
            let wz = h.step(quat(yaw), &walking(0.5, 0.0)).unwrap();
            // 70 % of what was asked, plus a steady drift to the left.
            yaw = wrap(yaw + (0.7 * wz + 0.02) * DT);
        }
        assert!(yaw.abs() < 0.035, "500 steps from 0.25 rad settled at {yaw}");
    }
}