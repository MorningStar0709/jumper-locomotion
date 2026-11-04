//! Where the robot is, from the feet it stands on.
//!
//! The controller is not told its position: [`RobotState`](crate::types::RobotState)
//! carries `q`, `qd`, `tau` and the IMU, and nothing else. Path tracking needs a
//! pose, so the pose has to be integrated. This module does the integration and
//! leaves the two things it cannot know to its caller: the foot positions in the
//! body frame (forward kinematics over `q`) and which feet are on the ground.
//!
//! The quantity is the same one a firmware would compute: take the feet that were
//! planted at both ends of a control step, average their displacement, negate it,
//! and turn it into the world frame with the IMU heading. A planted foot does not
//! move in the world, so doing this in the **world** frame measures nothing --
//! measured over 10 m walks it reads a pose that stays at the origin. The body
//! frame is the only one where the difference means "the robot moved".
//!
//! Measured against the simulator over twelve 10 m walks on paving and on 50 mm
//! ground (`benchmarks/foot-odometry-20261103.result.json`): the body-frame error
//! is 0.19-0.27 m at the median, 0.54 m at worst. That is the same order as the
//! lateral error the heading loop removes, so this pose supports slow straight
//! tracking and coarse path following; it does not support docking.
//!
//! Slip is invisible here and always will be: a foot that slides while planted
//! does not change its angle relative to the body, so no leg sensor can see it.
//! The estimate is good because this gait plants two or three feet per step, not
//! because slip was measured.

/// A planar pose: metres in the world frame, heading in radians.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose2d {
    pub x: f32,
    pub y: f32,
    pub yaw: f32,
}

impl Pose2d {
    pub fn new(x: f32, y: f32, yaw: f32) -> Self {
        Self { x, y, yaw }
    }

    /// Signed distance to the left of the line through `origin` along its heading.
    pub fn cross_track(&self, origin: &Pose2d) -> f32 {
        let (c, s) = (origin.yaw.cos(), origin.yaw.sin());
        let (dx, dy) = (self.x - origin.x, self.y - origin.y);
        -s * dx + c * dy
    }

    /// Heading error against `origin`, in `(-pi, pi]`.
    pub fn heading_error(&self, origin: &Pose2d) -> f32 {
        wrap(self.yaw - origin.yaw)
    }
}

/// Wraps an angle into `(-pi, pi]`. One implementation, shared with the heading loop:
/// two copies of this disagree at the seam and then disagree about which way to turn.
pub use crate::heading::wrap;

/// How many feet the robot has. Fixed, because both inputs come from one model; a
/// caller with another count gets `None` rather than a silently rearranged gait.
pub const FEET: usize = 6;

/// Integrates a pose from planted-foot displacement.
///
/// The caller owns the two inputs: `feet_body` is each foot's position in the body
/// frame (what forward kinematics over `q` gives), `planted` is that foot's contact
/// state this step. A foot counted as planted here and on the previous step is used;
/// one that changed state is not, because its displacement includes the swing.
///
/// The hot path allocates nothing: six feet, fixed arrays.
#[derive(Debug, Clone)]
pub struct FootOdometry {
    pose: Pose2d,
    /// Where each foot was in the body frame at the end of the last accepted step,
    /// and whether it was down. The snapshot is replaced on every call, including a
    /// rejected one, so that the "planted at both ends" rule always compares adjacent
    /// steps rather than reaching back over a gap.
    previous: [[f32; 2]; FEET],
    previous_planted: [bool; FEET],
    have_previous: bool,
    /// The heading the stored snapshot was taken at. The two ends of a step are
    /// expressed in body frames that differ whenever the robot turned during it, so
    /// the previous point is turned into the current frame before differencing; without
    /// that, a foot 0.3 m out contributes 0.3 × Δyaw of phantom travel per step.
    frame_yaw: f32,
    /// How fast a foot has to move, relative to the others, before it is not
    /// counted: `1.0 m/s`, which at the 50 Hz control rate the estimator was measured
    /// at is the 2 cm it has always been. Stated as a speed because the caller's step
    /// is not fixed -- the board feeds this every host tick, ~1 ms -- and a distance
    /// that never exceeds a millimetre per step would never reject anything.
    pub outlier_m_per_s: f32,
}

impl FootOdometry {
    pub fn new(x: f32, y: f32, yaw: f32) -> Self {
        Self {
            pose: Pose2d::new(x, y, yaw),
            previous: [[0.0; 2]; FEET],
            previous_planted: [false; FEET],
            have_previous: false,
            frame_yaw: 0.0,
            outlier_m_per_s: 1.0,
        }
    }

    pub fn pose(&self) -> Pose2d {
        self.pose
    }

    /// Adopts `yaw` as the heading this pose has always had. Used once, when the
    /// IMU's first believable heading becomes the frame's zero; `update` would
    /// otherwise start from a yaw of 0 that the robot was never facing.
    pub fn set_yaw(&mut self, yaw: f32) {
        self.pose.yaw = yaw;
        self.frame_yaw = yaw;
    }

    /// Forgets the walk: the next step only records where the feet are.
    ///
    /// The heading is not a parameter: `update` writes `pose.yaw` from the IMU on every
    /// step it accepts, so anything passed in would survive only until the first one.
    pub fn reset(&mut self) {
        self.pose = Pose2d::new(0.0, 0.0, 0.0);
        self.have_previous = false;
        self.frame_yaw = 0.0;
    }

    /// Drops the snapshot without moving the pose: the next accepted step only
    /// records where the feet are.
    pub fn forget(&mut self) {
        self.have_previous = false;
    }

    /// One control step, in the body frame. Returns the updated pose, or `None` on a
    /// step that could not move the estimate (no readings yet, no foot planted twice,
    /// or an input that is not a number).
    pub fn update(
        &mut self,
        yaw: f32,
        feet_body: &[[f32; 2]; FEET],
        planted: &[bool; FEET],
        dt: f32,
    ) -> Option<Pose2d> {
        let usable = yaw.is_finite()
            && dt.is_finite()
            && dt > 0.0
            && feet_body
                .iter()
                .all(|p| p[0].is_finite() && p[1].is_finite());
        if !usable {
            // Nothing is recorded: a NaN foot stored here would poison the next step,
            // and an old snapshot across a gap would include a whole swing.
            self.have_previous = false;
            return None;
        }
        let had = self.have_previous;
        let previous = self.previous;
        let previous_planted = self.previous_planted;
        let turned = wrap(yaw - self.frame_yaw);
        self.previous = *feet_body;
        self.previous_planted = *planted;
        self.have_previous = true;
        self.frame_yaw = yaw;
        self.pose.yaw = yaw;
        if !had {
            return None;
        }

        let mut kept: [[f32; 2]; FEET] = [[0.0; 2]; FEET];
        let mut n = 0usize;
        // Turn the old sample into the current body frame, then difference.
        let (rs, rc) = ((-turned).sin(), (-turned).cos());
        for i in 0..FEET {
            if planted[i] && previous_planted[i] {
                let px = rc * previous[i][0] - rs * previous[i][1];
                let py = rs * previous[i][0] + rc * previous[i][1];
                kept[n] = [px - feet_body[i][0], py - feet_body[i][1]];
                n += 1;
            }
        }
        if n == 0 {
            return None;
        }
        let tolerance = self.outlier_m_per_s * dt;
        if n > 1 && tolerance > 0.0 {
            n = drop_outliers(&mut kept, n, tolerance);
        }
        let mut dx = 0.0;
        let mut dy = 0.0;
        for k in kept[..n].iter() {
            dx += k[0];
            dy += k[1];
        }
        dx /= n as f32;
        dy /= n as f32;
        // The displacement is measured in the body frame, so it is only a world
        // displacement once it is turned by the heading the body had during it.
        let (s, c) = (yaw.sin(), yaw.cos());
        self.pose.x += c * dx - s * dy;
        self.pose.y += s * dx + c * dy;
        Some(self.pose)
    }
}

/// Drops steps that disagree with the median by more than `tolerance`, in place.
///
/// A foot that slips contributes a displacement the others do not share; the median
/// of the step's two components is the honest centre, and anything far from it is not
/// telling us about the body. Nothing is dropped when that would leave fewer than
/// two feet, because one foot cannot say which of them is wrong.
fn drop_outliers(steps: &mut [[f32; 2]; FEET], n: usize, tolerance: f32) -> usize {
    let mut xs = [0.0_f32; FEET];
    let mut ys = [0.0_f32; FEET];
    for i in 0..n {
        xs[i] = steps[i][0];
        ys[i] = steps[i][1];
    }
    let (mx, my) = (median(&mut xs[..n]), median(&mut ys[..n]));
    let mut kept = [[0.0_f32; 2]; FEET];
    let mut k = 0usize;
    for i in 0..n {
        if (steps[i][0] - mx).abs() <= tolerance && (steps[i][1] - my).abs() <= tolerance {
            kept[k] = steps[i];
            k += 1;
        }
    }
    if k < 2 {
        return n;
    }
    steps[..k].copy_from_slice(&kept[..k]);
    k
}

fn median(values: &mut [f32]) -> f32 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if n % 2 == 1 {
        values[n / 2]
    } else {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    }
}

/// The pose, for a controller that has joint angles and an IMU and nothing else.
///
/// This is the only place the two halves meet: [`FootKinematics`](crate::legs::FootKinematics)
/// turns `q` into feet, [`planted_from_height`](crate::legs::planted_from_height)
/// says which are down, and [`FootOdometry`] integrates what they did. A mode holds
/// one of these; it costs one forward-kinematics chain per control step and changes
/// no command, so a mode that does not steer by it behaves exactly as before.
#[derive(Debug, Clone)]
pub struct PoseTracker {
    kinematics: crate::legs::FootKinematics,
    odometry: FootOdometry,
    /// Whether the heading has ever been taken from the IMU. Before the first
    /// frame the pose reads yaw 0, which is a guess: the robot's own heading at
    /// power-on is whatever it is facing, and a loop that latches a line from
    /// yaw 0 aims `|imu yaw|` radians away from where the robot is pointing.
    seeded: bool,
}

impl PoseTracker {
    /// `None` when the contract's wire does not carry every joint the chain needs,
    /// because a leg missing a joint would report the rest pose as a measurement.
    pub fn new(joint_names: &[String]) -> Option<Self> {
        let kinematics = crate::legs::FootKinematics::new(joint_names);
        if !kinematics.usable() {
            return None;
        }
        Some(Self { kinematics, odometry: FootOdometry::new(0.0, 0.0, 0.0), seeded: false })
    }

    pub fn pose(&self) -> Pose2d {
        self.odometry.pose()
    }

    /// Whether any heading has been taken from the IMU yet. Until it has, the pose
    /// reads yaw 0, which is the robot's finding-out value and not its heading: a
    /// loop that latches a reference from it aims at a line the robot is not on.
    pub fn seeded(&self) -> bool {
        self.seeded
    }

    /// Forgets the walk. The heading comes from the next IMU frame rather than a
    /// number passed in, because the only caller that has one is the same IMU.
    pub fn reset(&mut self) {
        self.odometry.reset();
        self.seeded = false;
    }

    /// Drops the previous foot sample without moving the pose: the next step only
    /// records where the feet are. Used when the tick stream stalled, so the distance
    /// travelled during the gap is not attributed to the step after it.
    pub fn forget(&mut self) {
        self.odometry.forget();
    }

    /// One control step. The first step after a reset only records where the feet are.
    pub fn step(&mut self, quat: [f32; 4], q: &[f32], dt: f32) -> Option<Pose2d> {
        let yaw = crate::heading::yaw_of(quat);
        let feet = self.kinematics.feet(q);
        if !yaw.is_finite() || !feet.iter().flatten().all(|v| v.is_finite()) {
            // Not a step: forget where the feet were, so the next good reading cannot
            // measure displacement across the gap (which would include a swing).
            self.odometry.forget();
            return None;
        }
        if !self.seeded {
            // The first believable heading is the robot's frame: adopting it as
            // yaw 0 makes the pose relative to where the robot started, which is
            // what a consumer of `pose()` expects, and it stops the very first
            // measurement from mixing an absolute IMU heading into a relative pose.
            self.seeded = true;
            self.odometry.reset();
            self.odometry.set_yaw(yaw);
        }
        let planted = crate::legs::planted_from_height(&feet);
        let xy = crate::legs::feet_xy(&feet);
        self.odometry.update(yaw, &xy, &planted, dt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FEET: usize = 6;

    fn every_chain_joint() -> Vec<String> {
        let mut all: Vec<String> = Vec::new();
        for leg in crate::leg_table::LEGS {
            for seg in leg.chain {
                if let Some(j) = seg.joint {
                    all.push(j.to_string());
                }
            }
        }
        all.sort();
        all.dedup();
        all
    }

    #[test]
    fn a_tracker_needs_every_chain_joint() {
        let all = every_chain_joint();
        assert!(PoseTracker::new(&all).is_some(), "the full chain is usable");
        let short = &all[..all.len() - 1];
        assert!(
            PoseTracker::new(short).is_none(),
            "a chain missing a joint would report the rest pose as a measurement"
        );
    }

    #[test]
    fn a_tracker_at_rest_stays_at_the_origin() {
        let all = every_chain_joint();
        let mut tracker = PoseTracker::new(&all).expect("full chain");
        let q = vec![0.0_f32; all.len()];
        let ident = [1.0, 0.0, 0.0, 0.0];
        assert!(tracker.step(ident, &q, 0.02).is_none(), "the first step only records");
        let pose = tracker.step(ident, &q, 0.02).expect("second step");
        assert!(pose.x.abs() < 1e-6 && pose.y.abs() < 1e-6, "a still robot does not move");
        tracker.reset();
        // The heading is not handed in on reset: it comes from the next IMU frame, so
        // until one arrives the tracker reports the heading it last saw (here 0.0).
        assert!(tracker.pose().yaw.abs() < 1e-6);
        // A quaternion for yaw = 0.5: half the angle in the w and z slots.
        let yawed = [0.25_f32.cos(), 0.0, 0.0, 0.25_f32.sin()];
        tracker.step(yawed, &q, 0.02);
        let after = tracker.step(yawed, &q, 0.02).expect("a step after the reset");
        assert!((after.yaw - 0.5).abs() < 1e-5, "yaw was {}", after.yaw);
    }

    fn feet(centres: [f32; 2]) -> [[f32; 2]; FEET] {
        // Six feet spread around the body, all displaced by the same centre.
        let mut out = [[0.0_f32; 2]; FEET];
        for (i, slot) in out.iter_mut().enumerate() {
            let a = i as f32 * std::f32::consts::TAU / FEET as f32;
            *slot = [centres[0] + 0.2 * a.cos(), centres[1] + 0.2 * a.sin()];
        }
        out
    }

    #[test]
    fn a_step_back_in_the_body_frame_is_a_step_forward_in_the_world() {
        let mut odo = FootOdometry::new(0.0, 0.0, 0.0);
        odo.update(0.0, &feet([0.0, 0.0]), &[true; FEET], 0.02);
        let pose = odo
            .update(0.0, &feet([-0.01, 0.0]), &[true; FEET], 0.02)
            .expect("a step with every foot planted moves the estimate");
        assert!((pose.x - 0.01).abs() < 1e-6, "x was {}", pose.x);
        assert!(pose.y.abs() < 1e-6, "y was {}", pose.y);
    }

    #[test]
    fn the_same_step_turns_with_the_heading() {
        let mut odo = FootOdometry::new(0.0, 0.0, 0.0);
        odo.update(0.0, &feet([0.0, 0.0]), &[true; FEET], 0.02);
        // Turned a quarter turn left, a body-frame step of -x is a world step of +y.
        let pose = odo
            .update(std::f32::consts::FRAC_PI_2, &feet([-0.01, 0.0]), &[true; FEET], 0.02)
            .expect("the heading does not stop the step");
        assert!(pose.x.abs() < 1e-6, "x was {}", pose.x);
        assert!((pose.y - 0.01).abs() < 1e-6, "y was {}", pose.y);
    }

    #[test]
    fn a_foot_that_was_not_planted_before_is_not_a_body_displacement() {
        let mut odo = FootOdometry::new(0.0, 0.0, 0.0);
        // Foot 0 lifts and moves a long way; the other five are planted throughout.
        let before = feet([0.0, 0.0]);
        let mut after = feet([0.0, 0.0]);
        after[0] = [0.5, 0.5];
        odo.update(0.0, &before, &[false, true, true, true, true, true], 0.02);
        let pose = odo
            .update(0.0, &after, &[false, true, true, true, true, true], 0.02)
            .expect("five planted feet carry the step");
        assert!(pose.x.abs() < 1e-6 && pose.y.abs() < 1e-6, "pose {pose:?}");
    }

    #[test]
    fn a_foot_that_just_landed_is_not_a_body_displacement() {
        let mut odo = FootOdometry::new(0.0, 0.0, 0.0);
        let before = feet([0.0, 0.0]);
        let mut after = feet([0.0, 0.0]);
        after[0] = [0.4, 0.0]; // this foot swung in and landed on this step
        odo.update(0.0, &before, &[false, true, true, true, true, true], 0.02);
        let pose = odo
            .update(0.0, &after, &[true; FEET], 0.02)
            .expect("the five feet that stayed down carry the step");
        assert!(pose.x.abs() < 1e-6 && pose.y.abs() < 1e-6, "pose {pose:?}");
    }

    #[test]
    fn a_whole_group_landing_together_is_not_a_body_displacement() {
        let mut odo = FootOdometry::new(0.0, 0.0, 0.0);
        let before = feet([0.0, 0.0]);
        let mut after = feet([0.0, 0.0]);
        for foot in after.iter_mut().take(3) {
            foot[0] += 0.4; // three feet swing in and land on this step
        }
        odo.update(0.0, &before, &[false, false, false, true, true, true], 0.02);
        let pose = odo
            .update(0.0, &after, &[true; FEET], 0.02)
            .expect("three feet that stayed down carry the step");
        assert!(pose.x.abs() < 1e-6, "the landing group moved the pose: {pose:?}");
    }

    #[test]
    fn nothing_planted_twice_means_no_update() {
        let mut odo = FootOdometry::new(0.0, 0.0, 0.0);
        odo.update(0.0, &feet([0.0, 0.0]), &[false; FEET], 0.02);
        assert!(odo.update(0.0, &feet([0.1, 0.0]), &[false; FEET], 0.02).is_none());
    }

    #[test]
    fn one_slipping_foot_does_not_drag_the_estimate() {
        let mut odo = FootOdometry::new(0.0, 0.0, 0.0);
        odo.update(0.0, &feet([0.0, 0.0]), &[true; FEET], 0.02);
        let mut after = feet([-0.01, 0.0]);
        // Foot 2 slides 4 cm further than its neighbours did.
        after[2][0] -= 0.04;
        let pose = odo
            .update(0.0, &after, &[true; FEET], 0.02)
            .expect("the steps are still usable");
        assert!(
            (pose.x - 0.01).abs() < 2e-3,
            "a slipping foot moved the estimate to {}",
            pose.x
        );
    }

    #[test]
    fn a_lifted_foot_with_one_perturbed_neighbour_is_still_dropped() {
        let mut odo = FootOdometry::new(0.0, 0.0, 0.0);
        odo.update(0.0, &feet([0.0, 0.0]), &[true; FEET], 0.02);
        let mut after = feet([-0.01, 0.0]);
        after[1][1] += 0.05; // one foot dragged sideways
        let pose = odo
            .update(0.0, &after, &[true; FEET], 0.02)
            .expect("five good feet are enough");
        assert!((pose.y).abs() < 2e-3, "sideways slip leaked: {}", pose.y);
    }

    #[test]
    fn reset_forgets_the_walk_it_was_in_the_middle_of() {
        let mut odo = FootOdometry::new(0.0, 0.0, 0.0);
        odo.update(0.0, &feet([0.0, 0.0]), &[true; FEET], 0.02);
        odo.reset();
        // The first step after a reset only records where the feet are.
        assert!(odo.update(0.0, &feet([0.0, 0.0]), &[true; FEET], 0.02).is_none());
        // The robot walks 1 cm along a heading of 0.3 rad while turning to face it: the
        // feet stay in the world, so in the *new* body frame they appear rotated back.
        // A foot that is planted in the world does exactly this, and getting it wrong
        // is the lever-arm term the frame correction exists for.
        let d = [0.01 * 0.3f32.cos(), 0.01 * 0.3f32.sin()];
        let before = feet([0.0, 0.0]);
        let (s, c) = (-0.3f32).sin_cos();
        let mut after = [[0.0_f32; 2]; FEET];
        for (i, slot) in after.iter_mut().enumerate() {
            let dx = before[i][0] - d[0];
            let dy = before[i][1] - d[1];
            *slot = [c * dx - s * dy, s * dx + c * dy];
        }
        let pose = odo
            .update(0.3, &after, &[true; FEET], 0.02)
            .expect("the second step after a reset counts");
        // A reset starts at the origin; the walk is then turned by the heading it was
        // walked at, which recovers exactly `d`.
        let expected_x = d[0];
        let expected_y = d[1];
        assert!((pose.x - expected_x).abs() < 1e-6, "pose {pose:?}");
        assert!((pose.y - expected_y).abs() < 1e-6, "pose {pose:?}");
    }

    #[test]
    fn a_missing_heading_does_not_advance_the_estimate() {
        let mut odo = FootOdometry::new(0.0, 0.0, 0.0);
        odo.update(0.0, &feet([0.0, 0.0]), &[true; FEET], 0.02);
        assert!(odo.update(f32::NAN, &feet([-0.01, 0.0]), &[true; FEET], 0.02).is_none());
    }

    #[test]
    fn the_slip_filter_scales_with_the_step() {
        // The same disagreement is a small step at 50 Hz and a leap at 1 kHz, because
        // the tolerance is a speed. 3 mm of disagreement:
        let mut slow = FootOdometry::new(0.0, 0.0, 0.0);
        slow.update(0.0, &feet([0.0, 0.0]), &[true; FEET], 0.02);
        let mut moved = feet([-0.01, 0.0]);
        moved[5][0] += 0.003;
        let kept = slow.update(0.0, &moved, &[true; FEET], 0.02).expect("a step");
        assert!((kept.x - 0.01).abs() > 1e-6, "at 50 Hz the 3 mm foot should still count: {kept:?}");

        let mut fast = FootOdometry::new(0.0, 0.0, 0.0);
        fast.update(0.0, &feet([0.0, 0.0]), &[true; FEET], 0.001);
        let mut nudged = feet([-0.001, 0.0]);
        nudged[5][0] += 0.003;
        let p = fast.update(0.0, &nudged, &[true; FEET], 0.001).expect("a step");
        assert!((p.x - 0.001).abs() < 1e-6, "at 1 kHz 3 mm is a slip and must be dropped: {p:?}");
    }

    #[test]
    fn a_rejected_reading_is_not_kept_as_the_next_reference() {
        // A NaN reading must not become the snapshot: the next good step would subtract
        // it and the estimate would go NaN for the rest of the walk.
        let mut odo = FootOdometry::new(0.0, 0.0, 0.0);
        odo.update(0.0, &feet([0.0, 0.0]), &[true; FEET], 0.02);
        let mut bad = feet([-0.01, 0.0]);
        bad[2][0] = f32::NAN;
        assert!(odo.update(0.0, &bad, &[true; FEET], 0.02).is_none());
        // The next good step only records, because the reference was dropped.
        assert!(odo.update(0.0, &feet([-0.01, 0.0]), &[true; FEET], 0.02).is_none());
        let p = odo.update(0.0, &feet([-0.02, 0.0]), &[true; FEET], 0.02).expect("a step");
        assert!((p.x - 0.01).abs() < 1e-6, "pose {p:?}");
    }

    #[test]
    fn forgetting_keeps_the_pose_and_drops_only_the_sample() {
        // What the stall guard relies on: after a gap the walk continues from where the
        // robot thinks it is, and the step after the gap only records the feet.
        let mut odo = FootOdometry::new(0.0, 0.0, 0.0);
        odo.update(0.0, &feet([0.0, 0.0]), &[true; FEET], 0.02);
        let walked = odo
            .update(0.0, &feet([-0.01, 0.0]), &[true; FEET], 0.02)
            .expect("the second sample measures");
        assert!(walked.x > 0.0);
        odo.forget();
        assert!(
            odo.update(0.0, &feet([-0.02, 0.0]), &[true; FEET], 0.02).is_none(),
            "the first step after a gap must only record the feet"
        );
        assert!((odo.pose().x - walked.x).abs() < 1e-9, "forgetting moved the pose");
    }

    #[test]
    fn turning_in_place_does_not_move_the_pose() {
        // Feet on a ring average to the origin, so a ring cannot test the frame
        // correction: rotating it changes nothing to rotate. These feet all sit at one
        // offset (the lever arm), and the second sample is that same point seen from
        // the turned body -- `p_cur = R(-dYaw) p_prev` -- which is what a foot planted
        // in the world does while the robot spins over it.
        let mut odo = FootOdometry::new(0.0, 0.0, 0.0);
        let before = [[0.3_f32, 0.0]; FEET];
        odo.update(0.0, &before, &[true; FEET], 0.02);
        let (s, c) = (-0.2_f32).sin_cos();
        let mut after = [[0.0_f32; 2]; FEET];
        for slot in after.iter_mut() {
            *slot = [c * before[0][0] - s * before[0][1], s * before[0][0] + c * before[0][1]];
        }
        let p = odo.update(0.2, &after, &[true; FEET], 0.02).expect("a step");
        // Without the correction this reads 0.3 * 0.2 = 6 cm of phantom travel.
        assert!(p.x.abs() < 1e-5 && p.y.abs() < 1e-5, "a spin moved the pose: {p:?}");
    }

    #[test]
    fn cross_track_is_positive_to_the_left() {
        let origin = Pose2d::new(0.0, 0.0, 0.0);
        assert!((Pose2d::new(0.0, 0.5, 0.0).cross_track(&origin) - 0.5).abs() < 1e-6);
        let turned = Pose2d::new(0.0, 0.0, std::f32::consts::FRAC_PI_2);
        // Facing +y, left is -x.
        assert!((Pose2d::new(-0.5, 0.0, 0.0).cross_track(&turned) - 0.5).abs() < 1e-6);
    }
}
