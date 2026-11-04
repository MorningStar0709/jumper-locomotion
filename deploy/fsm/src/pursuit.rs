//! Path tracking: hold a straight line, not just a heading.
//!
//! The heading hold in [`crate::heading`] keeps the robot facing where it started,
//! which is enough while the ground does not move it sideways. It is not enough after
//! a slip or a shove: the robot faces the right way and is a metre off its line, and
//! nothing pulls it back. This loop closes that gap -- it aims at a point
//! `lookahead` metres ahead on the line and turns towards it.
//!
//! The line is latched from the pose at the moment the robot starts walking, the same
//! moment the heading hold takes its anchor. There is no input channel for "aim there"
//! yet; re-latching mid-walk needs a control the repository does not have, and inventing
//! one here would put a name in a contract that no host can press.

use crate::types::Command;

/// Below this planar speed the robot is standing, and the line is dropped rather than
/// held: a line latched while standing would drag the next walk back to it. Taken from
/// the heading hold so the two loops cannot drift apart on what "walking" means.
pub use crate::heading::STANDING_BELOW;

/// How far ahead on the line the loop aims, in metres. Shorter tracks the line harder
/// and oscillates when the position estimate is noisy; longer cuts corners.
pub const LOOKAHEAD_M: f32 = 1.0;

/// A straight line, latched from the pose at the first walking step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Line {
    pub x: f32,
    pub y: f32,
    pub yaw: f32,
}

impl Line {
    /// Signed distance from the line, positive to the left of the heading it was
    /// latched at.
    pub fn cross_track(&self, p: [f32; 2]) -> f32 {
        let (sin, cos) = self.yaw.sin_cos();
        -(p[0] - self.x) * sin + (p[1] - self.y) * cos
    }

    /// The point `ahead` metres along the line from the robot's projection onto it.
    pub fn aim(&self, p: [f32; 2], ahead: f32) -> [f32; 2] {
        let (sin, cos) = self.yaw.sin_cos();
        let along = (p[0] - self.x) * cos + (p[1] - self.y) * sin;
        [self.x + (along + ahead) * cos, self.y + (along + ahead) * sin]
    }
}

/// The loop that turns a pose and a commanded velocity into a yaw rate.
#[derive(Debug, Clone)]
pub struct Pursuit {
    kp: f32,
    max_rate: f32,
    yaw: [f32; 2],
    lookahead: f32,
    line: Option<Line>,
}

impl Pursuit {
    pub fn new(kp: f32, max_rate: f32, yaw: [f32; 2], lookahead: f32) -> Self {
        assert!(
            yaw[0].is_finite() && yaw[1].is_finite() && yaw[0] < yaw[1],
            "Pursuit needs a finite band, got {yaw:?}"
        );
        assert!(
            yaw[0] < 0.0 && 0.0 < yaw[1],
            "Pursuit needs a band with room on both sides of zero, got {yaw:?}"
        );
        assert!(lookahead > 0.0 && lookahead.is_finite(), "lookahead must be positive, got {lookahead}");
        assert!(
            kp > 0.0 && kp.is_finite(),
            "Pursuit needs a finite, positive gain, got {kp}"
        );
        assert!(
            max_rate > 0.0 && max_rate.is_finite(),
            "Pursuit needs a finite, positive rate limit, got {max_rate}"
        );
        Self { kp, max_rate, yaw, lookahead, line: None }
    }

    /// Forgets the line: the next walking step latches a new one from its own pose.
    pub fn reset(&mut self) {
        self.line = None;
    }

    pub fn line(&self) -> Option<Line> {
        self.line
    }

    /// One step. `None` while standing (nothing to hold), otherwise the yaw rate to
    /// send: the operator's request plus a correction towards the line.
    pub fn step(&mut self, pose: [f32; 3], command: &Command) -> Option<f32> {
        let walking = (command.lin_vel_x * command.lin_vel_x
            + command.lin_vel_y * command.lin_vel_y)
            .sqrt();
        if !walking.is_finite() || walking <= STANDING_BELOW {
            self.line = None;
            return None;
        }
        if !command.yaw_rate.is_finite() {
            return None;
        }
        let p = [pose[0], pose[1]];
        let line = match self.line {
            Some(line) => line,
            None => {
                let line = Line { x: p[0], y: p[1], yaw: pose[2] };
                self.line = Some(line);
                return Some(self.inside(command.yaw_rate));
            }
        };
        // Walking backwards inverts the steering: a yaw rate that turns the nose
        // towards the line drives a backwards-moving robot away from it. The aim
        // point stays ahead of the *heading*; the correction's sign follows the
        // direction of travel.
        let aim = line.aim(p, self.lookahead);
        let sign = if command.lin_vel_x < 0.0 { -1.0 } else { 1.0 };
        let (dx, dy) = (aim[0] - p[0], aim[1] - p[1]);
        if !dx.is_finite() || !dy.is_finite() || (dx == 0.0 && dy == 0.0) {
            return Some(self.inside(command.yaw_rate));
        }
        let error = crate::heading::wrap(dy.atan2(dx) - pose[2]);
        let correction = (sign * self.kp * error).clamp(-self.max_rate, self.max_rate);
        Some(self.inside(command.yaw_rate + correction))
    }

    fn inside(&self, rate: f32) -> f32 {
        rate.clamp(self.yaw[0], self.yaw[1])
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    fn walking(vx: f32, vy: f32, wz: f32) -> Command {
        Command { lin_vel_x: vx, lin_vel_y: vy, yaw_rate: wz, ..Command::default() }
    }

    fn at(x: f32, y: f32, yaw: f32) -> [f32; 3] {
        [x, y, yaw]
    }

    #[test]
    fn a_robot_on_its_line_along_it_is_not_corrected() {
        let mut p = Pursuit::new(2.0, 0.5, [-0.75, 0.75], LOOKAHEAD_M);
        p.step(at(0.0, 0.0, 0.0), &walking(0.5, 0.0, 0.0));
        let wz = p.step(at(0.4, 0.0, 0.0), &walking(0.5, 0.0, 0.0)).expect("walking");
        assert!(wz.abs() < 1e-6, "a lined-up robot was corrected: {wz}");
    }

    #[test]
    fn a_robot_left_of_its_line_turns_right() {
        let mut p = Pursuit::new(2.0, 0.5, [-0.75, 0.75], LOOKAHEAD_M);
        p.step(at(0.0, 0.0, 0.0), &walking(0.5, 0.0, 0.0));
        // 0.2 m to the left: negative yaw rate is a right turn.
        let wz = p.step(at(0.4, 0.2, 0.0), &walking(0.5, 0.0, 0.0)).expect("walking");
        assert!(wz < -0.05, "left of the line should turn right: {wz}");
    }

    #[test]
    fn facing_left_of_the_line_turns_right() {
        let mut p = Pursuit::new(2.0, 0.5, [-0.75, 0.75], LOOKAHEAD_M);
        p.step(at(0.0, 0.0, 0.0), &walking(0.5, 0.0, 0.0));
        let wz = p.step(at(0.4, 0.0, 0.2), &walking(0.5, 0.0, 0.0)).expect("walking");
        assert!(wz < -0.05, "facing left should turn right: {wz}");
    }

    #[test]
    fn walking_backwards_does_not_invert_the_correction() {
        // The aim point has to follow the direction of travel. With the aim ahead of
        // the *heading* while the robot walks backwards, a robot left of its line is
        // steered further left -- the loop diverges instead of converging.
        let mut p = Pursuit::new(2.0, 0.5, [-0.75, 0.75], 1.0);
        let back = Command { lin_vel_x: -0.4, ..Command::default() };
        p.step([0.0, 0.0, 0.0], &back);
        let wz = p.step([0.0, 0.3, 0.0], &back).expect("walking");
        assert!(wz > 0.0, "left of the line while walking backwards must turn left, got {wz}");
    }

    #[test]
    fn a_non_finite_request_never_reaches_the_policy() {
        let mut p = Pursuit::new(2.0, 0.5, [-0.75, 0.75], 1.0);
        let nan = Command { lin_vel_x: 0.4, yaw_rate: f32::NAN, ..Command::default() };
        assert_eq!(p.step([0.0, 0.0, 0.0], &nan), None, "a NaN request must be dropped");
    }

    #[test]
    #[should_panic(expected = "room on both sides of zero")]
    fn a_one_sided_band_is_refused_at_construction() {
        Pursuit::new(2.0, 0.5, [0.0, 0.75], 1.0);
    }

    #[test]
    fn standing_drops_the_line() {
        let mut p = Pursuit::new(2.0, 0.5, [-0.75, 0.75], LOOKAHEAD_M);
        p.step(at(0.0, 0.0, 0.0), &walking(0.5, 0.0, 0.0));
        assert!(p.line().is_some());
        assert!(p.step(at(0.4, 0.2, 0.0), &walking(0.0, 0.0, 0.0)).is_none());
        assert!(p.line().is_none(), "a standing robot kept its line");
    }

    #[test]
    fn the_line_is_latched_from_the_first_walking_pose() {
        let mut p = Pursuit::new(2.0, 0.5, [-0.75, 0.75], LOOKAHEAD_M);
        // Started 3 m away and turned 0.5 rad: the line goes through there.
        p.step(at(3.0, 1.0, 0.5), &walking(0.5, 0.0, 0.0));
        let line = p.line().expect("latching is part of the first step");
        assert!((line.x - 3.0).abs() < 1e-6 && (line.y - 1.0).abs() < 1e-6);
        assert!((line.yaw - 0.5).abs() < 1e-6);
    }

    #[test]
    fn the_request_is_passed_through_and_the_sum_stays_inside_the_axis() {
        let mut p = Pursuit::new(4.0, 0.5, [-0.75, 0.75], 0.3);
        p.step(at(0.0, 0.0, 0.0), &walking(0.5, 0.0, 0.0));
        // A hard-left error against a right request: the sum would be 1.5 rad/s.
        let wz = p.step(at(0.0, 0.2, 0.0), &walking(0.5, 0.0, -0.7)).expect("walking");
        assert!(wz >= -0.75 - 1e-6, "left the trained axis: {wz}");
        // The correction alone is capped at max_rate, and the request rides on top.
        let mut q = Pursuit::new(4.0, 0.5, [-2.0, 2.0], 0.3);
        q.step(at(0.0, 0.0, 0.0), &walking(0.5, 0.0, 0.0));
        let wz = q.step(at(1.0, 0.0, 1.2), &walking(0.5, 0.0, 0.0)).expect("walking");
        assert!((wz + 0.5).abs() < 1e-3, "the correction was not capped: {wz}");
    }

    #[test]
    fn the_loop_pulls_a_displaced_robot_back_to_its_line() {
        // Closed loop with a toy plant: the robot walks along its heading and answers
        // the yaw rate with a lag, and starts 0.4 m off the line it was told to keep.
        let mut p = Pursuit::new(2.0, 0.6, [-0.75, 0.75], LOOKAHEAD_M);
        let mut pose = at(0.0, 0.4, 0.0);
        let dt = 0.02f32;
        let mut last = p.step(pose, &walking(0.5, 0.0, 0.0)).unwrap_or(0.0);
        for _ in 0..1500 {
            pose[2] += last * dt;
            pose[0] += 0.5 * pose[2].cos() * dt;
            pose[1] += 0.5 * pose[2].sin() * dt;
            last = p.step(pose, &walking(0.5, 0.0, 0.0)).expect("walking");
        }
        let off = p.line().expect("a line").cross_track([pose[0], pose[1]]).abs();
        assert!(off < 0.05, "did not return to the line: {off} m off");
        assert!(pose[2].abs() < 0.1, "returned but not along it: {} rad", pose[2]);
    }
}
