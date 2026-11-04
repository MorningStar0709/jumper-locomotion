//! Foot positions in the body frame, from joint angles.
//!
//! [`odometry`](crate::odometry) needs to know where each foot is, expressed in the
//! body, at every control step. The robot sends joint angles; the model in
//! `assets/jumper/jumper.xml` says how they add up. `leg_table`
//! holds that chain, and this module runs it.
//!
//! The chain is serial for all six legs and every frame between the body and the
//! foot is unrotated, so a segment is a translation followed by one revolute joint:
//! the position of a frame is the previous frame's position plus the previous
//! rotation times the segment offset, and the rotation accumulates the joint.
//!
//! # Why the table is not typed by hand
//!
//! Every number in `leg_table.rs` was read out of the model, and the tests here
//! compare the chain against MuJoCo's own forward kinematics on four joint poses.
//! A model change that the table does not follow therefore fails a test rather than
//! leaving the controller walking on a stale leg.
//!
//! # Which feet are on the ground
//!
//! Nothing in [`RobotState`](crate::types::RobotState) says so; there are no contact
//! sensors in the contract. [`planted_from_height`] is the honest substitute: the
//! feet within a tolerance of the lowest one are the ones bearing weight. It is a
//! heuristic and it is written down as one -- on a slope it reads the downhill feet
//! as planted, and it cannot see a foot that has landed but not yet loaded.

use crate::leg_table::{LegTable, Segment, LEGS};

/// The six legs, in the order [`FootKinematics::feet`] returns them.
///
/// Must equal `leg_table::LEGS` by name and order; a test holds the two together,
/// because a reorder there would relabel every foot here without a compile error.
pub const FOOT_NAMES: [&str; 6] = ["LF", "RF", "LM", "RM", "LR", "RR"];

/// How close to the lowest foot another foot has to be to count as planted.
pub const PLANTED_TOLERANCE_M: f32 = 0.02;

/// Foot kinematics resolved against one robot's wire order.
///
/// Building it resolves each chain joint to its index in `q`, once, so a step costs
/// no string comparisons. A joint the wire does not carry resolves to `None` and the
/// whole thing is then reported as unusable, because a chain with a missing joint
/// would silently return the rest pose for that leg.
#[derive(Debug, Clone)]
pub struct FootKinematics {
    /// Wire index of each chain segment's joint, `None` where the segment has none.
    index: Vec<Vec<Option<usize>>>,
    usable: bool,
}

impl FootKinematics {
    pub fn new(joint_names: &[String]) -> Self {
        let index: Vec<Vec<Option<usize>>> = LEGS
            .iter()
            .map(|leg| {
                leg.chain
                    .iter()
                    .map(|segment| {
                        segment
                            .joint
                            .and_then(|name| joint_names.iter().position(|n| n == name))
                    })
                    .collect()
            })
            .collect();
        let usable = LEGS
            .iter()
            .zip(&index)
            .all(|(leg, idx)| leg.chain.iter().zip(idx).all(|(s, i)| s.joint.is_none() || i.is_some()));
        Self { index, usable }
    }

    /// Whether every chain joint was found on the wire.
    pub fn usable(&self) -> bool {
        self.usable
    }

    /// The six feet in the body frame, in [`FOOT_NAMES`] order.
    ///
    /// Gives `NaN` for a leg whose joints are missing from `q`, because a short
    /// reading is not "the joint is at zero": that would hand back a plausible foot
    /// at the rest pose, and a fabricated measurement is worse than none.
    /// [`usable`](Self::usable) says only that the *contract* names every joint.
    pub fn feet(&self, q: &[f32]) -> [[f32; 3]; 6] {
        let mut out = [[0.0f32; 3]; 6];
        for (leg_i, leg) in LEGS.iter().enumerate() {
            out[leg_i] = self.one_leg(leg, &self.index[leg_i], q);
        }
        out
    }

    fn one_leg(&self, leg: &LegTable, index: &[Option<usize>], q: &[f32]) -> [f32; 3] {
        let mut position = [0.0f32; 3];
        let mut rotation = [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        for (segment, idx) in leg.chain.iter().zip(index) {
            position = add(position, mul(rotation, segment.offset));
            match (segment, idx) {
                (Segment { axis, .. }, Some(i)) => {
                    let angle = q.get(*i).copied().unwrap_or(f32::NAN);
                    rotation = mulm(rotation, axis_angle(*axis, angle));
                }
                // The contract does not name this joint, so this chain cannot be
                // evaluated at all. Skipping the rotation would return a foot at the
                // rest pose -- the fabrication this function exists to refuse.
                // A segment with no joint is a fixed link: nothing to rotate.
                (Segment { joint: None, .. }, _) => {}
                // A segment that has a joint the contract does not name is a chain
                // this function cannot evaluate. Returning the rest pose here would be
                // the fabrication it exists to refuse.
                (Segment { joint: Some(_), .. }, None) => return [f32::NAN; 3],
            }
        }
        add(position, mul(rotation, leg.site_offset))
    }
}

/// The feet within [`PLANTED_TOLERANCE_M`] of the lowest one, in [`FOOT_NAMES`] order.
///
/// Calls the foot's body-frame `z` its height above the floor, which is only true when
/// the base is level. The two axes differ by an order of magnitude, because the feet
/// span ~0.10 m front to rear and ~0.56 m left to right:
///
/// - pitch: 3° over 0.10 m moves a foot ~5 mm in body `z`, well inside the tolerance,
///   so a nose-up stance still reads every foot as planted;
/// - roll: 3° over 0.56 m moves it ~3 cm, past the tolerance, so on a bank the uphill
///   feet are dropped, the downhill ones are believed, and the error points downhill.
///
/// A model with attitude in it (or contact sensors) is what would remove the roll case;
/// nothing here can.
pub fn planted_from_height(feet: &[[f32; 3]; 6]) -> [bool; 6] {
    planted_within(feet, PLANTED_TOLERANCE_M)
}

pub fn planted_within(feet: &[[f32; 3]; 6], tolerance: f32) -> [bool; 6] {
    let lowest = feet
        .iter()
        .map(|f| f[2])
        .fold(f32::INFINITY, |a, b| if b < a { b } else { a });
    let mut out = [false; 6];
    for (i, foot) in feet.iter().enumerate() {
        out[i] = foot[2] <= lowest + tolerance;
    }
    out
}

/// The feet as the planar positions [`odometry`](crate::odometry) integrates.
pub fn feet_xy(feet: &[[f32; 3]; 6]) -> [[f32; 2]; 6] {
    let mut out = [[0.0_f32; 2]; 6];
    for (i, f) in feet.iter().enumerate() {
        out[i] = [f[0], f[1]];
    }
    out
}

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn mul(m: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn mulm(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut out = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

fn axis_angle(axis: [f32; 3], angle: f32) -> [[f32; 3]; 3] {
    let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    if n <= f32::EPSILON {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let (x, y, z) = (axis[0] / n, axis[1] / n, axis[2] / n);
    let (s, c) = (angle.sin(), angle.cos());
    let t = 1.0 - c;
    [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    // Recorded from MuJoCo over the same joint poses: `docs` in `leg_table.rs` names
    // the model these came from.
    include!("leg_samples.rs");

    fn kinematics() -> FootKinematics {
        let names: Vec<String> = JOINT_NAMES.iter().map(|s| s.to_string()).collect();
        let kin = FootKinematics::new(&names);
        assert!(kin.usable(), "the model's own joint names do not resolve");
        kin
    }

    #[test]
    fn the_chain_agrees_with_mujoco_at_the_rest_pose_and_three_random_poses() {
        let kin = kinematics();
        for (n, (q, expected)) in [SAMPLE_0, SAMPLE_1, SAMPLE_2, SAMPLE_3].iter().enumerate() {
            let got = kin.feet(q);
            for (leg, (g, e)) in got.iter().zip(expected).enumerate() {
                for axis in 0..3 {
                    assert!(
                        (g[axis] - e[axis]).abs() < 2e-4,
                        "sample {n} leg {leg} axis {axis}: chain {g:?} vs mujoco {e:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_joint_missing_from_the_wire_leaves_the_chain_unusable() {
        let names: Vec<String> = JOINT_NAMES
            .iter()
            .filter(|n| **n != "LM_J1_joint")
            .map(|s| s.to_string())
            .collect();
        assert!(!FootKinematics::new(&names).usable());
        // And the chain itself must not quietly hand back a foot: `feet()` is public
        // and a caller outside this crate can call it without consulting `usable`.
        let kin = FootKinematics::new(&names);
        let q = SAMPLE_0.0;
        let feet = kin.feet(&q);
        // The missing joint is LM_J1, so LM is the leg that must refuse -- not every
        // leg, which would also pass if `feet()` simply broke.
        assert!(
            feet.iter().flatten().any(|v| !v.is_finite()),
            "a leg whose joint the contract omits came back as a plausible foot: {feet:?}"
        );
        let lm = FOOT_NAMES.iter().position(|n| *n == "LM").expect("LM is a foot");
        assert!(
            feet[lm].iter().any(|v| !v.is_finite()),
            "the leg missing a joint is LM, and it came back finite: {:?}",
            feet[lm]
        );
        let others: Vec<[f32; 3]> = feet
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != lm)
            .map(|(_, f)| *f)
            .collect();
        assert!(
            others.iter().flatten().all(|v| v.is_finite()),
            "the other five legs are complete and must still be measured: {others:?}"
        );
    }

    #[test]
    fn a_short_q_gives_nothing_rather_than_the_rest_pose() {
        // A reading that is missing joints is not "those joints are at zero": a foot
        // at the rest pose would look like a measurement, and the caller has no way
        // to tell. NaN is what the caller rejects.
        let kin = kinematics();
        let feet = kin.feet(&[0.0; 4]);
        assert!(
            feet.iter().flatten().any(|v| !v.is_finite()),
            "a short q produced plausible feet: {feet:?}"
        );
    }

    #[test]
    fn the_foot_names_are_the_leg_tables_names_in_the_same_order() {
        // Nothing at compile time keeps these two lists together, and a reorder in
        // `LEGS` would relabel every foot without a single error.
        let from_table: Vec<&str> = crate::leg_table::LEGS.iter().map(|l| l.name).collect();
        assert_eq!(from_table, FOOT_NAMES.to_vec());
        // The site is where the chain ends; it carries the same label as the leg,
        // which is what lets `FootKinematics` report a foot without a second table.
        let sites: Vec<&str> = crate::leg_table::LEGS.iter().map(|l| l.site).collect();
        assert_eq!(sites, FOOT_NAMES.to_vec());
    }

    #[test]
    fn standing_on_a_flat_floor_plants_all_six() {
        let feet = [[0.3, 0.0, -0.01], [-0.3, 0.0, -0.01], [0.0, 0.1, -0.01],
                    [0.0, -0.1, -0.01], [0.0, 0.2, -0.01], [0.0, -0.2, -0.01]];
        assert_eq!(planted_from_height(&feet), [true; 6]);
    }

    #[test]
    fn a_lifted_foot_is_not_planted() {
        let mut feet = [[0.3, 0.0, -0.01], [-0.3, 0.0, -0.01], [0.0, 0.1, -0.01],
                        [0.0, -0.1, -0.01], [0.0, 0.2, -0.01], [0.0, -0.2, -0.01]];
        feet[2][2] = 0.05;
        let planted = planted_from_height(&feet);
        assert!(!planted[2], "a foot 5 cm up counted as planted");
        assert!(planted[0] && planted[1] && planted[3]);
    }

    #[test]
    fn the_rest_pose_feet_sit_below_and_outside_the_body() {
        let kin = kinematics();
        let feet = kin.feet(&[0.0; 22]);
        for (i, foot) in feet.iter().enumerate() {
            assert!(foot[2] < 0.0, "leg {i} rest foot is not below the body: {foot:?}");
            assert!(foot[0].abs() + foot[1].abs() > 0.2, "leg {i} rest foot is under the body: {foot:?}");
        }
    }
}