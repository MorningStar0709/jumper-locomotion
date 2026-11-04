//! The kinematic chain from the body to each foot, read off `assets/jumper/jumper.xml`.
//!
//! Generated once from the model rather than typed: every offset below is the `pos` of a
//! body in its parent's frame, every axis is that body's joint axis, and none of the
//! intermediate frames is rotated relative to its parent (all quaternions are identity).
//! A test compares the result of the chain against MuJoCo's own forward kinematics, so a
//! change to the model that this table does not follow fails rather than drifts.

/// One body on the way from the body frame down to a foot.
pub struct Segment {
    /// Position of this body's frame in its parent's frame, metres.
    pub offset: [f32; 3],
    /// Rotation axis of the joint at this body, in this body's frame.
    pub axis: [f32; 3],
    /// Joint name in the robot's model, or `None` on the last body before the site.
    pub joint: Option<&'static str>,
}

pub struct LegTable {
    /// The foot this leg is, in `FOOT_NAMES` order. Checked in `legs.rs` against
    /// `FOOT_NAMES`, so a reorder in the generated table cannot silently relabel a foot.
    #[allow(dead_code)]
    pub name: &'static str,
    /// The MuJoCo site the chain ends at. Read by the table's self-check in
    /// `legs.rs`; the runtime uses `chain` and `site_offset` alone.
    #[allow(dead_code)]
    pub site: &'static str,
    pub chain: &'static [Segment],
    /// The foot site's position in the last chain body's frame.
    pub site_offset: [f32; 3],
}

pub const LF: LegTable = LegTable {
    name: "LF",
    site: "LF",
    chain: &[
        Segment { offset: [0.05067610, 0.06100090, 0.01070000], axis: [0.00000000, 0.00000000, 1.00000000], joint: Some("LF_J0_joint") },
        Segment { offset: [0.00000000, 0.00640000, -0.03116000], axis: [0.00000000, 1.00000000, 0.00000000], joint: Some("LF_J1_joint") },
        Segment { offset: [0.00090000, 0.04787000, 0.00011132], axis: [1.00000000, 0.00000000, 0.00000000], joint: Some("LF_J2_joint") },
        Segment { offset: [-0.00120000, 0.05000000, 0.00000000], axis: [1.00000000, 0.00000000, 0.00000000], joint: Some("LF_J3_joint") },
        Segment { offset: [-0.00740201, 0.14154600, -0.00058817], axis: [0.00000000, 0.00000000, 0.00000000], joint: None },
    ],
    site_offset: [-0.00647599, -0.02811370, 0.01025910],
};

pub const RF: LegTable = LegTable {
    name: "RF",
    site: "RF",
    chain: &[
        Segment { offset: [0.05067600, -0.06099900, 0.01070000], axis: [0.00000000, 0.00000000, 1.00000000], joint: Some("RF_J0_joint") },
        Segment { offset: [0.00000000, -0.00640000, -0.03116000], axis: [0.00000000, 1.00000000, 0.00000000], joint: Some("RF_J1_joint") },
        Segment { offset: [-0.00090000, -0.04787000, 0.00011132], axis: [1.00000000, 0.00000000, 0.00000000], joint: Some("RF_J2_joint") },
        Segment { offset: [0.00120000, -0.05000000, 0.00000000], axis: [1.00000000, 0.00000000, 0.00000000], joint: Some("RF_J3_joint") },
        Segment { offset: [-0.00880304, -0.14154600, -0.00058832], axis: [0.00000000, 0.00000000, 0.00000000], joint: None },
    ],
    site_offset: [-0.00647574, 0.02811340, 0.01025810],
};

pub const LM: LegTable = LegTable {
    name: "LM",
    site: "LM",
    chain: &[
        Segment { offset: [-0.00732390, 0.06650100, 0.00745000], axis: [0.00000000, 0.00000000, 1.00000000], joint: Some("LM_J0_joint") },
        Segment { offset: [0.00110010, 0.05613000, -0.02493000], axis: [1.00000000, 0.00000000, 0.00000000], joint: Some("LM_J1_joint") },
        Segment { offset: [-0.00119991, 0.05000000, 0.00000000], axis: [1.00000000, 0.00000000, 0.00000000], joint: Some("LM_J2_joint") },
        Segment { offset: [0.00078931, 0.12091700, 0.00624036], axis: [0.00000000, 0.00000000, 0.00000000], joint: None },
    ],
    site_offset: [-0.00000274, -0.01049010, -0.00482261],
};

pub const RM: LegTable = LegTable {
    name: "RM",
    site: "RM",
    chain: &[
        Segment { offset: [-0.00732390, -0.06649900, 0.00745000], axis: [0.00000000, 0.00000000, 1.00000000], joint: Some("RM_J0_joint") },
        Segment { offset: [-0.00110000, -0.05613000, -0.02493000], axis: [1.00000000, 0.00000000, 0.00000000], joint: Some("RM_J1_joint") },
        Segment { offset: [0.00120000, -0.05000000, 0.00000000], axis: [1.00000000, 0.00000000, 0.00000000], joint: Some("RM_J2_joint") },
        Segment { offset: [-0.00078909, -0.12091700, 0.00624036], axis: [0.00000000, 0.00000000, 0.00000000], joint: None },
    ],
    site_offset: [0.00000271, 0.01049010, -0.00482261],
};

pub const LR: LegTable = LegTable {
    name: "LR",
    site: "LR",
    chain: &[
        Segment { offset: [-0.06532400, 0.05550100, 0.00745000], axis: [0.00000000, 0.00000000, 1.00000000], joint: Some("LR_J0_joint") },
        Segment { offset: [0.00110010, 0.05613000, -0.02493000], axis: [1.00000000, 0.00000000, 0.00000000], joint: Some("LR_J1_joint") },
        Segment { offset: [-0.00119991, 0.05000000, 0.00000000], axis: [1.00000000, 0.00000000, 0.00000000], joint: Some("LR_J2_joint") },
        Segment { offset: [0.00078931, 0.12091700, 0.00624036], axis: [0.00000000, 0.00000000, 0.00000000], joint: None },
    ],
    site_offset: [-0.00000274, -0.01049010, -0.00482261],
};

pub const RR: LegTable = LegTable {
    name: "RR",
    site: "RR",
    chain: &[
        Segment { offset: [-0.06532390, -0.05549910, 0.00745000], axis: [0.00000000, 0.00000000, 1.00000000], joint: Some("RR_J0_joint") },
        Segment { offset: [-0.00110000, -0.05613000, -0.02493000], axis: [1.00000000, 0.00000000, 0.00000000], joint: Some("RR_J1_joint") },
        Segment { offset: [0.00120000, -0.05000000, 0.00000000], axis: [1.00000000, 0.00000000, 0.00000000], joint: Some("RR_J2_joint") },
        Segment { offset: [-0.00078909, -0.12091700, 0.00624036], axis: [0.00000000, 0.00000000, 0.00000000], joint: None },
    ],
    site_offset: [0.00000271, 0.01049010, -0.00482261],
};

pub const LEGS: [&LegTable; 6] = [&LF, &RF, &LM, &RM, &LR, &RR];
