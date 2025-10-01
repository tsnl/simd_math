//--------------------------------------------------------------------------------------------------
// SimdTransform:
//--------------------------------------------------------------------------------------------------

use crate::quaternion::SimdUnitQuat;
use crate::vector::SimdVec3;
use std::ops::Mul;

#[derive(Clone, Copy, Default)]
pub struct SimdTransform {
    position: SimdVec3,
    rotation: SimdUnitQuat,
}

impl SimdTransform {
    /// Create a new transform with the given translation and rotation.
    #[inline]
    pub const fn new(translation: SimdVec3, rotation: SimdUnitQuat) -> Self {
        SimdTransform {
            position: translation,
            rotation,
        }
    }

    /// Create a new identity transform (no translation, no rotation).
    #[inline]
    pub fn identity() -> Self {
        SimdTransform::default()
    }

    #[inline]
    pub fn position(&self) -> SimdVec3 {
        self.position
    }

    #[inline]
    pub fn rotation(&self) -> SimdUnitQuat {
        self.rotation
    }

    #[inline]
    pub fn inverse(self) -> Self {
        // The inverse of a transform is the inverse of the rotation and the negative of the
        // rotated position.
        let inv_rotation = self.rotation.inverse();
        let inv_position = -(inv_rotation * self.position);

        SimdTransform {
            position: inv_position,
            rotation: inv_rotation,
        }
    }
}

impl Mul<SimdTransform> for SimdTransform {
    type Output = SimdTransform;

    /// Compose two transforms: T1 * T2 = T1(T2(v)) for any vector3 v.
    fn mul(self, rhs: SimdTransform) -> Self::Output {
        // Apply the rotation of the first transform to the translation of the second,
        // then apply the rotation of the first to the rotation of the second.
        let new_translation = self.position + self.rotation * rhs.position;
        let new_rotation = self.rotation * rhs.rotation;

        SimdTransform {
            position: new_translation,
            rotation: new_rotation,
        }
    }
}

impl Mul<SimdVec3> for SimdTransform {
    type Output = SimdVec3;

    /// Apply the transform to a vector3: v' = T(v) = R(v) + T
    fn mul(self, rhs: SimdVec3) -> Self::Output {
        // Rotate the vector by the rotation quaternion and then translate it.
        self.rotation * rhs + self.position
    }
}

#[cfg(test)]
mod simd_transform_tests {
    use super::*;
    use std::f32::consts::PI;

    const EPSILON: f32 = 1e-6;

    fn assert_vec3_eq(a: SimdVec3, b: SimdVec3) {
        let ax = a.x();
        let bx = b.x();
        assert!((ax - bx).abs() < EPSILON, "x: {ax} != {bx}");
        let ay = a.y();
        let by = b.y();
        assert!((ay - by).abs() < EPSILON, "y: {ay} != {by}");
        let az = a.z();
        let bz = b.z();
        assert!((az - bz).abs() < EPSILON, "z: {az} != {bz}");
    }

    fn assert_quat_eq(a: SimdUnitQuat, b: SimdUnitQuat) {
        for i in 0..4 {
            let val_a = a.0[i];
            let val_b = b.0[i];
            assert!(
                (val_a - val_b).abs() < EPSILON,
                "component {i}: {val_a} != {val_b}"
            );
        }
    }

    fn assert_transform_eq(a: SimdTransform, b: SimdTransform) {
        assert_vec3_eq(a.position, b.position);
        assert_quat_eq(a.rotation, b.rotation);
    }

    #[test]
    fn test_transform_identity_default() {
        let t_ident = SimdTransform::identity();
        let t_default = SimdTransform::default();
        assert_transform_eq(t_ident, t_default);
        assert_vec3_eq(t_ident.position, SimdVec3::default());
        assert_quat_eq(t_ident.rotation, SimdUnitQuat::default());
    }

    #[test]
    fn test_transform_new() {
        let trans = SimdVec3::new(1.0, 2.0, 3.0);
        let rot = SimdUnitQuat::from_axis_angle(SimdVec3::new(0.0, 0.0, 1.0), PI / 2.0);
        let t = SimdTransform::new(trans, rot);
        assert_vec3_eq(t.position, trans);
        assert_quat_eq(t.rotation, rot);
    }

    #[test]
    fn test_transform_mul_transform() {
        let t_ident = SimdTransform::identity();

        let trans1 = SimdVec3::new(1.0, 2.0, 3.0);
        let rot1 = SimdUnitQuat::from_axis_angle(SimdVec3::new(0.0, 0.0, 1.0), PI / 2.0); // 90 deg Z rot
        let tf1 = SimdTransform::new(trans1, rot1);

        // Identity
        assert_transform_eq(tf1 * t_ident, tf1);
        assert_transform_eq(t_ident * tf1, tf1);

        let trans2 = SimdVec3::new(10.0, 0.0, 0.0);
        let rot2 = SimdUnitQuat::from_axis_angle(SimdVec3::new(1.0, 0.0, 0.0), PI); // 180 deg X rot
        let tf2 = SimdTransform::new(trans2, rot2);

        // Composition: T1 * T2
        // new_rotation = rot1 * rot2
        // new_translation = trans1 + rot1 * trans2
        let composed_tf = tf1 * tf2;

        let expected_rot = rot1 * rot2;
        let expected_trans = trans1 + rot1 * trans2;
        assert_transform_eq(
            composed_tf,
            SimdTransform::new(expected_trans, expected_rot),
        );

        // Apply to a point (implicitly, by checking components)
        // Point p = (1,0,0)
        // T2(p): rot2 * p + trans2 = (180X * (1,0,0)) + (10,0,0) = (1,0,0) + (10,0,0) = (11,0,0)
        // T1(T2(p)): rot1 * (11,0,0) + trans1 = (90Z * (11,0,0)) + (1,2,3)
        //            = (0,11,0) + (1,2,3) = (1,13,3)

        // Let's verify the new_translation and new_rotation directly
        // rot1 * trans2: 90Z * (10,0,0) = (0,10,0)
        // trans1 + rot1*trans2 = (1,2,3) + (0,10,0) = (1,12,3)
        assert_vec3_eq(composed_tf.position, SimdVec3::new(1.0, 12.0, 3.0));

        // rot1: 90Z, rot2: 180X
        // (s1,0,0,z1) * (s2,x2,0,0) = [s1s2, s1x2, z1s2, z1x2] (simplified)
        // q_z90 = [cos(pi/4), 0, 0, sin(pi/4)]
        // q_x180 = [cos(pi/2), sin(pi/2), 0, 0] = [0, 1, 0, 0]
        // rot1 * rot2 = [0, cos(pi/4), 0, sin(pi/4)]
        // Actually, it's rot1 * rot2, so q_rot1 is [c,0,0,s] and q_rot2 is [0,1,0,0]
        // s_res = c*0 - 0*1 - 0*0 - s*0 = 0
        // x_res = c*1 + 0*0 + 0*0 - s*0 = c
        // y_res = c*0 + 0*0 + 0*0 + s*1 = s
        // z_res = c*0 + 0*1 - 0*0 + 0*0 = 0
        // So, the resulting quaternion is [0, c, s, 0]
        let c45 = (PI / 4.0).cos();
        let s45 = (PI / 4.0).sin();
        // let expected_composed_q = quat(0.0, c45, s45, 0.0); // This was for q_x * q_z, but it's q_z * q_x
        assert_quat_eq(composed_tf.rotation, SimdUnitQuat::new(0.0, c45, s45, 0.0));
    }

    #[test]
    fn test_transform_inverse_1() {
        let t = SimdTransform::identity();
        let inv_t = t.inverse();
        assert_transform_eq(t, inv_t); // Identity inverse is itself
    }

    #[test]
    fn test_transform_inverse_2() {
        let axis = SimdVec3::new(0.0, 0.0, 1.0);
        let angle = PI / 4.0;
        let t = SimdTransform::new(
            SimdVec3::new(1.0, 2.0, 3.0),
            SimdUnitQuat::from_axis_angle(axis, angle),
        );
        let inv_t = t.inverse();
        assert_transform_eq(t * inv_t, SimdTransform::default()); // Identity inverse is itself
    }
}
