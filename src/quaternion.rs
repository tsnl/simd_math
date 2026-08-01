use super::*;
use std::ops::Mul;

/// A unit (rotation) quaternion stored as an `f32x4` in `[s, x, y, z]` order,
/// where `s` is the scalar (real) part. Think `s + iv`.
///
/// IMPORTANT: the norm is assumed to always be 1. Constructors ensure this.
/// Reference: <https://www.3dgep.com/understanding-quaternions/#Rotations>
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimdUnitQuat(pub(crate) f32x4); // s, x, y, z

impl SimdUnitQuat {
    /// The identity (no-rotation) quaternion.
    pub const IDENTITY: Self = Self(f32x4::from_array([1.0, 0.0, 0.0, 0.0]));
}

impl Default for SimdUnitQuat {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl From<[f32; 4]> for SimdUnitQuat {
    /// Builds a unit quaternion from `[s, x, y, z]`, normalizing the input.
    ///
    /// Panics if the input cannot be normalized; see [`SimdUnitQuat::new`].
    #[inline]
    fn from(q: [f32; 4]) -> Self {
        SimdUnitQuat::new(q[0], q[1], q[2], q[3])
    }
}

impl From<SimdUnitQuat> for [f32; 4] {
    #[inline]
    fn from(q: SimdUnitQuat) -> Self {
        *q.0.as_array()
    }
}

impl SimdUnitQuat {
    /// Creates a unit quaternion from `(s, x, y, z)` components, normalizing
    /// the result.
    ///
    /// # Panics
    ///
    /// Panics if the components cannot be normalized into a unit quaternion
    /// (all zero, or large/non-finite enough that the norm is not a positive
    /// finite number).
    #[inline]
    pub fn new(s: f32, x: f32, y: f32, z: f32) -> Self {
        let q = f32x4::from_array([s, x, y, z]);
        let norm = (q * q).reduce_sum().sqrt();
        assert!(
            norm > 0.0 && norm.is_finite(),
            "cannot normalize the given components into a unit quaternion"
        );
        SimdUnitQuat(q / f32x4::splat(norm))
    }

    /// Creates a quaternion representing a rotation of `angle` radians around
    /// `axis` (right-hand rule). The axis does not need to be normalized.
    ///
    /// # Panics
    ///
    /// Panics if `axis` is the zero vector.
    #[inline]
    pub fn from_axis_angle(axis: SimdVec3, angle: f32) -> Self {
        // Normalize the axis vector
        let axis = axis.normalized().expect("Axis vector cannot be zero");
        let half_angle = angle * 0.5;
        let sin_half_angle = half_angle.sin();
        let real = half_angle.cos();
        let imag = axis.0 * f32x4::splat(sin_half_angle);
        SimdUnitQuat::new(real, imag[0], imag[1], imag[2])
    }
}

impl SimdUnitQuat {
    /// Conjugate: q* = [s, -v]
    #[inline]
    #[must_use]
    pub fn conjugate(self) -> Self {
        SimdUnitQuat(self.0 * f32x4::from_array([1.0, -1.0, -1.0, -1.0]))
    }

    /// Inverse: for unit quaternions, q* = q⁻¹
    #[inline]
    #[must_use]
    pub fn inverse(self) -> Self {
        self.conjugate()
    }

    /// The scalar (real) part of the quaternion.
    #[inline]
    #[must_use]
    pub fn real(self) -> f32 {
        self.0[0]
    }

    /// The vector (imaginary) part of the quaternion.
    #[inline]
    #[must_use]
    pub fn imag(self) -> SimdVec3 {
        SimdVec3(simd_swizzle!(self.0, [1, 2, 3, 0]) * f32x4::from_array([1.0, 1.0, 1.0, 0.0]))
    }

    /// Recovers the rotation axis (unit length) and angle (radians, in
    /// `[0, 2π]`) represented by this quaternion.
    ///
    /// For the identity quaternion the axis is arbitrary; this returns
    /// `(SimdVec3::UNIT_X, 0.0)`.
    #[must_use]
    pub fn to_axis_angle(self) -> (SimdVec3, f32) {
        let angle = 2.0 * self.real().clamp(-1.0, 1.0).acos();
        match self.imag().normalized() {
            Some(axis) => (axis, angle),
            None => (SimdVec3::UNIT_X, 0.0),
        }
    }

    /// Spherical linear interpolation from `a` (at `t = 0`) to `b`
    /// (at `t = 1`), always taking the shorter arc.
    ///
    /// Falls back to normalized linear interpolation when the quaternions are
    /// nearly parallel, where slerp is numerically unstable.
    #[must_use]
    pub fn slerp(a: Self, b: Self, t: f32) -> Self {
        let mut dot = (a.0 * b.0).reduce_sum();

        // q and -q represent the same rotation; pick the sign of b that gives
        // the shorter arc.
        let mut b = b.0;
        if dot < 0.0 {
            b = -b;
            dot = -dot;
        }

        if dot > 0.9995 {
            // Nearly parallel: nlerp to avoid dividing by a tiny sin(theta).
            let q = a.0 + (b - a.0) * f32x4::splat(t);
            return SimdUnitQuat(q / f32x4::splat((q * q).reduce_sum().sqrt()));
        }

        let theta = dot.clamp(-1.0, 1.0).acos();
        let sin_theta = theta.sin();
        let k_a = ((1.0 - t) * theta).sin() / sin_theta;
        let k_b = (t * theta).sin() / sin_theta;
        SimdUnitQuat(a.0 * f32x4::splat(k_a) + b * f32x4::splat(k_b))
    }
}

impl Mul<SimdUnitQuat> for SimdUnitQuat {
    type Output = SimdUnitQuat;

    #[inline]
    fn mul(self, rhs: SimdUnitQuat) -> Self::Output {
        // Quaternion multiplication: q1 * q2 = [s1*s2 - v1⋅v2, s1*v2 + s2*v1 + v1×v2]
        let q1 = self.0;
        let q2 = rhs.0;

        // Scalar part:
        //      s1*s2 - v1⋅v2
        //    = s1*s2 - x1*x2 - y1*y2 - z1*z2
        //    = [s1, x1, y1, z1] ⋅ [s2, -x2, -y2, -z2]
        let s = (q1 * q2 * f32x4::from_array([1.0, -1.0, -1.0, -1.0])).reduce_sum();

        // Vector part:
        //      s1*v2 + s2*v1 + v1×v2
        let v = rhs.imag() * self.real() + self.imag() * rhs.real() + self.imag().cross(rhs.imag());

        // Combine the two by summing the scalar and vector parts:
        SimdUnitQuat(f32x4::from_array([s, 0.0, 0.0, 0.0]) + simd_swizzle!(v.0, [3, 0, 1, 2]))
    }
}

impl Mul<SimdVec3> for SimdUnitQuat {
    type Output = SimdVec3;

    /// Rotates a vector by this quaternion.
    ///
    /// Uses the optimized form of the sandwich product q * v * q⁻¹:
    /// with `t = 2 (v_q × v)`, the result is `v + s t + v_q × t`, which is
    /// cheaper than performing two full quaternion multiplications.
    #[inline]
    fn mul(self, rhs: SimdVec3) -> Self::Output {
        let qv = self.imag();
        let t = qv.cross(rhs) * 2.0;
        rhs + t * self.real() + qv.cross(t)
    }
}

#[cfg(test)]
mod simd_unit_quat_tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_4, PI};

    const EPSILON: f32 = 1e-6;

    fn assert_f32_eq(a: f32, b: f32) {
        assert!((a - b).abs() < EPSILON, "{a} != {b}");
    }

    fn assert_vec3_eq(a: SimdVec3, b: SimdVec3) {
        let ax = a[0];
        let bx = b[0];
        assert!((ax - bx).abs() < EPSILON, "x: {ax} != {bx}");
        let ay = a[1];
        let by = b[1];
        assert!((ay - by).abs() < EPSILON, "y: {ay} != {by}");
        let az = a[2];
        let bz = b[2];
        assert!((az - bz).abs() < EPSILON, "z: {az} != {bz}");
    }

    fn assert_quat_eq(a: SimdUnitQuat, b: SimdUnitQuat) {
        // Compare backing f32x4 directly to avoid issues with imag() creating new Vec3s
        for i in 0..4 {
            let val_a = a.0[i];
            let val_b = b.0[i];
            assert!(
                (val_a - val_b).abs() < EPSILON,
                "component {i}: {val_a} != {val_b}"
            );
        }
    }

    #[test]
    fn test_quat_new_normalizes() {
        let q = SimdUnitQuat::new(2.0, 0.0, 0.0, 0.0); // Should be normalized to [1,0,0,0]
        assert_quat_eq(q, SimdUnitQuat(f32x4::from_array([1.0, 0.0, 0.0, 0.0])));

        let q2 = SimdUnitQuat::new(1.0, 2.0, 3.0, 4.0);
        let norm_sq = q2.0[0] * q2.0[0] + q2.0[1] * q2.0[1] + q2.0[2] * q2.0[2] + q2.0[3] * q2.0[3];
        assert_f32_eq(norm_sq, 1.0);
    }

    #[test]
    #[should_panic(expected = "cannot normalize")]
    fn test_quat_new_zero_panics() {
        SimdUnitQuat::new(0.0, 0.0, 0.0, 0.0);
    }

    #[test]
    fn test_quat_default_is_identity() {
        assert_quat_eq(
            SimdUnitQuat::default(),
            SimdUnitQuat::new(1.0, 0.0, 0.0, 0.0),
        );
    }

    #[test]
    fn test_quat_from_axis_angle() {
        // 90 degree rotation around Z axis
        let axis = SimdVec3::from([0.0, 0.0, 1.0]);
        let angle = PI / 2.0;
        let q = SimdUnitQuat::from_axis_angle(axis, angle);
        // s = cos(pi/4), x=0, y=0, z = sin(pi/4)
        let expected_s = (PI / 4.0).cos();
        let expected_z = (PI / 4.0).sin();
        assert_quat_eq(q, SimdUnitQuat::new(expected_s, 0.0, 0.0, expected_z));

        // 180 degree rotation around Y axis
        let axis = SimdVec3::from([0.0, 1.0, 0.0]);
        let angle = PI;
        let q = SimdUnitQuat::from_axis_angle(axis, angle);
        // s = cos(pi/2) = 0, x=0, y=sin(pi/2)=1, z=0
        assert_quat_eq(q, SimdUnitQuat::new(0.0, 0.0, 1.0, 0.0));
    }

    #[test]
    #[should_panic]
    fn test_quat_from_axis_angle_zero_axis() {
        SimdUnitQuat::from_axis_angle(SimdVec3::from([0.0, 0.0, 0.0]), PI / 2.0);
    }

    #[test]
    fn test_quat_to_axis_angle_roundtrip() {
        let axis = SimdVec3::from([1.0, 2.0, 3.0]).normalized().unwrap();
        let angle = 1.234;
        let (recovered_axis, recovered_angle) =
            SimdUnitQuat::from_axis_angle(axis, angle).to_axis_angle();
        assert_vec3_eq(recovered_axis, axis);
        assert_f32_eq(recovered_angle, angle);

        // Identity has no rotation axis; angle must be zero.
        let (_, identity_angle) = SimdUnitQuat::IDENTITY.to_axis_angle();
        assert_f32_eq(identity_angle, 0.0);
    }

    #[test]
    fn test_quat_conjugate_inverse() {
        let q = SimdUnitQuat::from_axis_angle(SimdVec3::from([1.0, 2.0, 3.0]), PI / 3.0);
        let q_conj = q.conjugate();
        let q_inv = q.inverse();

        assert_quat_eq(q_conj, q_inv); // For unit quaternions, conjugate is inverse

        // q * q_conj should be identity [1,0,0,0]
        let identity = q * q_conj;
        assert_quat_eq(identity, SimdUnitQuat::default());

        assert_f32_eq(q_conj.0[0], q.0[0]);
        assert_f32_eq(q_conj.0[1], -q.0[1]);
        assert_f32_eq(q_conj.0[2], -q.0[2]);
        assert_f32_eq(q_conj.0[3], -q.0[3]);
    }

    #[test]
    fn test_quat_real_imag() {
        let s = 0.5f32;
        let x = 0.1f32;
        let y = 0.2f32;
        let z = (1.0f32 - s * s - x * x - y * y).sqrt(); // Ensure unit quaternion for simplicity
        let q = SimdUnitQuat::new(s, x, y, z);

        assert_f32_eq(q.real(), s);
        assert_vec3_eq(q.imag(), SimdVec3::from([x, y, z]));
    }

    #[test]
    fn test_quat_mul_quat() {
        // Identity * q = q
        let q = SimdUnitQuat::from_axis_angle(SimdVec3::from([1.0, 1.0, 1.0]), PI / 4.0);
        assert_quat_eq(SimdUnitQuat::default() * q, q);
        assert_quat_eq(q * SimdUnitQuat::default(), q);

        // Rotation by q1 then q2 is q2 * q1
        // 90 deg around Z: q_z = [cos(pi/4), 0, 0, sin(pi/4)]
        let q_z = SimdUnitQuat::from_axis_angle(SimdVec3::from([0.0, 0.0, 1.0]), PI / 2.0);
        assert_quat_eq(
            q_z,
            SimdUnitQuat(f32x4::from_array([
                FRAC_PI_4.cos(),
                0.0,
                0.0,
                FRAC_PI_4.sin(),
            ])),
        );
        // 90 deg around X: q_x = [cos(pi/4), sin(pi/4), 0, 0]
        let q_x = SimdUnitQuat::from_axis_angle(SimdVec3::from([1.0, 0.0, 0.0]), PI / 2.0);
        assert_quat_eq(
            q_x,
            SimdUnitQuat(f32x4::from_array([
                FRAC_PI_4.cos(),
                FRAC_PI_4.sin(),
                0.0,
                0.0,
            ])),
        );

        // Rotate (1,0,0) by q_z -> (0,1,0)
        // Then rotate (0,1,0) by q_x -> (0,0,1)
        // So (q_x * q_z) * (1,0,0) -> (0,0,1)

        let v = SimdVec3::from([1.0, 0.0, 0.0]);
        let rotated_v_manual = q_x * (q_z * v);

        let q_combined = q_x * q_z;
        let rotated_v_combined = q_combined * v;

        assert_vec3_eq(rotated_v_manual, rotated_v_combined);
        assert_vec3_eq(rotated_v_combined, SimdVec3::from([0.0, 0.0, 1.0]));
    }

    #[test]
    fn test_quat_mul_vec3_rotation() {
        // Rotate i=(1,0,0) by 90deg around Z-axis -> j=(0,1,0)
        let q_rot_z90 = SimdUnitQuat::from_axis_angle(SimdVec3::from([0.0, 0.0, 1.0]), PI / 2.0);
        let i = SimdVec3::from([1.0, 0.0, 0.0]);
        let j = SimdVec3::from([0.0, 1.0, 0.0]);
        assert_vec3_eq(q_rot_z90 * i, j);

        // Rotate j=(0,1,0) by 90deg around X-axis -> k=(0,0,1)
        let q_rot_x90 = SimdUnitQuat::from_axis_angle(SimdVec3::from([1.0, 0.0, 0.0]), PI / 2.0);
        let k = SimdVec3::from([0.0, 0.0, 1.0]);
        assert_vec3_eq(q_rot_x90 * j, k);

        // Rotate k=(0,0,1) by 90deg around Y-axis -> i=(1,0,0)
        let q_rot_y90 = SimdUnitQuat::from_axis_angle(SimdVec3::from([0.0, 1.0, 0.0]), PI / 2.0);
        assert_vec3_eq(q_rot_y90 * k, i);

        // Rotate (1,1,0) by -90deg around Z-axis -> (1,-1,0) (normalized)
        let v_in = SimdVec3::from([1.0, 1.0, 0.0]).normalized().unwrap();
        let q_rot_z_neg90 =
            SimdUnitQuat::from_axis_angle(SimdVec3::from([0.0, 0.0, 1.0]), -PI / 2.0);
        let v_expected = SimdVec3::from([1.0, -1.0, 0.0]).normalized().unwrap();
        assert_vec3_eq(q_rot_z_neg90 * v_in, v_expected);

        // Rotating a non-unit vector must scale correctly and preserve length.
        let v_long = SimdVec3::from([3.0, 4.0, 12.0]);
        let rotated = q_rot_z90 * v_long;
        assert_f32_eq(rotated.norm(), v_long.norm());
        assert_vec3_eq(rotated, SimdVec3::from([-4.0, 3.0, 12.0]));
    }

    #[test]
    fn test_quat_slerp() {
        let q0 = SimdUnitQuat::IDENTITY;
        let q1 = SimdUnitQuat::from_axis_angle(SimdVec3::from([0.0, 0.0, 1.0]), PI / 2.0);

        // Endpoints
        assert_quat_eq(SimdUnitQuat::slerp(q0, q1, 0.0), q0);
        assert_quat_eq(SimdUnitQuat::slerp(q0, q1, 1.0), q1);

        // Midpoint between identity and a 90 degree rotation is a 45 degree rotation.
        let midpoint = SimdUnitQuat::slerp(q0, q1, 0.5);
        let expected = SimdUnitQuat::from_axis_angle(SimdVec3::from([0.0, 0.0, 1.0]), PI / 4.0);
        assert_quat_eq(midpoint, expected);

        // Slerp between nearly identical quaternions (nlerp fallback path).
        let q_close = SimdUnitQuat::from_axis_angle(SimdVec3::from([0.0, 0.0, 1.0]), 1e-4);
        let q = SimdUnitQuat::slerp(q0, q_close, 0.5);
        let norm_sq = (q.0 * q.0).reduce_sum();
        assert_f32_eq(norm_sq, 1.0);

        // Slerp must take the short way around: interpolating towards -q1
        // (the same rotation as q1) should act like interpolating towards q1.
        let neg_q1 = SimdUnitQuat(-q1.0);
        let midpoint_neg = SimdUnitQuat::slerp(q0, neg_q1, 0.5);
        let v = SimdVec3::from([1.0, 0.0, 0.0]);
        assert_vec3_eq(midpoint_neg * v, expected * v);
    }

    #[test]
    fn test_quat_array_conversions() {
        let arr = [0.5, 0.1, 0.2, 0.8];
        let q = SimdUnitQuat::from(arr);

        // Check that the quaternion was normalized
        let norm_sq = q.0[0] * q.0[0] + q.0[1] * q.0[1] + q.0[2] * q.0[2] + q.0[3] * q.0[3];
        assert_f32_eq(norm_sq, 1.0);

        // Convert back to array
        let back_to_arr: [f32; 4] = q.into();
        let back_norm_sq = back_to_arr[0] * back_to_arr[0]
            + back_to_arr[1] * back_to_arr[1]
            + back_to_arr[2] * back_to_arr[2]
            + back_to_arr[3] * back_to_arr[3];
        assert_f32_eq(back_norm_sq, 1.0);
    }

    #[test]
    fn test_quat_identity_constant() {
        let identity = SimdUnitQuat::IDENTITY;
        assert_quat_eq(identity, SimdUnitQuat::new(1.0, 0.0, 0.0, 0.0));

        // Identity quaternion should not change any vector when rotating
        let test_vector = SimdVec3::from([1.0, 2.0, 3.0]);
        let rotated = identity * test_vector;
        assert_vec3_eq(rotated, test_vector);
    }
}
