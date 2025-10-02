use super::*;
use std::ops::Mul;

/// SimdMat4 is a 3x4 matrix represented as 4 SimdVec3 columns: the first three are the transformed basis
/// vectors, and the fourth is the translation vector. This is sufficient for representing any
/// affine transform in 3D space (rotation, scale, shear, translation).
#[derive(Clone, Copy)]
pub struct SimdMat4([SimdVec3; 4]);

impl Default for SimdMat4 {
    fn default() -> Self {
        Self::identity()
    }
}

impl SimdMat4 {
    #[inline]
    pub const fn new(
        im_v0: SimdVec3,
        im_v1: SimdVec3,
        im_v2: SimdVec3,
        translation: SimdVec3,
    ) -> Self {
        SimdMat4([im_v0, im_v1, im_v2, translation])
    }

    #[inline]
    pub fn identity() -> Self {
        SimdMat4([
            SimdVec3::from([1.0, 0.0, 0.0]),
            SimdVec3::from([0.0, 1.0, 0.0]),
            SimdVec3::from([0.0, 0.0, 1.0]),
            SimdVec3::from([0.0, 0.0, 0.0]),
        ])
    }

    #[inline]
    pub fn from_translation(translation: SimdVec3) -> Self {
        SimdMat4([
            SimdVec3::from([1.0, 0.0, 0.0]),
            SimdVec3::from([0.0, 1.0, 0.0]),
            SimdVec3::from([0.0, 0.0, 1.0]),
            translation,
        ])
    }

    #[inline]
    pub fn from_rotation(im_x: SimdVec3, im_y: SimdVec3, im_z: SimdVec3) -> Self {
        SimdMat4([im_x, im_y, im_z, SimdVec3::default()])
    }
}

impl From<SimdUnitQuat> for SimdMat4 {
    /// Convert a quaternion to a 4x4 matrix.
    #[inline]
    fn from(q: SimdUnitQuat) -> Self {
        let q = q.0;
        let s = q[0];
        let x = q[1];
        let y = q[2];
        let z = q[3];

        // Pre-calculate squared terms and products for clarity and potential optimization
        let x2 = x * x;
        let y2 = y * y;
        let z2 = z * z;

        let xy = x * y;
        let xz = x * z;
        let yz = y * z;

        let sx = s * x;
        let sy = s * y;
        let sz = s * z;

        SimdMat4([
            SimdVec3::from([
                1.0 - 2.0 * (y2 + z2),
                2.0 * (xy + sz), //
                2.0 * (xz - sy), //
            ]),
            SimdVec3::from([
                2.0 * (xy - sz), //
                1.0 - 2.0 * (x2 + z2),
                2.0 * (yz + sx), //
            ]),
            SimdVec3::from([
                2.0 * (xz + sy), //
                2.0 * (yz - sx), //
                1.0 - 2.0 * (x2 + y2),
            ]),
            SimdVec3::default(),
        ])
    }
}

impl Mul<SimdVec3> for SimdMat4 {
    type Output = SimdVec3;

    /// Matrix-vector3 multiplication: scale each transformed basis vector by the corresponding
    /// coefficient (X, Y, or Z) and sum, then add translation.
    /// Equivalent to converting the SimdVec3::new into a vec4 with a 1.0 in the W component.
    #[inline]
    fn mul(self, rhs: SimdVec3) -> Self::Output {
        let [i, j, k, t] = self.0;
        i * rhs[0] + j * rhs[1] + k * rhs[2] + t
    }
}

// f32x4 is a helper for matmul.
impl Mul<f32x4> for SimdMat4 {
    type Output = f32x4;

    /// Matrix-vector4 multiplication: scale each transformed basis vector by the corresponding
    /// coefficient (X, Y, Z, or W) and sum.
    #[inline]
    fn mul(self, rhs: f32x4) -> Self::Output {
        let [i, j, k, t] = self.0;
        i.0 * f32x4::splat(rhs[0])
            + j.0 * f32x4::splat(rhs[1])
            + k.0 * f32x4::splat(rhs[2])
            + t.0 * f32x4::splat(rhs[3])
    }
}

impl Mul<SimdMat4> for SimdMat4 {
    type Output = SimdMat4;

    /// Matrix-matrix multiplication: transform each basis vector of the rhs matrix to compose the
    /// two transforms. Note that we need to unwrap the rhs bases into f32x4 to get a 0 coefficient
    /// in W for all bases except the fourth.
    /// NOTE: This function depends on Vec3 having '0' as its fourth component in f32x4.
    #[inline]
    fn mul(self, rhs: SimdMat4) -> Self::Output {
        SimdMat4([
            SimdVec3(self * rhs.0[0].0),
            SimdVec3(self * rhs.0[1].0),
            SimdVec3(self * rhs.0[2].0),
            self * rhs.0[3],
        ])
    }
}

#[cfg(test)]
mod simd_mat4_tests {
    use super::*;
    use std::f32::consts::PI;

    // Assuming EPSILON and assert_vec3_eq are available (e.g. from quat_tests or a common test utils mod)
    const EPSILON: f32 = 1e-6;

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

    fn assert_mat4_eq(a: SimdMat4, b: SimdMat4) {
        for i in 0..4 {
            assert_vec3_eq(a.0[i], b.0[i]);
        }
    }

    #[test]
    fn test_mat4_identity_default() {
        let m_ident = SimdMat4::identity();
        let m_default = SimdMat4::default();
        assert_mat4_eq(m_ident, m_default);

        assert_vec3_eq(m_ident.0[0], SimdVec3::from([1.0, 0.0, 0.0]));
        assert_vec3_eq(m_ident.0[1], SimdVec3::from([0.0, 1.0, 0.0]));
        assert_vec3_eq(m_ident.0[2], SimdVec3::from([0.0, 0.0, 1.0]));
        assert_vec3_eq(m_ident.0[3], SimdVec3::from([0.0, 0.0, 0.0]));
    }

    #[test]
    fn test_mat4_new_and_constructors() {
        let t = SimdVec3::from([1.0, 2.0, 3.0]);
        let m_trans = SimdMat4::from_translation(t);
        assert_vec3_eq(m_trans.0[3], t);
        assert_vec3_eq(m_trans.0[0], SimdVec3::from([1.0, 0.0, 0.0]));

        let ix = SimdVec3::from([1.0, 0.0, 0.0]);
        let iy = SimdVec3::from([0.0, 1.0, 0.0]);
        let iz = SimdVec3::from([0.0, 0.0, 1.0]);
        let m_rot = SimdMat4::from_rotation(ix, iy, iz);
        assert_vec3_eq(m_rot.0[0], ix);
        assert_vec3_eq(m_rot.0[1], iy);
        assert_vec3_eq(m_rot.0[2], iz);
        assert_vec3_eq(m_rot.0[3], SimdVec3::default());

        let m_new = SimdMat4::new(ix, iy, iz, t);
        assert_vec3_eq(m_new.0[0], ix);
        assert_vec3_eq(m_new.0[1], iy);
        assert_vec3_eq(m_new.0[2], iz);
        assert_vec3_eq(m_new.0[3], t);
    }

    #[test]
    fn test_mat4_from_quat() {
        // Identity quaternion
        let q_ident = SimdUnitQuat::default();
        let m_ident_from_q = SimdMat4::from(q_ident);
        assert_mat4_eq(m_ident_from_q, SimdMat4::identity());

        // 90-degree rotation around Z-axis
        let q_rot_z90 = SimdUnitQuat::from_axis_angle(SimdVec3::from([0.0, 0.0, 1.0]), PI / 2.0);
        let m_rot_z90 = SimdMat4::from(q_rot_z90);

        // Expected matrix for 90-deg Z rotation:
        // cos(a) -sin(a)  0   0
        // sin(a)  cos(a)  0   0
        //   0       0     1   0
        //   0       0     0   1
        // For a=PI/2: cos(a)=0, sin(a)=1
        //  0 -1  0  0
        //  1  0  0  0
        //  0  0  1  0
        //  0  0  0  1
        let expected_m_rot_z90 = SimdMat4::new(
            SimdVec3::from([0.0, 1.0, 0.0]), // Column X (after rotation, original X becomes Y)
            SimdVec3::from([-1.0, 0.0, 0.0]), // Column Y (original Y becomes -X)
            SimdVec3::from([0.0, 0.0, 1.0]), // Column Z (Z remains Z)
            SimdVec3::default(),             // Translation
        );
        assert_mat4_eq(m_rot_z90, expected_m_rot_z90);
    }

    #[test]
    fn test_mat4_mul_vec3() {
        let m_ident = SimdMat4::identity();
        let v = SimdVec3::from([1.0, 2.0, 3.0]);
        assert_vec3_eq(m_ident * v, v);

        let t = SimdVec3::from([10.0, 20.0, 30.0]);
        let m_trans = SimdMat4::from_translation(t);
        assert_vec3_eq(m_trans * v, v + t);

        // Rotation: 90 deg around Z
        let q_rot_z90 = SimdUnitQuat::from_axis_angle(SimdVec3::from([0.0, 0.0, 1.0]), PI / 2.0);
        let m_rot_z90 = SimdMat4::from(q_rot_z90);
        // (1,2,3) rotated by 90 deg around Z -> (-2,1,3)
        assert_vec3_eq(m_rot_z90 * v, SimdVec3::from([-2.0, 1.0, 3.0]));

        // Combined rotation and translation
        let m_combined = m_trans * m_rot_z90; // Rotate then translate
        let rotated_v = m_rot_z90 * v;
        let translated_rotated_v = m_trans * rotated_v;
        assert_vec3_eq(m_combined * v, translated_rotated_v);
        assert_vec3_eq(m_combined * v, SimdVec3::from([-2.0, 1.0, 3.0]) + t);
    }

    #[test]
    fn test_mat4_mul_mat4() {
        let m_ident = SimdMat4::identity();
        let m_a = SimdMat4::from_translation(SimdVec3::from([1.0, 2.0, 3.0]));
        assert_mat4_eq(m_ident * m_a, m_a);
        assert_mat4_eq(m_a * m_ident, m_a);

        let t1 = SimdVec3::from([1.0, 2.0, 3.0]);
        let m_t1 = SimdMat4::from_translation(t1);
        let t2 = SimdVec3::from([10.0, 20.0, 30.0]);
        let m_t2 = SimdMat4::from_translation(t2);

        // M_t1 * M_t2 should result in a translation by t1 + t2
        let m_t1_t2 = m_t1 * m_t2;
        let expected_m_trans_sum = SimdMat4::from_translation(t1 + t2);
        assert_mat4_eq(m_t1_t2, expected_m_trans_sum);

        // Rotation matrices
        let q_rot_z90 = SimdUnitQuat::from_axis_angle(SimdVec3::from([0.0, 0.0, 1.0]), PI / 2.0);
        let m_rot_z90 = SimdMat4::from(q_rot_z90);
        let q_rot_x90 = SimdUnitQuat::from_axis_angle(SimdVec3::from([1.0, 0.0, 0.0]), PI / 2.0);
        let m_rot_x90 = SimdMat4::from(q_rot_x90);

        // M_rot_x90 * M_rot_z90 (rotate by Z then by X)
        let m_combined_rot = m_rot_x90 * m_rot_z90;

        // Equivalent quaternion rotation: q_rot_x90 * q_rot_z90
        let q_combined_rot = q_rot_x90 * q_rot_z90;
        let m_expected_combined_rot = SimdMat4::from(q_combined_rot);
        assert_mat4_eq(m_combined_rot, m_expected_combined_rot);

        // Test with a vector
        let v = SimdVec3::from([1.0, 0.0, 0.0]);
        // (1,0,0) -> Z-rot -> (0,1,0) -> X-rot -> (0,0,1)
        assert_vec3_eq(m_combined_rot * v, SimdVec3::from([0.0, 0.0, 1.0]));
    }

    #[test]
    fn test_mat4_mul_f32x4() {
        // Test matrix multiplication with f32x4 (internal helper)
        let m = SimdMat4::new(
            SimdVec3::from([1.0, 0.0, 0.0]),    // X basis
            SimdVec3::from([0.0, 1.0, 0.0]),    // Y basis
            SimdVec3::from([0.0, 0.0, 1.0]),    // Z basis
            SimdVec3::from([10.0, 20.0, 30.0]), // Translation
        );

        let vec4 = f32x4::from_array([2.0, 3.0, 4.0, 1.0]);
        let result = m * vec4;

        // Expected: 2*X_basis + 3*Y_basis + 4*Z_basis + 1*translation
        // = 2*(1,0,0,0) + 3*(0,1,0,0) + 4*(0,0,1,0) + 1*(10,20,30,0)
        // = (2,0,0,0) + (0,3,0,0) + (0,0,4,0) + (10,20,30,0)
        // = (12,23,34,0)
        let expected = f32x4::from_array([12.0, 23.0, 34.0, 0.0]);

        for i in 0..4 {
            assert!(
                (result[i] - expected[i]).abs() < EPSILON,
                "component {}: {} != {}",
                i,
                result[i],
                expected[i]
            );
        }
    }
}
