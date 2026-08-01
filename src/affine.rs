use super::*;
use std::ops::Mul;

/// A 3×4 affine transform matrix represented as 4 [`SimdVec3`] columns: the
/// first three are the transformed basis vectors, and the fourth is the
/// translation vector. This is sufficient for representing any affine
/// transform in 3D space (rotation, scale, shear, translation), and behaves
/// like a 4×4 matrix whose implicit bottom row is `[0, 0, 0, 1]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimdAffine3([SimdVec3; 4]);

/// Former name of [`SimdAffine3`].
#[deprecated(
    note = "renamed to `SimdAffine3`: the type is a 3x4 affine transform, not a 4x4 matrix"
)]
pub type SimdMat4 = SimdAffine3;

impl Default for SimdAffine3 {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl SimdAffine3 {
    /// The identity transform.
    pub const IDENTITY: Self = Self::new(
        SimdVec3::UNIT_X,
        SimdVec3::UNIT_Y,
        SimdVec3::UNIT_Z,
        SimdVec3::ZERO,
    );

    /// Creates a transform from three transformed basis-vector columns and a
    /// translation column.
    #[inline]
    pub const fn new(
        im_v0: SimdVec3,
        im_v1: SimdVec3,
        im_v2: SimdVec3,
        translation: SimdVec3,
    ) -> Self {
        SimdAffine3([im_v0, im_v1, im_v2, translation])
    }

    /// The identity transform. Same as [`SimdAffine3::IDENTITY`].
    #[inline]
    pub const fn identity() -> Self {
        Self::IDENTITY
    }

    /// Creates a pure-translation transform.
    #[inline]
    pub const fn from_translation(translation: SimdVec3) -> Self {
        Self::new(
            SimdVec3::UNIT_X,
            SimdVec3::UNIT_Y,
            SimdVec3::UNIT_Z,
            translation,
        )
    }

    /// Creates a linear transform (no translation) from three transformed
    /// basis-vector columns.
    #[inline]
    pub const fn from_rotation(im_x: SimdVec3, im_y: SimdVec3, im_z: SimdVec3) -> Self {
        Self::new(im_x, im_y, im_z, SimdVec3::ZERO)
    }

    /// Returns the four columns: the three transformed basis vectors followed
    /// by the translation vector.
    #[inline]
    #[must_use]
    pub const fn columns(&self) -> [SimdVec3; 4] {
        self.0
    }

    /// Returns the translation column.
    #[inline]
    #[must_use]
    pub const fn translation(&self) -> SimdVec3 {
        self.0[3]
    }

    /// Returns the determinant of the linear (upper-left 3×3) part.
    #[inline]
    #[must_use]
    pub fn determinant(&self) -> f32 {
        let [a, b, c, _] = self.0;
        a.dot(b.cross(c))
    }

    /// Returns the inverse transform, or `None` if the linear part is
    /// singular (determinant zero).
    #[must_use]
    pub fn inverse(&self) -> Option<Self> {
        let [a, b, c, t] = self.0;

        // Rows of the inverted linear part (before scaling), via the adjugate.
        let r0 = b.cross(c);
        let r1 = c.cross(a);
        let r2 = a.cross(b);

        let det = a.dot(r0);
        if det == 0.0 {
            return None;
        }
        let inv_det = 1.0 / det;

        // Transpose the rows back into columns.
        let ic0 = SimdVec3::from([r0[0], r1[0], r2[0]]) * inv_det;
        let ic1 = SimdVec3::from([r0[1], r1[1], r2[1]]) * inv_det;
        let ic2 = SimdVec3::from([r0[2], r1[2], r2[2]]) * inv_det;

        let inv_linear = SimdAffine3::from_rotation(ic0, ic1, ic2);
        let inv_translation = -(inv_linear * t);
        Some(SimdAffine3::new(ic0, ic1, ic2, inv_translation))
    }

    /// Multiplies by an `f32x4` column, treating lane 3 as the W coefficient
    /// for the translation column.
    #[inline]
    fn mul_f32x4(self, rhs: f32x4) -> f32x4 {
        let [i, j, k, t] = self.0;
        i.0 * f32x4::splat(rhs[0])
            + j.0 * f32x4::splat(rhs[1])
            + k.0 * f32x4::splat(rhs[2])
            + t.0 * f32x4::splat(rhs[3])
    }
}

impl From<SimdUnitQuat> for SimdAffine3 {
    /// Converts a rotation quaternion into the equivalent rotation matrix.
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

        SimdAffine3([
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

impl From<SimdTransform> for SimdAffine3 {
    /// Converts a rigid transform (rotation + translation) into the
    /// equivalent affine matrix.
    #[inline]
    fn from(t: SimdTransform) -> Self {
        let mut m = SimdAffine3::from(t.rotation());
        m.0[3] = t.position();
        m
    }
}

impl From<[[f32; 3]; 4]> for SimdAffine3 {
    /// Builds a transform from four column arrays: three transformed basis
    /// vectors followed by the translation.
    #[inline]
    fn from(cols: [[f32; 3]; 4]) -> Self {
        SimdAffine3(cols.map(SimdVec3::from))
    }
}

impl From<SimdAffine3> for [[f32; 3]; 4] {
    #[inline]
    fn from(m: SimdAffine3) -> Self {
        m.0.map(<[f32; 3]>::from)
    }
}

impl Mul<SimdVec3> for SimdAffine3 {
    type Output = SimdVec3;

    /// Matrix-vector3 multiplication: scale each transformed basis vector by the corresponding
    /// coefficient (X, Y, or Z) and sum, then add translation.
    /// Equivalent to extending the vector into a vec4 with a 1.0 in the W component.
    #[inline]
    fn mul(self, rhs: SimdVec3) -> Self::Output {
        let [i, j, k, t] = self.0;
        i * rhs[0] + j * rhs[1] + k * rhs[2] + t
    }
}

impl Mul<SimdVec4> for SimdAffine3 {
    type Output = SimdVec4;

    /// Matrix-vector4 multiplication, treating this transform as a 4×4 matrix
    /// with an implicit `[0, 0, 0, 1]` bottom row: the translation column is
    /// scaled by the input's W component, and the output's W component equals
    /// the input's.
    #[inline]
    fn mul(self, rhs: SimdVec4) -> Self::Output {
        let mut out = self.mul_f32x4(rhs.0);
        out[3] = rhs[3];
        SimdVec4(out)
    }
}

impl Mul<SimdAffine3> for SimdAffine3 {
    type Output = SimdAffine3;

    /// Matrix-matrix multiplication: transform each basis vector of the rhs
    /// matrix to compose the two transforms. The basis columns multiply with a
    /// W coefficient of 0 (no translation) and the translation column with a
    /// W coefficient of 1; the former relies on [`SimdVec3`]'s guarantee that
    /// its hidden fourth lane is always 0.
    #[inline]
    fn mul(self, rhs: SimdAffine3) -> Self::Output {
        SimdAffine3([
            SimdVec3(self.mul_f32x4(rhs.0[0].0)),
            SimdVec3(self.mul_f32x4(rhs.0[1].0)),
            SimdVec3(self.mul_f32x4(rhs.0[2].0)),
            self * rhs.0[3],
        ])
    }
}

#[cfg(test)]
mod simd_affine3_tests {
    use super::*;
    use std::f32::consts::PI;

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

    fn assert_affine_eq(a: SimdAffine3, b: SimdAffine3) {
        for i in 0..4 {
            assert_vec3_eq(a.0[i], b.0[i]);
        }
    }

    #[test]
    fn test_affine_identity_default() {
        let m_ident = SimdAffine3::identity();
        let m_default = SimdAffine3::default();
        assert_affine_eq(m_ident, m_default);
        assert_affine_eq(m_ident, SimdAffine3::IDENTITY);

        assert_vec3_eq(m_ident.0[0], SimdVec3::from([1.0, 0.0, 0.0]));
        assert_vec3_eq(m_ident.0[1], SimdVec3::from([0.0, 1.0, 0.0]));
        assert_vec3_eq(m_ident.0[2], SimdVec3::from([0.0, 0.0, 1.0]));
        assert_vec3_eq(m_ident.0[3], SimdVec3::from([0.0, 0.0, 0.0]));
    }

    #[test]
    fn test_affine_new_and_constructors() {
        let t = SimdVec3::from([1.0, 2.0, 3.0]);
        let m_trans = SimdAffine3::from_translation(t);
        assert_vec3_eq(m_trans.0[3], t);
        assert_vec3_eq(m_trans.0[0], SimdVec3::from([1.0, 0.0, 0.0]));

        let ix = SimdVec3::from([1.0, 0.0, 0.0]);
        let iy = SimdVec3::from([0.0, 1.0, 0.0]);
        let iz = SimdVec3::from([0.0, 0.0, 1.0]);
        let m_rot = SimdAffine3::from_rotation(ix, iy, iz);
        assert_vec3_eq(m_rot.0[0], ix);
        assert_vec3_eq(m_rot.0[1], iy);
        assert_vec3_eq(m_rot.0[2], iz);
        assert_vec3_eq(m_rot.0[3], SimdVec3::default());

        let m_new = SimdAffine3::new(ix, iy, iz, t);
        assert_vec3_eq(m_new.0[0], ix);
        assert_vec3_eq(m_new.0[1], iy);
        assert_vec3_eq(m_new.0[2], iz);
        assert_vec3_eq(m_new.0[3], t);
    }

    #[test]
    fn test_affine_accessors() {
        let ix = SimdVec3::from([1.0, 2.0, 3.0]);
        let iy = SimdVec3::from([4.0, 5.0, 6.0]);
        let iz = SimdVec3::from([7.0, 8.0, 9.0]);
        let t = SimdVec3::from([10.0, 11.0, 12.0]);
        let m = SimdAffine3::new(ix, iy, iz, t);

        let [c0, c1, c2, c3] = m.columns();
        assert_vec3_eq(c0, ix);
        assert_vec3_eq(c1, iy);
        assert_vec3_eq(c2, iz);
        assert_vec3_eq(c3, t);
        assert_vec3_eq(m.translation(), t);
    }

    #[test]
    fn test_affine_array_conversions() {
        let cols = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ];
        let m = SimdAffine3::from(cols);
        let back: [[f32; 3]; 4] = m.into();
        assert_eq!(cols, back);
    }

    #[test]
    fn test_affine_from_quat() {
        // Identity quaternion
        let q_ident = SimdUnitQuat::default();
        let m_ident_from_q = SimdAffine3::from(q_ident);
        assert_affine_eq(m_ident_from_q, SimdAffine3::identity());

        // 90-degree rotation around Z-axis
        let q_rot_z90 = SimdUnitQuat::from_axis_angle(SimdVec3::from([0.0, 0.0, 1.0]), PI / 2.0);
        let m_rot_z90 = SimdAffine3::from(q_rot_z90);

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
        let expected_m_rot_z90 = SimdAffine3::new(
            SimdVec3::from([0.0, 1.0, 0.0]), // Column X (after rotation, original X becomes Y)
            SimdVec3::from([-1.0, 0.0, 0.0]), // Column Y (original Y becomes -X)
            SimdVec3::from([0.0, 0.0, 1.0]), // Column Z (Z remains Z)
            SimdVec3::default(),             // Translation
        );
        assert_affine_eq(m_rot_z90, expected_m_rot_z90);
    }

    #[test]
    fn test_affine_from_transform() {
        let translation = SimdVec3::from([1.0, 2.0, 3.0]);
        let rotation = SimdUnitQuat::from_axis_angle(SimdVec3::from([0.0, 0.0, 1.0]), PI / 2.0);
        let transform = SimdTransform::new(translation, rotation);
        let m = SimdAffine3::from(transform);

        // The matrix must act on points exactly like the transform does.
        for point in [
            SimdVec3::from([1.0, 0.0, 0.0]),
            SimdVec3::from([0.0, 1.0, 0.0]),
            SimdVec3::from([1.0, 2.0, 3.0]),
        ] {
            assert_vec3_eq(m * point, transform * point);
        }
    }

    #[test]
    fn test_affine_mul_vec3() {
        let m_ident = SimdAffine3::identity();
        let v = SimdVec3::from([1.0, 2.0, 3.0]);
        assert_vec3_eq(m_ident * v, v);

        let t = SimdVec3::from([10.0, 20.0, 30.0]);
        let m_trans = SimdAffine3::from_translation(t);
        assert_vec3_eq(m_trans * v, v + t);

        // Rotation: 90 deg around Z
        let q_rot_z90 = SimdUnitQuat::from_axis_angle(SimdVec3::from([0.0, 0.0, 1.0]), PI / 2.0);
        let m_rot_z90 = SimdAffine3::from(q_rot_z90);
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
    fn test_affine_mul_affine() {
        let m_ident = SimdAffine3::identity();
        let m_a = SimdAffine3::from_translation(SimdVec3::from([1.0, 2.0, 3.0]));
        assert_affine_eq(m_ident * m_a, m_a);
        assert_affine_eq(m_a * m_ident, m_a);

        let t1 = SimdVec3::from([1.0, 2.0, 3.0]);
        let m_t1 = SimdAffine3::from_translation(t1);
        let t2 = SimdVec3::from([10.0, 20.0, 30.0]);
        let m_t2 = SimdAffine3::from_translation(t2);

        // M_t1 * M_t2 should result in a translation by t1 + t2
        let m_t1_t2 = m_t1 * m_t2;
        let expected_m_trans_sum = SimdAffine3::from_translation(t1 + t2);
        assert_affine_eq(m_t1_t2, expected_m_trans_sum);

        // Rotation matrices
        let q_rot_z90 = SimdUnitQuat::from_axis_angle(SimdVec3::from([0.0, 0.0, 1.0]), PI / 2.0);
        let m_rot_z90 = SimdAffine3::from(q_rot_z90);
        let q_rot_x90 = SimdUnitQuat::from_axis_angle(SimdVec3::from([1.0, 0.0, 0.0]), PI / 2.0);
        let m_rot_x90 = SimdAffine3::from(q_rot_x90);

        // M_rot_x90 * M_rot_z90 (rotate by Z then by X)
        let m_combined_rot = m_rot_x90 * m_rot_z90;

        // Equivalent quaternion rotation: q_rot_x90 * q_rot_z90
        let q_combined_rot = q_rot_x90 * q_rot_z90;
        let m_expected_combined_rot = SimdAffine3::from(q_combined_rot);
        assert_affine_eq(m_combined_rot, m_expected_combined_rot);

        // Test with a vector
        let v = SimdVec3::from([1.0, 0.0, 0.0]);
        // (1,0,0) -> Z-rot -> (0,1,0) -> X-rot -> (0,0,1)
        assert_vec3_eq(m_combined_rot * v, SimdVec3::from([0.0, 0.0, 1.0]));
    }

    #[test]
    fn test_affine_mul_vec4() {
        let m = SimdAffine3::new(
            SimdVec3::from([1.0, 0.0, 0.0]),    // X basis
            SimdVec3::from([0.0, 1.0, 0.0]),    // Y basis
            SimdVec3::from([0.0, 0.0, 1.0]),    // Z basis
            SimdVec3::from([10.0, 20.0, 30.0]), // Translation
        );

        // W = 1: a point; the translation applies and W is preserved.
        let point = SimdVec4::from([2.0, 3.0, 4.0, 1.0]);
        let result = m * point;
        assert_eq!(result, SimdVec4::from([12.0, 23.0, 34.0, 1.0]));

        // W = 0: a direction; the translation must not apply.
        let direction = SimdVec4::from([2.0, 3.0, 4.0, 0.0]);
        let result = m * direction;
        assert_eq!(result, SimdVec4::from([2.0, 3.0, 4.0, 0.0]));
    }

    #[test]
    fn test_affine_determinant() {
        assert!((SimdAffine3::identity().determinant() - 1.0).abs() < EPSILON);

        let m_scale = SimdAffine3::from_rotation(
            SimdVec3::from([2.0, 0.0, 0.0]),
            SimdVec3::from([0.0, 3.0, 0.0]),
            SimdVec3::from([0.0, 0.0, 4.0]),
        );
        assert!((m_scale.determinant() - 24.0).abs() < EPSILON);

        // Rotations have determinant 1.
        let q = SimdUnitQuat::from_axis_angle(SimdVec3::from([1.0, 2.0, 3.0]), PI / 3.0);
        assert!((SimdAffine3::from(q).determinant() - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_affine_inverse() {
        // Rotation + translation.
        let q = SimdUnitQuat::from_axis_angle(SimdVec3::from([1.0, 2.0, 3.0]), PI / 3.0);
        let m =
            SimdAffine3::from_translation(SimdVec3::from([4.0, 5.0, 6.0])) * SimdAffine3::from(q);
        let m_inv = m.inverse().expect("rotation+translation is invertible");
        assert_affine_eq(m * m_inv, SimdAffine3::identity());
        assert_affine_eq(m_inv * m, SimdAffine3::identity());

        // Non-uniform scale.
        let m_scale = SimdAffine3::from_rotation(
            SimdVec3::from([2.0, 0.0, 0.0]),
            SimdVec3::from([0.0, 4.0, 0.0]),
            SimdVec3::from([0.0, 0.0, 8.0]),
        );
        let m_scale_inv = m_scale.inverse().expect("scale is invertible");
        assert_affine_eq(m_scale * m_scale_inv, SimdAffine3::identity());

        // Inverting the matrix must invert its action on points.
        let p = SimdVec3::from([1.0, 2.0, 3.0]);
        assert_vec3_eq(m_inv * (m * p), p);

        // Singular matrix (zero column) has no inverse.
        let m_singular = SimdAffine3::from_rotation(
            SimdVec3::from([1.0, 0.0, 0.0]),
            SimdVec3::from([0.0, 1.0, 0.0]),
            SimdVec3::ZERO,
        );
        assert!(m_singular.inverse().is_none());
    }
}
