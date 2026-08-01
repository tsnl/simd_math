use super::*;
use num_traits::identities::{ConstOne, ConstZero};
use std::fmt::Debug;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

/// Common interface implemented by every vector type in this crate.
pub trait SimdVector: Sized {
    /// The scalar type stored in each lane of the vector.
    type LaneType: Sized + ConstZero + Copy + Debug + PartialEq + PartialOrd;
}

macro_rules! impl_float_simd_vec {
    ($name:ident : $simd_ty:ty [ $lane_ty:ty ; $dim:expr ]) => {
        impl_basic_simd_vec!($name : $simd_ty [ $lane_ty ; $dim ]);
        impl_simd_vec_neg_method!($name : $simd_ty [ $lane_ty ; $dim ]);
        impl_simd_vec_float_methods!($name : $simd_ty [ $lane_ty ; $dim ]);
    }
}
macro_rules! impl_int_simd_vec {
    ($name:ident : $simd_ty:ty [ $lane_ty:ty ; $dim:expr ]) => {
        impl_basic_simd_vec!($name : $simd_ty [ $lane_ty ; $dim ]);
        impl_simd_vec_neg_method!($name : $simd_ty [ $lane_ty ; $dim ]);
        impl Eq for $name {}
    };
}
macro_rules! impl_uint_simd_vec {
    ($name:ident : $simd_ty:ty [ $lane_ty:ty ; $dim:expr ]) => {
        impl_basic_simd_vec!($name : $simd_ty [ $lane_ty ; $dim ]);
        impl Eq for $name {}
    }
}

macro_rules! impl_basic_simd_vec {
    ($name:ident : $simd_ty:ty [ $lane_ty:ty ; $dim:expr ]) => {
        #[doc = concat!(
            "A ", stringify!($dim), "-dimensional vector of `", stringify!($lane_ty),
            "` lanes backed by a `", stringify!($simd_ty), "` register."
        )]
        ///
        /// When the backing register is wider than the vector, the extra
        /// (padding) lanes of every value of this type hold zero, and every
        /// operation preserves that invariant. Comparisons ignore the padding
        /// lanes entirely. A few operations (elementwise division, product
        /// reduction) internally substitute ones into the padding lanes of a
        /// *temporary* operand register to stay well-defined, but such
        /// registers are never returned or stored.
        #[derive(Clone, Copy)]
        pub struct $name(pub(crate) $simd_ty);

        impl Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_tuple(stringify!($name))
                    .field(&&self.0.as_array()[..$dim])
                    .finish()
            }
        }

        /// Equality over the visible lanes only; padding lanes are ignored.
        impl PartialEq for $name {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                self.0.as_array()[..$dim] == other.0.as_array()[..$dim]
            }
        }

        impl SimdVector for $name {
            type LaneType = $lane_ty;
        }

        impl $name {
            /// Returns a copy of `v` with the padding lanes (if any) replaced
            /// by ones, making elementwise division and product reduction
            /// well-defined for vectors narrower than their backing register.
            #[inline]
            fn with_padding_ones(v: $simd_ty) -> $simd_ty {
                let mut arr = *v.as_array();
                for lane in arr[$dim..].iter_mut() {
                    *lane = <$lane_ty>::ONE;
                }
                <$simd_ty>::from_array(arr)
            }
        }

        impl_simd_vec_ctor_methods!($name : $simd_ty [ $lane_ty ; $dim ]);
        impl_simd_vec_universal_methods!($name : $simd_ty [ $lane_ty ; $dim ]);
    };
}

macro_rules! impl_simd_vec_ctor_methods {
    ($name:ident : $simd_ty:ty [ $lane_ty:ty ; $dim:expr ]) => {
        impl $name {
            /// Creates a vector with every component set to `value`.
            #[inline]
            pub fn splat(value: $lane_ty) -> Self {
                let mut arr = [<$lane_ty>::ZERO; <$simd_ty>::LEN];
                arr[..$dim].fill(value);
                $name(<$simd_ty>::from_array(arr))
            }

            /// Creates a vector from an array of any scalar type that converts
            /// losslessly into the lane type, e.g.
            /// `SimdUVec2::from_convert([1u16, 2u16])`.
            #[inline]
            pub fn from_convert<T: Into<$lane_ty>>(arr: [T; $dim]) -> Self {
                Self::from(arr.map(Into::into))
            }
        }

        // Convert array into SimdVecN: SimdVecN::from([a1, a2, ..., aN])
        impl From<[$lane_ty; $dim]> for $name {
            #[inline]
            fn from(arr: [$lane_ty; $dim]) -> Self {
                let full_arr = {
                    let mut temp = [<$lane_ty>::ZERO; <$simd_ty>::LEN];
                    temp[..$dim].copy_from_slice(&arr);
                    temp
                };
                $name(<$simd_ty>::from_array(full_arr))
            }
        }

        // Convert SimdVecN into array:
        impl From<$name> for [$lane_ty; $dim] {
            #[inline]
            fn from(it: $name) -> [$lane_ty; $dim] {
                it.0.as_array()[..$dim].try_into().unwrap()
            }
        }
    };
}

macro_rules! impl_simd_vec_universal_methods {
    ($name:ident : $simd_ty:ty [ $lane_ty:ty ; $dim:expr ]) => {
        impl Default for $name {
            #[inline]
            fn default() -> Self {
                $name(<$simd_ty>::splat(<$lane_ty>::ZERO)) // Default to zero vector
            }
        }
        impl $name {
            /// Dot product of two vectors.
            #[inline]
            #[must_use]
            pub fn dot(self, other: $name) -> $lane_ty {
                (self.0 * other.0).reduce_sum()
            }
            /// Sum of all components.
            #[inline]
            #[must_use]
            pub fn reduce_sum(self) -> $lane_ty {
                self.0.reduce_sum()
            }
            /// Product of all components.
            #[inline]
            #[must_use]
            pub fn reduce_product(self) -> $lane_ty {
                Self::with_padding_ones(self.0).reduce_product()
            }
        }
        impl Mul<$lane_ty> for $name {
            type Output = $name;
            #[inline]
            fn mul(self, rhs: $lane_ty) -> Self::Output {
                $name(self.0 * <$simd_ty>::splat(rhs))
            }
        }
        impl Mul<$name> for $lane_ty {
            type Output = $name;
            /// Scalar-on-the-left multiplication: `s * v == v * s`.
            #[inline]
            fn mul(self, rhs: $name) -> Self::Output {
                rhs * self
            }
        }
        impl MulAssign<$lane_ty> for $name {
            #[inline]
            fn mul_assign(&mut self, rhs: $lane_ty) {
                self.0 *= <$simd_ty>::splat(rhs);
            }
        }
        impl Mul<$name> for $name {
            type Output = $name;
            /// Elementwise product
            #[inline]
            fn mul(self, rhs: $name) -> Self::Output {
                $name(self.0 * rhs.0)
            }
        }
        impl MulAssign<$name> for $name {
            #[inline]
            fn mul_assign(&mut self, rhs: $name) {
                self.0 *= rhs.0;
            }
        }
        impl Div<$lane_ty> for $name {
            type Output = $name;
            #[inline]
            fn div(self, rhs: $lane_ty) -> Self::Output {
                $name(self.0 / Self::with_padding_ones(<$simd_ty>::splat(rhs)))
            }
        }
        impl DivAssign<$lane_ty> for $name {
            #[inline]
            fn div_assign(&mut self, rhs: $lane_ty) {
                self.0 /= Self::with_padding_ones(<$simd_ty>::splat(rhs));
            }
        }
        impl Div<$name> for $name {
            type Output = $name;
            /// Element-wise division
            #[inline]
            fn div(self, rhs: $name) -> Self::Output {
                $name(self.0 / Self::with_padding_ones(rhs.0))
            }
        }
        impl DivAssign<$name> for $name {
            #[inline]
            fn div_assign(&mut self, rhs: $name) {
                self.0 /= Self::with_padding_ones(rhs.0);
            }
        }
        impl Add<$name> for $name {
            type Output = $name;
            #[inline]
            fn add(self, rhs: $name) -> Self::Output {
                $name(self.0 + rhs.0)
            }
        }
        impl AddAssign<$name> for $name {
            #[inline]
            fn add_assign(&mut self, rhs: $name) {
                self.0 += rhs.0;
            }
        }
        impl Sub<$name> for $name {
            type Output = $name;
            #[inline]
            fn sub(self, rhs: $name) -> Self::Output {
                $name(self.0 - rhs.0)
            }
        }
        impl SubAssign<$name> for $name {
            #[inline]
            fn sub_assign(&mut self, rhs: $name) {
                self.0 -= rhs.0;
            }
        }
        impl $name {
            /// Elementwise minimum of two vectors.
            #[inline]
            #[must_use]
            pub fn elementwise_min(self, other: $name) -> $name {
                $name(self.0.simd_min(other.0))
            }
            /// Elementwise maximum of two vectors.
            #[inline]
            #[must_use]
            pub fn elementwise_max(self, other: $name) -> $name {
                $name(self.0.simd_max(other.0))
            }
        }
        impl Index<usize> for $name {
            type Output = $lane_ty;
            #[inline]
            fn index(&self, index: usize) -> &Self::Output {
                assert!(
                    index < $dim,
                    "index out of bounds: the vector has {} components but the index is {}",
                    $dim,
                    index
                );
                &self.0[index]
            }
        }
        impl IndexMut<usize> for $name {
            #[inline]
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                assert!(
                    index < $dim,
                    "index out of bounds: the vector has {} components but the index is {}",
                    $dim,
                    index
                );
                &mut self.0[index]
            }
        }
    };
}

macro_rules! impl_simd_vec_neg_method {
    ($name:ident : $simd_ty:ty [ $lane_ty:ty ; $dim:expr ]) => {
        impl Neg for $name {
            type Output = $name;
            #[inline]
            fn neg(self) -> Self::Output {
                $name(-self.0)
            }
        }
    };
}

macro_rules! impl_simd_vec_float_methods {
    ($name:ident : $simd_ty:ty [ $lane_ty:ty ; $dim:expr ]) => {
        impl $name {
            /// Compute the squared Euclidean norm of the vector.
            /// This is more efficient than computing the norm itself.
            #[inline]
            #[must_use]
            pub fn norm_squared(self) -> $lane_ty {
                self.dot(self)
            }

            /// Compute the Euclidean norm (magnitude) of the vector.
            #[inline]
            #[must_use]
            pub fn norm(self) -> $lane_ty {
                self.norm_squared().sqrt()
            }

            /// Normalize the vector by dividing it by its norm.
            /// Returns `None` for the zero vector.
            #[inline]
            #[must_use]
            pub fn normalized(self) -> Option<Self> {
                let norm = self.norm();
                if norm == 0.0 {
                    // Avoid division by zero
                    None
                } else {
                    Some(Self(self.0 / <$simd_ty>::splat(norm)))
                }
            }

            /// Linear interpolation between two vectors a and b by factor t
            #[inline]
            #[must_use]
            pub fn lerp(a: Self, b: Self, t: $lane_ty) -> Self {
                let k_a = <$simd_ty>::splat(1.0 - t);
                let k_b = <$simd_ty>::splat(t);
                let res = a.0 * k_a + b.0 * k_b;
                Self(res)
            }

            /// Clamp each component of the vector between the corresponding components of min and max
            #[inline]
            #[must_use]
            pub fn clamp(self, min: Self, max: Self) -> Self {
                let clamped = self.0.simd_clamp(min.0, max.0);
                Self(clamped)
            }

            /// Element-wise power operation with scalar exponent
            #[inline]
            #[must_use]
            pub fn powf(self, exponent: $lane_ty) -> Self {
                let self_array = *self.0.as_array();
                let mut result = self_array; // Start with the same array structure
                for i in 0..$dim {
                    result[i] = self_array[i].powf(exponent);
                }
                Self(<$simd_ty>::from_array(result))
            }

            /// Element-wise power operation with SIMD exponent vector
            #[inline]
            #[must_use]
            pub fn powf_elementwise(self, exponent: Self) -> Self {
                let self_array = *self.0.as_array();
                let exp_array = *exponent.0.as_array();
                let mut result = self_array; // Start with the same array structure
                for i in 0..$dim {
                    result[i] = self_array[i].powf(exp_array[i]);
                }
                Self(<$simd_ty>::from_array(result))
            }
        }
    };
}

impl_uint_simd_vec!(SimdUVec4: u32x4 [u32; 4]);
impl_uint_simd_vec!(SimdUVec3: u32x4 [u32; 3]);
impl_uint_simd_vec!(SimdUVec2: u32x2 [u32; 2]);

impl_int_simd_vec!(SimdIVec4: i32x4 [i32; 4] );
impl_int_simd_vec!(SimdIVec3: i32x4 [i32; 3] );
impl_int_simd_vec!(SimdIVec2: i32x2 [i32; 2] );

impl_float_simd_vec!(SimdVec4: f32x4 [f32; 4]);
impl_float_simd_vec!(SimdVec3: f32x4 [f32; 3]);
impl_float_simd_vec!(SimdVec2: f32x2 [f32; 2]);

impl SimdVec2 {
    /// The zero vector.
    pub const ZERO: Self = Self(f32x2::from_array([0.0, 0.0]));
    /// The unit vector along the X axis.
    pub const UNIT_X: Self = Self(f32x2::from_array([1.0, 0.0]));
    /// The unit vector along the Y axis.
    pub const UNIT_Y: Self = Self(f32x2::from_array([0.0, 1.0]));
}
impl SimdVec3 {
    /// The zero vector.
    pub const ZERO: Self = Self(f32x4::from_array([0.0, 0.0, 0.0, 0.0]));
    /// The unit vector along the X axis.
    pub const UNIT_X: Self = Self(f32x4::from_array([1.0, 0.0, 0.0, 0.0]));
    /// The unit vector along the Y axis.
    pub const UNIT_Y: Self = Self(f32x4::from_array([0.0, 1.0, 0.0, 0.0]));
    /// The unit vector along the Z axis.
    pub const UNIT_Z: Self = Self(f32x4::from_array([0.0, 0.0, 1.0, 0.0]));
}
impl SimdVec4 {
    /// The zero vector.
    pub const ZERO: Self = Self(f32x4::from_array([0.0, 0.0, 0.0, 0.0]));
    /// The unit vector along the X axis.
    pub const UNIT_X: Self = Self(f32x4::from_array([1.0, 0.0, 0.0, 0.0]));
    /// The unit vector along the Y axis.
    pub const UNIT_Y: Self = Self(f32x4::from_array([0.0, 1.0, 0.0, 0.0]));
    /// The unit vector along the Z axis.
    pub const UNIT_Z: Self = Self(f32x4::from_array([0.0, 0.0, 1.0, 0.0]));
    /// The unit vector along the W axis.
    pub const UNIT_W: Self = Self(f32x4::from_array([0.0, 0.0, 0.0, 1.0]));
}

impl SimdVec2 {
    /// Convert spherical coordinate angles (azimuth, elevation) to equirectangular texture coordinates (u, v).
    /// Both azimuth and elevation are expected to be in radians.
    /// - azimuth: angle in the horizontal plane, range [-π, π]
    /// - elevation: angle in the vertical plane, range [-π/2, π/2]
    ///
    /// Returns UV coordinates where:
    /// - u is in range [0, 1] (left to right across the texture)
    /// - v is in range [0, 1] (top to bottom of the texture)
    #[must_use]
    pub fn spherical_coords_angles_into_equirectangular_coords(self) -> Self {
        let azimuth = self[0];
        let elevation = self[1];

        // Convert azimuth from [-π, π] to [0, 1]
        let u = (azimuth + std::f32::consts::PI) / (2.0 * std::f32::consts::PI);

        // Convert elevation from [-π/2, π/2] to [0, 1]
        // Note: we flip the v coordinate so that positive elevation maps to the top of the texture
        let v = 0.5 - (elevation / std::f32::consts::PI);

        SimdVec2::from([u, v])
    }

    /// Extend this vector into a [`SimdVec3`] with the given `z` component.
    #[inline]
    #[must_use]
    pub fn into_vec3(self, z: f32) -> SimdVec3 {
        SimdVec3::from([self[0], self[1], z])
    }
}

impl SimdVec4 {
    /// Truncate this vector into a [`SimdVec3`], dropping the `w` component.
    #[inline]
    #[must_use]
    pub fn into_vec3(self) -> SimdVec3 {
        SimdVec3::from([self[0], self[1], self[2]])
    }
}

impl SimdVec3 {
    /// Extend this vector into a [`SimdVec4`] with the given `w` component.
    #[inline]
    #[must_use]
    pub fn into_vec4(self, w: f32) -> SimdVec4 {
        let mut v = self.0;
        v[3] = w;
        SimdVec4(v)
    }

    /// Convert a Cartesian vector to spherical coordinates.
    /// Returns (azimuth, elevation, radius) where:
    /// - azimuth: angle in the horizontal plane, range [-π, π]
    /// - elevation: angle in the vertical plane, range [-π/2, π/2]
    /// - radius: distance from origin
    ///
    /// The convention is Y-up: azimuth is measured in the XZ plane from +X
    /// towards +Z, and elevation is measured towards +Y.
    #[must_use]
    pub fn into_spherical_coords(self) -> Self {
        let x = self[0];
        let y = self[1];
        let z = self[2];

        let radius = self.norm();

        if radius == 0.0 {
            return SimdVec3::from([0.0, 0.0, 0.0]);
        }

        let azimuth = z.atan2(x);
        let elevation = (y / radius).asin();

        SimdVec3::from([azimuth, elevation, radius])
    }

    /// Cross product of two 3D vectors.
    /// Returns a vector perpendicular to both input vectors.
    /// The magnitude of the result is |a| * |b| * sin(θ) where θ is the angle between the vectors.
    /// The direction follows the right-hand rule.
    #[must_use]
    pub fn cross(self, other: SimdVec3) -> SimdVec3 {
        let a = self;
        let b = other;

        let result_x = a[1] * b[2] - a[2] * b[1];
        let result_y = a[2] * b[0] - a[0] * b[2];
        let result_z = a[0] * b[1] - a[1] * b[0];

        SimdVec3::from([result_x, result_y, result_z])
    }
}

#[cfg(test)]
mod simd_vec3_tests {
    use super::*;

    #[test]
    fn test_cross() {
        // Test cross product of unit vectors
        let x_axis = SimdVec3::from([1.0, 0.0, 0.0]);
        let y_axis = SimdVec3::from([0.0, 1.0, 0.0]);
        let z_axis = SimdVec3::from([0.0, 0.0, 1.0]);

        // Right-hand rule: x × y = z
        let cross_xy = x_axis.cross(y_axis);
        assert!((cross_xy[0] - 0.0).abs() < 1e-6);
        assert!((cross_xy[1] - 0.0).abs() < 1e-6);
        assert!((cross_xy[2] - 1.0).abs() < 1e-6);

        // y × z = x
        let cross_yz = y_axis.cross(z_axis);
        assert!((cross_yz[0] - 1.0).abs() < 1e-6);
        assert!((cross_yz[1] - 0.0).abs() < 1e-6);
        assert!((cross_yz[2] - 0.0).abs() < 1e-6);

        // z × x = y
        let cross_zx = z_axis.cross(x_axis);
        assert!((cross_zx[0] - 0.0).abs() < 1e-6);
        assert!((cross_zx[1] - 1.0).abs() < 1e-6);
        assert!((cross_zx[2] - 0.0).abs() < 1e-6);
    }
}

#[cfg(test)]
mod spherical_coords_tests {
    use super::*;

    const EPSILON: f32 = 1e-6;

    fn assert_vec3_near(a: SimdVec3, b: SimdVec3, epsilon: f32) {
        assert!((a[0] - b[0]).abs() < epsilon, "x: {} vs {}", a[0], b[0]);
        assert!((a[1] - b[1]).abs() < epsilon, "y: {} vs {}", a[1], b[1]);
        assert!((a[2] - b[2]).abs() < epsilon, "z: {} vs {}", a[2], b[2]);
    }

    fn assert_vec2_near(a: SimdVec2, b: SimdVec2, epsilon: f32) {
        assert!((a[0] - b[0]).abs() < epsilon, "x: {} vs {}", a[0], b[0]);
        assert!((a[1] - b[1]).abs() < epsilon, "y: {} vs {}", a[1], b[1]);
    }

    #[test]
    fn test_into_spherical_coords_basic_directions() {
        // Test positive X axis
        let pos_x = SimdVec3::from([1.0, 0.0, 0.0]);
        let spherical = pos_x.into_spherical_coords();
        assert_vec3_near(spherical, SimdVec3::from([0.0, 0.0, 1.0]), EPSILON);

        // Test positive Z axis
        let pos_z = SimdVec3::from([0.0, 0.0, 1.0]);
        let spherical = pos_z.into_spherical_coords();
        assert_vec3_near(
            spherical,
            SimdVec3::from([std::f32::consts::FRAC_PI_2, 0.0, 1.0]),
            EPSILON,
        );

        // Test negative Z axis
        let neg_z = SimdVec3::from([0.0, 0.0, -1.0]);
        let spherical = neg_z.into_spherical_coords();
        assert_vec3_near(
            spherical,
            SimdVec3::from([-std::f32::consts::FRAC_PI_2, 0.0, 1.0]),
            EPSILON,
        );

        // Test positive Y axis (up)
        let pos_y = SimdVec3::from([0.0, 1.0, 0.0]);
        let spherical = pos_y.into_spherical_coords();
        assert_vec3_near(
            spherical,
            SimdVec3::from([0.0, std::f32::consts::FRAC_PI_2, 1.0]),
            EPSILON,
        );

        // Test negative Y axis (down)
        let neg_y = SimdVec3::from([0.0, -1.0, 0.0]);
        let spherical = neg_y.into_spherical_coords();
        assert_vec3_near(
            spherical,
            SimdVec3::from([0.0, -std::f32::consts::FRAC_PI_2, 1.0]),
            EPSILON,
        );
    }

    #[test]
    fn test_into_spherical_coords_zero_vector() {
        let zero = SimdVec3::from([0.0, 0.0, 0.0]);
        let spherical = zero.into_spherical_coords();
        assert_vec3_near(spherical, SimdVec3::from([0.0, 0.0, 0.0]), EPSILON);
    }

    #[test]
    fn test_into_spherical_coords_magnitude() {
        // Test that magnitude is preserved
        let vec = SimdVec3::from([3.0, 4.0, 0.0]);
        let spherical = vec.into_spherical_coords();
        assert!((spherical[2] - 5.0).abs() < EPSILON); // 3-4-5 triangle
    }

    #[test]
    fn test_spherical_coords_angles_into_equirectangular_coords() {
        // Test center of texture (azimuth=0, elevation=0)
        let center = SimdVec2::from([0.0, 0.0]);
        let uv = center.spherical_coords_angles_into_equirectangular_coords();
        assert_vec2_near(uv, SimdVec2::from([0.5, 0.5]), EPSILON);

        // Test positive azimuth (π/2, looking towards +Z)
        let pos_azimuth = SimdVec2::from([std::f32::consts::FRAC_PI_2, 0.0]);
        let uv = pos_azimuth.spherical_coords_angles_into_equirectangular_coords();
        assert_vec2_near(uv, SimdVec2::from([0.75, 0.5]), EPSILON);

        // Test negative azimuth (-π/2, looking towards -Z)
        let neg_azimuth = SimdVec2::from([-std::f32::consts::FRAC_PI_2, 0.0]);
        let uv = neg_azimuth.spherical_coords_angles_into_equirectangular_coords();
        assert_vec2_near(uv, SimdVec2::from([0.25, 0.5]), EPSILON);

        // Test positive elevation (π/2, looking up)
        let pos_elevation = SimdVec2::from([0.0, std::f32::consts::FRAC_PI_2]);
        let uv = pos_elevation.spherical_coords_angles_into_equirectangular_coords();
        assert_vec2_near(uv, SimdVec2::from([0.5, 0.0]), EPSILON);

        // Test negative elevation (-π/2, looking down)
        let neg_elevation = SimdVec2::from([0.0, -std::f32::consts::FRAC_PI_2]);
        let uv = neg_elevation.spherical_coords_angles_into_equirectangular_coords();
        assert_vec2_near(uv, SimdVec2::from([0.5, 1.0]), EPSILON);
    }

    #[test]
    fn test_spherical_coords_boundary_values() {
        // Test azimuth boundaries
        let left_edge = SimdVec2::from([-std::f32::consts::PI, 0.0]);
        let uv = left_edge.spherical_coords_angles_into_equirectangular_coords();
        assert_vec2_near(uv, SimdVec2::from([0.0, 0.5]), EPSILON);

        let right_edge = SimdVec2::from([std::f32::consts::PI, 0.0]);
        let uv = right_edge.spherical_coords_angles_into_equirectangular_coords();
        assert_vec2_near(uv, SimdVec2::from([1.0, 0.5]), EPSILON);
    }
}

#[cfg(test)]
mod vector_unit_tests {
    use super::*;

    const EPSILON: f32 = 1e-6;

    fn assert_f32_near(a: f32, b: f32) {
        assert!(
            (a - b).abs() < EPSILON,
            "Expected {}, got {} (difference: {})",
            b,
            a,
            (a - b).abs()
        );
    }

    macro_rules! test_simd_vec_ctor_methods {
        ($vec_type:ty, $lane_ty:ty, 2, $test_values:expr) => {
            #[test]
            fn test_new() {
                let v = <$vec_type>::from([$test_values[0], $test_values[1]]);
                assert_eq!(v[0], $test_values[0]);
                assert_eq!(v[1], $test_values[1]);
            }

            #[test]
            fn test_splat() {
                let v = <$vec_type>::splat($test_values[0]);
                assert_eq!(v[0], $test_values[0]);
                assert_eq!(v[1], $test_values[0]);
            }

            #[test]
            fn test_array_conversions() {
                let arr: [$lane_ty; 2] = [$test_values[0], $test_values[1]];
                let v = <$vec_type>::from(arr);
                assert_eq!(v[0], arr[0]);
                assert_eq!(v[1], arr[1]);

                let back_to_arr: [$lane_ty; 2] = v.into();
                assert_eq!(back_to_arr, arr);
            }
        };
        ($vec_type:ty, $lane_ty:ty, 3, $test_values:expr) => {
            #[test]
            fn test_new() {
                let v = <$vec_type>::from([$test_values[0], $test_values[1], $test_values[2]]);
                assert_eq!(v[0], $test_values[0]);
                assert_eq!(v[1], $test_values[1]);
                assert_eq!(v[2], $test_values[2]);
            }

            #[test]
            fn test_splat() {
                let v = <$vec_type>::splat($test_values[0]);
                assert_eq!(v[0], $test_values[0]);
                assert_eq!(v[1], $test_values[0]);
                assert_eq!(v[2], $test_values[0]);
            }

            #[test]
            fn test_array_conversions() {
                let arr: [$lane_ty; 3] = [$test_values[0], $test_values[1], $test_values[2]];
                let v = <$vec_type>::from(arr);
                assert_eq!(v[0], arr[0]);
                assert_eq!(v[1], arr[1]);
                assert_eq!(v[2], arr[2]);

                let back_to_arr: [$lane_ty; 3] = v.into();
                assert_eq!(back_to_arr, arr);
            }
        };
        ($vec_type:ty, $lane_ty:ty, 4, $test_values:expr) => {
            #[test]
            fn test_new() {
                let v = <$vec_type>::from([
                    $test_values[0],
                    $test_values[1],
                    $test_values[2],
                    $test_values[3],
                ]);
                assert_eq!(v[0], $test_values[0]);
                assert_eq!(v[1], $test_values[1]);
                assert_eq!(v[2], $test_values[2]);
                assert_eq!(v[3], $test_values[3]);
            }

            #[test]
            fn test_splat() {
                let v = <$vec_type>::splat($test_values[0]);
                assert_eq!(v[0], $test_values[0]);
                assert_eq!(v[1], $test_values[0]);
                assert_eq!(v[2], $test_values[0]);
                assert_eq!(v[3], $test_values[0]);
            }

            #[test]
            fn test_array_conversions() {
                let arr: [$lane_ty; 4] = [
                    $test_values[0],
                    $test_values[1],
                    $test_values[2],
                    $test_values[3],
                ];
                let v = <$vec_type>::from(arr);
                assert_eq!(v[0], arr[0]);
                assert_eq!(v[1], arr[1]);
                assert_eq!(v[2], arr[2]);
                assert_eq!(v[3], arr[3]);

                let back_to_arr: [$lane_ty; 4] = v.into();
                assert_eq!(back_to_arr, arr);
            }
        };
    }

    macro_rules! test_simd_vec_universal_methods_int {
        ($vec_type:ty, $lane_ty:ty, $dim:expr, $test_values:expr) => {
            // Use regular base methods but with integer types that support Ord
            test_simd_vec_universal_methods!($vec_type, $lane_ty, $dim, $test_values);
        };
    }

    macro_rules! test_simd_vec_universal_methods {
        ($vec_type:ty, $lane_ty:ty, $dim:expr, $test_values:expr) => {
            #[test]
            fn test_basic_operations() {
                // Just test that basic operations work - detailed testing is in other modules
                let v = <$vec_type>::default();
                let v2 = <$vec_type>::splat($test_values[0]);
                let _result = v + v2;
                let _result = v2.dot(v2);
                let _result = v2.elementwise_min(v2);
                let _result = v2.elementwise_max(v2);
            }
        };
    }

    macro_rules! test_simd_vec_neg_method {
        ($vec_type:ty, 2, $test_values:expr) => {
            #[test]
            fn test_negation() {
                let v = <$vec_type>::from([$test_values[0], $test_values[1]]);
                let neg_v = -v;
                assert_eq!(neg_v[0], -$test_values[0]);
                assert_eq!(neg_v[1], -$test_values[1]);
            }
        };
        ($vec_type:ty, 3, $test_values:expr) => {
            #[test]
            fn test_negation() {
                let v = <$vec_type>::from([$test_values[0], $test_values[1], $test_values[2]]);
                let neg_v = -v;
                assert_eq!(neg_v[0], -$test_values[0]);
                assert_eq!(neg_v[1], -$test_values[1]);
                assert_eq!(neg_v[2], -$test_values[2]);
            }
        };
        ($vec_type:ty, 4, $test_values:expr) => {
            #[test]
            fn test_negation() {
                let v = <$vec_type>::from([
                    $test_values[0],
                    $test_values[1],
                    $test_values[2],
                    $test_values[3],
                ]);
                let neg_v = -v;
                assert_eq!(neg_v[0], -$test_values[0]);
                assert_eq!(neg_v[1], -$test_values[1]);
                assert_eq!(neg_v[2], -$test_values[2]);
                assert_eq!(neg_v[3], -$test_values[3]);
            }
        };
    }

    macro_rules! test_simd_vec_float_methods {
        ($vec_type:ty, 2) => {
            #[test]
            fn test_norm() {
                let v = <$vec_type>::from([3.0, 4.0]);
                assert_f32_near(v.norm_squared(), 25.0);
                assert_f32_near(v.norm(), 5.0);
            }

            #[test]
            fn test_normalized() {
                let v = <$vec_type>::from([3.0, 4.0]);
                let normalized = v.normalized().unwrap();
                assert_f32_near(normalized.norm(), 1.0);

                // Test zero vector
                let zero = <$vec_type>::splat(0.0);
                assert!(zero.normalized().is_none());
            }

            #[test]
            fn test_lerp() {
                let a = <$vec_type>::splat(0.0);
                let b = <$vec_type>::splat(10.0);

                let lerp_0 = <$vec_type>::lerp(a, b, 0.0);
                let lerp_1 = <$vec_type>::lerp(a, b, 1.0);
                let lerp_half = <$vec_type>::lerp(a, b, 0.5);

                assert_f32_near(lerp_0[0], 0.0);
                assert_f32_near(lerp_0[1], 0.0);
                assert_f32_near(lerp_1[0], 10.0);
                assert_f32_near(lerp_1[1], 10.0);
                assert_f32_near(lerp_half[0], 5.0);
                assert_f32_near(lerp_half[1], 5.0);
            }

            #[test]
            fn test_clamp() {
                let v = <$vec_type>::splat(5.0);
                let min = <$vec_type>::splat(0.0);
                let max = <$vec_type>::splat(3.0);

                let clamped = v.clamp(min, max);
                assert_f32_near(clamped[0], 3.0);
                assert_f32_near(clamped[1], 3.0);
            }

            #[test]
            fn test_powf() {
                let v = <$vec_type>::splat(2.0);
                let powered = v.powf(3.0);
                assert_f32_near(powered[0], 8.0);
                assert_f32_near(powered[1], 8.0);
            }

            #[test]
            fn test_powf_elementwise() {
                let v = <$vec_type>::splat(2.0);
                let exp = <$vec_type>::splat(3.0);
                let powered = v.powf_elementwise(exp);
                assert_f32_near(powered[0], 8.0);
                assert_f32_near(powered[1], 8.0);
            }
        };
        ($vec_type:ty, 3) => {
            #[test]
            fn test_norm() {
                let v = <$vec_type>::from([3.0, 4.0, 0.0]);
                assert_f32_near(v.norm_squared(), 25.0);
                assert_f32_near(v.norm(), 5.0);
            }

            #[test]
            fn test_normalized() {
                let v = <$vec_type>::from([3.0, 4.0, 0.0]);
                let normalized = v.normalized().unwrap();
                assert_f32_near(normalized.norm(), 1.0);

                // Test zero vector
                let zero = <$vec_type>::splat(0.0);
                assert!(zero.normalized().is_none());
            }

            #[test]
            fn test_lerp() {
                let a = <$vec_type>::splat(0.0);
                let b = <$vec_type>::splat(10.0);

                let lerp_0 = <$vec_type>::lerp(a, b, 0.0);
                let lerp_1 = <$vec_type>::lerp(a, b, 1.0);
                let lerp_half = <$vec_type>::lerp(a, b, 0.5);

                assert_f32_near(lerp_0[0], 0.0);
                assert_f32_near(lerp_0[1], 0.0);
                assert_f32_near(lerp_0[2], 0.0);
                assert_f32_near(lerp_1[0], 10.0);
                assert_f32_near(lerp_1[1], 10.0);
                assert_f32_near(lerp_1[2], 10.0);
                assert_f32_near(lerp_half[0], 5.0);
                assert_f32_near(lerp_half[1], 5.0);
                assert_f32_near(lerp_half[2], 5.0);
            }

            #[test]
            fn test_clamp() {
                let v = <$vec_type>::splat(5.0);
                let min = <$vec_type>::splat(0.0);
                let max = <$vec_type>::splat(3.0);

                let clamped = v.clamp(min, max);
                assert_f32_near(clamped[0], 3.0);
                assert_f32_near(clamped[1], 3.0);
                assert_f32_near(clamped[2], 3.0);
            }

            #[test]
            fn test_powf() {
                let v = <$vec_type>::splat(2.0);
                let powered = v.powf(3.0);
                assert_f32_near(powered[0], 8.0);
                assert_f32_near(powered[1], 8.0);
                assert_f32_near(powered[2], 8.0);
            }

            #[test]
            fn test_powf_elementwise() {
                let v = <$vec_type>::splat(2.0);
                let exp = <$vec_type>::splat(3.0);
                let powered = v.powf_elementwise(exp);
                assert_f32_near(powered[0], 8.0);
                assert_f32_near(powered[1], 8.0);
                assert_f32_near(powered[2], 8.0);
            }
        };
        ($vec_type:ty, 4) => {
            #[test]
            fn test_norm() {
                let v = <$vec_type>::from([3.0, 4.0, 0.0, 0.0]);
                assert_f32_near(v.norm_squared(), 25.0);
                assert_f32_near(v.norm(), 5.0);
            }

            #[test]
            fn test_normalized() {
                let v = <$vec_type>::from([3.0, 4.0, 0.0, 0.0]);
                let normalized = v.normalized().unwrap();
                assert_f32_near(normalized.norm(), 1.0);

                // Test zero vector
                let zero = <$vec_type>::splat(0.0);
                assert!(zero.normalized().is_none());
            }

            #[test]
            fn test_lerp() {
                let a = <$vec_type>::splat(0.0);
                let b = <$vec_type>::splat(10.0);

                let lerp_0 = <$vec_type>::lerp(a, b, 0.0);
                let lerp_1 = <$vec_type>::lerp(a, b, 1.0);
                let lerp_half = <$vec_type>::lerp(a, b, 0.5);

                assert_f32_near(lerp_0[0], 0.0);
                assert_f32_near(lerp_0[1], 0.0);
                assert_f32_near(lerp_0[2], 0.0);
                assert_f32_near(lerp_0[3], 0.0);
                assert_f32_near(lerp_1[0], 10.0);
                assert_f32_near(lerp_1[1], 10.0);
                assert_f32_near(lerp_1[2], 10.0);
                assert_f32_near(lerp_1[3], 10.0);
                assert_f32_near(lerp_half[0], 5.0);
                assert_f32_near(lerp_half[1], 5.0);
                assert_f32_near(lerp_half[2], 5.0);
                assert_f32_near(lerp_half[3], 5.0);
            }

            #[test]
            fn test_clamp() {
                let v = <$vec_type>::splat(5.0);
                let min = <$vec_type>::splat(0.0);
                let max = <$vec_type>::splat(3.0);

                let clamped = v.clamp(min, max);
                assert_f32_near(clamped[0], 3.0);
                assert_f32_near(clamped[1], 3.0);
                assert_f32_near(clamped[2], 3.0);
                assert_f32_near(clamped[3], 3.0);
            }

            #[test]
            fn test_powf() {
                let v = <$vec_type>::splat(2.0);
                let powered = v.powf(3.0);
                assert_f32_near(powered[0], 8.0);
                assert_f32_near(powered[1], 8.0);
                assert_f32_near(powered[2], 8.0);
                assert_f32_near(powered[3], 8.0);
            }

            #[test]
            fn test_powf_elementwise() {
                let v = <$vec_type>::splat(2.0);
                let exp = <$vec_type>::splat(3.0);
                let powered = v.powf_elementwise(exp);
                assert_f32_near(powered[0], 8.0);
                assert_f32_near(powered[1], 8.0);
                assert_f32_near(powered[2], 8.0);
                assert_f32_near(powered[3], 8.0);
            }
        };
    }

    macro_rules! test_simd_vec_field_methods {
        ($vec_type:ty, [x: 0, y: 1]) => {
            #[test]
            fn test_field_accessors() {
                let v = <$vec_type>::from([1.0, 2.0]);
                assert_eq!(v[0], 1.0);
                assert_eq!(v[1], 2.0);
            }
        };
        ($vec_type:ty, [x: 0, y: 1, z: 2]) => {
            #[test]
            fn test_field_accessors() {
                let v = <$vec_type>::from([1.0, 2.0, 3.0]);
                assert_eq!(v[0], 1.0);
                assert_eq!(v[1], 2.0);
                assert_eq!(v[2], 3.0);
            }
        };
        ($vec_type:ty, [x: 0, y: 1, z: 2, w: 3]) => {
            #[test]
            fn test_field_accessors() {
                let v = <$vec_type>::from([1.0, 2.0, 3.0, 4.0]);
                assert_eq!(v[0], 1.0);
                assert_eq!(v[1], 2.0);
                assert_eq!(v[2], 3.0);
                assert_eq!(v[3], 4.0);
            }
        };
    }

    macro_rules! test_simd_vec_constants {
        ($vec_type:ty, 2, [
            ZERO => [0.0, 0.0],
            UNIT_X => [1.0, 0.0],
            UNIT_Y => [0.0, 1.0]
        ]) => {
            #[test]
            fn test_constants() {
                let zero = <$vec_type>::ZERO;
                assert_f32_near(zero[0], 0.0);
                assert_f32_near(zero[1], 0.0);

                let unit_x = <$vec_type>::UNIT_X;
                assert_f32_near(unit_x[0], 1.0);
                assert_f32_near(unit_x[1], 0.0);

                let unit_y = <$vec_type>::UNIT_Y;
                assert_f32_near(unit_y[0], 0.0);
                assert_f32_near(unit_y[1], 1.0);
            }
        };
        ($vec_type:ty, 3, [
            ZERO => [0.0, 0.0, 0.0],
            UNIT_X => [1.0, 0.0, 0.0],
            UNIT_Y => [0.0, 1.0, 0.0],
            UNIT_Z => [0.0, 0.0, 1.0]
        ]) => {
            #[test]
            fn test_constants() {
                let zero = <$vec_type>::ZERO;
                assert_f32_near(zero[0], 0.0);
                assert_f32_near(zero[1], 0.0);
                assert_f32_near(zero[2], 0.0);

                let unit_x = <$vec_type>::UNIT_X;
                assert_f32_near(unit_x[0], 1.0);
                assert_f32_near(unit_x[1], 0.0);
                assert_f32_near(unit_x[2], 0.0);

                let unit_y = <$vec_type>::UNIT_Y;
                assert_f32_near(unit_y[0], 0.0);
                assert_f32_near(unit_y[1], 1.0);
                assert_f32_near(unit_y[2], 0.0);

                let unit_z = <$vec_type>::UNIT_Z;
                assert_f32_near(unit_z[0], 0.0);
                assert_f32_near(unit_z[1], 0.0);
                assert_f32_near(unit_z[2], 1.0);
            }
        };
        ($vec_type:ty, 4, [
            ZERO => [0.0, 0.0, 0.0, 0.0],
            UNIT_X => [1.0, 0.0, 0.0, 0.0],
            UNIT_Y => [0.0, 1.0, 0.0, 0.0],
            UNIT_Z => [0.0, 0.0, 1.0, 0.0],
            UNIT_W => [0.0, 0.0, 0.0, 1.0]
        ]) => {
            #[test]
            fn test_constants() {
                let zero = <$vec_type>::ZERO;
                assert_f32_near(zero[0], 0.0);
                assert_f32_near(zero[1], 0.0);
                assert_f32_near(zero[2], 0.0);
                assert_f32_near(zero[3], 0.0);

                let unit_x = <$vec_type>::UNIT_X;
                assert_f32_near(unit_x[0], 1.0);
                assert_f32_near(unit_x[1], 0.0);
                assert_f32_near(unit_x[2], 0.0);
                assert_f32_near(unit_x[3], 0.0);

                let unit_y = <$vec_type>::UNIT_Y;
                assert_f32_near(unit_y[0], 0.0);
                assert_f32_near(unit_y[1], 1.0);
                assert_f32_near(unit_y[2], 0.0);
                assert_f32_near(unit_y[3], 0.0);

                let unit_z = <$vec_type>::UNIT_Z;
                assert_f32_near(unit_z[0], 0.0);
                assert_f32_near(unit_z[1], 0.0);
                assert_f32_near(unit_z[2], 1.0);
                assert_f32_near(unit_z[3], 0.0);

                let unit_w = <$vec_type>::UNIT_W;
                assert_f32_near(unit_w[0], 0.0);
                assert_f32_near(unit_w[1], 0.0);
                assert_f32_near(unit_w[2], 0.0);
                assert_f32_near(unit_w[3], 1.0);
            }
        };
    }

    // Generate tests for each vector type
    mod simd_vec2_tests {
        use super::*;

        test_simd_vec_ctor_methods!(SimdVec2, f32, 2, [1.0, 2.0, 3.0, 4.0]);
        test_simd_vec_universal_methods!(SimdVec2, f32, 2, [1.0, 2.0, 3.0, 4.0]);
        test_simd_vec_neg_method!(SimdVec2, 2, [1.0, -2.0, 3.0, 4.0]);
        test_simd_vec_float_methods!(SimdVec2, 2);
        test_simd_vec_field_methods!(SimdVec2, [x: 0, y: 1]);
        test_simd_vec_constants!(SimdVec2, 2, [
            ZERO => [0.0, 0.0],
            UNIT_X => [1.0, 0.0],
            UNIT_Y => [0.0, 1.0]
        ]);
    }

    mod simd_vec3_tests {
        use super::*;

        test_simd_vec_ctor_methods!(SimdVec3, f32, 3, [1.0, 2.0, 3.0, 4.0]);
        test_simd_vec_universal_methods!(SimdVec3, f32, 3, [1.0, 2.0, 3.0, 2.0]);
        test_simd_vec_neg_method!(SimdVec3, 3, [1.0, -2.0, 3.0, 4.0]);
        test_simd_vec_float_methods!(SimdVec3, 3);
        test_simd_vec_field_methods!(SimdVec3, [x: 0, y: 1, z: 2]);
        test_simd_vec_constants!(SimdVec3, 3, [
            ZERO => [0.0, 0.0, 0.0],
            UNIT_X => [1.0, 0.0, 0.0],
            UNIT_Y => [0.0, 1.0, 0.0],
            UNIT_Z => [0.0, 0.0, 1.0]
        ]);

        #[test]
        fn test_cross_additional() {
            // Test additional cross product cases beyond basic unit vectors
            let v1 = SimdVec3::from([1.0, 2.0, 3.0]);
            let v2 = SimdVec3::from([4.0, 5.0, 6.0]);
            let cross = v1.cross(v2);

            // Cross product should be perpendicular to both vectors
            assert_f32_near(cross.dot(v1), 0.0);
            assert_f32_near(cross.dot(v2), 0.0);

            // Test cross product with parallel vectors (should be zero)
            let parallel1 = SimdVec3::from([1.0, 2.0, 3.0]);
            let parallel2 = SimdVec3::from([2.0, 4.0, 6.0]);
            let zero_cross = parallel1.cross(parallel2);
            assert_f32_near(zero_cross.norm(), 0.0);
        }
    }

    mod simd_vec4_tests {
        use super::*;

        test_simd_vec_ctor_methods!(SimdVec4, f32, 4, [1.0, 2.0, 3.0, 4.0]);
        test_simd_vec_universal_methods!(SimdVec4, f32, 4, [1.0, 2.0, 3.0, 2.0]);
        test_simd_vec_neg_method!(SimdVec4, 4, [1.0, -2.0, 3.0, -4.0]);
        test_simd_vec_float_methods!(SimdVec4, 4);
        test_simd_vec_field_methods!(SimdVec4, [x: 0, y: 1, z: 2, w: 3]);
        test_simd_vec_constants!(SimdVec4, 4, [
            ZERO => [0.0, 0.0, 0.0, 0.0],
            UNIT_X => [1.0, 0.0, 0.0, 0.0],
            UNIT_Y => [0.0, 1.0, 0.0, 0.0],
            UNIT_Z => [0.0, 0.0, 1.0, 0.0],
            UNIT_W => [0.0, 0.0, 0.0, 1.0]
        ]);

        #[test]
        fn test_into_vec3() {
            let v4 = SimdVec4::from([1.0, 2.0, 3.0, 4.0]);
            let v3 = v4.into_vec3();
            assert_f32_near(v3[0], 1.0);
            assert_f32_near(v3[1], 2.0);
            assert_f32_near(v3[2], 3.0);
        }
    }

    // Integer vector tests - separate modules to avoid name conflicts
    mod simd_ivec2_tests {
        use super::*;
        test_simd_vec_ctor_methods!(SimdIVec2, i32, 2, [1, 2, 3, 4]);
        test_simd_vec_universal_methods_int!(SimdIVec2, i32, 2, [1, 2, 3, 2]);
        test_simd_vec_neg_method!(SimdIVec2, 2, [1, -2, 3, 4]);
    }

    mod simd_ivec3_tests {
        use super::*;
        test_simd_vec_ctor_methods!(SimdIVec3, i32, 3, [1, 2, 3, 4]);
        test_simd_vec_universal_methods_int!(SimdIVec3, i32, 3, [1, 2, 3, 2]);
        test_simd_vec_neg_method!(SimdIVec3, 3, [1, -2, 3, 4]);
    }

    mod simd_ivec4_tests {
        use super::*;
        test_simd_vec_ctor_methods!(SimdIVec4, i32, 4, [1, 2, 3, 4]);
        test_simd_vec_universal_methods_int!(SimdIVec4, i32, 4, [1, 2, 3, 2]);
        test_simd_vec_neg_method!(SimdIVec4, 4, [1, -2, 3, -4]);
    }

    // Unsigned integer vector tests - separate modules to avoid name conflicts
    mod simd_uvec2_tests {
        use super::*;
        test_simd_vec_ctor_methods!(SimdUVec2, u32, 2, [1, 2, 3, 4]);
        test_simd_vec_universal_methods_int!(SimdUVec2, u32, 2, [5, 4, 3, 2]);
    }

    mod simd_uvec3_tests {
        use super::*;
        test_simd_vec_ctor_methods!(SimdUVec3, u32, 3, [1, 2, 3, 4]);
        test_simd_vec_universal_methods_int!(SimdUVec3, u32, 3, [5, 4, 3, 2]);
    }

    mod simd_uvec4_tests {
        use super::*;
        test_simd_vec_ctor_methods!(SimdUVec4, u32, 4, [1, 2, 3, 4]);
        test_simd_vec_universal_methods_int!(SimdUVec4, u32, 4, [5, 4, 3, 2]);
    }
}

#[cfg(test)]
mod padding_lane_tests {
    //! Regression tests for the invariant that padding lanes (lanes beyond the
    //! vector's dimension in a wider backing register) always hold zero, and
    //! that no operation observes or corrupts them.
    use super::*;

    fn pad_f(v: SimdVec3) -> f32 {
        v.0[3]
    }

    #[test]
    fn test_splat_zeroes_padding() {
        let v = SimdVec3::splat(2.0);
        assert_eq!(pad_f(v), 0.0);
        assert_eq!(v.dot(v), 12.0);
        assert_eq!(v.reduce_sum(), 6.0);
        assert_eq!(v.norm_squared(), 12.0);
        assert_eq!(SimdIVec3::splat(2).reduce_sum(), 6);
        assert_eq!(SimdUVec3::splat(2).reduce_sum(), 6);
    }

    #[test]
    fn test_reduce_product_ignores_padding() {
        assert_eq!(SimdVec3::from([2.0, 3.0, 4.0]).reduce_product(), 24.0);
        assert_eq!(SimdIVec3::from([2, 3, 4]).reduce_product(), 24);
        assert_eq!(SimdUVec3::from([2, 3, 4]).reduce_product(), 24);
        assert_eq!(SimdVec4::from([1.0, 2.0, 3.0, 4.0]).reduce_product(), 24.0);
    }

    #[test]
    fn test_elementwise_int_div() {
        let q = SimdIVec3::from([4, 6, 8]) / SimdIVec3::from([2, 3, 4]);
        assert_eq!(q, SimdIVec3::from([2, 2, 2]));

        let mut q = SimdUVec3::from([4, 6, 8]);
        q /= SimdUVec3::from([2, 3, 4]);
        assert_eq!(q, SimdUVec3::from([2, 2, 2]));

        let q = SimdIVec3::from([9, 9, 9]) / 3;
        assert_eq!(q, SimdIVec3::from([3, 3, 3]));
    }

    #[test]
    fn test_elementwise_float_div_keeps_padding_zero() {
        let q = SimdVec3::from([4.0, 6.0, 8.0]) / SimdVec3::from([2.0, 3.0, 4.0]);
        assert_eq!(q, SimdVec3::from([2.0, 2.0, 2.0]));
        assert_eq!(pad_f(q), 0.0);
        assert!(!q.norm().is_nan());

        // Scalar division by zero produces infinities in the visible lanes
        // (IEEE-754), but the padding lane must not become NaN.
        let q = SimdVec3::from([1.0, 2.0, 3.0]) / 0.0;
        assert_eq!(pad_f(q), 0.0);
    }

    #[test]
    fn test_eq_ignores_padding() {
        assert_eq!(SimdVec3::splat(2.0), SimdVec3::from([2.0, 2.0, 2.0]));

        // Even a manually corrupted padding lane must not affect equality.
        let mut corrupted = SimdVec3::from([1.0, 2.0, 3.0]);
        corrupted.0[3] = 7.0;
        assert_eq!(corrupted, SimdVec3::from([1.0, 2.0, 3.0]));
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_index_out_of_bounds_panics() {
        let v = SimdVec3::from([1.0, 2.0, 3.0]);
        let _ = v[3];
    }

    #[test]
    fn test_index_mut() {
        let mut v = SimdVec3::from([1.0, 2.0, 3.0]);
        v[1] = 5.0;
        assert_eq!(v, SimdVec3::from([1.0, 5.0, 3.0]));
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_index_mut_out_of_bounds_panics() {
        let mut v = SimdUVec3::from([1, 2, 3]);
        v[3] = 4;
    }

    #[test]
    fn test_ops_preserve_padding_zero() {
        let a = SimdVec3::from([1.0, 2.0, 3.0]);
        let b = SimdVec3::from([4.0, 5.0, 6.0]);
        assert_eq!(pad_f(a + b), 0.0);
        assert_eq!(pad_f(a - b), 0.0);
        assert_eq!(pad_f(a * b), 0.0);
        assert_eq!(pad_f(a / b), 0.0);
        assert_eq!(pad_f(a * 2.0), 0.0);
        assert_eq!(pad_f(2.0 * a), 0.0);
        assert_eq!(pad_f(a / 2.0), 0.0);
        assert_eq!(pad_f(-a), 0.0);
        assert_eq!(pad_f(a.elementwise_min(b)), 0.0);
        assert_eq!(pad_f(a.elementwise_max(b)), 0.0);
        assert_eq!(pad_f(SimdVec3::lerp(a, b, 0.5)), 0.0);
        assert_eq!(pad_f(a.clamp(SimdVec3::ZERO, b)), 0.0);
        assert_eq!(pad_f(a.powf(2.0)), 0.0);
        assert_eq!(pad_f(a.powf_elementwise(b)), 0.0);
        assert_eq!(pad_f(a.cross(b)), 0.0);
        assert_eq!(pad_f(a.normalized().unwrap()), 0.0);
        assert_eq!(pad_f(SimdVec3::default()), 0.0);
        assert_eq!(pad_f(SimdVec3::splat(9.0)), 0.0);
    }
}

#[cfg(test)]
mod conversion_and_scalar_op_tests {
    use super::*;

    #[test]
    fn test_scalar_times_vector() {
        assert_eq!(
            2.0 * SimdVec3::from([1.0, 2.0, 3.0]),
            SimdVec3::from([2.0, 4.0, 6.0])
        );
        assert_eq!(2 * SimdIVec2::from([1, 2]), SimdIVec2::from([2, 4]));
        assert_eq!(
            2u32 * SimdUVec4::from([1, 2, 3, 4]),
            SimdUVec4::from([2, 4, 6, 8])
        );
    }

    #[test]
    fn test_from_convert() {
        assert_eq!(
            SimdUVec2::from_convert([1u16, 2u16]),
            SimdUVec2::from([1, 2])
        );
        assert_eq!(
            SimdIVec3::from_convert([1i16, -2i16, 3i16]),
            SimdIVec3::from([1, -2, 3])
        );
        assert_eq!(
            SimdVec3::from_convert([1u8, 2u8, 3u8]),
            SimdVec3::from([1.0, 2.0, 3.0])
        );
        assert_eq!(
            SimdVec4::from_convert([1i8, 2i8, 3i8, 4i8]),
            SimdVec4::from([1.0, 2.0, 3.0, 4.0])
        );
    }

    #[test]
    fn test_into_vec_conversions() {
        assert_eq!(
            SimdVec2::from([1.0, 2.0]).into_vec3(3.0),
            SimdVec3::from([1.0, 2.0, 3.0])
        );
        assert_eq!(
            SimdVec3::from([1.0, 2.0, 3.0]).into_vec4(4.0),
            SimdVec4::from([1.0, 2.0, 3.0, 4.0])
        );
        assert_eq!(
            SimdVec4::from([1.0, 2.0, 3.0, 4.0]).into_vec3(),
            SimdVec3::from([1.0, 2.0, 3.0])
        );
    }

    #[test]
    fn test_debug_format_shows_visible_lanes_only() {
        assert_eq!(
            format!("{:?}", SimdVec3::from([1.0, 2.0, 3.0])),
            "SimdVec3([1.0, 2.0, 3.0])"
        );
        assert_eq!(
            format!("{:?}", SimdUVec2::from([1, 2])),
            "SimdUVec2([1, 2])"
        );
    }
}
