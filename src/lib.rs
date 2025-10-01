#![feature(portable_simd)]

use std::simd::prelude::*;

use num_traits::identities::ConstZero;
use std::fmt::Debug;
use std::ops::{Add, BitAnd, BitAndAssign, BitOr, BitOrAssign, Div, Index, Mul, Neg, Sub};

//--------------------------------------------------------------------------------------------------
// Vector types:
//--------------------------------------------------------------------------------------------------

macro_rules! impl_float_simd_vec {
    ($short_name:ident : $name:ident : $simd_ty:ty [ $lane_ty:ty ; $dim:expr ] { $($field_name:ident),+ } [ $($init:expr),+ ]) => {
        impl_basic_simd_vec!($short_name : $name : $simd_ty [ $lane_ty ; $dim ] { $($field_name),+ } [ $($init),+ ]);
        impl_simd_vec_neg_method!($name : $simd_ty [ $lane_ty ; $dim ]);
        impl_simd_vec_float_methods!($name : $simd_ty [ $lane_ty ; $dim ]);
    }
}
macro_rules! impl_int_simd_vec {
    ($short_name:ident : $name:ident : $simd_ty:ty [ $lane_ty:ty ; $dim:expr ] { $($field_name:ident),+ } [ $($init:expr),+ ]) => {
        impl_basic_simd_vec!($short_name : $name : $simd_ty [ $lane_ty ; $dim ] { $($field_name),+ } [ $($init),+ ]);
        impl_simd_vec_neg_method!($name : $simd_ty [ $lane_ty ; $dim ]);
    };
}
macro_rules! impl_uint_simd_vec {
    ($short_name:ident : $name:ident : $simd_ty:ty [ $lane_ty:ty ; $dim:expr ] { $($field_name:ident),+ } [ $($init:expr),+ ]) => {
        impl_basic_simd_vec!($short_name : $name : $simd_ty [ $lane_ty ; $dim ] { $($field_name),+ } [ $($init),+ ]);
    }
}

macro_rules! impl_basic_simd_vec {
    ($short_name:ident : $name:ident : $simd_ty:ty [ $lane_ty:ty ; $dim:expr ] { $($field_name:ident),+ } [ $($init:expr),+ ]) => {
        #[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
        pub struct $name($simd_ty);

        impl_simd_vec_ctor_methods!($short_name : $name : $simd_ty [ $lane_ty ; $dim ] { $($field_name),+ } [ $($init),+ ]);
        impl_simd_vec_base_methods!($name : $simd_ty [ $lane_ty ; $dim ]);
        impl_simd_vec_field_methods!($name $lane_ty [ $($field_name),+ ] );
    };
}

macro_rules! help_apply_to_array {
    ($func:expr ; $array:expr ; [ $_0:ident , $_1:ident , $_2:ident , $_3:ident ]) => {
        $func($array[0], $array[1], $array[2], $array[3])
    };
    ($func:expr ; $array:expr ; [ $_0:ident , $_1:ident , $_2:ident ]) => {
        $func($array[0], $array[1], $array[2])
    };
    ($func:expr ; $array:expr ; [ $_0:ident , $_1:ident ]) => {
        $func($array[0], $array[1])
    };
}
macro_rules! impl_simd_vec_ctor_methods {
    ($short_name:ident : $name:ident : $simd_ty:ty [ $lane_ty:ty ; $dim:expr ] { $($field_name:ident),+ } [ $($init:expr),+ ]) => {
        pub fn $short_name( $($field_name: $lane_ty),+ ) -> $name {
            $name(<$simd_ty>::from_array([$($init),+]))
        }
        impl $name {
            #[inline]
            pub const fn new($($field_name: $lane_ty),+) -> Self {
                $name(<$simd_ty>::from_array([$($init),+]))
            }
            #[inline]
            pub fn splat(value: $lane_ty) -> Self {
                $name(<$simd_ty>::splat(value))
            }
        }
        impl From<[$lane_ty; $dim]> for $name {
            #[inline]
            fn from(arr: [$lane_ty; $dim]) -> Self {
                help_apply_to_array!($name::new ; arr ; [ $($field_name),+ ])
            }
        }
        impl From<$name> for [$lane_ty; $dim] {
            #[inline]
            fn from(it: $name) -> [$lane_ty; $dim] {
                it.0.as_array()[..$dim].try_into().unwrap()
            }
        }
    };
}

macro_rules! impl_simd_vec_base_methods {
    ($name:ident : $simd_ty:ty [ $lane_ty:ty ; $dim:expr ]) => {
        impl Default for $name {
            #[inline]
            fn default() -> Self {
                $name(<$simd_ty>::splat(<$lane_ty>::ZERO)) // Default to zero vector
            }
        }
        impl $name {
            #[inline]
            pub fn dot(self, other: $name) -> $lane_ty {
                (self.0 * other.0).reduce_sum()
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
        impl Mul<$lane_ty> for $name {
            type Output = $name;
            #[inline]
            fn mul(self, rhs: $lane_ty) -> Self::Output {
                $name(self.0 * <$simd_ty>::splat(rhs))
            }
        }
        impl Div<$lane_ty> for $name {
            type Output = $name;
            #[inline]
            fn div(self, rhs: $lane_ty) -> Self::Output {
                $name(self.0 / <$simd_ty>::splat(rhs))
            }
        }
        impl Div<$name> for $name {
            type Output = $name;
            /// Element-wise division
            #[inline]
            fn div(self, rhs: $name) -> Self::Output {
                $name(self.0 / rhs.0)
            }
        }
        impl Add<$name> for $name {
            type Output = $name;
            #[inline]
            fn add(self, rhs: $name) -> Self::Output {
                $name(self.0 + rhs.0)
            }
        }
        impl Sub<$name> for $name {
            type Output = $name;
            #[inline]
            fn sub(self, rhs: $name) -> Self::Output {
                $name(self.0 - rhs.0)
            }
        }
        impl $name {
            pub fn elementwise_min(self, other: $name) -> $name {
                $name(self.0.simd_min(other.0))
            }
            pub fn elementwise_max(self, other: $name) -> $name {
                $name(self.0.simd_max(other.0))
            }
        }
        impl Index<usize> for $name {
            type Output = $lane_ty;
            #[inline]
            fn index(&self, index: usize) -> &Self::Output {
                &self.0[index]
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
            #[inline]
            pub fn norm_squared(self) -> $lane_ty {
                self.dot(self)
            }

            #[inline]
            pub fn norm(self) -> $lane_ty {
                self.norm_squared().sqrt()
            }

            #[inline]
            pub fn normalized(self) -> Option<Self> {
                // Normalize the vector by dividing it by its norm.
                let norm = self.norm();
                if norm == 0.0 {
                    // Avoid division by zero
                    None
                } else {
                    Some(Self(self.0 / <$simd_ty>::splat(norm)))
                }
            }

            #[inline]
            pub fn lerp(a: Self, b: Self, t: $lane_ty) -> Self {
                // Linear interpolation between two vectors a and b by factor t
                let k_a = <$simd_ty>::splat(1.0 - t);
                let k_b = <$simd_ty>::splat(t);
                let res = a.0 * k_a + b.0 * k_b;
                Self(res)
            }

            #[inline]
            pub fn powf(self, exponent: $lane_ty) -> Self {
                // Element-wise power operation
                let self_array = *self.0.as_array();
                let mut result = self_array; // Start with the same array structure
                for i in 0..$dim {
                    result[i] = self_array[i].powf(exponent);
                }
                Self(<$simd_ty>::from_array(result))
            }

            #[inline]
            pub fn powf_elementwise(self, exponent: Self) -> Self {
                // Element-wise power operation with SIMD exponent vector
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

macro_rules! impl_simd_vec_field_methods {
    ( $arg0:ident $lane_ty:ty [ $name0:ident ] ) => {
        impl_simd_vec_field_methods!( $arg0 $lane_ty [ $name0 : 0 ] );
    };

    ( $arg0:ident $lane_ty:ty [ $name0:ident , $name1:ident ] ) => {
        impl_simd_vec_field_methods!( $arg0 $lane_ty [ $name0 : 0 , $name1 : 1 ] );
    };

    ( $arg0:ident $lane_ty:ty [ $name0:ident , $name1:ident , $name2:ident ] ) => {
        impl_simd_vec_field_methods!( $arg0 $lane_ty [ $name0 : 0 , $name1 : 1 , $name2 : 2 ] );
    };

    ( $arg0:ident $lane_ty:ty [ $name0:ident , $name1:ident , $name2:ident , $name3:ident ] ) => {
        impl_simd_vec_field_methods!( $arg0 $lane_ty [ $name0 : 0 , $name1 : 1 , $name2 : 2 , $name3 : 3 ] );
    };

    // Inner case:
    ($vec_name:ident $lane_ty:ty [ $( $field_name:ident : $index:expr ),+ ]) => {
        impl $vec_name {
            $(
                #[inline]
                pub fn $field_name(self) -> $lane_ty {
                    self.0[$index]
                }
            )+
        }
    };
}

impl_uint_simd_vec!(uvec4: SimdUVec4: u32x4 [u32; 4] { x, y, z, w } [ x, y, z, w ]);
impl_uint_simd_vec!(uvec3: SimdUVec3: u32x4 [u32; 3] { x, y, z } [ x, y, z, 0 ]);
impl_uint_simd_vec!(uvec2: SimdUVec2: u32x2 [u32; 2] { x, y } [ x, y ]);

impl_int_simd_vec!(ivec4: SimdIVec4: i32x4 [i32; 4] { x, y, z, w } [ x, y, z, w ]);
impl_int_simd_vec!(ivec3: SimdIVec3: i32x4 [i32; 3] { x, y, z } [ x, y, z, 0 ]);
impl_int_simd_vec!(ivec2: SimdIVec2: i32x2 [i32; 2] { x, y } [ x, y ]);

impl_float_simd_vec!(vec4: SimdVec4: f32x4 [f32; 4] { x, y, z, w } [ x, y, z, w ]);
impl_float_simd_vec!(vec3: SimdVec3: f32x4 [f32; 3] { x, y, z } [ x, y, z, 0.0 ]);
impl_float_simd_vec!(vec2: SimdVec2: f32x2 [f32; 2] { x, y } [ x, y ]);

impl SimdVec2 {
    pub const ZERO: Self = Self::new(0.0, 0.0);
    pub const UNIT_X: Self = Self::new(1.0, 0.0);
    pub const UNIT_Y: Self = Self::new(1.0, 0.0);
}
impl SimdVec3 {
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0);
    pub const UNIT_X: Self = Self::new(1.0, 0.0, 0.0);
    pub const UNIT_Y: Self = Self::new(0.0, 1.0, 0.0);
    pub const UNIT_Z: Self = Self::new(0.0, 0.0, 1.0);
}
impl SimdVec4 {
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0, 0.0);
    pub const UNIT_X: Self = Self::new(1.0, 0.0, 0.0, 0.0);
    pub const UNIT_Y: Self = Self::new(0.0, 1.0, 0.0, 0.0);
    pub const UNIT_Z: Self = Self::new(0.0, 0.0, 1.0, 0.0);
    pub const UNIT_W: Self = Self::new(0.0, 0.0, 0.0, 1.0);
}

impl SimdVec2 {
    /// Converts spherical coordinate angles into equirectangular UV coordinates.
    /// Input: x = azimuth in radians, y = elevation in radians
    /// Output: UV coordinates in [0,1]^2 space for equirectangular projection
    /// where (0,0) is top-left and (1,1) is bottom-right
    #[inline]
    pub fn spherical_coords_angles_into_equirectangular_coords(self) -> SimdVec2 {
        let azimuth = self.x();
        let elevation = self.y();

        // Convert azimuth from [-π, π] to [0, 1]
        // azimuth = -π maps to u = 0, azimuth = π maps to u = 1
        let u = (azimuth + std::f32::consts::PI) / (2.0 * std::f32::consts::PI);

        // Convert elevation from [-π/2, π/2] to [0, 1]
        // elevation = π/2 (up) maps to v = 0 (top of texture)
        // elevation = -π/2 (down) maps to v = 1 (bottom of texture)
        let v = 1.0 - ((elevation + std::f32::consts::FRAC_PI_2) / std::f32::consts::PI);

        SimdVec2::new(u, v)
    }
}

impl SimdVec4 {
    pub fn into_vec3(self) -> SimdVec3 {
        // Convert SimdVec4 to SimdVec3 by dropping the last component (w), setting to 0
        SimdVec3(simd_swizzle!(self.0, [0, 1, 2, 3]) * f32x4::from_array([1.0, 1.0, 1.0, 0.0]))
    }
}

impl SimdVec3 {
    /// Converts a 3D direction vector into spherical coordinates.
    /// Returns SimdVec3 where:
    /// - x = azimuth in radians (angle around Y axis, from -π to π)
    /// - y = elevation in radians (angle from XZ plane, from -π/2 to π/2)
    /// - z = magnitude (distance from origin)
    #[inline]
    pub fn into_spherical_coords(self) -> SimdVec3 {
        let x = self.x();
        let y = self.y();
        let z = self.z();

        // Calculate magnitude
        let magnitude = self.norm();

        // Handle zero vector case
        if magnitude < f32::EPSILON {
            return SimdVec3::new(0.0, 0.0, 0.0);
        }

        // Calculate azimuth (angle around Y axis)
        // atan2(z, x) gives angle from positive X axis towards positive Z axis
        let azimuth = z.atan2(x);

        // Calculate elevation (angle from XZ plane)
        // asin(y / magnitude) gives angle from XZ plane towards positive Y axis
        let elevation = (y / magnitude).asin();

        SimdVec3::new(azimuth, elevation, magnitude)
    }

    /// Computes the cross product of two vec3 simd registers.
    /// Last component is guaranteed to be zero after this operation.
    #[inline]
    pub fn cross(self, other: SimdVec3) -> SimdVec3 {
        //          i   j   k
        // lt   = [lx, ly, lz, _]
        // rt   = [rx, ry, rz, _]

        // t1   = diagonal from top-left to bottom-right:
        //      = [lx * ry, ly * rz, lz * rx, lw * rw]
        // t2   = diagonal from top-right to bottom-left:
        //      = [lx * rz, ly * rx, lz * ry, lw * rw]
        // t3   = t2 shuffled <<= 1 (keep w in place)
        //      = [ly * rx, lz * ry, lx * rz, lw * rw]
        // t4   = t1 - t3
        //      = [lx * ry, ly * rz, lz * rx, lw * rw
        //        -ly * rx|-lz * ry|-lx * rz|-lw * rw]
        //            k        i        j        0

        let t1 = self.0 * simd_swizzle!(other.0, [1, 2, 0, 3]);
        let t2 = self.0 * simd_swizzle!(other.0, [2, 0, 1, 3]);
        let t3 = simd_swizzle!(t2, [1, 2, 0, 3]);
        let t4 = t1 - t3;

        // Marshalling results:
        Self(simd_swizzle!(t4, [1, 2, 0, 3]))
    }
}

#[cfg(test)]
mod simd_vec3_tests {
    use super::*;

    #[test]
    fn test_cross() {
        // i x j = k
        let v1 = vec3(1.0, 0.0, 0.0);
        let v2 = vec3(0.0, 1.0, 0.0);
        assert_eq!(v1.cross(v2), vec3(0.0, 0.0, 1.0));

        // j x k = i
        let v1 = vec3(0.0, 1.0, 0.0);
        let v2 = vec3(0.0, 0.0, 1.0);
        assert_eq!(v1.cross(v2), vec3(1.0, 0.0, 0.0));

        // i x k = -j
        let v1 = vec3(1.0, 0.0, 0.0);
        let v2 = vec3(0.0, 0.0, 1.0);
        assert_eq!(v1.cross(v2), vec3(0.0, -1.0, 0.0));

        // Parallel vectors:
        let v1 = vec3(1.0, 0.0, 0.0);
        let v2 = vec3(1.0, 0.0, 0.0);
        assert_eq!(v1.cross(v2), vec3(0.0, 0.0, 0.0));

        // Cross with zero vector:
        let v1 = vec3(1.0, 2.0, 3.0);
        let v2 = vec3(0.0, 0.0, 0.0);
        assert_eq!(v1.cross(v2), vec3(0.0, 0.0, 0.0));
    }
}

//--------------------------------------------------------------------------------------------------
// SimdQuat: we only deal with unit quaternions.
//--------------------------------------------------------------------------------------------------

/// Quaternion represented as an f32x4 SIMD datatype [s, x, y, z]
/// IMPORTANT: norm is assumed to always be 1. Constructors ensure this.
/// Reference: https://www.3dgep.com/understanding-quaternions/#Rotations
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct SimdUnitQuat(f32x4); // s, x, y, z

impl SimdUnitQuat {
    pub const IDENTITY: Self = Self(f32x4::from_array([1.0, 0.0, 0.0, 0.0]));
}

impl Default for SimdUnitQuat {
    #[inline]
    fn default() -> Self {
        SimdUnitQuat::new(1.0, 0.0, 0.0, 0.0) // Identity quaternion
    }
}

impl From<[f32; 4]> for SimdUnitQuat {
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
    #[inline]
    pub fn new(s: f32, x: f32, y: f32, z: f32) -> Self {
        let q = f32x4::from_array([s, x, y, z]);
        let n = q / f32x4::splat((q * q).reduce_sum().sqrt());
        SimdUnitQuat(n)
    }

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
    pub fn conjugate(self) -> Self {
        SimdUnitQuat(self.0 * f32x4::from_array([1.0, -1.0, -1.0, -1.0]))
    }

    /// Inverse: for unit quaternions, q* = q⁻¹
    #[inline]
    pub fn inverse(self) -> Self {
        self.conjugate()
    }

    #[inline]
    pub fn real(self) -> f32 {
        self.0[0]
    }

    #[inline]
    pub fn imag(self) -> SimdVec3 {
        SimdVec3(simd_swizzle!(self.0, [1, 2, 3, 0]) * f32x4::from_array([1.0, 1.0, 1.0, 0.0]))
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

    #[inline]
    fn mul(self, rhs: SimdVec3) -> Self::Output {
        // Same logic as above, but since we cannot construct Quaternions of non-unit length, we
        // have to manually rewrite the multiplication. However, we can assume that s2 is 0 since
        // rhs is a pure-imaginary quaternion, which saves us some work.

        // Calculate p_intermediate = self * rhs_pure_quaternion
        // rhs_pure_quaternion has scalar part 0 and vector part rhs.
        // Scalar part of p_intermediate: -self.imag().dot(rhs)
        let s_p = -self.imag().dot(rhs);

        // Vector part of p_intermediate: self.real()*rhs + self.imag().cross(rhs)
        let v_p = rhs * self.real() + self.imag().cross(rhs);

        // Combine the two by summing the scalar and vector parts for p_intermediate:
        let p = SimdUnitQuat(
            f32x4::from_array([s_p, 0.0, 0.0, 0.0]) + simd_swizzle!(v_p.0, [3, 0, 1, 2]),
        );

        // Finally, post-multiply by the rotor's inverse (just the conjugate, so cheap) to obtain a
        // rotated vector.
        // This is equivalent to: result_pure_quat := p_intermediate * self.inverse()
        // then take imag(result_pure_quat)
        (p * self.inverse()).imag()
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
    fn test_quat_default_is_identity() {
        assert_quat_eq(
            SimdUnitQuat::default(),
            SimdUnitQuat::new(1.0, 0.0, 0.0, 0.0),
        );
    }

    #[test]
    fn test_quat_from_axis_angle() {
        // 90 degree rotation around Z axis
        let axis = vec3(0.0, 0.0, 1.0);
        let angle = PI / 2.0;
        let q = SimdUnitQuat::from_axis_angle(axis, angle);
        // s = cos(pi/4), x=0, y=0, z = sin(pi/4)
        let expected_s = (PI / 4.0).cos();
        let expected_z = (PI / 4.0).sin();
        assert_quat_eq(q, SimdUnitQuat::new(expected_s, 0.0, 0.0, expected_z));

        // 180 degree rotation around Y axis
        let axis = vec3(0.0, 1.0, 0.0);
        let angle = PI;
        let q = SimdUnitQuat::from_axis_angle(axis, angle);
        // s = cos(pi/2) = 0, x=0, y=sin(pi/2)=1, z=0
        assert_quat_eq(q, SimdUnitQuat::new(0.0, 0.0, 1.0, 0.0));
    }

    #[test]
    #[should_panic]
    fn test_quat_from_axis_angle_zero_axis() {
        SimdUnitQuat::from_axis_angle(vec3(0.0, 0.0, 0.0), PI / 2.0);
    }

    #[test]
    fn test_quat_conjugate_inverse() {
        let q = SimdUnitQuat::from_axis_angle(vec3(1.0, 2.0, 3.0), PI / 3.0);
        let q_conj = q.conjugate();
        let q_inv = q.inverse();

        assert_quat_eq(q_conj, q_inv); // For unit quaternions, conjugate is inverse

        // q * q_conj should be identity [1,0,0,0]
        let identity = q * q_conj; // Fixed typo: q_conjugate -> q_conj
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
        assert_vec3_eq(q.imag(), vec3(x, y, z));
    }

    #[test]
    fn test_quat_mul_quat() {
        // Identity * q = q
        let q = SimdUnitQuat::from_axis_angle(vec3(1.0, 1.0, 1.0), PI / 4.0);
        assert_quat_eq(SimdUnitQuat::default() * q, q);
        assert_quat_eq(q * SimdUnitQuat::default(), q);

        // Rotation by q1 then q2 is q2 * q1
        // 90 deg around Z: q_z = [cos(pi/4), 0, 0, sin(pi/4)]
        let q_z = SimdUnitQuat::from_axis_angle(vec3(0.0, 0.0, 1.0), PI / 2.0);
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
        let q_x = SimdUnitQuat::from_axis_angle(vec3(1.0, 0.0, 0.0), PI / 2.0);
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

        let v = vec3(1.0, 0.0, 0.0);
        let rotated_v_manual = q_x * (q_z * v);

        let q_combined = q_x * q_z;
        let rotated_v_combined = q_combined * v;

        assert_vec3_eq(rotated_v_manual, rotated_v_combined);
        assert_vec3_eq(rotated_v_combined, vec3(0.0, 0.0, 1.0));
    }

    #[test]
    fn test_quat_mul_vec3_rotation() {
        // Rotate i=(1,0,0) by 90deg around Z-axis -> j=(0,1,0)
        let q_rot_z90 = SimdUnitQuat::from_axis_angle(vec3(0.0, 0.0, 1.0), PI / 2.0);
        let i = vec3(1.0, 0.0, 0.0);
        let j = vec3(0.0, 1.0, 0.0);
        assert_vec3_eq(q_rot_z90 * i, j);

        // Rotate j=(0,1,0) by 90deg around X-axis -> k=(0,0,1)
        let q_rot_x90 = SimdUnitQuat::from_axis_angle(vec3(1.0, 0.0, 0.0), PI / 2.0);
        let k = vec3(0.0, 0.0, 1.0);
        assert_vec3_eq(q_rot_x90 * j, k);

        // Rotate k=(0,0,1) by 90deg around Y-axis -> i=(1,0,0)
        let q_rot_y90 = SimdUnitQuat::from_axis_angle(vec3(0.0, 1.0, 0.0), PI / 2.0);
        assert_vec3_eq(q_rot_y90 * k, i);

        // Rotate (1,1,0) by -90deg around Z-axis -> (1,-1,0) (normalized)
        let v_in = vec3(1.0, 1.0, 0.0).normalized().unwrap();
        let q_rot_z_neg90 = SimdUnitQuat::from_axis_angle(vec3(0.0, 0.0, 1.0), -PI / 2.0);
        let v_expected = vec3(1.0, -1.0, 0.0).normalized().unwrap();
        assert_vec3_eq(q_rot_z_neg90 * v_in, v_expected);
    }
}

//--------------------------------------------------------------------------------------------------
// SimdMat4:
//--------------------------------------------------------------------------------------------------

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
        im_x: SimdVec3,
        im_y: SimdVec3,
        im_z: SimdVec3,
        translation: SimdVec3,
    ) -> Self {
        SimdMat4([im_x, im_y, im_z, translation])
    }

    #[inline]
    pub fn identity() -> Self {
        SimdMat4([
            SimdVec3::new(1.0, 0.0, 0.0),
            SimdVec3::new(0.0, 1.0, 0.0),
            SimdVec3::new(0.0, 0.0, 1.0),
            SimdVec3::new(0.0, 0.0, 0.0),
        ])
    }

    #[inline]
    pub fn from_translation(translation: SimdVec3) -> Self {
        SimdMat4([
            SimdVec3::new(1.0, 0.0, 0.0),
            SimdVec3::new(0.0, 1.0, 0.0),
            SimdVec3::new(0.0, 0.0, 1.0),
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
            SimdVec3::new(
                1.0 - 2.0 * (y2 + z2),
                2.0 * (xy + sz), //
                2.0 * (xz - sy), //
            ),
            SimdVec3::new(
                2.0 * (xy - sz), //
                1.0 - 2.0 * (x2 + z2),
                2.0 * (yz + sx), //
            ),
            SimdVec3::new(
                2.0 * (xz + sy), //
                2.0 * (yz - sx), //
                1.0 - 2.0 * (x2 + y2),
            ),
            SimdVec3::default(),
        ])
    }
}

impl Mul<SimdVec3> for SimdMat4 {
    type Output = SimdVec3;

    /// Matrix-vector3 multiplication: scale each transformed basis vector by the corresponding
    /// coefficient (X, Y, or Z) and sum, then add translation.
    /// Equivalent to converting the vec3 into a vec4 with a 1.0 in the W component.
    #[inline]
    fn mul(self, rhs: SimdVec3) -> Self::Output {
        let [i, j, k, t] = self.0;
        i * rhs.x() + j * rhs.y() + k * rhs.z() + t
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

        assert_vec3_eq(m_ident.0[0], vec3(1.0, 0.0, 0.0));
        assert_vec3_eq(m_ident.0[1], vec3(0.0, 1.0, 0.0));
        assert_vec3_eq(m_ident.0[2], vec3(0.0, 0.0, 1.0));
        assert_vec3_eq(m_ident.0[3], vec3(0.0, 0.0, 0.0));
    }

    #[test]
    fn test_mat4_new_and_constructors() {
        let t = vec3(1.0, 2.0, 3.0);
        let m_trans = SimdMat4::from_translation(t);
        assert_vec3_eq(m_trans.0[3], t);
        assert_vec3_eq(m_trans.0[0], vec3(1.0, 0.0, 0.0));

        let ix = vec3(1.0, 0.0, 0.0);
        let iy = vec3(0.0, 1.0, 0.0);
        let iz = vec3(0.0, 0.0, 1.0);
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
        let q_rot_z90 = SimdUnitQuat::from_axis_angle(vec3(0.0, 0.0, 1.0), PI / 2.0);
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
            vec3(0.0, 1.0, 0.0),  // Column X (after rotation, original X becomes Y)
            vec3(-1.0, 0.0, 0.0), // Column Y (original Y becomes -X)
            vec3(0.0, 0.0, 1.0),  // Column Z (Z remains Z)
            SimdVec3::default(),  // Translation
        );
        assert_mat4_eq(m_rot_z90, expected_m_rot_z90);
    }

    #[test]
    fn test_mat4_mul_vec3() {
        let m_ident = SimdMat4::identity();
        let v = vec3(1.0, 2.0, 3.0);
        assert_vec3_eq(m_ident * v, v);

        let t = vec3(10.0, 20.0, 30.0);
        let m_trans = SimdMat4::from_translation(t);
        assert_vec3_eq(m_trans * v, v + t);

        // Rotation: 90 deg around Z
        let q_rot_z90 = SimdUnitQuat::from_axis_angle(vec3(0.0, 0.0, 1.0), PI / 2.0);
        let m_rot_z90 = SimdMat4::from(q_rot_z90);
        // (1,2,3) rotated by 90 deg around Z -> (-2,1,3)
        assert_vec3_eq(m_rot_z90 * v, vec3(-2.0, 1.0, 3.0));

        // Combined rotation and translation
        let m_combined = m_trans * m_rot_z90; // Rotate then translate
        let rotated_v = m_rot_z90 * v;
        let translated_rotated_v = m_trans * rotated_v;
        assert_vec3_eq(m_combined * v, translated_rotated_v);
        assert_vec3_eq(m_combined * v, vec3(-2.0, 1.0, 3.0) + t);
    }

    #[test]
    fn test_mat4_mul_mat4() {
        let m_ident = SimdMat4::identity();
        let m_a = SimdMat4::from_translation(vec3(1.0, 2.0, 3.0));
        assert_mat4_eq(m_ident * m_a, m_a);
        assert_mat4_eq(m_a * m_ident, m_a);

        let t1 = vec3(1.0, 2.0, 3.0);
        let m_t1 = SimdMat4::from_translation(t1);
        let t2 = vec3(10.0, 20.0, 30.0);
        let m_t2 = SimdMat4::from_translation(t2);

        // M_t1 * M_t2 should result in a translation by t1 + t2
        let m_t1_t2 = m_t1 * m_t2;
        let expected_m_trans_sum = SimdMat4::from_translation(t1 + t2);
        assert_mat4_eq(m_t1_t2, expected_m_trans_sum);

        // Rotation matrices
        let q_rot_z90 = SimdUnitQuat::from_axis_angle(vec3(0.0, 0.0, 1.0), PI / 2.0);
        let m_rot_z90 = SimdMat4::from(q_rot_z90);
        let q_rot_x90 = SimdUnitQuat::from_axis_angle(vec3(1.0, 0.0, 0.0), PI / 2.0);
        let m_rot_x90 = SimdMat4::from(q_rot_x90);

        // M_rot_x90 * M_rot_z90 (rotate by Z then by X)
        let m_combined_rot = m_rot_x90 * m_rot_z90;

        // Equivalent quaternion rotation: q_rot_x90 * q_rot_z90
        let q_combined_rot = q_rot_x90 * q_rot_z90;
        let m_expected_combined_rot = SimdMat4::from(q_combined_rot);
        assert_mat4_eq(m_combined_rot, m_expected_combined_rot);

        // Test with a vector
        let v = vec3(1.0, 0.0, 0.0);
        // (1,0,0) -> Z-rot -> (0,1,0) -> X-rot -> (0,0,1)
        assert_vec3_eq(m_combined_rot * v, vec3(0.0, 0.0, 1.0));
    }
}

//--------------------------------------------------------------------------------------------------
// SimdTransform:
//--------------------------------------------------------------------------------------------------

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
        let trans = vec3(1.0, 2.0, 3.0);
        let rot = SimdUnitQuat::from_axis_angle(vec3(0.0, 0.0, 1.0), PI / 2.0);
        let t = SimdTransform::new(trans, rot);
        assert_vec3_eq(t.position, trans);
        assert_quat_eq(t.rotation, rot);
    }

    #[test]
    fn test_transform_mul_transform() {
        let t_ident = SimdTransform::identity();

        let trans1 = vec3(1.0, 2.0, 3.0);
        let rot1 = SimdUnitQuat::from_axis_angle(vec3(0.0, 0.0, 1.0), PI / 2.0); // 90 deg Z rot
        let tf1 = SimdTransform::new(trans1, rot1);

        // Identity
        assert_transform_eq(tf1 * t_ident, tf1);
        assert_transform_eq(t_ident * tf1, tf1);

        let trans2 = vec3(10.0, 0.0, 0.0);
        let rot2 = SimdUnitQuat::from_axis_angle(vec3(1.0, 0.0, 0.0), PI); // 180 deg X rot
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
        assert_vec3_eq(composed_tf.position, vec3(1.0, 12.0, 3.0));

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
        let axis = vec3(0.0, 0.0, 1.0);
        let angle = PI / 4.0;
        let t = SimdTransform::new(
            vec3(1.0, 2.0, 3.0),
            SimdUnitQuat::from_axis_angle(axis, angle),
        );
        let inv_t = t.inverse();
        assert_transform_eq(t * inv_t, SimdTransform::default()); // Identity inverse is itself
    }
}

//--------------------------------------------------------------------------------------------------
// SimdAABB:
//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct SimdAABB {
    pub min: SimdVec3,
    pub max: SimdVec3,
}
impl SimdAABB {
    pub fn union_identity() -> Self {
        SimdAABB {
            min: SimdVec3::from([f32::INFINITY; 3]),
            max: SimdVec3::from([f32::NEG_INFINITY; 3]),
        }
    }
    pub fn intersection_identity() -> Self {
        SimdAABB {
            min: SimdVec3::from([f32::NEG_INFINITY; 3]),
            max: SimdVec3::from([f32::INFINITY; 3]),
        }
    }
}
impl SimdAABB {
    #[inline]
    pub const fn new(min: SimdVec3, max: SimdVec3) -> Self {
        SimdAABB { min, max }
    }

    #[inline]
    pub fn min(&self) -> SimdVec3 {
        self.min
    }

    #[inline]
    pub fn max(&self) -> SimdVec3 {
        self.max
    }
}
impl SimdAABB {
    #[inline]
    pub fn center(&self) -> SimdVec3 {
        (self.min + self.max) * 0.5
    }
    #[inline]
    pub fn extent(&self) -> SimdVec3 {
        self.max - self.min
    }
}
impl BitOrAssign for SimdAABB {
    fn bitor_assign(&mut self, other: Self) {
        *self |= other.min;
        *self |= other.max;
    }
}
impl BitOrAssign<SimdVec3> for SimdAABB {
    fn bitor_assign(&mut self, other: SimdVec3) {
        self.min = self.min.elementwise_min(other);
        self.max = self.max.elementwise_max(other);
    }
}
impl BitOr<SimdAABB> for SimdAABB {
    type Output = Self;

    fn bitor(mut self, other: Self) -> Self::Output {
        self |= other;
        self
    }
}
impl BitOr<SimdVec3> for SimdAABB {
    type Output = Self;

    fn bitor(mut self, other: SimdVec3) -> Self::Output {
        self |= other;
        self
    }
}
impl BitAndAssign for SimdAABB {
    fn bitand_assign(&mut self, other: Self) {
        *self &= other.min;
        *self &= other.max;
    }
}
impl BitAndAssign<SimdVec3> for SimdAABB {
    fn bitand_assign(&mut self, other: SimdVec3) {
        self.min = self.min.elementwise_max(other);
        self.max = self.max.elementwise_min(other);
    }
}
impl BitAnd<SimdAABB> for SimdAABB {
    type Output = Self;

    fn bitand(mut self, other: Self) -> Self::Output {
        self &= other;
        self
    }
}
impl BitAnd<SimdVec3> for SimdAABB {
    type Output = Self;

    fn bitand(mut self, other: SimdVec3) -> Self::Output {
        self &= other;
        self
    }
}

#[cfg(test)]
mod spherical_coords_tests {
    use super::*;

    const EPSILON: f32 = 1e-6;

    fn assert_vec3_near(a: SimdVec3, b: SimdVec3, epsilon: f32) {
        assert!((a.x() - b.x()).abs() < epsilon, "x: {} vs {}", a.x(), b.x());
        assert!((a.y() - b.y()).abs() < epsilon, "y: {} vs {}", a.y(), b.y());
        assert!((a.z() - b.z()).abs() < epsilon, "z: {} vs {}", a.z(), b.z());
    }

    fn assert_vec2_near(a: SimdVec2, b: SimdVec2, epsilon: f32) {
        assert!((a.x() - b.x()).abs() < epsilon, "x: {} vs {}", a.x(), b.x());
        assert!((a.y() - b.y()).abs() < epsilon, "y: {} vs {}", a.y(), b.y());
    }

    #[test]
    fn test_into_spherical_coords_basic_directions() {
        // Test positive X axis
        let pos_x = SimdVec3::new(1.0, 0.0, 0.0);
        let spherical = pos_x.into_spherical_coords();
        assert_vec3_near(spherical, SimdVec3::new(0.0, 0.0, 1.0), EPSILON);

        // Test positive Z axis
        let pos_z = SimdVec3::new(0.0, 0.0, 1.0);
        let spherical = pos_z.into_spherical_coords();
        assert_vec3_near(
            spherical,
            SimdVec3::new(std::f32::consts::FRAC_PI_2, 0.0, 1.0),
            EPSILON,
        );

        // Test negative Z axis
        let neg_z = SimdVec3::new(0.0, 0.0, -1.0);
        let spherical = neg_z.into_spherical_coords();
        assert_vec3_near(
            spherical,
            SimdVec3::new(-std::f32::consts::FRAC_PI_2, 0.0, 1.0),
            EPSILON,
        );

        // Test positive Y axis (up)
        let pos_y = SimdVec3::new(0.0, 1.0, 0.0);
        let spherical = pos_y.into_spherical_coords();
        assert_vec3_near(
            spherical,
            SimdVec3::new(0.0, std::f32::consts::FRAC_PI_2, 1.0),
            EPSILON,
        );

        // Test negative Y axis (down)
        let neg_y = SimdVec3::new(0.0, -1.0, 0.0);
        let spherical = neg_y.into_spherical_coords();
        assert_vec3_near(
            spherical,
            SimdVec3::new(0.0, -std::f32::consts::FRAC_PI_2, 1.0),
            EPSILON,
        );
    }

    #[test]
    fn test_into_spherical_coords_zero_vector() {
        let zero = SimdVec3::new(0.0, 0.0, 0.0);
        let spherical = zero.into_spherical_coords();
        assert_vec3_near(spherical, SimdVec3::new(0.0, 0.0, 0.0), EPSILON);
    }

    #[test]
    fn test_into_spherical_coords_magnitude() {
        // Test that magnitude is preserved
        let vec = SimdVec3::new(3.0, 4.0, 0.0);
        let spherical = vec.into_spherical_coords();
        assert!((spherical.z() - 5.0).abs() < EPSILON); // 3-4-5 triangle
    }

    #[test]
    fn test_spherical_coords_angles_into_equirectangular_coords() {
        // Test center of texture (azimuth=0, elevation=0)
        let center = SimdVec2::new(0.0, 0.0);
        let uv = center.spherical_coords_angles_into_equirectangular_coords();
        assert_vec2_near(uv, SimdVec2::new(0.5, 0.5), EPSILON);

        // Test positive azimuth (π/2, looking towards +Z)
        let pos_azimuth = SimdVec2::new(std::f32::consts::FRAC_PI_2, 0.0);
        let uv = pos_azimuth.spherical_coords_angles_into_equirectangular_coords();
        assert_vec2_near(uv, SimdVec2::new(0.75, 0.5), EPSILON);

        // Test negative azimuth (-π/2, looking towards -Z)
        let neg_azimuth = SimdVec2::new(-std::f32::consts::FRAC_PI_2, 0.0);
        let uv = neg_azimuth.spherical_coords_angles_into_equirectangular_coords();
        assert_vec2_near(uv, SimdVec2::new(0.25, 0.5), EPSILON);

        // Test positive elevation (π/2, looking up)
        let pos_elevation = SimdVec2::new(0.0, std::f32::consts::FRAC_PI_2);
        let uv = pos_elevation.spherical_coords_angles_into_equirectangular_coords();
        assert_vec2_near(uv, SimdVec2::new(0.5, 0.0), EPSILON);

        // Test negative elevation (-π/2, looking down)
        let neg_elevation = SimdVec2::new(0.0, -std::f32::consts::FRAC_PI_2);
        let uv = neg_elevation.spherical_coords_angles_into_equirectangular_coords();
        assert_vec2_near(uv, SimdVec2::new(0.5, 1.0), EPSILON);
    }

    #[test]
    fn test_spherical_coords_boundary_values() {
        // Test azimuth boundaries
        let left_edge = SimdVec2::new(-std::f32::consts::PI, 0.0);
        let uv = left_edge.spherical_coords_angles_into_equirectangular_coords();
        assert_vec2_near(uv, SimdVec2::new(0.0, 0.5), EPSILON);

        let right_edge = SimdVec2::new(std::f32::consts::PI, 0.0);
        let uv = right_edge.spherical_coords_angles_into_equirectangular_coords();
        assert_vec2_near(uv, SimdVec2::new(1.0, 0.5), EPSILON);
    }
}
