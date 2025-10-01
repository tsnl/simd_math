//--------------------------------------------------------------------------------------------------
// Vector types:
//--------------------------------------------------------------------------------------------------

use num_traits::identities::ConstZero;
use std::fmt::Debug;
use std::ops::{Add, Div, Index, Mul, Neg, Sub};
use std::simd::prelude::*;

macro_rules! impl_float_simd_vec {
    ($name:ident : $simd_ty:ty [ $lane_ty:ty ; $dim:expr ] { $($field_name:ident),+ } [ $($init:expr),+ ]) => {
        impl_basic_simd_vec!($name : $simd_ty [ $lane_ty ; $dim ] { $($field_name),+ } [ $($init),+ ]);
        impl_simd_vec_neg_method!($name : $simd_ty [ $lane_ty ; $dim ]);
        impl_simd_vec_float_methods!($name : $simd_ty [ $lane_ty ; $dim ]);
    }
}
macro_rules! impl_int_simd_vec {
    ($name:ident : $simd_ty:ty [ $lane_ty:ty ; $dim:expr ] { $($field_name:ident),+ } [ $($init:expr),+ ]) => {
        impl_basic_simd_vec!($name : $simd_ty [ $lane_ty ; $dim ] { $($field_name),+ } [ $($init),+ ]);
        impl_simd_vec_neg_method!($name : $simd_ty [ $lane_ty ; $dim ]);
    };
}
macro_rules! impl_uint_simd_vec {
    ($name:ident : $simd_ty:ty [ $lane_ty:ty ; $dim:expr ] { $($field_name:ident),+ } [ $($init:expr),+ ]) => {
        impl_basic_simd_vec!($name : $simd_ty [ $lane_ty ; $dim ] { $($field_name),+ } [ $($init),+ ]);
    }
}

macro_rules! impl_basic_simd_vec {
    ($name:ident : $simd_ty:ty [ $lane_ty:ty ; $dim:expr ] { $($field_name:ident),+ } [ $($init:expr),+ ]) => {
        #[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
        pub struct $name(pub(crate) $simd_ty);

        impl_simd_vec_ctor_methods!($name : $simd_ty [ $lane_ty ; $dim ] { $($field_name),+ } [ $($init),+ ]);
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
    ($name:ident : $simd_ty:ty [ $lane_ty:ty ; $dim:expr ] { $($field_name:ident),+ } [ $($init:expr),+ ]) => {
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

impl_uint_simd_vec!(SimdUVec4: u32x4 [u32; 4] { x, y, z, w } [ x, y, z, w ]);
impl_uint_simd_vec!(SimdUVec3: u32x4 [u32; 3] { x, y, z } [ x, y, z, 0 ]);
impl_uint_simd_vec!(SimdUVec2: u32x2 [u32; 2] { x, y } [ x, y ]);

impl_int_simd_vec!(SimdIVec4: i32x4 [i32; 4] { x, y, z, w } [ x, y, z, w ]);
impl_int_simd_vec!(SimdIVec3: i32x4 [i32; 3] { x, y, z } [ x, y, z, 0 ]);
impl_int_simd_vec!(SimdIVec2: i32x2 [i32; 2] { x, y } [ x, y ]);

impl_float_simd_vec!(SimdVec4: f32x4 [f32; 4] { x, y, z, w } [ x, y, z, w ]);
impl_float_simd_vec!(SimdVec3: f32x4 [f32; 3] { x, y, z } [ x, y, z, 0.0 ]);
impl_float_simd_vec!(SimdVec2: f32x2 [f32; 2] { x, y } [ x, y ]);

impl SimdVec2 {
    pub const ZERO: Self = Self(f32x2::from_array([0.0, 0.0]));
    pub const UNIT_X: Self = Self(f32x2::from_array([1.0, 0.0]));
    pub const UNIT_Y: Self = Self(f32x2::from_array([0.0, 1.0]));
}
impl SimdVec3 {
    pub const ZERO: Self = Self(f32x4::from_array([0.0, 0.0, 0.0, 0.0]));
    pub const UNIT_X: Self = Self(f32x4::from_array([1.0, 0.0, 0.0, 0.0]));
    pub const UNIT_Y: Self = Self(f32x4::from_array([0.0, 1.0, 0.0, 0.0]));
    pub const UNIT_Z: Self = Self(f32x4::from_array([0.0, 0.0, 1.0, 0.0]));
}
impl SimdVec4 {
    pub const ZERO: Self = Self(f32x4::from_array([0.0, 0.0, 0.0, 0.0]));
    pub const UNIT_X: Self = Self(f32x4::from_array([1.0, 0.0, 0.0, 0.0]));
    pub const UNIT_Y: Self = Self(f32x4::from_array([0.0, 1.0, 0.0, 0.0]));
    pub const UNIT_Z: Self = Self(f32x4::from_array([0.0, 0.0, 1.0, 0.0]));
    pub const UNIT_W: Self = Self(f32x4::from_array([0.0, 0.0, 0.0, 1.0]));
}

impl SimdVec2 {
    /// Convert spherical coordinate angles (azimuth, elevation) to equirectangular texture coordinates (u, v).
    ///
    /// Both azimuth and elevation are expected to be in radians.
    /// - azimuth: angle in the horizontal plane, range [-π, π]
    /// - elevation: angle in the vertical plane, range [-π/2, π/2]
    ///
    /// Returns UV coordinates where:
    /// - u is in range [0, 1] (left to right across the texture)
    /// - v is in range [0, 1] (top to bottom of the texture)
    pub fn spherical_coords_angles_into_equirectangular_coords(self) -> Self {
        let azimuth = self.x();
        let elevation = self.y();

        // Convert azimuth from [-π, π] to [0, 1]
        let u = (azimuth + std::f32::consts::PI) / (2.0 * std::f32::consts::PI);

        // Convert elevation from [-π/2, π/2] to [0, 1]
        // Note: we flip the v coordinate so that positive elevation maps to the top of the texture
        let v = 0.5 - (elevation / std::f32::consts::PI);

        SimdVec2::new(u, v)
    }
}

impl SimdVec4 {
    pub fn into_vec3(self) -> SimdVec3 {
        SimdVec3::new(self.x(), self.y(), self.z())
    }
}

impl SimdVec3 {
    /// Convert a Cartesian vector to spherical coordinates.
    /// Returns (azimuth, elevation, radius) where:
    /// - azimuth: angle in the horizontal plane, range [-π, π]
    /// - elevation: angle in the vertical plane, range [-π/2, π/2]
    /// - radius: distance from origin
    pub fn into_spherical_coords(self) -> Self {
        let x = self.x();
        let y = self.y();
        let z = self.z();

        let radius = self.norm();

        if radius == 0.0 {
            return SimdVec3::new(0.0, 0.0, 0.0);
        }

        let azimuth = z.atan2(x);
        let elevation = (y / radius).asin();

        SimdVec3::new(azimuth, elevation, radius)
    }

    /// Cross product of two 3D vectors.
    /// Returns a vector perpendicular to both input vectors.
    /// The magnitude of the result is |a| * |b| * sin(θ) where θ is the angle between the vectors.
    /// The direction follows the right-hand rule.
    pub fn cross(self, other: SimdVec3) -> SimdVec3 {
        let a = self;
        let b = other;

        let result_x = a.y() * b.z() - a.z() * b.y();
        let result_y = a.z() * b.x() - a.x() * b.z();
        let result_z = a.x() * b.y() - a.y() * b.x();

        SimdVec3::new(result_x, result_y, result_z)
    }
}

#[cfg(test)]
mod simd_vec3_tests {
    use super::*;

    #[test]
    fn test_cross() {
        // Test cross product of unit vectors
        let x_axis = SimdVec3::new(1.0, 0.0, 0.0);
        let y_axis = SimdVec3::new(0.0, 1.0, 0.0);
        let z_axis = SimdVec3::new(0.0, 0.0, 1.0);

        // Right-hand rule: x × y = z
        let cross_xy = x_axis.cross(y_axis);
        assert!((cross_xy.x() - 0.0).abs() < 1e-6);
        assert!((cross_xy.y() - 0.0).abs() < 1e-6);
        assert!((cross_xy.z() - 1.0).abs() < 1e-6);

        // y × z = x
        let cross_yz = y_axis.cross(z_axis);
        assert!((cross_yz.x() - 1.0).abs() < 1e-6);
        assert!((cross_yz.y() - 0.0).abs() < 1e-6);
        assert!((cross_yz.z() - 0.0).abs() < 1e-6);

        // z × x = y
        let cross_zx = z_axis.cross(x_axis);
        assert!((cross_zx.x() - 0.0).abs() < 1e-6);
        assert!((cross_zx.y() - 1.0).abs() < 1e-6);
        assert!((cross_zx.z() - 0.0).abs() < 1e-6);
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
