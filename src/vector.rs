use super::*;
use num_traits::identities::ConstZero;
use std::fmt::Debug;
use std::ops::{Add, Div, Index, Mul, Neg, Sub};

pub trait SimdVector: Sized {
    type LaneType: Sized + ConstZero + Copy + Debug + PartialEq + PartialOrd;
}

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

        impl SimdVector for $name {
            type LaneType = $lane_ty;
        }

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
            /// Compute the squared Euclidean norm of the vector.
            /// This is more efficient than computing the norm itself.
            #[inline]
            pub fn norm_squared(self) -> $lane_ty {
                self.dot(self)
            }

            /// Compute the Euclidean norm (magnitude) of the vector.
            #[inline]
            pub fn norm(self) -> $lane_ty {
                self.norm_squared().sqrt()
            }

            /// Normalize the vector by dividing it by its norm.
            #[inline]
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
            pub fn lerp(a: Self, b: Self, t: $lane_ty) -> Self {
                let k_a = <$simd_ty>::splat(1.0 - t);
                let k_b = <$simd_ty>::splat(t);
                let res = a.0 * k_a + b.0 * k_b;
                Self(res)
            }

            /// Clamp each component of the vector between the corresponding components of min and max
            #[inline]
            pub fn clamp(self, min: Self, max: Self) -> Self {
                let clamped = self.0.simd_clamp(min.0, max.0);
                Self(clamped)
            }

            /// Element-wise power operation with scalar exponent
            #[inline]
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
                let v = <$vec_type>::new($test_values[0], $test_values[1]);
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
                let v = <$vec_type>::new($test_values[0], $test_values[1], $test_values[2]);
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
                let v = <$vec_type>::new(
                    $test_values[0],
                    $test_values[1],
                    $test_values[2],
                    $test_values[3],
                );
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

    macro_rules! test_simd_vec_base_methods_int {
        ($vec_type:ty, $lane_ty:ty, $dim:expr, $test_values:expr) => {
            // Use regular base methods but with integer types that support Ord
            test_simd_vec_base_methods!($vec_type, $lane_ty, $dim, $test_values);
        };
    }

    macro_rules! test_simd_vec_base_methods {
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
                let v = <$vec_type>::new($test_values[0], $test_values[1]);
                let neg_v = -v;
                assert_eq!(neg_v[0], -$test_values[0]);
                assert_eq!(neg_v[1], -$test_values[1]);
            }
        };
        ($vec_type:ty, 3, $test_values:expr) => {
            #[test]
            fn test_negation() {
                let v = <$vec_type>::new($test_values[0], $test_values[1], $test_values[2]);
                let neg_v = -v;
                assert_eq!(neg_v[0], -$test_values[0]);
                assert_eq!(neg_v[1], -$test_values[1]);
                assert_eq!(neg_v[2], -$test_values[2]);
            }
        };
        ($vec_type:ty, 4, $test_values:expr) => {
            #[test]
            fn test_negation() {
                let v = <$vec_type>::new(
                    $test_values[0],
                    $test_values[1],
                    $test_values[2],
                    $test_values[3],
                );
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
                let v = <$vec_type>::new(3.0, 4.0);
                assert_f32_near(v.norm_squared(), 25.0);
                assert_f32_near(v.norm(), 5.0);
            }

            #[test]
            fn test_normalized() {
                let v = <$vec_type>::new(3.0, 4.0);
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
                let v = <$vec_type>::new(3.0, 4.0, 0.0);
                assert_f32_near(v.norm_squared(), 25.0);
                assert_f32_near(v.norm(), 5.0);
            }

            #[test]
            fn test_normalized() {
                let v = <$vec_type>::new(3.0, 4.0, 0.0);
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
                let v = <$vec_type>::new(3.0, 4.0, 0.0, 0.0);
                assert_f32_near(v.norm_squared(), 25.0);
                assert_f32_near(v.norm(), 5.0);
            }

            #[test]
            fn test_normalized() {
                let v = <$vec_type>::new(3.0, 4.0, 0.0, 0.0);
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
                let v = <$vec_type>::new(1.0, 2.0);
                assert_eq!(v.x(), 1.0);
                assert_eq!(v.y(), 2.0);
            }
        };
        ($vec_type:ty, [x: 0, y: 1, z: 2]) => {
            #[test]
            fn test_field_accessors() {
                let v = <$vec_type>::new(1.0, 2.0, 3.0);
                assert_eq!(v.x(), 1.0);
                assert_eq!(v.y(), 2.0);
                assert_eq!(v.z(), 3.0);
            }
        };
        ($vec_type:ty, [x: 0, y: 1, z: 2, w: 3]) => {
            #[test]
            fn test_field_accessors() {
                let v = <$vec_type>::new(1.0, 2.0, 3.0, 4.0);
                assert_eq!(v.x(), 1.0);
                assert_eq!(v.y(), 2.0);
                assert_eq!(v.z(), 3.0);
                assert_eq!(v.w(), 4.0);
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
        test_simd_vec_base_methods!(SimdVec2, f32, 2, [1.0, 2.0, 3.0, 4.0]);
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
        test_simd_vec_base_methods!(SimdVec3, f32, 3, [1.0, 2.0, 3.0, 2.0]);
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
            let v1 = SimdVec3::new(1.0, 2.0, 3.0);
            let v2 = SimdVec3::new(4.0, 5.0, 6.0);
            let cross = v1.cross(v2);

            // Cross product should be perpendicular to both vectors
            assert_f32_near(cross.dot(v1), 0.0);
            assert_f32_near(cross.dot(v2), 0.0);

            // Test cross product with parallel vectors (should be zero)
            let parallel1 = SimdVec3::new(1.0, 2.0, 3.0);
            let parallel2 = SimdVec3::new(2.0, 4.0, 6.0);
            let zero_cross = parallel1.cross(parallel2);
            assert_f32_near(zero_cross.norm(), 0.0);
        }
    }

    mod simd_vec4_tests {
        use super::*;

        test_simd_vec_ctor_methods!(SimdVec4, f32, 4, [1.0, 2.0, 3.0, 4.0]);
        test_simd_vec_base_methods!(SimdVec4, f32, 4, [1.0, 2.0, 3.0, 2.0]);
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
            let v4 = SimdVec4::new(1.0, 2.0, 3.0, 4.0);
            let v3 = v4.into_vec3();
            assert_f32_near(v3.x(), 1.0);
            assert_f32_near(v3.y(), 2.0);
            assert_f32_near(v3.z(), 3.0);
        }
    }

    // Integer vector tests - separate modules to avoid name conflicts
    mod simd_ivec2_tests {
        use super::*;
        test_simd_vec_ctor_methods!(SimdIVec2, i32, 2, [1, 2, 3, 4]);
        test_simd_vec_base_methods_int!(SimdIVec2, i32, 2, [1, 2, 3, 2]);
        test_simd_vec_neg_method!(SimdIVec2, 2, [1, -2, 3, 4]);
    }

    mod simd_ivec3_tests {
        use super::*;
        test_simd_vec_ctor_methods!(SimdIVec3, i32, 3, [1, 2, 3, 4]);
        test_simd_vec_base_methods_int!(SimdIVec3, i32, 3, [1, 2, 3, 2]);
        test_simd_vec_neg_method!(SimdIVec3, 3, [1, -2, 3, 4]);
    }

    mod simd_ivec4_tests {
        use super::*;
        test_simd_vec_ctor_methods!(SimdIVec4, i32, 4, [1, 2, 3, 4]);
        test_simd_vec_base_methods_int!(SimdIVec4, i32, 4, [1, 2, 3, 2]);
        test_simd_vec_neg_method!(SimdIVec4, 4, [1, -2, 3, -4]);
    }

    // Unsigned integer vector tests - separate modules to avoid name conflicts
    mod simd_uvec2_tests {
        use super::*;
        test_simd_vec_ctor_methods!(SimdUVec2, u32, 2, [1, 2, 3, 4]);
        test_simd_vec_base_methods_int!(SimdUVec2, u32, 2, [5, 4, 3, 2]);
    }

    mod simd_uvec3_tests {
        use super::*;
        test_simd_vec_ctor_methods!(SimdUVec3, u32, 3, [1, 2, 3, 4]);
        test_simd_vec_base_methods_int!(SimdUVec3, u32, 3, [5, 4, 3, 2]);
    }

    mod simd_uvec4_tests {
        use super::*;
        test_simd_vec_ctor_methods!(SimdUVec4, u32, 4, [1, 2, 3, 4]);
        test_simd_vec_base_methods_int!(SimdUVec4, u32, 4, [5, 4, 3, 2]);
    }
}
