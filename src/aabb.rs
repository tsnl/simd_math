use crate::vector::{SimdVec2, SimdVec3};
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign};

macro_rules! impl_simd_aabb {
    ($name:ident, $vec_type:ty, $num_elements:expr) => {
        #[derive(Clone, Copy, Debug)]
        pub struct $name {
            pub min: $vec_type,
            pub max: $vec_type,
        }

        impl $name {
            /// Creates a new AABB with the given min and max bounds.
            #[inline]
            pub const fn new(min: $vec_type, max: $vec_type) -> Self {
                $name { min, max }
            }

            /// Returns the minimum bounds of the AABB.
            #[inline]
            pub fn min(&self) -> $vec_type {
                self.min
            }

            /// Returns the maximum bounds of the AABB.
            #[inline]
            pub fn max(&self) -> $vec_type {
                self.max
            }

            /// Returns the center point of the AABB.
            #[inline]
            pub fn center(&self) -> $vec_type {
                (self.min + self.max) * 0.5
            }

            /// Returns the extent (size) of the AABB.
            #[inline]
            pub fn extent(&self) -> $vec_type {
                self.max - self.min
            }

            /// Returns the identity element for union operations.
            /// This AABB will not affect the result when unioned with any other AABB.
            pub fn union_identity() -> Self {
                $name {
                    min: <$vec_type>::from([f32::INFINITY; $num_elements]),
                    max: <$vec_type>::from([f32::NEG_INFINITY; $num_elements]),
                }
            }

            /// Returns the identity element for intersection operations.
            /// This AABB will not affect the result when intersected with any other AABB.
            pub fn intersection_identity() -> Self {
                $name {
                    min: <$vec_type>::from([f32::NEG_INFINITY; $num_elements]),
                    max: <$vec_type>::from([f32::INFINITY; $num_elements]),
                }
            }
        }

        impl BitOrAssign for $name {
            fn bitor_assign(&mut self, other: Self) {
                *self |= other.min;
                *self |= other.max;
            }
        }

        impl BitOrAssign<$vec_type> for $name {
            fn bitor_assign(&mut self, other: $vec_type) {
                self.min = self.min.elementwise_min(other);
                self.max = self.max.elementwise_max(other);
            }
        }

        impl BitOr<$name> for $name {
            type Output = Self;

            fn bitor(mut self, other: Self) -> Self::Output {
                self |= other;
                self
            }
        }

        impl BitOr<$vec_type> for $name {
            type Output = Self;

            fn bitor(mut self, other: $vec_type) -> Self::Output {
                self |= other;
                self
            }
        }

        impl BitAndAssign for $name {
            fn bitand_assign(&mut self, other: Self) {
                self.min = self.min.elementwise_max(other.min);
                self.max = self.max.elementwise_min(other.max);
            }
        }

        impl BitAndAssign<$vec_type> for $name {
            fn bitand_assign(&mut self, other: $vec_type) {
                self.min = self.min.elementwise_max(other);
                self.max = self.max.elementwise_min(other);
            }
        }

        impl BitAnd<$name> for $name {
            type Output = Self;

            fn bitand(mut self, other: Self) -> Self::Output {
                self &= other;
                self
            }
        }

        impl BitAnd<$vec_type> for $name {
            type Output = Self;

            fn bitand(mut self, other: $vec_type) -> Self::Output {
                self &= other;
                self
            }
        }
    };
}

// Define the 2D and 3D AABB types
impl_simd_aabb!(SimdAabb2, SimdVec2, 2);
impl_simd_aabb!(SimdAabb3, SimdVec3, 3);

// Legacy alias for backward compatibility
pub type SimdAABB = SimdAabb3;

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! impl_aabb2_tests {
        ($test_mod:ident) => {
            mod $test_mod {
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

                fn assert_vec2_near(a: SimdVec2, b: SimdVec2) {
                    assert_f32_near(a.x(), b.x());
                    assert_f32_near(a.y(), b.y());
                }

                fn assert_aabb_near(a: SimdAabb2, b: SimdAabb2) {
                    assert_vec2_near(a.min, b.min);
                    assert_vec2_near(a.max, b.max);
                }

                #[test]
                fn test_aabb_new() {
                    let min = SimdVec2::new(1.0, 2.0);
                    let max = SimdVec2::new(4.0, 5.0);
                    let aabb = SimdAabb2::new(min, max);

                    assert_vec2_near(aabb.min(), min);
                    assert_vec2_near(aabb.max(), max);
                }

                #[test]
                fn test_aabb_accessors() {
                    let min = SimdVec2::new(-1.0, -2.0);
                    let max = SimdVec2::new(1.0, 2.0);
                    let aabb = SimdAabb2::new(min, max);

                    assert_vec2_near(aabb.min(), min);
                    assert_vec2_near(aabb.max(), max);
                }

                #[test]
                fn test_aabb_center() {
                    let min = SimdVec2::new(0.0, 0.0);
                    let max = SimdVec2::new(4.0, 6.0);
                    let aabb = SimdAabb2::new(min, max);

                    let center = aabb.center();
                    let expected = SimdVec2::new(2.0, 3.0);
                    assert_vec2_near(center, expected);
                }

                #[test]
                fn test_aabb_extent() {
                    let min = SimdVec2::new(1.0, 2.0);
                    let max = SimdVec2::new(5.0, 8.0);
                    let aabb = SimdAabb2::new(min, max);

                    let extent = aabb.extent();
                    let expected = SimdVec2::new(4.0, 6.0);
                    assert_vec2_near(extent, expected);
                }

                #[test]
                fn test_aabb_union_identity() {
                    let identity = SimdAabb2::union_identity();

                    // Union identity should have infinite min and negative infinite max
                    assert_eq!(identity.min.x(), f32::INFINITY);
                    assert_eq!(identity.min.y(), f32::INFINITY);
                    assert_eq!(identity.max.x(), f32::NEG_INFINITY);
                    assert_eq!(identity.max.y(), f32::NEG_INFINITY);

                    // Test that union with any AABB gives that AABB
                    let test_aabb =
                        SimdAabb2::new(SimdVec2::new(1.0, 2.0), SimdVec2::new(4.0, 5.0));
                    let result = identity | test_aabb;
                    assert_aabb_near(result, test_aabb);
                }

                #[test]
                fn test_aabb_intersection_identity() {
                    let identity = SimdAabb2::intersection_identity();

                    // Intersection identity should have negative infinite min and infinite max
                    assert_eq!(identity.min.x(), f32::NEG_INFINITY);
                    assert_eq!(identity.min.y(), f32::NEG_INFINITY);
                    assert_eq!(identity.max.x(), f32::INFINITY);
                    assert_eq!(identity.max.y(), f32::INFINITY);

                    // Test that intersection with any AABB gives that AABB
                    let test_aabb =
                        SimdAabb2::new(SimdVec2::new(1.0, 2.0), SimdVec2::new(4.0, 5.0));
                    let result = identity & test_aabb;
                    assert_aabb_near(result, test_aabb);
                }

                #[test]
                fn test_aabb_union_with_aabb() {
                    let aabb1 = SimdAabb2::new(SimdVec2::new(0.0, 0.0), SimdVec2::new(2.0, 2.0));
                    let aabb2 = SimdAabb2::new(SimdVec2::new(1.0, 1.0), SimdVec2::new(3.0, 3.0));
                    let expected = SimdAabb2::new(SimdVec2::new(0.0, 0.0), SimdVec2::new(3.0, 3.0));

                    let union = aabb1 | aabb2;
                    assert_aabb_near(union, expected);

                    // Test |= operator
                    let mut aabb3 = aabb1;
                    aabb3 |= aabb2;
                    assert_aabb_near(aabb3, expected);
                }

                #[test]
                fn test_aabb_union_with_point() {
                    let aabb = SimdAabb2::new(SimdVec2::new(1.0, 1.0), SimdVec2::new(3.0, 3.0));

                    // Point inside AABB - should not change AABB
                    let point_inside = SimdVec2::new(2.0, 2.0);
                    let result1 = aabb | point_inside;
                    assert_aabb_near(result1, aabb);

                    // Point outside AABB - should expand AABB
                    let point_outside = SimdVec2::new(0.0, 4.0);
                    let expected = SimdAabb2::new(SimdVec2::new(0.0, 1.0), SimdVec2::new(3.0, 4.0));
                    let result2 = aabb | point_outside;
                    assert_aabb_near(result2, expected);

                    // Test |= operator
                    let mut aabb3 = aabb;
                    aabb3 |= point_outside;
                    assert_aabb_near(aabb3, expected);
                }

                #[test]
                fn test_aabb_intersection_with_aabb() {
                    let aabb1 = SimdAabb2::new(SimdVec2::new(0.0, 0.0), SimdVec2::new(4.0, 4.0));
                    let aabb2 = SimdAabb2::new(SimdVec2::new(2.0, 2.0), SimdVec2::new(6.0, 6.0));
                    let expected = SimdAabb2::new(SimdVec2::new(2.0, 2.0), SimdVec2::new(4.0, 4.0));

                    let intersection = aabb1 & aabb2;
                    assert_aabb_near(intersection, expected);

                    // Test &= operator
                    let mut aabb3 = aabb1;
                    aabb3 &= aabb2;
                    assert_aabb_near(aabb3, expected);
                }

                #[test]
                fn test_aabb_intersection_with_point() {
                    let aabb = SimdAabb2::new(SimdVec2::new(1.0, 1.0), SimdVec2::new(5.0, 5.0));

                    // Point inside AABB - should shrink AABB towards point
                    let point = SimdVec2::new(3.0, 2.0);
                    let expected = SimdAabb2::new(SimdVec2::new(3.0, 2.0), SimdVec2::new(3.0, 2.0));
                    let result = aabb & point;
                    assert_aabb_near(result, expected);

                    // Test &= operator
                    let mut aabb2 = aabb;
                    aabb2 &= point;
                    assert_aabb_near(aabb2, expected);
                }

                #[test]
                fn test_aabb_no_intersection() {
                    let aabb1 = SimdAabb2::new(SimdVec2::new(0.0, 0.0), SimdVec2::new(2.0, 2.0));
                    let aabb2 = SimdAabb2::new(SimdVec2::new(3.0, 3.0), SimdVec2::new(5.0, 5.0));

                    let intersection = aabb1 & aabb2;

                    // When there's no intersection, the result should have min > max
                    assert!(intersection.min.x() > intersection.max.x());
                    assert!(intersection.min.y() > intersection.max.y());
                }

                #[test]
                fn test_aabb_comprehensive_operations() {
                    // Start with union identity
                    let mut aabb = SimdAabb2::union_identity();

                    // Add some points to build an AABB
                    let points = vec![
                        SimdVec2::new(1.0, 2.0),
                        SimdVec2::new(-1.0, 4.0),
                        SimdVec2::new(3.0, 0.0),
                    ];

                    for point in points.iter() {
                        aabb |= *point;
                    }

                    let expected =
                        SimdAabb2::new(SimdVec2::new(-1.0, 0.0), SimdVec2::new(3.0, 4.0));
                    assert_aabb_near(aabb, expected);

                    // Test center and extent of the built AABB
                    let center = aabb.center();
                    let extent = aabb.extent();
                    let expected_center = SimdVec2::new(1.0, 2.0);
                    let expected_extent = SimdVec2::new(4.0, 4.0);
                    assert_vec2_near(center, expected_center);
                    assert_vec2_near(extent, expected_extent);
                }
            }
        };
    }

    macro_rules! impl_aabb3_tests {
        ($test_mod:ident) => {
            mod $test_mod {
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

                fn assert_vec3_near(a: SimdVec3, b: SimdVec3) {
                    assert_f32_near(a.x(), b.x());
                    assert_f32_near(a.y(), b.y());
                    assert_f32_near(a.z(), b.z());
                }

                fn assert_aabb_near(a: SimdAabb3, b: SimdAabb3) {
                    assert_vec3_near(a.min, b.min);
                    assert_vec3_near(a.max, b.max);
                }

                #[test]
                fn test_aabb_new() {
                    let min = SimdVec3::new(1.0, 2.0, 3.0);
                    let max = SimdVec3::new(4.0, 5.0, 6.0);
                    let aabb = SimdAabb3::new(min, max);

                    assert_vec3_near(aabb.min(), min);
                    assert_vec3_near(aabb.max(), max);
                }

                #[test]
                fn test_aabb_accessors() {
                    let min = SimdVec3::new(-1.0, -2.0, -3.0);
                    let max = SimdVec3::new(1.0, 2.0, 3.0);
                    let aabb = SimdAabb3::new(min, max);

                    assert_vec3_near(aabb.min(), min);
                    assert_vec3_near(aabb.max(), max);
                }

                #[test]
                fn test_aabb_center() {
                    let min = SimdVec3::new(0.0, 0.0, 0.0);
                    let max = SimdVec3::new(4.0, 6.0, 8.0);
                    let aabb = SimdAabb3::new(min, max);

                    let center = aabb.center();
                    let expected = SimdVec3::new(2.0, 3.0, 4.0);
                    assert_vec3_near(center, expected);
                }

                #[test]
                fn test_aabb_extent() {
                    let min = SimdVec3::new(1.0, 2.0, 3.0);
                    let max = SimdVec3::new(5.0, 8.0, 11.0);
                    let aabb = SimdAabb3::new(min, max);

                    let extent = aabb.extent();
                    let expected = SimdVec3::new(4.0, 6.0, 8.0);
                    assert_vec3_near(extent, expected);
                }

                #[test]
                fn test_aabb_union_identity() {
                    let identity = SimdAabb3::union_identity();

                    // Union identity should have infinite min and negative infinite max
                    assert_eq!(identity.min.x(), f32::INFINITY);
                    assert_eq!(identity.min.y(), f32::INFINITY);
                    assert_eq!(identity.min.z(), f32::INFINITY);
                    assert_eq!(identity.max.x(), f32::NEG_INFINITY);
                    assert_eq!(identity.max.y(), f32::NEG_INFINITY);
                    assert_eq!(identity.max.z(), f32::NEG_INFINITY);

                    // Test that union with any AABB gives that AABB
                    let test_aabb =
                        SimdAabb3::new(SimdVec3::new(1.0, 2.0, 3.0), SimdVec3::new(4.0, 5.0, 6.0));
                    let result = identity | test_aabb;
                    assert_aabb_near(result, test_aabb);
                }

                #[test]
                fn test_aabb_intersection_identity() {
                    let identity = SimdAabb3::intersection_identity();

                    // Intersection identity should have negative infinite min and infinite max
                    assert_eq!(identity.min.x(), f32::NEG_INFINITY);
                    assert_eq!(identity.min.y(), f32::NEG_INFINITY);
                    assert_eq!(identity.min.z(), f32::NEG_INFINITY);
                    assert_eq!(identity.max.x(), f32::INFINITY);
                    assert_eq!(identity.max.y(), f32::INFINITY);
                    assert_eq!(identity.max.z(), f32::INFINITY);

                    // Test that intersection with any AABB gives that AABB
                    let test_aabb =
                        SimdAabb3::new(SimdVec3::new(1.0, 2.0, 3.0), SimdVec3::new(4.0, 5.0, 6.0));
                    let result = identity & test_aabb;
                    assert_aabb_near(result, test_aabb);
                }

                #[test]
                fn test_aabb_union_with_aabb() {
                    let aabb1 =
                        SimdAabb3::new(SimdVec3::new(0.0, 0.0, 0.0), SimdVec3::new(2.0, 2.0, 2.0));
                    let aabb2 =
                        SimdAabb3::new(SimdVec3::new(1.0, 1.0, 1.0), SimdVec3::new(3.0, 3.0, 3.0));
                    let expected =
                        SimdAabb3::new(SimdVec3::new(0.0, 0.0, 0.0), SimdVec3::new(3.0, 3.0, 3.0));

                    let union = aabb1 | aabb2;
                    assert_aabb_near(union, expected);

                    // Test |= operator
                    let mut aabb3 = aabb1;
                    aabb3 |= aabb2;
                    assert_aabb_near(aabb3, expected);
                }

                #[test]
                fn test_aabb_union_with_point() {
                    let aabb =
                        SimdAabb3::new(SimdVec3::new(1.0, 1.0, 1.0), SimdVec3::new(3.0, 3.0, 3.0));

                    // Point inside AABB - should not change AABB
                    let point_inside = SimdVec3::new(2.0, 2.0, 2.0);
                    let result1 = aabb | point_inside;
                    assert_aabb_near(result1, aabb);

                    // Point outside AABB - should expand AABB
                    let point_outside = SimdVec3::new(0.0, 4.0, 2.0);
                    let expected =
                        SimdAabb3::new(SimdVec3::new(0.0, 1.0, 1.0), SimdVec3::new(3.0, 4.0, 3.0));
                    let result2 = aabb | point_outside;
                    assert_aabb_near(result2, expected);

                    // Test |= operator
                    let mut aabb3 = aabb;
                    aabb3 |= point_outside;
                    assert_aabb_near(aabb3, expected);
                }

                #[test]
                fn test_aabb_intersection_with_aabb() {
                    let aabb1 =
                        SimdAabb3::new(SimdVec3::new(0.0, 0.0, 0.0), SimdVec3::new(4.0, 4.0, 4.0));
                    let aabb2 =
                        SimdAabb3::new(SimdVec3::new(2.0, 2.0, 2.0), SimdVec3::new(6.0, 6.0, 6.0));
                    let expected =
                        SimdAabb3::new(SimdVec3::new(2.0, 2.0, 2.0), SimdVec3::new(4.0, 4.0, 4.0));

                    let intersection = aabb1 & aabb2;
                    assert_aabb_near(intersection, expected);

                    // Test &= operator
                    let mut aabb3 = aabb1;
                    aabb3 &= aabb2;
                    assert_aabb_near(aabb3, expected);
                }

                #[test]
                fn test_aabb_intersection_with_point() {
                    let aabb =
                        SimdAabb3::new(SimdVec3::new(1.0, 1.0, 1.0), SimdVec3::new(5.0, 5.0, 5.0));

                    // Point inside AABB - should shrink AABB towards point
                    let point = SimdVec3::new(3.0, 2.0, 4.0);
                    let expected =
                        SimdAabb3::new(SimdVec3::new(3.0, 2.0, 4.0), SimdVec3::new(3.0, 2.0, 4.0));
                    let result = aabb & point;
                    assert_aabb_near(result, expected);

                    // Test &= operator
                    let mut aabb2 = aabb;
                    aabb2 &= point;
                    assert_aabb_near(aabb2, expected);
                }

                #[test]
                fn test_aabb_no_intersection() {
                    let aabb1 =
                        SimdAabb3::new(SimdVec3::new(0.0, 0.0, 0.0), SimdVec3::new(2.0, 2.0, 2.0));
                    let aabb2 =
                        SimdAabb3::new(SimdVec3::new(3.0, 3.0, 3.0), SimdVec3::new(5.0, 5.0, 5.0));

                    let intersection = aabb1 & aabb2;

                    // When there's no intersection, the result should have min > max
                    assert!(intersection.min.x() > intersection.max.x());
                    assert!(intersection.min.y() > intersection.max.y());
                    assert!(intersection.min.z() > intersection.max.z());
                }

                #[test]
                fn test_aabb_comprehensive_operations() {
                    // Start with union identity
                    let mut aabb = SimdAabb3::union_identity();

                    // Add some points to build an AABB
                    let points = vec![
                        SimdVec3::new(1.0, 2.0, 3.0),
                        SimdVec3::new(-1.0, 4.0, 1.0),
                        SimdVec3::new(3.0, 0.0, 5.0),
                    ];

                    for point in points.iter() {
                        aabb |= *point;
                    }

                    let expected =
                        SimdAabb3::new(SimdVec3::new(-1.0, 0.0, 1.0), SimdVec3::new(3.0, 4.0, 5.0));
                    assert_aabb_near(aabb, expected);

                    // Test center and extent of the built AABB
                    let center = aabb.center();
                    let extent = aabb.extent();
                    let expected_center = SimdVec3::new(1.0, 2.0, 3.0);
                    let expected_extent = SimdVec3::new(4.0, 4.0, 4.0);
                    assert_vec3_near(center, expected_center);
                    assert_vec3_near(extent, expected_extent);
                }
            }
        };
    }

    impl_aabb2_tests!(simd_aabb2_tests);
    impl_aabb3_tests!(simd_aabb3_tests);

    // Legacy tests for backward compatibility
    mod simd_aabb_legacy_tests {
        use super::*;

        #[test]
        fn test_legacy_type_alias() {
            let aabb = SimdAABB::new(SimdVec3::new(1.0, 2.0, 3.0), SimdVec3::new(4.0, 5.0, 6.0));
            let aabb3 = SimdAabb3::new(SimdVec3::new(1.0, 2.0, 3.0), SimdVec3::new(4.0, 5.0, 6.0));

            // Ensure they're the same type
            assert_eq!(
                std::mem::size_of::<SimdAABB>(),
                std::mem::size_of::<SimdAabb3>()
            );
            assert_eq!(aabb.min.x(), aabb3.min.x());
            assert_eq!(aabb.min.y(), aabb3.min.y());
            assert_eq!(aabb.min.z(), aabb3.min.z());
            assert_eq!(aabb.max.x(), aabb3.max.x());
            assert_eq!(aabb.max.y(), aabb3.max.y());
            assert_eq!(aabb.max.z(), aabb3.max.z());
        }
    }
}
