use super::*;

use std::{
    f32,
    ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign},
};

macro_rules! impl_simd_rect {
    ($name:ident, $vec_type:ty, $num_elements:expr, $lo:expr, $hi:expr) => {
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
                (self.min + self.max) / <$vec_type>::splat(2 as <$vec_type as SimdVector>::LaneType)
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
                    min: <$vec_type>::splat($hi),
                    max: <$vec_type>::splat($lo),
                }
            }

            /// Returns the identity element for intersection operations.
            /// This AABB will not affect the result when intersected with any other AABB.
            pub fn intersection_identity() -> Self {
                $name {
                    min: <$vec_type>::splat($lo),
                    max: <$vec_type>::splat($hi),
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

impl_simd_rect!(SimdRect2, SimdVec2, 2, f32::NEG_INFINITY, f32::INFINITY);
impl_simd_rect!(SimdURect2, SimdUVec2, 2, u32::MIN, u32::MAX);
impl_simd_rect!(SimdIRect2, SimdIVec2, 2, i32::MIN, i32::MAX);

impl_simd_rect!(SimdRect3, SimdVec3, 3, f32::NEG_INFINITY, f32::INFINITY);
impl_simd_rect!(SimdURect3, SimdUVec3, 3, u32::MIN, u32::MAX);
impl_simd_rect!(SimdIRect3, SimdIVec3, 3, i32::MIN, i32::MAX);

//--------------------------------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    mod simd_rect2 {
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

        fn assert_rect_near(a: SimdRect2, b: SimdRect2) {
            assert_vec2_near(a.min, b.min);
            assert_vec2_near(a.max, b.max);
        }

        #[test]
        fn test_rect_new() {
            let min = SimdVec2::new(1.0, 2.0);
            let max = SimdVec2::new(4.0, 5.0);
            let rect = SimdRect2::new(min, max);

            assert_vec2_near(rect.min(), min);
            assert_vec2_near(rect.max(), max);
        }

        #[test]
        fn test_rect_accessors() {
            let min = SimdVec2::new(-1.0, -2.0);
            let max = SimdVec2::new(1.0, 2.0);
            let rect = SimdRect2::new(min, max);

            assert_vec2_near(rect.min(), min);
            assert_vec2_near(rect.max(), max);
        }

        #[test]
        fn test_rect_center() {
            let min = SimdVec2::new(0.0, 0.0);
            let max = SimdVec2::new(4.0, 6.0);
            let rect = SimdRect2::new(min, max);

            let center = rect.center();
            let expected = SimdVec2::new(2.0, 3.0);
            assert_vec2_near(center, expected);
        }

        #[test]
        fn test_rect_extent() {
            let min = SimdVec2::new(1.0, 2.0);
            let max = SimdVec2::new(5.0, 8.0);
            let rect = SimdRect2::new(min, max);

            let extent = rect.extent();
            let expected = SimdVec2::new(4.0, 6.0);
            assert_vec2_near(extent, expected);
        }

        #[test]
        fn test_rect_union_identity() {
            let identity = SimdRect2::union_identity();

            // Union identity should have infinite min and negative infinite max
            assert_eq!(identity.min.x(), f32::INFINITY);
            assert_eq!(identity.min.y(), f32::INFINITY);
            assert_eq!(identity.max.x(), f32::NEG_INFINITY);
            assert_eq!(identity.max.y(), f32::NEG_INFINITY);

            // Test that union with any AABB gives that AABB
            let test_rect = SimdRect2::new(SimdVec2::new(1.0, 2.0), SimdVec2::new(4.0, 5.0));
            let result = identity | test_rect;
            assert_rect_near(result, test_rect);
        }

        #[test]
        fn test_rect_intersection_identity() {
            let identity = SimdRect2::intersection_identity();

            // Intersection identity should have negative infinite min and infinite max
            assert_eq!(identity.min.x(), f32::NEG_INFINITY);
            assert_eq!(identity.min.y(), f32::NEG_INFINITY);
            assert_eq!(identity.max.x(), f32::INFINITY);
            assert_eq!(identity.max.y(), f32::INFINITY);

            // Test that intersection with any AABB gives that AABB
            let test_rect = SimdRect2::new(SimdVec2::new(1.0, 2.0), SimdVec2::new(4.0, 5.0));
            let result = identity & test_rect;
            assert_rect_near(result, test_rect);
        }

        #[test]
        fn test_rect_union_with_rect() {
            let rect1 = SimdRect2::new(SimdVec2::new(0.0, 0.0), SimdVec2::new(2.0, 2.0));
            let rect2 = SimdRect2::new(SimdVec2::new(1.0, 1.0), SimdVec2::new(3.0, 3.0));
            let expected = SimdRect2::new(SimdVec2::new(0.0, 0.0), SimdVec2::new(3.0, 3.0));

            let union = rect1 | rect2;
            assert_rect_near(union, expected);

            // Test |= operator
            let mut rect3 = rect1;
            rect3 |= rect2;
            assert_rect_near(rect3, expected);
        }

        #[test]
        fn test_rect_union_with_point() {
            let rect = SimdRect2::new(SimdVec2::new(1.0, 1.0), SimdVec2::new(3.0, 3.0));

            // Point inside AABB - should not change AABB
            let point_inside = SimdVec2::new(2.0, 2.0);
            let result1 = rect | point_inside;
            assert_rect_near(result1, rect);

            // Point outside AABB - should expand AABB
            let point_outside = SimdVec2::new(0.0, 4.0);
            let expected = SimdRect2::new(SimdVec2::new(0.0, 1.0), SimdVec2::new(3.0, 4.0));
            let result2 = rect | point_outside;
            assert_rect_near(result2, expected);

            // Test |= operator
            let mut rect3 = rect;
            rect3 |= point_outside;
            assert_rect_near(rect3, expected);
        }

        #[test]
        fn test_rect_intersection_with_rect() {
            let rect1 = SimdRect2::new(SimdVec2::new(0.0, 0.0), SimdVec2::new(4.0, 4.0));
            let rect2 = SimdRect2::new(SimdVec2::new(2.0, 2.0), SimdVec2::new(6.0, 6.0));
            let expected = SimdRect2::new(SimdVec2::new(2.0, 2.0), SimdVec2::new(4.0, 4.0));

            let intersection = rect1 & rect2;
            assert_rect_near(intersection, expected);

            // Test &= operator
            let mut rect3 = rect1;
            rect3 &= rect2;
            assert_rect_near(rect3, expected);
        }

        #[test]
        fn test_rect_intersection_with_point() {
            let rect = SimdRect2::new(SimdVec2::new(1.0, 1.0), SimdVec2::new(5.0, 5.0));

            // Point inside AABB - should shrink AABB towards point
            let point = SimdVec2::new(3.0, 2.0);
            let expected = SimdRect2::new(SimdVec2::new(3.0, 2.0), SimdVec2::new(3.0, 2.0));
            let result = rect & point;
            assert_rect_near(result, expected);

            // Test &= operator
            let mut rect2 = rect;
            rect2 &= point;
            assert_rect_near(rect2, expected);
        }

        #[test]
        fn test_rect_no_intersection() {
            let rect1 = SimdRect2::new(SimdVec2::new(0.0, 0.0), SimdVec2::new(2.0, 2.0));
            let rect2 = SimdRect2::new(SimdVec2::new(3.0, 3.0), SimdVec2::new(5.0, 5.0));

            let intersection = rect1 & rect2;

            // When there's no intersection, the result should have min > max
            assert!(intersection.min.x() > intersection.max.x());
            assert!(intersection.min.y() > intersection.max.y());
        }

        #[test]
        fn test_rect_comprehensive_operations() {
            // Start with union identity
            let mut rect = SimdRect2::union_identity();

            // Add some points to build an AABB
            let points = vec![
                SimdVec2::new(1.0, 2.0),
                SimdVec2::new(-1.0, 4.0),
                SimdVec2::new(3.0, 0.0),
            ];

            for point in points.iter() {
                rect |= *point;
            }

            let expected = SimdRect2::new(SimdVec2::new(-1.0, 0.0), SimdVec2::new(3.0, 4.0));
            assert_rect_near(rect, expected);

            // Test center and extent of the built AABB
            let center = rect.center();
            let extent = rect.extent();
            let expected_center = SimdVec2::new(1.0, 2.0);
            let expected_extent = SimdVec2::new(4.0, 4.0);
            assert_vec2_near(center, expected_center);
            assert_vec2_near(extent, expected_extent);
        }
    }

    mod simd_rect3 {
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

        fn assert_rect_near(a: SimdRect3, b: SimdRect3) {
            assert_vec3_near(a.min, b.min);
            assert_vec3_near(a.max, b.max);
        }

        #[test]
        fn test_rect_new() {
            let min = SimdVec3::new(1.0, 2.0, 3.0);
            let max = SimdVec3::new(4.0, 5.0, 6.0);
            let rect = SimdRect3::new(min, max);

            assert_vec3_near(rect.min(), min);
            assert_vec3_near(rect.max(), max);
        }

        #[test]
        fn test_rect_accessors() {
            let min = SimdVec3::new(-1.0, -2.0, -3.0);
            let max = SimdVec3::new(1.0, 2.0, 3.0);
            let rect = SimdRect3::new(min, max);

            assert_vec3_near(rect.min(), min);
            assert_vec3_near(rect.max(), max);
        }

        #[test]
        fn test_rect_center() {
            let min = SimdVec3::new(0.0, 0.0, 0.0);
            let max = SimdVec3::new(4.0, 6.0, 8.0);
            let rect = SimdRect3::new(min, max);

            let center = rect.center();
            let expected = SimdVec3::new(2.0, 3.0, 4.0);
            assert_vec3_near(center, expected);
        }

        #[test]
        fn test_rect_extent() {
            let min = SimdVec3::new(1.0, 2.0, 3.0);
            let max = SimdVec3::new(5.0, 8.0, 11.0);
            let rect = SimdRect3::new(min, max);

            let extent = rect.extent();
            let expected = SimdVec3::new(4.0, 6.0, 8.0);
            assert_vec3_near(extent, expected);
        }

        #[test]
        fn test_rect_union_identity() {
            let identity = SimdRect3::union_identity();

            // Union identity should have infinite min and negative infinite max
            assert_eq!(identity.min.x(), f32::INFINITY);
            assert_eq!(identity.min.y(), f32::INFINITY);
            assert_eq!(identity.min.z(), f32::INFINITY);
            assert_eq!(identity.max.x(), f32::NEG_INFINITY);
            assert_eq!(identity.max.y(), f32::NEG_INFINITY);
            assert_eq!(identity.max.z(), f32::NEG_INFINITY);

            // Test that union with any AABB gives that AABB
            let test_rect =
                SimdRect3::new(SimdVec3::new(1.0, 2.0, 3.0), SimdVec3::new(4.0, 5.0, 6.0));
            let result = identity | test_rect;
            assert_rect_near(result, test_rect);
        }

        #[test]
        fn test_rect_intersection_identity() {
            let identity = SimdRect3::intersection_identity();

            // Intersection identity should have negative infinite min and infinite max
            assert_eq!(identity.min.x(), f32::NEG_INFINITY);
            assert_eq!(identity.min.y(), f32::NEG_INFINITY);
            assert_eq!(identity.min.z(), f32::NEG_INFINITY);
            assert_eq!(identity.max.x(), f32::INFINITY);
            assert_eq!(identity.max.y(), f32::INFINITY);
            assert_eq!(identity.max.z(), f32::INFINITY);

            // Test that intersection with any AABB gives that AABB
            let test_rect =
                SimdRect3::new(SimdVec3::new(1.0, 2.0, 3.0), SimdVec3::new(4.0, 5.0, 6.0));
            let result = identity & test_rect;
            assert_rect_near(result, test_rect);
        }

        #[test]
        fn test_rect_union_with_rect() {
            let rect1 = SimdRect3::new(SimdVec3::new(0.0, 0.0, 0.0), SimdVec3::new(2.0, 2.0, 2.0));
            let rect2 = SimdRect3::new(SimdVec3::new(1.0, 1.0, 1.0), SimdVec3::new(3.0, 3.0, 3.0));
            let expected =
                SimdRect3::new(SimdVec3::new(0.0, 0.0, 0.0), SimdVec3::new(3.0, 3.0, 3.0));

            let union = rect1 | rect2;
            assert_rect_near(union, expected);

            // Test |= operator
            let mut rect3 = rect1;
            rect3 |= rect2;
            assert_rect_near(rect3, expected);
        }

        #[test]
        fn test_rect_union_with_point() {
            let rect = SimdRect3::new(SimdVec3::new(1.0, 1.0, 1.0), SimdVec3::new(3.0, 3.0, 3.0));

            // Point inside AABB - should not change AABB
            let point_inside = SimdVec3::new(2.0, 2.0, 2.0);
            let result1 = rect | point_inside;
            assert_rect_near(result1, rect);

            // Point outside AABB - should expand AABB
            let point_outside = SimdVec3::new(0.0, 4.0, 2.0);
            let expected =
                SimdRect3::new(SimdVec3::new(0.0, 1.0, 1.0), SimdVec3::new(3.0, 4.0, 3.0));
            let result2 = rect | point_outside;
            assert_rect_near(result2, expected);

            // Test |= operator
            let mut rect3 = rect;
            rect3 |= point_outside;
            assert_rect_near(rect3, expected);
        }

        #[test]
        fn test_rect_intersection_with_rect() {
            let rect1 = SimdRect3::new(SimdVec3::new(0.0, 0.0, 0.0), SimdVec3::new(4.0, 4.0, 4.0));
            let rect2 = SimdRect3::new(SimdVec3::new(2.0, 2.0, 2.0), SimdVec3::new(6.0, 6.0, 6.0));
            let expected =
                SimdRect3::new(SimdVec3::new(2.0, 2.0, 2.0), SimdVec3::new(4.0, 4.0, 4.0));

            let intersection = rect1 & rect2;
            assert_rect_near(intersection, expected);

            // Test &= operator
            let mut rect3 = rect1;
            rect3 &= rect2;
            assert_rect_near(rect3, expected);
        }

        #[test]
        fn test_rect_intersection_with_point() {
            let rect = SimdRect3::new(SimdVec3::new(1.0, 1.0, 1.0), SimdVec3::new(5.0, 5.0, 5.0));

            // Point inside AABB - should shrink AABB towards point
            let point = SimdVec3::new(3.0, 2.0, 4.0);
            let expected =
                SimdRect3::new(SimdVec3::new(3.0, 2.0, 4.0), SimdVec3::new(3.0, 2.0, 4.0));
            let result = rect & point;
            assert_rect_near(result, expected);

            // Test &= operator
            let mut rect2 = rect;
            rect2 &= point;
            assert_rect_near(rect2, expected);
        }

        #[test]
        fn test_rect_no_intersection() {
            let rect1 = SimdRect3::new(SimdVec3::new(0.0, 0.0, 0.0), SimdVec3::new(2.0, 2.0, 2.0));
            let rect2 = SimdRect3::new(SimdVec3::new(3.0, 3.0, 3.0), SimdVec3::new(5.0, 5.0, 5.0));

            let intersection = rect1 & rect2;

            // When there's no intersection, the result should have min > max
            assert!(intersection.min.x() > intersection.max.x());
            assert!(intersection.min.y() > intersection.max.y());
            assert!(intersection.min.z() > intersection.max.z());
        }

        #[test]
        fn test_rect_comprehensive_operations() {
            // Start with union identity
            let mut rect = SimdRect3::union_identity();

            // Add some points to build an AABB
            let points = vec![
                SimdVec3::new(1.0, 2.0, 3.0),
                SimdVec3::new(-1.0, 4.0, 1.0),
                SimdVec3::new(3.0, 0.0, 5.0),
            ];

            for point in points.iter() {
                rect |= *point;
            }

            let expected =
                SimdRect3::new(SimdVec3::new(-1.0, 0.0, 1.0), SimdVec3::new(3.0, 4.0, 5.0));
            assert_rect_near(rect, expected);

            // Test center and extent of the built AABB
            let center = rect.center();
            let extent = rect.extent();
            let expected_center = SimdVec3::new(1.0, 2.0, 3.0);
            let expected_extent = SimdVec3::new(4.0, 4.0, 4.0);
            assert_vec3_near(center, expected_center);
            assert_vec3_near(extent, expected_extent);
        }
    }
}
