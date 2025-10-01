use super::*;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign};

macro_rules! impl_simd_rect {
    ($name:ident, $vec_type:ty, $num_elements:expr, $lo:expr, $hi:expr) => {
        #[derive(Clone, Copy, Debug)]
        pub struct $name {
            pub min: $vec_type,
            pub max: $vec_type,
        }

        impl $name {
            /// Creates a new rect with the given min and max bounds.
            #[inline]
            pub const fn new(min: $vec_type, max: $vec_type) -> Self {
                $name { min, max }
            }

            /// Returns the minimum bounds of the rect.
            #[inline]
            pub fn min(&self) -> $vec_type {
                self.min
            }

            /// Returns the maximum bounds of the rect.
            #[inline]
            pub fn max(&self) -> $vec_type {
                self.max
            }

            /// Returns the center point of the rect.
            #[inline]
            pub fn center(&self) -> $vec_type {
                type LaneType = <$vec_type as SimdVector>::LaneType;
                (self.min + self.max) / <$vec_type>::splat(2 as LaneType)
            }

            /// Returns the extent (size) of the rect.
            #[inline]
            pub fn extent(&self) -> $vec_type {
                self.max - self.min
            }

            /// Returns the identity element for union operations.
            /// This rect will not affect the result when unioned with any other rects.
            pub fn union_identity() -> Self {
                $name {
                    min: <$vec_type>::splat($hi),
                    max: <$vec_type>::splat($lo),
                }
            }

            /// Returns the identity element for intersection operations.
            /// This rect will not affect the result when intersected with any other rects.
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

#[cfg(test)]
mod tests {
    use super::*;

    // Macro for generating comprehensive test suites for 2D rect types
    macro_rules! test_simd_rect2 {
        (
            $test_mod:ident,
            $rect_type:ty,
            $vec_type:ty,
            $lane_type:ty,
            $lo_val:expr,
            $hi_val:expr,
            $test_vals:expr,
            $assert_fn:ident
        ) => {
            mod $test_mod {
                use super::*;

                const EPSILON: $lane_type = if std::mem::size_of::<$lane_type>() == 4 {
                    1e-6 as $lane_type
                } else {
                    0 as $lane_type
                };

                fn $assert_fn(a: $lane_type, b: $lane_type) {
                    // Check if we're dealing with f32 specifically
                    if std::any::TypeId::of::<$lane_type>() == std::any::TypeId::of::<f32>() {
                        // Special handling for infinity values
                        if (a == $hi_val && b == $hi_val) || (a == $lo_val && b == $lo_val) {
                            // Both are the same infinity, they're equal
                            return;
                        }
                        // Floating point comparison
                        let diff = if a > b { a - b } else { b - a };
                        assert!(
                            diff < EPSILON,
                            "Expected {}, got {} (difference: {})",
                            b,
                            a,
                            diff
                        );
                    } else {
                        // Integer comparison
                        assert_eq!(a, b, "Expected {}, got {}", b, a);
                    }
                }

                fn assert_vec2_eq(a: $vec_type, b: $vec_type) {
                    $assert_fn(a.x(), b.x());
                    $assert_fn(a.y(), b.y());
                }

                fn assert_rect_eq(a: $rect_type, b: $rect_type) {
                    assert_vec2_eq(a.min, b.min);
                    assert_vec2_eq(a.max, b.max);
                }

                #[test]
                fn test_rect_new() {
                    let vals = $test_vals;
                    let min = <$vec_type>::new(vals[0], vals[1]);
                    let max = <$vec_type>::new(vals[2], vals[3]);
                    let rect = <$rect_type>::new(min, max);

                    assert_vec2_eq(rect.min(), min);
                    assert_vec2_eq(rect.max(), max);
                }

                #[test]
                fn test_rect_accessors() {
                    let vals = $test_vals;
                    let min = <$vec_type>::new(vals[0], vals[1]);
                    let max = <$vec_type>::new(vals[2], vals[3]);
                    let rect = <$rect_type>::new(min, max);

                    assert_vec2_eq(rect.min(), min);
                    assert_vec2_eq(rect.max(), max);
                }

                #[test]
                fn test_rect_center() {
                    let vals = $test_vals;
                    let min = <$vec_type>::new(vals[0], vals[1]);
                    let max = <$vec_type>::new(vals[4], vals[5]);
                    let rect = <$rect_type>::new(min, max);

                    let center = rect.center();
                    let expected = <$vec_type>::new(
                        (vals[0] + vals[4]) / (2 as $lane_type),
                        (vals[1] + vals[5]) / (2 as $lane_type),
                    );
                    assert_vec2_eq(center, expected);
                }

                #[test]
                fn test_rect_extent() {
                    let vals = $test_vals;
                    let min = <$vec_type>::new(vals[0], vals[1]);
                    let max = <$vec_type>::new(vals[4], vals[5]);
                    let rect = <$rect_type>::new(min, max);

                    let extent = rect.extent();
                    let expected = <$vec_type>::new(vals[4] - vals[0], vals[5] - vals[1]);
                    assert_vec2_eq(extent, expected);
                }

                #[test]
                fn test_rect_union_identity() {
                    let identity = <$rect_type>::union_identity();

                    // Union identity should have hi min and lo max
                    $assert_fn(identity.min.x(), $hi_val);
                    $assert_fn(identity.min.y(), $hi_val);
                    $assert_fn(identity.max.x(), $lo_val);
                    $assert_fn(identity.max.y(), $lo_val);

                    // Test that union with any rect gives that rect
                    let vals = $test_vals;
                    let test_rect = <$rect_type>::new(
                        <$vec_type>::new(vals[0], vals[1]),
                        <$vec_type>::new(vals[2], vals[3]),
                    );
                    let result = identity | test_rect;
                    assert_rect_eq(result, test_rect);
                }

                #[test]
                fn test_rect_intersection_identity() {
                    let identity = <$rect_type>::intersection_identity();

                    // Intersection identity should have lo min and hi max
                    $assert_fn(identity.min.x(), $lo_val);
                    $assert_fn(identity.min.y(), $lo_val);
                    $assert_fn(identity.max.x(), $hi_val);
                    $assert_fn(identity.max.y(), $hi_val);

                    // Test that intersection with any rect gives that rect
                    let vals = $test_vals;
                    let test_rect = <$rect_type>::new(
                        <$vec_type>::new(vals[0], vals[1]),
                        <$vec_type>::new(vals[2], vals[3]),
                    );
                    let result = identity & test_rect;
                    assert_rect_eq(result, test_rect);
                }

                #[test]
                fn test_rect_union_with_rect() {
                    let vals = $test_vals;
                    let rect1 = <$rect_type>::new(
                        <$vec_type>::new(vals[0], vals[1]),
                        <$vec_type>::new(vals[2], vals[3]),
                    );
                    let rect2 = <$rect_type>::new(
                        <$vec_type>::new(vals[6], vals[7]),
                        <$vec_type>::new(vals[8], vals[9]),
                    );

                    let min_x = if vals[0] < vals[6] { vals[0] } else { vals[6] };
                    let min_y = if vals[1] < vals[7] { vals[1] } else { vals[7] };
                    let max_x = if vals[2] > vals[8] { vals[2] } else { vals[8] };
                    let max_y = if vals[3] > vals[9] { vals[3] } else { vals[9] };

                    let expected = <$rect_type>::new(
                        <$vec_type>::new(min_x, min_y),
                        <$vec_type>::new(max_x, max_y),
                    );

                    let union = rect1 | rect2;
                    assert_rect_eq(union, expected);

                    // Test |= operator
                    let mut rect3 = rect1;
                    rect3 |= rect2;
                    assert_rect_eq(rect3, expected);
                }

                #[test]
                fn test_rect_union_with_point() {
                    let vals = $test_vals;
                    let rect = <$rect_type>::new(
                        <$vec_type>::new(vals[0], vals[1]),
                        <$vec_type>::new(vals[2], vals[3]),
                    );

                    // Point inside rect - should not change rect for most cases
                    let point_inside = <$vec_type>::new(vals[10], vals[11]);
                    let result1 = rect | point_inside;

                    // For union with point, we expand bounds if necessary
                    let min_x = if vals[0] < vals[10] {
                        vals[0]
                    } else {
                        vals[10]
                    };
                    let min_y = if vals[1] < vals[11] {
                        vals[1]
                    } else {
                        vals[11]
                    };
                    let max_x = if vals[2] > vals[10] {
                        vals[2]
                    } else {
                        vals[10]
                    };
                    let max_y = if vals[3] > vals[11] {
                        vals[3]
                    } else {
                        vals[11]
                    };
                    let expected1 = <$rect_type>::new(
                        <$vec_type>::new(min_x, min_y),
                        <$vec_type>::new(max_x, max_y),
                    );
                    assert_rect_eq(result1, expected1);

                    // Test |= operator
                    let mut rect2 = rect;
                    rect2 |= point_inside;
                    assert_rect_eq(rect2, expected1);
                }

                #[test]
                fn test_rect_intersection_with_rect() {
                    let vals = $test_vals;
                    let rect1 = <$rect_type>::new(
                        <$vec_type>::new(vals[0], vals[1]),
                        <$vec_type>::new(vals[4], vals[5]),
                    );
                    let rect2 = <$rect_type>::new(
                        <$vec_type>::new(vals[6], vals[7]),
                        <$vec_type>::new(vals[8], vals[9]),
                    );

                    let min_x = if vals[0] > vals[6] { vals[0] } else { vals[6] };
                    let min_y = if vals[1] > vals[7] { vals[1] } else { vals[7] };
                    let max_x = if vals[4] < vals[8] { vals[4] } else { vals[8] };
                    let max_y = if vals[5] < vals[9] { vals[5] } else { vals[9] };

                    let expected = <$rect_type>::new(
                        <$vec_type>::new(min_x, min_y),
                        <$vec_type>::new(max_x, max_y),
                    );

                    let intersection = rect1 & rect2;
                    assert_rect_eq(intersection, expected);

                    // Test &= operator
                    let mut rect3 = rect1;
                    rect3 &= rect2;
                    assert_rect_eq(rect3, expected);
                }

                #[test]
                fn test_rect_intersection_with_point() {
                    let vals = $test_vals;
                    let rect = <$rect_type>::new(
                        <$vec_type>::new(vals[0], vals[1]),
                        <$vec_type>::new(vals[4], vals[5]),
                    );

                    // Point - should create degenerate rect at point
                    let point = <$vec_type>::new(vals[10], vals[11]);
                    let expected = <$rect_type>::new(
                        <$vec_type>::new(vals[10], vals[11]),
                        <$vec_type>::new(vals[10], vals[11]),
                    );
                    let result = rect & point;
                    assert_rect_eq(result, expected);

                    // Test &= operator
                    let mut rect2 = rect;
                    rect2 &= point;
                    assert_rect_eq(rect2, expected);
                }

                #[test]
                fn test_rect_comprehensive_operations() {
                    // Start with union identity
                    let mut rect = <$rect_type>::union_identity();

                    // Add some points to build an AABB
                    let vals = $test_vals;
                    let points = vec![
                        <$vec_type>::new(vals[0], vals[1]),
                        <$vec_type>::new(vals[6], vals[7]),
                        <$vec_type>::new(vals[12], vals[13]),
                    ];

                    for point in points.iter() {
                        rect |= *point;
                    }

                    // Compute expected bounds
                    let min_x = [vals[0], vals[6], vals[12]]
                        .iter()
                        .cloned()
                        .fold(vals[0], |a, b| if a < b { a } else { b });
                    let min_y = [vals[1], vals[7], vals[13]]
                        .iter()
                        .cloned()
                        .fold(vals[1], |a, b| if a < b { a } else { b });
                    let max_x = [vals[0], vals[6], vals[12]]
                        .iter()
                        .cloned()
                        .fold(vals[0], |a, b| if a > b { a } else { b });
                    let max_y = [vals[1], vals[7], vals[13]]
                        .iter()
                        .cloned()
                        .fold(vals[1], |a, b| if a > b { a } else { b });

                    let expected = <$rect_type>::new(
                        <$vec_type>::new(min_x, min_y),
                        <$vec_type>::new(max_x, max_y),
                    );
                    assert_rect_eq(rect, expected);

                    // Test center and extent of the built AABB
                    let center = rect.center();
                    let extent = rect.extent();
                    let expected_center = <$vec_type>::new(
                        (min_x + max_x) / (2 as $lane_type),
                        (min_y + max_y) / (2 as $lane_type),
                    );
                    let expected_extent = <$vec_type>::new(max_x - min_x, max_y - min_y);
                    assert_vec2_eq(center, expected_center);
                    assert_vec2_eq(extent, expected_extent);
                }
            }
        };
    }

    // Macro for generating comprehensive test suites for 3D rect types
    macro_rules! test_simd_rect3 {
        (
            $test_mod:ident,
            $rect_type:ty,
            $vec_type:ty,
            $lane_type:ty,
            $lo_val:expr,
            $hi_val:expr,
            $test_vals:expr,
            $assert_fn:ident
        ) => {
            mod $test_mod {
                use super::*;

                const EPSILON: $lane_type = if std::mem::size_of::<$lane_type>() == 4 {
                    1e-6 as $lane_type
                } else {
                    0 as $lane_type
                };

                fn $assert_fn(a: $lane_type, b: $lane_type) {
                    // Check if we're dealing with f32 specifically
                    if std::any::TypeId::of::<$lane_type>() == std::any::TypeId::of::<f32>() {
                        // Special handling for infinity values
                        if (a == $hi_val && b == $hi_val) || (a == $lo_val && b == $lo_val) {
                            // Both are the same infinity, they're equal
                            return;
                        }
                        // Floating point comparison
                        let diff = if a > b { a - b } else { b - a };
                        assert!(
                            diff < EPSILON,
                            "Expected {}, got {} (difference: {})",
                            b,
                            a,
                            diff
                        );
                    } else {
                        // Integer comparison
                        assert_eq!(a, b, "Expected {}, got {}", b, a);
                    }
                }

                fn assert_vec3_eq(a: $vec_type, b: $vec_type) {
                    $assert_fn(a.x(), b.x());
                    $assert_fn(a.y(), b.y());
                    $assert_fn(a.z(), b.z());
                }

                fn assert_rect_eq(a: $rect_type, b: $rect_type) {
                    assert_vec3_eq(a.min, b.min);
                    assert_vec3_eq(a.max, b.max);
                }

                #[test]
                fn test_rect_new() {
                    let vals = $test_vals;
                    let min = <$vec_type>::new(vals[0], vals[1], vals[2]);
                    let max = <$vec_type>::new(vals[3], vals[4], vals[5]);
                    let rect = <$rect_type>::new(min, max);

                    assert_vec3_eq(rect.min(), min);
                    assert_vec3_eq(rect.max(), max);
                }

                #[test]
                fn test_rect_accessors() {
                    let vals = $test_vals;
                    let min = <$vec_type>::new(vals[0], vals[1], vals[2]);
                    let max = <$vec_type>::new(vals[3], vals[4], vals[5]);
                    let rect = <$rect_type>::new(min, max);

                    assert_vec3_eq(rect.min(), min);
                    assert_vec3_eq(rect.max(), max);
                }

                #[test]
                fn test_rect_center() {
                    let vals = $test_vals;
                    let min = <$vec_type>::new(vals[0], vals[1], vals[2]);
                    let max = <$vec_type>::new(vals[6], vals[7], vals[8]);
                    let rect = <$rect_type>::new(min, max);

                    let center = rect.center();
                    let expected = <$vec_type>::new(
                        (vals[0] + vals[6]) / (2 as $lane_type),
                        (vals[1] + vals[7]) / (2 as $lane_type),
                        (vals[2] + vals[8]) / (2 as $lane_type),
                    );
                    assert_vec3_eq(center, expected);
                }

                #[test]
                fn test_rect_extent() {
                    let vals = $test_vals;
                    let min = <$vec_type>::new(vals[0], vals[1], vals[2]);
                    let max = <$vec_type>::new(vals[6], vals[7], vals[8]);
                    let rect = <$rect_type>::new(min, max);

                    let extent = rect.extent();
                    let expected =
                        <$vec_type>::new(vals[6] - vals[0], vals[7] - vals[1], vals[8] - vals[2]);
                    assert_vec3_eq(extent, expected);
                }

                #[test]
                fn test_rect_union_identity() {
                    let identity = <$rect_type>::union_identity();

                    // Union identity should have hi min and lo max
                    $assert_fn(identity.min.x(), $hi_val);
                    $assert_fn(identity.min.y(), $hi_val);
                    $assert_fn(identity.min.z(), $hi_val);
                    $assert_fn(identity.max.x(), $lo_val);
                    $assert_fn(identity.max.y(), $lo_val);
                    $assert_fn(identity.max.z(), $lo_val);

                    // Test that union with any AABB gives that AABB
                    let vals = $test_vals;
                    let test_rect = <$rect_type>::new(
                        <$vec_type>::new(vals[0], vals[1], vals[2]),
                        <$vec_type>::new(vals[3], vals[4], vals[5]),
                    );
                    let result = identity | test_rect;
                    assert_rect_eq(result, test_rect);
                }

                #[test]
                fn test_rect_intersection_identity() {
                    let identity = <$rect_type>::intersection_identity();

                    // Intersection identity should have lo min and hi max
                    $assert_fn(identity.min.x(), $lo_val);
                    $assert_fn(identity.min.y(), $lo_val);
                    $assert_fn(identity.min.z(), $lo_val);
                    $assert_fn(identity.max.x(), $hi_val);
                    $assert_fn(identity.max.y(), $hi_val);
                    $assert_fn(identity.max.z(), $hi_val);

                    // Test that intersection with any AABB gives that AABB
                    let vals = $test_vals;
                    let test_rect = <$rect_type>::new(
                        <$vec_type>::new(vals[0], vals[1], vals[2]),
                        <$vec_type>::new(vals[3], vals[4], vals[5]),
                    );
                    let result = identity & test_rect;
                    assert_rect_eq(result, test_rect);
                }

                #[test]
                fn test_rect_union_with_rect() {
                    let vals = $test_vals;
                    let rect1 = <$rect_type>::new(
                        <$vec_type>::new(vals[0], vals[1], vals[2]),
                        <$vec_type>::new(vals[3], vals[4], vals[5]),
                    );
                    let rect2 = <$rect_type>::new(
                        <$vec_type>::new(vals[9], vals[10], vals[11]),
                        <$vec_type>::new(vals[12], vals[13], vals[14]),
                    );

                    let min_x = if vals[0] < vals[9] { vals[0] } else { vals[9] };
                    let min_y = if vals[1] < vals[10] {
                        vals[1]
                    } else {
                        vals[10]
                    };
                    let min_z = if vals[2] < vals[11] {
                        vals[2]
                    } else {
                        vals[11]
                    };
                    let max_x = if vals[3] > vals[12] {
                        vals[3]
                    } else {
                        vals[12]
                    };
                    let max_y = if vals[4] > vals[13] {
                        vals[4]
                    } else {
                        vals[13]
                    };
                    let max_z = if vals[5] > vals[14] {
                        vals[5]
                    } else {
                        vals[14]
                    };

                    let expected = <$rect_type>::new(
                        <$vec_type>::new(min_x, min_y, min_z),
                        <$vec_type>::new(max_x, max_y, max_z),
                    );

                    let union = rect1 | rect2;
                    assert_rect_eq(union, expected);

                    // Test |= operator
                    let mut rect3 = rect1;
                    rect3 |= rect2;
                    assert_rect_eq(rect3, expected);
                }

                #[test]
                fn test_rect_union_with_point() {
                    let vals = $test_vals;
                    let rect = <$rect_type>::new(
                        <$vec_type>::new(vals[0], vals[1], vals[2]),
                        <$vec_type>::new(vals[3], vals[4], vals[5]),
                    );

                    // Point - should expand bounds if necessary
                    let point = <$vec_type>::new(vals[15], vals[16], vals[17]);

                    let min_x = if vals[0] < vals[15] {
                        vals[0]
                    } else {
                        vals[15]
                    };
                    let min_y = if vals[1] < vals[16] {
                        vals[1]
                    } else {
                        vals[16]
                    };
                    let min_z = if vals[2] < vals[17] {
                        vals[2]
                    } else {
                        vals[17]
                    };
                    let max_x = if vals[3] > vals[15] {
                        vals[3]
                    } else {
                        vals[15]
                    };
                    let max_y = if vals[4] > vals[16] {
                        vals[4]
                    } else {
                        vals[16]
                    };
                    let max_z = if vals[5] > vals[17] {
                        vals[5]
                    } else {
                        vals[17]
                    };

                    let expected = <$rect_type>::new(
                        <$vec_type>::new(min_x, min_y, min_z),
                        <$vec_type>::new(max_x, max_y, max_z),
                    );
                    let result = rect | point;
                    assert_rect_eq(result, expected);

                    // Test |= operator
                    let mut rect2 = rect;
                    rect2 |= point;
                    assert_rect_eq(rect2, expected);
                }

                #[test]
                fn test_rect_intersection_with_rect() {
                    let vals = $test_vals;
                    let rect1 = <$rect_type>::new(
                        <$vec_type>::new(vals[0], vals[1], vals[2]),
                        <$vec_type>::new(vals[6], vals[7], vals[8]),
                    );
                    let rect2 = <$rect_type>::new(
                        <$vec_type>::new(vals[9], vals[10], vals[11]),
                        <$vec_type>::new(vals[12], vals[13], vals[14]),
                    );

                    let min_x = if vals[0] > vals[9] { vals[0] } else { vals[9] };
                    let min_y = if vals[1] > vals[10] {
                        vals[1]
                    } else {
                        vals[10]
                    };
                    let min_z = if vals[2] > vals[11] {
                        vals[2]
                    } else {
                        vals[11]
                    };
                    let max_x = if vals[6] < vals[12] {
                        vals[6]
                    } else {
                        vals[12]
                    };
                    let max_y = if vals[7] < vals[13] {
                        vals[7]
                    } else {
                        vals[13]
                    };
                    let max_z = if vals[8] < vals[14] {
                        vals[8]
                    } else {
                        vals[14]
                    };

                    let expected = <$rect_type>::new(
                        <$vec_type>::new(min_x, min_y, min_z),
                        <$vec_type>::new(max_x, max_y, max_z),
                    );

                    let intersection = rect1 & rect2;
                    assert_rect_eq(intersection, expected);

                    // Test &= operator
                    let mut rect3 = rect1;
                    rect3 &= rect2;
                    assert_rect_eq(rect3, expected);
                }

                #[test]
                fn test_rect_intersection_with_point() {
                    let vals = $test_vals;
                    let rect = <$rect_type>::new(
                        <$vec_type>::new(vals[0], vals[1], vals[2]),
                        <$vec_type>::new(vals[6], vals[7], vals[8]),
                    );

                    // Point - should create degenerate rect at point
                    let point = <$vec_type>::new(vals[15], vals[16], vals[17]);
                    let expected = <$rect_type>::new(
                        <$vec_type>::new(vals[15], vals[16], vals[17]),
                        <$vec_type>::new(vals[15], vals[16], vals[17]),
                    );
                    let result = rect & point;
                    assert_rect_eq(result, expected);

                    // Test &= operator
                    let mut rect2 = rect;
                    rect2 &= point;
                    assert_rect_eq(rect2, expected);
                }

                #[test]
                fn test_rect_comprehensive_operations() {
                    // Start with union identity
                    let mut rect = <$rect_type>::union_identity();

                    // Add some points to build an AABB
                    let vals = $test_vals;
                    let points = vec![
                        <$vec_type>::new(vals[0], vals[1], vals[2]),
                        <$vec_type>::new(vals[9], vals[10], vals[11]),
                        <$vec_type>::new(vals[18], vals[19], vals[20]),
                    ];

                    for point in points.iter() {
                        rect |= *point;
                    }

                    // Compute expected bounds
                    let min_x = [vals[0], vals[9], vals[18]]
                        .iter()
                        .cloned()
                        .fold(vals[0], |a, b| if a < b { a } else { b });
                    let min_y = [vals[1], vals[10], vals[19]]
                        .iter()
                        .cloned()
                        .fold(vals[1], |a, b| if a < b { a } else { b });
                    let min_z = [vals[2], vals[11], vals[20]]
                        .iter()
                        .cloned()
                        .fold(vals[2], |a, b| if a < b { a } else { b });
                    let max_x = [vals[0], vals[9], vals[18]]
                        .iter()
                        .cloned()
                        .fold(vals[0], |a, b| if a > b { a } else { b });
                    let max_y = [vals[1], vals[10], vals[19]]
                        .iter()
                        .cloned()
                        .fold(vals[1], |a, b| if a > b { a } else { b });
                    let max_z = [vals[2], vals[11], vals[20]]
                        .iter()
                        .cloned()
                        .fold(vals[2], |a, b| if a > b { a } else { b });

                    let expected = <$rect_type>::new(
                        <$vec_type>::new(min_x, min_y, min_z),
                        <$vec_type>::new(max_x, max_y, max_z),
                    );
                    assert_rect_eq(rect, expected);

                    // Test center and extent of the built AABB
                    let center = rect.center();
                    let extent = rect.extent();
                    let expected_center = <$vec_type>::new(
                        (min_x + max_x) / (2 as $lane_type),
                        (min_y + max_y) / (2 as $lane_type),
                        (min_z + max_z) / (2 as $lane_type),
                    );
                    let expected_extent =
                        <$vec_type>::new(max_x - min_x, max_y - min_y, max_z - min_z);
                    assert_vec3_eq(center, expected_center);
                    assert_vec3_eq(extent, expected_extent);
                }
            }
        };
    }

    // Generate tests for all 2D rect types
    test_simd_rect2!(
        simd_rect2,
        SimdRect2,
        SimdVec2,
        f32,
        f32::NEG_INFINITY,
        f32::INFINITY,
        [
            1.0, 2.0, 4.0, 5.0, 8.0, 10.0, 0.0, 0.0, 3.0, 3.0, 2.5, 3.0, -1.0, 4.0
        ],
        assert_f32_near
    );

    test_simd_rect2!(
        simd_urect2,
        SimdURect2,
        SimdUVec2,
        u32,
        u32::MIN,
        u32::MAX,
        [1, 2, 4, 5, 8, 10, 0, 0, 3, 3, 2, 3, 5, 7],
        assert_u32_eq
    );

    test_simd_rect2!(
        simd_irect2,
        SimdIRect2,
        SimdIVec2,
        i32,
        i32::MIN,
        i32::MAX,
        [1, 2, 4, 5, 8, 10, 0, 0, 3, 3, 2, 3, -1, 4],
        assert_i32_eq
    );

    // Generate tests for all 3D rect types
    test_simd_rect3!(
        simd_rect3,
        SimdRect3,
        SimdVec3,
        f32,
        f32::NEG_INFINITY,
        f32::INFINITY,
        [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 2.5, 3.0,
            4.0, -1.0, 4.0, 0.0
        ],
        assert_f32_near
    );

    test_simd_rect3!(
        simd_urect3,
        SimdURect3,
        SimdUVec3,
        u32,
        u32::MIN,
        u32::MAX,
        [
            1, 2, 3, 4, 5, 6, 8, 10, 12, 0, 0, 0, 3, 3, 3, 2, 3, 4, 5, 7, 9
        ],
        assert_u32_eq
    );

    test_simd_rect3!(
        simd_irect3,
        SimdIRect3,
        SimdIVec3,
        i32,
        i32::MIN,
        i32::MAX,
        [
            1, 2, 3, 4, 5, 6, 8, 10, 12, 0, 0, 0, 3, 3, 3, 2, 3, 4, -1, 4, 0
        ],
        assert_i32_eq
    );
}
