use super::*;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign};

macro_rules! impl_simd_rect {
    ($name:ident, $vec_type:ty, $lane_ty:ty, $dim:expr, $lo:expr, $hi:expr) => {
        #[doc = concat!(
                    "An axis-aligned bounding box with [`", stringify!($vec_type), "`] corners."
                )]
        ///
        /// `min` and `max` are inclusive corner bounds. Union is the `|`/`|=`
        /// operator and intersection is `&`/`&=`; both also accept a vector on
        /// the right-hand side, treated as a degenerate point-rect.
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct $name {
            /// The minimum (inclusive) corner of the box.
            pub min: $vec_type,
            /// The maximum (inclusive) corner of the box.
            pub max: $vec_type,
        }

        impl $name {
            /// Creates a new rect with the given min and max bounds.
            #[inline]
            pub const fn new(min: $vec_type, max: $vec_type) -> Self {
                $name { min, max }
            }

            /// Returns the center point of the rect.
            ///
            /// Assumes a well-formed rect (`min <= max` in every component).
            /// Computed as `min + extent / 2` so that it cannot overflow for
            /// well-formed integer rects.
            #[inline]
            #[must_use]
            pub fn center(&self) -> $vec_type {
                self.min + self.extent() / (2 as $lane_ty)
            }

            /// Returns the extent (size) of the rect along each axis.
            ///
            /// For unsigned rects this wraps if the rect is not well-formed
            /// (`min > max` in some component).
            #[inline]
            #[must_use]
            pub fn extent(&self) -> $vec_type {
                self.max - self.min
            }

            /// Returns `true` if `point` lies within the box (bounds inclusive).
            #[inline]
            #[must_use]
            pub fn contains(&self, point: $vec_type) -> bool {
                point.elementwise_max(self.min) == point && point.elementwise_min(self.max) == point
            }

            /// Returns `true` if the two boxes overlap (touching edges count,
            /// since bounds are inclusive).
            #[inline]
            #[must_use]
            pub fn intersects(&self, other: &Self) -> bool {
                let lo = self.min.elementwise_max(other.min);
                let hi = self.max.elementwise_min(other.max);
                lo.elementwise_min(hi) == lo
            }

            /// Returns `true` if the box contains no points, i.e. some
            /// component of `min` exceeds the corresponding component of `max`.
            #[inline]
            #[must_use]
            pub fn is_empty(&self) -> bool {
                self.min.elementwise_min(self.max) != self.min
            }

            /// Returns the identity element for union operations.
            /// This rect will not affect the result when unioned with any other rects.
            #[must_use]
            pub fn union_identity() -> Self {
                $name {
                    min: <$vec_type>::splat($hi),
                    max: <$vec_type>::splat($lo),
                }
            }

            /// Returns the identity element for intersection operations.
            /// This rect will not affect the result when intersected with any other rects.
            #[must_use]
            pub fn intersection_identity() -> Self {
                $name {
                    min: <$vec_type>::splat($lo),
                    max: <$vec_type>::splat($hi),
                }
            }
        }

        // Flatten a rect into a `[min_array, max_array]` pair, and back.
        impl From<$name> for [[$lane_ty; $dim]; 2] {
            #[inline]
            fn from(rect: $name) -> Self {
                [rect.min.into(), rect.max.into()]
            }
        }
        impl From<[[$lane_ty; $dim]; 2]> for $name {
            #[inline]
            fn from([min, max]: [[$lane_ty; $dim]; 2]) -> Self {
                $name {
                    min: min.into(),
                    max: max.into(),
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

impl_simd_rect!(
    SimdRect2,
    SimdVec2,
    f32,
    2,
    f32::NEG_INFINITY,
    f32::INFINITY
);
impl_simd_rect!(SimdURect2, SimdUVec2, u32, 2, u32::MIN, u32::MAX);
impl_simd_rect!(SimdIRect2, SimdIVec2, i32, 2, i32::MIN, i32::MAX);

impl_simd_rect!(
    SimdRect3,
    SimdVec3,
    f32,
    3,
    f32::NEG_INFINITY,
    f32::INFINITY
);
impl_simd_rect!(SimdURect3, SimdUVec3, u32, 3, u32::MIN, u32::MAX);
impl_simd_rect!(SimdIRect3, SimdIVec3, i32, 3, i32::MIN, i32::MAX);

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
                    $assert_fn(a[0], b[0]);
                    $assert_fn(a[1], b[1]);
                }

                fn assert_rect_eq(a: $rect_type, b: $rect_type) {
                    assert_vec2_eq(a.min, b.min);
                    assert_vec2_eq(a.max, b.max);
                }

                #[test]
                fn test_rect_new() {
                    let vals = $test_vals;
                    let min = <$vec_type>::from([vals[0], vals[1]]);
                    let max = <$vec_type>::from([vals[2], vals[3]]);
                    let rect = <$rect_type>::new(min, max);

                    assert_vec2_eq(rect.min, min);
                    assert_vec2_eq(rect.max, max);
                }

                #[test]
                fn test_rect_accessors() {
                    let vals = $test_vals;
                    let min = <$vec_type>::from([vals[0], vals[1]]);
                    let max = <$vec_type>::from([vals[2], vals[3]]);
                    let rect = <$rect_type>::new(min, max);

                    assert_vec2_eq(rect.min, min);
                    assert_vec2_eq(rect.max, max);
                }

                #[test]
                fn test_rect_center() {
                    let vals = $test_vals;
                    let min = <$vec_type>::from([vals[0], vals[1]]);
                    let max = <$vec_type>::from([vals[4], vals[5]]);
                    let rect = <$rect_type>::new(min, max);

                    let center = rect.center();
                    let expected = <$vec_type>::from([
                        (vals[0] + vals[4]) / (2 as $lane_type),
                        (vals[1] + vals[5]) / (2 as $lane_type),
                    ]);
                    assert_vec2_eq(center, expected);
                }

                #[test]
                fn test_rect_extent() {
                    let vals = $test_vals;
                    let min = <$vec_type>::from([vals[0], vals[1]]);
                    let max = <$vec_type>::from([vals[4], vals[5]]);
                    let rect = <$rect_type>::new(min, max);

                    let extent = rect.extent();
                    let expected = <$vec_type>::from([vals[4] - vals[0], vals[5] - vals[1]]);
                    assert_vec2_eq(extent, expected);
                }

                #[test]
                fn test_rect_union_identity() {
                    let identity = <$rect_type>::union_identity();

                    // Union identity should have hi min and lo max
                    $assert_fn(identity.min[0], $hi_val);
                    $assert_fn(identity.min[1], $hi_val);
                    $assert_fn(identity.max[0], $lo_val);
                    $assert_fn(identity.max[1], $lo_val);

                    // Test that union with any rect gives that rect
                    let vals = $test_vals;
                    let test_rect = <$rect_type>::new(
                        <$vec_type>::from([vals[0], vals[1]]),
                        <$vec_type>::from([vals[2], vals[3]]),
                    );
                    let result = identity | test_rect;
                    assert_rect_eq(result, test_rect);
                }

                #[test]
                fn test_rect_intersection_identity() {
                    let identity = <$rect_type>::intersection_identity();

                    // Intersection identity should have lo min and hi max
                    $assert_fn(identity.min[0], $lo_val);
                    $assert_fn(identity.min[1], $lo_val);
                    $assert_fn(identity.max[0], $hi_val);
                    $assert_fn(identity.max[1], $hi_val);

                    // Test that intersection with any rect gives that rect
                    let vals = $test_vals;
                    let test_rect = <$rect_type>::new(
                        <$vec_type>::from([vals[0], vals[1]]),
                        <$vec_type>::from([vals[2], vals[3]]),
                    );
                    let result = identity & test_rect;
                    assert_rect_eq(result, test_rect);
                }

                #[test]
                fn test_rect_union_with_rect() {
                    let vals = $test_vals;
                    let rect1 = <$rect_type>::new(
                        <$vec_type>::from([vals[0], vals[1]]),
                        <$vec_type>::from([vals[2], vals[3]]),
                    );
                    let rect2 = <$rect_type>::new(
                        <$vec_type>::from([vals[6], vals[7]]),
                        <$vec_type>::from([vals[8], vals[9]]),
                    );

                    let min_x = if vals[0] < vals[6] { vals[0] } else { vals[6] };
                    let min_y = if vals[1] < vals[7] { vals[1] } else { vals[7] };
                    let max_x = if vals[2] > vals[8] { vals[2] } else { vals[8] };
                    let max_y = if vals[3] > vals[9] { vals[3] } else { vals[9] };

                    let expected = <$rect_type>::new(
                        <$vec_type>::from([min_x, min_y]),
                        <$vec_type>::from([max_x, max_y]),
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
                        <$vec_type>::from([vals[0], vals[1]]),
                        <$vec_type>::from([vals[2], vals[3]]),
                    );

                    // Point inside rect - should not change rect for most cases
                    let point_inside = <$vec_type>::from([vals[10], vals[11]]);
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
                        <$vec_type>::from([min_x, min_y]),
                        <$vec_type>::from([max_x, max_y]),
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
                        <$vec_type>::from([vals[0], vals[1]]),
                        <$vec_type>::from([vals[4], vals[5]]),
                    );
                    let rect2 = <$rect_type>::new(
                        <$vec_type>::from([vals[6], vals[7]]),
                        <$vec_type>::from([vals[8], vals[9]]),
                    );

                    let min_x = if vals[0] > vals[6] { vals[0] } else { vals[6] };
                    let min_y = if vals[1] > vals[7] { vals[1] } else { vals[7] };
                    let max_x = if vals[4] < vals[8] { vals[4] } else { vals[8] };
                    let max_y = if vals[5] < vals[9] { vals[5] } else { vals[9] };

                    let expected = <$rect_type>::new(
                        <$vec_type>::from([min_x, min_y]),
                        <$vec_type>::from([max_x, max_y]),
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
                        <$vec_type>::from([vals[0], vals[1]]),
                        <$vec_type>::from([vals[4], vals[5]]),
                    );

                    // Point - should create degenerate rect at point
                    let point = <$vec_type>::from([vals[10], vals[11]]);
                    let expected = <$rect_type>::new(
                        <$vec_type>::from([vals[10], vals[11]]),
                        <$vec_type>::from([vals[10], vals[11]]),
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
                        <$vec_type>::from([vals[0], vals[1]]),
                        <$vec_type>::from([vals[6], vals[7]]),
                        <$vec_type>::from([vals[12], vals[13]]),
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
                        <$vec_type>::from([min_x, min_y]),
                        <$vec_type>::from([max_x, max_y]),
                    );
                    assert_rect_eq(rect, expected);

                    // Test center and extent of the built AABB
                    let center = rect.center();
                    let extent = rect.extent();
                    let expected_center = <$vec_type>::from([
                        (min_x + max_x) / (2 as $lane_type),
                        (min_y + max_y) / (2 as $lane_type),
                    ]);
                    let expected_extent = <$vec_type>::from([max_x - min_x, max_y - min_y]);
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
                    $assert_fn(a[0], b[0]);
                    $assert_fn(a[1], b[1]);
                    $assert_fn(a[2], b[2]);
                }

                fn assert_rect_eq(a: $rect_type, b: $rect_type) {
                    assert_vec3_eq(a.min, b.min);
                    assert_vec3_eq(a.max, b.max);
                }

                #[test]
                fn test_rect_new() {
                    let vals = $test_vals;
                    let min = <$vec_type>::from([vals[0], vals[1], vals[2]]);
                    let max = <$vec_type>::from([vals[3], vals[4], vals[5]]);
                    let rect = <$rect_type>::new(min, max);

                    assert_vec3_eq(rect.min, min);
                    assert_vec3_eq(rect.max, max);
                }

                #[test]
                fn test_rect_accessors() {
                    let vals = $test_vals;
                    let min = <$vec_type>::from([vals[0], vals[1], vals[2]]);
                    let max = <$vec_type>::from([vals[3], vals[4], vals[5]]);
                    let rect = <$rect_type>::new(min, max);

                    assert_vec3_eq(rect.min, min);
                    assert_vec3_eq(rect.max, max);
                }

                #[test]
                fn test_rect_center() {
                    let vals = $test_vals;
                    let min = <$vec_type>::from([vals[0], vals[1], vals[2]]);
                    let max = <$vec_type>::from([vals[6], vals[7], vals[8]]);
                    let rect = <$rect_type>::new(min, max);

                    let center = rect.center();
                    let expected = <$vec_type>::from([
                        (vals[0] + vals[6]) / (2 as $lane_type),
                        (vals[1] + vals[7]) / (2 as $lane_type),
                        (vals[2] + vals[8]) / (2 as $lane_type),
                    ]);
                    assert_vec3_eq(center, expected);
                }

                #[test]
                fn test_rect_extent() {
                    let vals = $test_vals;
                    let min = <$vec_type>::from([vals[0], vals[1], vals[2]]);
                    let max = <$vec_type>::from([vals[6], vals[7], vals[8]]);
                    let rect = <$rect_type>::new(min, max);

                    let extent = rect.extent();
                    let expected = <$vec_type>::from([
                        vals[6] - vals[0],
                        vals[7] - vals[1],
                        vals[8] - vals[2],
                    ]);
                    assert_vec3_eq(extent, expected);
                }

                #[test]
                fn test_rect_union_identity() {
                    let identity = <$rect_type>::union_identity();

                    // Union identity should have hi min and lo max
                    $assert_fn(identity.min[0], $hi_val);
                    $assert_fn(identity.min[1], $hi_val);
                    $assert_fn(identity.min[2], $hi_val);
                    $assert_fn(identity.max[0], $lo_val);
                    $assert_fn(identity.max[1], $lo_val);
                    $assert_fn(identity.max[2], $lo_val);

                    // Test that union with any AABB gives that AABB
                    let vals = $test_vals;
                    let test_rect = <$rect_type>::new(
                        <$vec_type>::from([vals[0], vals[1], vals[2]]),
                        <$vec_type>::from([vals[3], vals[4], vals[5]]),
                    );
                    let result = identity | test_rect;
                    assert_rect_eq(result, test_rect);
                }

                #[test]
                fn test_rect_intersection_identity() {
                    let identity = <$rect_type>::intersection_identity();

                    // Intersection identity should have lo min and hi max
                    $assert_fn(identity.min[0], $lo_val);
                    $assert_fn(identity.min[1], $lo_val);
                    $assert_fn(identity.min[2], $lo_val);
                    $assert_fn(identity.max[0], $hi_val);
                    $assert_fn(identity.max[1], $hi_val);
                    $assert_fn(identity.max[2], $hi_val);

                    // Test that intersection with any AABB gives that AABB
                    let vals = $test_vals;
                    let test_rect = <$rect_type>::new(
                        <$vec_type>::from([vals[0], vals[1], vals[2]]),
                        <$vec_type>::from([vals[3], vals[4], vals[5]]),
                    );
                    let result = identity & test_rect;
                    assert_rect_eq(result, test_rect);
                }

                #[test]
                fn test_rect_union_with_rect() {
                    let vals = $test_vals;
                    let rect1 = <$rect_type>::new(
                        <$vec_type>::from([vals[0], vals[1], vals[2]]),
                        <$vec_type>::from([vals[3], vals[4], vals[5]]),
                    );
                    let rect2 = <$rect_type>::new(
                        <$vec_type>::from([vals[9], vals[10], vals[11]]),
                        <$vec_type>::from([vals[12], vals[13], vals[14]]),
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
                        <$vec_type>::from([min_x, min_y, min_z]),
                        <$vec_type>::from([max_x, max_y, max_z]),
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
                        <$vec_type>::from([vals[0], vals[1], vals[2]]),
                        <$vec_type>::from([vals[3], vals[4], vals[5]]),
                    );

                    // Point - should expand bounds if necessary
                    let point = <$vec_type>::from([vals[15], vals[16], vals[17]]);

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
                        <$vec_type>::from([min_x, min_y, min_z]),
                        <$vec_type>::from([max_x, max_y, max_z]),
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
                        <$vec_type>::from([vals[0], vals[1], vals[2]]),
                        <$vec_type>::from([vals[6], vals[7], vals[8]]),
                    );
                    let rect2 = <$rect_type>::new(
                        <$vec_type>::from([vals[9], vals[10], vals[11]]),
                        <$vec_type>::from([vals[12], vals[13], vals[14]]),
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
                        <$vec_type>::from([min_x, min_y, min_z]),
                        <$vec_type>::from([max_x, max_y, max_z]),
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
                        <$vec_type>::from([vals[0], vals[1], vals[2]]),
                        <$vec_type>::from([vals[6], vals[7], vals[8]]),
                    );

                    // Point - should create degenerate rect at point
                    let point = <$vec_type>::from([vals[15], vals[16], vals[17]]);
                    let expected = <$rect_type>::new(
                        <$vec_type>::from([vals[15], vals[16], vals[17]]),
                        <$vec_type>::from([vals[15], vals[16], vals[17]]),
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
                        <$vec_type>::from([vals[0], vals[1], vals[2]]),
                        <$vec_type>::from([vals[9], vals[10], vals[11]]),
                        <$vec_type>::from([vals[18], vals[19], vals[20]]),
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
                        <$vec_type>::from([min_x, min_y, min_z]),
                        <$vec_type>::from([max_x, max_y, max_z]),
                    );
                    assert_rect_eq(rect, expected);

                    // Test center and extent of the built AABB
                    let center = rect.center();
                    let extent = rect.extent();
                    let expected_center = <$vec_type>::from([
                        (min_x + max_x) / (2 as $lane_type),
                        (min_y + max_y) / (2 as $lane_type),
                        (min_z + max_z) / (2 as $lane_type),
                    ]);
                    let expected_extent =
                        <$vec_type>::from([max_x - min_x, max_y - min_y, max_z - min_z]);
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

#[cfg(test)]
mod rect_api_tests {
    use super::*;

    #[test]
    fn test_contains() {
        let r = SimdRect2::new(SimdVec2::from([0.0, 0.0]), SimdVec2::from([2.0, 3.0]));
        assert!(r.contains(SimdVec2::from([1.0, 1.0])));
        assert!(r.contains(SimdVec2::from([0.0, 0.0]))); // bounds are inclusive
        assert!(r.contains(SimdVec2::from([2.0, 3.0]))); // bounds are inclusive
        assert!(!r.contains(SimdVec2::from([2.1, 1.0])));
        assert!(!r.contains(SimdVec2::from([-0.1, 1.0])));

        let r3 = SimdIRect3::new(SimdIVec3::from([0, 0, 0]), SimdIVec3::from([2, 2, 2]));
        assert!(r3.contains(SimdIVec3::from([1, 1, 1])));
        assert!(!r3.contains(SimdIVec3::from([1, 1, 3])));
    }

    #[test]
    fn test_intersects() {
        let a = SimdRect2::new(SimdVec2::from([0.0, 0.0]), SimdVec2::from([2.0, 2.0]));
        let b = SimdRect2::new(SimdVec2::from([1.0, 1.0]), SimdVec2::from([3.0, 3.0]));
        let c = SimdRect2::new(SimdVec2::from([5.0, 5.0]), SimdVec2::from([6.0, 6.0]));
        let touching = SimdRect2::new(SimdVec2::from([2.0, 0.0]), SimdVec2::from([3.0, 2.0]));
        assert!(a.intersects(&b));
        assert!(b.intersects(&a));
        assert!(!a.intersects(&c));
        assert!(a.intersects(&touching)); // touching edges count

        let a3 = SimdURect3::new(SimdUVec3::from([0, 0, 0]), SimdUVec3::from([2, 2, 2]));
        let b3 = SimdURect3::new(SimdUVec3::from([3, 0, 0]), SimdUVec3::from([4, 2, 2]));
        assert!(!a3.intersects(&b3));
    }

    #[test]
    fn test_is_empty() {
        assert!(SimdRect2::union_identity().is_empty());
        assert!(!SimdRect2::intersection_identity().is_empty());
        assert!(SimdURect3::union_identity().is_empty());

        // A degenerate point-rect still contains one point.
        let point_rect = SimdRect2::new(SimdVec2::ZERO, SimdVec2::ZERO);
        assert!(!point_rect.is_empty());

        // Intersection of disjoint rects is empty.
        let a = SimdRect3::new(SimdVec3::ZERO, SimdVec3::from([1.0, 1.0, 1.0]));
        let b = SimdRect3::new(
            SimdVec3::from([2.0, 0.0, 0.0]),
            SimdVec3::from([3.0, 1.0, 1.0]),
        );
        assert!((a & b).is_empty());
        assert!(!a.intersects(&b));
    }

    #[test]
    fn test_array_conversions() {
        let r = SimdIRect2::from([[0, 1], [4, 5]]);
        assert_eq!(r.min, SimdIVec2::from([0, 1]));
        assert_eq!(r.max, SimdIVec2::from([4, 5]));
        let arr: [[i32; 2]; 2] = r.into();
        assert_eq!(arr, [[0, 1], [4, 5]]);

        let arr3: [[f32; 3]; 2] = SimdRect3::new(
            SimdVec3::from([1.0, 2.0, 3.0]),
            SimdVec3::from([4.0, 5.0, 6.0]),
        )
        .into();
        assert_eq!(arr3, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    }

    #[test]
    fn test_int_center_no_overflow() {
        // (min + max) / 2 would overflow here; min + extent / 2 must not.
        let r = SimdIRect2::new(
            SimdIVec2::from([i32::MAX - 4, i32::MAX - 8]),
            SimdIVec2::from([i32::MAX, i32::MAX]),
        );
        let center = r.center();
        assert_eq!(center, SimdIVec2::from([i32::MAX - 2, i32::MAX - 4]));

        let r = SimdURect2::new(
            SimdUVec2::from([u32::MAX - 4, u32::MAX - 8]),
            SimdUVec2::from([u32::MAX, u32::MAX]),
        );
        let center = r.center();
        assert_eq!(center, SimdUVec2::from([u32::MAX - 2, u32::MAX - 4]));
    }
}
