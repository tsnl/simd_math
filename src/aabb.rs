use crate::vector::SimdVec3;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign};

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
        self.min = self.min.elementwise_max(other.min);
        self.max = self.max.elementwise_min(other.max);
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
mod simd_aabb_tests {
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

    fn assert_aabb_near(a: SimdAABB, b: SimdAABB) {
        assert_vec3_near(a.min, b.min);
        assert_vec3_near(a.max, b.max);
    }

    #[test]
    fn test_aabb_new() {
        let min = SimdVec3::new(1.0, 2.0, 3.0);
        let max = SimdVec3::new(4.0, 5.0, 6.0);
        let aabb = SimdAABB::new(min, max);

        assert_vec3_near(aabb.min(), min);
        assert_vec3_near(aabb.max(), max);
    }

    #[test]
    fn test_aabb_accessors() {
        let min = SimdVec3::new(-1.0, -2.0, -3.0);
        let max = SimdVec3::new(1.0, 2.0, 3.0);
        let aabb = SimdAABB::new(min, max);

        assert_vec3_near(aabb.min(), min);
        assert_vec3_near(aabb.max(), max);
    }

    #[test]
    fn test_aabb_center() {
        let min = SimdVec3::new(0.0, 0.0, 0.0);
        let max = SimdVec3::new(4.0, 6.0, 8.0);
        let aabb = SimdAABB::new(min, max);

        let center = aabb.center();
        assert_vec3_near(center, SimdVec3::new(2.0, 3.0, 4.0));
    }

    #[test]
    fn test_aabb_extent() {
        let min = SimdVec3::new(1.0, 2.0, 3.0);
        let max = SimdVec3::new(5.0, 8.0, 11.0);
        let aabb = SimdAABB::new(min, max);

        let extent = aabb.extent();
        assert_vec3_near(extent, SimdVec3::new(4.0, 6.0, 8.0));
    }

    #[test]
    fn test_aabb_union_identity() {
        let identity = SimdAABB::union_identity();

        // Union identity should have infinite min and negative infinite max
        assert_eq!(identity.min.x(), f32::INFINITY);
        assert_eq!(identity.min.y(), f32::INFINITY);
        assert_eq!(identity.min.z(), f32::INFINITY);
        assert_eq!(identity.max.x(), f32::NEG_INFINITY);
        assert_eq!(identity.max.y(), f32::NEG_INFINITY);
        assert_eq!(identity.max.z(), f32::NEG_INFINITY);

        // Test that union with any AABB gives that AABB
        let test_aabb = SimdAABB::new(SimdVec3::new(1.0, 2.0, 3.0), SimdVec3::new(4.0, 5.0, 6.0));
        let result = identity | test_aabb;
        assert_aabb_near(result, test_aabb);
    }

    #[test]
    fn test_aabb_intersection_identity() {
        let identity = SimdAABB::intersection_identity();

        // Intersection identity should have negative infinite min and infinite max
        assert_eq!(identity.min.x(), f32::NEG_INFINITY);
        assert_eq!(identity.min.y(), f32::NEG_INFINITY);
        assert_eq!(identity.min.z(), f32::NEG_INFINITY);
        assert_eq!(identity.max.x(), f32::INFINITY);
        assert_eq!(identity.max.y(), f32::INFINITY);
        assert_eq!(identity.max.z(), f32::INFINITY);

        // Test that intersection with any AABB gives that AABB
        let test_aabb = SimdAABB::new(SimdVec3::new(1.0, 2.0, 3.0), SimdVec3::new(4.0, 5.0, 6.0));
        let result = identity & test_aabb;
        assert_aabb_near(result, test_aabb);
    }

    #[test]
    fn test_aabb_union_with_aabb() {
        let aabb1 = SimdAABB::new(SimdVec3::new(0.0, 0.0, 0.0), SimdVec3::new(2.0, 2.0, 2.0));
        let aabb2 = SimdAABB::new(SimdVec3::new(1.0, 1.0, 1.0), SimdVec3::new(3.0, 3.0, 3.0));

        let union = aabb1 | aabb2;
        let expected = SimdAABB::new(SimdVec3::new(0.0, 0.0, 0.0), SimdVec3::new(3.0, 3.0, 3.0));
        assert_aabb_near(union, expected);

        // Test |= operator
        let mut aabb3 = aabb1;
        aabb3 |= aabb2;
        assert_aabb_near(aabb3, expected);
    }

    #[test]
    fn test_aabb_union_with_point() {
        let aabb = SimdAABB::new(SimdVec3::new(1.0, 1.0, 1.0), SimdVec3::new(3.0, 3.0, 3.0));

        // Point inside AABB - should not change AABB
        let point_inside = SimdVec3::new(2.0, 2.0, 2.0);
        let result1 = aabb | point_inside;
        assert_aabb_near(result1, aabb);

        // Point outside AABB - should expand AABB
        let point_outside = SimdVec3::new(0.0, 4.0, 2.0);
        let result2 = aabb | point_outside;
        let expected = SimdAABB::new(SimdVec3::new(0.0, 1.0, 1.0), SimdVec3::new(3.0, 4.0, 3.0));
        assert_aabb_near(result2, expected);

        // Test |= operator
        let mut aabb3 = aabb;
        aabb3 |= point_outside;
        assert_aabb_near(aabb3, expected);
    }

    #[test]
    fn test_aabb_intersection_with_aabb() {
        let aabb1 = SimdAABB::new(SimdVec3::new(0.0, 0.0, 0.0), SimdVec3::new(4.0, 4.0, 4.0));
        let aabb2 = SimdAABB::new(SimdVec3::new(2.0, 2.0, 2.0), SimdVec3::new(6.0, 6.0, 6.0));

        let intersection = aabb1 & aabb2;
        let expected = SimdAABB::new(SimdVec3::new(2.0, 2.0, 2.0), SimdVec3::new(4.0, 4.0, 4.0));
        assert_aabb_near(intersection, expected);

        // Test &= operator
        let mut aabb3 = aabb1;
        aabb3 &= aabb2;
        assert_aabb_near(aabb3, expected);
    }

    #[test]
    fn test_aabb_intersection_with_point() {
        let aabb = SimdAABB::new(SimdVec3::new(1.0, 1.0, 1.0), SimdVec3::new(5.0, 5.0, 5.0));

        // Point inside AABB - should shrink AABB towards point
        let point = SimdVec3::new(3.0, 2.0, 4.0);
        let result = aabb & point;
        let expected = SimdAABB::new(SimdVec3::new(3.0, 2.0, 4.0), SimdVec3::new(3.0, 2.0, 4.0));
        assert_aabb_near(result, expected);

        // Test &= operator
        let mut aabb2 = aabb;
        aabb2 &= point;
        assert_aabb_near(aabb2, expected);
    }

    #[test]
    fn test_aabb_no_intersection() {
        let aabb1 = SimdAABB::new(SimdVec3::new(0.0, 0.0, 0.0), SimdVec3::new(2.0, 2.0, 2.0));
        let aabb2 = SimdAABB::new(SimdVec3::new(3.0, 3.0, 3.0), SimdVec3::new(5.0, 5.0, 5.0));

        let intersection = aabb1 & aabb2;

        // When there's no intersection, the result should have min > max
        assert!(intersection.min.x() > intersection.max.x());
        assert!(intersection.min.y() > intersection.max.y());
        assert!(intersection.min.z() > intersection.max.z());
    }

    #[test]
    fn test_aabb_comprehensive_operations() {
        // Start with union identity
        let mut aabb = SimdAABB::union_identity();

        // Add some points to build an AABB
        let points = [
            SimdVec3::new(1.0, 2.0, 3.0),
            SimdVec3::new(-1.0, 4.0, 1.0),
            SimdVec3::new(3.0, 0.0, 5.0),
        ];

        for point in points.iter() {
            aabb |= *point;
        }

        let expected = SimdAABB::new(SimdVec3::new(-1.0, 0.0, 1.0), SimdVec3::new(3.0, 4.0, 5.0));
        assert_aabb_near(aabb, expected);

        // Test center and extent of the built AABB
        let center = aabb.center();
        let extent = aabb.extent();
        assert_vec3_near(center, SimdVec3::new(1.0, 2.0, 3.0));
        assert_vec3_near(extent, SimdVec3::new(4.0, 4.0, 4.0));
    }
}
