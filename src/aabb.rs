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
