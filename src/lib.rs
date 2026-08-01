#![feature(portable_simd)]
#![warn(missing_docs)]

//! A SIMD-accelerated math library for games, graphics, robotics, and other
//! spatial computing applications.
//!
//! Provides vector ([`SimdVec3`] and friends), quaternion ([`SimdUnitQuat`]),
//! affine-matrix ([`SimdAffine3`]), rigid-transform ([`SimdTransform`]), and
//! axis-aligned bounding box ([`SimdRect3`] and friends) types built on Rust's
//! portable SIMD.
//!
//! # Nightly requirement
//!
//! This crate uses the unstable [`portable_simd`] feature and therefore
//! requires a **nightly** toolchain. This repository pins a known-good nightly
//! via `rust-toolchain.toml`.
//!
//! [`portable_simd`]: https://doc.rust-lang.org/std/simd/index.html
//!
//! # Example
//!
//! ```
//! use simd_math::prelude::*;
//! use std::f32::consts::PI;
//!
//! // Create a 3D vector
//! let vector = SimdVec3::from([1.0, 0.0, 0.0]);
//!
//! // Create a rotation quaternion (90 degrees around Z-axis)
//! let rotation = SimdUnitQuat::from_axis_angle(SimdVec3::from([0.0, 0.0, 1.0]), PI / 2.0);
//!
//! // Apply rotation to vector; components are indexed, with named index
//! // constants available from the prelude.
//! let rotated = rotation * vector;
//! assert!((rotated[Y] - 1.0).abs() < 1e-6);
//! ```

mod affine;
pub use affine::*;

mod quaternion;
pub use quaternion::*;

mod rect;
pub use rect::*;

mod transform;
pub use transform::*;

mod vector;
pub use vector::*;

pub mod prelude {
    //! One-stop import: every public type, plus named component indices.
    //!
    //! The [`X`]/[`Y`]/[`Z`]/[`W`] constants let you index vector components
    //! by name instead of by bare integer: `v[X]` instead of `v[0]`.

    pub use crate::{
        SimdAffine3, SimdIRect2, SimdIRect3, SimdIVec2, SimdIVec3, SimdIVec4, SimdRect2, SimdRect3,
        SimdTransform, SimdURect2, SimdURect3, SimdUVec2, SimdUVec3, SimdUVec4, SimdUnitQuat,
        SimdVec2, SimdVec3, SimdVec4, SimdVector,
    };

    /// Index of the X component of a vector.
    pub const X: usize = 0;
    /// Index of the Y component of a vector.
    pub const Y: usize = 1;
    /// Index of the Z component of a vector.
    pub const Z: usize = 2;
    /// Index of the W component of a vector.
    pub const W: usize = 3;
}

use std::simd::prelude::*;
