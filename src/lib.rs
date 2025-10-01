#![feature(portable_simd)]

//! A SIMD-accelerated math library for 3D graphics and linear algebra operations.
//!
//! This library provides high-performance vector, quaternion, matrix, transform, and AABB
//! operations using Rust's portable SIMD functionality.

mod rect;
pub use rect::*;

mod matrix;
pub use matrix::*;

mod quaternion;
pub use quaternion::*;

mod transform;
pub use transform::*;

mod vector;
pub use vector::*;

use std::simd::prelude::*;
