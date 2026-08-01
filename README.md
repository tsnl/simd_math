# `simd_math`

[![CI](https://github.com/tsnl/simd_math/actions/workflows/ci.yml/badge.svg)](https://github.com/tsnl/simd_math/actions/workflows/ci.yml)
[![Publish](https://github.com/tsnl/simd_math/actions/workflows/publish.yml/badge.svg)](https://github.com/tsnl/simd_math/actions/workflows/publish.yml)

A Rust library providing SIMD-accelerated mathematical functions for games, graphics, robotics, and other spatial computing applications.

> [!IMPORTANT]
> This crate is built on Rust's unstable [`portable_simd`](https://doc.rust-lang.org/std/simd/index.html) feature and requires a **nightly** toolchain. This repository pins a known-good nightly via `rust-toolchain.toml`, so `cargo build` inside the repo just works.

## Example

```rust
// See tests/integration_test.rs test_readme_example() for full runnable code.
// See tests/integration_test.rs for more examples.

use simd_math::prelude::*;
use std::f32::consts::PI;

// Create a 3D vector
let vector = SimdVec3::from([1.0, 0.0, 0.0]);

// Create a rotation quaternion (90 degrees around Z-axis)
let rotation = SimdUnitQuat::from_axis_angle(SimdVec3::from([0.0, 0.0, 1.0]), PI / 2.0);

// Apply rotation to vector
let rotated_vector = rotation * vector;

// Components are accessed by index; the prelude provides X/Y/Z/W constants
assert!((rotated_vector[Y] - 1.0).abs() < 1e-6);
```

## Conventions and Quirks

### Arrays

We ensure that our algebraic types are convertible to and from Rust arrays. Use arrays in your APIs for simple, math-library-agnostic interfaces. Cf [Mujoco](https://github.com/google-deepmind/mujoco).

In the same spirit, we do not expose `.x`, `.y`, `.z` fields on vectors or quaternions. Use indexing: `v[0]`, `v[1]`, `v[2]`, or equivalently `v[X]`, `v[Y]`, `v[Z]` with the named index constants from `simd_math::prelude`.

This decision further eschews conventions about what is front, up, and right in 3D space, which vary between applications. It also makes it easier to use vectors for non-spatial data, e.g. color.

### Boring Algebraic Conventions

Vectors are column vectors. Matrix multiplication is `Matrix * Vector`.

Matrices are constructed in column-major order, i.e. each argument to `SimdMat3x4::new` is a column. This is consistent with OpenGL and GLM, but not with most textbook math. `SimdMat3x4` is a 3×4 affine transform: three transformed basis-vector columns plus a translation column, with an implicit `[0, 0, 0, 1]` bottom row.

Quaternions are represented as `(s, x, y, z)` where `s` is the scalar part. Think `s + iv`.

We assume a right-handed coordinate system.

Spherical coordinates are `(azimuth, elevation, radius)` with Y up: azimuth is measured in the XZ plane from +X towards +Z in `[-π, π]`, and elevation is measured towards +Y in `[-π/2, π/2]`.

### Hidden SIMD lanes

3-component vectors are stored in 4-lane registers. The extra lane is an internal detail: it always holds zero, every operation preserves that invariant, and equality, `Debug`, indexing, and reductions all ignore it. Indexing a `SimdVec3` with `v[3]` panics.
