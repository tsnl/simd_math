# `simd_math`

[![CI](https://github.com/tsnl/simd_math/actions/workflows/ci.yml/badge.svg)](https://github.com/tsnl/simd_math/actions/workflows/ci.yml)
[![Publish](https://github.com/tsnl/simd_math/actions/workflows/publish.yml/badge.svg)](https://github.com/tsnl/simd_math/actions/workflows/publish.yml)

A Rust library providing SIMD-accelerated mathematical functions for games, graphics, robotics, and other spatial computing applications.

## Example

```rust
// See tests/integration_test.rs test_readme_example() for full runnable code.
// See tests/integration_test.rs for more examples.

use simd_math::*;
use std::f32::consts::PI;

// Create a 3D vector
let vector = SimdVec3::from([1.0, 0.0, 0.0]);

// Create a rotation quaternion (90 degrees around Z-axis)
let rotation = SimdUnitQuat::from_axis_angle(SimdVec3::from([0.0, 0.0, 1.0]), PI / 2.0);

// Apply rotation to vector
let rotated_vector = rotation * vector;
```

## Conventions and Quirks

### Arrays

We ensure that our algebraic types are convertible to and from Rust arrays. Use arrays in your APIs for simple, math-library-agnostic interfaces. Cf [Mujoco](https://github.com/google-deepmind/mujoco).

In the same spirit, we do not expose `.x`, `.y`, `.z` fields on vectors or quaternions. Use indexing (`v[0]`, `v[1]`, `v[2]`).

This decision further eschews conventions about what is front, up, and right in 3D space, which vary between applications. It also makes it easier to use vectors for non-spatial data, e.g. color.

### Boring Algebraic Conventions

Vectors are column vectors. Matrix multiplication is `Matrix * Vector`.

Matrices are constructed in column-major order, i.e. each argument to `Mat3::new` is a column.
This is consistent with OpenGL and GLM, but not with most textbook math.

Quaternions are represented as `(s, x, y, z)` where `s` is the scalar part. Think `s + iv`.

We assume a right-handed coordinate system.
