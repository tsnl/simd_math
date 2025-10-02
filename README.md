# `simd_math`

[![CI](https://github.com/tsnl/simd_math/actions/workflows/ci.yml/badge.svg)](https://github.com/tsnl/simd_math/actions/workflows/ci.yml)

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

-   Vectors are column vectors. Matrix multiplication is `Matrix * Vector`.
-   Quaternions are represented as `(s, x, y, z)` where `s` is the scalar part. Think `s + iv`
-   We do not expose `.x`, `.y`, `.z` fields on vectors or quaternions. Use indexing (`v[0]`, `v[1]`, `v[2]`).
-   We generally remain intercompatible with Rust array types (e.g. `SimdVec3` can be created from `[f32; 3]` and converted back). This means you can use Rust array types and slices in your APIs for a math-library-agnostic interface. Cf [Mujoco](https://github.com/google-deepmind/mujoco).
