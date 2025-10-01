# `simd_math`

A Rust library providing SIMD-accelerated mathematical functions for games, graphics, robotics, and other spatial computing applications.

## Example

```rust
// See tests/integration_test.rs test_readme_example() for full runnable code.
// See tests/integration_test.rs for more examples.

use simd_math::*;
use std::f32::consts::PI;

// Create a 3D vector
let vector = SimdVec3::new(1.0, 0.0, 0.0);

// Create a rotation quaternion (90 degrees around Z-axis)
let rotation = SimdUnitQuat::from_axis_angle(SimdVec3::new(0.0, 0.0, 1.0), PI / 2.0);

// Apply rotation to vector
let rotated_vector = rotation * vector;
```

