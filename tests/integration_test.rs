//! Integration tests to verify that all public APIs are properly re-exported from the main library.

use simd_math::*;
use std::f32::consts::PI;

#[test]
fn test_vector_operations() {
    // Test vector creation and basic operations
    let v1 = SimdVec3::new(1.0, 2.0, 3.0);
    let v2 = SimdVec3::new(4.0, 5.0, 6.0);

    let sum = v1 + v2;
    assert_eq!(sum.x(), 5.0);
    assert_eq!(sum.y(), 7.0);
    assert_eq!(sum.z(), 9.0);

    // Test constants
    let zero = SimdVec3::ZERO;
    let unit_x = SimdVec3::UNIT_X;

    assert_eq!(zero.x(), 0.0);
    assert_eq!(unit_x.x(), 1.0);
    assert_eq!(unit_x.y(), 0.0);
}

#[test]
fn test_quaternion_operations() {
    // Test quaternion creation and rotation
    let _identity = SimdUnitQuat::IDENTITY;
    let axis = SimdVec3::new(0.0, 0.0, 1.0);
    let angle = PI / 2.0;
    let rotation = SimdUnitQuat::from_axis_angle(axis, angle);

    // Test vector rotation
    let v = SimdVec3::new(1.0, 0.0, 0.0);
    let rotated = rotation * v;

    // 90-degree rotation around Z should turn (1,0,0) into (0,1,0)
    assert!((rotated.x() - 0.0).abs() < 1e-6);
    assert!((rotated.y() - 1.0).abs() < 1e-6);
    assert!((rotated.z() - 0.0).abs() < 1e-6);
}

#[test]
fn test_matrix_operations() {
    // Test matrix creation and transformation
    let _identity = SimdMat4::identity();
    let translation = SimdVec3::new(1.0, 2.0, 3.0);
    let transform_matrix = SimdMat4::from_translation(translation);

    let point = SimdVec3::new(0.0, 0.0, 0.0);
    let transformed = transform_matrix * point;

    assert_eq!(transformed.x(), 1.0);
    assert_eq!(transformed.y(), 2.0);
    assert_eq!(transformed.z(), 3.0);
}

#[test]
fn test_transform_operations() {
    // Test transform composition
    let translation = SimdVec3::new(1.0, 0.0, 0.0);
    let rotation = SimdUnitQuat::from_axis_angle(SimdVec3::new(0.0, 0.0, 1.0), PI / 2.0);
    let transform = SimdTransform::new(translation, rotation);

    let point = SimdVec3::new(1.0, 0.0, 0.0);
    let transformed = transform * point;

    // Should rotate then translate: (1,0,0) -> (0,1,0) -> (1,1,0)
    assert!((transformed.x() - 1.0).abs() < 1e-6);
    assert!((transformed.y() - 1.0).abs() < 1e-6);
    assert!((transformed.z() - 0.0).abs() < 1e-6);
}

#[test]
fn test_aabb_operations() {
    // Test AABB creation and operations
    let min = SimdVec3::new(0.0, 0.0, 0.0);
    let max = SimdVec3::new(1.0, 1.0, 1.0);
    let aabb = SimdAABB::new(min, max);

    let center = aabb.center();
    let extent = aabb.extent();

    assert_eq!(center.x(), 0.5);
    assert_eq!(center.y(), 0.5);
    assert_eq!(center.z(), 0.5);

    assert_eq!(extent.x(), 1.0);
    assert_eq!(extent.y(), 1.0);
    assert_eq!(extent.z(), 1.0);
}

#[test]
fn test_spherical_coordinates() {
    // Test spherical coordinate conversion
    let cartesian = SimdVec3::new(1.0, 0.0, 0.0);
    let spherical = cartesian.into_spherical_coords();

    // Should be (azimuth=0, elevation=0, radius=1)
    assert!((spherical.x() - 0.0).abs() < 1e-6); // azimuth
    assert!((spherical.y() - 0.0).abs() < 1e-6); // elevation
    assert!((spherical.z() - 1.0).abs() < 1e-6); // radius

    // Test equirectangular coordinate conversion
    let angles = SimdVec2::new(0.0, 0.0); // center
    let uv = angles.spherical_coords_angles_into_equirectangular_coords();

    assert!((uv.x() - 0.5).abs() < 1e-6);
    assert!((uv.y() - 0.5).abs() < 1e-6);
}

#[test]
fn test_cross_product() {
    // Test cross product
    let x_axis = SimdVec3::new(1.0, 0.0, 0.0);
    let y_axis = SimdVec3::new(0.0, 1.0, 0.0);
    let z_axis = x_axis.cross(y_axis);

    assert!((z_axis.x() - 0.0).abs() < 1e-6);
    assert!((z_axis.y() - 0.0).abs() < 1e-6);
    assert!((z_axis.z() - 1.0).abs() < 1e-6);
}
