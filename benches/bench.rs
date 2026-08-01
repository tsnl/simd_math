//! Micro-benchmarks for the core operations, with scalar baselines to keep the
//! "SIMD-accelerated" claim honest. Run with `cargo bench`.
//!
//! Uses the nightly libtest bench harness, which this nightly-only crate gets
//! for free (no extra dependencies).
#![feature(test)]

extern crate test;

use simd_math::prelude::*;
use std::f32::consts::PI;
use test::{Bencher, black_box};

const N: usize = 1024;

fn make_vec3s(seed: f32) -> Vec<SimdVec3> {
    (0..N)
        .map(|i| {
            let x = (i as f32 * 0.37 + seed).sin();
            let y = (i as f32 * 0.11 + seed).cos();
            let z = (i as f32 * 0.53 + seed).sin();
            SimdVec3::from([x, y, z])
        })
        .collect()
}

fn make_arrays(seed: f32) -> Vec<[f32; 3]> {
    make_vec3s(seed).into_iter().map(Into::into).collect()
}

#[bench]
fn bench_vec3_dot_simd(b: &mut Bencher) {
    let xs = make_vec3s(0.0);
    let ys = make_vec3s(1.0);
    b.iter(|| {
        let mut acc = 0.0;
        for (x, y) in xs.iter().zip(&ys) {
            acc += black_box(*x).dot(black_box(*y));
        }
        acc
    });
}

#[bench]
fn bench_vec3_dot_scalar(b: &mut Bencher) {
    let xs = make_arrays(0.0);
    let ys = make_arrays(1.0);
    b.iter(|| {
        let mut acc = 0.0;
        for (x, y) in xs.iter().zip(&ys) {
            let x = black_box(*x);
            let y = black_box(*y);
            acc += x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
        }
        acc
    });
}

#[bench]
fn bench_vec3_add_mul_simd(b: &mut Bencher) {
    let xs = make_vec3s(0.0);
    let ys = make_vec3s(1.0);
    b.iter(|| {
        let mut acc = SimdVec3::ZERO;
        for (x, y) in xs.iter().zip(&ys) {
            acc += black_box(*x) * 0.5 + black_box(*y) * 2.0;
        }
        acc
    });
}

#[bench]
fn bench_vec3_add_mul_scalar(b: &mut Bencher) {
    let xs = make_arrays(0.0);
    let ys = make_arrays(1.0);
    b.iter(|| {
        let mut acc = [0.0f32; 3];
        for (x, y) in xs.iter().zip(&ys) {
            let x = black_box(*x);
            let y = black_box(*y);
            for i in 0..3 {
                acc[i] += x[i] * 0.5 + y[i] * 2.0;
            }
        }
        acc
    });
}

#[bench]
fn bench_vec3_cross(b: &mut Bencher) {
    let xs = make_vec3s(0.0);
    let ys = make_vec3s(1.0);
    b.iter(|| {
        let mut acc = SimdVec3::ZERO;
        for (x, y) in xs.iter().zip(&ys) {
            acc += black_box(*x).cross(black_box(*y));
        }
        acc
    });
}

#[bench]
fn bench_vec3_normalize(b: &mut Bencher) {
    let xs = make_vec3s(0.0);
    b.iter(|| {
        let mut acc = SimdVec3::ZERO;
        for x in &xs {
            if let Some(n) = black_box(*x).normalized() {
                acc += n;
            }
        }
        acc
    });
}

#[bench]
fn bench_quat_rotate_vec3(b: &mut Bencher) {
    let q = SimdUnitQuat::from_axis_angle(SimdVec3::from([1.0, 2.0, 3.0]), PI / 3.0);
    let xs = make_vec3s(0.0);
    b.iter(|| {
        let mut acc = SimdVec3::ZERO;
        for x in &xs {
            acc += black_box(q) * black_box(*x);
        }
        acc
    });
}

#[bench]
fn bench_quat_mul_quat(b: &mut Bencher) {
    let a = SimdUnitQuat::from_axis_angle(SimdVec3::from([1.0, 2.0, 3.0]), PI / 3.0);
    let c = SimdUnitQuat::from_axis_angle(SimdVec3::from([3.0, 1.0, 2.0]), PI / 5.0);
    b.iter(|| {
        let mut q = black_box(a);
        for _ in 0..N {
            q = q * black_box(c);
        }
        q
    });
}

#[bench]
fn bench_affine_transform_points_simd(b: &mut Bencher) {
    let q = SimdUnitQuat::from_axis_angle(SimdVec3::from([1.0, 2.0, 3.0]), PI / 3.0);
    let m = SimdAffine3::from_translation(SimdVec3::from([4.0, 5.0, 6.0])) * SimdAffine3::from(q);
    let xs = make_vec3s(0.0);
    b.iter(|| {
        let mut acc = SimdVec3::ZERO;
        for x in &xs {
            acc += black_box(m) * black_box(*x);
        }
        acc
    });
}

#[bench]
fn bench_affine_transform_points_scalar(b: &mut Bencher) {
    let q = SimdUnitQuat::from_axis_angle(SimdVec3::from([1.0, 2.0, 3.0]), PI / 3.0);
    let m = SimdAffine3::from_translation(SimdVec3::from([4.0, 5.0, 6.0])) * SimdAffine3::from(q);
    let cols: [[f32; 3]; 4] = m.into();
    let xs = make_arrays(0.0);
    b.iter(|| {
        let mut acc = [0.0f32; 3];
        for x in &xs {
            let cols = black_box(cols);
            let x = black_box(*x);
            for i in 0..3 {
                acc[i] += cols[0][i] * x[0] + cols[1][i] * x[1] + cols[2][i] * x[2] + cols[3][i];
            }
        }
        acc
    });
}

#[bench]
fn bench_affine_mul_affine(b: &mut Bencher) {
    let q = SimdUnitQuat::from_axis_angle(SimdVec3::from([1.0, 2.0, 3.0]), PI / 3.0);
    let m = SimdAffine3::from_translation(SimdVec3::from([4.0, 5.0, 6.0])) * SimdAffine3::from(q);
    b.iter(|| {
        let mut acc = SimdAffine3::IDENTITY;
        for _ in 0..N {
            acc = acc * black_box(m);
        }
        acc
    });
}

#[bench]
fn bench_rect3_union_points(b: &mut Bencher) {
    let xs = make_vec3s(0.0);
    b.iter(|| {
        let mut bounds = SimdRect3::union_identity();
        for x in &xs {
            bounds |= black_box(*x);
        }
        bounds
    });
}
