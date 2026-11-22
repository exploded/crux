//! Resampling a frame onto the reference grid: homography plus Lanczos-3
//! interpolation with PixInsight-style clamping (WBPP: c = 0.15).
//!
//! Pixel centres sit at integer coordinates in crux (PixInsight puts them
//! at +0.5; `Homography::from_pixinsight` converts).

use crate::image::Image;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// 3×3 projective transform, row-major: (x', y', w') = H · (x, y, 1).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Homography(pub [f64; 9]);

impl Homography {
    pub fn identity() -> Self {
        Homography([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    }

    pub fn from_affine(a: &crate::transparency::Affine) -> Self {
        Homography([a.a[1], a.a[2], a.a[0], a.b[1], a.b[2], a.b[0], 0.0, 0.0, 1.0])
    }

    #[inline]
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        let m = &self.0;
        let w = m[6] * x + m[7] * y + m[8];
        ((m[0] * x + m[1] * y + m[2]) / w, (m[3] * x + m[4] * y + m[5]) / w)
    }

    pub fn inverse(&self) -> Option<Homography> {
        let m = &self.0;
        let det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
        if det.abs() < 1e-15 {
            return None;
        }
        let inv = [
            (m[4] * m[8] - m[5] * m[7]) / det,
            (m[2] * m[7] - m[1] * m[8]) / det,
            (m[1] * m[5] - m[2] * m[4]) / det,
            (m[5] * m[6] - m[3] * m[8]) / det,
            (m[0] * m[8] - m[2] * m[6]) / det,
            (m[2] * m[3] - m[0] * m[5]) / det,
            (m[3] * m[7] - m[4] * m[6]) / det,
            (m[1] * m[6] - m[0] * m[7]) / det,
            (m[0] * m[4] - m[1] * m[3]) / det,
        ];
        let s = inv[8];
        Some(Homography(inv.map(|v| v / s)))
    }

    /// self ∘ other: apply `other` first.
    pub fn then_after(&self, other: &Homography) -> Homography {
        let (a, b) = (&self.0, &other.0);
        let mut r = [0f64; 9];
        for i in 0..3 {
            for j in 0..3 {
                r[i * 3 + j] = (0..3).map(|k| a[i * 3 + k] * b[k * 3 + j]).sum();
            }
        }
        let s = r[8];
        Homography(r.map(|v| v / s))
    }

    /// A PixInsight matrix whose pixel centres are at +`origin` (0.5).
    pub fn from_pixinsight(m: [f64; 9], origin: f64) -> Homography {
        let t = Homography([1.0, 0.0, origin, 0.0, 1.0, origin, 0.0, 0.0, 1.0]);
        let ti = Homography([1.0, 0.0, -origin, 0.0, 1.0, -origin, 0.0, 0.0, 1.0]);
        ti.then_after(&Homography(m)).then_after(&t)
    }

    /// Translation, rotation (degrees) and mean scale of the linear part.
    pub fn summary(&self) -> (f64, f64, f64, f64) {
        let m = &self.0;
        let rot = m[3].atan2(m[0]).to_degrees();
        let scale = ((m[0] * m[4] - m[1] * m[3]).abs()).sqrt();
        (m[2], m[5], rot, scale)
    }
}

const LUT_RES: usize = 4096;

/// Lanczos-3 kernel sampled every 1/LUT_RES pixel over [0, 3].
fn lanczos_lut() -> &'static [f32] {
    static LUT: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();
    LUT.get_or_init(|| {
        (0..=3 * LUT_RES + 1)
            .map(|i| {
                let x = i as f64 / LUT_RES as f64;
                if x == 0.0 {
                    1.0
                } else if x >= 3.0 {
                    0.0
                } else {
                    let px = std::f64::consts::PI * x;
                    (3.0 * px.sin() * (px / 3.0).sin() / (px * px)) as f32
                }
            })
            .collect()
    })
}

/// The kernel table, for the GPU's copy.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub(crate) fn lanczos_table() -> &'static [f32] {
    lanczos_lut()
}

/// Warps smaller than this stay on the CPU (uploading costs more than it
/// saves).
#[cfg(feature = "gpu")]
const GPU_MIN_PIXELS: usize = 1 << 20;

/// The warp on the GPU when there is one and the job is big enough; `None`
/// means do it on the CPU (a GPU failure also switches the GPU off).
#[cfg(feature = "gpu")]
fn on_gpu(src: &Image, w: usize, h: usize, step: usize, map: &(dyn Fn(f64, f64) -> Option<(f64, f64)> + Sync), nan_outside: bool, clamp: Option<f32>) -> Option<Vec<f32>> {
    if w * h < GPU_MIN_PIXELS {
        return None;
    }
    let gpu = crate::gpu::get()?;
    let (gw, nodes) = crate::gpu::warp::grid(w, h, step, map);
    match crate::gpu::warp::run(gpu, src, w, h, step, gw, &nodes, nan_outside, clamp) {
        Ok(v) => v,
        Err(e) => {
            crate::gpu::fail(&format!("warp: {e:#}"));
            None
        }
    }
}

#[inline]
fn lw(lut: &[f32], d: f32) -> f32 {
    let a = d.abs() * LUT_RES as f32;
    let i = a as usize;
    if i >= 3 * LUT_RES {
        return 0.0;
    }
    let f = a - i as f32;
    lut[i] + (lut[i + 1] - lut[i]) * f
}

/// One 1-D Lanczos sum, normalised by the weights. With clamping, when the
/// negative lobes take away more than `c` of the positive lobes'
/// contribution (ringing beside a bright star), only the positive lobes
/// are used. Off by default: plain Lanczos-3 matches WBPP's registered
/// frames to 0.5 ADU σ; its own clamping rule (c = 0.15) changes 0.28% of
/// pixels near stars by ~6 ADU and isn't reproduced here.
#[inline]
fn sum1(vals: &[f32; 6], w: &[f32; 6], clamp: Option<f32>) -> f32 {
    let (mut sp, mut sn, mut wp, mut wn) = (0f32, 0f32, 0f32, 0f32);
    for k in 0..6 {
        let s = vals[k] * w[k];
        if w[k] >= 0.0 {
            sp += s;
            wp += w[k];
        } else {
            sn += s;
            wn += w[k];
        }
    }
    if let Some(c) = clamp {
        if sp > 0.0 && -sn > c * sp {
            return sp / wp;
        }
    }
    (sp + sn) / (wp + wn)
}

/// Bilinear warp (for comparisons).
pub fn warp_bilinear(src: &Image, out_to_src: &Homography, w: usize, h: usize) -> Image {
    let (sw, sh) = (src.width as i64, src.height as i64);
    let mut out = vec![0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            let (sx, sy) = out_to_src.apply(x as f64, y as f64);
            if !(sx >= -0.5 && sy >= -0.5 && sx < sw as f64 - 0.5 && sy < sh as f64 - 0.5) {
                continue;
            }
            let (ix, iy) = (sx.floor() as i64, sy.floor() as i64);
            let (fx, fy) = ((sx - ix as f64) as f32, (sy - iy as f64) as f32);
            let g = |xx: i64, yy: i64| src.data[(yy.clamp(0, sh - 1) * sw + xx.clamp(0, sw - 1)) as usize];
            let a = g(ix, iy) * (1.0 - fx) + g(ix + 1, iy) * fx;
            let b = g(ix, iy + 1) * (1.0 - fx) + g(ix + 1, iy + 1) * fx;
            *o = a * (1.0 - fy) + b * fy;
        }
    });
    Image::from_data(w, h, out)
}

/// Warp `src` onto a `w`×`h` grid. `out_to_src` maps output pixels to
/// source pixels. Pixels mapped outside the source's pixel centres become
/// 0 (no data), as in StarAlignment.
pub fn warp(src: &Image, out_to_src: &Homography, w: usize, h: usize, clamp: Option<f32>) -> Image {
    // On the GPU the homography is evaluated in f64 every 8 px and
    // interpolated: its perspective terms bend it by ~1e-6 px over that.
    #[cfg(feature = "gpu")]
    if let Some(v) = on_gpu(src, w, h, 8, &|x, y| Some(out_to_src.apply(x, y)), false, clamp) {
        let mut img = Image::from_data(w, h, v);
        img.adu_scale = src.adu_scale;
        return img;
    }
    warp_cpu(src, out_to_src, w, h, clamp)
}

/// `warp` on the CPU.
pub fn warp_cpu(src: &Image, out_to_src: &Homography, w: usize, h: usize, clamp: Option<f32>) -> Image {
    let lut = lanczos_lut();
    let (sw, sh) = (src.width as i64, src.height as i64);
    let mut out = vec![0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let mut vals = [0f32; 6];
        let mut col = [0f32; 6];
        let mut wx = [0f32; 6];
        let mut wy = [0f32; 6];
        for (x, o) in row.iter_mut().enumerate() {
            let (sx, sy) = out_to_src.apply(x as f64, y as f64);
            if !(sx >= 0.0 && sy >= 0.0 && sx <= (sw - 1) as f64 && sy <= (sh - 1) as f64) {
                *o = 0.0;
                continue;
            }
            let (ix, iy) = (sx.floor() as i64, sy.floor() as i64);
            let (fx, fy) = ((sx - ix as f64) as f32, (sy - iy as f64) as f32);
            for k in 0..6 {
                wx[k] = lw(lut, fx - (k as f32 - 2.0));
                wy[k] = lw(lut, fy - (k as f32 - 2.0));
            }
            for (j, c) in col.iter_mut().enumerate() {
                let yy = (iy + j as i64 - 2).clamp(0, sh - 1) as usize;
                let r = &src.data[yy * src.width..(yy + 1) * src.width];
                for (k, v) in vals.iter_mut().enumerate() {
                    *v = r[(ix + k as i64 - 2).clamp(0, sw - 1) as usize];
                }
                *c = sum1(&vals, &wx, clamp);
            }
            *o = sum1(&col, &wy, clamp);
        }
    });
    let mut img = Image::from_data(w, h, out);
    img.adu_scale = src.adu_scale;
    img
}

/// Lanczos-3 warp through any smooth mapping (output pixel → source pixel,
/// `None` where it doesn't exist). The mapping is evaluated every `step`
/// pixels and interpolated bilinearly in between (sky projections and lens
/// distortion bend by far less than a pixel over 16 px). Output pixels
/// whose nearest four source pixels aren't all non-zero are NaN.
pub fn warp_with(src: &Image, w: usize, h: usize, step: usize, map: &(dyn Fn(f64, f64) -> Option<(f64, f64)> + Sync)) -> Vec<f32> {
    #[cfg(feature = "gpu")]
    if let Some(v) = on_gpu(src, w, h, step.max(1), map, true, None) {
        return v;
    }
    warp_with_cpu(src, w, h, step, map)
}

/// `warp_with` on the CPU.
pub fn warp_with_cpu(src: &Image, w: usize, h: usize, step: usize, map: &(dyn Fn(f64, f64) -> Option<(f64, f64)> + Sync)) -> Vec<f32> {
    let lut = lanczos_lut();
    let step = step.max(1);
    let (gw, gh) = (w.div_ceil(step) + 1, h.div_ceil(step) + 1);
    let grid: Vec<Option<(f64, f64)>> = (0..gw * gh).into_par_iter().map(|i| map(((i % gw) * step) as f64, ((i / gw) * step) as f64)).collect();
    let (sw, sh) = (src.width as i64, src.height as i64);
    let mut out = vec![f32::NAN; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let (gy, fy) = (y / step, (y % step) as f64 / step as f64);
        let mut vals = [0f32; 6];
        let mut col = [0f32; 6];
        let mut wx = [0f32; 6];
        let mut wy = [0f32; 6];
        for (x, o) in row.iter_mut().enumerate() {
            let (gx, fx) = (x / step, (x % step) as f64 / step as f64);
            let (Some(a), Some(b), Some(c), Some(d)) = (grid[gy * gw + gx], grid[gy * gw + gx + 1], grid[(gy + 1) * gw + gx], grid[(gy + 1) * gw + gx + 1]) else {
                continue;
            };
            let sx = (a.0 * (1.0 - fx) + b.0 * fx) * (1.0 - fy) + (c.0 * (1.0 - fx) + d.0 * fx) * fy;
            let sy = (a.1 * (1.0 - fx) + b.1 * fx) * (1.0 - fy) + (c.1 * (1.0 - fx) + d.1 * fx) * fy;
            if !(sx >= 0.0 && sy >= 0.0 && sx <= (sw - 1) as f64 && sy <= (sh - 1) as f64) {
                continue;
            }
            let (ix, iy) = (sx.floor() as i64, sy.floor() as i64);
            let near = |xx: i64, yy: i64| src.data[(yy.min(sh - 1) * sw + xx.min(sw - 1)) as usize];
            if near(ix, iy) == 0.0 || near(ix + 1, iy) == 0.0 || near(ix, iy + 1) == 0.0 || near(ix + 1, iy + 1) == 0.0 {
                continue;
            }
            let (fx, fy) = ((sx - ix as f64) as f32, (sy - iy as f64) as f32);
            for k in 0..6 {
                wx[k] = lw(lut, fx - (k as f32 - 2.0));
                wy[k] = lw(lut, fy - (k as f32 - 2.0));
            }
            for (j, c) in col.iter_mut().enumerate() {
                let yy = (iy + j as i64 - 2).clamp(0, sh - 1) as usize;
                let r = &src.data[yy * src.width..(yy + 1) * src.width];
                for (k, v) in vals.iter_mut().enumerate() {
                    *v = r[(ix + k as i64 - 2).clamp(0, sw - 1) as usize];
                }
                *c = sum1(&vals, &wx, None);
            }
            *o = sum1(&col, &wy, None);
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_and_compose() {
        let h = Homography([1.001, 0.002, 5.0, -0.0015, 0.999, -3.0, 1e-8, -2e-8, 1.0]);
        let i = h.inverse().unwrap();
        let (x, y) = h.apply(1000.0, 2000.0);
        let (bx, by) = i.apply(x, y);
        assert!((bx - 1000.0).abs() < 1e-6 && (by - 2000.0).abs() < 1e-6);
        let id = i.then_after(&h);
        for (a, b) in id.0.iter().zip(Homography::identity().0) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    /// A star field with an empty corner, warped both ways on the GPU and
    /// the CPU: same pixels.
    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_warps_match_the_cpu() {
        let Some(gpu) = crate::gpu::get() else {
            eprintln!("no GPU here: skipped");
            return;
        };
        let (w, h) = (900usize, 640usize);
        let mut data: Vec<f32> = (0..w * h).map(|i| 0.02 + 0.01 * ((i % w) as f32 * 0.05).sin() * ((i / w) as f32 * 0.03).cos()).collect();
        for k in 0..400 {
            let (sx, sy) = ((k * 7919 % w) as f32, (k * 104729 % h) as f32);
            let a = 0.05 + (k % 13) as f32 * 0.07;
            for y in (sy as usize).saturating_sub(6)..(sy as usize + 7).min(h) {
                for x in (sx as usize).saturating_sub(6)..(sx as usize + 7).min(w) {
                    data[y * w + x] += a * (-((x as f32 - sx).powi(2) + (y as f32 - sy).powi(2)) / 3.2).exp();
                }
            }
        }
        for y in 0..80 {
            for x in 0..120 {
                data[y * w + x] = 0.0;
            }
        }
        let src = Image::from_data(w, h, data);
        let hm = Homography([0.9993, 0.0371, 12.37, -0.0369, 1.0004, -7.81, 2e-7, -1e-7, 1.0]);
        for clamp in [None, Some(0.15)] {
            let (gw, nodes) = crate::gpu::warp::grid(w, h, 8, &|x, y| Some(hm.apply(x, y)));
            let g = crate::gpu::warp::run(gpu, &src, w, h, 8, gw, &nodes, false, clamp).unwrap().unwrap();
            let c = warp_cpu(&src, &hm, w, h, clamp);
            let worst = g.iter().zip(&c.data).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            // Clamping switches at a threshold: a pixel right on it can go
            // either way after rounding.
            let off = g.iter().zip(&c.data).filter(|(a, b)| (*a - *b).abs() > 1e-5).count();
            match clamp {
                None => assert!(worst < 1e-5, "homography: {worst:e}"),
                Some(_) => assert!(off * 10_000 <= w * h && worst < 1e-3, "homography, clamped: {off} pixels differ, by up to {worst:e}"),
            }
        }
        // A bent mapping with a hole and far-off coordinates.
        let map = |x: f64, y: f64| -> Option<(f64, f64)> {
            if x > 700.0 && y > 500.0 {
                return None;
            }
            Some((x * 0.98 + 0.00002 * y * y + 3.3, y * 1.01 - 0.00003 * x * x + 1.7))
        };
        let (gw, nodes) = crate::gpu::warp::grid(w, h, 16, &map);
        let g = crate::gpu::warp::run(gpu, &src, w, h, 16, gw, &nodes, true, None).unwrap().unwrap();
        let c = warp_with_cpu(&src, w, h, 16, &map);
        let mut worst = 0f32;
        for (a, b) in g.iter().zip(&c) {
            assert_eq!(a.is_nan(), b.is_nan());
            if !a.is_nan() {
                worst = worst.max((a - b).abs());
            }
        }
        assert!(worst < 1e-5, "mapping: {worst:e}");
        assert!(c.iter().filter(|v| v.is_nan()).count() > 10_000);
    }

    #[test]
    fn shift_is_exact_on_integers_and_smooth_otherwise() {
        let (w, h) = (64, 48);
        let data: Vec<f32> = (0..w * h).map(|i| 0.2 + 0.1 * ((i % w) as f32 * 0.3).sin() * ((i / w) as f32 * 0.2).cos()).collect();
        let src = Image::from_data(w, h, data);
        // Integer shift: exact copy.
        let t = Homography([1.0, 0.0, 3.0, 0.0, 1.0, 2.0, 0.0, 0.0, 1.0]);
        let out = warp(&src, &t, w, h, Some(0.15));
        assert!((out.get(10, 10) - src.get(13, 12)).abs() < 1e-6);
        assert_eq!(out.get(w - 1, 10), 0.0);
        // Half-pixel shift of a smooth signal: close to the true value.
        let t = Homography([1.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let out = warp(&src, &t, w, h, Some(0.15));
        let truth = 0.2 + 0.1 * (20.5f32 * 0.3).sin() * (20.0f32 * 0.2).cos();
        assert!((out.get(20, 20) - truth).abs() < 2e-3, "{} {}", out.get(20, 20), truth);
    }
}
