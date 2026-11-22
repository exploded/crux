//! Gradient removal on a linear master, like PixInsight's
//! AutomaticBackgroundExtractor: sample boxes on a grid, reject the ones
//! on stars, nebula or dust, fit a low-order polynomial surface to the
//! rest, and subtract it (or divide by it), keeping the background level.
//!
//! The fit always runs on the same binned copy of the full image, so the
//! preview and the full-size output use the same model.

use crate::image::Image;
use serde::{Deserialize, Serialize};

/// Longest side of the binned copy the boxes are measured on.
const FIT_SIZE: usize = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Correction {
    /// Additive gradients (light pollution, moonlight, airglow).
    #[default]
    Subtract,
    /// Multiplicative ones (vignetting a flat didn't remove).
    Divide,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GradientSettings {
    pub enabled: bool,
    /// Polynomial degree 1–4, or 0 for auto: per channel, the lowest
    /// degree a higher one doesn't improve by 15%, capped at 2; then one
    /// degree for every channel of a target (the median), so the channels
    /// don't bend differently at the edges and tint the corners.
    pub degree: u32,
    /// Sample boxes along the longer side.
    pub samples: usize,
    /// Boxes this many residual sigmas above the surface are rejected as
    /// nebula or stars (ABE's tolerance). Lower protects faint nebulosity
    /// more, at the cost of a less complete fit.
    pub tolerance: f32,
    pub correction: Correction,
    /// In sparse fields (galaxies, clusters, star fields: nearly every box
    /// is sky), also follow what the polynomial leaves (vignetting a flat
    /// didn't remove, rings, patches) with a smooth mesh through the sky
    /// boxes. Nebula-filled frames never get it.
    pub follow_sky: bool,
}

impl Default for GradientSettings {
    fn default() -> Self {
        GradientSettings { enabled: true, degree: 0, samples: 20, tolerance: 1.0, correction: Correction::Subtract, follow_sky: false }
    }
}

/// A smooth correction on top of the polynomial: values on an `nx` × `ny`
/// grid over normalised coordinates, interpolated bilinearly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Mesh {
    pub nx: usize,
    pub ny: usize,
    pub vals: Vec<f64>,
}

impl Mesh {
    fn at(&self, u: f64, v: f64) -> f64 {
        let gx = ((u + 1.0) / 2.0 * (self.nx - 1) as f64).clamp(0.0, (self.nx - 1) as f64);
        let gy = ((v + 1.0) / 2.0 * (self.ny - 1) as f64).clamp(0.0, (self.ny - 1) as f64);
        let (x0, y0) = (gx as usize, gy as usize);
        let (x1, y1) = ((x0 + 1).min(self.nx - 1), (y0 + 1).min(self.ny - 1));
        let (fx, fy) = (gx - x0 as f64, gy - y0 as f64);
        let g = |x: usize, y: usize| self.vals[y * self.nx + x];
        (g(x0, y0) * (1.0 - fx) + g(x1, y0) * fx) * (1.0 - fy) + (g(x0, y1) * (1.0 - fx) + g(x1, y1) * fx) * fy
    }
}

/// Share of the usable boxes the polynomial must keep for a field to count
/// as sparse (sky nearly everywhere).
const SPARSE_KEPT: f64 = 0.0;
/// Mesh nodes along the longer side, and the smoothing in node spacings.
const MESH_NODES: usize = 16;
const MESH_SIGMA: f64 = 0.8;
/// Boxes this many σ brighter than the median of their neighbours (within
/// this many node spacings) are objects, not sky.
const LOCAL_BRIGHT: f64 = 2.0;
const LOCAL_REACH: f64 = 2.5;

/// A fitted background surface over normalised coordinates (−1…1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Model {
    pub degree: u32,
    pub coef: Vec<f64>,
    pub width: usize,
    pub height: usize,
    /// The level the corrected image keeps (the surface's median).
    pub level: f64,
    pub correction: Correction,
    /// Sparse fields: what the polynomial left, through the sky boxes.
    #[serde(default)]
    pub mesh: Option<Mesh>,
}

/// What the fit found, for the stage card.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct GradientReport {
    pub degree: u32,
    pub boxes: usize,
    pub kept: usize,
    /// Surface peak-to-peak over the frame, in the image's units.
    pub amplitude: f64,
    /// … as a fraction of the background level.
    pub amplitude_frac: f64,
    /// Residual scatter of the kept boxes about the surface ÷ the surface's
    /// level: how flat the background is after correction.
    pub residual_frac: f64,
    /// Sample boxes (fractions of the frame) and whether each was kept.
    pub samples: Vec<(f32, f32, bool)>,
}

fn n_terms(d: u32) -> usize {
    ((d + 1) * (d + 2) / 2) as usize
}

fn terms(u: f64, v: f64, d: u32, out: &mut [f64]) {
    let mut k = 0;
    for t in 0..=d {
        for i in 0..=t {
            out[k] = u.powi((t - i) as i32) * v.powi(i as i32);
            k += 1;
        }
    }
}

impl Model {
    #[inline]
    fn uv(&self, x: f64, y: f64) -> (f64, f64) {
        (2.0 * x / self.width.max(1) as f64 - 1.0, 2.0 * y / self.height.max(1) as f64 - 1.0)
    }

    /// The surface at pixel (x, y).
    pub fn at(&self, x: f64, y: f64) -> f64 {
        let (u, v) = self.uv(x, y);
        let mut t = [0f64; 15];
        terms(u, v, self.degree, &mut t);
        let p: f64 = self.coef.iter().zip(&t).map(|(c, t)| c * t).sum();
        p + self.mesh.as_ref().map_or(0.0, |m| m.at(u, v))
    }

    /// Apply in place. Zero pixels (outside a registered frame) stay zero.
    pub fn apply(&self, img: &mut Image) {
        use rayon::prelude::*;
        let w = img.width;
        let lvl = self.level;
        img.data.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            // The surface is smooth: evaluate every 16 px and interpolate.
            let step = 16usize;
            let knots: Vec<f64> = (0..=w.div_ceil(step)).map(|i| self.at((i * step).min(w) as f64 + 0.5, y as f64 + 0.5)).collect();
            for (x, v) in row.iter_mut().enumerate() {
                if *v == 0.0 {
                    continue;
                }
                let (i, f) = (x / step, (x % step) as f64 / step as f64);
                let m = knots[i] + (knots[(i + 1).min(knots.len() - 1)] - knots[i]) * f;
                *v = match self.correction {
                    Correction::Subtract => (*v as f64 - m + lvl) as f32,
                    Correction::Divide => {
                        if m > 1e-9 {
                            (*v as f64 * lvl / m) as f32
                        } else {
                            *v
                        }
                    }
                };
            }
        });
    }

    /// The surface as an image of `w` × `h` (for showing the model).
    pub fn render(&self, w: usize, h: usize) -> Image {
        let mut img = Image::new(w, h);
        for y in 0..h {
            for x in 0..w {
                img.data[y * w + x] = self.at((x as f64 + 0.5) * self.width as f64 / w as f64, (y as f64 + 0.5) * self.height as f64 / h as f64) as f32;
            }
        }
        img
    }
}

/// Mean of non-zero pixels in `f` × `f` blocks; blocks under half covered
/// are NaN.
pub fn bin_valid(img: &Image, f: usize) -> (usize, usize, Vec<f32>) {
    let f = f.max(1);
    let (bw, bh) = (img.width / f, img.height / f);
    let mut out = vec![f32::NAN; bw * bh];
    for by in 0..bh {
        for bx in 0..bw {
            let (mut s, mut n) = (0f64, 0usize);
            for y in by * f..(by + 1) * f {
                for &v in &img.row(y)[bx * f..(bx + 1) * f] {
                    if v != 0.0 && v.is_finite() {
                        s += v as f64;
                        n += 1;
                    }
                }
            }
            if n * 2 >= f * f {
                out[by * bw + bx] = (s / n as f64) as f32;
            }
        }
    }
    (bw, bh, out)
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let m = v.len() / 2;
    v.select_nth_unstable_by(m, |a, b| a.total_cmp(b));
    v[m]
}

/// Least squares by normal equations with partial pivoting.
fn solve(a: &mut [Vec<f64>], b: &mut [f64]) -> Option<Vec<f64>> {
    let n = b.len();
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))?;
        if a[p][c].abs() < 1e-14 {
            return None;
        }
        a.swap(c, p);
        b.swap(c, p);
        for r in c + 1..n {
            let f = a[r][c] / a[c][c];
            for k in c..n {
                a[r][k] -= f * a[c][k];
            }
            b[r] -= f * b[c];
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let s: f64 = (r + 1..n).map(|k| a[r][k] * x[k]).sum();
        x[r] = (b[r] - s) / a[r][r];
    }
    Some(x)
}

fn fit_poly(pts: &[(f64, f64, f64)], d: u32) -> Option<Vec<f64>> {
    let n = n_terms(d);
    if pts.len() < n + 2 {
        return None;
    }
    let mut a = vec![vec![0.0; n]; n];
    let mut b = vec![0.0; n];
    let mut t = [0f64; 15];
    for &(u, v, z) in pts {
        terms(u, v, d, &mut t);
        for i in 0..n {
            b[i] += t[i] * z;
            for j in 0..n {
                a[i][j] += t[i] * t[j];
            }
        }
    }
    solve(&mut a, &mut b)
}

fn eval(c: &[f64], d: u32, u: f64, v: f64) -> f64 {
    let mut t = [0f64; 15];
    terms(u, v, d, &mut t);
    c.iter().zip(&t).map(|(c, t)| c * t).sum()
}

/// Sample boxes measured on one image, reusable for fits of any degree.
#[derive(Debug, Clone)]
pub struct Samples {
    /// (u, v, value) of every usable box.
    pts: Vec<(f64, f64, f64)>,
    total: usize,
    width: usize,
    height: usize,
}

/// Highest degree auto picks.
pub const AUTO_MAX: u32 = 2;

/// Measure the sample boxes of a linear image.
pub fn samples(img: &Image, n: usize) -> Samples {
    let b = boxes(img, n);
    Samples { pts: b.0, total: b.1, width: img.width, height: img.height }
}

fn boxes(img: &Image, samples: usize) -> (Vec<(f64, f64, f64)>, usize) {
    let f = img.width.max(img.height).div_ceil(FIT_SIZE).max(1);
    let (bw, bh, b) = bin_valid(img, f);
    let n_long = samples.clamp(4, 64);
    let size = (bw.max(bh) as f64 / n_long as f64).max(2.0);
    let (nx, ny) = (((bw as f64) / size).round().max(2.0) as usize, ((bh as f64) / size).round().max(2.0) as usize);
    let mut cand = Vec::new();
    let mut scatter = Vec::new();
    for j in 0..ny {
        for i in 0..nx {
            let (x0, x1) = (i * bw / nx, (i + 1) * bw / nx);
            let (y0, y1) = (j * bh / ny, (j + 1) * bh / ny);
            let mut v: Vec<f64> = Vec::with_capacity((x1 - x0) * (y1 - y0));
            for y in y0..y1 {
                for x in x0..x1 {
                    let p = b[y * bw + x];
                    if p.is_finite() {
                        v.push(p as f64);
                    }
                }
            }
            if v.len() * 10 < (x1 - x0) * (y1 - y0) * 7 {
                continue;
            }
            let m = median(&mut v);
            let mut dev: Vec<f64> = v.iter().map(|x| (x - m).abs()).collect();
            let mad = median(&mut dev);
            let (cx, cy) = ((x0 + x1) as f64 / 2.0, (y0 + y1) as f64 / 2.0);
            cand.push((2.0 * cx / bw as f64 - 1.0, 2.0 * cy / bh as f64 - 1.0, m, mad));
            scatter.push(mad);
        }
    }
    let total = nx * ny;
    // Boxes far rougher than typical hold stars or bright structure.
    let typical = median(&mut scatter.clone());
    let pts = cand.into_iter().filter(|c| typical <= 0.0 || c.3 <= 3.0 * typical).map(|c| (c.0, c.1, c.2)).collect();
    (pts, total)
}

/// The degree auto picks for one channel.
pub fn auto_degree(s: &Samples, tolerance: f32) -> u32 {
    let tol = tolerance.max(0.1) as f64;
    let mut best: Option<(u32, f64)> = None;
    for d in 1..=AUTO_MAX {
        let Some((_, _, sg)) = fit_degree(&s.pts, d, tol) else { break };
        match best {
            Some((_, b)) if sg > 0.85 * b => break,
            _ => best = Some((d, sg)),
        }
    }
    best.map(|b| b.0).unwrap_or(1)
}

/// Fit at one degree with iterative rejection; returns (coef, kept flags,
/// residual sigma).
fn fit_degree(pts: &[(f64, f64, f64)], d: u32, tol: f64) -> Option<(Vec<f64>, Vec<bool>, f64)> {
    let mut keep = vec![true; pts.len()];
    let mut coef = None;
    let mut sigma = 0.0;
    for _ in 0..8 {
        let sel: Vec<(f64, f64, f64)> = pts.iter().zip(&keep).filter(|(_, k)| **k).map(|(p, _)| *p).collect();
        let c = fit_poly(&sel, d)?;
        let mut res: Vec<f64> = sel.iter().map(|&(u, v, z)| z - eval(&c, d, u, v)).collect();
        let mut abs: Vec<f64> = res.iter_mut().map(|r| r.abs()).collect();
        sigma = (1.4826 * median(&mut abs)).max(1e-12);
        // Every box is reconsidered against the new surface.
        let next: Vec<bool> = pts.iter().map(|&(u, v, z)| {
            let r = z - eval(&c, d, u, v);
            r <= tol * sigma && r >= -3.0 * sigma
        }).collect();
        coef = Some(c);
        if next == keep || next.iter().filter(|k| **k).count() < n_terms(d) + 2 {
            break;
        }
        keep = next;
    }
    coef.map(|c| (c, keep, sigma))
}

/// Fit the background surface of a linear image (auto degree per image).
pub fn fit(img: &Image, s: &GradientSettings) -> Option<(Model, GradientReport)> {
    let smp = samples(img, s.samples);
    let d = if s.degree == 0 { auto_degree(&smp, s.tolerance) } else { s.degree };
    fit_samples(&smp, s, d)
}

/// Fit at a given degree from measured samples.
pub fn fit_samples(b: &Samples, s: &GradientSettings, degree: u32) -> Option<(Model, GradientReport)> {
    let tol = s.tolerance.max(0.1) as f64;
    let d = degree.clamp(1, 4);
    let (coef, keep, sigma) = fit_degree(&b.pts, d, tol)?;
    let (width, height) = (b.width, b.height);
    // Level: the surface's median over the frame.
    let mut grid: Vec<f64> = Vec::with_capacity(32 * 32);
    for j in 0..32 {
        for i in 0..32 {
            grid.push(eval(&coef, d, (i as f64 + 0.5) / 16.0 - 1.0, (j as f64 + 0.5) / 16.0 - 1.0));
        }
    }
    let mesh = (s.follow_sky && keep.iter().filter(|k| **k).count() as f64 >= SPARSE_KEPT * b.pts.len() as f64).then(|| mesh_of(&b.pts, &keep, &coef, d, b.width, b.height));
    if let Some(m) = &mesh {
        for (k, g) in grid.iter_mut().enumerate() {
            *g += m.at(((k % 32) as f64 + 0.5) / 16.0 - 1.0, ((k / 32) as f64 + 0.5) / 16.0 - 1.0);
        }
    }
    let (lo, hi) = grid.iter().fold((f64::MAX, f64::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    let level = median(&mut grid);
    let model = Model { degree: d, coef, width, height, level, correction: s.correction, mesh };
    let lvl = level.abs().max(1e-12);
    let report = GradientReport {
        degree: d,
        boxes: b.total,
        kept: keep.iter().filter(|k| **k).count(),
        amplitude: hi - lo,
        amplitude_frac: (hi - lo) / lvl,
        residual_frac: sigma / lvl,
        samples: b.pts.iter().zip(&keep).map(|(p, k)| (((p.0 + 1.0) / 2.0) as f32, ((p.1 + 1.0) / 2.0) as f32, *k)).collect(),
    };
    Some((model, report))
}

/// The mesh through the boxes' residuals from the polynomial. Every
/// usable box takes part except those standing out from their neighbours
/// (galaxies, clusters: brighter than the boxes around them), so smooth
/// residue the polynomial can't follow (rings, patches from flats) is
/// followed. Gaussian-weighted means at each node, fading to nothing far
/// from any box.
fn mesh_of(pts: &[(f64, f64, f64)], _keep: &[bool], coef: &[f64], d: u32, width: usize, height: usize) -> Mesh {
    let long = width.max(height).max(1) as f64;
    let (ax, ay) = (width as f64 / long, height as f64 / long);
    let nx = ((MESH_NODES as f64 * ax).round() as usize).max(3);
    let ny = ((MESH_NODES as f64 * ay).round() as usize).max(3);
    let res: Vec<(f64, f64, f64)> = pts.iter().map(|&(u, v, z)| (u, v, z - eval(coef, d, u, v))).collect();
    // Distances in units of the frame's long side.
    let dist2 = |a: &(f64, f64, f64), b: &(f64, f64, f64)| ((a.0 - b.0) * ax).powi(2) + ((a.1 - b.1) * ay).powi(2);
    let reach2 = (2.0 * LOCAL_REACH / MESH_NODES as f64).powi(2);
    let mut keep = vec![true; res.len()];
    for _ in 0..3 {
        let local: Vec<f64> = res
            .iter()
            .map(|p| {
                let mut v: Vec<f64> = res.iter().zip(&keep).filter(|(q, k)| **k && dist2(p, q) <= reach2).map(|(q, _)| q.2).collect();
                if v.is_empty() { p.2 } else { median(&mut v) }
            })
            .collect();
        let mut dev: Vec<f64> = res.iter().zip(&local).zip(&keep).filter(|(_, k)| **k).map(|((p, l), _)| (p.2 - l).abs()).collect();
        let sigma = 1.4826 * median(&mut dev).max(1e-12);
        let next: Vec<bool> = res.iter().zip(&local).map(|(p, l)| p.2 - l <= LOCAL_BRIGHT * sigma).collect();
        if next == keep {
            break;
        }
        keep = next;
    }
    let (su, sv) = (MESH_SIGMA * 2.0 / (nx - 1) as f64, MESH_SIGMA * 2.0 / (ny - 1) as f64);
    let mut vals = Vec::with_capacity(nx * ny);
    for j in 0..ny {
        for i in 0..nx {
            let (u, v) = (2.0 * i as f64 / (nx - 1) as f64 - 1.0, 2.0 * j as f64 / (ny - 1) as f64 - 1.0);
            let (mut sw, mut sz) = (0.0, 0.0);
            for (&(pu, pv, r), _) in res.iter().zip(&keep).filter(|(_, k)| **k) {
                let wgt = (-0.5 * (((pu - u) / su).powi(2) + ((pv - v) / sv).powi(2))).exp();
                sw += wgt;
                sz += wgt * r;
            }
            vals.push(if sw > 0.0 { sz / sw * sw.min(1.0) } else { 0.0 });
        }
    }
    Mesh { nx, ny, vals }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removes_a_known_gradient_and_keeps_nebula() {
        let (w, h) = (1200, 800);
        let mut img = Image::new(w, h);
        let mut seed = 1u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64 / (1u64 << 31) as f64) - 0.5
        };
        for y in 0..h {
            for x in 0..w {
                let (u, v) = (2.0 * x as f64 / w as f64 - 1.0, 2.0 * y as f64 / h as f64 - 1.0);
                let sky = 0.02 + 0.004 * u + 0.002 * v * v;
                // A bright blob of "nebula" the fit must ignore.
                let neb = 0.01 * (-((x as f64 - 300.0).powi(2) + (y as f64 - 300.0).powi(2)) / (2.0 * 90.0f64.powi(2))).exp();
                img.data[y * w + x] = (sky + neb + 0.0005 * rnd()) as f32;
            }
        }
        let (m, r) = fit(&img, &GradientSettings::default()).unwrap();
        assert_eq!(r.degree, 2);
        assert!(r.kept < r.boxes, "nebula boxes rejected");
        // The surface follows the sky, not the nebula.
        for (x, y) in [(100.0, 700.0), (1100.0, 100.0), (600.0, 400.0), (300.0, 300.0)] {
            let (u, v) = (2.0 * x / w as f64 - 1.0, 2.0 * y / h as f64 - 1.0);
            let sky = 0.02 + 0.004 * u + 0.002 * v * v;
            assert!((m.at(x, y) - sky).abs() < 3e-4, "at {x},{y}: {} vs {sky}", m.at(x, y));
        }
        m.apply(&mut img);
        // Corners now agree.
        let corner = |x0: usize, y0: usize| {
            let mut v: Vec<f64> = (y0..y0 + 50).flat_map(|y| (x0..x0 + 50).map(move |x| (x, y))).map(|(x, y)| img.data[y * w + x] as f64).collect();
            median(&mut v)
        };
        let c = [corner(1100, 700), corner(1100, 20), corner(600, 700), corner(900, 400)];
        let spread = c.iter().cloned().fold(f64::MIN, f64::max) - c.iter().cloned().fold(f64::MAX, f64::min);
        assert!(spread < 3e-4, "spread {spread}");
    }
}
