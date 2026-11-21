//! Background and noise from a mesh of sigma-clipped cells, as in
//! SExtractor: each cell's clipped median, a 3×3 median filter over the
//! mesh to drop cells swamped by stars, then bilinear interpolation.

use crate::image::Image;
use crate::stats::{clipped_median_sigma, fit_plane_robust, median, Plane};

#[derive(Debug, Clone)]
pub struct Background {
    pub cell: usize,
    pub nx: usize,
    pub ny: usize,
    pub width: usize,
    pub height: usize,
    /// Per-cell background level (normalised units), row-major.
    pub level: Vec<f32>,
    /// Per-cell noise sigma (normalised units).
    pub noise: Vec<f32>,
    /// Median of the cell levels.
    pub global_level: f32,
    /// Median of the cell noise.
    pub global_noise: f32,
}

impl Background {
    pub fn estimate(img: &Image, cell: usize) -> Background {
        let cell = cell.max(8);
        let (w, h) = (img.width, img.height);
        let nx = w.div_ceil(cell);
        let ny = h.div_ceil(cell);
        let mut level = vec![0f32; nx * ny];
        let mut noise = vec![0f32; nx * ny];
        // Sample sparsely: 64 px cells at step 3 still give ~480 pixels.
        let step = if cell >= 48 { 3 } else if cell >= 24 { 2 } else { 1 };
        let mut buf: Vec<f32> = Vec::with_capacity((cell / step + 1).pow(2));
        for cy in 0..ny {
            for cx in 0..nx {
                buf.clear();
                let (x0, y0) = (cx * cell, cy * cell);
                let (x1, y1) = ((x0 + cell).min(w), (y0 + cell).min(h));
                for y in (y0..y1).step_by(step) {
                    let row = &img.data[y * w..(y + 1) * w];
                    buf.extend(row[x0..x1].iter().step_by(step).copied());
                }
                let (m, s) = clipped_median_sigma(&mut buf, 3.0, 6);
                level[cy * nx + cx] = m;
                noise[cy * nx + cx] = s;
            }
        }
        let level = median_filter3(&level, nx, ny);
        let noise = median_filter3(&noise, nx, ny);
        let global_level = median(&level);
        let global_noise = median(&noise);
        Background { cell, nx, ny, width: w, height: h, level, noise, global_level, global_noise }
    }

    fn centre(&self, i: usize, n: usize, len: usize) -> f32 {
        // Cell centres; the last cell may be narrower.
        let start = i * self.cell;
        let end = ((i + 1) * self.cell).min(len);
        let _ = n;
        (start + end) as f32 * 0.5
    }

    /// Interpolated background at a pixel.
    pub fn at(&self, x: f32, y: f32) -> f32 {
        let (ix, fx) = self.locate(x, self.nx, self.width);
        let (iy, fy) = self.locate(y, self.ny, self.height);
        let g = |cx: usize, cy: usize| self.level[cy * self.nx + cx];
        let ix1 = (ix + 1).min(self.nx - 1);
        let iy1 = (iy + 1).min(self.ny - 1);
        let top = g(ix, iy) * (1.0 - fx) + g(ix1, iy) * fx;
        let bot = g(ix, iy1) * (1.0 - fx) + g(ix1, iy1) * fx;
        top * (1.0 - fy) + bot * fy
    }

    /// Cell index and fraction toward the next cell centre.
    fn locate(&self, v: f32, n: usize, len: usize) -> (usize, f32) {
        if n == 1 {
            return (0, 0.0);
        }
        let c = (v - self.cell as f32 * 0.5) / self.cell as f32;
        if c <= 0.0 {
            return (0, 0.0);
        }
        let i = (c.floor() as usize).min(n - 1);
        if i >= n - 1 {
            return (n - 1, 0.0);
        }
        let (c0, c1) = (self.centre(i, n, len), self.centre(i + 1, n, len));
        (i, ((v - c0) / (c1 - c0)).clamp(0.0, 1.0))
    }

    /// `img - background`, written into `out`.
    pub fn subtract(&self, img: &Image, out: &mut Vec<f32>) {
        let (w, h) = (img.width, img.height);
        out.resize(w * h, 0.0);
        // Interpolate along y once per row, then along x per pixel.
        let mut rowbg = vec![0f32; self.nx];
        let xs: Vec<(usize, f32)> = (0..w).map(|x| self.locate(x as f32 + 0.5, self.nx, w)).collect();
        for y in 0..h {
            let (iy, fy) = self.locate(y as f32 + 0.5, self.ny, h);
            let iy1 = (iy + 1).min(self.ny - 1);
            for (cx, r) in rowbg.iter_mut().enumerate() {
                *r = self.level[iy * self.nx + cx] * (1.0 - fy) + self.level[iy1 * self.nx + cx] * fy;
            }
            let src = &img.data[y * w..(y + 1) * w];
            let dst = &mut out[y * w..(y + 1) * w];
            for x in 0..w {
                let (ix, fx) = xs[x];
                let ix1 = (ix + 1).min(self.nx - 1);
                let b = rowbg[ix] * (1.0 - fx) + rowbg[ix1] * fx;
                dst[x] = src[x] - b;
            }
        }
    }

    /// Robust plane through the mesh (pixel coordinates).
    pub fn plane(&self) -> Option<Plane> {
        let mut pts = Vec::with_capacity(self.level.len());
        for cy in 0..self.ny {
            for cx in 0..self.nx {
                pts.push((
                    self.centre(cx, self.nx, self.width) as f64,
                    self.centre(cy, self.ny, self.height) as f64,
                    self.level[cy * self.nx + cx] as f64,
                ));
            }
        }
        fit_plane_robust(&pts, 3.0, 6)
    }
}

fn median_filter3(v: &[f32], nx: usize, ny: usize) -> Vec<f32> {
    let mut out = vec![0f32; v.len()];
    let mut win = Vec::with_capacity(9);
    for y in 0..ny {
        for x in 0..nx {
            win.clear();
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let (xx, yy) = (x as i32 + dx, y as i32 + dy);
                    if xx >= 0 && yy >= 0 && (xx as usize) < nx && (yy as usize) < ny {
                        win.push(v[yy as usize * nx + xx as usize]);
                    }
                }
            }
            out[y * nx + x] = median(&win);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_gradient_and_noise() {
        let (w, h) = (640, 480);
        let mut img = Image::new(w, h);
        let mut rng = crate::synth::Rng::new(7);
        for y in 0..h {
            for x in 0..w {
                img.data[y * w + x] = 0.01 + 0.002 * x as f32 / w as f32 + 0.0005 * rng.gauss() as f32;
            }
        }
        let bg = Background::estimate(&img, 64);
        assert!((bg.global_noise - 0.0005).abs() < 0.00005, "{}", bg.global_noise);
        assert!((bg.at(320.0, 240.0) - 0.011).abs() < 0.0001);
        let p = bg.plane().unwrap();
        assert!((p.b * w as f64 - 0.002).abs() < 0.0001, "{p:?}");
        let mut out = Vec::new();
        bg.subtract(&img, &mut out);
        let m = median(&out);
        assert!(m.abs() < 0.0001);
    }
}
