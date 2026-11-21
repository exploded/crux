//! Robust statistics shared by every check.

/// MAD → Gaussian sigma.
pub const MAD_TO_SIGMA: f64 = 1.4826;

/// Median by selection; reorders `v`. NaN-free input assumed.
pub fn median_mut(v: &mut [f32]) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    let n = v.len();
    let (_, m, _) = v.select_nth_unstable_by(n / 2, |a, b| a.total_cmp(b));
    *m
}

pub fn median(v: &[f32]) -> f32 {
    median_mut(&mut v.to_vec())
}

/// Median and MAD-sigma; reorders `v`.
pub fn median_sigma_mut(v: &mut [f32]) -> (f32, f32) {
    let m = median_mut(v);
    for x in v.iter_mut() {
        *x = (*x - m).abs();
    }
    let mad = median_mut(v);
    (m, mad * MAD_TO_SIGMA as f32)
}

/// Iterative sigma-clipped median and sigma (MAD based). Reorders `v`.
pub fn clipped_median_sigma(v: &mut Vec<f32>, k: f32, iters: usize) -> (f32, f32) {
    let mut work: Vec<f32> = Vec::with_capacity(v.len());
    let (mut m, mut s) = (f32::NAN, f32::NAN);
    for _ in 0..iters {
        if v.len() < 3 {
            break;
        }
        work.clear();
        work.extend_from_slice(v);
        let (nm, ns) = median_sigma_mut(&mut work);
        let converged = nm == m && ns == s;
        m = nm;
        s = ns;
        if converged || s <= 0.0 {
            break;
        }
        let (lo, hi) = (m - k * s, m + k * s);
        let before = v.len();
        v.retain(|&x| x >= lo && x <= hi);
        if v.len() == before {
            break;
        }
    }
    if m.is_nan() && !v.is_empty() {
        work.clear();
        work.extend_from_slice(v);
        let r = median_sigma_mut(&mut work);
        m = r.0;
        s = r.1;
    }
    (m, s)
}

pub fn median_f64(v: &[f64]) -> Option<f64> {
    let mut w: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if w.is_empty() {
        return None;
    }
    w.sort_by(|a, b| a.total_cmp(b));
    let n = w.len();
    Some(if n % 2 == 1 { w[n / 2] } else { 0.5 * (w[n / 2 - 1] + w[n / 2]) })
}

/// Median and MAD-sigma of finite values.
pub fn median_sigma_f64(v: &[f64]) -> Option<(f64, f64)> {
    let m = median_f64(v)?;
    let dev: Vec<f64> = v.iter().filter(|x| x.is_finite()).map(|x| (x - m).abs()).collect();
    Some((m, median_f64(&dev)? * MAD_TO_SIGMA))
}

/// p in [0,1], linear interpolation between order statistics.
pub fn percentile_f64(v: &[f64], p: f64) -> Option<f64> {
    let mut w: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if w.is_empty() {
        return None;
    }
    w.sort_by(|a, b| a.total_cmp(b));
    let pos = p.clamp(0.0, 1.0) * (w.len() - 1) as f64;
    let (i, f) = (pos.floor() as usize, pos.fract());
    Some(if i + 1 < w.len() { w[i] * (1.0 - f) + w[i + 1] * f } else { w[i] })
}

/// z = a + b·x + c·y.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Plane {
    pub a: f64,
    pub b: f64,
    pub c: f64,
}

impl Plane {
    pub fn at(&self, x: f64, y: f64) -> f64 {
        self.a + self.b * x + self.c * y
    }
}

/// Least-squares plane through (x, y, z), refit after clipping residuals
/// beyond `k` sigma (nebulae, bright stars).
pub fn fit_plane_robust(pts: &[(f64, f64, f64)], k: f64, iters: usize) -> Option<Plane> {
    let mut keep: Vec<bool> = vec![true; pts.len()];
    let mut plane = None;
    for _ in 0..iters.max(1) {
        let p = fit_plane(pts.iter().zip(&keep).filter(|(_, &k)| k).map(|(p, _)| *p))?;
        plane = Some(p);
        let res: Vec<f64> = pts.iter().map(|&(x, y, z)| z - p.at(x, y)).collect();
        let kept: Vec<f64> = res.iter().zip(&keep).filter(|(_, &k)| k).map(|(r, _)| *r).collect();
        let (_, s) = median_sigma_f64(&kept)?;
        if s <= 0.0 {
            break;
        }
        let new: Vec<bool> = res.iter().map(|r| r.abs() <= k * s).collect();
        if new == keep || new.iter().filter(|&&b| b).count() < 3 {
            break;
        }
        keep = new;
    }
    plane
}

fn fit_plane(pts: impl Iterator<Item = (f64, f64, f64)>) -> Option<Plane> {
    // Normal equations, centred for numerical stability.
    let pts: Vec<_> = pts.collect();
    let n = pts.len() as f64;
    if pts.len() < 3 {
        return None;
    }
    let (mx, my, mz) = pts.iter().fold((0.0, 0.0, 0.0), |a, p| (a.0 + p.0 / n, a.1 + p.1 / n, a.2 + p.2 / n));
    let (mut sxx, mut syy, mut sxy, mut sxz, mut syz) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for &(x, y, z) in &pts {
        let (dx, dy, dz) = (x - mx, y - my, z - mz);
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
        sxz += dx * dz;
        syz += dy * dz;
    }
    let det = sxx * syy - sxy * sxy;
    if det.abs() < 1e-12 {
        return None;
    }
    let b = (sxz * syy - syz * sxy) / det;
    let c = (syz * sxx - sxz * sxy) / det;
    Some(Plane { a: mz - b * mx - c * my, b, c })
}

/// Robust z-score with a floor on sigma (as a fraction of the median) so a
/// very consistent group doesn't turn tiny differences into huge z.
pub fn robust_z(x: f64, median: f64, sigma: f64, floor_frac: f64) -> f64 {
    let s = sigma.max(median.abs() * floor_frac).max(1e-12);
    (x - median) / s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_and_clip() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        let mut v: Vec<f32> = (0..1000).map(|i| (i % 10) as f32).collect();
        v.extend([1000.0; 50]);
        let (m, s) = clipped_median_sigma(&mut v, 3.0, 5);
        assert!((m - 4.5).abs() < 1.0, "{m}");
        assert!(s < 5.0, "{s}");
    }

    #[test]
    fn plane_with_outliers() {
        let mut pts = Vec::new();
        for y in 0..20 {
            for x in 0..20 {
                let (x, y) = (x as f64, y as f64);
                let z = if x > 15.0 && y > 15.0 { 1000.0 } else { 10.0 + 2.0 * x - 0.5 * y };
                pts.push((x, y, z));
            }
        }
        let p = fit_plane_robust(&pts, 3.0, 5).unwrap();
        assert!((p.b - 2.0).abs() < 1e-6 && (p.c + 0.5).abs() < 1e-6 && (p.a - 10.0).abs() < 1e-6, "{p:?}");
    }

    #[test]
    fn percentiles() {
        let v: Vec<f64> = (0..=10).map(f64::from).collect();
        assert_eq!(percentile_f64(&v, 0.1), Some(1.0));
        assert_eq!(median_f64(&[1.0, 2.0, 3.0, 4.0]), Some(2.5));
    }
}
