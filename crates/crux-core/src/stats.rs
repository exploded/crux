//! Robust statistics shared by every check.

/// MAD → Gaussian sigma.
pub const MAD_TO_SIGMA: f64 = 1.4826;

/// Median by selection; may reorder `v`. NaN-free input assumed.
pub fn median_mut(v: &mut [f32]) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    let n = v.len();
    if n >= PAR_MIN {
        return kth_par(v, n / 2);
    }
    let (_, m, _) = v.select_nth_unstable_by(n / 2, |a, b| a.total_cmp(b));
    *m
}

/// From this many values, selections run in parallel ([`kth_par`]).
const PAR_MIN: usize = 1 << 20;

/// `x`'s place in `total_cmp` order as an unsigned integer.
#[inline]
fn order_key(x: f32) -> u32 {
    let b = x.to_bits();
    if b & 0x8000_0000 != 0 {
        !b
    } else {
        b | 0x8000_0000
    }
}

#[inline]
fn from_order_key(k: u32) -> f32 {
    f32::from_bits(if k & 0x8000_0000 != 0 { k & 0x7fff_ffff } else { !k })
}

/// The `k`-th smallest value (from 0) in `total_cmp` order: exactly what
/// `select_nth_unstable_by(k, total_cmp)` leaves at `k`, found in parallel
/// from two 16-bit histograms of the values' bits, without reordering `v`.
/// A serial selection over a 176 MP mosaic took about a second.
pub fn kth_par(v: &[f32], k: usize) -> f32 {
    assert!(k < v.len(), "kth_par: {k} of {}", v.len());
    kth_par_by(v, k, Some)
}

/// [`kth_par`] over `f(x)` for every `x` of `v` that `f` keeps: the same
/// answer as collecting those values first, without the copy.
pub fn kth_par_by(v: &[f32], k: usize, f: impl Fn(f32) -> Option<f32> + Sync) -> f32 {
    use rayon::prelude::*;
    fn hist(v: &[f32], f: &(impl Fn(f32) -> Option<f32> + Sync), pick: impl Fn(u32) -> Option<usize> + Sync) -> Vec<u64> {
        v.par_chunks(1 << 16)
            .fold(
                || vec![0u64; 1 << 16],
                |mut h, c| {
                    for &x in c {
                        if let Some(y) = f(x) {
                            if let Some(b) = pick(order_key(y)) {
                                h[b] += 1;
                            }
                        }
                    }
                    h
                },
            )
            .reduce(
                || vec![0u64; 1 << 16],
                |mut a, b| {
                    a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
                    a
                },
            )
    }
    // The bin holding the k-th value, and how many values sit below it.
    let find = |h: &[u64], k: u64| -> (u32, u64) {
        let mut below = 0u64;
        for (b, &c) in h.iter().enumerate() {
            if below + c > k {
                return (b as u32, below);
            }
            below += c;
        }
        panic!("kth_par: asked for value {k} of {below}")
    };
    let (hi, below) = find(&hist(v, &f, |key| Some((key >> 16) as usize)), k as u64);
    let (lo, _) = find(&hist(v, &f, |key| ((key >> 16) == hi).then_some((key & 0xffff) as usize)), k as u64 - below);
    from_order_key((hi << 16) | lo)
}

/// `clipped_median_sigma` without copies: each round's values are those
/// inside every earlier round's bounds, i.e. inside their intersection,
/// counted and selected in parallel.
fn clipped_median_sigma_par(v: &[f32], k: f32, iters: usize) -> (f32, f32) {
    use rayon::prelude::*;
    let (mut lo, mut hi) = (f32::NEG_INFINITY, f32::INFINITY);
    let inside = |x: f32, lo: f32, hi: f32| x >= lo && x <= hi;
    let mut n = v.len();
    let (mut m, mut s) = (f32::NAN, f32::NAN);
    for _ in 0..iters {
        if n < 3 {
            break;
        }
        let (l, h) = (lo, hi);
        let nm = kth_par_by(v, n / 2, |x| inside(x, l, h).then_some(x));
        let mad = kth_par_by(v, n / 2, |x| inside(x, l, h).then(|| (x - nm).abs()));
        let ns = mad * MAD_TO_SIGMA as f32;
        let converged = nm == m && ns == s;
        m = nm;
        s = ns;
        if converged || s <= 0.0 {
            break;
        }
        lo = lo.max(m - k * s);
        hi = hi.min(m + k * s);
        let (l, h) = (lo, hi);
        let before = n;
        n = v.par_iter().filter(|&&x| inside(x, l, h)).count();
        if n == before {
            break;
        }
    }
    if m.is_nan() && n > 0 {
        let (l, h) = (lo, hi);
        let nm = kth_par_by(v, n / 2, |x| inside(x, l, h).then_some(x));
        let mad = kth_par_by(v, n / 2, |x| inside(x, l, h).then(|| (x - nm).abs()));
        m = nm;
        s = mad * MAD_TO_SIGMA as f32;
    }
    (m, s)
}

pub fn median(v: &[f32]) -> f32 {
    median_mut(&mut v.to_vec())
}

/// Median and MAD-sigma; leaves `v` holding the absolute deviations (in
/// any order).
pub fn median_sigma_mut(v: &mut [f32]) -> (f32, f32) {
    let m = median_mut(v);
    if v.len() >= PAR_MIN {
        use rayon::prelude::*;
        v.par_iter_mut().for_each(|x| *x = (*x - m).abs());
    } else {
        for x in v.iter_mut() {
            *x = (*x - m).abs();
        }
    }
    let mad = median_mut(v);
    (m, mad * MAD_TO_SIGMA as f32)
}

/// Iterative sigma-clipped median and sigma (MAD based). Reorders `v`
/// (big inputs are left as they are).
pub fn clipped_median_sigma(v: &mut Vec<f32>, k: f32, iters: usize) -> (f32, f32) {
    if v.len() >= PAR_MIN {
        return clipped_median_sigma_par(v, k, iters);
    }
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
    fn parallel_selection_matches_select_nth() {
        let mut seed = 99u64;
        let mut v: Vec<f32> = (0..300_001)
            .map(|i| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u = (seed >> 40) as f32 / (1u64 << 24) as f32;
                // Ties, negatives, zeros of both signs and a wide range.
                match i % 7 {
                    0 => 0.25,
                    1 => -u * 3.0,
                    2 => if i % 2 == 0 { 0.0 } else { -0.0 },
                    _ => u * u * 1e4,
                }
            })
            .collect();
        for k in [0usize, 1, 7, 150_000, 299_999, 300_000] {
            let want = *v.clone().select_nth_unstable_by(k, |a, b| a.total_cmp(b)).1;
            let got = kth_par(&v, k);
            assert_eq!(got.to_bits(), want.to_bits(), "k = {k}");
        }
        v.truncate(10);
        assert_eq!(kth_par(&v, 3).to_bits(), v.clone().select_nth_unstable_by(3, |a, b| a.total_cmp(b)).1.to_bits());
    }

    /// The copy-free clipping gives the copying version's answer.
    #[test]
    fn parallel_clipping_matches() {
        let mut seed = 5u64;
        let v: Vec<f32> = (0..(PAR_MIN + 12_345))
            .map(|i| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u = (seed >> 40) as f32 / (1u64 << 24) as f32;
                // Noise plus a few far outliers (stars).
                (u - 0.5) * 2.0 + if i % 997 == 0 { 40.0 * u } else { 0.0 }
            })
            .collect();
        for iters in [1, 3, 8] {
            let (pm, ps) = clipped_median_sigma_par(&v, 3.0, iters);
            // The serial algorithm, by hand on a copy.
            let mut w = v.clone();
            let (mut m, mut s) = (f32::NAN, f32::NAN);
            for _ in 0..iters {
                let mut work = w.clone();
                let mid = work.len() / 2;
                let nm = *work.select_nth_unstable_by(mid, |a, b| a.total_cmp(b)).1;
                for x in work.iter_mut() {
                    *x = (*x - nm).abs();
                }
                let ns = *work.select_nth_unstable_by(mid, |a, b| a.total_cmp(b)).1 * MAD_TO_SIGMA as f32;
                let conv = nm == m && ns == s;
                m = nm;
                s = ns;
                if conv {
                    break;
                }
                let before = w.len();
                w.retain(|&x| x >= m - 3.0 * s && x <= m + 3.0 * s);
                if w.len() == before {
                    break;
                }
            }
            assert_eq!((pm.to_bits(), ps.to_bits()), (m.to_bits(), s.to_bits()), "{iters} rounds");
        }
    }

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
