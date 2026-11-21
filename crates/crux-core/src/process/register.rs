//! Registration from the cached star lists (no pixel reads): match each
//! frame to the reference with `transparency::match_stars` (offset voting,
//! meridian flips), then fit a homography to every star pair within a
//! shrinking tolerance, dropping outliers. StarAlignment in WBPP fits the
//! same model (projective, no distortion correction).

use super::resample::Homography;
use crate::stars::Star;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Registration {
    /// Frame pixel → reference pixel.
    pub to_ref: Homography,
    pub matched: usize,
    /// RMS residual of the matched stars, pixels.
    pub rms: f64,
    pub shift: (f64, f64),
    pub rotation: f64,
    pub scale: f64,
    pub flipped: bool,
}

/// Least-squares homography mapping `src` → `dst` (≥ 4 pairs), with
/// coordinates normalised for conditioning.
pub fn fit_homography(pairs: &[((f64, f64), (f64, f64))]) -> Option<Homography> {
    if pairs.len() < 4 {
        return None;
    }
    let norm = |pts: &mut dyn Iterator<Item = (f64, f64)>| {
        let v: Vec<(f64, f64)> = pts.collect();
        let n = v.len() as f64;
        let (mx, my) = (v.iter().map(|p| p.0).sum::<f64>() / n, v.iter().map(|p| p.1).sum::<f64>() / n);
        let d = v.iter().map(|p| ((p.0 - mx).powi(2) + (p.1 - my).powi(2)).sqrt()).sum::<f64>() / n;
        let s = if d > 0.0 { std::f64::consts::SQRT_2 / d } else { 1.0 };
        Homography([s, 0.0, -s * mx, 0.0, s, -s * my, 0.0, 0.0, 1.0])
    };
    let ts = norm(&mut pairs.iter().map(|p| p.0));
    let td = norm(&mut pairs.iter().map(|p| p.1));
    // Normal equations for h0..h7 (h8 = 1).
    let mut a = [[0f64; 8]; 8];
    let mut b = [0f64; 8];
    for p in pairs {
        let (x, y) = ts.apply(p.0 .0, p.0 .1);
        let (u, v) = td.apply(p.1 .0, p.1 .1);
        let rows = [([x, y, 1.0, 0.0, 0.0, 0.0, -x * u, -y * u], u), ([0.0, 0.0, 0.0, x, y, 1.0, -x * v, -y * v], v)];
        for (r, t) in rows {
            for i in 0..8 {
                for j in 0..8 {
                    a[i][j] += r[i] * r[j];
                }
                b[i] += r[i] * t;
            }
        }
    }
    let h = solve(a, b)?;
    let hn = Homography([h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0]);
    // dst = td⁻¹ · hn · ts · src
    Some(td.inverse()?.then_after(&hn.then_after(&ts)))
}

/// Gaussian elimination with partial pivoting.
fn solve<const N: usize>(mut a: [[f64; N]; N], mut b: [f64; N]) -> Option<[f64; N]> {
    for c in 0..N {
        let p = (c..N).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))?;
        if a[p][c].abs() < 1e-12 {
            return None;
        }
        a.swap(c, p);
        b.swap(c, p);
        for r in c + 1..N {
            let f = a[r][c] / a[c][c];
            for k in c..N {
                a[r][k] -= f * a[c][k];
            }
            b[r] -= f * b[c];
        }
    }
    let mut x = [0f64; N];
    for r in (0..N).rev() {
        let s: f64 = (r + 1..N).map(|k| a[r][k] * x[k]).sum();
        x[r] = (b[r] - s) / a[r][r];
    }
    Some(x)
}

/// Spatial hash of reference stars for nearest-neighbour lookups.
struct Grid {
    cell: f64,
    map: std::collections::HashMap<(i32, i32), Vec<usize>>,
}

impl Grid {
    fn new(pts: &[(f64, f64)], cell: f64) -> Grid {
        let mut map: std::collections::HashMap<(i32, i32), Vec<usize>> = std::collections::HashMap::new();
        for (i, p) in pts.iter().enumerate() {
            map.entry(((p.0 / cell).floor() as i32, (p.1 / cell).floor() as i32)).or_default().push(i);
        }
        Grid { cell, map }
    }
    fn nearest(&self, pts: &[(f64, f64)], q: (f64, f64), tol: f64) -> Option<(usize, f64)> {
        let (cx, cy) = ((q.0 / self.cell).floor() as i32, (q.1 / self.cell).floor() as i32);
        let mut best: Option<(usize, f64)> = None;
        for dy in -1..=1 {
            for dx in -1..=1 {
                if let Some(v) = self.map.get(&(cx + dx, cy + dy)) {
                    for &i in v {
                        let d = ((pts[i].0 - q.0).powi(2) + (pts[i].1 - q.1).powi(2)).sqrt();
                        if d <= tol && best.is_none_or(|b| d < b.1) {
                            best = Some((i, d));
                        }
                    }
                }
            }
        }
        best
    }
}

/// Register `frame` to `reference` (both brightest first). `w`, `h`: frame size.
pub fn register(reference: &[Star], frame: &[Star], w: f64, h: f64) -> Option<Registration> {
    let m = crate::transparency::match_stars(reference, frame, w, h)?;
    // Initial model: rotate about the centre by m.angle, then the affine.
    let t = m.angle.to_radians();
    let (c, s, cx, cy) = (t.cos(), t.sin(), w / 2.0, h / 2.0);
    let rot = Homography([c, -s, cx - c * cx + s * cy, s, c, cy - s * cx - c * cy, 0.0, 0.0, 1.0]);
    let mut model = Homography::from_affine(&m.transform).then_after(&rot);
    refine(reference, frame, model).map(|(hm, matched, rms)| {
        model = hm;
        let (dx, dy, rotation, scale) = model.summary();
        Registration { to_ref: model, matched, rms, shift: (dx, dy), rotation, scale, flipped: m.flipped }
    })
}

/// Refit `model` (frame → reference) on mutual star pairs: all stars
/// first, then the unsaturated ones alone when there are enough (saturated
/// stars help find the match; their centroids are a little less precise).
pub fn refine(reference: &[Star], frame: &[Star], model: Homography) -> Option<(Homography, usize, f64)> {
    let all = refine_on(reference, frame, model)?;
    let clean = |v: &[Star]| v.iter().filter(|s| s.flags & crate::stars::SATURATED == 0).copied().collect::<Vec<Star>>();
    let (rc, fc) = (clean(reference), clean(frame));
    if (rc.len() < reference.len() || fc.len() < frame.len()) && rc.len() >= 60 && fc.len() >= 60 {
        if let Some(c) = refine_on(&rc, &fc, all.0).filter(|c| c.1 >= 30) {
            return Some(c);
        }
    }
    Some(all)
}

fn refine_on(reference: &[Star], frame: &[Star], mut model: Homography) -> Option<(Homography, usize, f64)> {
    let rp: Vec<(f64, f64)> = reference.iter().map(|s| (s.x as f64, s.y as f64)).collect();
    let fp: Vec<(f64, f64)> = frame.iter().map(|s| (s.x as f64, s.y as f64)).collect();
    let grid = Grid::new(&rp, 8.0);
    let mut last = (0usize, f64::MAX);
    for tol in [4.0, 2.5, 1.5, 1.5] {
        let mut pairs: Vec<((f64, f64), (f64, f64))> = Vec::new();
        let mut used = vec![false; rp.len()];
        for &p in &fp {
            let q = model.apply(p.0, p.1);
            if let Some((i, _)) = grid.nearest(&rp, q, tol) {
                if !used[i] {
                    used[i] = true;
                    pairs.push((p, rp[i]));
                }
            }
        }
        if pairs.len() < 8 {
            return None;
        }
        let fit = fit_homography(&pairs)?;
        // Drop pairs beyond 3× the RMS and refit once.
        let res: Vec<f64> = pairs.iter().map(|(a, b)| dist(fit.apply(a.0, a.1), *b)).collect();
        let rms = (res.iter().map(|r| r * r).sum::<f64>() / res.len() as f64).sqrt();
        let kept: Vec<_> = pairs.iter().zip(&res).filter(|(_, r)| **r <= 3.0 * rms).map(|(p, _)| *p).collect();
        model = if kept.len() >= 8 { fit_homography(&kept)? } else { fit };
        let res2: Vec<f64> = kept.iter().map(|(a, b)| dist(model.apply(a.0, a.1), *b)).collect();
        last = (kept.len(), (res2.iter().map(|r| r * r).sum::<f64>() / res2.len().max(1) as f64).sqrt());
    }
    let (_, _, _, scale) = model.summary();
    if (scale - 1.0).abs() > 0.02 {
        return None;
    }
    Some((model, last.0, last.1))
}

fn dist(a: (f64, f64), b: (f64, f64)) -> f64 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

/// A starting model from two plate solutions (frame → sky → reference),
/// for frames star matching can't pair with the reference.
pub fn seed_from_solutions(reference: &crate::solve::Solution, frame: &crate::solve::Solution, w: f64, h: f64) -> Option<Homography> {
    let mut pairs = Vec::new();
    for i in 0..=4 {
        for j in 0..=4 {
            let (x, y) = (w * i as f64 / 4.0, h * j as f64 / 4.0);
            let (ra, dec) = frame.pixel_to_sky(x, y);
            let (rx, ry) = reference.sky_to_pixel(ra, dec)?;
            pairs.push(((x, y), (rx, ry)));
        }
    }
    fit_homography(&pairs)
}

/// Reference quality: many sharp, round stars through clear sky.
pub fn reference_score(n_stars: u32, fwhm: Option<f32>, ecc: Option<f32>, transparency: Option<f64>) -> f64 {
    let f = fwhm.unwrap_or(10.0).max(0.5) as f64;
    let e = ecc.unwrap_or(0.6).clamp(0.0, 0.95) as f64;
    n_stars as f64 * transparency.unwrap_or(0.8).min(1.0) / (f * f) * (1.0 - 0.5 * e)
}

/// Largest displacement (px) between two frame→reference models over the frame.
pub fn max_difference(a: &Homography, b: &Homography, w: f64, h: f64) -> f64 {
    let mut mx: f64 = 0.0;
    for i in 0..=8 {
        for j in 0..=8 {
            let (x, y) = (w * i as f64 / 8.0, h * j as f64 / 8.0);
            mx = mx.max(dist(a.apply(x, y), b.apply(x, y)));
        }
    }
    mx
}

#[cfg(test)]
mod tests {
    use super::*;

    fn star(x: f64, y: f64, flux: f32) -> Star {
        Star { x: x as f32, y: y as f32, flux, peak: flux, fwhm: 3.0, hfr: 2.0, ecc: 0.2, pa: 0.0, snr: 50.0, area: 20, flags: 0 }
    }

    #[test]
    fn recovers_a_homography() {
        let (w, h) = (3000.0, 2000.0);
        let truth = Homography([0.9995, -0.0021, 31.2, 0.0019, 1.0003, -47.9, 2e-8, -1e-8, 1.0]);
        let inv = truth.inverse().unwrap();
        let mut rng = crate::synth::Rng::new(9);
        let mut reference = Vec::new();
        let mut frame = Vec::new();
        for k in 0..400 {
            let (x, y) = (rng.uniform() * w, rng.uniform() * h);
            let flux = 1000.0 - k as f32;
            reference.push(star(x, y, flux));
            let (fx, fy) = inv.apply(x, y);
            if fx > 10.0 && fy > 10.0 && fx < w - 10.0 && fy < h - 10.0 {
                frame.push(star(fx + 0.05 * rng.gauss(), fy + 0.05 * rng.gauss(), flux));
            }
        }
        let r = register(&reference, &frame, w, h).unwrap();
        assert!(r.matched > 300, "{}", r.matched);
        assert!(r.rms < 0.12, "{}", r.rms);
        assert!(max_difference(&r.to_ref, &truth, w, h) < 0.05, "{}", max_difference(&r.to_ref, &truth, w, h));
    }

    #[test]
    fn seeds_from_plate_solutions() {
        let (w, h) = (6248.0, 4176.0);
        let k = 0.6708 / 3600.0;
        let sol = |ra: f64, dec: f64, rot: f64| {
            let (c, s) = (rot.to_radians().cos(), rot.to_radians().sin());
            crate::solve::Solution { ra, dec, scale: 0.6708, rotation: rot, mirrored: false, cd: [-k * c, k * s, k * s, k * c], cx: w / 2.0, cy: h / 2.0, matched: 100, rms: 0.5, offset_arcmin: None, target_offset_arcmin: None, version: crate::solve::SOLVE_VERSION }
        };
        let r = sol(275.19, -16.17, 0.0);
        // The frame points 100 px east and is turned 180° (meridian flip).
        let f = sol(275.19 + 100.0 * k / (-16.17f64).to_radians().cos(), -16.17, 180.0);
        let seed = seed_from_solutions(&r, &f, w, h).unwrap();
        // Frame centre lands 100 px from the reference centre.
        let (x, y) = seed.apply(w / 2.0, h / 2.0);
        assert!(((x - w / 2.0).abs() - 100.0).abs() < 0.5 && (y - h / 2.0).abs() < 0.5, "{x} {y}");
        assert!((seed.summary().2.abs() - 180.0).abs() < 0.01, "{:?}", seed.summary());
    }

    #[test]
    fn meridian_flip() {
        let (w, h) = (3000.0, 2000.0);
        let mut rng = crate::synth::Rng::new(4);
        let t = 180.3f64.to_radians();
        let (c, s) = (t.cos(), t.sin());
        let mut reference = Vec::new();
        let mut frame = Vec::new();
        for k in 0..300 {
            let (x, y) = (rng.uniform() * w, rng.uniform() * h);
            reference.push(star(x, y, 1000.0 - k as f32));
            // frame = rotation of reference about the centre plus a shift
            let (dx, dy) = (x - w / 2.0, y - h / 2.0);
            let (fx, fy) = (w / 2.0 + c * dx - s * dy + 12.0, h / 2.0 + s * dx + c * dy - 7.0);
            if fx > 0.0 && fy > 0.0 && fx < w && fy < h {
                frame.push(star(fx, fy, 1000.0 - k as f32));
            }
        }
        let r = register(&reference, &frame, w, h).unwrap();
        assert!(r.flipped);
        assert!(r.rms < 0.01, "{}", r.rms);
        // frame→reference undoes the 180.3° turn.
        assert!((r.rotation.rem_euclid(360.0) - 179.7).abs() < 0.01, "{}", r.rotation);
    }
}
