//! Transparency by relative photometry on the same stars.
//!
//! Star count and sky darkness both mislead (thin cloud can leave plenty of
//! stars and even darken the sky), so instead: match each frame's brightest
//! unsaturated stars to a reference frame of the same field, and take the
//! median flux ratio. Normalised so the cohort's best frame is 1.0.
//!
//! Matching: vote on reference−frame offsets of star pairs (finds the dither
//! or re-pointing shift without a plate solve), trying the frame both as is
//! and rotated 180° (meridian flip). Then mutual-nearest pairs and an affine
//! refit, twice, with shrinking tolerance.

use crate::analysis::LightMetrics;
use crate::stars::{Star, SATURATED};
use crate::stats::{median_f64, median_sigma_f64};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const GRID: usize = 4;
const VOTE_STARS: usize = 150;
const BIN: f64 = 3.0;

/// x_ref = a0 + a1·x + a2·y, y_ref = b0 + b1·x + b2·y.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Affine {
    pub a: [f64; 3],
    pub b: [f64; 3],
}

impl Affine {
    pub fn translation(dx: f64, dy: f64) -> Self {
        Affine { a: [dx, 1.0, 0.0], b: [dy, 0.0, 1.0] }
    }
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        (self.a[0] + self.a[1] * x + self.a[2] * y, self.b[0] + self.b[1] * x + self.b[2] * y)
    }
    pub fn det(&self) -> f64 {
        self.a[1] * self.b[2] - self.a[2] * self.b[1]
    }
    pub fn inverse(&self) -> Option<Affine> {
        let d = self.det();
        if d.abs() < 1e-9 {
            return None;
        }
        let (a1, a2, b1, b2) = (self.b[2] / d, -self.a[2] / d, -self.b[1] / d, self.a[1] / d);
        Some(Affine { a: [-(a1 * self.a[0] + a2 * self.b[0]), a1, a2], b: [-(b1 * self.a[0] + b2 * self.b[0]), b1, b2] })
    }
}

#[derive(Debug, Clone)]
pub struct StarMatch {
    /// Frame (after rotating by `angle` about the centre) → reference coordinates.
    pub transform: Affine,
    /// Rotation applied before the transform, degrees (≈180 after a meridian flip).
    pub angle: f64,
    pub flipped: bool,
    /// (reference index, frame index).
    pub pairs: Vec<(usize, usize)>,
}

/// Rotation about the image centre.
#[derive(Clone, Copy)]
struct Rot {
    c: f64,
    s: f64,
    cx: f64,
    cy: f64,
}

impl Rot {
    fn new(deg: f64, w: f64, h: f64) -> Self {
        let t = deg.to_radians();
        Rot { c: t.cos(), s: t.sin(), cx: w / 2.0, cy: h / 2.0 }
    }
    fn pt(&self, s: &Star) -> (f64, f64) {
        self.xy(s.x as f64, s.y as f64)
    }
    fn xy(&self, x: f64, y: f64) -> (f64, f64) {
        let (x, y) = (x - self.cx, y - self.cy);
        (self.cx + self.c * x - self.s * y, self.cy + self.s * x + self.c * y)
    }
}

/// Best offset peak for one rotation: (votes in the 3×3 peak, peak bin).
fn vote(rv: &[Star], fv: &[Star], rot: Rot) -> (u32, (i32, i32)) {
    let mut votes: HashMap<(i32, i32), u32> = HashMap::with_capacity(rv.len() * fv.len());
    let fp: Vec<(f64, f64)> = fv.iter().map(|b| rot.pt(b)).collect();
    for a in rv {
        for &(fx, fy) in &fp {
            let k = (((a.x as f64 - fx) / BIN).floor() as i32, ((a.y as f64 - fy) / BIN).floor() as i32);
            *votes.entry(k).or_default() += 1;
        }
    }
    // Peak of 3×3 bin sums, so a shift on a bin edge isn't split.
    let mut best = (0u32, (0, 0));
    for &k in votes.keys() {
        let mut s = 0;
        for dy in -1..=1 {
            for dx in -1..=1 {
                s += votes.get(&(k.0 + dx, k.1 + dy)).copied().unwrap_or(0);
            }
        }
        if s > best.0 {
            best = (s, k);
        }
    }
    best
}

/// Match frame stars to reference stars. Both lists brightest first.
/// Tries no rotation first; if that's weak, searches ±2° around 0° and
/// 180° (meridian flips are never exactly 180°).
pub fn match_stars(r: &[Star], f: &[Star], w: f64, h: f64) -> Option<StarMatch> {
    let rv = &r[..r.len().min(VOTE_STARS)];
    let fv = &f[..f.len().min(VOTE_STARS)];
    let (mut peak, mut k) = vote(rv, fv, Rot::new(0.0, w, h));
    let mut angle = 0.0;
    if peak < 30 {
        // Coarse search on fewer stars, then the winner at full size.
        let rs = &rv[..rv.len().min(90)];
        let fs = &fv[..fv.len().min(90)];
        let mut cands: Vec<f64> = Vec::new();
        for base in [0.0, 180.0] {
            for i in -20..=20 {
                cands.push(base + i as f64 * 0.1);
            }
        }
        let mut best = (0u32, 0.0);
        for &a in &cands {
            let (p, _) = vote(rs, fs, Rot::new(a, w, h));
            if p > best.0 {
                best = (p, a);
            }
        }
        let (p2, k2) = vote(rv, fv, Rot::new(best.1, w, h));
        if p2 > peak {
            peak = p2;
            k = k2;
            angle = best.1;
        }
    }
    if peak < 8 {
        return None;
    }
    let rot = Rot::new(angle, w, h);
    // Refine the shift from the pairs inside the winning 3×3 bins.
    let (mut dxs, mut dys) = (Vec::new(), Vec::new());
    for a in rv {
        for b in fv {
            let (fx, fy) = rot.pt(b);
            let (dx, dy) = (a.x as f64 - fx, a.y as f64 - fy);
            let kk = ((dx / BIN).floor() as i32, (dy / BIN).floor() as i32);
            if (kk.0 - k.0).abs() <= 1 && (kk.1 - k.1).abs() <= 1 {
                dxs.push(dx);
                dys.push(dy);
            }
        }
    }
    let mut t = Affine::translation(median_f64(&dxs)?, median_f64(&dys)?);
    let mut pairs = Vec::new();
    for tol in [8.0, 4.0, 2.5] {
        pairs = mutual_nearest(r, f, &t, rot, tol);
        if pairs.len() < 6 {
            return None;
        }
        if pairs.len() >= 10 {
            if let Some(a) = fit_affine(r, f, &pairs, rot) {
                t = a;
            }
        }
    }
    // A real match between frames of one camera has scale ≈ 1.
    if (t.det().abs() - 1.0).abs() > 0.02 || pairs.len() < 10 {
        return None;
    }
    let flipped = (angle - 180.0).abs() < 45.0;
    Some(StarMatch { transform: t, angle, flipped, pairs })
}

fn mutual_nearest(r: &[Star], f: &[Star], t: &Affine, rot: Rot, tol: f64) -> Vec<(usize, usize)> {
    let fp: Vec<(f64, f64)> = f
        .iter()
        .map(|s| {
            let (x, y) = rot.pt(s);
            t.apply(x, y)
        })
        .collect();
    let tol2 = tol * tol;
    let nearest = |p: (f64, f64), pts: &mut dyn Iterator<Item = (usize, (f64, f64))>| {
        let mut best = (usize::MAX, f64::MAX);
        for (i, q) in pts {
            let d = (p.0 - q.0).powi(2) + (p.1 - q.1).powi(2);
            if d < best.1 {
                best = (i, d);
            }
        }
        best
    };
    let mut pairs = Vec::new();
    for (fi, &p) in fp.iter().enumerate() {
        let (ri, d) = nearest(p, &mut r.iter().map(|s| (s.x as f64, s.y as f64)).enumerate());
        if ri == usize::MAX || d > tol2 {
            continue;
        }
        let q = (r[ri].x as f64, r[ri].y as f64);
        let (back, _) = nearest(q, &mut fp.iter().copied().enumerate());
        if back == fi {
            pairs.push((ri, fi));
        }
    }
    pairs
}

fn fit_affine(r: &[Star], f: &[Star], pairs: &[(usize, usize)], rot: Rot) -> Option<Affine> {
    // Normal equations for [1 x y]·coef = target, solved for x and y targets.
    let mut m = [[0f64; 3]; 3];
    let (mut vx, mut vy) = ([0f64; 3], [0f64; 3]);
    for &(ri, fi) in pairs {
        let (x, y) = rot.pt(&f[fi]);
        let row = [1.0, x, y];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] += row[i] * row[j];
            }
            vx[i] += row[i] * r[ri].x as f64;
            vy[i] += row[i] * r[ri].y as f64;
        }
    }
    Some(Affine { a: solve3(m, vx)?, b: solve3(m, vy)? })
}

fn solve3(m: [[f64; 3]; 3], v: [f64; 3]) -> Option<[f64; 3]> {
    let det = |m: &[[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let d = det(&m);
    if d.abs() < 1e-12 {
        return None;
    }
    let mut out = [0f64; 3];
    for (c, o) in out.iter_mut().enumerate() {
        let mut mc = m;
        for r in 0..3 {
            mc[r][c] = v[r];
        }
        *o = det(&mc) / d;
    }
    Some(out)
}

/// One frame against the reference: raw flux ratio and the 4×4 grid.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RawTransparency {
    pub ratio: f64,
    /// Robust scatter of the per-star ratios.
    pub scatter: f64,
    pub n_matched: usize,
    pub flipped: bool,
    pub shift: (f64, f64),
    /// Per-cell median ratio (frame coordinates, row-major from the top-left).
    pub cells: Vec<Option<f64>>,
    /// Cells where the reference has plenty of stars but the frame matched almost none.
    pub lost: Vec<bool>,
    /// Frame pixel → reference pixel, rotation included.
    pub to_ref: Option<Affine>,
}

/// Affine for "rotate by `deg` about the frame centre, then `t`".
fn compose(t: &Affine, deg: f64, w: f64, h: f64) -> Affine {
    let r = Rot::new(deg, w, h);
    // Rotation as an affine: x' = cx + c(x−cx) − s(y−cy), y' = cy + s(x−cx) + c(y−cy).
    let ra = Affine { a: [r.cx - r.c * r.cx + r.s * r.cy, r.c, -r.s], b: [r.cy - r.s * r.cx - r.c * r.cy, r.s, r.c] };
    Affine {
        a: [t.a[0] + t.a[1] * ra.a[0] + t.a[2] * ra.b[0], t.a[1] * ra.a[1] + t.a[2] * ra.b[1], t.a[1] * ra.a[2] + t.a[2] * ra.b[2]],
        b: [t.b[0] + t.b[1] * ra.a[0] + t.b[2] * ra.b[0], t.b[1] * ra.a[1] + t.b[2] * ra.b[1], t.b[1] * ra.a[2] + t.b[2] * ra.b[2]],
    }
}

pub fn measure(r: &[Star], f: &[Star], w: f64, h: f64) -> Option<RawTransparency> {
    let m = match_stars(r, f, w, h)?;
    let ratios: Vec<(usize, f64)> = m
        .pairs
        .iter()
        .filter(|&&(ri, fi)| r[ri].flux > 0.0 && f[fi].flux > 0.0 && r[ri].flags & SATURATED == 0 && f[fi].flags & SATURATED == 0)
        .map(|&(ri, fi)| (fi, f[fi].flux as f64 / r[ri].flux as f64))
        .collect();
    let all: Vec<f64> = ratios.iter().map(|x| x.1).collect();
    let (ratio, scatter) = median_sigma_f64(&all)?;
    let cell_of = |x: f64, y: f64| -> Option<usize> {
        if x < 0.0 || y < 0.0 || x >= w || y >= h {
            return None;
        }
        Some(((y / h * GRID as f64) as usize).min(GRID - 1) * GRID + ((x / w * GRID as f64) as usize).min(GRID - 1))
    };
    let mut per: Vec<Vec<f64>> = vec![Vec::new(); GRID * GRID];
    for &(fi, q) in &ratios {
        if let Some(c) = cell_of(f[fi].x as f64, f[fi].y as f64) {
            per[c].push(q);
        }
    }
    // Where reference stars land in this frame, to spot cells that lost theirs.
    // Only count reference stars that should still make the frame's
    // brightest-N list at this frame's transparency (with 1.5× margin):
    // otherwise asymmetric vignetting after a meridian flip, which pushes a
    // corner's stars off the list, looks like lost stars.
    let cutoff = if f.len() >= 100 { f.iter().map(|s| s.flux).fold(f32::MAX, f32::min) as f64 } else { 0.0 };
    // Saturated stars carry no photometry, so they can't be "lost" either.
    let mut ref_count = vec![0usize; GRID * GRID];
    if let Some(inv) = m.transform.inverse() {
        let back = Rot::new(-m.angle, w, h);
        for s in r.iter().filter(|s| s.flags & SATURATED == 0 && s.flux as f64 * ratio > 1.5 * cutoff) {
            let (x, y) = inv.apply(s.x as f64, s.y as f64);
            let (x, y) = back.xy(x, y);
            if let Some(c) = cell_of(x, y) {
                ref_count[c] += 1;
            }
        }
    }
    let cells = per.iter().map(|v| if v.len() >= 3 { median_f64(v) } else { None }).collect();
    let lost = per.iter().zip(&ref_count).map(|(v, &n)| n >= 8 && v.len() < 3).collect();
    Some(RawTransparency {
        ratio,
        scatter,
        n_matched: all.len(),
        flipped: m.flipped,
        shift: (m.transform.a[0], m.transform.b[0]),
        cells,
        lost,
        to_ref: Some(compose(&m.transform, m.angle, w, h)),
    })
}

/// Transparency of a frame within its cohort, 1.0 = best frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameTransparency {
    pub value: f64,
    pub n_matched: usize,
    /// Per-cell transparency on the same scale.
    pub cells: Vec<Option<f64>>,
    pub lost: Vec<bool>,
    /// 1 − worst measured cell / median cell.
    pub patchiness: Option<f64>,
    pub flipped: bool,
    /// Frame index of the reference.
    pub reference: usize,
    /// Frame pixel → reference pixel (rotation included), for blinking
    /// against the reference.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub to_ref: Option<Affine>,
}

/// Transparency for every frame of one field (cohort of frame indices).
/// Frames that can't be matched are left out.
pub fn cohort(ids: &[usize], metrics: &HashMap<usize, LightMetrics>, min_matches: usize) -> HashMap<usize, FrameTransparency> {
    let mut out = HashMap::new();
    // Reference: the frame with the most stars (best for matching; its own
    // transparency doesn't matter because results are renormalised).
    let Some(&rid) = ids.iter().filter(|i| metrics.contains_key(i)).max_by_key(|i| metrics[i].n_stars) else {
        return out;
    };
    let rm = &metrics[&rid];
    let (w, h) = (rm.width as f64, rm.height as f64);
    let mut raw: Vec<(usize, RawTransparency)> = Vec::new();
    for &i in ids {
        let Some(m) = metrics.get(&i) else { continue };
        if i == rid {
            raw.push((
                i,
                RawTransparency {
                    ratio: 1.0,
                    scatter: 0.0,
                    n_matched: m.stars.len(),
                    flipped: false,
                    shift: (0.0, 0.0),
                    cells: vec![Some(1.0); GRID * GRID],
                    lost: vec![false; GRID * GRID],
                    to_ref: Some(Affine::translation(0.0, 0.0)),
                },
            ));
            continue;
        }
        if let Some(t) = measure(&rm.stars, &m.stars, w, h).filter(|t| t.n_matched >= min_matches) {
            raw.push((i, t));
        }
    }
    // Normalise to the best frame, unless it stands far above the runner-up
    // (a measurement outlier would otherwise drag every other frame down).
    let mut sorted: Vec<f64> = raw.iter().map(|(_, t)| t.ratio).collect();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let best = match sorted.as_slice() {
        [a, b, ..] if *a > 1.25 * b => *b,
        [a, ..] => *a,
        [] => 0.0,
    };
    if best <= 0.0 {
        return out;
    }
    for (i, t) in raw {
        let cells: Vec<Option<f64>> = t.cells.iter().map(|c| c.map(|v| v / best)).collect();
        let valid: Vec<f64> = cells.iter().flatten().copied().collect();
        // Only cells with measured stars count. "Lost" cells alone aren't
        // evidence of cloud: a defocused corner turns stars into donuts that
        // fail the blend test, and they vanish just the same (LMC panel 31,
        // 17 Nov 03:29). Genuinely patchy frames show it in measured cells.
        let patchiness = (valid.len() >= GRID * GRID / 2).then(|| {
            let med = median_f64(&valid).unwrap_or(0.0);
            let worst = valid.iter().copied().fold(f64::MAX, f64::min);
            if med > 0.0 { (1.0 - worst / med).max(0.0) } else { 0.0 }
        });
        out.insert(
            i,
            FrameTransparency {
                value: t.ratio / best,
                n_matched: t.n_matched,
                cells,
                lost: t.lost,
                patchiness,
                flipped: t.flipped,
                reference: rid,
                to_ref: t.to_ref,
            },
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::synth::{render, Params};

    fn stars(p: &Params) -> LightMetrics {
        let (img, _) = render(p);
        crate::analysis::analyze_light(&img, &Config::default())
    }

    #[test]
    fn affine_inverse() {
        let a = Affine { a: [10.0, 0.999, 0.01], b: [-5.0, -0.01, 1.001] };
        let inv = a.inverse().unwrap();
        let (x, y) = a.apply(123.0, 456.0);
        let (bx, by) = inv.apply(x, y);
        assert!((bx - 123.0).abs() < 1e-9 && (by - 456.0).abs() < 1e-9);
    }

    #[test]
    fn matches_dither_and_flip() {
        let base = Params { n_stars: 500, ..Default::default() };
        let r = stars(&base);
        let f = stars(&Params { shift: (37.0, -21.0), noise_seed: 9, ..base.clone() });
        let m = match_stars(&r.stars, &f.stars, 1024.0, 768.0).unwrap();
        assert!(!m.flipped);
        assert!((m.transform.a[0] + 37.0).abs() < 0.5 && (m.transform.b[0] - 21.0).abs() < 0.5, "{:?}", m.transform);
        assert!(m.pairs.len() > 100, "{}", m.pairs.len());

        // Meridian flip: render the field rotated 180°.
        let (img, _) = render(&Params { noise_seed: 11, ..base.clone() });
        let mut rot = img.clone();
        let n = rot.data.len();
        for i in 0..n {
            rot.data[i] = img.data[n - 1 - i];
        }
        let fm = crate::analysis::analyze_light(&rot, &Config::default());
        let m = match_stars(&r.stars, &fm.stars, 1024.0, 768.0).unwrap();
        assert!(m.flipped);
        assert!(m.pairs.len() > 100);
    }

    #[test]
    fn matches_inexact_meridian_flip() {
        // A flip that's 180.4° with a shift, on a full-size frame where
        // 0.4° moves corner stars by ~25 px.
        let (w, h) = (6248.0, 4176.0);
        let mut rng = crate::synth::Rng::new(3);
        let r: Vec<Star> = (0..300)
            .map(|i| Star {
                x: (rng.uniform() * w) as f32,
                y: (rng.uniform() * h) as f32,
                flux: 1000.0 - i as f32,
                peak: 1.0,
                fwhm: 4.0,
                hfr: 3.0,
                ecc: 0.3,
                pa: 0.0,
                snr: 100.0,
                area: 30,
                flags: 0,
            })
            .collect();
        let rot = Rot::new(180.4, w, h);
        let f: Vec<Star> = r
            .iter()
            .map(|s| {
                let (x, y) = rot.pt(s);
                Star { x: (x + 40.0) as f32, y: (y - 25.0) as f32, ..*s }
            })
            .filter(|s| s.x > 0.0 && s.y > 0.0 && (s.x as f64) < w && (s.y as f64) < h)
            .collect();
        let m = match_stars(&r, &f, w, h).expect("match");
        assert!(m.flipped);
        assert!(m.pairs.len() > 250, "{}", m.pairs.len());
    }

    #[test]
    fn transparency_drop_is_measured() {
        let base = Params { n_stars: 600, ..Default::default() };
        let mut metrics = HashMap::new();
        metrics.insert(0, stars(&base));
        metrics.insert(1, stars(&Params { noise_seed: 5, shift: (12.0, 8.0), ..base.clone() }));
        metrics.insert(2, stars(&Params { noise_seed: 6, shift: (-9.0, 4.0), transparency: 0.45, ..base.clone() }));
        let t = cohort(&[0, 1, 2], &metrics, 15);
        assert!((t[&0].value - 1.0).abs() < 0.05, "{:?}", t[&0].value);
        assert!((t[&1].value - 1.0).abs() < 0.05, "{:?}", t[&1].value);
        assert!((t[&2].value - 0.45).abs() < 0.05, "{:?}", t[&2].value);
        assert!(t[&1].patchiness.unwrap() < 0.25, "{:?}", t[&1].patchiness);
    }

    #[test]
    fn patchy_cloud_is_localised() {
        let base = Params { n_stars: 900, ..Default::default() };
        let r = stars(&base);
        // Cloud over the left quarter: dim stars there by 60%.
        let (mut img, _) = render(&Params { noise_seed: 8, ..base.clone() });
        let sky = 800.0 / 65535.0;
        for y in 0..img.height {
            for x in 0..img.width / 4 {
                let v = &mut img.data[y * img.width + x];
                *v = sky + (*v - sky) * 0.4;
            }
        }
        let f = crate::analysis::analyze_light(&img, &Config::default());
        let mut metrics = HashMap::new();
        metrics.insert(0, r);
        metrics.insert(1, f);
        let t = cohort(&[0, 1], &metrics, 15);
        let p = t[&1].patchiness.unwrap();
        assert!(p > 0.4, "patchiness {p}, cells {:?}", t[&1].cells);
        // The worst cells are in the left column.
        let left: Vec<f64> = (0..GRID).filter_map(|r| t[&1].cells[r * GRID]).collect();
        let right: Vec<f64> = (0..GRID).filter_map(|r| t[&1].cells[r * GRID + GRID - 1]).collect();
        assert!(left.iter().sum::<f64>() / left.len().max(1) as f64 <= 0.6 * right.iter().sum::<f64>() / right.len() as f64 || t[&1].lost.iter().any(|&l| l));
    }
}
