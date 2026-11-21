//! Satellite and aircraft trails.
//!
//! 1. Bin the background-subtracted image 2×2 (trails are long, so binning
//!    costs little signal and quarters the work).
//! 2. Mask compact blobs above 4σ (stars, halos); long thin shapes stay.
//! 3. Hough transform over pixels above 2σ; score each (ρ, θ) cell against
//!    the votes noise alone would give along a chord that long.
//! 4. Verify each peak: refit the line to nearby candidates, then walk it in
//!    16 px segments comparing a 3 px core band with side bands 5–6 px out.
//!    A real trail lights up a long stretch of segments; elongated stars,
//!    diffraction spikes and nebula filaments don't.
//! 5. Regular lit/unlit alternation along a trail = aircraft (blinking lights).

use crate::stats::median_sigma_mut;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrailKind {
    Satellite,
    Aircraft,
}

impl TrailKind {
    pub fn name(self) -> &'static str {
        match self {
            TrailKind::Satellite => "satellite",
            TrailKind::Aircraft => "aircraft",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Trail {
    pub kind: TrailKind,
    /// Direction of the trail, degrees in [0, 180), from +x toward +y.
    pub angle: f32,
    /// Ends of the lit part, full-resolution pixels.
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
    /// Length of the lit part, full-resolution pixels.
    pub length: f32,
    /// Median brightness above the local sky along the lit part, ADU.
    pub brightness_adu: f32,
    /// Lit fraction between the first and last lit segment.
    pub coverage: f32,
    /// Hough significance (σ above noise).
    pub score: f32,
}

#[derive(Debug, Clone)]
pub struct TrailParams {
    pub bin: usize,
    pub cand_sigma: f32,
    pub star_sigma: f32,
    /// Blobs no bigger than this (binned px) are masked as stars.
    pub star_extent: usize,
    pub angle_step_deg: f32,
    /// Hough significance needed to try verifying a line.
    pub min_score: f32,
    /// Shortest lit length, full-resolution pixels.
    pub min_length: f32,
    pub max_trails: usize,
    /// Segment length along the line, binned px.
    pub seg: usize,
    /// Segment is lit when core − side exceeds this many binned σ.
    pub lit_sigma: f32,
}

impl Default for TrailParams {
    fn default() -> Self {
        TrailParams {
            bin: 2,
            cand_sigma: 2.0,
            star_sigma: 4.0,
            star_extent: 12,
            angle_step_deg: 0.5,
            min_score: 8.0,
            min_length: 600.0,
            max_trails: 10,
            seg: 16,
            lit_sigma: 0.8,
        }
    }
}

struct Binned {
    w: usize,
    h: usize,
    v: Vec<f32>,
    mask: Vec<bool>,
    sigma: f32,
}

impl Binned {
    #[inline]
    fn get(&self, x: f32, y: f32) -> Option<(f32, bool)> {
        let (xi, yi) = (x.round() as isize, y.round() as isize);
        if xi < 0 || yi < 0 || xi >= self.w as isize || yi >= self.h as isize {
            return None;
        }
        let i = yi as usize * self.w + xi as usize;
        Some((self.v[i], self.mask[i]))
    }
}

fn bin_residual(resid: &[f32], w: usize, h: usize, b: usize) -> (Vec<f32>, usize, usize) {
    let (bw, bh) = (w / b, h / b);
    let mut v = vec![0f32; bw * bh];
    let k = 1.0 / (b * b) as f32;
    for by in 0..bh {
        for dy in 0..b {
            let row = &resid[(by * b + dy) * w..(by * b + dy + 1) * w];
            let out = &mut v[by * bw..(by + 1) * bw];
            for (bx, o) in out.iter_mut().enumerate() {
                let s: f32 = row[bx * b..bx * b + b].iter().sum();
                *o += s * k;
            }
        }
    }
    (v, bw, bh)
}

/// Mask compact bright blobs (stars and their halos), keep long thin shapes.
fn star_mask(v: &[f32], w: usize, h: usize, thr: f32, extent: usize) -> Vec<bool> {
    let mut mask = vec![false; w * h];
    let mut seen = vec![false; w * h];
    let mut stack: Vec<u32> = Vec::new();
    for start in 0..w * h {
        if seen[start] || v[start] <= thr {
            continue;
        }
        seen[start] = true;
        stack.clear();
        stack.push(start as u32);
        let (mut x0, mut y0, mut x1, mut y1, mut area) = (usize::MAX, usize::MAX, 0, 0, 0usize);
        let (mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0f64, 0f64, 0f64, 0f64, 0f64);
        let mut peak = f32::MIN;
        while let Some(i) = stack.pop() {
            let i = i as usize;
            let (x, y) = (i % w, i / w);
            area += 1;
            peak = peak.max(v[i]);
            let (fx, fy) = (x as f64, y as f64);
            sx += fx;
            sy += fy;
            sxx += fx * fx;
            syy += fy * fy;
            sxy += fx * fy;
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let (xx, yy) = (x as i32 + dx, y as i32 + dy);
                    if xx < 0 || yy < 0 || xx >= w as i32 || yy >= h as i32 {
                        continue;
                    }
                    let j = yy as usize * w + xx as usize;
                    if !seen[j] && v[j] > thr {
                        seen[j] = true;
                        stack.push(j as u32);
                    }
                }
            }
        }
        let (ew, eh) = (x1 - x0 + 1, y1 - y0 + 1);
        let big = ew.max(eh);
        let fill = area as f32 / (ew * eh) as f32;
        // Stars are roughly round; a trail, or a fragment of a bright one, is
        // thin at any angle. Judge by the pixels' second moments (a bounding
        // box makes a diagonal fragment look square).
        let n = area as f64;
        let (mx, my) = (sx / n, sy / n);
        let (cxx, cyy, cxy) = (sxx / n - mx * mx + 1.0 / 12.0, syy / n - my * my + 1.0 / 12.0, sxy / n - mx * my);
        let tr = cxx + cyy;
        let disc = ((cxx - cyy).powi(2) + 4.0 * cxy * cxy).sqrt();
        let elong = (tr + disc) / (tr - disc).max(1e-6);
        let round = elong <= 6.0;
        // Only bright blobs are masked: a moderately bright trail breaks
        // into 1–3 px fragments above the threshold, and those look round.
        let blob = round && peak >= 2.0 * thr && (big <= extent || fill >= 0.3);
        if blob {
            let m = 2 + big / 4;
            for y in y0.saturating_sub(m)..(y1 + m + 1).min(h) {
                for x in x0.saturating_sub(m)..(x1 + m + 1).min(w) {
                    mask[y * w + x] = true;
                }
            }
        }
    }
    mask
}

/// Parameter range t along the line where p0 + t·d stays inside the box.
fn chord(rho: f32, theta: f32, lo: f32, wmax: f32, hmax: f32) -> Option<(f32, f32)> {
    let (c, s) = (theta.cos(), theta.sin());
    let (px, py) = (rho * c, rho * s);
    let (dx, dy) = (-s, c);
    let (mut t0, mut t1) = (f32::MIN, f32::MAX);
    for (p, d, a, b) in [(px, dx, lo, wmax), (py, dy, lo, hmax)] {
        if d.abs() < 1e-6 {
            if p < a || p > b {
                return None;
            }
        } else {
            let (u, v) = ((a - p) / d, (b - p) / d);
            t0 = t0.max(u.min(v));
            t1 = t1.min(u.max(v));
        }
    }
    (t1 > t0).then_some((t0, t1))
}

/// Total-least-squares line through candidates within `band` of (ρ, θ).
fn refine(cands: &[(u16, u16)], rho: f32, theta: f32, band: f32) -> Option<(f32, f32)> {
    let (c, s) = (theta.cos(), theta.sin());
    let pts: Vec<(f32, f32)> = cands
        .iter()
        .map(|&(x, y)| (x as f32, y as f32))
        .filter(|&(x, y)| (x * c + y * s - rho).abs() <= band)
        .collect();
    if pts.len() < 20 {
        return None;
    }
    let n = pts.len() as f32;
    let (mx, my) = pts.iter().fold((0.0, 0.0), |a, p| (a.0 + p.0 / n, a.1 + p.1 / n));
    let (mut sxx, mut syy, mut sxy) = (0f32, 0f32, 0f32);
    for &(x, y) in &pts {
        sxx += (x - mx) * (x - mx);
        syy += (y - my) * (y - my);
        sxy += (x - mx) * (y - my);
    }
    // Direction of the line = major eigenvector; normal is perpendicular.
    let dir = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let mut th = dir + std::f32::consts::FRAC_PI_2;
    let mut r = mx * th.cos() + my * th.sin();
    if r < 0.0 {
        r = -r;
        th += std::f32::consts::PI;
    }
    // Keep θ comparable with the Hough range [0, π).
    let th = th.rem_euclid(2.0 * std::f32::consts::PI);
    let (th, r) = if th >= std::f32::consts::PI { (th - std::f32::consts::PI, -r) } else { (th, r) };
    Some((r, th))
}

struct Verified {
    rho: f32,
    theta: f32,
    t_first: f32,
    t_last: f32,
    lit: usize,
    coverage: f32,
    transitions: usize,
    brightness: f32,
}

fn verify(b: &Binned, rho: f32, theta: f32, p: &TrailParams) -> Option<Verified> {
    let (lo, hi_w, hi_h) = (3.0, b.w as f32 - 4.0, b.h as f32 - 4.0);
    let (t0, t1) = chord(rho, theta, lo, hi_w, hi_h)?;
    let (c, s) = (theta.cos(), theta.sin());
    let (px, py) = (rho * c, rho * s);
    let (dx, dy) = (-s, c);
    let seg = p.seg;
    // Per segment: (core − brighter side, local roughness of the sides).
    let mut segs: Vec<Option<(f32, f32)>> = Vec::new();
    let mut core: Vec<f32> = Vec::with_capacity(seg * 3);
    let mut left: Vec<f32> = Vec::with_capacity(seg * 2);
    let mut right: Vec<f32> = Vec::with_capacity(seg * 2);
    let (mut t, mut k, mut masked) = (t0, 0usize, 0usize);
    while t <= t1 {
        let (x, y) = (px + t * dx, py + t * dy);
        let mut any_mask = false;
        for o in [-1.0f32, 0.0, 1.0] {
            if let Some((v, m)) = b.get(x + o * c, y + o * s) {
                any_mask |= m;
                core.push(v);
            }
        }
        for o in [-6.0f32, -5.0, 5.0, 6.0] {
            if let Some((v, m)) = b.get(x + o * c, y + o * s) {
                if !m {
                    if o < 0.0 { left.push(v) } else { right.push(v) }
                }
            }
        }
        if any_mask {
            masked += 1;
        }
        k += 1;
        if k == seg {
            let v = if masked * 2 > seg || core.len() < seg || left.len() < seg / 2 || right.len() < seg / 2 {
                None
            } else {
                let (cm, _) = median_sigma_mut(&mut core);
                let (lm, ls) = median_sigma_mut(&mut left);
                let (rm, rs) = median_sigma_mut(&mut right);
                // A trail is a ridge: brighter than both sides. A nebula
                // edge or gradient is bright on one side only.
                Some((cm - lm.max(rm), 0.5 * (ls + rs)))
            };
            segs.push(v);
            core.clear();
            left.clear();
            right.clear();
            k = 0;
            masked = 0;
        }
        t += 1.0;
    }
    // Lit threshold rises with local roughness: smooth sky has side σ ≈ the
    // global σ; bright nebulosity is far rougher (LMC panel 14, 30 Doradus).
    let base = p.lit_sigma * b.sigma;
    let lit_flags: Vec<Option<bool>> = segs.iter().map(|v| v.map(|(x, rough)| x > base.max(p.lit_sigma * rough))).collect();
    let segs: Vec<Option<f32>> = segs.iter().map(|v| v.map(|(x, _)| x)).collect();
    let first = lit_flags.iter().position(|l| *l == Some(true))?;
    let last = lit_flags.iter().rposition(|l| *l == Some(true))?;
    let span: Vec<Option<bool>> = lit_flags[first..=last].to_vec();
    let lit = span.iter().filter(|l| **l == Some(true)).count();
    let known = span.iter().filter(|l| l.is_some()).count().max(1);
    let mut transitions = 0;
    let mut prev: Option<bool> = None;
    for l in span.iter().flatten() {
        if prev.is_some_and(|p| p != *l) {
            transitions += 1;
        }
        prev = Some(*l);
    }
    let mut lit_vals: Vec<f32> = segs[first..=last].iter().zip(&span).filter(|(_, l)| **l == Some(true)).filter_map(|(v, _)| *v).collect();
    let (brightness, _) = median_sigma_mut(&mut lit_vals);
    Some(Verified {
        rho,
        theta,
        t_first: t0 + (first * seg) as f32,
        t_last: t0 + ((last + 1) * seg) as f32,
        lit,
        coverage: lit as f32 / known as f32,
        transitions,
        brightness,
    })
}

/// Find trails in a background-subtracted image.
/// A diffraction spike, not a trail: the line runs through the centre of a
/// saturated star that sits in the middle stretch of the lit part. Spikes
/// from a Newtonian's spider (Alnitak beside IC 434) are long, straight and
/// centred on their star, and fade away from it; a satellite crosses the
/// frame regardless of stars. So a line spanning 70% or more of the frame
/// along its direction is a satellite even through a star (LMC panel 20,
/// 9 Nov 22:27: 82%, through a bright star; Alnitak's spike spans 41%).
pub fn is_diffraction_spike(t: &Trail, saturated: &[(f32, f32)], w: usize, h: usize) -> bool {
    let (dx, dy) = (t.x1 - t.x0, t.y1 - t.y0);
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1.0 {
        return false;
    }
    let (c, s) = ((dx / len).abs(), (dy / len).abs());
    let chord = (if c > 1e-6 { w as f32 / c } else { f32::INFINITY }).min(if s > 1e-6 { h as f32 / s } else { f32::INFINITY });
    if len >= 0.7 * chord {
        return false;
    }
    let (ux, uy) = (dx / len, dy / len);
    saturated.iter().any(|&(sx, sy)| {
        let (px, py) = (sx - t.x0, sy - t.y0);
        let along = px * ux + py * uy;
        let perp = (px * uy - py * ux).abs();
        perp <= 8.0 && along >= 0.2 * len && along <= 0.8 * len
    })
}

pub fn detect(resid: &[f32], w: usize, h: usize, adu_scale: f32, p: &TrailParams) -> Vec<Trail> {
    let timing = std::env::var_os("CRUX_TIMING").is_some();
    let t_all = std::time::Instant::now();
    let lap = |what: &str| {
        if timing {
            eprintln!("  trails {what}: {:.0} ms", t_all.elapsed().as_secs_f64() * 1e3);
        }
    };
    let (v, bw, bh) = bin_residual(resid, w, h, p.bin);
    lap("bin");
    if bw < 64 || bh < 64 {
        return Vec::new();
    }
    let mut sample: Vec<f32> = v.iter().step_by(7).copied().collect();
    let (_, sigma) = median_sigma_mut(&mut sample);
    if sigma <= 0.0 {
        return Vec::new();
    }
    let mask = star_mask(&v, bw, bh, p.star_sigma * sigma, p.star_extent);
    let b = Binned { w: bw, h: bh, v, mask, sigma };
    lap("mask");

    // Candidates at 2× (for line refinement) and at 4× (for the Hough):
    // a 3 px trail has better per-pixel SNR at 4×, and there are 4× fewer
    // pixels to vote.
    let border = 4;
    let thr = p.cand_sigma * sigma;
    let mut cands: Vec<(u16, u16)> = Vec::new();
    for y in border..bh - border {
        for x in border..bw - border {
            let i = y * bw + x;
            if !b.mask[i] && b.v[i] > thr {
                cands.push((x as u16, y as u16));
            }
        }
    }
    let (hw, hh) = (bw / 2, bh / 2);
    let mut v4 = vec![0f32; hw * hh];
    let mut m4 = vec![false; hw * hh];
    for y in 0..hh {
        for x in 0..hw {
            let (i, j) = ((2 * y) * bw + 2 * x, (2 * y + 1) * bw + 2 * x);
            v4[y * hw + x] = 0.25 * (b.v[i] + b.v[i + 1] + b.v[j] + b.v[j + 1]);
            m4[y * hw + x] = b.mask[i] || b.mask[i + 1] || b.mask[j] || b.mask[j + 1];
        }
    }
    let mut s4: Vec<f32> = v4.iter().step_by(7).copied().collect();
    let (_, sigma4) = median_sigma_mut(&mut s4);
    let thr4 = p.cand_sigma * sigma4;
    let border4 = 2;
    let mut hc: Vec<(f32, f32)> = Vec::new();
    let mut valid = 0usize;
    for y in border4..hh - border4 {
        for x in border4..hw - border4 {
            let i = y * hw + x;
            if m4[i] {
                continue;
            }
            valid += 1;
            if v4[i] > thr4 {
                hc.push((x as f32, y as f32));
            }
        }
    }
    drop(v4);
    drop(m4);
    if hc.is_empty() || valid == 0 {
        return Vec::new();
    }
    let density = hc.len() as f32 / valid as f32;
    if timing {
        eprintln!("  trails candidates: {} at 4× ({:.1}%)", hc.len(), density * 100.0);
    }

    // Hough accumulator, filled one θ row at a time so writes stay in cache.
    let step = p.angle_step_deg * 0.5;
    let n_theta = (180.0 / step).round() as usize;
    let diag = ((hw * hw + hh * hh) as f32).sqrt().ceil() as usize;
    let n_rho = 2 * diag + 1;
    let mut acc = vec![0u32; n_theta * n_rho];
    for (k, row) in acc.chunks_exact_mut(n_rho).enumerate() {
        let t = (k as f32 * step).to_radians();
        let (c, s) = (t.cos(), t.sin());
        for &(x, y) in &hc {
            let r = (x * c + y * s + diag as f32 + 0.5) as usize;
            row[r] += 1;
        }
    }
    lap("hough");
    // Peaks by significance over the expected noise votes along each chord.
    let min_votes = 12u32;
    let min_len_b = p.min_length / (2 * p.bin) as f32;
    let (bw_h, bh_h, border_h) = (hw, hh, border4);
    let mut peaks: Vec<(f32, usize, usize)> = Vec::new();
    for k in 0..n_theta {
        let theta = (k as f32 * step).to_radians();
        for ri in 0..n_rho {
            let votes = acc[k * n_rho + ri];
            if votes < min_votes {
                continue;
            }
            let rho = ri as f32 - diag as f32;
            let Some((t0, t1)) = chord(rho, theta, border_h as f32, (bw_h - border_h) as f32, (bh_h - border_h) as f32) else { continue };
            let len = t1 - t0;
            if len < min_len_b {
                continue;
            }
            let exp = density * len;
            let score = (votes as f32 - exp) / (exp + 1.0).sqrt();
            if score >= p.min_score {
                peaks.push((score, k, ri));
            }
        }
    }
    peaks.sort_by(|a, b| b.0.total_cmp(&a.0));
    // Non-maximum suppression in (θ, ρ).
    let mut picked: Vec<(f32, usize, usize)> = Vec::new();
    let dk = (3.0 / step).ceil() as isize;
    for pk in peaks {
        let near = picked.iter().any(|q| {
            let mut d = (pk.1 as isize - q.1 as isize).abs();
            d = d.min(n_theta as isize - d);
            d <= dk && (pk.2 as isize - q.2 as isize).abs() <= 5
        });
        if !near {
            picked.push(pk);
            if picked.len() >= p.max_trails * 3 {
                break;
            }
        }
    }

    lap("peaks");
    let mut trails: Vec<(Verified, f32)> = Vec::new();
    for (score, k, ri) in picked {
        // From 4× to 2× coordinates: x2 = 2·x4 + 0.5.
        let mut theta = (k as f32 * step).to_radians();
        let mut rho = 2.0 * (ri as f32 - diag as f32) + 0.5 * (theta.cos() + theta.sin());
        for band in [3.0, 2.0] {
            if let Some((r, t)) = refine(&cands, rho, theta, band) {
                rho = r;
                theta = t;
            }
        }
        let vf = verify(&b, rho, theta, p);
        if std::env::var_os("CRUX_DEBUG_TRAILS").is_some() {
            eprintln!(
                "trail candidate: score {score:.1} θ {:.2}° ρ {rho:.1} → {}",
                theta.to_degrees(),
                match &vf {
                    Some(v) => format!("lit {} cov {:.2} trans {} bright {:.2}σ", v.lit, v.coverage, v.transitions, v.brightness / b.sigma),
                    None => "no lit segments".into(),
                }
            );
        }
        let Some(vf) = vf else { continue };
        let lit_len = (vf.lit * p.seg * p.bin) as f32;
        if lit_len < p.min_length || vf.coverage < 0.25 {
            continue;
        }
        // Same line found twice, a near-parallel line lit where it grazes a
        // trail, or a plane's paired wing lights: close in angle and close at
        // the new line's lit midpoint.
        let tm = 0.5 * (vf.t_first + vf.t_last);
        let (mx, my) = (vf.rho * vf.theta.cos() - tm * vf.theta.sin(), vf.rho * vf.theta.sin() + tm * vf.theta.cos());
        let dup = trails.iter().any(|(t, _)| {
            let mut da = (t.theta - vf.theta).abs();
            da = da.min(std::f32::consts::PI - da);
            let dist = (mx * t.theta.cos() + my * t.theta.sin() - t.rho).abs();
            da < 3f32.to_radians() && dist < 12.0
        });
        if !dup {
            trails.push((vf, score));
        }
        if trails.len() >= p.max_trails {
            break;
        }
    }

    lap("verify");
    trails
        .into_iter()
        .map(|(vf, score)| {
            let (c, s) = (vf.theta.cos(), vf.theta.sin());
            let (px, py) = (vf.rho * c, vf.rho * s);
            let (dx, dy) = (-s, c);
            let k = p.bin as f32;
            let (x0, y0) = ((px + vf.t_first * dx) * k, (py + vf.t_first * dy) * k);
            let (x1, y1) = ((px + vf.t_last * dx) * k, (py + vf.t_last * dy) * k);
            // Aircraft lights blink: bright dashes with clear gaps. A faint
            // satellite also flickers in and out of the noise, so dashes
            // must be well above the lit threshold (LMC panel 9, 9 Nov 00:14:
            // two faint parallel satellites at 16 ADU).
            let dashed = vf.transitions >= 6 && vf.coverage < 0.8 && vf.brightness >= 3.0 * p.lit_sigma * b.sigma;
            Trail {
                kind: if dashed { TrailKind::Aircraft } else { TrailKind::Satellite },
                angle: dy.atan2(dx).to_degrees().rem_euclid(180.0),
                x0,
                y0,
                x1,
                y1,
                length: (vf.lit * p.seg) as f32 * k,
                brightness_adu: vf.brightness * adu_scale,
                coverage: vf.coverage,
                score,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {

    #[test]
    fn spikes_are_not_trails() {
        let t = Trail {
            kind: TrailKind::Satellite,
            angle: 90.0,
            x0: 3237.0,
            y0: 3944.0,
            x1: 3214.0,
            y1: 2248.0,
            length: 704.0,
            brightness_adu: 33.0,
            coverage: 0.5,
            score: 30.0,
        };
        // Alnitak on the line, mid-way: a spike.
        assert!(is_diffraction_spike(&t, &[(3225.0, 3100.0)], 6248, 4176));
        // A star near one end, or off the line: a satellite.
        assert!(!is_diffraction_spike(&t, &[(3236.0, 3900.0)], 6248, 4176));
        assert!(!is_diffraction_spike(&t, &[(3260.0, 3100.0)], 6248, 4176));
        // Through the star but spanning 82% of the frame: a satellite.
        let long = Trail { y0: 3900.0, y1: 476.0, ..t.clone() };
        assert!(!is_diffraction_spike(&long, &[(3225.0, 2200.0)], 6248, 4176));
    }

    use super::*;
    use crate::background::Background;
    use crate::synth::{render, Params};

    fn run(p: &Params) -> Vec<Trail> {
        let (img, _) = render(p);
        let bg = Background::estimate(&img, 64);
        let mut r = Vec::new();
        bg.subtract(&img, &mut r);
        detect(&r, img.width, img.height, img.adu_scale, &TrailParams { min_length: 400.0, ..Default::default() })
    }

    fn base() -> Params {
        Params { width: 1600, height: 1100, n_stars: 900, ..Default::default() }
    }

    #[test]
    fn clean_frame_has_no_trails() {
        assert!(run(&base()).is_empty());
    }

    #[test]
    fn elongated_stars_are_not_trails() {
        let t = run(&Params { ecc: 0.75, pa_deg: 30.0, ..base() });
        assert!(t.is_empty(), "{t:?}");
        let t = run(&Params { fwhm: 7.0, ..base() });
        assert!(t.is_empty(), "{t:?}");
    }

    #[test]
    fn finds_faint_satellite() {
        // Peak 1.5σ of the per-pixel noise, 3 px wide.
        let t = run(&Params { trail: Some((100.0, 200.0, 23.0, 30.0, 3.0)), ..base() });
        assert_eq!(t.len(), 1, "{t:?}");
        assert!((t[0].angle - 23.0).abs() < 1.0, "{t:?}");
        assert_eq!(t[0].kind, TrailKind::Satellite);
        assert!(t[0].length > 1000.0, "{t:?}");
    }

    #[test]
    fn two_trails_and_steep_angle() {
        let mut p = Params { trail: Some((800.0, 0.0, 95.0, 60.0, 3.0)), ..base() };
        let (mut img, _) = render(&p);
        p.trail = Some((0.0, 900.0, -12.0, 60.0, 3.0));
        p.n_stars = 0;
        p.sky_adu = 0.0;
        p.pedestal_adu = 0.0;
        p.noise_adu = 0.0;
        let (extra, _) = render(&p);
        for (a, b) in img.data.iter_mut().zip(&extra.data) {
            *a += b;
        }
        let bg = Background::estimate(&img, 64);
        let mut r = Vec::new();
        bg.subtract(&img, &mut r);
        let t = detect(&r, img.width, img.height, img.adu_scale, &TrailParams { min_length: 400.0, ..Default::default() });
        assert_eq!(t.len(), 2, "{t:?}");
        let mut angles: Vec<f32> = t.iter().map(|x| x.angle).collect();
        angles.sort_by(|a, b| a.total_cmp(b));
        assert!((angles[0] - 95.0).abs() < 1.0 && (angles[1] - 168.0).abs() < 1.0, "{angles:?}");
    }

    #[test]
    fn dashed_trail_is_aircraft() {
        let t = run(&Params { trail: Some((0.0, 100.0, 30.0, 120.0, 3.0)), trail_dash: Some(120.0), ..base() });
        assert!(!t.is_empty(), "{t:?}");
        assert_eq!(t[0].kind, TrailKind::Aircraft, "{t:?}");
    }
}
