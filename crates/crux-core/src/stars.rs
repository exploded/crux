//! Star detection and measurement.
//!
//! 1. Subtract the mesh background, smooth with a σ = 1 px Gaussian.
//! 2. Threshold at k σ of the smoothed noise; 8-connected components.
//! 3. Drop edge, oversized and blended components; flag saturation.
//! 4. Measure each star with adaptive Gaussian moments (the weight's
//!    covariance converges to twice the weighted moments), giving centroid,
//!    FWHM (geometric mean of the axes), eccentricity and position angle.
//!    HFR uses N.I.N.A.'s definition: Σ(r·flux)/Σflux.

use crate::background::Background;
use crate::image::Image;
use crate::stats::median_sigma_mut;
use serde::{Deserialize, Serialize};

pub const SATURATED: u8 = 1;
pub const EDGE: u8 = 2;
pub const BLENDED: u8 = 4;
/// Too sharp to be a star: hot pixel or cosmic ray.
pub const HOT: u8 = 8;
/// Moments didn't converge.
pub const FAILED: u8 = 16;

const FWHM_PER_SIGMA: f32 = 2.354_820;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Star {
    pub x: f32,
    pub y: f32,
    /// Background-subtracted flux in the aperture (normalised units).
    pub flux: f32,
    /// Peak above background (normalised units).
    pub peak: f32,
    pub fwhm: f32,
    pub hfr: f32,
    pub ecc: f32,
    /// Major-axis position angle, degrees in (-90, 90], from +x toward +y.
    pub pa: f32,
    pub snr: f32,
    pub area: u32,
    pub flags: u8,
}

impl Star {
    pub fn is_clean(&self) -> bool {
        self.flags == 0
    }
}

#[derive(Debug, Clone)]
pub struct DetectParams {
    /// Detection threshold in σ of the smoothed image.
    pub sigma: f32,
    pub min_area: usize,
    /// Larger components are nebula knots, galaxies or bloated bright stars.
    pub max_area: usize,
    /// Ignore stars within this many pixels of the border.
    pub edge: usize,
    /// Raw value (normalised) at or above which a star counts as saturated.
    pub saturation: f32,
    /// Measure at most this many components (brightest first).
    pub max_measure: usize,
}

impl Default for DetectParams {
    fn default() -> Self {
        DetectParams { sigma: 5.0, min_area: 5, max_area: 3000, edge: 16, saturation: 60000.0 / 65535.0, max_measure: 1500 }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Detection {
    pub stars: Vec<Star>,
    /// Components above threshold, before any filtering.
    pub n_components: usize,
    /// Components rejected as hot pixels or cosmic rays (not in `stars`).
    pub n_hot: usize,
    /// Star-like components (not hot, edge, oversized or blended), measured
    /// or not. The star count used for grading.
    pub n_starlike: usize,
    /// Noise σ of the unsmoothed residual (normalised).
    pub noise: f32,
}

/// Gaussian kernel, σ = 1 px, radius 3.
fn kernel() -> [f32; 7] {
    let mut k = [0f32; 7];
    for (i, v) in k.iter_mut().enumerate() {
        let d = i as f32 - 3.0;
        *v = (-0.5 * d * d).exp();
    }
    let s: f32 = k.iter().sum();
    k.map(|v| v / s)
}

/// Separable 7-tap blur in place, clamped edges. Vertical pass keeps a
/// small ring of original rows instead of a second full buffer.
pub fn blur7_inplace(data: &mut [f32], w: usize, h: usize) {
    let k = kernel();
    let r = 3usize;
    let mut row = vec![0f32; w + 2 * r];
    for y in 0..h {
        let line = &mut data[y * w..(y + 1) * w];
        row[r..r + w].copy_from_slice(line);
        for i in 0..r {
            row[i] = line[0];
            row[r + w + i] = line[w - 1];
        }
        for x in 0..w {
            let s = &row[x..x + 7];
            line[x] = s[0] * k[0] + s[1] * k[1] + s[2] * k[2] + s[3] * k[3] + s[4] * k[4] + s[5] * k[5] + s[6] * k[6];
        }
    }
    let mut hist = vec![vec![0f32; w]; r];
    let mut acc = vec![0f32; w];
    for y in 0..h {
        acc.fill(0.0);
        for (j, &kj) in k.iter().enumerate() {
            let yy = (y as isize + j as isize - r as isize).clamp(0, h as isize - 1) as usize;
            let src: &[f32] = if yy < y { &hist[yy % r] } else { &data[yy * w..(yy + 1) * w] };
            for (a, &s) in acc.iter_mut().zip(src) {
                *a += kj * s;
            }
        }
        hist[y % r].copy_from_slice(&data[y * w..(y + 1) * w]);
        data[y * w..(y + 1) * w].copy_from_slice(&acc);
    }
}

struct Component {
    area: usize,
    /// Sum of smoothed values above threshold: a flux proxy.
    sum: f32,
    peak: f32,
    px: usize,
    py: usize,
    raw_max: f32,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    maxima: Vec<(usize, usize, f32)>,
}

pub fn detect(img: &Image, bg: &Background, p: &DetectParams) -> Detection {
    let mut resid = Vec::new();
    bg.subtract(img, &mut resid);
    detect_residual(img, &resid, p)
}

/// Detection on a precomputed background-subtracted image (`img` is still
/// needed for raw values, to flag saturation).
pub fn detect_residual(img: &Image, resid: &[f32], p: &DetectParams) -> Detection {
    let (w, h) = (img.width, img.height);
    let mut smooth = resid.to_vec();
    blur7_inplace(&mut smooth, w, h);

    let mut sample: Vec<f32> = resid.iter().step_by(11).copied().collect();
    let (_, noise) = median_sigma_mut(&mut sample);
    let mut sample: Vec<f32> = smooth.iter().step_by(11).copied().collect();
    let (_, snoise) = median_sigma_mut(&mut sample);
    let t = p.sigma * snoise.max(1e-9);

    let t_start = std::time::Instant::now();
    let mut visited = vec![false; w * h];
    let mut stack: Vec<u32> = Vec::new();
    let mut comps: Vec<Component> = Vec::new();
    for start in 0..w * h {
        if visited[start] || smooth[start] <= t {
            continue;
        }
        visited[start] = true;
        stack.clear();
        stack.push(start as u32);
        let mut c = Component {
            area: 0,
            sum: 0.0,
            peak: f32::MIN,
            px: 0,
            py: 0,
            raw_max: f32::MIN,
            x0: usize::MAX,
            y0: usize::MAX,
            x1: 0,
            y1: 0,
            maxima: Vec::new(),
        };
        while let Some(i) = stack.pop() {
            let i = i as usize;
            let (x, y) = (i % w, i / w);
            c.area += 1;
            let v = smooth[i];
            c.sum += v;
            if v > c.peak {
                c.peak = v;
                c.px = x;
                c.py = y;
            }
            c.raw_max = c.raw_max.max(img.data[i]);
            c.x0 = c.x0.min(x);
            c.y0 = c.y0.min(y);
            c.x1 = c.x1.max(x);
            c.y1 = c.y1.max(y);
            let mut is_max = v > 2.0 * t;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let (xx, yy) = (x as i32 + dx, y as i32 + dy);
                    if xx < 0 || yy < 0 || xx >= w as i32 || yy >= h as i32 {
                        continue;
                    }
                    let j = yy as usize * w + xx as usize;
                    let vj = smooth[j];
                    if vj > v {
                        is_max = false;
                    }
                    if !visited[j] && vj > t {
                        visited[j] = true;
                        stack.push(j as u32);
                    }
                }
            }
            if is_max && c.maxima.len() < 16 {
                c.maxima.push((x, y, v));
            }
        }
        if c.area >= p.min_area {
            comps.push(c);
        }
    }
    let n_components = comps.len();
    let timing = std::env::var_os("CRUX_TIMING").is_some();
    if timing {
        eprintln!("  label: {:.0} ms", t_start.elapsed().as_secs_f64() * 1e3);
    }
    let t_measure = std::time::Instant::now();

    // Brightest first by integrated flux, not peak: peaks fall with poor
    // focus, so ranking by peak drops stars from soft corners (seen on the
    // LMC data as fake "lost" cells in patchy-cloud checks).
    comps.sort_by(|a, b| b.sum.total_cmp(&a.sum));
    let mut stars = Vec::new();
    let mut measured = 0usize;
    let mut measured_other = 0usize;
    let mut n_hot = 0usize;
    let mut n_starlike = 0usize;
    for c in &comps {
        let mut flags = 0u8;
        if c.x0 < p.edge || c.y0 < p.edge || c.x1 + p.edge >= w || c.y1 + p.edge >= h {
            flags |= EDGE;
        }
        if c.area > p.max_area {
            continue;
        }
        if c.raw_max >= p.saturation {
            flags |= SATURATED;
        }
        let secondary = c.maxima.iter().any(|&(x, y, v)| {
            let d2 = (x as f32 - c.px as f32).powi(2) + (y as f32 - c.py as f32).powi(2);
            d2 > 6.25 && v > 0.25 * c.peak
        });
        if secondary {
            flags |= BLENDED;
        }
        if flags & EDGE != 0 {
            continue;
        }
        // Sharpness: a real star (FWHM ≥ 2 px) has 4-neighbours at ≥ ~50%
        // of its peak; a hot pixel or cosmic ray has almost nothing. Checked
        // before measuring: in undarked lights hot pixels have the highest
        // peaks and would otherwise use up the measurement budget.
        let (px, py) = (c.px, c.py);
        let pk = resid[py * w + px];
        let nb = (resid[py * w + px - 1] + resid[py * w + px + 1] + resid[(py - 1) * w + px] + resid[(py + 1) * w + px]) / 4.0;
        if pk > 0.0 && nb < 0.2 * pk {
            n_hot += 1;
            continue;
        }
        if flags & BLENDED == 0 {
            n_starlike += 1;
        }
        // Only the brightest are measured; the rest are just counted.
        // Saturated and blended stars have their own budget: on a bright
        // sky (2012 Luminance, sky 16,700 ADU) 1,487 of the brightest 1,500
        // were saturated, leaving nothing for shape or matching.
        if flags & (SATURATED | BLENDED) == 0 {
            if measured >= p.max_measure {
                continue;
            }
            measured += 1;
        } else {
            if measured_other >= p.max_measure {
                continue;
            }
            measured_other += 1;
        }
        let s0 = ((c.area as f32 / std::f32::consts::PI).sqrt() / 1.5).clamp(1.0, 8.0);
        match measure(resid, w, h, c.px as f32, c.py as f32, s0, noise) {
            Some(mut s) => {
                s.flags |= flags;
                s.area = c.area as u32;
                stars.push(s);
            }
            None => stars.push(Star {
                x: c.px as f32,
                y: c.py as f32,
                flux: 0.0,
                peak: c.peak,
                fwhm: 0.0,
                hfr: 0.0,
                ecc: 0.0,
                pa: 0.0,
                snr: 0.0,
                area: c.area as u32,
                // Moments collapse on a single-pixel spike.
                flags: flags | FAILED | if c.area <= 30 { HOT } else { 0 },
            }),
        }
    }
    if timing {
        eprintln!("  measure: {:.0} ms ({measured} + {measured_other} measured)", t_measure.elapsed().as_secs_f64() * 1e3);
    }
    Detection { stars, n_components, n_hot, n_starlike, noise }
}

/// Adaptive-moment measurement around (x0, y0) on a background-subtracted image.
pub fn measure(r: &[f32], w: usize, h: usize, x0: f32, y0: f32, s0: f32, noise: f32) -> Option<Star> {
    let (mut cx, mut cy) = (x0 + 0.5, y0 + 0.5); // pixel centres at +0.5
    let (mut wxx, mut wyy, mut wxy) = (s0 * s0, s0 * s0, 0f32);
    let mut converged = false;
    let mut m = (0f32, 0f32, 0f32);
    let mut s_0 = 0f32;
    for _ in 0..25 {
        let det = wxx * wyy - wxy * wxy;
        if det <= 0.0 {
            return None;
        }
        let (ixx, iyy, ixy) = (wyy / det, wxx / det, -wxy / det);
        let rad = (4.0 * wxx.max(wyy).sqrt()).ceil().clamp(3.0, 30.0) as i32;
        let (xi, yi) = (cx.floor() as i32, cy.floor() as i32);
        let (mut s0w, mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
        for yy in (yi - rad).max(0)..=(yi + rad).min(h as i32 - 1) {
            let dy = yy as f32 + 0.5 - cy;
            let row = &r[yy as usize * w..];
            for xx in (xi - rad).max(0)..=(xi + rad).min(w as i32 - 1) {
                let dx = xx as f32 + 0.5 - cx;
                let q = ixx * dx * dx + 2.0 * ixy * dx * dy + iyy * dy * dy;
                if q > 16.0 {
                    continue;
                }
                let wv = ((-0.5 * q).exp() * row[xx as usize]) as f64;
                s0w += wv;
                sx += wv * dx as f64;
                sy += wv * dy as f64;
                sxx += wv * (dx * dx) as f64;
                syy += wv * (dy * dy) as f64;
                sxy += wv * (dx * dy) as f64;
            }
        }
        if s0w <= 0.0 {
            return None;
        }
        let (mx, my) = ((sx / s0w) as f32, (sy / s0w) as f32);
        let mxx = (sxx / s0w) as f32 - mx * mx;
        let myy = (syy / s0w) as f32 - my * my;
        let mxy = (sxy / s0w) as f32 - mx * my;
        if mxx <= 0.0 || myy <= 0.0 || mxx * myy - mxy * mxy <= 0.0 {
            return None;
        }
        cx += mx;
        cy += my;
        if (cx - x0 - 0.5).abs() > 6.0 || (cy - y0 - 0.5).abs() > 6.0 {
            return None;
        }
        let (nxx, nyy, nxy) = ((2.0 * mxx).max(0.25), (2.0 * myy).max(0.25), 2.0 * mxy);
        let change = (nxx - wxx).abs() / wxx + (nyy - wyy).abs() / wyy + (nxy - wxy).abs() / wxx.max(wyy);
        wxx = nxx;
        wyy = nyy;
        wxy = nxy;
        m = (mxx, myy, mxy);
        s_0 = s0w as f32;
        if change < 1e-3 && mx.abs() < 1e-2 && my.abs() < 1e-2 {
            converged = true;
            break;
        }
        if wxx > 400.0 || wyy > 400.0 {
            return None;
        }
    }
    let _ = s_0;
    // Covariance of the underlying Gaussian is twice the weighted moments.
    let (cxx, cyy, cxy) = (2.0 * m.0, 2.0 * m.1, 2.0 * m.2);
    let tr = cxx + cyy;
    let disc = ((cxx - cyy).powi(2) + 4.0 * cxy * cxy).sqrt();
    let l1 = 0.5 * (tr + disc);
    let l2 = (0.5 * (tr - disc)).max(1e-6);
    let fwhm = FWHM_PER_SIGMA * (l1 * l2).sqrt().sqrt();
    let ecc = (1.0 - l2 / l1).max(0.0).sqrt();
    let pa = 0.5 * (2.0 * cxy).atan2(cxx - cyy).to_degrees();

    // Aperture photometry and HFR.
    let ap = (2.5 * fwhm).max(5.0);
    let rad = ap.ceil() as i32;
    let (xi, yi) = (cx.floor() as i32, cy.floor() as i32);
    let (mut flux, mut pos, mut rsum, mut npix, mut peak) = (0f64, 0f64, 0f64, 0usize, f32::MIN);
    for yy in (yi - rad).max(0)..=(yi + rad).min(h as i32 - 1) {
        let dy = yy as f32 + 0.5 - cy;
        for xx in (xi - rad).max(0)..=(xi + rad).min(w as i32 - 1) {
            let dx = xx as f32 + 0.5 - cx;
            let d = (dx * dx + dy * dy).sqrt();
            if d > ap {
                continue;
            }
            let v = r[yy as usize * w + xx as usize];
            flux += v as f64;
            npix += 1;
            peak = peak.max(v);
            if v > 0.0 {
                pos += v as f64;
                rsum += (v * d) as f64;
            }
        }
    }
    let hfr = if pos > 0.0 { (rsum / pos) as f32 } else { 0.0 };
    let snr = flux as f32 / (noise.max(1e-9) * (npix as f32).sqrt());
    let mut flags = 0;
    if !converged {
        flags |= FAILED;
    }
    if fwhm < 1.2 {
        flags |= HOT;
    }
    Some(Star { x: cx, y: cy, flux: flux as f32, peak, fwhm, hfr, ecc, pa, snr, area: 0, flags })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::{render, Params};

    fn run(p: &Params) -> Detection {
        let (img, _) = render(p);
        let bg = Background::estimate(&img, 64);
        detect(&img, &bg, &DetectParams::default())
    }

    fn med(v: impl Iterator<Item = f32>) -> f32 {
        crate::stats::median(&v.collect::<Vec<_>>())
    }

    #[test]
    fn blur_preserves_flat_and_sum() {
        let mut d = vec![1f32; 50 * 40];
        blur7_inplace(&mut d, 50, 40);
        assert!(d.iter().all(|v| (v - 1.0).abs() < 1e-5));
        let mut d = vec![0f32; 50 * 40];
        d[20 * 50 + 25] = 1.0;
        blur7_inplace(&mut d, 50, 40);
        let s: f32 = d.iter().sum();
        assert!((s - 1.0).abs() < 1e-4);
    }

    #[test]
    fn measures_round_stars() {
        let d = run(&Params::default());
        let good: Vec<&Star> = d.stars.iter().filter(|s| s.is_clean() && s.snr > 30.0).collect();
        assert!(good.len() > 150, "{} good of {}", good.len(), d.stars.len());
        let fwhm = med(good.iter().map(|s| s.fwhm));
        let ecc = med(good.iter().map(|s| s.ecc));
        assert!((fwhm - 3.5).abs() < 0.25, "fwhm {fwhm}");
        assert!(ecc < 0.2, "ecc {ecc}");
        let hfr = med(good.iter().map(|s| s.hfr));
        assert!(hfr > 1.2 && hfr < 3.0, "hfr {hfr}");
    }

    #[test]
    fn measures_blur_and_elongation() {
        let d = run(&Params { fwhm: 6.0, ..Default::default() });
        let fwhm = med(d.stars.iter().filter(|s| s.is_clean() && s.snr > 30.0).map(|s| s.fwhm));
        assert!((fwhm - 6.0).abs() < 0.4, "fwhm {fwhm}");

        let d = run(&Params { ecc: 0.7, pa_deg: 30.0, ..Default::default() });
        let good: Vec<&Star> = d.stars.iter().filter(|s| s.is_clean() && s.snr > 30.0).collect();
        let ecc = med(good.iter().map(|s| s.ecc));
        let pa = med(good.iter().map(|s| s.pa));
        assert!((ecc - 0.7).abs() < 0.06, "ecc {ecc}");
        assert!((pa - 30.0).abs() < 3.0, "pa {pa}");
    }

    #[test]
    fn flags_hot_pixels_and_saturation() {
        let (mut img, _) = render(&Params { n_stars: 50, ..Default::default() });
        for i in 0..30 {
            let (x, y) = (100 + i * 25, 300);
            img.data[y * img.width + x] = 0.5;
        }
        let bg = Background::estimate(&img, 64);
        let d = detect(&img, &bg, &DetectParams { min_area: 1, ..Default::default() });
        assert!(d.n_hot >= 25, "hot {}", d.n_hot);

        let d = run(&Params { max_peak_adu: 120000.0, ..Default::default() });
        assert!(d.stars.iter().any(|s| s.flags & SATURATED != 0));
    }
}
