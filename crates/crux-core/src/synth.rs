//! Synthetic frames for tests: elliptical Gaussian stars on a noisy sky,
//! with optional blur, elongation, gradient, trail and transparency loss.
//! Deterministic for a given seed.

use crate::image::Image;

/// Small xorshift RNG; no external dependency, reproducible.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Standard normal (Box–Muller).
    pub fn gauss(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

#[derive(Debug, Clone)]
pub struct Params {
    pub width: usize,
    pub height: usize,
    pub n_stars: usize,
    /// Mean FWHM in pixels (geometric mean of the axes).
    pub fwhm: f64,
    /// Eccentricity sqrt(1 - b²/a²) applied to every star (tracking error).
    pub ecc: f64,
    /// Position angle of the major axis, degrees from +x toward +y.
    pub pa_deg: f64,
    /// Sky level in ADU above the pedestal.
    pub sky_adu: f64,
    pub pedestal_adu: f64,
    /// Gaussian noise sigma in ADU.
    pub noise_adu: f64,
    /// Fractional sky change across the frame width (left → right).
    pub gradient: f64,
    /// Star flux multiplier (1.0 = clear).
    pub transparency: f64,
    /// Brightest star peak in ADU at FWHM 3.5 px (flux is conserved as FWHM changes).
    pub max_peak_adu: f64,
    /// Optional trail: (x0, y0, angle_deg, peak_adu, width_px).
    pub trail: Option<(f64, f64, f64, f64, f64)>,
    /// Star field seed (positions and fluxes).
    pub field_seed: u64,
    /// Noise seed.
    pub noise_seed: u64,
    /// Shift of the whole star field (dither).
    pub shift: (f64, f64),
    /// Patchy cloud: (fraction of the width from the left, star flux factor there).
    pub cloud_patch: Option<(f64, f64)>,
    /// Dashed trail (aircraft lights): on/off period in pixels along the trail.
    pub trail_dash: Option<f64>,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            width: 1024,
            height: 768,
            n_stars: 400,
            fwhm: 3.5,
            ecc: 0.0,
            pa_deg: 0.0,
            sky_adu: 300.0,
            pedestal_adu: 500.0,
            noise_adu: 20.0,
            gradient: 0.0,
            transparency: 1.0,
            max_peak_adu: 30000.0,
            trail: None,
            field_seed: 1,
            noise_seed: 2,
            shift: (0.0, 0.0),
            cloud_patch: None,
            trail_dash: None,
        }
    }
}

/// A rendered star's truth.
#[derive(Debug, Clone, Copy)]
pub struct TrueStar {
    pub x: f64,
    pub y: f64,
    pub peak_adu: f64,
}

pub fn render(p: &Params) -> (Image, Vec<TrueStar>) {
    let (w, h) = (p.width, p.height);
    let mut adu = vec![0f64; w * h];
    for y in 0..h {
        for x in 0..w {
            adu[y * w + x] = p.pedestal_adu + p.sky_adu * (1.0 + p.gradient * (x as f64 / w as f64 - 0.5));
        }
    }
    // Covariance of an elliptical Gaussian with the given FWHM (geometric
    // mean of axes), eccentricity and position angle.
    let s = p.fwhm / 2.354_820_045;
    let ratio = (1.0 - p.ecc * p.ecc).sqrt(); // b/a
    let (sa, sb) = (s / ratio.sqrt(), s * ratio.sqrt());
    let t = p.pa_deg.to_radians();
    let (c, sn) = (t.cos(), t.sin());
    let cxx = sa * sa * c * c + sb * sb * sn * sn;
    let cyy = sa * sa * sn * sn + sb * sb * c * c;
    let cxy = (sa * sa - sb * sb) * c * sn;
    let det = cxx * cyy - cxy * cxy;
    let (ixx, iyy, ixy) = (cyy / det, cxx / det, -cxy / det);
    let rad = (4.0 * sa).ceil() as i64 + 1;

    let mut field = Rng::new(p.field_seed);
    let mut truth = Vec::with_capacity(p.n_stars);
    for _ in 0..p.n_stars {
        let x = field.uniform() * w as f64 + p.shift.0;
        let y = field.uniform() * h as f64 + p.shift.1;
        // Power-law brightness: many faint, few bright.
        // Seeing conserves flux: peaks scale with 1/FWHM² (reference 3.5 px).
        let mut peak = p.max_peak_adu * field.uniform().powf(3.0).max(0.002) * p.transparency * (3.5 / p.fwhm).powi(2);
        if let Some((frac, factor)) = p.cloud_patch {
            if x < frac * w as f64 {
                peak *= factor;
            }
        }
        truth.push(TrueStar { x, y, peak_adu: peak });
        let (xi, yi) = (x.round() as i64, y.round() as i64);
        for yy in (yi - rad).max(0)..(yi + rad + 1).min(h as i64) {
            for xx in (xi - rad).max(0)..(xi + rad + 1).min(w as i64) {
                // Supersample 2×2 for small FWHM.
                let mut v = 0.0;
                for (ox, oy) in [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)] {
                    let dx = xx as f64 + ox - x;
                    let dy = yy as f64 + oy - y;
                    v += (-0.5 * (ixx * dx * dx + 2.0 * ixy * dx * dy + iyy * dy * dy)).exp();
                }
                adu[yy as usize * w + xx as usize] += peak * v / 4.0;
            }
        }
    }
    if let Some((x0, y0, ang, peak, width)) = p.trail {
        let (dx, dy) = (ang.to_radians().cos(), ang.to_radians().sin());
        let sw = width / 2.3548;
        for y in 0..h {
            for x in 0..w {
                // Perpendicular distance to the line.
                let d = ((x as f64 - x0) * dy - (y as f64 - y0) * dx).abs();
                if let Some(period) = p.trail_dash {
                    let along = (x as f64 - x0) * dx + (y as f64 - y0) * dy;
                    if along.rem_euclid(period) > period / 2.0 {
                        continue;
                    }
                }
                if d < 4.0 * sw {
                    adu[y * w + x] += peak * (-0.5 * (d / sw).powi(2)).exp();
                }
            }
        }
    }
    let mut rng = Rng::new(p.noise_seed);
    let mut img = Image::new(w, h);
    for (o, a) in img.data.iter_mut().zip(&adu) {
        let v = a + p.noise_adu * rng.gauss();
        *o = (v.clamp(0.0, 65535.0) / 65535.0) as f32;
    }
    (img, truth)
}
