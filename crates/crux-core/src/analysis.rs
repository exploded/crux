//! Per-frame pixel metrics for lights: stars, focus, shape, sky, gradient.
//! Everything here is absolute; comparisons with other frames live in `grade`.

use crate::background::Background;
use crate::classify::FrameType;
use crate::config::Config;
use crate::image::Image;
use crate::stars::{detect_residual, DetectParams, Star, SATURATED};
use crate::stats::median;
use serde::{Deserialize, Serialize};

/// Bump when metrics change meaning; cached results are recomputed.
pub const ANALYSIS_VERSION: u32 = 13;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Region {
    pub fwhm: Option<f32>,
    pub ecc: Option<f32>,
    pub n: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LightMetrics {
    pub width: u32,
    pub height: u32,
    /// Components above the detection threshold.
    pub n_components: u32,
    /// Star-like components (not hot, edge, oversized or blended). Only the
    /// brightest `max_measure` of them are measured.
    pub n_stars: u32,
    /// Clean stars bright enough for shape statistics.
    pub n_shape: u32,
    pub n_saturated: u32,
    pub fwhm: Option<f32>,
    pub hfr: Option<f32>,
    pub ecc: Option<f32>,
    /// Mean major-axis direction (degrees, image coordinates).
    pub pa: Option<f32>,
    /// Alignment of elongation directions: 0 random, 1 all identical.
    pub coherence: Option<f32>,
    /// +1 elongation points away from the centre, −1 tangential, 0 neither.
    pub radial: Option<f32>,
    /// 3×3 grid, row-major from the top-left.
    pub regions: Vec<Region>,
    /// Median of the background mesh, ADU (includes the bias pedestal).
    pub sky_adu: f32,
    pub noise_adu: f32,
    /// Fitted sky plane: ADU change across the full width and height.
    pub grad_x_adu: f32,
    pub grad_y_adu: f32,
    /// Brightest clean stars, for transparency matching and overlays.
    pub stars: Vec<Star>,
    /// Satellite and aircraft trails.
    #[serde(default)]
    pub trails: Vec<crate::trails::Trail>,
    pub seconds: f32,
}

impl LightMetrics {
    pub fn centre(&self) -> &Region {
        &self.regions[4]
    }

    /// Mean of the four corner regions' eccentricity.
    pub fn corner_ecc(&self) -> Option<f32> {
        let v: Vec<f32> = [0, 2, 6, 8].iter().filter_map(|&i| self.regions.get(i).and_then(|r| r.ecc)).collect();
        (v.len() >= 3).then(|| v.iter().sum::<f32>() / v.len() as f32)
    }

    pub fn corner_fwhm(&self) -> Option<f32> {
        let v: Vec<f32> = [0, 2, 6, 8].iter().filter_map(|&i| self.regions.get(i).and_then(|r| r.fwhm)).collect();
        (v.len() >= 3).then(|| v.iter().sum::<f32>() / v.len() as f32)
    }

    /// What the pixels say this frame is. Only meaningful for gross
    /// mismatches (a "light" with no stars at the bias level).
    pub fn pixel_type(&self, pedestal_adu: f32, exptime: Option<f64>) -> FrameType {
        let signal = self.sky_adu - pedestal_adu;
        if self.n_stars >= 20 {
            FrameType::Light
        } else if signal > 0.15 * 65535.0 {
            FrameType::Flat
        } else if signal.abs() < 30.0 && exptime.unwrap_or(1.0) < 0.01 {
            FrameType::Bias
        } else if signal < 60.0 {
            FrameType::Dark
        } else {
            FrameType::Unknown
        }
    }
}

pub fn detect_params(cfg: &Config, adu_scale: f32) -> DetectParams {
    let a = &cfg.analysis;
    DetectParams {
        sigma: a.detect_sigma,
        min_area: a.min_star_area,
        max_area: a.max_star_area,
        edge: a.edge_margin,
        saturation: a.saturation_adu / adu_scale,
        max_measure: a.max_measure,
    }
}

/// Centres of saturated blobs of any size (a bright star's core can be far
/// bigger than the star detector's size limit). Saturated pixels are summed
/// in 32 px cells; touching cells form one blob.
pub fn saturated_blobs(img: &Image, level: f32) -> Vec<(f32, f32)> {
    const C: usize = 32;
    let (cw, ch) = (img.width.div_ceil(C), img.height.div_ceil(C));
    let mut cells = vec![(0f64, 0f64, 0u32); cw * ch];
    for (y, row) in img.data.chunks_exact(img.width).enumerate() {
        for (x, &v) in row.iter().enumerate() {
            if v >= level {
                let c = &mut cells[(y / C) * cw + x / C];
                c.0 += x as f64;
                c.1 += y as f64;
                c.2 += 1;
            }
        }
    }
    let mut seen = vec![false; cells.len()];
    let mut out = Vec::new();
    for start in 0..cells.len() {
        if seen[start] || cells[start].2 == 0 {
            continue;
        }
        let (mut sx, mut sy, mut n) = (0f64, 0f64, 0u32);
        let mut stack = vec![start];
        seen[start] = true;
        while let Some(i) = stack.pop() {
            let c = cells[i];
            sx += c.0;
            sy += c.1;
            n += c.2;
            let (cx, cy) = ((i % cw) as isize, (i / cw) as isize);
            for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)] {
                let (nx, ny) = (cx + dx, cy + dy);
                if nx >= 0 && ny >= 0 && (nx as usize) < cw && (ny as usize) < ch {
                    let j = ny as usize * cw + nx as usize;
                    if !seen[j] && cells[j].2 > 0 {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
        }
        // Single hot pixels aren't stars.
        if n >= 9 {
            out.push(((sx / n as f64) as f32, (sy / n as f64) as f32));
        }
    }
    out
}

pub fn analyze_light(img: &Image, cfg: &Config) -> LightMetrics {
    analyze_light_map(img, cfg).0
}

/// Metrics plus the 32x binned map used by the flats-against-lights check.
pub fn analyze_light_map(img: &Image, cfg: &Config) -> (LightMetrics, crate::calib::Map) {
    let t0 = std::time::Instant::now();
    let k = img.adu_scale;
    let bg = Background::estimate(img, cfg.analysis.mesh_cell);
    let mut resid = Vec::new();
    bg.subtract(img, &mut resid);
    let det = detect_residual(img, &resid, &detect_params(cfg, k));
    let trails = if cfg.analysis.trails {
        let tp = crate::trails::TrailParams {
            min_length: cfg.analysis.trail_min_length,
            min_score: cfg.analysis.trail_min_score,
            ..Default::default()
        };
        crate::trails::detect(&resid, img.width, img.height, k, &tp)
    } else {
        Vec::new()
    };
    drop(resid);
    let saturated = saturated_blobs(img, cfg.analysis.saturation_adu / k);
    let trails: Vec<crate::trails::Trail> = trails.into_iter().filter(|t| !crate::trails::is_diffraction_spike(t, &saturated, img.width, img.height)).collect();
    let (w, h) = (img.width as f32, img.height as f32);

    let clean: Vec<&Star> = det.stars.iter().filter(|s| s.is_clean()).collect();
    let shape: Vec<&Star> = clean.iter().copied().filter(|s| s.snr >= cfg.analysis.min_shape_snr).collect();
    let med_of = |v: Vec<f32>| (!v.is_empty()).then(|| median(&v));
    let fwhm = med_of(shape.iter().map(|s| s.fwhm).collect());
    let hfr = med_of(shape.iter().map(|s| s.hfr).collect());
    let ecc = med_of(shape.iter().map(|s| s.ecc).collect());

    // Axial statistics on 2θ, weighted by eccentricity.
    let (mut c2, mut s2, mut we) = (0f64, 0f64, 0f64);
    let (mut rad_sum, mut rad_w) = (0f64, 0f64);
    let half_diag = (w * w + h * h).sqrt() / 2.0;
    for s in &shape {
        let e = s.ecc as f64;
        let t = (s.pa as f64).to_radians();
        c2 += e * (2.0 * t).cos();
        s2 += e * (2.0 * t).sin();
        we += e;
        let (dx, dy) = ((s.x - w / 2.0) as f64, (s.y - h / 2.0) as f64);
        if (dx * dx + dy * dy).sqrt() > 0.3 * half_diag as f64 {
            let phi = dy.atan2(dx);
            rad_sum += e * (2.0 * (t - phi)).cos();
            rad_w += e;
        }
    }
    let enough = shape.len() >= 10 && we > 0.0;
    let coherence = enough.then(|| ((c2 * c2 + s2 * s2).sqrt() / we) as f32);
    let pa = enough.then(|| (0.5 * s2.atan2(c2)).to_degrees() as f32);
    let radial = (rad_w > 0.0 && shape.len() >= 10).then(|| (rad_sum / rad_w) as f32);

    let mut regions = vec![Region::default(); 9];
    let mut buckets: Vec<(Vec<f32>, Vec<f32>)> = vec![(Vec::new(), Vec::new()); 9];
    for s in &shape {
        let gx = ((s.x / w * 3.0) as usize).min(2);
        let gy = ((s.y / h * 3.0) as usize).min(2);
        buckets[gy * 3 + gx].0.push(s.fwhm);
        buckets[gy * 3 + gx].1.push(s.ecc);
    }
    for (r, (f, e)) in regions.iter_mut().zip(buckets) {
        r.n = f.len() as u32;
        if f.len() >= 5 {
            r.fwhm = Some(median(&f));
            r.ecc = Some(median(&e));
        }
    }

    let plane = bg.plane();
    // Kept for matching and solving: the brightest clean stars, plus the
    // brightest saturated ones (flagged; good positions, clipped fluxes, so
    // photometry skips them). On a bright sky the clean list alone starts
    // hundreds of stars down the catalogue and differs between filters.
    // In a crowded field with many clipped stars the brightest clean ones
    // are a narrow band just below clipping, and the band moves with focus
    // (a softer frame clips fewer), so two frames shared ~10% of their
    // lists (NGC 3372 on an ASI294MC: 24 of 300). Keep going while stars
    // are within 3× of the 15th brightest clean one (the very brightest
    // clean star jumps between frames), up to 4× `keep_stars`.
    let mut stars: Vec<Star> = clean.iter().map(|s| **s).collect();
    stars.sort_by(|a, b| b.flux.total_cmp(&a.flux));
    let keep = cfg.analysis.keep_stars;
    let floor = stars.get((keep / 20).min(stars.len().saturating_sub(1))).map_or(0.0, |s| s.flux / 3.0);
    let n = keep.max(stars.iter().take(4 * keep).take_while(|s| s.flux >= floor).count());
    stars.truncate(n);
    let mut sat: Vec<Star> = det.stars.iter().filter(|s| s.flags == SATURATED && s.flux > 0.0).copied().collect();
    sat.sort_by(|a, b| b.flux.total_cmp(&a.flux));
    sat.truncate(cfg.analysis.keep_saturated);
    stars.extend(sat);
    stars.sort_by(|a, b| b.flux.total_cmp(&a.flux));

    let map = crate::calib::Map::bin(img, crate::calib::MAP_BIN);
    let m = LightMetrics {
        width: img.width as u32,
        height: img.height as u32,
        n_components: det.n_components as u32,
        n_stars: det.n_starlike as u32,
        n_shape: shape.len() as u32,
        n_saturated: det.stars.iter().filter(|s| s.flags & SATURATED != 0).count() as u32,
        fwhm,
        hfr,
        ecc,
        pa,
        coherence,
        radial,
        regions,
        sky_adu: bg.global_level * k,
        noise_adu: det.noise * k,
        grad_x_adu: plane.map(|p| (p.b * img.width as f64) as f32 * k).unwrap_or(0.0),
        grad_y_adu: plane.map(|p| (p.c * img.height as f64) as f32 * k).unwrap_or(0.0),
        stars,
        trails,
        seconds: t0.elapsed().as_secs_f32(),
    };
    (m, map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::{render, Params};

    #[test]
    fn tracking_signature() {
        let cfg = Config::default();
        let (img, _) = render(&Params { ecc: 0.65, pa_deg: -40.0, ..Default::default() });
        let m = analyze_light(&img, &cfg);
        assert!(m.coherence.unwrap() > 0.85, "{:?}", m.coherence);
        assert!((m.pa.unwrap() + 40.0).abs() < 4.0, "{:?}", m.pa);
        assert!((m.ecc.unwrap() - 0.65).abs() < 0.06);

        let (img, _) = render(&Params::default());
        let m = analyze_light(&img, &cfg);
        assert!(m.coherence.unwrap() < 0.5, "{:?}", m.coherence);
        assert_eq!(m.pixel_type(500.0, Some(300.0)), FrameType::Light);
    }

    #[test]
    fn sky_and_gradient() {
        let cfg = Config::default();
        let (img, _) = render(&Params { sky_adu: 400.0, gradient: 0.2, ..Default::default() });
        let m = analyze_light(&img, &cfg);
        assert!((m.sky_adu - 900.0).abs() < 10.0, "{}", m.sky_adu);
        assert!((m.grad_x_adu - 80.0).abs() < 8.0, "{}", m.grad_x_adu);
        assert!(m.grad_y_adu.abs() < 8.0, "{}", m.grad_y_adu);
    }

    #[test]
    fn dark_frame_reads_as_dark() {
        let cfg = Config::default();
        let (img, _) = render(&Params { n_stars: 0, sky_adu: 2.0, noise_adu: 8.0, ..Default::default() });
        let m = analyze_light(&img, &cfg);
        assert_eq!(m.pixel_type(500.0, Some(300.0)), FrameType::Dark);
    }
}
