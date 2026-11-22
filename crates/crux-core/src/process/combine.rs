//! Light integration (milestone 12): registered lights of one field and
//! filter → a master light, rejection maps and a report.
//!
//! 1. Each frame is measured once: background mesh (for local
//!    normalisation), noise, and aperture photometry of the reference's
//!    stars, whose median flux ratio is the frame's signal scale (thin
//!    cloud, airmass and exposure all show up there).
//! 2. Values are normalised to the reference: `(v − B_i) / s_i + B_ref`,
//!    where `B` is each frame's background model (local, like PixInsight's
//!    LocalNormalization: sky gradients that change through the night are
//!    matched) or its median (global), and `s_i` the signal scale.
//! 3. Per pixel: rejection (WBPP's choice by stack size unless set), then
//!    the weighted mean. Weights default to SNR² = (s_i / σ_i)², in the
//!    spirit of PixInsight's PSF Signal Weight.
//!
//! Frames are read in strips of rows, so memory doesn't grow with the
//! number of frames. Zero pixels (outside a registered frame) don't count.

use super::integrate::{esd_table, reject, Rejection};
use crate::grade::Grade;
use crate::header::Header;
use crate::image::Image;
use anyhow::{Context, Result};
use chrono::NaiveDate;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Normalisation {
    /// Frame median and signal scale.
    Global,
    /// Background mesh and signal scale: matches changing sky gradients.
    Local,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Weighting {
    Equal,
    /// By exposure time.
    Exposure,
    /// (signal ÷ noise)², relative to the reference.
    Snr,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub normalisation: Normalisation,
    pub weighting: Weighting,
    /// `None`: WBPP's choice by stack size (`Rejection::auto_lights`).
    pub rejection: Option<Rejection>,
    /// Background mesh cell for local normalisation, pixels.
    pub local_cell: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { normalisation: Normalisation::Local, weighting: Weighting::Snr, rejection: None, local_cell: 256 }
    }
}

impl Settings {
    pub fn rejection_for(&self, n: usize) -> Rejection {
        self.rejection.unwrap_or_else(|| Rejection::auto_lights(n))
    }
}

/// One registered light.
#[derive(Debug, Clone)]
pub struct Input {
    pub path: PathBuf,
    /// The raw light's file name, for reports.
    pub name: String,
    pub exptime: f64,
    pub fwhm: Option<f32>,
    pub night: Option<NaiveDate>,
    pub grade: Grade,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameReport {
    pub path: PathBuf,
    pub name: String,
    pub exptime: f64,
    pub fwhm: Option<f32>,
    pub night: Option<NaiveDate>,
    pub grade: Grade,
    /// Median level, ADU.
    pub location: f64,
    /// Background noise, ADU.
    pub noise: f64,
    /// Star flux relative to the reference.
    pub signal: f64,
    /// Stars measured for `signal`.
    pub stars: usize,
    /// (signal ÷ noise) relative to the reference.
    pub snr: f64,
    /// 0–1; the best frame is 1.
    pub weight: f64,
    /// Fraction of the frame with data.
    pub coverage: f64,
    /// Fractions of its pixels rejected low and high.
    pub rejected_low: f64,
    pub rejected_high: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhatIf {
    pub label: String,
    pub frames: usize,
    /// SNR change as a fraction (−0.04 = 4% worse).
    pub snr_change: f64,
    /// Weighted FWHM of the frames, pixels.
    pub fwhm: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub object: String,
    pub filter: String,
    pub out: PathBuf,
    pub rejection_low: PathBuf,
    pub rejection_high: PathBuf,
    pub rejection: String,
    pub normalisation: Normalisation,
    pub weighting: Weighting,
    pub reference: usize,
    pub frames: Vec<FrameReport>,
    /// Seconds.
    pub total_exposure: f64,
    pub rejected_total: f64,
    /// Background noise of the reference frame and of the master, ADU
    /// (the master is in the reference's units).
    pub noise_reference: f64,
    pub noise_master: f64,
    /// The median frame's noise after scaling to the reference, ADU. The
    /// gain is measured against it: the reference is the one frame that
    /// isn't resampled, so its per-pixel noise is a little higher.
    #[serde(default)]
    pub noise_typical: f64,
    /// Noise reduction the weights predict (no rejection, no resampling).
    pub snr_gain_expected: f64,
    /// Measured: typical frame noise ÷ master noise.
    pub snr_gain: f64,
    /// The master is worth this many typical frames.
    pub equivalent_frames: f64,
    /// Weighted FWHM of the frames, pixels.
    pub fwhm: Option<f64>,
    pub what_ifs: Vec<WhatIf>,
    /// Pixels clipped to [0, 1].
    pub clamped: u64,
    pub seconds: f64,
}

/// Progress: frames measured, then strips combined.
#[derive(Debug, Clone, Copy)]
pub enum Step {
    Measure { done: usize, total: usize },
    Combine { done: usize, total: usize },
}

/// Background levels on a coarse mesh (clipped medians), empty pixels
/// ignored; cells with under half their pixels are NaN.
#[derive(Debug, Clone)]
struct Mesh {
    cell: usize,
    nx: usize,
    ny: usize,
    w: usize,
    h: usize,
    level: Vec<f32>,
}

impl Mesh {
    fn build(img: &Image, cell: usize) -> Mesh {
        let (w, h) = (img.width, img.height);
        let cell = cell.clamp(16, w.max(h));
        let (nx, ny) = (w.div_ceil(cell), h.div_ceil(cell));
        let step = (cell / 96).max(1);
        let mut level = vec![f32::NAN; nx * ny];
        let mut buf = Vec::new();
        for cy in 0..ny {
            for cx in 0..nx {
                buf.clear();
                let (x0, y0) = (cx * cell, cy * cell);
                let (x1, y1) = ((x0 + cell).min(w), (y0 + cell).min(h));
                let mut total = 0usize;
                for y in (y0..y1).step_by(step) {
                    for &v in img.data[y * w + x0..y * w + x1].iter().step_by(step) {
                        total += 1;
                        if v > 0.0 {
                            buf.push(v);
                        }
                    }
                }
                if buf.len() * 2 >= total && buf.len() >= 16 {
                    level[cy * nx + cx] = crate::stats::clipped_median_sigma(&mut buf, 3.0, 6).0;
                }
            }
        }
        Mesh { cell, nx, ny, w, h, level }
    }

    /// Cell centres in [-1, 1] coordinates, with their levels.
    fn cells(&self) -> impl Iterator<Item = (f64, f64, f32)> + '_ {
        (0..self.ny).flat_map(move |cy| {
            (0..self.nx).map(move |cx| {
                let xc = ((cx * self.cell + ((cx + 1) * self.cell).min(self.w)) as f64 * 0.5) / self.w as f64 * 2.0 - 1.0;
                let yc = ((cy * self.cell + ((cy + 1) * self.cell).min(self.h)) as f64 * 0.5) / self.h as f64 * 2.0 - 1.0;
                (xc, yc, self.level[cy * self.nx + cx])
            })
        })
    }
}

/// A cubic surface in [-1, 1] coordinates: the difference between a
/// frame's background (scaled) and the reference's. Ten terms fitted to
/// hundreds of mesh cells, so cell noise averages out instead of leaving
/// blotches; sky gradients are smooth anyway.
#[derive(Debug, Clone, Copy, Default)]
struct Surface([f64; 10]);

impl Surface {
    fn terms(u: f64, v: f64) -> [f64; 10] {
        [1.0, u, v, u * u, u * v, v * v, u * u * u, u * u * v, u * v * v, v * v * v]
    }

    /// Robust least squares (three rounds of 3σ clipping).
    fn fit(pts: &[(f64, f64, f64)]) -> Option<Surface> {
        let mut keep: Vec<bool> = vec![true; pts.len()];
        let mut sol = None;
        for _ in 0..4 {
            let used: Vec<&(f64, f64, f64)> = pts.iter().zip(&keep).filter(|(_, k)| **k).map(|(p, _)| p).collect();
            if used.len() < 20 {
                break;
            }
            let mut a = [[0f64; 10]; 10];
            let mut b = [0f64; 10];
            for &&(u, v, d) in &used {
                let t = Self::terms(u, v);
                for i in 0..10 {
                    b[i] += t[i] * d;
                    for j in 0..10 {
                        a[i][j] += t[i] * t[j];
                    }
                }
            }
            let Some(c) = solve10(a, b) else { break };
            let s = Surface(c);
            let res: Vec<f64> = pts.iter().map(|&(u, v, d)| d - s.at(u, v)).collect();
            let mut r: Vec<f64> = res.iter().zip(&keep).filter(|(_, k)| **k).map(|(r, _)| r.abs()).collect();
            r.sort_by(|x, y| x.total_cmp(y));
            let sd = 1.4826 * r[r.len() / 2];
            sol = Some(s);
            if sd <= 0.0 {
                break;
            }
            keep = res.iter().map(|r| r.abs() <= 3.0 * sd).collect();
        }
        sol
    }

    fn at(&self, u: f64, v: f64) -> f64 {
        Self::terms(u, v).iter().zip(&self.0).map(|(t, c)| t * c).sum()
    }

    /// Coefficients of the cubic in u along row v.
    fn along(&self, v: f64) -> [f64; 4] {
        let a = &self.0;
        [a[0] + a[2] * v + a[5] * v * v + a[9] * v * v * v, a[1] + a[4] * v + a[8] * v * v, a[3] + a[7] * v, a[6]]
    }
}

/// Gaussian elimination with partial pivoting.
fn solve10(mut a: [[f64; 10]; 10], mut b: [f64; 10]) -> Option<[f64; 10]> {
    for c in 0..10 {
        let p = (c..10).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))?;
        if a[p][c].abs() < 1e-12 {
            return None;
        }
        a.swap(c, p);
        b.swap(c, p);
        for r in c + 1..10 {
            let f = a[r][c] / a[c][c];
            for k in c..10 {
                a[r][k] -= f * a[c][k];
            }
            b[r] -= f * b[c];
        }
    }
    let mut x = [0f64; 10];
    for c in (0..10).rev() {
        let s: f64 = (c + 1..10).map(|k| a[c][k] * x[k]).sum();
        x[c] = (b[c] - s) / a[c][c];
    }
    Some(x)
}

/// Noise, the way PixInsight's k-sigma estimator does it: the finest
/// à trous wavelet layer (image minus its B3-spline smoothing) holds the
/// noise and little of the stars or nebula; its iteratively 3σ-clipped
/// standard deviation, divided by 0.8907 (that layer's share of white
/// noise), is the noise σ. Empty pixels and their neighbours are left out.
/// (A plain clipped σ also counts faint stars and nebula, which dominate
/// once a stack has beaten the noise down.)
pub fn noise_of(img: &Image) -> f32 {
    let (w, h) = (img.width, img.height);
    if w < 8 || h < 8 {
        return 0.0;
    }
    const K: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];
    // Rows, then columns, sampled at every third pixel.
    let rows: Vec<usize> = (2..h - 2).step_by(3).collect();
    let mut tmp = vec![0f32; w * h];
    tmp.par_chunks_mut(w).enumerate().for_each(|(y, out)| {
        let r = &img.data[y * w..(y + 1) * w];
        for x in 2..w - 2 {
            out[x] = K[0] * r[x - 2] + K[1] * r[x - 1] + K[2] * r[x] + K[3] * r[x + 1] + K[4] * r[x + 2];
        }
    });
    let coeffs: Vec<f32> = rows
        .par_iter()
        .flat_map_iter(|&y| {
            let img = &img.data;
            let tmp = &tmp;
            (2..w - 2).step_by(3).filter_map(move |x| {
                // Skip anything touching empty pixels.
                for dy in 0..5 {
                    let yy = y + dy - 2;
                    if img[yy * w + x - 2] <= 0.0 || img[yy * w + x + 2] <= 0.0 || img[yy * w + x] <= 0.0 {
                        return None;
                    }
                }
                let s = K[0] * tmp[(y - 2) * w + x] + K[1] * tmp[(y - 1) * w + x] + K[2] * tmp[y * w + x] + K[3] * tmp[(y + 1) * w + x] + K[4] * tmp[(y + 2) * w + x];
                Some(img[y * w + x] - s)
            })
        })
        .collect();
    if coeffs.len() < 100 {
        return 0.0;
    }
    let mut c = coeffs;
    let (_, sd) = crate::stats::clipped_median_sigma(&mut c, 3.0, 8);
    sd / 0.8907
}

/// One star's aperture and background pixels (indices into the frame),
/// chosen on the reference and used unchanged in every registered frame:
/// a neighbour's light clipped from the clean reference's annulus would
/// hide in a noisy frame's noise and bias its background.
struct Aperture {
    inside: Vec<u32>,
    annulus: Vec<u32>,
}

fn apertures(img: &Image, stars: &[(f32, f32)], r: f32) -> Vec<Option<Aperture>> {
    let (w, h) = (img.width as i64, img.height as i64);
    let (r_in, r_out) = (r + 2.0, r + 6.0);
    let ro = r_out.ceil() as i64;
    stars
        .iter()
        .map(|&(sx, sy)| {
            let (cx, cy) = (sx.round() as i64, sy.round() as i64);
            if cx - ro < 0 || cy - ro < 0 || cx + ro >= w || cy + ro >= h {
                return None;
            }
            let (mut inside, mut ann, mut vals) = (Vec::new(), Vec::new(), Vec::new());
            for y in cy - ro..=cy + ro {
                for x in cx - ro..=cx + ro {
                    let d = ((x as f32 - sx).powi(2) + (y as f32 - sy).powi(2)).sqrt();
                    let k = (y * w + x) as u32;
                    let v = img.data[k as usize];
                    if d <= r {
                        if v <= 0.0 {
                            return None;
                        }
                        inside.push(k);
                    } else if d >= r_in && d <= r_out && v > 0.0 {
                        ann.push(k);
                        vals.push(v);
                    }
                }
            }
            if vals.len() < 10 {
                return None;
            }
            let (m, sd) = crate::stats::clipped_median_sigma(&mut vals.clone(), 3.0, 4);
            let annulus: Vec<u32> = ann.into_iter().filter(|&k| (img.data[k as usize] - m).abs() <= 3.0 * sd.max(1e-9)).collect();
            (annulus.len() >= 10).then_some(Aperture { inside, annulus })
        })
        .collect()
}

/// Background-subtracted aperture fluxes (NaN where the aperture has
/// empty pixels). The background is the annulus mean within 5σ (drops
/// cosmic rays, keeps the noise symmetric).
fn fluxes(img: &Image, aps: &[Option<Aperture>]) -> Vec<f32> {
    let mut b = Vec::new();
    aps.iter()
        .map(|a| {
            let Some(a) = a else { return f32::NAN };
            if a.inside.iter().any(|&k| img.data[k as usize] <= 0.0) {
                return f32::NAN;
            }
            b.clear();
            b.extend(a.annulus.iter().map(|&k| img.data[k as usize]).filter(|v| *v > 0.0));
            if b.len() < 10 {
                return f32::NAN;
            }
            let (m, sd) = crate::stats::clipped_median_sigma(&mut b.clone(), 5.0, 3);
            let kept: Vec<f32> = b.iter().copied().filter(|v| (v - m).abs() <= 5.0 * sd.max(1e-9)).collect();
            let bg = kept.iter().sum::<f32>() / kept.len().max(1) as f32;
            a.inside.iter().map(|&k| img.data[k as usize] - bg).sum::<f32>()
        })
        .collect()
}

/// How `match_to` brought one stack onto another.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Matched {
    /// Star flux of `img` ÷ `target` (1 when too few stars).
    pub scale: f64,
    pub stars: usize,
    /// Background difference removed, normalised units: median and
    /// largest absolute value of the fitted surface at the mesh cells.
    pub offset: f64,
    pub offset_max: f64,
}

/// Bring `img` onto `target`'s brightness and background, for comparing
/// stacks of one field made from different frames (both registered to the
/// same geometry): divide by the median star flux ratio (apertures chosen
/// on `target`, as `integrate` does), then subtract a cubic surface fitted
/// to the background mesh differences. Empty (zero) pixels stay empty.
pub fn match_to(img: &mut Image, target: &Image, stars: &[(f32, f32)], fwhm: f32, cell: usize) -> Matched {
    let r = (2.0 * if fwhm > 0.0 { fwhm } else { 3.0 }).clamp(3.0, 15.0);
    let stars: Vec<(f32, f32)> = stars.iter().take(250).copied().collect();
    let aps = apertures(target, &stars, r);
    let ft = fluxes(target, &aps);
    let fi = fluxes(img, &aps);
    let ratios: Vec<f64> = fi.iter().zip(&ft).filter(|(a, b)| a.is_finite() && b.is_finite() && **a > 0.0 && **b > 0.0).map(|(a, b)| (*a / *b) as f64).collect();
    let scale = if ratios.len() >= 10 { crate::stats::median_f64(&ratios).unwrap_or(1.0) } else { 1.0 };
    let inv = (1.0 / scale.max(1e-9)) as f32;
    img.data.par_iter_mut().for_each(|v| {
        if *v > 0.0 {
            *v *= inv;
        }
    });
    let (mi, mt) = (Mesh::build(img, cell), Mesh::build(target, cell));
    let pts: Vec<(f64, f64, f64)> = mi.cells().zip(mt.cells()).filter(|((_, _, a), (_, _, b))| a.is_finite() && b.is_finite()).map(|((u, v, a), (_, _, b))| (u, v, a as f64 - b as f64)).collect();
    let surface = Surface::fit(&pts).unwrap_or_else(|| {
        let mut d: Vec<f64> = pts.iter().map(|p| p.2).collect();
        let mut s = Surface::default();
        if !d.is_empty() {
            d.sort_by(|a, b| a.total_cmp(b));
            s.0[0] = d[d.len() / 2];
        }
        s
    });
    let mut at: Vec<f64> = pts.iter().map(|&(u, v, _)| surface.at(u, v)).collect();
    let offset_max = at.iter().fold(0f64, |m, x| m.max(x.abs()));
    at.sort_by(|a, b| a.total_cmp(b));
    let offset = at.get(at.len() / 2).copied().unwrap_or(0.0);
    let (w, h) = (img.width, img.height);
    img.data.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let c = surface.along((y as f64 + 0.5) / h as f64 * 2.0 - 1.0);
        for (x, v) in row.iter_mut().enumerate() {
            if *v > 0.0 {
                let u = (x as f64 + 0.5) / w as f64 * 2.0 - 1.0;
                // Keep data pixels non-zero: zero means "no data".
                *v = (*v - (c[0] + u * (c[1] + u * (c[2] + u * c[3]))) as f32).max(1e-7);
            }
        }
    });
    Matched { scale, stars: ratios.len(), offset, offset_max }
}

enum Rows {
    Seek(crate::xisf::XisfLayout),
    Whole(Image),
}

struct Measured {
    rows: Rows,
    location: f32,
    noise: f32,
    mesh: Mesh,
    flux: Vec<f32>,
    coverage: f64,
    scale: f32,
}

fn measure(input: &Input, cell: usize, aps: &[Option<Aperture>]) -> Result<Measured> {
    let img = crate::read_image(&input.path).with_context(|| format!("read {}", input.path.display()))?;
    let valid: Vec<f32> = img.data.iter().step_by(7).copied().filter(|v| *v > 0.0).collect();
    let coverage = valid.len() as f64 / img.data.iter().step_by(7).count().max(1) as f64;
    let mut v = valid;
    let location = if v.is_empty() { 0.0 } else { crate::stats::median_mut(&mut v) };
    let noise = noise_of(&img);
    let mesh = Mesh::build(&img, cell);
    let flux = fluxes(&img, aps);
    let scale = img.adu_scale;
    let is_xisf = input.path.extension().is_some_and(|e| e.eq_ignore_ascii_case("xisf"));
    let rows = match is_xisf.then(|| crate::xisf::read_layout(&input.path).ok()).flatten() {
        Some(l) if l.compression.is_none() && l.channels <= 1 && l.width == img.width && l.height == img.height => Rows::Seek(l),
        _ => Rows::Whole(img),
    };
    Ok(Measured { rows, location, noise, mesh, flux, coverage, scale })
}

/// Integrate `inputs` (same size, registered to `inputs[reference]`).
/// `stars`: the reference's unsaturated stars (x, y), brightest first.
#[allow(clippy::too_many_arguments)]
pub fn integrate(
    inputs: &[Input],
    reference: usize,
    stars: &[(f32, f32)],
    s: &Settings,
    object: &str,
    filter: &str,
    out: &Path,
    extra: &Header,
    progress: &(dyn Fn(Step) + Sync),
    cancel: &AtomicBool,
) -> Result<Report> {
    let t0 = std::time::Instant::now();
    let laps = crate::timing::Laps::new(format!("integrate {object} {filter}"));
    let n = inputs.len();
    anyhow::ensure!(n > 0 && reference < n, "nothing to integrate");
    // Aperture: twice the worst FWHM, 3–15 px.
    let fwhm_max = inputs.iter().filter_map(|i| i.fwhm).fold(0f32, f32::max);
    let r = (2.0 * if fwhm_max > 0.0 { fwhm_max } else { 3.0 }).clamp(3.0, 15.0);
    let stars: Vec<(f32, f32)> = stars.iter().take(250).copied().collect();
    let aps = {
        let ri = crate::read_image(&inputs[reference].path).with_context(|| format!("read {}", inputs[reference].path.display()))?;
        apertures(&ri, &stars, r)
    };

    let done = AtomicUsize::new(0);
    let measured: Vec<Result<Measured>> = inputs
        .par_iter()
        .with_max_len(1)
        .map(|inp| {
            if cancel.load(Ordering::Relaxed) {
                anyhow::bail!("cancelled");
            }
            let m = measure(inp, s.local_cell, &aps);
            progress(Step::Measure { done: done.fetch_add(1, Ordering::Relaxed) + 1, total: n });
            m
        })
        .collect();
    let measured: Vec<Measured> = measured.into_iter().collect::<Result<_>>()?;
    laps.lap(&format!("{n} frames measured"));
    let (w, h) = match &measured[reference].rows {
        Rows::Seek(l) => (l.width, l.height),
        Rows::Whole(i) => (i.width, i.height),
    };
    for (m, inp) in measured.iter().zip(inputs) {
        let (mw, mh) = match &m.rows {
            Rows::Seek(l) => (l.width, l.height),
            Rows::Whole(i) => (i.width, i.height),
        };
        anyhow::ensure!((mw, mh) == (w, h), "{} is {mw}×{mh}, the reference {w}×{h}", inp.path.display());
    }

    // Signal scale: median flux ratio against the reference.
    let rf = &measured[reference].flux;
    let mut n_stars = vec![0usize; n];
    let signal: Vec<f64> = measured
        .iter()
        .enumerate()
        .map(|(i, m)| {
            if i == reference {
                n_stars[i] = rf.iter().filter(|f| f.is_finite() && **f > 0.0).count();
                return 1.0;
            }
            let ratios: Vec<f64> = m.flux.iter().zip(rf).filter(|(a, b)| a.is_finite() && b.is_finite() && **b > 0.0 && **a > 0.0).map(|(a, b)| (*a / *b) as f64).collect();
            n_stars[i] = ratios.len();
            if ratios.len() >= 10 {
                crate::stats::median_f64(&ratios).unwrap_or(1.0)
            } else {
                inputs[i].exptime / inputs[reference].exptime.max(1e-9)
            }
        })
        .collect();
    let noise_ref = measured[reference].noise.max(1e-12) as f64;
    // Noise in the reference's units after scaling.
    let sigma_n: Vec<f64> = measured.iter().zip(&signal).map(|(m, &sg)| m.noise as f64 / sg.max(1e-9)).collect();
    let snr: Vec<f64> = sigma_n.iter().map(|sn| noise_ref / sn.max(1e-12)).collect();
    let mut weight: Vec<f64> = match s.weighting {
        Weighting::Equal => vec![1.0; n],
        Weighting::Exposure => inputs.iter().map(|i| i.exptime).collect(),
        Weighting::Snr => snr.iter().map(|x| x * x).collect(),
    };
    let wmax = weight.iter().cloned().fold(0.0, f64::max).max(1e-12);
    weight.iter_mut().for_each(|x| *x /= wmax);

    let rejection = s.rejection_for(n);
    let esd: Vec<Vec<f32>> = (0..=n)
        .map(|k| match rejection {
            Rejection::Esd { fraction, alpha } => esd_table(k, fraction, alpha),
            _ => Vec::new(),
        })
        .collect();
    let inv_s: Vec<f32> = signal.iter().map(|x| (1.0 / x.max(1e-9)) as f32).collect();
    // v' = v / s_i − D_i: D_i = B_i / s_i − B_ref, a constant (global) or
    // a cubic surface fitted to the meshes (local).
    let global_d = |i: usize| measured[i].location as f64 / signal[i] - measured[reference].location as f64;
    let surfaces: Vec<Surface> = (0..n)
        .map(|i| {
            let mut c = Surface::default();
            c.0[0] = global_d(i);
            if s.normalisation == Normalisation::Global || i == reference {
                return if i == reference { Surface::default() } else { c };
            }
            let pts: Vec<(f64, f64, f64)> = measured[i]
                .mesh
                .cells()
                .zip(measured[reference].mesh.cells())
                .filter(|((_, _, a), (_, _, b))| a.is_finite() && b.is_finite())
                .map(|((u, v, a), (_, _, b))| (u, v, a as f64 / signal[i] - b as f64))
                .collect();
            Surface::fit(&pts).unwrap_or(c)
        })
        .collect();
    let wts: Vec<f32> = weight.iter().map(|x| *x as f32).collect();
    // Each frame's noise after scaling (normalised units), for the noise floor.
    let noise_n: Vec<f32> = sigma_n.iter().map(|x| *x as f32).collect();

    let strip = 32usize;
    let n_strips = h.div_ceil(strip);
    let mut img = vec![0f32; w * h];
    let mut map_lo = vec![0f32; w * h];
    let mut map_hi = vec![0f32; w * h];
    let combined = AtomicUsize::new(0);
    let per_strip: Vec<Result<(Vec<(u64, u64, u64)>, u64)>> = img
        .par_chunks_mut(strip * w)
        .zip(map_lo.par_chunks_mut(strip * w))
        .zip(map_hi.par_chunks_mut(strip * w))
        .enumerate()
        .map(|(si, ((o, lo_m), hi_m))| {
            if cancel.load(Ordering::Relaxed) {
                anyhow::bail!("cancelled");
            }
            let y0 = si * strip;
            let rows = o.len() / w;
            let mut bufs: Vec<Vec<f32>> = Vec::with_capacity(n);
            for (m, inp) in measured.iter().zip(inputs) {
                let mut b = vec![0f32; rows * w];
                match &m.rows {
                    Rows::Seek(l) => crate::xisf::read_rows(&inp.path, l, y0, y0 + rows, &mut b)?,
                    Rows::Whole(i) => b.copy_from_slice(&i.data[y0 * w..(y0 + rows) * w]),
                }
                bufs.push(b);
            }
            let mut counts = vec![(0u64, 0u64, 0u64); n];
            let mut clamped = 0u64;
            let us: Vec<f64> = (0..w).map(|x| (x as f64 + 0.5) / w as f64 * 2.0 - 1.0).collect();
            let mut drow: Vec<Vec<f32>> = vec![vec![0f32; w]; n];
            let mut vals: Vec<(f32, u16)> = Vec::with_capacity(n);
            let mut work: Vec<f32> = Vec::with_capacity(n);
            for yy in 0..rows {
                let y = y0 + yy;
                let vy = (y as f64 + 0.5) / h as f64 * 2.0 - 1.0;
                for (i, sf) in surfaces.iter().enumerate() {
                    let c = sf.along(vy);
                    for (d, &u) in drow[i].iter_mut().zip(&us) {
                        *d = (c[0] + u * (c[1] + u * (c[2] + u * c[3]))) as f32;
                    }
                }
                for x in 0..w {
                    vals.clear();
                    for i in 0..n {
                        let v = bufs[i][yy * w + x];
                        if v > 0.0 {
                            vals.push((v * inv_s[i] - drow[i][x], i as u16));
                            counts[i].2 += 1;
                        }
                    }
                    let k = yy * w + x;
                    if vals.is_empty() {
                        o[k] = 0.0;
                        continue;
                    }
                    let nv = vals.len();
                    let (a, b) = reject(&mut vals, &mut work, rejection, Some(&esd[nv]), Some(&noise_n));
                    for &(_, i) in &vals[..a] {
                        counts[i as usize].0 += 1;
                    }
                    for &(_, i) in &vals[b..] {
                        counts[i as usize].1 += 1;
                    }
                    lo_m[k] = a as f32 / n as f32;
                    hi_m[k] = (vals.len() - b) as f32 / n as f32;
                    let (mut sw, mut sv) = (0f64, 0f64);
                    for &(v, i) in &vals[a..b] {
                        let wt = wts[i as usize].max(1e-6) as f64;
                        sw += wt;
                        sv += wt * v as f64;
                    }
                    let mut v = if sw > 0.0 { (sv / sw) as f32 } else { 0.0 };
                    if !(0.0..=1.0).contains(&v) {
                        clamped += 1;
                        v = v.clamp(0.0, 1.0);
                    }
                    o[k] = v;
                }
            }
            progress(Step::Combine { done: combined.fetch_add(1, Ordering::Relaxed) + 1, total: n_strips });
            Ok((counts, clamped))
        })
        .collect();
    laps.lap("combined");
    let mut counts = vec![(0u64, 0u64, 0u64); n];
    let mut clamped = 0u64;
    for r in per_strip {
        let (c, k) = r?;
        clamped += k;
        for (t, x) in counts.iter_mut().zip(c) {
            t.0 += x.0;
            t.1 += x.1;
            t.2 += x.2;
        }
    }

    let scale = measured[reference].scale;
    let mut master = Image::from_data(w, h, img);
    master.adu_scale = scale;
    let noise_master = noise_of(&master) as f64;
    // Write: master, then the rejection maps (compressed; mostly zeros).
    let low_path = out.with_file_name(format!("{}_rejection_low.xisf", out.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default()));
    let high_path = out.with_file_name(format!("{}_rejection_high.xisf", out.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default()));
    let total_exposure: f64 = inputs.iter().map(|i| i.exptime).sum();
    let rejected_total = counts.iter().map(|c| (c.0 + c.1) as f64).sum::<f64>() / counts.iter().map(|c| c.2 as f64).sum::<f64>().max(1.0);
    let mut hdr = extra.clone();
    set(&mut hdr, "IMAGETYP", "Master Light", "Type of image");
    set(&mut hdr, "NCOMBINE", &n.to_string(), "Frames integrated");
    set(&mut hdr, "TOTALEXP", &format!("{total_exposure:.1}"), "Total exposure, s");
    set(&mut hdr, "CRUX-NRM", match s.normalisation { Normalisation::Local => "local", Normalisation::Global => "global" }, "crux normalisation");
    set(&mut hdr, "CRUX-WGT", match s.weighting { Weighting::Snr => "snr", Weighting::Exposure => "exposure", Weighting::Equal => "equal" }, "crux weights");
    set(&mut hdr, "CRUX-REJ", &rejection.name(), "crux rejection");
    set(&mut hdr, "CRUX-RJT", &format!("{:.3}", rejected_total * 100.0), "Values rejected, %");
    write(out, &master, &hdr, None)?;
    let mut rh = Header::default();
    rh.push("IMAGETYP", "Rejection Map", "");
    rh.push("NCOMBINE", n.to_string(), "Frames integrated");
    write(&low_path, &Image::from_data(w, h, map_lo), &rh, Some("lz4"))?;
    write(&high_path, &Image::from_data(w, h, map_hi), &rh, Some("lz4"))?;

    let frames: Vec<FrameReport> = inputs
        .iter()
        .enumerate()
        .map(|(i, inp)| {
            let c = counts[i];
            let px = c.2.max(1) as f64;
            FrameReport {
                path: inp.path.clone(),
                name: inp.name.clone(),
                exptime: inp.exptime,
                fwhm: inp.fwhm,
                night: inp.night,
                grade: inp.grade,
                location: measured[i].location as f64 * scale as f64,
                noise: measured[i].noise as f64 * scale as f64,
                signal: signal[i],
                stars: n_stars[i],
                snr: snr[i],
                weight: weight[i],
                coverage: measured[i].coverage,
                rejected_low: c.0 as f64 / px,
                rejected_high: c.1 as f64 / px,
            }
        })
        .collect();
    let all: Vec<usize> = (0..n).collect();
    let typical = {
        let mut v = sigma_n.clone();
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    };
    let gain_expected = gain(&sigma_n, &weight, &all) * typical;
    let snr_gain = typical / noise_master.max(1e-12);
    let what_ifs = what_ifs(&frames, &sigma_n, &weight, s.weighting);
    Ok(Report {
        object: object.to_string(),
        filter: filter.to_string(),
        out: out.to_path_buf(),
        rejection_low: low_path,
        rejection_high: high_path,
        rejection: rejection.name(),
        normalisation: s.normalisation,
        weighting: s.weighting,
        reference,
        total_exposure,
        rejected_total,
        noise_reference: noise_ref * scale as f64,
        noise_master: noise_master * scale as f64,
        noise_typical: typical * scale as f64,
        snr_gain_expected: gain_expected,
        snr_gain,
        equivalent_frames: snr_gain * snr_gain,
        fwhm: weighted_fwhm(&frames, &weight, &all),
        what_ifs,
        clamped,
        seconds: t0.elapsed().as_secs_f64(),
        frames,
    })
}

fn set(h: &mut Header, key: &str, value: &str, comment: &str) {
    h.cards.retain(|c| !c.key.eq_ignore_ascii_case(key));
    h.push(key, value, comment);
}

fn write(p: &Path, img: &Image, h: &Header, codec: Option<&str>) -> Result<()> {
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d).with_context(|| format!("create {}", d.display()))?;
    }
    let tmp = p.with_extension("xisf.tmp");
    crate::xisf::write_image(&tmp, img, h, codec)?;
    std::fs::rename(&tmp, p).with_context(|| format!("replace {}", p.display()))?;
    Ok(())
}

/// 1 ÷ noise of the weighted mean of `set` (signal 1 in every frame).
fn gain(sigma: &[f64], w: &[f64], set: &[usize]) -> f64 {
    let sw: f64 = set.iter().map(|&i| w[i]).sum();
    let var: f64 = set.iter().map(|&i| (w[i] * sigma[i]).powi(2)).sum();
    if sw <= 0.0 || var <= 0.0 {
        return 0.0;
    }
    sw / var.sqrt()
}

fn weighted_fwhm(frames: &[FrameReport], w: &[f64], set: &[usize]) -> Option<f64> {
    let (mut s, mut k) = (0.0, 0.0);
    for &i in set {
        if let Some(f) = frames[i].fwhm {
            s += w[i] * f as f64;
            k += w[i];
        }
    }
    (k > 0.0).then(|| s / k)
}

/// What leaving frames out (or weighting differently) would do, from the
/// measured noise and signal alone. Weights are re-derived for each set
/// the way the chosen weighting would.
fn what_ifs(frames: &[FrameReport], sigma: &[f64], w: &[f64], weighting: Weighting) -> Vec<WhatIf> {
    let n = frames.len();
    let all: Vec<usize> = (0..n).collect();
    let base = gain(sigma, w, &all);
    let mut out = Vec::new();
    let mut push = |label: String, set: Vec<usize>, ws: &[f64]| {
        if set.is_empty() || base <= 0.0 {
            return;
        }
        out.push(WhatIf { label, frames: set.len(), snr_change: gain(sigma, ws, &set) / base - 1.0, fwhm: weighted_fwhm(frames, ws, &set) });
    };
    let warn: Vec<usize> = all.iter().copied().filter(|&i| frames[i].grade == Grade::Warn).collect();
    if !warn.is_empty() && warn.len() < n {
        let keep: Vec<usize> = all.iter().copied().filter(|i| !warn.contains(i)).collect();
        push(format!("without the {} warn frame{}", warn.len(), if warn.len() == 1 { "" } else { "s" }), keep, w);
    }
    if n >= 5 {
        let k = (n / 10).max(1);
        let mut by_snr = all.clone();
        by_snr.sort_by(|&a, &b| frames[a].snr.total_cmp(&frames[b].snr));
        let keep: Vec<usize> = by_snr[k..].to_vec();
        push(format!("without the {k} lowest-SNR frame{}", if k == 1 { "" } else { "s" }), keep, w);
    }
    if weighting != Weighting::Equal {
        push("with equal weights".into(), all.clone(), &vec![1.0; n]);
    }
    if weighting != Weighting::Snr {
        let ws: Vec<f64> = frames.iter().map(|f| f.snr * f.snr).collect();
        push("weighted by SNR".into(), all.clone(), &ws);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("crux-combine-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// Frames of one scene with different transparency, sky gradient and
    /// noise, plus a satellite trail: the master recovers the scene, the
    /// trail is rejected, the noisy frame weighs least.
    #[test]
    fn integrates_a_synthetic_field() {
        let (w, h) = (256usize, 192usize);
        let stars: Vec<(f32, f32)> = (0..40).map(|k| (20.0 + (k * 37 % 216) as f32, 20.0 + (k * 53 % 152) as f32)).collect();
        let scene = |x: usize, y: usize| -> f32 {
            let mut v = 0.002 * (x as f32 / w as f32);
            for &(sx, sy) in &stars {
                let d2 = (x as f32 - sx).powi(2) + (y as f32 - sy).powi(2);
                v += 0.05 * (-d2 / (2.0 * 1.5f32.powi(2))).exp();
            }
            v
        };
        let d = tmp("synth");
        let mut inputs = Vec::new();
        let mut seed = 12345u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % 1_000_000) as f32 / 1_000_000.0 - 0.5
        };
        for f in 0..8 {
            let t = [1.0, 0.95, 0.9, 1.0, 0.6, 0.97, 1.0, 0.92][f];
            let noise = if f == 4 { 0.004 } else { 0.001 };
            let mut data = vec![0f32; w * h];
            for y in 0..h {
                for x in 0..w {
                    let sky = 0.05 + 0.002 * f as f32 * (y as f32 / h as f32);
                    let mut v = sky + t * scene(x, y) + noise * rnd() * 3.46;
                    if f == 2 && (x as i64 - y as i64).abs() < 2 {
                        v += 0.2;
                    }
                    data[y * w + x] = v;
                }
            }
            // Frame 6 lacks its left edge (registered off the frame).
            if f == 6 {
                for y in 0..h {
                    for x in 0..20 {
                        data[y * w + x] = 0.0;
                    }
                }
            }
            let p = d.join(format!("f{f}.xisf"));
            crate::xisf::write_image(&p, &Image::from_data(w, h, data), &Header::default(), None).unwrap();
            inputs.push(Input { path: p, name: format!("f{f}"), exptime: 300.0, fwhm: Some(3.5), night: None, grade: if f == 4 { Grade::Warn } else { Grade::Good } });
        }
        let out = d.join("master.xisf");
        let st = Settings { local_cell: 48, ..Settings::default() };
        let r = integrate(&inputs, 0, &stars, &st, "T", "H", &out, &Header::default(), &|_| {}, &AtomicBool::new(false)).unwrap();
        assert!((r.frames[4].signal - 0.6).abs() < 0.03, "signal {}", r.frames[4].signal);
        assert!((r.frames[2].signal - 0.9).abs() < 0.03);
        assert!(r.frames[4].weight < 0.1, "noisy frame weight {}", r.frames[4].weight);
        assert!(r.frames[2].rejected_high > 0.005, "trail not rejected: {}", r.frames[2].rejected_high);
        let m = crate::read_image(&out).unwrap();
        // The master is the reference's scene: sky 0.05 + scene, no trail.
        let mut err = 0f64;
        for y in 10..h - 10 {
            for x in 30..w - 10 {
                err = err.max((m.data[y * w + x] - (0.05 + scene(x, y))).abs() as f64);
            }
        }
        assert!(err < 0.004, "max error {err}");
        assert!(r.snr_gain > 2.0, "gain {}", r.snr_gain);
        assert!(r.what_ifs.iter().any(|x| x.label.contains("warn")));
        let _ = std::fs::remove_dir_all(&d);
    }
}
