//! Mosaic assembly (milestone 14): a target's panel masters (objects
//! "<target> Panel <n>", or any objects sharing a target name) into one
//! master per filter on a common sky grid.
//!
//! 1. **Solve** every panel master against the catalogue (hinted by the
//!    reference frame's header, which the master carries), then fit the
//!    lens distortion: a cubic in pixel position for the residuals of the
//!    matched catalogue stars. Panels are placed by the sky, never by name.
//! 2. **Canvas:** a gnomonic projection about the panels' mean centre,
//!    north up, the panels' parity, at the panels' scale × `bin` (auto:
//!    the smallest whole bin that keeps the canvas under `max_megapixels`).
//! 3. **Warp** each panel (block-binned first when `bin` > 1) with
//!    Lanczos-3 through canvas → sky → panel pixels.
//! 4. **Brightness:** each overlap's block medians give a scale (one panel
//!    ≈ slope × the other + plane, robust), solved for all panels at once
//!    in logs; then an offset and plane per panel from the scaled
//!    differences, solved together (sums fixed at zero, planes gently held
//!    back), so seams vanish without bending the whole mosaic.
//! 5. **Blend:** weights 1/noise², feathered over `feather_arcmin` from
//!    each panel's edge (smoothstep), so overlaps also gain SNR.
//!
//! Report: per panel the solve, distortion, brightness model and noise;
//! per overlap the seam level before and after (in noise σ, block medians)
//! and how well the stars line up (matched star pairs, rms in panel
//! pixels, with and without the distortion fit).

use crate::catalog::{self, Catalog};
use crate::header::Header;
use crate::image::Image;
use crate::solve::{self, Hint, Solution};
use anyhow::{Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// 2: string keywords were quoted twice (`''RA---TAN''`), so other
/// programs didn't see the WCS.
/// 3: data never clamped to 0 (empty) after the lift.
pub const MOSAIC_VERSION: u32 = 3;
/// Overlap block size, canvas pixels.
const BLOCK: usize = 24;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Settings {
    /// Assemble after integrating (`crux process`, the GUI's Process).
    pub after_integrate: bool,
    /// Canvas pixels per panel pixel side; 0 = auto.
    pub bin: usize,
    /// Auto bin keeps the canvas under this.
    pub max_megapixels: f64,
    /// Width of the blend at panel edges.
    pub feather_arcmin: f64,
    /// Fit lens distortion from catalogue stars.
    pub distortion: bool,
    /// Match panel brightness (scale, offset, plane) across overlaps.
    pub match_brightness: bool,
    /// Also a gentle curvature per panel (curved skies from a cloudy or
    /// dawn reference frame).
    pub curvature: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { after_integrate: true, bin: 0, max_megapixels: 200.0, feather_arcmin: 9.0, distortion: true, match_brightness: true, curvature: true }
    }
}

/// One panel master.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Input {
    pub object: String,
    pub filter: String,
    pub path: PathBuf,
    pub frames: usize,
    pub exposure: f64,
}

/// A target with panels: every filter's masters.
#[derive(Debug, Clone)]
pub struct Group {
    pub target: String,
    pub inputs: Vec<Input>,
}

impl Group {
    pub fn filters(&self) -> Vec<String> {
        let mut f: Vec<String> = self.inputs.iter().map(|i| i.filter.clone()).collect();
        f.sort();
        f.dedup();
        f
    }
    pub fn panels(&self) -> usize {
        let mut o: Vec<&str> = self.inputs.iter().map(|i| i.object.as_str()).collect();
        o.sort();
        o.dedup();
        o.len()
    }
}

/// Targets with two or more panel objects, from a processing output
/// folder's integration reports.
pub fn groups(out: &Path) -> Vec<Group> {
    let reps = super::job::load_integrations(out);
    let mut by: BTreeMap<String, Vec<Input>> = BTreeMap::new();
    for r in reps.values() {
        if !r.out.is_file() {
            continue;
        }
        by.entry(crate::index::target_of(&r.object)).or_default().push(Input {
            object: r.object.clone(),
            filter: r.filter.clone(),
            path: r.out.clone(),
            frames: r.frames.len(),
            exposure: r.total_exposure,
        });
    }
    by.into_iter()
        .map(|(target, mut inputs)| {
            inputs.sort_by(|a, b| (crate::index::panel_of(&a.object), &a.object, &a.filter).cmp(&(crate::index::panel_of(&b.object), &b.object, &b.filter)));
            Group { target, inputs }
        })
        .filter(|g| g.panels() >= 2)
        .collect()
}

// ---------------------------------------------------------------- geometry

/// Lens distortion: pixel offsets (dx, dy) as a polynomial in the
/// linearly predicted position, normalised by `norm` about (cx, cy).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Distortion {
    pub order: usize,
    pub cx: f64,
    pub cy: f64,
    pub norm: f64,
    pub ax: Vec<f64>,
    pub ay: Vec<f64>,
}

fn terms(u: f64, v: f64, order: usize, out: &mut Vec<f64>) {
    out.clear();
    for d in 0..=order {
        for i in 0..=d {
            out.push(u.powi((d - i) as i32) * v.powi(i as i32));
        }
    }
}

impl Distortion {
    pub fn at(&self, x: f64, y: f64) -> (f64, f64) {
        let mut t = Vec::with_capacity(10);
        terms((x - self.cx) / self.norm, (y - self.cy) / self.norm, self.order, &mut t);
        (t.iter().zip(&self.ax).map(|(a, b)| a * b).sum(), t.iter().zip(&self.ay).map(|(a, b)| a * b).sum())
    }
}

/// Where a panel sits on the sky.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Geometry {
    pub solution: Solution,
    pub distortion: Option<Distortion>,
    pub width: usize,
    pub height: usize,
    /// Catalogue stars matched for the distortion fit.
    pub matched: usize,
    /// Residual of the matched stars, pixels: linear solution, and with
    /// the distortion fit.
    #[serde(deserialize_with = "nan_or")]
    pub rms_linear: f64,
    #[serde(deserialize_with = "nan_or")]
    pub rms: f64,
    /// Largest distortion correction inside the frame, pixels.
    pub distortion_max: f64,
}

impl Geometry {
    pub fn sky_to_pixel(&self, ra: f64, dec: f64) -> Option<(f64, f64)> {
        let (x, y) = self.solution.sky_to_pixel(ra, dec)?;
        Some(match &self.distortion {
            Some(d) => {
                let (dx, dy) = d.at(x, y);
                (x + dx, y + dy)
            }
            None => (x, y),
        })
    }
    pub fn pixel_to_sky(&self, x: f64, y: f64) -> (f64, f64) {
        let (mut lx, mut ly) = (x, y);
        if let Some(d) = &self.distortion {
            for _ in 0..4 {
                let (dx, dy) = d.at(lx, ly);
                lx = x - dx;
                ly = y - dy;
            }
        }
        self.solution.pixel_to_sky(lx, ly)
    }
    fn without_distortion(&self) -> Geometry {
        Geometry { distortion: None, ..self.clone() }
    }
}

/// Mutual-nearest pairs within `r` px between predicted catalogue
/// positions and detected stars: (catalogue index, star index).
fn match_nearest(pred: &[(f64, f64)], stars: &[(f64, f64)], r: f64) -> Vec<(usize, usize)> {
    let cell = r.max(1.0);
    let mut grid: std::collections::HashMap<(i64, i64), Vec<usize>> = Default::default();
    for (i, s) in stars.iter().enumerate() {
        grid.entry(((s.0 / cell).floor() as i64, (s.1 / cell).floor() as i64)).or_default().push(i);
    }
    let nearest = |p: (f64, f64), pts: &[(f64, f64)], grid: &std::collections::HashMap<(i64, i64), Vec<usize>>| -> Option<(usize, f64)> {
        let (cx, cy) = ((p.0 / cell).floor() as i64, (p.1 / cell).floor() as i64);
        let mut best: Option<(usize, f64)> = None;
        for gy in cy - 1..=cy + 1 {
            for gx in cx - 1..=cx + 1 {
                for &i in grid.get(&(gx, gy)).map(|v| v.as_slice()).unwrap_or(&[]) {
                    let d = (pts[i].0 - p.0).hypot(pts[i].1 - p.1);
                    if d <= r && best.is_none_or(|b| d < b.1) {
                        best = Some((i, d));
                    }
                }
            }
        }
        best
    };
    let mut pgrid: std::collections::HashMap<(i64, i64), Vec<usize>> = Default::default();
    for (i, p) in pred.iter().enumerate() {
        pgrid.entry(((p.0 / cell).floor() as i64, (p.1 / cell).floor() as i64)).or_default().push(i);
    }
    let mut out = Vec::new();
    for (i, p) in pred.iter().enumerate() {
        if let Some((j, _)) = nearest(*p, stars, &grid) {
            if nearest(stars[j], pred, &pgrid).map(|b| b.0) == Some(i) {
                out.push((i, j));
            }
        }
    }
    out
}

/// Weighted least squares with sparse rows: Σ w·(row·x − rhs)² → min.
/// Normal equations, Gaussian elimination with partial pivoting.
fn lstsq(rows: &[(Vec<(usize, f64)>, f64, f64)], n: usize) -> Option<Vec<f64>> {
    let mut a = vec![0f64; n * n];
    let mut b = vec![0f64; n];
    for (row, rhs, w) in rows {
        let w2 = w * w;
        for &(i, vi) in row {
            b[i] += w2 * vi * rhs;
            for &(j, vj) in row {
                a[i * n + j] += w2 * vi * vj;
            }
        }
    }
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| a[i * n + c].abs().total_cmp(&a[j * n + c].abs()))?;
        if a[p * n + c].abs() < 1e-300 {
            return None;
        }
        if p != c {
            for k in 0..n {
                a.swap(c * n + k, p * n + k);
            }
            b.swap(c, p);
        }
        for r in c + 1..n {
            let f = a[r * n + c] / a[c * n + c];
            if f != 0.0 {
                for k in c..n {
                    a[r * n + k] -= f * a[c * n + k];
                }
                b[r] -= f * b[c];
            }
        }
    }
    let mut x = vec![0f64; n];
    for c in (0..n).rev() {
        let s: f64 = (c + 1..n).map(|k| a[c * n + k] * x[k]).sum();
        x[c] = (b[c] - s) / a[c * n + c];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

fn mad_sigma(v: &mut [f64]) -> f64 {
    let m = crate::stats::median_f64(v).unwrap_or(0.0);
    let mut d: Vec<f64> = v.iter().map(|x| (x - m).abs()).collect();
    crate::stats::median_f64(&mut d).unwrap_or(0.0) * crate::stats::MAD_TO_SIGMA
}

/// Fit the distortion from catalogue stars matched to the detected stars
/// (`stars` brightest first). A straight-line plate solution is several
/// pixels out at the corners (LMC: ~7 px), and was tens of pixels out
/// before the solver's rotation fix, so matching goes coarse to fine: the
/// brightest stars within 60 px for an affine model, 12 px for a
/// quadratic, then every star within 5, 3 and 2 px for a cubic.
fn fit_distortion(sol: &Solution, cat: &Catalog, stars: &[(f64, f64)], w: usize, h: usize) -> Result<Geometry> {
    let half_diag = 0.5 * ((w * w + h * h) as f64).sqrt() * sol.scale / 3600.0;
    let cs = cat.cone(sol.ra, sol.dec, half_diag * 1.05, (stars.len() * 3).clamp(2000, 20000))?;
    let lin: Vec<(f64, f64)> = cs.iter().filter_map(|c| sol.sky_to_pixel(c.ra, c.dec)).filter(|p| p.0 >= -20.0 && p.1 >= -20.0 && p.0 < w as f64 + 20.0 && p.1 < h as f64 + 20.0).collect();
    let norm = 0.5 * (w.max(h) as f64);
    let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
    let core_rms = |mut r: Vec<f64>| -> f64 {
        if r.is_empty() {
            return f64::NAN;
        }
        r.sort_by(f64::total_cmp);
        let core = &r[..(r.len() * 9 / 10).max(1)];
        (core.iter().map(|v| v * v).sum::<f64>() / core.len() as f64).sqrt()
    };
    let predict = |d: &Option<Distortion>| -> Vec<(f64, f64)> {
        lin.iter()
            .map(|p| match d {
                Some(d) => {
                    let (dx, dy) = d.at(p.0, p.1);
                    (p.0 + dx, p.1 + dy)
                }
                None => *p,
            })
            .collect()
    };
    let mut dist: Option<Distortion> = None;
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    let mut t = Vec::new();
    const ALL: usize = usize::MAX;
    for (n_cat, n_star, radius, order) in [(300usize, 400usize, 60.0, 1usize), (800, 1000, 12.0, 2), (ALL, ALL, 5.0, 3), (ALL, ALL, 3.0, 3), (ALL, ALL, 2.0, 3)] {
        let pred = predict(&dist);
        let pc = &pred[..pred.len().min(n_cat)];
        let st = &stars[..stars.len().min(n_star)];
        let m = match_nearest(pc, st, radius);
        let nt = (order + 1) * (order + 2) / 2;
        if std::env::var_os("CRUX_MOSAIC_DEBUG").is_some() {
            eprintln!("  distortion: {} catalogue, {} stars, radius {radius}: {} pairs", pc.len(), st.len(), m.len());
        }
        if m.len() < nt * 4 {
            break;
        }
        // Fit with 3σ clipping.
        let mut keep = m.clone();
        let mut fit: Option<Distortion> = None;
        for _ in 0..5 {
            let (mut rx, mut ry) = (Vec::new(), Vec::new());
            for &(i, j) in &keep {
                terms((lin[i].0 - cx) / norm, (lin[i].1 - cy) / norm, order, &mut t);
                let row: Vec<(usize, f64)> = t.iter().copied().enumerate().collect();
                rx.push((row.clone(), stars[j].0 - lin[i].0, 1.0));
                ry.push((row, stars[j].1 - lin[i].1, 1.0));
            }
            let (Some(ax), Some(ay)) = (lstsq(&rx, nt), lstsq(&ry, nt)) else { break };
            let d = Distortion { order, cx, cy, norm, ax, ay };
            let res = |&(i, j): &(usize, usize)| {
                let (dx, dy) = d.at(lin[i].0, lin[i].1);
                (lin[i].0 + dx - stars[j].0).hypot(lin[i].1 + dy - stars[j].1)
            };
            let mut r: Vec<f64> = keep.iter().map(res).collect();
            let s = mad_sigma(&mut r.clone()).max(0.03);
            let med = crate::stats::median_f64(&mut r).unwrap_or(0.0);
            let next: Vec<(usize, usize)> = keep.iter().copied().filter(|p| res(p) <= med + 3.0 * s).collect();
            fit = Some(d);
            if next.len() == keep.len() || next.len() < nt * 4 {
                break;
            }
            keep = next;
        }
        if fit.is_none() {
            break;
        }
        dist = fit;
        pairs = m;
    }
    let mut g = Geometry { solution: sol.clone(), distortion: None, width: w, height: h, matched: pairs.len(), rms_linear: f64::NAN, rms: f64::NAN, distortion_max: 0.0 };
    let Some(d) = dist else {
        let m = match_nearest(&lin, stars, 3.0);
        g.matched = m.len();
        g.rms_linear = core_rms(m.iter().map(|&(i, j)| (lin[i].0 - stars[j].0).hypot(lin[i].1 - stars[j].1)).collect());
        g.rms = g.rms_linear;
        return Ok(g);
    };
    let pred = predict(&Some(d.clone()));
    // Same star pairs, both models.
    g.rms = core_rms(pairs.iter().map(|&(i, j)| (pred[i].0 - stars[j].0).hypot(pred[i].1 - stars[j].1)).collect());
    g.rms_linear = core_rms(pairs.iter().map(|&(i, j)| (lin[i].0 - stars[j].0).hypot(lin[i].1 - stars[j].1)).collect());
    for gy in 0..=8 {
        for gx in 0..=8 {
            let (dx, dy) = d.at(gx as f64 * w as f64 / 8.0, gy as f64 * h as f64 / 8.0);
            g.distortion_max = g.distortion_max.max(dx.hypot(dy));
        }
    }
    g.distortion = Some(d);
    Ok(g)
}

/// Solve one panel master and fit its distortion.
pub fn solve_panel(path: &Path, cat: &Catalog, learned: &std::sync::Mutex<solve::Learned>, distortion: bool) -> Result<(Geometry, Vec<(f64, f64)>)> {
    let img = crate::read_image(path).with_context(|| format!("read {}", path.display()))?;
    let meta = crate::meta::read_meta(path)?;
    let hint = Hint::from_meta(&meta).with_context(|| format!("{}: no RA/DEC and pixel scale in the header to start the solve", path.display()))?;
    let det = {
        let bg = crate::background::Background::estimate(&img, 64);
        crate::stars::detect(&img, &bg, &crate::stars::DetectParams { max_measure: 4000, ..Default::default() })
    };
    let (w, h) = (img.width, img.height);
    drop(img);
    let hint2 = learned.lock().unwrap().hint(&hint);
    let sol = solve::solve(cat, &det.stars, w as f64, h as f64, &hint2)?.with_context(|| format!("{}: plate solve failed ({} stars)", path.display(), det.stars.len()))?;
    learned.lock().unwrap().learn(&hint, &sol);
    let centres: Vec<(f64, f64)> = det.stars.iter().filter(|s| s.is_clean()).map(|s| (s.x as f64, s.y as f64)).collect();
    let g = if distortion {
        fit_distortion(&sol, cat, &centres, w, h)?
    } else {
        let mut g = fit_distortion(&sol, cat, &centres, w, h)?;
        g.distortion = None;
        g.rms = g.rms_linear;
        g
    };
    Ok((g, centres))
}

// ---------------------------------------------------------------- canvas

/// The mosaic's sky grid.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Canvas {
    /// Canvas pixel ↔ sky (a linear gnomonic solution, north up).
    pub wcs: Solution,
    pub width: usize,
    pub height: usize,
    pub bin: usize,
    /// Arcseconds per canvas pixel.
    pub scale: f64,
}

fn unit(ra: f64, dec: f64) -> [f64; 3] {
    let (r, d) = (ra.to_radians(), dec.to_radians());
    [d.cos() * r.cos(), d.cos() * r.sin(), d.sin()]
}

/// The outline of a panel (corners and edge midpoints), sky.
fn outline(g: &Geometry) -> Vec<(f64, f64)> {
    let (w, h) = (g.width as f64, g.height as f64);
    [(0.0, 0.0), (w / 2.0, 0.0), (w, 0.0), (w, h / 2.0), (w, h), (w / 2.0, h), (0.0, h), (0.0, h / 2.0)].iter().map(|&(x, y)| g.pixel_to_sky(x, y)).collect()
}

pub fn make_canvas(geoms: &[&Geometry], s: &Settings) -> Canvas {
    let mut v = [0f64; 3];
    for g in geoms {
        let u = unit(g.solution.ra, g.solution.dec);
        for k in 0..3 {
            v[k] += u[k];
        }
    }
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-12);
    let (ra0, dec0) = (v[1].atan2(v[0]).to_degrees().rem_euclid(360.0), (v[2] / n).asin().to_degrees());
    let mut scales: Vec<f64> = geoms.iter().map(|g| g.solution.scale).collect();
    let pscale = crate::stats::median_f64(&mut scales).unwrap_or(1.0);
    // Parity of the panels (as they're displayed), north up.
    let pos = geoms.iter().filter(|g| g.solution.cd[0] * g.solution.cd[3] - g.solution.cd[1] * g.solution.cd[2] > 0.0).count();
    let det_pos = pos * 2 >= geoms.len();
    let extent = |bin: usize| -> (f64, f64, f64, f64, [f64; 4]) {
        let sd = pscale * bin as f64 / 3600.0;
        let cd = if det_pos { [-sd, 0.0, 0.0, -sd] } else { [sd, 0.0, 0.0, -sd] };
        let (mut x0, mut x1, mut y0, mut y1) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
        for g in geoms {
            for (ra, dec) in outline(g) {
                if let Some((xi, eta)) = catalog::project(ra0, dec0, ra, dec) {
                    // Inverse of the diagonal cd.
                    let (x, y) = (xi / cd[0], eta / cd[3]);
                    x0 = x0.min(x);
                    x1 = x1.max(x);
                    y0 = y0.min(y);
                    y1 = y1.max(y);
                }
            }
        }
        (x0, x1, y0, y1, cd)
    };
    let mut bin = s.bin.max(1);
    if s.bin == 0 {
        let (x0, x1, y0, y1, _) = extent(1);
        let mp = (x1 - x0) * (y1 - y0) / 1e6;
        bin = ((mp / s.max_megapixels.max(1.0)).sqrt().ceil() as usize).max(1);
    }
    let (x0, x1, y0, y1, cd) = extent(bin);
    let pad = 4.0;
    let (width, height) = ((x1 - x0).ceil() as usize + 2 * pad as usize, (y1 - y0).ceil() as usize + 2 * pad as usize);
    let wcs = Solution {
        ra: ra0,
        dec: dec0,
        scale: pscale * bin as f64,
        rotation: 0.0,
        mirrored: det_pos,
        cd,
        cx: -x0.floor() + pad,
        cy: -y0.floor() + pad,
        matched: 0,
        rms: 0.0,
        offset_arcmin: None,
        target_offset_arcmin: None,
        version: solve::SOLVE_VERSION,
    };
    Canvas { wcs, width, height, bin, scale: pscale * bin as f64 }
}

// ---------------------------------------------------------------- tiles

/// One panel warped onto the canvas.
struct Tile {
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    /// NaN where the panel has no data.
    data: Vec<f32>,
    /// Feather, 0–1.
    feather: Vec<f32>,
    noise: f64,
    /// Centre and size as canvas fractions (centre −½…½).
    cx: f64,
    cy: f64,
    sx: f64,
    sy: f64,
}

impl Tile {
    fn at(&self, x: usize, y: usize) -> f32 {
        if x < self.x0 || y < self.y0 || x >= self.x0 + self.w || y >= self.y0 + self.h {
            return f32::NAN;
        }
        self.data[(y - self.y0) * self.w + (x - self.x0)]
    }
}

/// Chamfer distance (1, √2) to the nearest invalid pixel or the edge.
fn distance(valid: &[bool], w: usize, h: usize) -> Vec<f32> {
    const D: f32 = std::f32::consts::SQRT_2;
    let mut d: Vec<f32> = valid.iter().map(|&v| if v { f32::MAX } else { 0.0 }).collect();
    let g = |d: &Vec<f32>, x: isize, y: isize| -> f32 {
        if x < 0 || y < 0 || x >= w as isize || y >= h as isize {
            0.0
        } else {
            d[y as usize * w + x as usize]
        }
    };
    for y in 0..h as isize {
        for x in 0..w as isize {
            let i = y as usize * w + x as usize;
            if d[i] == 0.0 {
                continue;
            }
            let m = (g(&d, x - 1, y) + 1.0).min(g(&d, x, y - 1) + 1.0).min(g(&d, x - 1, y - 1) + D).min(g(&d, x + 1, y - 1) + D);
            d[i] = d[i].min(m);
        }
    }
    for y in (0..h as isize).rev() {
        for x in (0..w as isize).rev() {
            let i = y as usize * w + x as usize;
            if d[i] == 0.0 {
                continue;
            }
            let m = (g(&d, x + 1, y) + 1.0).min(g(&d, x, y + 1) + 1.0).min(g(&d, x + 1, y + 1) + D).min(g(&d, x - 1, y + 1) + D);
            d[i] = d[i].min(m);
        }
    }
    d
}

fn bin_image(img: &Image, f: usize) -> Image {
    if f <= 1 {
        return img.clone();
    }
    let (w, h, d) = crate::render::gradient::bin_valid(img, f);
    Image::from_data(w, h, d.into_iter().map(|v| if v.is_finite() { v } else { 0.0 }).collect())
}

fn make_tile(path: &Path, g: &Geometry, c: &Canvas, feather_px: f64) -> Result<Tile> {
    let img = crate::read_image(path).with_context(|| format!("read {}", path.display()))?;
    let src = bin_image(&img, c.bin);
    drop(img);
    let (mut x0, mut x1, mut y0, mut y1) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
    for (ra, dec) in outline(g) {
        if let Some((x, y)) = c.wcs.sky_to_pixel(ra, dec) {
            x0 = x0.min(x);
            x1 = x1.max(x);
            y0 = y0.min(y);
            y1 = y1.max(y);
        }
    }
    let x0 = (x0.floor() as i64 - 2).max(0) as usize;
    let y0 = (y0.floor() as i64 - 2).max(0) as usize;
    let x1 = ((x1.ceil() as i64 + 3).max(0) as usize).min(c.width);
    let y1 = ((y1.ceil() as i64 + 3).max(0) as usize).min(c.height);
    anyhow::ensure!(x1 > x0 + 8 && y1 > y0 + 8, "{} falls outside the canvas", path.display());
    let (w, h) = (x1 - x0, y1 - y0);
    let b = c.bin as f64;
    let map = |x: f64, y: f64| -> Option<(f64, f64)> {
        let (ra, dec) = c.wcs.pixel_to_sky(x + x0 as f64, y + y0 as f64);
        let (px, py) = g.sky_to_pixel(ra, dec)?;
        Some(((px + 0.5) / b - 0.5, (py + 0.5) / b - 0.5))
    };
    let mut data = super::resample::warp_with(&src, w, h, 16, &map);
    // Lanczos reaches 3 px: drop that much at every edge, feather inside.
    let valid: Vec<bool> = data.iter().map(|v| v.is_finite()).collect();
    let dist = distance(&valid, w, h);
    let mut feather = vec![0f32; w * h];
    for i in 0..w * h {
        if dist[i] <= 3.0 {
            data[i] = f32::NAN;
        } else {
            let t = ((dist[i] - 3.0) as f64 / feather_px.max(1.0)).clamp(0.0, 1.0) as f32;
            feather[i] = (t * t * (3.0 - 2.0 * t)).max(1e-4);
        }
    }
    let noise = {
        let z: Vec<f32> = data.iter().map(|v| if v.is_finite() { *v } else { 0.0 }).collect();
        super::combine::noise_of(&Image::from_data(w, h, z)) as f64
    };
    let (cw, ch) = (c.width as f64, c.height as f64);
    Ok(Tile { x0, y0, w, h, data, feather, noise, cx: (x0 as f64 + w as f64 / 2.0) / cw - 0.5, cy: (y0 as f64 + h as f64 / 2.0) / ch - 0.5, sx: w as f64 / cw, sy: h as f64 / ch })
}

// ---------------------------------------------------------------- brightness

/// Block medians where both tiles have every pixel: (a, b, X, Y), X and Y
/// the block centre over the canvas size, −0.5…0.5.
fn overlap_blocks(a: &Tile, b: &Tile, c: &Canvas) -> Vec<(f64, f64, f64, f64)> {
    let (xa, xb) = (a.x0.max(b.x0), (a.x0 + a.w).min(b.x0 + b.w));
    let (ya, yb) = (a.y0.max(b.y0), (a.y0 + a.h).min(b.y0 + b.h));
    if xb < xa + 3 * BLOCK || yb < ya + 3 * BLOCK {
        return Vec::new();
    }
    let mut out = Vec::new();
    let (mut va, mut vb) = (Vec::with_capacity(BLOCK * BLOCK), Vec::with_capacity(BLOCK * BLOCK));
    let mut y = ya;
    while y + BLOCK <= yb {
        let mut x = xa;
        'blocks: while x + BLOCK <= xb {
            va.clear();
            vb.clear();
            for yy in y..y + BLOCK {
                for xx in x..x + BLOCK {
                    let (p, q) = (a.at(xx, yy), b.at(xx, yy));
                    if !p.is_finite() || !q.is_finite() {
                        x += BLOCK;
                        continue 'blocks;
                    }
                    va.push(p);
                    vb.push(q);
                }
            }
            let (ma, mb) = (crate::stats::median_mut(&mut va), crate::stats::median_mut(&mut vb));
            out.push((ma as f64, mb as f64, (x as f64 + BLOCK as f64 / 2.0) / c.width as f64 - 0.5, (y as f64 + BLOCK as f64 / 2.0) / c.height as f64 - 0.5));
            x += BLOCK;
        }
        y += BLOCK;
    }
    out
}

/// y ≈ cols·c, iterative 3σ clipping. Returns (coefficients, kept mask).
fn robust_fit(y: &[f64], cols: &[Vec<f64>]) -> Option<(Vec<f64>, Vec<bool>)> {
    let n = cols.len();
    let mut keep = vec![true; y.len()];
    let mut coef = None;
    for _ in 0..6 {
        let rows: Vec<(Vec<(usize, f64)>, f64, f64)> = (0..y.len()).filter(|&i| keep[i]).map(|i| ((0..n).map(|k| (k, cols[k][i])).collect(), y[i], 1.0)).collect();
        if rows.len() < n * 3 {
            return None;
        }
        let c = lstsq(&rows, n)?;
        let r: Vec<f64> = (0..y.len()).map(|i| y[i] - (0..n).map(|k| c[k] * cols[k][i]).sum::<f64>()).collect();
        let mut rk: Vec<f64> = (0..y.len()).filter(|&i| keep[i]).map(|i| r[i]).collect();
        let sd = mad_sigma(&mut rk).max(1e-12);
        let next: Vec<bool> = r.iter().map(|v| v.abs() < 3.0 * sd).collect();
        coef = Some(c);
        if next == keep {
            break;
        }
        keep = next;
    }
    coef.map(|c| (c, keep))
}

/// Per panel: value' = scale·value + offset + gx·X + gy·Y.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Brightness {
    pub scale: f64,
    pub offset: f64,
    pub gx: f64,
    pub gy: f64,
    /// Gentle curvature in the panel's own coordinates (u, v: −½…½ across
    /// the panel): q[0]·u² + q[1]·u·v + q[2]·v². A cloudy or dawn
    /// reference frame leaves a curved sky in a panel's master.
    #[serde(default)]
    pub q: [f64; 3],
    /// The panel's centre and size, canvas fractions (for u, v).
    #[serde(default)]
    pub frame: [f64; 4],
}

impl Default for Brightness {
    fn default() -> Self {
        Brightness { scale: 1.0, offset: 0.0, gx: 0.0, gy: 0.0, q: [0.0; 3], frame: [0.0, 0.0, 1.0, 1.0] }
    }
}

impl Brightness {
    fn uv(&self, x: f64, y: f64) -> (f64, f64) {
        ((x - self.frame[0]) / self.frame[2].max(1e-9), (y - self.frame[1]) / self.frame[3].max(1e-9))
    }
    /// The additive part at canvas fractions (x, y).
    pub fn additive(&self, x: f64, y: f64) -> f64 {
        let (u, v) = self.uv(x, y);
        self.offset + self.gx * x + self.gy * y + self.q[0] * u * u + self.q[1] * u * v + self.q[2] * v * v
    }
    pub fn apply(&self, value: f64, x: f64, y: f64) -> f64 {
        self.scale * value + self.additive(x, y)
    }
}

type Pairs = Vec<(usize, usize, Vec<(f64, f64, f64, f64)>)>;

/// Background-subtracted flux in a circle of radius `r` about (x, y),
/// background the median of the annulus r + 3 … r + 8. None near an edge
/// or with a pixel above 0.8 (saturated cores).
fn aperture(t: &Tile, x: f64, y: f64, r: f64) -> Option<f64> {
    let ro = r + 8.0;
    let (x0, x1) = ((x - ro).floor() as i64, (x + ro).ceil() as i64);
    let (y0, y1) = ((y - ro).floor() as i64, (y + ro).ceil() as i64);
    if x0 < 0 || y0 < 0 {
        return None;
    }
    let (mut inside, mut ann) = (Vec::new(), Vec::new());
    for yy in y0..=y1 {
        for xx in x0..=x1 {
            let d = (xx as f64 - x).hypot(yy as f64 - y);
            if d > ro {
                continue;
            }
            let v = t.at(xx as usize, yy as usize);
            if !v.is_finite() {
                return None;
            }
            if d <= r {
                if v > 0.8 {
                    return None;
                }
                inside.push(v as f64);
            } else if d >= r + 3.0 {
                ann.push(v as f64);
            }
        }
    }
    let bg = crate::stats::median_f64(&mut ann)?;
    Some(inside.iter().map(|v| v - bg).sum())
}

/// Signal ratio a/b in an overlap from the same stars measured in both
/// tiles (canvas positions): median of the flux ratios, with a weight.
/// Star light doesn't care about nebula, sky gradients or offsets.
fn star_ratio(ta: &Tile, tb: &Tile, pa: &[(f64, f64)], pb: &[(f64, f64)], bin: usize) -> Option<(f64, f64, usize)> {
    let r = (10.0 / bin as f64).max(3.0);
    let m = match_nearest(pa, pb, 3.0 / bin as f64 + 1.0);
    let area = std::f64::consts::PI * r * r;
    let (la, lb) = (20.0 * ta.noise * area.sqrt(), 20.0 * tb.noise * area.sqrt());
    let mut lr: Vec<f64> = m
        .iter()
        .filter_map(|&(i, j)| {
            let (x, y) = (0.5 * (pa[i].0 + pb[j].0), 0.5 * (pa[i].1 + pb[j].1));
            let fa = aperture(ta, x, y, r)?;
            let fb = aperture(tb, x, y, r)?;
            (fa > la && fb > lb).then(|| (fa / fb).ln())
        })
        .collect();
    if lr.len() < 8 {
        return None;
    }
    let n = lr.len();
    let med = crate::stats::median_f64(&mut lr.clone())?;
    let sd = mad_sigma(&mut lr).max(0.005);
    Some((med.exp(), (n as f64).sqrt() / sd, n))
}

fn match_brightness(tiles: &[Option<Tile>], pairs: &Pairs, ratios: &[Option<(f64, f64, usize)>], curvature: bool) -> Vec<Brightness> {
    const K: usize = 6;
    let np = tiles.len();
    let mut out = vec![Brightness::default(); np];
    for (k, t) in tiles.iter().enumerate() {
        if let Some(t) = t {
            out[k].frame = [t.cx, t.cy, t.sx, t.sy];
        }
    }
    if pairs.is_empty() {
        return out;
    }
    let noise = |i: usize| tiles[i].as_ref().map(|t| t.noise).unwrap_or(1.0).max(1e-9);
    // (a) Scales: Pi ≈ slope·Pj + plane → log s_j − log s_i = log slope.
    let mut rows: Vec<(Vec<(usize, f64)>, f64, f64)> = Vec::new();
    for (k, (i, j, bl)) in pairs.iter().enumerate() {
        if let Some((ratio, w, _)) = ratios[k] {
            rows.push((vec![(*j, 1.0), (*i, -1.0)], ratio.ln(), w));
            continue;
        }
        let a: Vec<f64> = bl.iter().map(|b| b.0).collect();
        let cols = vec![bl.iter().map(|b| b.1).collect::<Vec<f64>>(), vec![1.0; bl.len()], bl.iter().map(|b| b.2).collect(), bl.iter().map(|b| b.3).collect()];
        let Some((c, keep)) = robust_fit(&a, &cols) else { continue };
        if !(0.3..=3.0).contains(&c[0]) {
            continue;
        }
        let mut kb: Vec<f64> = bl.iter().zip(&keep).filter(|(_, k)| **k).map(|(b, _)| b.1).collect();
        kb.sort_by(f64::total_cmp);
        let spread = kb[kb.len() * 9 / 10] - kb[kb.len() / 10];
        let nk = kb.len() as f64;
        // A pixel fit is the fallback for overlaps without enough stars:
        // much weaker than star photometry.
        let w = 0.1 * nk.sqrt() * (spread / noise(*i).max(noise(*j))).min(50.0) / 50.0;
        rows.push((vec![(*j, 1.0), (*i, -1.0)], c[0].ln(), w));
    }
    rows.push(((0..np).map(|k| (k, 1.0)).collect(), 0.0, 1e3));
    for k in 0..np {
        rows.push((vec![(k, 1.0)], 0.0, 1e-3));
    }
    let logs = lstsq(&rows, np).unwrap_or(vec![0.0; np]);
    for k in 0..np {
        out[k].scale = logs[k].exp();
    }
    // (b) Offsets and planes on the scaled differences.
    let mut rows: Vec<(Vec<(usize, f64)>, f64, f64)> = Vec::new();
    for (i, j, bl) in pairs {
        let (si, sj) = (out[*i].scale, out[*j].scale);
        let d: Vec<f64> = bl.iter().map(|b| si * b.0 - sj * b.1).collect();
        let med = crate::stats::median_f64(&mut d.clone()).unwrap_or(0.0);
        let sd = mad_sigma(&mut d.clone()).max(1e-12);
        let idx: Vec<usize> = (0..d.len()).filter(|&k| (d[k] - med).abs() < 3.0 * sd).collect();
        let take = idx.len().min(400);
        for t in 0..take {
            let k = idx[t * idx.len() / take];
            let (x, y) = (bl[k].2, bl[k].3);
            let (ui, vi) = out[*i].uv(x, y);
            let (uj, vj) = out[*j].uv(x, y);
            let (a, b) = (K * i, K * j);
            let mut row = vec![(a, 1.0), (a + 1, x), (a + 2, y), (b, -1.0), (b + 1, -x), (b + 2, -y)];
            if curvature {
                row.extend([(a + 3, ui * ui), (a + 4, ui * vi), (a + 5, vi * vi), (b + 3, -uj * uj), (b + 4, -uj * vj), (b + 5, -vj * vj)]);
            }
            rows.push((row, -d[k], 1.0));
        }
    }
    let n_data = rows.len().max(1);
    for k in 0..3 {
        rows.push(((0..np).map(|p| (K * p + k, 1.0)).collect(), 0.0, 100.0));
    }
    let lam = 0.02 * (n_data as f64 / np as f64).sqrt();
    for p in 0..np {
        rows.push((vec![(K * p, 1.0)], 0.0, 1e-3));
        for k in 1..3 {
            rows.push((vec![(K * p + k, 1.0)], 0.0, lam));
        }
        for k in 3..K {
            rows.push((vec![(K * p + k, 1.0)], 0.0, if curvature { lam } else { 1e6 }));
        }
    }
    if let Some(x) = lstsq(&rows, K * np) {
        for p in 0..np {
            out[p].offset = x[K * p];
            out[p].gx = x[K * p + 1];
            out[p].gy = x[K * p + 2];
            if curvature {
                out[p].q = [x[K * p + 3], x[K * p + 4], x[K * p + 5]];
            }
        }
    }
    out
}

// ---------------------------------------------------------------- report

/// JSON has no NaN: serde writes `null`, read back as NaN.
fn nan_or<'de, D: serde::Deserializer<'de>>(d: D) -> std::result::Result<f64, D::Error> {
    Ok(Option::<f64>::deserialize(d)?.unwrap_or(f64::NAN))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanelReport {
    pub object: String,
    pub panel: Option<u32>,
    pub path: PathBuf,
    pub frames: usize,
    pub exposure: f64,
    /// Arcseconds per pixel, rotation (degrees), centre.
    pub scale: f64,
    pub rotation: f64,
    pub ra: f64,
    pub dec: f64,
    pub matched: usize,
    #[serde(deserialize_with = "nan_or")]
    pub rms_linear: f64,
    #[serde(deserialize_with = "nan_or")]
    pub rms: f64,
    pub distortion_max: f64,
    pub brightness: Brightness,
    /// Noise σ (normalised), as warped.
    pub noise: f64,
    /// Outline on the canvas: corners, clockwise from the first pixel.
    pub outline: Vec<[f64; 2]>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Seam {
    pub a: String,
    pub b: String,
    pub blocks: usize,
    /// Median difference in the overlap over the combined noise σ, before
    /// and after brightness matching.
    pub before: f64,
    pub after: f64,
    /// Stars matched in the overlap and their misalignment, panel pixels
    /// (rms), with and without the distortion fit.
    pub stars: usize,
    #[serde(deserialize_with = "nan_or")]
    pub star_rms: f64,
    #[serde(deserialize_with = "nan_or")]
    pub star_rms_linear: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub target: String,
    pub filter: String,
    pub out: PathBuf,
    pub canvas: Canvas,
    pub panels: Vec<PanelReport>,
    pub seams: Vec<Seam>,
    pub settings: Settings,
    /// Total exposure and frames over all panels.
    pub exposure: f64,
    pub frames: usize,
    pub seconds: f64,
    pub signature: String,
}

impl Report {
    /// Worst seam after matching, in noise σ.
    pub fn worst_seam(&self) -> f64 {
        self.seams.iter().map(|s| s.after.abs()).fold(0.0, f64::max)
    }
    /// Median star misalignment over the overlaps, panel pixels.
    pub fn star_rms(&self) -> f64 {
        let mut v: Vec<f64> = self.seams.iter().filter(|s| s.stars >= 5).map(|s| s.star_rms).collect();
        crate::stats::median_f64(&mut v).unwrap_or(f64::NAN)
    }
}

pub fn report_path(out: &Path) -> PathBuf {
    out.join("crux-mosaic.json")
}

pub fn load_reports(out: &Path) -> BTreeMap<String, Report> {
    std::fs::read(report_path(out)).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
}

fn key(target: &str, filter: &str) -> String {
    format!("{target}|{filter}")
}

pub fn output_path(out: &Path, target: &str, filter: &str) -> PathBuf {
    let t = super::job::safe_name(target);
    out.join("mosaic").join(&t).join(format!("mosaic_{t}_{}.xisf", super::job::safe_name(filter)))
}

fn signature(g: &Group, s: &Settings) -> String {
    let mut parts = vec![format!("v{MOSAIC_VERSION}"), serde_json::to_string(s).unwrap_or_default()];
    for i in &g.inputs {
        let (size, mt) = crate::meta::file_stamp(&i.path).unwrap_or((0, 0));
        parts.push(format!("{}|{}|{size}|{mt}", i.path.display(), i.filter));
    }
    let mut h = std::collections::hash_map::DefaultHasher::new();
    std::hash::Hash::hash(&parts, &mut h);
    format!("{:016x}", std::hash::Hasher::finish(&h))
}

// ---------------------------------------------------------------- run

#[derive(Debug, Clone)]
pub enum Step {
    Solve { done: usize, total: usize },
    Warp { filter: String, done: usize, total: usize },
    Match { filter: String },
    Blend { filter: String },
    Written { filter: String, path: PathBuf },
    Skipped { filter: String },
}

/// Assemble a target's mosaics (one per filter). Skips filters whose
/// inputs and settings are unchanged unless `force`. `fits` (`[process]
/// fits`) also writes a FITS copy beside each, unchanged ones included.
#[allow(clippy::too_many_arguments)]
pub fn assemble(g: &Group, out: &Path, s: &Settings, cat: &Catalog, force: bool, fits: bool, progress: &(dyn Fn(Step) + Sync), cancel: &AtomicBool) -> Result<Vec<Report>> {
    let reps = assemble_xisf(g, out, s, cat, force, progress, cancel)?;
    if fits {
        for r in &reps {
            crate::fits::ensure_copy(&r.out)?;
        }
    }
    Ok(reps)
}

fn assemble_xisf(g: &Group, out: &Path, s: &Settings, cat: &Catalog, force: bool, progress: &(dyn Fn(Step) + Sync), cancel: &AtomicBool) -> Result<Vec<Report>> {
    let t0 = std::time::Instant::now();
    let sig = signature(g, s);
    let mut reports = load_reports(out);
    let filters = g.filters();
    if !force {
        let all: Vec<Report> = filters.iter().filter_map(|f| reports.get(&key(&g.target, f)).cloned()).filter(|r| r.signature == sig && r.out.is_file()).collect();
        if all.len() == filters.len() {
            for f in &filters {
                progress(Step::Skipped { filter: f.clone() });
            }
            return Ok(all);
        }
    }
    // 1. Solve every master.
    let learned = std::sync::Mutex::new(solve::Learned::default());
    let done = AtomicUsize::new(0);
    let total = g.inputs.len();
    progress(Step::Solve { done: 0, total });
    // Solve one first so the rest start from its measured scale.
    let first = solve_panel(&g.inputs[0].path, cat, &learned, s.distortion);
    done.fetch_add(1, Ordering::Relaxed);
    let mut solved: Vec<Result<(Geometry, Vec<(f64, f64)>)>> = vec![first];
    let rest: Vec<Result<(Geometry, Vec<(f64, f64)>)>> = g.inputs[1..]
        .par_iter()
        .with_max_len(1)
        .map(|i| {
            if cancel.load(Ordering::Relaxed) {
                anyhow::bail!("cancelled");
            }
            let r = solve_panel(&i.path, cat, &learned, s.distortion);
            progress(Step::Solve { done: done.fetch_add(1, Ordering::Relaxed) + 1, total });
            r
        })
        .collect();
    solved.extend(rest);
    anyhow::ensure!(!cancel.load(Ordering::Relaxed), "cancelled");
    let geoms: Vec<&Geometry> = solved.iter().filter_map(|r| r.as_ref().ok().map(|x| &x.0)).collect();
    anyhow::ensure!(geoms.len() >= 2, "fewer than two panels solved: {}", solved.iter().filter_map(|r| r.as_ref().err().map(|e| e.to_string())).collect::<Vec<_>>().join("; "));
    let canvas = make_canvas(&geoms, s);
    let solve_secs = t0.elapsed().as_secs_f64();
    let feather_px = s.feather_arcmin * 60.0 / canvas.scale;

    let mut written = Vec::new();
    for filter in &filters {
        if cancel.load(Ordering::Relaxed) {
            anyhow::bail!("cancelled");
        }
        let tf = std::time::Instant::now();
        let idx: Vec<usize> = (0..g.inputs.len()).filter(|&k| &g.inputs[k].filter == filter).collect();
        let done = AtomicUsize::new(0);
        let n = idx.len();
        progress(Step::Warp { filter: filter.clone(), done: 0, total: n });
        let tiles: Vec<Option<Tile>> = idx
            .par_iter()
            .with_max_len(1)
            .map(|&k| {
                let t = match &solved[k] {
                    Ok((geo, _)) if !cancel.load(Ordering::Relaxed) => make_tile(&g.inputs[k].path, geo, &canvas, feather_px).ok(),
                    _ => None,
                };
                progress(Step::Warp { filter: filter.clone(), done: done.fetch_add(1, Ordering::Relaxed) + 1, total: n });
                t
            })
            .collect();
        anyhow::ensure!(!cancel.load(Ordering::Relaxed), "cancelled");
        progress(Step::Match { filter: filter.clone() });
        let mut pairs: Pairs = Vec::new();
        for a in 0..n {
            for b in a + 1..n {
                if let (Some(ta), Some(tb)) = (&tiles[a], &tiles[b]) {
                    let bl = overlap_blocks(ta, tb, &canvas);
                    if bl.len() >= 40 {
                        pairs.push((a, b, bl));
                    }
                }
            }
        }
        let fx = |x: usize| (x as f64 + 0.5) / canvas.width as f64 - 0.5;
        let fy = |y: usize| (y as f64 + 0.5) / canvas.height as f64 - 0.5;

        // Seams before and after, and star alignment in each overlap.
        let star_canvas = |k: usize, linear: bool| -> Vec<(f64, f64)> {
            let Ok((geo, centres)) = &solved[idx[k]] else { return Vec::new() };
            let gg = if linear { geo.without_distortion() } else { geo.clone() };
            centres.iter().filter_map(|&(x, y)| {
                let (ra, dec) = gg.pixel_to_sky(x, y);
                canvas.wcs.sky_to_pixel(ra, dec)
            }).collect()
        };
        let stars_d: Vec<Vec<(f64, f64)>> = (0..n).map(|k| star_canvas(k, false)).collect();
        let stars_l: Vec<Vec<(f64, f64)>> = (0..n).map(|k| star_canvas(k, true)).collect();
        let in_both = |a: usize, b: usize, st: &Vec<Vec<(f64, f64)>>| -> Vec<(f64, f64)> {
            let (Some(ta), Some(tb)) = (&tiles[a], &tiles[b]) else { return Vec::new() };
            st[a].iter().copied().filter(|&p| {
                let (x, y) = (p.0.round() as i64, p.1.round() as i64);
                x >= 0 && y >= 0 && ta.at(x as usize, y as usize).is_finite() && tb.at(x as usize, y as usize).is_finite()
            }).collect()
        };
        let ratios: Vec<Option<(f64, f64, usize)>> = pairs
            .par_iter()
            .map(|(a, b, _)| star_ratio(tiles[*a].as_ref()?, tiles[*b].as_ref()?, &in_both(*a, *b, &stars_d), &in_both(*b, *a, &stars_d), canvas.bin))
            .collect();
        let bright = if s.match_brightness { match_brightness(&tiles, &pairs, &ratios, s.curvature) } else { vec![Brightness::default(); n] };
        let in_tile = |t: &Tile, p: (f64, f64)| -> bool {
            let (x, y) = (p.0.round() as i64, p.1.round() as i64);
            x >= 0 && y >= 0 && t.at(x as usize, y as usize).is_finite()
        };
        let align = |a: usize, b: usize, st: &Vec<Vec<(f64, f64)>>| -> (usize, f64) {
            let (Some(ta), Some(tb)) = (&tiles[a], &tiles[b]) else { return (0, f64::NAN) };
            let pa: Vec<(f64, f64)> = st[a].iter().copied().filter(|&p| in_tile(ta, p) && in_tile(tb, p)).collect();
            let pb: Vec<(f64, f64)> = st[b].iter().copied().filter(|&p| in_tile(ta, p) && in_tile(tb, p)).collect();
            let m = match_nearest(&pa, &pb, 3.0 / canvas.bin as f64 + 1.0);
            if m.is_empty() {
                return (0, f64::NAN);
            }
            let mut r: Vec<f64> = m.iter().map(|&(i, j)| (pa[i].0 - pb[j].0).hypot(pa[i].1 - pb[j].1) * canvas.bin as f64).collect();
            r.sort_by(f64::total_cmp);
            let core = &r[..(r.len() * 9 / 10).max(1)];
            (m.len(), (core.iter().map(|v| v * v).sum::<f64>() / core.len() as f64).sqrt())
        };
        let seams: Vec<Seam> = pairs
            .par_iter()
            .map(|(a, b, bl)| {
                let (ta, tb) = (tiles[*a].as_ref().unwrap(), tiles[*b].as_ref().unwrap());
                let nz = ta.noise.hypot(tb.noise).max(1e-12);
                let (ba, bb) = (bright[*a], bright[*b]);
                let mut before: Vec<f64> = bl.iter().map(|k| k.0 - k.1).collect();
                let mut after: Vec<f64> = bl.iter().map(|k| ba.apply(k.0, k.2, k.3) - bb.apply(k.1, k.2, k.3)).collect();
                let (ns, rms) = align(*a, *b, &stars_d);
                let (_, rms_l) = align(*a, *b, &stars_l);
                Seam {
                    a: g.inputs[idx[*a]].object.clone(),
                    b: g.inputs[idx[*b]].object.clone(),
                    blocks: bl.len(),
                    before: crate::stats::median_f64(&mut before).unwrap_or(0.0) / nz,
                    after: crate::stats::median_f64(&mut after).unwrap_or(0.0) / nz,
                    stars: ns,
                    star_rms: rms,
                    star_rms_linear: rms_l,
                }
            })
            .collect();

        // 5. Blend, row by row.
        progress(Step::Blend { filter: filter.clone() });
        let (cw, ch) = (canvas.width, canvas.height);
        let weight: Vec<f64> = (0..n).map(|k| tiles[k].as_ref().map(|t| 1.0 / (bright[k].scale * t.noise).max(1e-12).powi(2)).unwrap_or(0.0)).collect();
        let wmax = weight.iter().cloned().fold(0.0, f64::max).max(1e-300);
        let mut mosaic = vec![0f32; cw * ch];
        mosaic.par_chunks_mut(cw).enumerate().for_each(|(y, row)| {
            let mut num = vec![0f64; cw];
            let mut den = vec![0f64; cw];
            for (k, t) in tiles.iter().enumerate() {
                let Some(t) = t else { continue };
                if y < t.y0 || y >= t.y0 + t.h {
                    continue;
                }
                let br = bright[k];
                let wk = weight[k] / wmax;
                let yy = fy(y);
                let r = (y - t.y0) * t.w;
                for x in 0..t.w {
                    let v = t.data[r + x];
                    if !v.is_finite() {
                        continue;
                    }
                    let cx = t.x0 + x;
                    let wv = wk * t.feather[r + x] as f64;
                    num[cx] += wv * br.apply(v as f64, fx(cx), yy);
                    den[cx] += wv;
                }
            }
            for x in 0..cw {
                row[x] = if den[x] > 0.0 { (num[x] / den[x]) as f32 } else { f32::NAN };
            }
        });
        let noises: Vec<f64> = tiles.iter().map(|t| t.as_ref().map(|t| t.noise).unwrap_or(0.0)).collect();
        drop(tiles);
        // Keep it positive (lifted by the 0.05th percentile if that's below
        // zero), and never 0 where there's data: 0 means empty.
        super::combine::keep_positive(&mut mosaic);

        let path = output_path(out, &g.target, filter);
        std::fs::create_dir_all(path.parent().unwrap())?;
        let mut h = Header::default();
        let first = crate::meta::read_meta(&g.inputs[idx[0]].path).map(|m| m.header).unwrap_or_default();
        for k in ["TELESCOP", "INSTRUME", "FOCALLEN", "XPIXSZ", "YPIXSZ", "GAIN", "OFFSET", "EGAIN", "SITELAT", "SITELONG"] {
            if let Some(v) = first.get(k) {
                h.push(k, v, "");
            }
        }
        // Plain values: the writers add the quotes.
        h.push("IMAGETYP", "Master Light", "");
        h.push("OBJECT", g.target.as_str(), "Mosaic");
        h.push("FILTER", filter.to_string(), "");
        h.push("CTYPE1", "RA---TAN", "");
        h.push("CTYPE2", "DEC--TAN", "");
        h.push("CRVAL1", format!("{:.9}", canvas.wcs.ra), "");
        h.push("CRVAL2", format!("{:.9}", canvas.wcs.dec), "");
        h.push("CRPIX1", format!("{:.3}", canvas.wcs.cx + 1.0), "first stored row is row 1");
        h.push("CRPIX2", format!("{:.3}", canvas.wcs.cy + 1.0), "");
        h.push("CD1_1", format!("{:.12e}", canvas.wcs.cd[0]), "");
        h.push("CD1_2", format!("{:.12e}", canvas.wcs.cd[1]), "");
        h.push("CD2_1", format!("{:.12e}", canvas.wcs.cd[2]), "");
        h.push("CD2_2", format!("{:.12e}", canvas.wcs.cd[3]), "");
        h.push("CRUX-MOS", idx.len().to_string(), "Panels in the mosaic");
        h.push("XBINNING", canvas.bin.to_string(), "");
        h.push("YBINNING", canvas.bin.to_string(), "");
        let exposure: f64 = idx.iter().map(|&k| g.inputs[k].exposure).sum();
        let frames: usize = idx.iter().map(|&k| g.inputs[k].frames).sum();
        h.push("NCOMBINE", frames.to_string(), "Frames over all panels");
        crate::xisf::write_image(&path, &Image::from_data(cw, ch, mosaic), &h, None)?;

        let panels: Vec<PanelReport> = idx
            .iter()
            .enumerate()
            .map(|(k, &i)| {
                let inp = &g.inputs[i];
                let (geo, err) = match &solved[i] {
                    Ok((geo, _)) => (Some(geo), None),
                    Err(e) => (None, Some(format!("{e:#}"))),
                };
                let outline = geo
                    .map(|geo| {
                        let (w, hh) = (geo.width as f64, geo.height as f64);
                        [(0.0, 0.0), (w, 0.0), (w, hh), (0.0, hh)]
                            .iter()
                            .filter_map(|&(x, y)| {
                                let (ra, dec) = geo.pixel_to_sky(x, y);
                                canvas.wcs.sky_to_pixel(ra, dec).map(|p| [p.0, p.1])
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                PanelReport {
                    object: inp.object.clone(),
                    panel: crate::index::panel_of(&inp.object),
                    path: inp.path.clone(),
                    frames: inp.frames,
                    exposure: inp.exposure,
                    scale: geo.map(|g| g.solution.scale).unwrap_or(0.0),
                    rotation: geo.map(|g| g.solution.rotation).unwrap_or(0.0),
                    ra: geo.map(|g| g.solution.ra).unwrap_or(0.0),
                    dec: geo.map(|g| g.solution.dec).unwrap_or(0.0),
                    matched: geo.map(|g| g.matched).unwrap_or(0),
                    rms_linear: geo.map(|g| g.rms_linear).unwrap_or(f64::NAN),
                    rms: geo.map(|g| g.rms).unwrap_or(f64::NAN),
                    distortion_max: geo.map(|g| g.distortion_max).unwrap_or(0.0),
                    brightness: bright[k],
                    noise: noises[k],
                    outline,
                    error: err,
                }
            })
            .collect();
        let r = Report {
            target: g.target.clone(),
            filter: filter.clone(),
            out: path.clone(),
            canvas: canvas.clone(),
            panels,
            seams,
            settings: s.clone(),
            exposure,
            frames,
            seconds: tf.elapsed().as_secs_f64() + solve_secs / filters.len() as f64,
            signature: sig.clone(),
        };
        reports.insert(key(&g.target, filter), r.clone());
        crate::paths::write_atomic(&report_path(out), &serde_json::to_vec_pretty(&reports)?)?;
        progress(Step::Written { filter: filter.clone(), path });
        written.push(r);
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn least_squares_recovers_a_plane() {
        let mut rows = Vec::new();
        for i in 0..50 {
            let (x, y) = (i as f64 * 0.1, (i * 7 % 11) as f64);
            rows.push((vec![(0, 1.0), (1, x), (2, y)], 2.0 + 3.0 * x - 0.5 * y, 1.0));
        }
        let c = lstsq(&rows, 3).unwrap();
        assert!((c[0] - 2.0).abs() < 1e-9 && (c[1] - 3.0).abs() < 1e-9 && (c[2] + 0.5).abs() < 1e-9, "{c:?}");
    }

    #[test]
    fn distance_is_zero_outside_and_grows_inside() {
        let (w, h) = (20, 10);
        let valid: Vec<bool> = (0..w * h).map(|i| (i % w) >= 5).collect();
        let d = distance(&valid, w, h);
        assert_eq!(d[3], 0.0);
        assert!((d[5 * w + 5] - 1.0).abs() < 1e-6);
        assert!((d[5 * w + 9] - 5.0).abs() < 1e-6, "{}", d[5 * w + 9]);
    }

    #[test]
    fn distortion_round_trips() {
        let sol = Solution { ra: 80.0, dec: -69.0, scale: 0.67, rotation: 0.0, mirrored: true, cd: [-0.67 / 3600.0, 0.0, 0.0, -0.67 / 3600.0], cx: 3124.0, cy: 2088.0, matched: 0, rms: 0.0, offset_arcmin: None, target_offset_arcmin: None, version: 1 };
        let d = Distortion { order: 3, cx: 3124.0, cy: 2088.0, norm: 3124.0, ax: vec![0.1, 0.5, -0.2, 0.3, 0.0, 0.1, 1.0, 0.0, 0.2, 0.0], ay: vec![-0.2, 0.1, 0.4, 0.0, 0.2, -0.1, 0.0, 0.8, 0.0, 0.3] };
        let g = Geometry { solution: sol, distortion: Some(d), width: 6248, height: 4176, matched: 0, rms_linear: 0.0, rms: 0.0, distortion_max: 0.0 };
        for &(x, y) in &[(0.0, 0.0), (6000.0, 300.0), (3124.0, 2088.0), (100.0, 4000.0)] {
            let (ra, dec) = g.pixel_to_sky(x, y);
            let (bx, by) = g.sky_to_pixel(ra, dec).unwrap();
            assert!((bx - x).abs() < 1e-3 && (by - y).abs() < 1e-3, "{x},{y} → {bx},{by}");
        }
    }
}
