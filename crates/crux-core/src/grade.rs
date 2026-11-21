//! Grade lights against their cohorts and explain every flag.
//!
//! Cohorts:
//! - **night**: same night, filter, exposure and binning, across all panels.
//!   Seeing, focus and tracking are properties of the night, and mosaic
//!   panels have only a handful of frames each.
//! - **panel**: same object, filter, exposure and binning, across nights.
//!   Sky level and gradient include the target's own nebulosity, so they're
//!   only comparable for the same field.

use crate::analysis::LightMetrics;
use crate::astro::SkyContext;
use crate::classify::{FrameType, Role};
use crate::config::Config;
use crate::scan::Frame;
use crate::stats::{median_f64, median_sigma_f64, percentile_f64, robust_z};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Grade {
    Good,
    Warn,
    Reject,
}

impl Grade {
    pub fn name(self) -> &'static str {
        match self {
            Grade::Good => "good",
            Grade::Warn => "warn",
            Grade::Reject => "reject",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Check {
    Read,
    Type,
    Stars,
    Focus,
    Tracking,
    Sky,
    Gradient,
    Transparency,
    Patchy,
    Trail,
    // Calibration frames.
    Level,
    Noise,
    Structure,
    HotPixels,
    Temperature,
    Exposure,
    Saturated,
    FlatChange,
    FlatMatch,
    Source,
    /// A set's members (how many were rejected).
    Frames,
    Banding,
    /// Against the flat library.
    Library,
    /// Your decision (GUI or `crux plan --reject`).
    Manual,
}

impl Check {
    pub fn name(self) -> &'static str {
        match self {
            Check::Read => "read",
            Check::Type => "type",
            Check::Stars => "stars",
            Check::Focus => "focus",
            Check::Tracking => "tracking",
            Check::Sky => "sky",
            Check::Gradient => "gradient",
            Check::Transparency => "transparency",
            Check::Patchy => "patchy cloud",
            Check::Trail => "trail",
            Check::Level => "level",
            Check::Noise => "noise",
            Check::Structure => "structure",
            Check::HotPixels => "hot pixels",
            Check::Temperature => "temperature",
            Check::Exposure => "exposure",
            Check::Saturated => "saturated",
            Check::FlatChange => "flat change",
            Check::FlatMatch => "flat vs lights",
            Check::Source => "source",
            Check::Frames => "frames",
            Check::Banding => "banding",
            Check::Library => "library",
            Check::Manual => "yours",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Flag {
    pub check: Check,
    pub grade: Grade,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameReport {
    pub frame: usize,
    pub grade: Grade,
    pub flags: Vec<Flag>,
    /// Context that doesn't change the grade (Moon, optics, cloud sensor).
    pub notes: Vec<String>,
    pub sky: Option<SkyContext>,
    /// Sky background, e-/px/s.
    pub sky_e_s: Option<f64>,
    /// Sky / panel's best.
    pub sky_ratio: Option<f64>,
    /// Gradient across the frame, % of sky.
    pub gradient_pct: Option<f64>,
    /// FWHM / night median.
    pub fwhm_ratio: Option<f64>,
    /// 1.0 = the field's best frame.
    pub transparency: Option<crate::transparency::FrameTransparency>,
}

impl FrameReport {
    pub fn new(frame: usize) -> Self {
        FrameReport {
            frame,
            grade: Grade::Good,
            flags: Vec::new(),
            notes: Vec::new(),
            sky: None,
            sky_e_s: None,
            sky_ratio: None,
            gradient_pct: None,
            fwhm_ratio: None,
            transparency: None,
        }
    }

    pub fn flag(&mut self, check: Check, grade: Grade, reason: String) {
        self.grade = self.grade.max(grade);
        self.flags.push(Flag { check, grade, reason });
    }

    /// One line: the worst flags' reasons.
    pub fn summary(&self) -> String {
        let mut f: Vec<&Flag> = self.flags.iter().filter(|f| f.grade != Grade::Good).collect();
        f.sort_by(|a, b| b.grade.cmp(&a.grade));
        f.iter().map(|f| f.reason.as_str()).collect::<Vec<_>>().join("; ")
    }
}

/// Per-frame values derived once, used by several checks.
struct Derived {
    sky_e_s: Option<f64>,
    grad: Option<(f64, f64)>,
}

fn key_night(f: &Frame) -> String {
    format!(
        "{:?}|{}|{:.2}|{}",
        f.class.night,
        f.meta.filter.clone().unwrap_or_default().to_uppercase(),
        f.meta.exptime.unwrap_or(0.0),
        f.meta.binning()
    )
}

fn key_panel(f: &Frame) -> String {
    format!(
        "{}|{}|{:.2}|{}",
        f.meta.object.clone().unwrap_or_default().to_lowercase(),
        f.meta.filter.clone().unwrap_or_default().to_uppercase(),
        f.meta.exptime.unwrap_or(0.0),
        f.meta.binning()
    )
}

fn direction(dx: f64, dy: f64) -> &'static str {
    // Image coordinates: +x right, +y down (row 0 is the top).
    let a = dy.atan2(dx).to_degrees();
    match a {
        a if (-22.5..22.5).contains(&a) => "right",
        a if (22.5..67.5).contains(&a) => "bottom right",
        a if (67.5..112.5).contains(&a) => "bottom",
        a if (112.5..157.5).contains(&a) => "bottom left",
        a if (-67.5..-22.5).contains(&a) => "top right",
        a if (-112.5..-67.5).contains(&a) => "top",
        a if (-157.5..-112.5).contains(&a) => "top left",
        _ => "left",
    }
}

/// Where a cell of the 4×4 transparency grid sits, in words.
fn cell_place(i: usize) -> &'static str {
    let g = crate::transparency::GRID;
    let (row, col) = (i / g, i % g);
    match (row, col) {
        (0, 0) => "top left",
        (0, c) if c == g - 1 => "top right",
        (0, _) => "top",
        (r, 0) if r == g - 1 => "bottom left",
        (r, c) if r == g - 1 && c == g - 1 => "bottom right",
        (r, _) if r == g - 1 => "bottom",
        (_, 0) => "left",
        (_, c) if c == g - 1 => "right",
        _ => "centre",
    }
}

/// Grade every raw light that has metrics (or failed analysis).
/// `pedestal(i)` is the bias level in ADU for frame i.
pub fn grade_lights(
    frames: &[Frame],
    metrics: &HashMap<usize, LightMetrics>,
    errors: &HashMap<usize, String>,
    pedestal: &dyn Fn(usize) -> f64,
    cfg: &Config,
) -> Vec<FrameReport> {
    let transp = transparency_all(frames, metrics, cfg);
    grade_lights_with(frames, metrics, errors, pedestal, cfg, &transp)
}

/// Two trails along the same line on the sensor: within 1° and 6 px, with
/// their lit parts overlapping.
fn same_line(a: &crate::trails::Trail, b: &crate::trails::Trail) -> bool {
    let da = (a.angle - b.angle).rem_euclid(180.0);
    if da.min(180.0 - da) > 1.0 {
        return false;
    }
    let (ux, uy) = (a.angle.to_radians().cos(), a.angle.to_radians().sin());
    let off = |x: f32, y: f32| (x - a.x0) * -uy + (y - a.y0) * ux;
    let (bm_x, bm_y) = (0.5 * (b.x0 + b.x1), 0.5 * (b.y0 + b.y1));
    if off(bm_x, bm_y).abs() > 6.0 {
        return false;
    }
    let along = |x: f32, y: f32| (x - a.x0) * ux + (y - a.y0) * uy;
    let (a0, a1) = (0.0f32.min(along(a.x1, a.y1)), 0.0f32.max(along(a.x1, a.y1)));
    let (b0, b1) = (along(b.x0, b.y0).min(along(b.x1, b.y1)), along(b.x0, b.y0).max(along(b.x1, b.y1)));
    a0.max(b0) < a1.min(b1)
}

/// Trails that are fixed on the sensor: the same line in at least 3 other
/// frames and a fifth of the frames of that camera and size. A satellite
/// never crosses the same pixels twice; amplifier glow, a warm row or a
/// column defect does, in raw lights before darks take it out (the ASI294MC
/// has a glow streak along row ~720). By frame: indices into its trails.
fn fixed_trails(frames: &[Frame], metrics: &HashMap<usize, LightMetrics>, lights: &[usize]) -> HashMap<usize, Vec<usize>> {
    let mut by_sensor: HashMap<(String, u32, u32), Vec<usize>> = HashMap::new();
    for &i in lights.iter().filter(|i| metrics.contains_key(i)) {
        let m = &frames[i].meta;
        by_sensor.entry((m.instrume.clone().unwrap_or_default(), m.width, m.height)).or_default().push(i);
    }
    let mut out: HashMap<usize, Vec<usize>> = HashMap::new();
    for ids in by_sensor.values() {
        let need = 3usize.max(ids.len() / 5);
        for &i in ids {
            for (k, t) in metrics[&i].trails.iter().enumerate() {
                let others = ids.iter().filter(|&&j| j != i && metrics[&j].trails.iter().any(|u| same_line(t, u) || same_line(u, t))).count();
                if others >= need {
                    out.entry(i).or_default().push(k);
                }
            }
        }
    }
    out
}

/// Raw lights with metrics, keyed by field (object, filter, exposure, binning).
fn panels(frames: &[Frame], metrics: &HashMap<usize, LightMetrics>) -> HashMap<String, Vec<usize>> {
    let mut panel: HashMap<String, Vec<usize>> = HashMap::new();
    let mut ids: Vec<usize> = metrics.keys().copied().filter(|&i| frames[i].class.frame_type == FrameType::Light && frames[i].class.role == Role::Raw).collect();
    ids.sort_unstable();
    for i in ids {
        panel.entry(key_panel(&frames[i])).or_default().push(i);
    }
    panel
}

/// Transparency of every light within its field. The slow part of grading
/// (star matching), and independent of the reject limits, so the GUI
/// computes it once and regrades with [`grade_lights_with`] as limits move.
pub fn transparency_all(
    frames: &[Frame],
    metrics: &HashMap<usize, LightMetrics>,
    cfg: &Config,
) -> HashMap<usize, crate::transparency::FrameTransparency> {
    use rayon::prelude::*;
    let th = &cfg.thresholds;
    let panel = panels(frames, metrics);
    let fields: Vec<&Vec<usize>> = panel.values().filter(|ids| ids.len() >= th.transparency.min_frames).collect();
    fields
        .par_iter()
        .map(|ids| crate::transparency::cohort(ids, metrics, th.transparency.min_matches))
        .flatten()
        .collect()
}

/// [`grade_lights`] with transparency already measured.
pub fn grade_lights_with(
    frames: &[Frame],
    metrics: &HashMap<usize, LightMetrics>,
    errors: &HashMap<usize, String>,
    pedestal: &dyn Fn(usize) -> f64,
    cfg: &Config,
    transp: &HashMap<usize, crate::transparency::FrameTransparency>,
) -> Vec<FrameReport> {
    let th = &cfg.thresholds;
    let lights: Vec<usize> = (0..frames.len())
        .filter(|&i| frames[i].class.frame_type == FrameType::Light && frames[i].class.role == Role::Raw)
        .filter(|i| metrics.contains_key(i) || errors.contains_key(i))
        .collect();

    let derived: HashMap<usize, Derived> = lights
        .iter()
        .filter_map(|&i| {
            let m = metrics.get(&i)?;
            let f = &frames[i];
            let signal = m.sky_adu as f64 - pedestal(i);
            let egain = f.meta.egain.unwrap_or(cfg.camera.egain);
            let sky_e_s = f.meta.exptime.filter(|&e| e > 0.0).map(|e| signal.max(0.0) * egain / e);
            let grad = (signal > 1.0).then(|| (m.grad_x_adu as f64 / signal * 100.0, m.grad_y_adu as f64 / signal * 100.0));
            Some((i, Derived { sky_e_s, grad }))
        })
        .collect();

    let mut night: HashMap<String, Vec<usize>> = HashMap::new();
    let mut panel: HashMap<String, Vec<usize>> = HashMap::new();
    for &i in lights.iter().filter(|i| metrics.contains_key(i)) {
        night.entry(key_night(&frames[i])).or_default().push(i);
        panel.entry(key_panel(&frames[i])).or_default().push(i);
    }
    let stat = |ids: &[usize], f: &dyn Fn(usize) -> Option<f64>| -> Option<(f64, f64, usize)> {
        let v: Vec<f64> = ids.iter().filter_map(|&i| f(i)).collect();
        (v.len() >= th.min_cohort).then(|| median_sigma_f64(&v).map(|(m, s)| (m, s, v.len()))).flatten()
    };

    let fixed = fixed_trails(frames, metrics, &lights);
    let mut out = Vec::with_capacity(lights.len());
    for &i in &lights {
        let f = &frames[i];
        let mut r = FrameReport { transparency: transp.get(&i).cloned(), ..FrameReport::new(i) };
        let Some(m) = metrics.get(&i) else {
            r.flag(Check::Read, Grade::Reject, format!("couldn't analyse: {}", errors.get(&i).map(String::as_str).unwrap_or("unknown error")));
            out.push(r);
            continue;
        };
        let d = &derived[&i];
        let ncoh = &night[&key_night(f)];
        let pcoh = &panel[&key_panel(f)];

        // Sky context from the exposure midpoint.
        if let Some(t) = f.meta.date_obs {
            let mid = t + chrono::Duration::milliseconds((f.meta.exptime.unwrap_or(0.0) * 500.0) as i64);
            let lat = f.meta.sitelat.unwrap_or(cfg.site.latitude);
            let lon = f.meta.sitelong.unwrap_or(cfg.site.longitude);
            let target = f.meta.ra.zip(f.meta.dec);
            r.sky = Some(SkyContext::at(mid, lat, lon, target));
        }

        // Type: a "light" with no stars at the bias level.
        let ptype = m.pixel_type(pedestal(i) as f32, f.meta.exptime);
        if matches!(ptype, FrameType::Dark | FrameType::Bias) {
            r.flag(
                Check::Type,
                Grade::Reject,
                format!(
                    "no stars and the sky is at the bias level ({:.0} ADU): looks like a {} (cap on or roof closed?)",
                    m.sky_adu,
                    ptype.name()
                ),
            );
        }

        // Star count: gross failures only.
        if let Some((med, _, _)) = stat(ncoh, &|j| Some(metrics[&j].n_stars as f64)) {
            let frac = m.n_stars as f64 / med.max(1.0);
            let msg = format!("only {} stars, night median {:.0}", m.n_stars, med);
            if frac < th.stars.reject_frac {
                r.flag(Check::Stars, Grade::Reject, msg);
            } else if frac < th.stars.warn_frac {
                r.flag(Check::Stars, Grade::Warn, msg);
            }
        }

        // Transparency and patchy cloud.
        if let Some(tr) = r.transparency.clone() {
            let tt = &th.transparency;
            let v = tr.value;
            if v < tt.warn {
                let mut why = Vec::new();
                // How much of the loss airmass explains, against the best frame.
                let best_air = pcoh
                    .iter()
                    .filter_map(|j| transp.get(j).map(|t| (t.value, frames[*j].meta.airmass)))
                    .max_by(|a, b| a.0.total_cmp(&b.0))
                    .and_then(|b| b.1);
                let mut expected = 1.0;
                if let (Some(x), Some(xb)) = (f.meta.airmass, best_air) {
                    expected = 10f64.powf(-0.4 * tt.extinction_k * (x - xb)).min(1.0);
                    if expected < 0.97 {
                        why.push(format!("airmass {x:.2} (best frame {xb:.2}) explains only {expected:.2}"));
                    }
                }
                if v / expected < tt.warn {
                    why.push("cloud or haze".into());
                }
                if let Some(cc) = f.meta.cloudcvr.filter(|c| *c >= 50.0) {
                    why.push(format!("cloud sensor {cc:.0}%"));
                }
                let msg = format!(
                    "transparency {v:.2}: stars at {:.0}% of their brightness in the field's best frame ({} matched) — {}",
                    v * 100.0,
                    tr.n_matched,
                    why.join(", ")
                );
                r.flag(Check::Transparency, if v < tt.reject { Grade::Reject } else { Grade::Warn }, msg);
            }
            if tr.patchiness.is_some_and(|p| p >= tt.patchy_warn) {
                let valid: Vec<(usize, f64)> = tr.cells.iter().enumerate().filter_map(|(i, c)| c.map(|v| (i, v))).collect();
                let lost: Vec<usize> = tr.lost.iter().enumerate().filter(|(_, l)| **l).map(|(i, _)| i).collect();
                let worst = valid.iter().min_by(|a, b| a.1.total_cmp(&b.1)).map(|c| c.0).unwrap_or(0);
                let (lo, hi) = valid.iter().fold((f64::MAX, 0f64), |(lo, hi), c| (lo.min(c.1), hi.max(c.1)));
                let mut msg = format!(
                    "patchy cloud: transparency varies {lo:.2}–{hi:.2} across the frame, worst toward the {}",
                    cell_place(worst)
                );
                if !lost.is_empty() {
                    msg.push_str(&format!(" ({} of 16 cells lost their stars)", lost.len()));
                }
                r.flag(Check::Patchy, Grade::Warn, msg);
            }
        }

        // Trails, less lines fixed on the sensor.
        let is_fixed = |k: usize| fixed.get(&i).is_some_and(|v| v.contains(&k));
        let trails: Vec<&crate::trails::Trail> = m.trails.iter().enumerate().filter(|(k, _)| !is_fixed(*k)).map(|(_, t)| t).collect();
        for t in m.trails.iter().enumerate().filter(|(k, _)| is_fixed(*k)).map(|(_, t)| t) {
            r.notes.push(format!(
                "line at {:.0}° from ({:.0}, {:.0}) to ({:.0}, {:.0}) is in the same place in other frames: a sensor feature (glow or a bad row), not a trail; darks remove it",
                t.angle, t.x0, t.y0, t.x1, t.y1
            ));
        }
        if !trails.is_empty() {
            let noise = (m.noise_adu as f64).max(1e-6);
            let worst = trails.iter().map(|t| t.brightness_adu as f64).fold(0.0, f64::max);
            let desc: Vec<String> = trails
                .iter()
                .map(|t| {
                    format!(
                        "{} at {:.0}°, {:.0} px long, {:.0} ADU ({:.1}× the pixel noise)",
                        t.kind.name(),
                        t.angle,
                        t.length,
                        t.brightness_adu,
                        t.brightness_adu as f64 / noise
                    )
                })
                .collect();
            let n = trails.len();
            let msg = format!("{n} trail{}: {}", if n == 1 { "" } else { "s" }, desc.join(", "));
            let g = if worst / noise >= th.trails.reject_sigma { Grade::Reject } else { Grade::Warn };
            r.flag(Check::Trail, g, msg);
        }

        // Focus.
        if let Some(fw) = m.fwhm {
            let fw = fw as f64;
            let focus_ctx = || {
                let fp = f.meta.focpos.map(|p| {
                    let med = median_f64(&ncoh.iter().filter_map(|&j| frames[j].meta.focpos).collect::<Vec<_>>());
                    match med {
                        Some(mp) => format!("FOCPOS {p:.0} (night median {mp:.0})"),
                        None => format!("FOCPOS {p:.0}"),
                    }
                });
                let ft = f.meta.foctemp.map(|t| format!("FOCTEMP {t:.1} °C"));
                [fp, ft].into_iter().flatten().collect::<Vec<_>>().join(", ")
            };
            let hfr = m.hfr.map(|h| format!(" (HFR {h:.1})")).unwrap_or_default();
            if let Some((med, sig, _)) = stat(ncoh, &|j| metrics[&j].fwhm.map(f64::from)) {
                let ratio = fw / med;
                r.fwhm_ratio = Some(ratio);
                let z = robust_z(fw, med, sig, th.sigma_floor);
                let msg = format!("FWHM {fw:.1} px{hfr}, night median {med:.1} px — {}", focus_ctx());
                if ratio >= th.focus.reject_ratio && z >= th.min_z {
                    r.flag(Check::Focus, Grade::Reject, msg);
                } else if ratio >= th.focus.warn_ratio && z >= th.min_z {
                    r.flag(Check::Focus, Grade::Warn, msg);
                }
            }
            if fw > th.focus.abs_warn_px && !r.flags.iter().any(|x| x.check == Check::Focus) {
                r.flag(Check::Focus, Grade::Warn, format!("FWHM {fw:.1} px{hfr} is soft for this scope — {}", focus_ctx()));
            }
        }

        // Tracking vs optics.
        if let (Some(e), Some(coh)) = (m.ecc.map(f64::from), m.coherence.map(f64::from)) {
            let centre = m.centre().ecc.map(f64::from);
            let corner = m.corner_ecc().map(f64::from);
            let uniform = coh >= th.tracking.min_coherence && match (centre, corner) {
                (Some(c), Some(k)) => c >= 0.75 * k,
                _ => true,
            };
            let pa = m.pa.unwrap_or(0.0);
            let ecc_stat = stat(ncoh, &|j| metrics[&j].ecc.map(f64::from));
            let (excess, z, med_s) = match ecc_stat {
                Some((med, sig, _)) => (e - med, robust_z(e, med, sig, th.sigma_floor), format!(", night median {med:.2}")),
                None => (0.0, 0.0, String::new()),
            };
            let rel = |lim: f64| excess >= lim && z >= th.min_z;
            let grade = if !uniform {
                None
            } else if e >= th.tracking.reject_ecc || rel(th.tracking.rel_reject) {
                Some(Grade::Reject)
            } else if e >= th.tracking.warn_ecc || rel(th.tracking.rel_warn) {
                Some(Grade::Warn)
            } else {
                None
            };
            if let Some(g) = grade {
                r.flag(
                    Check::Tracking,
                    g,
                    format!("stars elongated: eccentricity {e:.2}{med_s}, at PA {pa:.0}° across the whole field (coherence {coh:.2}) — tracking or wind"),
                );
            }
            if let (Some(c), Some(k), Some(rad)) = (centre, corner, m.radial) {
                if k - c >= th.tracking.optics_excess && rad.abs() >= 0.3 {
                    let pattern = if rad > 0.0 { "radial" } else { "tangential" };
                    r.notes.push(format!(
                        "corners more elongated than the centre ({k:.2} vs {c:.2}), {pattern} pattern: coma or tilt, not tracking"
                    ));
                }
            }
        }

        // Sky brightness, against the panel's best frames.
        r.sky_e_s = d.sky_e_s;
        if let Some(sky) = d.sky_e_s {
            let vals: Vec<f64> = pcoh.iter().filter_map(|j| derived[j].sky_e_s).collect();
            let (vals, scope) = if vals.len() >= th.min_cohort {
                (vals, "panel")
            } else {
                (ncoh.iter().filter_map(|j| derived[j].sky_e_s).collect(), "night")
            };
            if vals.len() >= th.min_cohort {
                if let Some(best) = percentile_f64(&vals, th.sky.best_percentile).filter(|b| *b > 0.0) {
                    let ratio = sky / best;
                    r.sky_ratio = Some(ratio);
                    if ratio >= th.sky.warn_ratio {
                        let mut why = r.sky.map(|s| s.describe()).unwrap_or_default();
                        if why.is_empty() {
                            why.push("no Moon or twilight: cloud, haze or light pollution".into());
                        }
                        if let Some(cc) = f.meta.cloudcvr.filter(|c| *c >= 50.0) {
                            why.push(format!("cloud sensor {cc:.0}%"));
                        }
                        let msg = format!("sky {sky:.3} e-/px/s, {ratio:.1}× the {scope}'s best ({best:.3}) — {}", why.join(", "));
                        let g = if ratio >= th.sky.reject_ratio { Grade::Reject } else { Grade::Warn };
                        r.flag(Check::Sky, g, msg);
                    }
                }
            }
        }

        // Gradient, against the panel's typical gradient.
        if let Some((gx, gy)) = d.grad {
            r.gradient_pct = Some((gx * gx + gy * gy).sqrt());
            let others: Vec<(f64, f64)> = pcoh.iter().filter_map(|j| derived[j].grad).collect();
            if others.len() >= th.min_cohort {
                let mx = median_f64(&others.iter().map(|g| g.0).collect::<Vec<_>>()).unwrap_or(0.0);
                let my = median_f64(&others.iter().map(|g| g.1).collect::<Vec<_>>()).unwrap_or(0.0);
                let (dx, dy) = (gx - mx, gy - my);
                let excess = (dx * dx + dy * dy).sqrt();
                if excess >= th.gradient.note_pct {
                    let mut why = r.sky.map(|s| s.describe()).unwrap_or_default();
                    if why.is_empty() {
                        why.push("no Moon or twilight".into());
                    }
                    let msg = format!(
                        "extra sky gradient of {excess:.0}% across the frame, brighter toward the {} (panel typical {:.0}%) — {}",
                        direction(dx, dy),
                        (mx * mx + my * my).sqrt(),
                        why.join(", ")
                    );
                    if excess >= th.gradient.warn_pct {
                        r.flag(Check::Gradient, Grade::Warn, msg);
                    } else {
                        r.notes.push(msg);
                    }
                }
            }
        }

        // Context notes.
        if let Some(s) = &r.sky {
            let d = s.describe().join(", ");
            let said = r.flags.iter().map(|x| &x.reason).chain(&r.notes).any(|t| t.contains(&d));
            if !d.is_empty() && !said {
                r.notes.push(d);
            }
        }
        if let Some(cc) = f.meta.cloudcvr.filter(|c| *c >= 50.0) {
            if !r.flags.iter().any(|x| x.reason.contains("cloud sensor")) {
                r.notes.push(format!("cloud sensor {cc:.0}% (hint only)"));
            }
        }
        out.push(r);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trails::{Trail, TrailKind};

    fn t(angle: f32, x0: f32, y0: f32, x1: f32, y1: f32) -> Trail {
        Trail { kind: TrailKind::Satellite, angle, x0, y0, x1, y1, length: ((x1 - x0).hypot(y1 - y0)), brightness_adu: 20.0, coverage: 0.5, score: 18.0 }
    }

    #[test]
    fn same_line_on_the_sensor() {
        // The ASI294MC glow streak in two frames: lit parts differ, line the same.
        let a = t(0.2, 4104.0, 725.0, 2152.0, 718.0);
        let b = t(0.0, 4104.0, 723.0, 3336.0, 723.0);
        assert!(same_line(&a, &b) && same_line(&b, &a));
        // Same angle, 40 px away: a different line.
        assert!(!same_line(&a, &t(0.0, 4104.0, 765.0, 3336.0, 765.0)));
        // Same line, lit parts apart.
        assert!(!same_line(&t(0.0, 0.0, 100.0, 500.0, 100.0), &t(0.0, 900.0, 100.0, 1500.0, 100.0)));
        // Across the angle wrap (179.6° vs 0.1°).
        assert!(same_line(&t(179.6, 100.0, 50.0, 900.0, 55.0), &t(0.1, 200.0, 52.0, 800.0, 53.0)));
        // Different angle.
        assert!(!same_line(&a, &t(17.0, 0.0, 0.0, 2000.0, 611.0)));
    }
}
