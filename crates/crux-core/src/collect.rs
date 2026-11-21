//! Training bundles (`crux collect`): data for training crux's own
//! denoising (and later deconvolution) networks, from anyone's registered
//! frames, made so they can be shared.
//!
//! **Noise pairs.** A field's registered frames are dealt into two (or
//! four) equal parts with the same spread of FWHM, each part integrated
//! like a master, and the second stack of each pair brought onto the
//! first's brightness and background (`combine::match_to`). The two stacks
//! are independent noisy views of one scene: a network trained to turn one
//! into the other learns to remove noise without ever seeing a clean image
//! (Noise2Noise, Lehtinen et al. 2018). Halves and quarters give two noise
//! levels. Patches (256 px, Float32, values as stored: ADU ÷ `adu_scale`)
//! are cut at the same places from both stacks, half of them where there's
//! signal.
//!
//! **Star cutouts.** Up to 200 unsaturated stars spread over the first
//! stack, with FWHM, eccentricity and position: the stack's PSF across the
//! field, for training deconvolution later.
//!
//! **What's left out.** File names and paths, site coordinates, observer,
//! and the day (dates are cut to the month). Only the camera and optics
//! details listed in `SetInfo` go in; no FITS header is copied.
//!
//! A bundle is a folder: `manifest.json` (written last, so a folder without
//! one is incomplete), Float32 FITS cubes (one plane per patch), and
//! `bundle.html` showing what's inside.

use crate::config::Config;
use crate::header::Header;
use crate::image::Image;
use crate::process::combine;
use anyhow::{Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

pub const FORMAT: &str = "crux-training-bundle";
/// Bump when the bundle layout or meaning changes.
pub const BUNDLE_VERSION: u32 = 1;
/// Fewest frames in each part of a pair.
pub const MIN_PART: usize = 3;

#[derive(Debug, Clone)]
pub struct Options {
    /// Patch side, pixels.
    pub patch: usize,
    /// Size limit for the bundle, MB.
    pub max_mb: f64,
    /// Make quarter pairs as well as halves (when there are enough frames).
    pub quarters: bool,
    pub psf_stars: usize,
    /// Keep the integrated half and quarter stacks in `<out>\.stacks`.
    pub keep_stacks: bool,
    pub seed: u64,
}

impl Default for Options {
    fn default() -> Self {
        Options { patch: 256, max_mb: 500.0, quarters: true, psf_stars: 200, keep_stacks: false, seed: 0x5eed_c0ffee }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Source {
    /// crux's own registered frames (`crux-integration.json`).
    Crux,
    /// PixInsight WBPP (`*_r.xisf`).
    Wbpp,
    /// Siril (`r_*.fit`).
    Siril,
    /// Frames another program aligned (MaxIm, DSS, …), named by the user
    /// with `--registered`.
    Aligned,
}

/// One field × filter: registered frames that share a geometry.
#[derive(Debug, Clone)]
pub struct Stack {
    pub object: String,
    pub filter: String,
    pub frames: Vec<combine::Input>,
    /// Relative SNR² weight of each frame when known (crux's report).
    pub weights: Vec<f64>,
    pub source: Source,
}

impl Stack {
    /// Pairs this stack can give: halves, and quarters when allowed.
    pub fn pair_kinds(&self, quarters: bool) -> Vec<(&'static str, usize)> {
        let n = self.frames.len();
        let mut k = Vec::new();
        if n >= 2 * MIN_PART {
            k.push(("half", 2));
        }
        if quarters && n >= 4 * MIN_PART {
            k.push(("quarter", 4));
        }
        k
    }

    pub fn label(&self) -> String {
        format!("{} · {}", self.object, if self.filter.is_empty() { "no filter" } else { &self.filter })
    }
}

/// Registered frames under `dir`: from crux's integration report (for a
/// data folder, its workspace folder; or a workspace folder itself), else
/// WBPP's or Siril's registered files, grouped by object, filter and size.
/// With `any_aligned`, every light frame directly in `dir` counts as
/// registered (aligned by another program); stacks, masters and
/// calibration frames are skipped. Also returns notes on what was skipped.
pub fn find_stacks(dir: &Path, workspace: &Path, any_aligned: bool) -> Result<(Vec<Stack>, Vec<String>)> {
    let mut notes = Vec::new();
    for out in [dir.to_path_buf(), crate::process::job::out_dir(dir, workspace)] {
        if any_aligned || !out.join("crux-integration.json").is_file() {
            continue;
        }
        let mut stacks = Vec::new();
        for r in crate::process::job::load_integrations(&out).into_values() {
            let present: Vec<&combine::FrameReport> = r.frames.iter().filter(|f| f.path.is_file()).collect();
            let frames: Vec<combine::Input> = present.iter().map(|f| combine::Input { path: f.path.clone(), name: f.name.clone(), exptime: f.exptime, fwhm: f.fwhm, night: f.night, grade: f.grade }).collect();
            let weights: Vec<f64> = present.iter().map(|f| f.weight).collect();
            if frames.len() < r.frames.len() {
                notes.push(format!("{} · {}: {} of {} registered frames are missing", r.object, r.filter, r.frames.len() - frames.len(), r.frames.len()));
            }
            stacks.push(Stack { object: r.object, filter: r.filter, frames, weights, source: Source::Crux });
        }
        stacks.sort_by(|a, b| (&a.object, &a.filter).cmp(&(&b.object, &b.filter)));
        return Ok((stacks, notes));
    }
    // Other programs' registered frames.
    let mut files = Vec::new();
    let depth = if any_aligned { 1 } else { usize::MAX };
    for e in walkdir::WalkDir::new(dir).max_depth(depth).into_iter().filter_entry(|e| !e.file_name().to_string_lossy().starts_with('.')).flatten() {
        if !e.file_type().is_file() {
            continue;
        }
        let name = e.file_name().to_string_lossy().to_ascii_lowercase();
        let (stem, ext) = match name.rsplit_once('.') {
            Some(x) => x,
            None => continue,
        };
        let source = match ext {
            "xisf" | "fit" | "fits" | "fts" if any_aligned => {
                if ["master", "median", "mean", "sum", "stack", "deconv", "color", "colour"].iter().any(|k| stem.contains(k)) {
                    continue;
                }
                Source::Aligned
            }
            "xisf" if stem.ends_with("_r") => Source::Wbpp,
            "fit" | "fits" | "fts" if stem.starts_with("r_") => Source::Siril,
            "fit" | "fits" | "fts" if stem.ends_with("_r") => Source::Wbpp,
            _ => continue,
        };
        files.push((e.path().to_path_buf(), source));
    }
    let metas: Vec<(PathBuf, Source, Result<crate::FrameMeta>)> = files.into_par_iter().map(|(p, s)| {
        let m = crate::meta::read_meta(&p);
        (p, s, m)
    }).collect();
    let mut groups: std::collections::BTreeMap<(String, String, u32, u32), (Source, Vec<crate::FrameMeta>)> = Default::default();
    let mut colour = 0;
    for (p, s, m) in metas {
        let m = match m {
            Ok(m) => m,
            Err(e) => {
                notes.push(format!("can't read {}: {e:#}", p.display()));
                continue;
            }
        };
        // Colour images and raw one-shot-colour frames (still a Bayer
        // mosaic). crux's own registered channels of OSC lights are mono
        // and carry no Bayer keywords, so they pass.
        if m.planes > 1 || m.bayer().is_some() {
            colour += 1;
            continue;
        }
        if s == Source::Aligned {
            let t = m.imagetyp.as_deref().unwrap_or("").to_ascii_lowercase();
            if m.combined || ["dark", "flat", "bias", "offset"].iter().any(|k| t.contains(k)) {
                continue;
            }
        }
        let object = m.object.as_deref().map(crate::targets::normalise).unwrap_or_default();
        let filter = m.filter.as_deref().map(crate::targets::filter_name).unwrap_or_default();
        groups.entry((object, filter, m.width, m.height)).or_insert_with(|| (s, Vec::new())).1.push(m);
    }
    if colour > 0 {
        notes.push(format!("{colour} colour frames left out: colour (one-shot colour) frames aren't supported yet"));
    }
    let mut stacks = Vec::new();
    for ((object, filter, _, _), (source, metas)) in groups {
        // FWHM, to deal the frames into parts with the same seeing.
        let mut cfg = Config::default();
        cfg.analysis.trails = false;
        let frames: Vec<combine::Input> = metas
            .par_iter()
            .map(|m| {
                let fwhm = crate::read_image(&m.path).ok().and_then(|img| crate::analysis::analyze_light(&img, &cfg).fwhm);
                combine::Input {
                    path: m.path.clone(),
                    name: m.path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(),
                    exptime: m.exptime.unwrap_or(1.0),
                    fwhm,
                    night: None,
                    grade: crate::grade::Grade::Good,
                }
            })
            .collect();
        stacks.push(Stack { object, filter, frames, weights: Vec::new(), source });
    }
    Ok((stacks, notes))
}

/// Deal frames (indices) into `parts` equal parts whose stacks will be
/// about equally noisy. With SNR weights (`weights`, one per frame, from
/// crux's integration report), the best frames go first, each to the part
/// with the smallest total weight that still has room. Without, frames are
/// sorted by FWHM and dealt in a snake (A B B A …), so every part gets the
/// same spread of seeing. Leftovers (the weakest or softest frames) are
/// left out.
pub fn split(frames: &[combine::Input], weights: &[f64], parts: usize) -> Vec<Vec<usize>> {
    let n = frames.len();
    let room = n / parts.max(1);
    let mut out = vec![Vec::new(); parts];
    let mut idx: Vec<usize> = (0..n).collect();
    if weights.len() == n && weights.iter().all(|w| w.is_finite() && *w > 0.0) {
        idx.sort_by(|&a, &b| weights[b].total_cmp(&weights[a]).then(a.cmp(&b)));
        let mut sum = vec![0f64; parts];
        for &i in idx.iter().take(room * parts) {
            let p = (0..parts).filter(|&p| out[p].len() < room).min_by(|&a, &b| sum[a].total_cmp(&sum[b])).unwrap_or(0);
            out[p].push(i);
            sum[p] += weights[i];
        }
        // Then swap frames between parts while that narrows a gap.
        for _ in 0..200 {
            let mut best: Option<(usize, usize, usize, usize, f64)> = None;
            for p in 0..parts {
                for q in p + 1..parts {
                    let gap = (sum[p] - sum[q]).abs();
                    for (a, &i) in out[p].iter().enumerate() {
                        for (b, &j) in out[q].iter().enumerate() {
                            let d = weights[i] - weights[j];
                            let gain = gap - (sum[p] - d - (sum[q] + d)).abs();
                            if gain > 1e-12 && best.is_none_or(|x| gain > x.4) {
                                best = Some((p, q, a, b, gain));
                            }
                        }
                    }
                }
            }
            let Some((p, q, a, b, _)) = best else { break };
            let (i, j) = (out[p][a], out[q][b]);
            out[p][a] = j;
            out[q][b] = i;
            sum[p] += weights[j] - weights[i];
            sum[q] += weights[i] - weights[j];
        }
        return out;
    }
    idx.sort_by(|&a, &b| frames[a].fwhm.unwrap_or(f32::INFINITY).total_cmp(&frames[b].fwhm.unwrap_or(f32::INFINITY)).then(a.cmp(&b)));
    for (k, &i) in idx.iter().take(room * parts).enumerate() {
        let (round, pos) = (k / parts, k % parts);
        out[if round % 2 == 0 { pos } else { parts - 1 - pos }].push(i);
    }
    out
}

// ---------------------------------------------------------------- manifest

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub format: String,
    pub version: u32,
    /// Month made, YYYY-MM.
    pub created: String,
    pub crux_version: String,
    pub patch: usize,
    /// Pixel values are as stored: multiply by a set's `adu_scale` for ADU.
    pub units: String,
    pub sets: Vec<SetInfo>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SetInfo {
    pub id: String,
    pub source: Option<Source>,
    pub object: String,
    pub filter: String,
    pub camera: Option<String>,
    pub telescope: Option<String>,
    /// Pixel size after binning, µm.
    pub pixel_um: Option<f64>,
    pub focal_mm: Option<f64>,
    pub binning: u32,
    /// Arcseconds per pixel (header focal length × pixel size).
    pub scale_arcsec: Option<f64>,
    pub gain: Option<f64>,
    pub offset: Option<f64>,
    /// e-/ADU.
    pub egain: Option<f64>,
    pub temp_c: Option<f64>,
    pub exposure_s: Option<f64>,
    /// YYYY-MM of the first frame.
    pub month: Option<String>,
    /// Field centre, degrees (from the header).
    pub ra: Option<f64>,
    pub dec: Option<f64>,
    pub width: usize,
    pub height: usize,
    pub adu_scale: f32,
    pub frames: usize,
    pub pairs: Vec<PairInfo>,
    pub psf: Option<PsfInfo>,
    /// Debayered from a one-shot-colour sensor with this pattern (bilinear,
    /// so the channel's noise is correlated between neighbours).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cfa: Option<String>,
    /// For colour frames: the filter in front of the sensor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub front_filter: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartInfo {
    pub frames: usize,
    pub exposure_s: f64,
    /// Weighted FWHM of the frames, pixels.
    pub fwhm: Option<f64>,
    /// Noise σ of the stack, normalised units (after matching, for `b`).
    pub noise: f64,
    pub rejection: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairInfo {
    /// "half" or "quarter".
    pub kind: String,
    pub file_a: String,
    pub file_b: String,
    pub a: PartInfo,
    pub b: PartInfo,
    pub matched: combine::Matched,
    /// Median of the field, normalised units (sky level for most fields).
    pub sky: f64,
    /// Top-left corner of each patch in the stacks.
    pub patches: Vec<(usize, usize)>,
    /// Each patch's median above `sky`, in noise σ of `a`.
    pub signal: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PsfInfo {
    pub file: String,
    /// Cutout side, pixels (odd; the star is at the centre pixel ± 0.5).
    pub size: usize,
    pub stars: Vec<PsfStar>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PsfStar {
    pub x: f32,
    pub y: f32,
    pub fwhm: f32,
    pub ecc: f32,
    pub pa: f32,
    /// Normalised units, background subtracted.
    pub flux: f32,
    pub peak: f32,
    /// Local background (annulus median), normalised units.
    pub background: f32,
}

// ------------------------------------------------------------------ collect

/// Progress lines for the caller to show.
pub type Progress<'a> = &'a (dyn Fn(&str) + Sync);

/// Patch pairs each pair gets within `max_mb` (at least 4, at most 96).
pub fn patches_per_pair(stacks: &[Stack], o: &Options) -> usize {
    let pairs: usize = stacks.iter().map(|s| s.pair_kinds(o.quarters).iter().map(|(_, p)| p / 2).sum::<usize>()).sum();
    if pairs == 0 {
        return 0;
    }
    let bytes = o.max_mb * 0.95 * 1e6;
    let per = bytes / (pairs as f64 * 2.0 * (o.patch * o.patch * 4) as f64);
    (per.floor() as usize).clamp(4, 96)
}

/// Make the bundle in `out` (created; must not hold a bundle already).
pub fn collect(stacks: &[Stack], out: &Path, o: &Options, notes: Vec<String>, progress: Progress, cancel: &AtomicBool) -> Result<Manifest> {
    std::fs::create_dir_all(out).with_context(|| format!("create {}", out.display()))?;
    let work = out.join(".stacks");
    let per_pair = patches_per_pair(stacks, o);
    let mut sets = Vec::new();
    let mut thumbs = Vec::new();
    let mut notes = notes;
    let mut rng = Rng(o.seed);
    for (k, st) in stacks.iter().enumerate() {
        if cancel.load(std::sync::atomic::Ordering::Relaxed) {
            anyhow::bail!("cancelled");
        }
        let kinds = st.pair_kinds(o.quarters);
        if kinds.is_empty() {
            notes.push(format!("{}: {} frames, too few for a pair (needs {})", st.label(), st.frames.len(), 2 * MIN_PART));
            continue;
        }
        // RC Astro's licence forbids using its outputs to develop models
        // like crux's: never collect anything it has processed.
        if let Some(f) = st.frames.iter().find(|f| touched_by_rc_astro(&f.path)) {
            let msg = format!("{}: left out, {} was processed by RC Astro software (BlurXTerminator, NoiseXTerminator or StarXTerminator), whose licence forbids using its output to train other tools", st.label(), f.name);
            progress(&msg);
            notes.push(msg);
            continue;
        }
        let id = format!("s{:02}", k + 1);
        progress(&format!("{} ({id}): {} frames", st.label(), st.frames.len()));
        match collect_set(st, &id, out, &work, o, per_pair, &mut rng, &mut thumbs, progress, cancel) {
            Ok(s) => sets.push(s),
            Err(e) => {
                let msg = format!("{}: {e:#}", st.label());
                progress(&format!("  failed: {msg}"));
                notes.push(msg);
            }
        }
    }
    if !o.keep_stacks {
        let _ = std::fs::remove_dir_all(&work);
    }
    let m = Manifest {
        format: FORMAT.into(),
        version: BUNDLE_VERSION,
        created: chrono::Local::now().format("%Y-%m").to_string(),
        crux_version: env!("CARGO_PKG_VERSION").into(),
        patch: o.patch,
        units: "normalised: ADU / adu_scale".into(),
        sets,
        notes,
    };
    std::fs::write(out.join("bundle.html"), html(&m, out, &thumbs)).context("write bundle.html")?;
    crate::paths::write_atomic(&out.join("manifest.json"), &serde_json::to_vec_pretty(&m)?)?;
    Ok(m)
}

#[allow(clippy::too_many_arguments)]
fn collect_set(st: &Stack, id: &str, out: &Path, work: &Path, o: &Options, per_pair: usize, rng: &mut Rng, thumbs: &mut Vec<Thumb>, progress: Progress, cancel: &AtomicBool) -> Result<SetInfo> {
    // The sharpest frame: metadata, and the stars for photometry. Every
    // registered frame shares its geometry.
    let best = (0..st.frames.len()).min_by(|&a, &b| st.frames[a].fwhm.unwrap_or(f32::INFINITY).total_cmp(&st.frames[b].fwhm.unwrap_or(f32::INFINITY))).unwrap_or(0);
    let meta = crate::meta::read_meta(&st.frames[best].path)?;
    let mut cfg = Config::default();
    cfg.analysis.trails = false;
    let rimg = crate::read_image(&st.frames[best].path)?;
    let rm = crate::analysis::analyze_light(&rimg, &cfg);
    let stars: Vec<(f32, f32)> = rm.stars.iter().filter(|s| s.is_clean()).map(|s| (s.x, s.y)).collect();
    let fwhm = rm.fwhm.unwrap_or(3.0);
    drop(rimg);
    let mut info = set_info(st, id, &meta);
    let settings = combine::Settings::default();
    for (kind, parts) in st.pair_kinds(o.quarters) {
        let split = split(&st.frames, &st.weights, parts);
        for p in 0..parts / 2 {
            if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                anyhow::bail!("cancelled");
            }
            let tag = if parts == 2 { kind.to_string() } else { format!("{kind}{}", p + 1) };
            let mut stacked = Vec::new();
            for (side, part) in [("a", &split[2 * p]), ("b", &split[2 * p + 1])] {
                let inputs: Vec<combine::Input> = part.iter().map(|&i| st.frames[i].clone()).collect();
                let ri = (0..inputs.len()).min_by(|&a, &b| inputs[a].fwhm.unwrap_or(f32::INFINITY).total_cmp(&inputs[b].fwhm.unwrap_or(f32::INFINITY))).unwrap_or(0);
                let path = work.join(format!("{id}_{tag}_{side}.xisf"));
                let r = combine::integrate(&inputs, ri, &stars, &settings, &st.object, &st.filter, &path, &Header::default(), &|_| {}, cancel)?;
                stacked.push((path, r));
            }
            let (pa, ra) = &stacked[0];
            let (pb, rb) = &stacked[1];
            let a = crate::read_image(pa)?;
            let mut b = crate::read_image(pb)?;
            anyhow::ensure!((a.width, a.height) == (b.width, b.height), "the stacks differ in size");
            let matched = combine::match_to(&mut b, &a, &stars, fwhm, settings.local_cell);
            let (na, nb) = (combine::noise_of(&a) as f64, combine::noise_of(&b) as f64);
            let sky = {
                let mut v: Vec<f32> = a.data.iter().step_by(11).copied().filter(|v| *v > 0.0).collect();
                crate::stats::median_mut(&mut v) as f64
            };
            let picks = pick_patches(&a, &b, o.patch, per_pair, sky as f32, na as f32, rng);
            let (fa, fb) = (format!("{id}_{tag}_a.fits"), format!("{id}_{tag}_b.fits"));
            write_patches(&out.join(&fa), &a, &picks, o.patch, kind, na)?;
            write_patches(&out.join(&fb), &b, &picks, o.patch, kind, nb)?;
            if let Some(best) = picks.iter().max_by(|x, y| x.1.total_cmp(&y.1)) {
                if let Some(png) = thumb(&a, &b, best.0, o.patch) {
                    thumbs.push(Thumb { caption: format!("{} · {}: A | B", st.label(), tag), png });
                }
            }
            progress(&format!(
                "  {tag}: {} + {} frames, noise {:.2} / {:.2} ADU, scale {:.3}, background {:+.2} ADU, {} patches",
                ra.frames.len(),
                rb.frames.len(),
                na * a.adu_scale as f64,
                nb * a.adu_scale as f64,
                matched.scale,
                matched.offset * a.adu_scale as f64,
                picks.len()
            ));
            let part = |r: &combine::Report, noise: f64| PartInfo { frames: r.frames.len(), exposure_s: r.total_exposure, fwhm: r.fwhm, noise, rejection: r.rejection.clone() };
            if info.psf.is_none() {
                info.psf = psf_cutouts(&a, &cfg, o.psf_stars, &out.join(format!("{id}_psf.fits")))?;
            }
            info.adu_scale = a.adu_scale;
            info.width = a.width;
            info.height = a.height;
            info.pairs.push(PairInfo {
                kind: kind.into(),
                file_a: fa,
                file_b: fb,
                a: part(ra, na),
                b: part(rb, nb),
                matched,
                sky,
                signal: picks.iter().map(|p| p.1).collect(),
                patches: picks.iter().map(|p| p.0).collect(),
            });
            if !o.keep_stacks {
                for (p, r) in &stacked {
                    let _ = std::fs::remove_file(p);
                    let _ = std::fs::remove_file(&r.rejection_low);
                    let _ = std::fs::remove_file(&r.rejection_high);
                }
            }
        }
    }
    Ok(info)
}

/// Camera and optics details only (nothing that says where or who).
fn set_info(st: &Stack, id: &str, m: &crate::FrameMeta) -> SetInfo {
    let clean = |s: &Option<String>| s.as_ref().map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
    let bin = m.binning().max(1);
    let scale = match (m.focallen, m.xpixsz) {
        (Some(f), Some(p)) if f > 0.0 && p > 0.0 => Some(206.264_806 * p * bin as f64 / f),
        _ => None,
    };
    let month = st.frames.iter().filter_map(|f| f.night).min().map(|d| d.format("%Y-%m").to_string()).or_else(|| m.date_loc.or(m.date_obs).map(|d| d.format("%Y-%m").to_string()));
    SetInfo {
        id: id.into(),
        source: Some(st.source),
        object: st.object.clone(),
        filter: st.filter.clone(),
        camera: clean(&m.instrume),
        telescope: clean(&m.telescop),
        pixel_um: m.xpixsz.map(|p| p * bin as f64).filter(|p| *p > 0.0),
        focal_mm: m.focallen.filter(|f| *f > 0.0),
        binning: bin,
        scale_arcsec: scale,
        gain: m.gain,
        offset: m.offset,
        egain: m.egain,
        temp_c: m.ccd_temp,
        exposure_s: m.exptime,
        month,
        ra: m.ra,
        dec: m.dec,
        frames: st.frames.len(),
        cfa: m.header.str("CRUX-CFA"),
        front_filter: m.header.str("CRUX-OSF"),
        ..Default::default()
    }
}

/// Patch corners and signal (median above `sky` in σ). Candidates sit on a
/// half-patch grid where both stacks have data everywhere; half the picks
/// favour signal (weight 1 + signal, capped at 20σ), the rest are uniform.
fn pick_patches(a: &Image, b: &Image, patch: usize, want: usize, sky: f32, sigma: f32, rng: &mut Rng) -> Vec<((usize, usize), f32)> {
    let (w, h) = (a.width, a.height);
    if w < patch || h < patch || want == 0 {
        return Vec::new();
    }
    let step = (patch / 2).max(1);
    let corners: Vec<(usize, usize)> = (0..=(h - patch)).step_by(step).flat_map(|y| (0..=(w - patch)).step_by(step).map(move |x| (x, y))).collect();
    let mut cands: Vec<((usize, usize), f32)> = corners
        .par_iter()
        .filter_map(|&(x, y)| {
            let mut s = Vec::with_capacity(patch * patch / 16);
            for yy in y..y + patch {
                let (ra, rb) = (&a.data[yy * w + x..yy * w + x + patch], &b.data[yy * w + x..yy * w + x + patch]);
                if ra.iter().chain(rb).any(|v| *v <= 0.0) {
                    return None;
                }
                if yy % 4 == 0 {
                    s.extend(ra.iter().step_by(4));
                }
            }
            let m = crate::stats::median_mut(&mut s);
            Some(((x, y), (m - sky) / sigma.max(1e-12)))
        })
        .collect();
    let mut picks = Vec::new();
    let favoured = want / 2;
    while picks.len() < want && !cands.is_empty() {
        let k = if picks.len() < favoured {
            let total: f64 = cands.iter().map(|c| 1.0 + c.1.clamp(0.0, 20.0) as f64).sum();
            let mut t = rng.unit() * total;
            let mut k = cands.len() - 1;
            for (i, c) in cands.iter().enumerate() {
                t -= 1.0 + c.1.clamp(0.0, 20.0) as f64;
                if t <= 0.0 {
                    k = i;
                    break;
                }
            }
            k
        } else {
            (rng.unit() * cands.len() as f64) as usize % cands.len()
        };
        let c = cands.swap_remove(k);
        // No two patches overlapping by more than half.
        cands.retain(|o| o.0 .0.abs_diff(c.0 .0) >= step || o.0 .1.abs_diff(c.0 .1) >= step);
        picks.push(c);
    }
    picks
}

fn write_patches(path: &Path, img: &Image, picks: &[((usize, usize), f32)], patch: usize, kind: &str, noise: f64) -> Result<()> {
    if picks.is_empty() {
        anyhow::bail!("no patch fits inside both stacks");
    }
    let planes: Vec<Vec<f32>> = picks
        .iter()
        .map(|&((x, y), _)| (y..y + patch).flat_map(|yy| img.data[yy * img.width + x..yy * img.width + x + patch].iter().copied()).collect())
        .collect();
    let refs: Vec<&[f32]> = planes.iter().map(|p| p.as_slice()).collect();
    let mut h = Header::default();
    h.push("CRUXBNDL", BUNDLE_VERSION.to_string(), "crux training bundle version");
    h.push("PAIRKIND", kind, "half or quarter stack");
    h.push("NOISE", format!("{noise:.6e}"), "noise sigma, normalised units");
    h.push("ADUSCALE", format!("{}", img.adu_scale), "ADU = value x ADUSCALE");
    crate::fits::write_cube(path, patch, patch, &refs, &h)
}

/// Unsaturated stars spread over a 5 × 5 grid (brightest first in each
/// cell, taken in turns), cut out at a size that holds the wings.
fn psf_cutouts(img: &Image, cfg: &Config, want: usize, path: &Path) -> Result<Option<PsfInfo>> {
    let m = crate::analysis::analyze_light(img, cfg);
    let fwhm = m.fwhm.unwrap_or(3.0);
    let size = ((8.0 * fwhm).ceil() as usize | 1).clamp(31, 63);
    let r = size / 2;
    let (w, h) = (img.width, img.height);
    let mut cells: Vec<Vec<&crate::stars::Star>> = vec![Vec::new(); 25];
    for s in m.stars.iter().filter(|s| s.is_clean()) {
        let (cx, cy) = (s.x.round() as i64, s.y.round() as i64);
        if cx < r as i64 + 1 || cy < r as i64 + 1 || cx + r as i64 + 1 >= w as i64 || cy + r as i64 + 1 >= h as i64 {
            continue;
        }
        let gx = ((s.x / w as f32 * 5.0) as usize).min(4);
        let gy = ((s.y / h as f32 * 5.0) as usize).min(4);
        cells[gy * 5 + gx].push(s);
    }
    let mut chosen = Vec::new();
    let mut round = 0;
    while chosen.len() < want && cells.iter().any(|c| c.len() > round) {
        for c in &cells {
            if let Some(s) = c.get(round) {
                if chosen.len() < want {
                    chosen.push(*s);
                }
            }
        }
        round += 1;
    }
    let mut planes = Vec::new();
    let mut stars = Vec::new();
    for s in chosen {
        let (cx, cy) = (s.x.round() as usize, s.y.round() as usize);
        let cut: Vec<f32> = (cy - r..=cy + r).flat_map(|y| img.data[y * w + cx - r..=y * w + cx + r].iter().copied()).collect();
        if cut.iter().any(|v| *v <= 0.0) {
            continue;
        }
        let mut ring: Vec<f32> = Vec::new();
        for y in 0..size {
            for x in 0..size {
                let d = ((x as f32 - r as f32).powi(2) + (y as f32 - r as f32).powi(2)).sqrt();
                if d >= r as f32 - 3.0 && d <= r as f32 {
                    ring.push(cut[y * size + x]);
                }
            }
        }
        let background = crate::stats::median_mut(&mut ring);
        stars.push(PsfStar { x: s.x, y: s.y, fwhm: s.fwhm, ecc: s.ecc, pa: s.pa, flux: s.flux, peak: s.peak, background });
        planes.push(cut);
    }
    if planes.is_empty() {
        return Ok(None);
    }
    let refs: Vec<&[f32]> = planes.iter().map(|p| p.as_slice()).collect();
    let mut hd = Header::default();
    hd.push("CRUXBNDL", BUNDLE_VERSION.to_string(), "crux training bundle version");
    hd.push("ADUSCALE", format!("{}", img.adu_scale), "ADU = value x ADUSCALE");
    crate::fits::write_cube(path, size, size, &refs, &hd)?;
    Ok(Some(PsfInfo { file: path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(), size, stars }))
}

/// Whether a file's header (FITS cards, or the XISF XML header with its
/// processing history) mentions an RC Astro XTerminator.
pub fn touched_by_rc_astro(path: &Path) -> bool {
    use std::io::Read;
    let Ok(mut f) = std::fs::File::open(path) else { return false };
    let mut head = Vec::new();
    let mut sig = [0u8; 16];
    if f.read_exact(&mut sig).is_err() {
        return false;
    }
    if &sig[..8] == b"XISF0100" {
        let len = u32::from_le_bytes([sig[8], sig[9], sig[10], sig[11]]) as usize;
        head.resize(len.min(64 << 20), 0);
        if f.read_exact(&mut head).is_err() {
            return false;
        }
    } else {
        head.extend_from_slice(&sig);
        let mut block = [0u8; 2880];
        for _ in 0..200 {
            if f.read_exact(&mut block[..2880 - if head.len() == 16 { 16 } else { 0 }]).is_err() {
                break;
            }
            let n = if head.len() == 16 { 2864 } else { 2880 };
            head.extend_from_slice(&block[..n]);
            if head.chunks(80).any(|c| c.starts_with(b"END ") || c == b"END") {
                break;
            }
        }
    }
    let text = String::from_utf8_lossy(&head).to_ascii_lowercase();
    ["xterminator", "rc-astro", "rc astro"].iter().any(|k| text.contains(k))
}

/// splitmix64: reproducible picks without another dependency.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// [0, 1).
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// --------------------------------------------------------------------- html

/// What's in the bundle, for the contributor to check before sending it.
fn html(m: &Manifest, out: &Path, thumb_list: &[Thumb]) -> String {
    let esc = |s: &str| s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;");
    let opt = |v: Option<f64>, d: usize, unit: &str| v.map(|x| format!("{x:.d$}{unit}")).unwrap_or_else(|| "—".into());
    let bytes: u64 = std::fs::read_dir(out).map(|d| d.flatten().filter(|e| e.path().is_file()).filter_map(|e| e.metadata().ok()).map(|m| m.len()).sum()).unwrap_or(0);
    let patches: usize = m.sets.iter().flat_map(|s| &s.pairs).map(|p| p.patches.len()).sum();
    let stars: usize = m.sets.iter().filter_map(|s| s.psf.as_ref()).map(|p| p.stars.len()).sum();
    let mut cams: Vec<String> = m.sets.iter().filter_map(|s| s.camera.clone()).collect();
    cams.sort();
    cams.dedup();
    let mut rows = String::new();
    let mut thumbs = String::new();
    for s in &m.sets {
        let pairs: Vec<String> = s.pairs.iter().map(|p| format!("{} ({}+{} frames)", p.kind, p.a.frames, p.b.frames)).collect();
        rows.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>\n",
            esc(&s.object),
            esc(if s.filter.is_empty() { "—" } else { &s.filter }),
            esc(s.camera.as_deref().unwrap_or("—")),
            opt(s.scale_arcsec, 2, "″"),
            s.frames,
            esc(&pairs.join(", ")),
            s.pairs.iter().map(|p| p.patches.len()).sum::<usize>(),
            s.psf.as_ref().map(|p| p.stars.len()).unwrap_or(0),
            esc(s.month.as_deref().unwrap_or("—")),
        ));
    }
    for t in thumb_list {
        thumbs.push_str(&format!("<figure><img alt=\"{c}\" src=\"data:image/png;base64,{png}\"><figcaption>{c}</figcaption></figure>\n", c = esc(&t.caption), png = t.png));
    }
    let notes: String = m.notes.iter().map(|n| format!("<li>{}</li>", esc(n))).collect();
    format!(
        r#"<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>crux training bundle</title>
<style>
:root {{ --bg:#fff; --fg:#1b1d21; --muted:#5d636e; --line:#dde1e7; --card:#f5f6f8; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#15171a; --fg:#e6e8eb; --muted:#9aa1ab; --line:#2c3036; --card:#1d2024; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; margin:0 auto; max-width:1100px; padding:24px 16px; }}
h1 {{ font-size:24px; margin:0 0 4px; }} h2 {{ font-size:18px; margin:28px 0 8px; }}
.muted {{ color:var(--muted); }} table {{ border-collapse:collapse; width:100%; font-size:14px; }}
th, td {{ text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); vertical-align:top; }}
.wrap {{ overflow-x:auto; }} .grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(260px, 1fr)); gap:12px; }}
figure {{ margin:0; background:var(--card); padding:8px; border-radius:6px; }} img {{ width:100%; image-rendering:pixelated; display:block; }}
figcaption {{ font-size:13px; color:var(--muted); margin-top:4px; }} ul {{ padding-left:20px; }}
</style></head><body>
<h1>crux training bundle</h1>
<p class="muted">Format {fmt} version {ver} · made {created} · crux {cv}</p>
<p>{nsets} stacks, {patches} patch pairs, {stars} star cutouts, {mb:.1} MB. Cameras: {cams}.</p>
<h2>What's in it</h2>
<ul>
<li>Pairs of image patches, {patch} × {patch} px, cut at the same places from two stacks of the same field made from different frames.</li>
<li>Star cutouts from one stack per field, with their size, shape and position.</li>
<li>Camera and optics details per field: camera, telescope name, pixel size, focal length, binning, gain, offset, sensor temperature, exposure, object, field centre, and the month.</li>
</ul>
<h2>What's not in it</h2>
<ul>
<li>File names, folder names or paths.</li>
<li>Site latitude and longitude, observer name, or the day and time of any frame.</li>
<li>Any FITS header. Only the details above are copied, into <code>manifest.json</code>.</li>
</ul>
<h2>Stacks</h2>
<div class="wrap"><table>
<tr><th>Object</th><th>Filter</th><th>Camera</th><th>Scale</th><th>Frames</th><th>Pairs</th><th>Patches</th><th>Stars</th><th>Month</th></tr>
{rows}</table></div>
{notes_block}
<h2>Sample patches</h2>
<p class="muted">One patch per pair, both stacks with the same stretch.</p>
<div class="grid">{thumbs}</div>
</body></html>
"#,
        fmt = FORMAT,
        ver = m.version,
        created = esc(&m.created),
        cv = esc(&m.crux_version),
        nsets = m.sets.len(),
        mb = bytes as f64 / 1e6,
        cams = if cams.is_empty() { "not recorded".into() } else { esc(&cams.join(", ")) },
        patch = m.patch,
        notes_block = if notes.is_empty() { String::new() } else { format!("<h2>Notes</h2><ul>{notes}</ul>") },
    )
}

/// A thumbnail for bundle.html: caption and base64 PNG.
pub struct Thumb {
    pub caption: String,
    pub png: String,
}

/// One patch, A | B with one auto stretch, as base64 PNG.
fn thumb(a: &Image, b: &Image, (x0, y0): (usize, usize), patch: usize) -> Option<String> {
    let cut = |img: &Image| Image::from_data(patch, patch, (y0..y0 + patch).flat_map(|y| img.data[y * img.width + x0..y * img.width + x0 + patch].iter().copied()).collect());
    let (a, b) = (cut(a), cut(b));
    let stf = crate::stretch::Stf::auto(&a);
    let gap = 6;
    let (w, h) = (patch * 2 + gap, patch);
    let mut px = vec![40u8; w * h];
    for y in 0..h {
        for x in 0..patch {
            px[y * w + x] = (stf.apply(a.data[y * patch + x]) * 255.0).round().clamp(0.0, 255.0) as u8;
            px[y * w + patch + gap + x] = (stf.apply(b.data[y * patch + x]) * 255.0).round().clamp(0.0, 255.0) as u8;
        }
    }
    let mut buf = Vec::new();
    {
        let mut e = png::Encoder::new(&mut buf, w as u32, h as u32);
        e.set_color(png::ColorType::Grayscale);
        e.set_depth(png::BitDepth::Eight);
        let mut wr = e.write_header().ok()?;
        wr.write_image_data(&px).ok()?;
    }
    Some(base64(&buf))
}

fn base64(b: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(b.len().div_ceil(3) * 4);
    for c in b.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        s.push(T[(n >> 18) as usize & 63] as char);
        s.push(T[(n >> 12) as usize & 63] as char);
        s.push(if c.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        s.push(if c.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grade::Grade;

    fn input(fwhm: f32) -> combine::Input {
        combine::Input { path: PathBuf::new(), name: String::new(), exptime: 300.0, fwhm: Some(fwhm), night: None, grade: Grade::Good }
    }

    #[test]
    fn split_balances_seeing_and_size() {
        let frames: Vec<_> = [3.0, 5.0, 3.5, 4.0, 6.0, 3.2, 4.5, 6.5].iter().map(|&f| input(f)).collect();
        let parts = split(&frames, &[], 2);
        assert_eq!(parts[0].len(), 4);
        assert_eq!(parts[1].len(), 4);
        let mean = |p: &Vec<usize>| p.iter().map(|&i| frames[i].fwhm.unwrap() as f64).sum::<f64>() / p.len() as f64;
        assert!((mean(&parts[0]) - mean(&parts[1])).abs() < 0.6, "{:?}", parts);
        let q = split(&frames[..7], &[], 4);
        assert!(q.iter().all(|p| p.len() == 1));
        // The softest frames are the ones left over.
        assert!(!q.iter().flatten().any(|&i| i == 4));
    }

    #[test]
    fn split_by_weight_balances_noise() {
        let frames: Vec<_> = (0..9).map(|_| input(4.0)).collect();
        let w = [1.0, 0.95, 0.9, 0.6, 0.55, 0.5, 0.3, 0.25, 0.1];
        let parts = split(&frames, &w, 2);
        assert_eq!(parts[0].len(), 4);
        assert_eq!(parts[1].len(), 4);
        let sum = |p: &Vec<usize>| p.iter().map(|&i| w[i]).sum::<f64>();
        // Stack noise goes as 1/√(Σ weight): within 4%.
        let (a, b) = (sum(&parts[0]), sum(&parts[1]));
        assert!((a.max(b) / a.min(b)).sqrt() < 1.04, "{parts:?}");
        // The weakest frame is the one left over.
        assert!(!parts.iter().flatten().any(|&i| i == 8));
    }

    #[test]
    fn spots_rc_astro_history() {
        let dir = std::env::temp_dir().join(format!("crux-collect-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let img = Image::from_data(8, 8, vec![0.1; 64]);
        let clean = dir.join("clean.fits");
        crate::fits::write_image(&clean, &img, -32, &Header::default()).unwrap();
        let mut h = Header::default();
        h.push("HISTORY", "", "NoiseXTerminator.denoise = 0.90");
        let dirty = dir.join("dirty.fits");
        crate::fits::write_image(&dirty, &img, -32, &h).unwrap();
        let xisf = dir.join("dirty.xisf");
        let mut hx = Header::default();
        hx.push("HISTORY", "", "BlurXTerminator.sharpen_stars = 0.5");
        crate::xisf::write_image(&xisf, &img, &hx, None).unwrap();
        assert!(!touched_by_rc_astro(&clean));
        assert!(touched_by_rc_astro(&dirty));
        assert!(touched_by_rc_astro(&xisf));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn base64_matches_known_values() {
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn patches_stay_inside_data_and_apart() {
        let (w, h) = (1200usize, 900usize);
        let mut data = vec![0.1f32; w * h];
        // A dead border on the left, like a registered frame's edge.
        for y in 0..h {
            for x in 0..100 {
                data[y * w + x] = 0.0;
            }
        }
        let a = Image::from_data(w, h, data.clone());
        let b = Image::from_data(w, h, data);
        let mut rng = Rng(1);
        let p = pick_patches(&a, &b, 256, 20, 0.1, 0.01, &mut rng);
        assert!(!p.is_empty());
        for &((x, y), _) in &p {
            assert!(x >= 100 && x + 256 <= w && y + 256 <= h);
        }
        for (i, a) in p.iter().enumerate() {
            for b in &p[i + 1..] {
                assert!(a.0 .0.abs_diff(b.0 .0) >= 128 || a.0 .1.abs_diff(b.0 .1) >= 128);
            }
        }
    }
}
