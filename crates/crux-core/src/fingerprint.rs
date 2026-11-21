//! Darks, bias and dark flats matched by fingerprint across the library.
//!
//! These frames have no stars, so they can't be plate solved, and WBPP
//! masters carry few keywords (no GAIN, OFFSET or CCD-TEMP). What they do
//! carry is the sensor's hot pixels, which also show in every undarked light
//! (thousands of isolated one-pixel spikes per sub). Hot pixels identify a
//! sensor, their brightness follows temperature and exposure, and the set
//! drifts slowly as new ones appear. So a dark is scored against a sample of
//! the lights it would calibrate:
//!
//! - **coverage:** the share of the lights' persistent hot pixels that the
//!   dark also has (an old dark misses the ones that appeared since);
//! - **brightness:** shared hot pixels in the lights against the dark,
//!   scaled by exposure (about 1 when the temperature matches).
//!
//! Bias and dark flats have few hot pixels; they're ranked on headers, level
//! (a dark's level includes the bias, so the right bias sits just below it)
//! and date.

use crate::cache::Cache;
use crate::classify::{FrameType, Role};
use crate::grade::{FrameReport, Grade};
use crate::image::Image;
use crate::pipeline::Triage;
use crate::scan::Frame;
use chrono::NaiveDate;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

/// Bump when spike detection changes; cached fingerprints are then remade.
pub const FP_VERSION: u32 = 1;
/// Spikes kept per file, strongest first.
const MAX_SPIKES: usize = 200_000;
/// Detection threshold, in pixel noise.
const SPIKE_SIGMA: f32 = 6.0;
/// A hot pixel "shows" in the lights above this many light-noise sigmas.
const VISIBLE_SIGMA: f32 = 10.0;
/// Lights sampled per family.
const LIGHT_SAMPLE: usize = 8;
/// Raw darks or bias sampled per set.
const SET_SAMPLE: usize = 5;
/// Dark current doubles about every this many °C (typical CMOS; assumed).
const DOUBLING_C: f64 = 6.0;

/// Isolated one-pixel spikes in a frame, sorted by pixel index.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Spikes {
    pub w: u32,
    pub h: u32,
    /// Pixel noise, ADU.
    pub sigma_adu: f32,
    /// Median level, ADU.
    pub level_adu: f32,
    pub idx: Vec<u32>,
    /// Height above the 8 neighbours' median, ADU.
    pub excess: Vec<f32>,
}

impl Spikes {
    pub fn get(&self, i: u32) -> Option<f32> {
        self.idx.binary_search(&i).ok().map(|k| self.excess[k])
    }

    pub fn len(&self) -> usize {
        self.idx.len()
    }

    pub fn is_empty(&self) -> bool {
        self.idx.is_empty()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(28 + self.idx.len() * 8);
        b.extend_from_slice(b"CRUXSPK1");
        for v in [self.w, self.h, self.idx.len() as u32] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.extend_from_slice(&self.sigma_adu.to_le_bytes());
        b.extend_from_slice(&self.level_adu.to_le_bytes());
        for i in &self.idx {
            b.extend_from_slice(&i.to_le_bytes());
        }
        for e in &self.excess {
            b.extend_from_slice(&e.to_le_bytes());
        }
        b
    }

    pub fn from_bytes(b: &[u8]) -> Option<Spikes> {
        if b.len() < 28 || &b[..8] != b"CRUXSPK1" {
            return None;
        }
        let u = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let f = |o: usize| f32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let (w, h, n) = (u(8), u(12), u(16) as usize);
        if b.len() != 28 + n * 8 {
            return None;
        }
        let idx = (0..n).map(|k| u(28 + 4 * k)).collect();
        let excess = (0..n).map(|k| f(28 + 4 * n + 4 * k)).collect();
        Some(Spikes { w, h, sigma_adu: f(20), level_adu: f(24), idx, excess })
    }
}

/// Pixel noise and median level from a sparse grid, normalised units.
fn noise_and_level(img: &Image) -> (f32, f32) {
    let (w, h) = (img.width, img.height);
    let mut d = Vec::new();
    let mut lv = Vec::new();
    let step = ((w * h) / 400_000).max(1);
    let mut k = 0usize;
    for y in (1..h.saturating_sub(1)).step_by(3) {
        for x in (1..w.saturating_sub(1)).step_by(3) {
            k += 1;
            if k % step != 0 {
                continue;
            }
            let p = img.get(x, y);
            let m4 = (img.get(x - 1, y) + img.get(x + 1, y) + img.get(x, y - 1) + img.get(x, y + 1)) / 4.0;
            d.push(p - m4);
            lv.push(p);
        }
    }
    if d.is_empty() {
        return (0.0, 0.0);
    }
    let med = |v: &mut Vec<f32>| {
        let m = v.len() / 2;
        *v.select_nth_unstable_by(m, |a, b| a.total_cmp(b)).1
    };
    let level = med(&mut lv);
    let dm = med(&mut d.clone());
    let mut dev: Vec<f32> = d.iter().map(|x| (x - dm).abs()).collect();
    let mad = med(&mut dev);
    // p − mean of 4 neighbours has variance σ²(1 + 1/4).
    (mad * 1.4826 / 1.25f32.sqrt(), level)
}

/// Find isolated one-pixel spikes: at least `SPIKE_SIGMA` noise above the
/// median of the 8 neighbours, with no neighbour above 40% of the spike
/// (star cores are wider than one pixel).
pub fn spikes(img: &Image) -> Spikes {
    let (w, h) = (img.width, img.height);
    let (sigma, level) = noise_and_level(img);
    let mut out = Spikes { w: w as u32, h: h as u32, sigma_adu: sigma * img.adu_scale, level_adu: level * img.adu_scale, ..Default::default() };
    if w < 3 || h < 3 || sigma <= 0.0 {
        return out;
    }
    let t = SPIKE_SIGMA * sigma;
    if w < 5 || h < 5 {
        return out;
    }
    let mut found: Vec<(u32, f32)> = (2..h - 2)
        .into_par_iter()
        .flat_map_iter(|y| {
            let (a, r, b) = (img.row(y - 1), img.row(y), img.row(y + 1));
            let (a2, b2) = (img.row(y - 2), img.row(y + 2));
            let mut v = Vec::new();
            for x in 2..w - 2 {
                let p = r[x];
                let n = [a[x - 1], a[x], a[x + 1], r[x - 1], r[x + 1], b[x - 1], b[x], b[x + 1]];
                let mean: f32 = n.iter().sum::<f32>() / 8.0;
                if p - mean <= t {
                    continue;
                }
                let mut s = n;
                s.sort_unstable_by(|a, b| a.total_cmp(b));
                let med = 0.5 * (s[3] + s[4]);
                let e = p - med;
                if e <= t {
                    continue;
                }
                // Background from the ring two pixels out: a star's
                // neighbours stand well above it, a hot pixel's don't.
                let mut ring = [
                    a2[x - 2], a2[x - 1], a2[x], a2[x + 1], a2[x + 2], b2[x - 2], b2[x - 1], b2[x], b2[x + 1], b2[x + 2],
                    a[x - 2], r[x - 2], b[x - 2], a[x + 2], r[x + 2], b[x + 2],
                ];
                ring.sort_unstable_by(|a, b| a.total_cmp(b));
                let bg = 0.5 * (ring[7] + ring[8]);
                let height = p - bg;
                if med - bg > 0.15 * height || s[7] - bg > 0.5 * height {
                    continue;
                }
                v.push(((y * w + x) as u32, e * img.adu_scale));
            }
            v
        })
        .collect();
    found.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    found.truncate(MAX_SPIKES);
    found.sort_unstable_by_key(|x| x.0);
    out.idx = found.iter().map(|x| x.0).collect();
    out.excess = found.iter().map(|x| x.1).collect();
    out
}

/// Spikes of a raw one-shot-colour frame: [`spikes`] on each of the four
/// colour sub-grids (every other pixel), so a pixel is judged against
/// neighbours of its own colour. On a light, red, green and blue sit at
/// different sky levels, and against mixed neighbours every pixel of the
/// brightest colour would be a "spike".
pub fn spikes_cfa(img: &Image) -> Spikes {
    let (w, h) = (img.width, img.height);
    let mut out = Spikes { w: w as u32, h: h as u32, ..Default::default() };
    if w < 10 || h < 10 {
        return out;
    }
    let mut found: Vec<(u32, f32)> = Vec::new();
    let (mut sig, mut lev) = (Vec::new(), Vec::new());
    for (ox, oy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
        let (sw, sh) = ((w - ox).div_ceil(2), (h - oy).div_ceil(2));
        let mut sub = Image::new(sw, sh);
        sub.adu_scale = img.adu_scale;
        for y in 0..sh {
            for x in 0..sw {
                sub.data[y * sw + x] = img.get(2 * x + ox, 2 * y + oy);
            }
        }
        let s = spikes(&sub);
        sig.push(s.sigma_adu);
        lev.push(s.level_adu);
        for (&i, &e) in s.idx.iter().zip(&s.excess) {
            let (x, y) = (i as usize % sw, i as usize / sw);
            found.push((((2 * y + oy) * w + 2 * x + ox) as u32, e));
        }
    }
    out.sigma_adu = crate::stats::median_mut(&mut sig);
    out.level_adu = crate::stats::median_mut(&mut lev);
    found.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    found.truncate(MAX_SPIKES);
    found.sort_unstable_by_key(|x| x.0);
    out.idx = found.iter().map(|x| x.0).collect();
    out.excess = found.iter().map(|x| x.1).collect();
    out
}

/// Spikes present in at least `min` of the frames, at their median height.
pub fn persistent(frames: &[&Spikes], min: usize) -> Spikes {
    let Some(first) = frames.first() else { return Spikes::default() };
    let mut seen: HashMap<u32, Vec<f32>> = HashMap::new();
    for f in frames {
        for (i, e) in f.idx.iter().zip(&f.excess) {
            seen.entry(*i).or_default().push(*e);
        }
    }
    let mut v: Vec<(u32, f32)> = seen
        .into_iter()
        .filter(|(_, e)| e.len() >= min.max(1))
        .map(|(i, mut e)| {
            e.sort_unstable_by(|a, b| a.total_cmp(b));
            (i, e[e.len() / 2])
        })
        .collect();
    v.sort_unstable_by_key(|x| x.0);
    let mut sig: Vec<f32> = frames.iter().map(|f| f.sigma_adu).collect();
    sig.sort_unstable_by(|a, b| a.total_cmp(b));
    let mut lv: Vec<f32> = frames.iter().map(|f| f.level_adu).collect();
    lv.sort_unstable_by(|a, b| a.total_cmp(b));
    Spikes {
        w: first.w,
        h: first.h,
        sigma_adu: sig[sig.len() / 2],
        level_adu: lv[lv.len() / 2],
        idx: v.iter().map(|x| x.0).collect(),
        excess: v.iter().map(|x| x.1).collect(),
    }
}

/// A dark's hot pixels against the lights'.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotMatch {
    /// Hot pixels in most of the sampled lights, bright enough to matter.
    pub lights_hot: usize,
    /// Share of those the dark also has.
    pub coverage: f64,
    /// The dark's hot pixels that should show in the lights.
    pub dark_hot: usize,
    /// Share of those that do show.
    pub shown: f64,
    /// Median brightness of shared hot pixels, lights ÷ dark, exposure-scaled.
    pub ratio: Option<f64>,
}

impl HotMatch {
    /// Both directions at once: the dark must have the lights' hot pixels
    /// (or they stay) and the lights must have the dark's (or subtracting
    /// them leaves dark holes).
    pub fn score(&self) -> f64 {
        self.coverage * self.shown
    }

    /// Temperature difference implied by the brightness ratio, °C (lights − dark).
    pub fn delta_c(&self) -> Option<f64> {
        self.ratio.filter(|r| *r > 0.0).map(|r| DOUBLING_C * r.log2())
    }
}

/// `lights`: persistent spikes of the light sample; `any`: spikes in any of
/// them; `scale`: light exposure ÷ dark exposure.
pub fn compare(dark: &Spikes, lights: &Spikes, any: &Spikes, scale: f64) -> HotMatch {
    let visible = VISIBLE_SIGMA * lights.sigma_adu;
    let strong: Vec<(u32, f32)> = lights.idx.iter().zip(&lights.excess).filter(|(_, e)| **e > visible).map(|(i, e)| (*i, *e)).collect();
    let mut ratios = Vec::new();
    let mut covered = 0;
    for (i, e) in &strong {
        if let Some(d) = dark.get(*i) {
            covered += 1;
            if d > 0.0 {
                ratios.push(*e as f64 / (d as f64 * scale));
            }
        }
    }
    let dstrong: Vec<u32> = dark.idx.iter().zip(&dark.excess).filter(|(_, e)| (**e as f64 * scale) > visible as f64).map(|(i, _)| *i).collect();
    let shown = dstrong.iter().filter(|i| any.get(**i).is_some()).count();
    ratios.sort_unstable_by(|a, b| a.total_cmp(b));
    HotMatch {
        lights_hot: strong.len(),
        coverage: if strong.is_empty() { 0.0 } else { covered as f64 / strong.len() as f64 },
        dark_hot: dstrong.len(),
        shown: if dstrong.is_empty() { 0.0 } else { shown as f64 / dstrong.len() as f64 },
        ratio: (ratios.len() >= 20).then(|| ratios[ratios.len() / 2]),
    }
}

/// A dark, bias or dark-flat set (or master) somewhere in the library.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    pub kind: FrameType,
    pub master: bool,
    pub label: String,
    /// Members (one file for a master).
    pub files: Vec<PathBuf>,
    /// The same master file elsewhere (same name, size and time).
    pub copies: Vec<PathBuf>,
    pub folder: PathBuf,
    pub exptime: Option<f64>,
    pub gain: Option<f64>,
    pub offset: Option<f64>,
    pub binning: u32,
    pub width: u32,
    pub height: u32,
    pub temp: Option<f64>,
    pub date: Option<NaiveDate>,
    pub level_adu: Option<f64>,
    pub noise_adu: Option<f64>,
    /// Hot pixels found.
    pub hot: usize,
    pub error: Option<String>,
}

impl Candidate {
    /// Where it lives: the folder name, plus how many copies.
    pub fn place(&self) -> String {
        let name = |p: &Path| p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        if self.copies.is_empty() {
            name(&self.folder)
        } else {
            format!("{} (+{} identical)", name(&self.folder), self.copies.len())
        }
    }

    pub fn contains(&self, p: &Path) -> bool {
        let k = crate::paths::key(p);
        self.files.iter().chain(&self.copies).any(|f| crate::paths::key(f) == k)
    }
}

fn kind_word(k: FrameType) -> &'static str {
    match k {
        FrameType::Dark => "Dark",
        FrameType::Bias => "Bias",
        FrameType::DarkFlat => "Dark flat",
        _ => "Frame",
    }
}

fn exp_str(e: Option<f64>) -> String {
    match e {
        Some(e) if e >= 1.0 => format!("{e:.0} s"),
        Some(e) if e > 0.0 => format!("{e:.2} s"),
        _ => String::new(),
    }
}

/// Every dark, bias and dark-flat set or master in these folders. Identical
/// masters (same name, size and time) become one candidate with copies; the
/// first folder given wins.
pub fn candidates(folders: &[(PathBuf, &[Frame])]) -> Vec<Candidate> {
    let mut out: Vec<Candidate> = Vec::new();
    let mut masters: HashMap<(String, u64, i64), usize> = HashMap::new();
    for (folder, frames) in folders {
        let mut sets: BTreeMap<String, Vec<&Frame>> = BTreeMap::new();
        for f in frames.iter() {
            let k = f.class.frame_type;
            if !matches!(k, FrameType::Dark | FrameType::Bias | FrameType::DarkFlat) || f.class.in_reject_dir || f.class.role == Role::Processed {
                continue;
            }
            if f.class.role == Role::Master {
                let name = f.meta.path.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
                let id = (name, f.meta.file_size, f.meta.mtime_ns);
                if let Some(&c) = masters.get(&id) {
                    out[c].copies.push(f.meta.path.clone());
                    continue;
                }
                masters.insert(id, out.len());
                out.push(new_candidate(folder, &[f], true));
            } else {
                let m = &f.meta;
                let key = format!(
                    "{:?}|{:.2}|{:?}|{:?}|{}|{}x{}|{:?}|{:?}",
                    k,
                    m.exptime.unwrap_or(0.0),
                    m.gain,
                    m.offset,
                    m.xbin.unwrap_or(1),
                    m.width,
                    m.height,
                    m.set_temp.or(m.ccd_temp).map(|t| t.round() as i64),
                    f.class.night
                );
                sets.entry(key).or_default().push(f);
            }
        }
        for (_, v) in sets {
            out.push(new_candidate(folder, &v, false));
        }
    }
    out
}

fn new_candidate(folder: &Path, frames: &[&Frame], master: bool) -> Candidate {
    let m = &frames[0].meta;
    let mut temps: Vec<f64> = frames.iter().filter_map(|f| f.meta.ccd_temp.or(f.meta.set_temp)).collect();
    temps.sort_by(|a, b| a.total_cmp(b));
    let date = frames.iter().filter_map(|f| f.class.night.or(f.meta.date_obs.map(|d| d.date()))).min();
    let kind = frames[0].class.frame_type;
    let label = format!(
        "{}{}{}",
        if master { format!("Master {}", kind_word(kind).to_lowercase()) } else { kind_word(kind).to_string() },
        match exp_str(m.exptime) {
            s if s.is_empty() || kind == FrameType::Bias => String::new(),
            s => format!(" {s}"),
        },
        date.map(|d| format!(" · {}", d.format("%-d %b %Y"))).unwrap_or_default()
    );
    Candidate {
        kind,
        master,
        label,
        files: frames.iter().map(|f| f.meta.path.clone()).collect(),
        copies: Vec::new(),
        folder: folder.to_path_buf(),
        exptime: m.exptime,
        gain: m.gain,
        offset: m.offset,
        binning: m.xbin.unwrap_or(1),
        width: m.width,
        height: m.height,
        temp: (!temps.is_empty()).then(|| temps[temps.len() / 2]),
        date,
        level_adu: None,
        noise_adu: None,
        hot: 0,
        error: None,
    }
}

fn spk_name(key: &str, size: u64, mtime_ns: i64) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    let s = format!("{key}|{size}|{mtime_ns}|spikes{FP_VERSION}");
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}.spk")
}

/// A file's spikes, from the cache folder or measured (and then cached).
pub fn file_spikes(path: &Path, cache: &Cache, memory: &mut HashMap<String, Spikes>) -> Result<Spikes, String> {
    let (size, mtime) = crate::meta::file_stamp(path).map_err(|e| format!("{e:#}"))?;
    let name = spk_name(&crate::paths::key(path), size, mtime);
    if let Some(s) = memory.get(&name) {
        return Ok(s.clone());
    }
    let dir = cache.dir().map(|d| d.join("spikes"));
    if let Some(s) = dir.as_ref().and_then(|d| std::fs::read(d.join(&name)).ok()).and_then(|b| Spikes::from_bytes(&b)) {
        return Ok(s);
    }
    let img = crate::read_image(path).map_err(|e| format!("{e:#}"))?;
    let cfa = crate::meta::read_meta(path).ok().and_then(|m| m.bayer()).is_some();
    let s = if cfa { spikes_cfa(&img) } else { spikes(&img) };
    let written = dir.is_some_and(|d| {
        std::fs::create_dir_all(&d).is_ok() && {
            let tmp = d.join(format!("{name}.tmp"));
            std::fs::write(&tmp, s.to_bytes()).is_ok() && std::fs::rename(&tmp, d.join(&name)).is_ok()
        }
    });
    if !written {
        memory.insert(name, s.clone());
    }
    Ok(s)
}

/// Up to `n` items spread evenly through `v`.
fn spread<T: Clone>(v: &[T], n: usize) -> Vec<T> {
    if v.len() <= n {
        return v.to_vec();
    }
    (0..n).map(|k| v[k * (v.len() - 1) / (n - 1).max(1)].clone()).collect()
}

/// Lights that one dark could calibrate: same exposure, binning, size,
/// gain and offset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Family {
    pub label: String,
    /// `FrameType::Light`, or `Flat` for the flats (bias and dark flats only).
    pub kind: FrameType,
    pub exptime: Option<f64>,
    pub binning: u32,
    pub width: u32,
    pub height: u32,
    pub gain: Option<f64>,
    pub offset: Option<f64>,
    pub temp: Option<f64>,
    pub first: Option<NaiveDate>,
    pub last: Option<NaiveDate>,
    pub n: usize,
    /// Groups in this folder.
    pub groups: Vec<usize>,
    /// Lights sampled for hot pixels.
    pub sample: Vec<PathBuf>,
    /// Persistent hot pixels in the sample (lights only).
    pub lights_hot: usize,
    pub light_noise_adu: Option<f64>,
    /// Ranked candidates for the dark (dark flat for flats) and the bias.
    pub darks: Vec<Ranked>,
    pub bias: Vec<Ranked>,
    /// What the folder's own matching picked (`group.rs`), as candidate indices.
    pub current_dark: Option<usize>,
    pub current_bias: Option<usize>,
}

impl Family {
    /// The best usable dark, if it beats the current one clearly.
    pub fn better_dark(&self) -> Option<&Ranked> {
        let best = self.darks.first().filter(|r| r.usable)?;
        match self.current_dark {
            Some(c) if c == best.cand => None,
            Some(c) => {
                let cur = self.darks.iter().find(|r| r.cand == c)?;
                let (b, k) = (best.hot.as_ref()?.score(), cur.hot.as_ref().map(|h| h.score()).unwrap_or(0.0));
                (b - k >= 0.03 || !cur.usable).then_some(best)
            }
            None => Some(best),
        }
    }
}

/// One candidate's place in a family's ranking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ranked {
    pub cand: usize,
    pub usable: bool,
    pub hot: Option<HotMatch>,
    /// One sentence: why it's ranked here.
    pub summary: String,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Fingerprints {
    pub candidates: Vec<Candidate>,
    pub families: Vec<Family>,
    pub seconds: f64,
}

fn close(a: Option<f64>, b: Option<f64>, tol: f64) -> Option<bool> {
    Some((a? - b?).abs() <= tol)
}

/// Where else to look for darks, bias and dark flats, headers only: `also`,
/// the folders beside `dir`, and every folder the calibration library
/// found them in (`library::refresh`). `dir` and folders inside it are left
/// out (its own scan has them).
pub fn other_folders(dir: &Path, cfg: &crate::config::Config, also: &[PathBuf]) -> Vec<(PathBuf, Vec<Frame>)> {
    let mut folders: Vec<PathBuf> = also.to_vec();
    if let Some(parent) = dir.parent() {
        let mut v: Vec<PathBuf> = std::fs::read_dir(parent)
            .map(|rd| rd.flatten().filter(|e| e.file_type().is_ok_and(|t| t.is_dir())).map(|e| e.path()).collect())
            .unwrap_or_default();
        v.sort();
        folders.extend(v.into_iter().filter(|p| !p.file_name().is_some_and(|n| n.to_string_lossy().starts_with('.'))));
    }
    folders.extend(crate::library::offset_folders(cfg));
    let mut seen = std::collections::HashSet::new();
    folders
        .into_iter()
        .filter(|d| !crate::library::inside(d, dir) && seen.insert(crate::paths::key(d)))
        .map(|d| {
            let mut c = Cache::disabled();
            let o = crate::pipeline::run(&d, cfg, &mut c, false, &|_, _, _| {});
            (d, o.scan.frames)
        })
        .collect()
}

/// Match every light family (and the flats) in `t` against the darks, bias
/// and dark flats in this folder and `others`.
pub fn build(t: &Triage, root: &Path, others: &[(PathBuf, Vec<Frame>)], cache: &Cache, progress: &(dyn Fn(usize, usize) + Sync)) -> Fingerprints {
    build_from(&t.scan.frames, &t.grouping, &t.reports, root, others, cache, progress)
}

/// `build` from the parts of a triage (the GUI keeps those after sending
/// the triage itself to the UI).
pub fn build_from(
    frames: &[Frame],
    grouping: &crate::group::Grouping,
    reports: &[FrameReport],
    root: &Path,
    others: &[(PathBuf, Vec<Frame>)],
    cache: &Cache,
    progress: &(dyn Fn(usize, usize) + Sync),
) -> Fingerprints {
    let t0 = std::time::Instant::now();
    let mut folders: Vec<(PathBuf, &[Frame])> = vec![(root.to_path_buf(), frames)];
    for (d, f) in others {
        if crate::paths::key(d) != crate::paths::key(root) {
            folders.push((d.clone(), f.as_slice()));
        }
    }
    let mut cands = candidates(&folders);
    let grade: HashMap<usize, Grade> = reports.iter().map(|r: &FrameReport| (r.frame, r.grade)).collect();

    // Light families.
    let mut fams: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for g in grouping.by_kind(FrameType::Light) {
        if g.master {
            continue;
        }
        let m = &frames[g.members[0]].meta;
        let key = format!("{:.2}|{}|{}x{}|{:?}|{:?}", m.exptime.unwrap_or(0.0), m.xbin.unwrap_or(1), m.width, m.height, m.gain, m.offset);
        fams.entry(key).or_default().push(g.id);
    }
    let mut families: Vec<Family> = Vec::new();
    for gids in fams.values() {
        let members: Vec<usize> = gids.iter().flat_map(|g| grouping.groups[*g].members.iter().copied()).filter(|i| frames[*i].class.role == Role::Raw).collect();
        let mut good: Vec<usize> = members.iter().copied().filter(|i| !frames[*i].class.in_reject_dir && grade.get(i).is_none_or(|g| *g != Grade::Reject)).collect();
        good.sort_by_key(|i| frames[*i].class.local_time);
        families.push(new_family(FrameType::Light, frames, gids.clone(), &members, spread(&good, LIGHT_SAMPLE)));
    }
    // Flats: one family per size and binning (they need bias or dark flats).
    let mut ffam: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for g in grouping.by_kind(FrameType::Flat) {
        if g.master || g.retired {
            continue;
        }
        let m = &frames[g.members[0]].meta;
        ffam.entry(format!("{}|{}x{}", m.xbin.unwrap_or(1), m.width, m.height)).or_default().push(g.id);
    }
    for gids in ffam.values() {
        let members: Vec<usize> = gids.iter().flat_map(|g| grouping.groups[*g].members.iter().copied()).collect();
        families.push(new_family(FrameType::Flat, frames, gids.clone(), &members, Vec::new()));
    }

    // Fingerprint the candidates that could serve some family, and the light samples.
    let fits = |c: &Candidate, f: &Family| c.width == f.width && c.height == f.height && c.binning == f.binning;
    let wanted: Vec<usize> = (0..cands.len()).filter(|&c| families.iter().any(|f| fits(&cands[c], f))).collect();
    let light_files: Vec<PathBuf> = families.iter().flat_map(|f| f.sample.clone()).collect();
    let jobs: Vec<PathBuf> = wanted
        .iter()
        .flat_map(|&c| if cands[c].master { cands[c].files.clone() } else { spread(&cands[c].files, SET_SAMPLE) })
        .chain(light_files)
        .collect();
    let total = jobs.len();
    let done = std::sync::atomic::AtomicUsize::new(0);
    let measured: HashMap<String, Result<Spikes, String>> = jobs
        .par_iter()
        .with_max_len(1)
        .map(|p| {
            let mut mem = HashMap::new();
            let r = file_spikes(p, cache, &mut mem);
            progress(done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1, total);
            (crate::paths::key(p), r)
        })
        .collect();
    let get = |p: &Path| measured.get(&crate::paths::key(p));
    let mut cand_spikes: HashMap<usize, Spikes> = HashMap::new();
    for &c in &wanted {
        let files = if cands[c].master { cands[c].files.clone() } else { spread(&cands[c].files, SET_SAMPLE) };
        let ok: Vec<&Spikes> = files.iter().filter_map(|p| get(p).and_then(|r| r.as_ref().ok())).collect();
        if ok.is_empty() {
            cands[c].error = files.iter().find_map(|p| get(p).and_then(|r| r.as_ref().err().cloned()));
            continue;
        }
        let s = if ok.len() == 1 { ok[0].clone() } else { persistent(&ok, ok.len().div_ceil(2)) };
        cands[c].level_adu = Some(s.level_adu as f64);
        cands[c].noise_adu = Some(s.sigma_adu as f64);
        cands[c].hot = s.len();
        cand_spikes.insert(c, s);
    }

    for fam in &mut families {
        let ls: Vec<&Spikes> = fam.sample.iter().filter_map(|p| get(p).and_then(|r| r.as_ref().ok())).collect();
        let lights = (!ls.is_empty()).then(|| persistent(&ls, (ls.len() / 2).max(2).min(ls.len())));
        let any = (!ls.is_empty()).then(|| persistent(&ls, 1));
        if let Some(l) = &lights {
            fam.light_noise_adu = Some(l.sigma_adu as f64);
            fam.lights_hot = l.idx.iter().zip(&l.excess).filter(|(_, e)| **e > VISIBLE_SIGMA * l.sigma_adu).count();
        }
        // What this folder's own matching uses.
        let cal = if fam.kind == FrameType::Light { &grouping.light_calibration } else { &grouping.flat_calibration };
        let mine: Vec<&crate::group::Calibration> = cal.iter().filter(|c| fam.groups.contains(&c.group)).collect();
        let pick = |ids: &[usize]| ids.first().and_then(|g| grouping.groups.get(*g)).and_then(|g| g.members.first()).map(|i| frames[*i].meta.path.clone());
        let dark_of = mine.iter().find_map(|c| pick(if fam.kind == FrameType::Light { &c.darks } else { &c.dark_flats }));
        let bias_of = mine.iter().find_map(|c| pick(&c.bias));
        fam.current_dark = dark_of.and_then(|p| cands.iter().position(|c| c.contains(&p)));
        fam.current_bias = bias_of.and_then(|p| cands.iter().position(|c| c.contains(&p)));

        let dark_kind = if fam.kind == FrameType::Light { FrameType::Dark } else { FrameType::DarkFlat };
        let mut darks: Vec<Ranked> = Vec::new();
        for (ci, c) in cands.iter().enumerate() {
            if c.kind != dark_kind || !fits(c, fam) {
                continue;
            }
            darks.push(rank_dark(ci, c, fam, cand_spikes.get(&ci), lights.as_ref(), any.as_ref()));
        }
        darks.sort_by(|a, b| {
            b.usable.cmp(&a.usable).then_with(|| {
                let (x, y) = (a.hot.as_ref().map(|h| h.score()).unwrap_or(-1.0), b.hot.as_ref().map(|h| h.score()).unwrap_or(-1.0));
                let (ca, cb) = (&cands[a.cand], &cands[b.cand]);
                if (x - y).abs() > 0.01 {
                    y.total_cmp(&x)
                } else {
                    days_from(ca, fam)
                        .cmp(&days_from(cb, fam))
                        .then(cb.master.cmp(&ca.master))
                        .then((crate::paths::key(&cb.folder) == crate::paths::key(root)).cmp(&(crate::paths::key(&ca.folder) == crate::paths::key(root))))
                }
            })
        });
        let dark_level = darks.first().filter(|r| r.usable).and_then(|r| cands[r.cand].level_adu);
        let mut bias: Vec<Ranked> = Vec::new();
        for (ci, c) in cands.iter().enumerate() {
            if c.kind != FrameType::Bias || !fits(c, fam) {
                continue;
            }
            bias.push(rank_bias(ci, c, fam, dark_level));
        }
        bias.sort_by(|a, b| {
            b.usable.cmp(&a.usable).then_with(|| {
                // Within 1 ADU of the dark is as good as it gets; then nearest
                // in time, then a ready master, then this folder's own.
                let lv = |r: &Ranked| match (dark_level, cands[r.cand].level_adu) {
                    (Some(d), Some(l)) => (((d - l).abs() - 1.0).max(0.0) * 10.0) as i64,
                    _ => 1_000_000,
                };
                let (ca, cb) = (&cands[a.cand], &cands[b.cand]);
                lv(a)
                    .cmp(&lv(b))
                    .then(days_from(ca, fam).cmp(&days_from(cb, fam)))
                    .then(cb.master.cmp(&ca.master))
                    .then((crate::paths::key(&cb.folder) == crate::paths::key(root)).cmp(&(crate::paths::key(&ca.folder) == crate::paths::key(root))))
            })
        });
        fam.darks = darks;
        fam.bias = bias;
    }
    Fingerprints { candidates: cands, families, seconds: t0.elapsed().as_secs_f64() }
}

fn new_family(kind: FrameType, frames: &[Frame], groups: Vec<usize>, members: &[usize], sample: Vec<usize>) -> Family {
    let m = &frames[members[0]].meta;
    let nights: Vec<NaiveDate> = members.iter().filter_map(|i| frames[*i].class.night).collect();
    let mut temps: Vec<f64> = members.iter().filter_map(|i| frames[*i].meta.ccd_temp.or(frames[*i].meta.set_temp)).collect();
    temps.sort_by(|a, b| a.total_cmp(b));
    let label = match kind {
        FrameType::Light => format!("Lights {}", exp_str(m.exptime)),
        _ => "Flats".to_string(),
    };
    Family {
        label,
        kind,
        exptime: m.exptime,
        binning: m.xbin.unwrap_or(1),
        width: m.width,
        height: m.height,
        gain: m.gain,
        offset: m.offset,
        temp: (!temps.is_empty()).then(|| temps[temps.len() / 2]),
        first: nights.iter().min().copied(),
        last: nights.iter().max().copied(),
        n: members.len(),
        groups,
        sample: sample.iter().map(|i| frames[*i].meta.path.clone()).collect(),
        lights_hot: 0,
        light_noise_adu: None,
        darks: Vec::new(),
        bias: Vec::new(),
        current_dark: None,
        current_bias: None,
    }
}

/// Days between a candidate and the family's nights (0 inside them).
fn days_from(c: &Candidate, f: &Family) -> i64 {
    match (c.date, f.first, f.last) {
        (Some(d), Some(a), Some(_)) if d < a => (a - d).num_days(),
        (Some(d), Some(_), Some(b)) if d > b => (d - b).num_days(),
        (Some(_), Some(_), Some(_)) => 0,
        _ => 100_000,
    }
}

fn when(c: &Candidate, f: &Family) -> Option<String> {
    let d = c.date?;
    let days = days_from(c, f);
    let a = f.first?;
    Some(if days == 0 {
        format!("made {} (during the {})", d.format("%-d %b"), noun(f))
    } else if d < a {
        format!("made {}, {} before the {}", d.format("%-d %b %Y"), span(days), noun(f))
    } else {
        format!("made {}, {} after the {}", d.format("%-d %b %Y"), span(days), noun(f))
    })
}

fn noun(f: &Family) -> &'static str {
    if f.kind == FrameType::Light { "lights" } else { "flats" }
}

fn span(days: i64) -> String {
    match days {
        0..=13 => format!("{days} days"),
        14..=59 => format!("{} weeks", days / 7),
        _ => format!("{} months", (days as f64 / 30.4).round() as i64),
    }
}

fn rank_dark(ci: usize, c: &Candidate, f: &Family, s: Option<&Spikes>, lights: Option<&Spikes>, any: Option<&Spikes>) -> Ranked {
    let mut reasons = Vec::new();
    let mut usable = true;
    let mut problem = None;
    if let Some(e) = &c.error {
        usable = false;
        problem = Some(format!("couldn't read it: {e}"));
    }
    match close(c.exptime, f.exptime, f.exptime.unwrap_or(0.0) * 0.005 + 0.001) {
        Some(false) => {
            usable = false;
            problem.get_or_insert(format!("exposure {} vs {} (would need scaling with a bias)", exp_str(c.exptime), exp_str(f.exptime)));
        }
        None => reasons.push("exposure unknown on one side".to_string()),
        _ => {}
    }
    for (what, a, b) in [("gain", c.gain, f.gain), ("offset", c.offset, f.offset)] {
        match close(a, b, 0.5) {
            Some(false) => {
                usable = false;
                problem.get_or_insert(format!("{what} {:.0} vs {:.0}", a.unwrap_or(0.0), b.unwrap_or(0.0)));
            }
            None if c.master => {}
            None => reasons.push(format!("{what} unknown on one side")),
            _ => {}
        }
    }
    if c.master && (c.gain.is_none() || c.temp.is_none()) {
        reasons.push("master has no GAIN, OFFSET or CCD-TEMP: matched on its hot pixels".to_string());
    }
    if let Some(w) = when(c, f) {
        reasons.push(w);
    }
    let hot = match (s, lights, any) {
        (Some(s), Some(l), Some(a)) if f.kind == FrameType::Light => {
            let scale = match (f.exptime, c.exptime) {
                (Some(a), Some(b)) if b > 0.0 => a / b,
                _ => 1.0,
            };
            Some(compare(s, l, a, scale))
        }
        _ => None,
    };
    let summary = if let Some(p) = problem {
        format!("Not usable: {p}.")
    } else if let Some(h) = &hot {
        let mut t = format!("Has {:.0}% of the lights' {} hot pixels", h.coverage * 100.0, h.lights_hot);
        match (h.ratio, h.delta_c()) {
            (Some(r), Some(dc)) if dc.abs() >= 1.5 => {
                t += &format!(", {r:.2}× as bright in the lights (about {:.0} °C {})", dc.abs(), if dc > 0.0 { "warmer" } else { "colder" });
                reasons.push(format!("temperature from hot-pixel brightness, assuming dark current doubles every {DOUBLING_C:.0} °C"));
            }
            (Some(r), _) => t += &format!(", {r:.2}× as bright (same temperature)"),
            _ => {}
        }
        t += &format!(". {:.0}% of its {} bright hot pixels show in the lights", h.shown * 100.0, h.dark_hot);
        if h.coverage < 0.8 {
            t += ". The lights have hot pixels this dark lacks: the sensor has changed since, or it's another camera";
        }
        if h.shown < 0.8 {
            t += ". It has hot pixels the lights lack: subtracting them would leave dark holes (the sensor gained them later, or it's another camera)";
        }
        t + "."
    } else if f.kind == FrameType::Light {
        "Matches on headers; hot pixels not compared.".to_string()
    } else {
        "Matches on headers.".to_string()
    };
    if let (Some(a), Some(b)) = (c.temp, f.temp) {
        if (a - b).abs() >= 1.0 {
            reasons.push(format!("headers say {a:.0} °C vs the lights' {b:.0} °C"));
        }
    }
    Ranked { cand: ci, usable, hot, summary, reasons }
}

fn rank_bias(ci: usize, c: &Candidate, f: &Family, dark_level: Option<f64>) -> Ranked {
    let mut reasons = Vec::new();
    let mut usable = true;
    let mut problem = None;
    if let Some(e) = &c.error {
        usable = false;
        problem = Some(format!("couldn't read it: {e}"));
    }
    for (what, a, b) in [("gain", c.gain, f.gain), ("offset", c.offset, f.offset)] {
        if close(a, b, 0.5) == Some(false) {
            usable = false;
            problem.get_or_insert(format!("{what} {:.0} vs {:.0}", a.unwrap_or(0.0), b.unwrap_or(0.0)));
        }
    }
    if let Some(w) = when(c, f) {
        reasons.push(w);
    }
    let summary = match (problem, dark_level, c.level_adu) {
        (Some(p), _, _) => format!("Not usable: {p}."),
        (None, Some(d), Some(l)) if (d - l).abs() <= 1.0 => format!("Level {l:.1} ADU, matching the dark's {d:.1}: same offset setting."),
        (None, Some(d), Some(l)) if l < d && d - l <= 5.0 => format!("Level {l:.1} ADU, just under the dark's {d:.1}: same offset setting."),
        (None, Some(d), Some(l)) => {
            reasons.push("a dark's level is the bias plus dark current, so the right bias sits just below it".into());
            format!("Level {l:.1} ADU vs the dark's {d:.1}: probably a different offset setting.")
        }
        (None, None, Some(l)) => format!("Level {l:.1} ADU."),
        _ => "Matches on headers.".to_string(),
    };
    Ranked { cand: ci, usable, hot: None, summary, reasons }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(seed: u64, hot: &[(usize, f32)], star: Option<(usize, usize)>) -> Image {
        let (w, h) = (200, 150);
        let mut rng = crate::synth::Rng::new(seed);
        let mut d: Vec<f32> = (0..w * h).map(|_| (500.0 + 5.0 * rng.gauss() as f32) / 65535.0).collect();
        for (i, a) in hot {
            d[*i] += a / 65535.0;
        }
        if let Some((cx, cy)) = star {
            for y in cy - 4..=cy + 4 {
                for x in cx - 4..=cx + 4 {
                    let r2 = ((x as f32 - cx as f32).powi(2) + (y as f32 - cy as f32).powi(2)) / (2.0 * 1.7f32.powi(2));
                    d[y * w + x] += 3000.0 * (-r2).exp() / 65535.0;
                }
            }
        }
        let mut img = Image::from_data(w, h, d);
        img.adu_scale = 65535.0;
        img
    }

    #[test]
    fn hot_pixels_match_and_drift() {
        let old: Vec<(usize, f32)> = (0..300).map(|k| (1000 + k * 93, 400.0 + k as f32)).collect();
        let mut new = old.clone();
        new.extend((0..100).map(|k| (1050 + k * 211, 600.0)));
        // A star is not a spike.
        let s = spikes(&frame(1, &old, Some((100, 75))));
        assert!(s.get((75 * 200 + 100) as u32).is_none());
        assert!(s.len() >= 290 && s.len() <= 310, "{}", s.len());
        assert!((s.sigma_adu - 5.0).abs() < 0.5, "{}", s.sigma_adu);
        // Lights with the new hot pixels, twice as bright (warmer).
        let lights: Vec<Spikes> = (0..4).map(|k| {
            let hot: Vec<(usize, f32)> = new.iter().map(|(i, a)| (*i, a * 2.0)).collect();
            spikes(&frame(10 + k, &hot, None))
        }).collect();
        let refs: Vec<&Spikes> = lights.iter().collect();
        let (l, any) = (persistent(&refs, 2), persistent(&refs, 1));
        let m_old = compare(&s, &l, &any, 1.0);
        let m_new = compare(&spikes(&frame(2, &new, None)), &l, &any, 1.0);
        assert!((m_old.coverage - 0.75).abs() < 0.03, "{m_old:?}");
        assert!(m_new.coverage > 0.97, "{m_new:?}");
        assert!((m_new.ratio.unwrap() - 2.0).abs() < 0.1, "{m_new:?}");
        assert!((m_new.delta_c().unwrap() - 6.0).abs() < 0.5);
        assert!(m_old.shown > 0.95, "{m_old:?} {}", s.len());
        // Round trip.
        assert_eq!(Spikes::from_bytes(&s.to_bytes()).unwrap(), s);
    }
}
