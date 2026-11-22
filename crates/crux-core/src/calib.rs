//! Calibration-frame measurements and the flat-checking engine.
//!
//! The flat maths is a port of `FlatChecker.js` (C:\Projects\pixinsight):
//! - **Flat stability:** each raw flat against the median of its set's
//!   unsaturated flats, smoothed at 64 px. Large-scale change between frames
//!   means the light source changed (cloud, shifting twilight).
//! - **Flats against lights:** dark-subtracted lights are median-stacked at
//!   low resolution; dithering (and, in a mosaic, different fields) removes
//!   the stars and target, leaving the illumination the optics delivered.
//!   Dividing it by the flat gives the error every calibrated light carries.
//!
//! FlatChecker bins by 8 and smooths at 8 and 18 binned pixels. crux keeps a
//! 32× binned map of every frame (cached), so the same physical scales are
//! 2 and 4.5 binned pixels.

use crate::image::Image;
use crate::stats::{median_mut, median_sigma_mut};
use serde::{Deserialize, Serialize};

/// Bump when `CalMetrics` changes; calibration frames are re-measured
/// without touching the (slower) light analysis.
pub const CAL_VERSION: u32 = 2;

/// Binning factor of the stored maps.
pub const MAP_BIN: usize = 32;
/// Smoothing for flat-to-flat comparisons, full-resolution pixels (FlatChecker: 8 × 8).
pub const DRIFT_SIGMA_PX: f32 = 64.0;
/// Smoothing for the flats-against-lights error map (FlatChecker: 18 × 8).
pub const ERROR_SIGMA_PX: f32 = 144.0;

/// Map width (bins) at which the pixel scales above were set: a 6248 px
/// ASI2600 frame. Narrower maps smooth proportionally less, so a small
/// sensor isn't smoothed over a larger fraction of its frame. (At 9% of the
/// width, the blur lags a strong edge gradient and the target mask eats it.)
const REFERENCE_MAP_W: f32 = 195.0;

/// Smoothing sigma in bins for a map `w` bins wide.
pub(crate) fn sigma_bins(px: f32, w: usize) -> f32 {
    let s = px / MAP_BIN as f32;
    s * (w as f32 / REFERENCE_MAP_W).min(1.0)
}

/// A block-averaged image in ADU. Rows and columns that don't fill a whole
/// block are dropped.
#[derive(Debug, Clone, PartialEq)]
pub struct Map {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

const MAP_MAGIC: &[u8; 8] = b"CRUXMAP1";

impl Map {
    pub fn bin(img: &Image, n: usize) -> Map {
        let (bw, bh) = ((img.width / n).max(1), (img.height / n).max(1));
        let mut data = vec![0f32; bw * bh];
        let mut acc = vec![0f64; bw];
        for by in 0..bh {
            acc.iter_mut().for_each(|a| *a = 0.0);
            for y in by * n..((by + 1) * n).min(img.height) {
                let row = img.row(y);
                for (bx, a) in acc.iter_mut().enumerate() {
                    let s: f32 = row[bx * n..((bx + 1) * n).min(img.width)].iter().sum();
                    *a += s as f64;
                }
            }
            let k = img.adu_scale as f64 / (n * n) as f64;
            for (o, a) in data[by * bw..(by + 1) * bw].iter_mut().zip(&acc) {
                *o = (*a * k) as f32;
            }
        }
        Map { w: bw, h: bh, data }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(16 + self.data.len() * 4);
        b.extend_from_slice(MAP_MAGIC);
        b.extend_from_slice(&(self.w as u32).to_le_bytes());
        b.extend_from_slice(&(self.h as u32).to_le_bytes());
        for v in &self.data {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b
    }

    pub fn from_bytes(b: &[u8]) -> Option<Map> {
        if b.len() < 16 || &b[..8] != MAP_MAGIC {
            return None;
        }
        let w = u32::from_le_bytes(b[8..12].try_into().ok()?) as usize;
        let h = u32::from_le_bytes(b[12..16].try_into().ok()?) as usize;
        if b.len() != 16 + w * h * 4 {
            return None;
        }
        let data = b[16..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        Some(Map { w, h, data })
    }
}

/// Per-frame numbers for bias, darks and flats (raw or master).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CalMetrics {
    /// Mean within 5σ of the median, ADU.
    pub level_adu: f32,
    /// Standard deviation within 5σ (read noise plus fixed pattern), ADU.
    pub noise_adu: f32,
    /// Fraction of pixels at or above 99.5% of full scale.
    pub saturated: f32,
    /// Fraction of pixels more than 10σ (and 50 ADU) above the median.
    pub hot: f32,
    /// Large-scale structure: 1st–99th percentile spread of the 64 px
    /// smoothed map, ADU. Light leaks and amp glow show here.
    pub structure_adu: f32,
    /// Row-to-row and column-to-column noise of the row and column means
    /// (banding), ADU.
    #[serde(default)]
    pub row_noise_adu: f32,
    #[serde(default)]
    pub col_noise_adu: f32,
    pub seconds: f32,
}

/// Normalised level at which a sample counts as saturated (FlatChecker).
pub const SAT_LEVEL: f32 = 0.995;

pub fn analyze_cal(img: &Image) -> (CalMetrics, Map) {
    let t0 = std::time::Instant::now();
    let k = img.adu_scale;
    let s: Vec<f32> = img.data.iter().step_by(13).copied().collect();
    let (level, noise) = clipped_mean_sd(s, 5.0);
    let hot_thr = level + (10.0 * noise).max(50.0 / k);
    let (mut sat, mut hot) = (0usize, 0usize);
    // Row and column means with outliers (hot pixels, stars) replaced by the level.
    let (lo, hi) = (level - 5.0 * noise, level + 5.0 * noise);
    let mut rows = vec![0f64; img.height];
    let mut cols = vec![0f64; img.width];
    for (y, row) in img.data.chunks_exact(img.width).enumerate() {
        let mut rs = 0f64;
        for (c, &v) in cols.iter_mut().zip(row) {
            sat += (v >= SAT_LEVEL) as usize;
            hot += (v > hot_thr) as usize;
            let v = if v < lo || v > hi { level } else { v } as f64;
            rs += v;
            *c += v;
        }
        rows[y] = rs / img.width as f64;
    }
    cols.iter_mut().for_each(|c| *c /= img.height as f64);
    let map = Map::bin(img, MAP_BIN);
    let sm = gaussian_blur(&map.data, map.w, map.h, sigma_bins(DRIFT_SIGMA_PX, map.w));
    let n = img.data.len().max(1) as f32;
    let m = CalMetrics {
        level_adu: level * k,
        noise_adu: noise * k,
        saturated: sat as f32 / n,
        hot: hot as f32 / n,
        structure_adu: percentile_spread(&sm),
        row_noise_adu: step_noise(&rows) * k,
        col_noise_adu: step_noise(&cols) * k,
        seconds: t0.elapsed().as_secs_f32(),
    };
    (m, map)
}

/// Robust noise of a sequence from its first differences (slow trends cancel).
fn step_noise(v: &[f64]) -> f32 {
    if v.len() < 3 {
        return 0.0;
    }
    let mut d: Vec<f32> = v.windows(2).map(|w| (w[1] - w[0]) as f32).collect();
    let (_, sig) = median_sigma_mut(&mut d);
    sig / std::f32::consts::SQRT_2
}

/// Read noise from two frames: the clipped SD of their difference over √2,
/// so fixed pattern cancels. Normalised units.
pub fn pair_noise(a: &Image, b: &Image) -> Option<f32> {
    if a.data.len() != b.data.len() || a.data.is_empty() {
        return None;
    }
    let d: Vec<f32> = a.data.iter().zip(&b.data).step_by(7).map(|(x, y)| x - y).collect();
    let (_, sd) = clipped_mean_sd(d, 5.0);
    Some(sd / std::f32::consts::SQRT_2)
}

/// Mean and standard deviation within `k` MAD-sigmas of the median. Raw
/// frames hold integers, so the median and MAD of a bias are the same
/// whole numbers in every frame; the clipped mean and SD resolve 0.01 ADU.
fn clipped_mean_sd(mut s: Vec<f32>, k: f32) -> (f32, f32) {
    let (med, sig) = median_sigma_mut(&mut s.clone());
    let sig = sig.max(1e-6);
    let (mut n, mut sum, mut sum2) = (0f64, 0f64, 0f64);
    for v in s.drain(..) {
        if (v - med).abs() <= k * sig {
            let d = (v - med) as f64;
            n += 1.0;
            sum += d;
            sum2 += d * d;
        }
    }
    if n < 2.0 {
        return (med, sig);
    }
    let mean = sum / n;
    (med + mean as f32, ((sum2 / n - mean * mean).max(0.0)).sqrt() as f32)
}

// ---------------------------------------------------------------------------
// FlatChecker engine. Maps are row-major w × h.
// ---------------------------------------------------------------------------

fn quantile_sorted(s: &[f32], q: f64) -> f32 {
    let i = (q * (s.len() as f64 - 1.0)).round() as usize;
    s[i.min(s.len() - 1)]
}

fn sorted(a: &[f32]) -> Vec<f32> {
    let mut s = a.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    s
}

/// Divide by the median in place; returns the median (None if not positive).
pub fn normalise(a: &mut [f32]) -> Option<f32> {
    let m = median_mut(&mut a.to_vec());
    if !(m > 0.0) {
        return None;
    }
    a.iter_mut().for_each(|v| *v /= m);
    Some(m)
}

/// Per-pixel median of equally sized maps.
pub fn median_stack(maps: &[&[f32]]) -> Vec<f32> {
    let n = maps.first().map_or(0, |m| m.len());
    let k = maps.len();
    let mut v = vec![0f32; k];
    (0..n)
        .map(|i| {
            for (j, m) in maps.iter().enumerate() {
                v[j] = m[i];
            }
            v.sort_by(|a, b| a.total_cmp(b));
            if k % 2 == 1 {
                v[k / 2]
            } else {
                0.5 * (v[k / 2 - 1] + v[k / 2])
            }
        })
        .collect()
}

/// Separable Gaussian blur, edges extended.
pub fn gaussian_blur(a: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    let r = (3.0 * sigma).ceil() as isize;
    let mut k: Vec<f64> = (-r..=r).map(|i| (-0.5 * (i * i) as f64 / (sigma * sigma) as f64).exp()).collect();
    let sum: f64 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= sum);
    let clamp = |v: isize, n: usize| v.clamp(0, n as isize - 1) as usize;
    let mut tmp = vec![0f32; w * h];
    for y in 0..h {
        let row = &a[y * w..(y + 1) * w];
        for x in 0..w {
            let mut s = 0f64;
            for j in -r..=r {
                s += k[(j + r) as usize] * row[clamp(x as isize + j, w)] as f64;
            }
            tmp[y * w + x] = s as f32;
        }
    }
    let mut out = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut s = 0f64;
            for j in -r..=r {
                s += k[(j + r) as usize] * tmp[clamp(y as isize + j, h) * w + x] as f64;
            }
            out[y * w + x] = s as f32;
        }
    }
    out
}

/// Block mean by an integer factor.
fn downsample(a: &[f32], w: usize, h: usize, f: usize) -> (Vec<f32>, usize, usize) {
    let (w2, h2) = ((w / f).max(1), (h / f).max(1));
    let mut out = vec![0f32; w2 * h2];
    for y in 0..(h2 * f).min(h) {
        for x in 0..(w2 * f).min(w) {
            out[(y / f) * w2 + x / f] += a[y * w + x];
        }
    }
    let k = 1.0 / (f * f) as f32;
    out.iter_mut().for_each(|v| *v *= k);
    (out, w2, h2)
}

/// One row (length `full_w`) of a map binned by `n`, bilinear.
pub fn upsample_row(map: &[f32], bw: usize, bh: usize, n: usize, y: usize, full_w: usize, out: &mut [f32]) {
    let fy = ((y as f32 + 0.5) / n as f32 - 0.5).clamp(0.0, (bh - 1) as f32);
    let y0 = fy.floor() as usize;
    let y1 = (y0 + 1).min(bh - 1);
    let ty = fy - y0 as f32;
    for (x, o) in out.iter_mut().enumerate().take(full_w) {
        let fx = ((x as f32 + 0.5) / n as f32 - 0.5).clamp(0.0, (bw - 1) as f32);
        let x0 = fx.floor() as usize;
        let x1 = (x0 + 1).min(bw - 1);
        let tx = fx - x0 as f32;
        let a = map[y0 * bw + x0] + tx * (map[y0 * bw + x1] - map[y0 * bw + x0]);
        let b = map[y1 * bw + x0] + tx * (map[y1 * bw + x1] - map[y1 * bw + x0]);
        *o = a + ty * (b - a);
    }
}

/// Smooth version of `a`, weighting each pixel by `mask` (0..1). Where the
/// mask leaves too little data within reach, blends in the same thing at 4×
/// coarser scale.
fn masked_blur(a: &[f32], mask: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    let n = a.len();
    let am: Vec<f32> = a.iter().zip(mask).map(|(v, m)| v * m).collect();
    let total: f32 = mask.iter().sum();
    if total < 1.0 {
        return vec![1.0; n];
    }
    let mut num = gaussian_blur(&am, w, h, sigma);
    let den = gaussian_blur(mask, w, h, sigma);
    const ENOUGH: f32 = 0.2;
    let mut sparse = false;
    for i in 0..n {
        num[i] /= den[i].max(1e-6);
        sparse |= den[i] < ENOUGH;
    }
    if !sparse || w < 8 || h < 8 {
        return num;
    }
    let f = 4;
    let (dm, dw, dh) = downsample(mask, w, h, f);
    let (mut da, _, _) = downsample(&am, w, h, f);
    for (v, m) in da.iter_mut().zip(&dm) {
        *v = if *m > 0.0 { *v / m } else { 0.0 };
    }
    let coarse = masked_blur(&da, &dm, dw, dh, sigma);
    let mut row = vec![0f32; w];
    for y in 0..h {
        upsample_row(&coarse, dw, dh, f, y, w, &mut row);
        for x in 0..w {
            let i = y * w + x;
            if den[i] < ENOUGH {
                let t = den[i] / ENOUGH;
                num[i] = t * num[i] + (1.0 - t) * row[x];
            }
        }
    }
    num
}

/// Smooth, large-scale version of a ratio map, ignoring pixels that stand
/// out from it (target and leftover stars above, defects below). Returns the
/// map (median 1) and the fraction masked.
pub fn smooth_masked(a: &[f32], w: usize, h: usize, sigma: f32) -> (Vec<f32>, f64) {
    let n = a.len();
    let mut mask = vec![1f32; n];
    for _ in 0..6 {
        let sm = masked_blur(a, &mask, w, h, sigma);
        let mut res: Vec<f32> = (0..n).filter(|&i| mask[i] > 0.0).map(|i| (a[i] - sm[i]).abs()).collect();
        // Floor at 0.2%: in a very smooth map the blur's small bias at the
        // edges of a strong gradient would otherwise be masked as target.
        // Stars and nebulosity stand out by far more.
        let s = (1.4826 * median_mut(&mut res)).max(2e-3);
        for i in 0..n {
            let d = a[i] - sm[i];
            mask[i] = if d < 2.0 * s && d > -4.0 * s { 1.0 } else { 0.0 };
        }
    }
    let mut sm = masked_blur(a, &mask, w, h, sigma);
    let _ = normalise(&mut sm);
    let kept: f32 = mask.iter().sum();
    (sm, 1.0 - kept as f64 / n as f64)
}

/// a / b, normalised to median 1.
pub fn ratio_map(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut out: Vec<f32> = a.iter().zip(b).map(|(x, y)| if *y > 0.0 { x / y } else { 0.0 }).collect();
    let _ = normalise(&mut out);
    out
}

/// 1st to 99th percentile spread, in the map's units.
pub fn percentile_spread(map: &[f32]) -> f32 {
    if map.is_empty() {
        return 0.0;
    }
    let s = sorted(map);
    quantile_sorted(&s, 0.99) - quantile_sorted(&s, 0.01)
}

/// Peak-to-peak spread of a median-1 map, %.
pub fn spread_percent(map: &[f32]) -> f64 {
    100.0 * percentile_spread(map) as f64
}

fn region_mean(map: &[f32], w: usize, h: usize, fx0: f64, fy0: f64, fx1: f64, fy1: f64) -> f64 {
    let x0 = (fx0 * w as f64).floor() as usize;
    let x1 = ((fx1 * w as f64).floor() as usize).max(x0 + 1).min(w);
    let y0 = (fy0 * h as f64).floor() as usize;
    let y1 = ((fy1 * h as f64).floor() as usize).max(y0 + 1).min(h);
    let mut s = 0f64;
    for y in y0..y1 {
        for x in x0..x1 {
            s += map[y * w + x] as f64;
        }
    }
    s / ((x1 - x0) * (y1 - y0)) as f64
}

/// Edges and corners of a map, % relative to its centre.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Edges {
    pub left: f64,
    pub right: f64,
    pub top: f64,
    pub bottom: f64,
    pub tl: f64,
    pub tr: f64,
    pub bl: f64,
    pub br: f64,
}

impl Edges {
    /// Left minus right, %.
    pub fn left_right(&self) -> f64 {
        self.left - self.right
    }

    pub fn top_bottom(&self) -> f64 {
        self.top - self.bottom
    }

    /// The largest edge or corner deviation, with its name.
    pub fn worst(&self) -> (&'static str, f64) {
        [
            ("left", self.left),
            ("right", self.right),
            ("top", self.top),
            ("bottom", self.bottom),
            ("top left", self.tl),
            ("top right", self.tr),
            ("bottom left", self.bl),
            ("bottom right", self.br),
        ]
        .into_iter()
        .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
        .expect("non-empty")
    }

    /// "left +6.1%, right −6.4%": the edges and corners beyond `min` %.
    pub fn describe(&self, min: f64) -> String {
        let mut v: Vec<(&str, f64)> = vec![
            ("left", self.left),
            ("right", self.right),
            ("top", self.top),
            ("bottom", self.bottom),
            ("top left", self.tl),
            ("top right", self.tr),
            ("bottom left", self.bl),
            ("bottom right", self.br),
        ];
        v.retain(|e| e.1.abs() >= min);
        v.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
        v.truncate(3);
        v.iter().map(|(n, x)| format!("{n} {}", pct(*x))).collect::<Vec<_>>().join(", ")
    }
}

/// Signed percentage with a real minus sign: "+3.2%", "−1.0%".
pub fn pct(v: f64) -> String {
    if v < 0.0 {
        format!("−{:.1}%", -v)
    } else {
        format!("+{v:.1}%")
    }
}

pub fn edge_profile(map: &[f32], w: usize, h: usize) -> Edges {
    let c = region_mean(map, w, h, 0.45, 0.45, 0.55, 0.55);
    let p = |v: f64| 100.0 * (v / c - 1.0);
    let e = 0.06;
    Edges {
        left: p(region_mean(map, w, h, 0.0, 0.35, e, 0.65)),
        right: p(region_mean(map, w, h, 1.0 - e, 0.35, 1.0, 0.65)),
        top: p(region_mean(map, w, h, 0.35, 0.0, 0.65, e)),
        bottom: p(region_mean(map, w, h, 0.35, 1.0 - e, 0.65, 1.0)),
        tl: p(region_mean(map, w, h, 0.0, 0.0, e, e)),
        tr: p(region_mean(map, w, h, 1.0 - e, 0.0, 1.0, e)),
        bl: p(region_mean(map, w, h, 0.0, 1.0 - e, e, 1.0)),
        br: p(region_mean(map, w, h, 1.0 - e, 1.0 - e, 1.0, 1.0)),
    }
}

/// Flat stability for one set. `frames`: offset-subtracted maps, each
/// normalised to median 1. `use_`: false for frames to leave out of the
/// reference (saturated). Returns the reference and each frame's change, %.
pub fn flat_drift(frames: &[&[f32]], w: usize, h: usize, use_: &[bool]) -> (Vec<f32>, Vec<f64>) {
    let good: Vec<&[f32]> = frames.iter().zip(use_).filter(|(_, u)| **u).map(|(f, _)| *f).collect();
    let reference = median_stack(if good.is_empty() { frames } else { &good });
    let sigma = sigma_bins(DRIFT_SIGMA_PX, w);
    let drift = frames.iter().map(|f| spread_percent(&gaussian_blur(&ratio_map(f, &reference), w, h, sigma))).collect();
    (reference, drift)
}

/// The error a flat leaves in calibrated lights.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlatError {
    /// Smoothed error map, median 1 (w × h at `MAP_BIN`).
    #[serde(skip)]
    pub map: Vec<f32>,
    pub w: usize,
    pub h: usize,
    /// Fraction masked as target or stars.
    pub masked: f64,
    /// 1st–99th percentile spread, %.
    pub spread: f64,
    pub edges: Edges,
}

/// `illumination`: median stack of offset-subtracted, normalised lights.
/// `flat`: offset-subtracted flat, normalised.
pub fn flat_against_lights(illumination: &[f32], flat: &[f32], w: usize, h: usize) -> FlatError {
    let (map, masked) = smooth_masked(&ratio_map(illumination, flat), w, h, sigma_bins(ERROR_SIGMA_PX, w));
    let spread = spread_percent(&map);
    let edges = edge_profile(&map, w, h);
    FlatError { map, w, h, masked, spread, edges }
}

/// Spread (%) of a median-1 map once a straight tilt (a + bx + cy, least
/// squares) is taken out: what gradient removal can't fix. A sky gradient
/// left in the lights, and most of a cloudy flat's error, are tilts;
/// vignetting and dust aren't.
pub fn untilted_spread(map: &[f32], w: usize, h: usize) -> f64 {
    if map.len() != w * h || w < 2 || h < 2 {
        return spread_percent(map);
    }
    // Normal equations on centred coordinates (x and y are then orthogonal
    // to the constant over a full grid).
    let (cx, cy) = ((w - 1) as f64 / 2.0, (h - 1) as f64 / 2.0);
    let (mut sv, mut sxv, mut syv, mut sxx, mut syy) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for y in 0..h {
        for x in 0..w {
            let (dx, dy, v) = (x as f64 - cx, y as f64 - cy, map[y * w + x] as f64);
            sv += v;
            sxv += dx * v;
            syv += dy * v;
            sxx += dx * dx;
            syy += dy * dy;
        }
    }
    let n = (w * h) as f64;
    let (a, b, c) = (sv / n, sxv / sxx, syv / syy);
    let res: Vec<f32> = (0..w * h)
        .map(|i| {
            let (dx, dy) = ((i % w) as f64 - cx, (i / w) as f64 - cy);
            (map[i] as f64 - (b * dx + c * dy) - a + 1.0) as f32
        })
        .collect();
    spread_percent(&res)
}

/// Dust shadows on the ASI2600 at f/4 are 300–600 px across. A band-pass
/// from 64 to 256 px keeps them and drops vignetting (larger) and noise,
/// stars and target residue (smaller).
pub const DUST_LO_PX: f32 = 64.0;
pub const DUST_HI_PX: f32 = 256.0;

fn band_pass(a: &[f32], w: usize, h: usize) -> Vec<f32> {
    let lo = gaussian_blur(a, w, h, sigma_bins(DUST_LO_PX, w));
    let hi = gaussian_blur(a, w, h, sigma_bins(DUST_HI_PX, w));
    lo.iter().zip(&hi).map(|(l, h)| if *h > 0.0 { l / h - 1.0 } else { 0.0 }).collect()
}

/// A flat's dust fingerprint: its band-passed reference (median 1 in, about
/// zero mean out).
pub fn dust_fingerprint(flat: &[f32], w: usize, h: usize) -> Vec<f32> {
    band_pass(flat, w, h)
}

/// The same band-pass of the lights' illumination, with bright residue
/// (stars, nebula) clipped: dust only ever darkens.
pub fn dust_of_lights(illumination: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut v = band_pass(illumination, w, h);
    let mut t = v.clone();
    let (m, s) = median_sigma_mut(&mut t);
    v.iter_mut().for_each(|x| *x = x.clamp(m - 5.0 * s, m + 2.0 * s));
    v
}

/// Pearson correlation of two maps, leaving out a border as wide as the
/// largest blur (edge-extended smoothing biases it).
pub fn correlate(a: &[f32], b: &[f32], w: usize, h: usize) -> Option<f64> {
    if a.len() != b.len() || a.len() != w * h {
        return None;
    }
    let e = (sigma_bins(DUST_HI_PX, w).ceil() as usize).min(w / 4).min(h / 4);
    let (mut n, mut sa, mut sb, mut saa, mut sbb, mut sab) = (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
    for y in e..h - e {
        for x in e..w - e {
            let (p, q) = (a[y * w + x] as f64, b[y * w + x] as f64);
            n += 1.0;
            sa += p;
            sb += q;
            saa += p * p;
            sbb += q * q;
            sab += p * q;
        }
    }
    if n < 10.0 {
        return None;
    }
    let cov = sab / n - sa * sb / (n * n);
    let va = saa / n - sa * sa / (n * n);
    let vb = sbb / n - sb * sb / (n * n);
    (va > 0.0 && vb > 0.0).then(|| cov / (va * vb).sqrt())
}

/// The smoothed ratio map behind `compare_flats` and the stability check
/// (median 1), for heat maps.
pub fn ratio_smoothed(a: &[f32], b: &[f32], w: usize, h: usize) -> Vec<f32> {
    gaussian_blur(&ratio_map(a, b), w, h, sigma_bins(DRIFT_SIGMA_PX, w))
}

/// Two flats (normalised) compared at the stability scale.
pub fn compare_flats(a: &[f32], b: &[f32], w: usize, h: usize) -> (f64, Edges) {
    let r = gaussian_blur(&ratio_map(a, b), w, h, sigma_bins(DRIFT_SIGMA_PX, w));
    (spread_percent(&r), edge_profile(&r, w, h))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A map from f(x, y) on [0, 1]², with 0.1% noise.
    fn ramp(w: usize, h: usize, f: impl Fn(f32, f32) -> f32) -> Vec<f32> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEED: AtomicU64 = AtomicU64::new(1);
        let mut rng = crate::synth::Rng::new(SEED.fetch_add(1, Ordering::Relaxed));
        let mut v = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                v.push(f(x as f32 / (w - 1) as f32, y as f32 / (h - 1) as f32) * (1.0 + 0.001 * rng.gauss() as f32));
            }
        }
        v
    }

    #[test]
    fn map_roundtrip_and_bin() {
        let mut img = Image::new(70, 40);
        img.adu_scale = 65535.0;
        for (i, v) in img.data.iter_mut().enumerate() {
            *v = (i % 70) as f32 / 65535.0;
        }
        let m = Map::bin(&img, 32);
        assert_eq!((m.w, m.h), (2, 1));
        assert!((m.data[0] - 15.5).abs() < 1e-3 && (m.data[1] - 47.5).abs() < 1e-3, "{:?}", m.data);
        assert_eq!(Map::from_bytes(&m.to_bytes()), Some(m));
    }

    #[test]
    fn untilted_spread_keeps_vignetting_and_drops_a_tilt() {
        let (w, h) = (195, 130);
        let tilt = ramp(w, h, |x, y| 1.0 + 0.08 * (x - 0.5) - 0.03 * (y - 0.5));
        assert!(spread_percent(&tilt) > 7.0);
        assert!(untilted_spread(&tilt, w, h) < 1.0, "{}", untilted_spread(&tilt, w, h));
        let vig = ramp(w, h, |x, y| 1.0 - 0.3 * ((x - 0.5).powi(2) + (y - 0.5).powi(2)));
        let both = ramp(w, h, |x, y| (1.0 - 0.3 * ((x - 0.5).powi(2) + (y - 0.5).powi(2))) * (1.0 + 0.08 * (x - 0.5)));
        let (sv, sb) = (untilted_spread(&vig, w, h), untilted_spread(&both, w, h));
        assert!(sv > 10.0 && (sb - sv).abs() < 1.5, "{sv} {sb}");
    }

    #[test]
    fn vignetting_cancels_and_tilt_shows() {
        let (w, h) = (195, 130);
        let vig = |x: f32, y: f32| 1.0 - 0.3 * ((x - 0.5).powi(2) + (y - 0.5).powi(2));
        let flat = ramp(w, h, vig);
        // Lights see the same vignetting: no error.
        let e = flat_against_lights(&flat, &flat, w, h);
        assert!(e.spread < 0.2, "{}", e.spread);
        // A flat with a 12% left-right tilt (twilight gradient).
        let bad = ramp(w, h, |x, y| vig(x, y) * (1.06 - 0.12 * x));
        let e = flat_against_lights(&flat, &bad, w, h);
        assert!((e.edges.left_right() + 11.0).abs() < 1.5, "{:?}", e.edges);
        assert!(e.spread > 9.0, "{}", e.spread);
        // A target (bright blob) is masked, not reported as error.
        let lit = ramp(w, h, |x, y| vig(x, y) * (1.0 + 0.5 * (-((x - 0.3).powi(2) + (y - 0.6).powi(2)) / 0.002).exp()));
        let e = flat_against_lights(&lit, &flat, w, h);
        assert!(e.spread < 1.5 && e.masked > 0.0, "{} {}", e.spread, e.masked);
    }

    #[test]
    fn drift_finds_the_changed_flat() {
        let (w, h) = (100, 66);
        let a = ramp(w, h, |x, _| 1.0 - 0.1 * x);
        let b = ramp(w, h, |x, y| (1.0 - 0.1 * x) * (1.0 + 0.05 * y));
        let frames: Vec<&[f32]> = vec![&a, &a, &a, &b];
        let (_, d) = flat_drift(&frames, w, h, &[true; 4]);
        assert!(d[0] < 0.1 && d[3] > 3.5, "{d:?}");
        let (s, e) = compare_flats(&b, &a, w, h);
        assert!(s > 3.5 && e.top_bottom() < -3.0, "{s} {e:?}");
    }
}
