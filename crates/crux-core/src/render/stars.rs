//! Natural star colours for narrowband images: the native version of
//! James's StarXTerminator → SetiAstro NBtoRGB Stars → screen combine.
//!
//! The palette mixer boosts OIII and SII several times to match Ha, and
//! their star profiles differ, so mixed stars come out pink or magenta
//! (the mixer's README says to mix starless masters and add RGB stars
//! back). Here:
//!
//! 1. **Mask:** compact structures in the small-scale wavelet layers
//!    (B3 à trous, scales 1–8 px) of every channel, each a soft disc a
//!    little larger than its footprint. Elongated or large structures
//!    (nebula filaments) are left out.
//! 2. **Starless:** each linear channel with the discs filled from their
//!    surroundings (push-pull interpolation) plus matching noise.
//! 3. **Mix:** starless and original channels both go through the palette
//!    with the same coefficients (the original statistics, so nothing but
//!    the stars changes).
//! 4. **Star share:** what the stars added to each pixel of the finished
//!    image, `x = 1 − (1 − full) / (1 − starless)` (unscreened), as a
//!    share of its brightness.
//! 5. **Recolour** that share: the pixel keeps its brightness, and the star
//!    part takes the star's colour, one per star from its integrated
//!    linear light, NBtoRGB-style: R = Ha (½Ha + ½SII with SII),
//!    G = r·R + (1 − r)·B, B = OIII, OIII scaled so the median star is
//!    white (green from R rather than Ha, so it always lies between red
//!    and blue). Per-pixel colour gave cyan cores and orange rims (OIII stars
//!    are sharper than Ha's); r = ½ puts green halfway between red and
//!    blue, as a blackbody does (lower gives cyan stars). Doubtful stars
//!    (low SNR, flat like a nebula knot) and nebula under a star barely
//!    change, and disc edges, where the star adds nothing, can't show.
//!
//! **Stretched separately** (`StarSettings::separate`, the default; James's
//! PixInsight way): steps 3–5 change. The starless channels get the whole
//! palette (mix, stretch, finishing) from their own statistics, so the
//! nebula is stretched as nebula. The star layer is each channel's
//! `max(original − starless, 0)` above the layer's noise floor, coloured one
//! colour per star as above (doubtful stars take the palette's colour), and
//! stretched by its own curve (`Curve`: a star on dark sky as the mixer's
//! stretch would show it, with `size` setting how hard), then screened onto
//! the finished starless image: `1 − (1 − starless)(1 − stars)`. Stars over
//! bright nebula keep their brightness instead of being compressed with it.

use crate::image::Image;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum StarMode {
    /// Stars as the palette mixes them.
    Keep,
    /// Starless mix plus naturally coloured stars (narrowband with OIII).
    #[default]
    Natural,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct StarSettings {
    pub mode: StarMode,
    /// Share of Ha in the green of star colours (NBtoRGB's ratio). 0.5
    /// follows a blackbody; lower gives cyan stars.
    pub ha_to_green: f64,
    /// Star colour saturation (1 = as measured).
    pub saturation: f64,
    /// Mask size multiplier (1 = crux's default margin).
    pub grow: f64,
    /// Detection threshold in noise sigmas of the small-scale layers.
    pub sigma: f64,
    /// How the starless channels are made.
    pub engine: StarEngine,
    /// Stretch the starless image and the stars separately, then screen
    /// the stars on (false: one stretch for both, stars recoloured in place).
    pub separate: bool,
    /// Separate stretch: how hard the star layer is stretched (1 = a star on
    /// dark sky as the nebula stretch would show it; lower gives smaller,
    /// fainter stars).
    pub size: f64,
    /// Separate stretch: star layer brightness (1 = as stretched).
    pub brightness: f64,
    /// Separate stretch: light under this many noise sigmas of the star
    /// layer is left out (2: crux's default). Higher drops faint stars and
    /// halos; the bright stars barely change.
    pub faint_cut: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StarEngine {
    /// crux's push-pull fill over the star discs, with matching noise.
    Classic,
    /// crux's star remover network (`starnet`).
    Network,
}

/// The star remover when this build has crux's networks (James,
/// 2026-11-21), else the classic fill.
impl Default for StarEngine {
    fn default() -> Self {
        if cfg!(feature = "ml") {
            StarEngine::Network
        } else {
            StarEngine::Classic
        }
    }
}

/// The star layer's default floor in noise sigmas (`StarSettings::faint_cut`).
pub const STAR_FLOOR_SIGMAS: f64 = 2.0;

/// The star layer's default stretch (`StarSettings::size`).
pub const DEFAULT_SIZE: f64 = 0.6;

impl Default for StarSettings {
    fn default() -> Self {
        StarSettings { mode: StarMode::Natural, ha_to_green: 0.5, saturation: 1.0, grow: 1.0, sigma: 4.0, engine: StarEngine::default(), separate: true, size: DEFAULT_SIZE, brightness: 1.0, faint_cut: STAR_FLOOR_SIGMAS }
    }
}

impl StarSettings {
    /// The settings that change the starless channels (not just the render).
    pub fn mask_key(&self) -> String {
        format!("{:?}|{}|{}|{:?}", self.mode, self.grow, self.sigma, self.engine)
    }
}

/// A star's disc: centre and radius in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Disc {
    pub x: f32,
    pub y: f32,
    pub r: f32,
    /// Peak ÷ mean of its small-scale light: stars are peaked (2.5+ when
    /// bright), nebula knots flat.
    pub peaked: f32,
}

const K: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];

/// One B3 à trous smoothing step with holes of `step` pixels (mirrored
/// edges).
fn b3(src: &[f32], w: usize, h: usize, step: usize) -> Vec<f32> {
    let mirror = |i: isize, n: usize| -> usize {
        let n = n as isize;
        let mut i = i;
        if i < 0 {
            i = -i;
        }
        if i >= n {
            i = 2 * n - 2 - i;
        }
        i.clamp(0, n - 1) as usize
    };
    let s = step as isize;
    let mut tmp = vec![0f32; w * h];
    tmp.par_chunks_mut(w).enumerate().for_each(|(y, out)| {
        let r = &src[y * w..(y + 1) * w];
        for x in 0..w {
            let xi = x as isize;
            out[x] = (0..5).map(|k| K[k] * r[mirror(xi + (k as isize - 2) * s, w)]).sum();
        }
    });
    let mut out = vec![0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, o)| {
        let yi = y as isize;
        let rows: [usize; 5] = std::array::from_fn(|k| mirror(yi + (k as isize - 2) * s, h));
        for x in 0..w {
            o[x] = (0..5).map(|k| K[k] * tmp[rows[k] * w + x]).sum();
        }
    });
    out
}

/// Star discs in one linear image.
pub fn detect(img: &Image, s: &StarSettings) -> Vec<Disc> {
    let (w, h) = (img.width, img.height);
    if w < 16 || h < 16 {
        return Vec::new();
    }
    // Small scales: the image less its smoothing at 1, 2 and 4 px holes
    // (structures up to ~8 px across).
    let mut c = img.data.clone();
    for step in [1usize, 2, 4] {
        c = b3(&c, w, h, step);
    }
    let detail: Vec<f32> = img.data.par_iter().zip(c.par_iter()).map(|(v, s)| if *v == 0.0 { 0.0 } else { v - s }).collect();
    drop(c);
    // Noise of the detail layer: MAD of a sample (exact selections, in
    // parallel on big images).
    let mut smp: Vec<f32> = detail.iter().step_by(7).copied().filter(|v| *v != 0.0).collect();
    if smp.is_empty() {
        return Vec::new();
    }
    let m = smp.len() / 2;
    let med = crate::stats::kth_par(&smp, m);
    smp.par_iter_mut().for_each(|v| *v = (*v - med).abs());
    let sigma = (1.4826 * crate::stats::kth_par(&smp, m)).max(1e-9);
    let thr = med + s.sigma as f32 * sigma;
    // Connected components above the threshold (8-connected).
    let mut seen = vec![false; w * h];
    let mut discs = Vec::new();
    let mut stack = Vec::new();
    for start in 0..w * h {
        if seen[start] || detail[start] <= thr {
            continue;
        }
        seen[start] = true;
        stack.push(start);
        let (mut area, mut sx, mut sy, mut sw, mut peak) = (0usize, 0f64, 0f64, 0f64, 0f32);
        let (mut x0, mut x1, mut y0, mut y1) = (usize::MAX, 0usize, usize::MAX, 0usize);
        while let Some(i) = stack.pop() {
            let (x, y) = (i % w, i / w);
            let v = detail[i] - med;
            area += 1;
            sx += x as f64 * v as f64;
            sy += y as f64 * v as f64;
            sw += v as f64;
            peak = peak.max(v);
            x0 = x0.min(x);
            x1 = x1.max(x);
            y0 = y0.min(y);
            y1 = y1.max(y);
            if area > 20_000 {
                // A huge region: nebula, not a star; drain it.
                continue;
            }
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                    if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                        continue;
                    }
                    let j = ny as usize * w + nx as usize;
                    if !seen[j] && detail[j] > thr {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
        }
        let (bw, bh) = ((x1 - x0 + 1) as f32, (y1 - y0 + 1) as f32);
        let fill = area as f32 / (bw * bh);
        let aspect = bw.max(bh) / bw.min(bh);
        if area < 3 || area > 6_000 || fill < 0.3 || aspect > 2.5 || peak < 1.5 * s.sigma as f32 * sigma || sw <= 0.0 {
            continue;
        }
        let r0 = bw.max(bh) / 2.0;
        let mean = (sw / area as f64) as f32;
        discs.push(Disc { x: (sx / sw) as f32, y: (sy / sw) as f32, r: (r0 * 1.6 + 1.5) * s.grow as f32, peaked: if mean > 0.0 { peak / mean } else { 1.0 } });
    }
    discs
}

/// One disc per star: detections from several channels within half the
/// larger radius of each other merge (the larger disc, the more peaked).
pub fn dedupe(mut discs: Vec<Disc>) -> Vec<Disc> {
    discs.sort_by(|a, b| b.r.total_cmp(&a.r));
    let cell = 16.0f32;
    let mut grid: std::collections::HashMap<(i32, i32), Vec<usize>> = Default::default();
    let mut out: Vec<Disc> = Vec::new();
    for d in discs {
        let (cx, cy) = ((d.x / cell) as i32, (d.y / cell) as i32);
        let reach = ((d.r / cell).ceil() as i32).max(1);
        let mut dup = None;
        'look: for gy in cy - reach..=cy + reach {
            for gx in cx - reach..=cx + reach {
                if let Some(v) = grid.get(&(gx, gy)) {
                    for &k in v {
                        let o = &out[k];
                        let dist = ((o.x - d.x).powi(2) + (o.y - d.y).powi(2)).sqrt();
                        if dist < 0.5 * o.r.max(d.r) {
                            dup = Some(k);
                            break 'look;
                        }
                    }
                }
            }
        }
        match dup {
            Some(k) => out[k].peaked = out[k].peaked.max(d.peaked),
            None => {
                grid.entry((cx, cy)).or_default().push(out.len());
                out.push(d);
            }
        }
    }
    out
}

/// A star's integrated linear light per channel and how sure crux is of it.
#[derive(Debug, Clone)]
pub struct StarInfo {
    pub disc: Disc,
    pub flux: Vec<f32>,
    /// 0–1: colour confidence (SNR and peakedness).
    pub confidence: f32,
}

/// Integrated star light (image − starless) in each disc, per channel.
/// `h`: the channel whose noise sets the SNR.
pub fn measure(discs: &[Disc], with: &[&Image], without: &[&Image], h: usize, noise_h: f32) -> Vec<StarInfo> {
    let (w, hh) = (with[0].width, with[0].height);
    discs
        .par_iter()
        .map(|d| {
            let mut flux = vec![0f32; with.len()];
            let mut under = vec![0f32; with.len()];
            let mut top = vec![0f32; with.len()];
            let mut n = 0usize;
            let (xa, xb) = ((d.x - d.r).floor().max(0.0) as usize, ((d.x + d.r).ceil() as usize).min(w.saturating_sub(1)));
            let (ya, yb) = ((d.y - d.r).floor().max(0.0) as usize, ((d.y + d.r).ceil() as usize).min(hh.saturating_sub(1)));
            for y in ya..=yb {
                for x in xa..=xb {
                    if (x as f32 - d.x).powi(2) + (y as f32 - d.y).powi(2) > d.r * d.r {
                        continue;
                    }
                    let i = y * w + x;
                    n += 1;
                    for k in 0..with.len() {
                        let v = with[k].data[i] - without[k].data[i];
                        flux[k] += v;
                        top[k] = top[k].max(v);
                        under[k] += without[k].data[i];
                    }
                }
            }
            let snr = flux[h] / (noise_h * (n.max(1) as f32).sqrt()).max(1e-12);
            // Peak contrast against what's under it: a star's core
            // outshines its background (≫ 1 on dark sky); a nebula knot is
            // a small bump on bright nebula.
            let nn = n.max(1) as f32;
            let contrast = top.iter().zip(&under).map(|(t, u)| t / (u / nn).max(1e-9)).fold(0f32, f32::max);
            let confidence = (snr / 15.0).clamp(0.0, 1.0) * ((d.peaked - 1.15) / 0.5).clamp(0.0, 1.0) * ((contrast - 0.1) / 0.3).clamp(0.0, 1.0);
            if std::env::var("CRUX_STAR_DEBUG").is_ok() {
                eprintln!("STAR {:.1} {:.1} r {:.1} snr {:.1} peaked {:.2} contrast {:.3} conf {:.2}", d.x, d.y, d.r, snr, d.peaked, contrast, confidence);
            }
            StarInfo { disc: *d, flux, confidence }
        })
        .collect()
}

/// Per-pixel star light and confidence painted from the stars, for a
/// render at `scale` (render pixels per full pixel) whose origin is at
/// full-resolution (x0, y0).
#[derive(Debug, Clone)]
pub struct Paint {
    pub flux: Vec<Vec<f32>>,
    pub confidence: Vec<f32>,
}

/// Rows per band when painting discs in parallel.
const BAND: usize = 64;

/// For each band of `BAND` rows, the items (in their original order) whose
/// rows `ya..=yb` touch it. Painting band by band in that order gives
/// every pixel exactly what painting item by item gives.
fn bands(spans: &[Option<(usize, usize)>], height: usize) -> Vec<Vec<u32>> {
    let mut b = vec![Vec::new(); height.div_ceil(BAND).max(1)];
    for (i, s) in spans.iter().enumerate() {
        if let Some((ya, yb)) = s {
            for band in b.iter_mut().take(yb / BAND + 1).skip(ya / BAND) {
                band.push(i as u32);
            }
        }
    }
    b
}

pub fn paint(stars: &[StarInfo], n: usize, width: usize, height: usize, scale: f32, x0: f32, y0: f32) -> Paint {
    let mut flux = vec![vec![0f32; width * height]; n];
    let mut conf = vec![0f32; width * height];
    if width == 0 || height == 0 {
        return Paint { flux, confidence: conf };
    }
    struct Spot {
        cx: f32,
        cy: f32,
        r: f32,
        key: f32,
        xa: usize,
        xb: usize,
    }
    let spots: Vec<Option<(Spot, usize, usize)>> = stars
        .iter()
        .map(|s| {
            let (cx, cy, r) = ((s.disc.x - x0) * scale, (s.disc.y - y0) * scale, (s.disc.r * scale).max(0.75) + 1.5 * scale.min(1.0));
            if cx + r < 0.0 || cy + r < 0.0 || cx - r >= width as f32 || cy - r >= height as f32 {
                return None;
            }
            let key = s.flux.iter().copied().fold(0f32, f32::max);
            let (xa, xb) = ((cx - r).floor().max(0.0) as usize, ((cx + r).ceil() as usize).min(width - 1));
            let (ya, yb) = ((cy - r).floor().max(0.0) as usize, ((cy + r).ceil() as usize).min(height - 1));
            Some((Spot { cx, cy, r, key, xa, xb }, ya, yb))
        })
        .collect();
    let lists = bands(&spots.iter().map(|s| s.as_ref().map(|(_, a, b)| (*a, *b))).collect::<Vec<_>>(), height);
    // Every plane cut into the same bands.
    let mut flux_bands: Vec<Vec<&mut [f32]>> = (0..lists.len()).map(|_| Vec::with_capacity(n)).collect();
    for plane in flux.iter_mut() {
        for (k, c) in plane.chunks_mut(BAND * width).enumerate() {
            flux_bands[k].push(c);
        }
    }
    conf.par_chunks_mut(BAND * width).zip(flux_bands.into_par_iter()).zip(lists.par_iter()).enumerate().for_each(|(band, ((conf, mut fl), list))| {
        let row0 = band * BAND;
        let rows = conf.len() / width;
        let mut best = vec![0f32; conf.len()];
        for &si in list {
            let Some((sp, ya, yb)) = &spots[si as usize] else { continue };
            let s = &stars[si as usize];
            for y in (*ya).max(row0)..=(*yb).min(row0 + rows - 1) {
                for x in sp.xa..=sp.xb {
                    if (x as f32 + 0.5 - sp.cx).powi(2) + (y as f32 + 0.5 - sp.cy).powi(2) > sp.r * sp.r {
                        continue;
                    }
                    let i = (y - row0) * width + x;
                    // Overlaps: the brighter star wins.
                    if sp.key > best[i] {
                        best[i] = sp.key;
                        for (k, f) in fl.iter_mut().enumerate() {
                            f[i] = s.flux[k];
                        }
                        conf[i] = s.confidence;
                    }
                }
            }
        }
    });
    Paint { flux, confidence: conf }
}

/// A soft mask (1 inside a disc, fading over 1.5 px) from discs.
pub fn mask(discs: &[Disc], w: usize, h: usize) -> Vec<f32> {
    let mut m = vec![0f32; w * h];
    if w == 0 || h == 0 {
        return m;
    }
    let spans: Vec<Option<(usize, usize)>> = discs
        .iter()
        .map(|d| {
            let ext = d.r + 1.5;
            Some(((d.y - ext).floor().max(0.0) as usize, ((d.y + ext).ceil() as usize).min(h.saturating_sub(1))))
        })
        .collect();
    let lists = bands(&spans, h);
    m.par_chunks_mut(BAND * w).zip(lists.par_iter()).enumerate().for_each(|(band, (m, list))| {
        let row0 = band * BAND;
        let rows = m.len() / w;
        for &di in list {
            let d = &discs[di as usize];
            let ext = d.r + 1.5;
            let (xa, xb) = ((d.x - ext).floor().max(0.0) as usize, ((d.x + ext).ceil() as usize).min(w.saturating_sub(1)));
            let (ya, yb) = spans[di as usize].unwrap();
            if yb < row0 || ya >= row0 + rows {
                continue;
            }
            for y in ya.max(row0)..=yb.min(row0 + rows - 1) {
                for x in xa..=xb {
                    let dist = ((x as f32 - d.x).powi(2) + (y as f32 - d.y).powi(2)).sqrt();
                    let v = ((ext - dist) / 1.5).clamp(0.0, 1.0);
                    let i = (y - row0) * w + x;
                    if v > m[i] {
                        m[i] = v;
                    }
                }
            }
        }
    });
    m
}

/// Stars whose leftover glow is looked for, brightest first.
const GLOW_STARS: usize = 400;
/// A star's glow is looked for out to this many disc radii.
const GLOW_REACH: f32 = 8.0;
/// A glow brighter than this share of its star's own light is nebula.
const GLOW_MAX_RATIO: f32 = 0.02;

/// Take the round glows the starless image keeps around bright stars out of
/// it (they go to the star layer instead, with the star's colour). Per star
/// the glow is the starless image's median at each radius over the median
/// of a ring just outside, kept only while it falls outwards: the round part
/// alone, so nebula crossing a star stays. Narrowband glows differ by
/// filter (OIII and SII glows are wider than Ha's), so in a palette they
/// came out coloured (SHO: magenta halos).
/// One star's glow in one channel's starless image.
struct GlowFit {
    /// Index into the discs.
    i: usize,
    /// More glow than a star's own (nebula the star sits in).
    nebula: bool,
    /// Too faint to bother with.
    weak: bool,
    /// The glow per radius, falling outwards, zero at the reach.
    ex: Vec<f32>,
    bg: f32,
    reach: f32,
}

/// Measure the glows of the brightest stars (see [`remove_glows`]).
fn glow_fits(sl: &Image, orig: &Image, discs: &[Disc], noise: f32) -> Vec<GlowFit> {
    let (w, h) = (sl.width, sl.height);
    if (orig.width, orig.height) != (w, h) || noise <= 0.0 {
        return Vec::new();
    }
    let at = |img: &Image, x: f32, y: f32| img.data[(y as usize).min(h - 1) * w + (x as usize).min(w - 1)];
    let mut bright: Vec<(f32, usize)> = discs.iter().enumerate().map(|(i, d)| (at(orig, d.x, d.y) - at(sl, d.x, d.y), i)).filter(|(c, _)| *c > 50.0 * noise).collect();
    bright.sort_by(|a, b| b.0.total_cmp(&a.0));
    bright.truncate(GLOW_STARS);
    let mut fits = Vec::with_capacity(bright.len());
    for (c, i) in bright {
        let d = &discs[i];
        let reach = (GLOW_REACH * d.r).clamp(16.0, 160.0);
        let outer = reach * 1.3;
        let n = reach.ceil() as usize;
        let mut rings: Vec<Vec<f32>> = vec![Vec::new(); n + 1];
        let mut out_ring: Vec<f32> = Vec::new();
        let (xa, xb) = ((d.x - outer).floor().max(0.0) as usize, ((d.x + outer).ceil() as usize).min(w - 1));
        let (ya, yb) = ((d.y - outer).floor().max(0.0) as usize, ((d.y + outer).ceil() as usize).min(h - 1));
        for y in ya..=yb {
            for x in xa..=xb {
                let r = ((x as f32 + 0.5 - d.x).powi(2) + (y as f32 + 0.5 - d.y).powi(2)).sqrt();
                let v = sl.data[y * w + x];
                if r < reach {
                    rings[r as usize].push(v);
                } else if r < outer {
                    out_ring.push(v);
                }
            }
        }
        if out_ring.len() < 20 {
            continue;
        }
        let bg = crate::stats::median_mut(&mut out_ring);
        // Excess per radius (zero at the reach).
        let e: Vec<f32> = (0..=n).map(|k| if k < n && rings[k].len() >= 3 { (crate::stats::median_mut(&mut rings[k]) - bg).max(0.0) } else { 0.0 }).collect();
        // From the brightest radius near the centre outwards it may only
        // fall: a nebula bump further out is left alone.
        let k0 = (0..((d.r.ceil() as usize).max(3)).min(n)).max_by(|a, b| e[*a].total_cmp(&e[*b])).unwrap_or(0);
        // A star's own glow is faint beside the star (at most 0.6% on
        // RCW 14); far more is nebula the star sits in (M 42's Trapezium:
        // 50×; NGC 6164: 5%), which stays. Judged just outside the disc:
        // nebula crossing the star fills the innermost rings.
        let edge = e[(d.r.ceil() as usize).min(n)];
        if std::env::var_os("CRUX_GLOW_DEBUG").is_some() {
            eprintln!("glow {:.0},{:.0} r {:.1} star {:.5} glow {:.5} ratio {:.3} reach {:.0}", d.x, d.y, d.r, c, edge, edge / c.max(1e-9), reach);
        }
        let mut ex = vec![0f32; n + 1];
        let mut run = e[k0];
        for k in 0..=n {
            if k > k0 {
                run = run.min(e[k]);
            }
            ex[k] = run * (1.0 - (k as f32 / reach).min(1.0).powi(2));
        }
        fits.push(GlowFit { i, nebula: edge > GLOW_MAX_RATIO * c, weak: e[k0] < 2.0 * noise, ex, bg, reach });
    }
    fits
}

/// Per disc: does this channel see more glow than a star's own (nebula
/// the star sits in)? Combine the channels' answers (any) before
/// [`remove_glows`], so every channel keeps or loses a star's glow
/// together: a saturated core makes the glow look large beside it in that
/// channel alone, and taking the glow from only some channels coloured the
/// halo (cyan on a DSLR's saturated stars).
pub fn glow_nebula(sl: &Image, orig: &Image, discs: &[Disc], noise: f32) -> Vec<bool> {
    let mut v = vec![false; discs.len()];
    for f in glow_fits(sl, orig, discs, noise) {
        v[f.i] = f.nebula;
    }
    v
}

/// Take the round glows the starless image keeps around bright stars out of
/// it (they go to the star layer instead, with the star's colour). Per star
/// the glow is the starless image's median at each radius over the median
/// of a ring just outside, kept only while it falls outwards: the round part
/// alone, so nebula crossing a star stays. Narrowband glows differ by
/// filter (OIII and SII glows are wider than Ha's), so in a palette they
/// came out coloured (SHO: magenta halos). `keep`: per disc, leave its glow
/// (from [`glow_nebula`] over every channel).
pub fn remove_glows(sl: &mut Image, orig: &Image, discs: &[Disc], noise: f32, keep: &[bool]) {
    let w = sl.width;
    let h = sl.height;
    for f in glow_fits(sl, orig, discs, noise) {
        if f.nebula || f.weak || keep.get(f.i).copied().unwrap_or(false) {
            continue;
        }
        let d = &discs[f.i];
        let n = f.ex.len() - 1;
        let (xa, xb) = ((d.x - f.reach).floor().max(0.0) as usize, ((d.x + f.reach).ceil() as usize).min(w - 1));
        let (ya, yb) = ((d.y - f.reach).floor().max(0.0) as usize, ((d.y + f.reach).ceil() as usize).min(h - 1));
        for y in ya..=yb {
            for x in xa..=xb {
                let r = ((x as f32 + 0.5 - d.x).powi(2) + (y as f32 + 0.5 - d.y).powi(2)).sqrt();
                if r < f.reach {
                    let k = r as usize;
                    let t = r - k as f32;
                    let e = f.ex[k.min(n)] * (1.0 - t) + f.ex[(k + 1).min(n)] * t;
                    let i = y * w + x;
                    sl.data[i] = (sl.data[i] - e).max(f.bg.min(sl.data[i]));
                }
            }
        }
    }
}

/// Fill where `weight` < 1 from the surroundings: push-pull over a 2×
/// pyramid (weighted means down, bilinear blending up).
pub fn fill(data: &[f32], weight: &[f32], w: usize, h: usize) -> Vec<f32> {
    if w <= 1 && h <= 1 {
        return data.to_vec();
    }
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let mut cv = vec![0f32; cw * ch];
    let mut cwt = vec![0f32; cw * ch];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let j = (y / 2) * cw + x / 2;
            cv[j] += data[i] * weight[i];
            cwt[j] += weight[i];
        }
    }
    for j in 0..cw * ch {
        if cwt[j] > 0.0 {
            cv[j] /= cwt[j];
        }
        cwt[j] = cwt[j].min(1.0);
    }
    let coarse = fill(&cv, &cwt, cw, ch);
    let mut out = data.to_vec();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let wt = weight[i].clamp(0.0, 1.0);
            if wt >= 1.0 {
                continue;
            }
            // Bilinear sample of the coarse level at this pixel's centre.
            let (fx, fy) = (((x as f32 + 0.5) / 2.0 - 0.5).clamp(0.0, (cw - 1) as f32), ((y as f32 + 0.5) / 2.0 - 0.5).clamp(0.0, (ch - 1) as f32));
            let (ix, iy) = (fx.floor() as usize, fy.floor() as usize);
            let (ax, ay) = (fx - ix as f32, fy - iy as f32);
            let (jx, jy) = ((ix + 1).min(cw - 1), (iy + 1).min(ch - 1));
            let c = coarse[iy * cw + ix] * (1.0 - ax) * (1.0 - ay) + coarse[iy * cw + jx] * ax * (1.0 - ay) + coarse[jy * cw + ix] * (1.0 - ax) * ay + coarse[jy * cw + jx] * ax * ay;
            out[i] = wt * data[i] + (1.0 - wt) * c;
        }
    }
    out
}

/// A repeatable Gaussian-ish value for pixel `i` (sum of four uniforms).
fn noise_at(i: usize, salt: u64) -> f32 {
    let mut s = (i as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ salt;
    let mut sum = 0f32;
    for _ in 0..4 {
        s ^= s >> 33;
        s = s.wrapping_mul(0xff51afd7ed558ccd);
        s ^= s >> 33;
        sum += (s >> 40) as f32 / (1u64 << 24) as f32;
    }
    // Sum of 4 uniforms: mean 2, variance 1/3.
    (sum - 2.0) * 3f32.sqrt()
}

/// The image with the masked stars removed. `noise`: the image's noise σ,
/// added inside the mask so the fill doesn't look smoother than the sky.
pub fn starless(img: &Image, m: &[f32], noise: f32, salt: u64) -> Image {
    let (w, h) = (img.width, img.height);
    // Registration borders (zero) don't feed the fill.
    let weight: Vec<f32> = img.data.iter().zip(m).map(|(v, m)| if *v == 0.0 { 0.0 } else { 1.0 - m }).collect();
    let f = fill(&img.data, &weight, w, h);
    let data: Vec<f32> = (0..w * h)
        .into_par_iter()
        .map(|i| {
            let v = img.data[i];
            if v == 0.0 || m[i] <= 0.0 {
                v
            } else {
                v * (1.0 - m[i]) + (f[i] + noise * noise_at(i, salt)) * m[i]
            }
        })
        .collect();
    let mut out = Image::from_data(w, h, data);
    out.adu_scale = img.adu_scale;
    out
}

/// Star-layer colour constants for one render.
#[derive(Debug, Clone)]
pub struct Colour {
    /// Channel indices of H, O and (optionally) S.
    pub h: usize,
    pub o: usize,
    pub s: Option<usize>,
    /// Gains so the median star is white (H = 1).
    pub gain_o: f32,
    pub gain_s: f32,
    pub ha_to_green: f32,
    pub saturation: f32,
    /// Broadband: the R, G and B channels and their median-star-white
    /// gains; the star's colour is its own light in each (H, O and S
    /// above are unused).
    pub rgb: Option<([usize; 3], [f32; 3])>,
}

/// Combine one pixel: `full` and `bare` are the finished renders with and
/// without stars, `lin` the star's integrated light per channel, `conf`
/// how sure crux is of the star (0 keeps the mixed pixel).
#[inline]
pub fn combine(full: [f32; 3], bare: [f32; 3], lin: &dyn Fn(usize) -> f32, conf: f32, c: &Colour) -> [f32; 3] {
    let yf = (full[0] + full[1] + full[2]) / 3.0;
    let yb = (bare[0] + bare[1] + bare[2]) / 3.0;
    let x = if yb < 0.9999 { (1.0 - (1.0 - yf) / (1.0 - yb)).clamp(0.0, 1.0) } else { 0.0 };
    // The stars' share of this pixel's brightness.
    // Over bright nebula stars barely show and knots pass for stars, so
    // recolouring fades out as the starless picture goes from 0.35 to 0.7.
    let dark = ((0.7 - yb) / 0.35).clamp(0.0, 1.0);
    let a = if yf > 1e-6 { (x / yf).clamp(0.0, 1.0) * conf.clamp(0.0, 1.0) * dark } else { 0.0 };
    if a <= 0.0 {
        return full;
    }
    // Same brightness; the star part takes the star's colour.
    let col = unclipped(colour_of(lin, c), yf);
    std::array::from_fn(|k| ((1.0 - a) * full[k] + a * yf * col[k]).clamp(0.0, 1.0))
}

/// Linear R, G, B of star light, NBtoRGB-style: R = Ha (½Ha + ½SII with
/// SII), B = OIII, green between them (from red, not Ha: with weak SII in
/// red, green from Ha would outshine both and give green "stars").
#[inline]
pub fn rgb_of(lin: &dyn Fn(usize) -> f32, c: &Colour) -> [f32; 3] {
    if let Some((ch, g)) = c.rgb {
        return std::array::from_fn(|k| lin(ch[k]).max(0.0) * g[k]);
    }
    let h = lin(c.h).max(0.0);
    let o = lin(c.o).max(0.0) * c.gain_o;
    let r = match c.s {
        Some(si) => 0.5 * h + 0.5 * lin(si).max(0.0) * c.gain_s,
        None => h,
    };
    [r, c.ha_to_green * r + (1.0 - c.ha_to_green) * o, o]
}

/// A star's colour with mean 1, saturation applied (white when it has no
/// light).
#[inline]
pub fn colour_of(lin: &dyn Fn(usize) -> f32, c: &Colour) -> [f32; 3] {
    let v = rgb_of(lin, c);
    let mean = (v[0] + v[1] + v[2]) / 3.0;
    if mean > 0.0 {
        std::array::from_fn(|k| (1.0 + c.saturation * (v[k] / mean - 1.0)).max(0.0))
    } else {
        [1.0; 3]
    }
}

/// Desaturate a mean-1 colour just enough that `y · colour` doesn't clip
/// (clipping one channel swings the hue: a blue star's core turned cyan).
/// Bright cores go white.
#[inline]
pub fn unclipped(mut col: [f32; 3], y: f32) -> [f32; 3] {
    let top = col[0].max(col[1]).max(col[2]);
    if top > 1.0 && y * top > 1.0 {
        let s = ((1.0 / y - 1.0) / (top - 1.0)).clamp(0.0, 1.0);
        for v in col.iter_mut() {
            *v = 1.0 + s * (*v - 1.0);
        }
    }
    col
}

/// The star layer's stretch when stretched separately: a star of linear
/// light `I` on dark sky, as the mixer's stretch shows it above the sky.
/// The sky sits `clip` noise sigmas above the black point (`x0`) and goes
/// to `b` (the background level × `size`), so a smaller size stretches
/// less: smaller, fainter stars.
#[derive(Debug, Clone, Copy)]
pub struct Curve {
    pub x0: f64,
    pub m: f64,
    pub b: f64,
    pub brightness: f32,
}

impl Curve {
    /// `noise`: the reference channel's noise σ (linear, full resolution).
    pub fn new(noise: f64, clip: f64, background: f64, size: f64, brightness: f64) -> Curve {
        let x0 = (clip.max(0.5) * noise.max(1.0e-7)).min(0.5);
        let b = (background * size).clamp(0.005, 0.45);
        Curve { x0, m: super::palette::mtf(b, x0), b, brightness: brightness.max(0.0) as f32 }
    }

    #[inline]
    pub fn apply(&self, light: f32) -> f32 {
        if light <= 0.0 {
            return 0.0;
        }
        let y = (super::palette::mtf(self.m, (self.x0 + light as f64).min(1.0)) - self.b) / (1.0 - self.b);
        (y.max(0.0) as f32 * self.brightness).min(1.0)
    }
}

/// How much of the palette's colour light outside measured stars keeps.
const HUE_KEEP: f32 = 0.25;

/// One pixel of the separate stretch. `base`: the finished starless pixel;
/// `light`: the star layer's linear light per channel (above its floor);
/// `star`: the painted star's light, with `conf` its confidence (0 where no
/// measured star covers the pixel); `hue`: the palette's colour of the
/// original pixel, for light no measured star explains (doubtful stars,
/// nebula knots the starless lost).
#[inline]
pub fn screen(base: [f32; 3], light: &dyn Fn(usize) -> f32, star: &dyn Fn(usize) -> f32, conf: f32, hue: [f32; 3], c: &Colour, curve: &Curve) -> [f32; 3] {
    let v = rgb_of(light, c);
    let y = curve.apply((v[0] + v[1] + v[2]) / 3.0);
    if y <= 0.0 {
        return base;
    }
    let conf = conf.clamp(0.0, 1.0);
    let sc = if conf > 0.0 { colour_of(star, c) } else { [1.0; 3] };
    let hm = (hue[0] + hue[1] + hue[2]) / 3.0;
    // Mostly neutral: at a star's halo the palette's colour is the mix's
    // (SHO puts Ha in green, scaled down, so halos came out magenta).
    let hc: [f32; 3] = if hm > 1.0e-4 { std::array::from_fn(|k| 1.0 + HUE_KEEP * (hue[k] / hm - 1.0)) } else { [1.0; 3] };
    let col = unclipped(std::array::from_fn(|k| conf * sc[k] + (1.0 - conf) * hc[k]), y);
    std::array::from_fn(|k| {
        let s = (y * col[k]).clamp(0.0, 1.0);
        (1.0 - (1.0 - base[k]) * (1.0 - s)).clamp(0.0, 1.0)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn takes_a_round_glow_out_of_the_starless_image() {
        let (w, h) = (300usize, 300usize);
        let mut rng = 12345u64;
        let mut noise = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 0.002
        };
        let (cx, cy) = (150.5f32, 150.5f32);
        let mut sl = Image::new(w, h);
        let mut orig = Image::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let r2 = (x as f32 + 0.5 - cx).powi(2) + (y as f32 + 0.5 - cy).powi(2);
                // Sky, a nebula band crossing the star, and the glow.
                let band = if (y as f32 - 150.0).abs() < 4.0 { 0.02 } else { 0.0 };
                let v = 0.1 + band + 0.01 * (-r2 / (2.0 * 12.0f32.powi(2))).exp() + noise();
                sl.data[y * w + x] = v;
                orig.data[y * w + x] = v + 0.8 * (-r2 / (2.0 * 2.0f32.powi(2))).exp();
            }
        }
        let disc = Disc { x: cx, y: cy, r: 6.0, peaked: 3.0 };
        remove_glows(&mut sl, &orig, &[disc], 0.0006, &[false]);
        // The glow (1.25% of the star) goes, 4 px from the star, off the band …
        assert!((sl.get(154, 135) - 0.1).abs() < 0.004, "glow left: {}", sl.get(154, 135));
        // … the band crossing it stays, and so does the far sky.
        assert!(sl.get(130, 150) - 0.1 > 0.015, "band lost: {}", sl.get(130, 150));
        assert!((sl.get(10, 10) - 0.1).abs() < 0.002);
    }

    #[test]
    fn removes_stars_and_keeps_the_sky() {
        let p = crate::synth::Params { width: 600, height: 400, n_stars: 150, noise_adu: 10.0, ..Default::default() };
        let (img, truth) = crate::synth::render(&p);
        let discs = detect(&img, &StarSettings::default());
        // Most stars found; nothing like the whole frame.
        let found = truth.iter().filter(|t| t.peak_adu > 200.0).filter(|t| discs.iter().any(|d| ((d.x as f64 - t.x).powi(2) + (d.y as f64 - t.y).powi(2)).sqrt() < 3.0)).count();
        let bright = truth.iter().filter(|t| t.peak_adu > 200.0).count();
        assert!(found as f64 >= 0.9 * bright as f64, "found {found} of {bright}");
        let m = mask(&discs, img.width, img.height);
        let cover = m.iter().filter(|v| **v > 0.5).count() as f64 / m.len() as f64;
        assert!(cover < 0.25, "mask covers {cover}");
        let noise = crate::process::combine::noise_of(&img);
        let sl = starless(&img, &m, noise, 1);
        // At the brightest star, the starless value is back near the sky.
        let sky = (p.pedestal_adu + p.sky_adu) as f32 / 65535.0;
        let top = truth.iter().max_by(|a, b| a.peak_adu.total_cmp(&b.peak_adu)).unwrap();
        let v = sl.get(top.x as usize, top.y as usize);
        assert!((v - sky).abs() < 6.0 * noise, "starless {v} vs sky {sky} (noise {noise})");
        assert!(img.get(top.x as usize, top.y as usize) - sky > 100.0 * noise);
    }

    #[test]
    fn natural_colour_is_white_for_a_median_star_and_neutral_off_stars() {
        let c = Colour { h: 0, o: 1, s: None, gain_o: 2.0, gain_s: 1.0, ha_to_green: 0.3, saturation: 1.0, rgb: None };
        // A star whose OIII is half its Ha (gain 2 makes them equal): white.
        let lin = |i: usize| if i == 0 { 0.01 } else { 0.005 };
        let out = combine([0.8, 0.5, 0.7], [0.0, 0.0, 0.0], &lin, 1.0, &c);
        assert!((out[0] - out[1]).abs() < 1e-5 && (out[1] - out[2]).abs() < 1e-5, "{out:?}");
        // Brightness kept.
        assert!(((out[0] + out[1] + out[2]) / 3.0 - (0.8 + 0.5 + 0.7) / 3.0).abs() < 1e-5);
        // No star light added: the starless pixel unchanged.
        let none = combine([0.2, 0.2, 0.2], [0.2, 0.2, 0.2], &|_| 0.0, 1.0, &c);
        assert_eq!(none, [0.2, 0.2, 0.2]);
        // A doubtful star keeps the mixed pixel.
        assert_eq!(combine([0.8, 0.5, 0.7], [0.2, 0.2, 0.2], &lin, 0.0, &c), [0.8, 0.5, 0.7]);
    }

    #[test]
    fn separate_stars_screen_onto_the_starless_image() {
        let c = Colour { h: 0, o: 1, s: None, gain_o: 2.0, gain_s: 1.0, ha_to_green: 0.5, saturation: 1.0, rgb: None };
        let curve = Curve::new(1.0e-3, 2.8, 0.15, 1.0, 1.0);
        // No star light: the starless pixel unchanged.
        let base = [0.3, 0.2, 0.25];
        assert_eq!(screen(base, &|_| 0.0, &|_| 0.0, 0.0, [0.5; 3], &c, &curve), base);
        // The curve rises with light, from 0, and never passes 1.
        assert_eq!(curve.apply(0.0), 0.0);
        let (a, b, d) = (curve.apply(1.0e-3), curve.apply(1.0e-2), curve.apply(0.5));
        assert!(0.0 < a && a < b && b < d && d <= 1.0, "{a} {b} {d}");
        // A smaller size gives fainter stars.
        assert!(Curve::new(1.0e-3, 2.8, 0.15, 0.5, 1.0).apply(1.0e-2) < b);
        // A median star (OIII half its Ha) comes out white and screened.
        let lin = |i: usize| if i == 0 { 0.01 } else { 0.005 };
        let out = screen(base, &lin, &lin, 1.0, [1.0, 0.0, 0.0], &c, &curve);
        let y = curve.apply(0.01);
        for k in 0..3 {
            let want = 1.0 - (1.0 - base[k]) * (1.0 - y);
            assert!((out[k] - want).abs() < 1e-5, "{k}: {} vs {want}", out[k]);
        }
        // Light no measured star explains takes the palette's hue.
        let red = screen([0.0; 3], &lin, &|_| 0.0, 0.0, [0.6, 0.2, 0.1], &c, &curve);
        assert!(red[0] > red[1] && red[1] > red[2], "{red:?}");
    }
}
