//! Light calibration, as PixInsight's ImageCalibration does it in WBPP:
//!
//! calibrated = (light − dark) ÷ flat × mean(flat), then hot pixels (found
//! in the master dark) replaced from their neighbours, then an output
//! pedestal if too many pixels would fall below zero, then [0, 1].
//!
//! When the dark's exposure differs from the light's, the dark current is
//! scaled: light − bias − (dark − bias) × t_light ÷ t_dark (needs a bias).

use crate::image::Image;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;

/// A master frame in memory.
#[derive(Debug, Clone)]
pub struct Master {
    pub path: PathBuf,
    pub label: String,
    pub image: Arc<Image>,
    pub exptime: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct CalParams {
    /// Replace pixels this many sigmas above the master dark's median
    /// (None: no hot-pixel correction). WBPP: 10.
    pub hot_sigma: Option<f32>,
    /// Neighbourhood radius for the replacement (1 = 3×3).
    pub hot_radius: usize,
    /// Add a pedestal when more than this fraction of pixels would be
    /// below zero (None: never). WBPP's "auto": 0.0001.
    pub pedestal_limit: Option<f64>,
    /// Flat values below this (normalised to the flat's mean) are treated
    /// as this, so dead or vignetted corners don't explode.
    pub min_flat: f32,
}

impl Default for CalParams {
    fn default() -> Self {
        CalParams { hot_sigma: Some(10.0), hot_radius: 1, pedestal_limit: Some(0.0001), min_flat: 0.005 }
    }
}

/// What one calibration did.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct CalStats {
    /// Added pedestal, normalised units.
    pub pedestal: f32,
    pub hot_fixed: usize,
    /// Pixels clipped at 0 and 1.
    pub clipped_low: usize,
    pub clipped_high: usize,
}

/// Masters and derived constants for one light family.
#[derive(Debug, Clone)]
pub struct Calibrator {
    pub bias: Option<Master>,
    pub dark: Option<Master>,
    pub flat: Option<Master>,
    /// Dark current scale (t_light ÷ t_dark), 1 without scaling.
    pub dark_scale: f32,
    /// True when the dark is scaled (bias subtracted separately).
    pub scaled: bool,
    /// Mean of the flat (PixInsight's flat scaling factor).
    pub flat_mean: f32,
    /// Hot pixels (indices), sorted.
    pub hot: Arc<Vec<u32>>,
    pub params: CalParams,
    /// Plain-English notes: what was used and assumed.
    pub notes: Vec<String>,
    /// The lights are raw one-shot-colour frames: hot pixels take the
    /// median of their same-colour neighbours (2 px apart).
    pub cfa: bool,
}

/// The flat scaling factor: mean of the flat with the lowest and highest
/// 5% of pixels left out. Matches ImageCalibration's `f` (flat scale
/// clipping factor 0.05) to six decimals on James's masters.
pub fn flat_mean(f: &Image) -> f32 {
    let mut v: Vec<f32> = f.data.iter().copied().filter(|x| x.is_finite()).collect();
    let n = v.len();
    if n == 0 {
        return 1.0;
    }
    let k = (n as f64 * 0.05) as usize;
    if k == 0 || 2 * k >= n {
        return (v.iter().map(|x| *x as f64).sum::<f64>() / n as f64) as f32;
    }
    let (_, lo, rest) = v.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
    let lo = *lo;
    // Upper cut within the remainder: element n−k−1 overall.
    let idx = n - k - 1 - (k + 1);
    let (_, hi, _) = rest.select_nth_unstable_by(idx, |a, b| a.total_cmp(b));
    let hi = *hi;
    // Sum values in [lo, hi], handling ties at the cuts exactly by count.
    let (mut s, mut c) = (0f64, 0usize);
    for &x in f.data.iter().filter(|x| x.is_finite()) {
        if x > lo && x < hi {
            s += x as f64;
            c += 1;
        }
    }
    let want = n - 2 * k;
    let fill = want.saturating_sub(c);
    // Ties at the cut values make up the rest (split evenly between them).
    let (nlo, nhi) = f.data.iter().fold((0usize, 0usize), |(a, b), &x| (a + (x == lo) as usize, b + (x == hi) as usize));
    let take_lo = fill.min(nlo) / if lo == hi { 1 } else { 2 };
    let take_hi = (fill - take_lo).min(nhi);
    s += lo as f64 * take_lo as f64 + hi as f64 * take_hi as f64;
    c += take_lo + take_hi;
    (s / c.max(1) as f64) as f32
}

/// Median, and the standard deviation within 5σ of it (iterated). The
/// clipped SD is what reproduces ImageCalibration's hot-pixel count on
/// James's master dark (87,697 vs 86,436; the MAD gives 95,000+).
pub fn dark_level(data: &[f32]) -> (f32, f32) {
    let mut v: Vec<f32> = data.iter().step_by(3).copied().filter(|x| x.is_finite()).collect();
    let (m, s) = crate::stats::median_sigma_mut(&mut v.clone());
    let (mut lo, mut hi) = (m as f64 - 5.0 * s as f64, m as f64 + 5.0 * s as f64);
    let mut sd = s as f64;
    for _ in 0..10 {
        let (mut n, mut s1, mut s2) = (0f64, 0f64, 0f64);
        for &x in v.iter() {
            let x = x as f64;
            if x >= lo && x <= hi {
                n += 1.0;
                s1 += x;
                s2 += x * x;
            }
        }
        if n < 2.0 {
            break;
        }
        let mu = s1 / n;
        let nsd = ((s2 / n - mu * mu).max(0.0)).sqrt();
        let done = (nsd - sd).abs() <= 1e-4 * sd;
        sd = nsd;
        lo = mu - 5.0 * sd;
        hi = mu + 5.0 * sd;
        if done {
            break;
        }
    }
    v.clear();
    (m, sd as f32)
}

/// Hot pixels of a master dark: more than `k` clipped SDs above its median.
pub fn hot_map(dark: &Image, k: f32) -> Vec<u32> {
    let (m, s) = dark_level(&dark.data);
    let s = s.max(0.5 / dark.adu_scale.max(1.0));
    let t = m + k * s;
    let mut v: Vec<u32> = dark
        .data
        .par_iter()
        .enumerate()
        .filter(|(_, &x)| x > t)
        .map(|(i, _)| i as u32)
        .collect();
    v.sort_unstable();
    v
}

impl Calibrator {
    pub fn new(bias: Option<Master>, dark: Option<Master>, flat: Option<Master>, light_exptime: Option<f64>, params: CalParams) -> anyhow::Result<Calibrator> {
        let size = |m: &Option<Master>| m.as_ref().map(|m| (m.image.width, m.image.height));
        let sizes: Vec<(usize, usize)> = [size(&bias), size(&dark), size(&flat)].into_iter().flatten().collect();
        anyhow::ensure!(sizes.windows(2).all(|w| w[0] == w[1]), "masters differ in size");
        let mut notes = Vec::new();
        let (mut dark_scale, mut scaled) = (1.0f32, false);
        if let Some(d) = &dark {
            notes.push(format!("dark: {}", d.label));
            match (d.exptime, light_exptime) {
                (Some(td), Some(tl)) if td > 0.0 && ((tl - td) / td).abs() > 0.005 => {
                    if bias.is_some() {
                        dark_scale = (tl / td) as f32;
                        scaled = true;
                        notes.push(format!("dark current scaled ×{:.3} ({tl:.0} s light, {td:.0} s dark)", dark_scale));
                    } else {
                        notes.push(format!("dark is {td:.0} s for {tl:.0} s lights and there's no bias to scale it: subtracted as is"));
                    }
                }
                (None, _) | (_, None) => notes.push("dark or light exposure unknown: dark subtracted as is".into()),
                _ => {}
            }
        }
        if let Some(b) = &bias {
            if dark.is_none() || scaled {
                notes.push(format!("bias: {}", b.label));
            }
        }
        let flat_mean = match &flat {
            Some(f) => {
                notes.push(format!("flat: {}", f.label));
                flat_mean(&f.image)
            }
            None => 1.0,
        };
        let hot = match (&dark, params.hot_sigma) {
            (Some(d), Some(k)) => {
                let h = hot_map(&d.image, k);
                let n = d.image.data.len().max(1);
                notes.push(format!("Replaced {} hot pixels ({:.3}%) with the median of their neighbours. Hot means over {k}σ in the master dark", h.len(), h.len() as f64 * 100.0 / n as f64));
                h
            }
            _ => Vec::new(),
        };
        Ok(Calibrator { bias, dark, flat, dark_scale, scaled, flat_mean, hot: Arc::new(hot), params, notes, cfa: false })
    }

    /// Calibrate one raw light (normalised [0, 1]).
    pub fn apply(&self, raw: &Image) -> anyhow::Result<(Image, CalStats)> {
        let (w, h) = (raw.width, raw.height);
        for m in [&self.bias, &self.dark, &self.flat].into_iter().flatten() {
            anyhow::ensure!(m.image.width == w && m.image.height == h, "{} is {}×{}, the light {}×{}", m.label, m.image.width, m.image.height, w, h);
        }
        let mut out = raw.data.clone();
        let dark = self.dark.as_ref().map(|d| &d.image.data);
        let bias = self.bias.as_ref().map(|b| &b.image.data);
        let flat = self.flat.as_ref().map(|f| &f.image.data);
        let (k, scaled, fm) = (self.dark_scale, self.scaled, self.flat_mean);
        let minf = self.params.min_flat * fm;
        out.par_chunks_mut(65536).enumerate().for_each(|(c, chunk)| {
            let o = c * 65536;
            for (j, v) in chunk.iter_mut().enumerate() {
                let i = o + j;
                let mut x = *v;
                match (dark, bias, scaled) {
                    (Some(d), Some(b), true) => x = x - b[i] - (d[i] - b[i]) * k,
                    (Some(d), _, _) => x -= d[i],
                    (None, Some(b), _) => x -= b[i],
                    (None, None, _) => {}
                }
                if let Some(f) = flat {
                    let fv = f[i];
                    let fv = if fv.is_finite() && fv > minf { fv } else { minf.max(f32::MIN_POSITIVE) };
                    x = x / fv * fm;
                }
                *v = x;
            }
        });
        let mut st = CalStats::default();
        if !self.hot.is_empty() {
            st.hot_fixed = fix_hot_step(&mut out, w, h, &self.hot, self.params.hot_radius, if self.cfa { 2 } else { 1 });
        }
        if let Some(limit) = self.params.pedestal_limit {
            st.pedestal = auto_pedestal(&out, limit, raw.adu_scale);
            if st.pedestal > 0.0 {
                let p = st.pedestal;
                out.par_iter_mut().for_each(|v| *v += p);
            }
        }
        let (lo, hi) = out
            .par_chunks_mut(65536)
            .map(|c| {
                let (mut lo, mut hi) = (0usize, 0usize);
                for v in c.iter_mut() {
                    if *v < 0.0 || v.is_nan() {
                        *v = 0.0;
                        lo += 1;
                    } else if *v > 1.0 {
                        *v = 1.0;
                        hi += 1;
                    }
                }
                (lo, hi)
            })
            .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
        st.clipped_low = lo;
        st.clipped_high = hi;
        let mut img = Image::from_data(w, h, out);
        img.adu_scale = raw.adu_scale;
        Ok((img, st))
    }
}

/// Replace each hot pixel with the median of its neighbours (all of them,
/// hot or not, averaging the middle two of an even count), read before any
/// replacement. This is ImageCalibration's rule: on James's LMC frames it
/// reproduces WBPP's values to 1e-6.
pub fn fix_hot(data: &mut [f32], w: usize, h: usize, hot: &[u32], r: usize) -> usize {
    fix_hot_step(data, w, h, hot, r, 1)
}

/// [`fix_hot`] with neighbours `step` pixels apart: 2 for a raw
/// one-shot-colour frame, so only pixels of the same colour count.
pub fn fix_hot_step(data: &mut [f32], w: usize, h: usize, hot: &[u32], r: usize, step: usize) -> usize {
    let r = r.max(1) as i64;
    let step = step.max(1) as i64;
    let src: &[f32] = data;
    let repl: Vec<(u32, f32)> = hot
        .par_iter()
        .filter_map(|&i| {
            let (x, y) = ((i as usize % w) as i64, (i as usize / w) as i64);
            let mut v: Vec<f32> = Vec::with_capacity(((2 * r + 1) * (2 * r + 1)) as usize);
            for dy in -r..=r {
                for dx in -r..=r {
                    let (xx, yy) = (x + dx * step, y + dy * step);
                    if (dx, dy) == (0, 0) || xx < 0 || yy < 0 || xx >= w as i64 || yy >= h as i64 {
                        continue;
                    }
                    v.push(src[yy as usize * w + xx as usize]);
                }
            }
            if v.is_empty() {
                return None;
            }
            v.sort_unstable_by(|a, b| a.total_cmp(b));
            let n = v.len();
            let m = if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) };
            Some((i, m))
        })
        .collect();
    for &(i, v) in &repl {
        data[i as usize] = v;
    }
    repl.len()
}

/// Smallest whole-ADU pedestal that leaves at most `limit` of the pixels
/// below zero (0 when already fine).
pub fn auto_pedestal(data: &[f32], limit: f64, adu_scale: f32) -> f32 {
    let n = data.len();
    let neg = data.par_iter().filter(|v| **v < 0.0).count();
    if (neg as f64) <= limit * n as f64 {
        return 0.0;
    }
    let mut v: Vec<f32> = data.par_iter().copied().filter(|v| *v < 0.0).collect();
    let keep = (limit * n as f64).floor() as usize;
    // The (keep)-th most negative value must reach zero.
    let idx = keep.min(v.len() - 1);
    let (_, q, _) = v.select_nth_unstable_by(idx, |a, b| a.total_cmp(b));
    let p = -*q;
    let s = adu_scale.max(1.0);
    (p * s).ceil() / s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(img: Image, label: &str, t: Option<f64>) -> Master {
        Master { path: PathBuf::from(label), label: label.into(), image: Arc::new(img), exptime: t }
    }

    #[test]
    fn formula_and_hot_pixels() {
        let (w, h) = (32, 32);
        let dark: Vec<f32> = (0..w * h).map(|i| if i == 300 { 0.5 } else { 0.0076 + (i % 3) as f32 * 1e-5 }).collect();
        let flat: Vec<f32> = (0..w * h).map(|i| 0.3 + 0.1 * (i % w) as f32 / w as f32).collect();
        let fm = flat.iter().sum::<f32>() / flat.len() as f32;
        let sky = 0.01f32;
        let light: Vec<f32> = (0..w * h).map(|i| dark[i] + sky * flat[i] / fm).collect();
        let c = Calibrator::new(None, Some(m(Image::from_data(w, h, dark), "d", Some(300.0))), Some(m(Image::from_data(w, h, flat), "f", None)), Some(300.0), CalParams::default()).unwrap();
        assert_eq!(c.hot.as_slice(), &[300]);
        let (out, st) = c.apply(&Image::from_data(w, h, light)).unwrap();
        assert_eq!(st.hot_fixed, 1);
        assert_eq!(st.pedestal, 0.0);
        for (i, v) in out.data.iter().enumerate() {
            assert!((v - sky).abs() < 1e-5, "{i}: {v}");
        }
    }

    #[test]
    fn scales_dark_with_bias() {
        let (w, h) = (8, 8);
        let bias = Image::from_data(w, h, vec![0.0076; w * h]);
        let dark = Image::from_data(w, h, vec![0.0076 + 0.002; w * h]);
        let light = Image::from_data(w, h, vec![0.0076 + 0.001 + 0.05; w * h]);
        let c = Calibrator::new(Some(m(bias, "b", Some(0.0))), Some(m(dark, "d", Some(600.0))), None, Some(300.0), CalParams { hot_sigma: None, ..Default::default() }).unwrap();
        assert!(c.scaled);
        let (out, _) = c.apply(&light).unwrap();
        assert!((out.data[0] - 0.05).abs() < 1e-6);
    }

    #[test]
    fn pedestal_when_negative() {
        let mut d = vec![0.001f32; 10_000];
        for v in d.iter_mut().take(50) {
            *v = -0.0002;
        }
        let p = auto_pedestal(&d, 0.0001, 65535.0);
        assert!(p >= 0.0002 && p < 0.0002 + 2.0 / 65535.0, "{p}");
        assert_eq!(auto_pedestal(&vec![0.001f32; 100], 0.0001, 65535.0), 0.0);
    }
}
