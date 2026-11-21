//! Per-pixel stacking with normalisation and outlier rejection. Used for
//! masters now (flats, darks, bias) and lights in milestone 12.
//!
//! Frames are normalised to the first frame's level (multiplicative for
//! flats, additive for darks and bias), then each pixel's values are
//! cleaned by the chosen rejection and averaged. Rejection follows
//! PixInsight's ImageIntegration: sigma clipping, winsorised sigma
//! clipping, and the generalised extreme Studentized deviate test (ESD,
//! WBPP's choice for flats).

use crate::image::Image;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Normalise {
    None,
    /// Scale each frame to the reference's median (flats).
    Multiplicative,
    /// Shift each frame to the reference's median (darks, bias).
    Additive,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Rejection {
    None,
    /// Iterative clipping around the median, in sigmas.
    Sigma { low: f32, high: f32 },
    /// Like sigma clipping, but the spread comes from winsorised values.
    Winsorized { low: f32, high: f32 },
    /// Generalised ESD: up to `fraction` of the values, significance `alpha`.
    Esd { fraction: f32, alpha: f32 },
    /// Values further than `low`/`high` (fractions of the median) from the
    /// median: PixInsight's percentile clipping, for very small stacks.
    Percentile { low: f32, high: f32 },
    /// Percentile clipping that never rejects a value within `sigma_low` /
    /// `sigma_high` of its frame's measured noise. On a faint narrowband
    /// sky 10% of the median is inside the noise, and plain percentile
    /// clipping rejected 23% of all values in WBPP's 3-frame LMC stacks
    /// (crux's copy: 19–40% on its 4–5-frame panels).
    PercentileNoise { low: f32, high: f32, sigma_low: f32, sigma_high: f32 },
    /// The median (mean of the middle two for an even count). For masters
    /// of 3–7 frames: sigma clipping can't reject anything there (one
    /// outlier among n values is at most (n−1)/√n sigmas out: 1.8 for 5),
    /// so a star drifting through 5 sky flats stayed in WBPP-style masters
    /// at 1–2% and printed dark blobs on every light.
    Median,
}

impl Rejection {
    /// WBPP's defaults by stack size, except the median for 3–7 frames
    /// (WBPP: sigma clipping, which rejects nothing at that size).
    pub fn auto(n: usize) -> Rejection {
        match n {
            0..=2 => Rejection::None,
            3..=7 => Rejection::Median,
            8..=14 => Rejection::Winsorized { low: 4.0, high: 3.0 },
            _ => Rejection::Esd { fraction: 0.3, alpha: 0.05 },
        }
    }

    /// WBPP 3.2's choice for lights by stack size (from its LMC logs: 3 and
    /// 5 frames percentile clipping, 7 winsorised, 17 ESD), except that
    /// small stacks get a noise floor under the percentile rule.
    pub fn auto_lights(n: usize) -> Rejection {
        match n {
            0..=2 => Rejection::None,
            3..=5 => Rejection::PercentileNoise { low: 0.2, high: 0.1, sigma_low: 4.0, sigma_high: 3.0 },
            6..=15 => Rejection::Winsorized { low: 4.0, high: 3.0 },
            _ => Rejection::Esd { fraction: 0.3, alpha: 0.05 },
        }
    }

    pub fn name(&self) -> String {
        match self {
            Rejection::None => "no rejection".into(),
            Rejection::Sigma { low, high } => format!("sigma clipping {low}/{high}σ"),
            Rejection::Winsorized { low, high } => format!("winsorised sigma clipping {low}/{high}σ"),
            Rejection::Esd { fraction, alpha } => format!("ESD (up to {:.0}%, α {alpha})", fraction * 100.0),
            Rejection::Percentile { low, high } => format!("percentile clipping {:.0}%/{:.0}%", low * 100.0, high * 100.0),
            Rejection::PercentileNoise { low, high, sigma_low, sigma_high } => format!("percentile clipping {:.0}%/{:.0}% above the noise ({sigma_low}/{sigma_high}σ)", low * 100.0, high * 100.0),
            Rejection::Median => "median".into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Params {
    pub normalise: Normalise,
    pub rejection: Rejection,
}

#[derive(Debug, Clone)]
pub struct Stacked {
    pub image: Image,
    /// Per frame: (scale, offset) applied before combining.
    pub norm: Vec<(f32, f32)>,
    /// Per frame: fraction of its pixels rejected low and high.
    pub rejected: Vec<(f64, f64)>,
    /// Fraction of all values rejected.
    pub rejected_total: f64,
}

/// Median of a subsample (every `step`th value).
pub fn sample_median(data: &[f32], step: usize) -> f32 {
    let mut v: Vec<f32> = data.iter().step_by(step.max(1)).copied().filter(|x| x.is_finite()).collect();
    crate::stats::median_mut(&mut v)
}

/// Stack same-sized frames. Fails if there are none or sizes differ.
pub fn stack(frames: &[Image], p: &Params) -> anyhow::Result<Stacked> {
    let n = frames.len();
    anyhow::ensure!(n > 0, "nothing to stack");
    let (w, h) = (frames[0].width, frames[0].height);
    anyhow::ensure!(frames.iter().all(|f| f.width == w && f.height == h), "frames differ in size");
    let meds: Vec<f32> = frames.par_iter().map(|f| sample_median(&f.data, 17)).collect();
    let norm: Vec<(f32, f32)> = meds
        .iter()
        .map(|&m| match p.normalise {
            Normalise::None => (1.0, 0.0),
            Normalise::Multiplicative => (if m > 0.0 { meds[0] / m } else { 1.0 }, 0.0),
            Normalise::Additive => (1.0, meds[0] - m),
        })
        .collect();
    let esd = match p.rejection {
        Rejection::Esd { fraction, alpha } => Some(esd_table(n, fraction, alpha)),
        _ => None,
    };
    let mut out = vec![0f32; w * h];
    // Per row: rejected low/high counts per frame.
    let counts: Vec<Vec<(u32, u32)>> = out
        .par_chunks_mut(w)
        .enumerate()
        .map(|(y, row)| {
            let mut cnt = vec![(0u32, 0u32); n];
            let mut vals: Vec<(f32, u16)> = Vec::with_capacity(n);
            let mut work: Vec<f32> = Vec::with_capacity(n);
            for (x, o) in row.iter_mut().enumerate() {
                vals.clear();
                for (i, f) in frames.iter().enumerate() {
                    let v = f.data[y * w + x];
                    if v.is_finite() {
                        vals.push((v * norm[i].0 + norm[i].1, i as u16));
                    }
                }
                if vals.is_empty() {
                    *o = 0.0;
                    continue;
                }
                let (lo, hi) = reject(&mut vals, &mut work, p.rejection, esd.as_deref(), None);
                let kept = &vals[lo..hi];
                *o = kept.iter().map(|v| v.0 as f64).sum::<f64>() as f32 / kept.len() as f32;
                if p.rejection == Rejection::Median {
                    // Not rejection as such: nothing to count.
                    continue;
                }
                for &(_, i) in &vals[..lo] {
                    cnt[i as usize].0 += 1;
                }
                for &(_, i) in &vals[hi..] {
                    cnt[i as usize].1 += 1;
                }
            }
            cnt
        })
        .collect();
    let mut rej = vec![(0u64, 0u64); n];
    for row in &counts {
        for (i, c) in row.iter().enumerate() {
            rej[i].0 += c.0 as u64;
            rej[i].1 += c.1 as u64;
        }
    }
    let px = (w * h) as f64;
    let total = rej.iter().map(|r| (r.0 + r.1) as f64).sum::<f64>() / (px * n as f64);
    let mut image = Image::from_data(w, h, out);
    image.adu_scale = frames[0].adu_scale;
    Ok(Stacked { image, norm, rejected: rej.iter().map(|r| (r.0 as f64 / px, r.1 as f64 / px)).collect(), rejected_total: total })
}

/// Percentile clipping never rejects the median: with an even count it is
/// the mean of the middle two, and both could fall outside the band (4
/// frames at a star's edge, seeing differing), leaving nothing and a zero
/// pixel (13,000–32,000 per 4-frame stack on NGC 3372 H).
fn keep_median((lo, hi): (usize, usize), n: usize) -> (usize, usize) {
    (lo.min((n - 1) / 2), hi.max(n / 2 + 1))
}

/// Sort `vals` and return the kept range `lo..hi`. `esd` is the critical
/// value table for `vals.len()` values; `noise` each frame's noise (by
/// the frame index in `vals`), for `PercentileNoise`.
pub(crate) fn reject(vals: &mut [(f32, u16)], work: &mut Vec<f32>, r: Rejection, esd: Option<&[f32]>, noise: Option<&[f32]>) -> (usize, usize) {
    vals.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    let n = vals.len();
    if n < 3 {
        return (0, n);
    }
    match r {
        Rejection::None => (0, n),
        Rejection::Median => {
            if n % 2 == 1 {
                (n / 2, n / 2 + 1)
            } else {
                (n / 2 - 1, n / 2 + 1)
            }
        }
        Rejection::Percentile { low, high } => {
            let m = median_sorted(vals);
            if m <= 0.0 {
                return (0, n);
            }
            keep_median(clip_range(vals, 0, n, m - low * m, m + high * m), n)
        }
        Rejection::PercentileNoise { low, high, sigma_low, sigma_high } => {
            let m = median_sorted(vals);
            let sd = |i: u16| noise.and_then(|v| v.get(i as usize)).copied().unwrap_or(0.0);
            // Sorted, so the rejected values are at the ends.
            let mut lo = 0;
            while lo < n && m - vals[lo].0 > (low * m.max(0.0)).max(sigma_low * sd(vals[lo].1)) {
                lo += 1;
            }
            let mut hi = n;
            while hi > lo && vals[hi - 1].0 - m > (high * m.max(0.0)).max(sigma_high * sd(vals[hi - 1].1)) {
                hi -= 1;
            }
            keep_median((lo, hi), n)
        }
        Rejection::Sigma { low, high } => {
            let (mut lo, mut hi) = (0, n);
            loop {
                if hi - lo < 3 {
                    break;
                }
                let s = &vals[lo..hi];
                let m = median_sorted(s);
                let sd = stddev(s.iter().map(|v| v.0));
                if sd <= 0.0 {
                    break;
                }
                let (nlo, nhi) = clip_range(vals, lo, hi, m - low * sd, m + high * sd);
                if (nlo, nhi) == (lo, hi) {
                    break;
                }
                lo = nlo;
                hi = nhi;
            }
            (lo, hi)
        }
        Rejection::Winsorized { low, high } => {
            let (mut lo, mut hi) = (0, n);
            loop {
                if hi - lo < 3 {
                    break;
                }
                let s = &vals[lo..hi];
                let m = median_sorted(s);
                // Winsorise at 1.5σ until the spread settles (Huber's method).
                work.clear();
                work.extend(s.iter().map(|v| v.0));
                let mut sd = stddev(work.iter().copied());
                for _ in 0..10 {
                    let (a, b) = (m - 1.5 * sd, m + 1.5 * sd);
                    for v in work.iter_mut() {
                        *v = v.clamp(a, b);
                    }
                    let nsd = 1.134 * stddev(work.iter().copied());
                    if (nsd - sd).abs() <= sd * 5e-4 {
                        sd = nsd;
                        break;
                    }
                    sd = nsd;
                }
                if sd <= 0.0 {
                    break;
                }
                let (nlo, nhi) = clip_range(vals, lo, hi, m - low * sd, m + high * sd);
                if (nlo, nhi) == (lo, hi) {
                    break;
                }
                lo = nlo;
                hi = nhi;
            }
            (lo, hi)
        }
        Rejection::Esd { .. } => {
            let lam = esd.unwrap_or(&[]);
            // Remove the most extreme value up to r times; the number of
            // outliers is the largest i whose statistic beats its critical value.
            let (mut lo, mut hi) = (0usize, n);
            let mut removed: Vec<(usize, usize)> = Vec::new();
            let mut outliers = 0;
            let maxr = lam.len().min(n.saturating_sub(3));
            for i in 0..maxr {
                let s = &vals[lo..hi];
                let k = s.len() as f64;
                let mean = s.iter().map(|v| v.0 as f64).sum::<f64>() / k;
                let sd = (s.iter().map(|v| (v.0 as f64 - mean).powi(2)).sum::<f64>() / (k - 1.0)).sqrt();
                if sd <= 0.0 {
                    break;
                }
                let dl = mean - s[0].0 as f64;
                let dh = s[s.len() - 1].0 as f64 - mean;
                let stat = dl.max(dh) / sd;
                if dl > dh {
                    lo += 1;
                } else {
                    hi -= 1;
                }
                removed.push((lo, hi));
                if stat > lam[i] as f64 {
                    outliers = i + 1;
                }
            }
            if outliers == 0 {
                (0, n)
            } else {
                removed[outliers - 1]
            }
        }
    }
}

fn clip_range(vals: &[(f32, u16)], lo: usize, hi: usize, a: f32, b: f32) -> (usize, usize) {
    let mut nlo = lo;
    while nlo < hi && vals[nlo].0 < a {
        nlo += 1;
    }
    let mut nhi = hi;
    while nhi > nlo && vals[nhi - 1].0 > b {
        nhi -= 1;
    }
    (nlo, nhi)
}

fn median_sorted(s: &[(f32, u16)]) -> f32 {
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2].0
    } else {
        0.5 * (s[n / 2 - 1].0 + s[n / 2].0)
    }
}

fn stddev(it: impl Iterator<Item = f32> + Clone) -> f32 {
    let (mut n, mut s, mut s2) = (0f64, 0f64, 0f64);
    for v in it {
        n += 1.0;
        s += v as f64;
        s2 += (v as f64) * (v as f64);
    }
    if n < 2.0 {
        return 0.0;
    }
    (((s2 - s * s / n) / (n - 1.0)).max(0.0)).sqrt() as f32
}

/// ESD critical values λ_i for i = 1..⌊fraction·n⌋ (Rosner 1983).
pub fn esd_table(n: usize, fraction: f32, alpha: f32) -> Vec<f32> {
    let r = ((fraction as f64) * n as f64).floor() as usize;
    (1..=r)
        .map(|i| {
            let nn = n as f64;
            let ii = i as f64;
            let p = 1.0 - alpha as f64 / (2.0 * (nn - ii + 1.0));
            let nu = nn - ii - 1.0;
            if nu < 1.0 {
                return f32::INFINITY;
            }
            let t = t_quantile(p, nu);
            ((nn - ii) * t / ((nu + t * t) * (nn - ii + 1.0)).sqrt()) as f32
        })
        .collect()
}

/// Student's t quantile by bisection on the CDF.
pub fn t_quantile(p: f64, nu: f64) -> f64 {
    let (mut a, mut b) = (0.0f64, 1000.0f64);
    for _ in 0..200 {
        let m = 0.5 * (a + b);
        if t_cdf(m, nu) < p {
            a = m;
        } else {
            b = m;
        }
    }
    0.5 * (a + b)
}

pub fn t_cdf(t: f64, nu: f64) -> f64 {
    let x = nu / (nu + t * t);
    let tail = 0.5 * betai(0.5 * nu, 0.5, x);
    if t >= 0.0 {
        1.0 - tail
    } else {
        tail
    }
}

/// Regularised incomplete beta I_x(a, b) (Numerical Recipes).
fn betai(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let bt = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

fn betacf(a: f64, b: f64, x: f64) -> f64 {
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let tiny = 1e-300;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..300 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-12 {
            break;
        }
    }
    h
}

fn ln_gamma(x: f64) -> f64 {
    // Lanczos approximation (g = 7, n = 9).
    const G: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = G[0];
    let t = x + 7.5;
    for (i, g) in G.iter().enumerate().skip(1) {
        a += g / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_keeps_the_middle_of_an_even_stack() {
        let mut work = Vec::new();
        let noise = [1.0f32; 4];
        for r in [Rejection::Percentile { low: 0.2, high: 0.1 }, Rejection::PercentileNoise { low: 0.2, high: 0.1, sigma_low: 4.0, sigma_high: 3.0 }] {
            // Middle two 1,000 and 3,000: median 2,000; both outside 20%/10%.
            let mut v = vec![(900.0f32, 0u16), (1000.0, 1), (3000.0, 2), (3100.0, 3)];
            let (lo, hi) = reject(&mut v, &mut work, r, None, Some(&noise));
            assert!(lo <= 1 && hi >= 3, "{r:?}: kept {lo}..{hi}");
        }
    }

    #[test]
    fn t_quantiles() {
        // Standard table values.
        assert!((t_quantile(0.975, 10.0) - 2.228).abs() < 0.002);
        assert!((t_quantile(0.995, 5.0) - 4.032).abs() < 0.002);
        assert!((t_quantile(0.95, 1.0) - 6.314).abs() < 0.002);
    }

    #[test]
    fn esd_matches_rosner() {
        // Rosner's example in the NIST handbook: n = 54, α = 0.05:
        // λ1 = 3.158, λ5 = 3.128, λ10 = 3.085.
        let l = esd_table(54, 10.0 / 54.0 + 1e-6, 0.05);
        assert!((l[0] - 3.158).abs() < 0.002, "{}", l[0]);
        assert!((l[4] - 3.128).abs() < 0.002, "{}", l[4]);
        assert!((l[9] - 3.085).abs() < 0.002, "{}", l[9]);
    }

    #[test]
    fn rejects_outliers() {
        let mut frames = Vec::new();
        let mut rng = crate::synth::Rng::new(3);
        for i in 0..21 {
            let mut d = vec![0f32; 64 * 64];
            for v in d.iter_mut() {
                *v = 0.4 + 0.01 * rng.gauss() as f32;
            }
            if i == 5 {
                d[100] = 0.9; // a cosmic ray
            }
            frames.push(Image::from_data(64, 64, d.iter().map(|v| v * (1.0 + i as f32 * 0.01)).collect()));
        }
        for rej in [Rejection::Esd { fraction: 0.3, alpha: 0.05 }, Rejection::Winsorized { low: 4.0, high: 3.0 }, Rejection::Sigma { low: 4.0, high: 3.0 }] {
            let s = stack(&frames, &Params { normalise: Normalise::Multiplicative, rejection: rej }).unwrap();
            assert!((s.image.data[100] - 0.4).abs() < 0.02, "{rej:?} {}", s.image.data[100]);
            assert!(s.rejected[5].1 > 0.0);
            // Normalised to the first frame.
            assert!((sample_median(&s.image.data, 1) - 0.4).abs() < 0.003);
            assert!(s.rejected_total < 0.01, "{rej:?} {}", s.rejected_total);
        }
    }
}
