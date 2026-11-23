//! Colour mix, stretch and finishing: a port of James's Narrowband Palette
//! Mixer (`C:\Projects\pixinsight\NarrowbandPaletteMixer.js`, v1.6, the
//! code between `@@ENGINE-BEGIN` and `@@ENGINE-END`), generalised from
//! H/O/S to any set of channels:
//!
//! - narrowband (H, O, S): the mixer exactly, including its presets;
//! - broadband (L, R, G, B, optionally H): R, G and B balanced so the
//!   average star is white, luminance from L;
//! - one channel: a mono stretch.
//!
//! Each output channel is `lin_c = dark_ref + Σ_k w_kc · g_k · (x_k −
//! dark_k)`, then one linked stretch, then keep-colour, green removal,
//! luminance, saturation and the black level, in the mixer's order. The
//! same code renders the preview, the preset tiles and the full image.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Signal level: this percentile of the proxy (mixer: 0.95).
pub const SIGNAL_QUANTILE: f64 = 0.95;
/// Pixels at or below this are clipped (registration borders).
pub const ZERO_FLOOR: f32 = 1.0e-6;
/// Normalisation stops boosting a line once its background is this many
/// times as rough as the reference's (mixer: 3).
pub const BOOST_NOISE_LIMIT: f64 = 3.0;

/// The mixer's midtones transfer function (STF, HistogramTransformation).
#[inline]
pub fn mtf(m: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    ((m - 1.0) * x) / ((2.0 * m - 1.0) * x - m)
}

pub fn quantile_of_sorted(s: &[f32], q: f64) -> f64 {
    if s.is_empty() {
        return 0.0;
    }
    let i = (q * (s.len() - 1) as f64).round() as isize;
    s[i.clamp(0, s.len() as isize - 1) as usize] as f64
}

/// Drop sorted samples below `floor` (clipped pixels), unless that would
/// leave under 5%.
pub fn trim_clipped(sorted: &[f32], floor: f32) -> &[f32] {
    let i = sorted.partition_point(|v| *v < floor);
    if (sorted.len() - i) as f64 >= 0.05 * sorted.len() as f64 {
        &sorted[i..]
    } else {
        sorted
    }
}

/// One channel's statistics, as the mixer measures them.
#[derive(Debug, Clone, Default)]
pub struct Stats {
    /// Full-resolution median and raw MAD, zero pixels excluded.
    pub median: f64,
    pub mad: f64,
    /// ImageIntegration's NOISE00, when known (0 otherwise).
    pub noise: f64,
    /// The master's noise as integrated, before any denoising (its
    /// integration report; 0 if unknown). Only tells a frame-filling nebula
    /// from sky (`auto_stretch`): a denoised master's own noise is too low
    /// for that, and `noise` would weaken the mixer's stretch.
    pub sky_noise: f64,
    /// Proxy median and signal level (95th percentile), and the sorted
    /// proxy samples for the dark reference.
    pub p_median: f64,
    pub p_high: f64,
    pub sorted: Vec<f32>,
}

impl Stats {
    /// `full`: every pixel of the channel; `proxy`: its preview copy.
    pub fn measure(full: &[f32], proxy: &[f32]) -> Stats {
        use rayon::prelude::*;
        // Exact selections over the pixels with data, in parallel and
        // without copying them (the same values select_nth gives).
        let keep = |x: f32| x > ZERO_FLOOR && x.is_finite();
        let n = full.par_iter().filter(|x| keep(**x)).count();
        let (median, mad) = if n == 0 {
            (0.0, 0.0)
        } else {
            let med = crate::stats::kth_par_by(full, n / 2, |x| keep(x).then_some(x));
            (med as f64, crate::stats::kth_par_by(full, n / 2, |x| keep(x).then(|| (x - med).abs())) as f64)
        };
        let mut all: Vec<f32> = proxy.iter().copied().filter(|x| x.is_finite()).collect();
        all.sort_by(|a, b| a.total_cmp(b));
        let sorted = trim_clipped(&all, ZERO_FLOOR).to_vec();
        Stats { median, mad, noise: 0.0, sky_noise: 0.0, p_median: quantile_of_sorted(&sorted, 0.5), p_high: quantile_of_sorted(&sorted, SIGNAL_QUANTILE), sorted }
    }
}

/// The noise that sets the black point: the larger of the MAD-based sigma
/// and the integration's own estimate.
pub fn stretch_sigma(s: &Stats) -> f64 {
    (1.4826 * s.mad).max(if s.noise > 0.0 { s.noise } else { 0.0 }).max(1.0e-7)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stretch {
    pub bp: f64,
    pub m: f64,
}

/// Shadows clipping and midtones balance that map `bg` to the target.
pub fn make_stretch(bg: f64, sigma: f64, o: &Stretching) -> Stretch {
    if !o.linear {
        return Stretch { bp: 0.0, m: 0.5 };
    }
    let mut bp = bg - o.clip * sigma;
    if bp >= bg {
        bp = bg - 1.0e-6;
    }
    let x0 = (bg - bp) / (1.0 - bp);
    Stretch { bp, m: mtf(o.background, x0) }
}

/// The darkest structure: this percentile of the proxy. The proxy is
/// block-averaged, so this is real structure (dust lanes, the darkest sky),
/// not noise.
pub const LOW_QUANTILE: f64 = 0.005;
/// Where a frame-filling target's darkest structure goes, as a fraction of
/// the background target (a sky frame's dark structure sits a little under
/// its median).
pub const LOW_FRACTION: f64 = 0.85;

/// The mixer's stretch (median → background, black point `clip` MAD-sigmas
/// below), unless the target fills the frame. Then the median is nebula and
/// the MAD measures its structure rather than the noise, which put the black
/// point far below the darkest pixel and the nebula at sky level (NGC 2244 H:
/// black point 45 ADU, darkest 0.1% at 109, median 227 → 0.15).
///
/// `noise` is the master's noise as integrated (0 if unknown: the mixer's
/// stretch). When the darkest structure `lo` sits more than `clip` noise
/// sigmas above the mixer's black point, no pixel comes near that black
/// point: it rises to `clip` sigmas under `lo`, and the stretch puts `lo` at
/// `LOW_FRACTION` of the background (or the median at the background, if
/// that's stronger). In a sky frame the MAD sigma is the noise and `lo` the
/// sky, so the black point never rises and nothing changes; a denoised
/// master's lower MAD makes that more certain.
pub fn auto_stretch(med: f64, lo: f64, sigma: f64, noise: f64, o: &Stretching) -> Stretch {
    let s = make_stretch(med, sigma, o);
    if !o.linear || noise <= 0.0 || !(lo < med) || !lo.is_finite() {
        return s;
    }
    let bp = lo - o.clip * noise;
    if bp <= s.bp {
        return s;
    }
    let x = |v: f64| ((v - bp) / (1.0 - bp)).clamp(1.0e-9, 1.0);
    // A smaller midtones balance stretches harder. At the crossover the
    // median's term equals the mixer's balance, but `lo`'s can be far
    // smaller, so it comes in over one clip width of black point rise:
    // taken at once, a mix weight moving across the line jumped the whole
    // image's brightness (NGC 2244, OIII's blue 176% → 184%).
    let m_med = mtf(o.background, x(med));
    let m_lo = mtf(LOW_FRACTION * o.background, x(lo));
    let t = ((bp - s.bp) / (o.clip * noise).max(1.0e-12)).clamp(0.0, 1.0);
    let m = if m_lo < m_med { m_med + t * (m_lo - m_med) } else { m_med };
    Stretch { bp, m }
}

/// How the channels are scaled before mixing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Balance {
    /// Each line's signal above background matched to the reference's
    /// (the mixer's "Normalise OIII and SII to Ha"), with the boost limit.
    #[default]
    Signal,
    /// The average star white (broadband colour).
    Stars,
    /// As they are.
    None,
}

/// Stretch settings (the mixer's Stretch box).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Stretching {
    /// Uncheck for already-stretched channels.
    pub linear: bool,
    /// Background level after the stretch (mixer default 0.15).
    pub background: f64,
    /// Fraction of the background pulled down to black.
    pub black: f64,
    /// Shadow clip in noise sigmas below the background.
    pub clip: f64,
    /// Neutral point at each channel's median (small targets on black sky).
    pub neutral_sky: bool,
    /// Percentile of each channel treated as neutral black.
    pub dark_ref: f64,
    /// Fraction of colour-preserving stretch.
    pub keep_colour: f64,
    /// Highlight protection, 0–1: a shoulder above the background pulls
    /// bright nebula down (1: full scale lands at about half way between
    /// the background and white), keeping each pixel's colour. 0 is the
    /// mixer's curve.
    pub highlights: f64,
}

impl Default for Stretching {
    fn default() -> Self {
        Stretching { linear: true, background: 0.11, black: 0.0, clip: 2.8, neutral_sky: false, dark_ref: 0.10, keep_colour: 0.0, highlights: 0.0 }
    }
}

/// Finishing (the mixer's Finishing box).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Finishing {
    /// Fraction of brightness taken from the luminance channel (H for
    /// narrowband, L for broadband).
    pub luminance: f64,
    /// Green removal (SCNR-like).
    pub green: f64,
    /// Saturation (1 = unchanged).
    pub saturation: f64,
    /// Nebula detail, 0–1: large-scale local contrast on the starless
    /// image (stars stretched separately only); 0 leaves it as stretched.
    pub detail: f64,
    /// Colour noise reduction, 0–1: the faint parts of the starless image
    /// take the colour of their surroundings (stars stretched separately
    /// only); bright nebula keeps its own.
    pub chroma: f64,
    /// Neutral background, 0–1: takes the faint sky's colour tint out,
    /// measured region by region (a palette's boosted weak lines tint the
    /// faint sky), fading out towards the nebula (stars stretched
    /// separately, narrowband only).
    pub neutral: f64,
}

impl Default for Finishing {
    fn default() -> Self {
        Finishing { luminance: 0.5, green: 0.0, saturation: 1.3, detail: 0.0, chroma: 0.0, neutral: 1.0 }
    }
}

/// The colour mix: per channel, the fraction sent to red, green and blue.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct Mix {
    /// The preset these weights came from ("" once changed).
    pub preset: String,
    pub balance: Balance,
    pub weights: BTreeMap<String, [f64; 3]>,
}

/// A preset: weights, plus stretch and finishing changes for the ones
/// that need them.
#[derive(Debug, Clone)]
pub struct Preset {
    pub name: &'static str,
    pub weights: Vec<(&'static str, [f64; 3])>,
    pub balance: Balance,
    pub stretch: Option<Stretching>,
    pub finish: Option<Finishing>,
}

/// What kind of target the channels make.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    Narrowband,
    Broadband,
    Mono,
}

/// Canonical channel name for a filter label.
pub fn canonical(filter: &str) -> String {
    let f = filter.trim().to_uppercase();
    let f = f.trim_start_matches("FILTER-").trim_start_matches("FILTER_");
    match f {
        "H" | "HA" | "H-ALPHA" | "HALPHA" | "H_ALPHA" => "H".into(),
        "O" | "OIII" | "O3" | "O-III" => "O".into(),
        "S" | "SII" | "S2" | "S-II" => "S".into(),
        "L" | "LUM" | "LUMINANCE" | "CLEAR" | "C" => "L".into(),
        "R" | "RED" => "R".into(),
        "G" | "GREEN" => "G".into(),
        "B" | "BLUE" => "B".into(),
        o => o.to_string(),
    }
}

pub fn kind_of(channels: &[String]) -> Kind {
    let has = |c: &str| channels.iter().any(|x| x == c);
    if has("R") && has("G") && has("B") {
        Kind::Broadband
    } else if ["H", "O", "S"].iter().filter(|c| has(c)).count() >= 2 {
        Kind::Narrowband
    } else {
        Kind::Mono
    }
}

fn p(name: &'static str, h: [f64; 3], s: [f64; 3], o: [f64; 3]) -> Preset {
    let f = |v: [f64; 3]| [v[0] / 100.0, v[1] / 100.0, v[2] / 100.0];
    Preset { name, weights: vec![("H", f(h)), ("S", f(s)), ("O", f(o))], balance: Balance::Signal, stretch: None, finish: None }
}

/// Presets for a kind of target. Narrowband: the mixer's nine, in its order.
pub fn presets(kind: Kind, channels: &[String]) -> Vec<Preset> {
    match kind {
        Kind::Narrowband => vec![
            p("Copper & Blue", [90.0, 45.0, 0.0], [80.0, 30.0, 0.0], [0.0, 50.0, 140.0]),
            p("Red & Blue", [100.0, 20.0, 5.0], [100.0, 50.0, 0.0], [0.0, 45.0, 150.0]),
            p("Gold & Blue", [70.0, 45.0, 0.0], [100.0, 40.0, 0.0], [0.0, 35.0, 140.0]),
            p("Ha Red / OIII Teal", [100.0, 0.0, 0.0], [60.0, 40.0, 0.0], [0.0, 80.0, 100.0]),
            p("Realistic", [70.0, 0.0, 0.0], [30.0, 30.0, 0.0], [0.0, 70.0, 100.0]),
            p("HOO", [100.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 100.0, 100.0]),
            p("SHO", [0.0, 100.0, 0.0], [100.0, 0.0, 0.0], [0.0, 0.0, 100.0]),
            p("HSO", [100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]),
            p("Hubble Helix", [100.0, 60.0, 0.0], [100.0, 0.0, 0.0], [20.0, 40.0, 70.0]),
        ],
        Kind::Broadband => {
            let rgb = vec![("R", [1.0, 0.0, 0.0]), ("G", [0.0, 1.0, 0.0]), ("B", [0.0, 0.0, 1.0])];
            let has_h = channels.iter().any(|c| c == "H");
            let has_l = channels.iter().any(|c| c == "L");
            let fin = |lum: f64, sat: f64| Some(Finishing { luminance: lum, green: 0.0, saturation: sat, detail: 0.0, chroma: 0.0, neutral: 0.0 });
            let mut v = vec![
                Preset { name: "Natural", weights: rgb.clone(), balance: Balance::Stars, stretch: None, finish: fin(if has_l { 0.8 } else { 0.0 }, 1.2) },
                Preset { name: "Vivid", weights: rgb.clone(), balance: Balance::Stars, stretch: None, finish: fin(if has_l { 0.8 } else { 0.0 }, 1.6) },
                Preset { name: "RGB only", weights: rgb.clone(), balance: Balance::Stars, stretch: None, finish: fin(0.0, 1.2) },
            ];
            if has_h {
                let mut w = rgb.clone();
                w.push(("H", [0.5, 0.0, 0.03]));
                v.insert(1, Preset { name: "Ha + RGB", weights: w, balance: Balance::Stars, stretch: None, finish: fin(if has_l { 0.8 } else { 0.0 }, 1.3) });
            }
            v
        }
        Kind::Mono => {
            let c: &'static str = match channels.first().map(|s| s.as_str()) {
                Some("H") => "H",
                Some("O") => "O",
                Some("S") => "S",
                Some("L") => "L",
                Some("R") => "R",
                Some("G") => "G",
                Some("B") => "B",
                _ => "*",
            };
            let st = |bg: f64, black: f64| Some(Stretching { background: bg, black, ..Stretching::default() });
            let mono = |name, bg, black| Preset { name, weights: vec![(c, [1.0, 1.0, 1.0])], balance: Balance::None, stretch: st(bg, black), finish: Some(Finishing { luminance: 0.0, green: 0.0, saturation: 1.0, detail: 0.0, chroma: 0.0, neutral: 0.0 }) };
            vec![mono("Natural", 0.15, 0.0), mono("Gentle", 0.10, 0.0), mono("Strong", 0.25, 0.0), mono("Black sky", 0.12, 0.6)]
        }
    }
}

/// The mix for a preset over these channels: weights for channels that
/// aren't there fold into their stand-in (S → H and H → S, as the mixer
/// substitutes one for the other; "*" → the only channel).
pub fn mix_of(preset: &Preset, channels: &[String]) -> Mix {
    let has = |c: &str| channels.iter().any(|x| x == c);
    let mut w: BTreeMap<String, [f64; 3]> = channels.iter().map(|c| (c.clone(), [0.0; 3])).collect();
    for (c, v) in &preset.weights {
        let target = if *c == "*" {
            channels.first().cloned()
        } else if has(c) {
            Some(c.to_string())
        } else if *c == "S" && has("H") {
            Some("H".into())
        } else if *c == "H" && has("S") {
            Some("S".into())
        } else {
            None
        };
        if let Some(t) = target {
            let e = w.entry(t).or_insert([0.0; 3]);
            for i in 0..3 {
                e[i] += v[i];
            }
        }
    }
    Mix { preset: preset.name.to_string(), balance: preset.balance, weights: w }
}

/// The channel scaling, stretch and finishing constants for one render.
#[derive(Debug, Clone)]
pub struct Coef {
    /// Per output colour, per input channel.
    pub a: [Vec<f64>; 3],
    pub k: [f64; 3],
    pub bp: [f64; 3],
    pub m: [f64; 3],
    pub lum: Option<(usize, Stretch)>,
    pub keep_colour: f64,
    pub black: f64,
    pub luminance: f64,
    pub green: f64,
    pub saturation: f64,
    /// Highlight shoulder: strength (0–0.5) and knee (the background's
    /// output level).
    pub shoulder: (f64, f64),
    pub gain: Vec<f64>,
    /// (channel, wanted gain, used gain) where the boost limit applied.
    pub limited: Vec<(usize, f64, f64)>,
}

/// Which channel the others are scaled to: H, else S, else G, else L,
/// else the first.
pub fn reference(channels: &[String], kind: Kind) -> usize {
    let order: &[&str] = match kind {
        Kind::Narrowband => &["H", "S", "O"],
        Kind::Broadband => &["G", "L"],
        Kind::Mono => &[],
    };
    order.iter().find_map(|c| channels.iter().position(|x| x == c)).unwrap_or(0)
}

/// Which channel supplies luminance: H for narrowband, L (else none) for
/// broadband, the channel itself for mono.
pub fn luminance_channel(channels: &[String], kind: Kind) -> Option<usize> {
    match kind {
        Kind::Narrowband => channels.iter().position(|x| x == "H").or_else(|| channels.iter().position(|x| x == "S")),
        Kind::Broadband => channels.iter().position(|x| x == "L"),
        Kind::Mono => Some(0),
    }
}

/// The mixer's `computeCoefficients`, over any channels. `stats[i]` and
/// `star_gain[i]` (Balance::Stars: reference flux ÷ channel flux) per
/// channel; channels with no weight in the mix are left out.
pub fn coefficients(channels: &[String], stats: &[&Stats], star_gain: Option<&[f64]>, mix: &Mix, st: &Stretching, fin: &Finishing) -> Coef {
    let kind = kind_of(channels);
    let n = channels.len();
    let r = reference(channels, kind);
    let refs = stats[r];
    let ref_range = refs.p_high - refs.p_median;
    let dark = |s: &Stats| quantile_of_sorted(&s.sorted, if st.neutral_sky { 0.5 } else { st.dark_ref });
    let spread = |s: &Stats| (s.p_median - quantile_of_sorted(&s.sorted, 0.1587)).max(1.0e-9);
    let w: Vec<[f64; 3]> = channels.iter().map(|c| mix.weights.get(c).copied().unwrap_or([0.0; 3])).collect();
    // A channel takes part if the mix uses it, or it's the reference.
    let used: Vec<bool> = (0..n).map(|i| i == r || w[i].iter().any(|x| *x != 0.0)).collect();
    let mut gain = vec![0.0; n];
    let mut limited = Vec::new();
    for i in 0..n {
        if !used[i] {
            continue;
        }
        let s = stats[i];
        let mut g = 1.0;
        match mix.balance {
            Balance::Signal => {
                let d = s.p_high - s.p_median;
                if d > 1.0e-8 && ref_range > 1.0e-8 {
                    g = ref_range / d;
                }
                let gmax = (BOOST_NOISE_LIMIT * spread(refs) / spread(s)).max(1.0);
                if g > gmax {
                    limited.push((i, g, gmax));
                    g = gmax;
                }
            }
            Balance::Stars => {
                if let Some(sg) = star_gain.and_then(|v| v.get(i)).copied().filter(|x| x.is_finite() && *x > 0.0) {
                    g = sg;
                }
            }
            Balance::None => {}
        }
        gain[i] = g;
    }
    let mut a: [Vec<f64>; 3] = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
    let mut k = [0.0; 3];
    for c in 0..3 {
        let mut k0 = dark(refs);
        for i in 0..n {
            if !used[i] {
                continue;
            }
            let ai = w[i][c] * gain[i];
            a[c][i] = ai;
            k0 -= ai * dark(stats[i]);
        }
        k[c] = k0;
    }
    // One linked stretch, anchored where every line sits at its median.
    let mut mix_bg = 0.0;
    for c in 0..3 {
        let mut v = k[c];
        for i in 0..n {
            if used[i] {
                v += a[c][i] * stats[i].median;
            }
        }
        mix_bg += v / 3.0;
    }
    // The mix's darkest structure (each line at its low percentile) and
    // noise (0 unless every line's noise is known).
    let lo_of = |s: &Stats| quantile_of_sorted(&s.sorted, LOW_QUANTILE);
    let (mut mix_lo, mut mix_noise) = (0.0, 0.0);
    let noise_known = (0..n).filter(|i| used[*i]).all(|i| stats[i].sky_noise > 0.0);
    for c in 0..3 {
        let (mut v, mut var) = (k[c], 0.0);
        for i in 0..n {
            if used[i] {
                v += a[c][i] * lo_of(stats[i]);
                var += (a[c][i] * stats[i].sky_noise).powi(2);
            }
        }
        mix_lo += v / 3.0;
        mix_noise += var.sqrt() / 3.0;
    }
    if !noise_known {
        mix_noise = 0.0;
    }
    let sigma = stretch_sigma(refs);
    let s = auto_stretch(mix_bg, mix_lo, sigma, mix_noise, st);
    let lum = luminance_channel(channels, kind).map(|li| {
        let ls = stats[li];
        (li, auto_stretch(ls.median, lo_of(ls), stretch_sigma(ls), ls.sky_noise, st))
    });
    Coef {
        a,
        k,
        bp: [s.bp; 3],
        m: [s.m; 3],
        lum,
        keep_colour: st.keep_colour,
        black: if st.linear { st.black * st.background } else { 0.0 },
        luminance: fin.luminance,
        green: fin.green,
        saturation: fin.saturation,
        shoulder: {
            let blk = if st.linear { st.black * st.background } else { 0.0 };
            (0.5 * st.highlights.clamp(0.0, 1.0), ((st.background - blk) / (1.0 - blk)).clamp(0.0, 0.95))
        },
        gain,
        limited,
    }
}

/// The mixer's `processPixels`: `ch[i]` is channel i's pixels; writes RGB.
pub fn process(ch: &[&[f32]], coef: &Coef, out_r: &mut [f32], out_g: &mut [f32], out_b: &mut [f32]) {
    let n = ch.len();
    let npx = out_r.len();
    let inv = [1.0 / (1.0 - coef.bp[0]), 1.0 / (1.0 - coef.bp[1]), 1.0 / (1.0 - coef.bp[2])];
    let (keep, green, lum, sat) = (coef.keep_colour, coef.green, coef.luminance, coef.saturation);
    let blk = coef.black;
    let (hs, knee) = coef.shoulder;
    let iblk = 1.0 / (1.0 - blk);
    let lum_ch = coef.lum.filter(|_| lum > 0.0);
    let active: Vec<usize> = (0..n).filter(|&i| coef.a.iter().any(|row| row[i] != 0.0)).collect();
    for p in 0..npx {
        let mut lin = [coef.k[0], coef.k[1], coef.k[2]];
        for &i in &active {
            let x = ch[i][p] as f64;
            lin[0] += coef.a[0][i] * x;
            lin[1] += coef.a[1][i] * x;
            lin[2] += coef.a[2][i] * x;
        }
        let mut lr = (lin[0] - coef.bp[0]) * inv[0];
        let mut lg = (lin[1] - coef.bp[1]) * inv[1];
        let mut lb = (lin[2] - coef.bp[2]) * inv[2];
        let (mut r, mut g, mut b) = (mtf(coef.m[0], lr), mtf(coef.m[1], lg), mtf(coef.m[2], lb));
        if keep > 0.0 {
            lr = lr.max(0.0);
            lg = lg.max(0.0);
            lb = lb.max(0.0);
            let l = (lr + lg + lb) / 3.0;
            let f = if l > 1.0e-9 { mtf(coef.m[0], l) / l } else { 0.0 };
            let (mut cr, mut cg, mut cb) = (lr * f, lg * f, lb * f);
            let mx = cr.max(cg).max(cb);
            if mx > 1.0 {
                cr /= mx;
                cg /= mx;
                cb /= mx;
            }
            r += keep * (cr - r);
            g += keep * (cg - g);
            b += keep * (cb - b);
        }
        if green > 0.0 {
            let nn = 0.5 * (r + b);
            if g > nn {
                g -= green * (g - nn);
            }
        }
        if let Some((li, ls)) = lum_ch {
            let l = (r + g + b) / 3.0;
            let hl = mtf(ls.m, (ch[li][p] as f64 - ls.bp) / (1.0 - ls.bp));
            let f = (l + lum * (hl - l) + 0.1) / (l + 0.1);
            r *= f;
            g *= f;
            b *= f;
        }
        if sat != 1.0 {
            let l = (r + g + b) / 3.0;
            r = l + sat * (r - l);
            g = l + sat * (g - l);
            b = l + sat * (b - l);
        }
        if blk > 0.0 {
            r = (r - blk) * iblk;
            g = (g - blk) * iblk;
            b = (b - blk) * iblk;
        }
        if hs > 0.0 {
            let m = r.max(g).max(b);
            if m > knee {
                let y = m.min(1.0);
                let f = (y - hs * (y - knee) * (y - knee) / (1.0 - knee)) / m;
                r *= f;
                g *= f;
                b *= f;
            }
        }
        out_r[p] = r.clamp(0.0, 1.0) as f32;
        out_g[p] = g.clamp(0.0, 1.0) as f32;
        out_b[p] = b.clamp(0.0, 1.0) as f32;
    }
}

/// Star colour balance: each channel's gain so the median star has the
/// reference channel's flux. Stars are local maxima of the reference
/// (proxy resolution) well above the noise and below saturation in every
/// channel; flux is a 5×5 sum less the median of the ring 4–6 px out.
pub fn star_gains(proxies: &[&[f32]], w: usize, h: usize, r: usize, sigma: f64) -> Vec<f64> {
    let n = proxies.len();
    let rf = proxies[r];
    let mut ratios: Vec<Vec<f64>> = vec![Vec::new(); n];
    let flux = |img: &[f32], x: usize, y: usize| -> Option<f64> {
        let mut ring = Vec::with_capacity(80);
        for dy in -6i64..=6 {
            for dx in -6i64..=6 {
                let d2 = dx * dx + dy * dy;
                if (16..=36).contains(&d2) {
                    ring.push(img[(y as i64 + dy) as usize * w + (x as i64 + dx) as usize] as f64);
                }
            }
        }
        ring.sort_by(|a, b| a.total_cmp(b));
        let bg = ring[ring.len() / 2];
        let mut s = 0.0;
        for dy in -2i64..=2 {
            for dx in -2i64..=2 {
                let v = img[(y as i64 + dy) as usize * w + (x as i64 + dx) as usize];
                if v >= 0.98 {
                    return None;
                }
                s += v as f64 - bg;
            }
        }
        Some(s)
    };
    let mut found = 0;
    'outer: for y in 8..h.saturating_sub(8) {
        for x in 8..w.saturating_sub(8) {
            let v = rf[y * w + x];
            // Local maximum in a 7×7 window.
            let mut is_max = true;
            for dy in -3i64..=3 {
                for dx in -3i64..=3 {
                    if (dx != 0 || dy != 0) && rf[(y as i64 + dy) as usize * w + (x as i64 + dx) as usize] >= v {
                        is_max = false;
                    }
                }
            }
            if !is_max {
                continue;
            }
            let Some(fr) = flux(rf, x, y) else { continue };
            if fr < 50.0 * sigma {
                continue;
            }
            let fl: Vec<Option<f64>> = proxies.iter().map(|p| flux(p, x, y)).collect();
            if fl.iter().any(|f| f.is_none_or(|v| v <= 0.0)) {
                continue;
            }
            for i in 0..n {
                ratios[i].push(fr / fl[i].unwrap());
            }
            found += 1;
            if found >= 2000 {
                break 'outer;
            }
        }
    }
    ratios
        .into_iter()
        .map(|mut v| {
            if v.len() < 10 {
                return 1.0;
            }
            let m = v.len() / 2;
            v.select_nth_unstable_by(m, |a, b| a.total_cmp(b));
            v[m]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixer_presets_and_stand_ins() {
        let ch = vec!["H".to_string(), "O".to_string()];
        let pr = presets(Kind::Narrowband, &ch);
        assert_eq!(pr.len(), 9);
        // No S: its weights go to H, as the mixer uses Ha in its place.
        let m = mix_of(&pr[0], &ch);
        let h = m.weights["H"];
        assert!((h[0] - 1.7).abs() < 1e-9 && (h[1] - 0.75).abs() < 1e-9);
        assert_eq!(m.weights["O"], [0.0, 0.5, 1.4]);
        assert_eq!(canonical("Luminance"), "L");
        assert_eq!(canonical("HA"), "H");
        assert_eq!(kind_of(&["L".into(), "R".into(), "G".into(), "B".into(), "H".into()]), Kind::Broadband);
    }

    #[test]
    fn highlights_pull_bright_pixels_down_keeping_colour() {
        let ch = vec!["H".to_string()];
        let px: Vec<f32> = (0..4000).map(|i| 0.01 + 0.5 * (i as f32 / 4000.0).powi(3)).collect();
        let s = Stats::measure(&px, &px);
        let mix = Mix { preset: String::new(), balance: Balance::None, weights: [("H".to_string(), [1.0, 0.6, 0.3])].into_iter().collect() };
        let fin = Finishing { luminance: 0.0, green: 0.0, saturation: 1.0, detail: 0.0, chroma: 0.0, neutral: 0.0 };
        let run = |h: f64| {
            let st = Stretching { highlights: h, background: 0.15, ..Default::default() };
            let c = coefficients(&ch, &[&s], None, &mix, &st, &fin);
            let (mut r, mut g, mut b) = (vec![0f32; px.len()], vec![0f32; px.len()], vec![0f32; px.len()]);
            process(&[&px], &c, &mut r, &mut g, &mut b);
            (r, g, b)
        };
        let (r0, g0, _) = run(0.0);
        let (r1, g1, _) = run(1.0);
        let mut changed = 0;
        for i in 0..px.len() {
            // Never brighter; dark pixels (under the background) untouched.
            assert!(r1[i] <= r0[i] + 1e-6);
            if r0[i] < 0.14 {
                assert!((r1[i] - r0[i]).abs() < 1e-6);
            }
            // Hue kept: the red to green ratio is unchanged.
            if r0[i] > 0.3 && r0[i] < 0.99 {
                assert!((g1[i] / r1[i] - g0[i] / r0[i]).abs() < 1e-3);
                changed += 1;
            }
        }
        assert!(changed > 100);
        // The brightest pixel follows the shoulder: y − ½(y − k)²/(1 − k),
        // k the background (0.15).
        let y = r0.iter().cloned().fold(0.0f32, f32::max) as f64;
        let top = r1.iter().cloned().fold(0.0f32, f32::max) as f64;
        let want = y - 0.5 * (y - 0.15).powi(2) / 0.85;
        assert!((top - want).abs() < 0.01, "{top} {want}");
    }

    #[test]
    fn neutral_where_every_line_is_dark() {
        // At each line's dark level all three outputs are equal, for any mix.
        let ch = vec!["H".to_string(), "O".to_string(), "S".to_string()];
        let mk = |base: f32, sig: f32| {
            let px: Vec<f32> = (0..4000).map(|i| base + sig * ((i * 37 % 101) as f32 / 100.0)).collect();
            Stats::measure(&px, &px)
        };
        let stats = vec![mk(0.02, 0.02), mk(0.01, 0.003), mk(0.015, 0.002)];
        let st = Stretching { keep_colour: 0.0, ..Default::default() };
        let fin = Finishing { luminance: 0.0, green: 0.0, saturation: 1.3, detail: 0.0, chroma: 0.0, neutral: 0.0 };
        for pr in presets(Kind::Narrowband, &ch) {
            let mix = mix_of(&pr, &ch);
            let sr: Vec<&Stats> = stats.iter().collect();
            let c = coefficients(&ch, &sr, None, &mix, &st, &fin);
            let d: Vec<f32> = stats.iter().map(|s| quantile_of_sorted(&s.sorted, 0.10) as f32).collect();
            let (h, o, s) = ([d[0]], [d[1]], [d[2]]);
            let (mut r, mut g, mut b) = ([0f32], [0f32], [0f32]);
            process(&[&h, &o, &s], &c, &mut r, &mut g, &mut b);
            assert!((r[0] - g[0]).abs() < 1e-5 && (g[0] - b[0]).abs() < 1e-5, "{}: {r:?} {g:?} {b:?}", pr.name);
        }
    }

    #[test]
    fn frame_filling_targets_stretch_from_the_darkest_structure() {
        let st = Stretching { background: 0.15, ..Default::default() };
        let noise = 1.0e-4;
        // Sky-dominated: MAD sigma = noise, darkest structure 0.5σ under the
        // sky. The mixer's stretch is kept exactly.
        let sky = 0.003;
        let s = auto_stretch(sky, sky - 0.5 * noise, noise, noise, &st);
        assert_eq!(s, make_stretch(sky, noise, &st));
        // Unknown noise: the mixer's stretch, whatever the data.
        assert_eq!(auto_stretch(0.0035, 0.0019, 0.001, 0.0, &st), make_stretch(0.0035, 0.001, &st));
        // NGC 2244 H in ADU/65535: median 227, darkest structure 122, MAD
        // sigma 65, noise 6.7. The old black point (45) is far under every
        // pixel and the median sits at the background level.
        let a = |v: f64| v / 65535.0;
        let old = make_stretch(a(227.0), a(65.0), &st);
        let at = |s: &Stretch, v: f64| mtf(s.m, (v - s.bp) / (1.0 - s.bp));
        assert!((at(&old, a(227.0)) - 0.15).abs() < 1e-9);
        let new = auto_stretch(a(227.0), a(122.0), a(65.0), a(6.7), &st);
        assert!((new.bp - a(122.0 - 2.8 * 6.7)).abs() < 1e-9, "black point {}", new.bp * 65535.0);
        assert!((at(&new, a(122.0)) - LOW_FRACTION * 0.15).abs() < 1e-9);
        let med = at(&new, a(227.0));
        assert!(med > 0.35 && med < 0.5, "median at {med}");
        // Brighter in → brighter out: the background setting still works.
        let brighter = auto_stretch(a(227.0), a(122.0), a(65.0), a(6.7), &Stretching { background: 0.2, ..st.clone() });
        assert!(at(&brighter, a(227.0)) > med);
    }

    #[test]
    fn stretch_is_continuous_where_the_black_point_starts_to_rise() {
        // Move the darkest structure up through the crossover in small
        // steps: the median's output level must change in small steps too.
        let st = Stretching { background: 0.15, ..Default::default() };
        let (med, sigma, noise) = (0.004, 0.0008, 0.0001);
        let at = |s: &Stretch, v: f64| mtf(s.m, (v - s.bp) / (1.0 - s.bp));
        let cross = med - st.clip * sigma + st.clip * noise;
        let mut prev = None;
        for k in -50..=50 {
            let lo = cross + k as f64 * 2.0e-7;
            let y = at(&auto_stretch(med, lo, sigma, noise, &st), med);
            if let Some(p) = prev {
                let step: f64 = y - p;
                assert!(step.abs() < 0.005, "jump of {step:.4} at lo {lo}");
            }
            prev = Some(y);
        }
    }
}
