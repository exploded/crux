//! The Image stage (milestone 13): from a target's integrated masters to a
//! finished picture. Gradient removal per channel (`gradient`), then the
//! palette mixer's colour mix, stretch and finishing (`palette`), natural
//! star colours for narrowband (`stars`), with all the settings in a
//! `recipe`.
//!
//! Previews and preset tiles render from a proxy (longest side
//! `PROXY_MAX`) with exactly the code and constants the full-size render
//! uses; only the pixel count differs.
//!
//! Outputs: `<workspace>\<folder>\image\<object>\<object> <recipe>.xisf`
//! (Float32 RGB, or Gray for one channel), `.png` (16-bit), `.jpg`
//! (8-bit, quality 92), `preview.png` (≤ 1,200 px) and `recipe.toml`,
//! the target's last-used recipe.

#[cfg(feature = "ml")]
pub mod net;
pub mod enhance;
pub mod gradient;
pub mod palette;
pub mod prepcache;
pub mod recipe;
pub mod starnet;
pub mod stars;

use crate::header::Header;
use crate::image::Image;
use anyhow::{Context, Result};
use gradient::{GradientReport, GradientSettings, Model};
use palette::{Kind, Stats};
use recipe::Recipe;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

/// Longest side of the preview proxy (the mixer's PREVIEW_MAX).
pub const PROXY_MAX: usize = 1200;
/// Longest side of a preset tile (the mixer's TILE_MAX).
pub const TILE_MAX: usize = 320;
/// Display order of channels.
const ORDER: [&str; 7] = ["H", "O", "S", "L", "R", "G", "B"];

/// One integrated master.
#[derive(Debug, Clone, PartialEq)]
pub struct Master {
    /// Canonical channel name (H, O, S, L, R, G, B, or the filter as is).
    pub channel: String,
    /// The filter as the headers name it.
    pub filter: String,
    pub path: PathBuf,
    pub frames: usize,
    pub exposure: f64,
    /// The network starless version of this master (`starnet`), when the
    /// star remover is chosen; `None` uses crux's classic fill.
    pub starless: Option<PathBuf>,
    /// Background noise as integrated, in image units (the integration
    /// report's; 0 if unknown). Kept when enhancing changes `path`.
    pub noise: f64,
}

/// A target with its masters, ready for the Image stage.
#[derive(Debug, Clone, PartialEq)]
pub struct Target {
    pub object: String,
    pub masters: Vec<Master>,
}

impl Target {
    pub fn channels(&self) -> Vec<String> {
        self.masters.iter().map(|m| m.channel.clone()).collect()
    }
    pub fn kind(&self) -> Kind {
        palette::kind_of(&self.channels())
    }
    /// Where this target's images go.
    pub fn image_dir(&self, out: &Path) -> PathBuf {
        out.join("image").join(crate::process::job::safe_name(&self.object))
    }
}

/// Targets from a processing output folder's integration and mosaic
/// reports, masters that exist only, channels in H O S L R G B order.
pub fn targets(out: &Path) -> Vec<Target> {
    let reps = crate::process::job::load_integrations(out);
    let mut by: std::collections::BTreeMap<String, Vec<Master>> = Default::default();
    for r in reps.values() {
        if !r.out.is_file() {
            continue;
        }
        by.entry(r.object.clone()).or_default().push(Master {
            channel: palette::canonical(&r.filter),
            filter: r.filter.clone(),
            path: r.out.clone(),
            frames: r.frames.len(),
            exposure: r.total_exposure,
            starless: None,
            noise: if r.noise_master.is_finite() && r.noise_master > 0.0 { r.noise_master / 65535.0 } else { 0.0 },
        });
    }
    // Mosaics (milestone 14): one target per mosaic, named after it.
    for r in crate::process::mosaic::load_reports(out).values() {
        if !r.out.is_file() {
            continue;
        }
        by.entry(r.target.clone()).or_default().push(Master {
            channel: palette::canonical(&r.filter),
            filter: r.filter.clone(),
            path: r.out.clone(),
            frames: r.frames,
            exposure: r.exposure,
            starless: None,
            noise: 0.0,
        });
    }
    by.into_iter()
        .map(|(object, mut masters)| {
            masters.sort_by_key(|m| ORDER.iter().position(|c| *c == m.channel).unwrap_or(99));
            masters.dedup_by(|a, b| a.channel == b.channel);
            Target { object, masters }
        })
        .collect()
}

/// Masters stacked by another program (PixInsight WBPP's `masterLight_*`),
/// read in place: `dir` itself or its `master` subfolder. One target named
/// `object`; the channel from FILTER (else the file name's `FILTER-X`), the
/// noise from NOISE00 (ImageIntegration's estimate, image units). Cropped
/// copies (`_autocrop`), drizzled masters and rejection maps are skipped:
/// every master kept shares the registration reference's geometry.
pub fn targets_from_masters(dir: &Path, object: &str) -> Vec<Target> {
    let sub = dir.join("master");
    let dir = if sub.is_dir() { sub } else { dir.to_path_buf() };
    let mut masters: Vec<Master> = Vec::new();
    let Ok(rd) = std::fs::read_dir(&dir) else { return Vec::new() };
    let mut files: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_file()).collect();
    files.sort();
    for p in files {
        let name = p.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
        if !name.starts_with("masterlight") || ["autocrop", "drizzle", "rejection", "slope", "weight"].iter().any(|s| name.contains(s)) {
            continue;
        }
        let header = match crate::meta::format_of(&p) {
            Some(crate::meta::FileFormat::Xisf) => crate::xisf::read_layout(&p).map(|l| l.header),
            Some(crate::meta::FileFormat::Fits) => crate::fits::read_layout(&p).map(|l| l.header),
            _ => continue,
        };
        let Ok(h) = header else { continue };
        let filter = h.str("FILTER").filter(|f| !f.trim().is_empty()).or_else(|| {
            let i = name.find("filter-")? + 7;
            Some(name[i..].split(['_', '.']).next()?.to_uppercase())
        });
        let Some(filter) = filter else { continue };
        let noise = h.f64("NOISE00").filter(|v| v.is_finite() && *v > 0.0 && *v < 0.1).unwrap_or(0.0);
        masters.push(Master {
            channel: palette::canonical(&filter),
            filter,
            path: p.clone(),
            frames: h.i64("NCOMBINE").unwrap_or(0).max(0) as usize,
            exposure: h.f64("EXPTIME").unwrap_or(0.0),
            starless: None,
            noise,
        });
    }
    if masters.is_empty() {
        return Vec::new();
    }
    masters.sort_by_key(|m| ORDER.iter().position(|c| *c == m.channel).unwrap_or(99));
    masters.dedup_by(|a, b| a.channel == b.channel);
    vec![Target { object: object.to_string(), masters }]
}

/// Block mean down to `max` on the longer side; blocks more than half
/// outside the frame (zero) stay zero, like registration borders.
pub fn proxy(img: &Image, max: usize) -> Image {
    let f = img.width.max(img.height).div_ceil(max).max(1);
    let (w, h, d) = gradient::bin_valid(img, f);
    Image::from_data(w, h, d.into_iter().map(|v| if v.is_finite() { v } else { 0.0 }).collect())
}

/// One channel after gradient removal.
pub struct Channel {
    pub name: String,
    pub master: Master,
    /// Full resolution (kept only for the final render).
    pub full: Option<Image>,
    /// Proxy before and after the gradient correction.
    pub raw_proxy: Image,
    pub proxy: Image,
    pub stats: Stats,
    pub model: Option<Model>,
    pub report: Option<GradientReport>,
    /// With stars removed (narrowband, natural star colours): full
    /// resolution when kept, the proxy, and its statistics.
    pub starless: Option<Image>,
    pub starless_proxy: Option<Image>,
    pub stats_starless: Option<Stats>,
    /// A 1:1 crop (`Prepared::crop`), with and without stars.
    pub crop: Image,
    pub crop_starless: Option<Image>,
    /// Noise σ at full resolution and in the proxy.
    pub noise_full: f32,
    pub noise_proxy: f32,
}

/// A target's channels, measured and corrected.
pub struct Prepared {
    pub object: String,
    pub kind: Kind,
    pub channels: Vec<Channel>,
    pub width: usize,
    pub height: usize,
    /// Star colour balance gains (1 where not measured).
    pub star_gain: Vec<f64>,
    /// The gradient settings this was prepared with.
    pub gradient: GradientSettings,
    /// The star settings, the stars masked, and the share of the frame
    /// the mask covers.
    pub stars: stars::StarSettings,
    pub n_stars: usize,
    pub star_cover: f64,
    /// Edges covered by few frames (full-resolution blocks), for the auto crop.
    pub thin: Option<Thin>,
    /// The 1:1 crop: x, y, width, height.
    pub crop: (usize, usize, usize, usize),
    /// Each star's light and confidence, and that painted for the proxy,
    /// the crop and (when kept) full resolution.
    pub star_info: Vec<stars::StarInfo>,
    pub paint_proxy: Option<stars::Paint>,
    pub paint_crop: Option<stars::Paint>,
    pub paint_full: Option<stars::Paint>,
}

impl Prepared {
    pub fn names(&self) -> Vec<String> {
        self.channels.iter().map(|c| c.name.clone()).collect()
    }
}

/// Can this target's stars be handled on their own (narrowband with OIII
/// and Ha or SII, broadband with R, G and B: stars from their own light;
/// one channel: grey stars, stretched separately)?
pub fn stars_possible(names: &[String]) -> bool {
    match palette::kind_of(names) {
        Kind::Narrowband => names.iter().any(|c| c == "O") && names.iter().any(|c| c == "H" || c == "S"),
        Kind::Broadband => true,
        Kind::Mono => names.len() == 1,
    }
}

/// The target as a recipe wants it: masters sharpened and denoised
/// (`enhance`) and, for natural star colours with the star remover, their
/// starless versions (`starnet`). Both are cached on disk; `progress` gets
/// a short line now and then.
pub fn for_recipe(t: &Target, out: &Path, r: &Recipe, progress: &(dyn Fn(&str) + Sync), cancel: &AtomicBool) -> Result<Target> {
    let passes = Passes::new(t, &r.enhance, &r.stars);
    let te = enhance::apply(t, out, &r.enhance, &|p| progress(&passes.enhance_line(&p)), cancel)?;
    if passes.starless {
        let names = te.channels();
        starnet::apply(&te, out, &|i, _, f| progress(&passes.starless_line(i, names.get(i).map(String::as_str).unwrap_or(""), f)), cancel)
    } else {
        Ok(te)
    }
}

/// Whether `for_recipe` runs the star remover for these settings.
pub fn wants_starless(t: &Target, st: &stars::StarSettings) -> bool {
    st.engine == stars::StarEngine::Network && st.mode == stars::StarMode::Natural && stars_possible(&t.channels())
}

/// Progress lines for `for_recipe`'s network passes, numbered across every
/// channel ("Pass 4 of 9 · O: correcting stars 40%"). Each step runs one
/// channel at a time, so "H 1 of 3" alone started again at every step.
/// Passes found in the cache are skipped, so the count can jump ahead.
pub struct Passes {
    masters: usize,
    /// The enhance steps each master goes through, in order.
    steps: Vec<&'static str>,
    pub starless: bool,
}

impl Passes {
    pub fn new(t: &Target, e: &enhance::EnhanceSettings, st: &stars::StarSettings) -> Passes {
        use enhance::{Engine, Sharpen};
        let mut steps = Vec::new();
        if e.active() {
            match e.engine {
                Engine::Crux => {
                    // The denoiser runs even for sharpening alone: the blend needs it.
                    steps.push("denoise");
                    match e.sharpen {
                        Sharpen::Off => {}
                        Sharpen::Correct => steps.push("correct"),
                        Sharpen::Tighten => steps.push("tighten"),
                    }
                }
                Engine::RcAstro => {
                    if e.sharpen != Sharpen::Off {
                        steps.push("BlurXTerminator");
                    }
                    if e.denoise {
                        steps.push("NoiseXTerminator");
                    }
                }
            }
        }
        Passes { masters: t.masters.len(), steps, starless: wants_starless(t, st) }
    }

    pub fn total(&self) -> usize {
        self.masters * (self.steps.len() + self.starless as usize)
    }

    fn line(&self, pass: usize, channel: &str, what: &str, done: f32) -> String {
        format!("Pass {} of {} · {channel}: {what} {:.0}%", pass.min(self.total()).max(1), self.total(), done * 100.0)
    }

    pub fn enhance_line(&self, p: &enhance::Progress) -> String {
        let per = self.steps.len();
        // Blending ends the master's last pass.
        let k = self.steps.iter().position(|s| *s == p.step).unwrap_or(per.saturating_sub(1));
        self.line(p.master * per + k + 1, &p.channel, &step_word(&p.step), p.done)
    }

    pub fn starless_line(&self, master: usize, channel: &str, done: f32) -> String {
        self.line(self.masters * self.steps.len() + master + 1, channel, "removing stars", done)
    }
}

/// A network step as the progress line says it.
pub fn step_word(s: &str) -> String {
    match s {
        "denoise" => "removing noise".into(),
        "correct" => "correcting stars".into(),
        "tighten" => "sharpening".into(),
        "blend" => "blending".into(),
        other => other.to_string(),
    }
}

/// Read and correct a target's masters. `keep_full` keeps the
/// full-resolution channels for a final render.
pub fn prepare(t: &Target, g: &GradientSettings, st: &stars::StarSettings, keep_full: bool, cancel: &AtomicBool) -> Result<Prepared> {
    anyhow::ensure!(!t.masters.is_empty(), "{} has no masters", t.object);
    let timing = std::env::var_os("CRUX_TIMING").is_some();
    let t0 = std::time::Instant::now();
    let lap = |what: &str| {
        if timing {
            eprintln!("prepare {}: {what} at {:.2} s", t.object, t0.elapsed().as_secs_f64());
        }
    };
    let mut channels = Vec::new();
    let mut size = None;
    let mut fulls = Vec::new();
    // Following the sky closely is for broadband (galaxies, clusters, star
    // fields; narrowband frames are mostly nebula), sampled on the star
    // remover's starless masters when there are some (in a star field a
    // box's median follows how many faint stars it holds) and on a finer
    // grid.
    let names0: Vec<String> = t.masters.iter().map(|m| m.channel.clone()).collect();
    let follow = g.enabled && g.follow_sky && palette::kind_of(&names0) == Kind::Broadband && st.engine == stars::StarEngine::Network && t.masters.iter().all(|m| m.starless.is_some());
    for m in &t.masters {
        if cancel.load(Ordering::Relaxed) {
            anyhow::bail!("cancelled");
        }
        let img = crate::xisf::read_image(&m.path).with_context(|| format!("read {}", m.path.display()))?;
        match size {
            None => size = Some((img.width, img.height)),
            Some(s) => anyhow::ensure!(s == (img.width, img.height), "{} is {}×{}, the other masters {}×{}: they must be registered to one reference", m.path.display(), img.width, img.height, s.0, s.1),
        }
        let smp = if follow {
            let sl = m.starless.as_ref().and_then(|p| crate::read_image(p).ok()).filter(|s| (s.width, s.height) == (img.width, img.height));
            Some(gradient::samples(sl.as_ref().unwrap_or(&img), 2 * g.samples))
        } else {
            g.enabled.then(|| gradient::samples(&img, g.samples))
        };
        fulls.push((m, img, smp));
    }
    lap("read and sampled");
    // One degree for every channel.
    let degree = if g.degree > 0 {
        g.degree
    } else {
        let mut d: Vec<u32> = fulls.iter().filter_map(|(_, _, s)| s.as_ref().map(|s| gradient::auto_degree(s, g.tolerance))).collect();
        d.sort();
        d.get(d.len().saturating_sub(1) / 2).copied().unwrap_or(1)
    };
    let (width, height) = size.unwrap_or((0, 0));
    let gfit = &GradientSettings { follow_sky: follow, ..g.clone() };
    let mut corrected = Vec::new();
    for (m, mut img, smp) in fulls {
        let raw_proxy = proxy(&img, PROXY_MAX);
        let fit = smp.and_then(|s| gradient::fit_samples(&s, gfit, degree));
        if let Some((md, _)) = &fit {
            md.apply(&mut img);
        }
        corrected.push((m, img, raw_proxy, fit));
    }
    lap("gradient fitted");
    let names: Vec<String> = corrected.iter().map(|c| c.0.channel.clone()).collect();
    let kind = palette::kind_of(&names);
    // Stars: one mask from every channel's stars.
    let with_stars = st.mode == stars::StarMode::Natural && stars_possible(&names);
    let (mask, discs) = if with_stars {
        // Every channel at once; collected in channel order, so the merge
        // is the same as one after another.
        let found: Vec<Vec<_>> = {
            use rayon::prelude::*;
            corrected.par_iter().map(|c| stars::detect(&c.1, st)).collect()
        };
        let discs = stars::dedupe(found.into_iter().flatten().collect());
        (Some(stars::mask(&discs, width, height)), discs)
    } else {
        (None, Vec::new())
    };
    let n_stars = discs.len();
    lap("stars found");
    let mut fulls_kept: Vec<Image> = Vec::new();
    let mut starless_kept: Vec<Image> = Vec::new();
    let star_cover = mask.as_ref().map(|m| m.iter().filter(|v| **v > 0.5).count() as f64 / m.len().max(1) as f64).unwrap_or(0.0);
    // The 1:1 crop: the centre, up to 1,000 × 700.
    let (cw, ch) = (width.min(1000), height.min(700));
    let crop_rect = ((width - cw) / 2, (height - ch) / 2, cw, ch);
    let cut = |img: &Image| -> Image {
        let (x0, y0, w, h) = crop_rect;
        let mut d = Vec::with_capacity(w * h);
        for y in y0..y0 + h {
            d.extend_from_slice(&img.data[y * img.width + x0..y * img.width + x0 + w]);
        }
        Image::from_data(w, h, d)
    };
    // Each channel's statistics, noise and starless copy, every channel at
    // once (one after another this was 8.6 s of IC 434's 13 s).
    // First every channel's statistics and starless copy, then the stars
    // whose glows are nebula in any channel (decided together, so a halo
    // never loses its glow in some channels only), then the glows out.
    let first: Vec<_> = {
        use rayon::prelude::*;
        corrected
            .into_par_iter()
            .enumerate()
            .map(|(k, (m, img, raw_proxy, fit))| {
                let (model, report) = match fit {
                    Some((md, rp)) => (Some(md), Some(rp)),
                    None => (None, None),
                };
                let laps = crate::timing::Laps::new(format!("prepare {}", m.channel));
                let px = if model.is_some() { proxy(&img, PROXY_MAX) } else { raw_proxy.clone() };
                laps.lap("proxy");
                let mut stats = Stats::measure(&img.data, &px.data);
                stats.sky_noise = m.noise;
                laps.lap("stats");
                let noise_full = crate::process::combine::noise_of(&img);
                let noise_proxy = crate::process::combine::noise_of(&px);
                laps.lap("noise");
                // The star remover's starless master when chosen (made by
                // `starnet`), corrected like the master; else the classic fill.
                let net_sl = (st.engine == stars::StarEngine::Network)
                    .then(|| m.starless.as_ref().and_then(|p| crate::read_image(p).ok()))
                    .flatten()
                    .filter(|s| (s.width, s.height) == (img.width, img.height))
                    .map(|mut s| {
                        if let Some(md) = &model {
                            md.apply(&mut s);
                        }
                        s
                    });
                let sl = mask.as_ref().map(|mk| net_sl.unwrap_or_else(|| stars::starless(&img, mk, noise_full, k as u64 + 1)));
                laps.lap("starless");
                let nebula = sl.as_ref().map(|s| stars::glow_nebula(s, &img, &discs, noise_full)).unwrap_or_default();
                laps.lap("glows around nebula");
                (m, img, raw_proxy, model, report, px, stats, noise_full, noise_proxy, sl, nebula)
            })
            .collect()
    };
    let mut keep = vec![false; discs.len()];
    for f in &first {
        for (k, v) in keep.iter_mut().zip(&f.10) {
            *k |= *v;
        }
    }
    let per_channel: Vec<_> = {
        use rayon::prelude::*;
        first
            .into_par_iter()
            .map(|(m, img, raw_proxy, model, report, px, stats, noise_full, noise_proxy, sl, _)| {
                let laps = crate::timing::Laps::new(format!("prepare {}", m.channel));
                let sl = sl.map(|mut s| {
                    stars::remove_glows(&mut s, &img, &discs, noise_full, &keep);
                    s
                });
                laps.lap("glows removed");
                let sl_proxy = sl.as_ref().map(|s| proxy(s, PROXY_MAX));
                let stats_starless = sl.as_ref().zip(sl_proxy.as_ref()).map(|(s, sp)| Stats { sky_noise: m.noise, ..Stats::measure(&s.data, &sp.data) });
                laps.lap("starless stats");
                let crop = cut(&img);
                let crop_starless = sl.as_ref().map(&cut);
                let thin = thin_edges(&img);
                laps.lap("thin edges");
                (m, img, raw_proxy, model, report, px, stats, noise_full, noise_proxy, sl, sl_proxy, stats_starless, crop, crop_starless, thin)
            })
            .collect()
    };
    let mut thin: Option<Thin> = None;
    for (m, img, raw_proxy, model, report, px, stats, noise_full, noise_proxy, sl, sl_proxy, stats_starless, crop, crop_starless, th) in per_channel {
        thin = match (thin, th) {
            (Some(mut a), Some(b)) if (a.w, a.h) == (b.w, b.h) => {
                a.blocks.iter_mut().zip(&b.blocks).for_each(|(x, y)| *x |= *y);
                Some(a)
            }
            (a, b) => a.or(b),
        };
        // Star light is measured on full resolution, for every channel.
        let (full, sl) = if mask.is_some() {
            fulls_kept.push(img);
            starless_kept.push(sl.unwrap());
            (None, None)
        } else {
            (keep_full.then_some(img), None)
        };
        channels.push(Channel {
            name: m.channel.clone(),
            master: m.clone(),
            full,
            raw_proxy,
            proxy: px,
            stats,
            model,
            report,
            starless: sl,
            starless_proxy: sl_proxy,
            stats_starless,
            crop,
            crop_starless,
            noise_full,
            noise_proxy,
        });
    }
    lap("stats, noise and starless");
    // Each star's integrated light; paint it for each render size.
    let (mut star_info, mut paint_proxy, mut paint_crop, mut paint_full) = (Vec::new(), None, None, None);
    if !fulls_kept.is_empty() {
        let hi = names.iter().position(|c| c == "H").or(names.iter().position(|c| c == "S")).unwrap_or(0);
        let with: Vec<&Image> = fulls_kept.iter().collect();
        let without: Vec<&Image> = starless_kept.iter().collect();
        let laps = crate::timing::Laps::new("prepare stars");
        star_info = stars::measure(&discs, &with, &without, hi, channels[hi].noise_full);
        laps.lap("measured");
        let n = channels.len();
        let (pw, ph) = (channels[0].proxy.width, channels[0].proxy.height);
        paint_proxy = Some(stars::paint(&star_info, n, pw, ph, pw as f32 / width as f32, 0.0, 0.0));
        let (x0, y0, cw, ch) = crop_rect;
        paint_crop = Some(stars::paint(&star_info, n, cw, ch, 1.0, x0 as f32, y0 as f32));
        laps.lap("painted at proxy and crop size");
        if keep_full {
            paint_full = Some(stars::paint(&star_info, n, width, height, 1.0, 0.0, 0.0));
            laps.lap("painted at full size");
            for ((c, f), s) in channels.iter_mut().zip(fulls_kept).zip(starless_kept) {
                c.full = Some(f);
                c.starless = Some(s);
            }
        }
    }
    lap("stars measured and painted");
    let star_gain = if channels.len() >= 2 {
        let r = palette::reference(&names, kind);
        let (pw, ph) = (channels[0].proxy.width, channels[0].proxy.height);
        let refs = &channels[r].proxy.data;
        let mut s: Vec<f32> = refs.iter().copied().filter(|v| *v > palette::ZERO_FLOOR).collect();
        let (_, sigma) = crate::stats::median_sigma_mut(&mut s);
        let proxies: Vec<&[f32]> = channels.iter().map(|c| c.proxy.data.as_slice()).collect();
        palette::star_gains(&proxies, pw, ph, r, sigma as f64)
    } else {
        vec![1.0; channels.len()]
    };
    Ok(Prepared { object: t.object.clone(), kind, channels, width, height, star_gain, gradient: g.clone(), stars: st.clone(), n_stars, star_cover, thin, crop: crop_rect, star_info, paint_proxy, paint_crop, paint_full })
}

/// An RGB render.
#[derive(Clone)]
pub struct Rgb {
    pub width: usize,
    pub height: usize,
    pub r: Vec<f32>,
    pub g: Vec<f32>,
    pub b: Vec<f32>,
}

impl Rgb {
    pub fn is_grey(&self) -> bool {
        self.r.iter().zip(&self.g).zip(&self.b).all(|((r, g), b)| r == g && g == b)
    }

    /// The part a crop keeps.
    pub fn cropped(&self, c: &recipe::Crop) -> Rgb {
        let (x0, y0, w, h) = c.pixels(self.width, self.height);
        let cut = |src: &[f32]| -> Vec<f32> {
            let mut d = Vec::with_capacity(w * h);
            for y in y0..y0 + h {
                d.extend_from_slice(&src[y * self.width + x0..y * self.width + x0 + w]);
            }
            d
        };
        Rgb { width: w, height: h, r: cut(&self.r), g: cut(&self.g), b: cut(&self.b) }
    }

    /// Turned `q` quarter turns clockwise.
    pub fn rotated(self, q: u8) -> Rgb {
        use rayon::prelude::*;
        let q = q % 4;
        if q == 0 {
            return self;
        }
        let (w, h) = (self.width, self.height);
        let (nw, nh) = if q == 2 { (w, h) } else { (h, w) };
        // Source pixel for output (x, y).
        let src = move |x: usize, y: usize| -> usize {
            match q {
                1 => (h - 1 - x) * w + y,
                2 => (h - 1 - y) * w + (w - 1 - x),
                _ => x * w + (w - 1 - y),
            }
        };
        let turn = |s: &[f32]| -> Vec<f32> {
            let mut d = vec![0.0f32; nw * nh];
            d.par_chunks_mut(nw.max(1)).enumerate().for_each(|(y, row)| {
                for (x, v) in row.iter_mut().enumerate() {
                    *v = s[src(x, y)];
                }
            });
            d
        };
        Rgb { width: nw, height: nh, r: turn(&self.r), g: turn(&self.g), b: turn(&self.b) }
    }

    /// 8-bit RGBA, for the GUI.
    pub fn rgba8(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(self.r.len() * 4);
        for i in 0..self.r.len() {
            v.extend_from_slice(&[(self.r[i] * 255.0).round() as u8, (self.g[i] * 255.0).round() as u8, (self.b[i] * 255.0).round() as u8, 255]);
        }
        v
    }
}

/// Are natural star colours on for this render (and possible)?
pub fn stars_on(p: &Prepared, r: &Recipe) -> bool {
    r.stars.mode == stars::StarMode::Natural && p.channels.iter().all(|c| c.starless_proxy.is_some())
}

/// The coefficients a recipe gives for these channels, from the original
/// statistics: the whole picture when stars are kept or recoloured in
/// place, and the colour of light no measured star explains when they're
/// stretched separately.
pub fn coefficients(p: &Prepared, r: &Recipe) -> palette::Coef {
    let stats: Vec<&Stats> = p.channels.iter().map(|c| &c.stats).collect();
    palette::coefficients(&p.names(), &stats, Some(&p.star_gain), &r.mix, &r.stretch, &r.finish)
}

/// Are the starless image and the stars stretched separately?
pub fn separate_on(p: &Prepared, r: &Recipe) -> bool {
    r.stars.separate && stars_on(p, r) && p.channels.iter().all(|c| c.stats_starless.is_some())
}

/// The starless image's coefficients when stretched separately: the same
/// mix, stretch and finishing from the starless statistics, so the nebula
/// is stretched as nebula (stars no longer lift the signal level or the
/// spread). None otherwise.
pub fn coefficients_starless(p: &Prepared, r: &Recipe) -> Option<palette::Coef> {
    if !separate_on(p, r) {
        return None;
    }
    let stats: Vec<&Stats> = p.channels.iter().map(|c| c.stats_starless.as_ref()).collect::<Option<_>>()?;
    Some(palette::coefficients(&p.names(), &stats, Some(&p.star_gain), &r.mix, &r.stretch, &r.finish))
}

/// The star layer below this many of its own noise sigmas is left out, so
/// the starless image's small residuals don't speckle the stars' stretch.
pub const STAR_FLOOR_SIGMAS: f32 = stars::STAR_FLOOR_SIGMAS as f32;

/// What the separate stretch needs for one render.
pub struct Split {
    pub coef: palette::Coef,
    pub curve: stars::Curve,
    /// Per channel: the star layer's floor at this render's resolution.
    pub floor: Vec<f32>,
    /// Nebula detail for this render, when asked for.
    pub detail: Option<Detail>,
    /// Colour noise reduction for this render, when asked for.
    pub chroma: Option<Chroma>,
    /// Neutral background, when asked for.
    pub neutral: Option<Neutral>,
}

/// Nebula detail: a band-pass of the stretched starless brightness, made
/// once from the proxy (fine noise and the picture's overall shape left
/// out) and sampled at each rendered pixel's place in the full frame, so
/// the preview, the 1:1 crop and the full render agree.
pub struct Detail {
    map: Vec<f32>,
    mw: usize,
    mh: usize,
    /// Rendered pixel to map pixel: `m = (o + x·s) / f`.
    sx: f32,
    sy: f32,
    ox: f32,
    oy: f32,
    f: f32,
    amount: f32,
}

impl Detail {
    #[inline]
    fn at(&self, x: usize, y: usize) -> f32 {
        let mx = ((self.ox + (x as f32 + 0.5) * self.sx) / self.f - 0.5).clamp(0.0, (self.mw - 1) as f32);
        let my = ((self.oy + (y as f32 + 0.5) * self.sy) / self.f - 0.5).clamp(0.0, (self.mh - 1) as f32);
        let (x0, y0) = (mx as usize, my as usize);
        let (x1, y1) = ((x0 + 1).min(self.mw - 1), (y0 + 1).min(self.mh - 1));
        let (fx, fy) = (mx - x0 as f32, my - y0 as f32);
        let m = &self.map;
        let top = m[y0 * self.mw + x0] * (1.0 - fx) + m[y0 * self.mw + x1] * fx;
        let bot = m[y1 * self.mw + x0] * (1.0 - fx) + m[y1 * self.mw + x1] * fx;
        top * (1.0 - fy) + bot * fy
    }

    /// Lift or lower one starless pixel's brightness by its local detail,
    /// keeping its colour.
    #[inline]
    fn apply(&self, x: usize, y: usize, rgb: [f32; 3]) -> [f32; 3] {
        let l = (rgb[0] + rgb[1] + rgb[2]) / 3.0;
        if l <= 1.0e-4 {
            return rgb;
        }
        let k = ((l + self.amount * self.at(x, y)) / l).clamp(0.0, 3.0);
        rgb.map(|v| (v * k).clamp(0.0, 1.0))
    }
}

/// Where a render's pixels sit in the full frame: `X = ox + x·sx`.
#[derive(Clone, Copy)]
struct Frame {
    sx: f32,
    sy: f32,
    ox: f32,
    oy: f32,
}

impl Frame {
    /// A render of the whole frame at `w` by `h`.
    fn whole(p: &Prepared, w: usize, h: usize) -> Frame {
        Frame { sx: p.width as f32 / w.max(1) as f32, sy: p.height as f32 / h.max(1) as f32, ox: 0.0, oy: 0.0 }
    }
}

/// Detail band: fine and coarse scales, as fractions of the frame width.
const DETAIL_FINE: f32 = 0.0025;
const DETAIL_COARSE: f32 = 0.02;

/// Box blur of radius `r` along one line, edges extended.
fn box_line(src: &[f32], r: usize, out: &mut [f32]) {
    let n = src.len() as isize;
    let at = |i: isize| src[i.clamp(0, n - 1) as usize];
    let win = (2 * r + 1) as f32;
    let r = r as isize;
    let mut s: f32 = (-r..=r).map(at).sum();
    for i in 0..n {
        out[i as usize] = s / win;
        s += at(i + r + 1) - at(i - r);
    }
}

/// A Gaussian of `sigma` pixels from three box blurs, rows then columns.
fn gauss3(a: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    use rayon::prelude::*;
    // Three boxes of width d have variance 3·(d² − 1)/12 = σ².
    let r = (((4.0 * sigma * sigma + 1.0).sqrt() - 1.0) / 2.0).round().max(0.0) as usize;
    let mut out = a.to_vec();
    if r == 0 {
        return out;
    }
    out.par_chunks_mut(w).for_each(|row| {
        let mut tmp = row.to_vec();
        for _ in 0..3 {
            box_line(&tmp.clone(), r, &mut tmp);
        }
        row.copy_from_slice(&tmp);
    });
    let cols: Vec<Vec<f32>> = (0..w)
        .into_par_iter()
        .map(|x| {
            let mut c: Vec<f32> = (0..h).map(|y| out[y * w + x]).collect();
            for _ in 0..3 {
                box_line(&c.clone(), r, &mut c);
            }
            c
        })
        .collect();
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, v) in row.iter_mut().enumerate() {
            *v = cols[x][y];
        }
    });
    out
}

/// Colour noise reduction: the stretched starless image's colour smoothed
/// on the proxy (blurred R, G, B over blurred brightness), sampled at each
/// rendered pixel's place like [`Detail`]. Faint pixels take it; bright
/// nebula keeps its own colour.
pub struct Chroma {
    /// Smoothed R, G and B as fractions of the brightness (mean 1).
    maps: [Vec<f32>; 3],
    mw: usize,
    mh: usize,
    sx: f32,
    sy: f32,
    ox: f32,
    oy: f32,
    f: f32,
    amount: f32,
}

/// Colour smoothing scale, a fraction of the proxy's width.
const CHROMA_SIGMA: f32 = 0.004;

impl Chroma {
    #[inline]
    fn apply(&self, x: usize, y: usize, rgb: [f32; 3]) -> [f32; 3] {
        let l = (rgb[0] + rgb[1] + rgb[2]) / 3.0;
        if l <= 1.0e-4 {
            return rgb;
        }
        // Full strength up to a quarter brightness, none above 0.6.
        let t = ((l - 0.25) / 0.35).clamp(0.0, 1.0);
        let k = self.amount * (1.0 - t * t * (3.0 - 2.0 * t));
        if k <= 0.0 {
            return rgb;
        }
        let mx = ((self.ox + (x as f32 + 0.5) * self.sx) / self.f - 0.5).clamp(0.0, (self.mw - 1) as f32);
        let my = ((self.oy + (y as f32 + 0.5) * self.sy) / self.f - 0.5).clamp(0.0, (self.mh - 1) as f32);
        let (x0, y0) = (mx as usize, my as usize);
        let (x1, y1) = ((x0 + 1).min(self.mw - 1), (y0 + 1).min(self.mh - 1));
        let (fx, fy) = (mx - x0 as f32, my - y0 as f32);
        std::array::from_fn(|c| {
            let m = &self.maps[c];
            let top = m[y0 * self.mw + x0] * (1.0 - fx) + m[y0 * self.mw + x1] * fx;
            let bot = m[y1 * self.mw + x0] * (1.0 - fx) + m[y1 * self.mw + x1] * fx;
            let smooth = l * (top * (1.0 - fy) + bot * fy);
            (rgb[c] * (1.0 - k) + smooth * k).clamp(0.0, 1.0)
        })
    }
}

/// Neutral background: the faint sky's colour cast, measured region by
/// region (a palette's boosted weak lines tint the faint sky, more towards
/// the edges where the flats or the gradient fit leave a little), taken out
/// in full up to twice the dark pixels' brightness and fading to nothing by
/// four times (at most 0.6), so nebula keeps its colour.
pub struct Neutral {
    /// Per proxy pixel: the dark sky's colour less its mean (sums to zero,
    /// so brightness doesn't change), smoothed over a large scale.
    cast: [Vec<f32>; 3],
    mw: usize,
    mh: usize,
    sx: f32,
    sy: f32,
    ox: f32,
    oy: f32,
    f: f32,
    level: f32,
    amount: f32,
}

/// The darker share of the starless proxy taken as sky, and the scale its
/// colour is smoothed over (a fraction of the width).
const NEUTRAL_DARK: f64 = 0.4;
const NEUTRAL_SIGMA: f32 = 0.06;
/// Where the share of dark pixels nearby falls towards this, the sky colour
/// blends smoothly into the whole picture's.
const NEUTRAL_PRIOR: f32 = 0.05;

impl Neutral {
    #[inline]
    fn apply(&self, x: usize, y: usize, rgb: [f32; 3]) -> [f32; 3] {
        let l = (rgb[0] + rgb[1] + rgb[2]) / 3.0;
        let (a, b) = (2.0 * self.level, (4.0 * self.level).clamp(2.0 * self.level + 0.05, 0.6));
        let t = ((l - a) / (b - a)).clamp(0.0, 1.0);
        let k = self.amount * (1.0 - t * t * (3.0 - 2.0 * t)) * (l / self.level.max(1.0e-4)).min(1.0);
        if k <= 0.0 {
            return rgb;
        }
        let mx = ((self.ox + (x as f32 + 0.5) * self.sx) / self.f - 0.5).clamp(0.0, (self.mw - 1) as f32);
        let my = ((self.oy + (y as f32 + 0.5) * self.sy) / self.f - 0.5).clamp(0.0, (self.mh - 1) as f32);
        let (x0, y0) = (mx as usize, my as usize);
        let (x1, y1) = ((x0 + 1).min(self.mw - 1), (y0 + 1).min(self.mh - 1));
        let (fx, fy) = (mx - x0 as f32, my - y0 as f32);
        std::array::from_fn(|c| {
            let m = &self.cast[c];
            let top = m[y0 * self.mw + x0] * (1.0 - fx) + m[y0 * self.mw + x1] * fx;
            let bot = m[y1 * self.mw + x0] * (1.0 - fx) + m[y1 * self.mw + x1] * fx;
            (rgb[c] - k * (top * (1.0 - fy) + bot * fy)).clamp(0.0, 1.0)
        })
    }
}

fn neutral_of(p: &Prepared, r: &Recipe, coef: &palette::Coef, frame: Frame) -> Option<Neutral> {
    // Narrowband only: a broadband sky's leftover tint follows its
    // brightness gradient (NGC 2997: the corners turned teal).
    if r.finish.neutral <= 0.0 || p.kind != Kind::Narrowband {
        return None;
    }
    let sp: Vec<&[f32]> = p.channels.iter().map(|c| c.starless_proxy.as_ref().map(|i| i.data.as_slice())).collect::<Option<_>>()?;
    let first = p.channels[0].starless_proxy.as_ref()?;
    let (mw, mh) = (first.width, first.height);
    let n = mw * mh;
    let (mut rr, mut gg, mut bb) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
    palette::process_par(&sp, coef, &mut rr, &mut gg, &mut bb);
    let lum: Vec<f32> = (0..n).map(|i| (rr[i] + gg[i] + bb[i]) / 3.0).collect();
    // Pixels with data (crop edges and empty corners are zero).
    let mut ls: Vec<f32> = lum.iter().copied().filter(|l| *l > 0.005).collect();
    if ls.len() < 100 {
        return None;
    }
    let cut = ((ls.len() as f64 * NEUTRAL_DARK) as usize).min(ls.len() - 1);
    let (_, thr, _) = ls.select_nth_unstable_by(cut, |a, b| a.total_cmp(b));
    let thr = *thr;
    let dark: Vec<f32> = lum.iter().map(|l| if *l > 0.005 && *l <= thr { 1.0 } else { 0.0 }).collect();
    let level = {
        let (s, c) = lum.iter().zip(&dark).fold((0.0f64, 0.0f64), |(s, c), (l, d)| (s + (*l * *d) as f64, c + *d as f64));
        (s / c.max(1.0)) as f32
    };
    let sigma = NEUTRAL_SIGMA * mw as f32;
    // The four blurs side by side (each is short of work for every core).
    let masked = |c: &[f32]| -> Vec<f32> { c.iter().zip(&dark).map(|(v, d)| v * d).collect() };
    let ((wsum, br), (bg, bb2)) = rayon::join(
        || rayon::join(|| gauss3(&dark, mw, mh, sigma), || gauss3(&masked(&rr), mw, mh, sigma)),
        || rayon::join(|| gauss3(&masked(&gg), mw, mh, sigma), || gauss3(&masked(&bb), mw, mh, sigma)),
    );
    // Far from dark sky (inside a bright nebula) the local mean leans on
    // the whole picture's dark sky instead: a hard switch there drew the
    // weight's contour as a sharp-edged patch (M 42's top left).
    let dark_n: f32 = dark.iter().sum::<f32>().max(1.0);
    let global = |c: &[f32]| c.iter().zip(&dark).map(|(v, d)| v * d).sum::<f32>() / dark_n;
    let (gr, gg2, gb) = (global(&rr), global(&gg), global(&bb));
    let mean_of = |b: Vec<f32>, g: f32| -> Vec<f32> { b.iter().zip(&wsum).map(|(v, w)| (v + NEUTRAL_PRIOR * g) / (w + NEUTRAL_PRIOR)).collect() };
    let (mr, mg, mb) = (mean_of(br, gr), mean_of(bg, gg2), mean_of(bb2, gb));
    let mut cast = [vec![0f32; n], vec![0f32; n], vec![0f32; n]];
    for i in 0..n {
        let m = (mr[i] + mg[i] + mb[i]) / 3.0;
        cast[0][i] = mr[i] - m;
        cast[1][i] = mg[i] - m;
        cast[2][i] = mb[i] - m;
    }
    Some(Neutral { cast, mw, mh, sx: frame.sx, sy: frame.sy, ox: frame.ox, oy: frame.oy, f: p.width as f32 / mw as f32, level, amount: r.finish.neutral.clamp(0.0, 1.0) as f32 })
}

/// The colour map for a recipe (None when off, or without a starless proxy).
fn chroma_of(p: &Prepared, r: &Recipe, coef: &palette::Coef, frame: Frame) -> Option<Chroma> {
    if r.finish.chroma <= 0.0 {
        return None;
    }
    let sp: Vec<&[f32]> = p.channels.iter().map(|c| c.starless_proxy.as_ref().map(|i| i.data.as_slice())).collect::<Option<_>>()?;
    let first = p.channels[0].starless_proxy.as_ref()?;
    let (mw, mh) = (first.width, first.height);
    let n = mw * mh;
    let (mut rr, mut gg, mut bb) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
    palette::process_par(&sp, coef, &mut rr, &mut gg, &mut bb);
    let l: Vec<f32> = (0..n).map(|i| (rr[i] + gg[i] + bb[i]) / 3.0).collect();
    let sigma = (CHROMA_SIGMA * mw as f32).max(1.5);
    let ls = gauss3(&l, mw, mh, sigma);
    let ratio = |c: &[f32]| -> Vec<f32> { gauss3(c, mw, mh, sigma).iter().zip(&ls).map(|(v, l)| if *l > 1.0e-5 { v / l } else { 1.0 }).collect() };
    Some(Chroma { maps: [ratio(&rr), ratio(&gg), ratio(&bb)], mw, mh, sx: frame.sx, sy: frame.sy, ox: frame.ox, oy: frame.oy, f: p.width as f32 / mw as f32, amount: r.finish.chroma.clamp(0.0, 1.0) as f32 })
}

/// The detail map for a recipe (None when off, or without a starless
/// proxy).
fn detail_of(p: &Prepared, r: &Recipe, coef: &palette::Coef, frame: Frame) -> Option<Detail> {
    if r.finish.detail <= 0.0 {
        return None;
    }
    let sp: Vec<&[f32]> = p.channels.iter().map(|c| c.starless_proxy.as_ref().map(|i| i.data.as_slice())).collect::<Option<_>>()?;
    let first = p.channels[0].starless_proxy.as_ref()?;
    let (mw, mh) = (first.width, first.height);
    let n = mw * mh;
    let (mut rr, mut gg, mut bb) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
    palette::process_par(&sp, coef, &mut rr, &mut gg, &mut bb);
    let l: Vec<f32> = (0..n).map(|i| (rr[i] + gg[i] + bb[i]) / 3.0).collect();
    let fine = gauss3(&l, mw, mh, (DETAIL_FINE * mw as f32).max(0.7));
    let coarse = gauss3(&l, mw, mh, DETAIL_COARSE * mw as f32);
    let map: Vec<f32> = fine.iter().zip(&coarse).map(|(a, b)| a - b).collect();
    Some(Detail { map, mw, mh, sx: frame.sx, sy: frame.sy, ox: frame.ox, oy: frame.oy, f: p.width as f32 / mw as f32, amount: 2.0 * r.finish.detail.clamp(0.0, 1.0) as f32 })
}

/// Noise σ of `full − bare` (1.4826 × MAD), registration borders left out:
/// the star layer's own noise at that resolution (0 where the starless
/// image is the original away from stars, as the classic fill's is).
fn layer_sigma(full: &[f32], bare: &[f32]) -> f32 {
    let step = (full.len() / 1_000_000).max(1);
    let mut d: Vec<f32> = full.iter().zip(bare).step_by(step).filter(|(f, _)| **f > palette::ZERO_FLOOR).map(|(f, b)| f - b).collect();
    if d.is_empty() {
        return 0.0;
    }
    let m = d.len() / 2;
    d.select_nth_unstable_by(m, |a, b| a.total_cmp(b));
    let med = d[m];
    for v in d.iter_mut() {
        *v = (*v - med).abs();
    }
    d.select_nth_unstable_by(m, |a, b| a.total_cmp(b));
    1.4826 * d[m]
}

/// The separate stretch's constants, with the floor measured on `planes`
/// and `bare` (the render's own resolution, or a 1:1 crop standing in for
/// full size).
fn split_of(p: &Prepared, r: &Recipe, planes: &[&[f32]], bare: &[&[f32]], frame: Frame) -> Option<Split> {
    let coef = coefficients_starless(p, r)?;
    let detail = detail_of(p, r, &coef, frame);
    let chroma = chroma_of(p, r, &coef, frame);
    let neutral = neutral_of(p, r, &coef, frame);
    let names = p.names();
    // The star layer's stretch follows the reference channel's noise (H,
    // else S; G, else L for broadband).
    let ch = &p.channels[palette::reference(&names, p.kind)];
    let noise = if ch.stats.sky_noise > 0.0 { ch.stats.sky_noise } else { ch.noise_full as f64 };
    let curve = stars::Curve::new(noise, r.stretch.clip, r.stretch.background, r.stars.size, r.stars.brightness);
    // The floor: the star layer's own noise (none where the starless image
    // keeps the original's noise, as the star remover's and the classic
    // fill's do away from stars) times the cut, up to the default 2σ; a
    // higher cut adds the channel's noise at this resolution for each σ
    // beyond, which thins out faint stars and halos.
    let cut = r.stars.faint_cut.clamp(0.0, 50.0) as f32;
    let at_proxy = planes.first().is_some_and(|pl| pl.len() == p.channels[0].proxy.data.len());
    use rayon::prelude::*;
    let floor = planes
        .par_iter()
        .zip(bare)
        .zip(&p.channels)
        .map(|((f, b), c)| {
            let own = cut.min(STAR_FLOOR_SIGMAS) * layer_sigma(f, b);
            let noise = if at_proxy { c.noise_proxy } else { c.noise_full };
            own + (cut - STAR_FLOOR_SIGMAS).max(0.0) * noise
        })
        .collect();
    Some(Split { coef, curve, floor, detail, chroma, neutral })
}

/// Star colour constants.
fn colour(p: &Prepared, r: &Recipe) -> Option<stars::Colour> {
    if !stars_on(p, r) {
        return None;
    }
    let names = p.names();
    let at = |n: &str| names.iter().position(|c| c == n);
    if p.kind == Kind::Mono {
        return (names.len() == 1).then(|| stars::Colour { h: 0, o: 0, s: None, gain_o: 1.0, gain_s: 1.0, ha_to_green: 0.0, saturation: 0.0, rgb: Some(([0, 0, 0], [1.0; 3])) });
    }
    if p.kind == Kind::Broadband {
        let (ri, gi, bi) = (at("R")?, at("G")?, at("B")?);
        let gain = |i: usize| (p.star_gain.get(i).copied().unwrap_or(1.0) / p.star_gain.get(gi).copied().unwrap_or(1.0)) as f32;
        return Some(stars::Colour { h: gi, o: bi, s: None, gain_o: 1.0, gain_s: 1.0, ha_to_green: 0.0, saturation: r.stars.saturation as f32, rgb: Some(([ri, gi, bi], [gain(ri), 1.0, gain(bi)])) });
    }
    let h = at("H").or(at("S"))?;
    let o = at("O")?;
    let s = at("S").filter(|si| *si != h);
    let gain = |i: usize| (p.star_gain.get(i).copied().unwrap_or(1.0) / p.star_gain.get(h).copied().unwrap_or(1.0)) as f32;
    Some(stars::Colour { h, o, s, gain_o: gain(o), gain_s: s.map(gain).unwrap_or(1.0), ha_to_green: r.stars.ha_to_green as f32, saturation: r.stars.saturation as f32, rgb: None })
}

/// Render planes; with `bare` (the same planes without stars) and a
/// colour, the starless mix plus naturally coloured stars: recoloured in
/// place, or with `split` the starless image finished on its own and the
/// star layer screened on.
fn render_planes(planes: &[&[f32]], bare: Option<&[&[f32]]>, w: usize, h: usize, coef: &palette::Coef, col: Option<(&stars::Colour, &stars::Paint)>, split: Option<&Split>) -> Rgb {
    use rayon::prelude::*;
    let n = w * h;
    let (mut r, mut g, mut b) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
    let rows = 64usize;
    r.par_chunks_mut(rows * w).zip(g.par_chunks_mut(rows * w)).zip(b.par_chunks_mut(rows * w)).enumerate().for_each(|(k, ((rr, gg), bb))| {
        let s = k * rows * w;
        let e = s + rr.len();
        let ch: Vec<&[f32]> = planes.iter().map(|p| &p[s..e]).collect();
        match (bare, col) {
            (Some(bare), Some((c, pt))) if split.is_some() => {
                let sp = split.unwrap();
                let bc: Vec<&[f32]> = bare.iter().map(|p| &p[s..e]).collect();
                let len = e - s;
                // The original under its own stretch gives the colour of
                // light no measured star explains.
                let (mut fr, mut fg, mut fb) = (vec![0f32; len], vec![0f32; len], vec![0f32; len]);
                palette::process(&ch, coef, &mut fr, &mut fg, &mut fb);
                palette::process(&bc, &sp.coef, rr, gg, bb);
                if let Some(nt) = &sp.neutral {
                    for i in 0..len {
                        let o = nt.apply((s + i) % w, (s + i) / w, [rr[i], gg[i], bb[i]]);
                        rr[i] = o[0];
                        gg[i] = o[1];
                        bb[i] = o[2];
                    }
                }
                if let Some(cm) = &sp.chroma {
                    for i in 0..len {
                        let o = cm.apply((s + i) % w, (s + i) / w, [rr[i], gg[i], bb[i]]);
                        rr[i] = o[0];
                        gg[i] = o[1];
                        bb[i] = o[2];
                    }
                }
                if let Some(d) = &sp.detail {
                    for i in 0..len {
                        let o = d.apply((s + i) % w, (s + i) / w, [rr[i], gg[i], bb[i]]);
                        rr[i] = o[0];
                        gg[i] = o[1];
                        bb[i] = o[2];
                    }
                }
                for i in 0..len {
                    let light = |k: usize| {
                        let v = ch[k][i];
                        if v <= palette::ZERO_FLOOR { 0.0 } else { (v - bc[k][i] - sp.floor[k]).max(0.0) }
                    };
                    let star = |k: usize| pt.flux[k][s + i];
                    let o = stars::screen([rr[i], gg[i], bb[i]], &light, &star, pt.confidence[s + i], [fr[i], fg[i], fb[i]], c, &sp.curve);
                    rr[i] = o[0];
                    gg[i] = o[1];
                    bb[i] = o[2];
                }
            }
            (Some(bare), Some((c, pt))) => {
                let bc: Vec<&[f32]> = bare.iter().map(|p| &p[s..e]).collect();
                let len = e - s;
                let (mut fr, mut fg, mut fb) = (vec![0f32; len], vec![0f32; len], vec![0f32; len]);
                palette::process(&ch, coef, &mut fr, &mut fg, &mut fb);
                palette::process(&bc, coef, rr, gg, bb);
                for i in 0..len {
                    let lin = |k: usize| pt.flux[k][s + i];
                    let o = stars::combine([fr[i], fg[i], fb[i]], [rr[i], gg[i], bb[i]], &lin, pt.confidence[s + i], c);
                    rr[i] = o[0];
                    gg[i] = o[1];
                    bb[i] = o[2];
                }
            }
            _ => palette::process(&ch, coef, rr, gg, bb),
        }
    });
    Rgb { width: w, height: h, r, g, b }
}

/// Render the proxy.
pub fn render_proxy(p: &Prepared, r: &Recipe) -> Rgb {
    let coef = coefficients(p, r);
    let (w, h) = (p.channels[0].proxy.width, p.channels[0].proxy.height);
    let planes: Vec<&[f32]> = p.channels.iter().map(|c| c.proxy.data.as_slice()).collect();
    let col = colour(p, r);
    let bare: Option<Vec<&[f32]>> = col.as_ref().and_then(|_| p.channels.iter().map(|c| c.starless_proxy.as_ref().map(|i| i.data.as_slice())).collect());
    let split = bare.as_ref().and_then(|b| split_of(p, r, &planes, b, Frame::whole(p, w, h)));
    render_planes(&planes, bare.as_deref(), w, h, &coef, col.as_ref().zip(p.paint_proxy.as_ref()), split.as_ref())
}

/// Render the 1:1 crop (`Prepared::crop`).
pub fn render_crop(p: &Prepared, r: &Recipe) -> Rgb {
    let coef = coefficients(p, r);
    let (w, h) = (p.channels[0].crop.width, p.channels[0].crop.height);
    let planes: Vec<&[f32]> = p.channels.iter().map(|c| c.crop.data.as_slice()).collect();
    let col = colour(p, r);
    let bare: Option<Vec<&[f32]>> = col.as_ref().and_then(|_| p.channels.iter().map(|c| c.crop_starless.as_ref().map(|i| i.data.as_slice())).collect());
    let frame = Frame { sx: 1.0, sy: 1.0, ox: p.crop.0 as f32, oy: p.crop.1 as f32 };
    let split = bare.as_ref().and_then(|b| split_of(p, r, &planes, b, frame));
    render_planes(&planes, bare.as_deref(), w, h, &coef, col.as_ref().zip(p.paint_crop.as_ref()), split.as_ref())
}

/// The starless picture alone (proxy), for showing what the stars sit on.
pub fn render_starless(p: &Prepared, r: &Recipe) -> Option<Rgb> {
    if !stars_on(p, r) {
        return None;
    }
    let coef = coefficients_starless(p, r).unwrap_or_else(|| coefficients(p, r));
    let sp: Vec<&[f32]> = p.channels.iter().map(|c| c.starless_proxy.as_ref().map(|i| i.data.as_slice())).collect::<Option<_>>()?;
    let first = p.channels[0].starless_proxy.as_ref()?;
    Some(render_planes(&sp, None, first.width, first.height, &coef, None, None))
}

/// Render a small tile (preset previews), from the proxy.
pub fn render_tile(p: &Prepared, r: &Recipe, max: usize) -> Rgb {
    let coef = coefficients(p, r);
    let small: Vec<Image> = p.channels.iter().map(|c| proxy(&c.proxy, max)).collect();
    let planes: Vec<&[f32]> = small.iter().map(|c| c.data.as_slice()).collect();
    let col = colour(p, r);
    let bare_imgs: Option<Vec<Image>> = col.as_ref().and_then(|_| p.channels.iter().map(|c| c.starless_proxy.as_ref().map(|i| proxy(i, max))).collect());
    let bare: Option<Vec<&[f32]>> = bare_imgs.as_ref().map(|v| v.iter().map(|i| i.data.as_slice()).collect());
    let (tw, th) = (small[0].width, small[0].height);
    let pt = col.as_ref().map(|_| stars::paint(&p.star_info, p.channels.len(), tw, th, tw as f32 / p.width.max(1) as f32, 0.0, 0.0));
    let split = bare.as_ref().and_then(|b| split_of(p, r, &planes, b, Frame::whole(p, tw, th)));
    render_planes(&planes, bare.as_deref(), tw, th, &coef, col.as_ref().zip(pt.as_ref()), split.as_ref())
}

/// Render at full resolution (needs `prepare(.., keep_full = true)`),
/// cropped as the recipe says.
pub fn render_full(p: &Prepared, r: &Recipe) -> Result<Rgb> {
    let laps = crate::timing::Laps::new("render_full");
    let coef = coefficients(p, r);
    let planes: Vec<&[f32]> = p.channels.iter().map(|c| c.full.as_ref().map(|i| i.data.as_slice())).collect::<Option<_>>().ok_or_else(|| anyhow::anyhow!("full-resolution channels weren't kept"))?;
    let col = colour(p, r);
    let bare: Option<Vec<&[f32]>> = col.as_ref().and_then(|_| p.channels.iter().map(|c| c.starless.as_ref().map(|i| i.data.as_slice())).collect());
    // The floor from the 1:1 crop: the same resolution, a fraction of the
    // pixels.
    let crop: Vec<&[f32]> = p.channels.iter().map(|c| c.crop.data.as_slice()).collect();
    let crop_bare: Option<Vec<&[f32]>> = p.channels.iter().map(|c| c.crop_starless.as_ref().map(|i| i.data.as_slice())).collect();
    let split = bare.as_ref().and(crop_bare.as_ref()).and_then(|b| split_of(p, r, &crop, b, Frame::whole(p, p.width, p.height)));
    laps.lap("coefficients and split");
    let rgb = render_planes(&planes, bare.as_deref(), p.width, p.height, &coef, col.as_ref().zip(p.paint_full.as_ref()), split.as_ref());
    laps.lap("planes");
    // The crop is in the turned picture: cut first (turned back), then turn
    // the smaller result.
    let q = r.rotate % 4;
    let rgb = match crop_of(p, r) {
        Some(c) if !c.is_whole() => rgb.cropped(&c.turned(4 - q)),
        _ => rgb,
    };
    let rgb = rgb.rotated(q);
    laps.lap("crop and turn");
    Ok(rgb)
}

/// The crop the recipe asks for, in the turned picture: its own, or with
/// none set, the empty edges trimmed (the default).
pub fn crop_of(p: &Prepared, r: &Recipe) -> Option<recipe::Crop> {
    r.crop.or_else(|| auto_crop(p).map(|c| c.turned(r.rotate % 4)))
}

/// The largest rectangle with data in every channel: trims the empty
/// edges a mosaic or a registration leaves. None when there are none.
/// Worked out on the proxies (before gradient removal, where empty
/// pixels are still exactly zero), one proxy pixel in from the edge of
/// the data, since a proxy pixel can be partly empty.
pub fn auto_crop(p: &Prepared) -> Option<recipe::Crop> {
    let first = &p.channels.first()?.raw_proxy;
    let (w, h) = (first.width, first.height);
    if w < 4 || h < 4 {
        return None;
    }
    let empty: Vec<bool> = (0..w * h).map(|i| !p.channels.iter().all(|c| c.raw_proxy.data.get(i).is_some_and(|v| *v > 0.0))).collect();
    let mut valid = outside_empty(&empty, w, h).into_iter().map(|e| !e).collect::<Vec<bool>>();
    if let Some(t) = &p.thin {
        for y in 0..h {
            for x in 0..w {
                let (bx, by) = ((x * t.w / w).min(t.w - 1), (y * t.h / h).min(t.h - 1));
                if t.blocks[by * t.w + bx] {
                    valid[y * w + x] = false;
                }
            }
        }
    }
    if valid.iter().all(|v| *v) {
        return None;
    }
    let (x0, y0, x1, y1) = largest_rect(&valid, w, h)?;
    // One pixel in where the edge isn't the picture's own.
    let x0 = if x0 > 0 { x0 + 1 } else { 0 };
    let y0 = if y0 > 0 { y0 + 1 } else { 0 };
    let x1 = if x1 < w { x1 - 1 } else { w };
    let y1 = if y1 < h { y1 - 1 } else { h };
    if x1 <= x0 + 1 || y1 <= y0 + 1 {
        return None;
    }
    Some(recipe::Crop { left: x0 as f64 / w as f64, top: y0 as f64 / h as f64, right: x1 as f64 / w as f64, bottom: y1 as f64 / h as f64 })
}

/// The empty pixels joined to the picture's edge through empty pixels: a
/// stray empty pixel inside (a hole in a master) mustn't cut the crop down
/// to one side of it.
fn outside_empty(empty: &[bool], w: usize, h: usize) -> Vec<bool> {
    let mut out = vec![false; w * h];
    let mut stack: Vec<usize> = (0..w * h).filter(|&i| (i % w == 0 || i % w == w - 1 || i / w == 0 || i / w == h - 1) && empty[i]).collect();
    while let Some(i) = stack.pop() {
        if out[i] {
            continue;
        }
        out[i] = true;
        let (x, y) = (i % w, i / w);
        let near = [(x > 0).then(|| i - 1), (x + 1 < w).then(|| i + 1), (y > 0).then(|| i - w), (y + 1 < h).then(|| i + w)];
        stack.extend(near.into_iter().flatten().filter(|&j| empty[j] && !out[j]));
    }
    out
}

/// Blocks (32 px of the master) whose fine noise is this many times the
/// picture's typical block count as thinly covered.
const THIN_NOISE: f32 = 2.5;
const THIN_BLOCK: usize = 32;
/// Rows or columns along an edge whose median block is this many times the
/// typical block's are thinly covered too.
const THIN_STRIP: f32 = 1.4;

/// Edges covered by few frames, as a grid of [`THIN_BLOCK`] px blocks.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Thin {
    pub w: usize,
    pub h: usize,
    pub blocks: Vec<bool>,
}

/// Edges covered by few frames (a stack's frames framed differently on
/// different nights): blocks whose fine noise is [`THIN_NOISE`] × the
/// typical block's, reached from the picture's edge through such blocks
/// (bright nebula inside the picture can be as rough, but isn't joined to
/// the edge). None when there are none.
fn thin_edges(img: &Image) -> Option<Thin> {
    use rayon::prelude::*;
    const B: usize = THIN_BLOCK;
    let (w, h) = (img.width, img.height);
    let (bw, bh) = (w / B, h / B);
    if bw < 4 || bh < 4 {
        return None;
    }
    let d = &img.data;
    let noise: Vec<f32> = (0..bw * bh)
        .into_par_iter()
        .map(|k| {
            let (bx, by) = (k % bw, k / bw);
            let mut v: Vec<f32> = Vec::with_capacity(B * B);
            for y in (by * B).max(1)..((by + 1) * B).min(h - 1) {
                for x in (bx * B).max(1)..((bx + 1) * B).min(w - 1) {
                    let i = y * w + x;
                    if d[i] > 0.0 {
                        v.push(d[i] - 0.25 * (d[i - 1] + d[i + 1] + d[i - w] + d[i + w]));
                    }
                }
            }
            if v.len() < 64 {
                return 0.0;
            }
            let m = crate::stats::median_mut(&mut v);
            v.iter_mut().for_each(|x| *x = (*x - m).abs());
            crate::stats::median_mut(&mut v)
        })
        .collect();
    let mut sorted: Vec<f32> = noise.iter().copied().filter(|v| *v > 0.0).collect();
    if sorted.len() < 16 {
        return None;
    }
    let typical = crate::stats::median_mut(&mut sorted);
    let rough: Vec<bool> = noise.iter().map(|v| *v > THIN_NOISE * typical).collect();
    // Flood from the edge blocks through rough blocks.
    let mut thin = vec![false; bw * bh];
    let mut stack: Vec<usize> = (0..bw * bh).filter(|&k| (k % bw == 0 || k % bw == bw - 1 || k / bw == 0 || k / bw == bh - 1) && rough[k]).collect();
    while let Some(k) = stack.pop() {
        if thin[k] {
            continue;
        }
        thin[k] = true;
        let (bx, by) = (k % bw, k / bw);
        let near = [(bx > 0).then(|| k - 1), (bx + 1 < bw).then(|| k + 1), (by > 0).then(|| k - bw), (by + 1 < bh).then(|| k + bw)];
        stack.extend(near.into_iter().flatten().filter(|&j| rough[j] && !thin[j]));
    }
    // Whole strips along an edge a little rougher than the rest (half the
    // frames: √2): rows and columns judged by their median block, so
    // nebula filling part of one doesn't count.
    let strip = |vals: Vec<f32>| crate::stats::median(&vals) > THIN_STRIP * typical;
    let col = |x: usize| strip((0..bh).map(|y| noise[y * bw + x]).collect());
    let row = |y: usize| strip(noise[y * bw..(y + 1) * bw].to_vec());
    let mut mark = |k: usize| thin[k] = true;
    for x in (0..bw / 4).take_while(|&x| col(x)).chain((bw - bw / 4..bw).rev().take_while(|&x| col(x))) {
        (0..bh).for_each(|y| mark(y * bw + x));
    }
    for y in (0..bh / 4).take_while(|&y| row(y)).chain((bh - bh / 4..bh).rev().take_while(|&y| row(y))) {
        (0..bw).for_each(|x| mark(y * bw + x));
    }
    thin.iter().any(|t| *t).then_some(Thin { w: bw, h: bh, blocks: thin })
}

/// Largest all-true rectangle (x0, y0, x1, y1, ends exclusive): the
/// histogram-and-stack method, one pass over the rows.
fn largest_rect(valid: &[bool], w: usize, h: usize) -> Option<(usize, usize, usize, usize)> {
    let mut heights = vec![0usize; w];
    let mut best = (0usize, None);
    let mut stack: Vec<usize> = Vec::with_capacity(w + 1);
    for y in 0..h {
        for x in 0..w {
            heights[x] = if valid[y * w + x] { heights[x] + 1 } else { 0 };
        }
        stack.clear();
        for x in 0..=w {
            let cur = if x < w { heights[x] } else { 0 };
            while let Some(&t) = stack.last() {
                if heights[t] <= cur {
                    break;
                }
                stack.pop();
                let ht = heights[t];
                let left = stack.last().map_or(0, |&s| s + 1);
                let area = ht * (x - left);
                if area > best.0 {
                    best = (area, Some((left, y + 1 - ht, x, y + 1)));
                }
            }
            if x < w {
                stack.push(x);
            }
        }
    }
    best.1
}

/// The linear proxy of the mix before stretching, auto-stretched per
/// channel from the image shown: what the channels look like with no
/// palette work. `gradient_corrected` picks the proxy after gradient
/// removal instead of before it.
pub fn linear_view(p: &Prepared, gradient_corrected: bool) -> Rgb {
    linear_rgb(p, gradient_corrected, gradient_corrected)
}

/// Before and after gradient removal, both stretched from the
/// uncorrected proxy. One stretch shows the gradient on both sides, and
/// taking it from the uncorrected side keeps "before" the same whatever
/// the gradient settings (stretching from the corrected side made it
/// change with the degree).
pub fn gradient_pair(p: &Prepared) -> (Rgb, Rgb) {
    (linear_rgb(p, false, false), linear_rgb(p, true, false))
}

/// Shows the corrected or uncorrected proxy (`show_corrected`) with each
/// channel's auto STF taken from its corrected or uncorrected proxy
/// (`stretch_corrected`).
fn linear_rgb(p: &Prepared, show_corrected: bool, stretch_corrected: bool) -> Rgb {
    fn pick(c: &Channel, corrected: bool) -> &Image {
        if corrected {
            &c.proxy
        } else {
            &c.raw_proxy
        }
    }
    let first = &p.channels[0];
    let (w, h) = (pick(first, show_corrected).width, pick(first, show_corrected).height);
    let grab = |n: &str| p.channels.iter().find(|c| c.name == n);
    let (ri, gi, bi) = match p.kind {
        Kind::Broadband => (grab("R"), grab("G"), grab("B")),
        Kind::Narrowband => (grab("S").or(grab("H")), grab("H"), grab("O")),
        Kind::Mono => (Some(first), Some(first), Some(first)),
    };
    let stretch = |c: Option<&Channel>| -> Vec<f32> {
        let Some(c) = c else { return vec![0.0; w * h] };
        let stf = crate::stretch::Stf::auto(pick(c, stretch_corrected));
        pick(c, show_corrected).data.iter().map(|v| stf.apply(*v)).collect()
    };
    Rgb { width: w, height: h, r: stretch(ri), g: stretch(gi), b: stretch(bi) }
}

/// What was written.
#[derive(Debug, Clone)]
pub struct Written {
    pub xisf: PathBuf,
    /// The FITS copy, when asked for (`[process] fits`).
    pub fits: Option<PathBuf>,
    pub png: PathBuf,
    pub jpg: PathBuf,
    /// `preview.png`, longest side at most `PROXY_MAX` (Home's gallery,
    /// the autopilot report).
    pub preview: PathBuf,
    pub recipe: PathBuf,
    pub seconds: f64,
    /// Pixels written (after the crop).
    pub width: usize,
    pub height: usize,
}

/// The output files' name without the extension: "<object> <recipe>".
pub fn base_name(object: &str, recipe: &Recipe) -> String {
    format!("{} {}", crate::process::job::safe_name(object), recipe.slug())
}

/// JPEG quality for full-size images.
pub const JPEG_QUALITY: u8 = 92;

/// Write the full render: XISF (Float32, recipe embedded), 16-bit PNG,
/// 8-bit JPEG, `preview.png` and the target's `recipe.toml`; with `fits`,
/// also a Float32 FITS copy (NAXIS3 = 3 for colour; the recipe's name is
/// in CRUX-RCP, the recipe itself in `recipe.toml` beside it).
pub fn write(rgb: &Rgb, dir: &Path, object: &str, recipe: &Recipe, extra: &Header, fits: bool) -> Result<Written> {
    let t0 = std::time::Instant::now();
    std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    let base = base_name(object, recipe);
    let xisf = dir.join(format!("{base}.xisf"));
    let png = dir.join(format!("{base}.png"));
    let jpg = dir.join(format!("{base}.jpg"));
    let preview = dir.join("preview.png");
    let mut h = extra.clone();
    h.push("OBJECT", object, "");
    h.push("IMAGETYP", "Processed", "");
    h.push("CRUX-RCP", &recipe.name, "crux recipe");
    h.push("CRUX-RCV", &recipe.version.to_string(), "crux recipe format");
    let laps = crate::timing::Laps::new(format!("write {object}"));
    let grey = rgb.is_grey();
    let planes: Vec<&[f32]> = if grey { vec![&rgb.r] } else { vec![&rgb.r, &rgb.g, &rgb.b] };
    // The files are independent: written at the same time (the PNG and
    // JPEG encoders are single-threaded).
    let (fits, r_png, r_jpg, r_prev) = std::thread::scope(|sc| -> Result<_> {
        let png_t = sc.spawn(|| write_png16(&png, rgb, grey));
        let jpg_t = sc.spawn(|| write_jpeg(&jpg, rgb, grey, JPEG_QUALITY));
        let prev_t = sc.spawn(|| write_png8(&preview, &shrink(rgb, PROXY_MAX)));
        crate::xisf::write_planes(&xisf, rgb.width, rgb.height, &planes, &h, &[("Crux:Recipe", &recipe.to_toml())])?;
        let fits = if fits {
            let p = crate::fits::copy_path(&xisf);
            crate::fits::write_file(&p, |t| crate::fits::write_planes(t, rgb.width, rgb.height, &planes, &h))?;
            Some(p)
        } else {
            None
        };
        let join = |t: std::thread::ScopedJoinHandle<'_, Result<()>>| t.join().unwrap_or_else(|_| Err(anyhow::anyhow!("a writer thread panicked")));
        Ok((fits, join(png_t), join(jpg_t), join(prev_t)))
    })?;
    r_png?;
    r_jpg?;
    r_prev?;
    laps.lap("XISF, PNG and JPEG");
    let rp = dir.join("recipe.toml");
    recipe.save(&rp)?;
    Ok(Written { xisf, fits, png, jpg, preview, recipe: rp, seconds: t0.elapsed().as_secs_f64(), width: rgb.width, height: rgb.height })
}

/// A small JPEG only (longest side at most `max`, block mean) plus the
/// recipe: for comparing many renders without writing full-size files.
pub fn write_small(rgb: &Rgb, dir: &Path, object: &str, recipe: &Recipe, max: usize) -> Result<PathBuf> {
    std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    let jpg = dir.join(format!("{}.jpg", base_name(object, recipe)));
    let small = shrink(rgb, max);
    write_jpeg(&jpg, &small, small.is_grey(), JPEG_QUALITY)?;
    recipe.save(&dir.join("recipe.toml"))?;
    Ok(jpg)
}

/// Block-mean copy with the longest side at most `max` (whole factors).
pub fn shrink(rgb: &Rgb, max: usize) -> Rgb {
    let f = rgb.width.max(rgb.height).div_ceil(max.max(1)).max(1);
    if f == 1 {
        return rgb.clone();
    }
    let (w, h) = ((rgb.width / f).max(1), (rgb.height / f).max(1));
    let plane = |src: &[f32]| -> Vec<f32> {
        use rayon::prelude::*;
        let mut out = vec![0f32; w * h];
        out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for (x, o) in row.iter_mut().enumerate() {
                let mut s = 0f64;
                for yy in y * f..(y * f + f).min(rgb.height) {
                    let r = yy * rgb.width;
                    for xx in x * f..(x * f + f).min(rgb.width) {
                        s += src[r + xx] as f64;
                    }
                }
                *o = (s / (f * f) as f64) as f32;
            }
        });
        out
    };
    Rgb { width: w, height: h, r: plane(&rgb.r), g: plane(&rgb.g), b: plane(&rgb.b) }
}

/// An 8-bit JPEG (grey when every pixel is).
pub fn write_jpeg(path: &Path, rgb: &Rgb, grey: bool, quality: u8) -> Result<()> {
    use image::ImageEncoder;
    let f = std::io::BufWriter::new(std::fs::File::create(path).with_context(|| format!("create {}", path.display()))?);
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    let (data, kind): (Vec<u8>, image::ExtendedColorType) = if grey {
        (pack(rgb, 1, |i, o| o[0] = q(rgb.r[i])), image::ExtendedColorType::L8)
    } else {
        (pack(rgb, 3, |i, o| o.copy_from_slice(&[q(rgb.r[i]), q(rgb.g[i]), q(rgb.b[i])])), image::ExtendedColorType::Rgb8)
    };
    image::codecs::jpeg::JpegEncoder::new_with_quality(f, quality)
        .write_image(&data, rgb.width as u32, rgb.height as u32, kind)
        .with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

/// An 8-bit RGB PNG (previews, the autopilot gallery).
pub fn write_png8(path: &Path, rgb: &Rgb) -> Result<()> {
    let f = std::io::BufWriter::new(std::fs::File::create(path).with_context(|| format!("create {}", path.display()))?);
    let mut e = png::Encoder::new(f, rgb.width as u32, rgb.height as u32);
    e.set_color(png::ColorType::Rgb);
    e.set_depth(png::BitDepth::Eight);
    let mut w = e.write_header()?;
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    let data = pack(rgb, 3, |i, o| o.copy_from_slice(&[q(rgb.r[i]), q(rgb.g[i]), q(rgb.b[i])]));
    w.write_image_data(&data)?;
    Ok(())
}

/// Interleaved bytes, `per` per pixel, filled by `f(pixel, bytes)` in
/// parallel.
fn pack(rgb: &Rgb, per: usize, f: impl Fn(usize, &mut [u8]) + Sync) -> Vec<u8> {
    use rayon::prelude::*;
    let mut data = vec![0u8; rgb.r.len() * per];
    data.par_chunks_mut(per * 4096).enumerate().for_each(|(c, chunk)| {
        for (j, o) in chunk.chunks_mut(per).enumerate() {
            f(c * 4096 + j, o);
        }
    });
    data
}

/// 16-bit PNG (grey when every pixel is).
pub fn write_png16(path: &Path, rgb: &Rgb, grey: bool) -> Result<()> {
    let f = std::io::BufWriter::new(std::fs::File::create(path).with_context(|| format!("create {}", path.display()))?);
    let mut e = png::Encoder::new(f, rgb.width as u32, rgb.height as u32);
    e.set_color(if grey { png::ColorType::Grayscale } else { png::ColorType::Rgb });
    e.set_depth(png::BitDepth::Sixteen);
    e.set_compression(png::Compression::Fast);
    let mut w = e.write_header()?;
    let ch = if grey { 1 } else { 3 };
    let q = |v: f32| ((v.clamp(0.0, 1.0) * 65535.0).round() as u16).to_be_bytes();
    let data = pack(rgb, 2 * ch, |i, o| {
        o[..2].copy_from_slice(&q(rgb.r[i]));
        if !grey {
            o[2..4].copy_from_slice(&q(rgb.g[i]));
            o[4..6].copy_from_slice(&q(rgb.b[i]));
        }
    });
    w.write_image_data(&data)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_edges_reach_in_from_the_border_only() {
        let (w, h) = (8usize, 6usize);
        let mut empty = vec![false; w * h];
        // A ragged left edge two pixels deep on rows 0–2, and a hole inside.
        for y in 0..3 {
            empty[y * w] = true;
            empty[y * w + 1] = true;
        }
        empty[3 * w + 5] = true;
        let out = outside_empty(&empty, w, h);
        assert!(out[0] && out[w + 1] && out[2 * w + 1], "edge region");
        assert!(!out[3 * w + 5], "the hole inside isn't an empty edge");
        assert_eq!(out.iter().filter(|v| **v).count(), 6);
    }

    #[test]
    fn finds_thinly_covered_edges() {
        let (w, h) = (1024usize, 640usize);
        let mut rng = 99u64;
        let mut img = Image::new(w, h);
        for y in 0..h {
            for x in 0..w {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let n = ((rng >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 0.01;
                // The right 10% saw half the frames: √2 the noise.
                let k = if x >= 922 { 1.5 } else { 1.0 };
                img.data[y * w + x] = 0.1 + k * n;
            }
        }
        let t = thin_edges(&img).expect("thin strip found");
        let col = |bx: usize| (0..t.h).all(|by| t.blocks[by * t.w + bx]);
        assert!(col(t.w - 1) && col(t.w - 2), "right strip");
        assert!(!col(0) && !col(t.w / 2), "left and middle stay");
    }

    #[test]
    fn renders_and_writes_a_synthetic_target() {
        let dir = std::env::temp_dir().join(format!("crux-render-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (w, h) = (400, 300);
        let mut masters = Vec::new();
        for (c, lvl) in [("H", 0.03f32), ("O", 0.01), ("S", 0.012)] {
            let mut img = Image::new(w, h);
            for y in 0..h {
                for x in 0..w {
                    let neb = 0.02 * (-(((x as f32 - 200.0).powi(2) + (y as f32 - 150.0).powi(2)) / 5000.0)).exp();
                    img.data[y * w + x] = lvl + neb * lvl * 30.0 + 0.0003 * ((x * 7 + y * 13) % 17) as f32 + 0.00002 * x as f32;
                }
            }
            let p = dir.join(format!("m_{c}.xisf"));
            crate::xisf::write_image(&p, &img, &Header::default(), None).unwrap();
            masters.push(Master { channel: c.into(), filter: c.into(), path: p, frames: 1, exposure: 300.0, starless: None, noise: 0.0 });
        }
        let t = Target { object: "Test".into(), masters };
        let p = prepare(&t, &GradientSettings::default(), &stars::StarSettings { mode: stars::StarMode::Keep, ..Default::default() }, true, &AtomicBool::new(false)).unwrap();
        assert_eq!(p.kind, Kind::Narrowband);
        let r = Recipe::for_channels(&p.names());
        let prox = render_proxy(&p, &r);
        let full = render_full(&p, &r).unwrap();
        assert_eq!((full.width, full.height), (w, h));
        assert_eq!((prox.width, prox.height), (w, h)); // smaller than PROXY_MAX: same size
        // Same code, same pixels.
        assert_eq!(prox.r, full.r);
        let out = write(&full, &dir.join("image"), "Test", &r, &Header::default(), true).unwrap();
        let l = crate::xisf::read_layout(&out.xisf).unwrap();
        assert_eq!((l.width, l.height), (w, h));
        assert_eq!(l.header.get("CRUX-RCP"), Some("Copper & Blue"));
        let f = crate::fits::read_layout(out.fits.as_ref().unwrap()).unwrap();
        assert_eq!(f.naxis, if full.is_grey() { vec![w, h] } else { vec![w, h, 3] });
        assert_eq!(f.header.get("CRUX-RCP"), Some("Copper & Blue"));
        assert!(out.png.is_file() && out.jpg.is_file() && out.preview.is_file() && out.recipe.is_file());
        let shrunk = shrink(&full, 150);
        assert_eq!((shrunk.width, shrunk.height), (133, 100));
        // A crop in the recipe crops the full-size render.
        let mut rc = r.clone();
        rc.crop = Some(recipe::Crop { left: 0.25, top: 0.0, right: 0.75, bottom: 0.5 });
        let c = render_full(&p, &rc).unwrap();
        assert_eq!((c.width, c.height), (200, 150));
        assert_eq!(c.r[0], full.r[100]);
        assert_eq!(c.r[200 + 7], full.r[w + 100 + 7]);
        // Turned right, the same crop in the turned picture: the full
        // picture's left half, top half, turned.
        let mut rt = rc.clone();
        rt.rotate = 1;
        rt.crop = rc.crop.map(|k| k.turned(1));
        let t = render_full(&p, &rt).unwrap();
        assert_eq!((t.width, t.height), (150, 200));
        // Turned clockwise: the cropped picture's bottom-left pixel is now top left.
        assert_eq!(t.r[0], c.r[(c.height - 1) * c.width]);
        assert_eq!(t.r[t.width - 1], c.r[0]);
        assert_eq!(c.clone().rotated(2).rotated(2).r, c.r);
        // No empty edges: nothing to trim.
        assert_eq!(auto_crop(&p), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn finds_the_largest_rectangle_with_data() {
        // 10 × 8 with an empty top-left notch and an empty right column.
        let (w, h) = (10, 8);
        let valid: Vec<bool> = (0..w * h).map(|i| {
            let (x, y) = (i % w, i / w);
            !(x == 9 || (x < 3 && y < 2))
        }).collect();
        // Rows 2–7 across columns 0–8 (54) beat columns 3–8 down every row (48).
        assert_eq!(largest_rect(&valid, w, h), Some((0, 2, 9, 8)));
        assert_eq!(largest_rect(&[false; 4], 2, 2), None);
    }
}
