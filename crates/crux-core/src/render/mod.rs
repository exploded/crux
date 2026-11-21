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

/// Can this target have natural star colours (narrowband with OIII and
/// Ha or SII)?
pub fn stars_possible(names: &[String]) -> bool {
    palette::kind_of(names) == Kind::Narrowband && names.iter().any(|c| c == "O") && names.iter().any(|c| c == "H" || c == "S")
}

/// The target as a recipe wants it: masters sharpened and denoised
/// (`enhance`) and, for natural star colours with the star remover, their
/// starless versions (`starnet`). Both are cached on disk; `progress` gets
/// a short line now and then.
pub fn for_recipe(t: &Target, out: &Path, r: &Recipe, progress: &(dyn Fn(&str) + Sync), cancel: &AtomicBool) -> Result<Target> {
    let te = enhance::apply(t, out, &r.enhance, &|p| progress(&format!("{} {} of {}: {} {:.0}%", p.channel, p.master + 1, p.masters, p.step, p.done * 100.0)), cancel)?;
    if r.stars.engine == stars::StarEngine::Network && r.stars.mode == stars::StarMode::Natural && stars_possible(&te.channels()) {
        starnet::apply(&te, out, &|i, n, f| progress(&format!("removing stars {} of {}: {:.0}%", i + 1, n, f * 100.0)), cancel)
    } else {
        Ok(te)
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
    for m in &t.masters {
        if cancel.load(Ordering::Relaxed) {
            anyhow::bail!("cancelled");
        }
        let img = crate::xisf::read_image(&m.path).with_context(|| format!("read {}", m.path.display()))?;
        match size {
            None => size = Some((img.width, img.height)),
            Some(s) => anyhow::ensure!(s == (img.width, img.height), "{} is {}×{}, the other masters {}×{}: they must be registered to one reference", m.path.display(), img.width, img.height, s.0, s.1),
        }
        let smp = g.enabled.then(|| gradient::samples(&img, g.samples));
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
    let mut corrected = Vec::new();
    for (m, mut img, smp) in fulls {
        let raw_proxy = proxy(&img, PROXY_MAX);
        let fit = smp.and_then(|s| gradient::fit_samples(&s, g, degree));
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
    let per_channel: Vec<_> = {
        use rayon::prelude::*;
        corrected
            .into_par_iter()
            .enumerate()
            .map(|(k, (m, img, raw_proxy, fit))| {
                let (model, report) = match fit {
                    Some((md, rp)) => (Some(md), Some(rp)),
                    None => (None, None),
                };
                let px = if model.is_some() { proxy(&img, PROXY_MAX) } else { raw_proxy.clone() };
                let mut stats = Stats::measure(&img.data, &px.data);
                stats.sky_noise = m.noise;
                let noise_full = crate::process::combine::noise_of(&img);
                let noise_proxy = crate::process::combine::noise_of(&px);
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
                let sl_proxy = sl.as_ref().map(|s| proxy(s, PROXY_MAX));
                let stats_starless = sl.as_ref().zip(sl_proxy.as_ref()).map(|(s, sp)| Stats::measure(&s.data, &sp.data));
                let crop = cut(&img);
                let crop_starless = sl.as_ref().map(&cut);
                (m, img, raw_proxy, model, report, px, stats, noise_full, noise_proxy, sl, sl_proxy, stats_starless, crop, crop_starless)
            })
            .collect()
    };
    for (m, img, raw_proxy, model, report, px, stats, noise_full, noise_proxy, sl, sl_proxy, stats_starless, crop, crop_starless) in per_channel {
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
        star_info = stars::measure(&discs, &with, &without, hi, channels[hi].noise_full);
        let n = channels.len();
        let (pw, ph) = (channels[0].proxy.width, channels[0].proxy.height);
        paint_proxy = Some(stars::paint(&star_info, n, pw, ph, pw as f32 / width as f32, 0.0, 0.0));
        let (x0, y0, cw, ch) = crop_rect;
        paint_crop = Some(stars::paint(&star_info, n, cw, ch, 1.0, x0 as f32, y0 as f32));
        if keep_full {
            paint_full = Some(stars::paint(&star_info, n, width, height, 1.0, 0.0, 0.0));
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
    Ok(Prepared { object: t.object.clone(), kind, channels, width, height, star_gain, gradient: g.clone(), stars: st.clone(), n_stars, star_cover, crop: crop_rect, star_info, paint_proxy, paint_crop, paint_full })
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

/// The coefficients a recipe gives for these channels (always from the
/// original statistics: the Stars stage changes only the stars).
pub fn coefficients(p: &Prepared, r: &Recipe) -> palette::Coef {
    let stats: Vec<&Stats> = p.channels.iter().map(|c| &c.stats).collect();
    palette::coefficients(&p.names(), &stats, Some(&p.star_gain), &r.mix, &r.stretch, &r.finish)
}

/// Star colour constants.
fn colour(p: &Prepared, r: &Recipe) -> Option<stars::Colour> {
    if !stars_on(p, r) {
        return None;
    }
    let names = p.names();
    let at = |n: &str| names.iter().position(|c| c == n);
    let h = at("H").or(at("S"))?;
    let o = at("O")?;
    let s = at("S").filter(|si| *si != h);
    let gain = |i: usize| (p.star_gain.get(i).copied().unwrap_or(1.0) / p.star_gain.get(h).copied().unwrap_or(1.0)) as f32;
    Some(stars::Colour { h, o, s, gain_o: gain(o), gain_s: s.map(gain).unwrap_or(1.0), ha_to_green: r.stars.ha_to_green as f32, saturation: r.stars.saturation as f32 })
}

/// Render planes; with `bare` (the same planes without stars) and a
/// colour, the starless mix plus naturally coloured stars.
fn render_planes(planes: &[&[f32]], bare: Option<&[&[f32]]>, w: usize, h: usize, coef: &palette::Coef, col: Option<(&stars::Colour, &stars::Paint)>) -> Rgb {
    use rayon::prelude::*;
    let n = w * h;
    let (mut r, mut g, mut b) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
    let rows = 64usize;
    r.par_chunks_mut(rows * w).zip(g.par_chunks_mut(rows * w)).zip(b.par_chunks_mut(rows * w)).enumerate().for_each(|(k, ((rr, gg), bb))| {
        let s = k * rows * w;
        let e = s + rr.len();
        let ch: Vec<&[f32]> = planes.iter().map(|p| &p[s..e]).collect();
        match (bare, col) {
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
    render_planes(&planes, bare.as_deref(), w, h, &coef, col.as_ref().zip(p.paint_proxy.as_ref()))
}

/// Render the 1:1 crop (`Prepared::crop`).
pub fn render_crop(p: &Prepared, r: &Recipe) -> Rgb {
    let coef = coefficients(p, r);
    let (w, h) = (p.channels[0].crop.width, p.channels[0].crop.height);
    let planes: Vec<&[f32]> = p.channels.iter().map(|c| c.crop.data.as_slice()).collect();
    let col = colour(p, r);
    let bare: Option<Vec<&[f32]>> = col.as_ref().and_then(|_| p.channels.iter().map(|c| c.crop_starless.as_ref().map(|i| i.data.as_slice())).collect());
    render_planes(&planes, bare.as_deref(), w, h, &coef, col.as_ref().zip(p.paint_crop.as_ref()))
}

/// The starless picture alone (proxy), for showing what the stars sit on.
pub fn render_starless(p: &Prepared, r: &Recipe) -> Option<Rgb> {
    if !stars_on(p, r) {
        return None;
    }
    let coef = coefficients(p, r);
    let sp: Vec<&[f32]> = p.channels.iter().map(|c| c.starless_proxy.as_ref().map(|i| i.data.as_slice())).collect::<Option<_>>()?;
    let first = p.channels[0].starless_proxy.as_ref()?;
    Some(render_planes(&sp, None, first.width, first.height, &coef, None))
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
    render_planes(&planes, bare.as_deref(), tw, th, &coef, col.as_ref().zip(pt.as_ref()))
}

/// Render at full resolution (needs `prepare(.., keep_full = true)`),
/// cropped as the recipe says.
pub fn render_full(p: &Prepared, r: &Recipe) -> Result<Rgb> {
    let coef = coefficients(p, r);
    let planes: Vec<&[f32]> = p.channels.iter().map(|c| c.full.as_ref().map(|i| i.data.as_slice())).collect::<Option<_>>().ok_or_else(|| anyhow::anyhow!("full-resolution channels weren't kept"))?;
    let col = colour(p, r);
    let bare: Option<Vec<&[f32]>> = col.as_ref().and_then(|_| p.channels.iter().map(|c| c.starless.as_ref().map(|i| i.data.as_slice())).collect());
    let rgb = render_planes(&planes, bare.as_deref(), p.width, p.height, &coef, col.as_ref().zip(p.paint_full.as_ref()));
    Ok(match &r.crop {
        Some(c) if !c.is_whole() => rgb.cropped(c),
        _ => rgb,
    })
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
    let valid: Vec<bool> = (0..w * h).map(|i| p.channels.iter().all(|c| c.raw_proxy.data.get(i).is_some_and(|v| *v > 0.0))).collect();
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

/// The linear proxy of the mix before stretching, auto-stretched (one
/// linked STF), for "before" views: what the channels look like with no
/// palette work.
pub fn linear_view(p: &Prepared, gradient_corrected: bool) -> Rgb {
    fn pick(c: &Channel, g: bool) -> &Image {
        if g {
            &c.proxy
        } else {
            &c.raw_proxy
        }
    }
    let pick = |c| pick(c, gradient_corrected);
    let first = pick(&p.channels[0]);
    let (w, h) = (first.width, first.height);
    let names = p.names();
    let grab = |n: &str| p.channels.iter().position(|c| c.name == n).map(|i| pick(&p.channels[i]));
    let (ri, gi, bi) = match p.kind {
        Kind::Broadband => (grab("R"), grab("G"), grab("B")),
        Kind::Narrowband => (grab("S").or(grab("H")), grab("H"), grab("O")),
        Kind::Mono => (Some(first), Some(first), Some(first)),
    };
    let _ = names;
    // The stretch comes from the corrected proxy, so before and after
    // views share it and the gradient shows.
    let stretch = |img: Option<&Image>| -> Vec<f32> {
        let Some(img) = img else { return vec![0.0; w * h] };
        let corrected = p.channels.iter().find(|c| std::ptr::eq(&c.raw_proxy, img) || std::ptr::eq(&c.proxy, img)).map(|c| &c.proxy).unwrap_or(img);
        let stf = crate::stretch::Stf::auto(corrected);
        img.data.iter().map(|v| stf.apply(*v)).collect()
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
    let grey = rgb.is_grey();
    let planes: Vec<&[f32]> = if grey { vec![&rgb.r] } else { vec![&rgb.r, &rgb.g, &rgb.b] };
    crate::xisf::write_planes(&xisf, rgb.width, rgb.height, &planes, &h, &[("Crux:Recipe", &recipe.to_toml())])?;
    let fits = if fits {
        let p = crate::fits::copy_path(&xisf);
        crate::fits::write_file(&p, |t| crate::fits::write_planes(t, rgb.width, rgb.height, &planes, &h))?;
        Some(p)
    } else {
        None
    };
    write_png16(&png, rgb, grey)?;
    write_jpeg(&jpg, rgb, grey, JPEG_QUALITY)?;
    write_png8(&preview, &shrink(rgb, PROXY_MAX))?;
    let rp = dir.join("recipe.toml");
    recipe.save(&rp)?;
    Ok(Written { xisf, fits, png, jpg, preview, recipe: rp, seconds: t0.elapsed().as_secs_f64(), width: rgb.width, height: rgb.height })
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
        (rgb.r.iter().map(|v| q(*v)).collect(), image::ExtendedColorType::L8)
    } else {
        ((0..rgb.r.len()).flat_map(|i| [q(rgb.r[i]), q(rgb.g[i]), q(rgb.b[i])]).collect(), image::ExtendedColorType::Rgb8)
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
    let data: Vec<u8> = (0..rgb.r.len()).flat_map(|i| [q(rgb.r[i]), q(rgb.g[i]), q(rgb.b[i])]).collect();
    w.write_image_data(&data)?;
    Ok(())
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
    let mut data = Vec::with_capacity(rgb.r.len() * 2 * ch);
    let q = |v: f32| ((v.clamp(0.0, 1.0) * 65535.0).round() as u16).to_be_bytes();
    for i in 0..rgb.r.len() {
        data.extend_from_slice(&q(rgb.r[i]));
        if !grey {
            data.extend_from_slice(&q(rgb.g[i]));
            data.extend_from_slice(&q(rgb.b[i]));
        }
    }
    w.write_image_data(&data)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
