//! The Image stage's prepared target, kept on disk so reopening a target
//! doesn't read, correct and measure its masters again (9–13 s for the
//! IC 434 mosaic). Only the GUI's preparation is kept (`keep_full = false`:
//! proxies, crops and measurements, no full-resolution planes).
//!
//! File: `<workspace>\<folder>\image\<object>\prepared.cache`: the magic,
//! a JSON header (everything but pixels, plus the signature), then every
//! pixel array as little-endian f32 in a fixed order. The signature covers
//! the masters (path, size, modified time), the gradient and star settings
//! and [`VERSION`]; any difference means prepare again.

use super::gradient::{GradientReport, GradientSettings, Model};
use super::palette::{Kind, Stats};
use super::stars::{Disc, Paint, StarInfo, StarSettings};
use super::{Channel, Master, Prepared, Target};
use crate::image::Image;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Bump whenever `prepare` produces different results.
pub const VERSION: u32 = 18;
const MAGIC: &[u8; 8] = b"CRUXPREP";

/// Where a target's prepared cache lives.
pub fn path(t: &Target, out: &Path) -> PathBuf {
    t.image_dir(out).join("prepared.cache")
}

/// What a preparation was made from.
pub fn signature(t: &Target, g: &GradientSettings, st: &StarSettings) -> String {
    let masters: Vec<String> = t
        .masters
        .iter()
        .map(|m| {
            let (size, mtime) = std::fs::metadata(&m.path)
                .map(|md| (md.len(), md.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_nanos()).unwrap_or(0)))
                .unwrap_or((0, 0));
            let sl = m.starless.as_ref().map(|p| crate::paths::key(p)).unwrap_or_default();
            format!("{}:{}:{}:{}:{}", m.channel, crate::paths::key(&m.path), size, mtime, sl)
        })
        .collect();
    format!("v{VERSION}|{}|{}|{}", masters.join(","), serde_json::to_string(g).unwrap_or_default(), st.mask_key())
}

#[derive(Serialize, Deserialize)]
struct Head {
    sig: String,
    object: String,
    kind: Kind,
    width: usize,
    height: usize,
    star_gain: Vec<f64>,
    gradient: GradientSettings,
    stars: StarSettings,
    n_stars: usize,
    star_cover: f64,
    #[serde(default)]
    thin: Option<super::Thin>,
    crop: (usize, usize, usize, usize),
    channels: Vec<ChanHead>,
    star_info: Vec<StarHead>,
    paint_proxy: Option<PaintHead>,
    paint_crop: Option<PaintHead>,
}

#[derive(Serialize, Deserialize)]
struct ImgHead {
    w: usize,
    h: usize,
    adu: f32,
}

#[derive(Serialize, Deserialize)]
struct StatsHead {
    median: f64,
    mad: f64,
    noise: f64,
    #[serde(default)]
    sky_noise: f64,
    p_median: f64,
    p_high: f64,
    sorted: usize,
}

#[derive(Serialize, Deserialize)]
struct ChanHead {
    name: String,
    channel: String,
    filter: String,
    path: PathBuf,
    frames: usize,
    exposure: f64,
    raw_proxy: ImgHead,
    proxy: ImgHead,
    stats: StatsHead,
    model: Option<Model>,
    report: Option<GradientReport>,
    starless_proxy: Option<ImgHead>,
    stats_starless: Option<StatsHead>,
    crop: ImgHead,
    crop_starless: Option<ImgHead>,
    noise_full: f32,
    noise_proxy: f32,
}

#[derive(Serialize, Deserialize)]
struct StarHead {
    disc: Disc,
    flux: Vec<f32>,
    confidence: f32,
}

#[derive(Serialize, Deserialize)]
struct PaintHead {
    channels: usize,
    len: usize,
}

fn img_head(i: &Image) -> ImgHead {
    ImgHead { w: i.width, h: i.height, adu: i.adu_scale }
}

fn stats_head(s: &Stats) -> StatsHead {
    StatsHead { median: s.median, mad: s.mad, noise: s.noise, sky_noise: s.sky_noise, p_median: s.p_median, p_high: s.p_high, sorted: s.sorted.len() }
}

fn paint_head(p: &Paint) -> PaintHead {
    PaintHead { channels: p.flux.len(), len: p.confidence.len() }
}

/// Save a GUI preparation (one with full-resolution planes isn't kept).
pub fn save(file: &Path, sig: &str, p: &Prepared) -> Result<()> {
    anyhow::ensure!(p.channels.iter().all(|c| c.full.is_none() && c.starless.is_none()) && p.paint_full.is_none(), "full-resolution preparations aren't cached");
    let head = Head {
        sig: sig.to_string(),
        object: p.object.clone(),
        kind: p.kind,
        width: p.width,
        height: p.height,
        star_gain: p.star_gain.clone(),
        gradient: p.gradient.clone(),
        stars: p.stars.clone(),
        n_stars: p.n_stars,
        star_cover: p.star_cover,
        thin: p.thin.clone(),
        crop: p.crop,
        channels: p
            .channels
            .iter()
            .map(|c| ChanHead {
                name: c.name.clone(),
                channel: c.master.channel.clone(),
                filter: c.master.filter.clone(),
                path: c.master.path.clone(),
                frames: c.master.frames,
                exposure: c.master.exposure,
                raw_proxy: img_head(&c.raw_proxy),
                proxy: img_head(&c.proxy),
                stats: stats_head(&c.stats),
                model: c.model.clone(),
                report: c.report.clone(),
                starless_proxy: c.starless_proxy.as_ref().map(img_head),
                stats_starless: c.stats_starless.as_ref().map(stats_head),
                crop: img_head(&c.crop),
                crop_starless: c.crop_starless.as_ref().map(img_head),
                noise_full: c.noise_full,
                noise_proxy: c.noise_proxy,
            })
            .collect(),
        star_info: p.star_info.iter().map(|s| StarHead { disc: s.disc, flux: s.flux.clone(), confidence: s.confidence }).collect(),
        paint_proxy: p.paint_proxy.as_ref().map(paint_head),
        paint_crop: p.paint_crop.as_ref().map(paint_head),
    };
    let json = serde_json::to_vec(&head)?;
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(MAGIC);
    buf.extend_from_slice(&(json.len() as u64).to_le_bytes());
    buf.extend_from_slice(&json);
    let mut put = |v: &[f32]| {
        buf.reserve(v.len() * 4);
        for x in v {
            buf.extend_from_slice(&x.to_le_bytes());
        }
    };
    for c in &p.channels {
        put(&c.raw_proxy.data);
        put(&c.proxy.data);
        put(&c.stats.sorted);
        if let Some(i) = &c.starless_proxy {
            put(&i.data);
        }
        if let Some(s) = &c.stats_starless {
            put(&s.sorted);
        }
        put(&c.crop.data);
        if let Some(i) = &c.crop_starless {
            put(&i.data);
        }
    }
    for pt in [&p.paint_proxy, &p.paint_crop].into_iter().flatten() {
        for f in &pt.flux {
            put(f);
        }
        put(&pt.confidence);
    }
    if let Some(d) = file.parent() {
        std::fs::create_dir_all(d)?;
    }
    let tmp = file.with_extension("cache.tmp");
    {
        let mut f = std::fs::File::create(&tmp).with_context(|| format!("create {}", tmp.display()))?;
        f.write_all(&buf)?;
    }
    std::fs::rename(&tmp, file).with_context(|| format!("rename to {}", file.display()))?;
    Ok(())
}

/// The cached preparation, if it was made from exactly this (`sig`).
pub fn load(file: &Path, sig: &str) -> Option<Prepared> {
    let mut f = std::fs::File::open(file).ok()?;
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic).ok()?;
    if &magic != MAGIC {
        return None;
    }
    let mut len = [0u8; 8];
    f.read_exact(&mut len).ok()?;
    let mut json = vec![0u8; u64::from_le_bytes(len) as usize];
    f.read_exact(&mut json).ok()?;
    let head: Head = serde_json::from_slice(&json).ok()?;
    if head.sig != sig {
        return None;
    }
    let mut rest = Vec::new();
    f.read_to_end(&mut rest).ok()?;
    let mut pos = 0usize;
    let mut take = |n: usize| -> Option<Vec<f32>> {
        let end = pos.checked_add(n.checked_mul(4)?)?;
        let bytes = rest.get(pos..end)?;
        pos = end;
        Some(bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
    };
    let mut img = |h: &ImgHead| -> Option<Image> { Some(Image { width: h.w, height: h.h, data: take(h.w * h.h)?, adu_scale: h.adu }) };
    let stats = |h: &StatsHead, sorted: Vec<f32>| Stats { median: h.median, mad: h.mad, noise: h.noise, sky_noise: h.sky_noise, p_median: h.p_median, p_high: h.p_high, sorted };
    let mut channels = Vec::new();
    for c in &head.channels {
        let raw_proxy = img(&c.raw_proxy)?;
        let proxy = img(&c.proxy)?;
        let st = stats(&c.stats, take_vec(&mut img, c.stats.sorted)?);
        let starless_proxy = match &c.starless_proxy {
            Some(h) => Some(img(h)?),
            None => None,
        };
        let stats_starless = match &c.stats_starless {
            Some(h) => Some(stats(h, take_vec(&mut img, h.sorted)?)),
            None => None,
        };
        let crop = img(&c.crop)?;
        let crop_starless = match &c.crop_starless {
            Some(h) => Some(img(h)?),
            None => None,
        };
        channels.push(Channel {
            name: c.name.clone(),
            master: Master { channel: c.channel.clone(), filter: c.filter.clone(), path: c.path.clone(), frames: c.frames, exposure: c.exposure, starless: None, noise: st.sky_noise },
            full: None,
            raw_proxy,
            proxy,
            stats: st,
            model: c.model.clone(),
            report: c.report.clone(),
            starless: None,
            starless_proxy,
            stats_starless,
            crop,
            crop_starless,
            noise_full: c.noise_full,
            noise_proxy: c.noise_proxy,
        });
    }
    let mut paint = |h: &Option<PaintHead>| -> Option<Option<Paint>> {
        match h {
            None => Some(None),
            Some(h) => {
                let mut flux = Vec::with_capacity(h.channels);
                for _ in 0..h.channels {
                    flux.push(take_vec(&mut img, h.len)?);
                }
                let confidence = take_vec(&mut img, h.len)?;
                Some(Some(Paint { flux, confidence }))
            }
        }
    };
    let paint_proxy = paint(&head.paint_proxy)?;
    let paint_crop = paint(&head.paint_crop)?;
    Some(Prepared {
        object: head.object,
        kind: head.kind,
        channels,
        width: head.width,
        height: head.height,
        star_gain: head.star_gain,
        gradient: head.gradient,
        stars: head.stars,
        n_stars: head.n_stars,
        star_cover: head.star_cover,
        thin: head.thin,
        crop: head.crop,
        star_info: head.star_info.into_iter().map(|s| StarInfo { disc: s.disc, flux: s.flux, confidence: s.confidence }).collect(),
        paint_proxy,
        paint_crop,
        paint_full: None,
    })
}

/// `n` floats through the image reader (one cursor for every array).
fn take_vec(img: &mut impl FnMut(&ImgHead) -> Option<Image>, n: usize) -> Option<Vec<f32>> {
    img(&ImgHead { w: n, h: 1, adu: 1.0 }).map(|i| i.data)
}

/// `prepare` for the GUI, from the cache when the masters and settings are
/// unchanged; a fresh preparation is saved for next time.
pub fn prepare(t: &Target, out: &Path, g: &GradientSettings, st: &StarSettings, cancel: &std::sync::atomic::AtomicBool) -> Result<Prepared> {
    let slots = slots(&path(t, out));
    let sig = signature(t, g, st);
    for (i, f) in slots.iter().enumerate() {
        if let Some(p) = load(f, &sig) {
            promote(&slots, i);
            return Ok(p);
        }
    }
    let p = super::prepare(t, g, st, false, cancel)?;
    // Make room at the front: the oldest goes, the rest move down one.
    let _ = std::fs::remove_file(&slots[SLOTS - 1]);
    for k in (1..SLOTS).rev() {
        let _ = std::fs::rename(&slots[k - 1], &slots[k]);
    }
    let _ = save(&slots[0], &sig, &p);
    Ok(p)
}

/// How many preparations a target keeps (switching between a few enhance
/// or star settings finds each one).
const SLOTS: usize = 3;

/// `prepared.cache` (the most recent), then `prepared.1.cache`, … .
pub fn slots(front: &Path) -> Vec<PathBuf> {
    (0..SLOTS).map(|k| if k == 0 { front.to_path_buf() } else { front.with_extension(format!("{k}.cache")) }).collect()
}

/// Move slot `i` to the front, the ones before it down one.
fn promote(slots: &[PathBuf], i: usize) {
    if i == 0 {
        return;
    }
    let tmp = slots[0].with_extension("cache.swap");
    if std::fs::rename(&slots[i], &tmp).is_err() {
        return;
    }
    for k in (1..=i).rev() {
        let _ = std::fs::rename(&slots[k - 1], &slots[k]);
    }
    let _ = std::fs::rename(&tmp, &slots[0]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::Header;
    use std::sync::atomic::AtomicBool;

    #[test]
    fn round_trips_and_notices_changes() {
        let dir = std::env::temp_dir().join(format!("crux-prepcache-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (w, h) = (400, 300);
        let mut masters = Vec::new();
        for (c, lvl) in [("H", 0.03f32), ("O", 0.01), ("S", 0.012)] {
            let mut img = Image::new(w, h);
            for y in 0..h {
                for x in 0..w {
                    let neb = 0.02 * (-(((x as f32 - 200.0).powi(2) + (y as f32 - 150.0).powi(2)) / 5000.0)).exp();
                    let mut v = lvl + neb * lvl * 30.0 + 0.0003 * ((x * 7 + y * 13) % 17) as f32;
                    // A few stars, so the star paths are cached too.
                    for (sx, sy) in [(60.0f32, 50.0f32), (320.0, 80.0), (100.0, 240.0), (300.0, 250.0), (210.0, 40.0)] {
                        v += 0.5 * (-((x as f32 - sx).powi(2) + (y as f32 - sy).powi(2)) / 4.0).exp();
                    }
                    img.data[y * w + x] = v;
                }
            }
            let p = dir.join(format!("m_{c}.xisf"));
            crate::xisf::write_image(&p, &img, &Header::default(), None).unwrap();
            masters.push(Master { channel: c.into(), filter: c.into(), path: p, frames: 1, exposure: 300.0, starless: None, noise: 0.0 });
        }
        let t = Target { object: "Test".into(), masters };
        let (g, st) = (GradientSettings::default(), StarSettings::default());
        let fresh = super::super::prepare(&t, &g, &st, false, &AtomicBool::new(false)).unwrap();
        let file = dir.join("prepared.cache");
        let sig = signature(&t, &g, &st);
        save(&file, &sig, &fresh).unwrap();
        let back = load(&file, &sig).expect("cache loads");
        // Same renders from the cache as from a fresh preparation.
        let r = super::super::recipe::Recipe::for_channels(&fresh.names());
        let (a, b) = (super::super::render_proxy(&fresh, &r), super::super::render_proxy(&back, &r));
        assert_eq!((a.r, a.g, a.b), (b.r, b.g, b.b));
        assert!(fresh.n_stars > 0 && fresh.paint_proxy.is_some(), "the test should exercise the star paths");
        assert_eq!(back.n_stars, fresh.n_stars);
        assert_eq!(back.star_info.len(), fresh.star_info.len());
        assert_eq!(back.channels[0].crop.data, fresh.channels[0].crop.data);
        assert_eq!(back.channels[1].stats.sorted, fresh.channels[1].stats.sorted);
        // Other settings, or a master written again: not used.
        let g2 = GradientSettings { degree: 3, ..g.clone() };
        assert!(load(&file, &signature(&t, &g2, &st)).is_none());
        // Render-only star settings keep it; the mask's don't.
        let st2 = StarSettings { size: 1.2, brightness: 0.8, separate: false, saturation: 1.4, ..st.clone() };
        assert!(load(&file, &signature(&t, &g, &st2)).is_some());
        assert!(load(&file, &signature(&t, &g, &StarSettings { grow: 1.5, ..st.clone() })).is_none());
        std::thread::sleep(std::time::Duration::from_millis(20));
        let img = crate::read_image(&t.masters[0].path).unwrap();
        crate::xisf::write_image(&t.masters[0].path, &img, &Header::default(), None).unwrap();
        assert!(load(&file, &signature(&t, &g, &st)).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn promote_moves_a_slot_to_the_front() {
        let dir = std::env::temp_dir().join(format!("crux-prepslots-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let s = slots(&dir.join("prepared.cache"));
        assert!(s[1].ends_with("prepared.1.cache") && s[2].ends_with("prepared.2.cache"));
        for (k, f) in s.iter().enumerate() {
            std::fs::write(f, [k as u8]).unwrap();
        }
        promote(&s, 2);
        let read = |k: usize| std::fs::read(&s[k]).unwrap()[0];
        assert_eq!((read(0), read(1), read(2)), (2, 0, 1));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
