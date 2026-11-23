//! Enhance: crux's networks (or RC Astro's tools) on each linear master,
//! before gradient removal — star correction / sharpening and noise
//! reduction. The Image stage then reads the enhanced masters as if they
//! were the originals, so its prepared cache (whose signature covers the
//! master files) needs no changes.
//!
//! **crux engine.** Per master, the denoiser's output D and (when asked
//! for) the sharpener's output S are made at full strength and cached
//! (`<image dir>\enhanced\<channel>_<master hash>_<step>.xisf`). The result
//! is a blend, so moving a slider costs a second, not a network run:
//!
//! `final = original − denoise·(original − D) + sharpen·(S − D)`
//!
//! Sharpening alone changes star shapes and keeps the noise; denoising
//! alone is the usual blend; both together do both.
//!
//! **RC Astro engine** (`rcastro`): BlurXTerminator then NoiseXTerminator
//! on the master, through the user's own licence. Its outputs carry the
//! RCASTRO marker and never reach crux's training (`collect`).

use super::Target;
use crate::header::Header;
use crate::image::Image;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Engine {
    #[default]
    Crux,
    RcAstro,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Sharpen {
    #[default]
    Off,
    /// Correct star shapes (coma, elongation); sizes kept.
    Correct,
    /// Correct and tighten stars.
    Tighten,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct EnhanceSettings {
    pub engine: Engine,
    pub sharpen: Sharpen,
    /// 0–1: how much of the sharpening to apply.
    pub sharpen_amount: f32,
    pub denoise: bool,
    /// 0–1: how much of the noise to remove.
    pub denoise_amount: f32,
}

impl Default for EnhanceSettings {
    /// Correct star shapes and remove noise (James, 2026-11-23) when the
    /// build has crux's networks; off without them. Recipes save every
    /// field, so ones written earlier keep what they had.
    fn default() -> Self {
        let on = cfg!(feature = "ml");
        EnhanceSettings { engine: Engine::Crux, sharpen: if on { Sharpen::Correct } else { Sharpen::Off }, sharpen_amount: 1.0, denoise: on, denoise_amount: 0.8 }
    }
}

impl EnhanceSettings {
    pub fn active(&self) -> bool {
        self.denoise || self.sharpen != Sharpen::Off
    }

    /// One line for reports and headers.
    pub fn describe(&self) -> String {
        let mut p = Vec::new();
        match self.sharpen {
            Sharpen::Off => {}
            Sharpen::Correct => p.push(format!("correct stars {:.0}%", self.sharpen_amount * 100.0)),
            Sharpen::Tighten => p.push(format!("sharpen {:.0}%", self.sharpen_amount * 100.0)),
        }
        if self.denoise {
            p.push(format!("denoise {:.0}%", self.denoise_amount * 100.0));
        }
        if p.is_empty() {
            return "off".into();
        }
        format!("{} ({})", p.join(", "), match self.engine { Engine::Crux => "crux", Engine::RcAstro => "RC Astro" })
    }
}

/// What can run here, for the UI: `Ok` or why not.
pub fn availability(engine: Engine, sharpen: bool) -> std::result::Result<(), String> {
    match engine {
        Engine::Crux => {
            #[cfg(feature = "ml")]
            {
                if sharpen && super::net::builtin(super::net::Kind::Sharpen).is_none() {
                    return Err("crux's star corrector isn't in this build yet".into());
                }
                Ok(())
            }
            #[cfg(not(feature = "ml"))]
            {
                let _ = sharpen;
                Err("this build of crux has no networks (built without the ml feature)".into())
            }
        }
        Engine::RcAstro => {
            let exe = crate::rcastro::find().ok_or("RC Astro's CLI (rc-astro) isn't installed")?;
            let cli = crate::rcastro::probe(&exe).map_err(|e| format!("{e:#}"))?;
            let need: &[&str] = if sharpen { &["bxt"] } else { &["nxt"] };
            for k in need {
                if !cli.licensed(k) {
                    return Err(format!("{} isn't licensed on this computer", if *k == "bxt" { "BlurXTerminator" } else { "NoiseXTerminator" }));
                }
            }
            Ok(())
        }
    }
}

#[derive(Debug, Clone)]
pub struct Progress {
    /// Master index and count.
    pub master: usize,
    pub masters: usize,
    pub channel: String,
    /// "denoise", "sharpen", "blend", "BlurXTerminator" …
    pub step: String,
    /// 0–1 within the step.
    pub done: f32,
}

/// Serialises `apply` and `starnet::apply`, which share the cache folders.
pub(crate) static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// A temporary file name nobody else is using, beside `p`.
pub(crate) fn tmp_path(p: &Path) -> PathBuf {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    p.with_extension(format!("{}-{n}.tmp", std::process::id()))
}

fn hex(s: &str) -> String {
    let d = Sha256::digest(s.as_bytes());
    d.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

fn master_sig(p: &Path) -> String {
    let (size, mtime) = std::fs::metadata(p)
        .map(|m| (m.len(), m.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_nanos()).unwrap_or(0)))
        .unwrap_or((0, 0));
    format!("{}:{size}:{mtime}", crate::paths::key(p))
}

/// The enhanced target: same channels, masters replaced by enhanced files
/// (made now, or found in the cache). Inactive settings return `t` as is.
pub fn apply(t: &Target, out: &Path, s: &EnhanceSettings, progress: &(dyn Fn(Progress) + Sync), cancel: &AtomicBool) -> Result<Target> {
    if !s.active() {
        return Ok(t.clone());
    }
    // One at a time (the preview and a full-size write can ask together):
    // the second finds the first one's files in the cache.
    let _one = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = t.image_dir(out).join("enhanced");
    std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
    let mut res = t.clone();
    let n = t.masters.len();
    for (i, m) in t.masters.iter().enumerate() {
        if cancel.load(std::sync::atomic::Ordering::Relaxed) {
            anyhow::bail!("cancelled");
        }
        let ch = crate::process::job::safe_name(&m.channel);
        let mh = hex(&master_sig(&m.path));
        let tags = model_tags(s);
        let fin = dir.join(format!("{ch}_{mh}_final-{}.xisf", hex(&format!("{}|{tags}", serde_json::to_string(s).unwrap_or_default()))));
        let p = |step: &str, done: f32| progress(Progress { master: i, masters: n, channel: m.channel.clone(), step: step.into(), done });
        if fin.is_file() {
            touch(&fin);
        } else {
            match s.engine {
                Engine::Crux => crux_engine(&m.path, &dir, &ch, &mh, s, &fin, &p, cancel)?,
                Engine::RcAstro => rc_engine(&m.path, s, &fin, &p, cancel)?,
            }
        }
        tidy(&dir, &ch, &mh, &fin);
        res.masters[i].path = fin;
    }
    Ok(res)
}

fn model_tags(s: &EnhanceSettings) -> String {
    #[cfg(feature = "ml")]
    {
        use super::net::{builtin, Kind};
        let t = |k: Kind| builtin(k).map(|b| b.2).unwrap_or("none");
        match s.engine {
            Engine::Crux => format!("{}+{}", t(Kind::Denoise), t(Kind::Sharpen)),
            Engine::RcAstro => "rc-astro".into(),
        }
    }
    #[cfg(not(feature = "ml"))]
    {
        let _ = s;
        "no-ml".into()
    }
}

/// How many results per channel are kept: switching between a few settings
/// (crux and RC Astro, say) finds each one on disk.
pub(crate) const KEEP_RESULTS: usize = 3;

/// Mark a reused cache file as just used, so tidying keeps it. Sets the
/// last-access time only: the modified time is part of the starless and
/// prepared caches' signatures, so changing it would make them miss.
pub(crate) fn touch(p: &Path) {
    if let Ok(f) = std::fs::OpenOptions::new().write(true).open(p) {
        let _ = f.set_times(std::fs::FileTimes::new().set_accessed(std::time::SystemTime::now()));
    }
}

/// Of `files`, all but the `keep` most recently used (always keeping `current`).
pub(crate) fn oldest_beyond(mut files: Vec<PathBuf>, current: &Path, keep: usize) -> Vec<PathBuf> {
    let used = |p: &PathBuf| std::fs::metadata(p).and_then(|m| m.accessed().or_else(|_| m.modified())).unwrap_or(std::time::UNIX_EPOCH);
    files.retain(|p| p != current);
    files.sort_by_key(|p| std::cmp::Reverse(used(p)));
    files.into_iter().skip(keep.saturating_sub(1)).collect()
}

/// Remove this channel's files from other masters (an older master or a
/// replaced file), and finals beyond the most recently used few.
fn tidy(dir: &Path, ch: &str, mh: &str, current: &Path) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut finals = Vec::new();
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        let Some(rest) = name.strip_prefix(&format!("{ch}_")) else { continue };
        if !rest.starts_with(mh) || rest.ends_with(".tmp") {
            let _ = std::fs::remove_file(e.path());
        } else if rest.contains("_final-") && rest.ends_with(".xisf") {
            finals.push(e.path());
        }
    }
    for p in oldest_beyond(finals, current, KEEP_RESULTS) {
        let _ = std::fs::remove_file(p);
    }
}

#[allow(clippy::too_many_arguments)]
fn crux_engine(master: &Path, dir: &Path, ch: &str, mh: &str, s: &EnhanceSettings, fin: &Path, p: &(dyn Fn(&str, f32) + Sync), cancel: &AtomicBool) -> Result<()> {
    #[cfg(feature = "ml")]
    {
        use super::net::{Kind, Network, Settings};
        use rayon::prelude::*;
        let laps = crate::timing::Laps::new(format!("enhance {ch}"));
        let (orig, hdr) = crate::process::read_with_header(master)?;
        laps.lap("read");
        let (noise, sky, fwhm) = super::net::measure(&orig);
        laps.lap(&format!("noise, sky and FWHM ({fwhm:.2} px)"));
        let cached = |kind: Kind, control: f32, step: &str| -> Result<Image> {
            let (_, _, tag) = super::net::builtin(kind).ok_or_else(|| anyhow::anyhow!("this build of crux has no {} network", kind.format()))?;
            let file = dir.join(format!("{ch}_{mh}_{step}-{tag}.xisf"));
            if let Ok(img) = crate::read_image(&file) {
                if (img.width, img.height) == (orig.width, orig.height) {
                    return Ok(img);
                }
            }
            p(step, 0.0);
            let net = Network::shared(kind)?;
            let out = net.run(&orig, noise, sky, fwhm, &[control], &Settings::default(), &|k, n| p(step, k as f32 / n.max(1) as f32), cancel)?;
            laps.lap(&format!("{step} on the {}", net.engine()));
            write(&file, &out, &Header::default())?;
            laps.lap(&format!("{step} written"));
            Ok(out)
        };
        let d = cached(Kind::Denoise, 0.0, "denoise")?;
        let sh = match s.sharpen {
            Sharpen::Off => None,
            Sharpen::Correct => Some(cached(Kind::Sharpen, 0.0, "correct")?),
            Sharpen::Tighten => Some(cached(Kind::Sharpen, 1.0, "tighten")?),
        };
        p("blend", 0.0);
        let ad = if s.denoise { s.denoise_amount.clamp(0.0, 1.0) } else { 0.0 };
        let as_ = s.sharpen_amount.clamp(0.0, 1.0);
        let data: Vec<f32> = (0..orig.data.len())
            .into_par_iter()
            .map(|k| {
                let o = orig.data[k];
                if o <= 0.0 {
                    return 0.0;
                }
                let dv = d.data[k];
                let mut v = o - ad * (o - dv);
                if let Some(sv) = &sh {
                    v += as_ * (sv.data[k] - dv);
                }
                v.clamp(1e-7, 1.0)
            })
            .collect();
        let mut img = Image::from_data(orig.width, orig.height, data);
        img.adu_scale = orig.adu_scale;
        let mut h = hdr;
        h.push("HISTORY", "", &format!("crux enhance: {}", s.describe()));
        laps.lap("blend");
        write(fin, &img, &h)?;
        laps.lap("final written");
        p("blend", 1.0);
        Ok(())
    }
    #[cfg(not(feature = "ml"))]
    {
        let _ = (master, dir, ch, mh, s, fin, p, cancel);
        anyhow::bail!("this build of crux has no networks (built without the ml feature)")
    }
}

fn rc_engine(master: &Path, s: &EnhanceSettings, fin: &Path, p: &(dyn Fn(&str, f32) + Sync), cancel: &AtomicBool) -> Result<()> {
    use crate::rcastro::{self, Bxt, Nxt, Step};
    let exe = rcastro::find().ok_or_else(|| anyhow::anyhow!("RC Astro's CLI (rc-astro) isn't installed"))?;
    let cli = rcastro::probe(&exe)?;
    let a = s.sharpen_amount.clamp(0.0, 1.0);
    let mut steps = Vec::new();
    match s.sharpen {
        Sharpen::Off => {}
        Sharpen::Correct => steps.push(Step::Bxt(Bxt { correct_only: true, ..Default::default() })),
        Sharpen::Tighten => steps.push(Step::Bxt(Bxt { sharpen_stars: 0.5 * a, sharpen_nonstellar: 0.5 * a, ..Default::default() })),
    }
    if s.denoise {
        steps.push(Step::Nxt(Nxt { denoise: s.denoise_amount.clamp(0.0, 1.0), ..Default::default() }));
    }
    rcastro::run_chain(&cli, &steps, master, fin, None, &|k, e| {
        if let rcastro::Event::Progress(d) = e {
            p(steps[k].name(), d / 100.0);
        }
    }, cancel)?;
    Ok(())
}

#[cfg_attr(not(feature = "ml"), allow(dead_code))]
fn write(p: &Path, img: &Image, h: &Header) -> Result<()> {
    let tmp = tmp_path(p);
    crate::xisf::write_image(&tmp, img, h, None)?;
    std::fs::rename(&tmp, p).with_context(|| format!("replace {}", p.display()))?;
    Ok(())
}

/// The enhanced master that would be used for `master` under `s`, if made.
pub fn cached_path(t: &Target, out: &Path, s: &EnhanceSettings) -> Vec<Option<PathBuf>> {
    let dir = t.image_dir(out).join("enhanced");
    t.masters
        .iter()
        .map(|m| {
            let ch = crate::process::job::safe_name(&m.channel);
            let mh = hex(&master_sig(&m.path));
            let f = dir.join(format!("{ch}_{mh}_final-{}.xisf", hex(&format!("{}|{}", serde_json::to_string(s).unwrap_or_default(), model_tags(s)))));
            f.is_file().then_some(f)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_describe_and_round_trip() {
        let s = EnhanceSettings { denoise: true, sharpen: Sharpen::Correct, ..Default::default() };
        assert_eq!(s.describe(), "correct stars 100%, denoise 80% (crux)");
        assert_eq!(EnhanceSettings::default().active(), cfg!(feature = "ml"));
        let t = toml::to_string(&s).unwrap();
        let back: EnhanceSettings = toml::from_str(&t).unwrap();
        assert_eq!(back, s);
        // Old recipes without the section still load.
        let empty: EnhanceSettings = toml::from_str("").unwrap();
        assert_eq!(empty, EnhanceSettings::default());
    }

    #[cfg(feature = "ml")]
    #[test]
    fn denoise_is_cached_and_blended() {
        let dir = std::env::temp_dir().join(format!("crux-enhance-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (w, h) = (300usize, 200usize);
        let mut seed = 7u64;
        let data: Vec<f32> = (0..w * h)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                0.01 + 0.0003 * (((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5) * 3.46
            })
            .collect();
        let master = dir.join("m.xisf");
        crate::xisf::write_image(&master, &Image::from_data(w, h, data), &Header::default(), None).unwrap();
        let t = Target { object: "Test".into(), masters: vec![super::super::Master { channel: "L".into(), filter: "L".into(), path: master.clone(), frames: 3, exposure: 900.0, starless: None, noise: 0.0 }] };
        let s = EnhanceSettings { denoise: true, denoise_amount: 1.0, sharpen: Sharpen::Off, ..Default::default() };
        let e = apply(&t, &dir, &s, &|_| {}, &AtomicBool::new(false)).unwrap();
        let full = crate::read_image(&e.masters[0].path).unwrap();
        let s2 = EnhanceSettings { denoise_amount: 0.5, ..s.clone() };
        let e2 = apply(&t, &dir, &s2, &|_| {}, &AtomicBool::new(false)).unwrap();
        let half = crate::read_image(&e2.masters[0].path).unwrap();
        let orig = crate::read_image(&master).unwrap();
        // Half strength sits halfway between the original and full strength.
        for k in [0usize, 1234, 40000] {
            let want = 0.5 * (orig.data[k] + full.data[k]);
            assert!((half.data[k] - want).abs() < 1e-6, "{k}");
        }
        // Both finals are kept, so going back to the first is free.
        let finals = || std::fs::read_dir(t.image_dir(&dir).join("enhanced")).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().contains("_final-")).map(|e| e.path()).collect::<Vec<_>>();
        assert_eq!(finals().len(), 2);
        let mtime = |p: &Path| std::fs::metadata(p).unwrap().modified().unwrap();
        let before = mtime(&e.masters[0].path);
        let again = apply(&t, &dir, &s, &|_| {}, &AtomicBool::new(false)).unwrap();
        assert_eq!(again.masters[0].path, e.masters[0].path);
        // Reuse leaves the modified time (later caches' signatures) alone.
        assert_eq!(mtime(&again.masters[0].path), before);
        // Past KEEP_RESULTS, the least recently used goes (here s2's).
        std::thread::sleep(std::time::Duration::from_millis(20));
        for a in [0.2f32, 0.3] {
            apply(&t, &dir, &EnhanceSettings { denoise_amount: a, ..s.clone() }, &|_| {}, &AtomicBool::new(false)).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        let left = finals();
        assert_eq!(left.len(), KEEP_RESULTS);
        assert!(left.contains(&e.masters[0].path) && !left.contains(&e2.masters[0].path));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
