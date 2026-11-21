//! Autopilot (milestone 14): "process all of `\\nas\astronomy` again".
//!
//! Finds the data folders under a root, estimates the work (files and
//! bytes to read, time), then for each folder: triage (scan, measure,
//! grade, calibration checks), calibrate, register and integrate, mosaic
//! when there are panels, and an image per target with its last recipe or
//! crux's default. Everything goes to the workspace; nothing in the data
//! folders changes. Ends with `autopilot.json` and a gallery page
//! (`autopilot.html`) of the finished images and the decisions worth a look.

use crate::cache::Cache;
use crate::config::Config;
use crate::grade::Grade;
use crate::process::job;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

/// Is this folder name dated ("2026-11-05_LMC_Mosaic", "20120419 …")?
fn dated(name: &str) -> bool {
    let b = name.as_bytes();
    let digits = |r: std::ops::Range<usize>| r.clone().all(|i| b.get(i).is_some_and(|c| c.is_ascii_digit()));
    (b.len() >= 10 && digits(0..4) && b[4] == b'-' && digits(5..7) && b[7] == b'-' && digits(8..10)) || (b.len() >= 8 && digits(0..8) && b.get(8).is_none_or(|c| !c.is_ascii_digit()))
}

pub(crate) fn hidden_or_output(name: &str) -> bool {
    let l = name.to_lowercase();
    l.starts_with('.') || l.starts_with('_') || matches!(l.as_str(), "calibrated" | "registered" | "logs" | "master.old" | "cosmetized" | "debayered" | "fastintegration") || l.starts_with("mosaic")
}

/// Data folders under `root`: dated folder names (James's layout and
/// N.I.N.A.'s), found up to four levels down (year folders on the NAS);
/// else the root's subfolders; else the root itself.
pub fn discover(root: &Path) -> Vec<PathBuf> {
    fn walk(d: &Path, depth: usize, out: &mut Vec<PathBuf>) {
        let Ok(rd) = std::fs::read_dir(d) else { return };
        let mut subs: Vec<PathBuf> = rd.flatten().filter(|e| e.file_type().is_ok_and(|t| t.is_dir())).map(|e| e.path()).collect();
        subs.sort();
        for s in subs {
            let name = s.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            if hidden_or_output(&name) {
                continue;
            }
            if dated(&name) {
                out.push(s);
            } else if depth > 1 {
                walk(&s, depth - 1, out);
            }
        }
    }
    let mut out = Vec::new();
    let root_name = root.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    if dated(&root_name) {
        return vec![root.to_path_buf()];
    }
    walk(root, 4, &mut out);
    if out.is_empty() {
        if let Ok(rd) = std::fs::read_dir(root) {
            out = rd.flatten().filter(|e| e.file_type().is_ok_and(|t| t.is_dir())).map(|e| e.path()).filter(|p| !p.file_name().is_some_and(|n| hidden_or_output(&n.to_string_lossy()))).collect();
            out.sort();
        }
    }
    if out.is_empty() {
        out.push(root.to_path_buf());
    }
    out
}

/// Where a folder's cache goes: under `explicit` (one subfolder each, so
/// folders never share a cache) or crux's usual place.
pub fn cache_for(explicit: Option<&Path>, dir: &Path, cfg: &Config) -> PathBuf {
    match explicit {
        Some(e) => e.join(job::safe_name(&dir.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| "folder".into()))),
        None => crate::cachedir::resolve(None, dir, cfg.cache.in_folder),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FolderEstimate {
    pub dir: PathBuf,
    pub files: usize,
    pub bytes: u64,
    /// Raw lights outside reject folders, and how many crux has measured.
    pub lights: usize,
    pub light_bytes: u64,
    pub measured: usize,
    pub measured_bytes: u64,
    pub targets: Vec<String>,
    /// Integrated before (the workspace has its reports).
    pub processed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Estimate {
    pub root: PathBuf,
    pub workspace: PathBuf,
    pub folders: Vec<FolderEstimate>,
    pub files: usize,
    pub bytes: u64,
    pub lights: usize,
    pub to_measure: usize,
    /// Bytes read in all: unmeasured lights once to measure, every light
    /// once to calibrate.
    pub read_bytes: u64,
    pub seconds: f64,
    /// On a network share (slower reads).
    pub remote: bool,
}

pub fn human_bytes(b: u64) -> String {
    let g = b as f64 / 1e9;
    if g >= 1.0 {
        format!("{g:.0} GB")
    } else {
        format!("{:.0} MB", b as f64 / 1e6)
    }
}

pub fn human_time(s: f64) -> String {
    if s >= 3600.0 {
        format!("{:.1} h", s / 3600.0)
    } else if s >= 60.0 {
        format!("{:.0} min", s / 60.0)
    } else {
        format!("{s:.0} s")
    }
}

impl Estimate {
    /// "6,400 subs in 14 folders, ~330 GB to read over the network, about
    /// 4 h; output to D:\crux-work; nothing in the data folders changes."
    pub fn summary(&self) -> String {
        format!(
            "{} subs in {} folder{} ({} to measure), {} to read{}, about {}; output to {}; nothing in the data folders changes.",
            self.lights,
            self.folders.len(),
            if self.folders.len() == 1 { "" } else { "s" },
            self.to_measure,
            human_bytes(self.read_bytes),
            if self.remote { " over the network" } else { "" },
            human_time(self.seconds),
            self.workspace.display()
        )
    }
}

fn is_raw_light(f: &crate::Frame) -> bool {
    f.class.frame_type == crate::classify::FrameType::Light && f.class.role == crate::classify::Role::Raw && !f.class.in_reject_dir
}

/// Header-only look at each folder (fast: headers come from the cache
/// after the first time), and the time it should take.
pub fn estimate(root: &Path, folders: &[PathBuf], cfg: &Config, cache_root: Option<&Path>, workspace: &Path) -> Estimate {
    let mut e = Estimate { root: root.to_path_buf(), workspace: workspace.to_path_buf(), remote: root.to_string_lossy().starts_with("\\\\"), ..Default::default() };
    for d in folders {
        let (fcfg, _) = Config::load(None, Some(d)).unwrap_or_else(|_| (cfg.clone(), None));
        let cdir = cache_for(cache_root, d, &fcfg);
        let mut cache = Cache::load(&cdir);
        let t = crate::pipeline::run(d, &fcfg, &mut cache, false, &|_, _, _| {});
        let _ = cache.save();
        let mut fe = FolderEstimate { dir: d.clone(), files: t.scan.frames.len(), ..Default::default() };
        let mut targets = std::collections::BTreeSet::new();
        for f in &t.scan.frames {
            fe.bytes += f.meta.file_size;
            if is_raw_light(f) {
                fe.lights += 1;
                fe.light_bytes += f.meta.file_size;
                if cache.get(&f.meta.path, f.meta.file_size, f.meta.mtime_ns).is_some_and(|en| en.analysis.is_some()) {
                    fe.measured += 1;
                    fe.measured_bytes += f.meta.file_size;
                }
                if let Some(o) = &f.meta.object {
                    targets.insert(crate::index::target_of(o));
                }
            }
        }
        fe.targets = targets.into_iter().collect();
        fe.processed = job::out_dir(d, workspace).join("crux-integration.json").is_file();
        if fe.lights > 0 {
            e.folders.push(fe);
        }
    }
    let rate = if e.remote { 80e6 } else { 400e6 };
    for f in &e.folders {
        e.files += f.files;
        e.bytes += f.bytes;
        e.lights += f.lights;
        e.to_measure += f.lights - f.measured;
        e.read_bytes += (f.light_bytes - f.measured_bytes) + f.light_bytes;
    }
    // Measured on the LMC data (8 cores): 0.16 s per light to measure, ~1 s
    // to calibrate and register, ~0.4 s to integrate; ~10 s per target for
    // masters, the mosaic and the image.
    let n_targets: usize = e.folders.iter().map(|f| f.targets.len()).sum();
    let cpu = e.to_measure as f64 * 0.16 + e.lights as f64 * 1.4 + n_targets as f64 * 10.0;
    e.seconds = cpu.max(e.read_bytes as f64 / rate) + e.folders.len() as f64 * 5.0;
    e
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetResult {
    pub object: String,
    pub channels: Vec<String>,
    pub frames: usize,
    pub hours: f64,
    pub recipe: String,
    pub image: Option<PathBuf>,
    pub preview: Option<PathBuf>,
    pub mosaic: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FolderResult {
    pub dir: PathBuf,
    pub out: PathBuf,
    pub lights: usize,
    pub good: usize,
    pub warn: usize,
    pub reject: usize,
    pub processed: usize,
    pub up_to_date: usize,
    pub failed: usize,
    pub integrated: usize,
    pub targets: Vec<TargetResult>,
    /// Decisions crux wasn't sure about: worth a look in the morning.
    pub inbox: Vec<String>,
    pub error: Option<String>,
    pub seconds: f64,
}

#[derive(Debug, Clone)]
pub enum Event {
    Folder { index: usize, total: usize, dir: PathBuf },
    /// What's happening now, one line.
    Stage(String),
    /// Progress within the stage.
    Progress { done: usize, total: usize },
    Finished(Box<FolderResult>),
}

#[derive(Debug, Clone, Default)]
pub struct Options {
    pub workspace: PathBuf,
    /// Caches under this folder (tests); crux's usual place otherwise.
    pub cache_root: Option<PathBuf>,
    pub config: Option<PathBuf>,
    /// Stop after integrating (no mosaics or images).
    pub no_images: bool,
}

/// Process one data folder end to end.
pub fn run_folder(dir: &Path, opts: &Options, cat: Option<&crate::catalog::Catalog>, progress: &(dyn Fn(Event) + Sync), cancel: &AtomicBool) -> FolderResult {
    let t0 = std::time::Instant::now();
    let mut r = FolderResult { dir: dir.to_path_buf(), out: job::out_dir(dir, &opts.workspace), ..Default::default() };
    if let Err(e) = run_folder_inner(dir, opts, cat, progress, cancel, &mut r) {
        r.error = Some(format!("{e:#}"));
    }
    r.seconds = t0.elapsed().as_secs_f64();
    r
}

fn run_folder_inner(dir: &Path, opts: &Options, cat: Option<&crate::catalog::Catalog>, progress: &(dyn Fn(Event) + Sync), cancel: &AtomicBool, r: &mut FolderResult) -> Result<()> {
    let (mut cfg, _) = Config::load(opts.config.as_deref(), Some(dir))?;
    let cdir = cache_for(opts.cache_root.as_deref(), dir, &cfg);
    let mut cache = Cache::load(&cdir);
    let decisions = crate::decisions::Decisions::load(&cdir).ok();
    if let Some(d) = &decisions {
        d.apply_thresholds(&mut cfg);
        d.apply_layout(&mut cfg);
    }
    // 1. Triage.
    progress(Event::Stage("Measuring and grading".into()));
    let mut t = crate::pipeline::run_with(dir, &cfg, &mut cache, true, &|_, d, n| progress(Event::Progress { done: d, total: n }), cancel);
    if let Some(d) = &decisions {
        d.apply_overrides(&t.scan.frames, &mut t.reports);
        d.apply_flat_choices(dir, &mut t.cal, &t.grouping, t.library.as_ref());
    }
    let _ = cache.save();
    anyhow::ensure!(!cancel.load(Ordering::Relaxed), "cancelled");
    for rep in &t.reports {
        // Lights filed as rejects count only when crux stacks them (graded good).
        let f = &t.scan.frames[rep.frame];
        if !(is_raw_light(f) || (f.class.in_reject_dir && rep.grade == Grade::Good && f.class.frame_type == crate::classify::FrameType::Light && f.class.role == crate::classify::Role::Raw)) {
            continue;
        }
        r.lights += 1;
        match rep.grade {
            Grade::Good => r.good += 1,
            Grade::Warn => r.warn += 1,
            Grade::Reject => r.reject += 1,
        }
    }
    if r.lights == 0 {
        r.inbox.push("No lights here (calibration frames only?).".into());
        return Ok(());
    }
    // Flats that exist but are doubtful (missing ones are in the plan's notes).
    for c in &t.cal.choices {
        if c.grade != Grade::Good && c.chosen.is_some() {
            r.inbox.push(format!("Flats for {} {}: {}", c.filter, c.night, c.reason));
        }
    }
    if r.warn > 0 && cfg.process.include_warn {
        r.inbox.push(format!("{} warn frame{} went into the stacks: check them in Frames.", r.warn, if r.warn == 1 { "" } else { "s" }));
    }

    // 2. Calibrate, register, integrate.
    progress(Event::Stage("Planning".into()));
    let mut s = job::Settings::from_config(&cfg);
    s.workspace = opts.workspace.clone();
    let lacking = t.grouping.light_calibration.iter().any(|c| c.darks.is_empty() || c.bias.is_empty());
    let fp = if lacking {
        progress(Event::Stage("Looking for darks and bias in the folders beside it and the calibration library".into()));
        let others = crate::fingerprint::other_folders(dir, &cfg, &[]);
        Some(crate::fingerprint::build(&t, dir, &others, &cache, &|_, _| {}))
    } else {
        None
    };
    let plan = job::plan(&t, dir, decisions.as_ref(), &s, cat, fp.as_ref())?;
    for n in &plan.notes {
        r.inbox.push(n.clone());
    }
    progress(Event::Stage(format!("Calibrating and registering {} lights", plan.n_lights())));
    let last = std::sync::Mutex::new(0usize);
    let rep = job::run(
        &plan,
        &|e| match e {
            job::Event::Light { done, total, .. } => progress(Event::Progress { done, total }),
            job::Event::Integrate { index, total, .. } => {
                let mut l = last.lock().unwrap();
                if *l == 0 {
                    progress(Event::Stage(format!("Integrating {total} master light{}", if total == 1 { "" } else { "s" })));
                }
                if index != *l {
                    *l = index;
                    progress(Event::Progress { done: index - 1, total });
                }
            }
            _ => {}
        },
        cancel,
    )?;
    r.processed = rep.processed;
    r.up_to_date = rep.up_to_date;
    r.failed = rep.failed;
    r.integrated = rep.integrated + rep.integrations_up_to_date;
    if rep.failed > 0 {
        r.inbox.push(format!("{} light{} failed to process.", rep.failed, if rep.failed == 1 { "" } else { "s" }));
    }
    anyhow::ensure!(!cancel.load(Ordering::Relaxed), "cancelled");
    if opts.no_images {
        return Ok(());
    }

    // 3. Mosaics.
    let mut mosaic_targets = std::collections::BTreeSet::new();
    if cfg.mosaic.after_integrate {
        for g in crate::process::mosaic::groups(&plan.out) {
            let Some(cat) = cat else {
                r.inbox.push(format!("{}: {} panels, but no star database to place them (install ASTAP with D50).", g.target, g.panels()));
                continue;
            };
            progress(Event::Stage(format!("Mosaic {}: {} panels", g.target, g.panels())));
            match crate::process::mosaic::assemble(&g, &plan.out, &cfg.mosaic, cat, false, cfg.process.fits, &|_| {}, cancel) {
                Ok(reps) => {
                    mosaic_targets.insert(g.target.clone());
                    for m in reps {
                        for p in m.panels.iter().filter(|p| p.error.is_some()) {
                            r.inbox.push(format!("Mosaic {} {}: left out {}: {}", m.target, m.filter, p.object, p.error.as_deref().unwrap_or("")));
                        }
                        if m.worst_seam() > 1.0 {
                            r.inbox.push(format!("Mosaic {} {}: a seam is {:.1}σ out after matching; check it.", m.target, m.filter, m.worst_seam()));
                        }
                    }
                }
                Err(e) => r.inbox.push(format!("Mosaic {} failed: {e:#}", g.target)),
            }
        }
    }

    // 4. Images: a mosaic stands in for its panels.
    for tg in crate::render::targets(&plan.out) {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let is_mosaic = mosaic_targets.contains(&tg.object);
        if !is_mosaic && mosaic_targets.contains(&crate::index::target_of(&tg.object)) {
            continue;
        }
        progress(Event::Stage(format!("Image {}", tg.object)));
        r.targets.push(image_target(&tg, &plan.out, is_mosaic, cfg.process.fits, cancel));
    }
    Ok(())
}

/// Render a target with its last recipe (or crux's default): full image
/// plus a 1,200 px preview (and a FITS copy with `fits`).
pub fn image_target(tg: &crate::render::Target, out: &Path, is_mosaic: bool, fits: bool, cancel: &AtomicBool) -> TargetResult {
    use crate::render::{self, recipe::Recipe};
    let names = tg.channels();
    let dir_out = tg.image_dir(out);
    let mut rec = Recipe::load(&dir_out.join("recipe.toml")).unwrap_or_else(|_| Recipe::for_channels(&names));
    rec.fit_to(&names);
    if rec.name.is_empty() {
        rec.name = if rec.mix.preset.is_empty() { "Custom".into() } else { rec.mix.preset.clone() };
    }
    let mut res = TargetResult {
        object: tg.object.clone(),
        channels: names.clone(),
        frames: tg.masters.iter().map(|m| m.frames).sum(),
        hours: tg.masters.iter().map(|m| m.exposure).sum::<f64>() / 3600.0,
        recipe: rec.name.clone(),
        image: None,
        preview: None,
        mosaic: is_mosaic,
        error: None,
    };
    let go = || -> Result<(PathBuf, PathBuf)> {
        let tg = &render::for_recipe(tg, out, &rec, &|_| {}, cancel)?;
        let p = render::prepare(tg, &rec.gradient, &rec.stars, true, cancel)?;
        let rgb = render::render_full(&p, &rec)?;
        let mut extra = crate::header::Header::default();
        extra.push("NCHANNEL", names.len().to_string(), "masters mixed");
        let w = render::write(&rgb, &dir_out, &tg.object, &rec, &extra, fits)?;
        Ok((w.png, w.preview))
    };
    match go() {
        Ok((png, prev)) => {
            res.image = Some(png);
            res.preview = Some(prev);
        }
        Err(e) => res.error = Some(format!("{e:#}")),
    }
    res
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Report {
    pub root: PathBuf,
    pub workspace: PathBuf,
    pub estimate: Estimate,
    pub folders: Vec<FolderResult>,
    pub seconds: f64,
    pub cancelled: bool,
}

/// Run every folder of an estimate, then write `autopilot.json` and
/// `autopilot.html` to the workspace.
pub fn run_all(est: &Estimate, opts: &Options, progress: &(dyn Fn(Event) + Sync), cancel: &AtomicBool) -> Result<Report> {
    let t0 = std::time::Instant::now();
    let cat = crate::catalog::Catalog::find();
    let mut rep = Report { root: est.root.clone(), workspace: opts.workspace.clone(), estimate: est.clone(), ..Default::default() };
    let total = est.folders.len();
    for (k, f) in est.folders.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            rep.cancelled = true;
            break;
        }
        progress(Event::Folder { index: k + 1, total, dir: f.dir.clone() });
        let fr = run_folder(&f.dir, opts, cat.as_ref(), progress, cancel);
        progress(Event::Finished(Box::new(fr.clone())));
        rep.folders.push(fr);
        rep.seconds = t0.elapsed().as_secs_f64();
        write_report(&rep)?;
    }
    rep.seconds = t0.elapsed().as_secs_f64();
    rep.cancelled |= cancel.load(Ordering::Relaxed);
    write_report(&rep)?;
    Ok(rep)
}

pub fn report_path(workspace: &Path) -> PathBuf {
    workspace.join("autopilot.json")
}

pub fn gallery_path(workspace: &Path) -> PathBuf {
    workspace.join("autopilot.html")
}

pub fn load_report(workspace: &Path) -> Option<Report> {
    std::fs::read(report_path(workspace)).ok().and_then(|b| serde_json::from_slice(&b).ok())
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

/// A file URL relative to the workspace when it's inside it.
fn rel(ws: &Path, p: &Path) -> String {
    let s = match p.strip_prefix(ws) {
        Ok(r) => r.to_string_lossy().to_string(),
        Err(_) => format!("file:///{}", p.to_string_lossy()),
    };
    s.replace('\\', "/").split('/').map(|c| c.replace(' ', "%20").replace('#', "%23")).collect::<Vec<_>>().join("/")
}

fn write_report(rep: &Report) -> Result<()> {
    std::fs::create_dir_all(&rep.workspace)?;
    crate::paths::write_atomic(&report_path(&rep.workspace), &serde_json::to_vec_pretty(rep)?)?;
    std::fs::write(gallery_path(&rep.workspace), gallery_html(rep))?;
    Ok(())
}

/// The morning gallery: finished images, then what needs you.
pub fn gallery_html(rep: &Report) -> String {
    let ws = &rep.workspace;
    let mut cards = String::new();
    let mut inbox = String::new();
    for f in &rep.folders {
        let folder = f.dir.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        for t in &f.targets {
            let img = match (&t.preview, &t.error) {
                (Some(p), None) => format!("<a href=\"{}\"><img src=\"{}\" alt=\"{}\" loading=\"lazy\"></a>", rel(ws, t.image.as_deref().unwrap_or(p)), rel(ws, p), esc(&t.object)),
                (_, Some(e)) => format!("<div class=\"fail\">{}</div>", esc(e)),
                _ => String::new(),
            };
            cards.push_str(&format!(
                "<figure>{img}<figcaption><span><b>{}</b>{}</span><span>{} · {} frames · {:.1} h · {}</span><span class=\"dim\">{}</span></figcaption></figure>\n",
                esc(&t.object),
                if t.mosaic { " <i>mosaic</i>" } else { "" },
                esc(&t.channels.join(" ")),
                t.frames,
                t.hours,
                esc(&t.recipe),
                esc(&folder)
            ));
        }
        let mut items: Vec<String> = f.inbox.clone();
        if let Some(e) = &f.error {
            items.insert(0, format!("Failed: {e}"));
        }
        if !items.is_empty() {
            inbox.push_str(&format!("<h3>{}</h3><p class=\"dim\">{} lights: {} good, {} warn, {} reject</p><ul>", esc(&folder), f.lights, f.good, f.warn, f.reject));
            for i in items {
                inbox.push_str(&format!("<li>{}</li>", esc(&i)));
            }
            inbox.push_str("</ul>");
        }
    }
    if inbox.is_empty() {
        inbox = "<p class=\"dim\">Nothing to check.</p>".into();
    }
    format!(
        r#"<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>crux autopilot</title>
<style>
:root{{--bg:#0d1117;--card:#161b22;--ink:#e6edf3;--dim:#8b949e;--accent:#58a6ff;--bad:#f85149}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,"Segoe UI",sans-serif}}
main{{max-width:1500px;margin:0 auto;padding:24px 16px}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:17px;margin:28px 0 10px}} h3{{font-size:15px;margin:18px 0 2px}}
.dim{{color:var(--dim)}} .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px}}
figure{{margin:0;background:var(--card);border-radius:8px;overflow:hidden}} figure img{{width:100%;display:block;background:#000;aspect-ratio:3/2;object-fit:contain}}
figcaption{{padding:8px 10px;display:flex;flex-direction:column;gap:2px}} figcaption i{{color:var(--accent);font-style:normal;font-size:12px}}
.fail{{padding:30px 10px;color:var(--bad)}} ul{{margin:4px 0;padding-left:20px}} a{{color:var(--accent)}}
</style></head><body><main>
<h1>crux autopilot</h1><p class="dim">{} → {} · {} folders in {}{}</p>
<h2>Finished images</h2><div class="grid">{}</div>
<h2>Needs checking</h2>{}
</main></body></html>"#,
        esc(&rep.root.display().to_string()),
        esc(&rep.workspace.display().to_string()),
        rep.folders.len(),
        human_time(rep.seconds),
        if rep.cancelled { " (stopped early)" } else { "" },
        if cards.is_empty() { "<p class=\"dim\">None yet.</p>".to_string() } else { cards },
        inbox
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dated_names() {
        assert!(dated("2026-11-05_LMC_Mosaic"));
        assert!(dated("2012-04-19 NGC 6188b"));
        assert!(dated("20120415_1813"));
        assert!(!dated("2012"));
        assert!(!dated("LIGHT"));
        assert!(!dated("123456789"));
    }

    #[test]
    fn discovers_dated_folders_below_year_folders() {
        let root = std::env::temp_dir().join(format!("crux-auto-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        for d in ["2012/2012-04-19 NGC 6188b/DAY_2", "2012/2012-04-15 ngc3372", "2026-11-05_LMC/LIGHT", ".local", "2026-11-05_LMC/mosaic_x"] {
            std::fs::create_dir_all(root.join(d)).unwrap();
        }
        let found: Vec<String> = discover(&root).iter().map(|p| p.strip_prefix(&root).unwrap().to_string_lossy().replace('\\', "/")).collect();
        assert_eq!(found, vec!["2012/2012-04-15 ngc3372", "2012/2012-04-19 NGC 6188b", "2026-11-05_LMC"]);
        let _ = std::fs::remove_dir_all(&root);
    }
}
