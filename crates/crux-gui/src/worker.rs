//! Background jobs: open a folder (headers, calibration, lights, grading),
//! read the folders beside it (headers only) and decode the target's
//! picture and the library's gallery images.
//! Results come back over a channel; the UI never waits.

use crate::session::{FolderSummary, Timings};
use crux_core::cache::{Cache, Thumb};
use crux_core::health::FolderSpace;
use crux_core::index::{Index, IndexFile};
use crux_core::config::Config;
use crux_core::pipeline::{self, Stage, Triage};
use crux_core::transparency::FrameTransparency;
use crux_core::{FrameType, Role};
use eframe::egui;
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::time::Instant;

pub struct Loaded {
    pub root: PathBuf,
    pub cache_dir: Option<PathBuf>,
    pub cfg: Config,
    pub triage: Triage,
    pub transp: HashMap<usize, FrameTransparency>,
    pub thumbs: HashMap<usize, Thumb>,
    pub timings: Timings,
}

/// Quick facts from the header scan, shown while pixels are measured.
#[derive(Clone, Default)]
pub struct HeaderFacts {
    pub files: usize,
    pub lights: usize,
    pub filters: Vec<String>,
    pub nights: usize,
    pub panels: usize,
    pub flats: usize,
    pub masters: usize,
    pub seconds: f64,
}

pub enum Msg {
    Headers(HeaderFacts),
    Progress { stage: Stage, done: usize, total: usize },
    Loaded(Box<Loaded>),
    Failed(String),
    Folders(Vec<FolderSummary>, f64),
    /// The open folder's finished image, if it has one (decoded next).
    TargetImage(Option<PathBuf>),
    /// The open folder's parent looks like a library (`looks_like_library_dir`).
    LibraryGuess(PathBuf),
    Gallery(PathBuf, egui::ColorImage),
    /// The library index (every folder) and each folder's space.
    /// `area`: the folder Library health measured (the library folder
    /// holding the open folder, else the open folder itself).
    Library { files: Vec<IndexFile>, spaces: Vec<FolderSpace>, empty: Vec<PathBuf>, area: PathBuf, error: Option<String> },
    /// A move plan being applied or undone.
    MoveProgress { done: usize, total: usize, file: PathBuf },
    MoveDone { undo: bool, result: Result<crux_core::moves::Outcome, String>, root: PathBuf },
    DupProgress { done: usize, total: usize },
    Duplicates(Vec<crux_core::health::Duplicate>),
    Exported(Result<crux_core::export::Outcome, String>, PathBuf),
    /// Darks, bias and dark flats ranked across the library.
    FpProgress { done: usize, total: usize },
    Fingerprints(Box<crux_core::fingerprint::Fingerprints>),
    /// Processing (Stack page).
    Process(crux_core::process::job::Event),
    ProcessDone(Result<crux_core::process::job::RunReport, String>),
    Preview(Box<crate::stack::PreviewImgs>),
    PreviewFailed((PathBuf, crate::stack::Stage), String),
    Solved { done: usize, total: usize, results: Vec<(PathBuf, Option<crux_core::solve::Solution>)> },
    SolveFailed(String),
    /// Image stages: a target's masters read and gradient-corrected (key:
    /// target and gradient settings), and a full-size image written.
    ImagePrepared(String, Result<std::sync::Arc<crux_core::render::Prepared>, String>),
    ImageWritten(Result<crux_core::render::Written, String>),
    /// Image stage: what the background preparation is doing.
    ImageProgress(String),
    /// Sharpen and denoise stage: before and after crops and the settings
    /// the after crop was made with (key, result).
    ImageEnhanceView(String, Result<(crux_core::render::Rgb, crux_core::render::Rgb, String), String>),
    /// Image stages: the preview rendered in the background (key, before,
    /// after, labels, milliseconds), and the preset tiles (key, tiles).
    ImageView(Box<crate::image::ViewDone>),
    ImageTiles(String, Vec<(String, crux_core::render::Rgb)>),
    /// Mosaic stage: progress, the finished reports, and the preview
    /// (key, grey image, canvas pixels per preview pixel).
    MosaicStep(String),
    MosaicDone(Result<Vec<crux_core::process::mosaic::Report>, String>),
    MosaicPreview(String, Result<(egui::ColorImage, f32), String>),
    /// The folder watcher found new subs.
    WatchNew(PathBuf, Vec<PathBuf>),
    /// Autopilot: estimate, events, and the finished run.
    AutoEstimate(Result<crux_core::autopilot::Estimate, String>),
    AutoEvent(crux_core::autopilot::Event),
    AutoDone(Result<crux_core::autopilot::Report, String>),
    /// A folder of targets opened as the library (milestone 15).
    LibraryView(Box<crate::libview::LibView>),
    /// Refresh calibration library: events and the finished report.
    Refresh(crux_core::library::RefreshEvent),
    RefreshDone(Result<crux_core::library::RefreshReport, String>),
}

pub struct OpenJob {
    pub root: PathBuf,
    pub cache_dir: Option<PathBuf>,
    pub library: Option<PathBuf>,
    pub index: Option<PathBuf>,
    pub cancel: Arc<AtomicBool>,
}

pub fn spawn_open(job: OpenJob, tx: Sender<Msg>, ctx: egui::Context) {
    std::thread::Builder::new()
        .name("crux-open".into())
        .spawn(move || {
            let r = open(&job, &tx, &ctx);
            if let Err(e) = r {
                let _ = tx.send(Msg::Failed(format!("{e:#}")));
            }
            ctx.request_repaint();
        })
        .expect("spawn worker");
}

fn open(job: &OpenJob, tx: &Sender<Msg>, ctx: &egui::Context) -> anyhow::Result<()> {
    let root = &job.root;
    if !root.is_dir() {
        anyhow::bail!("{} is not a folder", root.display());
    }
    let (mut cfg, _) = Config::load(None, Some(root))?;
    if let Some(l) = &job.library {
        cfg.library.use_folder(l);
    }
    // A target folder opened directly, inside a folder of several targets
    // that isn't one of your library folders: offer the way back anyway.
    if library_parent(root, &cfg).is_none() {
        if let Some(parent) = root.parent().filter(|p| looks_like_library_dir(p)) {
            let _ = tx.send(Msg::LibraryGuess(parent.to_path_buf()));
        }
    }
    let mut cache = match &job.cache_dir {
        Some(d) => Cache::load(d),
        None => Cache::disabled(),
    };
    // Headers first: answers in a fraction of a second.
    let t0 = Instant::now();
    let h = pipeline::run(root, &cfg, &mut cache, false, &|_, _, _| {});
    let header_s = t0.elapsed().as_secs_f64();
    let _ = tx.send(Msg::Headers(facts(&h, header_s)));
    ctx.request_repaint();

    // A library folder (several targets in folders of their own, or one of
    // your library folders): show its targets; grade nothing yet.
    let configured = cfg.library.search_roots().iter().any(|r| crux_core::paths::key(r) == crux_core::paths::key(root));
    if configured || crux_core::targets::looks_like_library(root, &h) {
        let ix = job.index.as_ref().and_then(|p| Index::load(p).ok());
        let solved = |p: &Path, size: u64, mtime: i64| ix.as_ref().and_then(|ix| ix.solution(p, size, mtime)).flatten().map(|s| (s.ra, s.dec));
        let targets = crux_core::targets::find(&[(root.clone(), &h)], &solved);
        let mut images = HashMap::new();
        let mut made = HashMap::new();
        // One picture and status per target folder (`Unit::folder_for`): a
        // unit holding several targets in folders of their own gets one each.
        let dirs: BTreeSet<PathBuf> = targets.units.iter().flat_map(|u| u.targets.keys().map(move |t| u.folder_for(t).to_path_buf())).collect();
        for d in &dirs {
            if let Some(img) = crux_image(d, &cfg.process.workspace).or_else(|| find_image(d)) {
                images.insert(crux_core::paths::key(d), img);
            }
            if let Some(m) = crate::libview::made(d, &cfg.process.workspace) {
                made.insert(crux_core::paths::key(d), m);
            }
        }
        let library = crux_core::library::Library::load(&cfg.library).ok().flatten();
        let pics: Vec<PathBuf> = images.values().cloned().collect();
        let _ = tx.send(Msg::LibraryView(Box::new(crate::libview::LibView {
            root: root.clone(),
            cfg: cfg.clone(),
            targets,
            configured,
            library,
            images,
            made,
        })));
        ctx.request_repaint();
        let _ = cache.save();
        for p in pics {
            if job.cancel.load(Ordering::Relaxed) {
                break;
            }
            if let Some(img) = decode_gallery(&p) {
                let _ = tx.send(Msg::Gallery(p, img));
                ctx.request_repaint();
            }
        }
        return Ok(());
    }

    let send = |stage: Stage, done: usize, total: usize| {
        let _ = tx.send(Msg::Progress { stage, done, total });
        ctx.request_repaint();
    };
    let triage = pipeline::run_with(root, &cfg, &mut cache, true, &send, &job.cancel);
    if job.cancel.load(Ordering::Relaxed) {
        let _ = cache.save();
        anyhow::bail!("cancelled");
    }
    let _ = cache.save();
    if std::env::var_os("CRUX_TIMING").is_some() {
        eprintln!("open: triage and cache saved at {:.2} s", t0.elapsed().as_secs_f64());
    }
    let t1 = Instant::now();
    // Worked out once by the triage's grading; regrading reuses it.
    let transp = triage.transparency.clone();
    let transparency_s = t1.elapsed().as_secs_f64();
    let mut thumbs = HashMap::new();
    for (i, f) in triage.scan.frames.iter().enumerate() {
        if f.class.frame_type == FrameType::Light && f.class.role == Role::Raw {
            if let Some(t) = cache.get_thumb(&f.meta.path, f.meta.file_size, f.meta.mtime_ns) {
                thumbs.insert(i, t);
            }
        }
    }
    let n_cal = triage.cal_metrics.len() + triage.cal_errors.len();
    let n_lights = triage.metrics.len() + triage.errors.len();
    let timings = Timings {
        header_files: triage.scan.stats.files,
        header_s,
        cal_frames: n_cal,
        cal_measured: triage.cal_analysed,
        cal_s: triage.cal_seconds,
        lights: n_lights,
        lights_measured: triage.analysed,
        lights_s: triage.analysis_seconds,
        transparency_s,
    };
    if std::env::var_os("CRUX_TIMING").is_some() {
        eprintln!("open: headers {:.2} s, calibration frames {:.2} s ({} measured), lights {:.2} s ({} measured), transparency {:.2} s, total {:.2} s", header_s, triage.cal_seconds, triage.cal_analysed, triage.analysis_seconds, triage.analysed, transparency_s, t0.elapsed().as_secs_f64());
    }
    let mut reports = triage.reports.clone();
    if let Some(d) = job.cache_dir.as_deref().and_then(|d| crux_core::decisions::Decisions::load(d).ok()) {
        d.apply_overrides(&triage.scan.frames, &mut reports);
    }
    let own_records = crux_core::index::records(&triage, Some(&reports));
    // Kept for the fingerprint pass after the triage goes to the UI.
    let (own_frames, own_grouping) = (triage.scan.frames.clone(), triage.grouping.clone());
    let _ = tx.send(Msg::Loaded(Box::new(Loaded {
        root: root.clone(),
        cache_dir: job.cache_dir.clone(),
        cfg: cfg.clone(),
        triage,
        transp,
        thumbs,
        timings,
    })));
    ctx.request_repaint();

    // This target's picture for its Home card (the folders beside it get
    // none: Home shows only the open target).
    let image = crux_image(root, &cfg.process.workspace).or_else(|| find_image(root));
    let _ = tx.send(Msg::TargetImage(image.clone()));
    ctx.request_repaint();
    if let Some(p) = image {
        if let Some(img) = decode_gallery(&p) {
            let _ = tx.send(Msg::Gallery(p, img));
            ctx.request_repaint();
        }
    }

    // Then the folders beside it, headers only, for the library index and
    // the fingerprint search.
    let t2 = Instant::now();
    let mut folders = Vec::new();
    let mut recs: Vec<(PathBuf, Vec<IndexFile>)> = Vec::new();
    let mut spaces = Vec::new();
    let mut others: Vec<(PathBuf, Vec<crux_core::Frame>)> = Vec::new();
    // The folders beside it, only when it's inside one of your library
    // folders (crux never looks outside them), and never beside a drive
    // root or the home folder.
    let parent = library_parent(root, &cfg);
    if let Some(parent) = parent.as_deref() {
        let mut dirs: Vec<PathBuf> = std::fs::read_dir(parent)
            .map(|rd| rd.flatten().filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false)).map(|e| e.path()).collect())
            .unwrap_or_default();
        dirs.sort();
        for d in dirs {
            if job.cancel.load(Ordering::Relaxed) {
                return Ok(());
            }
            let name = d.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            if crux_core::targets::skip_unit(&name) || hidden_or_system(&d) {
                continue;
            }
            let mut c = Cache::disabled();
            let t = pipeline::run(&d, &cfg, &mut c, false, &|_, _, _| {});
            folders.push(summarise(&d, &t));
            spaces.push(crux_core::health::space(&d, &t.scan.frames));
            if crux_core::cache::key(&d) != crux_core::cache::key(root) {
                recs.push((d.clone(), crux_core::index::records(&t, None)));
                others.push((d.clone(), t.scan.frames));
            }
        }
    }
    let secs = t2.elapsed().as_secs_f64();
    let _ = tx.send(Msg::Folders(folders, secs));
    ctx.request_repaint();
    // The library index: this folder in full, the others from headers.
    let (files, error) = match &job.index {
        Some(ip) => match Index::load(ip) {
            Ok(mut ix) => {
                ix.set_root(root, own_records, true);
                for (d, r) in recs {
                    ix.set_root(&d, r, false);
                }
                let e = ix.save().err().map(|e| format!("{e:#}"));
                (ix.files, e)
            }
            Err(e) => (Vec::new(), Some(format!("{e:#}"))),
        },
        None => {
            let mut v = own_records;
            v.extend(recs.into_iter().flat_map(|x| x.1));
            (v, None)
        }
    };
    // Outside your library folders (or beside a drive root) only the open
    // folder is measured; without this Library health waited forever.
    let area = parent.clone().unwrap_or_else(|| root.clone());
    if spaces.is_empty() {
        spaces.push(crux_core::health::space(root, &own_frames));
    }
    let empty = crux_core::health::empty_dirs(&area);
    let _ = tx.send(Msg::Library { files, spaces, empty, area, error });
    ctx.request_repaint();
    // Darks, bias and dark flats across the library, by fingerprint: the
    // folders beside this one, plus those the calibration library knows.
    for d in crux_core::library::offset_folders(&cfg) {
        if !crux_core::library::inside(&d, root) && !others.iter().any(|(o, _)| crux_core::paths::key(o) == crux_core::paths::key(&d)) {
            let mut c = Cache::disabled();
            let t = pipeline::run(&d, &cfg, &mut c, false, &|_, _, _| {});
            others.push((d, t.scan.frames));
        }
    }
    if std::env::var_os("CRUX_TIMING").is_some() {
        eprintln!("open: folders beside it and the index done at {:.2} s", t0.elapsed().as_secs_f64());
    }
    if !job.cancel.load(Ordering::Relaxed) {
        let fp = crux_core::fingerprint::build_from(&own_frames, &own_grouping, &reports, root, &others, &cache, &|done, total| {
            let _ = tx.send(Msg::FpProgress { done, total });
            ctx.request_repaint();
        });
        let _ = tx.send(Msg::Fingerprints(Box::new(fp)));
        ctx.request_repaint();
        if std::env::var_os("CRUX_TIMING").is_some() {
            eprintln!("open: fingerprints done at {:.2} s", t0.elapsed().as_secs_f64());
        }
    }
    Ok(())
}

fn facts(t: &Triage, seconds: f64) -> HeaderFacts {
    let fr = &t.scan.frames;
    let lights: Vec<_> = fr.iter().filter(|f| f.class.frame_type == FrameType::Light && f.class.role == Role::Raw).collect();
    let mut filters: Vec<String> = lights.iter().filter_map(|f| f.meta.filter.clone()).map(|f| f.to_uppercase()).collect();
    filters.sort();
    filters.dedup();
    let mut nights: Vec<_> = lights.iter().filter_map(|f| f.class.night).collect();
    nights.sort();
    nights.dedup();
    let mut panels: Vec<String> = lights.iter().filter_map(|f| f.meta.object.clone()).collect();
    panels.sort();
    panels.dedup();
    HeaderFacts {
        files: t.scan.stats.files,
        lights: lights.len(),
        filters,
        nights: nights.len(),
        panels: panels.len(),
        flats: fr.iter().filter(|f| f.class.frame_type == FrameType::Flat && f.class.role == Role::Raw).count(),
        masters: fr.iter().filter(|f| f.class.role == Role::Master).count(),
        seconds,
    }
}

fn summarise(dir: &Path, t: &Triage) -> FolderSummary {
    let mut s = FolderSummary {
        path: dir.to_path_buf(),
        ..Default::default()
    };
    let mut flat_nights = std::collections::BTreeSet::new();
    for f in &t.scan.frames {
        if f.class.role != Role::Raw {
            continue;
        }
        match f.class.frame_type {
            FrameType::Light => {
                let e = f.meta.exptime.unwrap_or(0.0);
                *s.light_secs.entry(f.meta.filter.clone().unwrap_or_else(|| "?".into()).to_uppercase()).or_default() += e;
                if let Some(n) = f.class.night {
                    *s.light_nights.entry(n).or_default() += e;
                }
                if let Some(o) = &f.meta.object {
                    if !s.objects.contains(o) {
                        s.objects.push(o.clone());
                    }
                }
            }
            FrameType::Flat => {
                s.flats += 1;
                if let Some(n) = f.class.night {
                    flat_nights.insert(n);
                }
            }
            _ => {}
        }
    }
    s.flat_nights = flat_nights.into_iter().collect();
    s
}

/// crux's newest image for a data folder, from the workspace
/// (`<workspace>\<folder>\image\<object>\`): the target folder written
/// most recently, then the newest of its `preview.png` and JPGs. The
/// 16-bit PNGs are never used (hundreds of MB to decode for a card), so
/// a target rendered before crux wrote JPGs and previews is skipped.
pub fn crux_image(dir: &Path, workspace: &Path) -> Option<PathBuf> {
    let root = crux_core::process::job::out_dir(dir, workspace).join("image");
    let mtime = |p: &Path| p.metadata().and_then(|m| m.modified()).ok();
    let ext = |p: &Path| p.extension().map(|x| x.to_string_lossy().to_lowercase()).unwrap_or_default();
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    for target in std::fs::read_dir(&root).ok()?.flatten().map(|e| e.path()).filter(|p| p.is_dir()) {
        let pics: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(&target)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| matches!(ext(p).as_str(), "jpg" | "jpeg" | "png"))
            .filter_map(|p| Some((mtime(&p)?, p)))
            .collect();
        let Some(newest) = pics.iter().map(|x| x.0).max() else { continue };
        let usable = |p: &Path| p.file_name().is_some_and(|n| n == "preview.png") || matches!(ext(p).as_str(), "jpg" | "jpeg");
        let Some(pick) = pics.iter().filter(|(_, p)| usable(p)).max_by_key(|x| x.0).map(|x| x.1.clone()) else { continue };
        if best.as_ref().is_none_or(|b| newest > b.0) {
            best = Some((newest, pick));
        }
    }
    best.map(|b| b.1)
}

/// The folder whose subfolders sit beside `root` for the gallery: its
/// parent, when that's one of your library folders or inside one.
pub fn library_parent(root: &Path, cfg: &Config) -> Option<PathBuf> {
    let parent = root.parent().filter(|p| !crux_core::paths::too_broad(p))?;
    cfg.library.search_roots().iter().any(|r| crux_core::library::inside(parent, r)).then(|| parent.to_path_buf())
}

/// Whether `dir` looks like a folder of targets: at least two of its
/// subfolders hold an image file (FITS, XISF, CR2) within three levels.
/// Cheap: stops at each subfolder's first image, looks at no more than 2,000
/// entries per subfolder and 100 subfolders, never at drive roots or the
/// home folder, never opens online-only files.
pub fn looks_like_library_dir(dir: &Path) -> bool {
    if crux_core::paths::too_broad(dir) {
        return false;
    }
    let Ok(rd) = std::fs::read_dir(dir) else { return false };
    let mut found = 0;
    for sub in rd.flatten().map(|e| e.path()).filter(|p| p.is_dir()).take(100) {
        let name = sub.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        if crux_core::targets::skip_unit(&name) || hidden_or_system(&sub) {
            continue;
        }
        let has = walkdir::WalkDir::new(&sub)
            .max_depth(3)
            .into_iter()
            .take(2_000)
            .flatten()
            .any(|e| e.file_type().is_file() && crux_core::meta::format_of(e.path()).is_some() && e.metadata().is_ok_and(|m| !crux_core::paths::cloud_only(&m)));
        if has {
            found += 1;
            if found >= 2 {
                return true;
            }
        }
    }
    false
}

/// Where "back to the library" leads from `root`: its library parent, else
/// the library folder you opened it from (`from`) when `root` is inside it.
pub fn back_to_library(root: &Path, cfg: &Config, from: Option<&Path>) -> Option<PathBuf> {
    library_parent(root, cfg).or_else(|| {
        let f = from?;
        let inside = crux_core::library::inside(root, f) && crux_core::paths::key(root) != crux_core::paths::key(f);
        inside.then(|| f.to_path_buf())
    })
}

/// Hidden or system folders (Windows attributes; dot names elsewhere are
/// caught by the caller).
fn hidden_or_system(d: &Path) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        const HIDDEN: u32 = 0x2;
        const SYSTEM: u32 = 0x4;
        d.symlink_metadata().is_ok_and(|m| m.file_attributes() & (HIDDEN | SYSTEM) != 0)
    }
    #[cfg(not(windows))]
    {
        let _ = d;
        false
    }
}

/// A finished image to show for a target: JPG or PNG within three levels,
/// preferring names with "final", then "preview", then the biggest file.
pub fn find_image(dir: &Path) -> Option<PathBuf> {
    let mut best: Option<(i64, PathBuf)> = None;
    for e in walkdir::WalkDir::new(dir).max_depth(3).into_iter().flatten() {
        let p = e.path();
        let ext = p.extension().map(|x| x.to_string_lossy().to_lowercase()).unwrap_or_default();
        if !matches!(ext.as_str(), "jpg" | "jpeg" | "png") || !e.file_type().is_file() {
            continue;
        }
        let name = p.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
        let Ok(md) = e.metadata() else { continue };
        // Decoding an online-only file would download it.
        if crux_core::paths::cloud_only(&md) {
            continue;
        }
        let size = md.len() as i64;
        if size < 50_000 {
            continue;
        }
        let score = size.min(50_000_000) + if name.contains("final") { 200_000_000 } else { 0 } + if name.contains("preview") { 100_000_000 } else { 0 };
        if best.as_ref().is_none_or(|b| score > b.0) {
            best = Some((score, p.to_path_buf()));
        }
    }
    best.map(|b| b.1)
}

/// Decode and shrink an image to 640 px wide for a gallery card.
pub fn decode_gallery(p: &Path) -> Option<egui::ColorImage> {
    let img = image::ImageReader::open(p).ok()?.with_guessed_format().ok()?.decode().ok()?;
    let img = img.thumbnail(640, 640).to_rgba8();
    let size = [img.width() as usize, img.height() as usize];
    Some(egui::ColorImage::from_rgba_unmultiplied(size, img.as_raw()))
}

/// Apply (or undo) a plan in the background, reporting each file.
pub fn spawn_move(plan: Option<(crux_core::plan::Plan, PathBuf)>, undo_of: Option<PathBuf>, root: PathBuf, tx: Sender<Msg>, ctx: egui::Context) {
    std::thread::Builder::new()
        .name("crux-move".into())
        .spawn(move || {
            let tx2 = tx.clone();
            let c2 = ctx.clone();
            let progress: Arc<dyn Fn(usize, usize, &Path) + Send + Sync> = Arc::new(move |done, total, file: &Path| {
                let _ = tx2.send(Msg::MoveProgress { done, total, file: file.to_path_buf() });
                c2.request_repaint();
            });
            let opts = crux_core::moves::Options { force_copy: false, progress: Some(progress) };
            let (undo, result) = match (plan, undo_of) {
                (Some((p, path)), _) => {
                    let journal = path.with_extension("journal.jsonl");
                    let r = (|| -> anyhow::Result<crux_core::moves::Outcome> {
                        if let Some(d) = path.parent() {
                            std::fs::create_dir_all(d)?;
                        }
                        std::fs::write(&path, serde_json::to_vec_pretty(&p)?)?;
                        crux_core::moves::apply(&p, Some(&path), &journal, &opts)
                    })();
                    (false, r.map_err(|e| format!("{e:#}")))
                }
                (None, Some(j)) => (true, crux_core::moves::undo(&j, &crux_core::moves::undo_path(&j), &opts).map_err(|e| format!("{e:#}"))),
                _ => return,
            };
            let _ = tx.send(Msg::MoveDone { undo, result, root });
            ctx.request_repaint();
        })
        .expect("spawn mover");
}

/// Hash duplicate candidates in the background.
pub fn spawn_duplicates(cands: Vec<Vec<IndexFile>>, tx: Sender<Msg>, ctx: egui::Context) {
    std::thread::Builder::new()
        .name("crux-dups".into())
        .spawn(move || {
            let refs: Vec<Vec<&IndexFile>> = cands.iter().map(|v| v.iter().collect()).collect();
            let tx2 = tx.clone();
            let c2 = ctx.clone();
            let d = crux_core::health::confirm_duplicates(&refs, &move |done, total| {
                let _ = tx2.send(Msg::DupProgress { done, total });
                c2.request_repaint();
            });
            let _ = tx.send(Msg::Duplicates(d));
            ctx.request_repaint();
        })
        .expect("spawn hasher");
}

/// Hard-link files into a folder in the background.
pub fn spawn_export(files: Vec<PathBuf>, dest: PathBuf, tx: Sender<Msg>, ctx: egui::Context) {
    std::thread::Builder::new()
        .name("crux-export".into())
        .spawn(move || {
            let r = crux_core::export::hard_links(&files, &dest).map_err(|e| format!("{e:#}"));
            let _ = tx.send(Msg::Exported(r, dest));
            ctx.request_repaint();
        })
        .expect("spawn export");
}
