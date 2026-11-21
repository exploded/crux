//! Watching the open folder for new subs, and the Autopilot (milestone
//! 14), as cards on Home. Both run in the background on their own channel,
//! which outlives opening another folder (the folder channel doesn't).

use crate::app::UiState;
use crate::theme::{body, mono, semibold, Pal};
use crate::widgets::{self as w, Kind};
use crate::worker::Msg;
use crux_core::autopilot::{self, Estimate, Event, FolderResult, Report};
use crux_core::library::RefreshEvent;
use eframe::egui::{self, Color32, RichText, Sense, Vec2};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;

pub struct Bg {
    pub tx: Sender<Msg>,
    pub watch: WatchUi,
    pub auto: AutoUi,
    pub refresh: RefreshUi,
    /// Caches for the autopilot's and the refresh's folders (under
    /// --cache-dir in tests).
    pub cache_root: Option<PathBuf>,
}

impl Bg {
    pub fn new(tx: Sender<Msg>, cache_root: Option<PathBuf>) -> Bg {
        Bg { tx, watch: WatchUi::default(), auto: AutoUi::default(), refresh: RefreshUi::default(), cache_root }
    }
}

/// Refresh calibration library: every folder under the search folders
/// scanned for flats, darks and bias (in place).
#[derive(Default)]
pub struct RefreshUi {
    pub running: bool,
    cancel: Arc<AtomicBool>,
    pub folder: Option<(usize, usize, String)>,
    pub progress: Option<(usize, usize)>,
    pub report: Option<crux_core::library::RefreshReport>,
    pub error: Option<String>,
}

pub fn start_refresh(a: &mut UiState, cfg: &crux_core::config::Config, ctx: &egui::Context) {
    let r = &mut a.bg.refresh;
    if r.running {
        return;
    }
    r.running = true;
    r.error = None;
    r.report = None;
    r.folder = None;
    r.progress = None;
    r.cancel = Arc::new(AtomicBool::new(false));
    let (tx, ctx, cancel, cache_root, cfg) = (a.bg.tx.clone(), ctx.clone(), r.cancel.clone(), a.bg.cache_root.clone(), cfg.clone());
    std::thread::Builder::new()
        .name("crux-refresh".into())
        .spawn(move || {
            let roots = cfg.library.search_roots();
            let txe = tx.clone();
            let ctxe = ctx.clone();
            let res = if roots.is_empty() {
                Err("nothing to search: add search folders in Settings".to_string())
            } else {
                crux_core::library::refresh(
                    &cfg,
                    &roots,
                    cache_root.as_deref(),
                    &move |e| {
                        let _ = txe.send(Msg::Refresh(e));
                        ctxe.request_repaint();
                    },
                    &cancel,
                )
                .map_err(|e| format!("{e:#}"))
            };
            let _ = tx.send(Msg::RefreshDone(res));
            ctx.request_repaint();
        })
        .expect("spawn refresh");
}

/// The refresh button, its progress and what the last refresh found.
pub fn refresh_ui(ui: &mut egui::Ui, a: &mut UiState, cfg: &crux_core::config::Config, lib: Option<&crux_core::library::Library>) {
    let p = a.pal;
    let mut go = false;
    let mut stop = false;
    if a.bg.refresh.running {
        if w::button(ui, &p, "Stop", Kind::Danger, None).clicked() {
            stop = true;
        }
    } else if w::button(ui, &p, "Refresh calibration library", Kind::Normal, None).on_hover_text("Finds the flats, darks and bias in every search folder. The files aren't moved or changed.").clicked() {
        go = true;
    }
    let roots = cfg.library.search_roots();
    let mut facts = vec![format!("Searches {}", roots.iter().map(|r| r.display().to_string()).collect::<Vec<_>>().join(", "))];
    if let Some(l) = lib {
        if let Some(r) = &l.refreshed {
            facts.push(format!("last refreshed {r}"));
        }
        let offsets = l.folders.iter().filter(|f| f.has_offsets()).count();
        if offsets > 0 {
            facts.push(format!("darks and bias in {offsets} folder{}", if offsets == 1 { "" } else { "s" }));
        }
    }
    ui.add(egui::Label::new(RichText::new(facts.join(" · ")).font(body(12.0)).color(p.muted)).wrap());
    if go {
        start_refresh(a, cfg, ui.ctx());
    }
    if stop {
        a.bg.refresh.cancel.store(true, Ordering::Relaxed);
    }
    let r = &a.bg.refresh;
    if r.running {
        let line = match (&r.folder, r.progress) {
            (Some((i, n, name)), Some((d, t))) => format!("Folder {i} of {n}: {name} · calibration frames {d} of {t}"),
            (Some((i, n, name)), None) => format!("Folder {i} of {n}: {name}"),
            _ => "Starting…".into(),
        };
        ui.label(RichText::new(line).font(body(12.0)).color(p.warn));
        if let Some((i, n, _)) = &r.folder {
            ui.add(egui::ProgressBar::new(*i as f32 / (*n).max(1) as f32).desired_height(6.0));
        }
    }
    if let Some(e) = &r.error {
        ui.label(RichText::new(e).font(body(12.0)).color(p.rej));
    }
    if let Some(rep) = &r.report {
        ui.add(egui::Label::new(RichText::new(rep.summary()).font(body(12.0)).color(p.good)).wrap());
        for (dir, n) in &rep.loose {
            ui.add(egui::Label::new(RichText::new(format!("{n} image file{} directly in {} weren't scanned: put them in a folder inside it.", if *n == 1 { "" } else { "s" }, dir.display())).font(body(12.0)).color(p.warn)).wrap());
        }
        if !rep.errors.is_empty() {
            ui.label(RichText::new(format!("{} file{} couldn't be read", rep.errors.len(), if rep.errors.len() == 1 { "" } else { "s" })).font(body(12.0)).color(p.warn)).on_hover_text(rep.errors.iter().take(20).cloned().collect::<Vec<_>>().join("\n"));
        }
    }
}

#[derive(Default)]
pub struct WatchUi {
    pub root: Option<PathBuf>,
    stop: Arc<AtomicBool>,
    pub since: Option<String>,
    pub new_files: Vec<PathBuf>,
}

#[derive(Default)]
pub struct AutoUi {
    pub root: String,
    pub estimating: bool,
    pub estimate: Option<Estimate>,
    pub error: Option<String>,
    pub running: bool,
    cancel: Arc<AtomicBool>,
    pub folder: Option<(usize, usize, String)>,
    pub stage: String,
    pub progress: Option<(usize, usize)>,
    pub results: Vec<FolderResult>,
    pub report: Option<Report>,
    loading: std::collections::HashSet<PathBuf>,
    /// Harness: estimate (and run) once.
    pub pending: Option<bool>,
}

pub fn start_watch(a: &mut UiState, root: &Path, cfg: &crux_core::config::Config, ctx: &egui::Context) {
    stop_watch(a);
    let stop = Arc::new(AtomicBool::new(false));
    let w = &mut a.bg.watch;
    w.root = Some(root.to_path_buf());
    w.stop = stop.clone();
    w.since = Some(chrono::Local::now().format("%H:%M").to_string());
    w.new_files.clear();
    let (root, cfg, tx, ctx) = (root.to_path_buf(), cfg.clone(), a.bg.tx.clone(), ctx.clone());
    std::thread::Builder::new()
        .name("crux-watch".into())
        .spawn(move || {
            let mut wt = crux_core::watch::Watcher::new(&root, &cfg);
            while !stop.load(Ordering::Relaxed) {
                for _ in 0..20 {
                    if stop.load(Ordering::Relaxed) {
                        return;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(500));
                }
                let new = wt.poll();
                if !new.is_empty() {
                    let _ = tx.send(Msg::WatchNew(root.clone(), new));
                    ctx.request_repaint();
                }
            }
        })
        .expect("spawn watcher");
}

pub fn stop_watch(a: &mut UiState) {
    a.bg.watch.stop.store(true, Ordering::Relaxed);
    a.bg.watch.root = None;
}

fn estimate(a: &mut UiState, ctx: &egui::Context) {
    let root = PathBuf::from(a.bg.auto.root.trim());
    let au = &mut a.bg.auto;
    au.estimating = true;
    au.estimate = None;
    au.error = None;
    let (tx, ctx, cache_root) = (a.bg.tx.clone(), ctx.clone(), a.bg.cache_root.clone());
    std::thread::Builder::new()
        .name("crux-estimate".into())
        .spawn(move || {
            let r = (|| {
                anyhow::ensure!(root.is_dir(), "{} isn't a folder", root.display());
                let (cfg, _) = crux_core::config::Config::load(None, Some(&root))?;
                let folders = autopilot::discover(&root);
                Ok(autopilot::estimate(&root, &folders, &cfg, cache_root.as_deref(), &cfg.process.workspace))
            })()
            .map_err(|e: anyhow::Error| format!("{e:#}"));
            let _ = tx.send(Msg::AutoEstimate(r));
            ctx.request_repaint();
        })
        .expect("spawn estimate");
}

fn run(a: &mut UiState, ctx: &egui::Context) {
    let Some(est) = a.bg.auto.estimate.clone() else { return };
    let au = &mut a.bg.auto;
    au.running = true;
    au.results.clear();
    au.report = None;
    au.error = None;
    au.cancel = Arc::new(AtomicBool::new(false));
    let opts = autopilot::Options { workspace: est.workspace.clone(), cache_root: a.bg.cache_root.clone(), config: None, no_images: false };
    let (tx, ctx, cancel) = (a.bg.tx.clone(), ctx.clone(), au.cancel.clone());
    std::thread::Builder::new()
        .name("crux-autopilot".into())
        .spawn(move || {
            let txe = tx.clone();
            let ctxe = ctx.clone();
            let r = autopilot::run_all(
                &est,
                &opts,
                &move |e| {
                    let _ = txe.send(Msg::AutoEvent(e));
                    ctxe.request_repaint();
                },
                &cancel,
            )
            .map_err(|e| format!("{e:#}"));
            let _ = tx.send(Msg::AutoDone(r));
            ctx.request_repaint();
        })
        .expect("spawn autopilot");
}

fn load_previews(a: &mut UiState, results: &[FolderResult]) {
    for t in results.iter().flat_map(|f| f.targets.iter()) {
        let Some(p) = t.preview.clone() else { continue };
        if a.gallery.contains_key(&p) || !a.bg.auto.loading.insert(p.clone()) {
            continue;
        }
        let tx = a.bg.tx.clone();
        std::thread::spawn(move || {
            if let Some(img) = crate::worker::decode_gallery(&p) {
                let _ = tx.send(Msg::Gallery(p, img));
            }
        });
    }
}

/// Messages from the background channel. Returns a folder to reopen (the
/// watcher found new subs in the open folder).
pub fn handle(a: &mut UiState, m: Msg, ctx: &egui::Context, open_root: Option<&Path>) -> Option<PathBuf> {
    match m {
        Msg::WatchNew(root, files) => {
            if a.bg.watch.root.as_deref() != Some(root.as_path()) {
                return None;
            }
            let n = files.len();
            a.bg.watch.new_files.extend(files);
            a.toast(ctx, format!("{n} new sub{} in {}: grading", if n == 1 { "" } else { "s" }, root.file_name().map(|x| x.to_string_lossy().to_string()).unwrap_or_default()), false);
            if open_root.is_some_and(|r| crux_core::paths::key(r) == crux_core::paths::key(&root)) {
                return Some(root);
            }
        }
        Msg::AutoEstimate(r) => {
            a.bg.auto.estimating = false;
            match r {
                Ok(e) => {
                    a.bg.auto.estimate = Some(e);
                    if a.bg.auto.pending.take() == Some(true) {
                        run(a, ctx);
                    }
                }
                Err(e) => a.bg.auto.error = Some(e),
            }
        }
        Msg::AutoEvent(e) => match e {
            Event::Folder { index, total, dir } => {
                a.bg.auto.folder = Some((index, total, dir.file_name().map(|x| x.to_string_lossy().to_string()).unwrap_or_default()));
                a.bg.auto.progress = None;
            }
            Event::Stage(s) => {
                a.bg.auto.stage = s;
                a.bg.auto.progress = None;
            }
            Event::Progress { done, total } => a.bg.auto.progress = Some((done, total)),
            Event::Finished(f) => {
                let f = *f;
                load_previews(a, std::slice::from_ref(&f));
                a.bg.auto.results.push(f);
            }
        },
        Msg::AutoDone(r) => {
            a.bg.auto.running = false;
            a.bg.auto.folder = None;
            match r {
                Ok(rep) => {
                    let n: usize = rep.folders.iter().map(|f| f.targets.iter().filter(|t| t.image.is_some()).count()).sum();
                    a.toast(ctx, format!("Autopilot finished: {n} image{} in {}", if n == 1 { "" } else { "s" }, autopilot::human_time(rep.seconds)), false);
                    a.bg.auto.report = Some(rep);
                }
                Err(e) => a.bg.auto.error = Some(e),
            }
        }
        Msg::Gallery(p, img) => {
            let tex = ctx.load_texture(format!("gallery{}", p.display()), img, egui::TextureOptions::LINEAR);
            a.gallery.insert(p, tex);
        }
        Msg::Refresh(e) => match e {
            RefreshEvent::Folder { index, total, dir } => {
                a.bg.refresh.folder = Some((index, total, dir.file_name().map(|x| x.to_string_lossy().to_string()).unwrap_or_default()));
                a.bg.refresh.progress = None;
            }
            RefreshEvent::Progress { done, total } => a.bg.refresh.progress = Some((done, total)),
        },
        Msg::RefreshDone(r) => {
            let r2 = &mut a.bg.refresh;
            r2.running = false;
            r2.folder = None;
            r2.progress = None;
            match r {
                Ok(rep) => {
                    let msg = format!("Calibration library refreshed: {}", rep.summary());
                    a.bg.refresh.report = Some(rep);
                    a.toast(ctx, msg, false);
                    // Reopen so flat matching and the dark/bias search see it.
                    return open_root.map(Path::to_path_buf);
                }
                Err(e) if e == "cancelled" => {}
                Err(e) => r2.error = Some(e),
            }
        }
        _ => {}
    }
    None
}

/// Home card: watch the open folder.
pub fn watch_card(ui: &mut egui::Ui, a: &mut UiState, root: &Path, cfg: &crux_core::config::Config) {
    let p = a.pal;
    // Opening another folder ends the watch.
    if a.bg.watch.root.as_ref().is_some_and(|r| crux_core::paths::key(r) != crux_core::paths::key(root)) {
        stop_watch(a);
    }
    let on = a.bg.watch.root.is_some();
    w::card(ui, &p, |ui| {
        ui.horizontal(|ui| {
            w::h2(ui, "Watch this folder");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if on {
                    if w::button(ui, &p, "Stop", Kind::Normal, None).clicked() {
                        stop_watch(a);
                    }
                } else if w::button(ui, &p, "Start watching", Kind::Primary, None).clicked() {
                    start_watch(a, root, cfg, ui.ctx());
                }
            });
        });
        let wt = &a.bg.watch;
        if on {
            let n = wt.new_files.len();
            ui.label(RichText::new(format!("Watching since {} · {n} new sub{}", wt.since.clone().unwrap_or_default(), if n == 1 { "" } else { "s" })).color(p.good));
            for f in wt.new_files.iter().rev().take(4) {
                ui.label(RichText::new(f.file_name().map(|x| x.to_string_lossy().to_string()).unwrap_or_default()).font(mono(11.5)).color(p.muted));
            }
        } else {
            ui.add(egui::Label::new(RichText::new("Watches N.I.N.A.'s output folder, including one on the network. Every 10 seconds, crux grades new subs that have finished writing and updates the night's summary.").color(p.muted)).wrap());
        }
    });
}

/// Home card: process everything under a root.
pub fn autopilot_card(ui: &mut egui::Ui, a: &mut UiState, default_root: Option<&Path>) {
    let p = a.pal;
    if a.bg.auto.root.is_empty() {
        if let Some(r) = default_root {
            a.bg.auto.root = r.display().to_string();
        }
    }
    if a.bg.auto.pending == Some(false) || a.bg.auto.pending == Some(true) && a.bg.auto.estimate.is_none() && !a.bg.auto.estimating {
        if a.bg.auto.pending == Some(false) {
            a.bg.auto.pending = None;
        }
        if !a.bg.auto.estimating {
            estimate(a, ui.ctx());
        }
    }
    w::card(ui, &p, |ui| {
        w::h2(ui, "Autopilot");
        ui.add(egui::Label::new(RichText::new("Processes every data folder under a folder: grades, calibrates, registers and integrates the lights, assembles mosaics, and renders each target with its last recipe or the default. Output goes to the workspace; the data folders aren't changed.").color(p.muted)).wrap());
        ui.add_space(4.0);
        let busy = a.bg.auto.running || a.bg.auto.estimating;
        ui.horizontal(|ui| {
            let te = egui::TextEdit::singleline(&mut a.bg.auto.root).desired_width((ui.available_width() - 190.0).max(120.0)).font(mono(12.0));
            ui.add_enabled(!busy, te);
            if w::button_enabled(ui, &p, "Browse…", Kind::Ghost, None, !busy).clicked() {
                if let Some(d) = rfd::FileDialog::new().set_title("Process everything under").pick_folder() {
                    a.bg.auto.root = d.display().to_string();
                    a.bg.auto.estimate = None;
                }
            }
            if w::button_enabled(ui, &p, if a.bg.auto.estimating { "Estimating…" } else { "Estimate" }, Kind::Normal, None, !busy && !a.bg.auto.root.trim().is_empty()).clicked() {
                estimate(a, ui.ctx());
            }
        });
        let au = &a.bg.auto;
        if let Some(e) = &au.error {
            ui.label(RichText::new(e).color(p.rej));
        }
        if let Some(est) = &au.estimate {
            ui.add_space(4.0);
            ui.add(egui::Label::new(RichText::new(est.summary())).wrap());
            for f in est.folders.iter().take(12) {
                ui.label(
                    RichText::new(format!(
                        "{} · {} lights ({} measured){}{}",
                        f.dir.file_name().map(|x| x.to_string_lossy().to_string()).unwrap_or_default(),
                        f.lights,
                        f.measured,
                        if f.targets.is_empty() { String::new() } else { format!(" · {}", f.targets.join(", ")) },
                        if f.processed { " · processed before" } else { "" }
                    ))
                    .font(body(12.0))
                    .color(p.muted),
                );
            }
            if est.folders.len() > 12 {
                ui.label(RichText::new(format!("and {} more", est.folders.len() - 12)).font(body(12.0)).color(p.muted));
            }
        }
        ui.add_space(6.0);
        let mut go = false;
        let mut stop = false;
        ui.horizontal(|ui| {
            if a.bg.auto.running {
                if w::button(ui, &p, "Stop", Kind::Danger, None).clicked() {
                    stop = true;
                }
            } else if w::button_enabled(ui, &p, "Process all", Kind::Primary, None, a.bg.auto.estimate.as_ref().is_some_and(|e| !e.folders.is_empty()) && !a.bg.auto.estimating).clicked() {
                go = true;
            }
            if let Some(rep) = &a.bg.auto.report {
                if w::button(ui, &p, "Open gallery page", Kind::Ghost, None).clicked() {
                    let _ = crate::stack::open_folder(&autopilot::gallery_path(&rep.workspace));
                }
            }
        });
        if go {
            run(a, ui.ctx());
        }
        if stop {
            a.bg.auto.cancel.store(true, Ordering::Relaxed);
        }
        let au = &a.bg.auto;
        if au.running {
            if let Some((i, n, name)) = &au.folder {
                ui.label(RichText::new(format!("Folder {i} of {n}: {name}")).font(semibold(13.0)));
            }
            let line = match au.progress {
                Some((d, t)) => format!("{} · {d} of {t}", au.stage),
                None => au.stage.clone(),
            };
            ui.label(RichText::new(line).color(p.warn));
            if let Some((d, t)) = au.progress {
                ui.add(egui::ProgressBar::new(d as f32 / t.max(1) as f32).desired_height(6.0));
            }
        }
        results(ui, a, &p);
    });
}

fn results(ui: &mut egui::Ui, a: &mut UiState, p: &Pal) {
    let results: Vec<FolderResult> = match &a.bg.auto.report {
        Some(r) => r.folders.clone(),
        None => a.bg.auto.results.clone(),
    };
    if results.is_empty() {
        return;
    }
    load_previews(a, &results);
    ui.add_space(8.0);
    let targets: Vec<&autopilot::TargetResult> = results.iter().flat_map(|f| f.targets.iter()).collect();
    if !targets.is_empty() {
        let cw = 170.0;
        let cols = ((ui.available_width() + 10.0) / (cw + 10.0)).floor().max(1.0) as usize;
        egui::Grid::new("auto-gallery").spacing([10.0, 10.0]).show(ui, |ui| {
            for (k, t) in targets.iter().enumerate() {
                ui.vertical(|ui| {
                    ui.set_width(cw);
                    let (r, resp) = ui.allocate_exact_size(Vec2::new(cw, cw * 0.66), Sense::click());
                    ui.painter().rect_filled(r, 6.0, p.well);
                    if let Some(tex) = t.preview.as_ref().and_then(|pp| a.gallery.get(pp)) {
                        let [tw, th] = tex.size();
                        let s = (r.width() / tw as f32).min(r.height() / th as f32);
                        let ir = egui::Rect::from_center_size(r.center(), Vec2::new(tw as f32 * s, th as f32 * s));
                        ui.painter().image(tex.id(), ir, egui::Rect::from_min_max(egui::Pos2::ZERO, egui::Pos2::new(1.0, 1.0)), Color32::WHITE);
                    } else if let Some(e) = &t.error {
                        ui.painter().text(r.center(), egui::Align2::CENTER_CENTER, "failed", body(12.0), p.rej);
                        resp.clone().on_hover_text(e);
                    }
                    if resp.clicked() {
                        if let Some(img) = &t.image {
                            let _ = crate::stack::open_folder(img);
                        }
                    }
                    ui.label(RichText::new(format!("{}{}", t.object, if t.mosaic { " · mosaic" } else { "" })).font(semibold(12.5)));
                    ui.label(RichText::new(format!("{} · {:.1} h · {}", t.channels.join(" "), t.hours, t.recipe)).font(body(11.5)).color(p.muted));
                });
                if (k + 1) % cols == 0 {
                    ui.end_row();
                }
            }
        });
    }
    let inbox: Vec<(String, String)> = results.iter().flat_map(|f| {
        let name = f.dir.file_name().map(|x| x.to_string_lossy().to_string()).unwrap_or_default();
        f.error.iter().map(|e| format!("Failed: {e}")).chain(f.inbox.iter().cloned()).map(move |i| (name.clone(), i)).collect::<Vec<_>>()
    }).collect();
    if !inbox.is_empty() {
        ui.add_space(8.0);
        ui.label(RichText::new("Needs checking").font(semibold(13.0)));
        for (f, i) in inbox.iter().take(12) {
            ui.add(egui::Label::new(RichText::new(format!("{f}: {i}")).font(body(12.0)).color(p.muted)).wrap());
        }
        if inbox.len() > 12 {
            ui.label(RichText::new(format!("and {} more on the gallery page", inbox.len() - 12)).font(body(12.0)).color(p.muted));
        }
    }
}
