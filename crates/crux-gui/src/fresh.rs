//! Start fresh (Settings, and the library's Home): clear what crux has
//! worked out and made, then scan again from the start
//! (`crux_core::fresh`). The choices are measured first (sizes), clearing
//! runs in the background, then the calibration library is refreshed and
//! the open folder reopened.

use crate::app::UiState;
use crate::theme::{body, mono, semibold, Pal};
use crate::widgets::{self as w, Kind};
use crux_core::cachedir::human;
use crux_core::config::Config;
use crux_core::fresh::{Done, Options, Survey, Work};
use eframe::egui::{self, RichText, Ui};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, TryRecvError};
use std::time::Duration;

pub struct FreshUi {
    /// The choices are showing.
    open: bool,
    /// Show the choices on the next frame (the screenshot harness's `fresh`).
    pub pending_open: bool,
    /// Scroll the choices into view once they're measured.
    scroll: bool,
    /// The harness's `fresh-go`: press "Clear and start again" once
    /// measured (only with --cache-dir, never on a real workspace).
    pub harness_go: bool,
    survey: Option<Survey>,
    surveying: Option<Receiver<Survey>>,
    opts: Options,
    refresh_after: bool,
    confirm: bool,
    running: Option<Receiver<Result<Done, String>>>,
    /// For when clearing ends: the settings to refresh with, the folder to reopen.
    after: Option<(Config, Option<PathBuf>)>,
    /// Reopen this once the refresh has finished.
    reopen: Option<PathBuf>,
    error: Option<String>,
    /// The last run's summary and anything it couldn't remove.
    last: Option<(String, Vec<String>)>,
}

impl Default for FreshUi {
    fn default() -> Self {
        FreshUi { open: false, pending_open: false, scroll: false, harness_go: false, survey: None, surveying: None, opts: Options::default(), refresh_after: true, confirm: false, running: None, after: None, reopen: None, error: None, last: None }
    }
}

impl FreshUi {
    /// Measuring the choices (the screenshot harness waits).
    pub fn measuring(&self) -> bool {
        self.pending_open || self.surveying.is_some() || self.harness_go || self.running.is_some() || self.reopen.is_some()
    }
}

/// What's running that uses the caches, the index or the workspace.
pub fn busy(a: &UiState) -> Option<&'static str> {
    if a.fresh.running.is_some() {
        Some("Wait until clearing has finished.")
    } else if a.loading.is_some() || a.open_request.is_some() {
        Some("Wait until the folder has opened.")
    } else if a.stack.running.is_some() {
        Some("Wait until processing has finished.")
    } else if a.stack.solving.is_some() {
        Some("Wait until plate solving has finished.")
    } else if a.stack.mos.running.is_some() {
        Some("Wait until the mosaic has been made.")
    } else if a.stack.img.busy() {
        Some("Wait until the image stage has finished.")
    } else if a.bg.refresh.running {
        Some("Wait until the calibration library refresh has finished.")
    } else if a.bg.auto.running {
        Some("Wait until Autopilot has finished.")
    } else if a.bg.watch.root.is_some() {
        Some("Stop the folder watcher on Home first.")
    } else if a.org.dup_progress.is_some() {
        Some("Wait until the duplicate check has finished.")
    } else {
        None
    }
}

fn note(ui: &mut Ui, p: &Pal, t: impl Into<String>) {
    ui.add(egui::Label::new(RichText::new(t.into()).font(body(12.5)).color(p.muted)).wrap());
}

/// A choice with its explanation indented under it.
fn choice(ui: &mut Ui, p: &Pal, on: &mut bool, title: String, help: &str) {
    ui.checkbox(on, RichText::new(title).font(body(13.0)));
    ui.horizontal(|ui| {
        ui.add_space(26.0);
        note(ui, p, help);
    });
}

/// The card. Returns true when "Clear and start again" was pressed; the
/// caller then runs [`go`].
pub fn card(ui: &mut Ui, p: &Pal, f: &mut FreshUi, busy: Option<&'static str>, index: Option<&Path>, explicit: Option<&Path>, cfg: &Config) -> bool {
    if let Some(rx) = &f.surveying {
        match rx.try_recv() {
            Ok(s) => {
                f.survey = Some(s);
                f.surveying = None;
                f.scroll = true;
            }
            Err(TryRecvError::Empty) => ui.ctx().request_repaint_after(Duration::from_millis(100)),
            Err(TryRecvError::Disconnected) => f.surveying = None,
        }
    }
    let mut go = false;
    w::card(ui, p, |ui| {
        w::h2(ui, "Start fresh");
        note(ui, p, "Clear what crux has worked out and made, then scan your library folders again from the start. Your data isn't touched.");
        ui.add_space(6.0);
        if f.running.is_some() {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(RichText::new("Clearing…").font(body(13.0)));
            });
            return;
        }
        if let Some((s, errs)) = &f.last {
            ui.add(egui::Label::new(RichText::new(s).font(body(12.5)).color(p.good)).wrap());
            for e in errs.iter().take(5) {
                ui.add(egui::Label::new(RichText::new(e).font(mono(11.5)).color(p.warn)).wrap());
            }
        }
        if let Some(e) = &f.error {
            ui.add(egui::Label::new(RichText::new(e).font(body(12.5)).color(p.rej)).wrap());
        }
        if !f.open {
            if w::button(ui, p, "Choose what to clear…", Kind::Normal, None).clicked() || std::mem::take(&mut f.pending_open) {
                f.open = true;
                f.confirm = false;
                f.error = None;
                f.survey = None;
                let (tx, rx) = channel();
                f.surveying = Some(rx);
                let (index, explicit, ws, roots, ctx) = (index.map(Path::to_path_buf), explicit.map(Path::to_path_buf), vec![cfg.process.workspace.clone()], cfg.library.search_roots(), ui.ctx().clone());
                std::thread::spawn(move || {
                    let _ = tx.send(crux_core::fresh::survey(index.as_deref(), explicit.as_deref(), &ws, &roots));
                    ctx.request_repaint();
                });
            }
            return;
        }
        let Some(s) = f.survey.as_ref() else {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(RichText::new("Measuring what's there…").font(body(13.0)).color(p.muted));
            });
            return;
        };
        let o = &mut f.opts;
        let existing = s.caches.iter().filter(|c| c.folder_exists).count() + usize::from(s.explicit.is_some());
        let plural = |n: usize| if n == 1 { "" } else { "s" };
        choice(ui, p, &mut o.measurements, format!("Measurements, maps and thumbnails · {existing} folder{} · {}", plural(existing), human(s.measurement_bytes())), "Every frame is measured again when its folder opens.");
        let gone = s.missing().count();
        if gone > 0 {
            choice(ui, p, &mut o.missing, format!("Caches of folders that no longer exist · {gone} · {}", human(s.missing_bytes())), "Their decisions and move journals go too.");
        } else {
            o.missing = false;
        }
        if s.index.is_some() {
            choice(ui, p, &mut o.index, "The library index and flat library".into(), "Moved aside as a backup, then rebuilt as folders open and the calibration library is refreshed.");
        } else {
            o.index = false;
        }
        ui.add_space(6.0);
        let wher = s.workspaces.iter().map(|w| w.display().to_string()).collect::<Vec<_>>().join(", ");
        if s.work.is_empty() {
            if !wher.is_empty() {
                note(ui, p, format!("Nothing crux made in {wher}."));
            } else if s.refused.is_empty() {
                note(ui, p, "No workspace to clear.");
            }
            o.work = Work::Keep;
        } else {
            ui.label(RichText::new(format!("Processing outputs in {wher}")).font(semibold(13.0)));
            let n = s.work.len();
            for (val, title, help) in [
                (Work::Keep, "Keep them".to_string(), ""),
                (Work::Intermediates, format!("Intermediate files · {}", human(s.work_bytes(Work::Intermediates))), "Registered and calibrated frames, network and preview caches. Masters, mosaics, images and recipes stay; processing again rebuilds the rest."),
                (Work::All, format!("Everything crux made · {n} folder{} · {}", plural(n), human(s.work_bytes(Work::All))), "Masters, mosaics, finished images and recipes too. Training bundles and files crux didn't make stay."),
            ] {
                ui.radio_value(&mut o.work, val, RichText::new(title).font(body(13.0)));
                if !help.is_empty() {
                    ui.horizontal(|ui| {
                        ui.add_space(26.0);
                        note(ui, p, help);
                    });
                }
            }
        }
        for r in &s.refused {
            ui.add(egui::Label::new(RichText::new(format!("Left alone: {r}")).font(body(12.5)).color(p.warn)).wrap());
        }
        ui.add_space(6.0);
        let kept = s.decision_bytes();
        choice(ui, p, &mut o.decisions, format!("Your decisions · {}", human(kept)), "Kept and rejected frames, flat choices, panel corrections, layout rules and saved views. Leave this off to keep them.");
        let can_refresh = !cfg.library.search_roots().is_empty();
        if can_refresh {
            choice(ui, p, &mut f.refresh_after, "Refresh the calibration library afterwards".into(), "Then the open folder opens again.");
        }
        ui.add_space(8.0);
        let anything = o.anything();
        let frees = s.bytes(o);
        let backup = o.index && s.index.is_some();
        let real = s.workspaces.iter().any(|w| crux_core::paths::key(w).starts_with(r"c:\pixinsight") || crux_core::paths::key(w).starts_with(r"c:\crux-work") || crux_core::paths::key(w).starts_with(r"d:\crux-work"));
        let mut cancel = false;
        ui.horizontal_wrapped(|ui| {
            if !f.confirm {
                let label = if frees > 0 { format!("Start fresh… (frees {})", human(frees)) } else { "Start fresh…".into() };
                if w::button_enabled(ui, p, &label, Kind::Normal, None, anything && busy.is_none()).clicked() {
                    f.confirm = true;
                }
                if w::button(ui, p, "Cancel", Kind::Ghost, None).clicked() {
                    cancel = true;
                }
            } else {
                if w::button_enabled(ui, p, "Clear and start again", Kind::Danger, None, busy.is_none()).clicked() {
                    go = true;
                }
                if w::button(ui, p, "Back", Kind::Ghost, None).clicked() {
                    f.confirm = false;
                }
                let undo = if backup { "The index is kept as a backup; nothing else can be undone." } else { "This can't be undone." };
                ui.label(RichText::new(undo).font(body(12.5)).color(p.warn));
            }
        });
        if let Some(b) = busy {
            ui.label(RichText::new(b).font(body(12.5)).color(p.warn));
        }
        if std::mem::take(&mut f.harness_go) && explicit.is_some() && !real && busy.is_none() {
            go = true;
        }
        if cancel {
            f.open = false;
            f.survey = None;
        }
        if std::mem::take(&mut f.scroll) {
            ui.scroll_to_cursor(Some(egui::Align::BOTTOM));
        }
    });
    go
}

/// Clear in the background; [`tick`] finishes up.
pub fn go(a: &mut UiState, cfg: &Config, current: Option<&Path>) {
    let f = &mut a.fresh;
    let Some(s) = f.survey.take() else { return };
    let o = f.opts.clone();
    f.open = false;
    f.confirm = false;
    f.error = None;
    f.last = None;
    f.after = Some((cfg.clone(), current.map(Path::to_path_buf)));
    let (tx, rx) = channel();
    f.running = Some(rx);
    std::thread::Builder::new()
        .name("crux-fresh".into())
        .spawn(move || {
            let _ = tx.send(crux_core::fresh::run(&s, &o).map_err(|e| format!("{e:#}")));
        })
        .expect("spawn fresh");
}

/// Every frame: finish a clear (refresh, reopen) and reopen once the
/// refresh is done.
pub fn tick(a: &mut UiState, ctx: &egui::Context) {
    if let Some(rx) = &a.fresh.running {
        let r = match rx.try_recv() {
            Ok(r) => r,
            Err(TryRecvError::Empty) => {
                ctx.request_repaint_after(Duration::from_millis(200));
                return;
            }
            Err(TryRecvError::Disconnected) => Err("clearing stopped unexpectedly".into()),
        };
        a.fresh.running = None;
        let (cfg, current) = a.fresh.after.take().unwrap_or_else(|| (Config::default(), None));
        match r {
            Ok(d) => {
                let summary = d.summary();
                a.fresh.last = Some((summary.clone(), d.errors.clone()));
                a.library.clear();
                if a.fresh.refresh_after && !cfg.library.search_roots().is_empty() {
                    crate::auto::start_refresh(a, &cfg, ctx);
                    a.fresh.reopen = current;
                    a.toast(ctx, format!("{summary}. Refreshing the calibration library, then reopening."), false);
                } else {
                    a.open_request = current;
                    a.toast(ctx, summary, false);
                }
            }
            Err(e) => a.fresh.error = Some(format!("Nothing was cleared: {e}")),
        }
    }
    // The refresh reopens the open folder itself; in library mode (no
    // folder graded) it doesn't, so reopen here.
    if a.fresh.reopen.is_some() && !a.bg.refresh.running {
        let r = a.fresh.reopen.take().unwrap();
        if a.open_request.as_deref() != Some(r.as_path()) && a.loading.as_deref() != Some(r.as_path()) {
            a.open_request = Some(r);
        }
    }
}
