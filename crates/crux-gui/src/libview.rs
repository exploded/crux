//! Library mode (milestone 15): a folder holding several targets in
//! folders of their own (`C:\PixInsight`) opens as the library, not as one
//! target. Home then shows every target found from the headers, the
//! calibration frames crux can use for them (anywhere in the library
//! folders), targets split over several folders with an offer to combine
//! them, and what disagrees (names, sky positions). Nothing is graded
//! until you open a target.

use crate::app::UiState;
use crate::home::{target_card, Target};
use crate::theme::{body, mono, semibold};
use crate::widgets::{self as w, d_short, Kind};
use chrono::Datelike;
use crux_core::config::Config;
use crux_core::targets::LibraryTargets;
use eframe::egui::{self, RichText, ScrollArea};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// A library folder opened (from the worker).
pub struct LibView {
    pub root: PathBuf,
    pub cfg: Config,
    pub targets: LibraryTargets,
    /// The folder is one of the library folders (`[library] search`).
    pub configured: bool,
    pub library: Option<crux_core::library::Library>,
    /// Target folder key (`Unit::folder_for`) → its picture (crux's image,
    /// else a JPG/PNG in it).
    pub images: HashMap<String, PathBuf>,
    /// Target folder key → what crux has made for it in the workspace.
    pub made: HashMap<String, Made>,
}

/// What crux has made for a folder in the workspace: integrated masters
/// and finished images (newest of each), read when the library opens.
#[derive(Debug, Clone, Copy)]
pub struct Made {
    pub masters: usize,
    pub masters_at: std::time::SystemTime,
    pub image_at: Option<std::time::SystemTime>,
}

/// None when nothing has been stacked for `dir` yet.
pub fn made(dir: &Path, workspace: &Path) -> Option<Made> {
    let out = crux_core::process::job::out_dir(dir, workspace);
    let targets = crux_core::render::targets(&out);
    let mtime = |p: &Path| p.metadata().and_then(|m| m.modified()).ok();
    let masters: Vec<std::time::SystemTime> = targets.iter().flat_map(|t| t.masters.iter()).filter_map(|m| mtime(&m.path)).collect();
    let masters_at = masters.iter().max().copied()?;
    // Finished images (JPEG, PNG or XISF; older runs wrote no JPEG), not
    // the preview.
    let image_at = targets
        .iter()
        .filter_map(|t| std::fs::read_dir(t.image_dir(&out)).ok())
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.file_name().is_some_and(|n| n != "preview.png"))
        .filter(|p| p.extension().is_some_and(|x| ["jpg", "png", "xisf"].iter().any(|e| x.eq_ignore_ascii_case(e))))
        .filter_map(|p| mtime(&p))
        .max();
    Some(Made { masters: masters.len(), masters_at, image_at })
}

/// A card's status from what's been made: green once an image is newer
/// than every master.
pub fn made_status(m: &Made, p: &crate::theme::Pal) -> (Option<egui::Color32>, String) {
    let day = |t: std::time::SystemTime| chrono::DateTime::<chrono::Local>::from(t).date_naive();
    let masters = format!("{} master{}", m.masters, if m.masters == 1 { "" } else { "s" });
    match m.image_at {
        Some(i) if i >= m.masters_at => (Some(p.good), format!("Image made {} · {masters}", d_short(day(i)))),
        Some(_) => (Some(p.accent), format!("Stacked again {} · image older", d_short(day(m.masters_at)))),
        None => (Some(p.accent), format!("Stacked {} · {masters}, no image yet", d_short(day(m.masters_at)))),
    }
}

#[derive(Default)]
pub struct LibUi {
    pub view: Option<LibView>,
    /// `--cache-dir`: every folder shares that one cache, so a combine's
    /// entries are renamed there in place.
    pub cache_override: Option<PathBuf>,
    pub error: Option<String>,
}

/// Where library-level plans and journals go (combines).
pub fn plans_dir(cache_root: Option<&Path>) -> PathBuf {
    match cache_root {
        Some(r) => r.join("plans"),
        None => crux_core::paths::data_dir().unwrap_or_else(std::env::temp_dir).join("plans"),
    }
}

pub fn show(ui: &mut egui::Ui, a: &mut UiState) {
    // Out of the state while drawing (the cards change `a`), back after.
    let Some(view) = a.lib.view.take() else { return };
    let v = &view;
    let p = a.pal;
    let root = v.root.clone();
    let lt = &v.targets;
    let n_t = lt.targets.len();
    let cal_units = lt.units.iter().filter(|u| u.lights() == 0 && u.calibration() > 0).count();
    let mut open: Option<PathBuf> = None;
    let mut add_root = false;
    let mut combine: Option<usize> = None;
    let mut undo: Option<crux_core::moves::HistoryEntry> = None;
    // Owned copies of what the cards draw, so `a` is free for them.
    let cards: Vec<Target> = lt.targets.iter().map(|t| card_for(t, v, &p)).collect();
    let combinable = lt.combinable();
    let notes: Vec<(String, String)> = lt
        .notes
        .iter()
        .map(|n| (String::new(), n.clone()))
        .chain(lt.targets.iter().flat_map(|t| t.notes.iter().map(move |n| (t.name.clone(), n.clone()))))
        .collect();
    let units_cal: Vec<(String, String, PathBuf)> = lt
        .units
        .iter()
        .filter(|u| u.lights() == 0 && u.calibration() > 0)
        .map(|u| {
            let mut what = Vec::new();
            for (n, w) in [(u.flats, "flats"), (u.darks, "darks"), (u.bias, "bias"), (u.dark_flats, "dark flats")] {
                if n > 0 {
                    what.push(format!("{n} {w}"));
                }
            }
            (u.name.clone(), what.join(", "), u.path.clone())
        })
        .collect();
    let configured = v.configured;
    let cfg = v.cfg.clone();
    let library = v.library.clone();
    let history: Vec<crux_core::moves::HistoryEntry> = crux_core::moves::history(&plans_dir(a.lib.cache_override.as_deref()))
        .into_iter()
        .filter(|e| crux_core::paths::key(&e.root) == crux_core::paths::key(&root) && e.finished && e.undone.is_none())
        .take(3)
        .collect();

    ScrollArea::vertical().id_salt("library").auto_shrink([false, false]).show(ui, |ui| {
        ui.set_width(ui.available_width() - 8.0);
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                w::h1(ui, "Your library");
                ui.label(
                    RichText::new(format!(
                        "{n_t} target{} in {}. Open a target to grade it.",
                        if n_t == 1 { "" } else { "s" },
                        root.display()
                    ))
                    .color(p.muted),
                );
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Max), |ui| {
                if w::button(ui, &p, "+ Open folder", Kind::Normal, None).clicked() {
                    crate::home::pick_folder(a);
                }
            });
        });
        ui.add_space(16.0);
        if !configured {
            w::card(ui, &p, |ui| {
                w::eyebrow(ui, &p, "Library folder");
                ui.label(RichText::new(format!("Use {} as a library folder?", root.display())).font(semibold(16.0)));
                ui.add(
                    egui::Label::new(
                        RichText::new("crux then searches it for flats, darks and bias for every target. It searches only your library folders. You can list several (C:\\2025, C:\\2026, \\\\nas\\astronomy\\…) in Settings.")
                            .color(p.muted),
                    )
                    .wrap(),
                );
                ui.add_space(6.0);
                if w::button(ui, &p, "Add to my library folders", Kind::Primary, None).clicked() {
                    add_root = true;
                }
            });
            ui.add_space(16.0);
        }
        let wide = ui.available_width() >= 900.0;
        let total = ui.available_width();
        let (lw, rw) = if wide { ((total - 16.0) * 1.4 / 2.4, (total - 16.0) / 2.4) } else { (total, total) };
        let left = |ui: &mut egui::Ui, a: &mut UiState, open: &mut Option<PathBuf>| {
            ui.set_width(lw);
            ui.horizontal(|ui| {
                w::h2(ui, "Targets");
            });
            ui.add_space(8.0);
            if cards.is_empty() {
                ui.label(RichText::new("No lights found in this folder's subfolders.").color(p.muted));
                return;
            }
            let gap = 14.0;
            let wd = ui.available_width();
            let cols = ((wd + gap) / (230.0 + gap)).floor().max(1.0) as usize;
            let cw = (wd - gap * (cols as f32 - 1.0)) / cols as f32;
            for row in cards.chunks(cols) {
                ui.horizontal_top(|ui| {
                    ui.spacing_mut().item_spacing.x = gap;
                    for t in row {
                        target_card(ui, a, t, cw);
                    }
                });
                ui.add_space(gap);
            }
            let _ = open;
        };
        let right = |ui: &mut egui::Ui, a: &mut UiState, combine: &mut Option<usize>, undo: &mut Option<crux_core::moves::HistoryEntry>| {
            ui.set_width(rw);
            if !combinable.is_empty() || !history.is_empty() {
                w::card(ui, &p, |ui| {
                    w::h2(ui, "In several folders");
                    ui.add(
                        egui::Label::new(RichText::new("The same target in more than one folder. Combining moves each folder, whole and unchanged, into one folder named after the target. You review the plan before anything moves, and you can undo it.").font(body(12.5)).color(p.muted))
                            .wrap(),
                    );
                    for (ti, c) in &combinable {
                        let t = &lt.targets[*ti];
                        ui.add_space(8.0);
                        egui::Frame::new().fill(p.panel2).corner_radius(8).inner_margin(egui::Margin::symmetric(10, 8)).show(ui, |ui| {
                            ui.set_width(ui.available_width());
                            ui.label(RichText::new(format!("{} · {} folders", t.name, c.moving.len() + c.existing.is_some() as usize)).font(semibold(13.5)));
                            ui.add(egui::Label::new(RichText::new(format!("into {}", c.dest.display())).font(body(12.0)).color(p.muted)).wrap());
                            for &u in c.existing.iter().chain(&c.moving) {
                                ui.label(RichText::new(format!("  {}", lt.units[u].name)).font(mono(12.0)).color(p.muted));
                            }
                            for (u, why) in &c.blocked {
                                ui.label(RichText::new(format!("  {} stays: {why}", lt.units[*u].name)).font(body(12.0)).color(p.warn));
                            }
                            if w::button(ui, &p, "Combine…", Kind::Normal, None).clicked() {
                                *combine = Some(*ti);
                            }
                        });
                    }
                    for e in &history {
                        ui.add_space(8.0);
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(format!("Combined {} files on {}", e.moves, e.started)).font(body(12.0)).color(p.muted));
                            if w::button(ui, &p, "Undo", Kind::Ghost, None).clicked() {
                                *undo = Some(e.clone());
                            }
                        });
                    }
                });
                ui.add_space(16.0);
            }
            w::card(ui, &p, |ui| {
                w::h2(ui, "Calibration");
                let sets = library.as_ref().map(|l| l.sets.len()).unwrap_or(0);
                let usable = library.as_ref().map(|l| l.sets.iter().filter(|s| !s.retired && s.grade != crux_core::grade::Grade::Reject).count()).unwrap_or(0);
                ui.add(
                    egui::Label::new(
                        RichText::new(if sets > 0 {
                            format!("{sets} flat sets in the calibration library ({usable} usable). Each target takes the best flats, darks and bias from anywhere in your library folders when it's processed.")
                        } else {
                            "The calibration library is empty: refresh it to find every flat, dark and bias in your library folders.".into()
                        })
                        .font(body(12.5))
                        .color(p.muted),
                    )
                    .wrap(),
                );
                ui.add_space(4.0);
                if configured {
                    crate::auto::refresh_ui(ui, a, &cfg, library.as_ref());
                } else {
                    ui.label(RichText::new("Add this folder to your library folders to search it.").font(body(12.0)).color(p.warn));
                }
                if !units_cal.is_empty() {
                    ui.add_space(8.0);
                    egui::CollapsingHeader::new(RichText::new(format!("{cal_units} folder{} with calibration frames only", if cal_units == 1 { "" } else { "s" })).font(semibold(12.5)))
                        .id_salt("lib-cal-folders")
                        .show(ui, |ui| {
                            for (name, what, _) in &units_cal {
                                ui.horizontal(|ui| {
                                    ui.label(RichText::new(name).font(mono(12.0)));
                                    ui.label(RichText::new(what).font(body(12.0)).color(p.muted));
                                });
                            }
                        });
                }
            });
            if !notes.is_empty() {
                ui.add_space(16.0);
                w::card(ui, &p, |ui| {
                    w::h2(ui, "Needs checking");
                    for (t, n) in &notes {
                        ui.add_space(4.0);
                        let line = if t.is_empty() { n.clone() } else { format!("{t}: {n}") };
                        ui.add(egui::Label::new(RichText::new(line).font(body(12.5)).color(p.muted)).wrap());
                    }
                });
            }
            ui.add_space(16.0);
            let busy = crate::fresh::busy(a);
            let (ix, explicit) = (a.index_path.clone(), a.lib.cache_override.clone());
            if crate::fresh::card(ui, &p, &mut a.fresh, busy, ix.as_deref(), explicit.as_deref(), &v.cfg) {
                crate::fresh::go(a, &v.cfg, Some(&root));
            }
        };
        if wide {
            ui.horizontal_top(|ui| {
                ui.vertical(|ui| left(ui, a, &mut open));
                ui.add_space(8.0);
                ui.vertical(|ui| right(ui, a, &mut combine, &mut undo));
            });
        } else {
            left(ui, a, &mut open);
            ui.add_space(16.0);
            right(ui, a, &mut combine, &mut undo);
        }
        ui.add_space(40.0);
    });
    // Unless something in the page opened another folder meanwhile.
    if a.lib.view.is_none() {
        a.lib.view = Some(view);
    }

    if let Some(o) = open {
        a.from_library = Some(root.clone());
        a.open_request = Some(o);
    }
    if add_root {
        match add_library_folder(&root) {
            Ok(_) => {
                if let Some(v) = a.lib.view.as_mut() {
                    v.configured = true;
                    v.cfg.library.search = Config::load(None, None).map(|c| c.0.library.search).unwrap_or_default();
                    let cfg = v.cfg.clone();
                    crate::auto::start_refresh(a, &cfg, ui.ctx());
                }
            }
            Err(e) => a.lib.error = Some(format!("{e:#}")),
        }
    }
    if let Some(ti) = combine {
        let Some(v) = a.lib.view.as_ref() else { return };
        match v.targets.combine_plan(ti, &v.cfg, None) {
            Ok(mut plan) => {
                if let Some(c) = &a.lib.cache_override {
                    plan.cache_dir = Some(c.clone());
                }
                let title = format!("Combine {} into {}?", v.targets.targets[ti].name, crux_core::targets::folder_name(&v.targets.targets[ti].name));
                a.request = Some(crate::organise::Request::Apply { plan, title });
            }
            Err(e) => a.lib.error = Some(format!("{e:#}")),
        }
    }
    if let Some(e) = undo {
        a.request = Some(crate::organise::Request::Undo(e));
    }
}

/// A target card: its newest folder opens on click.
fn card_for(t: &crux_core::targets::Target, v: &LibView, p: &crate::theme::Pal) -> Target {
    let lt = &v.targets;
    let units: Vec<&crux_core::targets::Unit> = t.units.iter().map(|&u| &lt.units[u]).collect();
    let key = |u: &crux_core::targets::Unit| crux_core::paths::key(u.folder_for(&t.name));
    let image = units.iter().rev().find_map(|u| v.images.get(&key(u)).cloned());
    let span = match (t.nights.first(), t.nights.last()) {
        (Some(a), Some(b)) if a.year() != b.year() => format!("{}–{}", a.format("%b %Y"), b.format("%b %Y")),
        (Some(a), Some(b)) if a.month() != b.month() => format!("{}–{}", a.format("%b"), b.format("%b %Y")),
        (Some(a), _) => format!("{} {}", d_short(*a), a.year()),
        _ => String::new(),
    };
    let panels = t.panels();
    let sub = format!(
        "{}{} night{} · {span}",
        if panels > 1 { format!("{panels} panels · ") } else { String::new() },
        t.nights.len(),
        if t.nights.len() == 1 { "" } else { "s" }
    );
    let made = units.first().and_then(|u| v.made.get(&key(u)));
    let status = if units.len() > 1 {
        (Some(p.warn), format!("in {} folders", units.len()))
    } else if let Some(m) = made {
        made_status(m, p)
    } else if !t.notes.is_empty() {
        (Some(p.accent), t.notes[0].split(" (").next().unwrap_or(&t.notes[0]).to_string())
    } else {
        (None, format!("{} light{} · open to grade", t.lights, if t.lights == 1 { "" } else { "s" }))
    };
    // The folder holding most of its lights, newest first on a tie.
    let open = t
        .units
        .iter()
        .max_by_key(|&&u| (lt.units[u].targets.get(&t.name).copied().unwrap_or(0), u))
        .map(|&u| lt.units[u].folder_for(&t.name).to_path_buf());
    Target { name: t.name.clone(), sub, secs: t.secs.clone(), status, image, thumb: None, open, is_current: false }
}

/// Add a folder to `[library] search` in the per-user crux.toml.
pub fn add_library_folder(dir: &Path) -> anyhow::Result<PathBuf> {
    let (mut c, _) = Config::load(None, None)?;
    // An empty list means "the folder holding the library root": keep it.
    let mut roots = if c.library.search.is_empty() { c.library.search_roots() } else { c.library.search.clone() };
    if !roots.iter().any(|r| crux_core::paths::key(r) == crux_core::paths::key(dir)) {
        roots.push(dir.to_path_buf());
    }
    c.library.search = roots;
    c.save_user()
}
