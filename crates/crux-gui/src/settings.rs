//! Settings page: the per-user `crux.toml` (`%APPDATA%\crux\crux.toml`),
//! the rail's Settings entry from `docs/ui-design.md`. Only values that
//! differ from crux's defaults are saved (`Config::save_user`), so later
//! default changes still apply. A folder's own `crux.toml` is used instead
//! of this file; the page says so when the open folder has one.
//!
//! Saving applies processing, mosaic, library and cache settings to the
//! open folder at once; site and camera values (nights, Moon, sky levels)
//! are worked out when a folder opens, so those offer a reopen.

use crate::app::UiState;
use crate::session::Session;
use crate::theme::{body, mono, Pal};
use crate::widgets::{self as w, Kind};
use crux_core::config::Config;
use crux_core::process::job::Settings;
use eframe::egui::{self, RichText, Ui};
use std::path::PathBuf;

#[derive(Default)]
pub struct SettingsUi {
    /// Being edited; loaded from the file the first time the page shows.
    draft: Option<Config>,
    /// What the file holds (defaults when there's no file).
    saved: Config,
    file: Option<PathBuf>,
    /// The file couldn't be read: saving would overwrite it, so it's off.
    load_error: Option<String>,
    error: Option<String>,
    note: Option<String>,
    /// `library.search`, one folder per line.
    search_text: String,
    workspace_text: String,
    root_text: String,
    /// A saved site or camera change waits for the folder to reopen.
    reopen: bool,
}

impl SettingsUi {
    fn load(&mut self) {
        self.file = Config::user_path();
        let (cfg, err) = match Config::load(None, None) {
            Ok((c, _)) => (c, None),
            Err(e) => (Config::default(), Some(format!("{e:#}"))),
        };
        self.load_error = err;
        self.sync_text(&cfg);
        self.saved = cfg.clone();
        self.draft = Some(cfg);
    }

    fn sync_text(&mut self, c: &Config) {
        self.search_text = c.library.search.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join("\n");
        self.workspace_text = c.process.workspace.display().to_string();
        self.root_text = c.library.root.display().to_string();
    }
}

fn same<T: serde::Serialize>(a: &T, b: &T) -> bool {
    serde_json::to_value(a).ok() == serde_json::to_value(b).ok()
}

pub fn show(ui: &mut Ui, a: &mut UiState, s: Option<&mut Session>) {
    let p = a.pal;
    if a.settings.draft.is_none() {
        a.settings.load();
    }
    let root = s.as_ref().map(|s| s.root.clone());
    let own_file = root.as_ref().map(|r| r.join("crux.toml")).filter(|f| f.is_file());
    w::h1(ui, "Settings");
    let file = a.settings.file.as_ref().map(|f| f.display().to_string()).unwrap_or_else(|| "(no settings folder)".into());
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new("Saved to").color(p.muted));
        ui.label(RichText::new(&file).font(mono(12.0)).color(p.muted));
        ui.label(RichText::new("— only the values you change; everything else follows crux's defaults.").color(p.muted));
    });
    if let Some(f) = &own_file {
        ui.label(RichText::new(format!("This folder has its own {}, which crux uses instead of these settings.", f.display())).color(p.warn));
    }
    if let Some(e) = &a.settings.load_error {
        ui.label(RichText::new(format!("Couldn't read the settings file, so saving is off until it's fixed: {e}")).color(p.rej));
    }
    ui.add_space(8.0);

    let def = Config::default();
    let mut d = a.settings.draft.take().unwrap_or_default();
    let ixp = a.index_path.clone();
    let current = root.clone().or_else(|| a.lib.view.as_ref().map(|v| v.root.clone()));
    let fresh_cfg = s.as_ref().map(|s| s.cfg.clone()).unwrap_or_else(|| a.settings.saved.clone());
    let busy = crate::fresh::busy(a);
    let explicit = a.lib.cache_override.clone();
    let fresh = &mut a.fresh;
    let st = &mut a.settings;
    let mut go_fresh = false;
    egui::ScrollArea::vertical().auto_shrink([false, false]).max_height(ui.available_height() - 56.0).show(ui, |ui| {
        processing(ui, &p, st, &mut d, &def);
        ui.add_space(10.0);
        library(ui, &p, st, &mut d, &def);
        ui.add_space(10.0);
        site(ui, &p, &mut d, &def);
        ui.add_space(10.0);
        go_fresh = crate::fresh::card(ui, &p, fresh, busy, ixp.as_deref(), explicit.as_deref(), &fresh_cfg);
        ui.add_space(10.0);
    });

    // Footer: save, discard, open the folder.
    let dirty = !same(&d, &st.saved);
    let tz_bad = !d.site.timezone_ok();
    ui.add_space(8.0);
    ui.horizontal(|ui| {
        let can = dirty && st.load_error.is_none() && !tz_bad;
        if w::button_enabled(ui, &p, "Save", Kind::Primary, None, can).clicked() {
            match d.save_user() {
                Ok(path) => {
                    crux_core::gpu::set_enabled(d.gpu.r#use == crux_core::config::GpuUse::Auto);
                    let old = std::mem::replace(&mut st.saved, d.clone());
                    st.file = Some(path);
                    st.error = None;
                    st.note = Some(match (s, &own_file) {
                        (Some(_), Some(_)) => "Saved. The open folder keeps using its own crux.toml.".into(),
                        (Some(sess), None) => {
                            st.reopen |= apply(&mut a.stack, sess, &old, &d);
                            "Saved and applied to the open folder.".into()
                        }
                        (None, _) => "Saved. Folders use it when they open.".into(),
                    });
                }
                Err(e) => st.error = Some(format!("{e:#}")),
            }
        }
        if w::button_enabled(ui, &p, "Discard changes", Kind::Ghost, None, dirty).clicked() {
            d = st.saved.clone();
            st.sync_text(&d);
            st.note = None;
        }
        if let Some(dir) = st.file.as_ref().and_then(|f| f.parent()) {
            if w::button(ui, &p, "Open settings folder", Kind::Ghost, None).clicked() {
                let _ = std::fs::create_dir_all(dir);
                let _ = crate::stack::open_folder(dir);
            }
        }
        if dirty {
            ui.label(RichText::new(if tz_bad { "Fix the time zone to save" } else { "Unsaved changes" }).font(body(12.0)).color(if tz_bad { p.rej } else { p.warn }));
        } else if let Some(n) = &st.note {
            ui.label(RichText::new(n).font(body(12.0)).color(p.good));
        }
        if let Some(e) = &st.error {
            ui.label(RichText::new(e).font(body(12.0)).color(p.rej));
        }
    });
    if let (true, Some(root)) = (st.reopen, root) {
        ui.horizontal(|ui| {
            ui.label(RichText::new("Site and camera values are worked out when a folder opens (nights, Moon, sky levels).").font(body(12.0)).color(p.muted));
            if w::button(ui, &p, "Reopen the folder now", Kind::Normal, None).clicked() {
                st.reopen = false;
                a.open_request = Some(root);
            }
        });
    }
    a.settings.draft = Some(d);
    if go_fresh {
        crate::fresh::go(a, &fresh_cfg, current.as_deref());
    }
}

/// Put saved settings into the open folder's session. Returns whether
/// something that's only read when a folder opens changed.
fn apply(stack: &mut crate::stack::StackUi, s: &mut Session, old: &Config, new: &Config) -> bool {
    let before = s.base_cfg.clone();
    let b = &mut s.base_cfg;
    b.process = new.process.clone();
    b.mosaic = new.mosaic.clone();
    b.cache = new.cache.clone();
    // `--library DIR` points the index somewhere else: keep that.
    if same(&b.library.dir, &old.library.dir) {
        b.library = new.library.clone();
    } else {
        b.library.search = new.library.search.clone();
    }
    b.site = new.site.clone();
    b.camera = new.camera.clone();
    let reopen = !same(&before.site, &b.site) || !same(&before.camera, &b.camera);
    s.regrade();
    // The Stack page's settings follow the new defaults unless changed there.
    if stack.settings.as_ref().is_some_and(|st| same(st, &Settings::from_config(&before))) {
        stack.settings = None;
    }
    reopen
}

// ------------------------------------------------------------- rows

fn source(ui: &mut Ui, p: &Pal, changed: bool) -> bool {
    if changed {
        ui.label(RichText::new("changed").font(body(11.0)).color(p.accent));
        ui.small_button("Reset").on_hover_text("Back to crux's default").clicked()
    } else {
        ui.label(RichText::new("default").font(body(11.0)).color(p.faint));
        false
    }
}

fn label(ui: &mut Ui, text: &str, help: &str) {
    ui.label(RichText::new(text).font(body(13.0))).on_hover_text(help);
}

/// A row with a checkbox; resets to `def`.
fn check(ui: &mut Ui, p: &Pal, text: &str, help: &str, v: &mut bool, def: bool) {
    label(ui, text, help);
    ui.checkbox(v, "");
    ui.horizontal(|ui| {
        if source(ui, p, *v != def) {
            *v = def;
        }
    });
    ui.end_row();
}

fn choice(ui: &mut Ui, p: &Pal, text: &str, help: &str, v: &mut String, def: &str, opts: &[(&str, &str)]) {
    label(ui, text, help);
    let cur = opts.iter().find(|(k, _)| v.eq_ignore_ascii_case(k)).map(|(_, l)| *l).unwrap_or(v.as_str()).to_string();
    egui::ComboBox::from_id_salt(("set", text)).selected_text(cur).width(220.0).show_ui(ui, |ui| {
        for (k, l) in opts {
            if ui.selectable_label(v.eq_ignore_ascii_case(k), *l).clicked() {
                *v = k.to_string();
            }
        }
    });
    ui.horizontal(|ui| {
        if source(ui, p, !v.eq_ignore_ascii_case(def)) {
            *v = def.to_string();
        }
    });
    ui.end_row();
}

fn number(ui: &mut Ui, p: &Pal, text: &str, help: &str, v: &mut f64, def: f64, range: std::ops::RangeInclusive<f64>, speed: f64, decimals: usize, suffix: &str) {
    label(ui, text, help);
    ui.horizontal(|ui| {
        ui.set_min_width(CONTROL_W);
        ui.add(egui::DragValue::new(v).range(range).speed(speed).max_decimals(decimals).min_decimals(decimals.min(1)).suffix(suffix));
    });
    ui.horizontal(|ui| {
        if source(ui, p, (*v - def).abs() > 1e-12) {
            *v = def;
        }
    });
    ui.end_row();
}

/// A folder path typed or picked; `text` mirrors it while typing.
fn folder(ui: &mut Ui, p: &Pal, label_text: &str, help: &str, v: &mut PathBuf, def: &PathBuf, text: &mut String) {
    label(ui, label_text, help);
    ui.horizontal(|ui| {
        if ui.add(egui::TextEdit::singleline(text).desired_width(320.0).font(mono(12.0))).changed() {
            *v = PathBuf::from(text.trim());
        }
        if ui.button("Browse…").clicked() {
            if let Some(f) = rfd::FileDialog::new().set_title(label_text).set_directory(&*v).pick_folder() {
                *v = f;
                *text = v.display().to_string();
            }
        }
    });
    ui.horizontal(|ui| {
        if source(ui, p, v != def) {
            *v = def.clone();
            *text = v.display().to_string();
        }
    });
    ui.end_row();
}

/// Width of the widest control row (a path box and its Browse button), so
/// the default/changed column lines up in every card.
const CONTROL_W: f32 = 360.0;

fn grid(ui: &mut Ui, id: &str, add: impl FnOnce(&mut Ui)) {
    // The same first-column width in every card, so the controls line up.
    egui::Grid::new(id).num_columns(3).spacing([18.0, 8.0]).min_col_width(220.0).show(ui, add);
}

// ------------------------------------------------------------- cards

fn processing(ui: &mut Ui, p: &Pal, st: &mut SettingsUi, d: &mut Config, def: &Config) {
    w::card(ui, p, |ui| {
        w::h2(ui, "Processing");
        w::muted(ui, p, "Defaults for the stacking steps and Autopilot. The Stack page can change them for one run.");
        ui.add_space(6.0);
        grid(ui, "set-proc", |ui| {
            folder(ui, p, "Workspace", "Where calibrated and registered frames, masters, mosaics and images go. Never inside your data folders.", &mut d.process.workspace, &def.process.workspace, &mut st.workspace_text);
            check(ui, p, "FITS copies", "Also write a 32-bit float .fits beside each integrated master, mosaic and finished image, for programs that don't read XISF. XISF stays the main output.", &mut d.process.fits, def.process.fits);
            check(ui, p, "Include warn frames", "Process lights graded warn as well as good.", &mut d.process.include_warn, def.process.include_warn);
            check(ui, p, "Keep calibrated frames", "Also write the calibrated frames before registration (calibrated\\ in the workspace).", &mut d.process.keep_calibrated, def.process.keep_calibrated);
            label(ui, "Fix hot pixels", "Replace hot pixels with the median of their neighbours. A pixel is hot when it's this many standard deviations above the master dark's median (WBPP uses 10).");
            ui.horizontal(|ui| {
                let mut on = d.process.hot_sigma > 0.0;
                ui.checkbox(&mut on, "");
                let mut k = if on { d.process.hot_sigma } else { def.process.hot_sigma };
                ui.add_enabled(on, egui::Slider::new(&mut k, 5.0..=20.0).step_by(0.5).suffix(" σ"));
                d.process.hot_sigma = if on { k } else { 0.0 };
            });
            ui.horizontal(|ui| {
                if source(ui, p, d.process.hot_sigma != def.process.hot_sigma) {
                    d.process.hot_sigma = def.process.hot_sigma;
                }
            });
            ui.end_row();
            check(ui, p, "Integrate", "Stack each field and filter into a master light after registering.", &mut d.process.integrate, def.process.integrate);
            choice(ui, p, "Normalisation", "How frames are matched before stacking.", &mut d.process.normalisation, &def.process.normalisation, &[("local", "Local (follows changing gradients)"), ("global", "Global")]);
            choice(ui, p, "Weights", "How much each frame counts.", &mut d.process.weights, &def.process.weights, &[("snr", "Signal to noise"), ("exposure", "Exposure"), ("equal", "Equal")]);
            choice(
                ui,
                p,
                "Rejection",
                "Outlier rejection. Auto picks by stack size: median or percentile for small stacks, winsorised sigma, then ESD.",
                &mut d.process.rejection,
                &def.process.rejection,
                &[("auto", "Auto (by stack size)"), ("esd", "ESD"), ("winsorized", "Winsorised sigma"), ("sigma", "Sigma"), ("percentile", "Percentile"), ("none", "None")],
            );
            check(ui, p, "Mosaic after integrating", "Assemble mosaics of panel targets after the stacking steps or Autopilot integrate them.", &mut d.mosaic.after_integrate, def.mosaic.after_integrate);
            let mut on = d.gpu.r#use == crux_core::config::GpuUse::Auto;
            check(
                ui,
                p,
                "Use the graphics card",
                "Run Sharpen and denoise, the star remover and other heavy steps on the graphics card when crux finds one (NVIDIA, AMD, Intel or Apple). Off: always the CPU, which is much slower but gives the same result.",
                &mut on,
                true,
            );
            d.gpu.r#use = if on { crux_core::config::GpuUse::Auto } else { crux_core::config::GpuUse::Off };
            ui.label("");
            w::muted(ui, p, &format!("Now computing on: {}", crux_core::gpu::describe()));
            ui.label("");
            ui.end_row();
        });
    });
}

fn library(ui: &mut Ui, p: &Pal, st: &mut SettingsUi, d: &mut Config, def: &Config) {
    w::card(ui, p, |ui| {
        w::h2(ui, "Calibration library and cache");
        ui.add_space(6.0);
        grid(ui, "set-lib", |ui| {
            label(ui, "Search folders", "Refresh calibration library scans every folder inside each of these for flats, darks, bias and dark flats, which stay where they are. One folder per line; empty = the folder holding the library root.");
            ui.vertical(|ui| {
                if ui.add(egui::TextEdit::multiline(&mut st.search_text).desired_rows(2).desired_width(320.0).font(mono(12.0)).hint_text("one folder per line")).changed() {
                    d.library.search = st.search_text.lines().map(str::trim).filter(|l| !l.is_empty()).map(PathBuf::from).collect();
                }
                ui.horizontal(|ui| {
                    if ui.button("Add folder…").clicked() {
                        if let Some(f) = rfd::FileDialog::new().set_title("Search this folder for calibration frames").pick_folder() {
                            d.library.search.push(f);
                            st.search_text = d.library.search.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join("\n");
                        }
                    }
                    let roots = d.library.search_roots();
                    let s = if roots.is_empty() { "nothing to search".to_string() } else { roots.iter().map(|r| r.display().to_string()).collect::<Vec<_>>().join(", ") };
                    ui.label(RichText::new(format!("Searching: {s}")).font(body(11.5)).color(p.muted));
                });
            });
            ui.horizontal(|ui| {
                if source(ui, p, d.library.search != def.library.search) {
                    d.library.search.clear();
                    st.search_text.clear();
                }
            });
            ui.end_row();
            folder(ui, p, "Library root", "Where crux files flat sets when a plan sends them to the library (<root>\\<filter>\\<night>). Nothing goes here otherwise.", &mut d.library.root, &def.library.root, &mut st.root_text);
            label(ui, "Empty unused caches after", "Empty the rebuildable part of a folder's cache (measurements, thumbnails, maps) when the folder hasn't been opened for this long. Decisions and plans stay. 0 = never.");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut d.cache.unused_days).range(0..=3650).suffix(" days"));
                if d.cache.unused_days == 0 {
                    ui.label(RichText::new("never").font(body(11.5)).color(p.muted));
                }
            });
            ui.horizontal(|ui| {
                if source(ui, p, d.cache.unused_days != def.cache.unused_days) {
                    d.cache.unused_days = def.cache.unused_days;
                }
            });
            ui.end_row();
            check(ui, p, "Cache inside each data folder", "Keep each folder's cache in <folder>\\.crux, as crux did before, instead of in your user data folder. Off keeps the data folders untouched.", &mut d.cache.in_folder, def.cache.in_folder);
        });
    });
}

fn site(ui: &mut Ui, p: &Pal, d: &mut Config, def: &Config) {
    w::card(ui, p, |ui| {
        w::h2(ui, "Site and camera");
        w::muted(ui, p, "Used when a frame's header doesn't say: the site for nights, twilight and the Moon; the camera for sky levels and bias checks.");
        ui.add_space(6.0);
        grid(ui, "set-site", |ui| {
            number(ui, p, "Latitude", "Degrees, north positive (SITELAT).", &mut d.site.latitude, def.site.latitude, -90.0..=90.0, 0.001, 4, "°");
            number(ui, p, "Longitude", "Degrees, east positive (SITELONG).", &mut d.site.longitude, def.site.longitude, -180.0..=180.0, 0.001, 4, "°");
            label(ui, "Time zone", "IANA name, such as Australia/Melbourne or Europe/London. Turns UTC times into local ones when a frame has no DATE-LOC; a night runs noon to noon local time.");
            ui.horizontal(|ui| {
                ui.set_min_width(CONTROL_W);
                ui.add(egui::TextEdit::singleline(&mut d.site.timezone).desired_width(220.0).font(mono(12.0)));
                if d.site.timezone_ok() {
                    let now = d.site.to_local(chrono::Utc::now().naive_utc());
                    ui.label(RichText::new(format!("local time now {}", now.format("%H:%M"))).font(body(11.5)).color(p.muted));
                } else {
                    ui.label(RichText::new("not a time zone crux knows").font(body(11.5)).color(p.rej));
                }
            });
            ui.horizontal(|ui| {
                if source(ui, p, d.site.timezone != def.site.timezone) {
                    d.site.timezone = def.site.timezone.clone();
                }
            });
            ui.end_row();
            number(ui, p, "Camera gain", "Electrons per ADU, when EGAIN is missing.", &mut d.camera.egain, def.camera.egain, 0.01..=20.0, 0.001, 3, " e-/ADU");
            number(ui, p, "Expected bias level", "The bias offset you usually shoot at, in ADU. Bias sets more than 50 ADU from it are flagged.", &mut d.camera.bias_level_adu, def.camera.bias_level_adu, 0.0..=65535.0, 1.0, 0, " ADU");
        });
    });
}
