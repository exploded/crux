//! Stack page (milestones 11–13): stages on the left (Calibrate, Cosmetic,
//! Register, Plate solve, Integrate, then the image stages in `image.rs`), a
//! before/after split preview on the right (the selected frame, or for
//! Integrate the master beside its reference frame, or its rejection
//! maps), the stage's settings and the frame list. Processing reads the data in place
//! and writes to the workspace (`process.workspace`, `C:\crux-work`).

use crate::app::UiState;
use crate::session::Session;
use crate::theme::{body, mono, semibold, Pal};
use crate::widgets::{self as w, Kind};
use crate::worker::Msg;
use crux_core::process::combine;
use crux_core::process::job::{self, Event, LightResult, Plan, RunReport, Settings, Source};
use crux_core::solve::Solution;
use eframe::egui::{self, Color32, Pos2, Rect, RichText, ScrollArea, Sense, Stroke, StrokeKind, Ui, Vec2};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Stage {
    Calibrate,
    Cosmetic,
    Register,
    Solve,
    Integrate,
    Mosaic,
    /// Sharpen and denoise the linear masters (crux's networks or RC Astro).
    Enhance,
    Gradient,
    Palette,
    Finish,
    Stars,
    /// Crop and save: the last image stage.
    Crop,
}

impl Stage {
    pub fn parse(s: &str) -> Option<Stage> {
        match s {
            "calibrate" => Some(Stage::Calibrate),
            "cosmetic" => Some(Stage::Cosmetic),
            "register" => Some(Stage::Register),
            "solve" => Some(Stage::Solve),
            "integrate" => Some(Stage::Integrate),
            "enhance" => Some(Stage::Enhance),
            "gradient" => Some(Stage::Gradient),
            "palette" => Some(Stage::Palette),
            "finish" => Some(Stage::Finish),
            "stars" => Some(Stage::Stars),
            "mosaic" => Some(Stage::Mosaic),
            "crop" => Some(Stage::Crop),
            _ => None,
        }
    }

    /// Gradient, Palette and stretch, Finish, Stars, Crop and save
    /// (milestone 13, `image.rs`).
    pub fn is_image(self) -> bool {
        matches!(self, Stage::Enhance | Stage::Gradient | Stage::Palette | Stage::Finish | Stage::Stars | Stage::Crop)
    }
}

/// Two stretched images for the split view.
pub struct PreviewImgs {
    pub key: (PathBuf, Stage),
    pub before: egui::ColorImage,
    pub after: egui::ColorImage,
    pub labels: (String, String),
    pub info: Vec<String>,
    /// Circles to draw, as fractions of the image (reference stars).
    pub marks: Vec<(f32, f32)>,
}

pub struct Preview {
    key: (PathBuf, Stage),
    before: egui::TextureHandle,
    after: egui::TextureHandle,
    labels: (String, String),
    info: Vec<String>,
    marks: Vec<(f32, f32)>,
}

pub struct StackUi {
    pub stage: Stage,
    pub settings: Option<Settings>,
    pub plan: Option<Plan>,
    pub plan_error: Option<String>,
    plan_key: String,
    /// Selected light (path).
    pub sel: Option<PathBuf>,
    /// Set by the screenshot harness (CRUX_GUI_SEL).
    pub pinned_sel: bool,
    pub split: f32,
    pub preview: Option<Preview>,
    pub preview_error: Option<String>,
    /// The preview being computed, and the one wanted next.
    computing: Option<(PathBuf, Stage)>,
    /// Masters in memory, shared by previews and runs.
    loaded: Arc<Mutex<job::Loaded>>,
    pub running: Option<(usize, usize, String)>,
    pub cancel: Arc<AtomicBool>,
    pub results: HashMap<PathBuf, LightResult>,
    pub report: Option<Result<RunReport, String>>,
    pub solves: HashMap<PathBuf, Option<Solution>>,
    pub solving: Option<(usize, usize)>,
    pub solve_error: Option<String>,
    pub generation: u64,
    /// Integration reports by master path key (`crux-integration.json`,
    /// then live from a run).
    pub integrations: std::collections::BTreeMap<String, combine::Report>,
    /// Selected master (index into `Plan::stacks`).
    pub istack: usize,
    /// Integrate preview: 0 master vs reference, 1 rejected low, 2 high.
    pub imap: u8,
    /// A preview that failed, so it isn't retried every frame.
    failed_key: Option<(PathBuf, Stage)>,
    /// The image stages.
    pub img: crate::image::ImageUi,
    /// The mosaic stage.
    pub mos: crate::mosaic::MosaicUi,
    /// A run just finished: assemble the mosaic if Process includes it.
    pub mosaic_next: bool,
    /// A run just finished cleanly: open the first image step once the
    /// stacking steps (and any mosaic after them) are done.
    pub open_image_next: bool,
    /// What Process would do now (lights and masters out of date).
    pub pending: Option<job::Pending>,
}

impl Default for StackUi {
    fn default() -> Self {
        StackUi {
            stage: Stage::Calibrate,
            settings: None,
            plan: None,
            plan_error: None,
            plan_key: String::new(),
            sel: None,
            pinned_sel: false,
            split: 0.5,
            preview: None,
            preview_error: None,
            computing: None,
            loaded: Arc::new(Mutex::new(HashMap::new())),
            running: None,
            cancel: Arc::new(AtomicBool::new(false)),
            results: HashMap::new(),
            report: None,
            solves: HashMap::new(),
            solving: None,
            solve_error: None,
            generation: 0,
            integrations: Default::default(),
            istack: 0,
            imap: 0,
            failed_key: None,
            img: Default::default(),
            mos: Default::default(),
            mosaic_next: false,
            open_image_next: false,
            pending: None,
        }
    }
}

impl StackUi {
    pub fn reset(&mut self) {
        crate::image::cancel(&self.img);
        let loaded = self.loaded.clone();
        *self = StackUi::default();
        let _ = loaded.lock().map(|mut l| l.clear());
    }

    pub fn ready_for_shot(&self) -> bool {
        // A run (and the mosaic after it) ends first: `stack-run` shoots
        // what it leads to.
        if self.running.is_some() || self.mosaic_next || self.open_image_next {
            return false;
        }
        if self.stage == Stage::Mosaic {
            return self.mos.ready_for_shot() || self.plan_error.is_some();
        }
        if self.stage == Stage::Enhance {
            return self.img.enhance_ready() || self.plan_error.is_some();
        }
        if self.stage.is_image() {
            return self.img.ready_for_shot() || self.plan_error.is_some();
        }
        self.plan.as_ref().is_some_and(|p| p.n_lights() == 0) || self.preview.is_some() || self.preview_error.is_some() || self.plan_error.is_some()
    }

    /// Lights in plan order (field, then setup).
    pub fn lights(&self) -> Vec<&job::LightItem> {
        self.plan.as_ref().map(|p| p.fields.iter().flat_map(|f| f.lights.iter()).collect()).unwrap_or_default()
    }

    pub fn handle(&mut self, m: Msg, ctx: &egui::Context) -> Option<Msg> {
        let m = self.img.handle(m)?;
        let m = self.mos.handle(m, ctx)?;
        match m {
            Msg::Process(Event::Master { label }) => {
                if let Some(r) = &mut self.running {
                    r.2 = label;
                }
            }
            Msg::Process(Event::Light { done, total, path, result }) => {
                self.running = Some((done, total, path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default()));
                self.results.insert(path, result);
            }
            Msg::Process(Event::Integrate { index, total, label, step }) => {
                let (what, d, n) = match step {
                    combine::Step::Measure { done, total } => ("measuring", done, total),
                    combine::Step::Combine { done, total } => ("combining", done, total),
                };
                self.running = Some((index - 1, total, format!("Integrating {label}: {what} {d}/{n}")));
            }
            Msg::Process(Event::Integrated { index, total, label, report, error }) => {
                self.running = Some((index, total, format!("Integrated {label}")));
                if let Some(r) = report {
                    let k = crux_core::paths::key(&r.out);
                    self.integrations.insert(k, *r);
                }
                if let Some(e) = error {
                    self.preview_error = Some(format!("{label}: {e}"));
                }
                // The master may have changed under the preview.
                if self.stage == Stage::Integrate {
                    self.preview = None;
                    self.failed_key = None;
                }
            }
            Msg::ProcessDone(r) => {
                self.running = None;
                self.mosaic_next = matches!(&r, Ok(rep) if !rep.cancelled && rep.integration_failed == 0);
                self.open_image_next = matches!(&r, Ok(rep) if !rep.cancelled && rep.integration_failed == 0 && rep.failed == 0);
                self.pending = self.plan.as_ref().map(job::pending);
                self.report = Some(r);
                if self.stage == Stage::Integrate {
                    self.preview = None;
                    self.failed_key = None;
                }
            }
            Msg::Preview(p) => {
                let key = p.key.clone();
                self.computing = None;
                let opts = egui::TextureOptions::LINEAR;
                self.preview = Some(Preview {
                    key: key.clone(),
                    before: ctx.load_texture("stack-before", p.before, opts),
                    after: ctx.load_texture("stack-after", p.after, opts),
                    labels: p.labels,
                    info: p.info,
                    marks: p.marks,
                });
                self.preview_error = None;
            }
            Msg::PreviewFailed(key, e) => {
                self.computing = None;
                self.preview_error = Some(e);
                self.preview = None;
                self.failed_key = Some(key);
            }
            Msg::Solved { done, total, results } => {
                self.solving = if done < total { Some((done, total)) } else { None };
                self.solves.extend(results);
            }
            Msg::SolveFailed(e) => {
                self.solving = None;
                self.solve_error = Some(e);
            }
            other => return Some(other),
        }
        None
    }

    /// Run the plan in the background.
    pub fn start_run(&mut self, tx: Sender<Msg>, ctx: egui::Context) {
        let Some(plan) = self.plan.clone() else { return };
        self.cancel = Arc::new(AtomicBool::new(false));
        self.running = Some((0, plan.n_lights(), "Starting".into()));
        self.report = None;
        self.results.clear();
        let (loaded, cancel) = (self.loaded.clone(), self.cancel.clone());
        std::thread::Builder::new()
            .name("crux-process".into())
            .spawn(move || {
                let mut l = loaded.lock().unwrap_or_else(|e| e.into_inner());
                let txp = tx.clone();
                let ctxp = ctx.clone();
                let r = job::run_with(
                    &plan,
                    &mut l,
                    &|e| {
                        let _ = txp.send(Msg::Process(e));
                        ctxp.request_repaint();
                    },
                    &cancel,
                );
                let _ = tx.send(Msg::ProcessDone(r.map_err(|e| format!("{e:#}"))));
                ctx.request_repaint();
            })
            .expect("spawn process");
    }

    /// Plate-solve every planned light (from the index when solved before).
    pub fn start_solve(&mut self, index: Option<PathBuf>, s: &Session, tx: Sender<Msg>, ctx: egui::Context) {
        let items: Vec<(PathBuf, u64, i64, Option<crux_core::solve::Hint>, Vec<crux_core::stars::Star>, u32, u32)> = self
            .lights()
            .iter()
            .filter_map(|l| {
                let f = &s.triage.scan.frames[l.frame];
                let m = s.triage.metrics.get(&l.frame)?;
                Some((l.path.clone(), f.meta.file_size, f.meta.mtime_ns, crux_core::solve::Hint::from_meta(&f.meta), m.stars.clone(), m.width, m.height))
            })
            .collect();
        self.solving = Some((0, items.len()));
        self.solve_error = None;
        std::thread::Builder::new()
            .name("crux-solve".into())
            .spawn(move || {
                let Some(cat) = crux_core::catalog::Catalog::find() else {
                    let _ = tx.send(Msg::SolveFailed("No star database found: install ASTAP with its D50 database, or choose its folder in Settings (Add-ons).".into()));
                    ctx.request_repaint();
                    return;
                };
                let mut ix = index.as_ref().and_then(|p| crux_core::index::Index::load(p).ok());
                let total = items.len();
                let mut batch = Vec::new();
                // Learned: header focal lengths can be a few per cent out;
                // early failures are retried with the measured scale.
                let mut learned = crux_core::solve::Learned::default();
                let mut retry = Vec::new();
                for (k, (path, size, mtime, hint, stars, w, h)) in items.into_iter().enumerate() {
                    let known = ix.as_ref().and_then(|x| x.solution(&path, size, mtime)).map(|s| s.cloned());
                    let sol = match known {
                        Some(s) => {
                            if let (Some(hh), Some(x)) = (&hint, &s) {
                                learned.learn(hh, x);
                            }
                            s
                        }
                        None => {
                            let tried = hint.as_ref().is_some_and(|hh| learned.knows(hh));
                            let s = hint.as_ref().and_then(|hh| learned.solve(&cat, &stars, w as f64, h as f64, hh).ok().flatten());
                            if let Some(x) = ix.as_mut() {
                                x.set_solution(&path, size, mtime, s.clone());
                            }
                            if s.is_none() && !tried {
                                retry.push((path.clone(), size, mtime, hint.clone(), stars, w, h));
                            }
                            s
                        }
                    };
                    batch.push((path, sol));
                    if batch.len() >= 8 || k + 1 == total {
                        // Not finished until the retries are done.
                        let _ = tx.send(Msg::Solved { done: (k + 1).min(total.saturating_sub(1)), total, results: std::mem::take(&mut batch) });
                        ctx.request_repaint();
                    }
                }
                for (path, size, mtime, hint, stars, w, h) in retry {
                    let Some(hh) = hint.filter(|hh| learned.knows(hh)) else { continue };
                    let s = learned.solve(&cat, &stars, w as f64, h as f64, &hh).ok().flatten();
                    if s.is_some() {
                        if let Some(x) = ix.as_mut() {
                            x.set_solution(&path, size, mtime, s.clone());
                        }
                        batch.push((path, s));
                    }
                }
                let _ = tx.send(Msg::Solved { done: total, total, results: batch });
                ctx.request_repaint();
                if let Some(x) = &ix {
                    let _ = x.save();
                }
            })
            .expect("spawn solve");
    }
}

fn plan_key(s: &Session, st: &Settings) -> String {
    format!("{}|{}|{}", s.generation, serde_json::to_string(st).unwrap_or_default(), s.root.display())
}

/// Recompute the plan when grades, decisions or settings change.
fn refresh_plan(a: &mut UiState, s: &Session) {
    let st = a.stack.settings.get_or_insert_with(|| Settings::from_config(&s.cfg)).clone();
    // Fingerprints arrive later and can supply darks from other folders.
    let k = format!("{}|{}", plan_key(s, &st), a.cal.fp.is_some());
    if k == a.stack.plan_key {
        return;
    }
    a.stack.plan_key = k;
    let tp = std::time::Instant::now();
    let cat = crux_core::catalog::Catalog::find();
    let planned = job::plan(&s.triage, &s.root, Some(&s.decisions), &st, cat.as_ref(), a.cal.fp.as_ref());
    if std::env::var_os("CRUX_TIMING").is_some() {
        eprintln!("stack: plan in {:.2} s", tp.elapsed().as_secs_f64());
    }
    match planned {
        Ok(p) => {
            if a.stack.sel.as_ref().is_none_or(|sel| !p.fields.iter().any(|f| f.lights.iter().any(|l| &l.path == sel))) {
                a.stack.sel = p.fields.iter().find(|f| !f.lights.is_empty()).map(|f| f.reference_path.clone());
            }
            a.stack.integrations = job::load_integrations(&p.out);
            if a.stack.istack >= p.stacks.len() {
                a.stack.istack = 0;
            }
            let tp = std::time::Instant::now();
            a.stack.pending = Some(job::pending(&p));
            if std::env::var_os("CRUX_TIMING").is_some() {
                eprintln!("stack: pending in {:.2} s", tp.elapsed().as_secs_f64());
            }
            a.stack.plan = Some(p);
            a.stack.plan_error = None;
        }
        Err(e) => {
            a.stack.plan = None;
            a.stack.pending = None;
            a.stack.plan_error = Some(format!("{e:#}"));
        }
    }
    // The preview may show other settings now.
    a.stack.preview = None;
    a.stack.failed_key = None;
}

/// The Integrate preview: the master beside its reference frame at 1:1
/// (one stretch, so the noise drop shows), or a rejection map full frame.
fn want_integrate_preview(a: &mut UiState, tx: &Sender<Msg>, ctx: &egui::Context) {
    let Some(plan) = &a.stack.plan else { return };
    let Some(st) = plan.stacks.get(a.stack.istack).cloned() else { return };
    let mode = a.stack.imap;
    let key = (st.out.with_extension(format!("view{mode}")), Stage::Integrate);
    if a.stack.preview.as_ref().is_some_and(|p| p.key == key) || a.stack.computing.is_some() || a.stack.running.is_some() || a.stack.failed_key.as_ref() == Some(&key) {
        return;
    }
    let field = &plan.fields[st.field];
    let refpath = st.input(field, st.lights[st.reference]);
    let report = a.stack.integrations.get(&crux_core::paths::key(&st.out)).cloned();
    a.stack.computing = Some(key.clone());
    a.stack.preview_error = None;
    let (tx, ctx) = (tx.clone(), ctx.clone());
    std::thread::Builder::new()
        .name("crux-preview".into())
        .spawn(move || {
            let r = (|| -> anyhow::Result<PreviewImgs> {
                anyhow::ensure!(st.out.is_file(), "not integrated yet: run the stacking steps");
                let master = crux_core::read_image(&st.out)?;
                let n = report.as_ref().map(|r| r.frames.len()).unwrap_or(st.lights.len());
                let mut info = Vec::new();
                if let Some(r) = &report {
                    info.push(format!("{} frames, {:.2} h; noise {:.1} ADU against {:.1} for the median frame: SNR ×{:.2}", n, r.total_exposure / 3600.0, r.noise_master, r.noise_typical, r.snr_gain));
                }
                if mode == 0 {
                    let reference = crux_core::read_image(&refpath)?;
                    let stf = crux_core::stretch::Stf::auto(&master);
                    let (cw, ch) = (700.min(master.width), 466.min(master.height));
                    let (x0, y0) = busiest(&st.stars, master.width, master.height, cw, ch);
                    info.push("Same stretch on both sides: the master is in the reference frame's units.".into());
                    return Ok(PreviewImgs {
                        key: (st.out.with_extension(format!("view{mode}")), Stage::Integrate),
                        before: crop_at(&reference, &stf, x0, y0, cw, ch),
                        after: crop_at(&master, &stf, x0, y0, cw, ch),
                        labels: ("Reference frame".into(), format!("Master, {n} frames")),
                        info,
                        marks: Vec::new(),
                    });
                }
                let low = mode == 1;
                let path = report.as_ref().map(|r| if low { r.rejection_low.clone() } else { r.rejection_high.clone() }).unwrap_or_else(|| {
                    let stem = st.out.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
                    st.out.with_file_name(format!("{stem}_rejection_{}.xisf", if low { "low" } else { "high" }))
                });
                let map = crux_core::read_image(&path)?;
                let (bw, bh, f, b) = crux_core::stretch::preview(&master, 1400);
                // Block mean against the frame's mean: ESD and sigma clipping
                // reject a scatter of noise tails everywhere, so the map
                // shows where rejections are denser than that (trails,
                // planes, moving dust): the frame's average is dark grey,
                // four times it white.
                let mean = map.data.iter().map(|v| *v as f64).sum::<f64>() / map.data.len().max(1) as f64;
                let k = if mean > 0.0 { 1.0 / (4.0 * mean) } else { 1.0 };
                let mut px = vec![0u8; bw * bh];
                for y in 0..bh {
                    for x in 0..bw {
                        let mut sum = 0f64;
                        for dy in 0..f {
                            let row = &map.data[(y * f + dy) * map.width + x * f..];
                            sum += row[..f].iter().map(|v| *v as f64).sum::<f64>();
                        }
                        px[y * bw + x] = ((sum / (f * f) as f64 * k).min(1.0) * 255.0).round() as u8;
                    }
                }
                let total = map.data.iter().filter(|v| **v > 0.0).count();
                info.push(format!(
                    "{:.2}% of pixels had a value rejected {}; dark grey is the frame's average, white four times it",
                    total as f64 * 100.0 / map.data.len().max(1) as f64,
                    if low { "low (dark: cold pixels, dust motes that moved)" } else { "high (bright: satellites, planes, cosmic rays, hot pixels)" }
                ));
                Ok(PreviewImgs {
                    key: (st.out.with_extension(format!("view{mode}")), Stage::Integrate),
                    before: egui::ColorImage::from_gray([bw, bh], &b),
                    after: egui::ColorImage::from_gray([bw, bh], &px),
                    labels: ("Master".into(), if low { "Rejected low".into() } else { "Rejected high".into() }),
                    info,
                    marks: Vec::new(),
                })
            })();
            let _ = match r {
                Ok(p) => tx.send(Msg::Preview(Box::new(p))),
                Err(e) => tx.send(Msg::PreviewFailed(key, format!("{e:#}"))),
            };
            ctx.request_repaint();
        })
        .expect("spawn preview");
}

/// Compute the preview for the selection in the background.
fn want_preview(a: &mut UiState, s: &Session, tx: &Sender<Msg>, ctx: &egui::Context) {
    if a.stack.stage == Stage::Integrate {
        want_integrate_preview(a, tx, ctx);
        return;
    }
    let (Some(plan), Some(sel)) = (&a.stack.plan, &a.stack.sel) else { return };
    let stage = if a.stack.stage == Stage::Solve { Stage::Register } else { a.stack.stage };
    let key = (sel.clone(), stage);
    if a.stack.preview.as_ref().is_some_and(|p| p.key == key) || a.stack.computing.is_some() || a.stack.running.is_some() || a.stack.preview_error.is_some() && a.stack.preview.is_none() && a.stack.computing.is_none() && false {
        return;
    }
    let Some(item) = plan.fields.iter().flat_map(|f| f.lights.iter()).find(|l| &l.path == sel).cloned() else { return };
    let field = plan.fields.iter().find(|f| f.lights.iter().any(|l| l.path == item.path)).cloned();
    let ref_stars: Vec<(f32, f32)> = field.as_ref().and_then(|f| s.triage.metrics.get(&f.reference)).map(|m| m.stars.iter().map(|x| (x.x, x.y)).collect()).unwrap_or_default();
    let plan = plan.clone();
    a.stack.computing = Some(key.clone());
    a.stack.preview_error = None;
    let loaded = a.stack.loaded.clone();
    let (tx, ctx) = (tx.clone(), ctx.clone());
    std::thread::Builder::new()
        .name("crux-preview".into())
        .spawn(move || {
            let r = (|| -> anyhow::Result<PreviewImgs> {
                let tq = std::time::Instant::now();
                let timing = std::env::var_os("CRUX_TIMING").is_some();
                let mut l = loaded.lock().unwrap_or_else(|e| e.into_inner());
                let cal = job::calibrator(&plan, item.cal, plan.settings.cal.clone(), &mut l, &|_| {})?;
                if timing {
                    eprintln!("preview: calibrator at {:.2} s", tq.elapsed().as_secs_f64());
                }
                let [raw, plain, img, reg] = job::stages(&item, &cal, plan.settings.clamp)?;
                if timing {
                    eprintln!("preview: stages at {:.2} s", tq.elapsed().as_secs_f64());
                }
                let mut info = cal.notes.clone();
                match stage {
                    Stage::Calibrate | Stage::Solve | Stage::Integrate | Stage::Enhance | Stage::Gradient | Stage::Palette | Stage::Finish | Stage::Stars | Stage::Crop | Stage::Mosaic => {
                        let (bw, bh, _, b) = crux_core::stretch::preview(&raw, 1400);
                        let (aw, ah, _, af) = crux_core::stretch::preview(&img, 1400);
                        Ok(PreviewImgs {
                            key: (item.path.clone(), stage),
                            before: egui::ColorImage::from_gray([bw, bh], &b),
                            after: egui::ColorImage::from_gray([aw, ah], &af),
                            labels: ("Raw".into(), "Calibrated".into()),
                            info,
                            marks: Vec::new(),
                        })
                    }
                    Stage::Cosmetic => {
                        // 1:1 crop at the centre, one stretch for both sides.
                        let stf = crux_core::stretch::Stf::auto(&img);
                        info.retain(|n| n.contains("hot pixels"));
                        Ok(PreviewImgs {
                            key: (item.path.clone(), stage),
                            before: crop(&plain, &stf, 700, 466),
                            after: crop(&img, &stf, 700, 466),
                            labels: ("Before the hot-pixel fix".into(), "After".into()),
                            info,
                            marks: Vec::new(),
                        })
                    }
                    Stage::Register => {
                        // The reference beside this frame, registered: stars
                        // should run straight across the split.
                        let refimg = match field.as_ref().and_then(|f| f.lights.iter().find(|x| x.frame == f.reference)) {
                            Some(r) if r.path != item.path => {
                                let rc = job::calibrator(&plan, r.cal, plan.settings.cal.clone(), &mut l, &|_| {})?;
                                job::stages(r, &rc, plan.settings.clamp)?.into_iter().nth(3).unwrap()
                            }
                            _ => reg.clone(),
                        };
                        let stf = crux_core::stretch::Stf::auto(&refimg);
                        let stf2 = crux_core::stretch::Stf::auto(&reg);
                        info = vec![format!(
                            "{} stars matched, residual {:.3} px; shift ({:+.1}, {:+.1}) px, rotation {:+.3}°, scale {:.5}{}",
                            item.reg.matched,
                            item.reg.rms,
                            item.reg.shift.0,
                            item.reg.shift.1,
                            item.reg.rotation,
                            item.reg.scale,
                            if item.reg.flipped { ", meridian flip" } else { "" }
                        )];
                        // Reference stars inside the centre crop.
                        let (cw, ch) = (700.min(reg.width), 466.min(reg.height));
                        let (cx0, cy0) = busiest(&ref_stars, reg.width, reg.height, cw, ch);
                        let (x0, y0) = (cx0 as f32, cy0 as f32);
                        let marks = ref_stars.iter().map(|&(x, y)| ((x - x0 + 0.5) / cw as f32, (y - y0 + 0.5) / ch as f32)).filter(|&(u, v)| (0.0..=1.0).contains(&u) && (0.0..=1.0).contains(&v)).collect();
                        info.push("Circles: the reference's stars. On a registered frame each star sits in its circle.".into());
                        Ok(PreviewImgs {
                            key: (item.path.clone(), stage),
                            before: crop_at(&refimg, &stf, cx0, cy0, cw, ch),
                            after: crop_at(&reg, &stf2, cx0, cy0, cw, ch),
                            labels: ("Reference".into(), "This frame, registered".into()),
                            info,
                            marks,
                        })
                    }
                }
            })();
            let _ = match r {
                Ok(p) => tx.send(Msg::Preview(Box::new(p))),
                Err(e) => tx.send(Msg::PreviewFailed(key, format!("{e:#}"))),
            };
            ctx.request_repaint();
        })
        .expect("spawn preview");
}

/// Centre crop at 1:1 with a given stretch.
fn crop(img: &crux_core::Image, stf: &crux_core::stretch::Stf, w: usize, h: usize) -> egui::ColorImage {
    let (w, h) = (w.min(img.width), h.min(img.height));
    crop_at(img, stf, (img.width - w) / 2, (img.height - h) / 2, w, h)
}

/// The w×h window holding the most of `pts` (100 px steps).
fn busiest(pts: &[(f32, f32)], iw: usize, ih: usize, w: usize, h: usize) -> (usize, usize) {
    let mut best = ((iw.saturating_sub(w)) / 2, (ih.saturating_sub(h)) / 2, 0usize);
    let (mut y, step) = (0usize, 100usize);
    while y + h <= ih {
        let mut x = 0;
        while x + w <= iw {
            let n = pts.iter().filter(|p| p.0 >= x as f32 && p.0 < (x + w) as f32 && p.1 >= y as f32 && p.1 < (y + h) as f32).count();
            if n > best.2 {
                best = (x, y, n);
            }
            x += step;
        }
        y += step;
    }
    (best.0, best.1)
}

fn crop_at(img: &crux_core::Image, stf: &crux_core::stretch::Stf, x0: usize, y0: usize, w: usize, h: usize) -> egui::ColorImage {
    let mut px = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            px.push((stf.apply(img.data[(y0 + y) * img.width + x0 + x]) * 255.0).round() as u8);
        }
    }
    egui::ColorImage::from_gray([w, h], &px)
}

pub fn view(ui: &mut Ui, a: &mut UiState, s: &mut Session, tx: &Sender<Msg>) {
    let p = a.pal;
    if a.stack.generation != s.generation {
        a.stack.generation = s.generation;
    }
    refresh_plan(a, s);
    // The mosaic stage's state also feeds the stage list and the image
    // targets, so it ticks on every stage.
    crate::mosaic::tick(a, s, tx, ui.ctx());
    // Targets first, so a run that just made masters can open the image.
    crate::image::load(a, s);
    open_image_after_run(a, s, ui.ctx());
    if a.stack.stage.is_image() {
        crate::image::tick(a, s, tx, ui.ctx());
    } else {
        // Once stacking is done, prepare the image in the background so
        // the image steps are ready when the user opens them.
        if stacking_done(a) {
            crate::image::ensure_prepared(a, s, tx, ui.ctx());
        }
        if a.stack.stage != Stage::Mosaic {
            want_preview(a, s, tx, ui.ctx());
        }
    }
    // Bound by what's visible: a wide row must never widen the page.
    let vis = ui.available_width();
    // Image steps on a big enough window: the preview stays put while the
    // settings scroll under it, so a slider and its effect are both in view.
    let full = (vis - 14.0).min(1500.0);
    if a.stack.stage.is_image() && a.stack.plan_error.is_none() && full >= 900.0 && ui.available_height() >= 560.0 {
        pinned_image(ui, a, s, &p, tx, full);
        return;
    }
    ScrollArea::vertical().id_salt("stack").auto_shrink([false, false]).show(ui, |ui| {
        ui.set_max_width((vis - 14.0).min(1500.0));
        if a.stack.stage.is_image() {
            crate::image::header(ui, a, &p, tx);
        } else if a.stack.stage == Stage::Mosaic {
            crate::mosaic::header(ui, a, &p, tx);
        } else {
            header(ui, a, s, &p);
        }
        ui.add_space(14.0);
        let full = ui.available_width().min((vis - 14.0).min(1500.0));
        let two = full >= 900.0;
        let lw = if two { 280.0 } else { full };
        let gap = 16.0;
        if two {
            let top = ui.cursor().min;
            let mk = |x: f32, wd: f32| egui::UiBuilder::new().max_rect(Rect::from_min_size(Pos2::new(top.x + x, top.y), Vec2::new(wd, f32::INFINITY))).layout(egui::Layout::top_down(egui::Align::Min));
            let mut lu = ui.new_child(mk(0.0, lw));
            lu.set_width(lw);
            stages(&mut lu, a, s, &p, tx);
            let rw = full - lw - gap;
            let mut ru = ui.new_child(mk(lw + gap, rw));
            ru.set_width(rw);
            main(&mut ru, a, s, &p, tx);
            ui.advance_cursor_after_rect(lu.min_rect().union(ru.min_rect()));
        } else {
            stages(ui, a, s, &p, tx);
            main(ui, a, s, &p, tx);
        }
        ui.add_space(24.0);
    });
}

/// The image steps' layout: the header, then the step list in its own
/// scroll area beside the preview, which stays in place above its own
/// scrolling settings.
fn pinned_image(ui: &mut Ui, a: &mut UiState, s: &mut Session, p: &Pal, tx: &Sender<Msg>, full: f32) {
    ui.scope(|ui| {
        ui.set_max_width(full);
        crate::image::header(ui, a, p, tx);
    });
    ui.add_space(14.0);
    let top = ui.cursor().min;
    let h = (ui.available_height() - 6.0).max(200.0);
    let (lw, gap) = (280.0, 16.0);
    let col = |x: f32, wd: f32| egui::UiBuilder::new().max_rect(Rect::from_min_size(Pos2::new(top.x + x, top.y), Vec2::new(wd, h))).layout(egui::Layout::top_down(egui::Align::Min));
    // The step list, with room for its scroll bar.
    let mut lu = ui.new_child(col(0.0, lw + 10.0));
    ScrollArea::vertical().id_salt("stack-steps").auto_shrink([false, false]).show(&mut lu, |ui| {
        ui.set_max_width(lw);
        stages(ui, a, s, p, tx);
        ui.add_space(24.0);
    });
    let rw = full - lw - gap;
    let mut ru = ui.new_child(col(lw + gap, rw));
    ru.set_width(rw);
    crate::image::main_pinned(&mut ru, a, s, p, tx);
    ui.advance_cursor_after_rect(Rect::from_min_size(top, Vec2::new(full, h)));
}

/// Nothing of the stacking steps is running or waiting to run (a mosaic
/// assembled after integrating included).
fn stacking_idle(a: &UiState) -> bool {
    a.stack.running.is_none() && a.stack.mos.running.is_none() && !a.stack.mos.pending_run && !a.stack.mosaic_next
}

/// The stacking steps are done and up to date and masters exist, so the
/// image steps are what comes next.
pub fn stacking_done(a: &UiState) -> bool {
    let n = a.stack.plan.as_ref().map_or(0, |pl| pl.n_lights());
    n > 0 && matches!(a.stack.pending, Some(pd) if pd.is_empty()) && stacking_idle(a) && !a.stack.img.targets.is_empty()
}

/// After a clean run, once the stacking steps (and a mosaic after them)
/// have finished: open the first image step, unless the user went to a
/// step outside the run meanwhile or the mosaic failed.
fn open_image_after_run(a: &mut UiState, s: &Session, ctx: &egui::Context) {
    if !a.stack.open_image_next || !stacking_idle(a) {
        return;
    }
    a.stack.open_image_next = false;
    let on_run_step = process_steps(a, s).contains(&a.stack.stage);
    if stacking_done(a) && on_run_step && a.stack.mos.error.is_none() {
        a.stack.stage = Stage::Enhance;
        a.toast(ctx, "Stacking done. Check the image steps, then press Create full-size image.", false);
    }
}

/// The stages Process runs, in order.
pub fn process_steps(a: &UiState, s: &Session) -> Vec<Stage> {
    let Some(plan) = &a.stack.plan else { return Vec::new() };
    let mut v = vec![Stage::Calibrate];
    if plan.settings.cal.hot_sigma.is_some() {
        v.push(Stage::Cosmetic);
    }
    v.push(Stage::Register);
    if !plan.stacks.is_empty() {
        v.push(Stage::Integrate);
        if mosaic_follows(a, s) {
            v.push(Stage::Mosaic);
        }
    }
    v
}

/// Whether Process goes on to assemble a mosaic: "Mosaic after
/// integrating" is on and the plan integrates two or more panels of one
/// target.
fn mosaic_follows(a: &UiState, s: &Session) -> bool {
    let Some(plan) = &a.stack.plan else { return false };
    a.stack.mos.settings.as_ref().unwrap_or(&s.cfg.mosaic).after_integrate && has_panels(plan)
}

/// Two or more objects of one target ("LMC Panel 1", "LMC Panel 2").
fn has_panels(plan: &Plan) -> bool {
    let mut seen: HashMap<String, &str> = HashMap::new();
    plan.stacks.iter().any(|st| {
        let t = crux_core::index::target_of(&st.object);
        match seen.get(&t) {
            Some(o) => *o != st.object,
            None => {
                seen.insert(t, &st.object);
                false
            }
        }
    })
}

fn stage_name(st: Stage) -> &'static str {
    match st {
        Stage::Calibrate => "Calibrate",
        Stage::Cosmetic => "Cosmetic",
        Stage::Register => "Register",
        Stage::Solve => "Plate solve",
        Stage::Integrate => "Integrate",
        Stage::Mosaic => "Mosaic",
        Stage::Enhance => "Sharpen and denoise",
        Stage::Gradient => "Gradient",
        Stage::Palette => "Palette and stretch",
        Stage::Finish => "Finish",
        Stage::Stars => "Stars",
        Stage::Crop => "Crop and save",
    }
}

/// The Process button: what a click does now, and whether there's anything
/// to do.
fn process_label(n: usize, steps: usize, pending: Option<job::Pending>) -> (String, bool) {
    let s = |k: usize| if k == 1 { "" } else { "s" };
    match pending {
        Some(pd) if pd.is_empty() && n > 0 => ("Up to date".into(), false),
        Some(pd) if pd.lights == 0 && pd.stacks > 0 => (format!("Integrate {} master{}", pd.stacks, s(pd.stacks)), true),
        Some(pd) if pd.lights > 0 && pd.lights < n => (format!("Run steps 1–{steps} on {} of {n} lights", pd.lights), true),
        _ => (format!("Run steps 1–{steps} on {n} light{}", s(n)), n > 0),
    }
}

/// The pipeline strip's line for Stack.
pub fn strip_line(a: &UiState) -> String {
    let Some(plan) = &a.stack.plan else { return "calibrate, register".into() };
    match a.stack.pending {
        Some(pd) if pd.is_empty() && plan.n_lights() > 0 => "up to date".into(),
        Some(pd) if pd.lights == 0 && pd.stacks > 0 => format!("{} to integrate", pd.stacks),
        Some(pd) => format!("{} to process", pd.lights),
        None => format!("{} to process", plan.n_lights()),
    }
}

/// "A, B and C".
fn and_list(v: &[&str]) -> String {
    match v {
        [] => String::new(),
        [one] => one.to_string(),
        [rest @ .., last] => format!("{} and {last}", rest.join(", ")),
    }
}

fn header(ui: &mut Ui, a: &mut UiState, s: &Session, p: &Pal) {
    let full = ui.available_width();
    ui.horizontal(|ui| {
        ui.allocate_ui_with_layout(Vec2::new((full - 340.0).max(200.0), 0.0), egui::Layout::top_down(egui::Align::Min), |ui| {
            w::h1(ui, "Stack");
            let out = a.stack.plan.as_ref().map(|pl| pl.out.display().to_string()).unwrap_or_default();
            let k = process_steps(a, s).len();
            let mut t = format!(
                "Click a step on the left to see its settings and try them on the selected frame. Nothing is saved until you run steps 1 to {k}, which writes to {out}. Your data folder isn't changed."
            );
            let n = a.stack.plan.as_ref().map_or(0, |pl| pl.n_lights());
            match a.stack.pending {
                Some(pd) if pd.is_empty() && n > 0 => {
                    t.push_str(" Every light is processed and unchanged. Changing a setting, a grade or a master marks the lights it affects for processing again.");
                    if stacking_done(a) {
                        t.push_str(&format!(" Next, make the picture: run steps {}–{} under the image steps, or open any of them to adjust it first.", k + 1, k + IMAGE_STEPS.len()));
                    }
                }
                Some(pd) if pd.lights < n => t.push_str(&format!(" {} of {n} lights are already processed and unchanged, so the run skips them.", n - pd.lights)),
                _ => {}
            }
            ui.add(egui::Label::new(RichText::new(t).color(p.muted)).wrap());
        });
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if let Some(out) = a.stack.plan.as_ref().map(|pl| pl.out.clone()) {
                if a.stack.running.is_none() && a.stack.mos.running.is_none() && a.stack.report.is_some() && w::button(ui, p, "Open output folder", Kind::Ghost, None).clicked() {
                    let _ = open_folder(&out);
                }
            }
        });
    });
    let steps = process_steps(a, s);
    let mosaic = mosaic_step_running(a, &steps);
    match &a.stack.report {
        Some(Ok(r)) => {
            ui.add_space(6.0);
            let done = if mosaic.is_some() { format!("Steps 1–{} done", steps.len() - 1) } else { "Done".to_string() };
            let mut t = format!("{done} in {:.0} s: {} processed, {} already up to date", r.seconds, r.processed, r.up_to_date);
            if r.failed > 0 {
                t.push_str(&format!(", {} failed", r.failed));
            }
            if !r.masters_built.is_empty() {
                t.push_str(&format!("; built {} master{}", r.masters_built.len(), if r.masters_built.len() == 1 { "" } else { "s" }));
            }
            if r.integrated + r.integrations_up_to_date > 0 {
                t.push_str(&format!("; {} master light{} integrated, {} up to date", r.integrated, if r.integrated == 1 { "" } else { "s" }, r.integrations_up_to_date));
            }
            if r.integration_failed > 0 {
                t.push_str(&format!(", {} failed", r.integration_failed));
            }
            if r.cancelled {
                t.push_str(" (cancelled)");
            }
            let mosaic_failed = steps.contains(&Stage::Mosaic) && a.stack.mos.error.is_some();
            if let Some(m) = &mosaic {
                t.push_str(&format!(". Now assembling the mosaic: {m}"));
            } else if mosaic_failed {
                t.push_str(&format!(". The mosaic failed: see step {}", steps.len()));
            }
            let warn = r.failed > 0 || mosaic.is_some() || mosaic_failed;
            ui.label(RichText::new(t).color(if warn { p.warn } else { p.good }));
        }
        Some(Err(e)) => {
            ui.label(RichText::new(format!("Processing stopped: {e}")).color(p.rej));
        }
        None => {}
    }
}

pub fn open_folder(p: &std::path::Path) -> std::io::Result<()> {
    #[cfg(windows)]
    {
        std::process::Command::new("explorer").arg(p).spawn().map(|_| ())
    }
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open").arg(p).spawn().map(|_| ())
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        std::process::Command::new("xdg-open").arg(p).spawn().map(|_| ())
    }
}

/// Open a web page in the default browser.
pub fn open_url(url: &str) -> std::io::Result<()> {
    #[cfg(windows)]
    {
        std::process::Command::new("explorer").arg(url).spawn().map(|_| ())
    }
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open").arg(url).spawn().map(|_| ())
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        std::process::Command::new("xdg-open").arg(url).spawn().map(|_| ())
    }
}

/// Open the file manager on a file's folder with the file selected.
pub fn show_in_folder(p: &std::path::Path) -> std::io::Result<()> {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // Explorer wants /select,"path" with the quotes around the path only.
        std::process::Command::new("explorer").raw_arg(format!("/select,\"{}\"", p.display())).spawn().map(|_| ())
    }
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open").arg("-R").arg(p).spawn().map(|_| ())
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        open_folder(p.parent().unwrap_or(p))
    }
}

fn median(mut v: Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    Some(v[v.len() / 2])
}

fn stage_summary(a: &UiState, st: Stage) -> (Color32, String) {
    let p = &a.pal;
    if st.is_image() {
        return crate::image::summary(a, st);
    }
    if st == Stage::Mosaic {
        return crate::mosaic::summary(a, p);
    }
    let Some(plan) = &a.stack.plan else { return (p.faint, a.stack.plan_error.clone().unwrap_or_else(|| "No plan".into())) };
    let done = (!a.stack.results.is_empty() && a.stack.results.values().all(|r| r.error.is_none())) || a.stack.pending.is_some_and(|pd| pd.lights == 0 && plan.n_lights() > 0);
    // Some lights processed, some not (new lights, or a setting changed).
    let partial = !done && a.stack.pending.is_some_and(|pd| pd.lights < plan.n_lights());
    let dot = if a.stack.running.is_some() || partial { p.warn } else if done { p.good } else { p.faint };
    match st {
        Stage::Calibrate => {
            // Count distinct master flats, not filter-nights (several nights
            // can share one).
            let (mut built, mut masters, mut none) = (std::collections::HashSet::new(), std::collections::HashSet::new(), 0);
            for c in &plan.cals {
                match &c.flat {
                    Some(Source::Build { out, .. }) => {
                        built.insert(crux_core::paths::key(out));
                    }
                    Some(Source::File { path }) => {
                        masters.insert(crux_core::paths::key(path));
                    }
                    None => none += 1,
                }
            }
            let pl = |k: usize| if k == 1 { "" } else { "s" };
            let mut uses = Vec::new();
            match plan.cals.iter().find_map(|c| c.dark.as_ref()) {
                Some(Source::File { .. }) => uses.push("the existing master dark".to_string()),
                Some(Source::Build { .. }) => uses.push("a dark built from raw frames".to_string()),
                None => {}
            }
            if !built.is_empty() {
                uses.push(format!("{} flat{} built from raw frames", built.len(), pl(built.len())));
            }
            if !masters.is_empty() {
                uses.push(format!("{} existing master flat{}", masters.len(), pl(masters.len())));
            }
            let mut t = "Subtract the dark, divide by the flat.".to_string();
            if !uses.is_empty() {
                t.push_str(&format!(" Uses {}.", and_list(&uses.iter().map(|s| s.as_str()).collect::<Vec<_>>())));
            }
            if plan.cals.iter().all(|c| c.dark.is_none()) {
                t.push_str(" No dark.");
            }
            if none > 0 {
                t.push_str(&format!(" No flats for {none} filter-night{}.", pl(none)));
            }
            (dot, t)
        }
        Stage::Cosmetic => {
            let sig = plan.settings.cal.hot_sigma;
            let fixed = median(a.stack.results.values().filter(|r| !r.skipped_up_to_date).map(|r| r.cal.hot_fixed as f64).collect());
            match (sig, fixed) {
                (None, _) => (p.faint, "Off".into()),
                (Some(_), Some(n)) => (dot, format!("Replaced {n:.0} hot pixels per frame with their neighbours")),
                (Some(_), None) => (dot, "Replace hot pixels with the median of their neighbours".into()),
            }
        }
        Stage::Register => {
            let n = plan.n_lights();
            let rms = median(plan.fields.iter().flat_map(|f| f.lights.iter().filter(move |l| l.frame != f.reference).map(|l| l.reg.rms)).collect());
            let r = plan.fields.iter().find(|f| !f.lights.is_empty()).map(|f| f.reference_path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default()).unwrap_or_default();
            let fields = plan.fields.iter().filter(|f| !f.lights.is_empty()).count();
            let to = if fields > 1 { format!(", each of {fields} panels to its own reference frame") } else { format!(" to the {} frame", short_name(&r)) };
            (dot, format!("Align {n} frames{to}{}", rms.map(|x| format!(". Stars match to {x:.2} px")).unwrap_or_default()))
        }
        Stage::Integrate => {
            if plan.stacks.is_empty() {
                return (p.faint, "Off".into());
            }
            let reps: Vec<&combine::Report> = plan.stacks.iter().filter_map(|s| a.stack.integrations.get(&crux_core::paths::key(&s.out))).collect();
            let k = plan.stacks.len();
            if reps.is_empty() {
                let objects: std::collections::HashSet<&str> = plan.stacks.iter().map(|s| s.object.as_str()).collect();
                let per = if objects.len() > 1 { "one per panel and filter" } else { "one per filter" };
                return (p.faint, format!("Stack the aligned frames into {k} master light{} ({per})", if k == 1 { "" } else { "s" }));
            }
            let gain = median(reps.iter().map(|r| r.snr_gain).collect()).unwrap_or(0.0);
            // Masters from an earlier run aren't done if their frames or
            // settings have changed since (`job::pending` counts those).
            let stale = a.stack.pending.map_or(0, |pd| pd.stacks).min(reps.len());
            let current = reps.len() - stale;
            let dot = if current == k { p.good } else { p.warn };
            let t = if stale == 0 {
                format!("{current} of {k} master lights stacked. Signal-to-noise ×{gain:.1} over a single frame")
            } else {
                format!(
                    "{stale} of {k} master lights are from an earlier run and will be stacked again. Signal-to-noise was ×{gain:.1} over a single frame"
                )
            };
            (dot, t)
        }
        Stage::Enhance | Stage::Gradient | Stage::Palette | Stage::Finish | Stage::Stars | Stage::Crop | Stage::Mosaic => (p.faint, String::new()),
        Stage::Solve => {
            let n = a.stack.solves.len();
            let ok = a.stack.solves.values().filter(|s| s.is_some()).count();
            if let Some((d, t)) = a.stack.solving {
                (p.warn, format!("Solving {d} of {t}"))
            } else if n == 0 {
                (p.faint, "Find where each frame points on the sky. Not needed for stacking. Not run yet".into())
            } else {
                (if ok == n { p.good } else { p.warn }, format!("{ok} of {n} solved"))
            }
        }
    }
}

/// "2026-10-06 21:45 S" from a N.I.N.A. file name.
fn short_name(n: &str) -> String {
    let parts: Vec<&str> = n.split('_').collect();
    if parts.len() >= 3 && parts[0].len() == 10 {
        format!("{} {} {}", parts[0], parts[1].get(..5).unwrap_or(parts[1]).replace('-', ":"), parts[2])
    } else {
        n.to_string()
    }
}

/// The stage list in three groups: what Process runs (numbered in the
/// order it runs them; Cosmetic stays in the group, unnumbered, when it's
/// off), what runs from its own card, and the image stages.
fn stages(ui: &mut Ui, a: &mut UiState, s: &Session, p: &Pal, tx: &Sender<Msg>) {
    ui.spacing_mut().item_spacing.y = 6.0;
    let steps = process_steps(a, s);
    let k = steps.len();
    let card = |ui: &mut Ui, a: &mut UiState, st: Stage| {
        // Image steps carry on from the stacking steps' numbers.
        let n = steps.iter().position(|&x| x == st).or_else(|| IMAGE_STEPS.iter().position(|&x| x == st).map(|i| k + i));
        let name = match n {
            Some(i) => format!("{}. {}", i + 1, stage_name(st)),
            None => stage_name(st).to_string(),
        };
        let (dot, mut sum) = stage_summary(a, st);
        // Steps the run button covers: say whether they've run.
        if steps.contains(&st) && a.stack.running.is_none() && sum != "Off" {
            let n = a.stack.plan.as_ref().map_or(0, |pl| pl.n_lights());
            match a.stack.pending {
                Some(pd) if matches!(st, Stage::Calibrate | Stage::Cosmetic | Stage::Register) && pd.lights > 0 && pd.lights < n => sum.push_str(&format!("\nDone for {} of {n} lights", n - pd.lights)),
                _ if dot == a.pal.faint => sum.push_str("\nNot run yet"),
                _ => {}
            }
        }
        // The first image step, while no image has been made: mark it.
        let img = &a.stack.img;
        if st == Stage::Enhance && !a.stack.stage.is_image() && stacking_done(a) && !matches!(img.written, Some(Ok(_))) {
            sum.push_str("\nStart here to make the image");
        }
        if stage_button(ui, p, &name, &sum, dot, a.stack.stage == st, true).clicked() {
            a.stack.stage = st;
        }
    };
    w::eyebrow(ui, p, "Stacking steps");
    let mut first = vec![Stage::Calibrate, Stage::Cosmetic, Stage::Register, Stage::Integrate];
    if steps.contains(&Stage::Mosaic) {
        first.push(Stage::Mosaic);
    }
    for st in first {
        card(ui, a, st);
    }
    run_button(ui, a, p, &steps, tx);
    ui.add_space(8.0);
    w::eyebrow(ui, p, "Optional");
    card(ui, a, Stage::Solve);
    if !steps.contains(&Stage::Mosaic) {
        card(ui, a, Stage::Mosaic);
    }
    ui.add_space(8.0);
    w::eyebrow(ui, p, "Image steps");
    for st in IMAGE_STEPS {
        card(ui, a, st);
    }
    let ready = stacking_done(a);
    crate::image::run_button(ui, a, p, k + 1, k + IMAGE_STEPS.len(), ready, tx);
}

/// The image steps in the order the full-size image applies them.
const IMAGE_STEPS: [Stage; 6] = [Stage::Enhance, Stage::Gradient, Stage::Palette, Stage::Finish, Stage::Stars, Stage::Crop];

/// The run button under the stacking steps, or Cancel and progress while
/// they run.
fn run_button(ui: &mut Ui, a: &mut UiState, p: &Pal, steps: &[Stage], tx: &Sender<Msg>) {
    let k = steps.len();
    ui.add_space(2.0);
    if let Some((done, total, what)) = &a.stack.running {
        let f = if *total > 0 { *done as f32 / *total as f32 } else { 0.0 };
        ui.add(egui::ProgressBar::new(f).desired_height(6.0).fill(p.accent));
        ui.add(egui::Label::new(RichText::new(format!("{done} of {total} · {what}")).font(body(12.0)).color(p.muted)).wrap());
        if w::button(ui, p, "Cancel", Kind::Danger, None).clicked() {
            a.stack.cancel.store(true, Ordering::Relaxed);
        }
    } else if let Some(r) = mosaic_step_running(a, steps) {
        // The run's last step: the mosaic, assembled after integrating.
        ui.horizontal(|ui| {
            ui.add(egui::Spinner::new().size(12.0).color(p.accent));
            ui.add(egui::Label::new(RichText::new(format!("Step {k} of {k} · Mosaic · {r}")).font(body(12.0)).color(p.muted)).wrap());
        });
        if w::button(ui, p, "Cancel", Kind::Danger, None).clicked() {
            a.stack.mos.cancel.store(true, Ordering::Relaxed);
        }
    } else if stacking_done(a) {
        // Nothing left to stack; the image steps' button is what's next.
        // A setting, grade or master that changes brings the button back.
        ui.label(RichText::new(format!("Steps 1–{k} are up to date")).font(body(12.0)).color(p.muted));
    } else if let Some(n) = a.stack.plan.as_ref().map(|pl| pl.n_lights()) {
        let (label, enabled) = process_label(n, k, a.stack.pending);
        if w::button_enabled(ui, p, &label, Kind::Primary, None, enabled).clicked() {
            a.stack.start_run(tx.clone(), ui.ctx().clone());
        }
    }
}

/// What the mosaic is doing, when it's running and is one of the run's
/// steps (it starts once integration ends, as a job of its own).
fn mosaic_step_running(a: &UiState, steps: &[Stage]) -> Option<String> {
    if steps.contains(&Stage::Mosaic) { a.stack.mos.running.clone() } else { None }
}

/// One entry in the step list. Clicking it shows that step on the right
/// (it never runs anything), so it's drawn as a tab: the selected one is
/// filled with an accent bar on its left edge.
fn stage_button(ui: &mut Ui, p: &Pal, name: &str, sub: &str, dot: Color32, on: bool, enabled: bool) -> egui::Response {
    let wd = ui.available_width();
    let sub_g = ui.painter().layout(sub.to_string(), body(12.0), p.muted, wd - 40.0);
    let h = 22.0 + sub_g.size().y + 16.0;
    let (r, resp) = ui.allocate_exact_size(Vec2::new(wd, h), if enabled { Sense::click() } else { Sense::hover() });
    let pa = ui.painter();
    let fill = if on || (resp.hovered() && enabled) { p.panel2 } else { p.panel };
    pa.rect(r, 10.0, fill, Stroke::new(1.0, if on { p.accent } else { p.line }), StrokeKind::Inside);
    if on {
        pa.rect_filled(Rect::from_min_size(r.min + Vec2::new(1.0, 10.0), Vec2::new(3.0, r.height() - 20.0)), 1.5, p.accent);
    }
    pa.circle_filled(Pos2::new(r.left() + 16.0, r.top() + 18.0), 4.5, dot);
    pa.text(Pos2::new(r.left() + 30.0, r.top() + 10.0), egui::Align2::LEFT_TOP, name, semibold(14.0), if enabled { p.fg } else { p.faint });
    pa.galley(Pos2::new(r.left() + 30.0, r.top() + 30.0), sub_g, p.muted);
    resp
}

fn main(ui: &mut Ui, a: &mut UiState, s: &mut Session, p: &Pal, tx: &Sender<Msg>) {
    ui.spacing_mut().item_spacing.y = 14.0;
    if let Some(e) = &a.stack.plan_error {
        w::card(ui, p, |ui| {
            ui.label(RichText::new(e).color(p.rej));
        });
        return;
    }
    if a.stack.stage.is_image() {
        crate::image::main(ui, a, s, p, tx);
        return;
    }
    if a.stack.stage == Stage::Mosaic {
        crate::mosaic::main(ui, a, p);
        return;
    }
    split_view(ui, a, p);
    w::card(ui, p, |ui| settings_card(ui, a, s, p, tx));
    w::card(ui, p, |ui| frame_table(ui, a, s, p));
}

fn split_view(ui: &mut Ui, a: &mut UiState, p: &Pal) {
    let wd = ui.available_width();
    let Some(pv) = &a.stack.preview else {
        let h = (wd * 0.55).min(ui.ctx().content_rect().height() * 0.58);
        let (r, _) = ui.allocate_exact_size(Vec2::new(wd, h), Sense::hover());
        ui.painter().rect_filled(r, 10.0, p.well);
        let msg = if let Some(e) = &a.stack.preview_error {
            format!("Couldn't make the preview: {e}")
        } else if a.stack.running.is_some() {
            "Preview paused while processing".into()
        } else if a.stack.computing.is_some() {
            "Calibrating the selected frame…".into()
        } else {
            "Select a frame".into()
        };
        ui.painter().text(r.center(), egui::Align2::CENTER_CENTER, msg, body(13.0), p.muted);
        return;
    };
    let [tw, th] = pv.after.size();
    // Fit within the width and about 60% of the window height.
    let max_h = (ui.ctx().content_rect().height() * 0.58).max(280.0);
    let (iw, ih) = if wd * th as f32 / tw as f32 > max_h { (max_h * tw as f32 / th as f32, max_h) } else { (wd, wd * th as f32 / tw as f32) };
    let (outer, _) = ui.allocate_exact_size(Vec2::new(wd, ih), Sense::hover());
    ui.painter().rect_filled(outer, 10.0, p.well);
    let r = Rect::from_center_size(outer.center(), Vec2::new(iw, ih));
    let resp = ui.interact(r, ui.scope_id().with("split"), Sense::click_and_drag());
    if let Some(pos) = resp.interact_pointer_pos() {
        a.stack.split = ((pos.x - r.left()) / r.width()).clamp(0.0, 1.0);
    }
    let pa = ui.painter_at(r);
    let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
    pa.image(pv.after.id(), r, uv, Color32::WHITE);
    let sx = r.left() + r.width() * a.stack.split;
    let left = Rect::from_min_max(r.min, Pos2::new(sx, r.bottom()));
    pa.with_clip_rect(left).image(pv.before.id(), r, uv, Color32::WHITE);
    pa.line_segment([Pos2::new(sx, r.top()), Pos2::new(sx, r.bottom())], Stroke::new(2.0, Color32::WHITE));
    pa.circle_filled(Pos2::new(sx, r.center().y), 13.0, Color32::WHITE);
    pa.text(Pos2::new(sx, r.center().y), egui::Align2::CENTER_CENTER, "⟷", body(13.0), Color32::BLACK);
    let lab = |pos: Pos2, align: egui::Align2, t: &str| {
        let g = pa.layout_no_wrap(t.to_string(), mono(11.0), Color32::WHITE);
        let rr = align.anchor_size(pos, g.size() + Vec2::new(12.0, 4.0));
        pa.rect_filled(rr, 4.0, Color32::from_black_alpha(140));
        pa.galley(rr.min + Vec2::new(6.0, 2.0), g, Color32::WHITE);
    };
    lab(r.left_top() + Vec2::new(8.0, 8.0), egui::Align2::LEFT_TOP, &pv.labels.0);
    lab(r.right_top() + Vec2::new(-8.0, 8.0), egui::Align2::RIGHT_TOP, &pv.labels.1);
    for &(u, v) in &pv.marks {
        let c = Pos2::new(r.left() + u * r.width(), r.top() + v * r.height());
        pa.circle_stroke(c, 7.0, Stroke::new(1.2, Color32::from_rgb(0x5d, 0xd3, 0x9e)));
    }
    match pv.key.1 {
        Stage::Cosmetic => lab(r.left_bottom() + Vec2::new(8.0, -8.0), egui::Align2::LEFT_BOTTOM, "1:1, frame centre"),
        Stage::Register => lab(r.left_bottom() + Vec2::new(8.0, -8.0), egui::Align2::LEFT_BOTTOM, "1:1, where the reference has most stars"),
        Stage::Integrate if a.stack.imap == 0 => lab(r.left_bottom() + Vec2::new(8.0, -8.0), egui::Align2::LEFT_BOTTOM, "1:1, where the reference has most stars"),
        _ => {}
    }
    ui.scope(|ui| {
        ui.spacing_mut().item_spacing.y = 2.0;
        for line in &pv.info {
            ui.label(RichText::new(line).font(body(12.0)).color(p.muted));
        }
    });
}

fn settings_card(ui: &mut Ui, a: &mut UiState, s: &mut Session, p: &Pal, tx: &Sender<Msg>) {
    let Some(st) = a.stack.settings.clone() else { return };
    let mut st2 = st.clone();
    let base = Settings::from_config(&s.cfg);
    match a.stack.stage {
        Stage::Calibrate => {
            w::h2(ui, "Calibrate");
            w::muted(ui, p, "Subtracts the dark from each light and divides by the flat scaled to its mean, as WBPP's ImageCalibration does. crux uses WBPP's masters when they pass its checks, and restacks flats from the raw frames without the ones it rejects.");
            if let Some(plan) = &a.stack.plan {
                for c in &plan.cals {
                    let flat = c.flat.as_ref().map(|f| f.label()).unwrap_or_else(|| "no flats".into());
                    let notes = if c.notes.is_empty() { String::new() } else { format!(" — {}", c.notes.join("; ")) };
                    ui.add(egui::Label::new(RichText::new(format!("{}: {flat}{notes}", c.label)).font(body(12.0)).color(p.muted)).wrap());
                }
            }
            ui.horizontal(|ui| {
                let mut ped = st2.cal.pedestal_limit.is_some();
                ui.checkbox(&mut ped, "Output pedestal when needed");
                st2.cal.pedestal_limit = ped.then_some(0.0001);
                w::reset(ui, p, &mut st2.cal.pedestal_limit, &base.cal.pedestal_limit, "default");
            });
            ui.horizontal(|ui| {
                ui.checkbox(&mut st2.include_warn, "Include frames graded warn");
                w::reset(ui, p, &mut st2.include_warn, &base.include_warn, "default");
                ui.add_space(16.0);
                ui.checkbox(&mut st2.keep_calibrated, "Also keep calibrated frames");
                w::reset(ui, p, &mut st2.keep_calibrated, &base.keep_calibrated, "default");
            });
        }
        Stage::Cosmetic => {
            w::h2(ui, "Cosmetic correction");
            w::muted(ui, p, "Hot pixels are found in the master dark (more than k clipped standard deviations above its median) and replaced with the median of their eight neighbours, the same rule as ImageCalibration.");
            ui.horizontal(|ui| {
                let mut on = st2.cal.hot_sigma.is_some();
                ui.checkbox(&mut on, "Fix hot pixels");
                let mut k = st2.cal.hot_sigma.unwrap_or(10.0);
                ui.add_enabled(on, egui::Slider::new(&mut k, 5.0..=20.0).step_by(0.5).text("σ"));
                st2.cal.hot_sigma = on.then_some(k);
                w::reset(ui, p, &mut st2.cal.hot_sigma, &base.cal.hot_sigma, "default");
            });
        }
        Stage::Register | Stage::Solve if a.stack.stage == Stage::Register => {
            w::h2(ui, "Register");
            w::muted(ui, p, "crux fits a homography to the stars it measured during triage, without reading the pixels again, then resamples with Lanczos-3. Each field uses one reference for every filter, so the H, O and S stacks line up.");
            if let Some(plan) = &a.stack.plan {
                for f in plan.fields.iter().filter(|f| !f.lights.is_empty()) {
                    // Reference candidates: the five best scores.
                    let mut cands: Vec<(f64, String)> = f
                        .lights
                        .iter()
                        .filter_map(|l| {
                            let m = s.triage.metrics.get(&l.frame)?;
                            let t = s.triage.report_for(l.frame).and_then(|r| r.transparency.as_ref()).map(|x| x.value);
                            Some((crux_core::process::register::reference_score(m.n_stars, m.fwhm, m.ecc, t), l.path.file_name()?.to_string_lossy().to_string()))
                        })
                        .collect();
                    cands.sort_by(|a, b| b.0.total_cmp(&a.0));
                    let cur = f.reference_path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(format!("{} reference", f.object)).font(body(13.0)));
                        egui::ComboBox::from_id_salt(("ref", &f.object)).selected_text(short_name(&cur)).width(260.0).show_ui(ui, |ui| {
                            for (sc, n) in cands.iter().take(6) {
                                if ui.selectable_label(n == &cur, format!("{}  ·  score {sc:.0}", short_name(n))).clicked() {
                                    st2.reference = Some(n.trim_end_matches(".fits").trim_end_matches(".xisf").to_string());
                                }
                            }
                        });
                        w::reset(ui, p, &mut st2.reference, &None, "default");
                        if st2.reference.is_some() && ui.small_button("Back to crux's choice").clicked() {
                            st2.reference = None;
                        }
                    });
                }
            }
            ui.horizontal(|ui| {
                let mut on = st2.clamp.is_some();
                ui.checkbox(&mut on, "Lanczos clamping");
                let mut c = st2.clamp.unwrap_or(0.3);
                ui.add_enabled(on, egui::Slider::new(&mut c, 0.05..=1.0).text("threshold"));
                st2.clamp = on.then_some(c);
                w::reset(ui, p, &mut st2.clamp, &base.clamp, "default");
            });
        }
        Stage::Integrate => integrate_card(ui, a, p, &mut st2, &base),
        _ => {
            w::h2(ui, "Plate solve");
            w::muted(ui, p, "Solves each light by matching its measured stars to ASTAP's D50 star database, starting from the header's position and pixel scale. Solutions are saved in the library index.");
            w::muted(ui, p, "Optional. Stacking registers the lights to each other, and the mosaic solves its integrated masters itself, so neither needs this. Solving the lights finds frames pointed at the wrong place and gives the library each target's position.");
            ui.horizontal(|ui| {
                let busy = a.stack.solving.is_some();
                if w::button_enabled(ui, p, if a.stack.solves.is_empty() { "Solve all" } else { "Solve again" }, Kind::Primary, None, !busy).clicked() {
                    let ix = a.index_path.clone();
                    a.stack.start_solve(ix, s, tx.clone(), ui.ctx().clone());
                }
                if let Some((d, t)) = a.stack.solving {
                    ui.add(egui::ProgressBar::new(d as f32 / t.max(1) as f32).desired_width(200.0).fill(p.accent));
                }
            });
            if let Some(e) = &a.stack.solve_error {
                ui.label(RichText::new(e).color(p.rej));
            }
            solve_summary(ui, a, s, p);
        }
    }
    if st2 != st {
        a.stack.settings = Some(st2);
    }
}

/// A selectable chip whose selected text stays readable on the accent
/// (the theme's override text colour would make it light on light).
fn chip<T: PartialEq>(ui: &mut Ui, p: &Pal, cur: &mut T, v: T, text: &str) {
    let on = *cur == v;
    if ui.selectable_label(on, RichText::new(text).color(if on { p.accent_ink } else { p.fg })).clicked() {
        *cur = v;
    }
}

fn capital(s: &str) -> String {
    let mut c = s.chars();
    c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default()
}

fn integrate_card(ui: &mut Ui, a: &mut UiState, p: &Pal, st2: &mut Settings, base: &Settings) {
    w::h2(ui, "Integrate");
    w::muted(
        ui,
        p,
        "Builds one master per field and filter. Each frame is scaled by its stars' brightness against the reference, which allows for thin cloud, airmass and exposure, and its background is matched to the reference's. Local normalisation fits a smooth surface, so sky gradients that change through the night don't leave patches. Outliers are then rejected pixel by pixel, and the rest are averaged, weighted by signal-to-noise squared.",
    );
    ui.horizontal(|ui| {
        ui.checkbox(&mut st2.integrate, "Integrate master lights");
        w::reset(ui, p, &mut st2.integrate, &base.integrate, "default");
    });
    let c = &mut st2.combine;
    ui.horizontal(|ui| {
        ui.label(RichText::new("Normalisation").font(body(13.0)));
        chip(ui, p, &mut c.normalisation, combine::Normalisation::Local, "Local");
        chip(ui, p, &mut c.normalisation, combine::Normalisation::Global, "Global");
        w::reset(ui, p, &mut c.normalisation, &base.combine.normalisation, "default");
    });
    ui.horizontal(|ui| {
        ui.label(RichText::new("Weights").font(body(13.0)));
        chip(ui, p, &mut c.weighting, combine::Weighting::Snr, "SNR");
        chip(ui, p, &mut c.weighting, combine::Weighting::Exposure, "Exposure");
        chip(ui, p, &mut c.weighting, combine::Weighting::Equal, "Equal");
        w::reset(ui, p, &mut c.weighting, &base.combine.weighting, "default");
    });
    use crux_core::process::integrate::Rejection;
    let choices: [(Option<Rejection>, &str); 7] = [
        (None, "Auto (by stack size, like WBPP)"),
        (Some(Rejection::PercentileNoise { low: 0.2, high: 0.1, sigma_low: 4.0, sigma_high: 3.0 }), "Percentile above the noise"),
        (Some(Rejection::Esd { fraction: 0.3, alpha: 0.05 }), "ESD"),
        (Some(Rejection::Winsorized { low: 4.0, high: 3.0 }), "Winsorised sigma 4/3"),
        (Some(Rejection::Sigma { low: 4.0, high: 3.0 }), "Sigma 4/3"),
        (Some(Rejection::Percentile { low: 0.2, high: 0.1 }), "Percentile 20%/10%"),
        (Some(Rejection::None), "None"),
    ];
    ui.horizontal(|ui| {
        ui.label(RichText::new("Rejection").font(body(13.0)));
        let cur = choices.iter().find(|x| x.0 == c.rejection).map(|x| x.1).unwrap_or("Custom");
        egui::ComboBox::from_id_salt("int-rej").selected_text(cur).width(240.0).show_ui(ui, |ui| {
            for (r, n) in choices {
                if ui.selectable_label(c.rejection == r, n).clicked() {
                    c.rejection = r;
                }
            }
        });
        w::reset(ui, p, &mut c.rejection, &base.combine.rejection, "default");
    });
    let Some(plan) = a.stack.plan.clone() else { return };
    if plan.stacks.is_empty() {
        return;
    }
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.label(RichText::new("Master").font(body(13.0)));
        let label = |s: &job::StackPlan| format!("{} · {}", s.object, if s.filter.is_empty() { "no filter" } else { &s.filter });
        let cur = plan.stacks.get(a.stack.istack).map(label).unwrap_or_default();
        egui::ComboBox::from_id_salt("int-stack").selected_text(cur).width(260.0).show_ui(ui, |ui| {
            for (k, s) in plan.stacks.iter().enumerate() {
                if ui.selectable_label(k == a.stack.istack, label(s)).clicked() {
                    a.stack.istack = k;
                }
            }
        });
        ui.add_space(12.0);
        for (m, n) in [(0u8, "Master"), (1, "Rejected low"), (2, "Rejected high")] {
            chip(ui, p, &mut a.stack.imap, m, n);
        }
    });
    let Some(stp) = plan.stacks.get(a.stack.istack) else { return };
    let field = &plan.fields[stp.field];
    let secs: f64 = stp.lights.iter().map(|&i| field.lights[i].exptime).sum();
    let refname = field.lights[stp.lights[stp.reference]].path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    ui.label(RichText::new(format!("{} lights, {:.2} h; {}; normalised to {}", stp.lights.len(), secs / 3600.0, st2.combine.rejection_for(stp.lights.len()).name(), short_name(&refname))).font(body(12.0)).color(p.muted));
    let Some(r) = a.stack.integrations.get(&crux_core::paths::key(&stp.out)).cloned() else {
        ui.label(RichText::new("Not integrated yet: Process builds it after the lights.").font(body(12.0)).color(p.faint));
        return;
    };
    ui.label(
        RichText::new(format!(
            "Noise {:.1} ADU against {:.1} for the median frame: SNR ×{:.2} (the weights predicted ×{:.2}), worth {:.1} frames. {:.2}% of values rejected{}.",
            r.noise_master,
            r.noise_typical,
            r.snr_gain,
            r.snr_gain_expected,
            r.equivalent_frames,
            r.rejected_total * 100.0,
            r.fwhm.map(|f| format!("; FWHM {f:.2} px")).unwrap_or_default()
        ))
        .font(body(13.0)),
    );
    for wi in &r.what_ifs {
        let good = wi.snr_change >= 0.0;
        ui.label(RichText::new(format!("{}: SNR {:+.1}%{}", capital(&wi.label), wi.snr_change * 100.0, wi.fwhm.map(|f| format!(", FWHM {f:.2} px")).unwrap_or_default())).font(body(12.0)).color(if good { p.good } else { p.muted }));
    }
    // Per-frame weights.
    ui.add_space(4.0);
    let widths = [230.0, 160.0, 70.0, 80.0, 70.0, 120.0];
    let row = |ui: &mut Ui, cells: [(String, egui::FontId, Color32); 6], bar: Option<f32>| {
        let (rr, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 20.0), Sense::hover());
        let mut x = rr.left() + 6.0;
        for (k, ((t, f, col), wdt)) in cells.into_iter().zip(widths).enumerate() {
            if k == 1 {
                if let Some(b) = bar {
                    let br = Rect::from_min_size(Pos2::new(x, rr.center().y - 4.0), Vec2::new((wdt - 60.0) * b.clamp(0.0, 1.0), 8.0));
                    ui.painter().rect_filled(br, 3.0, p.accent);
                }
                let g = ui.painter().layout_no_wrap(t, f, col);
                ui.painter().galley(Pos2::new(x + wdt - 52.0, rr.center().y - g.size().y / 2.0), g, col);
            } else {
                let g = ui.painter().layout_no_wrap(t, f, col);
                ui.painter().with_clip_rect(Rect::from_min_size(Pos2::new(x, rr.top()), Vec2::new(wdt - 6.0, rr.height()))).galley(Pos2::new(x, rr.center().y - g.size().y / 2.0), g, col);
            }
            x += wdt;
        }
    };
    row(ui, ["Frame", "Weight", "Signal", "Noise ADU", "FWHM", "Rejected low / high"].map(|h| (h.to_string(), body(11.0), p.muted)), None);
    for (k, f) in r.frames.iter().enumerate() {
        let name = short_name(&f.name);
        let tag = if k == r.reference { " (reference)" } else if f.grade == crux_core::grade::Grade::Warn { " (warn)" } else { "" };
        row(
            ui,
            [
                (format!("{name}{tag}"), mono(12.0), p.fg),
                (format!("{:.2}", f.weight), body(12.0), p.fg),
                (format!("{:.3}", f.signal), body(12.0), p.muted),
                (format!("{:.1}", f.noise), body(12.0), p.muted),
                (f.fwhm.map(|x| format!("{x:.2}")).unwrap_or_else(|| "–".into()), body(12.0), p.muted),
                (format!("{:.2}% / {:.2}%", f.rejected_low * 100.0, f.rejected_high * 100.0), body(12.0), p.muted),
            ],
            Some(f.weight as f32),
        );
    }
    ui.label(RichText::new(r.out.display().to_string()).font(mono(11.0)).color(p.faint));
}

fn solve_summary(ui: &mut Ui, a: &UiState, s: &Session, p: &Pal) {
    let sols: Vec<(&PathBuf, &Solution)> = a.stack.solves.iter().filter_map(|(k, v)| v.as_ref().map(|v| (k, v))).collect();
    if sols.is_empty() {
        return;
    }
    let ra = median(sols.iter().map(|x| x.1.ra).collect()).unwrap_or(0.0);
    let dec = median(sols.iter().map(|x| x.1.dec).collect()).unwrap_or(0.0);
    let scale = median(sols.iter().map(|x| x.1.scale).collect()).unwrap_or(0.0);
    let rot: Vec<f64> = sols.iter().map(|x| x.1.rotation).collect();
    let (rmin, rmax) = rot.iter().fold((f64::MAX, f64::MIN), |a, &r| (a.0.min(r), a.1.max(r)));
    ui.label(RichText::new(format!("Centre {} {}, {:.4}″/px, rotation {:+.2}° to {:+.2}°", hms(ra), dms(dec), scale, rmin, rmax)).font(body(13.0)));
    let failed = a.stack.solves.values().filter(|v| v.is_none()).count();
    if failed > 0 {
        ui.label(RichText::new(format!("{failed} not solved: usually cloud or too few stars")).font(body(12.0)).color(p.warn));
    }
    let far: Vec<(&PathBuf, f64)> = sols.iter().filter_map(|(k, v)| v.target_offset_arcmin.filter(|x| *x > 2.0).map(|x| (*k, x))).collect();
    for (k, x) in far {
        let n = k.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        ui.label(RichText::new(format!("{}: {x:.1}′ from its planned target ({}) — mispointed or mislabelled", short_name(&n), s.triage.scan.frames.iter().find(|f| &f.meta.path == k).and_then(|f| f.meta.object.clone()).unwrap_or_default())).font(body(12.0)).color(p.warn));
    }
    let mount = median(sols.iter().filter_map(|x| x.1.offset_arcmin).collect());
    if let Some(m) = mount.filter(|m| *m > 5.0) {
        ui.label(RichText::new(format!("The mount's reported position is a median {m:.0}′ from where it was really pointing (its pointing model or sync)")).font(body(12.0)).color(p.muted));
    }
}

fn hms(deg: f64) -> String {
    let t = (deg.rem_euclid(360.0) / 15.0 * 36000.0).round() as u64;
    // t is in tenths of a second of time.
    format!("{:02}h{:02}m{:02}.{}s", t / 36000, t / 600 % 60, t / 10 % 60, t % 10)
}

fn dms(deg: f64) -> String {
    let t = (deg.abs() * 3600.0).round() as u64;
    format!("{}{:02}°{:02}′{:02}″", if deg < 0.0 { "−" } else { "+" }, t / 3600, t / 60 % 60, t % 60)
}

fn frame_table(ui: &mut Ui, a: &mut UiState, s: &Session, p: &Pal) {
    let Some(plan) = a.stack.plan.clone() else { return };
    ui.horizontal(|ui| {
        w::h2(ui, "Frames");
        ui.label(RichText::new(format!("{} in the plan, {} left out", plan.n_lights(), plan.n_skipped())).color(p.muted));
    });
    let header = ["Frame", "Filter", "Flat", "Registration", "Solve", "Status"];
    let widths = [230.0, 50.0, 230.0, 190.0, 150.0, 130.0];
    let row = |ui: &mut Ui, cells: [(String, egui::FontId, Color32); 6], sel: bool| -> egui::Response {
        let (r, resp) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 22.0), Sense::click());
        if sel {
            ui.painter().rect_filled(r, 4.0, p.panel2);
        } else if resp.hovered() {
            ui.painter().rect_filled(r, 4.0, crate::theme::alpha(p.panel2, 0.6));
        }
        let mut x = r.left() + 6.0;
        for ((t, f, c), wdt) in cells.into_iter().zip(widths) {
            let g = ui.painter().layout_no_wrap(t, f, c);
            ui.painter().with_clip_rect(Rect::from_min_size(Pos2::new(x, r.top()), Vec2::new(wdt - 6.0, r.height()))).galley(Pos2::new(x, r.center().y - g.size().y / 2.0), g, c);
            x += wdt;
        }
        resp
    };
    row(ui, header.map(|h| (h.to_string(), body(11.0), p.muted)), false);
    for f in &plan.fields {
        for l in &f.lights {
            let name = l.path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            let flat = plan.cals[l.cal].flat.as_ref().map(|x| x.label()).unwrap_or_else(|| "none".into());
            let reg = if l.frame == f.reference { "reference".to_string() } else { format!("{:.2} px, {} stars{}", l.reg.rms, l.reg.matched, if l.reg.flipped { ", flip" } else { "" }) };
            let sol = match a.stack.solves.get(&l.path) {
                Some(Some(x)) => format!("{:+.2}°{}", x.rotation, x.target_offset_arcmin.map(|o| format!(", {o:.1}′ off")).unwrap_or_default()),
                Some(None) => "not solved".into(),
                None => "–".into(),
            };
            let (status, sc) = match a.stack.results.get(&l.path) {
                Some(r) if r.error.is_some() => ("failed".to_string(), p.rej),
                Some(r) if r.skipped_up_to_date => ("up to date".to_string(), p.good),
                Some(r) => (format!("done {:.1} s", r.seconds), p.good),
                None => (if l.grade == crux_core::grade::Grade::Warn { "warn · to do" } else { "to do" }.to_string(), p.muted),
            };
            let cells = [
                (short_name(&name), mono(12.0), p.fg),
                (l.filter.clone(), body(12.0), p.filter(&l.filter)),
                (flat, body(12.0), p.muted),
                (reg, body(12.0), p.fg),
                (sol, body(12.0), p.muted),
                (status, body(12.0), sc),
            ];
            let sel = a.stack.sel.as_ref() == Some(&l.path);
            if row(ui, cells, sel).clicked() {
                a.stack.sel = Some(l.path.clone());
            }
        }
        let mut why: std::collections::BTreeMap<&str, usize> = Default::default();
        for (_, w) in &f.skipped {
            *why.entry(w.as_str()).or_default() += 1;
        }
        for (wy, n) in why {
            ui.label(RichText::new(format!("{}: {n} left out, {wy}", f.object)).font(body(12.0)).color(p.faint));
        }
    }
    let _ = s;
}
