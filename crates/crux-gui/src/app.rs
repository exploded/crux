//! The app shell: left rail, top bar with the pipeline strip and Activity,
//! one view at a time, toasts, the keyboard and the help sheet.

use crate::session::Session;
use crate::theme::{self, alpha, body, mono, semibold, Pal};
use crate::viewer::{self, Loader, Resident};
use crate::widgets::{self as w, Kind};
use crate::worker::{self, HeaderFacts, Msg, OpenJob};
use crate::{frames, home, inbox};
use chrono::NaiveDate;
use crux_core::grade::Grade;
use crux_core::pipeline::Stage;
use eframe::egui::{self, Color32, CornerRadius, Key, Pos2, Rect, Sense, Stroke, StrokeKind, Vec2};
use eframe::egui_wgpu;
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum View {
    Home,
    Inbox,
    Frames,
    Organise,
    Calibrate,
    Image,
    Settings,
}

impl View {
    /// Views that show an open folder's frames; without one they're greyed.
    pub fn needs_folder(self) -> bool {
        !matches!(self, View::Home | View::Settings)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Show {
    All,
    Flagged,
    Mine,
    Disagree,
    Border,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Sort {
    Time,
    Worst,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Metric {
    T,
    Fwhm,
    Ecc,
    Sky,
}

#[derive(Clone)]
pub struct Brush {
    pub night: NaiveDate,
    pub a: f32,
    pub b: f32,
    pub ids: Vec<usize>,
}

pub struct FramesUi {
    pub night: Option<NaiveDate>,
    pub show: Show,
    pub sort: Sort,
    pub metric: Metric,
    pub sel: Option<usize>,
    pub brush: Option<Brush>,
    pub marks: bool,
    pub grid: bool,
    pub mode: viewer::Mode,
    pub view: viewer::View,
    pub scroll_to_sel: bool,
    pub dragging_limit: bool,
    pub brush_from: Option<f32>,
}

pub struct Toast {
    pub msg: String,
    pub undo: bool,
    pub until: f64,
}

#[derive(Clone)]
pub struct Job {
    pub label: String,
    pub detail: String,
    pub progress: Option<(usize, usize)>,
    pub running: bool,
}

/// Everything the views share except the session.
pub struct UiState {
    pub pal: Pal,
    pub view: View,
    pub fu: FramesUi,
    pub toast: Option<Toast>,
    pub thumbs: HashMap<usize, egui::TextureHandle>,
    pub gallery: HashMap<PathBuf, egui::TextureHandle>,
    pub folders: Vec<crate::session::FolderSummary>,
    pub folders_s: f64,
    /// The open folder's finished image: None until the worker has looked,
    /// then Some(None) when it has none.
    pub target_image: Option<Option<PathBuf>>,
    pub facts: Option<HeaderFacts>,
    pub progress: Option<(Stage, usize, usize)>,
    pub loading: Option<PathBuf>,
    pub failed: Option<String>,
    pub resident: HashMap<usize, Resident>,
    pub load_errors: HashMap<usize, String>,
    pub gpu: Option<egui_wgpu::RenderState>,
    pub curves: HashMap<NaiveDate, Vec<[f64; 4]>>,
    /// Open another folder (set by views, handled by the app).
    pub open_request: Option<PathBuf>,
    pub cancel: Arc<AtomicBool>,
    pub started: f64,
    /// The library index (every folder), each folder's space, empty folders.
    pub library: Vec<crux_core::index::IndexFile>,
    pub spaces: Vec<crux_core::health::FolderSpace>,
    pub empty_dirs: Vec<PathBuf>,
    pub index_error: Option<String>,
    pub org: crate::organise::OrgUi,
    pub cal: crate::calibrate::CalUi,
    pub stack: crate::stack::StackUi,
    /// The library index (`index.json` path; SQLite beside it).
    pub index_path: Option<PathBuf>,
    /// Something a view wants the app to do (confirm and run a move, export…).
    pub request: Option<crate::organise::Request>,
    pub load_ms: Vec<f32>,
    /// Folder watcher and Autopilot (their own channel).
    pub bg: crate::auto::Bg,
    /// The Settings page (per-user crux.toml).
    pub settings: crate::settings::SettingsUi,
    /// Library mode (a folder of targets).
    pub lib: crate::libview::LibUi,
    /// Settings' Start fresh card.
    pub fresh: crate::fresh::FreshUi,
}

impl UiState {
    pub fn toast(&mut self, ctx: &egui::Context, msg: impl Into<String>, undo: bool) {
        self.toast = Some(Toast { msg: msg.into(), undo, until: ctx.input(|i| i.time) + 5.0 });
    }
}

pub struct Args {
    pub dir: Option<PathBuf>,
    pub cache_dir: Option<PathBuf>,
    pub library: Option<PathBuf>,
}

/// Screenshot harness for checking the UI without a person at the screen:
/// CRUX_GUI_SHOT=<png> plus CRUX_GUI_VIEW, CRUX_GUI_NIGHT, CRUX_GUI_SEL
/// (file-name fragment), CRUX_GUI_OPTS (marks,grid,blink,diff,flagged,zoom)
/// and CRUX_GUI_KEYS (keys pressed once everything has loaded, e.g. "r,j,a").
struct Shot {
    path: PathBuf,
    view: Option<String>,
    night: Option<String>,
    sel: Option<String>,
    opts: String,
    applied: bool,
    frames_ready: u32,
    sent: bool,
    done: Arc<AtomicBool>,
    keys: Vec<Key>,
    /// Actions run once loaded: auto-apply / auto-undo (scratch folders
    /// only, never under C:\PixInsight), confirm (open the modal), hash.
    pending: Vec<String>,
}

pub struct App {
    pub ui: UiState,
    pub session: Option<Session>,
    args: Args,
    tx: Sender<Msg>,
    rx: Receiver<Msg>,
    /// Watcher and Autopilot messages: never replaced.
    bg_rx: Receiver<Msg>,
    loader: Loader,
    order: VecDeque<usize>,
    last_want: Vec<usize>,
    help: bool,
    drawer: bool,
    g_at: Option<f64>,
    shot: Option<Shot>,
    /// The move waiting for confirmation (the app's only modal).
    confirm: Option<crate::organise::Request>,
    /// A move in progress: (done, total, last file).
    moving: Option<(usize, usize, PathBuf)>,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>, args: Args) -> App {
        let ctx = &cc.egui_ctx;
        theme::install_fonts(ctx);
        ctx.set_theme(egui::Theme::Dark);
        let pal = Pal::dark();
        theme::apply(ctx, &pal);
        let gpu = cc.wgpu_render_state.clone();
        if let Some(rs) = &gpu {
            let g = viewer::Gpu::new(rs);
            rs.renderer.write().callback_resources.insert(g);
        }
        let (tx, rx) = std::sync::mpsc::channel();
        let (bg_tx, bg_rx) = std::sync::mpsc::channel();
        let shot = std::env::var_os("CRUX_GUI_SHOT").map(|p| Shot {
            path: PathBuf::from(p),
            view: std::env::var("CRUX_GUI_VIEW").ok(),
            night: std::env::var("CRUX_GUI_NIGHT").ok(),
            sel: std::env::var("CRUX_GUI_SEL").ok(),
            opts: std::env::var("CRUX_GUI_OPTS").unwrap_or_default(),
            applied: false,
            frames_ready: 0,
            sent: false,
            done: Arc::new(AtomicBool::new(false)),
            keys: std::env::var("CRUX_GUI_KEYS")
                .unwrap_or_default()
                .split(',')
                .filter_map(|k| {
                    let k = k.trim();
                    let title: String = k.chars().take(1).flat_map(char::to_uppercase).chain(k.chars().skip(1)).collect();
                    Key::from_name(k).or_else(|| Key::from_name(&title))
                })
                .collect(),
            pending: Vec::new(),
        });
        let mut app = App {
            ui: UiState {
                pal,
                view: View::Home,
                fu: FramesUi {
                    night: None,
                    show: Show::All,
                    sort: Sort::Time,
                    metric: Metric::T,
                    sel: None,
                    brush: None,
                    marks: false,
                    grid: false,
                    mode: viewer::Mode::Frame,
                    view: viewer::View::default(),
                    scroll_to_sel: false,
                    dragging_limit: false,
                    brush_from: None,
                },
                toast: None,
                thumbs: HashMap::new(),
                gallery: HashMap::new(),
                folders: Vec::new(),
                folders_s: 0.0,
                target_image: None,
                facts: None,
                progress: None,
                loading: None,
                failed: None,
                resident: HashMap::new(),
                load_errors: HashMap::new(),
                gpu,
                curves: HashMap::new(),
                open_request: None,
                cancel: Arc::new(AtomicBool::new(false)),
                started: 0.0,
                load_ms: Vec::new(),
                bg: crate::auto::Bg::new(bg_tx, args.cache_dir.as_ref().map(|d| d.join("autopilot"))),
                library: Vec::new(),
                spaces: Vec::new(),
                empty_dirs: Vec::new(),
                index_error: None,
                org: crate::organise::OrgUi::default(),
                cal: crate::calibrate::CalUi::default(),
                stack: crate::stack::StackUi::default(),
                settings: crate::settings::SettingsUi::default(),
                lib: crate::libview::LibUi { cache_override: args.cache_dir.clone(), ..Default::default() },
                fresh: Default::default(),
                index_path: None,
                request: None,
            },
            session: None,
            loader: Loader::new(ctx.clone()),
            order: VecDeque::new(),
            last_want: Vec::new(),
            tx,
            rx,
            bg_rx,
            help: false,
            drawer: false,
            g_at: None,
            confirm: None,
            moving: None,
            shot,
            args,
        };
        // Open the compute device now (after the user's `[gpu] use`), so the
        // top bar can say GPU or CPU without waiting for the first job.
        let c = ctx.clone();
        std::thread::spawn(move || {
            let _ = crux_core::config::Config::load(None, None);
            crux_core::gpu::open();
            c.request_repaint();
        });
        if let Some(d) = app.args.dir.clone() {
            app.open(ctx, d);
        }
        if app.args.cache_dir.is_none() {
            // Empty caches of folders not opened for a while (rebuildable parts only).
            std::thread::spawn(|| {
                if let Ok((c, _)) = crux_core::config::Config::load(None, None) {
                    crux_core::cachedir::auto_clean(c.cache.unused_days);
                }
            });
        }
        app
    }

    fn cache_dir_for(&self, dir: &std::path::Path) -> PathBuf {
        let in_folder = self.args.cache_dir.is_none() && crux_core::config::Config::load(None, Some(dir)).is_ok_and(|(c, _)| c.cache.in_folder);
        crux_core::cachedir::resolve(self.args.cache_dir.as_deref(), dir, in_folder)
    }

    pub fn open(&mut self, ctx: &egui::Context, dir: PathBuf) {
        self.ui.cancel.store(true, Ordering::Relaxed);
        self.ui.cancel = Arc::new(AtomicBool::new(false));
        // A fresh channel drops anything still arriving from the old folder.
        let (tx, rx) = std::sync::mpsc::channel();
        self.tx = tx;
        self.rx = rx;
        self.session = None;
        self.ui.thumbs.clear();
        self.ui.curves.clear();
        self.ui.facts = None;
        self.ui.progress = None;
        self.ui.failed = None;
        self.ui.fu.sel = None;
        self.ui.fu.brush = None;
        self.ui.fu.night = None;
        self.ui.loading = Some(dir.clone());
        self.ui.started = ctx.input(|i| i.time);
        self.drop_resident();
        let index = crux_core::cachedir::default_index(self.args.cache_dir.as_deref());
        self.ui.org.invalidate();
        self.ui.cal.reset();
        self.ui.stack.reset();
        self.ui.index_path = index.clone();
        self.ui.org.dups = None;
        self.ui.org.dup_progress = None;
        self.ui.spaces.clear();
        // The old folder's neighbours, until the new folder's arrive.
        self.ui.folders.clear();
        self.ui.target_image = None;
        self.ui.lib.view = None;
        self.ui.lib.error = None;
        let job = OpenJob { cache_dir: Some(self.cache_dir_for(&dir)), library: self.args.library.clone(), index, root: dir.clone(), cancel: self.ui.cancel.clone() };
        worker::spawn_open(job, self.tx.clone(), ctx.clone());
        crate::remember_folder(&dir);
    }

    fn drop_resident(&mut self) {
        if let Some(rs) = &self.ui.gpu {
            let mut r = rs.renderer.write();
            if let Some(g) = r.callback_resources.get_mut::<viewer::Gpu>() {
                for li in self.order.drain(..) {
                    g.textures_remove(li);
                }
            }
        }
        self.order.clear();
        self.ui.resident.clear();
        self.ui.load_errors.clear();
        self.last_want.clear();
    }

    fn poll(&mut self, ctx: &egui::Context) {
        while let Ok(m) = self.bg_rx.try_recv() {
            let open = self.session.as_ref().map(|s| s.root.clone());
            if let Some(root) = crate::auto::handle(&mut self.ui, m, ctx, open.as_deref()) {
                // New subs in the open folder: reopen it (only they get measured).
                self.ui.open_request = Some(root);
            }
        }
        while let Ok(m) = self.rx.try_recv() {
            let Some(m) = self.ui.stack.handle(m, ctx) else { continue };
            match m {
                Msg::Headers(f) => self.ui.facts = Some(f),
                Msg::Progress { stage, done, total } => self.ui.progress = Some((stage, done, total)),
                Msg::Failed(e) => {
                    self.ui.loading = None;
                    if e != "cancelled" {
                        self.ui.failed = Some(e);
                    }
                }
                Msg::Loaded(l) => {
                    let l = *l;
                    let s = Session::new(l.root, l.cache_dir, l.cfg, l.triage, l.transp, l.thumbs, l.timings);
                    for (&li, t) in &s.thumbs {
                        let img = egui::ColorImage::from_gray([t.w as usize, t.h as usize], &t.data);
                        let tex = ctx.load_texture(format!("thumb{li}"), img, egui::TextureOptions::LINEAR);
                        self.ui.thumbs.insert(li, tex);
                    }
                    // Frames opens on every night (James, 2026-11-23).
                    self.ui.fu.night = None;
                    self.ui.loading = None;
                    self.session = Some(s);
                }
                Msg::LibraryView(v) => {
                    self.ui.loading = None;
                    self.ui.view = View::Home;
                    self.ui.lib.view = Some(*v);
                }
                Msg::Folders(f, secs) => {
                    self.ui.folders = f;
                    self.ui.folders_s = secs;
                }
                Msg::TargetImage(p) => self.ui.target_image = Some(p),
                Msg::Gallery(p, img) => {
                    let tex = ctx.load_texture(format!("gallery{}", p.display()), img, egui::TextureOptions::LINEAR);
                    self.ui.gallery.insert(p, tex);
                }
                Msg::Library { files, spaces, empty, error } => {
                    self.ui.library = files;
                    self.ui.spaces = spaces;
                    self.ui.empty_dirs = empty;
                    self.ui.index_error = error;
                }
                Msg::MoveProgress { done, total, file } => self.moving = Some((done, total, file)),
                Msg::MoveDone { undo, result, root } => {
                    self.moving = None;
                    self.confirm = None;
                    match result {
                        Ok(o) => {
                            let what = if undo { "Moved back" } else { "Moved" };
                            self.ui.toast(ctx, format!("{what} {} files ({:.1} GB). Journal written.", o.moved, o.bytes as f64 / 1e9), false);
                        }
                        Err(e) => self.ui.failed = Some(format!("Move stopped: {e}")),
                    }
                    self.ui.org.invalidate();
                    self.open(ctx, root);
                }
                Msg::FpProgress { done, total } => self.ui.cal.fp_progress = Some((done, total)),
                // Handled by the Stack page above.
                Msg::Process(_) | Msg::ProcessDone(_) | Msg::Preview(_) | Msg::PreviewFailed(..) | Msg::Solved { .. } | Msg::SolveFailed(_) | Msg::ImagePrepared(..) | Msg::ImageWritten(_) | Msg::ImageProgress(_) | Msg::ImageEnhanceView(..) | Msg::ImageView(_) | Msg::ImageTiles(..) | Msg::MosaicStep(_) | Msg::MosaicDone(_) | Msg::MosaicPreview(..) => {}
                // These come on the background channel.
                Msg::WatchNew(..) | Msg::AutoEstimate(_) | Msg::AutoEvent(_) | Msg::AutoDone(_) | Msg::Refresh(_) | Msg::RefreshDone(_) => {}
                Msg::Fingerprints(f) => {
                    self.ui.cal.fp = Some(*f);
                    self.ui.cal.fp_progress = None;
                }
                Msg::DupProgress { done, total } => self.ui.org.dup_progress = Some((done, total)),
                Msg::Duplicates(d) => {
                    self.ui.org.dups = Some(d);
                    self.ui.org.dup_progress = None;
                }
                Msg::Exported(r, dest) => match r {
                    Ok(o) if o.list.is_some() => self.ui.toast(ctx, format!("The export folder is on another drive, so crux wrote a file list to {}", o.list.unwrap().display()), false),
                    Ok(o) => self.ui.toast(
                        ctx,
                        format!("Linked {} files into {} ({} already there).", o.linked, dest.display(), o.existing),
                        false,
                    ),
                    Err(e) => self.ui.toast(ctx, format!("Export failed: {e}"), false),
                },
            }
        }
        while let Some(r) = self.loader.try_recv() {
            match r {
                Ok(d) => {
                    self.ui.load_ms.push(d.ms);
                    let Some(rs) = self.ui.gpu.clone() else { continue };
                    let keep = self.keep_list();
                    for li in viewer::upload(&rs, &d, &mut self.order, &keep) {
                        self.ui.resident.remove(&li);
                    }
                    self.ui.resident.insert(d.li, Resident { w: d.w, h: d.h, c0: d.c0, m: d.m, median: d.median, sigma: d.sigma });
                }
                Err((li, e)) => {
                    self.ui.load_errors.insert(li, e);
                }
            }
        }
    }

    /// Frames worth having on the GPU now: the selection, its reference and neighbours.
    fn keep_list(&self) -> Vec<usize> {
        let (Some(s), Some(sel)) = (&self.session, self.ui.fu.sel) else { return Vec::new() };
        let mut v = vec![sel];
        if let Some(r) = s.reference_of(sel) {
            v.push(r);
        }
        let list = frames::visible(s, &self.ui.fu);
        if let Some(p) = list.iter().position(|&x| x == sel) {
            for d in [1isize, -1, 2] {
                let j = p as isize + d;
                if j >= 0 && (j as usize) < list.len() {
                    v.push(list[j as usize]);
                }
            }
        }
        v.dedup();
        v
    }

    fn request_frames(&mut self) {
        if self.ui.view != View::Frames || self.ui.gpu.is_none() {
            return;
        }
        let want = self.keep_list();
        if want == self.last_want {
            return;
        }
        self.last_want = want.clone();
        let Some(s) = &self.session else { return };
        let list: Vec<(usize, PathBuf)> = want.iter().map(|&li| (li, s.lights[li].path.clone())).collect();
        let resident = &self.ui.resident;
        let errs = &self.ui.load_errors;
        self.loader.want(list, &|li| resident.contains_key(&li) || errs.contains_key(&li));
    }

    fn keys(&mut self, ctx: &egui::Context) {
        if ctx.egui_wants_keyboard_input() {
            return;
        }
        let now = ctx.input(|i| i.time);
        let pressed = |k: Key| ctx.input(|i| i.key_pressed(k) && !i.modifiers.ctrl && !i.modifiers.alt);
        if self.help {
            if pressed(Key::Escape) || pressed(Key::Questionmark) {
                self.help = false;
            }
            return;
        }
        if let Some(t) = self.g_at {
            if now - t < 0.9 {
                let to = [(Key::H, View::Home), (Key::I, View::Inbox), (Key::F, View::Frames), (Key::O, View::Organise), (Key::C, View::Calibrate)];
                for (k, v) in to {
                    if pressed(k) && (self.session.is_some() || !v.needs_folder()) {
                        self.ui.view = v;
                        self.g_at = None;
                        return;
                    }
                }
            } else {
                self.g_at = None;
            }
        }
        if pressed(Key::G) {
            self.g_at = Some(now);
            return;
        }
        if pressed(Key::Questionmark) || ctx.input(|i| i.events.iter().any(|e| matches!(e, egui::Event::Text(t) if t == "?"))) {
            self.help = true;
            return;
        }
        if pressed(Key::U) {
            if let Some(s) = &mut self.session {
                if s.undo_last() {
                    self.ui.toast(ctx, "Undone.", false);
                }
            }
        }
        if self.ui.view != View::Frames {
            return;
        }
        let Some(s) = &mut self.session else { return };
        let fu = &mut self.ui.fu;
        let list = frames::visible(s, fu);
        let pos = fu.sel.and_then(|x| list.iter().position(|&y| y == x));
        let step = |fu: &mut FramesUi, d: isize| {
            let j = pos.map(|p| p as isize + d).unwrap_or(0);
            if j >= 0 && (j as usize) < list.len() {
                fu.sel = Some(list[j as usize]);
                fu.scroll_to_sel = true;
                if fu.mode == viewer::Mode::Reference {
                    fu.mode = viewer::Mode::Frame;
                }
            }
        };
        if pressed(Key::J) || pressed(Key::ArrowRight) {
            step(fu, 1);
        } else if pressed(Key::K) || pressed(Key::ArrowLeft) {
            step(fu, -1);
        } else if pressed(Key::A) || pressed(Key::R) {
            if let Some(sel) = fu.sel {
                let g = if pressed(Key::A) { Grade::Good } else { Grade::Reject };
                s.set_override(&[sel], Some(g));
                let msg = format!("{} {}", s.lights[sel].hhmm(), if g == Grade::Reject { "marked reject" } else { "marked keep" });
                step(fu, 1);
                self.ui.toast(ctx, msg, true);
            }
        } else if pressed(Key::Space) {
            fu.mode = if fu.mode == viewer::Mode::Reference { viewer::Mode::Frame } else { viewer::Mode::Reference };
        } else if pressed(Key::D) {
            fu.mode = if fu.mode == viewer::Mode::Difference { viewer::Mode::Frame } else { viewer::Mode::Difference };
        } else if pressed(Key::Num1) {
            fu.marks = !fu.marks;
        } else if pressed(Key::Num2) {
            fu.grid = !fu.grid;
        } else if pressed(Key::F) {
            fu.view.scale = None;
        } else if pressed(Key::Escape) {
            fu.brush = None;
        }
    }

    fn shot_step(&mut self, ctx: &egui::Context) {
        let Some(sh) = &mut self.shot else { return };
        if sh.done.load(Ordering::Relaxed) {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }
        ctx.request_repaint();
        if self.session.is_none() {
            self.shot_library(ctx);
            return;
        }
        let Some(sh) = &mut self.shot else { return };
        let Some(s) = &self.session else { return };
        if !sh.applied {
            sh.applied = true;
            self.ui.view = match sh.view.as_deref() {
                Some("frames") => View::Frames,
                Some("inbox") => View::Inbox,
                Some("organise") => View::Organise,
                Some("calibrate") => View::Calibrate,
                Some("stack") | Some("image") => View::Image,
                Some("settings") => View::Settings,
                _ => View::Home,
            };
            if let Some(n) = sh.night.as_deref() {
                self.ui.fu.night = if n == "all" { None } else { n.parse().ok() };
            }
            if let Some(f) = sh.sel.as_deref() {
                self.ui.fu.sel = s.lights.iter().position(|l| l.name.contains(f));
                if let Some(i) = self.ui.fu.sel {
                    self.ui.fu.night = Some(s.lights[i].night);
                }
            }
            let o = sh.opts.clone();
            let has = |k: &str| o.split(',').any(|x| x.trim() == k);
            self.ui.fu.marks = has("marks");
            // Settings: show Start fresh's choices (measured only; never cleared here).
            if has("fresh") || has("fresh-go") {
                self.ui.fresh.pending_open = true;
                self.ui.fresh.harness_go = has("fresh-go");
            }
            self.ui.fu.grid = has("grid");
            if has("blink") {
                self.ui.fu.mode = viewer::Mode::Reference;
            }
            if has("diff") {
                self.ui.fu.mode = viewer::Mode::Difference;
            }
            if has("flagged") {
                self.ui.fu.show = Show::Flagged;
            }
            if has("worst") {
                self.ui.fu.sort = Sort::Worst;
            }
            if has("zoom") {
                self.ui.fu.view = viewer::View { scale: Some(0.5), centre: egui::vec2(3124.0, 2088.0) };
            }
            if has("help") {
                self.help = true;
            }
            if has("drawer") {
                self.drawer = true;
            }
            for (k, t) in [("org-browse", crate::organise::Tab::Browse), ("org-layout", crate::organise::Tab::Layout), ("org-health", crate::organise::Tab::Health), ("org-history", crate::organise::Tab::History)] {
                if has(k) {
                    self.ui.view = View::Organise;
                    self.ui.org.tab = t;
                }
            }
            // Calibration: cal-set:<name fragment>, cal-pair, cal-night:<n>, cal-row:<n>.
            let s = self.session.as_ref().unwrap();
            for x in o.split(',') {
                if let Some(f) = x.strip_prefix("cal-set:") {
                    self.ui.cal.sel_set = s.triage.cal.sets.iter().map(|x| x.group).find(|g| s.triage.grouping.groups[*g].label.to_lowercase().contains(&f.to_lowercase()));
                }
                if x == "cal-pair" {
                    let pr = s.triage.cal.pairs.iter().find(|p| p.spread < 2.0).or(s.triage.cal.pairs.first());
                    self.ui.cal.sel_pair = pr.map(|p| (p.a, p.b));
                    self.ui.cal.sel_set = None;
                }
                if let Some(n) = x.strip_prefix("cal-night:").and_then(|n| n.parse::<usize>().ok()) {
                    self.ui.cal.sel_choice = Some(n);
                }
                if let Some(n) = x.strip_prefix("cal-scroll:").and_then(|n| n.parse::<f32>().ok()) {
                    self.ui.cal.scroll_to = Some(n);
                }
                if let Some(n) = x.strip_prefix("cal-row:").and_then(|n| n.parse::<usize>().ok()) {
                    self.ui.cal.open_row = Some((0, n));
                }
            }
            // Stack: stack-stage:<calibrate|cosmetic|register|solve|integrate>;
            // CRUX_GUI_SEL picks the frame; stack-master:<n>, stack-map:<low|high>.
            for x in o.split(',') {
                if let Some(st) = x.strip_prefix("stack-stage:").and_then(crate::stack::Stage::parse) {
                    self.ui.stack.stage = st;
                }
                if let Some(n) = x.strip_prefix("stack-master:").and_then(|n| n.parse::<usize>().ok()) {
                    self.ui.stack.istack = n;
                }
                match x {
                    "stack-map:low" => self.ui.stack.imap = 1,
                    "stack-map:high" => self.ui.stack.imap = 2,
                    // Image stages: img-preset:<name>, img-target:<n>, img-view:<f>, img-fold, img-model,
                    // img-chan:<n>, img-expert, img-auto, img-write.
                    "img-model" => self.ui.stack.img.gmodel = true,
                    "img-expert" => self.ui.stack.img.depth = crate::image::Depth::Expert,
                    "img-auto" => self.ui.stack.img.depth = crate::image::Depth::Auto,
                    "img-write" => self.ui.stack.img.pending_write = true,
                    "img-whole" => self.ui.stack.img.sview = 0,
                    "img-starless" => self.ui.stack.img.sview = 2,
                    // Sharpen and denoise stage: turn steps on, show the corner.
                    "enh-denoise" => self.ui.stack.img.pending_enhance.push("denoise"),
                    "enh-correct" => self.ui.stack.img.pending_enhance.push("correct"),
                    "enh-tighten" => self.ui.stack.img.pending_enhance.push("tighten"),
                    "enh-corner" => self.ui.stack.img.ewhere = 1,
                    // Crop and save stage: trim the empty edges.
                    "img-autocrop" => self.ui.stack.img.pending_autocrop = true,
                    // Turn the picture a quarter turn clockwise.
                    "img-turn" => self.ui.stack.img.pending_turn = true,
                    // Milestone 14: mos-run assembles the mosaic; auto-estimate
                    // and auto-run start the Autopilot on Home; watch starts the watcher.
                    "mos-run" => self.ui.stack.mos.pending_run = true,
                    "auto-estimate" => self.ui.bg.auto.pending = Some(false),
                    "auto-run" => self.ui.bg.auto.pending = Some(true),
                    // Refresh calibration library (Home), then reopen.
                    "lib-refresh" => {
                        let cfg = s.cfg.clone();
                        crate::auto::start_refresh(&mut self.ui, &cfg, ctx);
                    }
                    "watch" => {
                        let (root, cfg) = (s.root.clone(), s.cfg.clone());
                        crate::auto::start_watch(&mut self.ui, &root, &cfg, ctx);
                    }
                    _ => {}
                }
                if let Some(n) = x.strip_prefix("img-preset:") {
                    self.ui.stack.img.pending_preset = Some(n.to_string());
                }
                if let Some(f) = x.strip_prefix("img-view:").and_then(|f| f.parse::<f32>().ok()) {
                    self.ui.stack.img.set_view_frac(f);
                }
                if x == "img-fold" {
                    self.ui.stack.img.fold_presets();
                }
                if let Some(n) = x.strip_prefix("img-chan:").and_then(|n| n.parse::<usize>().ok()) {
                    self.ui.stack.img.gchan = n;
                }
                if let Some(n) = x.strip_prefix("img-target:").and_then(|n| n.parse::<usize>().ok()) {
                    self.ui.stack.img.sel = n;
                }
            }
            if let (View::Image, Some(f)) = (self.ui.view, self.shot.as_ref().and_then(|x| x.sel.clone()).as_deref()) {
                self.ui.stack.sel = s.lights.iter().find(|l| l.name.contains(f)).map(|l| l.path.clone());
                self.ui.stack.pinned_sel = true;
            }
            if let Ok(q) = std::env::var("CRUX_GUI_QUERY") {
                self.ui.org.query = q;
            }
            if let Some(n) = o.split(',').find_map(|x| x.strip_prefix("preset:")).and_then(|n| n.parse::<usize>().ok()) {
                self.ui.org.set_preset(n);
            }
            let pending: Vec<String> = o.split(',').map(|x| x.to_string()).filter(|x| matches!(x.as_str(), "auto-apply" | "auto-undo" | "auto-reclaim" | "auto-export" | "confirm" | "hash") || x.starts_with("cal-pick:") || x == "cal-unpick" || x == "stack-solve" || x == "stack-run").collect();
            if let Some(sh) = &mut self.shot {
                sh.pending = pending;
            }
            return;
        }
        let Some(sh) = &mut self.shot else { return };
        // Wait for the frame (and the reference when blinking) to reach the GPU.
        let need: Vec<usize> = match (self.ui.view, self.ui.fu.sel) {
            (View::Frames, Some(sel)) => {
                let mut v = vec![sel];
                if self.ui.fu.mode != viewer::Mode::Frame {
                    v.extend(s.reference_of(sel));
                }
                v
            }
            _ => Vec::new(),
        };
        let ready = need.iter().all(|li| self.ui.resident.contains_key(li) || self.ui.load_errors.contains_key(li))
            && (match &self.ui.target_image {
                Some(None) => true,
                Some(Some(p)) => self.ui.gallery.contains_key(p),
                None => false,
            } || ctx.input(|i| i.time) > 60.0)
            && (self.ui.view != View::Calibrate || self.ui.cal.fp.is_some() || ctx.input(|i| i.time) > 120.0)
            && (self.ui.view != View::Image || self.ui.stack.ready_for_shot() || ctx.input(|i| i.time) > 180.0);
        if ready {
            sh.frames_ready += 1;
        }
        // Scripted Organise actions (moves only outside C:\PixInsight).
        if ready && sh.frames_ready == 10 && !sh.pending.is_empty() && self.moving.is_none() {
            let act = sh.pending.remove(0);
            let real = crux_core::cache::key(&s.root).starts_with(r"c:\pixinsight");
            sh.frames_ready = 1;
            match act.as_str() {
                "auto-apply" | "confirm" => {
                    let lay = self.ui.org.layout.clone();
                    let sess = self.session.as_mut().unwrap();
                    let plan = sess.plan_with(lay.as_ref(), true);
                    let title = format!("Move {} files?", plan.moves.len());
                    self.confirm = Some(crate::organise::Request::Apply { plan, title });
                    if act == "auto-apply" && !real {
                        self.start_confirmed(ctx);
                    }
                }
                "auto-undo" if !real => {
                    let h = self.session.as_ref().and_then(|s| s.cache_dir.as_ref()).map(|d| crux_core::moves::history(&d.join("plans"))).unwrap_or_default();
                    if let Some(e) = h.into_iter().find(|e| e.undone.is_none()) {
                        self.confirm = Some(crate::organise::Request::Undo(e));
                        self.start_confirmed(ctx);
                    }
                }
                "auto-reclaim" if !real && !self.ui.spaces.is_empty() => {
                    let s = self.session.as_ref().unwrap();
                    let parent = s.root.parent().unwrap_or(&s.root).to_path_buf();
                    let plan = crux_core::health::reclaim_plan(&parent, &self.ui.spaces, s.cache_dir.as_deref());
                    self.confirm = Some(crate::organise::Request::Apply { title: "reclaim".into(), plan });
                    self.start_confirmed(ctx);
                }
                "auto-export" if !real => {
                    let s = self.session.as_ref().unwrap();
                    let kept: Vec<PathBuf> = s.lights.iter().enumerate().filter(|(i, _)| s.grade(*i) != Grade::Reject).map(|(_, l)| l.path.clone()).collect();
                    let dest = s.root.parent().unwrap_or(&s.root).join("_exports").join("harness_kept");
                    self.confirm = Some(crate::organise::Request::Export { files: kept, dest });
                    self.start_confirmed(ctx);
                }
                // The selected night (cal-night) uses the first library set
                // whose label contains the text: what a drag does.
                x if x.starts_with("cal-pick:") => {
                    let frag = x.trim_start_matches("cal-pick:").to_lowercase();
                    let ci = self.ui.cal.sel_choice.unwrap_or(0);
                    let sess = self.session.as_mut().unwrap();
                    let c = sess.triage.cal.choices.get(ci).cloned();
                    let set = sess.triage.library.as_ref().and_then(|l| l.sets.iter().find(|x| c.as_ref().is_some_and(|c| x.filter.eq_ignore_ascii_case(&c.filter)) && x.label().to_lowercase().contains(&frag)).cloned());
                    if let (Some(c), Some(set)) = (c, set) {
                        sess.pick_flat(&c.filter, c.night, Some((Some(set.id.clone()), None, set.night)));
                    }
                }
                "stack-solve" => {
                    let ix = self.ui.index_path.clone();
                    let sess = self.session.as_ref().unwrap();
                    self.ui.stack.start_solve(ix, sess, self.tx.clone(), ctx.clone());
                }
                "stack-run" => self.ui.stack.start_run(self.tx.clone(), ctx.clone()),
                "cal-unpick" => {
                    let ci = self.ui.cal.sel_choice.unwrap_or(0);
                    let sess = self.session.as_mut().unwrap();
                    if let Some(c) = sess.triage.cal.choices.get(ci).cloned() {
                        sess.pick_flat(&c.filter, c.night, None);
                    }
                }
                "hash" => {
                    let lib = self.ui.library.clone();
                    let c: Vec<Vec<crux_core::index::IndexFile>> = crux_core::health::duplicate_candidates(&lib).into_iter().map(|v| v.into_iter().cloned().collect()).collect();
                    self.ui.org.dup_progress = Some((0, c.iter().map(|x| x.len()).sum()));
                    worker::spawn_duplicates(c, self.tx.clone(), ctx.clone());
                }
                _ => {}
            }
            return;
        }
        if self.moving.is_some() || self.ui.loading.is_some() || self.ui.org.dup_progress.is_some() || self.ui.stack.running.is_some() || self.ui.stack.solving.is_some() || self.ui.bg.refresh.running || self.ui.open_request.is_some() || self.ui.fresh.measuring() {
            sh.frames_ready = 1;
            return;
        }
        // Press the scripted keys, one every few frames, before the screenshot.
        if ready && sh.frames_ready % 4 == 0 && !sh.keys.is_empty() {
            let key = sh.keys.remove(0);
            sh.frames_ready = 1;
            ctx.input_mut(|i| {
                i.events.push(egui::Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers: egui::Modifiers::NONE })
            });
        }
        if sh.frames_ready > 20 && sh.keys.is_empty() && sh.pending.is_empty() && !sh.sent {
            shoot(sh, ctx);
        }
    }

    /// The harness in library mode (no folder graded): wait for the card
    /// pictures, run `lib-combine` (open the confirm), `lib-auto-combine`
    /// and `lib-auto-undo` (never under C:\PixInsight), then shoot.
    fn shot_library(&mut self, ctx: &egui::Context) {
        let Some(sh) = &mut self.shot else { return };
        let Some(v) = &self.ui.lib.view else {
            sh.frames_ready = 1;
            return;
        };
        if !sh.applied {
            sh.applied = true;
            self.ui.view = if sh.view.as_deref() == Some("settings") { View::Settings } else { View::Home };
            if sh.opts.split(',').any(|x| x.trim() == "fresh") {
                self.ui.fresh.pending_open = true;
            }
            sh.pending = sh.opts.split(',').map(|x| x.trim().to_string()).filter(|x| matches!(x.as_str(), "lib-combine" | "lib-auto-combine" | "lib-auto-undo")).collect();
        }
        let ready = self.ui.gallery.len() >= v.images.len() || ctx.input(|i| i.time) > 60.0;
        if ready {
            sh.frames_ready += 1;
        }
        if self.moving.is_some() || self.ui.loading.is_some() || self.ui.bg.refresh.running || self.ui.open_request.is_some() || self.ui.fresh.measuring() {
            sh.frames_ready = 1;
            return;
        }
        if ready && sh.frames_ready == 10 && !sh.pending.is_empty() {
            let act = sh.pending.remove(0);
            sh.frames_ready = 1;
            let real = crux_core::cache::key(&v.root).starts_with(r"c:\pixinsight");
            match act.as_str() {
                "lib-combine" | "lib-auto-combine" => {
                    if let Some((ti, _)) = v.targets.combinable().first() {
                        if let Ok(mut plan) = v.targets.combine_plan(*ti, &v.cfg, None) {
                            if let Some(c) = &self.ui.lib.cache_override {
                                plan.cache_dir = Some(c.clone());
                            }
                            let title = format!("Combine {}?", v.targets.targets[*ti].name);
                            self.confirm = Some(crate::organise::Request::Apply { plan, title });
                            if act == "lib-auto-combine" && !real {
                                self.start_confirmed(ctx);
                            }
                        }
                    }
                }
                "lib-auto-undo" if !real => {
                    let root = v.root.clone();
                    let h = crux_core::moves::history(&crate::libview::plans_dir(self.ui.lib.cache_override.as_deref()));
                    if let Some(e) = h.into_iter().find(|e| e.undone.is_none() && crux_core::paths::key(&e.root) == crux_core::paths::key(&root)) {
                        self.confirm = Some(crate::organise::Request::Undo(e));
                        self.start_confirmed(ctx);
                    }
                }
                _ => {}
            }
            return;
        }
        if sh.frames_ready > 20 && sh.pending.is_empty() && !sh.sent {
            shoot(sh, ctx);
        }
    }
}

/// Save a screenshot to the harness's path, then let the app close.
fn shoot(sh: &mut Shot, ctx: &egui::Context) {
    sh.sent = true;
    let path = sh.path.clone();
    let done = sh.done.clone();
    let cb = egui::ScreenshotCallback::new(move |img: Arc<egui::ColorImage>| {
        let [wd, ht] = img.size;
        let mut buf = Vec::with_capacity(wd * ht * 4);
        for p in &img.pixels {
            buf.extend_from_slice(&p.to_array());
        }
        if let Some(b) = image::RgbaImage::from_raw(wd as u32, ht as u32, buf) {
            let _ = b.save(&path);
        }
        done.store(true, Ordering::Relaxed);
    });
    ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(cb));
}

impl eframe::App for App {
    fn clear_color(&self, _v: &egui::Visuals) -> [f32; 4] {
        self.ui.pal.bg.to_normalized_gamma_f32()
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll(&ctx);
        crate::fresh::tick(&mut self.ui, &ctx);
        self.shot_step(&ctx);
        self.keys(&ctx);
        // Folders dropped on the window.
        let dropped: Vec<PathBuf> = ctx.input(|i| i.raw.dropped_files.iter().map(|f| f.path().to_path_buf()).collect());
        if let Some(d) = dropped.into_iter().find(|p| p.is_dir()) {
            self.open(&ctx, d);
        }
        if let Some(d) = self.ui.open_request.take() {
            self.open(&ctx, d);
        }
        if self.ui.loading.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(200));
        }
        let p = self.ui.pal;
        egui::Panel::left("rail")
            .exact_size(76.0)
            .resizable(false)
            .frame(egui::Frame::new().fill(p.panel).stroke(Stroke::new(1.0, p.line)))
            .show(ui, |ui| self.rail(ui));
        egui::Panel::top("top")
            .frame(egui::Frame::new().fill(p.panel).stroke(Stroke::new(1.0, p.line)).inner_margin(egui::Margin { left: 20, right: 16, top: 8, bottom: 8 }))
            .show(ui, |ui| self.top(ui));
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(p.bg).inner_margin(egui::Margin { left: 20, right: 16, top: 20, bottom: 0 }))
            .show(ui, |ui| {
                let view = self.ui.view;
                match (&mut self.session, view) {
                    (_, View::Home) => home::show(ui, &mut self.ui, self.session.as_mut()),
                    (_, View::Settings) => crate::settings::show(ui, &mut self.ui, self.session.as_mut()),
                    (Some(s), View::Inbox) => inbox::show(ui, &mut self.ui, s),
                    (Some(s), View::Frames) => frames::show(ui, &mut self.ui, s),
                    (Some(s), View::Organise) => crate::organise::show(ui, &mut self.ui, s),
                    (Some(s), View::Calibrate) => crate::calibrate::view(ui, &mut self.ui, s),
                    (Some(s), View::Image) => crate::stack::view(ui, &mut self.ui, s, &self.tx),
                    (None, _) => home::show(ui, &mut self.ui, None),
                }
            });
        self.request_frames();
        self.handle_request(&ctx);
        if self.confirm.is_some() {
            self.confirm_modal(&ctx);
        }
        self.toasts(&ctx);
        if self.help {
            self.help_sheet(&ctx);
        }
        if self.drawer {
            self.activity(&ctx);
        }
    }
}

impl App {
    fn handle_request(&mut self, ctx: &egui::Context) {
        use crate::organise::Request;
        let Some(r) = self.ui.request.take() else { return };
        match r {
            Request::HashDuplicates(c) => worker::spawn_duplicates(c, self.tx.clone(), ctx.clone()),
            other => self.confirm = Some(other),
        }
    }

    /// The only modal: confirm a move (or an export), then show its progress.
    fn confirm_modal(&mut self, ctx: &egui::Context) {
        use crate::organise::Request;
        let p = self.ui.pal;
        let screen = ctx.content_rect();
        let busy = self.moving.is_some();
        egui::Area::new(egui::Id::unique("modal-dim")).fixed_pos(screen.min).order(egui::Order::Foreground).show(ctx, |ui| {
            let (r, _) = ui.allocate_exact_size(screen.size(), Sense::click());
            ui.painter().rect_filled(r, CornerRadius::ZERO, Color32::from_rgba_unmultiplied(5, 8, 14, 150));
        });
        let mut go = false;
        let mut cancel = false;
        egui::Area::new(egui::Id::unique("modal")).anchor(egui::Align2::CENTER_CENTER, Vec2::ZERO).order(egui::Order::Tooltip).show(ctx, |ui| {
            egui::Frame::new().fill(p.panel).stroke(Stroke::new(1.0, p.line)).corner_radius(12).inner_margin(20).show(ui, |ui| {
                ui.set_width(540.0);
                let Some(req) = &self.confirm else { return };
                let (title, text, yes): (String, String, &str) = match req {
                    Request::Apply { plan, title } => {
                        let cross = plan.moves.iter().any(|m| m.from.components().next() != m.to.components().next());
                        let how = if cross { "Some go to another drive: each is copied, checked by hash, then the original removed." } else { "Everything is on one drive, so each move is a rename." };
                        let kinds: Vec<String> = [crux_core::plan::MoveKind::Reject, crux_core::plan::MoveKind::Retire, crux_core::plan::MoveKind::Organise, crux_core::plan::MoveKind::Reclaim]
                            .iter()
                            .filter(|k| plan.count(**k) > 0)
                            .map(|k| format!("{} {}", plan.count(*k), k.name()))
                            .collect();
                        (
                            title.clone(),
                            format!(
                                "{:.1} GB ({}). {how} Each move is recorded in the journal before it happens, so you can undo it. crux doesn't delete files.",
                                plan.total_bytes() as f64 / 1e9,
                                kinds.join(", ")
                            ),
                            "Move files",
                        )
                    }
                    Request::Undo(e) => (format!("Move {} files back?", e.moves), format!("Reverses the apply from {} using its journal. Folders it created are removed if they're empty.", e.started), "Move them back"),
                    Request::Export { files, dest } => (
                        format!("Link {} files for export?", files.len()),
                        format!("Creates hard links in {}. They use no extra space, and the originals stay where they are. If that folder is on another drive, crux writes a file list instead.", dest.display()),
                        "Export",
                    ),
                    Request::HashDuplicates(_) => return,
                };
                w::h2(ui, title);
                ui.add_space(6.0);
                ui.add(egui::Label::new(egui::RichText::new(text).color(p.muted)).wrap());
                if let Some((d, t, f)) = &self.moving {
                    ui.add_space(10.0);
                    let (r, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 8.0), Sense::hover());
                    ui.painter().rect_filled(r, CornerRadius::same(4), p.line);
                    let fr = if *t > 0 { *d as f32 / *t as f32 } else { 0.0 };
                    ui.painter().rect_filled(Rect::from_min_size(r.min, Vec2::new(r.width() * fr, 8.0)), CornerRadius::same(4), p.good);
                    ui.label(egui::RichText::new(format!("{d}/{t} · {}", f.display())).font(mono(11.5)).color(p.muted));
                }
                ui.add_space(12.0);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if w::button_enabled(ui, &p, yes, Kind::Primary, None, !busy).clicked() {
                        go = true;
                    }
                    if w::button_enabled(ui, &p, "Cancel", Kind::Normal, None, !busy).clicked() {
                        cancel = true;
                    }
                });
            });
        });
        if !busy && ctx.input(|i| i.key_pressed(Key::Escape)) {
            cancel = true;
        }
        if cancel {
            self.confirm = None;
        }
        if go {
            self.start_confirmed(ctx);
        }
    }

    fn start_confirmed(&mut self, ctx: &egui::Context) {
        use crate::organise::Request;
        let (root, plans) = match (&self.session, &self.ui.lib.view) {
            (Some(s), _) => (s.root.clone(), s.cache_dir.clone().unwrap_or_else(std::env::temp_dir).join("plans")),
            (None, Some(v)) => (v.root.clone(), crate::libview::plans_dir(self.ui.lib.cache_override.as_deref())),
            _ => return,
        };
        match self.confirm.take() {
            Some(Request::Apply { plan, title }) => {
                let path = plans.join(format!("plan-{}.json", chrono::Local::now().format("%Y%m%d-%H%M%S")));
                self.moving = Some((0, plan.moves.len(), PathBuf::new()));
                self.confirm = Some(Request::Apply { plan: plan.clone(), title });
                worker::spawn_move(Some((plan, path)), None, root, self.tx.clone(), ctx.clone());
            }
            Some(Request::Undo(e)) => {
                self.moving = Some((0, e.moves, PathBuf::new()));
                let j = e.journal.clone();
                self.confirm = Some(Request::Undo(e));
                worker::spawn_move(None, Some(j), root, self.tx.clone(), ctx.clone());
            }
            Some(Request::Export { files, dest }) => worker::spawn_export(files, dest, self.tx.clone(), ctx.clone()),
            _ => {}
        }
    }

    /// One rail button. Views that need a folder are greyed and inert
    /// without one.
    fn rail_item(&mut self, ui: &mut egui::Ui, r: Rect, v: View, label: &str, icon: Icon) {
        let p = self.ui.pal;
        let enabled = self.session.is_some() || !v.needs_folder();
        let resp = ui.interact(r, ui.scope_id().with(("rail", label)), if enabled { Sense::click() } else { Sense::hover() });
        // Without a folder the content falls back to Home, so mark Home.
        let shown = if enabled || !self.ui.view.needs_folder() { self.ui.view } else { View::Home };
        let on = enabled && shown == v;
        let hot = enabled && resp.hovered();
        let pa = ui.painter();
        if on || hot {
            pa.rect_filled(r, CornerRadius::same(10), p.panel2);
        }
        if on {
            pa.rect_filled(Rect::from_min_size(Pos2::new(r.left() - 6.0, r.top() + 10.0), Vec2::new(3.0, r.height() - 20.0)), CornerRadius::same(2), p.accent);
        }
        let col = if !enabled { alpha(p.faint, 0.6) } else if on || hot { p.fg } else { p.muted };
        draw_icon(pa, icon, Rect::from_center_size(Pos2::new(r.center().x, r.top() + 20.0), Vec2::splat(20.0)), col);
        pa.text(Pos2::new(r.center().x, r.bottom() - 13.0), egui::Align2::CENTER_CENTER, label, body(11.0), col);
        let inbox_n = if v == View::Inbox { self.session.as_ref().map(|s| s.inbox().len()).unwrap_or(0) } else { 0 };
        if inbox_n > 0 {
            let b = Rect::from_center_size(Pos2::new(r.right() - 16.0, r.top() + 9.0), Vec2::new(if inbox_n > 9 { 22.0 } else { 16.0 }, 16.0));
            pa.rect_filled(b, CornerRadius::same(8), p.rej);
            pa.text(b.center(), egui::Align2::CENTER_CENTER, inbox_n.to_string(), semibold(10.0), Color32::WHITE);
        }
        if !enabled {
            resp.on_hover_text("Open a folder first");
        } else if resp.on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
            self.ui.view = v;
        }
    }

    fn rail(&mut self, ui: &mut egui::Ui) {
        let p = self.ui.pal;
        ui.add_space(14.0);
        ui.vertical_centered(|ui| {
            let (r, _) = ui.allocate_exact_size(Vec2::new(70.0, 30.0), Sense::hover());
            let pa = ui.painter();
            let c = Pos2::new(r.left() + 16.0, r.center().y);
            let st = Stroke::new(2.0, p.accent);
            for (a, b) in [((0.0, -10.0), (0.0, -4.0)), ((0.0, 4.0), (0.0, 10.0)), ((-10.0, 0.0), (-4.0, 0.0)), ((4.0, 0.0), (10.0, 0.0))] {
                pa.line_segment([c + Vec2::new(a.0, a.1), c + Vec2::new(b.0, b.1)], st);
            }
            pa.circle_filled(c, 2.2, p.fg);
            pa.text(Pos2::new(r.left() + 31.0, r.center().y), egui::Align2::LEFT_CENTER, "crux", semibold(18.0), p.fg);
            ui.add_space(10.0);
        });
        // The pipeline steps, top down.
        let x = ui.max_rect().center().x;
        let mut y = ui.cursor().top();
        for (v, label, icon) in [
            (View::Home, "Home", Icon::Home),
            (View::Inbox, "Inbox", Icon::Inbox),
            (View::Frames, "Frames", Icon::Frames),
            (View::Calibrate, "Calibrate", Icon::Calibrate),
            (View::Image, "Image", Icon::Image),
        ] {
            self.rail_item(ui, Rect::from_min_size(Pos2::new(x - 32.0, y), Vec2::new(64.0, 56.0)), v, label, icon);
            y += 60.0;
        }
        // Tools that aren't steps, bottom up: keyboard help, Settings, Organise.
        let mut b = ui.max_rect().bottom() - 14.0;
        let kr = Rect::from_center_size(Pos2::new(x, b - 16.0), Vec2::splat(32.0));
        let resp = ui.interact(kr, ui.scope_id().with("rail-keys"), Sense::click()).on_hover_text("Keyboard shortcuts (?)");
        let pa = ui.painter();
        pa.rect(kr, CornerRadius::same(8), if resp.hovered() { p.panel2 } else { Color32::TRANSPARENT }, Stroke::new(1.0, p.line), StrokeKind::Inside);
        draw_icon(pa, Icon::Keyboard, kr.shrink(7.0), if resp.hovered() { p.fg } else { p.muted });
        if resp.clicked() {
            self.help = true;
        }
        b = kr.top() - 10.0;
        for (v, label, icon) in [(View::Settings, "Settings", Icon::Settings), (View::Organise, "Organise", Icon::Organise)] {
            let r = Rect::from_min_size(Pos2::new(x - 32.0, b - 56.0), Vec2::new(64.0, 56.0));
            // Keep clear of the steps on a short window.
            if r.top() >= y {
                self.rail_item(ui, r, v, label, icon);
            }
            b -= 60.0;
        }
    }

    fn top(&mut self, ui: &mut egui::Ui) {
        let p = self.ui.pal;
        ui.horizontal(|ui| {
            ui.set_min_height(34.0);
            ui.spacing_mut().item_spacing.x = 8.0;
            let root = self.session.as_ref().map(|s| s.root.clone()).or(self.ui.loading.clone()).or(self.ui.lib.view.as_ref().map(|v| v.root.clone()));
            match &root {
                Some(r) => {
                    // The parent opens on click when it's one of your
                    // library folders (or inside one): back to the library.
                    let cfg = self.session.as_ref().map(|s| &s.cfg).or(self.ui.lib.view.as_ref().map(|v| &v.cfg));
                    let lib_parent = cfg.and_then(|c| crate::worker::library_parent(r, c));
                    let parent = r.parent().map(|x| x.display().to_string()).unwrap_or_default();
                    let name = r.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                    match &lib_parent {
                        Some(lp) => {
                            if ui.link(egui::RichText::new(parent).font(body(13.0)).color(p.muted)).on_hover_text("Back to the library").clicked() {
                                self.ui.open_request = Some(lp.clone());
                            }
                        }
                        None => {
                            ui.label(egui::RichText::new(parent).font(body(13.0)).color(p.muted));
                        }
                    }
                    ui.label(egui::RichText::new("›").font(body(13.0)).color(p.muted));
                    let shown = if self.ui.lib.view.is_some() && self.session.is_none() { format!("{} (library)", name) } else { crate::home::pretty_name(&name) };
                    ui.label(egui::RichText::new(shown).font(semibold(13.0)).color(p.fg));
                }
                None => {
                    ui.label(egui::RichText::new("No folder open").font(body(13.0)).color(p.muted));
                }
            }
            ui.add_space(20.0);
            if let Some(s) = &self.session {
                let c = s.counts(0..s.lights.len());
                let cal_bad = s.triage.cal.sets.iter().filter(|x| x.grade == Grade::Reject).count();
                let stages: [(View, &str, Option<Color32>, String); 5] = [
                    (View::Frames, "Frames", Some(if c[2] > 0 { p.warn } else { p.good }), format!("{} to review", c[1] + c[2])),
                    (View::Calibrate, "Calibration", Some(if cal_bad > 0 { p.rej } else { p.good }), if cal_bad > 0 { format!("{cal_bad} problems") } else { "ok".into() }),
                    (View::Organise, "Organise", Some(if s.plan_moves > 0 { p.warn } else { p.good }), if s.plan_moves > 0 { format!("{} moves planned", s.plan_moves) } else { "no moves".into() }),
                    (View::Image, "Stack", Some(if self.ui.stack.report.as_ref().is_some_and(|r| r.is_ok()) || self.ui.stack.pending.is_some_and(|pd| pd.is_empty()) { p.good } else { p.faint }), crate::stack::strip_line(&self.ui)),
                    (View::Image, "Image", Some(if matches!(self.ui.stack.img.written, Some(Ok(_))) { p.good } else if self.ui.stack.img.recipe.is_some() { p.accent } else { p.faint }), self.ui.stack.img.recipe.as_ref().map(|r| if r.name.is_empty() { "Custom".to_string() } else { r.name.clone() }).unwrap_or_else(|| "not started".into())),
                ];
                for (i, (v, name, dot, st)) in stages.into_iter().enumerate() {
                    if i > 0 {
                        ui.label(egui::RichText::new("›").color(p.faint));
                    }
                    let img_stage = self.ui.stack.stage.is_image();
                    let on = self.ui.view == v && (if name == "Image" { img_stage } else if name == "Stack" { !img_stage } else { true });
                    let g1 = ui.painter().layout_no_wrap(name.to_string(), body(13.0), p.fg);
                    let compact = ui.ctx().content_rect().width() < 1360.0;
                    let g2 = ui.painter().layout_no_wrap(if compact { String::new() } else { st.clone() }, body(11.0), p.faint);
                    let wd = 12.0 + 8.0 + 8.0 + g1.size().x + 8.0 + g2.size().x + 12.0;
                    let (r, resp) = ui.allocate_exact_size(Vec2::new(wd, 30.0), Sense::click());
                    let pa = ui.painter();
                    if on {
                        pa.rect(r, CornerRadius::same(8), p.panel2, Stroke::new(1.0, p.line), StrokeKind::Inside);
                    } else if resp.hovered() {
                        pa.rect_filled(r, CornerRadius::same(8), p.panel2);
                    }
                    let c0 = Pos2::new(r.left() + 16.0, r.center().y);
                    match dot {
                        Some(col) => pa.circle_filled(c0, 4.0, col),
                        None => pa.circle_stroke(c0, 3.5, Stroke::new(1.5, p.faint)),
                    };
                    let fg = if on || resp.hovered() { p.fg } else { p.muted };
                    let x1 = r.left() + 28.0;
                    let g1w = g1.size().x;
                    pa.galley_with_override_text_color(Pos2::new(x1, r.center().y - g1.size().y / 2.0), g1, fg);
                    pa.galley(Pos2::new(x1 + g1w + 8.0, r.center().y - g2.size().y / 2.0 + 1.0), g2, p.faint);
                    if resp.on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
                        self.ui.view = v;
                        if name == "Image" && !img_stage {
                            self.ui.stack.stage = crate::stack::Stage::Palette;
                        } else if name == "Stack" && img_stage {
                            self.ui.stack.stage = crate::stack::Stage::Calibrate;
                        }
                    }
                }
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let (txt, live) = self.job_text();
                let g = ui.painter().layout_no_wrap(txt, body(12.0), p.muted);
                let (r, resp) = ui.allocate_exact_size(Vec2::new(g.size().x + 40.0, 30.0), Sense::click());
                let pa = ui.painter();
                pa.rect(r, CornerRadius::same(15), p.panel2, Stroke::new(1.0, p.line), StrokeKind::Inside);
                let c = Pos2::new(r.left() + 16.0, r.center().y);
                let col = if live { p.accent } else { p.good };
                if live {
                    let t = ui.ctx().input(|i| i.time) as f32;
                    let ph = (t * 0.5).fract();
                    pa.circle_filled(c, 3.5 + 6.0 * ph, alpha(col, 0.5 * (1.0 - ph)));
                    ui.ctx().request_repaint();
                }
                pa.circle_filled(c, 3.5, col);
                pa.galley(Pos2::new(r.left() + 28.0, r.center().y - g.size().y / 2.0), g, p.muted);
                if resp.on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
                    self.drawer = !self.drawer;
                }
                self.compute_badge(ui);
            });
        });
    }

    /// GPU or CPU: where the networks, warps and other heavy steps run.
    /// Click for Settings, where the graphics card can be switched off.
    fn compute_badge(&mut self, ui: &mut egui::Ui) {
        use crux_core::gpu::Compute;
        let p = self.ui.pal;
        let (label, col, tip) = match crux_core::gpu::compute_now() {
            Some(Compute::Gpu(name)) => ("GPU", p.good, format!("Using the graphics card: {name}")),
            Some(Compute::Cpu(why)) => ("CPU", p.muted, format!("Using the CPU: {why}")),
            None => {
                ui.ctx().request_repaint_after(std::time::Duration::from_millis(250));
                ("…", p.faint, "Looking for a graphics card".to_string())
            }
        };
        let g = ui.painter().layout_no_wrap(label.to_string(), semibold(11.5), col);
        let (r, resp) = ui.allocate_exact_size(Vec2::new(g.size().x + 40.0, 30.0), Sense::click());
        let pa = ui.painter();
        pa.rect(r, CornerRadius::same(15), if resp.hovered() { p.panel2 } else { Color32::TRANSPARENT }, Stroke::new(1.0, p.line), StrokeKind::Inside);
        draw_icon(pa, Icon::Chip, Rect::from_center_size(Pos2::new(r.left() + 17.0, r.center().y), Vec2::splat(16.0)), col);
        pa.galley(Pos2::new(r.left() + 29.0, r.center().y - g.size().y / 2.0), g, col);
        if resp.on_hover_text(format!("{tip}. Change it in Settings.")).on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
            self.ui.view = View::Settings;
        }
    }

    fn job_text(&self) -> (String, bool) {
        if let Some(e) = &self.ui.failed {
            return (format!("Couldn't open: {}", e.chars().take(60).collect::<String>()), false);
        }
        if self.ui.loading.is_some() {
            return match self.ui.progress {
                Some((Stage::Pixels, d, t)) => (format!("Measuring lights {d}/{t}"), true),
                Some((Stage::Calibration, d, t)) => (format!("Checking calibration {d}/{t}"), true),
                Some((Stage::Headers, d, t)) => (format!("Reading headers {d}/{t}"), true),
                None => ("Reading headers".into(), true),
            };
        }
        match &self.session {
            Some(s) => (format!("Ready · {} lights graded", s.lights.len()), false),
            None => ("Idle".into(), false),
        }
    }

    pub fn jobs(&self) -> Vec<Job> {
        let mut v = Vec::new();
        if let Some(f) = &self.ui.facts {
            v.push(Job { label: format!("Header scan, {} files", f.files), detail: format!("{:.2} s", f.seconds), progress: None, running: false });
        }
        if self.ui.loading.is_some() {
            if let Some((st, d, t)) = self.ui.progress {
                let label = match st {
                    Stage::Pixels => "Measuring lights",
                    Stage::Calibration => "Checking calibration frames",
                    Stage::Headers => "Reading headers",
                };
                v.push(Job { label: label.into(), detail: format!("{d}/{t}"), progress: Some((d, t)), running: true });
            }
        }
        if let Some(s) = &self.session {
            let t = &s.timings;
            if self.ui.facts.is_none() {
                v.push(Job { label: format!("Header scan, {} files", t.header_files), detail: format!("{:.2} s", t.header_s), progress: None, running: false });
            }
            v.push(Job {
                label: format!("Checked {} calibration frames", t.cal_frames),
                detail: format!("{:.1} s · {} measured", t.cal_s, t.cal_measured),
                progress: Some((1, 1)),
                running: false,
            });
            v.push(Job {
                label: format!("Measured {} lights", t.lights),
                detail: format!("{:.1} s · {} measured, rest cached", t.lights_s, t.lights_measured),
                progress: Some((1, 1)),
                running: false,
            });
            v.push(Job { label: "Transparency (star matching)".into(), detail: format!("{:.2} s", t.transparency_s), progress: None, running: false });
            v.push(Job { label: "Regrade with your limits".into(), detail: format!("{:.0} ms", s.regrade_ms), progress: None, running: false });
        }
        if !self.ui.folders.is_empty() {
            v.push(Job { label: format!("Headers of {} folders beside it", self.ui.folders.len()), detail: format!("{:.1} s", self.ui.folders_s), progress: None, running: false });
        }
        v.push(Job {
            label: format!("Full-resolution frames on the GPU: {}", self.ui.resident.len()),
            detail: if self.ui.load_ms.is_empty() {
                String::new()
            } else {
                let mut v = self.ui.load_ms.clone();
                v.sort_by(f32::total_cmp);
                format!("{:.0} ms each", v[v.len() / 2])
            },
            progress: None,
            running: false,
        });
        v
    }

    fn activity(&mut self, ctx: &egui::Context) {
        let p = self.ui.pal;
        let jobs = self.jobs();
        let screen = ctx.content_rect();
        let resp = egui::Area::new(egui::Id::unique("activity"))
            .fixed_pos(Pos2::new(screen.right() - 16.0 - 360.0, 52.0))
            .order(egui::Order::Foreground)
            .show(ctx, |ui| {
                egui::Frame::new().fill(p.panel).stroke(Stroke::new(1.0, p.line)).corner_radius(12).inner_margin(12).show(ui, |ui| {
                    ui.set_width(336.0);
                    w::eyebrow(ui, &p, "Activity");
                    for j in &jobs {
                        ui.add_space(4.0);
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new(&j.label).font(body(12.5)));
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.label(egui::RichText::new(&j.detail).font(mono(11.5)).color(p.muted));
                            });
                        });
                        if let Some((d, t)) = j.progress {
                            let (r, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 4.0), Sense::hover());
                            ui.painter().rect_filled(r, CornerRadius::same(2), p.line);
                            let f = if t > 0 { d as f32 / t as f32 } else { 0.0 };
                            ui.painter().rect_filled(Rect::from_min_size(r.min, Vec2::new(r.width() * f, 4.0)), CornerRadius::same(2), if j.running { p.accent } else { p.good });
                        }
                        ui.add_space(2.0);
                        let (r, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 1.0), Sense::hover());
                        ui.painter().rect_filled(r, CornerRadius::ZERO, p.line);
                    }
                    if self.ui.loading.is_some() {
                        ui.add_space(6.0);
                        if w::button(ui, &p, "Cancel", Kind::Danger, None).clicked() {
                            self.ui.cancel.store(true, Ordering::Relaxed);
                        }
                    }
                    if let Some(s) = &self.session {
                        if let Some(d) = &s.cache_dir {
                            ui.add_space(6.0);
                            ui.label(egui::RichText::new(format!("Cache and your decisions: {}", d.display())).font(body(11.5)).color(p.muted));
                        }
                    }
                });
            });
        if ctx.input(|i| i.pointer.any_click()) && !resp.response.hovered() && !ctx.is_pointer_over_egui() {
            self.drawer = false;
        }
        if ctx.input(|i| i.key_pressed(Key::Escape)) {
            self.drawer = false;
        }
    }

    fn toasts(&mut self, ctx: &egui::Context) {
        let now = ctx.input(|i| i.time);
        let Some(t) = &self.ui.toast else { return };
        if now > t.until {
            self.ui.toast = None;
            return;
        }
        ctx.request_repaint_after(std::time::Duration::from_millis(250));
        let p = self.ui.pal;
        let (msg, undo) = (t.msg.clone(), t.undo);
        let screen = ctx.content_rect();
        let mut clicked_undo = false;
        egui::Area::new(egui::Id::unique("toast"))
            .anchor(egui::Align2::CENTER_BOTTOM, Vec2::new(38.0, -24.0))
            .order(egui::Order::Foreground)
            .show(ctx, |ui| {
                egui::Frame::new().fill(p.fg).corner_radius(10).inner_margin(egui::Margin::symmetric(14, 10)).show(ui, |ui| {
                    ui.set_max_width(screen.width() - 64.0);
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new(&msg).color(p.bg).font(body(13.5)));
                        if undo {
                            ui.add_space(8.0);
                            let r = ui.add(egui::Label::new(egui::RichText::new("Undo").color(p.bg).font(semibold(13.5)).underline()).sense(Sense::click()));
                            if r.on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
                                clicked_undo = true;
                            }
                        }
                    });
                });
            });
        if clicked_undo {
            if let Some(s) = &mut self.session {
                s.undo_last();
            }
            self.ui.toast = None;
        }
    }

    fn help_sheet(&mut self, ctx: &egui::Context) {
        let p = self.ui.pal;
        let screen = ctx.content_rect();
        egui::Area::new(egui::Id::unique("help-dim")).fixed_pos(screen.min).order(egui::Order::Foreground).show(ctx, |ui| {
            let (r, resp) = ui.allocate_exact_size(screen.size(), Sense::click());
            ui.painter().rect_filled(r, CornerRadius::ZERO, Color32::from_rgba_unmultiplied(5, 8, 14, 150));
            if resp.clicked() {
                self.help = false;
            }
        });
        egui::Area::new(egui::Id::unique("help")).anchor(egui::Align2::CENTER_CENTER, Vec2::ZERO).order(egui::Order::Tooltip).show(ctx, |ui| {
            egui::Frame::new().fill(p.panel).stroke(Stroke::new(1.0, p.line)).corner_radius(12).inner_margin(20).show(ui, |ui| {
                ui.set_width(440.0);
                w::h2(ui, "Keyboard");
                ui.add_space(8.0);
                let rows = [
                    ("J / K", "Next / previous frame"),
                    ("A", "Keep (pinned)"),
                    ("R", "Reject (pinned)"),
                    ("U", "Undo"),
                    ("Space", "Blink with the reference"),
                    ("D", "Difference from the reference"),
                    ("1", "Star and trail marks"),
                    ("2", "Cloud grid"),
                    ("F", "Fit the frame (double-click: 1:1)"),
                    ("G then H/I/F/C/O", "Home, Inbox, Frames, Calibrate, Organise"),
                    ("?", "This list"),
                ];
                egui::Grid::new("keys").num_columns(2).spacing([14.0, 8.0]).show(ui, |ui| {
                    for (k, t) in rows {
                        w::kbd(ui, &p, k);
                        ui.label(t);
                        ui.end_row();
                    }
                });
                ui.add_space(10.0);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if w::button(ui, &p, "Close", Kind::Primary, None).clicked() {
                        self.help = false;
                    }
                });
            });
        });
    }
}

#[derive(Clone, Copy)]
enum Icon {
    Home,
    Inbox,
    Frames,
    Organise,
    Calibrate,
    Image,
    Settings,
    Keyboard,
    Chip,
}

/// The mockup's rail icons, drawn on a 24-unit grid.
fn draw_icon(pa: &egui::Painter, icon: Icon, r: Rect, c: Color32) {
    let s = r.width() / 24.0;
    let pt = |x: f32, y: f32| Pos2::new(r.left() + x * s, r.top() + y * s);
    let st = Stroke::new(1.7, c);
    let poly = |pts: &[(f32, f32)], closed: bool| {
        let v: Vec<Pos2> = pts.iter().map(|&(x, y)| pt(x, y)).collect();
        if closed {
            pa.add(egui::Shape::closed_line(v, st));
        } else {
            pa.add(egui::Shape::line(v, st));
        }
    };
    let rect = |x: f32, y: f32, wd: f32, ht: f32, rad: f32| {
        pa.rect_stroke(Rect::from_min_size(pt(x, y), Vec2::new(wd * s, ht * s)), CornerRadius::same((rad * s) as u8), st, StrokeKind::Middle);
    };
    match icon {
        Icon::Home => poly(&[(3.0, 11.0), (12.0, 4.0), (21.0, 11.0), (21.0, 20.0), (15.0, 20.0), (15.0, 14.0), (9.0, 14.0), (9.0, 20.0), (3.0, 20.0)], true),
        Icon::Inbox => {
            poly(&[(3.0, 13.0), (6.0, 5.0), (18.0, 5.0), (21.0, 13.0), (21.0, 20.0), (3.0, 20.0)], true);
            poly(&[(3.0, 13.0), (8.0, 13.0), (9.0, 16.0), (15.0, 16.0), (16.0, 13.0), (21.0, 13.0)], false);
        }
        Icon::Frames => {
            rect(3.0, 4.0, 8.0, 7.0, 1.0);
            rect(13.0, 4.0, 8.0, 7.0, 1.0);
            rect(3.0, 13.0, 8.0, 7.0, 1.0);
            rect(13.0, 13.0, 8.0, 7.0, 1.0);
        }
        Icon::Organise => {
            poly(&[(3.0, 6.0), (9.0, 5.0), (11.0, 7.0), (21.0, 7.0), (21.0, 19.0), (3.0, 19.0)], true);
            poly(&[(8.0, 13.0), (16.0, 13.0)], false);
            poly(&[(12.0, 9.0), (12.0, 17.0)], false);
        }
        Icon::Calibrate => {
            pa.circle_stroke(pt(12.0, 12.0), 8.0 * s, st);
            for (a, b) in [((12.0, 4.0), (12.0, 8.0)), ((12.0, 16.0), (12.0, 20.0)), ((4.0, 12.0), (8.0, 12.0)), ((16.0, 12.0), (20.0, 12.0))] {
                poly(&[a, b], false);
            }
        }
        Icon::Image => {
            rect(3.0, 3.0, 18.0, 18.0, 2.0);
            poly(&[(4.0, 18.0), (9.0, 12.0), (13.0, 16.0), (16.0, 13.0), (20.0, 18.0)], false);
            pa.circle_stroke(pt(16.0, 8.0), 2.0 * s, st);
        }
        Icon::Settings => {
            // Gear: eight teeth around a ring, a hole in the middle.
            let n = 8;
            let mut v = Vec::new();
            for i in 0..n * 4 {
                let a = std::f32::consts::TAU * (i as f32 - 0.5) / (n * 4) as f32;
                let r = if (i % 4) < 2 { 9.0 } else { 6.6 };
                v.push(pt(12.0 + r * a.cos(), 12.0 + r * a.sin()));
            }
            pa.add(egui::Shape::closed_line(v, st));
            pa.circle_stroke(pt(12.0, 12.0), 2.8 * s, st);
        }
        Icon::Keyboard => {
            rect(2.0, 6.0, 20.0, 12.0, 2.0);
            for x in [6.0, 10.0, 14.0, 18.0] {
                pa.circle_filled(pt(x, 10.0), 0.9 * s, c);
            }
            poly(&[(7.0, 14.0), (17.0, 14.0)], false);
        }
        Icon::Chip => {
            // A processor: a square with pins on every side.
            rect(6.0, 6.0, 12.0, 12.0, 2.0);
            rect(9.5, 9.5, 5.0, 5.0, 0.5);
            for k in [9.5, 14.5] {
                poly(&[(k, 2.5), (k, 6.0)], false);
                poly(&[(k, 18.0), (k, 21.5)], false);
                poly(&[(2.5, k), (6.0, k)], false);
                poly(&[(18.0, k), (21.5, k)], false);
            }
        }
    }
}

pub fn grade_name(g: Grade) -> &'static str {
    match g {
        Grade::Good => "Good",
        Grade::Warn => "Warn",
        Grade::Reject => "Reject",
    }
}
