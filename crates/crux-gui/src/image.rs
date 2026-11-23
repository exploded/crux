//! Image stages (milestone 13) on the Stack page: Sharpen and denoise
//! (crux's networks or RC Astro on the linear masters, `render::enhance`;
//! its own 1:1 original | enhanced view of the centre or a corner),
//! Gradient, Palette and stretch, Finish, Stars (classic fill or the star
//! remover network), Crop and save. The palette mixer's layout: the picture as the hero
//! with a before/after split, preset tiles rendered from your own data,
//! sliders at three depths (Auto, Adjust, Expert), each showing where its
//! value comes from (crux's default, a recipe, or changed here).
//!
//! Masters are read and gradient-corrected once in the background
//! (`render::prepare`); every slider then re-renders the 1,200 px proxy on
//! the UI thread with the same code the full-size image uses.

use crate::app::UiState;
use crate::session::Session;
use crate::stack::Stage;
use crate::theme::{body, mono, semibold, Pal};
use crate::widgets::{self as w, Kind};
use crate::worker::Msg;
use crux_core::render::{self, gradient, palette, recipe, recipe::Recipe, Prepared, Target};
use eframe::egui::{self, Color32, Pos2, Rect, RichText, Sense, Stroke, Ui, Vec2};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;

/// Longest side of a preset tile in the GUI.
const TILE: usize = 220;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Depth {
    Auto,
    Adjust,
    Expert,
}

pub struct ImageUi {
    pub targets: Vec<Target>,
    targets_key: String,
    pub sel: usize,
    pub recipe: Option<Recipe>,
    /// What the values are compared with, and what to call it.
    base: Option<(Recipe, String)>,
    prepared: Option<Arc<Prepared>>,
    prep_key: String,
    preparing: Option<String>,
    pub error: Option<String>,
    view_key: String,
    before: Option<egui::TextureHandle>,
    after: Option<egui::TextureHandle>,
    labels: (String, String),
    tiles: Vec<(String, egui::TextureHandle)>,
    tiles_key: String,
    /// The preview renders in the background, one at a time, so sliders
    /// never wait for it: whether one is running, and its result waiting
    /// to become textures.
    view_busy: bool,
    view_pending: Option<Box<ViewDone>>,
    /// Preset tiles likewise, started once the settings have held still
    /// for `TILES_SETTLE` (the key and when it last changed).
    tiles_busy: bool,
    tiles_pending: Option<(String, Vec<(String, render::Rgb)>)>,
    tiles_seen: (String, std::time::Instant),
    pub depth: Depth,
    /// Gradient stage: the channel shown, and whether to show its surface.
    pub gchan: usize,
    pub gmodel: bool,
    pub gsamples: bool,
    /// Stars stage: 0 whole image, 1 the 1:1 crop, 2 starless.
    pub sview: u8,
    pub writing: bool,
    pub written: Option<Result<render::Written, String>>,
    save_name: String,
    saved: Option<Vec<(String, PathBuf)>>,
    pub split: f32,
    /// Pinned layout: the preview's height limit this frame, and the share
    /// of the column it gets (dragged with the handle under it).
    pin_h: Option<f32>,
    view_frac: f32,
    /// Palette stage: whether the preset pictures are shown.
    tiles_open: bool,
    cancel: Arc<AtomicBool>,
    /// Set by the screenshot harness.
    pub pending_preset: Option<String>,
    pub pending_write: bool,
    pub render_ms: f64,
    /// `[process] fits`: also write a FITS copy.
    fits: bool,
    /// Crop stage: the drag in progress (grip, crop and pointer at the
    /// start), the shape chip chosen and its ratio (pixels, width ÷
    /// height), and a note from the last button.
    crop_drag: Option<(Grip, recipe::Crop, (f64, f64))>,
    crop_shape: usize,
    crop_ratio: Option<f64>,
    crop_note: Option<String>,
    /// Harness: trim the empty edges once the masters are read.
    pub pending_autocrop: bool,
    /// What the background preparation is doing (enhance progress).
    pub progress: Option<String>,
    /// Sharpen and denoise stage: before and after crops, their key, the
    /// request in flight, where to look (0 centre, 1 top-left corner), and
    /// whether each engine can run (crux, RC Astro; checked once).
    enh_view: Option<(egui::TextureHandle, egui::TextureHandle)>,
    enh_pending: Option<(String, Result<(render::Rgb, render::Rgb), String>)>,
    enh_key: String,
    enh_loading: Option<String>,
    enh_error: Option<String>,
    pub ewhere: u8,
    enh_avail: Option<[(Result<(), String>, Result<(), String>); 2]>,
    /// Harness: steps to switch on ("denoise", "correct", "tighten").
    pub pending_enhance: Vec<&'static str>,
}

impl ImageUi {
    /// Harness: the preview's share of the column (`img-view:<f>`).
    pub fn set_view_frac(&mut self, f: f32) {
        self.view_frac = f.clamp(0.15, 0.85);
    }

    /// Harness: fold the preset pictures away (`img-fold`).
    pub fn fold_presets(&mut self) {
        self.tiles_open = false;
    }
}

/// Pinned layout: the share of the column the preview starts with.
const VIEW_FRAC: f32 = 0.6;

/// A preview rendered in the background: key, before, after, labels, ms.
pub type ViewDone = (String, render::Rgb, render::Rgb, (String, String), f64);

/// How long the settings must hold still before the preset tiles render
/// again (nine renders per change made sliders stick).
const TILES_SETTLE: std::time::Duration = std::time::Duration::from_millis(250);

/// The preview's height limit: the pinned layout's, else a share of the
/// window (`frac`).
fn max_view_h(ui: &Ui, img: &ImageUi, frac: f32) -> f32 {
    img.pin_h.unwrap_or_else(|| (ui.ctx().content_rect().height() * frac).max(280.0))
}

/// What a drag on the crop view moves.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Grip {
    Move,
    /// A new box from this corner (fractions).
    New(f64, f64),
    /// Edges: x −1 left, 1 right, 0 neither; y likewise (both: a corner).
    Edge(i8, i8),
}

/// Shape chips: name and landscape ratio (None free, 0 the picture's own).
const SHAPES: [(&str, Option<f64>); 6] = [("Free", None), ("Original", Some(0.0)), ("1:1", Some(1.0)), ("4:3", Some(4.0 / 3.0)), ("3:2", Some(1.5)), ("16:9", Some(16.0 / 9.0))];

impl Default for ImageUi {
    fn default() -> Self {
        ImageUi {
            targets: Vec::new(),
            targets_key: String::new(),
            sel: 0,
            recipe: None,
            base: None,
            prepared: None,
            prep_key: String::new(),
            preparing: None,
            error: None,
            view_key: String::new(),
            before: None,
            after: None,
            labels: (String::new(), String::new()),
            tiles: Vec::new(),
            tiles_key: String::new(),
            view_busy: false,
            view_pending: None,
            tiles_busy: false,
            tiles_pending: None,
            tiles_seen: (String::new(), std::time::Instant::now()),
            depth: Depth::Adjust,
            gchan: 0,
            gmodel: false,
            gsamples: true,
            sview: 1,
            writing: false,
            written: None,
            save_name: String::new(),
            saved: None,
            split: 0.5,
            pin_h: None,
            view_frac: VIEW_FRAC,
            tiles_open: true,
            cancel: Arc::new(AtomicBool::new(false)),
            pending_preset: None,
            pending_write: false,
            render_ms: 0.0,
            fits: false,
            crop_drag: None,
            crop_shape: 0,
            crop_ratio: None,
            crop_note: None,
            pending_autocrop: false,
            progress: None,
            enh_view: None,
            enh_pending: None,
            enh_key: String::new(),
            enh_loading: None,
            enh_error: None,
            ewhere: 0,
            enh_avail: None,
            pending_enhance: Vec::new(),
        }
    }
}

impl ImageUi {
    pub fn ready_for_shot(&self) -> bool {
        self.after.is_some() && (self.prep_key.is_empty() || self.preparing.is_none()) && !self.writing && !self.pending_write && !self.view_busy && self.view_pending.is_none() && !self.tiles_busy && self.tiles_pending.is_none() || self.error.is_some()
    }

    /// Preparing, enhancing or writing in the background (files in the
    /// workspace are in use).
    pub fn busy(&self) -> bool {
        self.writing || self.preparing.is_some() || self.enh_loading.is_some()
    }

    /// Sharpen and denoise stage: its crops are on screen (or failed).
    pub fn enhance_ready(&self) -> bool {
        (self.enh_view.is_some() && self.enh_loading.is_none() && self.preparing.is_none() && self.pending_enhance.is_empty()) || self.enh_error.is_some() || self.error.is_some()
    }

    pub fn handle(&mut self, m: Msg) -> Option<Msg> {
        match m {
            Msg::ImageProgress(t) => self.progress = Some(t),
            Msg::ImageView(v) => {
                self.view_busy = false;
                self.view_pending = Some(v);
            }
            Msg::ImageTiles(key, t) => {
                self.tiles_busy = false;
                self.tiles_pending = Some((key, t));
            }
            Msg::ImageEnhanceView(key, r) => {
                self.enh_loading = None;
                self.enh_pending = Some((key, r));
            }
            Msg::ImagePrepared(key, r) => {
                self.preparing = None;
                self.progress = None;
                self.prep_key = key;
                match r {
                    Ok(p) => {
                        self.prepared = Some(p);
                        self.error = None;
                    }
                    Err(e) => {
                        self.prepared = None;
                        self.error = Some(e);
                    }
                }
                self.view_key.clear();
                self.tiles_key.clear();
            }
            Msg::ImageWritten(r) => {
                self.writing = false;
                self.written = Some(r);
            }
            other => return Some(other),
        }
        None
    }

    fn target(&self) -> Option<&Target> {
        self.targets.get(self.sel)
    }

    fn select(&mut self, i: usize) {
        if i != self.sel {
            self.sel = i;
            self.recipe = None;
            self.base = None;
            self.prepared = None;
            self.prep_key.clear();
            self.written = None;
            self.gchan = 0;
            self.crop_note = None;
        }
    }
}

/// The full-size image this recipe makes, when it's already on disk and
/// newer than every master (made in an earlier session). Its time to make
/// isn't known (`seconds` is NaN).
fn on_disk(t: &Target, out: &std::path::Path, r: &Recipe) -> Option<render::Written> {
    let r = named(r);
    let dir = t.image_dir(out);
    let base = render::base_name(&t.object, &r);
    let (xisf, png, jpg) = (dir.join(format!("{base}.xisf")), dir.join(format!("{base}.png")), dir.join(format!("{base}.jpg")));
    let modified = |p: &std::path::Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
    let made = modified(&jpg)?.min(modified(&xisf)?);
    if t.masters.iter().any(|m| modified(&m.path).is_none_or(|x| x > made)) {
        return None;
    }
    let l = crux_core::xisf::read_layout(&xisf).ok()?;
    let fits = crux_core::fits::copy_path(&xisf);
    Some(render::Written { fits: fits.is_file().then_some(fits), xisf, png, jpg, preview: dir.join("preview.png"), recipe: dir.join("recipe.toml"), seconds: f64::NAN, width: l.width, height: l.height })
}

/// The recipe as written: named after its preset when it has no name.
fn named(r: &Recipe) -> Recipe {
    let mut r = r.clone();
    if r.name.is_empty() {
        r.name = if r.mix.preset.is_empty() { "Custom".into() } else { r.mix.preset.clone() };
    }
    r
}

/// Where the full-size image goes and its file name without extension.
fn output_of(img: &ImageUi, out: &std::path::Path) -> Option<(PathBuf, String)> {
    let t = img.target()?;
    let r = img.recipe.as_ref()?;
    Some((t.image_dir(out), render::base_name(&t.object, &named(r))))
}

fn want_key(t: &Target, r: &Recipe) -> String {
    format!("{}|{}|{}|{}", t.object, serde_json::to_string(&r.gradient).unwrap_or_default(), r.stars.mask_key(), enhance_key(&r.enhance))
}

/// The enhance settings that change the masters (none when it's off).
fn enhance_key(e: &render::enhance::EnhanceSettings) -> String {
    if e.active() {
        serde_json::to_string(e).unwrap_or_default()
    } else {
        String::new()
    }
}

/// Sharpen and denoise the masters (cached on disk), reporting progress.
fn enhanced(t: &Target, out: &std::path::Path, e: &render::enhance::EnhanceSettings, passes: &render::Passes, tx: &Sender<Msg>, ctx: &egui::Context, cancel: &AtomicBool) -> anyhow::Result<Target> {
    render::enhance::apply(t, out, e, &|p| {
        let _ = tx.send(Msg::ImageProgress(passes.enhance_line(&p)));
        ctx.request_repaint();
    }, cancel)
}

/// The star remover's starless masters, when chosen for natural colours.
fn with_starless(t: Target, out: &std::path::Path, passes: &render::Passes, tx: &Sender<Msg>, ctx: &egui::Context, cancel: &AtomicBool) -> anyhow::Result<Target> {
    if !passes.starless {
        return Ok(t);
    }
    let names = t.channels();
    render::starnet::apply(&t, out, &|i, _, f| {
        let _ = tx.send(Msg::ImageProgress(passes.starless_line(i, names.get(i).map(String::as_str).unwrap_or(""), f)));
        ctx.request_repaint();
    }, cancel)
}

/// Run a preview render on the UI thread's own rayon pool. On the global
/// pool it queued behind whatever the background was doing (a network pass
/// held a palette change up for 12 s), since rayon workers finish their own
/// queued jobs before taking work from outside the pool.
fn quick<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        let n = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .thread_name(|i| format!("crux-preview-{i}"))
            .build()
            .expect("preview thread pool")
    })
    .install(f)
}

fn tex(ctx: &egui::Context, name: &str, rgb: &render::Rgb) -> egui::TextureHandle {
    let img = egui::ColorImage::from_rgba_unmultiplied([rgb.width, rgb.height], &rgb.rgba8());
    ctx.load_texture(name, img, egui::TextureOptions::LINEAR)
}

/// The recipe with the stars as mixed (the stages before Stars).
fn kept(r: &Recipe) -> Recipe {
    let mut k = r.clone();
    k.stars.mode = render::stars::StarMode::Keep;
    k
}

/// The recipe without finishing (Palette stage "after", Finish "before").
fn unfinished(r: &Recipe) -> Recipe {
    let mut u = r.clone();
    u.finish = palette::Finishing { luminance: 0.0, green: 0.0, saturation: 1.0, detail: 0.0, chroma: 0.0, neutral: 0.0 };
    u
}

/// Keep targets, recipe, prepared channels and the preview up to date.
/// Called every frame while an image stage is shown.
pub fn tick(a: &mut UiState, s: &Session, tx: &Sender<Msg>, ctx: &egui::Context) {
    let Some(plan) = &a.stack.plan else { return };
    let out = plan.out.clone();
    let st = a.stack.stage;
    let img = &mut a.stack.img;
    img.fits = s.cfg.process.fits;
    let tk = format!("{}|{}|{}", out.display(), a.stack.integrations.len(), a.stack.mos.generation);
    if tk != img.targets_key {
        img.targets_key = tk;
        img.targets = render::targets(&out);
        if img.sel >= img.targets.len() {
            img.sel = 0;
        }
        img.recipe = None;
        img.prepared = None;
        img.prep_key.clear();
    }
    let Some(t) = img.target().cloned() else { return };
    let names = t.channels();
    if img.recipe.is_none() {
        let last = t.image_dir(&out).join("recipe.toml");
        let (mut r, label) = match Recipe::load(&last) {
            Ok(r) => {
                let n = r.name.clone();
                (r, format!("recipe “{n}”"))
            }
            Err(_) => (Recipe::for_channels(&names), "default".to_string()),
        };
        r.fit_to(&names);
        // Made in an earlier session with these settings: show it as saved.
        img.written = on_disk(&t, &out, &r).map(Ok);
        img.base = Some((r.clone(), label));
        img.recipe = Some(r);
    }
    let r = img.recipe.clone().unwrap();
    // Read and correct the masters in the background.
    let want = want_key(&t, &r);
    if want != img.prep_key && img.preparing.is_none() {
        img.preparing = Some(want.clone());
        img.cancel = Arc::new(AtomicBool::new(false));
        let (tx, ctx2, cancel, g, stars) = (tx.clone(), ctx.clone(), img.cancel.clone(), r.gradient.clone(), r.stars.clone());
        let (t2, wk, out2, enh) = (t.clone(), want.clone(), out.clone(), r.enhance.clone());
        img.progress = None;
        std::thread::Builder::new()
            .name("crux-image".into())
            .spawn(move || {
                // Sharpened and denoised masters first (cached), then the
                // prepared target from `prepared.cache` when nothing changed.
                let passes = render::Passes::new(&t2, &enh, &stars);
                let res = enhanced(&t2, &out2, &enh, &passes, &tx, &ctx2, &cancel)
                    .and_then(|te| with_starless(te, &out2, &passes, &tx, &ctx2, &cancel))
                    .and_then(|te| render::prepcache::prepare(&te, &out2, &g, &stars, &cancel))
                    .map(Arc::new)
                    .map_err(|e| format!("{e:#}"));
                let _ = tx.send(Msg::ImagePrepared(wk, res));
                ctx2.request_repaint();
            })
            .expect("spawn image");
    }
    // Harness: switch enhance steps on.
    for step in std::mem::take(&mut img.pending_enhance) {
        let rr = img.recipe.as_mut().unwrap();
        match step {
            "denoise" => rr.enhance.denoise = true,
            "correct" => rr.enhance.sharpen = render::enhance::Sharpen::Correct,
            "tighten" => rr.enhance.sharpen = render::enhance::Sharpen::Tighten,
            _ => {}
        }
    }
    if st == Stage::Enhance {
        enhance_tick(img, &t, &out, tx, ctx);
    }
    // Background renders that have landed become textures.
    if let Some(v) = img.view_pending.take() {
        let (key, before, after, labels, ms) = *v;
        img.view_key = key;
        img.render_ms = ms;
        img.before = Some(tex(ctx, "img-before", &before));
        img.after = Some(tex(ctx, "img-after", &after));
        img.labels = labels;
    }
    if let Some((key, rendered)) = img.tiles_pending.take() {
        img.tiles_key = key;
        img.tiles = rendered.iter().map(|(n, rgb)| (n.clone(), tex(ctx, &format!("tile-{n}"), rgb))).collect();
    }
    // An older preparation stays on screen until the new one lands.
    let Some(p) = img.prepared.clone() else { return };
    if let Some(name) = img.pending_preset.take() {
        let ps = palette::presets(p.kind, &p.names());
        if let Some(pr) = ps.iter().find(|x| x.name.eq_ignore_ascii_case(&name)) {
            let rr = img.recipe.as_mut().unwrap();
            rr.apply_preset(pr, &p.names());
            rr.name = pr.name.to_string();
        }
    }
    if std::mem::take(&mut img.pending_autocrop) {
        let rr = img.recipe.as_mut().unwrap();
        rr.crop = render::auto_crop(&p);
    }
    let r = img.recipe.clone().unwrap();
    // The preview (the crop doesn't change it: the Crop stage draws the box).
    let mut rk = r.clone();
    rk.crop = None;
    let vk = format!("{:?}|{}|{}|{}|{}|{}|{}", st, img.prep_key, serde_json::to_string(&rk).unwrap_or_default(), img.gchan, img.gmodel, img.sview, p.channels.len());
    // One render at a time, off the UI thread, always from the settings as
    // they are when it starts: a slider drag shows the latest position as
    // soon as the render before it ends, and the slider itself never waits.
    if vk != img.view_key && !img.view_busy {
        img.view_busy = true;
        let (gmodel, gchan, sview) = (img.gmodel, img.gchan, img.sview);
        let (p, r, tx, ctx) = (p.clone(), r.clone(), tx.clone(), ctx.clone());
        let spawned = std::thread::Builder::new().name("crux-preview".into()).spawn(move || {
            let t0 = std::time::Instant::now();
            let (before, after, labels) = quick(|| match st {
                Stage::Enhance => {
                    let v = render::linear_view(&p, false);
                    (v.clone(), v, (String::new(), String::new()))
                }
                Stage::Gradient => {
                    let b = render::linear_view(&p, false);
                    let a2 = if gmodel {
                        let c = &p.channels[gchan.min(p.channels.len() - 1)];
                        match &c.model {
                            Some(m) => {
                                let (w, h) = (c.raw_proxy.width, c.raw_proxy.height);
                                let surf = m.render(w, h);
                                // Its own range, so a few per cent of tilt shows.
                                let (lo, hi) = surf.data.iter().fold((f32::MAX, f32::MIN), |(a, b), &x| (a.min(x), b.max(x)));
                                let k = if hi > lo { 0.7 / (hi - lo) } else { 0.0 };
                                let v: Vec<f32> = surf.data.iter().map(|x| 0.15 + (x - lo) * k).collect();
                                render::Rgb { width: w, height: h, r: v.clone(), g: v.clone(), b: v }
                            }
                            None => render::linear_view(&p, true),
                        }
                    } else {
                        render::linear_view(&p, true)
                    };
                    let right = if gmodel {
                        let c = &p.channels[gchan.min(p.channels.len() - 1)];
                        format!("{} background surface, {:.1}% across (contrast stretched)", c.name, c.report.as_ref().map(|g| g.amplitude_frac * 100.0).unwrap_or(0.0))
                    } else {
                        "Gradient removed".into()
                    };
                    (b, a2, ("Before".to_string(), right))
                }
                Stage::Palette => (render::linear_view(&p, true), render::render_proxy(&p, &kept(&unfinished(&r))), ("Linear, auto-stretched".to_string(), "Mixed and stretched".to_string())),
                Stage::Stars => {
                    let k = kept(&r);
                    match sview {
                        1 => (render::render_crop(&p, &k), render::render_crop(&p, &r), ("Stars as mixed, 1:1".to_string(), "Natural star colours, 1:1".to_string())),
                        2 => (render::render_proxy(&p, &k), render::render_starless(&p, &r).unwrap_or_else(|| render::render_proxy(&p, &k)), ("Stars as mixed".to_string(), "Starless mix".to_string())),
                        _ => (render::render_proxy(&p, &k), render::render_proxy(&p, &r), ("Stars as mixed".to_string(), "Natural star colours".to_string())),
                    }
                }
                Stage::Crop => {
                    let f = render::render_proxy(&p, &r);
                    (f.clone(), f, (String::new(), String::new()))
                }
                _ => (render::render_proxy(&p, &kept(&unfinished(&r))), render::render_proxy(&p, &kept(&r)), ("Before finishing".to_string(), "Finished".to_string())),
            });
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            let _ = tx.send(Msg::ImageView(Box::new((vk, before, after, labels, ms))));
            ctx.request_repaint();
        });
        if spawned.is_err() {
            img.view_busy = false;
        }
    }
    // Preset tiles, rendered from this target with the current stretch,
    // once the settings hold still (nine renders per slider step stuck the
    // sliders) and after the preview.
    if st == Stage::Palette {
        let tk = format!("{}|{}|{}", img.prep_key, serde_json::to_string(&r.stretch).unwrap_or_default(), serde_json::to_string(&r.finish).unwrap_or_default());
        if tk != img.tiles_seen.0 {
            img.tiles_seen = (tk.clone(), std::time::Instant::now());
        }
        if tk != img.tiles_key && !img.tiles_busy {
            let waited = img.tiles_seen.1.elapsed();
            // The first tiles for a target don't wait.
            let first = img.tiles.is_empty();
            if (first || waited >= TILES_SETTLE) && !img.view_busy {
                img.tiles_busy = true;
                let (p, r, tx, ctx) = (p.clone(), r.clone(), tx.clone(), ctx.clone());
                let spawned = std::thread::Builder::new().name("crux-tiles".into()).spawn(move || {
                    let rendered: Vec<(String, render::Rgb)> = quick(|| {
                        palette::presets(p.kind, &p.names())
                            .iter()
                            .map(|pr| {
                                let mut rr = r.clone();
                                rr.apply_preset(pr, &p.names());
                                (pr.name.to_string(), render::render_tile(&p, &rr, TILE))
                            })
                            .collect()
                    });
                    let _ = tx.send(Msg::ImageTiles(tk, rendered));
                    ctx.request_repaint();
                });
                if spawned.is_err() {
                    img.tiles_busy = false;
                }
            } else {
                ctx.request_repaint_after(TILES_SETTLE.saturating_sub(waited).max(std::time::Duration::from_millis(30)));
            }
        }
    }
    if img.pending_write && !img.writing {
        img.pending_write = false;
        start_write(img, &t, &out, tx, ctx);
    }
}

fn start_write(img: &mut ImageUi, t: &Target, out: &std::path::Path, tx: &Sender<Msg>, ctx: &egui::Context) {
    let Some(r) = img.recipe.as_ref().map(named) else { return };
    img.writing = true;
    img.written = None;
    let (tx, ctx, t, dir, cancel, fits, out) = (tx.clone(), ctx.clone(), t.clone(), t.image_dir(out), img.cancel.clone(), img.fits, out.to_path_buf());
    std::thread::Builder::new()
        .name("crux-image-write".into())
        .spawn(move || {
            let t0 = std::time::Instant::now();
            let res = (|| -> anyhow::Result<render::Written> {
                let t = render::enhance::apply(&t, &out, &r.enhance, &|_| {}, &cancel)?;
                let t = if r.stars.engine == render::stars::StarEngine::Network && r.stars.mode == render::stars::StarMode::Natural && render::stars_possible(&t.channels()) {
                    render::starnet::apply(&t, &out, &|_, _, _| {}, &cancel)?
                } else {
                    t
                };
                let p = render::prepare(&t, &r.gradient, &r.stars, true, &cancel)?;
                let rgb = render::render_full(&p, &r)?;
                let mut extra = crux_core::header::Header::default();
                extra.push("NCHANNEL", &t.masters.len().to_string(), "masters mixed");
                let mut wr = render::write(&rgb, &dir, &t.object, &r, &extra, fits)?;
                // Reading, correcting and rendering included.
                wr.seconds = t0.elapsed().as_secs_f64();
                Ok(wr)
            })()
            .map_err(|e| format!("{e:#}"));
            let _ = tx.send(Msg::ImageWritten(res));
            ctx.request_repaint();
        })
        .expect("spawn image write");
}

/// One line for the stage list.
pub fn summary(a: &UiState, st: Stage) -> (Color32, String) {
    let p = &a.pal;
    let img = &a.stack.img;
    let dot = if img.writing || img.preparing.is_some() {
        p.warn
    } else if matches!(img.written, Some(Ok(_))) {
        p.good
    } else if img.prepared.is_some() {
        p.accent
    } else {
        p.faint
    };
    if img.targets.is_empty() {
        return (p.faint, "Needs the master lights from stacking".into());
    }
    let Some(r) = &img.recipe else { return (p.faint, "Not started".into()) };
    match st {
        Stage::Enhance => {
            if !r.enhance.active() {
                return (p.faint, "Off".into());
            }
            match &img.progress {
                Some(t) if img.preparing.is_some() => (p.warn, t.clone()),
                _ => (dot, upper_first(&r.enhance.describe())),
            }
        }
        Stage::Gradient => {
            if !r.gradient.enabled {
                return (p.faint, "Off".into());
            }
            match &img.prepared {
                Some(pp) => {
                    let parts: Vec<String> = pp.channels.iter().filter_map(|c| c.report.as_ref().map(|g| format!("{} {:.1}%", c.name, g.amplitude_frac * 100.0))).collect();
                    let d = pp.channels.iter().find_map(|c| c.report.as_ref().map(|g| g.degree)).unwrap_or(0);
                    (dot, format!("{}, degree {d}: {} removed", if r.gradient.degree == 0 { "Auto" } else { "Fixed" }, parts.join(", ")))
                }
                None => (dot, "Measuring…".into()),
            }
        }
        Stage::Palette => (dot, format!("{}, background {:.2}", if r.mix.preset.is_empty() { "Custom mix" } else { &r.mix.preset }, r.stretch.background)),
        Stage::Stars => {
            let possible = img.target().is_some_and(|t| render::stars_possible(&t.channels()));
            if !possible {
                (p.faint, "Broadband and mono stars are already natural".into())
            } else if r.stars.mode == render::stars::StarMode::Keep {
                (dot, "As mixed".into())
            } else {
                match &img.prepared {
                    Some(pp) if pp.n_stars > 0 => (dot, format!("Natural colours{}: {} stars, {:.1}% of the frame", if r.stars.separate { ", stretched separately" } else { "" }, pp.n_stars, pp.star_cover * 100.0)),
                    _ => (dot, if r.stars.separate { "Natural colours, stretched separately".into() } else { "Natural colours".into() }),
                }
            }
        }
        Stage::Crop => {
            let shape = match (&r.crop, &img.prepared) {
                (Some(c), Some(pp)) if !c.is_whole() => {
                    let (_, _, w, h) = c.pixels(pp.width, pp.height);
                    format!("Cropped to {w} × {h} px")
                }
                (Some(c), None) if !c.is_whole() => "Cropped".to_string(),
                _ => "Whole picture".to_string(),
            };
            match &img.written {
                Some(Ok(wr)) => (p.good, format!("Saved {}", wr.jpg.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default())),
                _ => (if img.writing { p.warn } else { p.faint }, format!("{shape}; not saved yet")),
            }
        }
        _ => (dot, format!("Luminance {:.0}%, saturation {:.0}%{}", r.finish.luminance * 100.0, r.finish.saturation * 100.0, if r.finish.green > 0.0 { format!(", green −{:.0}%", r.finish.green * 100.0) } else { String::new() })),
    }
}

pub fn header(ui: &mut Ui, a: &mut UiState, p: &Pal, tx: &Sender<Msg>) {
    let full = ui.available_width();
    let out = a.stack.plan.as_ref().map(|pl| pl.out.clone());
    ui.horizontal(|ui| {
        ui.allocate_ui_with_layout(Vec2::new((full - 380.0).max(200.0), 0.0), egui::Layout::top_down(egui::Align::Min), |ui| {
            w::h1(ui, "Image");
            let dir = out.as_ref().and_then(|o| output_of(&a.stack.img, o)).map(|(d, _)| d.display().to_string()).unwrap_or_default();
            ui.add(egui::Label::new(RichText::new(format!("Sharpens and denoises the integrated masters if you choose, removes gradients, then mixes the colour, stretches and finishes them. The preview and the full-size image use the same processing. Nothing is saved until you press Create full-size image, which writes to {dir}.")).color(p.muted)).wrap());
        });
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let img = &mut a.stack.img;
            let ok = img.prepared.is_some() && !img.writing;
            if w::button_enabled(ui, p, if img.writing { "Creating…" } else { "Create full-size image" }, Kind::Primary, None, ok).clicked() {
                if let (Some(t), Some(o)) = (img.target().cloned(), out.clone()) {
                    start_write(img, &t, &o, tx, ui.ctx());
                }
            }
            if let Some(Ok(wr)) = &img.written {
                if w::button(ui, p, "Open output folder", Kind::Ghost, None).clicked() {
                    let _ = crate::stack::open_folder(wr.xisf.parent().unwrap_or(&wr.xisf));
                }
            }
        });
    });
    let img = &a.stack.img;
    match &img.written {
        Some(Ok(wr)) => {
            ui.add_space(6.0);
            written_line(ui, p, wr);
        }
        Some(Err(e)) => {
            ui.label(RichText::new(format!("Couldn't create the image: {e}")).color(p.rej));
        }
        None => {}
    }
}

/// "Saved <jpg> (and .png, .xisf), W × H px, in N s": the file name opens
/// the JPEG, the folder button shows it in the file manager.
fn written_line(ui: &mut Ui, p: &Pal, wr: &render::Written) {
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0;
        ui.label(RichText::new("Saved").color(p.good));
        if w::link(ui, p, &wr.jpg.display().to_string(), p.good).on_hover_text("Open the JPEG").clicked() {
            let _ = crate::stack::open_folder(&wr.jpg);
        }
        if w::folder_button(ui, p, "Show in the folder").clicked() {
            let _ = crate::stack::show_in_folder(&wr.jpg);
        }
        let more = if wr.fits.is_some() { ".png, .xisf and .fits" } else { ".png and .xisf" };
        let took = if wr.seconds.is_finite() { format!(", in {:.1} s", wr.seconds) } else { String::new() };
        ui.label(RichText::new(format!("with {more} beside it, {} × {} px{took}", wr.width, wr.height)).color(p.good));
    });
}

/// The image steps in one scrolling page (small windows).
pub fn main(ui: &mut Ui, a: &mut UiState, s: &mut Session, p: &Pal, tx: &Sender<Msg>) {
    let _ = s;
    ui.spacing_mut().item_spacing.y = 14.0;
    a.stack.img.pin_h = None;
    if !no_targets(ui, a, p) {
        target_picker(ui, a);
        preview(ui, a, p);
        below(ui, a, p, tx);
    }
}

/// The image steps with the preview held in place: target picker and
/// preview at the top, a handle to share the height, then the settings
/// in their own scroll area.
pub fn main_pinned(ui: &mut Ui, a: &mut UiState, s: &mut Session, p: &Pal, tx: &Sender<Msg>) {
    let _ = s;
    ui.spacing_mut().item_spacing.y = 14.0;
    if no_targets(ui, a, p) {
        return;
    }
    target_picker(ui, a);
    let room = ui.available_height();
    let img = &mut a.stack.img;
    let pv = (room * img.view_frac).clamp(160.0, (room - 140.0).max(160.0));
    img.pin_h = Some(pv);
    let y0 = ui.cursor().min.y;
    ui.spacing_mut().item_spacing.y = 0.0;
    preview(ui, a, p);
    let img = &mut a.stack.img;
    img.pin_h = None;
    // The handle: drag to give the preview or the settings more room,
    // double-click for the starting split.
    let shown = ui.cursor().min.y - y0;
    let wd = ui.available_width();
    let (r, resp) = ui.allocate_exact_size(Vec2::new(wd, 14.0), Sense::click_and_drag());
    let resp = resp.on_hover_cursor(egui::CursorIcon::ResizeVertical).on_hover_text("Drag to share the height between the preview and the settings. Double-click to reset.");
    let hot = resp.hovered() || resp.dragged();
    ui.painter().rect_filled(Rect::from_center_size(r.center(), Vec2::new(56.0, 4.0)), 2.0, if hot { p.accent } else { p.line });
    if resp.dragged() {
        // From what's shown, so the handle follows the pointer even when
        // the picture is limited by its width.
        img.view_frac = ((shown.min(pv) + resp.drag_delta().y) / room).clamp(0.15, 0.85);
    }
    if resp.double_clicked() {
        img.view_frac = VIEW_FRAC;
    }
    ui.spacing_mut().item_spacing.y = 14.0;
    egui::ScrollArea::vertical().id_salt("img-settings").auto_shrink([false, false]).show(ui, |ui| {
        // Room for the scroll bar.
        ui.set_max_width(ui.available_width() - 12.0);
        ui.spacing_mut().item_spacing.y = 14.0;
        below(ui, a, p, tx);
        ui.add_space(24.0);
    });
}

/// The card shown before any masters exist; true when it was shown.
fn no_targets(ui: &mut Ui, a: &UiState, p: &Pal) -> bool {
    if !a.stack.img.targets.is_empty() {
        return false;
    }
    w::card(ui, p, |ui| {
        w::h2(ui, "No master lights yet");
        w::muted(ui, p, "The image steps start from the integrated masters. Run the stacking steps through Integrate first; the masters then appear here.");
    });
    true
}

/// The stage's picture: crop box, enhance crops, or the before/after split.
fn preview(ui: &mut Ui, a: &mut UiState, p: &Pal) {
    match a.stack.stage {
        Stage::Crop => crop_view(ui, a, p),
        Stage::Enhance => enhance_view(ui, a, p),
        _ => split(ui, a, p),
    }
}

/// Everything under the picture: presets, settings, recipe or save.
fn below(ui: &mut Ui, a: &mut UiState, p: &Pal, tx: &Sender<Msg>) {
    if a.stack.stage == Stage::Palette {
        tiles(ui, a, p);
    }
    w::card(ui, p, |ui| settings(ui, a, p));
    if a.stack.stage == Stage::Crop {
        w::card(ui, p, |ui| save_card(ui, a, p, tx));
    } else {
        w::card(ui, p, |ui| recipe_card(ui, a, p));
    }
}

fn target_picker(ui: &mut Ui, a: &mut UiState) {
    if a.stack.img.targets.len() > 1 {
        let names: Vec<String> = a.stack.img.targets.iter().map(|t| format!("{}  ·  {}", t.object, t.channels().join(" "))).collect();
        let mut sel = a.stack.img.sel;
        ui.horizontal(|ui| {
            ui.label(RichText::new("Target").font(semibold(13.0)));
            egui::ComboBox::from_id_salt("img-target").selected_text(names[sel].clone()).width(320.0).show_ui(ui, |ui| {
                for (i, n) in names.iter().enumerate() {
                    ui.selectable_value(&mut sel, i, n);
                }
            });
        });
        a.stack.img.select(sel);
    }
}

/// The finished picture with the crop box: drag an edge or corner to
/// resize, inside to move, outside to draw a new box.
fn crop_view(ui: &mut Ui, a: &mut UiState, p: &Pal) {
    let wd = ui.available_width();
    let img = &mut a.stack.img;
    let Some(tex) = img.after.clone() else {
        let h = (wd * 0.55).min(max_view_h(ui, img, 0.58));
        let (r, _) = ui.allocate_exact_size(Vec2::new(wd, h), Sense::hover());
        ui.painter().rect_filled(r, 10.0, p.well);
        let msg = match &img.error {
            Some(e) => format!("Couldn't read the masters: {e}"),
            None => "Reading the masters…".into(),
        };
        ui.painter().text(r.center(), egui::Align2::CENTER_CENTER, msg, body(13.0), p.muted);
        return;
    };
    let [tw, th] = tex.size();
    let max_h = max_view_h(ui, img, 0.62) - if img.pin_h.is_some() { 20.0 } else { 0.0 };
    let (iw, ih) = if wd * th as f32 / tw as f32 > max_h { (max_h * tw as f32 / th as f32, max_h) } else { (wd, wd * th as f32 / tw as f32) };
    let (outer, _) = ui.allocate_exact_size(Vec2::new(wd, ih + 20.0), Sense::hover());
    ui.painter().rect_filled(outer, 10.0, p.well);
    let r = Rect::from_center_size(outer.center(), Vec2::new(iw, ih));
    let (fw, fh) = img.prepared.as_ref().map(|pp| (pp.width as f64, pp.height as f64)).unwrap_or((tw as f64, th as f64));
    let Some(recipe) = img.recipe.as_mut() else { return };
    let cur = recipe.crop.unwrap_or(recipe::Crop::WHOLE).clamped();
    let frac = |pos: Pos2| (((pos.x - r.left()) / r.width()) as f64, ((pos.y - r.top()) / r.height()) as f64);
    let cr = Rect::from_min_max(
        Pos2::new(r.left() + cur.left as f32 * r.width(), r.top() + cur.top as f32 * r.height()),
        Pos2::new(r.left() + cur.right as f32 * r.width(), r.top() + cur.bottom as f32 * r.height()),
    );
    let grip_at = |pos: Pos2| -> Grip {
        let near = 9.0;
        let in_y = pos.y > cr.top() - near && pos.y < cr.bottom() + near;
        let in_x = pos.x > cr.left() - near && pos.x < cr.right() + near;
        let sx = if !in_y { 0 } else if (pos.x - cr.left()).abs() < near { -1 } else if (pos.x - cr.right()).abs() < near { 1 } else { 0 };
        let sy = if !in_x { 0 } else if (pos.y - cr.top()).abs() < near { -1 } else if (pos.y - cr.bottom()).abs() < near { 1 } else { 0 };
        if sx != 0 || sy != 0 {
            Grip::Edge(sx, sy)
        } else if cr.contains(pos) {
            Grip::Move
        } else {
            let (x, y) = frac(pos);
            Grip::New(x.clamp(0.0, 1.0), y.clamp(0.0, 1.0))
        }
    };
    let resp = ui.interact(r.expand(10.0), ui.scope_id().with("img-crop"), Sense::click_and_drag());
    if let Some(pos) = resp.hover_pos() {
        let icon = match img.crop_drag.map(|d| d.0).unwrap_or_else(|| grip_at(pos)) {
            Grip::Move => egui::CursorIcon::Grab,
            Grip::New(..) => egui::CursorIcon::Crosshair,
            Grip::Edge(x, 0) if x != 0 => egui::CursorIcon::ResizeHorizontal,
            Grip::Edge(0, _) => egui::CursorIcon::ResizeVertical,
            Grip::Edge(x, y) if x == y => egui::CursorIcon::ResizeNwSe,
            Grip::Edge(..) => egui::CursorIcon::ResizeNeSw,
        };
        ui.ctx().set_cursor_icon(icon);
    }
    if resp.drag_started() {
        if let Some(pos) = resp.interact_pointer_pos() {
            img.crop_drag = Some((grip_at(pos), cur, frac(pos)));
        }
    }
    if resp.dragged() {
        if let (Some((g, start, s0)), Some(pos)) = (img.crop_drag, resp.interact_pointer_pos()) {
            let c = drag_crop(g, start, s0, frac(pos), img.crop_ratio, fw, fh);
            if Some(c) != recipe.crop {
                recipe.crop = Some(c);
                img.written = None;
                img.crop_note = None;
            }
        }
    }
    if resp.drag_stopped() {
        img.crop_drag = None;
    }
    // Draw: the picture, the outside dimmed, the box with thirds and grips.
    let pa = ui.painter_at(outer);
    pa.image(tex.id(), r, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
    let dim = Color32::from_black_alpha(165);
    for d in [
        Rect::from_min_max(r.left_top(), Pos2::new(r.right(), cr.top())),
        Rect::from_min_max(Pos2::new(r.left(), cr.bottom()), r.right_bottom()),
        Rect::from_min_max(Pos2::new(r.left(), cr.top()), Pos2::new(cr.left(), cr.bottom())),
        Rect::from_min_max(Pos2::new(cr.right(), cr.top()), Pos2::new(r.right(), cr.bottom())),
    ] {
        if d.width() > 0.0 && d.height() > 0.0 {
            pa.rect_filled(d, 0.0, dim);
        }
    }
    let thin = Stroke::new(1.0, Color32::from_white_alpha(70));
    for k in 1..3 {
        let x = cr.left() + cr.width() * k as f32 / 3.0;
        let y = cr.top() + cr.height() * k as f32 / 3.0;
        pa.line_segment([Pos2::new(x, cr.top()), Pos2::new(x, cr.bottom())], thin);
        pa.line_segment([Pos2::new(cr.left(), y), Pos2::new(cr.right(), y)], thin);
    }
    pa.rect_stroke(cr, 0.0, Stroke::new(1.5, Color32::WHITE), egui::StrokeKind::Middle);
    for c in [cr.left_top(), cr.right_top(), cr.left_bottom(), cr.right_bottom()] {
        pa.rect_filled(Rect::from_center_size(c, Vec2::splat(9.0)), 1.0, Color32::WHITE);
    }
    for c in [cr.center_top(), cr.center_bottom(), cr.left_center(), cr.right_center()] {
        pa.rect_filled(Rect::from_center_size(c, Vec2::splat(7.0)), 1.0, Color32::WHITE);
    }
    let (_, _, w, h) = cur.pixels(fw as usize, fh as usize);
    let t = format!("{w} × {h} px of {} × {}", fw as usize, fh as usize);
    let g = pa.layout_no_wrap(t, mono(11.0), Color32::WHITE);
    // Beside the picture when there's room, so it doesn't cover a grip.
    let size = g.size() + Vec2::new(12.0, 4.0);
    let lr = if r.left() - outer.left() > size.x + 24.0 {
        egui::Align2::RIGHT_BOTTOM.anchor_size(Pos2::new(r.left() - 14.0, r.bottom()), size)
    } else {
        egui::Align2::LEFT_TOP.anchor_size(r.left_top() + Vec2::new(14.0, 14.0), size)
    };
    pa.rect_filled(lr, 4.0, Color32::from_black_alpha(140));
    pa.galley(lr.min + Vec2::new(6.0, 2.0), g, Color32::WHITE);
}

/// The crop after dragging grip `g` from `s0` to `q` (fractions), held to
/// `ratio` (pixels, width ÷ height) when there is one.
fn drag_crop(g: Grip, start: recipe::Crop, s0: (f64, f64), q: (f64, f64), ratio: Option<f64>, fw: f64, fh: f64) -> recipe::Crop {
    const MIN: f64 = 0.01;
    let (dx, dy) = (q.0 - s0.0, q.1 - s0.1);
    let (qx, qy) = (q.0.clamp(0.0, 1.0), q.1.clamp(0.0, 1.0));
    match g {
        Grip::Move => {
            let (w, h) = (start.right - start.left, start.bottom - start.top);
            let l = (start.left + dx).clamp(0.0, 1.0 - w);
            let t = (start.top + dy).clamp(0.0, 1.0 - h);
            recipe::Crop { left: l, top: t, right: l + w, bottom: t + h }
        }
        Grip::New(x0, y0) => {
            let c = recipe::Crop { left: x0.min(qx), top: y0.min(qy), right: x0.max(qx), bottom: y0.max(qy) };
            if c.right - c.left < MIN || c.bottom - c.top < MIN {
                return start;
            }
            let anchor = (if qx >= x0 { 0.0 } else { 1.0 }, if qy >= y0 { 0.0 } else { 1.0 });
            match ratio {
                Some(a) => with_ratio(c, a, fw, fh, anchor, true),
                None => c,
            }
        }
        Grip::Edge(sx, sy) => {
            let mut c = start;
            if sx < 0 {
                c.left = (start.left + dx).clamp(0.0, start.right - MIN);
            } else if sx > 0 {
                c.right = (start.right + dx).clamp(start.left + MIN, 1.0);
            }
            if sy < 0 {
                c.top = (start.top + dy).clamp(0.0, start.bottom - MIN);
            } else if sy > 0 {
                c.bottom = (start.bottom + dy).clamp(start.top + MIN, 1.0);
            }
            let side = |s: i8| if s < 0 { 1.0 } else if s > 0 { 0.0 } else { 0.5 };
            match ratio {
                Some(a) => with_ratio(c, a, fw, fh, (side(sx), side(sy)), sx != 0),
                None => c,
            }
        }
    }
}

/// `c` reshaped to `ratio`, keeping its width (or height) and the point at
/// `anchor` (0 left/top edge, 1 right/bottom, 0.5 centre) where it is,
/// shrunk if it would leave the picture.
fn with_ratio(c: recipe::Crop, ratio: f64, fw: f64, fh: f64, anchor: (f64, f64), keep_width: bool) -> recipe::Crop {
    let (fx, fy) = anchor;
    let px = c.left + fx * (c.right - c.left);
    let py = c.top + fy * (c.bottom - c.top);
    let (mut cw, mut ch) = (c.right - c.left, c.bottom - c.top);
    if keep_width {
        ch = cw * fw / (ratio * fh);
    } else {
        cw = ch * ratio * fh / fw;
    }
    let room = |p: f64, f: f64| if f == 0.0 { 1.0 - p } else if f == 1.0 { p } else { 2.0 * p.min(1.0 - p) };
    let s = 1f64.min(room(px, fx) / cw).min(room(py, fy) / ch);
    cw *= s;
    ch *= s;
    let left = px - fx * cw;
    let top = py - fy * ch;
    recipe::Crop { left, top, right: left + cw, bottom: top + ch }
}

/// The largest box of `ratio` inside `c`, centred.
fn fit_inside(c: recipe::Crop, ratio: f64, fw: f64, fh: f64) -> recipe::Crop {
    let (w, h) = (c.right - c.left, c.bottom - c.top);
    let (mut cw, mut ch) = (w, w * fw / (ratio * fh));
    if ch > h {
        ch = h;
        cw = ch * ratio * fh / fw;
    }
    let (cx, cy) = ((c.left + c.right) / 2.0, (c.top + c.bottom) / 2.0);
    recipe::Crop { left: cx - cw / 2.0, top: cy - ch / 2.0, right: cx + cw / 2.0, bottom: cy + ch / 2.0 }
}

/// What Create full-size image writes, and the button.
fn save_card(ui: &mut Ui, a: &mut UiState, p: &Pal, tx: &Sender<Msg>) {
    let Some(out) = a.stack.plan.as_ref().map(|pl| pl.out.clone()) else { return };
    let img = &mut a.stack.img;
    let Some((dir, base)) = output_of(img, &out) else { return };
    w::h2(ui, "Save");
    w::muted(ui, p, "Create full-size image renders the picture at full resolution with every stage's settings and this crop. It replaces an earlier image made with the same recipe name. It writes these files:");
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new(dir.display().to_string()).font(mono(12.0)));
        if w::folder_button(ui, p, "Open this folder").clicked() {
            let _ = std::fs::create_dir_all(&dir);
            let _ = crate::stack::open_folder(&dir);
        }
    });
    let mut files = vec![
        (format!("{base}.jpg"), "8-bit JPEG, for sharing"),
        (format!("{base}.png"), "16-bit PNG, for further editing"),
        (format!("{base}.xisf"), "32-bit float, for PixInsight, with the recipe inside"),
    ];
    if img.fits {
        files.push((format!("{base}.fits"), "32-bit float FITS copy"));
    }
    files.push(("preview.png".into(), "1,200 px preview for Home and the autopilot report"));
    files.push(("recipe.toml".into(), "the settings, used next time you open this target"));
    egui::Grid::new("save-files").num_columns(2).spacing([16.0, 3.0]).show(ui, |ui| {
        for (f, what) in files {
            ui.label(RichText::new(f).font(mono(12.0)));
            ui.label(RichText::new(what).font(body(12.5)).color(p.muted));
            ui.end_row();
        }
    });
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        let ok = img.prepared.is_some() && !img.writing;
        if w::button_enabled(ui, p, if img.writing { "Creating…" } else { "Create full-size image" }, Kind::Primary, None, ok).clicked() {
            if let Some(t) = img.target().cloned() {
                start_write(img, &t, &out, tx, ui.ctx());
            }
        }
    });
    match &img.written {
        Some(Ok(wr)) => written_line(ui, p, wr),
        Some(Err(e)) => {
            ui.label(RichText::new(format!("Couldn't create the image: {e}")).color(p.rej));
        }
        None => {}
    }
}

fn split(ui: &mut Ui, a: &mut UiState, p: &Pal) {
    let wd = ui.available_width();
    let img = &mut a.stack.img;
    let (Some(before), Some(after)) = (img.before.clone(), img.after.clone()) else {
        let h = (wd * 0.55).min(max_view_h(ui, img, 0.58));
        let (r, _) = ui.allocate_exact_size(Vec2::new(wd, h), Sense::hover());
        ui.painter().rect_filled(r, 10.0, p.well);
        let msg = match (&img.error, &img.preparing) {
            (Some(e), _) => format!("Couldn't read the masters: {e}"),
            (_, Some(_)) => match &img.progress {
                Some(t) => format!("Sharpening and denoising the masters: {t}"),
                None => "Reading the masters and removing gradients…".into(),
            },
            _ => "Preparing…".into(),
        };
        ui.painter().text(r.center(), egui::Align2::CENTER_CENTER, msg, body(13.0), p.muted);
        return;
    };
    let [tw, th] = after.size();
    let max_h = max_view_h(ui, img, 0.58);
    let (iw, ih) = if wd * th as f32 / tw as f32 > max_h { (max_h * tw as f32 / th as f32, max_h) } else { (wd, wd * th as f32 / tw as f32) };
    let (outer, _) = ui.allocate_exact_size(Vec2::new(wd, ih), Sense::hover());
    ui.painter().rect_filled(outer, 10.0, p.well);
    let r = Rect::from_center_size(outer.center(), Vec2::new(iw, ih));
    let resp = ui.interact(r, ui.scope_id().with("img-split"), Sense::click_and_drag());
    if let Some(pos) = resp.interact_pointer_pos() {
        img.split = ((pos.x - r.left()) / r.width()).clamp(0.0, 1.0);
    }
    let pa = ui.painter_at(r);
    let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
    pa.image(after.id(), r, uv, Color32::WHITE);
    let sx = r.left() + r.width() * img.split;
    pa.with_clip_rect(Rect::from_min_max(r.min, Pos2::new(sx, r.bottom()))).image(before.id(), r, uv, Color32::WHITE);
    // Gradient samples of the chosen channel: kept green, rejected red.
    if a.stack.stage == Stage::Gradient && img.gsamples {
        if let Some(rep) = img.prepared.as_ref().and_then(|pp| pp.channels.get(img.gchan)).and_then(|c| c.report.as_ref()) {
            for &(u, v, keep) in &rep.samples {
                let c = Pos2::new(r.left() + u * r.width(), r.top() + v * r.height());
                if keep {
                    pa.circle_stroke(c, 4.0, Stroke::new(1.3, Color32::from_rgb(0x5d, 0xd3, 0x9e)));
                } else {
                    pa.circle_filled(c, 2.0, Color32::from_rgb(0xf0, 0x6a, 0x6a));
                }
            }
        }
    }
    pa.line_segment([Pos2::new(sx, r.top()), Pos2::new(sx, r.bottom())], Stroke::new(2.0, Color32::WHITE));
    pa.circle_filled(Pos2::new(sx, r.center().y), 13.0, Color32::WHITE);
    pa.text(Pos2::new(sx, r.center().y), egui::Align2::CENTER_CENTER, "⟷", body(13.0), Color32::BLACK);
    let lab = |pos: Pos2, align: egui::Align2, t: &str| {
        let g = pa.layout_no_wrap(t.to_string(), mono(11.0), Color32::WHITE);
        let rr = align.anchor_size(pos, g.size() + Vec2::new(12.0, 4.0));
        pa.rect_filled(rr, 4.0, Color32::from_black_alpha(140));
        pa.galley(rr.min + Vec2::new(6.0, 2.0), g, Color32::WHITE);
    };
    lab(r.left_top() + Vec2::new(8.0, 8.0), egui::Align2::LEFT_TOP, &img.labels.0);
    lab(r.right_top() + Vec2::new(-8.0, 8.0), egui::Align2::RIGHT_TOP, &img.labels.1);
    if let Some(pp) = &img.prepared {
        let t = pp.channels.iter().map(|c| format!("{} {}×", c.name, c.master.frames)).collect::<Vec<_>>().join(" · ");
        lab(r.left_bottom() + Vec2::new(8.0, -8.0), egui::Align2::LEFT_BOTTOM, &format!("{} · {}×{} proxy · {:.0} ms", t, before.size()[0], before.size()[1], img.render_ms));
    }
}

fn tiles(ui: &mut Ui, a: &mut UiState, p: &Pal) {
    let img = &mut a.stack.img;
    if img.tiles.is_empty() {
        return;
    }
    let cur = img.recipe.as_ref().map(|r| r.mix.preset.clone()).unwrap_or_default();
    // Folds away to one line, so the sliders sit right under the preview.
    let head = format!("{}  Presets", if img.tiles_open { "▾" } else { "▸" });
    let line = if cur.is_empty() { head } else { format!("{head}  ·  {cur}") };
    let hint = if img.tiles_open { "Hide the preset pictures" } else { "Show the preset pictures" };
    let resp = ui.add(egui::Label::new(RichText::new(line).font(semibold(13.0)).color(p.muted)).sense(Sense::click()));
    if resp.on_hover_cursor(egui::CursorIcon::PointingHand).on_hover_text(hint).clicked() {
        img.tiles_open = !img.tiles_open;
    }
    if !img.tiles_open {
        return;
    }
    let mut pick = None;
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = Vec2::new(10.0, 10.0);
        for (name, t) in &img.tiles {
            let [tw, th] = t.size();
            let wd = 150.0;
            let ht = wd * th as f32 / tw as f32;
            let (r, resp) = ui.allocate_exact_size(Vec2::new(wd, ht + 22.0), Sense::click());
            let on = *name == cur;
            let pa = ui.painter();
            let ir = Rect::from_min_size(r.min, Vec2::new(wd, ht));
            pa.image(t.id(), ir, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
            pa.rect_stroke(ir, 6.0, Stroke::new(if on { 2.0 } else { 1.0 }, if on { p.accent } else if resp.hovered() { p.faint } else { p.line }), egui::StrokeKind::Inside);
            pa.text(Pos2::new(r.left() + 2.0, ir.bottom() + 4.0), egui::Align2::LEFT_TOP, name, if on { semibold(12.0) } else { body(12.0) }, if on { p.fg } else { p.muted });
            if resp.on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
                pick = Some(name.clone());
            }
        }
    });
    if let Some(n) = pick {
        img.pending_preset = Some(n);
    }
}

/// "default", "recipe “X”" or "changed here".
fn src(ui: &mut Ui, p: &Pal, changed: bool, base: &str) {
    if changed {
        ui.label(RichText::new("changed here").font(body(11.0)).color(p.accent));
    } else {
        ui.label(RichText::new(base).font(body(11.0)).color(p.faint));
    }
}

/// A labelled slider with its source tag; `pct` shows a fraction as %.
#[allow(clippy::too_many_arguments)]
/// Image-stage sliders: wide enough that one pixel is a small step.
const SLIDER_WIDTH: f32 = 300.0;

fn slider(ui: &mut Ui, p: &Pal, label: &str, v: &mut f64, range: std::ops::RangeInclusive<f64>, base: f64, base_label: &str, pct: bool, help: &str) {
    slider_d(ui, p, label, v, range, base, base_label, pct, help, 2)
}

#[allow(clippy::too_many_arguments)]
fn slider_d(ui: &mut Ui, p: &Pal, label: &str, v: &mut f64, range: std::ops::RangeInclusive<f64>, base: f64, base_label: &str, pct: bool, help: &str, decimals: usize) {
    ui.horizontal(|ui| {
        ui.add_sized([150.0, 20.0], egui::Label::new(RichText::new(label).font(body(13.0)))).on_hover_text(help);
        // Wide, and no snapping to round numbers while dragging (egui's
        // smart aim made values jump 45 → 50 → 55): fine adjustments.
        ui.spacing_mut().slider_width = SLIDER_WIDTH;
        let s = if pct {
            let mut x = *v * 100.0;
            let (a, b) = (*range.start() * 100.0, *range.end() * 100.0);
            let r = ui.add(egui::Slider::new(&mut x, a..=b).suffix("%").fixed_decimals(1).smart_aim(false)).on_hover_text(help);
            if r.changed() {
                *v = x / 100.0;
            }
            r
        } else {
            ui.add(egui::Slider::new(v, range).fixed_decimals(decimals + 1).smart_aim(false)).on_hover_text(help)
        };
        let _ = s;
        src(ui, p, (*v - base).abs() > 1e-9, base_label);
    });
}

fn upper_first(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

/// Side of the Sharpen and denoise stage's 1:1 crops.
const ENH_CROP: (usize, usize) = (720, 480);

/// Sharpen and denoise stage: keep the before and after crops up to date
/// (read from the original and the enhanced master in the background).
fn enhance_tick(img: &mut ImageUi, t: &Target, out: &std::path::Path, tx: &Sender<Msg>, ctx: &egui::Context) {
    if let Some((key, r)) = img.enh_pending.take() {
        match r {
            Ok((b, a)) => {
                img.enh_view = Some((tex(ctx, "enh-before", &b), tex(ctx, "enh-after", &a)));
                img.enh_error = None;
            }
            Err(e) => {
                img.enh_view = None;
                img.enh_error = Some(e);
            }
        }
        img.enh_key = key;
    }
    let Some(r) = img.recipe.clone() else { return };
    // Only once the masters are enhanced (the preparation has caught up).
    if img.preparing.is_some() || img.prep_key != want_key(t, &r) || t.masters.is_empty() {
        return;
    }
    let ch = img.gchan.min(t.masters.len() - 1);
    let key = format!("{}|{}|{}|{}", t.object, enhance_key(&r.enhance), ch, img.ewhere);
    if key == img.enh_key || img.enh_loading.as_deref() == Some(key.as_str()) {
        return;
    }
    img.enh_loading = Some(key.clone());
    let (tx, ctx, t, out, e, wh) = (tx.clone(), ctx.clone(), t.clone(), out.to_path_buf(), r.enhance.clone(), img.ewhere);
    std::thread::Builder::new()
        .name("crux-enhance-view".into())
        .spawn(move || {
            let res = (|| -> anyhow::Result<(render::Rgb, render::Rgb)> {
                let m = &t.masters[ch];
                let after = if e.active() { render::enhance::cached_path(&t, &out, &e)[ch].clone().unwrap_or_else(|| m.path.clone()) } else { m.path.clone() };
                let a = crux_core::read_image(&m.path)?;
                let b = crux_core::read_image(&after)?;
                anyhow::ensure!((a.width, a.height) == (b.width, b.height), "the enhanced master differs in size");
                let (cw, ch2) = (ENH_CROP.0.min(a.width), ENH_CROP.1.min(a.height));
                // The corner sits a little inside the frame: registration
                // leaves an empty edge.
                let (x0, y0) = if wh == 1 { ((a.width / 25).min(a.width - cw), (a.height / 25).min(a.height - ch2)) } else { ((a.width - cw) / 2, (a.height - ch2) / 2) };
                let cut = |im: &crux_core::image::Image| crux_core::image::Image::from_data(cw, ch2, (y0..y0 + ch2).flat_map(|y| im.data[y * im.width + x0..y * im.width + x0 + cw].iter().copied()).collect());
                let (ca, cb) = (cut(&a), cut(&b));
                let stf = crux_core::stretch::Stf::auto(&ca);
                let grey = |im: &crux_core::image::Image| {
                    let v: Vec<f32> = im.data.iter().map(|x| stf.apply(*x)).collect();
                    render::Rgb { width: im.width, height: im.height, r: v.clone(), g: v.clone(), b: v }
                };
                Ok((grey(&ca), grey(&cb)))
            })()
            .map_err(|e| format!("{e:#}"));
            let _ = tx.send(Msg::ImageEnhanceView(key, res));
            ctx.request_repaint();
        })
        .expect("spawn enhance view");
}

/// The Sharpen and denoise stage's picture: a 1:1 crop of one master,
/// original | enhanced, one stretch, with a split to drag.
fn enhance_view(ui: &mut Ui, a: &mut UiState, p: &Pal) {
    let wd = ui.available_width();
    let img = &mut a.stack.img;
    let on = img.recipe.as_ref().is_some_and(|r| r.enhance.active());
    let ready = img.enh_view.clone().filter(|_| img.enh_key.ends_with(&format!("|{}|{}", img.gchan.min(img.target().map(|t| t.masters.len().saturating_sub(1)).unwrap_or(0)), img.ewhere)));
    let Some((before, after)) = ready else {
        let h = (wd * 0.55).min(max_view_h(ui, img, 0.58));
        let (r, _) = ui.allocate_exact_size(Vec2::new(wd, h), Sense::hover());
        ui.painter().rect_filled(r, 10.0, p.well);
        let msg = match (&img.error, &img.enh_error, &img.preparing, &img.progress) {
            (Some(e), _, _, _) => format!("Couldn't prepare the masters: {e}"),
            (_, Some(e), _, _) => format!("Couldn't show the masters: {e}"),
            (_, _, Some(_), Some(t)) => format!("Sharpening and denoising: {t}. The first run on a target takes about half a minute per master."),
            _ => "Reading the masters…".into(),
        };
        ui.painter().text(r.center(), egui::Align2::CENTER_CENTER, msg, body(13.0), p.muted);
        return;
    };
    let [tw, th] = after.size();
    let max_h = max_view_h(ui, img, 0.58);
    let (iw, ih) = if wd * th as f32 / tw as f32 > max_h { (max_h * tw as f32 / th as f32, max_h) } else { (wd, wd * th as f32 / tw as f32) };
    let (outer, _) = ui.allocate_exact_size(Vec2::new(wd, ih), Sense::hover());
    ui.painter().rect_filled(outer, 10.0, p.well);
    let r = Rect::from_center_size(outer.center(), Vec2::new(iw, ih));
    let resp = ui.interact(r, ui.scope_id().with("enh-split"), Sense::click_and_drag());
    if let Some(pos) = resp.interact_pointer_pos() {
        img.split = ((pos.x - r.left()) / r.width()).clamp(0.0, 1.0);
    }
    let pa = ui.painter_at(r);
    let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
    pa.image(after.id(), r, uv, Color32::WHITE);
    let sx = r.left() + r.width() * img.split;
    pa.with_clip_rect(Rect::from_min_max(r.min, Pos2::new(sx, r.bottom()))).image(before.id(), r, uv, Color32::WHITE);
    pa.line_segment([Pos2::new(sx, r.top()), Pos2::new(sx, r.bottom())], Stroke::new(2.0, Color32::WHITE));
    pa.circle_filled(Pos2::new(sx, r.center().y), 13.0, Color32::WHITE);
    pa.text(Pos2::new(sx, r.center().y), egui::Align2::CENTER_CENTER, "⟷", body(13.0), Color32::BLACK);
    let lab = |pos: Pos2, align: egui::Align2, t: &str| {
        let g = pa.layout_no_wrap(t.to_string(), mono(11.0), Color32::WHITE);
        let rr = align.anchor_size(pos, g.size() + Vec2::new(12.0, 4.0));
        pa.rect_filled(rr, 4.0, Color32::from_black_alpha(140));
        pa.galley(rr.min + Vec2::new(6.0, 2.0), g, Color32::WHITE);
    };
    let what = img.recipe.as_ref().map(|r| upper_first(&r.enhance.describe())).unwrap_or_default();
    let ch = img.target().and_then(|t| t.masters.get(img.gchan.min(t.masters.len().saturating_sub(1)))).map(|m| m.channel.clone()).unwrap_or_default();
    let place = if img.ewhere == 1 { "top-left corner" } else { "centre" };
    lab(r.left_top() + Vec2::new(8.0, 8.0), egui::Align2::LEFT_TOP, "Original");
    lab(r.right_top() + Vec2::new(-8.0, 8.0), egui::Align2::RIGHT_TOP, if on { &what } else { "Off" });
    lab(r.left_bottom() + Vec2::new(8.0, -8.0), egui::Align2::LEFT_BOTTOM, &format!("{ch} · 1:1, {place} · {tw} × {th} px · one stretch"));
}

/// Settings for the Sharpen and denoise stage.
fn enhance_settings(ui: &mut Ui, img: &mut ImageUi, p: &Pal, r: &mut Recipe, base: &Recipe, base_label: &str, depth: Depth) {
    use render::enhance::{availability, Engine, Sharpen};
    let avail = img.enh_avail.get_or_insert_with(|| [(availability(Engine::Crux, false), availability(Engine::Crux, true)), (availability(Engine::RcAstro, false), availability(Engine::RcAstro, true))]).clone();
    w::muted(ui, p, "Runs on each linear master before gradient removal. crux's own networks do it by default: a denoiser trained on pairs of stacks of your subs, and a star corrector trained on the centres of your sharpest masters. Each master is processed once and kept, so moving a slider only blends.");
    let e = &mut r.enhance;
    let be = &base.enhance;
    // Who does it.
    ui.horizontal(|ui| {
        ui.add_sized([150.0, 20.0], egui::Label::new(RichText::new("Done by").font(body(13.0))));
        if w::chip(ui, p, "crux", None, e.engine == Engine::Crux).clicked() {
            e.engine = Engine::Crux;
        }
        let rc_ok = avail[1].0.is_ok() || avail[1].1.is_ok();
        let resp = ui.add_enabled_ui(rc_ok, |ui| w::chip(ui, p, "RC Astro", None, e.engine == Engine::RcAstro)).inner;
        let resp = if let Err(why) = &avail[1].0 { resp.on_disabled_hover_text(why.clone()) } else { resp };
        if resp.clicked() {
            e.engine = Engine::RcAstro;
        }
        src(ui, p, e.engine != be.engine, base_label);
    });
    let ix = if e.engine == Engine::Crux { 0 } else { 1 };
    if e.engine == Engine::RcAstro {
        w::muted(ui, p, "Uses your RC Astro licence: BlurXTerminator for stars, NoiseXTerminator for noise. Their output is marked and never used to train crux's networks.");
    }
    // Stars.
    let sharpen_ok = avail[ix].1.clone();
    ui.horizontal(|ui| {
        ui.add_sized([150.0, 20.0], egui::Label::new(RichText::new("Stars").font(body(13.0))));
        if w::chip(ui, p, "Leave", None, e.sharpen == Sharpen::Off).clicked() {
            e.sharpen = Sharpen::Off;
        }
        for (m, n, tip) in [(Sharpen::Correct, "Correct shapes", "Makes coma and elongated stars round, keeping their size."), (Sharpen::Tighten, "Correct and sharpen", "Also tightens stars and fine detail.")] {
            let resp = ui.add_enabled_ui(sharpen_ok.is_ok(), |ui| w::chip(ui, p, n, None, e.sharpen == m)).inner.on_hover_text(tip);
            let resp = if let Err(why) = &sharpen_ok { resp.on_disabled_hover_text(why.clone()) } else { resp };
            if resp.clicked() {
                e.sharpen = m;
            }
        }
        src(ui, p, e.sharpen != be.sharpen, base_label);
    });
    if let Err(why) = &sharpen_ok {
        if e.sharpen != Sharpen::Off {
            ui.label(RichText::new(format!("Can't sharpen: {why}.")).font(body(12.5)).color(p.warn));
        }
    }
    if e.sharpen != Sharpen::Off && (depth != Depth::Auto || e.engine == Engine::Crux) {
        let mut v = e.sharpen_amount as f64;
        slider(ui, p, "Amount", &mut v, 0.0..=1.0, be.sharpen_amount as f64, base_label, true, "How much of the correction to apply.");
        e.sharpen_amount = v as f32;
    }
    // Noise.
    ui.horizontal(|ui| {
        ui.add_enabled_ui(avail[ix].0.is_ok(), |ui| ui.checkbox(&mut e.denoise, "Remove noise"));
        src(ui, p, e.denoise != be.denoise, base_label);
    });
    if let Err(why) = &avail[ix].0 {
        ui.label(RichText::new(format!("Can't remove noise: {why}.")).font(body(12.5)).color(p.warn));
    }
    if e.denoise {
        let mut v = e.denoise_amount as f64;
        slider(ui, p, "Amount", &mut v, 0.0..=1.0, be.denoise_amount as f64, base_label, true, "How much of the noise to remove. Around 80% keeps a little grain, which usually looks more natural than none.");
        e.denoise_amount = v as f32;
    }
    // What to look at.
    if let Some(t) = img.targets.get(img.sel) {
        let names: Vec<String> = t.masters.iter().map(|m| m.channel.clone()).collect();
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new("Show").font(body(12.5)).color(p.muted));
            for (i, n) in names.iter().enumerate() {
                if w::chip(ui, p, n, None, img.gchan == i).clicked() {
                    img.gchan = i;
                }
            }
            ui.add_space(8.0);
            if w::chip(ui, p, "Centre", None, img.ewhere == 0).clicked() {
                img.ewhere = 0;
            }
            if w::chip(ui, p, "Top-left corner", None, img.ewhere == 1).clicked() {
                img.ewhere = 1;
            }
        });
    }
}

fn settings(ui: &mut Ui, a: &mut UiState, p: &Pal) {
    let st = a.stack.stage;
    let img = &mut a.stack.img;
    let Some(mut r) = img.recipe.clone() else { return };
    let (base, base_label) = img.base.clone().unwrap_or_else(|| (r.clone(), "default".into()));
    let title = match st {
        Stage::Enhance => "Sharpen and denoise",
        Stage::Gradient => "Gradient",
        Stage::Palette => "Palette and stretch",
        Stage::Stars => "Stars",
        Stage::Crop => "Crop",
        _ => "Finish",
    };
    ui.horizontal(|ui| {
        w::h2(ui, title);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if let Some(d) = w::seg(ui, p, &[(Depth::Auto, "Auto"), (Depth::Adjust, "Adjust"), (Depth::Expert, "Expert")], img.depth) {
                img.depth = d;
            }
        });
    });
    let depth = img.depth;
    let pp = img.prepared.clone();
    match st {
        Stage::Enhance => enhance_settings(ui, img, p, &mut r, &base, &base_label, depth),
        Stage::Gradient => {
            w::muted(ui, p, "crux measures sample boxes on a grid, rejects the ones on stars, nebula or dust (red), fits a smooth surface to the rest (green) and subtracts it, keeping the background level. Every channel gets the same degree, so the corners don't take on a tint. This runs on each linear master before mixing.");
            if let Some(pp) = &pp {
                for c in &pp.channels {
                    let t = match &c.report {
                        Some(g) => format!("{}: degree {}, {} of {} boxes kept, {:.1}% of the background across the frame removed, {:.2}% left", c.name, g.degree, g.kept, g.boxes, g.amplitude_frac * 100.0, g.residual_frac * 100.0),
                        None => format!("{}: not corrected", c.name),
                    };
                    ui.label(RichText::new(t).font(body(12.5)).color(p.muted));
                }
                ui.horizontal_wrapped(|ui| {
                    ui.label(RichText::new("Show").font(body(12.5)).color(p.muted));
                    for (i, c) in pp.channels.iter().enumerate() {
                        if w::chip(ui, p, &c.name, None, img.gchan == i).clicked() {
                            img.gchan = i;
                        }
                    }
                    ui.add_space(8.0);
                    if w::chip(ui, p, "Corrected", None, !img.gmodel).clicked() {
                        img.gmodel = false;
                    }
                    if w::chip(ui, p, "Background surface", None, img.gmodel).clicked() {
                        img.gmodel = true;
                    }
                    if w::chip(ui, p, "Samples", None, img.gsamples).clicked() {
                        img.gsamples = !img.gsamples;
                    }
                });
            }
            if depth != Depth::Auto {
                ui.horizontal(|ui| {
                    ui.checkbox(&mut r.gradient.enabled, "Remove gradients");
                    src(ui, p, r.gradient.enabled != base.gradient.enabled, &base_label);
                });
                ui.horizontal(|ui| {
                    ui.add_sized([150.0, 20.0], egui::Label::new(RichText::new("Degree").font(body(13.0))));
                    for (d, n) in [(0u32, "Auto"), (1, "1 · tilt"), (2, "2 · curve"), (3, "3"), (4, "4")] {
                        if d >= 3 && depth != Depth::Expert {
                            continue;
                        }
                        if w::chip(ui, p, n, None, r.gradient.degree == d).clicked() {
                            r.gradient.degree = d;
                        }
                    }
                    src(ui, p, r.gradient.degree != base.gradient.degree, &base_label);
                });
                let mut tol = r.gradient.tolerance as f64;
                slider(ui, p, "Tolerance", &mut tol, 0.3..=3.0, base.gradient.tolerance as f64, &base_label, false, "Boxes this many residual sigmas above the surface are treated as nebula. Lower protects faint nebulosity more.");
                r.gradient.tolerance = tol as f32;
                if depth == Depth::Expert {
                    let mut n = r.gradient.samples as f64;
                    slider_d(ui, p, "Sample boxes across", &mut n, 8.0..=40.0, base.gradient.samples as f64, &base_label, false, "Boxes along the longer side.", 0);
                    r.gradient.samples = n.round() as usize;
                    ui.horizontal(|ui| {
                        ui.add_sized([150.0, 20.0], egui::Label::new(RichText::new("Correction").font(body(13.0))));
                        if w::chip(ui, p, "Subtract", None, r.gradient.correction == gradient::Correction::Subtract).clicked() {
                            r.gradient.correction = gradient::Correction::Subtract;
                        }
                        if w::chip(ui, p, "Divide", None, r.gradient.correction == gradient::Correction::Divide).clicked() {
                            r.gradient.correction = gradient::Correction::Divide;
                        }
                        src(ui, p, r.gradient.correction != base.gradient.correction, &base_label);
                    });
                }
            }
        }
        Stage::Palette => {
            let kind = pp.as_ref().map(|x| x.kind);
            let auto = match kind {
                Some(palette::Kind::Narrowband) => "Each emission line's signal above the background is scaled to match Ha's (limited by 3× Ha's background roughness) and mixed with neutral darks. One linked stretch then puts the background at the level you set.",
                Some(palette::Kind::Broadband) => "R, G and B are balanced so the average star is white, mixed, then one linked stretch. Luminance (Finish) comes from L.",
                _ => "One channel: a stretch that puts the background at the level you set, shadows clipped from the measured noise.",
            };
            w::muted(ui, p, auto);
            if let Some(pp) = &pp {
                let coef = render::coefficients(pp, &r);
                let names = pp.names();
                let g: Vec<String> = names.iter().zip(&coef.gain).filter(|(_, g)| **g > 0.0).map(|(n, g)| format!("{n} ×{g:.2}")).collect();
                ui.label(RichText::new(format!("{:?} balance: {}", r.mix.balance, g.join(", "))).font(body(12.5)).color(p.muted));
                for (i, want, used) in &coef.limited {
                    ui.label(RichText::new(format!("{} would need ×{want:.1} to match; held at ×{used:.1} so its noise and gradients don't colour the sky", names[*i])).font(body(12.5)).color(p.warn));
                }
                if depth != Depth::Auto && kind != Some(palette::Kind::Mono) {
                    ui.add_space(4.0);
                    let mut changed = false;
                    egui::Grid::new("mix").num_columns(4).spacing([12.0, 4.0]).show(ui, |ui| {
                        ui.spacing_mut().slider_width = 200.0;
                        ui.label("");
                        for (c, n) in [(p.f_r, "Red"), (p.f_g, "Green"), (p.f_b, "Blue")] {
                            ui.label(RichText::new(n).font(semibold(12.0)).color(c));
                        }
                        ui.end_row();
                        for n in &names {
                            let e = r.mix.weights.entry(n.clone()).or_insert([0.0; 3]);
                            ui.label(RichText::new(n).font(semibold(13.0)));
                            for v in e.iter_mut() {
                                let mut x = *v * 100.0;
                                if ui.add(egui::Slider::new(&mut x, 0.0..=200.0).suffix("%").fixed_decimals(1).smart_aim(false)).changed() {
                                    *v = x / 100.0;
                                    changed = true;
                                }
                            }
                            ui.end_row();
                        }
                    });
                    if changed {
                        r.mix.preset.clear();
                        r.name = "Custom".into();
                    }
                }
                if depth == Depth::Expert {
                    ui.horizontal(|ui| {
                        ui.add_sized([150.0, 20.0], egui::Label::new(RichText::new("Balance").font(body(13.0))));
                        for (b, n) in [(palette::Balance::Signal, "Signal (mixer)"), (palette::Balance::Stars, "Stars white"), (palette::Balance::None, "None")] {
                            if w::chip(ui, p, n, None, r.mix.balance == b).clicked() {
                                r.mix.balance = b;
                            }
                        }
                        src(ui, p, r.mix.balance != base.mix.balance, &base_label);
                    });
                }
            }
            if depth != Depth::Auto {
                ui.add_space(4.0);
                slider(ui, p, "Background", &mut r.stretch.background, 0.03..=0.45, base.stretch.background, &base_label, false, "Brightness of the background after the stretch. Lower leaves room to stretch further later (the mixer's default 0.15; STF uses 0.25).");
                slider(ui, p, "Black level", &mut r.stretch.black, 0.0..=1.0, base.stretch.black, &base_label, true, "Pulls the background towards black without weakening the stretch of the nebula.");
                slider(ui, p, "Highlights", &mut r.stretch.highlights, 0.0..=1.0, base.stretch.highlights, &base_label, true, "Holds bright nebula back from white, keeping its colour, so cores stay coloured instead of washing out. Stars aren't affected.");
            }
            if depth == Depth::Expert {
                slider(ui, p, "Shadow clip σ", &mut r.stretch.clip, 1.0..=20.0, base.stretch.clip, &base_label, false, "How far below the background the black point sits, in noise sigmas.");
                slider(ui, p, "Keep colour", &mut r.stretch.keep_colour, 0.0..=1.0, base.stretch.keep_colour, &base_label, true, "Stretch brightness only and keep the linear mix's colour ratios, so cores where lines overlap don't wash out to white.");
                slider(ui, p, "Dark reference", &mut r.stretch.dark_ref, 0.0..=0.5, base.stretch.dark_ref, &base_label, true, "The percentile of each channel treated as neutral black. Lower it if dark dust lanes take a colour cast.");
                ui.horizontal(|ui| {
                    ui.checkbox(&mut r.stretch.neutral_sky, "Neutral sky background (small targets)");
                    src(ui, p, r.stretch.neutral_sky != base.stretch.neutral_sky, &base_label);
                    ui.add_space(12.0);
                    ui.checkbox(&mut r.stretch.linear, "Linear input");
                    src(ui, p, r.stretch.linear != base.stretch.linear, &base_label);
                });
            }
        }
        Stage::Stars => {
            let broadband = img.target().is_some_and(|t| t.kind() == palette::Kind::Broadband);
            if broadband {
                w::muted(ui, p, "crux finds each star, takes it out of every master, and stretches the stars on their own: the starless image gets the full stretch, so faint nebula and galaxy halos come up without the stars swelling, and the stars get a gentler stretch screened on top. Each star keeps the colour of its own red, green and blue light, with the median star white. With one stretch, stars and the rest are stretched together, as before.");
            } else {
            w::muted(ui, p, "The palette boosts OIII and SII to match Ha, and stars have a different profile in each filter, so mixed stars turn pink or magenta. crux finds each star, measures its light in every linear master, and gives it one natural colour from that light (R = Ha, B = OIII, G between them, with the median star white), keeping its size and brightness. This replaces StarXTerminator and NBtoRGB Stars. Stretched separately (the default), the starless image gets its own stretch, mix and finishing, and the stars their own gentler stretch, screened on top; light no measured star explains keeps the mixed colour. With one stretch, only the stars change, and faint or doubtful stars and stars on bright nebula keep the mixed colour.");
            }
            let possible = img.target().is_some_and(|t| render::stars_possible(&t.channels()));
            if !possible {
                w::muted(ui, p, "This target has no colour channels to measure star colours from, so its stars stay as they are.");
            } else {
                if let Some(pp) = &pp {
                    if pp.n_stars > 0 {
                        let names = pp.names();
                        let at = |n: &str| names.iter().position(|c| c == n);
                        let gain = if broadband {
                            let gg = |n: &str| at(n).zip(at("G")).map(|(i, j)| pp.star_gain[i] / pp.star_gain[j]).unwrap_or(1.0);
                            format!("red ×{:.2}, blue ×{:.2}", gg("R"), gg("B"))
                        } else {
                            format!("OIII ×{:.2}", at("O").map(|o| pp.star_gain[o] / at("H").map(|h| pp.star_gain[h]).unwrap_or(1.0)).unwrap_or(1.0))
                        };
                        ui.label(RichText::new(format!("{} stars masked across the channels ({:.1}% of the frame); {gain} so the median star is white", pp.n_stars, pp.star_cover * 100.0)).font(body(12.5)).color(p.muted));
                    }
                }
                ui.horizontal_wrapped(|ui| {
                    ui.label(RichText::new("Show").font(body(12.5)).color(p.muted));
                    for (v, n) in [(1u8, "1:1 centre"), (0, "Whole image"), (2, "Starless")] {
                        if w::chip(ui, p, n, None, img.sview == v).clicked() {
                            img.sview = v;
                        }
                    }
                });
                ui.horizontal(|ui| {
                    ui.add_sized([150.0, 20.0], egui::Label::new(RichText::new("Stars").font(body(13.0))));
                    if w::chip(ui, p, "Natural colours", None, r.stars.mode == render::stars::StarMode::Natural).clicked() {
                        r.stars.mode = render::stars::StarMode::Natural;
                    }
                    if w::chip(ui, p, "As mixed", None, r.stars.mode == render::stars::StarMode::Keep).clicked() {
                        r.stars.mode = render::stars::StarMode::Keep;
                    }
                    src(ui, p, r.stars.mode != base.stars.mode, &base_label);
                });
                if r.stars.mode == render::stars::StarMode::Natural {
                    let ok = render::starnet::availability();
                    ui.horizontal(|ui| {
                        ui.add_sized([150.0, 20.0], egui::Label::new(RichText::new("Starless made by").font(body(13.0))));
                        if w::chip(ui, p, "Classic fill", None, r.stars.engine == render::stars::StarEngine::Classic).on_hover_text("Fills each star disc from its surroundings, with matching noise.").clicked() {
                            r.stars.engine = render::stars::StarEngine::Classic;
                        }
                        let resp = ui.add_enabled_ui(ok.is_ok(), |ui| w::chip(ui, p, "Star remover", None, r.stars.engine == render::stars::StarEngine::Network)).inner.on_hover_text("crux's star remover network: cleaner around bright stars and over nebula. The first run takes a minute or two per master.");
                        let resp = if let Err(why) = &ok { resp.on_disabled_hover_text(why.clone()) } else { resp };
                        if resp.clicked() {
                            r.stars.engine = render::stars::StarEngine::Network;
                        }
                        src(ui, p, r.stars.engine != base.stars.engine, &base_label);
                    });
                }
                if r.stars.mode == render::stars::StarMode::Natural {
                    ui.horizontal(|ui| {
                        ui.add_sized([150.0, 20.0], egui::Label::new(RichText::new("Stretch").font(body(13.0))));
                        if w::chip(ui, p, "Stars separately", None, r.stars.separate).on_hover_text("The starless image gets its own stretch, mix and finishing; the stars get a gentler stretch of their own and are screened on top. Stars over bright nebula keep their brightness.").clicked() {
                            r.stars.separate = true;
                        }
                        if w::chip(ui, p, "One stretch", None, !r.stars.separate).on_hover_text("Stars and nebula stretched together; the stars are then recoloured in place.").clicked() {
                            r.stars.separate = false;
                        }
                        src(ui, p, r.stars.separate != base.stars.separate, &base_label);
                    });
                    if r.stars.separate {
                        slider(ui, p, "Star size", &mut r.stars.size, 0.05..=1.5, base.stars.size, &base_label, true, "How hard the stars are stretched. 100% shows a star on dark sky as the nebula stretch would; lower gives smaller, fainter stars, mostly by dimming the faint ones and the halos.");
                        slider(ui, p, "Star brightness", &mut r.stars.brightness, 0.0..=1.5, base.stars.brightness, &base_label, true, "100% screens the stretched stars on as they are; 0% leaves the starless image.");
                        slider_d(ui, p, "Faint star cut", &mut r.stars.faint_cut, 0.0..=30.0, base.stars.faint_cut, &base_label, false, "Star light under this many noise sigmas is left out of the star layer. Raise it to thin out a crowded field's faint stars and halos; the bright stars barely change.", 1);
                    }
                }
                if depth != Depth::Auto && r.stars.mode == render::stars::StarMode::Natural {
                    if !broadband {
                    slider(ui, p, "Red in green", &mut r.stars.ha_to_green, 0.0..=1.0, base.stars.ha_to_green, &base_label, true, "NBtoRGB's ratio: how much of the stars' green comes from red (Ha) rather than blue (OIII). 50% puts green halfway, as in a blackbody; lower gives bluer stars, higher yellower ones.");
                    }
                    slider(ui, p, "Star saturation", &mut r.stars.saturation, 0.0..=2.0, base.stars.saturation, &base_label, true, "100% is the colour the stars measure.");
                }
                if depth == Depth::Expert && r.stars.mode == render::stars::StarMode::Natural {
                    slider(ui, p, "Mask size", &mut r.stars.grow, 0.6..=2.0, base.stars.grow, &base_label, true, "Larger takes more of each star's halo into the star layer (re-measures the masters).");
                    slider_d(ui, p, "Detection σ", &mut r.stars.sigma, 2.5..=8.0, base.stars.sigma, &base_label, false, "Stars must be this many noise sigmas above the small-scale background (re-measures the masters).", 1);
                }
            }
        }
        Stage::Crop => {
            w::muted(ui, p, "Drag the box's edges or corners to resize it, drag inside it to move it, or drag outside it to draw a new one. The crop applies to the full-size image; the other stages show the whole picture. It's kept with this target's recipe, not in saved recipes.");
            let (fw, fh) = pp.as_ref().map(|x| (x.width as f64, x.height as f64)).unwrap_or((1.0, 1.0));
            let cur = r.crop.unwrap_or(recipe::Crop::WHOLE);
            ui.horizontal_wrapped(|ui| {
                ui.add_sized([150.0, 20.0], egui::Label::new(RichText::new("Shape").font(body(13.0))));
                for (i, (name, ratio)) in SHAPES.iter().enumerate() {
                    if w::chip(ui, p, name, None, img.crop_shape == i).clicked() {
                        img.crop_shape = i;
                        // Landscape ratios turn to portrait for a tall crop.
                        let tall = (cur.bottom - cur.top) * fh > (cur.right - cur.left) * fw;
                        img.crop_ratio = ratio.map(|x| if x == 0.0 { fw / fh } else if tall && x != 1.0 { 1.0 / x } else { x });
                        if let Some(a) = img.crop_ratio {
                            r.crop = Some(fit_inside(cur, a, fw, fh));
                        }
                    }
                }
            });
            ui.horizontal_wrapped(|ui| {
                ui.add_sized([150.0, 20.0], egui::Label::new(""));
                if w::button_enabled(ui, p, "Trim empty edges", Kind::Normal, None, pp.is_some()).on_hover_text("The largest box with data in every channel: removes the black edges a mosaic or registration leaves.").clicked() {
                    match pp.as_ref().and_then(|x| render::auto_crop(x)) {
                        Some(c) => {
                            r.crop = Some(match img.crop_ratio {
                                Some(a) => fit_inside(c, a, fw, fh),
                                None => c,
                            });
                            img.crop_note = None;
                        }
                        None => img.crop_note = Some("No empty edges to trim: every channel has data to the edge.".into()),
                    }
                }
                if w::button(ui, p, "Whole picture", Kind::Ghost, None).clicked() {
                    r.crop = None;
                    img.crop_shape = 0;
                    img.crop_ratio = None;
                    img.crop_note = None;
                }
            });
            if let Some(n) = &img.crop_note {
                ui.label(RichText::new(n).font(body(12.5)).color(p.muted));
            }
            if pp.is_some() {
                let (x, y, w, h) = r.crop.unwrap_or(recipe::Crop::WHOLE).pixels(fw as usize, fh as usize);
                let t = if r.crop.is_none_or(|c| c.is_whole()) { format!("Keeps the whole picture, {w} × {h} px.") } else { format!("Keeps {w} × {h} px of {} × {}, from x {x}, y {y}.", fw as usize, fh as usize) };
                ui.label(RichText::new(t).font(body(12.5)).color(p.muted));
            }
        }
        _ => {
            w::muted(ui, p, "Luminance from the channel with the best signal (Ha for narrowband, L for LRGB) for detail, then green removal and saturation, in the mixer's order; the black level is applied last.");
            if depth != Depth::Auto {
                if pp.as_ref().is_none_or(|x| palette::luminance_channel(&x.names(), x.kind).is_some()) {
                    slider(ui, p, "Luminance", &mut r.finish.luminance, 0.0..=1.0, base.finish.luminance, &base_label, true, "How much of the brightness comes from the luminance channel. The colour still comes from the mix.");
                }
                slider(ui, p, "Green removal", &mut r.finish.green, 0.0..=1.0, base.finish.green, &base_label, true, "Reduces green where it exceeds the mean of red and blue, like SCNR.");
                slider(ui, p, "Saturation", &mut r.finish.saturation, 0.0..=2.5, base.finish.saturation, &base_label, true, "Colour saturation. 100% leaves it unchanged.");
                slider(ui, p, "Nebula detail", &mut r.finish.detail, 0.0..=1.0, base.finish.detail, &base_label, true, "Brings out the nebula's structure: bright filaments and dark lanes gain contrast against their surroundings. Works on the starless image, so stars and fine noise aren't sharpened (stars stretched separately only).");
                slider(ui, p, "Colour noise", &mut r.finish.chroma, 0.0..=1.0, base.finish.chroma, &base_label, true, "Smooths colour speckle in the faint background: faint parts take the colour of their surroundings, bright nebula keeps its own (stars stretched separately only).");
                slider(ui, p, "Neutral background", &mut r.finish.neutral, 0.0..=1.0, base.finish.neutral, &base_label, true, "Takes the colour tint out of the faint sky, region by region, fading out towards the nebula. Narrowband palettes boost the weak lines, which tints the faint sky (stars stretched separately, narrowband only).");
            }
        }
    }
    if Some(&r) != img.recipe.as_ref() {
        img.recipe = Some(r);
        img.written = None;
    }
}

fn recipe_card(ui: &mut Ui, a: &mut UiState, p: &Pal) {
    let img = &mut a.stack.img;
    let names = img.target().map(|t| t.channels()).unwrap_or_default();
    let saved = img.saved.get_or_insert_with(recipe::list).clone();
    let Some(r) = img.recipe.as_mut() else { return };
    ui.horizontal(|ui| {
        w::h2(ui, "Recipe");
        ui.label(RichText::new(format!("“{}”", if r.name.is_empty() { "Custom" } else { &r.name })).font(semibold(13.0)));
        ui.label(RichText::new(format!("format {}", r.version)).font(body(11.5)).color(p.faint));
    });
    w::muted(ui, p, "A recipe holds every setting on the image stages in one TOML file. The full-size image stores the recipe that made it, and each target keeps its last recipe.");
    let mut load: Option<Recipe> = None;
    ui.horizontal_wrapped(|ui| {
        egui::ComboBox::from_id_salt("img-recipe").selected_text("Load a recipe").width(200.0).show_ui(ui, |ui| {
            if ui.selectable_label(false, "crux's default for this target").clicked() {
                load = Some(Recipe::for_channels(&names));
            }
            for (n, path) in &saved {
                if ui.selectable_label(false, n).clicked() {
                    if let Ok(x) = Recipe::load(path) {
                        load = Some(x);
                    }
                }
            }
        });
        ui.add_space(12.0);
        ui.add(egui::TextEdit::singleline(&mut img.save_name).hint_text("Name").desired_width(160.0));
        let ok = !img.save_name.trim().is_empty();
        if w::button_enabled(ui, p, "Save recipe", Kind::Normal, None, ok).clicked() {
            r.name = img.save_name.trim().to_string();
            match recipe::save_named(r) {
                Ok(_) => {
                    img.saved = None;
                    img.base = Some((r.clone(), format!("recipe “{}”", r.name)));
                }
                Err(e) => img.error = Some(format!("{e:#}")),
            }
        }
        if let Some(d) = recipe::dir() {
            ui.label(RichText::new(d.display().to_string()).font(body(11.5)).color(p.faint));
        }
    });
    if let Some(mut x) = load {
        x.fit_to(&names);
        // The crop belongs to the target, not the recipe.
        x.crop = img.recipe.as_ref().and_then(|r| r.crop);
        let label = if x.name.is_empty() || saved.iter().all(|(n, _)| n != &x.name) { "default".to_string() } else { format!("recipe “{}”", x.name) };
        img.base = Some((x.clone(), label));
        img.recipe = Some(x);
        img.written = None;
    }
}

pub fn cancel(img: &ImageUi) {
    img.cancel.store(true, Ordering::Relaxed);
}
