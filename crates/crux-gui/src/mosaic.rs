//! Mosaic stage (milestone 14) on the Stack page: a target's panel masters
//! placed by plate-solving each one, distortion-corrected, brightness-
//! matched and feathered into one master per filter
//! (`crux_core::process::mosaic`). The preview shows the mosaic with each
//! panel's outline, coloured by how well its stars line up with its
//! neighbours'.

use crate::app::UiState;
use crate::theme::{body, mono, semibold, Pal};
use crate::widgets::{self as w, Kind};
use crate::worker::Msg;
use crux_core::process::mosaic::{self, Group, Report, Settings};
use eframe::egui::{self, Color32, Pos2, Rect, RichText, Sense, Stroke, Ui, Vec2};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;

#[derive(Default)]
pub struct MosaicUi {
    pub groups: Vec<Group>,
    groups_key: String,
    pub sel: usize,
    /// Filter shown.
    pub filter: usize,
    pub reports: BTreeMap<String, Report>,
    /// Settings being edited, and the config's (for "default / changed").
    pub settings: Option<Settings>,
    base: Settings,
    pub running: Option<String>,
    pub error: Option<String>,
    pub cancel: Arc<AtomicBool>,
    /// Bumped when a mosaic is written (the image stages list it).
    pub generation: u64,
    preview: Option<(String, egui::TextureHandle, f32)>,
    preview_loading: Option<String>,
    preview_error: Option<String>,
    /// Harness: assemble once the page is up.
    pub pending_run: bool,
    /// `[process] fits`: also write FITS copies.
    fits: bool,
}

impl MosaicUi {
    pub fn handle(&mut self, m: Msg, ctx: &egui::Context) -> Option<Msg> {
        match m {
            Msg::MosaicStep(s) => self.running = Some(s),
            Msg::MosaicDone(r) => {
                self.running = None;
                match r {
                    Ok(_) => {
                        self.error = None;
                        self.generation += 1;
                        self.groups_key.clear();
                        self.preview = None;
                    }
                    Err(e) => self.error = Some(e),
                }
            }
            Msg::MosaicPreview(key, r) => {
                self.preview_loading = None;
                match r {
                    Ok((img, factor)) => {
                        let t = ctx.load_texture("mosaic", img, egui::TextureOptions::LINEAR);
                        self.preview = Some((key, t, factor));
                        self.preview_error = None;
                    }
                    Err(e) => self.preview_error = Some(e),
                }
            }
            other => return Some(other),
        }
        None
    }

    pub fn ready_for_shot(&self) -> bool {
        self.running.is_none() && !self.pending_run && self.preview_loading.is_none() && (self.preview.is_some() || self.preview_error.is_some() || self.report().is_none())
    }

    fn group(&self) -> Option<&Group> {
        self.groups.get(self.sel)
    }

    pub fn report(&self) -> Option<&Report> {
        let g = self.group()?;
        let f = g.filters();
        let f = f.get(self.filter.min(f.len().saturating_sub(1)))?;
        self.reports.get(&format!("{}|{}", g.target, f))
    }
}

/// Keep groups and reports current and the preview loaded.
pub fn tick(a: &mut UiState, s: &crate::session::Session, tx: &Sender<Msg>, ctx: &egui::Context) {
    // Process just integrated: go on to the mosaic when it's one of its steps.
    if std::mem::take(&mut a.stack.mosaic_next) && crate::stack::process_steps(a, s).contains(&crate::stack::Stage::Mosaic) {
        a.stack.mos.pending_run = true;
    }
    let Some(plan) = &a.stack.plan else { return };
    let out = plan.out.clone();
    let m = &mut a.stack.mos;
    m.fits = s.cfg.process.fits;
    let key = format!("{}|{}|{}", out.display(), a.stack.integrations.len(), m.generation);
    if key != m.groups_key {
        m.groups_key = key;
        m.groups = mosaic::groups(&out);
        m.reports = mosaic::load_reports(&out);
        if m.sel >= m.groups.len() {
            m.sel = 0;
        }
        m.base = s.cfg.mosaic.clone();
        if m.settings.is_none() {
            m.settings = Some(m.base.clone());
        }
    }
    if m.pending_run && m.running.is_none() && !m.groups.is_empty() {
        m.pending_run = false;
        start(m, &out, tx, ctx);
    }
    // Preview of the shown mosaic.
    let Some(r) = m.report() else { return };
    let pk = format!("{}|{}", r.out.display(), r.signature);
    let path = r.out.clone();
    if m.preview.as_ref().is_some_and(|p| p.0 == pk) || m.preview_loading.as_deref() == Some(&pk) || m.running.is_some() {
        return;
    }
    if m.preview_error.is_some() && m.preview_loading.is_none() && m.preview.is_none() {
        return;
    }
    m.preview_loading = Some(pk.clone());
    let (tx, ctx) = (tx.clone(), ctx.clone());
    std::thread::Builder::new()
        .name("crux-mosaic-preview".into())
        .spawn(move || {
            let res = crux_core::read_image(&path).map_err(|e| format!("{e:#}")).map(|img| {
                let (bw, bh, f, b) = crux_core::stretch::preview(&img, 1600);
                (egui::ColorImage::from_gray([bw, bh], &b), f as f32)
            });
            let _ = tx.send(Msg::MosaicPreview(pk, res));
            ctx.request_repaint();
        })
        .expect("spawn mosaic preview");
}

fn start(m: &mut MosaicUi, out: &std::path::Path, tx: &Sender<Msg>, ctx: &egui::Context) {
    let Some(g) = m.group().cloned() else { return };
    let s = m.settings.clone().unwrap_or_default();
    let fits = m.fits;
    m.cancel = Arc::new(AtomicBool::new(false));
    m.running = Some(format!("Solving {} panel masters", g.inputs.len()));
    m.error = None;
    let (out, tx, ctx, cancel) = (out.to_path_buf(), tx.clone(), ctx.clone(), m.cancel.clone());
    std::thread::Builder::new()
        .name("crux-mosaic".into())
        .spawn(move || {
            let res = (|| {
                let cat = crux_core::catalog::Catalog::find().ok_or_else(|| "no star database to place the panels: install ASTAP with its D50 database, or set CRUX_CATALOG".to_string())?;
                let txp = tx.clone();
                let ctxp = ctx.clone();
                mosaic::assemble(
                    &g,
                    &out,
                    &s,
                    &cat,
                    true,
                    fits,
                    &move |st| {
                        let text = match st {
                            mosaic::Step::Solve { done, total } => format!("Solving panel masters: {done} of {total}"),
                            mosaic::Step::Warp { filter, done, total } => format!("{filter}: placing panels {done} of {total}"),
                            mosaic::Step::Match { filter } => format!("{filter}: matching brightness in the overlaps"),
                            mosaic::Step::Blend { filter } => format!("{filter}: blending"),
                            mosaic::Step::Written { filter, .. } => format!("{filter}: written"),
                            mosaic::Step::Skipped { filter } => format!("{filter}: unchanged"),
                        };
                        let _ = txp.send(Msg::MosaicStep(text));
                        ctxp.request_repaint();
                    },
                    &cancel,
                )
                .map_err(|e| format!("{e:#}"))
            })();
            let _ = tx.send(Msg::MosaicDone(res));
            ctx.request_repaint();
        })
        .expect("spawn mosaic");
}

/// The stage list's line.
pub fn summary(a: &UiState, p: &Pal) -> (Color32, String) {
    let m = &a.stack.mos;
    if let Some(r) = &m.running {
        return (p.warn, format!("Running: {r}"));
    }
    if m.groups.is_empty() {
        return (p.faint, "Only for targets shot as panels".into());
    }
    let g = &m.groups[m.sel.min(m.groups.len() - 1)];
    match m.report() {
        Some(r) => (if r.worst_seam() < 1.0 && r.star_rms() < 0.5 { p.good } else { p.warn }, format!("{}: {} panels, stars line up to {:.2} px", g.target, r.panels.len(), r.star_rms())),
        None => (p.faint, format!("{}: {} panels, not assembled", g.target, g.panels())),
    }
}

pub fn header(ui: &mut Ui, a: &mut UiState, p: &Pal, tx: &Sender<Msg>) {
    let full = ui.available_width();
    let out = a.stack.plan.as_ref().map(|pl| pl.out.clone());
    ui.horizontal(|ui| {
        ui.allocate_ui_with_layout(Vec2::new((full - 380.0).max(200.0), 0.0), egui::Layout::top_down(egui::Align::Min), |ui| {
            w::h1(ui, "Mosaic");
            ui.add(egui::Label::new(RichText::new("Joins the panel masters into one image per filter. crux places each panel by plate-solving its master and fitting the lens distortion to catalogue stars (not by the panel's name), matches brightness in the overlaps, then feathers the seams.").color(p.muted)).wrap());
        });
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let m = &mut a.stack.mos;
            if m.running.is_some() {
                if w::button(ui, p, "Cancel", Kind::Danger, None).clicked() {
                    m.cancel.store(true, Ordering::Relaxed);
                }
            } else {
                let label = if m.report().is_some() { "Assemble again" } else { "Assemble mosaic" };
                if w::button_enabled(ui, p, label, Kind::Primary, None, !m.groups.is_empty()).clicked() {
                    if let Some(o) = &out {
                        start(m, o, tx, ui.ctx());
                    }
                }
                if let Some(r) = m.report() {
                    let dir = r.out.parent().map(PathBuf::from);
                    if w::button(ui, p, "Open output folder", Kind::Ghost, None).clicked() {
                        if let Some(d) = dir {
                            let _ = crate::stack::open_folder(&d);
                        }
                    }
                }
            }
        });
    });
    let m = &a.stack.mos;
    if let Some(r) = &m.running {
        ui.add_space(6.0);
        ui.label(RichText::new(r).color(p.warn));
    }
    if let Some(e) = &m.error {
        ui.add_space(6.0);
        ui.label(RichText::new(format!("Couldn't assemble the mosaic: {e}")).color(p.rej));
    }
}

fn star_colour(p: &Pal, rms: f64) -> Color32 {
    if !rms.is_finite() {
        p.faint
    } else if rms < 0.3 {
        p.good
    } else if rms < 1.0 {
        p.warn
    } else {
        p.rej
    }
}

pub fn main(ui: &mut Ui, a: &mut UiState, p: &Pal) {
    ui.spacing_mut().item_spacing.y = 14.0;
    let m = &mut a.stack.mos;
    if m.groups.is_empty() {
        w::card(ui, p, |ui| {
            w::h2(ui, "No mosaic here");
            ui.label(RichText::new("A mosaic needs integrated masters of two or more panels of one target (objects named \"<target> Panel <n>\"). Run the stacking steps through Integrate first.").color(p.muted));
        });
        return;
    }
    // Target and filter chips.
    if m.groups.len() > 1 || m.groups[m.sel.min(m.groups.len() - 1)].filters().len() > 1 {
        ui.horizontal_wrapped(|ui| {
            for i in 0..m.groups.len() {
                let name = format!("{} ({} panels)", m.groups[i].target, m.groups[i].panels());
                if w::chip(ui, p, &name, None, m.sel == i).clicked() && m.sel != i {
                    m.sel = i;
                    m.filter = 0;
                    m.preview = None;
                }
            }
            let fs = m.groups[m.sel.min(m.groups.len() - 1)].filters();
            if fs.len() > 1 {
                ui.add_space(12.0);
                for (k, f) in fs.iter().enumerate() {
                    if w::chip(ui, p, f, None, m.filter == k).clicked() && m.filter != k {
                        m.filter = k;
                        m.preview = None;
                    }
                }
            }
        });
    }
    preview(ui, m, p);
    if let Some(r) = m.report().cloned() {
        w::card(ui, p, |ui| numbers(ui, &r, p));
    }
    w::card(ui, p, |ui| settings(ui, m, p));
    if let Some(r) = m.report().cloned() {
        w::card(ui, p, |ui| panel_table(ui, &r, p));
    }
}

fn preview(ui: &mut Ui, m: &mut MosaicUi, p: &Pal) {
    let wd = ui.available_width();
    let Some(r) = m.report().cloned() else {
        let (rect, _) = ui.allocate_exact_size(Vec2::new(wd, (wd * 0.5).min(420.0)), Sense::hover());
        ui.painter().rect_filled(rect, 10.0, p.well);
        let g = m.group().map(|g| format!("{}: {} panels, {}", g.target, g.panels(), g.filters().join(" "))).unwrap_or_default();
        ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, format!("{g}\nNot assembled yet"), body(13.0), p.muted);
        return;
    };
    let (cw, ch) = (r.canvas.width as f32, r.canvas.height as f32);
    let max_h = (ui.ctx().content_rect().height() * 0.6).max(300.0);
    let (iw, ih) = if wd * ch / cw > max_h { (max_h * cw / ch, max_h) } else { (wd, wd * ch / cw) };
    let (outer, _) = ui.allocate_exact_size(Vec2::new(wd, ih), Sense::hover());
    ui.painter().rect_filled(outer, 10.0, p.well);
    let rect = Rect::from_center_size(outer.center(), Vec2::new(iw, ih));
    let pa = ui.painter_at(rect);
    match &m.preview {
        Some((_, t, _)) => {
            pa.image(t.id(), rect, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
        }
        None => {
            let msg = m.preview_error.clone().map(|e| format!("Couldn't read the mosaic: {e}")).unwrap_or_else(|| "Reading the mosaic…".into());
            pa.text(rect.center(), egui::Align2::CENTER_CENTER, msg, body(13.0), p.muted);
        }
    }
    // Panel outlines, coloured by their worst overlap's star alignment.
    let k = iw / cw;
    let to = |q: &[f64; 2]| Pos2::new(rect.left() + q[0] as f32 * k, rect.top() + q[1] as f32 * k);
    let hover = ui.input(|i| i.pointer.hover_pos());
    let mut tip: Option<String> = None;
    for pr in &r.panels {
        if pr.outline.len() < 4 {
            continue;
        }
        let worst = r.seams.iter().filter(|s| (s.a == pr.object || s.b == pr.object) && s.stars >= 5).map(|s| s.star_rms).fold(0.0, f64::max);
        let col = star_colour(p, worst);
        let pts: Vec<Pos2> = pr.outline.iter().map(to).collect();
        for i in 0..pts.len() {
            pa.line_segment([pts[i], pts[(i + 1) % pts.len()]], Stroke::new(1.0, col.gamma_multiply(0.8)));
        }
        let c = pts.iter().fold(Vec2::ZERO, |acc, q| acc + q.to_vec2()) / pts.len() as f32;
        let label = pr.panel.map(|n| n.to_string()).unwrap_or_else(|| pr.object.clone());
        pa.text(c.to_pos2(), egui::Align2::CENTER_CENTER, label, mono(11.0), col);
        if let Some(h) = hover {
            let (x0, x1) = pts.iter().fold((f32::MAX, f32::MIN), |(a, b), q| (a.min(q.x), b.max(q.x)));
            let (y0, y1) = pts.iter().fold((f32::MAX, f32::MIN), |(a, b), q| (a.min(q.y), b.max(q.y)));
            if h.x >= x0 && h.x <= x1 && h.y >= y0 && h.y <= y1 && (h - c.to_pos2()).length() < 0.3 * (x1 - x0).min(y1 - y0) {
                tip = Some(format!(
                    "{}: {} frames, {:.1} h\nbrightness ×{:.3}, offset {:+.5}\ncatalogue stars {:.2} px rms ({} matched)\nworst overlap: stars {:.2} px",
                    pr.object,
                    pr.frames,
                    pr.exposure / 3600.0,
                    pr.brightness.scale,
                    pr.brightness.offset,
                    pr.rms,
                    pr.matched,
                    worst
                ));
            }
        }
    }
    if let Some(t) = tip {
        egui::Tooltip::always_open(ui.ctx().clone(), ui.layer_id(), ui.scope_id().with("mos-tip"), egui::PopupAnchor::Pointer).show(|ui| {
            ui.label(RichText::new(t).font(body(12.0)));
        });
    }
    ui.horizontal(|ui| {
        for (c, t) in [(p.good, "stars within 0.3 px"), (p.warn, "0.3–1 px"), (p.rej, "over 1 px")] {
            ui.label(RichText::new("■").color(c));
            ui.label(RichText::new(t).font(body(12.0)).color(p.muted));
            ui.add_space(8.0);
        }
    });
}

fn numbers(ui: &mut Ui, r: &Report, p: &Pal) {
    w::h2(ui, format!("{} {}", r.target, r.filter));
    let med = |mut v: Vec<f64>| crux_core::stats::median_f64(&mut v).unwrap_or(f64::NAN);
    let ok: Vec<&mosaic::PanelReport> = r.panels.iter().filter(|x| x.error.is_none()).collect();
    let rows = [
        ("Canvas", format!("{} × {} px at {:.2}″/px (bin {}), {:.0} MP", r.canvas.width, r.canvas.height, r.canvas.scale, r.canvas.bin, r.canvas.width as f64 * r.canvas.height as f64 / 1e6)),
        ("Panels", format!("{} of {} placed, {} frames, {:.1} h", ok.len(), r.panels.len(), r.frames, r.exposure / 3600.0)),
        ("Stars in the overlaps", format!("line up to {:.2} px rms (median of {} overlaps, panel pixels)", r.star_rms(), r.seams.len())),
        ("Catalogue stars", format!("{:.2} px rms per panel with the distortion fit (median), up to {:.0} px corrected at the corners", med(ok.iter().map(|x| x.rms).collect()), med(ok.iter().map(|x| x.distortion_max).collect()))),
        ("Seams", format!("brightness step {:.3}× the noise before matching, {:.3}× after (median); worst {:.2}×", med(r.seams.iter().map(|s| s.before.abs()).collect()), med(r.seams.iter().map(|s| s.after.abs()).collect()), r.worst_seam())),
        ("Time", format!("{:.0} s", r.seconds)),
    ];
    egui::Grid::new("mos-numbers").num_columns(2).spacing([18.0, 6.0]).show(ui, |ui| {
        for (k, v) in rows {
            ui.label(RichText::new(k).color(p.muted));
            ui.label(RichText::new(v));
            ui.end_row();
        }
    });
    for x in r.panels.iter().filter(|x| x.error.is_some()) {
        ui.label(RichText::new(format!("{} left out: {}", x.object, x.error.as_deref().unwrap_or(""))).color(p.rej));
    }
}


fn settings(ui: &mut Ui, m: &mut MosaicUi, p: &Pal) {
    w::h2(ui, "Settings");
    let base = m.base.clone();
    let built = m.report().map(|r| r.settings.clone());
    let Some(s) = m.settings.as_mut() else { return };
    ui.horizontal(|ui| {
        ui.add_sized([150.0, 20.0], egui::Label::new(RichText::new("Resolution").font(body(13.0)))).on_hover_text("Canvas pixels per panel pixel. Auto picks the smallest bin that keeps the canvas under 200 megapixels.");
        for (b, name) in [(0usize, "Auto"), (1, "Full"), (2, "½"), (3, "⅓"), (4, "¼")] {
            if w::chip(ui, p, name, None, s.bin == b).clicked() {
                s.bin = b;
            }
        }
        w::reset(ui, p, &mut s.bin, &base.bin, "default");
    });
    ui.horizontal(|ui| {
        ui.add_sized([150.0, 20.0], egui::Label::new(RichText::new("Feather").font(body(13.0)))).on_hover_text("How far in from each panel's edge the blend reaches. Wider hides residual differences; narrower keeps more of each panel's sharpest centre.");
        ui.add(egui::Slider::new(&mut s.feather_arcmin, 1.0..=20.0).suffix("′").fixed_decimals(0));
        if w::source(ui, p, (s.feather_arcmin - base.feather_arcmin).abs() > 1e-9, "default") {
            s.feather_arcmin = base.feather_arcmin;
        }
    });
    ui.horizontal(|ui| {
        ui.add_sized([150.0, 20.0], egui::Label::new(RichText::new("Lens distortion").font(body(13.0)))).on_hover_text("Fit each panel's distortion from catalogue stars (a cubic). Without it, stars at the panel corners can miss by tens of pixels.");
        ui.checkbox(&mut s.distortion, "");
        w::reset(ui, p, &mut s.distortion, &base.distortion, "default");
    });
    ui.horizontal(|ui| {
        ui.add_sized([150.0, 20.0], egui::Label::new(RichText::new("Match brightness").font(body(13.0)))).on_hover_text("Scale each panel by star photometry in the overlaps, then fit an offset and plane per panel so the seams don't show.");
        ui.checkbox(&mut s.match_brightness, "");
        w::reset(ui, p, &mut s.match_brightness, &base.match_brightness, "default");
    });
    ui.horizontal(|ui| {
        ui.add_sized([150.0, 20.0], egui::Label::new(RichText::new("Curved skies").font(body(13.0)))).on_hover_text("Fit a gentle curve per panel as well as a plane. This helps panels integrated against a cloudy or dawn reference frame, which keep a curved sky.");
        ui.add_enabled(s.match_brightness, egui::Checkbox::new(&mut s.curvature, ""));
        w::reset(ui, p, &mut s.curvature, &base.curvature, "default");
    });
    if built.is_some_and(|b| b != *s) {
        ui.label(RichText::new("Changed: press Assemble again to apply.").font(body(12.0)).color(p.accent));
    }
}

fn panel_table(ui: &mut Ui, r: &Report, p: &Pal) {
    w::h2(ui, "Panels");
    egui::ScrollArea::vertical().id_salt("mos-panels").max_height(320.0).show(ui, |ui| {
        egui::Grid::new("mos-table").num_columns(8).striped(true).spacing([14.0, 4.0]).show(ui, |ui| {
            for h in ["Panel", "Frames", "Hours", "Brightness", "Offset", "Catalogue rms", "Worst overlap", "Noise"] {
                ui.label(RichText::new(h).font(semibold(12.0)).color(p.muted));
            }
            ui.end_row();
            for x in &r.panels {
                let worst = r.seams.iter().filter(|s| (s.a == x.object || s.b == x.object) && s.stars >= 5).map(|s| s.star_rms).fold(0.0, f64::max);
                ui.label(RichText::new(x.panel.map(|n| n.to_string()).unwrap_or_else(|| x.object.clone())).font(mono(12.0)));
                ui.label(RichText::new(x.frames.to_string()).font(mono(12.0)));
                ui.label(RichText::new(format!("{:.1}", x.exposure / 3600.0)).font(mono(12.0)));
                ui.label(RichText::new(format!("×{:.3}", x.brightness.scale)).font(mono(12.0)).color(if (x.brightness.scale - 1.0).abs() > 0.3 { p.warn } else { p.fg }));
                ui.label(RichText::new(format!("{:+.5}", x.brightness.offset)).font(mono(12.0)));
                ui.label(RichText::new(if x.rms.is_finite() { format!("{:.2} px", x.rms) } else { "—".into() }).font(mono(12.0)));
                ui.label(RichText::new(format!("{worst:.2} px")).font(mono(12.0)).color(star_colour(p, worst)));
                ui.label(RichText::new(format!("{:.2e}", x.noise)).font(mono(12.0)));
                ui.end_row();
            }
        });
    });
}
