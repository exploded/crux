//! Calibration (milestone 10): what each light group is calibrated with,
//! and why. The matching graph, flat heat maps (against the lights, set
//! against set, frame drift), the set table and detail, flat pairs, the flat
//! library browser (drag a set onto a night to use it there) and the darks
//! and bias ranked across the library by fingerprint.

use crate::app::UiState;
use crate::session::Session;
use crate::theme::{alpha, body, mix, mono, semibold, Pal};
use crate::widgets::{self as w, d_short};
use chrono::NaiveDate;
use crux_core::calcheck::{FlatChoice, SetReport};
use crux_core::calib;
use crux_core::fingerprint::{Family, Fingerprints, Ranked};
use crux_core::grade::Grade;
use crux_core::FrameType;
use eframe::egui::{self, Color32, CornerRadius, Pos2, Rect, RichText, ScrollArea, Sense, Stroke, StrokeKind, Ui, Vec2};
use std::collections::{BTreeMap, HashMap};

/// What a heat map shows.
#[derive(Clone, PartialEq)]
enum Heat {
    /// Flat set (group) against the lights.
    VsLights(usize),
    /// Set a ÷ set b.
    Pair(usize, usize),
    /// A flat set's worst frame against the set.
    Drift(usize),
}

#[derive(Default)]
pub struct CalUi {
    pub sel_set: Option<usize>,
    pub sel_pair: Option<(usize, usize)>,
    pub sel_choice: Option<usize>,
    pub fp: Option<Fingerprints>,
    pub fp_progress: Option<(usize, usize)>,
    pub fam: usize,
    /// Expanded candidate row in the darks list (family, candidate).
    pub open_row: Option<(usize, usize)>,
    /// Scroll here once (screenshot harness).
    pub scroll_to: Option<f32>,
    textures: Vec<(Heat, egui::TextureHandle, f64, String)>,
    /// The drift heat map's frame and change.
    drift: HashMap<usize, Option<(Vec<f32>, usize, usize, f64, String)>>,
    generation: u64,
}

impl CalUi {
    /// A new folder: forget everything.
    pub fn reset(&mut self) {
        *self = CalUi::default();
    }
}

/// A flat set dragged from the library browser.
#[derive(Clone)]
struct DragFlat {
    filter: String,
    library_id: Option<String>,
    group: Option<String>,
    night: NaiveDate,
    label: String,
}

/// Diverging colour for an error of `t` (−1 … +1 = −10% … +10%), as in the mockup.
fn div(t: f32) -> Color32 {
    let t = t.clamp(-1.0, 1.0);
    let lerp = |a: [f32; 3], b: [f32; 3], f: f32| Color32::from_rgb((a[0] + (b[0] - a[0]) * f) as u8, (a[1] + (b[1] - a[1]) * f) as u8, (a[2] + (b[2] - a[2]) * f) as u8);
    let (c0, c1, c2, c3, c4) = ([44.0, 107.0, 255.0], [159.0, 184.0, 255.0], [244.0, 244.0, 244.0], [255.0, 179.0, 138.0], [255.0, 59.0, 48.0]);
    if t < -0.5 {
        lerp(c0, c1, (t + 1.0) * 2.0)
    } else if t < 0.0 {
        lerp(c1, c2, (t + 0.5) * 2.0)
    } else if t < 0.5 {
        lerp(c2, c3, t * 2.0)
    } else {
        lerp(c3, c4, (t - 0.5) * 2.0)
    }
}

/// A median-1 map as a ±`range`% heat map.
fn heat_image(map: &[f32], w: usize, h: usize, range: f32) -> egui::ColorImage {
    let px: Vec<Color32> = map.iter().map(|v| div(if *v > 0.0 { (v - 1.0) * 100.0 / range } else { 0.0 })).collect();
    egui::ColorImage { size: [w, h], pixels: px, source_size: Vec2::new(w as f32, h as f32) }
}

fn short(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(n - 1).collect::<String>())
    }
}

/// "H flats · 19 Oct", "library H 1 Nov", "master flat H".
fn set_name(s: &Session, gid: usize) -> String {
    let g = &s.triage.grouping.groups[gid];
    if g.master {
        let f = g.filter.clone().unwrap_or_default().to_uppercase();
        return match g.kind {
            FrameType::Flat => format!("Master flat {f}{}", if g.retired { " (from retired set)" } else { "" }),
            FrameType::Dark => format!("Master dark {}", g.exptime.map(|e| format!("{e:.0} s")).unwrap_or_default()),
            FrameType::Bias => "Master bias".into(),
            _ => g.label.clone(),
        };
    }
    match (g.kind, g.night) {
        (FrameType::Flat, Some(n)) => {
            let folder = s.triage.scan.frames[g.members[0]].meta.path.parent().and_then(|p| p.file_name()).map(|x| x.to_string_lossy().to_string()).unwrap_or_default();
            let bad = if g.retired { format!(" ({folder})") } else { String::new() };
            format!("{} flats · {}{bad}", g.filter.clone().unwrap_or_default().to_uppercase(), d_short(n))
        }
        _ => g.label.clone(),
    }
}

fn opt_name(o: &crux_core::calcheck::FlatOption, s: &Session) -> String {
    match (o.group, &o.library_id) {
        (Some(g), _) => set_name(s, g),
        (None, Some(_)) => o.label.replace("library ", "Library "),
        _ => o.label.clone(),
    }
}

pub fn view(ui: &mut Ui, a: &mut UiState, s: &mut Session) {
    let p = a.pal;
    if a.cal.generation != s.generation {
        a.cal.generation = s.generation;
    }
    let mut sa = ScrollArea::vertical().id_salt("calibrate").auto_shrink([false, false]);
    if let Some(y) = a.cal.scroll_to.filter(|_| a.cal.fp.is_some()) {
        a.cal.scroll_to = None;
        sa = sa.vertical_scroll_offset(y);
    }
    sa.show(ui, |ui| {
        ui.set_max_width(1400.0);
        w::h1(ui, "Calibration");
        ui.label(RichText::new("The flats, darks and bias for each light group, and why crux chose them. Every flat set gets the FlatChecker checks. Darks and bias are matched to the lights by their hot pixels, from anywhere in the library.").color(p.muted));
        ui.add_space(14.0);
        let full = ui.available_width();
        let two = full >= 980.0;
        let gap = 16.0;
        let lw = if two { (full - gap) * 0.53 } else { full };
        let rw = if two { full - gap - lw } else { full };
        if two {
            // Two child columns; the page reserves the taller one.
            let top = ui.cursor().min;
            let mk = |x: f32, wd: f32| egui::UiBuilder::new().max_rect(Rect::from_min_size(Pos2::new(top.x + x, top.y), Vec2::new(wd, f32::INFINITY))).layout(egui::Layout::top_down(egui::Align::Min));
            let mut lu = ui.new_child(mk(0.0, lw));
            lu.set_width(lw);
            left(&mut lu, a, s, &p);
            let mut ru = ui.new_child(mk(lw + gap, rw));
            ru.set_width(rw);
            right(&mut ru, a, s, &p);
            ui.advance_cursor_after_rect(lu.min_rect().union(ru.min_rect()));
        } else {
            left(ui, a, s, &p);
            right(ui, a, s, &p);
        }
        ui.add_space(24.0);
    });
}

fn left(ui: &mut Ui, a: &mut UiState, s: &mut Session, p: &Pal) {
    ui.spacing_mut().item_spacing.y = 16.0;
    w::card(ui, p, |ui| graph(ui, a, s, p));
    w::card(ui, p, |ui| heat_card(ui, a, s, p));
    w::card(ui, p, |ui| darks_card(ui, a, s, p));
}

fn right(ui: &mut Ui, a: &mut UiState, s: &mut Session, p: &Pal) {
    ui.spacing_mut().item_spacing.y = 16.0;
    w::card(ui, p, |ui| sets_table(ui, a, s, p));
    if a.cal.sel_set.is_some() || a.cal.sel_pair.is_some() {
        w::card(ui, p, |ui| detail(ui, a, s, p));
    }
    if !s.triage.cal.pairs.is_empty() {
        w::card(ui, p, |ui| pairs(ui, a, s, p));
    }
    w::card(ui, p, |ui| library(ui, a, s, p));
}

// ---------------------------------------------------------------- matching graph

/// The dark (or bias) the lights will use: yours, else the fingerprint's
/// current match, as (label, is yours, better available).
fn offset_node(a: &UiState, s: &Session, kind: &str) -> Option<(String, bool, Option<String>)> {
    let fp = a.cal.fp.as_ref()?;
    let fam = fp.families.iter().find(|f| f.kind == FrameType::Light)?;
    let list = if kind == "dark" { &fam.darks } else { &fam.bias };
    let cur = if kind == "dark" { fam.current_dark } else { fam.current_bias };
    let yours = s.decisions.offsets.get(&crux_core::decisions::offset_key(&s.root, kind, fam));
    let pick = yours.and_then(|k| list.iter().find(|r| fp.candidates[r.cand].files.iter().chain(&fp.candidates[r.cand].copies).any(|f| &crux_core::paths::key(f) == k)));
    let r = pick.or_else(|| cur.and_then(|c| list.iter().find(|r| r.cand == c))).or(list.first())?;
    let better = if kind == "dark" && pick.is_none() { fam.better_dark().map(|b| fp.candidates[b.cand].label.clone()) } else { None };
    Some((fp.candidates[r.cand].label.clone(), pick.is_some(), better))
}

fn graph(ui: &mut Ui, a: &mut UiState, s: &mut Session, p: &Pal) {
    ui.horizontal(|ui| {
        w::h2(ui, "Matching");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(RichText::new("dashed: check this · drag a library set onto a night").font(body(11.5)).color(p.muted));
        });
    });
    ui.add_space(6.0);
    let choices: Vec<FlatChoice> = s.triage.cal.choices.clone();
    if choices.is_empty() {
        ui.label(RichText::new("No light groups to calibrate in this folder.").color(p.muted));
        return;
    }
    let row_h = 30.0;
    let top = 26.0;
    let wd = ui.available_width().max(360.0);
    let ht = top + row_h * choices.len() as f32 + 8.0;
    let (rect, _) = ui.allocate_exact_size(Vec2::new(wd, ht), Sense::hover());
    let pa = ui.painter_at(rect.expand(2.0));
    let (c1, c2, c3) = (rect.left(), rect.left() + wd * 0.40, rect.left() + wd * 0.74);
    let lw = wd * 0.27;
    for (x, t) in [(c1, "LIGHT GROUPS"), (c2, "FLATS"), (c3, "DARK AND BIAS")] {
        pa.text(Pos2::new(x, rect.top() + 6.0), egui::Align2::LEFT_TOP, t, body(10.5), p.muted);
    }
    let ly = |i: usize| rect.top() + top + row_h * i as f32 + row_h / 2.0;
    // Flat nodes: one per chosen set, at the mean height of its nights.
    let mut nodes: BTreeMap<String, (Vec<usize>, String, Grade, Option<usize>, bool)> = BTreeMap::new();
    for (i, c) in choices.iter().enumerate() {
        let (key, name, g, gid) = match &c.chosen {
            Some(o) => (o.library_id.clone().or(o.group.map(|g| format!("g{g}"))).unwrap_or_else(|| o.label.clone()), opt_name(o, s), o.grade, o.group),
            None => ("none".into(), "No usable flats".into(), Grade::Reject, None),
        };
        let e = nodes.entry(key).or_insert((Vec::new(), name, g, gid, false));
        e.0.push(i);
        e.4 |= c.pinned;
    }
    let mut node_y: HashMap<String, f32> = HashMap::new();
    let mut ordered: Vec<(&String, f32)> = nodes.iter().map(|(k, v)| (k, v.0.iter().map(|i| ly(*i)).sum::<f32>() / v.0.len() as f32)).collect();
    ordered.sort_by(|a, b| a.1.total_cmp(&b.1));
    // Keep nodes at least a row apart.
    let mut last = f32::MIN;
    for (k, y) in ordered {
        let y = y.max(last + row_h);
        last = y;
        node_y.insert(k.clone(), y);
    }
    let dark = offset_node(a, s, "dark");
    let bias = offset_node(a, s, "bias");
    let dark_y = ly(0);
    let bias_y = ly(choices.len().saturating_sub(1)).max(dark_y + row_h * 1.5);
    let nw = c3 - c2 - 18.0;
    let ow = rect.right() - c3 - 2.0;
    let curve = |pa: &egui::Painter, a: Pos2, b: Pos2, st: Stroke, dashed: bool| {
        let pts: Vec<Pos2> = (0..=16)
            .map(|k| {
                let t = k as f32 / 16.0;
                let (mx1, mx2) = (a.x + (b.x - a.x) * 0.5, a.x + (b.x - a.x) * 0.5);
                let u = 1.0 - t;
                let x = u * u * u * a.x + 3.0 * u * u * t * mx1 + 3.0 * u * t * t * mx2 + t * t * t * b.x;
                let y = u * u * u * a.y + 3.0 * u * u * t * a.y + 3.0 * u * t * t * b.y + t * t * t * b.y;
                Pos2::new(x, y)
            })
            .collect();
        if dashed {
            pa.extend(egui::Shape::dashed_line(&pts, st, 4.0, 3.0));
        } else {
            pa.add(egui::Shape::line(pts, st));
        }
    };
    // Links first, nodes on top.
    for (i, c) in choices.iter().enumerate() {
        let y = ly(i);
        let key = c.chosen.as_ref().map(|o| o.library_id.clone().or(o.group.map(|g| format!("g{g}"))).unwrap_or_else(|| o.label.clone())).unwrap_or_else(|| "none".into());
        let ny = node_y.get(&key).copied().unwrap_or(y);
        let col = if c.pinned { p.accent } else { p.grade(c.grade) };
        curve(&pa, Pos2::new(c1 + lw, y), Pos2::new(c2, ny), Stroke::new(1.5, alpha(col, 0.75)), c.grade != Grade::Good);
        if dark.is_some() {
            curve(&pa, Pos2::new(c1 + lw, y), Pos2::new(c3, dark_y), Stroke::new(1.0, alpha(p.faint, 0.45)), false);
        }
    }
    if bias.is_some() {
        for (k, y) in &node_y {
            if k != "none" {
                curve(&pa, Pos2::new(c2 + nw, *y), Pos2::new(c3, bias_y), Stroke::new(1.0, alpha(p.faint, 0.45)), false);
            }
        }
    }
    // Light rows: click to select, drop a library set to use it.
    let mut picked: Option<(usize, DragFlat)> = None;
    for (i, c) in choices.iter().enumerate() {
        let y = ly(i);
        let r = Rect::from_min_size(Pos2::new(c1, y - 12.0), Vec2::new(lw, 24.0));
        let resp = ui.interact(r, ui.scope_id().with(("calrow", i)), Sense::click());
        let hover_drop = resp.dnd_hover_payload::<DragFlat>().filter(|d| d.filter == c.filter);
        let sel = a.cal.sel_choice == Some(i);
        let fill = if hover_drop.is_some() { alpha(p.accent, 0.25) } else if sel { alpha(p.accent, 0.14) } else { p.panel2 };
        pa.rect(r, CornerRadius::same(6), fill, Stroke::new(1.0, if sel || hover_drop.is_some() { p.accent } else { p.line }), StrokeKind::Inside);
        let fc = p.filter(&c.filter);
        pa.rect_filled(Rect::from_min_size(Pos2::new(r.left() + 6.0, y - 7.0), Vec2::new(14.0, 14.0)), CornerRadius::same(3), fc);
        pa.text(Pos2::new(r.left() + 13.0, y), egui::Align2::CENTER_CENTER, &c.filter, mono(10.0), p.bg);
        pa.text(Pos2::new(r.left() + 26.0, y), egui::Align2::LEFT_CENTER, format!("{} · {} lights", d_short(c.night), c.n_lights), body(11.5), p.fg);
        if let Some(d) = resp.dnd_release_payload::<DragFlat>() {
            if d.filter == c.filter {
                picked = Some((i, (*d).clone()));
            }
        }
        let tip = format!("{}{}", c.reason, if c.pinned { "\n\nYour choice. Select the row and press \"Back to crux's choice\" to undo." } else { "" });
        if resp.on_hover_text(tip).clicked() {
            a.cal.sel_choice = if sel { None } else { Some(i) };
        }
    }
    // Flat nodes.
    for (k, (_rows, name, g, gid, pinned)) in &nodes {
        let y = node_y[k];
        let r = Rect::from_min_size(Pos2::new(c2, y - 13.0), Vec2::new(nw, 26.0));
        let col = if *pinned { p.accent } else { p.grade(*g) };
        pa.rect(r, CornerRadius::same(6), mix(col, 0.14, p.panel), Stroke::new(1.0, col), StrokeKind::Inside);
        pa.text(Pos2::new(r.left() + 8.0, y), egui::Align2::LEFT_CENTER, short(name, (nw / 6.6) as usize), body(11.5), p.fg);
        if let Some(gid) = gid {
            let resp = ui.interact(r, ui.scope_id().with(("calnode", k)), Sense::click());
            if resp.on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
                a.cal.sel_set = Some(*gid);
                a.cal.sel_pair = None;
            }
        }
    }
    // Dark and bias nodes.
    for (y, node, what) in [(dark_y, &dark, "dark"), (bias_y, &bias, "bias")] {
        let r = Rect::from_min_size(Pos2::new(c3, y - 12.0), Vec2::new(ow, 24.0));
        match node {
            Some((label, yours, better)) => {
                let col = if *yours { p.accent } else if better.is_some() { p.warn } else { p.line };
                pa.rect(r, CornerRadius::same(6), p.panel2, Stroke::new(1.0, col), StrokeKind::Inside);
                if better.is_some() {
                    w::dashed_rect(&pa, r.expand(3.0), p.warn);
                }
                pa.text(Pos2::new(r.left() + 8.0, y), egui::Align2::LEFT_CENTER, short(label, (ow / 6.6) as usize), body(11.5), p.fg);
                if let Some(b) = better {
                    ui.interact(r, ui.scope_id().with(("caloff", what)), Sense::hover()).on_hover_text(format!("{b} matches the lights' hot pixels better. See Darks and bias below."));
                }
            }
            None => {
                pa.rect(r, CornerRadius::same(6), Color32::TRANSPARENT, Stroke::new(1.0, p.line), StrokeKind::Inside);
                let t = if a.cal.fp_progress.is_some() || a.cal.fp.is_none() { "matching…" } else { "none in the library" };
                pa.text(Pos2::new(r.left() + 8.0, y), egui::Align2::LEFT_CENTER, format!("{} {t}", if what == "dark" { "Dark" } else { "Bias" }), body(11.5), p.muted);
            }
        }
    }
    if let Some((i, d)) = picked {
        let c = &choices[i];
        s.pick_flat(&c.filter, c.night, Some((d.library_id.clone(), d.group.clone(), d.night)));
        a.toast(ui.ctx(), format!("{} lights on {} will use {}. Saved in your decisions; moves and processing use it.", c.filter, d_short(c.night), d.label), false);
        a.cal.sel_choice = Some(i);
    }
    // The selected night: crux's reason, and the way back.
    if let Some(c) = a.cal.sel_choice.and_then(|i| s.triage.cal.choices.get(i)).cloned() {
        ui.add_space(8.0);
        ui.horizontal_wrapped(|ui| {
            w::filter_badge(ui, p, &c.filter);
            ui.label(RichText::new(format!("{} · {} lights", d_short(c.night), c.n_lights)).font(semibold(13.0)));
            ui.label(RichText::new(&c.reason).font(body(12.5)).color(p.muted));
        });
        if c.pinned {
            if w::button(ui, p, "Back to crux's choice", w::Kind::Ghost, None).clicked() {
                s.pick_flat(&c.filter, c.night, None);
                a.toast(ui.ctx(), "Back to crux's flat choice.", false);
            }
        } else if !c.others.is_empty() {
            ui.label(RichText::new(format!("Also usable: {}", c.others.iter().take(4).map(|o| format!("{} ({:+} nights{})", opt_name(o, s), o.days, o.dust.map(|d| format!(", dust {d:.2}")).unwrap_or_default())).collect::<Vec<_>>().join("; "))).font(body(12.0)).color(p.muted));
        }
    }
}

// ---------------------------------------------------------------- heat maps

fn texture<'a>(a: &'a mut UiState, ctx: &egui::Context, key: Heat, make: impl FnOnce() -> Option<(Vec<f32>, usize, usize, f64, String)>) -> Option<&'a (Heat, egui::TextureHandle, f64, String)> {
    if let Some(i) = a.cal.textures.iter().position(|t| t.0 == key) {
        return a.cal.textures.get(i);
    }
    let (map, mw, mh, spread, caption) = make()?;
    if map.len() != mw * mh || mw == 0 {
        return None;
    }
    let range = match key {
        Heat::Drift(_) => 3.0,
        Heat::Pair(..) if spread < 5.0 => 3.0,
        _ => 10.0,
    };
    let tex = ctx.load_texture(format!("calheat{}", a.cal.textures.len()), heat_image(&map, mw, mh, range), egui::TextureOptions::LINEAR);
    a.cal.textures.push((key, tex, range as f64, caption));
    a.cal.textures.last()
}

fn heat_tile(ui: &mut Ui, p: &Pal, tex: &egui::TextureHandle, wd: f32, caption: RichText, sel: bool) -> egui::Response {
    let [tw, th] = tex.size();
    let ht = wd * th as f32 / tw.max(1) as f32;
    let resp = ui
        .vertical(|ui| {
            ui.set_width(wd);
            let (r, resp) = ui.allocate_exact_size(Vec2::new(wd, ht), Sense::click());
            ui.painter().image(tex.id(), r, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
            ui.painter().rect_stroke(r, CornerRadius::same(6), Stroke::new(if sel { 2.0 } else { 1.0 }, if sel { p.accent } else { p.line }), StrokeKind::Inside);
            ui.add(egui::Label::new(caption).wrap());
            resp
        })
        .inner;
    resp.on_hover_cursor(egui::CursorIcon::PointingHand)
}

fn scale_bar(ui: &mut Ui, p: &Pal) {
    ui.horizontal(|ui| {
        ui.label(RichText::new("−10%").font(mono(11.0)).color(p.muted));
        let wd = (ui.available_width() - 50.0).max(80.0);
        let (r, _) = ui.allocate_exact_size(Vec2::new(wd, 8.0), Sense::hover());
        let n = 64;
        for k in 0..n {
            let x0 = r.left() + r.width() * k as f32 / n as f32;
            let x1 = r.left() + r.width() * (k + 1) as f32 / n as f32 + 0.5;
            ui.painter().rect_filled(Rect::from_min_max(Pos2::new(x0, r.top()), Pos2::new(x1, r.bottom())), 0.0, div(-1.0 + 2.0 * (k as f32 + 0.5) / n as f32));
        }
        ui.label(RichText::new("+10%").font(mono(11.0)).color(p.muted));
    });
}

fn heat_card(ui: &mut Ui, a: &mut UiState, s: &mut Session, p: &Pal) {
    let sets: Vec<&SetReport> = s.triage.cal.sets.iter().filter(|x| x.vs_lights.as_ref().is_some_and(|v| !v.error.map.is_empty())).collect();
    let n_lights = sets.iter().filter_map(|x| x.vs_lights.as_ref().map(|v| v.n_lights)).max().unwrap_or(0);
    let filters: Vec<String> = {
        let mut f: Vec<String> = sets.iter().filter_map(|x| s.triage.grouping.groups[x.group].filter.clone()).map(|f| f.to_uppercase()).collect();
        f.sort();
        f.dedup();
        f
    };
    w::h2(ui, format!("Flats against the lights{}", if sets.is_empty() { String::new() } else { format!(" ({}, up to {n_lights} lights)", filters.join(", ")) }));
    ui.add_space(8.0);
    if sets.is_empty() {
        ui.label(RichText::new("No flat set could be checked against lights here (needs 5 good lights of the same filter within 60 nights).").color(p.muted));
        return;
    }
    let ids: Vec<(usize, Grade, f64, String, bool)> = sets
        .iter()
        .map(|x| {
            let v = x.vs_lights.as_ref().unwrap();
            (x.group, x.grade, v.error.spread, set_name(s, x.group), v.reliable)
        })
        .collect();
    let avail = ui.available_width();
    let cols = ((avail + 12.0) / 212.0).floor().max(1.0) as usize;
    let tw = (avail - 12.0 * (cols as f32 - 1.0)) / cols as f32;
    let ctx = ui.ctx().clone();
    let mut clicked = None;
    for row in ids.chunks(cols) {
        ui.horizontal_top(|ui| {
            ui.spacing_mut().item_spacing.x = 12.0;
            for (gid, g, spread, name, reliable) in row {
                let set = s.triage.cal.set(*gid).cloned();
                let Some(t) = texture(a, &ctx, Heat::VsLights(*gid), || set.and_then(|x| x.vs_lights).map(|v| (v.error.map.clone(), v.error.w, v.error.h, v.error.spread, String::new()))) else { continue };
                let tex = t.1.clone();
                let col = if *g == Grade::Reject { p.rej } else { p.fg };
                let cap = RichText::new(format!("{spread:.1}% p-p{} · {name}", if *reliable { "" } else { " (unreliable)" })).font(body(12.0)).color(col);
                if heat_tile(ui, p, &tex, tw, cap, a.cal.sel_set == Some(*gid)).clicked() {
                    clicked = Some(*gid);
                }
            }
        });
        ui.add_space(6.0);
    }
    if let Some(g) = clicked {
        a.cal.sel_set = Some(g);
        a.cal.sel_pair = None;
    }
    scale_bar(ui, p);
    ui.label(RichText::new("The error each flat would leave in the calibrated lights: the lights' illumination divided by the flat, smoothed (144 px). Blue edges are darker than the centre after calibration, red brighter.").font(body(11.5)).color(p.muted));
}

// ---------------------------------------------------------------- sets

fn set_order(s: &Session) -> Vec<&SetReport> {
    let g = &s.triage.grouping;
    let mut v: Vec<&SetReport> = s.triage.cal.sets.iter().collect();
    let rank = |k: FrameType| match k {
        FrameType::Flat => 0,
        FrameType::DarkFlat => 1,
        FrameType::Dark => 2,
        _ => 3,
    };
    v.sort_by_key(|x| {
        let gr = &g.groups[x.group];
        (rank(gr.kind), gr.filter.clone().unwrap_or_default().to_uppercase(), gr.master, gr.night)
    });
    v
}

fn sets_table(ui: &mut Ui, a: &mut UiState, s: &mut Session, p: &Pal) {
    w::h2(ui, "Sets");
    ui.add_space(8.0);
    let rows: Vec<(usize, String, Option<String>, String, String, String, String, Grade)> = set_order(s)
        .into_iter()
        .map(|x| {
            let gr = &s.triage.grouping.groups[x.group];
            let frames = if gr.master { "master".to_string() } else { gr.members.len().to_string() };
            let level = match (gr.kind, x.flat_level, x.level_adu) {
                (FrameType::Flat, Some(l), _) => format!("{:.0}%", l * 100.0),
                (_, _, Some(l)) => format!("{l:.1}"),
                _ => String::new(),
            };
            let worst = x.worst_change.map(|v| format!("{v:.1}%")).unwrap_or_default();
            let vsl = x.vs_lights.as_ref().map(|v| format!("{:.1}%", v.error.spread)).unwrap_or_default();
            (x.group, set_name(s, x.group), gr.filter.clone().map(|f| f.to_uppercase()), frames, level, worst, vsl, x.grade)
        })
        .collect();
    let mut clicked = None;
    egui::Grid::new("calsets").num_columns(6).spacing([14.0, 6.0]).min_col_width(10.0).show(ui, |ui| {
        for h in ["Set", "Frames", "Level", "Worst change", "vs lights", "Grade"] {
            ui.label(RichText::new(h).font(body(11.5)).color(p.muted));
        }
        ui.end_row();
        for (gid, name, filter, frames, level, worst, vsl, g) in rows {
            let sel = a.cal.sel_set == Some(gid);
            let col = if g == Grade::Reject { p.rej } else { p.fg };
            let r = ui
                .horizontal(|ui| {
                    if let Some(f) = &filter {
                        w::filter_badge(ui, p, f);
                    }
                    let t = RichText::new(short(&name, 34)).font(body(12.5)).color(col);
                    ui.add(egui::Label::new(if sel { t.strong() } else { t }).sense(Sense::click()))
                })
                .inner;
            for t in [frames, level, worst, vsl] {
                ui.label(RichText::new(t).font(mono(12.0)).color(col));
            }
            w::pill(ui, p, g, false);
            ui.end_row();
            if r.on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
                clicked = Some(gid);
            }
        }
    });
    if let Some(g) = clicked {
        a.cal.sel_set = if a.cal.sel_set == Some(g) { None } else { Some(g) };
        a.cal.sel_pair = None;
    }
}

/// The worst frame of a raw flat set against the set's median flat.
fn drift_of(s: &Session, gid: usize) -> Option<(Vec<f32>, usize, usize, f64, String)> {
    let set = s.triage.cal.set(gid)?;
    let (&fi, &change) = set.changes.iter().max_by(|a, b| a.1.total_cmp(b.1))?;
    let (reference, rw, rh) = s.triage.cal.references.get(&gid)?;
    let mut c = crux_core::cache::Cache::disabled();
    c.file = Some(s.cache_dir.as_ref()?.join("cache.json"));
    let f = &s.triage.scan.frames[fi];
    let m = c.get_map(&f.meta.path, f.meta.file_size, f.meta.mtime_ns)?;
    if m.w != *rw || m.h != *rh {
        return None;
    }
    let bias = s.cfg.camera.bias_level_adu as f32;
    let mut v: Vec<f32> = m.data.iter().map(|x| x - bias).collect();
    calib::normalise(&mut v)?;
    let r = calib::ratio_smoothed(&v, reference, m.w, m.h);
    Some((r, m.w, m.h, change, f.meta.file_name().to_string()))
}

fn detail(ui: &mut Ui, a: &mut UiState, s: &mut Session, p: &Pal) {
    let ctx = ui.ctx().clone();
    if let Some((ga, gb)) = a.cal.sel_pair {
        let (na, nb) = (set_name(s, ga), set_name(s, gb));
        let pair = s.triage.cal.pairs.iter().find(|x| (x.a, x.b) == (ga, gb) || (x.a, x.b) == (gb, ga)).cloned();
        w::h2(ui, format!("{na} ÷ {nb}"));
        if let Some(pr) = &pair {
            ui.label(
                RichText::new(format!(
                    "{:.1}% peak to peak{}{}",
                    pr.spread,
                    pr.dust.map(|d| format!(" · dust {d:.2}")).unwrap_or_default(),
                    if pr.combinable { " · can be combined" } else { "" }
                ))
                .color(p.muted),
            );
        }
        ui.add_space(6.0);
        let refs = &s.triage.cal.references;
        let made = match (refs.get(&ga), refs.get(&gb)) {
            (Some((x, w1, h1)), Some((y, w2, h2))) if (w1, h1) == (w2, h2) => Some((calib::ratio_smoothed(x, y, *w1, *h1), *w1, *h1, pair.as_ref().map(|x| x.spread).unwrap_or(0.0), String::new())),
            _ => None,
        };
        if let Some(t) = texture(a, &ctx, Heat::Pair(ga, gb), || made) {
            let tex = t.1.clone();
            let wd = ui.available_width().min(360.0);
            let range = t.2;
            heat_tile(ui, p, &tex, wd, RichText::new(format!("Smoothed ratio (64 px), ±{range:.0}%")).font(body(11.5)).color(p.muted), false);
        }
        return;
    }
    let Some(gid) = a.cal.sel_set else { return };
    let Some(set) = s.triage.cal.set(gid).cloned() else { return };
    let gr = s.triage.grouping.groups[gid].clone();
    ui.horizontal(|ui| {
        w::h2(ui, set_name(s, gid));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| w::pill(ui, p, set.grade, false));
    });
    ui.label(RichText::new(&gr.label).font(body(11.5)).color(p.faint));
    for f in &set.flags {
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new(f.check.name()).font(semibold(12.0)).color(p.grade(f.grade)));
            ui.label(RichText::new(&f.reason).font(body(12.5)));
        });
    }
    for n in &set.notes {
        ui.label(RichText::new(n).font(body(12.0)).color(p.muted));
    }
    if gr.kind == FrameType::Flat && !gr.master && !set.changes.is_empty() {
        ui.add_space(8.0);
        w::eyebrow(ui, p, "Change of each frame from the set median");
        let ch: Vec<f64> = set.changes.values().copied().collect();
        let wd = ui.available_width();
        let ht = 90.0;
        let (r, _) = ui.allocate_exact_size(Vec2::new(wd, ht), Sense::hover());
        let pa = ui.painter_at(r);
        let mx = ch.iter().copied().fold(3.5, f64::max) as f32;
        let yv = |v: f32| r.bottom() - 14.0 - (ht - 24.0) * v / mx;
        let bw = wd / ch.len() as f32;
        for (k, c) in ch.iter().enumerate() {
            let c = *c as f32;
            let col = if c > 3.0 { p.rej } else if c > 1.0 { p.warn } else { p.accent };
            pa.rect_filled(Rect::from_min_max(Pos2::new(r.left() + k as f32 * bw + 1.0, yv(c)), Pos2::new(r.left() + (k + 1) as f32 * bw - 1.0, yv(0.0))), CornerRadius::same(1), col);
        }
        for (v, c) in [(1.0, p.warn), (3.0, p.rej)] {
            pa.extend(egui::Shape::dashed_line(&[Pos2::new(r.left(), yv(v)), Pos2::new(r.right(), yv(v))], Stroke::new(1.0, c), 3.0, 3.0));
            let t = pa.layout_no_wrap(format!("{v:.0}%"), body(10.0), c);
            let at = Pos2::new(r.right() - t.size().x - 4.0, yv(v) - t.size().y - 2.0);
            pa.rect_filled(Rect::from_min_size(at - Vec2::new(2.0, 0.0), t.size() + Vec2::new(4.0, 0.0)), 2.0, p.panel);
            pa.galley(at, t, c);
        }
        pa.text(Pos2::new(r.left(), r.bottom()), egui::Align2::LEFT_BOTTOM, "frame 1", body(10.0), p.faint);
        pa.text(Pos2::new(r.right(), r.bottom()), egui::Align2::RIGHT_BOTTOM, format!("frame {}", ch.len()), body(10.0), p.faint);
        // The worst frame's drift as a heat map.
        if !a.cal.drift.contains_key(&gid) {
            let d = drift_of(s, gid);
            a.cal.drift.insert(gid, d);
        }
        let made = a.cal.drift.get(&gid).cloned().flatten();
        if let Some((_, _, _, change, name)) = made.clone() {
            if let Some(t) = texture(a, &ctx, Heat::Drift(gid), || made) {
                let tex = t.1.clone();
                ui.add_space(6.0);
                let wd = ui.available_width().min(320.0);
                heat_tile(ui, p, &tex, wd, RichText::new(format!("Worst frame against the set, ±3%: {change:.1}% peak to peak ({name})")).font(body(11.5)).color(p.muted), false);
            }
        }
    } else if gr.master {
        ui.label(RichText::new("A single integrated master: no per-frame stability to show.").font(body(12.5)).color(p.muted));
    }
    if let Some(v) = &set.vs_lights {
        ui.add_space(6.0);
        let e = &v.error.edges;
        ui.label(
            RichText::new(format!(
                "Against {} lights: {:.1}% p-p · left {:+.1}% right {:+.1}% top {:+.1}% bottom {:+.1}%{}",
                v.n_lights,
                v.error.spread,
                e.left,
                e.right,
                e.top,
                e.bottom,
                v.dust.map(|d| format!(" · dust {d:.2}")).unwrap_or_default()
            ))
            .font(body(12.0))
            .color(p.muted),
        );
    }
}

fn pairs(ui: &mut Ui, a: &mut UiState, s: &mut Session, p: &Pal) {
    w::h2(ui, "Flat sets against each other");
    ui.add_space(8.0);
    let g = &s.triage.grouping;
    let mut by_filter: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for pr in &s.triage.cal.pairs {
        for id in [pr.a, pr.b] {
            let f = g.groups[id].filter.clone().unwrap_or_default().to_uppercase();
            let v = by_filter.entry(f).or_default();
            if !v.contains(&id) {
                v.push(id);
            }
        }
    }
    let lab = |id: usize| {
        let gr = &g.groups[id];
        if gr.master {
            "master".to_string()
        } else if gr.retired {
            "retired".to_string()
        } else {
            gr.night.map(d_short).unwrap_or_default()
        }
    };
    let mut clicked = None;
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = Vec2::new(18.0, 12.0);
        for (f, ids) in &by_filter {
            let mut ids = ids.clone();
            ids.sort_by_key(|id| (g.groups[*id].master, g.groups[*id].retired, g.groups[*id].night));
            ui.vertical(|ui| {
                egui::Grid::new(("pairs", f)).spacing([4.0, 4.0]).show(ui, |ui| {
                    w::filter_badge(ui, p, f);
                    for b in &ids {
                        ui.label(RichText::new(lab(*b)).font(body(10.5)).color(p.muted));
                    }
                    ui.end_row();
                    for ia in &ids {
                        ui.label(RichText::new(lab(*ia)).font(body(10.5)).color(p.muted));
                        for ib in &ids {
                            if ia == ib {
                                ui.label(RichText::new("·").font(mono(12.0)).color(p.faint));
                                continue;
                            }
                            let pr = s.triage.cal.pairs.iter().find(|x| (x.a, x.b) == (*ia, *ib) || (x.a, x.b) == (*ib, *ia));
                            match pr {
                                Some(pr) => {
                                    let c = if pr.spread < 2.0 { p.good } else if pr.spread < 4.0 { p.warn } else { p.rej };
                                    let sel = a.cal.sel_pair == Some((*ia, *ib));
                                    let t = RichText::new(format!("{:.1}", pr.spread)).font(mono(12.0)).color(c);
                                    let r = ui.add(egui::Label::new(if sel { t.underline() } else { t }).sense(Sense::click()));
                                    if r.on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
                                        clicked = Some((*ia, *ib));
                                    }
                                }
                                None => {
                                    ui.label("");
                                }
                            }
                        }
                        ui.end_row();
                    }
                });
            });
        }
    });
    if let Some(x) = clicked {
        a.cal.sel_pair = Some(x);
        a.cal.sel_set = None;
    }
    ui.add_space(6.0);
    ui.label(RichText::new("Smoothed ratio, % peak to peak. Under 2% (green) the sets can be combined. Click a number for its map.").font(body(11.5)).color(p.muted));
}

// ---------------------------------------------------------------- flat library

fn library(ui: &mut Ui, a: &mut UiState, s: &mut Session, p: &Pal) {
    w::h2(ui, "Flat library");
    ui.add_space(4.0);
    let lib = s.triage.library.as_ref();
    ui.label(
        RichText::new(match lib {
            Some(l) => format!("{} sets in {}. Drag a set onto a night in Matching to use it there.", l.sets.len(), l.root.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| l.root.display().to_string())),
            None => "No flat library yet: Refresh calibration library finds every flat set in your folders. This folder's own sets are below.".into(),
        })
        .font(body(12.0))
        .color(p.muted),
    );
    ui.add_space(4.0);
    let cfg = s.cfg.clone();
    crate::auto::refresh_ui(ui, a, &cfg, s.triage.library.as_ref());
    ui.add_space(8.0);
    // Which sets the nights use now, and the selected night's options (with dust).
    let used: Vec<String> = s.triage.cal.choices.iter().filter_map(|c| c.chosen.as_ref()).map(|o| o.library_id.clone().unwrap_or_else(|| format!("g{}", o.group.unwrap_or(usize::MAX)))).collect();
    let sel = a.cal.sel_choice.and_then(|i| s.triage.cal.choices.get(i)).cloned();
    let dust_for = |id: &str| -> Option<f64> {
        let c = sel.as_ref()?;
        c.chosen.iter().chain(&c.others).find(|o| o.library_id.clone().unwrap_or_else(|| format!("g{}", o.group.unwrap_or(usize::MAX))) == id).and_then(|o| o.dust)
    };
    struct Chip {
        id: String,
        filter: String,
        night: NaiveDate,
        label: String,
        tip: String,
        grade: Grade,
        usable: bool,
        drag: DragFlat,
    }
    let mut chips: Vec<Chip> = Vec::new();
    let g = &s.triage.grouping;
    let mut local_ids = std::collections::HashSet::new();
    for set in &s.triage.cal.sets {
        let gr = &g.groups[set.group];
        if gr.kind != FrameType::Flat || gr.master {
            continue;
        }
        let Some(n) = gr.night else { continue };
        let f = gr.filter.clone().unwrap_or_default().to_uppercase();
        local_ids.insert(crux_core::library::set_id(&f, n, gr.binning.unwrap_or(1), gr.width, gr.height, None, gr.retired));
        chips.push(Chip {
            id: format!("g{}", gr.id),
            filter: f.clone(),
            night: n,
            label: format!("{}{}", d_short(n), if gr.retired { " ✕" } else { "" }),
            tip: format!("{} (this folder) · {} frames · {}{}", set_name(s, gr.id), gr.members.len(), set.grade.name(), if set.flags.is_empty() { String::new() } else { format!("\n{}", set.summary()) }),
            grade: set.grade,
            usable: !gr.retired && set.grade != Grade::Reject,
            drag: DragFlat { filter: f, library_id: None, group: Some(gr.label.clone()), night: n, label: set_name(s, gr.id) },
        });
    }
    if let Some(l) = lib {
        for x in l.sets.iter().filter(|x| !local_ids.contains(&x.id)) {
            let f = x.filter.to_uppercase();
            let home = x.home.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            chips.push(Chip {
                id: x.id.clone(),
                filter: f.clone(),
                night: x.night,
                label: format!("{}{}", d_short(x.night), if x.retired { " ✕" } else if x.master { " M" } else { "" }),
                tip: format!(
                    "Library {} · {} frames · {} · {home}{}",
                    x.label(),
                    x.frames,
                    x.grade.name(),
                    if x.reasons.is_empty() { String::new() } else { format!("\n{}", x.reasons.join("; ")) }
                ),
                grade: x.grade,
                usable: x.usable(),
                drag: DragFlat { filter: f, library_id: Some(x.id.clone()), group: None, night: x.night, label: format!("library {}", x.label()) },
            });
        }
    }
    chips.sort_by(|a, b| a.filter.cmp(&b.filter).then(a.night.cmp(&b.night)));
    let mut by_f: BTreeMap<String, Vec<&Chip>> = BTreeMap::new();
    for c in &chips {
        by_f.entry(c.filter.clone()).or_default().push(c);
    }
    for (f, list) in by_f {
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = Vec2::new(6.0, 6.0);
            w::filter_badge(ui, p, &f);
            ui.add_space(4.0);
            for c in list {
                let dust = dust_for(&c.id);
                let text = match dust {
                    Some(d) => format!("{} · {d:.2}", c.label),
                    None => c.label.clone(),
                };
                let in_use = used.contains(&c.id);
                let for_sel = sel.as_ref().and_then(|x| x.chosen.as_ref()).is_some_and(|o| o.library_id.clone().unwrap_or_else(|| format!("g{}", o.group.unwrap_or(usize::MAX))) == c.id);
                let col = if !c.usable { p.rej } else { p.grade(c.grade) };
                let gal = ui.painter().layout_no_wrap(text, body(11.5), if c.usable { p.fg } else { p.muted });
                let (r, resp) = ui.allocate_exact_size(Vec2::new(gal.size().x + 16.0, 22.0), if c.usable { Sense::click_and_drag() } else { Sense::hover() });
                let fill = if for_sel { alpha(p.accent, 0.3) } else if in_use { mix(col, 0.18, p.panel) } else { p.panel2 };
                ui.painter().rect(r, CornerRadius::same(11), fill, Stroke::new(if in_use { 1.5 } else { 1.0 }, if for_sel { p.accent } else { alpha(col, 0.7) }), StrokeKind::Inside);
                ui.painter().galley(Pos2::new(r.left() + 8.0, r.center().y - gal.size().y / 2.0), gal, p.fg);
                if c.usable {
                    resp.dnd_set_drag_payload(c.drag.clone());
                }
                let tip = format!("{}{}{}", c.tip, if in_use { "\nUsed now." } else { "" }, if c.usable { "\nDrag onto a night in Matching to use it there." } else { "\nNot usable." });
                if resp.dragged() {
                    if let Some(pos) = ui.ctx().pointer_interact_pos() {
                        egui::Area::new(ui.scope_id().with("caldrag")).fixed_pos(pos + Vec2::new(10.0, 6.0)).order(egui::Order::Tooltip).show(ui.ctx(), |ui| {
                            egui::Frame::popup(ui.style()).show(ui, |ui| {
                                ui.label(RichText::new(&c.drag.label).font(body(12.0)));
                            });
                        });
                    }
                } else {
                    resp.on_hover_text(tip);
                }
            }
        });
        ui.add_space(4.0);
    }
    ui.add_space(4.0);
    ui.label(
        RichText::new(if sel.is_some() { "Numbers: dust match with the selected night's lights (0.3 or more matches). Filled: in use; ✕ retired; M master." } else { "Select a night in Matching to see each set's dust match with its lights. Filled: in use; ✕ retired; M master." })
            .font(body(11.5))
            .color(p.muted),
    );
}

// ---------------------------------------------------------------- darks and bias

fn darks_card(ui: &mut Ui, a: &mut UiState, s: &mut Session, p: &Pal) {
    w::h2(ui, "Darks and bias across the library");
    ui.add_space(4.0);
    let Some(fp) = a.cal.fp.clone() else {
        let t = match a.cal.fp_progress {
            Some((d, n)) => format!("Reading hot pixels: {d} of {n} files…"),
            None => "Waiting for the library scan…".into(),
        };
        ui.label(RichText::new(t).color(p.muted));
        return;
    };
    ui.label(
        RichText::new(format!(
            "{} darks, bias and dark flats found ({}). Each dark is scored by how well its hot pixels match the lights' hot pixels, and by its brightness, which rises with temperature. Masters have no GAIN or CCD-TEMP, so this is the only way to match them.",
            fp.candidates.len(),
            if fp.seconds < 0.05 { "from cache".to_string() } else { format!("{:.1} s", fp.seconds) }
        ))
        .font(body(12.0))
        .color(p.muted),
    );
    ui.add_space(6.0);
    if fp.families.is_empty() {
        return;
    }
    if fp.families.len() > 1 {
        let opts: Vec<(usize, String)> = fp.families.iter().enumerate().map(|(i, f)| (i, f.label.clone())).collect();
        let refs: Vec<(usize, &str)> = opts.iter().map(|(i, t)| (*i, t.as_str())).collect();
        if let Some(k) = w::seg(ui, p, &refs, a.cal.fam.min(fp.families.len() - 1)) {
            a.cal.fam = k;
        }
        ui.add_space(6.0);
    }
    let fi = a.cal.fam.min(fp.families.len() - 1);
    let fam = &fp.families[fi];
    let nights = match (fam.first, fam.last) {
        (Some(x), Some(y)) if x != y => format!("{} – {}", d_short(x), d_short(y)),
        (Some(x), _) => d_short(x),
        _ => String::new(),
    };
    ui.label(
        RichText::new(if fam.kind == FrameType::Light {
            format!("{} · {} frames · {nights} · {} hot pixels in {} sampled lights (noise {:.1} ADU)", fam.label, fam.n, fam.lights_hot, fam.sample.len(), fam.light_noise_adu.unwrap_or(0.0))
        } else {
            format!("{} · {} frames · {nights}", fam.label, fam.n)
        })
        .font(semibold(13.0)),
    );
    ui.add_space(4.0);
    let mut action: Option<(String, Option<std::path::PathBuf>)> = None;
    for (title, list, cur, kind) in [
        (if fam.kind == FrameType::Light { "Darks" } else { "Dark flats" }, &fam.darks, fam.current_dark, "dark"),
        ("Bias", &fam.bias, fam.current_bias, "bias"),
    ] {
        ui.add_space(6.0);
        w::eyebrow(ui, p, title);
        if list.is_empty() {
            ui.label(RichText::new("None in the library.").font(body(12.5)).color(p.muted));
            continue;
        }
        let okey = crux_core::decisions::offset_key(&s.root, kind, fam);
        let yours = s.decisions.offsets.get(&okey).cloned();
        for (k, r) in list.iter().enumerate() {
            offset_row(ui, a, p, &fp, fam, fi, k, r, cur, yours.as_deref(), &okey, &mut action);
        }
    }
    if let Some((key, file)) = action {
        let msg = match &file {
            Some(f) => format!("Using {} for these lights. Saved in your decisions.", f.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default()),
            None => "Back to crux's choice.".into(),
        };
        s.pick_offset(key, file.as_deref());
        a.toast(ui.ctx(), msg, false);
    }
}

#[allow(clippy::too_many_arguments)]
fn offset_row(
    ui: &mut Ui,
    a: &mut UiState,
    p: &Pal,
    fp: &Fingerprints,
    _fam: &Family,
    fi: usize,
    k: usize,
    r: &Ranked,
    cur: Option<usize>,
    yours: Option<&str>,
    okey: &str,
    action: &mut Option<(String, Option<std::path::PathBuf>)>,
) {
    let c = &fp.candidates[r.cand];
    let is_yours = yours.is_some_and(|y| c.files.iter().chain(&c.copies).any(|f| crux_core::paths::key(f) == y));
    let open = a.cal.open_row == Some((fi, r.cand));
    let col = if !r.usable { p.muted } else { p.fg };
    let resp = ui
        .horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 8.0;
            let mark = if is_yours { "●" } else if k == 0 && r.usable && yours.is_none() { "★" } else { " " };
            ui.label(RichText::new(mark).font(body(12.0)).color(if is_yours { p.accent } else { p.good }));
            let t = ui.add(egui::Label::new(RichText::new(&c.label).font(semibold(12.5)).color(col)).sense(Sense::click()));
            ui.label(RichText::new(c.place()).font(body(11.5)).color(p.muted));
            if let Some(h) = &r.hot {
                let cc = |v: f64| if v >= 0.95 { p.good } else if v >= 0.8 { p.warn } else { p.rej };
                ui.label(RichText::new(format!("has {:.0}%", h.coverage * 100.0)).font(mono(11.5)).color(cc(h.coverage)))
                    .on_hover_text(format!("Of the lights' {} hot pixels, the share this dark also has. Missing ones stay in the calibrated lights.", h.lights_hot));
                ui.label(RichText::new(format!("shows {:.0}%", h.shown * 100.0)).font(mono(11.5)).color(cc(h.shown)))
                    .on_hover_text(format!("Of this dark's {} hot pixels bright enough to show in the lights, the share that do. The rest would leave dark holes.", h.dark_hot));
                if let Some(x) = h.ratio {
                    ui.label(RichText::new(format!("{x:.2}×")).font(mono(11.5)).color(p.muted)).on_hover_text("Hot-pixel brightness, lights ÷ dark (exposure-scaled). About 1 at the same temperature.");
                }
            } else if let Some(l) = c.level_adu {
                ui.label(RichText::new(format!("{l:.1} ADU")).font(mono(11.5)).color(p.muted));
            }
            if cur == Some(r.cand) {
                w::tag(ui, p, "matched now");
            }
            if is_yours {
                w::tag(ui, p, "yours");
            }
            t
        })
        .inner;
    if resp.on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
        a.cal.open_row = if open { None } else { Some((fi, r.cand)) };
    }
    if open {
        ui.indent(("off", fi, r.cand), |ui| {
            ui.add(egui::Label::new(RichText::new(&r.summary).font(body(12.5))).wrap());
            for x in &r.reasons {
                ui.label(RichText::new(format!("· {x}")).font(body(12.0)).color(p.muted));
            }
            let first = c.files.first().cloned();
            ui.horizontal(|ui| {
                if is_yours {
                    if w::button(ui, p, "Back to crux's choice", w::Kind::Ghost, None).clicked() {
                        *action = Some((okey.to_string(), None));
                    }
                } else if r.usable && !(yours.is_none() && cur == Some(r.cand)) && w::button(ui, p, "Use this one", w::Kind::Normal, None).clicked() {
                    *action = Some((okey.to_string(), first));
                }
                ui.label(RichText::new(c.folder.display().to_string()).font(body(11.0)).color(p.faint));
            });
        });
    }
}
