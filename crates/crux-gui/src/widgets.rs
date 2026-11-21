//! Small building blocks styled like the mockup: cards, pills, chips,
//! segmented controls, buttons and key caps.

use crate::theme::{self, alpha, body, mix, mono, semibold, Pal};
use crux_core::grade::Grade;
use eframe::egui::{self, Color32, CornerRadius, Frame, Margin, Response, RichText, Sense, Stroke, StrokeKind, Ui, Vec2};

pub fn card<R>(ui: &mut Ui, p: &Pal, add: impl FnOnce(&mut Ui) -> R) -> R {
    Frame::new()
        .fill(p.panel)
        .stroke(Stroke::new(1.0, p.line))
        .corner_radius(CornerRadius::same(12))
        .inner_margin(Margin::same(16))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            add(ui)
        })
        .inner
}

pub fn h1(ui: &mut Ui, t: impl Into<String>) {
    ui.label(RichText::new(t).font(semibold(22.0)));
}
pub fn h2(ui: &mut Ui, t: impl Into<String>) -> Response {
    ui.label(RichText::new(t).font(semibold(16.0)))
}
pub fn muted(ui: &mut Ui, p: &Pal, t: impl Into<String>) -> Response {
    ui.label(RichText::new(t).color(p.muted))
}
pub fn eyebrow(ui: &mut Ui, p: &Pal, t: &str) -> Response {
    ui.label(RichText::new(t.to_uppercase()).font(body(11.0)).color(p.muted).extra_letter_spacing(0.9))
}
#[derive(Clone, Copy, PartialEq)]
pub enum Kind {
    Normal,
    Primary,
    Danger,
    Ghost,
}

/// A button like the mockup's `.btn`, optionally with a key cap.
pub fn button(ui: &mut Ui, p: &Pal, text: &str, kind: Kind, key: Option<&str>) -> Response {
    button_enabled(ui, p, text, kind, key, true)
}

pub fn button_enabled(ui: &mut Ui, p: &Pal, text: &str, kind: Kind, key: Option<&str>, enabled: bool) -> Response {
    let (fill, fg, stroke) = match kind {
        Kind::Normal => (p.panel2, p.fg, p.line),
        Kind::Primary => (p.accent, p.accent_ink, p.accent),
        Kind::Danger => (p.panel2, p.rej, p.line),
        Kind::Ghost => (Color32::TRANSPARENT, p.fg, p.line),
    };
    let font = body(13.0);
    let galley = ui.painter().layout_no_wrap(text.to_string(), font.clone(), fg);
    let kg = key.map(|k| ui.painter().layout_no_wrap(k.to_string(), mono(10.5), if kind == Kind::Primary { p.accent_ink } else { p.muted }));
    let kw = kg.as_ref().map(|g| g.size().x + 8.0 + 6.0).unwrap_or(0.0);
    let size = Vec2::new(galley.size().x + 24.0 + kw, 30.0);
    let sense = if enabled { Sense::click() } else { Sense::hover() };
    let (rect, resp) = ui.allocate_exact_size(size, sense);
    if ui.is_rect_visible(rect) {
        let hovered = enabled && resp.hovered();
        let st = if hovered && kind != Kind::Primary { p.faint } else { stroke };
        let fill = if hovered && kind == Kind::Primary { mix(p.fg, 0.15, p.accent) } else { fill };
        let pa = ui.painter();
        pa.rect(rect, CornerRadius::same(8), fill, Stroke::new(1.0, st), StrokeKind::Inside);
        let col = if enabled { fg } else { p.faint };
        let tx = rect.left() + 12.0;
        pa.galley_with_override_text_color(egui::pos2(tx, rect.center().y - galley.size().y / 2.0), galley.clone(), col);
        if let Some(g) = kg {
            let kr = egui::Rect::from_min_size(
                egui::pos2(rect.right() - 12.0 - g.size().x - 8.0, rect.center().y - 8.5),
                Vec2::new(g.size().x + 8.0, 17.0),
            );
            let kfill = if kind == Kind::Primary { mix(p.accent_ink, 0.15, p.accent) } else { p.panel };
            pa.rect(kr, CornerRadius::same(4), kfill, Stroke::new(1.0, p.line), StrokeKind::Inside);
            pa.galley(egui::pos2(kr.left() + 4.0, kr.center().y - g.size().y / 2.0), g, p.muted);
        }
    }
    if enabled {
        resp.on_hover_cursor(egui::CursorIcon::PointingHand)
    } else {
        resp
    }
}

/// Text that works as a link: underlined, pointing hand, brighter on hover.
pub fn link(ui: &mut Ui, p: &Pal, text: &str, colour: Color32) -> Response {
    let resp = ui.add(egui::Label::new(RichText::new(text).color(colour).underline()).sense(Sense::click()));
    if resp.hovered() {
        ui.painter().line_segment([resp.rect.left_bottom(), resp.rect.right_bottom()], Stroke::new(1.0, mix(p.fg, 0.4, colour)));
    }
    resp.on_hover_cursor(egui::CursorIcon::PointingHand)
}

/// A small square button with a folder outline.
pub fn folder_button(ui: &mut Ui, p: &Pal, tip: &str) -> Response {
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(26.0, 22.0), Sense::click());
    if ui.is_rect_visible(rect) {
        let hovered = resp.hovered();
        let pa = ui.painter();
        pa.rect(rect, CornerRadius::same(6), if hovered { p.panel2 } else { Color32::TRANSPARENT }, Stroke::new(1.0, if hovered { p.faint } else { p.line }), StrokeKind::Inside);
        let c = if hovered { p.fg } else { p.muted };
        let s = 14.0 / 24.0;
        let o = rect.center() - Vec2::new(7.0, 7.0);
        let pt = |x: f32, y: f32| o + Vec2::new(x * s, y * s);
        let v = vec![pt(2.0, 5.0), pt(9.0, 5.0), pt(11.0, 8.0), pt(22.0, 8.0), pt(22.0, 20.0), pt(2.0, 20.0)];
        pa.add(egui::Shape::closed_line(v, Stroke::new(1.5, c)));
        pa.line_segment([pt(2.0, 11.0), pt(22.0, 11.0)], Stroke::new(1.5, c));
    }
    resp.on_hover_text(tip).on_hover_cursor(egui::CursorIcon::PointingHand)
}

/// Toggle button for viewer tools: accent outline when on.
pub fn tool(ui: &mut Ui, p: &Pal, text: &str, key: &str, on: bool, enabled: bool) -> Response {
    let r = button_enabled(ui, p, text, Kind::Normal, Some(key), enabled);
    if on {
        ui.painter().rect_stroke(r.rect, CornerRadius::same(8), Stroke::new(1.0, p.accent), StrokeKind::Inside);
    }
    r
}

pub fn kbd(ui: &mut Ui, p: &Pal, k: &str) {
    let g = ui.painter().layout_no_wrap(k.to_string(), mono(10.5), p.muted);
    let (rect, _) = ui.allocate_exact_size(Vec2::new(g.size().x + 9.0, 18.0), Sense::hover());
    let pa = ui.painter();
    pa.rect(rect, CornerRadius::same(4), p.panel, Stroke::new(1.0, p.line), StrokeKind::Inside);
    pa.line_segment([rect.left_bottom() + egui::vec2(2.0, -0.5), rect.right_bottom() + egui::vec2(-2.0, -0.5)], Stroke::new(1.5, p.line));
    pa.galley(egui::pos2(rect.left() + 4.5, rect.center().y - g.size().y / 2.0 - 0.5), g, p.muted);
}

/// Rounded chip; filled with the foreground colour when `on`.
pub fn chip(ui: &mut Ui, p: &Pal, text: &str, count: Option<usize>, on: bool) -> Response {
    let (fg, fill, stroke) = if on { (p.bg, p.fg, p.fg) } else { (p.fg, p.panel, p.line) };
    let g = ui.painter().layout_no_wrap(text.to_string(), body(12.5), fg);
    let cg = count.map(|c| ui.painter().layout_no_wrap(c.to_string(), body(11.0), if on { p.bg } else { p.muted }));
    let w = g.size().x + 20.0 + cg.as_ref().map(|c| c.size().x + 6.0).unwrap_or(0.0);
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(w, 26.0), Sense::click());
    let hovered = resp.hovered();
    let pa = ui.painter();
    pa.rect(rect, CornerRadius::same(13), fill, Stroke::new(1.0, if hovered && !on { p.faint } else { stroke }), StrokeKind::Inside);
    let x = rect.left() + 10.0;
    let gw = g.size().x;
    pa.galley(egui::pos2(x, rect.center().y - g.size().y / 2.0), g, fg);
    if let Some(c) = cg {
        pa.galley(egui::pos2(x + gw + 6.0, rect.center().y - c.size().y / 2.0 + 0.5), c, p.muted);
    }
    resp.on_hover_cursor(egui::CursorIcon::PointingHand)
}

/// Static chip for evidence lists.
pub fn tag(ui: &mut Ui, p: &Pal, text: &str) {
    let g = ui.painter().layout_no_wrap(text.to_string(), body(12.5), p.fg);
    let (rect, _) = ui.allocate_exact_size(Vec2::new(g.size().x + 20.0, 24.0), Sense::hover());
    ui.painter().rect(rect, CornerRadius::same(12), p.panel, Stroke::new(1.0, p.line), StrokeKind::Inside);
    ui.painter().galley(egui::pos2(rect.left() + 10.0, rect.center().y - g.size().y / 2.0), g, p.fg);
}

/// Segmented control; returns the newly picked key.
pub fn seg<K: PartialEq + Copy>(ui: &mut Ui, p: &Pal, opts: &[(K, &str)], cur: K) -> Option<K> {
    let mut picked = None;
    let gals: Vec<_> = opts.iter().map(|(_, t)| ui.painter().layout_no_wrap(t.to_string(), body(13.0), p.fg)).collect();
    let w: f32 = gals.iter().map(|g| g.size().x + 24.0).sum();
    let (rect, _) = ui.allocate_exact_size(Vec2::new(w, 32.0), Sense::hover());
    ui.painter().rect(rect, CornerRadius::same(8), p.panel, Stroke::new(1.0, p.line), StrokeKind::Inside);
    let mut x = rect.left();
    for ((k, _), g) in opts.iter().zip(gals) {
        let bw = g.size().x + 24.0;
        let r = egui::Rect::from_min_size(egui::pos2(x, rect.top()), Vec2::new(bw, rect.height())).shrink(1.0);
        let resp = ui.interact(r, ui.scope_id().with(("seg", x as i32, rect.top() as i32)), Sense::click());
        let on = *k == cur;
        if on {
            ui.painter().rect_filled(r, CornerRadius::same(7), p.panel2);
            ui.painter().line_segment([r.left_bottom() + egui::vec2(0.0, -1.0), r.right_bottom() + egui::vec2(0.0, -1.0)], Stroke::new(2.0, p.accent));
        } else if resp.hovered() {
            ui.painter().rect_filled(r, CornerRadius::same(7), theme::alpha(p.panel2, 0.6));
        }
        let col = if on { p.fg } else { p.muted };
        ui.painter().galley_with_override_text_color(egui::pos2(r.left() + 11.0, r.center().y - g.size().y / 2.0), g, col);
        if resp.on_hover_cursor(egui::CursorIcon::PointingHand).clicked() && !on {
            picked = Some(*k);
        }
        x += bw;
    }
    picked
}

/// Grade pill: Good outlined green, Warn amber, Reject red; dashed ring when pinned.
pub fn pill(ui: &mut Ui, p: &Pal, g: Grade, pinned: bool) {
    let text = match g {
        Grade::Good => "Good",
        Grade::Warn => "Warn",
        Grade::Reject => "Reject",
    };
    let text = if pinned { format!("{text} · yours") } else { text.to_string() };
    let c = p.grade(g);
    let gal = ui.painter().layout_no_wrap(text, semibold(12.0), c);
    let (rect, _) = ui.allocate_exact_size(Vec2::new(gal.size().x + 18.0, 22.0), Sense::hover());
    let pa = ui.painter();
    let fill = if g == Grade::Good { Color32::TRANSPARENT } else { alpha(c, 0.13) };
    let stroke = if g == Grade::Good { Stroke::new(1.0, alpha(c, 0.4)) } else { Stroke::NONE };
    pa.rect(rect, CornerRadius::same(11), fill, stroke, StrokeKind::Inside);
    if pinned {
        dashed_rect(pa, rect.expand(2.0), c);
    }
    pa.galley(egui::pos2(rect.left() + 9.0, rect.center().y - gal.size().y / 2.0), gal, c);
}

pub fn dashed_rect(pa: &egui::Painter, r: egui::Rect, c: Color32) {
    let st = Stroke::new(1.2, c);
    for (a, b) in [(r.left_top(), r.right_top()), (r.right_top(), r.right_bottom()), (r.right_bottom(), r.left_bottom()), (r.left_bottom(), r.left_top())] {
        pa.extend(egui::Shape::dashed_line(&[a, b], st, 3.0, 2.5));
    }
}

/// Filter badge (white text on the filter colour).
pub fn filter_badge(ui: &mut Ui, p: &Pal, f: &str) {
    let g = ui.painter().layout_no_wrap(f.to_string(), mono(11.0), p.bg);
    let (rect, _) = ui.allocate_exact_size(Vec2::new(g.size().x + 10.0, 17.0), Sense::hover());
    ui.painter().rect_filled(rect, CornerRadius::same(4), p.filter(f));
    ui.painter().galley(egui::pos2(rect.left() + 5.0, rect.center().y - g.size().y / 2.0), g, p.bg);
}

/// Wrapping row: lays children left to right, wrapping to new lines.
pub fn wrap_row<R>(ui: &mut Ui, add: impl FnOnce(&mut Ui) -> R) -> R {
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = egui::vec2(8.0, 8.0);
        add(ui)
    })
    .inner
}

/// "1 h 05 m" from seconds.
pub fn fmt_h(secs: f64) -> String {
    let m = (secs / 60.0).round() as i64;
    format!("{} h {:02} m", m / 60, m % 60)
}

pub fn d_short(d: chrono::NaiveDate) -> String {
    use chrono::Datelike;
    const MON: [&str; 12] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    format!("{} {}", d.day(), MON[d.month0() as usize])
}
