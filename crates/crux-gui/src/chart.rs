//! Charts painted directly: the night timeline (the signature view), the
//! Home sparkline and the inspector's "why" histogram.

use crate::app::{Brush, Metric, UiState};
use crate::session::Session;
use crate::theme::{alpha, body, mono, Pal};
use crate::widgets::{self as w, d_short, Kind};
use chrono::{Duration, NaiveDate, NaiveDateTime, NaiveTime};
use crux_core::astro::SkyContext;
use crux_core::grade::{Check, Grade};
use eframe::egui::{self, Align2, CornerRadius, Pos2, Rect, RichText, Sense, Shape, Stroke, Vec2};

/// Hours after 19:00 local on the night's evening.
fn hours(night: NaiveDate, t: NaiveDateTime) -> f32 {
    let t0 = night.and_time(NaiveTime::from_hms_opt(19, 0, 0).unwrap());
    (t - t0).num_seconds() as f32 / 3600.0
}

/// [hours from 19:00, target alt, Moon alt, Sun alt] every 5 minutes, 19:00–07:00.
fn curves(s: &Session, night: NaiveDate) -> Vec<[f64; 4]> {
    let frames = &s.triage.scan.frames;
    let tonight: Vec<&crux_core::Frame> = s.lights.iter().filter(|l| l.night == night).map(|l| &frames[l.fi]).collect();
    // Local − UTC from the frames themselves.
    let offs: Vec<i64> = tonight.iter().filter_map(|f| Some((f.class.local_time? - f.meta.date_obs?).num_minutes())).collect();
    let off = offs.get(offs.len() / 2).copied().unwrap_or(600);
    let med = |v: Vec<f64>| -> Option<f64> {
        let mut v = v;
        v.sort_by(f64::total_cmp);
        v.get(v.len() / 2).copied()
    };
    let target = med(tonight.iter().filter_map(|f| f.meta.ra).collect()).zip(med(tonight.iter().filter_map(|f| f.meta.dec).collect()));
    let lat = med(tonight.iter().filter_map(|f| f.meta.sitelat).collect()).unwrap_or(s.cfg.site.latitude);
    let lon = med(tonight.iter().filter_map(|f| f.meta.sitelong).collect()).unwrap_or(s.cfg.site.longitude);
    let t0 = night.and_time(NaiveTime::from_hms_opt(19, 0, 0).unwrap());
    (0..=144)
        .map(|k| {
            let local = t0 + Duration::minutes(5 * k);
            let sc = SkyContext::at(local - Duration::minutes(off), lat, lon, target);
            [k as f64 / 12.0, sc.target_alt.unwrap_or(f64::NAN), sc.moon_alt, sc.sun_alt]
        })
        .collect()
}

fn median(mut v: Vec<f64>) -> Option<f64> {
    v.sort_by(f64::total_cmp);
    v.get(v.len() / 2).copied()
}

pub fn timeline_card(ui: &mut egui::Ui, a: &mut UiState, s: &mut Session) {
    let p = a.pal;
    let night = a.fu.night.or(a.fu.sel.map(|i| s.lights[i].night)).or(s.nights.first().copied());
    let Some(night) = night else { return };
    ui.horizontal(|ui| {
        ui.label(RichText::new(format!("Night of {}", d_short(night))).font(crate::theme::semibold(16.0)));
        if a.fu.night.is_none() {
            ui.label(RichText::new("(selected frame's night)").font(body(13.0)).color(p.muted));
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let opts = [(Metric::Sky, "Sky"), (Metric::Ecc, "Eccentricity"), (Metric::Fwhm, "FWHM"), (Metric::T, "Transparency")];
            let ordered: Vec<(Metric, &str)> = opts.iter().rev().copied().collect();
            if let Some(k) = w::seg(ui, &p, &ordered, a.fu.metric) {
                a.fu.metric = k;
            }
        });
    });
    ui.add_space(6.0);
    let cur = a.curves.entry(night).or_insert_with(|| curves(s, night)).clone();
    timeline(ui, a, s, night, &cur);
    ui.add_space(6.0);
    brush_bar(ui, a, s);
}

fn timeline(ui: &mut egui::Ui, a: &mut UiState, s: &mut Session, night: NaiveDate, cur: &[[f64; 4]]) {
    let p = a.pal;
    let wd = ui.available_width().max(300.0);
    let ht = 240.0;
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(wd, ht), Sense::click_and_drag());
    let (pl, pr, pt, pb) = (42.0, 38.0, 14.0, 26.0);
    let plot = Rect::from_min_max(rect.min + Vec2::new(pl, pt), rect.max - Vec2::new(pr, pb));
    let hx = |h: f32| plot.left() + plot.width() * h / 12.0;
    let ids: Vec<usize> = (0..s.lights.len()).filter(|&i| s.lights[i].night == night).collect();
    let val = |i: usize| -> Option<f64> {
        let l = &s.lights[i];
        match a.fu.metric {
            Metric::T => l.t,
            Metric::Fwhm => l.fwhm,
            Metric::Ecc => l.ecc,
            Metric::Sky => l.sky,
        }
    };
    let vmax = ids.iter().filter_map(|&i| val(i)).fold(0.0, f64::max);
    let (lo, hi, label, dec) = match a.fu.metric {
        Metric::T => (0.0, 1.1, "transparency", 2),
        Metric::Fwhm => (0.0, vmax.max(8.0) * 1.05, "FWHM px", 1),
        Metric::Ecc => (0.2, 0.95, "eccentricity", 2),
        Metric::Sky => (0.0, vmax.max(0.3) * 1.08, "sky e-/px/s", 3),
    };
    let y = |v: f64| plot.top() + plot.height() * (1.0 - ((v - lo) / (hi - lo)) as f32);
    let ya = |alt: f64| plot.top() + plot.height() * (1.0 - (alt.max(0.0) / 90.0) as f32);
    let pa = ui.painter_at(rect);
    // Twilight bands.
    // Contiguous runs of the same twilight band, so there are no seams.
    let band = |sun: f64| if sun > 0.0 { 4 } else if sun > -6.0 { 3 } else if sun > -12.0 { 2 } else if sun > -18.0 { 1 } else { 0 };
    let mut k = 0;
    while k + 1 < cur.len() {
        let b = band(cur[k][3]);
        let mut e = k + 1;
        while e + 1 < cur.len() && band(cur[e][3]) == b {
            e += 1;
        }
        if b > 0 {
            let op = [0.0, 0.08, 0.18, 0.3, 0.42][b];
            let r = Rect::from_min_max(Pos2::new(hx(cur[k][0] as f32), plot.top()), Pos2::new(hx(cur[e][0] as f32), plot.bottom()));
            pa.rect_filled(r, CornerRadius::ZERO, alpha(p.accent, op * 0.55));
        }
        k = e;
    }
    // Grid and axes.
    for f in [0.0, 0.25, 0.5, 0.75, 1.0] {
        let v = lo + f * (hi - lo);
        pa.line_segment([Pos2::new(plot.left(), y(v)), Pos2::new(plot.right(), y(v))], Stroke::new(1.0, p.line));
        pa.text(Pos2::new(plot.left() - 6.0, y(v)), Align2::RIGHT_CENTER, format!("{v:.dec$}"), mono(10.0), p.faint);
    }
    for alt in [0.0, 30.0, 60.0, 90.0] {
        pa.text(Pos2::new(plot.right() + 6.0, ya(alt)), Align2::LEFT_CENTER, format!("{alt:.0}°"), mono(10.0), p.faint);
    }
    for hh in [19, 21, 23, 25, 27, 29, 31] {
        pa.text(Pos2::new(hx((hh - 19) as f32), rect.bottom() - 8.0), Align2::CENTER_CENTER, format!("{:02}:00", hh % 24), mono(10.0), p.faint);
    }
    pa.text(Pos2::new(plot.left(), plot.top() - 3.0), Align2::LEFT_BOTTOM, label, body(10.0), p.faint);
    pa.text(Pos2::new(plot.right(), plot.top() - 3.0), Align2::RIGHT_BOTTOM, "altitude", body(10.0), p.faint);
    // Target and Moon altitude.
    let tline: Vec<Pos2> = cur.iter().filter(|c| c[1].is_finite()).map(|c| Pos2::new(hx(c[0] as f32), ya(c[1]))).collect();
    if tline.len() > 1 {
        pa.add(Shape::line(tline, Stroke::new(1.5, alpha(p.muted, 0.7))));
        if let Some(top) = cur.iter().filter(|c| c[1].is_finite()).max_by(|a, b| a[1].total_cmp(&b[1])) {
            let target = s.lights.iter().find(|l| l.night == night).map(|l| l.target.clone()).unwrap_or_default();
            pa.text(Pos2::new(hx(top[0] as f32), ya(top[1]) - 6.0), Align2::CENTER_BOTTOM, format!("{target} altitude"), body(10.5), p.muted);
        }
    }
    let up: Vec<&[f64; 4]> = cur.iter().filter(|c| c[2] > 0.0).collect();
    if up.len() > 1 {
        let pts: Vec<Pos2> = up.iter().map(|c| Pos2::new(hx(c[0] as f32), ya(c[2]))).collect();
        pa.extend(Shape::dashed_line(&pts, Stroke::new(1.3, alpha(p.warn, 0.9)), 2.0, 4.0));
        let illum = ids.iter().find_map(|&i| s.lights[i].millum);
        let setting = std::ptr::eq(up[0], &cur[0]);
        let txt = format!("Moon{}{}", illum.map(|m| format!(" {:.0}% lit", m * 100.0)).unwrap_or_default(), if setting { ", setting" } else { ", rising" });
        pa.text(Pos2::new(hx(up[0][0] as f32) + 4.0, ya(up[0][2]) - 6.0), Align2::LEFT_BOTTOM, txt, body(10.5), p.warn);
    }
    // Brush.
    if let Some(b) = a.fu.brush.as_ref().filter(|b| b.night == night) {
        let r = Rect::from_min_max(Pos2::new(hx(b.a), plot.top()), Pos2::new(hx(b.b).max(hx(b.a) + 1.0), plot.bottom()));
        pa.rect(r, CornerRadius::ZERO, alpha(p.accent, 0.18), Stroke::new(1.0, p.accent), egui::StrokeKind::Inside);
    }
    // Limit line: transparency (draggable) or FWHM.
    let mut limit_y = None;
    match a.fu.metric {
        Metric::T => {
            let yy = y(s.limits[0]);
            limit_y = Some(yy);
            pa.extend(Shape::dashed_line(&[Pos2::new(plot.left(), yy), Pos2::new(plot.right(), yy)], Stroke::new(1.5, p.rej), 5.0, 4.0));
            pa.text(Pos2::new(plot.right() - 4.0, yy - 5.0), Align2::RIGHT_BOTTOM, format!("reject below {:.2} · drag to change", s.limits[0]), body(11.0), p.rej);
        }
        Metric::Fwhm => {
            if let Some(m) = median(ids.iter().filter_map(|&i| s.lights[i].fwhm).collect()) {
                let yy = y(m * s.limits[2]);
                pa.extend(Shape::dashed_line(&[Pos2::new(plot.left(), yy), Pos2::new(plot.right(), yy)], Stroke::new(1.5, p.rej), 5.0, 4.0));
                pa.text(Pos2::new(plot.right() - 4.0, yy - 5.0), Align2::RIGHT_BOTTOM, format!("{:.2}× night median ({m:.1} px)", s.limits[2]), body(11.0), p.rej);
            }
        }
        _ => {}
    }
    // Points.
    let mut hovered: Option<usize> = None;
    let hover = resp.hover_pos();
    let mut pts: Vec<(usize, Pos2)> = Vec::new();
    for &i in &ids {
        let Some(v) = val(i) else { continue };
        let pos = Pos2::new(hx(hours(night, s.lights[i].time)), y(v.min(hi)));
        pts.push((i, pos));
    }
    for &(i, pos) in &pts {
        if hover.is_some_and(|h| h.distance(pos) < 7.0) {
            hovered = Some(i);
        }
    }
    for &(i, pos) in &pts {
        let g = s.grade(i);
        let sel = a.fu.sel == Some(i);
        let r = if sel { 7.0 } else { 4.5 };
        pa.circle_filled(pos, r, p.grade(g));
        let stroke = if sel { Stroke::new(2.0, p.fg) } else if s.pinned(i).is_some() { Stroke::new(1.2, p.fg) } else { Stroke::new(1.0, p.panel) };
        pa.circle_stroke(pos, r, stroke);
    }
    if let Some(i) = hovered {
        let l = &s.lights[i];
        let v = val(i).map(|v| format!("{v:.dec$}")).unwrap_or_default();
        resp.clone().on_hover_text(format!("{} {} · {label} {v} · {}", l.obj, l.hhmm(), crate::app::grade_name(s.grade(i))));
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    // Interaction: click a point, drag the limit, or brush a time range.
    let near_limit = |pos: Pos2| limit_y.is_some_and(|ly| (pos.y - ly).abs() < 7.0 && plot.x_range().contains(pos.x));
    if let Some(h) = hover {
        if hovered.is_none() && near_limit(h) {
            ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeVertical);
        }
    }
    if resp.drag_started() {
        if let Some(pos) = resp.interact_pointer_pos() {
            if hovered.is_none() && near_limit(pos) {
                a.fu.dragging_limit = true;
            } else if plot.x_range().contains(pos.x) {
                a.fu.brush_from = Some(pos.x);
            }
        }
    }
    if resp.dragged() {
        if let Some(pos) = resp.interact_pointer_pos() {
            if a.fu.dragging_limit {
                let v = lo + (1.0 - ((pos.y - plot.top()) / plot.height()) as f64) * (hi - lo);
                let v = (v.clamp(0.2, 0.95) * 100.0).round() / 100.0;
                if (v - s.limits[0]).abs() > 1e-9 {
                    s.limits[0] = v;
                    s.regrade();
                }
            } else if let Some(x0) = a.fu.brush_from {
                let r = Rect::from_min_max(Pos2::new(x0.min(pos.x), plot.top()), Pos2::new(x0.max(pos.x), plot.bottom()));
                pa.rect(r, CornerRadius::ZERO, alpha(p.accent, 0.18), Stroke::new(1.0, p.accent), egui::StrokeKind::Inside);
            }
        }
    }
    if resp.drag_stopped() {
        if a.fu.dragging_limit {
            a.fu.dragging_limit = false;
            s.save_limits();
        } else if let (Some(x0), Some(pos)) = (a.fu.brush_from.take(), resp.interact_pointer_pos()) {
            if (pos.x - x0).abs() < 6.0 {
                a.fu.brush = None;
            } else {
                let th = |x: f32| (x - plot.left()) / plot.width() * 12.0;
                let (ha, hb) = (th(x0.min(pos.x)), th(x0.max(pos.x)));
                let sel: Vec<usize> = ids.iter().copied().filter(|&i| (ha..=hb).contains(&hours(night, s.lights[i].time))).collect();
                a.fu.brush = Some(Brush { night, a: ha, b: hb, ids: sel });
            }
        }
    }
    if resp.clicked() {
        match hovered {
            Some(i) => {
                a.fu.sel = Some(i);
                a.fu.scroll_to_sel = true;
            }
            None => a.fu.brush = None,
        }
    }
}

fn brush_bar(ui: &mut egui::Ui, a: &mut UiState, s: &mut Session) {
    let p = a.pal;
    let Some(b) = a.fu.brush.clone() else {
        ui.add(
            egui::Label::new(
                RichText::new("Shaded bands are twilight; the grey line is the target's altitude, the dotted line the Moon. Drag across the chart to select a time range.")
                    .font(body(13.0))
                    .color(p.muted),
            )
            .wrap(),
        );
        return;
    };
    let hm = |h: f32| {
        let m = (h * 60.0).round() as i64 + 19 * 60;
        format!("{:02}:{:02}", (m / 60) % 24, m % 60)
    };
    let mut act: Option<Option<Grade>> = None;
    let mut clear = false;
    w::wrap_row(ui, |ui| {
        ui.label(RichText::new(format!("{} frames", b.ids.len())).font(crate::theme::semibold(13.0)));
        ui.label(RichText::new(format!("{}–{}", hm(b.a), hm(b.b))).font(mono(13.0)));
        if w::button(ui, &p, "Reject all", Kind::Danger, None).clicked() {
            act = Some(Some(Grade::Reject));
        }
        if w::button(ui, &p, "Keep all", Kind::Normal, None).clicked() {
            act = Some(Some(Grade::Good));
        }
        if w::button(ui, &p, "Back to crux's grades", Kind::Ghost, None).clicked() {
            act = Some(None);
        }
        if w::button(ui, &p, "Clear", Kind::Ghost, None).clicked() {
            clear = true;
        }
    });
    if let Some(g) = act {
        s.set_override(&b.ids, g);
        let msg = match g {
            Some(Grade::Reject) => format!("{} frames from {} rejected", b.ids.len(), hm(b.a)),
            Some(_) => format!("{} frames kept", b.ids.len()),
            None => format!("{} frames back to crux's grades", b.ids.len()),
        };
        a.toast(ui.ctx(), msg, true);
    }
    if clear {
        a.fu.brush = None;
    }
}

/// Transparency through a night, with the reject limit (Home's last-night card).
pub fn sparkline(ui: &mut egui::Ui, p: &Pal, s: &Session, ids: &[usize]) {
    let wd = ui.available_width().max(280.0);
    let ht = 64.0;
    let (rect, _) = ui.allocate_exact_size(Vec2::new(wd, ht + 12.0), Sense::hover());
    let Some(&first) = ids.first() else { return };
    let night = s.lights[first].night;
    let x = |i: usize| rect.left() + 8.0 + (wd - 16.0) * hours(night, s.lights[i].time) / 12.0;
    let y = |v: f64| rect.top() + ht - 6.0 - (ht - 12.0) * (v.min(1.05) / 1.05) as f32;
    let pa = ui.painter_at(rect);
    let ly = y(s.limits[0]);
    pa.extend(Shape::dashed_line(&[Pos2::new(rect.left() + 8.0, ly), Pos2::new(rect.right() - 8.0, ly)], Stroke::new(1.0, alpha(p.rej, 0.6)), 3.0, 4.0));
    pa.text(Pos2::new(rect.right() - 8.0, ly - 4.0), Align2::RIGHT_BOTTOM, format!("T {:.2}", s.limits[0]), body(10.0), p.rej);
    let pts: Vec<(usize, Pos2)> = ids.iter().filter_map(|&i| s.lights[i].t.map(|t| (i, Pos2::new(x(i), y(t))))).collect();
    if pts.len() > 1 {
        pa.add(Shape::line(pts.iter().map(|q| q.1).collect(), Stroke::new(1.5, p.accent)));
    }
    for (i, q) in &pts {
        pa.circle_filled(*q, 3.0, p.grade(s.grade(*i)));
    }
    for hh in [20, 22, 24, 26, 28, 30] {
        pa.text(Pos2::new(rect.left() + 8.0 + (wd - 16.0) * (hh - 19) as f32 / 12.0, rect.bottom() - 2.0), Align2::CENTER_BOTTOM, format!("{:02}:00", hh % 24), mono(10.0), p.faint);
    }
}

/// Distribution of the main reason's measure across the frame's cohort,
/// with the limit and this frame marked.
pub fn why(ui: &mut egui::Ui, p: &Pal, s: &Session, i: usize) {
    let l = &s.lights[i];
    let main = l.flags.first().map(|f| f.check).unwrap_or(Check::Transparency);
    type Get = fn(&crate::session::Light) -> Option<f64>;
    let same_panel = |j: usize| s.lights[j].obj == l.obj;
    let same_night = |j: usize| s.lights[j].night == l.night;
    let (get, title, cohort, limit, key): (Get, &str, Box<dyn Fn(usize) -> bool>, Option<f64>, &str) = match main {
        Check::Trail => (|x| x.tx, "Trail brightness (× pixel noise), all trailed frames", Box::new(|_| true), Some(s.limits[1]), "trails.reject_sigma"),
        Check::Focus => (|x| x.fr, "FWHM / night median, this night", Box::new(same_night), Some(s.limits[2]), "focus.reject_ratio"),
        Check::Sky => (|x| x.skyr, "Sky / panel's best, this panel", Box::new(same_panel), Some(s.limits[3]), "sky.reject_ratio"),
        Check::Stars => (|x| x.stars.map(|v| v as f64), "Star count, this night", Box::new(same_night), None, "stars.reject_frac"),
        Check::Tracking => (|x| x.ecc, "Eccentricity, this night", Box::new(same_night), None, "tracking.reject_ecc"),
        Check::Gradient => (|x| x.grad, "Gradient %, this panel", Box::new(same_panel), None, "gradient.warn_pct"),
        _ => (|x| x.t, "Transparency vs this panel, all nights", Box::new(same_panel), Some(s.limits[0]), "transparency.reject"),
    };
    let vals: Vec<f64> = (0..s.lights.len()).filter(|&j| cohort(j)).filter_map(|j| get(&s.lights[j])).collect();
    let Some(me) = get(l) else { return };
    if vals.is_empty() {
        return;
    }
    let lo = vals.iter().copied().chain(limit).fold(f64::MAX, f64::min);
    let hi = vals.iter().copied().chain(limit).fold(f64::MIN, f64::max);
    let nb = 24;
    let bw = ((hi - lo) / nb as f64).max(1e-9);
    let mut bins = vec![0usize; nb];
    for v in &vals {
        bins[(((v - lo) / bw) as usize).min(nb - 1)] += 1;
    }
    w::eyebrow(ui, p, &format!("Why: {title}"));
    let wd = ui.available_width();
    let ht = 84.0;
    let (rect, _) = ui.allocate_exact_size(Vec2::new(wd, ht), Sense::hover());
    let pa = ui.painter_at(rect);
    let mx = *bins.iter().max().unwrap_or(&1) as f32;
    let x = |v: f64| rect.left() + 4.0 + (wd - 8.0) * ((v - lo) / (hi - lo).max(1e-9)) as f32;
    for (k, &b) in bins.iter().enumerate() {
        let h = (ht - 26.0) * b as f32 / mx;
        let r = Rect::from_min_size(Pos2::new(x(lo + k as f64 * bw) + 0.5, rect.bottom() - 16.0 - h), Vec2::new((wd - 8.0) / nb as f32 - 1.5, h));
        pa.rect_filled(r, CornerRadius::same(1), alpha(p.faint, 0.55));
    }
    if let Some(t) = limit {
        let xx = x(t);
        pa.extend(Shape::dashed_line(&[Pos2::new(xx, rect.top() + 4.0), Pos2::new(xx, rect.bottom() - 16.0)], Stroke::new(1.0, p.rej), 3.0, 3.0));
        pa.text(Pos2::new(xx + 3.0, rect.top() + 12.0), Align2::LEFT_BOTTOM, format!("limit {t:.2}"), body(10.0), p.rej);
    }
    let xm = x(me);
    pa.line_segment([Pos2::new(xm, rect.top() + 4.0), Pos2::new(xm, rect.bottom() - 16.0)], Stroke::new(2.5, p.accent));
    pa.text(Pos2::new((xm + 4.0).min(rect.right() - 60.0), rect.bottom() - 20.0), Align2::LEFT_BOTTOM, "this frame", body(10.5), p.accent);
    let fmt = |v: f64| if hi > 100.0 { format!("{v:.0}") } else { format!("{v:.2}") };
    pa.text(Pos2::new(rect.left() + 4.0, rect.bottom() - 2.0), Align2::LEFT_BOTTOM, fmt(lo), mono(10.0), p.faint);
    pa.text(Pos2::new(rect.right() - 4.0, rect.bottom() - 2.0), Align2::RIGHT_BOTTOM, fmt(hi), mono(10.0), p.faint);
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        ui.label(RichText::new(format!("{} frames in the cohort · setting", vals.len())).font(body(11.5)).color(p.muted));
        ui.label(RichText::new(format!("thresholds.{key}")).font(mono(11.5)).color(p.muted));
        ui.label(RichText::new("in crux.toml").font(body(11.5)).color(p.muted));
    });
}
