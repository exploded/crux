//! Home: last night at a glance, the target gallery, the Inbox summary,
//! a calendar of imaging nights and the library.

use crate::app::{Show, UiState, View};
use crate::chart;
use crate::session::{Session, Sev};
use crate::theme::{body, mix, mono, semibold};
use crate::widgets::{self as w, d_short, fmt_h, Kind};
use chrono::{Datelike, Duration, NaiveDate};
use crux_core::grade::{Check, Grade};
use crux_core::FrameType;
use eframe::egui::{self, Align2, Color32, CornerRadius, Pos2, Rect, RichText, ScrollArea, Sense, Stroke, StrokeKind, Vec2};
use std::collections::BTreeMap;

/// "LMC Mosaic" from "2026-11-05_LMC_Mosaic".
pub fn pretty_name(folder: &str) -> String {
    let b = folder.as_bytes();
    let rest = if b.len() > 11 && b[4] == b'-' && b[7] == b'-' && (b[10] == b'_' || b[10] == b' ') { &folder[11..] } else { folder };
    rest.replace('_', " ")
}

/// A target card's name: the target when every light names the same one
/// ("LMC" for "LMC Panel 1" … "LMC Panel 36"), else the folder's name
/// (several targets, or none in the headers).
/// Also the count line's start: "36 panels · " for a mosaic, "5 targets · "
/// for a folder of several.
fn card_name<'a>(objects: impl Iterator<Item = &'a String>, folder: &str) -> (String, String) {
    let mut objs: Vec<&String> = objects.collect();
    objs.sort();
    objs.dedup();
    let mut t: Vec<String> = objs.iter().filter(|o| !o.trim().is_empty()).map(|o| crux_core::targets::normalise(o)).collect();
    t.sort_by_key(|x| x.to_lowercase());
    t.dedup_by_key(|x| x.to_lowercase());
    match t.as_slice() {
        [one] => (one.clone(), if objs.len() > 1 { format!("{} panels · ", objs.len()) } else { String::new() }),
        [] => (pretty_name(folder), String::new()),
        many => (pretty_name(folder), format!("{} targets · ", many.len())),
    }
}

pub(crate) fn pick_folder(a: &mut UiState) {
    if let Some(d) = rfd::FileDialog::new().set_title("Open a folder of subs").pick_folder() {
        a.open_request = Some(d);
    }
}

pub fn show(ui: &mut egui::Ui, a: &mut UiState, s: Option<&mut Session>) {
    if s.is_none() && a.lib.view.is_some() {
        return crate::libview::show(ui, a);
    }
    let p = a.pal;
    ScrollArea::vertical().id_salt("home").auto_shrink([false, false]).show(ui, |ui| {
        ui.set_width(ui.available_width() - 8.0);
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                w::h1(ui, "Home");
                let sub = match (&s, &a.loading) {
                    (Some(s), _) => format!(
                        "crux graded {} light{} in {}.",
                        s.lights.len(),
                        if s.lights.len() == 1 { "" } else { "s" },
                        s.root.display()
                    ),
                    (None, Some(d)) => format!("Reading {}…", d.display()),
                    (None, None) => "Open a folder of N.I.N.A. subs. crux grades each frame and gives the reasons.".into(),
                };
                ui.label(RichText::new(sub).color(p.muted));
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Max), |ui| {
                if w::button(ui, &p, "+ Open folder", Kind::Normal, None).clicked() {
                    pick_folder(a);
                }
            });
        });
        ui.add_space(16.0);
        let wide = ui.available_width() >= 900.0;
        let total = ui.available_width();
        let (lw, rw) = if wide { ((total - 16.0) * 1.4 / 2.4, (total - 16.0) / 2.4) } else { (total, total) };
        let mut s = s;
        let left_col = |ui: &mut egui::Ui, a: &mut UiState, s: &mut Option<&mut Session>| {
            ui.set_width(lw);
            match s {
                Some(s) => last_night(ui, a, s),
                None => loading_card(ui, a),
            }
            ui.add_space(16.0);
            gallery(ui, a, s.as_deref());
        };
        let right_col = |ui: &mut egui::Ui, a: &mut UiState, s: &Option<&mut Session>| {
            ui.set_width(rw);
            if let Some(s) = s {
                inbox_card(ui, a, s);
                ui.add_space(16.0);
                crate::auto::watch_card(ui, a, &s.root, &s.cfg);
                ui.add_space(16.0);
            }
            calendar_card(ui, a, s.as_deref());
            ui.add_space(16.0);
            library_card(ui, a, s.as_deref());
            ui.add_space(16.0);
            // The open folder's parent, unless that's a drive root or the
            // home folder: then the open folder itself.
            let parent = s.as_deref().map(|s| s.root.parent().filter(|x| !crux_core::paths::too_broad(x)).unwrap_or(&s.root).to_path_buf());
            crate::auto::autopilot_card(ui, a, parent.as_deref());
        };
        if wide {
            ui.horizontal_top(|ui| {
                ui.vertical(|ui| left_col(ui, a, &mut s));
                ui.add_space(8.0);
                ui.vertical(|ui| right_col(ui, a, &s));
            });
        } else {
            left_col(ui, a, &mut s);
            ui.add_space(16.0);
            right_col(ui, a, &s);
        }
        ui.add_space(40.0);
    });
}

fn loading_card(ui: &mut egui::Ui, a: &mut UiState) {
    let p = a.pal;
    w::card(ui, &p, |ui| {
        if let Some(e) = &a.failed {
            w::h2(ui, "Couldn't open that folder");
            ui.label(RichText::new(e).color(p.rej));
            ui.add_space(8.0);
            if w::button(ui, &p, "Open another folder", Kind::Primary, None).clicked() {
                pick_folder(a);
            }
            return;
        }
        if a.loading.is_none() {
            w::eyebrow(ui, &p, "Start here");
            ui.label(RichText::new("Open a folder").font(semibold(28.0)));
            ui.label(RichText::new("Drop a folder on this window, or pick one. crux reads the headers first, then measures the pixels in the background. It doesn't change anything in the folder.").color(p.muted));
            ui.add_space(10.0);
            if w::button(ui, &p, "Open folder", Kind::Primary, None).clicked() {
                pick_folder(a);
            }
            return;
        }
        w::eyebrow(ui, &p, "Reading");
        if let Some(f) = &a.facts {
            ui.label(RichText::new(format!("{} lights", f.lights)).font(semibold(40.0)));
            let mut bits = vec![];
            if !f.filters.is_empty() {
                bits.push(f.filters.join("/"));
            }
            bits.push(format!("{} night{}", f.nights, if f.nights == 1 { "" } else { "s" }));
            if f.panels > 1 {
                bits.push(format!("{} panels", f.panels));
            }
            bits.push(format!("{} flats, {} masters", f.flats, f.masters));
            ui.label(RichText::new(format!("{} · {} files read in {:.2} s", bits.join(" · "), f.files, f.seconds)).color(p.muted));
        } else {
            ui.label(RichText::new("Reading headers…").font(semibold(28.0)));
        }
        ui.add_space(10.0);
        if let Some((st, d, t)) = a.progress {
            let what = match st {
                crux_core::pipeline::Stage::Headers => "Reading headers",
                crux_core::pipeline::Stage::Calibration => "Checking calibration frames",
                crux_core::pipeline::Stage::Pixels => "Measuring lights (cached frames are skipped)",
            };
            ui.label(format!("{what}: {d} of {t}"));
            let (r, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 8.0), Sense::hover());
            ui.painter().rect_filled(r, CornerRadius::same(4), p.line);
            let f = if t > 0 { d as f32 / t as f32 } else { 0.0 };
            ui.painter().rect_filled(Rect::from_min_size(r.min, Vec2::new(r.width() * f, 8.0)), CornerRadius::same(4), p.accent);
        }
        ui.add_space(8.0);
        if w::button(ui, &p, "Cancel", Kind::Ghost, None).clicked() {
            a.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    });
}

fn last_night(ui: &mut egui::Ui, a: &mut UiState, s: &Session) {
    let p = a.pal;
    let Some(&last) = s.nights.last() else {
        w::card(ui, &p, |ui| {
            w::h2(ui, "No lights in this folder");
            w::muted(ui, &p, "crux found calibration frames only. The Calibrate page lists their checks.");
        });
        return;
    };
    let ids: Vec<usize> = (0..s.lights.len()).filter(|&i| s.lights[i].night == last).collect();
    let c = s.counts(ids.iter().copied());
    let secs: f64 = ids.iter().map(|&i| s.lights[i].exp).sum();
    let mut filters: Vec<String> = ids.iter().map(|&i| s.lights[i].filter.clone()).collect();
    filters.sort();
    filters.dedup();
    let moon = ids.iter().find_map(|&i| s.lights[i].millum);
    let target = ids.first().map(|&i| s.lights[i].target.clone()).unwrap_or_default();
    let cloud = ids
        .iter()
        .filter(|&&i| s.lights[i].flags.iter().any(|f| matches!(f.check, Check::Transparency | Check::Sky | Check::Stars | Check::Patchy)))
        .map(|&i| s.lights[i].hhmm())
        .next();
    let g = &s.triage.grouping;
    let flats_tonight: Vec<&crux_core::group::Group> = g.groups.iter().filter(|x| x.kind == FrameType::Flat && !x.master && x.night == Some(last)).collect();
    // Background gradient behind the card, like the mockup.
    let bg = ui.painter().add(egui::Shape::Noop);
    let inner = egui::Frame::new().stroke(Stroke::new(1.0, p.line)).corner_radius(12).inner_margin(16).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal_top(|ui| {
            ui.vertical(|ui| {
                w::eyebrow(ui, &p, &format!("Last night · {} into {}", d_short(last), d_short(last + Duration::days(1))));
                ui.label(RichText::new(format!("{} subs", ids.len())).font(semibold(40.0)));
                let moon_s = moon.map(|m| format!(" · Moon {:.0}% lit", m * 100.0)).unwrap_or_default();
                ui.label(RichText::new(format!("{target} · {} · {}{moon_s}", filters.join("/"), fmt_h(secs))).color(p.muted));
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
                let (r, _) = ui.allocate_exact_size(Vec2::new(170.0, 52.0), Sense::hover());
                let pa = ui.painter();
                for (k, (n, label, col)) in [(c[0], "good", p.good), (c[1], "warn", p.warn), (c[2], "reject", p.rej)].into_iter().enumerate() {
                    let x = r.left() + 8.0 + k as f32 * 58.0;
                    pa.text(Pos2::new(x, r.top()), Align2::LEFT_TOP, n.to_string(), semibold(22.0), col);
                    pa.text(Pos2::new(x, r.top() + 32.0), Align2::LEFT_TOP, label, body(12.0), p.muted);
                }
            });
        });
        ui.add_space(6.0);
        chart::sparkline(ui, &p, s, &ids);
        ui.add_space(4.0);
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 18.0;
            match &cloud {
                Some(t) => {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        ui.label(RichText::new("☁ Cloud, haze or bright sky from").font(body(13.0)).color(p.muted));
                        ui.label(RichText::new(t).font(mono(13.0)).color(p.fg));
                    });
                }
                None => {
                    ui.label(RichText::new("Clear all night").font(body(13.0)).color(p.muted));
                }
            }
            for fg in &flats_tonight {
                let dir = fg.members.first().and_then(|&m| s.triage.scan.frames[m].meta.path.parent()).and_then(|d| d.strip_prefix(&s.root).ok()).map(|d| d.display().to_string()).unwrap_or_default();
                let grade = s.triage.cal.set(fg.id).map(|x| x.grade).unwrap_or(Grade::Good);
                let verdict = match grade {
                    Grade::Good => "checked, good",
                    Grade::Warn => "checked, warnings",
                    Grade::Reject => "checked, rejected",
                };
                ui.label(RichText::new(format!("{} dawn {} flats in {dir} ({verdict})", fg.members.len(), fg.filter.clone().unwrap_or_default())).font(body(13.0)).color(p.muted));
            }
        });
        ui.add_space(8.0);
        let mut go = None;
        ui.horizontal(|ui| {
            if w::button(ui, &p, &format!("Review {} flagged", c[1] + c[2]), Kind::Primary, None).clicked() {
                go = Some(View::Frames);
            }
            if w::button(ui, &p, "Organise files", Kind::Normal, None).clicked() {
                go = Some(View::Organise);
            }
        });
        if let Some(v) = go {
            if v == View::Frames {
                a.fu.night = Some(last);
                a.fu.show = Show::Flagged;
                a.fu.sel = None;
            }
            a.view = v;
        }
    });
    let r = inner.response.rect;
    let cols = [mix(p.accent, 0.10, p.panel), mix(p.accent, 0.04, p.panel), p.panel, mix(p.accent, 0.06, p.panel)];
    ui.painter().set(bg, rounded_mesh(r.shrink(1.0), cols));
}

/// A gradient-filled rounded rectangle (corner radius 12) as a triangle fan.
fn rounded_mesh(r: Rect, cols: [Color32; 4]) -> egui::Shape {
    let rad = 11.0f32;
    let mut pts = Vec::new();
    let corners = [(r.right() - rad, r.top() + rad, -90.0f32), (r.right() - rad, r.bottom() - rad, 0.0), (r.left() + rad, r.bottom() - rad, 90.0), (r.left() + rad, r.top() + rad, 180.0)];
    for (cx, cy, a0) in corners {
        for k in 0..=6 {
            let a = (a0 + 15.0 * k as f32).to_radians();
            pts.push(Pos2::new(cx + rad * a.cos(), cy + rad * a.sin()));
        }
    }
    let colour = |q: Pos2| {
        let u = ((q.x - r.left()) / r.width()).clamp(0.0, 1.0);
        let v = ((q.y - r.top()) / r.height()).clamp(0.0, 1.0);
        let lerp = |a: Color32, b: Color32, t: f32| mix(b, t, a);
        lerp(lerp(cols[0], cols[1], u), lerp(cols[3], cols[2], u), v)
    };
    let mut mesh = egui::Mesh::default();
    mesh.colored_vertex(r.center(), colour(r.center()));
    for q in &pts {
        mesh.colored_vertex(*q, colour(*q));
    }
    let n = pts.len() as u32;
    for k in 0..n {
        mesh.add_triangle(0, 1 + k, 1 + (k + 1) % n);
    }
    egui::Shape::mesh(mesh)
}

pub(crate) struct Target {
    pub name: String,
    pub sub: String,
    pub secs: BTreeMap<String, f64>,
    pub status: (Option<Color32>, String),
    pub image: Option<std::path::PathBuf>,
    pub thumb: Option<usize>,
    pub open: Option<std::path::PathBuf>,
    pub is_current: bool,
}

fn gallery(ui: &mut egui::Ui, a: &mut UiState, s: Option<&Session>) {
    let p = a.pal;
    let mut targets: Vec<Target> = Vec::new();
    if let Some(s) = s {
        let c = s.counts(0..s.lights.len());
        let mut secs = BTreeMap::new();
        for l in &s.lights {
            *secs.entry(l.filter.clone()).or_insert(0.0) += l.exp;
        }
        let months: Vec<String> = s.nights.iter().map(|n| n.format("%b %Y").to_string()).collect();
        let span = match (months.first(), months.last()) {
            (Some(a), Some(b)) if a != b => format!("{a}–{b}"),
            (Some(a), _) => a.clone(),
            _ => String::new(),
        };
        let (name, panels) = card_name(s.lights.iter().map(|l| &l.obj), &s.root.file_name().unwrap_or_default().to_string_lossy());
        // Best sub as the picture when there's no finished image.
        let best = (0..s.lights.len()).filter(|&i| s.grade(i) == Grade::Good).max_by(|&x, &y| s.lights[x].t.unwrap_or(0.0).total_cmp(&s.lights[y].t.unwrap_or(0.0)));
        let image = a.target_image.clone().flatten();
        targets.push(Target {
            name,
            sub: format!("{panels}{} night{} · {span}", s.nights.len(), if s.nights.len() == 1 { "" } else { "s" }),
            secs,
            status: (Some(if c[2] > 0 { p.warn } else { p.good }), format!("{} frames to review", c[1] + c[2])),
            image,
            thumb: best,
            open: None,
            is_current: true,
        });
    }
    // Only the open target: the others are on the library's Home.
    if targets.is_empty() {
        return;
    }
    let library = s.and_then(|s| crate::worker::library_parent(&s.root, &s.cfg));
    ui.horizontal(|ui| {
        w::h2(ui, "Target");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if let Some(lib) = &library {
                let r = ui.add(
                    egui::Label::new(RichText::new(format!("All targets in {}", lib.display())).font(body(12.0)).color(p.muted).underline())
                        .sense(Sense::click()),
                );
                if r.on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
                    a.open_request = Some(lib.clone());
                }
            }
        });
    });
    ui.add_space(8.0);
    let gap = 14.0;
    let wd = ui.available_width();
    let cols = ((wd + gap) / (230.0 + gap)).floor().max(1.0) as usize;
    let cw = (wd - gap * (cols as f32 - 1.0)) / cols as f32;
    for row in targets.chunks(cols) {
        ui.horizontal_top(|ui| {
            ui.spacing_mut().item_spacing.x = gap;
            for t in row {
                target_card(ui, a, t, cw);
            }
        });
        ui.add_space(gap);
    }
}

pub(crate) fn target_card(ui: &mut egui::Ui, a: &mut UiState, t: &Target, cw: f32) {
    let p = a.pal;
    let img_h = cw * 2.0 / 3.0;
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(cw, img_h + 98.0), Sense::click());
    let pa = ui.painter();
    pa.rect(rect, CornerRadius::same(12), p.panel, Stroke::new(1.0, if resp.hovered() { p.faint } else { p.line }), StrokeKind::Inside);
    let ir = Rect::from_min_size(rect.min + Vec2::splat(1.0), Vec2::new(cw - 2.0, img_h));
    let ir_cr = CornerRadius { nw: 11, ne: 11, sw: 0, se: 0 };
    pa.rect_filled(ir, ir_cr, p.well);
    let tint = Color32::WHITE;
    let tex = t.image.as_ref().and_then(|i| a.gallery.get(i)).map(|x| (x.id(), x.size_vec2())).or(t.thumb.and_then(|i| a.thumbs.get(&i)).map(|x| (x.id(), x.size_vec2())));
    match tex {
        Some((id, size)) => {
            // Cover: crop the texture to the card's 3:2.
            let (ta, ca) = (size.x / size.y, ir.width() / ir.height());
            let uv = if ta > ca {
                let f = ca / ta;
                Rect::from_min_max(Pos2::new(0.5 - f / 2.0, 0.0), Pos2::new(0.5 + f / 2.0, 1.0))
            } else {
                let f = ta / ca;
                Rect::from_min_max(Pos2::new(0.0, 0.5 - f / 2.0), Pos2::new(1.0, 0.5 + f / 2.0))
            };
            pa.add(egui::epaint::RectShape::filled(ir, ir_cr, tint).with_texture(id, uv));
        }
        None => {
            pa.text(ir.center(), Align2::CENTER_CENTER, "No image yet", body(12.0), Color32::from_rgb(0x7d, 0x87, 0x9c));
        }
    }
    let mut y = ir.bottom() + 12.0;
    let x = rect.left() + 12.0;
    let tot: f64 = t.secs.values().sum::<f64>().max(1.0);
    pa.text(Pos2::new(x, y), Align2::LEFT_TOP, &t.name, semibold(13.0), p.fg);
    pa.text(Pos2::new(rect.right() - 12.0, y + 1.0), Align2::RIGHT_TOP, format!("{:.1} h", tot / 3600.0), mono(12.0), p.muted);
    y += 22.0;
    pa.text(Pos2::new(x, y), Align2::LEFT_TOP, &t.sub, body(12.0), p.muted);
    y += 22.0;
    let bar = Rect::from_min_size(Pos2::new(x, y), Vec2::new(cw - 24.0, 6.0));
    pa.rect_filled(bar, CornerRadius::same(3), p.line);
    let mut bx = bar.left();
    for (f, sec) in &t.secs {
        let bw = bar.width() * (*sec / tot) as f32;
        pa.rect_filled(Rect::from_min_size(Pos2::new(bx, bar.top()), Vec2::new(bw, 6.0)), CornerRadius::ZERO, p.filter(f));
        bx += bw;
    }
    y += 16.0;
    let c = Pos2::new(x + 3.5, y + 8.0);
    match t.status.0 {
        Some(col) => pa.circle_filled(c, 3.5, col),
        None => pa.circle_stroke(c, 3.0, Stroke::new(1.3, p.faint)),
    };
    pa.text(Pos2::new(x + 14.0, y + 8.0), Align2::LEFT_CENTER, &t.status.1, body(12.0), p.muted);
    let hours: Vec<String> = t.secs.iter().map(|(f, s)| format!("{f} {}", fmt_h(*s))).collect();
    let resp = resp.on_hover_cursor(egui::CursorIcon::PointingHand).on_hover_text(hours.join(" · "));
    if resp.clicked() {
        if t.is_current {
            a.view = View::Frames;
        } else if let Some(o) = &t.open {
            a.open_request = Some(o.clone());
        }
    }
}

fn inbox_card(ui: &mut egui::Ui, a: &mut UiState, s: &Session) {
    let p = a.pal;
    let items = s.inbox();
    w::card(ui, &p, |ui| {
        ui.horizontal(|ui| {
            w::h2(ui, "Inbox");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let r = ui.add(egui::Label::new(RichText::new("Open").font(body(13.0)).color(p.muted).underline()).sense(Sense::click()));
                if r.on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
                    a.view = View::Inbox;
                }
            });
        });
        ui.add_space(4.0);
        if items.is_empty() {
            w::muted(ui, &p, "No frames or sets need checking.");
        }
        for it in items.iter().take(3) {
            ui.horizontal_top(|ui| {
                let col = match it.sev {
                    Sev::Rej => p.rej,
                    Sev::Warn => p.warn,
                    Sev::Info => p.info,
                    Sev::Good => p.good,
                };
                ui.add_space(2.0);
                let (r, _) = ui.allocate_exact_size(Vec2::new(10.0, 18.0), Sense::hover());
                ui.painter().circle_filled(Pos2::new(r.center().x, r.top() + 9.0), 4.0, col);
                ui.add(egui::Label::new(RichText::new(&it.title).font(body(13.0))).wrap());
            });
        }
        if items.len() > 3 {
            ui.label(RichText::new(format!("and {} more", items.len() - 3)).font(body(12.0)).color(p.muted));
        }
    });
}

fn calendar_card(ui: &mut egui::Ui, a: &mut UiState, s: Option<&Session>) {
    let p = a.pal;
    let mut lights: BTreeMap<NaiveDate, f64> = BTreeMap::new();
    let mut flats: std::collections::BTreeSet<NaiveDate> = Default::default();
    // The open target's nights only.
    if let Some(s) = s {
        for l in &s.lights {
            *lights.entry(l.night).or_default() += l.exp;
        }
        if let Some(f) = a.folders.iter().find(|f| f.path == s.root) {
            flats.extend(f.flat_nights.iter().copied());
        }
    }
    w::card(ui, &p, |ui| {
        w::h2(ui, "Imaging nights");
        ui.add_space(6.0);
        let Some(&latest) = lights.keys().chain(flats.iter()).max() else {
            w::muted(ui, &p, "No nights yet.");
            return;
        };
        let weeks = (((ui.available_width() + 4.0) / 19.0).floor() as i64).clamp(8, 26);
        let end = latest + Duration::days(6 - latest.weekday().num_days_from_monday() as i64);
        let start = end - Duration::days(weeks * 7 - 1);
        let (rect, resp) = ui.allocate_exact_size(Vec2::new(weeks as f32 * 19.0 - 4.0, 7.0 * 19.0 - 4.0), Sense::hover());
        let pa = ui.painter();
        let mut tip = None;
        for k in 0..weeks * 7 {
            let d = start + Duration::days(k);
            let (col, row) = (k / 7, k % 7);
            let r = Rect::from_min_size(Pos2::new(rect.left() + col as f32 * 19.0, rect.top() + row as f32 * 19.0), Vec2::splat(15.0));
            let sec = lights.get(&d).copied();
            let fill = match sec {
                Some(x) if x > 5.0 * 3600.0 => p.accent,
                Some(x) if x > 2.5 * 3600.0 => mix(p.accent, 0.65, p.panel2),
                Some(_) => mix(p.accent, 0.35, p.panel2),
                None if flats.contains(&d) => mix(p.f_o, 0.35, p.panel2),
                None => p.panel2,
            };
            pa.rect_filled(r, CornerRadius::same(4), fill);
            if resp.hover_pos().is_some_and(|h| r.contains(h)) {
                tip = Some(format!(
                    "{} {}{}",
                    d_short(d),
                    d.year(),
                    match sec {
                        Some(x) => format!(" · {:.1} h lights", x / 3600.0),
                        None if flats.contains(&d) => " · flats".into(),
                        None => String::new(),
                    }
                ));
            }
        }
        if let Some(t) = tip {
            resp.on_hover_text(t);
        }
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            let sw = |ui: &mut egui::Ui, c: Color32| {
                let (r, _) = ui.allocate_exact_size(Vec2::splat(10.0), Sense::hover());
                ui.painter().rect_filled(r, CornerRadius::same(3), c);
            };
            sw(ui, p.accent);
            ui.label(RichText::new("lights").font(body(12.0)).color(p.muted));
            ui.add_space(8.0);
            sw(ui, mix(p.f_o, 0.35, p.panel2));
            ui.label(RichText::new("flats only").font(body(12.0)).color(p.muted));
            ui.add_space(8.0);
            ui.label(RichText::new(format!("{} – {}", start.format("%b"), end.format("%b %Y"))).font(body(12.0)).color(p.muted));
        });
    });
}

fn library_card(ui: &mut egui::Ui, a: &mut UiState, s: Option<&Session>) {
    let p = a.pal;
    w::card(ui, &p, |ui| {
        w::h2(ui, "Library");
        ui.add_space(4.0);
        // What it is, then where (long paths wrap instead of running into
        // the label).
        let row = |ui: &mut egui::Ui, left: String, right: String| {
            egui::Frame::new().fill(p.panel2).corner_radius(8).inner_margin(egui::Margin::symmetric(10, 8)).show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.label(RichText::new(right).font(body(12.0)).color(p.muted));
                ui.add(egui::Label::new(RichText::new(left).font(mono(12.0))).wrap());
            });
        };
        if let Some(s) = s {
            if let Some(parent) = crate::worker::library_parent(&s.root, &s.cfg) {
                let n = a.folders.len();
                row(ui, parent.display().to_string(), if n > 0 { format!("library folder · {n} folders") } else { "library folder · reading…".into() });
            }
            let lib = crux_core::library::Library::db_path(&s.cfg.library.dir);
            let sets = s.triage.library.as_ref().map(|l| l.sets.len());
            row(ui, lib.display().to_string(), match sets {
                Some(n) => format!("flat library · {n} sets"),
                None => "flat library · empty".into(),
            });
            if let Some(d) = &s.cache_dir {
                row(ui, d.display().to_string(), "cache and your decisions".into());
            }
            ui.add_space(4.0);
            crate::auto::refresh_ui(ui, a, &s.cfg, s.triage.library.as_ref());
        }
        ui.add_space(4.0);
        egui::Frame::new().stroke(Stroke::new(1.5, p.line)).corner_radius(12).inner_margin(14).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal_wrapped(|ui| {
                ui.vertical(|ui| {
                    ui.label(RichText::new("Another folder").font(semibold(13.0)));
                    ui.label(RichText::new("Drop a folder on this window, or pick one.").font(body(12.5)).color(p.muted));
                });
            });
            ui.add_space(6.0);
            if w::button(ui, &p, "Open folder", Kind::Primary, None).clicked() {
                pick_folder(a);
            }
        });
    });
}
