//! Frames (replaces Blink): night chips, live reject limits, the night
//! timeline, a contact sheet worst-first or by panel, and the inspector with
//! the full-resolution viewer.

use crate::app::{FramesUi, Show, Sort, UiState};
use crate::chart;
use crate::session::{Session, LIMITS};
use crate::theme::{alpha, body, mono, semibold};
use crate::viewer::{self, Draw, Mode};
use crate::widgets::{self as w, d_short, fmt_h, Kind};
use crux_core::grade::Grade;
use eframe::egui::{self, Align2, Color32, CornerRadius, Pos2, Rect, RichText, ScrollArea, Sense, Stroke, StrokeKind, Vec2};
use eframe::egui_wgpu;

/// Lights shown with the current night, filter and sort.
pub fn visible(s: &Session, fu: &FramesUi) -> Vec<usize> {
    let n = s.lights.len();
    let mut v: Vec<usize> = match fu.show {
        Show::Disagree => (0..n).filter(|&i| s.disagrees(i)).collect(),
        Show::Border => (0..n).filter(|&i| s.borderline(i)).collect(),
        _ => (0..n).filter(|&i| fu.night.is_none_or(|nn| s.lights[i].night == nn)).collect(),
    };
    match fu.show {
        Show::Flagged => v.retain(|&i| s.grade(i) != Grade::Good),
        Show::Mine => v.retain(|&i| s.pinned(i).is_some()),
        _ => {}
    }
    if fu.sort == Sort::Worst {
        v.sort_by(|&a, &b| {
            s.grade(b).cmp(&s.grade(a)).then(s.lights[a].t.unwrap_or(1.0).total_cmp(&s.lights[b].t.unwrap_or(1.0)))
        });
    } else if fu.show == Show::All || fu.show == Show::Flagged || fu.show == Show::Mine {
        // By night, then panel, then time (contact-sheet groups).
        v.sort_by(|&a, &b| {
            let (x, y) = (&s.lights[a], &s.lights[b]);
            x.night.cmp(&y.night).then(x.obj.cmp(&y.obj)).then(x.time.cmp(&y.time))
        });
    }
    v
}

pub fn show(ui: &mut egui::Ui, a: &mut UiState, s: &mut Session) {
    let p = a.pal;
    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            w::h1(ui, "Frames");
            ui.label(
                RichText::new("Each frame's grade and the reasons for it. Drag a reject limit to regrade the frames as you drag. To select part of the night, drag across the chart.")
                    .color(p.muted),
            );
        });
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Max), |ui| {
            ui.spacing_mut().item_spacing.x = 5.0;
            // Right to left: "J K move · A R keep / reject · Space blink".
            for (keys, t) in [(&["Space"][..], "blink"), (&["A", "R"][..], "keep / reject"), (&["J", "K"][..], "move")] {
                ui.label(RichText::new(t).font(body(12.0)).color(p.muted));
                for k in keys.iter().rev() {
                    w::kbd(ui, &p, k);
                }
                ui.add_space(6.0);
            }
        });
    });
    ui.add_space(12.0);
    let list = visible(s, &a.fu);
    if a.fu.sel.is_none_or(|x| !list.contains(&x)) {
        a.fu.sel = list.iter().copied().find(|&i| s.grade(i) != Grade::Good).or(list.first().copied());
    }
    let avail = ui.available_size();
    let wide = avail.x >= 1000.0;
    if wide {
        let insp_w = 430.0;
        let left_w = avail.x - insp_w - 16.0;
        ui.horizontal_top(|ui| {
            ui.allocate_ui_with_layout(Vec2::new(left_w, avail.y), egui::Layout::top_down(egui::Align::Min), |ui| {
                ScrollArea::vertical().id_salt("fleft").auto_shrink([false, false]).show(ui, |ui| {
                    ui.set_width(left_w - 12.0);
                    left(ui, a, s, &list);
                    ui.add_space(40.0);
                });
            });
            ui.add_space(4.0);
            ui.allocate_ui_with_layout(Vec2::new(insp_w, avail.y), egui::Layout::top_down(egui::Align::Min), |ui| {
                ScrollArea::vertical().id_salt("insp").auto_shrink([false, false]).show(ui, |ui| {
                    ui.set_width(insp_w - 10.0);
                    w::card(ui, &p, |ui| inspector(ui, a, s));
                    ui.add_space(30.0);
                });
            });
        });
    } else {
        ScrollArea::vertical().id_salt("fall").auto_shrink([false, false]).show(ui, |ui| {
            left(ui, a, s, &list);
            ui.add_space(16.0);
            w::card(ui, &p, |ui| inspector(ui, a, s));
            ui.add_space(40.0);
        });
    }
}

fn left(ui: &mut egui::Ui, a: &mut UiState, s: &mut Session, list: &[usize]) {
    let p = a.pal;
    // Nights.
    w::wrap_row(ui, |ui| {
        if w::chip(ui, &p, "All nights", None, a.fu.night.is_none() && !matches!(a.fu.show, Show::Disagree | Show::Border)).clicked() {
            a.fu.night = None;
            a.fu.brush = None;
            if matches!(a.fu.show, Show::Disagree | Show::Border) {
                a.fu.show = Show::All;
            }
        }
        for &n in &s.nights.clone() {
            let count = s.lights.iter().filter(|l| l.night == n).count();
            let on = a.fu.night == Some(n) && !matches!(a.fu.show, Show::Disagree | Show::Border);
            if w::chip(ui, &p, &d_short(n), Some(count), on).clicked() {
                a.fu.night = Some(n);
                a.fu.brush = None;
                if matches!(a.fu.show, Show::Disagree | Show::Border) {
                    a.fu.show = Show::All;
                }
            }
        }
    });
    ui.add_space(6.0);
    w::wrap_row(ui, |ui| {
        let mut opts = vec![(Show::All, "All"), (Show::Flagged, "Flagged"), (Show::Mine, "Changed by me")];
        if a.fu.show == Show::Disagree {
            opts.push((Show::Disagree, "Disagreements"));
        }
        if a.fu.show == Show::Border {
            opts.push((Show::Border, "Borderline"));
        }
        if let Some(k) = w::seg(ui, &p, &opts, a.fu.show) {
            a.fu.show = k;
        }
        if let Some(k) = w::seg(ui, &p, &[(Sort::Time, "By time"), (Sort::Worst, "Worst first")], a.fu.sort) {
            a.fu.sort = k;
        }
        ui.label(RichText::new(format!("{} frames", list.len())).font(body(12.5)).color(p.muted));
    });
    ui.add_space(10.0);
    limits_card(ui, a, s);
    ui.add_space(14.0);
    w::card(ui, &p, |ui| chart::timeline_card(ui, a, s));
    ui.add_space(14.0);
    sheet(ui, a, s, list);
}

fn limits_card(ui: &mut egui::Ui, a: &mut UiState, s: &mut Session) {
    let p = a.pal;
    w::card(ui, &p, |ui| {
        ui.horizontal(|ui| {
            w::h2(ui, "Reject limits");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let changed = (0..4).any(|i| (s.limits[i] - s.defaults[i]).abs() > 1e-9);
                if w::button_enabled(ui, &p, "Reset to defaults", Kind::Ghost, None, changed).clicked() {
                    s.limits = s.defaults;
                    s.regrade();
                    s.save_limits();
                }
                if s.regrade_ms > 0.0 {
                    ui.label(RichText::new(format!("regrade {:.0} ms", s.regrade_ms)).font(body(11.0)).color(p.faint));
                }
            });
        });
        ui.add_space(8.0);
        let cols = if ui.available_width() > 720.0 { 4 } else { 2 };
        let mut regrade = false;
        let mut save = false;
        egui::Grid::new("limits").num_columns(cols).spacing([18.0, 12.0]).min_col_width((ui.available_width() - 18.0 * (cols as f32 - 1.0)) / cols as f32).show(ui, |ui| {
            for (i, l) in LIMITS.iter().enumerate() {
                ui.vertical(|ui| {
                    let wd = ui.available_width().min((ui.max_rect().width()).max(120.0));
                    ui.set_width(wd);
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(l.label).font(body(12.5)).color(p.muted));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let dirty = (s.limits[i] - s.defaults[i]).abs() > 1e-9;
                            let v = format!("{:.*}{}{}", l.decimals, s.limits[i], l.unit, if dirty { " •" } else { "" });
                            ui.label(RichText::new(v).font(mono(13.0)).color(p.fg));
                        });
                    });
                    ui.spacing_mut().slider_width = wd - 4.0;
                    let mut v = s.limits[i];
                    let r = ui.add(egui::Slider::new(&mut v, if l.strict_low { l.max..=l.min } else { l.min..=l.max }).step_by(l.step).show_value(false));
                    if r.changed() {
                        s.limits[i] = v;
                        regrade = true;
                    }
                    if r.drag_stopped() || (r.changed() && !r.dragged()) {
                        save = true;
                    }
                });
                if (i + 1) % cols == 0 {
                    ui.end_row();
                }
            }
        });
        if regrade {
            s.regrade();
        }
        if save {
            s.save_limits();
        }
        ui.add_space(8.0);
        let (r, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 1.0), Sense::hover());
        ui.painter().rect_filled(r, CornerRadius::ZERO, p.line);
        ui.add_space(8.0);
        let scope: Vec<usize> = (0..s.lights.len()).filter(|&i| a.fu.night.is_none_or(|n| s.lights[i].night == n)).collect();
        let rej: Vec<usize> = scope.iter().copied().filter(|&i| s.grade(i) == Grade::Reject).collect();
        let sec_r: f64 = rej.iter().map(|&i| s.lights[i].exp).sum();
        let sec_t: f64 = scope.iter().map(|&i| s.lights[i].exp).sum();
        let differ = scope.iter().filter(|&&i| (s.lights[i].auto == Grade::Reject) != (s.auto0[i] == Grade::Reject)).count();
        w::wrap_row(ui, |ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            ui.label("Rejecting");
            ui.label(RichText::new(rej.len().to_string()).font(mono(13.5)).color(p.rej));
            ui.label(format!("of {} frames", scope.len()));
            ui.add_space(10.0);
            ui.label(RichText::new(fmt_h(sec_r)).font(mono(13.5)));
            ui.label(format!("of {}", fmt_h(sec_t)));
            ui.add_space(10.0);
            ui.label("Keeping");
            ui.label(RichText::new(fmt_h(sec_t - sec_r)).font(mono(13.5)));
            let snr = if sec_t > 0.0 { ((sec_t - sec_r) / sec_t).sqrt() * 100.0 } else { 0.0 };
            ui.label(format!("(SNR {snr:.0}% of all frames)"));
            if differ > 0 {
                ui.add_space(10.0);
                ui.label(RichText::new(format!("{differ} differ from crux's defaults")).color(p.muted));
            }
        });
    });
}

fn sheet(ui: &mut egui::Ui, a: &mut UiState, s: &mut Session, list: &[usize]) {
    let p = a.pal;
    let flat = a.fu.sort == Sort::Worst || matches!(a.fu.show, Show::Disagree | Show::Border);
    let mut groups: Vec<(Option<(chrono::NaiveDate, String)>, Vec<usize>)> = Vec::new();
    if flat {
        groups.push((None, list.to_vec()));
    } else {
        for &i in list {
            let k = (s.lights[i].night, s.lights[i].obj.clone());
            match groups.last_mut() {
                Some((Some(g), v)) if *g == k => v.push(i),
                _ => groups.push((Some(k), vec![i])),
            }
        }
    }
    if list.is_empty() {
        egui::Frame::new().stroke(Stroke::new(1.5, p.line)).corner_radius(12).inner_margin(28).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.vertical_centered(|ui| w::muted(ui, &p, "No frames match."));
        });
        return;
    }
    let tw = 120.0;
    for (k, ids) in groups {
        if let Some((night, obj)) = &k {
            let c = s.counts(ids.iter().copied());
            let l0 = &s.lights[ids[0]];
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 8.0;
                let name = l0.panel.map(|n| format!("Panel {n}")).unwrap_or_else(|| obj.clone());
                ui.label(RichText::new(name).font(semibold(12.5)).color(p.fg));
                ui.label(RichText::new(d_short(*night)).font(body(12.5)).color(p.muted));
                w::filter_badge(ui, &p, if l0.filter.is_empty() { "?" } else { &l0.filter });
                ui.label(RichText::new(format!("{} × {:.0} s", ids.len(), l0.exp)).font(body(12.5)).color(p.muted));
                if c[2] > 0 {
                    ui.label(RichText::new(format!("{} reject", c[2])).font(body(12.5)).color(p.rej));
                }
                if c[1] > 0 {
                    ui.label(RichText::new(format!("{} warn", c[1])).font(body(12.5)).color(p.warn));
                }
            });
            ui.add_space(2.0);
        }
        w::wrap_row(ui, |ui| {
            ui.spacing_mut().item_spacing = Vec2::splat(6.0);
            for &i in &ids {
                tile(ui, a, s, i, tw, k.is_none());
            }
        });
        ui.add_space(12.0);
    }
}

fn tile(ui: &mut egui::Ui, a: &mut UiState, s: &Session, i: usize, tw: f32, long_label: bool) {
    let p = a.pal;
    let l = &s.lights[i];
    let aspect = if l.width > 0 { l.height as f32 / l.width as f32 } else { 2.0 / 3.0 };
    let (r, resp) = ui.allocate_exact_size(Vec2::new(tw, (tw * aspect).round()), Sense::click());
    let g = s.grade(i);
    let sel = a.fu.sel == Some(i);
    if sel && a.fu.scroll_to_sel {
        ui.scroll_to_rect(r, None);
        a.fu.scroll_to_sel = false;
    }
    if ui.is_rect_visible(r) {
        let pa = ui.painter();
        pa.rect_filled(r, CornerRadius::same(6), p.well);
        if let Some(t) = a.thumbs.get(&i) {
            pa.add(egui::epaint::RectShape::filled(r, CornerRadius::same(6), Color32::WHITE).with_texture(t.id(), Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0))));
        }
        if g == Grade::Reject {
            pa.rect(r, CornerRadius::same(6), Color32::from_rgba_unmultiplied(255, 60, 80, 46), Stroke::new(1.5, alpha(p.rej, 0.8)), StrokeKind::Inside);
        }
        let outline = if sel { Stroke::new(2.0, p.accent) } else if resp.hovered() { Stroke::new(1.0, p.faint) } else { Stroke::new(1.0, p.line) };
        pa.rect_stroke(if sel { r.expand(2.0) } else { r }, CornerRadius::same(if sel { 8 } else { 6 }), outline, StrokeKind::Middle);
        if g != Grade::Good {
            let c = Pos2::new(r.left() + 8.5, r.top() + 8.5);
            pa.circle_filled(c, 6.5, Color32::from_black_alpha(150));
            pa.circle_filled(c, 4.5, p.grade(g));
        }
        if s.pinned(i).is_some() {
            pin(pa, Pos2::new(r.right() - 9.0, r.top() + 9.0), p.fg);
        }
        let label = if long_label { format!("{} {}", l.panel.map(|n| format!("P{n}")).unwrap_or_default(), d_short(l.night)) } else { l.hhmm() };
        let pos = Pos2::new(r.left() + 5.0, r.bottom() - 3.0);
        pa.text(pos + Vec2::new(1.0, 1.0), Align2::LEFT_BOTTOM, &label, mono(10.0), Color32::BLACK);
        pa.text(pos, Align2::LEFT_BOTTOM, &label, mono(10.0), Color32::from_rgb(0xcf, 0xd6, 0xe4));
    }
    let resp = resp.on_hover_text(format!("{} · {} · {}", l.obj, l.hhmm(), crate::app::grade_name(g)));
    if resp.clicked() {
        a.fu.sel = Some(i);
        if a.fu.mode == Mode::Reference {
            a.fu.mode = Mode::Frame;
        }
    }
}

/// A small pin marking your override.
pub fn pin(pa: &egui::Painter, c: Pos2, col: Color32) {
    pa.circle_filled(c + Vec2::new(0.0, -2.0), 3.6, Color32::from_black_alpha(160));
    pa.circle_filled(c + Vec2::new(0.0, -2.0), 2.6, col);
    pa.line_segment([c + Vec2::new(0.0, 0.5), c + Vec2::new(0.0, 5.0)], Stroke::new(1.6, col));
}

fn inspector(ui: &mut egui::Ui, a: &mut UiState, s: &mut Session) {
    let p = a.pal;
    let Some(i) = a.fu.sel else {
        w::muted(ui, &p, "Select a frame.");
        return;
    };
    let l = s.lights[i].clone();
    let g = s.grade(i);
    let pinned = s.pinned(i).is_some();
    ui.horizontal_top(|ui| {
        ui.vertical(|ui| {
            ui.set_max_width(ui.available_width() - 110.0);
            ui.label(RichText::new(format!("{} · {}", l.obj, l.hhmm())).font(semibold(16.0)));
            ui.add(egui::Label::new(RichText::new(&l.rel).font(mono(11.5)).color(p.muted)).wrap());
        });
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| w::pill(ui, &p, g, pinned));
    });
    ui.add_space(8.0);
    well(ui, a, s, i);
    ui.add_space(8.0);
    let reference = s.reference_of(i);
    let res_ok = a.gpu.is_some();
    w::wrap_row(ui, |ui| {
        ui.spacing_mut().item_spacing = Vec2::splat(6.0);
        if w::tool(ui, &p, "Blink with reference", "Space", a.fu.mode == Mode::Reference, reference.is_some() && res_ok).clicked() {
            a.fu.mode = if a.fu.mode == Mode::Reference { Mode::Frame } else { Mode::Reference };
        }
        if w::tool(ui, &p, "Difference", "D", a.fu.mode == Mode::Difference, reference.is_some() && res_ok).clicked() {
            a.fu.mode = if a.fu.mode == Mode::Difference { Mode::Frame } else { Mode::Difference };
        }
        if w::tool(ui, &p, "Stars + trails", "1", a.fu.marks, s.triage.metrics.contains_key(&l.fi)).clicked() {
            a.fu.marks = !a.fu.marks;
        }
        if w::tool(ui, &p, "Cloud grid", "2", a.fu.grid, l.cells.is_some()).clicked() {
            a.fu.grid = !a.fu.grid;
        }
    });
    ui.add_space(10.0);
    // Reasons.
    if l.flags.is_empty() {
        reason_row(ui, &p, "All clear", p.good, "Sharp, round stars, clear sky, no trails.");
    }
    for f in &l.flags {
        let c = match f.grade {
            Grade::Reject => p.rej,
            Grade::Warn => p.warn,
            Grade::Good => p.muted,
        };
        reason_row(ui, &p, f.check.name(), c, &f.reason);
    }
    if l.bad {
        let how = if l.bad_name { "Named BAD_" } else { "In a reject folder" };
        let text = if g == Grade::Good {
            format!("{how}, but crux grades it good, so it's stacked. To leave it out, press R.")
        } else {
            format!("{how}, so it's left out of the stack.")
        };
        reason_row(ui, &p, "Filed", p.accent, &text);
    }
    if pinned {
        let crux = crate::app::grade_name(l.auto);
        reason_row(ui, &p, "Yours", p.accent, &format!("pinned {} (crux says {crux}); kept across rescans and used when moving files", crate::app::grade_name(g).to_lowercase()));
    }
    if !l.notes.is_empty() {
        ui.add(egui::Label::new(RichText::new(l.notes.join(" · ")).font(body(12.0)).color(p.muted)).wrap());
    }
    ui.add_space(8.0);
    metrics_table(ui, &p, &l);
    ui.add_space(10.0);
    chart::why(ui, &p, s, i);
    ui.add_space(10.0);
    let mut act = None;
    ui.horizontal(|ui| {
        if w::button(ui, &p, "Keep", Kind::Normal, Some("A")).clicked() {
            act = Some(Some(Grade::Good));
        }
        if w::button(ui, &p, "Reject", Kind::Danger, Some("R")).clicked() {
            act = Some(Some(Grade::Reject));
        }
        if pinned && w::button(ui, &p, &format!("Use crux's grade ({})", crate::app::grade_name(l.auto)), Kind::Ghost, None).clicked() {
            act = Some(None);
        }
    });
    if let Some(gr) = act {
        s.set_override(&[i], gr);
        let msg = match gr {
            Some(Grade::Reject) => format!("{} marked reject", l.hhmm()),
            Some(_) => format!("{} marked keep", l.hhmm()),
            None => "Back to crux's grade".into(),
        };
        a.toast(ui.ctx(), msg, true);
    }
}

fn reason_row(ui: &mut egui::Ui, p: &crate::theme::Pal, cat: &str, c: Color32, text: &str) {
    ui.horizontal_top(|ui| {
        let (r, _) = ui.allocate_exact_size(Vec2::new(90.0, 18.0), Sense::hover());
        ui.painter().text(r.left_top() + Vec2::new(0.0, 2.0), Align2::LEFT_TOP, cat.to_uppercase(), semibold(11.0), c);
        ui.add(egui::Label::new(RichText::new(text).font(body(13.0)).color(p.fg)).wrap());
    });
    ui.add_space(2.0);
}

fn metrics_table(ui: &mut egui::Ui, p: &crate::theme::Pal, l: &crate::session::Light) {
    let f2 = |v: Option<f64>| v.map(|x| format!("{x:.2}"));
    let cells: [(&str, Option<String>); 9] = [
        ("Transparency", f2(l.t)),
        ("FWHM", l.fwhm.map(|x| format!("{x:.2} px"))),
        ("Eccentricity", f2(l.ecc)),
        ("Stars", l.stars.map(|x| group_thousands(x as u64))),
        ("Sky e-/px/s", l.sky.map(|x| format!("{x:.3}{}", l.skyr.map(|r| format!(" ({r:.2}×)")).unwrap_or_default()))),
        ("Gradient", l.grad.map(|x| format!("{x:.1}%"))),
        ("Altitude", l.talt.map(|x| format!("{x:.1}°"))),
        ("Moon", l.malt.map(|x| if x > 0.0 { format!("{x:.1}° up") } else { "below horizon".into() })),
        ("Pier side", l.pier.clone()),
    ];
    let wd = ui.available_width();
    let (r, _) = ui.allocate_exact_size(Vec2::new(wd, 3.0 * 46.0), Sense::hover());
    let pa = ui.painter();
    pa.rect(r, CornerRadius::same(8), p.line, Stroke::new(1.0, p.line), StrokeKind::Inside);
    let cw = (wd - 2.0) / 3.0;
    for (k, (label, v)) in cells.iter().enumerate() {
        let (row, col) = (k / 3, k % 3);
        let c = Rect::from_min_size(Pos2::new(r.left() + 1.0 + col as f32 * cw, r.top() + 1.0 + row as f32 * 45.3), Vec2::new(cw - 1.0, 44.3));
        let cr = match k {
            0 => CornerRadius { nw: 7, ..Default::default() },
            2 => CornerRadius { ne: 7, ..Default::default() },
            6 => CornerRadius { sw: 7, ..Default::default() },
            8 => CornerRadius { se: 7, ..Default::default() },
            _ => CornerRadius::ZERO,
        };
        pa.rect_filled(c, cr, p.panel);
        pa.text(c.left_top() + Vec2::new(9.0, 6.0), Align2::LEFT_TOP, *label, body(11.0), p.muted);
        pa.text(c.left_top() + Vec2::new(9.0, 22.0), Align2::LEFT_TOP, v.clone().unwrap_or_else(|| "—".into()), mono(13.0), p.fg);
    }
}

pub fn group_thousands(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// The image well: full-resolution viewer with pan, zoom and overlays.
fn well(ui: &mut egui::Ui, a: &mut UiState, s: &Session, i: usize) {
    let p = a.pal;
    let l = &s.lights[i];
    let (iw, ih) = (l.width.max(1) as f32, l.height.max(1) as f32);
    let wd = ui.available_width();
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(wd, (wd * ih / iw).round()), Sense::click_and_drag());
    let pa = ui.painter_at(rect);
    pa.rect_filled(rect, CornerRadius::same(10), p.well);
    let reference = s.reference_of(i);
    let mode = if reference.is_some() { a.fu.mode } else { Mode::Frame };
    let showing = if mode == Mode::Reference { reference.unwrap_or(i) } else { i };
    // Zoom and pan (shared across frames).
    let (min, _max, scale) = a.fu.view.corners(rect, iw, ih);
    if resp.hovered() {
        let (scroll, hover) = ui.ctx().input(|inp| (inp.smooth_scroll_delta.y, inp.pointer.hover_pos()));
        if scroll.abs() > 0.0 {
            if let Some(hp) = hover {
                let f = (scroll / 200.0).exp();
                let fit = viewer::View::fit_scale(rect, iw, ih);
                let ns = (scale * f).clamp(fit, 8.0);
                let img = egui::vec2(min[0] + (hp.x - rect.left()) / scale, min[1] + (hp.y - rect.top()) / scale);
                let centre = img + (rect.center() - hp) / ns;
                a.fu.view = if (ns - fit).abs() < 1e-4 { viewer::View::default() } else { viewer::View { scale: Some(ns), centre } };
            }
        }
    }
    if resp.dragged() {
        let d = resp.drag_delta();
        let (c, sc) = match a.fu.view.scale {
            Some(sc) => (a.fu.view.centre, sc),
            None => (egui::vec2(iw / 2.0, ih / 2.0), scale),
        };
        if a.fu.view.scale.is_some() {
            a.fu.view.centre = c - d / sc;
        }
    }
    if resp.double_clicked() {
        a.fu.view = match a.fu.view.scale {
            Some(_) => viewer::View::default(),
            None => {
                let hp = resp.interact_pointer_pos().unwrap_or(rect.center());
                let img = egui::vec2(min[0] + (hp.x - rect.left()) / scale, min[1] + (hp.y - rect.top()) / scale);
                viewer::View { scale: Some(1.0 / ui.ctx().pixels_per_point()), centre: img }
            }
        };
    }
    let (min, max, scale) = a.fu.view.corners(rect, iw, ih);
    let fr = a.resident.get(&i).copied();
    let rf = reference.and_then(|r| a.resident.get(&r).copied());
    let ready = fr.is_some() && (mode == Mode::Frame || rf.is_some());
    if let (true, Some(fr), Some(_)) = (ready, fr, a.gpu.as_ref()) {
        let k = match (l.t, reference.and_then(|r| s.lights[r].t)) {
            (Some(t), Some(rt)) if rt > 0.0 => (t / rt) as f32,
            _ => 1.0,
        };
        let bg = p.well.to_normalized_gamma_f32();
        let draw = Draw {
            frame: i,
            reference,
            img_min: min,
            img_max: max,
            size: [iw, ih],
            fr,
            rf,
            to_ref: l.to_ref,
            mode,
            bg,
            k,
        };
        pa.add(egui_wgpu::Callback::new_paint_callback(rect, draw));
    } else if let Some(t) = a.thumbs.get(&showing) {
        // The thumbnail stands in until the full frame arrives.
        let a0 = viewer::View::to_screen(rect, min, scale, 0.0, 0.0);
        let a1 = viewer::View::to_screen(rect, min, scale, iw, ih);
        pa.image(t.id(), Rect::from_min_max(a0, a1), Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
    }
    // Overlays (frame coordinates; the reference is shown in this frame's geometry).
    let to_s = |x: f32, y: f32| viewer::View::to_screen(rect, min, scale, x + 0.5, y + 0.5);
    if a.fu.marks && mode == Mode::Frame {
        if let Some(m) = s.triage.metrics.get(&l.fi) {
            let st = Stroke::new(1.2, alpha(p.good, 0.85));
            for star in &m.stars {
                let c = to_s(star.x, star.y);
                if !rect.expand(20.0).contains(c) {
                    continue;
                }
                let q = (1.0 - (star.ecc as f64).powi(2)).max(0.05).sqrt() as f32;
                let rad = (1.6 * star.fwhm * scale).max(3.0);
                let (ra, rb) = (rad / q.sqrt(), rad * q.sqrt());
                let th = star.pa.to_radians();
                let pts: Vec<Pos2> = (0..20)
                    .map(|k| {
                        let t = k as f32 / 20.0 * std::f32::consts::TAU;
                        let (x, y) = (ra * t.cos(), rb * t.sin());
                        c + Vec2::new(x * th.cos() - y * th.sin(), x * th.sin() + y * th.cos())
                    })
                    .collect();
                pa.add(egui::Shape::closed_line(pts, st));
            }
            for t in &m.trails {
                let (a0, a1) = (to_s(t.x0, t.y0), to_s(t.x1, t.y1));
                pa.line_segment([a0, a1], Stroke::new(2.5, alpha(p.rej, 0.9)));
                let noise = (m.noise_adu as f64).max(1e-6);
                let label = format!("{} {:.1}× noise", t.kind.name(), t.brightness_adu as f64 / noise);
                let mid = a0 + (a1 - a0) * 0.5;
                pa.text(mid + Vec2::new(6.0, -6.0), Align2::LEFT_BOTTOM, label, semibold(11.0), p.rej);
            }
        }
    }
    if a.fu.grid && mode == Mode::Frame {
        if let Some(cells) = &l.cells {
            let (cw, chh) = (iw / 4.0, ih / 4.0);
            for (k, c) in cells.iter().enumerate() {
                let (cx, cy) = ((k % 4) as f32 * cw, (k / 4) as f32 * chh);
                let r = Rect::from_min_max(viewer::View::to_screen(rect, min, scale, cx, cy), viewer::View::to_screen(rect, min, scale, cx + cw, cy + chh));
                let fill = match c {
                    None => Color32::from_rgba_unmultiplied(120, 120, 120, 90),
                    Some(v) if *v < s.limits[0] => Color32::from_rgba_unmultiplied(255, 60, 80, 90),
                    Some(v) if *v < 0.75 => Color32::from_rgba_unmultiplied(242, 193, 78, 72),
                    _ => Color32::TRANSPARENT,
                };
                pa.rect(r, CornerRadius::ZERO, fill, Stroke::new(1.0, Color32::from_white_alpha(46)), StrokeKind::Inside);
                let t = c.map(|v| format!("{v:.2}")).unwrap_or_else(|| "—".into());
                pa.text(r.center() + Vec2::new(1.0, 1.0), Align2::CENTER_CENTER, &t, semibold(12.0), Color32::BLACK);
                pa.text(r.center(), Align2::CENTER_CENTER, &t, semibold(12.0), Color32::WHITE);
            }
        }
    }
    // Tag, top left.
    let tag = match mode {
        Mode::Reference => {
            let rl = &s.lights[showing];
            format!("REFERENCE · {} {} · T {}", d_short(rl.night), rl.hhmm(), rl.t.map(|t| format!("{t:.2}")).unwrap_or_else(|| "—".into()))
        }
        Mode::Difference => "FRAME − REFERENCE".into(),
        Mode::Frame => format!("{} {}", d_short(l.night), l.hhmm()),
    };
    let tg = pa.layout_no_wrap(tag, mono(11.0), if mode == Mode::Reference { p.accent_ink } else { Color32::WHITE });
    let tr = Rect::from_min_size(rect.left_top() + Vec2::new(8.0, 8.0), tg.size() + Vec2::new(12.0, 4.0));
    pa.rect_filled(tr, CornerRadius::same(4), if mode == Mode::Reference { p.accent } else { Color32::from_black_alpha(153) });
    pa.galley(tr.left_top() + Vec2::new(6.0, 2.0), tg, Color32::WHITE);
    // Note, bottom right.
    let note = if let Some(e) = a.load_errors.get(&i) {
        format!("couldn't read: {e}")
    } else if !ready {
        if a.gpu.is_some() { "loading full resolution…".into() } else { "GPU viewer unavailable: thumbnail".into() }
    } else {
        let z = scale * ui.ctx().pixels_per_point();
        format!("{} · {}×{} · {:.0}%", if a.fu.view.scale.is_none() { "fit" } else { "zoom" }, l.width, l.height, z * 100.0)
    };
    let ng = pa.layout_no_wrap(note, body(11.0), Color32::from_rgb(0x9a, 0xa4, 0xb8));
    let nr = Rect::from_min_size(rect.right_bottom() - ng.size() - Vec2::new(20.0, 12.0), ng.size() + Vec2::new(12.0, 4.0));
    pa.rect_filled(nr, CornerRadius::same(4), Color32::from_black_alpha(140));
    pa.galley(nr.left_top() + Vec2::new(6.0, 2.0), ng, Color32::WHITE);
    if a.fu.view.scale.is_some() {
        ui.ctx().set_cursor_icon(if resp.dragged() { egui::CursorIcon::Grabbing } else if resp.hovered() { egui::CursorIcon::Grab } else { egui::CursorIcon::Default });
    }
}
