//! Inbox: only the decisions crux wasn't sure about.

use crate::app::{Show, Sort, UiState, View};
use crate::session::{Act, Session, Sev};
use crate::theme::semibold;
use crate::widgets::{self as w, Kind};
use eframe::egui::{self, Color32, CornerRadius, Pos2, Rect, RichText, ScrollArea, Sense, Stroke, StrokeKind, Vec2};

pub fn show(ui: &mut egui::Ui, a: &mut UiState, s: &mut Session) {
    let p = a.pal;
    let items = s.inbox();
    ScrollArea::vertical().id_salt("inbox").auto_shrink([false, false]).show(ui, |ui| {
        ui.set_max_width(980.0);
        w::h1(ui, "Inbox");
        ui.label(RichText::new("Frames and sets that crux couldn't grade with confidence. Choosing an action removes the item. Your choices are saved.").color(p.muted));
        ui.add_space(16.0);
        if items.is_empty() {
            egui::Frame::new().stroke(Stroke::new(1.5, p.line)).corner_radius(12).inner_margin(28).show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.vertical_centered(|ui| w::muted(ui, &p, "No frames or sets need checking."));
            });
        }
        let mut action: Option<(String, Act, String)> = None;
        let mut open_frame = None;
        for it in &items {
            let sev = match it.sev {
                Sev::Rej => p.rej,
                Sev::Warn => p.warn,
                Sev::Info => p.info,
                Sev::Good => p.good,
            };
            let bg = ui.painter().add(egui::Shape::Noop);
            let inner = egui::Frame::new().inner_margin(egui::Margin { left: 20, right: 16, top: 14, bottom: 14 }).show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.label(RichText::new(&it.title).font(semibold(16.0)));
                ui.add_space(2.0);
                ui.add(egui::Label::new(RichText::new(&it.body).color(p.muted)).wrap());
                if !it.evidence.is_empty() {
                    ui.add_space(4.0);
                    w::wrap_row(ui, |ui| {
                        for e in it.evidence.iter().take(12) {
                            w::tag(ui, &p, e);
                        }
                    });
                }
                if !it.thumbs.is_empty() {
                    ui.add_space(6.0);
                    w::wrap_row(ui, |ui| {
                        ui.spacing_mut().item_spacing = Vec2::splat(6.0);
                        for &li in it.thumbs.iter().take(14) {
                            let l = &s.lights[li];
                            let (r, resp) = ui.allocate_exact_size(Vec2::new(96.0, 64.0), Sense::click());
                            let pa = ui.painter();
                            pa.rect_filled(r, CornerRadius::same(6), p.well);
                            if let Some(t) = a.thumbs.get(&li) {
                                pa.add(egui::epaint::RectShape::filled(r, CornerRadius::same(6), Color32::WHITE).with_texture(t.id(), Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0))));
                            }
                            pa.rect_stroke(r, CornerRadius::same(6), Stroke::new(1.0, if resp.hovered() { p.faint } else { p.line }), StrokeKind::Inside);
                            let g = s.grade(li);
                            if g != crux_core::grade::Grade::Good {
                                pa.circle_filled(r.left_top() + Vec2::splat(8.0), 4.5, p.grade(g));
                            }
                            let resp = resp.on_hover_text(format!("{} {} · {}", l.obj, l.hhmm(), crate::app::grade_name(g)));
                            if resp.on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
                                open_frame = Some(li);
                            }
                        }
                        if it.thumbs.len() > 14 {
                            ui.label(RichText::new(format!("+{}", it.thumbs.len() - 14)).color(p.muted));
                        }
                    });
                }
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    for (label, primary, act) in &it.acts {
                        if w::button(ui, &p, label, if *primary { Kind::Primary } else { Kind::Normal }, None).clicked() {
                            action = Some((it.id.clone(), *act, label.clone()));
                        }
                    }
                });
            });
            let r = inner.response.rect;
            ui.painter().set(
                bg,
                egui::Shape::Vec(vec![
                    egui::Shape::rect_filled(r, CornerRadius::same(12), p.panel),
                    egui::Shape::rect_filled(Rect::from_min_size(r.min, Vec2::new(4.0, r.height())), CornerRadius { nw: 12, sw: 12, ne: 0, se: 0 }, sev),
                ]),
            );
            ui.painter().rect_stroke(r, CornerRadius::same(12), Stroke::new(1.0, p.line), StrokeKind::Inside);
            ui.add_space(10.0);
        }
        if let Some(li) = open_frame {
            a.fu.sel = Some(li);
            a.fu.night = Some(s.lights[li].night);
            a.fu.show = Show::All;
            a.fu.scroll_to_sel = true;
            a.view = View::Frames;
        }
        if let Some((id, act, label)) = action {
            match act {
                Act::Dismiss => {
                    s.decisions.dismissed.insert(id.clone());
                    s.save_decisions();
                    let msg = if label == "Acknowledge" { "Acknowledged.".to_string() } else { format!("Done: {}.", label.to_lowercase()) };
                    a.toast(ui.ctx(), msg, false);
                }
                Act::ReviewDisagree | Act::ReviewBorder => {
                    a.fu.show = if act == Act::ReviewDisagree { Show::Disagree } else { Show::Border };
                    a.fu.sort = Sort::Time;
                    a.fu.sel = None;
                    a.view = View::Frames;
                }
                Act::OpenCalibration => a.view = View::Calibrate,
            }
        }
    });
}
