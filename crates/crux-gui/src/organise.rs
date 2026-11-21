//! Organise: browse the whole library any way you like without moving a
//! thing, design a layout once, see exactly what would move, apply it with
//! a journal, undo any apply, and keep the library healthy.

use crate::app::UiState;
use crate::session::Session;
use crate::theme::{alpha, body, mix, mono, semibold, Pal};
use crate::widgets::{self as w, d_short, Kind};
use crux_core::config::Layout;
use crux_core::grade::Grade;
use crux_core::health::{FolderSpace, Part};
use crux_core::index::IndexFile;
use crux_core::plan::{MoveKind, Plan};
use crux_core::{FrameType, Role};
use eframe::egui::{self, Align2, Color32, CornerRadius, Pos2, Rect, RichText, ScrollArea, Sense, Stroke, StrokeKind, Vec2};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tab {
    Browse,
    Layout,
    Health,
    History,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Dim {
    Type,
    Target,
    Panel,
    Night,
    Filter,
    Grade,
    Folder,
}

impl Dim {
    const ALL: [Dim; 7] = [Dim::Type, Dim::Target, Dim::Panel, Dim::Night, Dim::Filter, Dim::Grade, Dim::Folder];
    fn name(self) -> &'static str {
        match self {
            Dim::Type => "Type",
            Dim::Target => "Target",
            Dim::Panel => "Panel",
            Dim::Night => "Night",
            Dim::Filter => "Filter",
            Dim::Grade => "Grade",
            Dim::Folder => "Current folder",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Rule {
    Light,
    Reject,
    Flat,
    Retired,
    Dark,
    Bias,
}

impl Rule {
    const ALL: [Rule; 6] = [Rule::Light, Rule::Reject, Rule::Flat, Rule::Retired, Rule::Dark, Rule::Bias];
    fn name(self) -> &'static str {
        match self {
            Rule::Light => "Kept lights",
            Rule::Reject => "Rejected lights",
            Rule::Flat => "Flats",
            Rule::Retired => "Retired flats",
            Rule::Dark => "Darks",
            Rule::Bias => "Bias",
        }
    }
    fn get(self, l: &Layout) -> &str {
        match self {
            Rule::Light => &l.light,
            Rule::Reject => &l.reject,
            Rule::Flat => &l.flat,
            Rule::Retired => &l.flat_retired,
            Rule::Dark => &l.dark,
            Rule::Bias => &l.bias,
        }
    }
    fn set(self, l: &mut Layout, v: String) {
        match self {
            Rule::Light => l.light = v,
            Rule::Reject => l.reject = v,
            Rule::Flat => l.flat = v,
            Rule::Retired => l.flat_retired = v,
            Rule::Dark => l.dark = v,
            Rule::Bias => l.bias = v,
        }
    }
}

/// Layout presets: your current layout first.
fn presets() -> Vec<(&'static str, Layout)> {
    let l = |light: &str, reject: &str, flat: &str, retired: &str, dark: &str, bias: &str| Layout {
        light: light.into(),
        reject: reject.into(),
        flat: flat.into(),
        flat_retired: retired.into(),
        dark: dark.into(),
        bias: bias.into(),
        master: "master".into(),
    };
    vec![
        ("crux default (LIGHT\\PANEL)", Layout::default()),
        ("By night then panel", l("{night}/{panel}", "{night}/BAD/{reason}", "FLAT/{night}", "FLAT.bad/{night}", "dark", "bias")),
        ("By filter", l("LIGHT/{filter}/{panel}", "BAD/{filter}/{reason}", "FLAT/{filter}/{night}", "FLAT.bad/{filter}/{night}", "dark", "bias")),
        ("WBPP-friendly", l("LIGHT/{filter}", "BAD/{reason}", "FLAT/{filter}", "FLAT.bad", "DARK", "BIAS")),
        ("Siril-friendly", l("lights", "rejected/{reason}", "flats", "flats_retired", "darks", "biases")),
    ]
}

pub struct OrgUi {
    pub tab: Tab,
    dims: Vec<Dim>,
    pub query: String,
    node: Option<Vec<String>>,
    open: HashSet<String>,
    selected: BTreeSet<PathBuf>,
    pub layout: Option<Layout>,
    last_rule: Rule,
    literal: String,
    plan: Option<(u64, Layout, Plan)>,
    pub history: Vec<crux_core::moves::HistoryEntry>,
    history_loaded: bool,
    pub dups: Option<Vec<crux_core::health::Duplicate>>,
    pub dup_progress: Option<(usize, usize)>,
    /// crux's own caches (measured when the Health tab opens).
    caches: Option<Vec<crux_core::cachedir::CacheFolder>>,
    /// A destructive cache action waiting for its second click.
    armed: Option<PathBuf>,
    cache_note: Option<String>,
}

impl Default for OrgUi {
    fn default() -> Self {
        OrgUi {
            tab: Tab::Browse,
            dims: vec![Dim::Target, Dim::Panel],
            query: String::new(),
            node: None,
            open: HashSet::new(),
            selected: BTreeSet::new(),
            layout: None,
            last_rule: Rule::Light,
            literal: String::new(),
            plan: None,
            history: Vec::new(),
            history_loaded: false,
            dups: None,
            dup_progress: None,
            caches: None,
            armed: None,
            cache_note: None,
        }
    }
}

impl OrgUi {
    /// Pick a layout preset by position (for the screenshot harness).
    pub fn set_preset(&mut self, n: usize) {
        if let Some((_, l)) = presets().into_iter().nth(n) {
            self.layout = Some(l);
            self.plan = None;
        }
    }

    /// Forget cached plans and history (after an apply or undo).
    pub fn invalidate(&mut self) {
        self.plan = None;
        self.history_loaded = false;
        self.selected.clear();
    }
}

/// What a view asks the app to do (it owns the worker channel and modal).
pub enum Request {
    Apply { plan: Plan, title: String },
    Undo(crux_core::moves::HistoryEntry),
    Export { files: Vec<PathBuf>, dest: PathBuf },
    HashDuplicates(Vec<Vec<IndexFile>>),
}

fn gb(b: u64) -> String {
    if b >= 1_000_000_000 {
        format!("{:.1} GB", b as f64 / 1e9)
    } else {
        format!("{:.0} MB", b as f64 / 1e6)
    }
}

/// The library as the views see it: the index, with this folder's live
/// grades (your limits and pins) and panel corrections laid over it.
fn records(a: &UiState, s: &Session) -> Vec<IndexFile> {
    let root = crux_core::cache::key(&s.root);
    let mut out: Vec<IndexFile> = if a.library.is_empty() {
        let reports = s.live_reports();
        crux_core::index::records(&s.triage, Some(&reports))
    } else {
        a.library.clone()
    };
    for f in out.iter_mut() {
        if crux_core::cache::key(&f.root) != root {
            if let Some(n) = s.decisions.panel(&f.path) {
                f.panel = Some(n);
            }
            if let Some(g) = s.decisions.get(&f.path) {
                f.grade = Some(g);
            }
            continue;
        }
        if let Some(i) = s.by_path(&f.path) {
            let l = &s.lights[i];
            f.grade = Some(s.grade(i));
            f.panel = l.panel;
            f.t = l.t;
            f.sky = l.sky;
            f.alt = l.talt;
            f.reason = l.flags.iter().find(|x| x.grade == Grade::Reject).map(|x| x.check.name().to_string());
        }
    }
    out
}

fn dim_val(f: &IndexFile, d: Dim, parent: &Path) -> String {
    match d {
        Dim::Type => match (f.role, f.kind) {
            (Role::Master, k) => format!("Master {}", k.name().to_lowercase()),
            (Role::Processed, _) => "Processed".into(),
            (_, FrameType::Light) => "Lights".into(),
            (_, FrameType::Flat) => "Flats".into(),
            (_, FrameType::Dark | FrameType::DarkFlat) => "Darks".into(),
            (_, FrameType::Bias) => "Bias".into(),
            _ => "Unknown".into(),
        },
        Dim::Target => if f.kind == FrameType::Light { f.target.clone().unwrap_or_else(|| "Unknown".into()) } else { "Calibration".into() },
        Dim::Panel => match f.panel {
            Some(p) => format!("Panel {p:02}"),
            None if f.kind == FrameType::Light => "No panel".into(),
            None => dim_val(f, Dim::Type, parent),
        },
        Dim::Night => f.night.map(|n| n.to_string()).unwrap_or_else(|| "—".into()),
        Dim::Filter => f.filter.clone().unwrap_or_else(|| "—".into()),
        Dim::Grade => match f.grade {
            Some(g) => crate::app::grade_name(g).into(),
            None => "Not graded".into(),
        },
        Dim::Folder => f.path.parent().map(|p| p.strip_prefix(parent).unwrap_or(p).display().to_string()).unwrap_or_default(),
    }
}

pub fn show(ui: &mut egui::Ui, a: &mut UiState, s: &mut Session) {
    let p = a.pal;
    ScrollArea::vertical().id_salt("organise").auto_shrink([false, false]).show(ui, |ui| {
        ui.set_width(ui.available_width() - 8.0);
        w::h1(ui, "Organise");
        ui.label(
            RichText::new("Browse your files in any grouping without moving them. Set up a folder layout, then review every move before you apply it.")
                .color(p.muted),
        );
        ui.add_space(12.0);
        let hist = if a.org.history.is_empty() { "History".to_string() } else { format!("History · {}", a.org.history.len()) };
        if let Some(t) = w::seg(ui, &p, &[(Tab::Browse, "Browse"), (Tab::Layout, "Layout and plan"), (Tab::Health, "Library health"), (Tab::History, hist.as_str())], a.org.tab) {
            a.org.tab = t;
        }
        if !a.org.history_loaded {
            a.org.history = s.cache_dir.as_ref().map(|d| crux_core::moves::history(&d.join("plans"))).unwrap_or_default();
            a.org.history_loaded = true;
        }
        ui.add_space(14.0);
        match a.org.tab {
            Tab::Browse => browse(ui, a, s),
            Tab::Layout => layout(ui, a, s),
            Tab::Health => health(ui, a, s),
            Tab::History => history(ui, a, s),
        }
        ui.add_space(40.0);
    });
}

// ---------------------------------------------------------------- Browse

#[derive(Clone)]
struct DragFiles(Vec<PathBuf>);

fn browse(ui: &mut egui::Ui, a: &mut UiState, s: &mut Session) {
    let p = a.pal;
    let parent = s.root.parent().unwrap_or(&s.root).to_path_buf();
    let all = records(a, s);
    let q = crux_core::query::parse(&a.org.query);
    let files: Vec<&IndexFile> = all.iter().filter(|f| q.matches(f)).collect();
    let wide = ui.available_width() > 860.0;
    let left_w = if wide { 300.0 } else { ui.available_width() };
    let mut drop: Option<(Dim, String, Vec<PathBuf>)> = None;
    let mut body = |ui: &mut egui::Ui, a: &mut UiState, s: &mut Session, left: bool| {
        if left {
            ui.set_width(left_w);
            w::card(ui, &p, |ui| {
                w::eyebrow(ui, &p, "Group by (drag to reorder)");
                ui.add_space(4.0);
                let mut remove = None;
                let mut swap = None;
                w::wrap_row(ui, |ui| {
                    for (i, d) in a.org.dims.clone().iter().enumerate() {
                        let r = dim_chip(ui, &p, d.name());
                        if r.1 {
                            remove = Some(i);
                        }
                        r.0.dnd_set_drag_payload(i);
                        if let Some(from) = r.0.dnd_release_payload::<usize>() {
                            swap = Some((*from, i));
                        }
                        if r.0.dnd_hover_payload::<usize>().is_some() {
                            ui.painter().rect_stroke(r.0.rect.expand(2.0), CornerRadius::same(14), Stroke::new(1.5, p.accent), StrokeKind::Outside);
                        }
                    }
                });
                if let Some(i) = remove {
                    a.org.dims.remove(i);
                    a.org.node = None;
                }
                if let Some((from, to)) = swap {
                    if from != to && from < a.org.dims.len() {
                        let d = a.org.dims.remove(from);
                        a.org.dims.insert(to.min(a.org.dims.len()), d);
                        a.org.node = None;
                    }
                }
                ui.add_space(6.0);
                w::eyebrow(ui, &p, "Add");
                ui.add_space(2.0);
                w::wrap_row(ui, |ui| {
                    let missing: Vec<Dim> = Dim::ALL.iter().copied().filter(|d| !a.org.dims.contains(d)).collect();
                    for d in missing.iter() {
                        if w::chip(ui, &p, &format!("+ {}", d.name()), None, false).clicked() {
                            a.org.dims.push(*d);
                            a.org.node = None;
                        }
                    }
                });
            });
            ui.add_space(12.0);
            w::card(ui, &p, |ui| {
                drop = tree(ui, a, &files, &parent);
            });
            ui.add_space(12.0);
            w::card(ui, &p, |ui| {
                w::eyebrow(ui, &p, "Saved views");
                ui.add_space(2.0);
                let mut views: Vec<(String, String, bool)> = s.decisions.views.iter().map(|v| (v.name.clone(), v.query.clone(), true)).collect();
                for (n, q) in [("Clean subs, FWHM under 4.5 px", "grade:good fwhm<4.5"), ("Everything crux rejects", "grade:reject"), ("Thin cloud (T under 0.8)", "t<0.8"), ("Flats", "type:flat")] {
                    if !views.iter().any(|v| v.1 == q) {
                        views.push((n.into(), q.into(), false));
                    }
                }
                let mut del = None;
                for (i, (name, query, mine)) in views.iter().enumerate() {
                    let n = all.iter().filter(|f| crux_core::query::parse(query).matches(f)).count();
                    ui.horizontal(|ui| {
                        let r = ui.add(egui::Label::new(RichText::new(name).font(body(13.0))).sense(Sense::click()));
                        if r.on_hover_text(query).on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
                            a.org.query = query.clone();
                            a.org.node = None;
                        }
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if *mine && ui.add(egui::Label::new(RichText::new("×").color(p.muted)).sense(Sense::click())).on_hover_text("Remove this view").clicked() {
                                del = Some(i);
                            }
                            ui.label(RichText::new(n.to_string()).font(mono(11.0)).color(p.muted));
                        });
                    });
                }
                if let Some(i) = del {
                    s.decisions.views.remove(i);
                    s.save_decisions();
                }
            });
        } else {
            files_panel(ui, a, s, &files, &parent);
        }
    };
    if wide {
        ui.horizontal_top(|ui| {
            ui.vertical(|ui| body(ui, a, s, true));
            ui.add_space(8.0);
            ui.vertical(|ui| body(ui, a, s, false));
        });
    } else {
        body(ui, a, s, true);
        ui.add_space(12.0);
        body(ui, a, s, false);
    }
    // Drag to correct: onto a panel or a grade.
    if let Some((dim, val, paths)) = drop {
        let n = paths.len();
        match dim {
            Dim::Grade => {
                let g = match val.as_str() {
                    "Reject" => Some(Grade::Reject),
                    "Good" | "Warn" => Some(Grade::Good),
                    _ => None,
                };
                if let Some(g) = g {
                    s.set_override_paths(&paths, Some(g));
                    a.toast(ui.ctx(), format!("{n} file{} marked {}", if n == 1 { "" } else { "s" }, if g == Grade::Reject { "reject" } else { "keep" }), true);
                }
            }
            Dim::Panel => {
                if let Ok(num) = val.trim_start_matches("Panel ").parse::<u32>() {
                    s.set_panel(&paths, Some(num));
                    a.toast(ui.ctx(), format!("{n} file{} reassigned to Panel {num}. Recorded as a correction; the FITS headers are untouched.", if n == 1 { "" } else { "s" }), true);
                }
            }
            _ => {}
        }
        a.org.selected.clear();
    }
}

/// A removable chip; returns (response, × clicked).
fn dim_chip(ui: &mut egui::Ui, p: &Pal, text: &str) -> (egui::Response, bool) {
    let g = ui.painter().layout_no_wrap(text.to_string(), body(12.5), p.bg);
    let wd = g.size().x + 20.0 + 14.0;
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(wd, 26.0), Sense::click_and_drag());
    let pa = ui.painter();
    pa.rect_filled(rect, CornerRadius::same(13), p.fg);
    pa.galley(Pos2::new(rect.left() + 10.0, rect.center().y - g.size().y / 2.0), g, p.bg);
    let x = Rect::from_center_size(Pos2::new(rect.right() - 11.0, rect.center().y), Vec2::splat(14.0));
    pa.text(x.center(), Align2::CENTER_CENTER, "×", body(13.0), p.bg);
    let removed = resp.clicked() && resp.interact_pointer_pos().is_some_and(|q| x.contains(q));
    (resp.on_hover_cursor(egui::CursorIcon::Grab), removed)
}

struct Node {
    files: Vec<usize>,
    kids: BTreeMap<String, Node>,
}

/// Sort key that puts PANEL_2 before PANEL_10.
fn nat(s: &str) -> String {
    let mut out = String::new();
    let mut num = String::new();
    for c in s.chars().chain(std::iter::once('\0')) {
        if c.is_ascii_digit() {
            num.push(c);
            continue;
        }
        if !num.is_empty() {
            out.push_str(&format!("{:0>10}", num));
            num.clear();
        }
        if c != '\0' {
            out.extend(c.to_lowercase());
        }
    }
    out
}

fn sorted_kids(n: &Node) -> Vec<(&String, &Node)> {
    let mut v: Vec<(&String, &Node)> = n.kids.iter().collect();
    v.sort_by_key(|(k, _)| nat(k));
    v
}

/// The group-by tree; returns a drop (dimension, value, files) if rows were dropped on a node.
fn tree(ui: &mut egui::Ui, a: &mut UiState, files: &[&IndexFile], parent: &Path) -> Option<(Dim, String, Vec<PathBuf>)> {
    let p = a.pal;
    let mut root = Node { files: Vec::new(), kids: BTreeMap::new() };
    for (i, f) in files.iter().enumerate() {
        let mut n = &mut root;
        n.files.push(i);
        for d in &a.org.dims {
            n = n.kids.entry(dim_val(f, *d, parent)).or_insert_with(|| Node { files: Vec::new(), kids: BTreeMap::new() });
            n.files.push(i);
        }
    }
    let bytes: u64 = files.iter().map(|f| f.size).sum();
    let mut drop = None;
    let sel_all = a.org.node.is_none();
    if node_row(ui, &p, 0, "All files", &format!("{} · {}", files.len(), gb(bytes)), sel_all, false, None, true).0.clicked() {
        a.org.node = None;
    }
    let mut stack: Vec<(usize, Vec<String>, &Node)> = Vec::new();
    for (k, n) in sorted_kids(&root).into_iter().rev() {
        stack.push((0, vec![k.clone()], n));
    }
    let mut shown = 0;
    while let Some((depth, path, n)) = stack.pop() {
        shown += 1;
        if shown > 600 {
            ui.label(RichText::new("…more groups; narrow the view with the query bar").font(body(12.0)).color(p.muted));
            break;
        }
        let key = path.join("/");
        let open = a.org.open.contains(&key);
        let secs: f64 = n.files.iter().filter(|&&i| files[i].kind == FrameType::Light && files[i].role == Role::Raw).filter_map(|&i| files[i].exptime).sum();
        let ct = if secs > 0.0 { format!("{} · {:.1} h", n.files.len(), secs / 3600.0) } else { n.files.len().to_string() };
        let dim = a.org.dims[depth];
        let droppable = matches!(dim, Dim::Panel | Dim::Grade);
        let sel = a.org.node.as_ref() == Some(&path);
        let (r, dropped) = node_row(ui, &p, depth, path.last().unwrap(), &ct, sel, !n.kids.is_empty(), Some(open), droppable);
        if let Some(paths) = dropped {
            drop = Some((dim, path.last().unwrap().clone(), paths));
        }
        if r.clicked() {
            if !n.kids.is_empty() {
                if open {
                    a.org.open.remove(&key);
                } else {
                    a.org.open.insert(key.clone());
                }
            }
            a.org.node = Some(path.clone());
        }
        if open {
            for (k, c) in sorted_kids(n).into_iter().rev() {
                let mut pp = path.clone();
                pp.push(k.clone());
                stack.push((depth + 1, pp, c));
            }
        }
    }
    drop
}

#[allow(clippy::too_many_arguments)]
fn node_row(ui: &mut egui::Ui, p: &Pal, depth: usize, name: &str, count: &str, sel: bool, kids: bool, open: Option<bool>, droppable: bool) -> (egui::Response, Option<Vec<PathBuf>>) {
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 26.0), Sense::click());
    let pa = ui.painter();
    let hover_drop = droppable && resp.dnd_hover_payload::<DragFiles>().is_some();
    if sel {
        pa.rect_filled(rect, CornerRadius::same(6), alpha(p.accent, 0.16));
    } else if resp.hovered() {
        pa.rect_filled(rect, CornerRadius::same(6), p.panel2);
    }
    if hover_drop {
        w::dashed_rect(pa, rect.shrink(1.0), p.accent);
    }
    let x = rect.left() + 8.0 + depth as f32 * 14.0;
    if kids {
        pa.text(Pos2::new(x + 4.0, rect.center().y), Align2::CENTER_CENTER, if open == Some(true) { "▾" } else { "▸" }, body(10.0), p.faint);
    }
    let font = if open.is_none() { semibold(13.0) } else { body(13.0) };
    pa.text(Pos2::new(x + 16.0, rect.center().y), Align2::LEFT_CENTER, name, font, p.fg);
    pa.text(Pos2::new(rect.right() - 8.0, rect.center().y), Align2::RIGHT_CENTER, count, mono(12.0), p.muted);
    let dropped = if droppable { resp.dnd_release_payload::<DragFiles>().map(|d| d.0.clone()) } else { None };
    (resp.on_hover_cursor(egui::CursorIcon::PointingHand), dropped)
}

fn files_panel(ui: &mut egui::Ui, a: &mut UiState, s: &mut Session, files: &[&IndexFile], parent: &Path) {
    let p = a.pal;
    // Query bar.
    egui::Frame::new().fill(p.panel).stroke(Stroke::new(1.0, p.line)).corner_radius(10).inner_margin(egui::Margin::symmetric(10, 6)).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal(|ui| {
            ui.label(RichText::new("⌕").color(p.muted));
            let te = egui::TextEdit::singleline(&mut a.org.query)
                .font(mono(13.0))
                .hint_text("panel:12 night:2026-11 fwhm<4.5 grade:good")
                .frame(egui::Frame::NONE)
                .desired_width(ui.available_width() - 110.0);
            if ui.add(te).changed() {
                a.org.node = None;
            }
            if w::button(ui, &p, "Save view", Kind::Ghost, None).clicked() {
                let q = a.org.query.trim().to_string();
                if q.is_empty() {
                    a.toast(ui.ctx(), "Type a filter first, for example grade:good fwhm<4.5", false);
                } else if !s.decisions.views.iter().any(|v| v.query == q) {
                    s.decisions.views.push(crux_core::decisions::SavedView { name: q.clone(), query: q });
                    s.save_decisions();
                    a.toast(ui.ctx(), "View saved.", false);
                }
            }
        });
    });
    let q = crux_core::query::parse(&a.org.query);
    ui.add_space(6.0);
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = Vec2::splat(6.0);
        ui.label(RichText::new("Try").font(body(12.0)).color(p.muted));
        for t in ["grade:reject", "t<0.8", "panel:26", "type:flat", "pier:east", "reason:trail"] {
            if w::chip(ui, &p, t, None, false).clicked() {
                a.org.query = t.into();
                a.org.node = None;
            }
        }
        ui.label(RichText::new("Drag rows onto a panel or grade in the tree to correct them.").font(body(12.0)).color(p.muted));
    });
    if !q.unknown.is_empty() {
        ui.label(RichText::new(format!("Not a filter crux knows: {}", q.unknown.join(", "))).font(body(12.0)).color(p.warn));
    }
    ui.add_space(8.0);
    // Files in the selected node.
    let in_node: Vec<&IndexFile> = match &a.org.node {
        None => files.to_vec(),
        Some(path) => files.iter().copied().filter(|f| path.iter().enumerate().all(|(i, v)| a.org.dims.get(i).is_some_and(|d| dim_val(f, *d, parent) == *v))).collect(),
    };
    let bytes: u64 = in_node.iter().map(|f| f.size).sum();
    let mut export = false;
    ui.horizontal(|ui| {
        ui.label(RichText::new(format!("{} files · {}", in_node.len(), gb(bytes))).font(semibold(14.0)));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            export = w::button_enabled(ui, &p, "Export as hard links", Kind::Normal, None, !in_node.is_empty()).clicked();
            ui.label(RichText::new("Browsing doesn't move files").font(body(12.0)).color(p.muted));
        });
    });
    if export {
        let name = export_name(&a.org.query, a.org.node.as_deref());
        let dest = parent.join("_exports").join(name);
        a.request = Some(Request::Export { files: in_node.iter().map(|f| f.path.clone()).collect(), dest });
    }
    ui.add_space(4.0);
    // Table.
    let cols = [("File", 0.40), ("Night", 0.09), ("Grade", 0.09), ("T", 0.06), ("FWHM", 0.07), ("Where it is now", 0.29)];
    let wd = ui.available_width();
    egui::Frame::new().stroke(Stroke::new(1.0, p.line)).corner_radius(10).show(ui, |ui| {
        ui.set_width(wd);
        let (hr, _) = ui.allocate_exact_size(Vec2::new(wd, 30.0), Sense::hover());
        let mut x = hr.left() + 8.0;
        for (name, f) in cols {
            ui.painter().text(Pos2::new(x, hr.center().y), Align2::LEFT_CENTER, name, body(11.5), p.muted);
            x += (wd - 16.0) * f;
        }
        ui.painter().line_segment([hr.left_bottom(), hr.right_bottom()], Stroke::new(1.0, p.line));
        let row_h = 28.0;
        ScrollArea::vertical().id_salt("ftable").max_height(560.0).auto_shrink([false, true]).show_rows(ui, row_h, in_node.len(), |ui, range| {
            for i in range {
                let f = in_node[i];
                let (r, resp) = ui.allocate_exact_size(Vec2::new(wd, row_h), Sense::click_and_drag());
                let sel = a.org.selected.contains(&f.path);
                let pa = ui.painter();
                if sel {
                    pa.rect_filled(r, CornerRadius::ZERO, alpha(p.accent, 0.14));
                } else if resp.hovered() {
                    pa.rect_filled(r, CornerRadius::ZERO, p.panel2);
                }
                pa.line_segment([r.left_bottom(), r.right_bottom()], Stroke::new(1.0, p.line));
                let mut x = r.left() + 8.0;
                let colw = |k: usize| (wd - 16.0) * cols[k].1;
                let name = f.path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                let name = clip(&name, 46);
                pa.text(Pos2::new(x, r.center().y), Align2::LEFT_CENTER, &name, mono(12.0), p.fg);
                if let Some(n) = s.decisions.panel(&f.path) {
                    let px = x + 8.0 + pa.layout_no_wrap(name.clone(), mono(12.0), p.fg).size().x;
                    pa.text(Pos2::new(px, r.center().y), Align2::LEFT_CENTER, format!("P{n}"), semibold(11.0), p.warn);
                }
                x += colw(0);
                pa.text(Pos2::new(x, r.center().y), Align2::LEFT_CENTER, f.night.map(d_short).unwrap_or_else(|| "—".into()), body(12.5), p.fg);
                x += colw(1);
                if let Some(g) = f.grade {
                    let c = p.grade(g);
                    let gtxt = crate::app::grade_name(g);
                    let gal = pa.layout_no_wrap(gtxt.to_string(), semibold(11.5), c);
                    let pr = Rect::from_min_size(Pos2::new(x, r.center().y - 10.0), Vec2::new(gal.size().x + 16.0, 20.0));
                    if g == Grade::Good {
                        pa.rect_stroke(pr, CornerRadius::same(10), Stroke::new(1.0, alpha(c, 0.4)), StrokeKind::Inside);
                    } else {
                        pa.rect_filled(pr, CornerRadius::same(10), alpha(c, 0.13));
                    }
                    pa.galley(Pos2::new(pr.left() + 8.0, pr.center().y - gal.size().y / 2.0), gal, c);
                }
                x += colw(2);
                pa.text(Pos2::new(x, r.center().y), Align2::LEFT_CENTER, f.t.map(|v| format!("{v:.2}")).unwrap_or_default(), mono(12.0), p.fg);
                x += colw(3);
                pa.text(Pos2::new(x, r.center().y), Align2::LEFT_CENTER, f.fwhm.map(|v| format!("{v:.2}")).unwrap_or_default(), mono(12.0), p.fg);
                x += colw(4);
                let dir = f.path.parent().map(|d| d.strip_prefix(parent).unwrap_or(d).display().to_string()).unwrap_or_default();
                pa.text(Pos2::new(x, r.center().y), Align2::LEFT_CENTER, clip(&dir, 40), mono(12.0), p.muted);
                if resp.clicked() {
                    let ctrl = ui.input(|i| i.modifiers.command || i.modifiers.ctrl);
                    if !ctrl {
                        a.org.selected.clear();
                    }
                    if sel && ctrl {
                        a.org.selected.remove(&f.path);
                    } else {
                        a.org.selected.insert(f.path.clone());
                    }
                }
                let payload = if sel { a.org.selected.iter().cloned().collect() } else { vec![f.path.clone()] };
                if resp.drag_started() {
                    resp.dnd_set_drag_payload(DragFiles(payload));
                }
            }
        });
    });
    if let Some(d) = egui::DragAndDrop::payload::<DragFiles>(ui.ctx()) {
        if let Some(pos) = ui.ctx().pointer_interact_pos() {
            let txt = format!("{} file{}", d.0.len(), if d.0.len() == 1 { "" } else { "s" });
            ui.ctx().debug_painter().text(pos + Vec2::new(14.0, 10.0), Align2::LEFT_TOP, txt, semibold(12.0), p.accent);
        }
    }
    if a.org.selected.len() > 1 {
        ui.label(RichText::new(format!("{} selected · drag them onto a panel or grade", a.org.selected.len())).font(body(12.0)).color(p.muted));
    }
}

fn clip(s: &str, n: usize) -> String {
    if s.chars().count() > n {
        format!("{}…", s.chars().take(n - 1).collect::<String>())
    } else {
        s.to_string()
    }
}

fn export_name(query: &str, node: Option<&[String]>) -> String {
    let base = match node {
        Some(n) if !n.is_empty() => n.join(" "),
        _ if !query.trim().is_empty() => query.trim().to_string(),
        _ => "all".into(),
    };
    let clean: String = base.chars().map(|c| if c.is_alphanumeric() || c == '-' || c == '.' { c } else { '_' }).collect();
    format!("{}_{}", chrono::Local::now().format("%Y-%m-%d"), clean.trim_matches('_'))
}

// ---------------------------------------------------------------- Layout and plan

fn tokens(t: &str) -> Vec<String> {
    t.split(['/', '\\']).filter(|x| !x.trim().is_empty()).map(|x| x.to_string()).collect()
}

fn layout(ui: &mut egui::Ui, a: &mut UiState, s: &mut Session) {
    let p = a.pal;
    let mut cfg_layout = s.cfg.layout.clone();
    s.decisions.apply_layout_to(&mut cfg_layout);
    if a.org.layout.is_none() {
        a.org.layout = Some(cfg_layout.clone());
    }
    let mut lay = a.org.layout.clone().unwrap();
    let mut changed = false;
    let wide = ui.available_width() > 980.0;
    let lw = if wide { (ui.available_width() - 16.0) * 0.44 } else { ui.available_width() };
    let rules_card = |ui: &mut egui::Ui, a: &mut UiState, s: &mut Session, lay: &mut Layout, changed: &mut bool| {
        ui.set_width(lw);
        w::card(ui, &p, |ui| {
            ui.horizontal(|ui| {
                w::h2(ui, "Layout rules");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let current = presets().into_iter().find(|(_, l)| same_layout(l, lay)).map(|x| x.0).unwrap_or("Custom");
                    egui::ComboBox::from_id_salt("preset").selected_text(current).width(220.0).show_ui(ui, |ui| {
                        for (name, l) in presets() {
                            if ui.selectable_label(name == current, name).clicked() {
                                *lay = l;
                                *changed = true;
                            }
                        }
                    });
                });
            });
            ui.add_space(4.0);
            ui.add(
                egui::Label::new(
                    RichText::new(format!("Paths are relative to {}. Click a token to add it to the rule you last touched; drag tokens to reorder. `..` steps out of the folder.", s.root.display()))
                        .font(body(13.0))
                        .color(p.muted),
                )
                .wrap(),
            );
            ui.add_space(6.0);
            w::wrap_row(ui, |ui| {
                ui.spacing_mut().item_spacing = Vec2::splat(6.0);
                for t in crux_core::plan::TOKENS {
                    if w::button(ui, &p, t, Kind::Normal, None).clicked() {
                        let r = a.org.last_rule;
                        let mut v = tokens(r.get(lay));
                        v.push(t.to_string());
                        r.set(lay, v.join("/"));
                        *changed = true;
                    }
                }
                let te = egui::TextEdit::singleline(&mut a.org.literal).hint_text("text").desired_width(90.0).font(mono(12.5));
                ui.add(te);
                if w::button_enabled(ui, &p, "+ text", Kind::Normal, None, !a.org.literal.trim().is_empty()).clicked() {
                    let r = a.org.last_rule;
                    let mut v = tokens(r.get(lay));
                    v.push(a.org.literal.trim().replace(['/', '\\'], "_"));
                    r.set(lay, v.join("/"));
                    a.org.literal.clear();
                    *changed = true;
                }
            });
            ui.add_space(6.0);
            let opts = s.plan_options(true);
            for rule in Rule::ALL {
                ui.add_space(6.0);
                w::eyebrow(ui, &p, rule.name());
                ui.add_space(2.0);
                let toks = tokens(rule.get(lay));
                let on = a.org.last_rule == rule;
                let fr = egui::Frame::new()
                    .fill(p.panel2)
                    .stroke(Stroke::new(1.0, if on { p.accent } else { p.line }))
                    .corner_radius(8)
                    .inner_margin(6)
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        let mut remove = None;
                        let mut mv = None;
                        w::wrap_row(ui, |ui| {
                            ui.spacing_mut().item_spacing = Vec2::splat(4.0);
                            for (i, t) in toks.iter().enumerate() {
                                if i > 0 {
                                    ui.label(RichText::new("\\").font(mono(13.0)).color(p.faint));
                                }
                                let (r, x) = token_chip(ui, &p, t);
                                if x {
                                    remove = Some(i);
                                }
                                r.dnd_set_drag_payload((rule as u8, i));
                                if let Some(from) = r.dnd_release_payload::<(u8, usize)>() {
                                    if from.0 == rule as u8 {
                                        mv = Some((from.1, i));
                                    }
                                }
                                if r.clicked() {
                                    a.org.last_rule = rule;
                                }
                            }
                            if toks.is_empty() {
                                ui.label(RichText::new("(the folder itself)").font(body(12.0)).color(p.muted));
                            }
                        });
                        (remove, mv)
                    });
                if fr.response.interact(Sense::click()).clicked() {
                    a.org.last_rule = rule;
                }
                let (remove, mv) = fr.inner;
                if let Some(i) = remove {
                    let mut v = toks.clone();
                    v.remove(i);
                    rule.set(lay, v.join("/"));
                    a.org.last_rule = rule;
                    *changed = true;
                }
                if let Some((from, to)) = mv {
                    if from != to {
                        let mut v = toks.clone();
                        let t = v.remove(from);
                        v.insert(to.min(v.len()), t);
                        rule.set(lay, v.join("/"));
                        a.org.last_rule = rule;
                        *changed = true;
                    }
                }
                if let Some((fi, reason)) = sample(s, rule) {
                    let dest = crux_core::plan::dest_for(&s.triage, &opts, rule.get(lay), fi, &reason);
                    let rel = rel_to(&dest, &s.root);
                    let name = s.triage.scan.frames[fi].meta.file_name().to_string();
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing.x = 0.0;
                        ui.label(RichText::new("e.g. ").font(mono(12.0)).color(p.muted));
                        ui.label(RichText::new(format!("{rel}\\")).font(mono(12.0)).color(p.fg));
                        ui.label(RichText::new(name).font(mono(12.0)).color(p.muted));
                    });
                }
            }
            ui.add_space(8.0);
            ui.label(RichText::new("Masters stay where they are. Your rules are saved with your decisions.").font(body(12.0)).color(p.muted));
            if s.decisions.layout.is_some() && w::button(ui, &p, "Back to the crux.toml layout", Kind::Ghost, None).clicked() {
                s.decisions.layout = None;
                s.save_decisions();
                a.org.layout = None;
                a.org.plan = None;
            }
        });
    };
    if wide {
        ui.horizontal_top(|ui| {
            ui.vertical(|ui| rules_card(ui, a, s, &mut lay, &mut changed));
            ui.add_space(8.0);
            ui.vertical(|ui| plan_col(ui, a, s, &lay));
        });
    } else {
        rules_card(ui, a, s, &mut lay, &mut changed);
        ui.add_space(12.0);
        plan_col(ui, a, s, &lay);
    }
    if changed {
        a.org.layout = Some(lay.clone());
        let base = s.cfg.layout.clone();
        s.decisions.layout = if same_layout(&lay, &base) { None } else { Some(lay) };
        s.save_decisions();
        a.org.plan = None;
    }
}

fn same_layout(a: &Layout, b: &Layout) -> bool {
    Rule::ALL.iter().all(|r| tokens(r.get(a)) == tokens(r.get(b)))
}

fn rel_to(p: &Path, root: &Path) -> String {
    if let Ok(r) = p.strip_prefix(root) {
        return r.display().to_string();
    }
    if let Some(parent) = root.parent() {
        if let Ok(r) = p.strip_prefix(parent) {
            return format!("..\\{}", r.display());
        }
    }
    p.display().to_string()
}

/// A frame to show as the example for a rule.
fn sample(s: &Session, rule: Rule) -> Option<(usize, String)> {
    let fr = &s.triage.scan.frames;
    let raw = |i: usize| fr[i].class.role == Role::Raw;
    match rule {
        Rule::Light => s.lights.iter().enumerate().find(|(i, l)| !l.bad && s.grade(*i) != Grade::Reject).map(|(_, l)| (l.fi, String::new())),
        Rule::Reject => s
            .lights
            .iter()
            .enumerate()
            .find(|(i, l)| s.grade(*i) == Grade::Reject && l.flags.iter().any(|f| f.grade == Grade::Reject))
            .map(|(_, l)| (l.fi, crux_core::plan::reason_slug(l.flags.iter().find(|f| f.grade == Grade::Reject).unwrap().check).to_string())),
        Rule::Flat => (0..fr.len()).find(|&i| raw(i) && fr[i].class.frame_type == FrameType::Flat && !fr[i].class.in_reject_dir).map(|i| (i, String::new())),
        Rule::Retired => (0..fr.len()).find(|&i| raw(i) && fr[i].class.frame_type == FrameType::Flat && fr[i].class.in_reject_dir).map(|i| (i, "retired".into())),
        Rule::Dark => (0..fr.len()).find(|&i| raw(i) && fr[i].class.frame_type == FrameType::Dark).map(|i| (i, String::new())),
        Rule::Bias => (0..fr.len()).find(|&i| raw(i) && fr[i].class.frame_type == FrameType::Bias).map(|i| (i, String::new())),
    }
}

fn token_chip(ui: &mut egui::Ui, p: &Pal, t: &str) -> (egui::Response, bool) {
    let is_tok = t.starts_with('{');
    let g = ui.painter().layout_no_wrap(t.to_string(), mono(12.5), p.fg);
    let wd = g.size().x + 12.0 + 18.0;
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(wd, 26.0), Sense::click_and_drag());
    let pa = ui.painter();
    let (fill, st) = if is_tok { (mix(p.accent, 0.18, p.panel), mix(p.accent, 0.4, p.panel)) } else { (p.panel, p.line) };
    pa.rect(rect, CornerRadius::same(6), fill, Stroke::new(1.0, st), StrokeKind::Inside);
    if resp.dnd_hover_payload::<(u8, usize)>().is_some() {
        pa.rect_stroke(rect.expand(1.5), CornerRadius::same(7), Stroke::new(1.5, p.accent), StrokeKind::Outside);
    }
    pa.galley(Pos2::new(rect.left() + 8.0, rect.center().y - g.size().y / 2.0), g, p.fg);
    let x = Rect::from_center_size(Pos2::new(rect.right() - 10.0, rect.center().y), Vec2::splat(14.0));
    pa.text(x.center(), Align2::CENTER_CENTER, "×", body(13.0), p.muted);
    let removed = resp.clicked() && resp.interact_pointer_pos().is_some_and(|q| x.contains(q));
    (resp.on_hover_cursor(egui::CursorIcon::Grab), removed)
}

fn plan_col(ui: &mut egui::Ui, a: &mut UiState, s: &mut Session, lay: &Layout) {
    let p = a.pal;
    let fresh = a.org.plan.as_ref().is_some_and(|(g, l, _)| *g == s.generation && same_layout(l, lay));
    if !fresh {
        let plan = s.plan_with(Some(lay), true);
        a.org.plan = Some((s.generation, lay.clone(), plan));
    }
    let plan = a.org.plan.as_ref().unwrap().2.clone();
    let root_files: Vec<&crux_core::Frame> = s.triage.scan.frames.iter().collect();
    let bytes = plan.total_bytes();
    let cross = plan.moves.iter().any(|m| m.from.components().next() != m.to.components().next());
    w::card(ui, &p, |ui| {
        ui.horizontal(|ui| {
            w::h2(ui, "The plan");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(RichText::new("Updates as you edit the rules").font(body(12.0)).color(p.muted));
            });
        });
        ui.add_space(6.0);
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 22.0;
            let stat = |ui: &mut egui::Ui, v: String, l: &str, c: Color32| {
                ui.vertical(|ui| {
                    ui.label(RichText::new(v).font(semibold(20.0)).color(c));
                    ui.label(RichText::new(l).font(body(12.0)).color(p.muted));
                });
            };
            stat(ui, plan.moves.len().to_string(), "files move", p.fg);
            stat(ui, (root_files.len().saturating_sub(plan.moves.len())).to_string(), "don't move", p.fg);
            stat(ui, plan.conflicts.len().to_string(), "name clashes", if plan.conflicts.is_empty() { p.good } else { p.rej });
            stat(ui, gb(bytes), if cross { "other drives: copied and checked" } else { "one drive: renamed in place" }, p.fg);
        });
        let kinds: Vec<String> = [MoveKind::Reject, MoveKind::Retire, MoveKind::Organise].iter().filter(|k| plan.count(**k) > 0).map(|k| format!("{} {}", plan.count(*k), k.name())).collect();
        if !kinds.is_empty() {
            ui.label(RichText::new(kinds.join(" · ")).font(body(12.0)).color(p.muted));
        }
        for c in plan.conflicts.iter().take(5) {
            ui.label(RichText::new(c).font(body(12.0)).color(p.rej));
        }
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            let can = !plan.moves.is_empty() && plan.conflicts.is_empty();
            if w::button_enabled(ui, &p, &format!("Apply {} moves", plan.moves.len()), Kind::Primary, None, can).clicked() {
                a.request = Some(Request::Apply { plan: plan.clone(), title: format!("Move {} files?", plan.moves.len()) });
            }
            let kept: Vec<PathBuf> = s.lights.iter().enumerate().filter(|(i, _)| s.grade(*i) != Grade::Reject).map(|(_, l)| l.path.clone()).collect();
            if w::button_enabled(ui, &p, "Export kept subs as hard links", Kind::Normal, None, !kept.is_empty()).clicked() {
                let parent = s.root.parent().unwrap_or(&s.root).to_path_buf();
                let name = format!("{}_{}_kept", chrono::Local::now().format("%Y-%m-%d"), crate::home::pretty_name(&s.root.file_name().unwrap_or_default().to_string_lossy()).replace(' ', "_"));
                a.request = Some(Request::Export { files: kept, dest: parent.join("_exports").join(name) });
            }
        });
    });
    ui.add_space(12.0);
    // Before/after tree diff.
    let mut before: BTreeMap<String, usize> = BTreeMap::new();
    let mut after: BTreeMap<String, usize> = BTreeMap::new();
    let moved: HashMap<String, &PathBuf> = plan.moves.iter().map(|m| (crux_core::cache::key(&m.from), &m.to)).collect();
    for f in &root_files {
        let from = f.meta.path.parent().map(|d| rel_to(d, &s.root)).unwrap_or_default();
        let to = moved.get(&crux_core::cache::key(&f.meta.path)).and_then(|t| t.parent()).map(|d| rel_to(d, &s.root)).unwrap_or_else(|| from.clone());
        *before.entry(from).or_default() += 1;
        *after.entry(to).or_default() += 1;
    }
    let mut keys: Vec<&String> = before.keys().chain(after.keys()).collect::<BTreeSet<_>>().into_iter().collect();
    keys.sort_by_key(|k| nat(k));
    let half = (ui.available_width() - 12.0) / 2.0;
    ui.horizontal_top(|ui| {
        for side in 0..2 {
            ui.vertical(|ui| {
                ui.set_width(half);
                w::eyebrow(ui, &p, if side == 0 { "Now" } else { "After" });
                ui.add_space(4.0);
                egui::Frame::new().stroke(Stroke::new(1.0, p.line)).corner_radius(10).inner_margin(8).show(ui, |ui| {
                    ui.set_width(half - 18.0);
                    ScrollArea::vertical().id_salt(("diff", side)).max_height(520.0).show(ui, |ui| {
                        for k in &keys {
                            let (b, af) = (before.get(*k).copied(), after.get(*k).copied());
                            let (n, colour, bg, strike) = match side {
                                0 => match (b, af) {
                                    (None, _) => continue,
                                    (Some(b), None) => (b, p.faint, None, true),
                                    (Some(b), Some(x)) if x != b => (b, p.fg, Some(alpha(p.warn, 0.10)), false),
                                    (Some(b), _) => (b, p.fg, None, false),
                                },
                                _ => match (b, af) {
                                    (_, None) => continue,
                                    (None, Some(x)) => (x, p.good, Some(alpha(p.good, 0.12)), false),
                                    (Some(b), Some(x)) if x != b => (x, p.fg, Some(alpha(p.warn, 0.10)), false),
                                    (_, Some(x)) => (x, p.fg, None, false),
                                },
                            };
                            let (r, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 22.0), Sense::hover());
                            if let Some(bg) = bg {
                                ui.painter().rect_filled(r, CornerRadius::same(4), bg);
                            }
                            let label = if k.is_empty() { ".\\".to_string() } else { format!("{k}\\") };
                            let g = ui.painter().layout_no_wrap(clip(&label, 52), mono(12.0), colour);
                            let gw = g.size().x;
                            ui.painter().galley(Pos2::new(r.left() + 6.0, r.center().y - g.size().y / 2.0), g, colour);
                            if strike {
                                ui.painter().line_segment([Pos2::new(r.left() + 6.0, r.center().y), Pos2::new(r.left() + 6.0 + gw, r.center().y)], Stroke::new(1.0, colour));
                            }
                            ui.painter().text(Pos2::new(r.right() - 6.0, r.center().y), Align2::RIGHT_CENTER, n.to_string(), mono(12.0), p.muted);
                        }
                    });
                });
            });
        }
    });
}

// ---------------------------------------------------------------- Library health

fn part_colour(p: &Pal, part: Part) -> Color32 {
    match part {
        Part::Raw => mix(p.accent, 0.7, Color32::WHITE),
        Part::Calibration => mix(p.f_o, 0.7, Color32::WHITE),
        Part::Masters => crate::theme::hex(0xc9b5ff),
        Part::Outputs => crate::theme::hex(0x9fe0b6),
        Part::Rebuildable => crate::theme::hex(0xf2c14e),
        Part::Other => mix(p.faint, 0.8, Color32::WHITE),
    }
}

/// Squarified treemap layout: (index, rect) for each value.
fn squarify(values: &[f64], rect: Rect) -> Vec<(usize, Rect)> {
    let mut idx: Vec<usize> = (0..values.len()).filter(|&i| values[i] > 0.0).collect();
    idx.sort_by(|&a, &b| values[b].total_cmp(&values[a]));
    let total: f64 = idx.iter().map(|&i| values[i]).sum();
    let mut out = Vec::new();
    if total <= 0.0 || rect.width() < 1.0 || rect.height() < 1.0 {
        return out;
    }
    let scale = (rect.width() * rect.height()) as f64 / total;
    let mut r = rect;
    let mut rest = &idx[..];
    while !rest.is_empty() {
        let short = r.width().min(r.height()) as f64;
        let mut n = 0;
        let mut best = f64::INFINITY;
        for k in 1..=rest.len() {
            let row = &rest[..k];
            let s: f64 = row.iter().map(|&i| values[i] * scale).sum();
            let worst = row.iter().map(|&i| {
                let a = values[i] * scale;
                ((short * short * a) / (s * s)).max((s * s) / (short * short * a))
            }).fold(0.0, f64::max);
            if worst > best {
                break;
            }
            best = worst;
            n = k;
        }
        let row = &rest[..n];
        let s: f64 = row.iter().map(|&i| values[i] * scale).sum();
        let thick = (s / short) as f32;
        let mut off = 0.0;
        for &i in row {
            let len = (values[i] * scale) as f32 / thick;
            let cell = if r.width() >= r.height() {
                Rect::from_min_size(Pos2::new(r.left(), r.top() + off), Vec2::new(thick, len))
            } else {
                Rect::from_min_size(Pos2::new(r.left() + off, r.top()), Vec2::new(len, thick))
            };
            out.push((i, cell));
            off += len;
        }
        r = if r.width() >= r.height() { Rect::from_min_max(Pos2::new(r.left() + thick, r.top()), r.max) } else { Rect::from_min_max(Pos2::new(r.left(), r.top() + thick), r.max) };
        rest = &rest[n..];
    }
    out
}

fn health(ui: &mut egui::Ui, a: &mut UiState, s: &mut Session) {
    let p = a.pal;
    let parent = s.root.parent().unwrap_or(&s.root).to_path_buf();
    if a.spaces.is_empty() {
        w::card(ui, &p, |ui| w::muted(ui, &p, "Measuring the library…"));
        return;
    }
    let total: u64 = a.spaces.iter().map(|x| x.total()).sum();
    let reb: u64 = a.spaces.iter().map(|x| x.get(Part::Rebuildable)).sum();
    let mut reclaim = false;
    w::card(ui, &p, |ui| {
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                w::h2(ui, "Where the space goes");
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    ui.label(RichText::new(format!("{} · {} measured ·", parent.display(), gb(total))).font(body(13.0)).color(p.muted));
                    ui.label(RichText::new(gb(reb)).font(semibold(13.0)).color(p.warn));
                    ui.label(RichText::new("is WBPP intermediates that can be rebuilt from raw subs").font(body(13.0)).color(p.muted));
                });
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                reclaim = w::button_enabled(ui, &p, &format!("Move {} to {}", gb(reb), crux_core::health::RECLAIM_DIR), Kind::Primary, None, reb > 0).clicked();
            });
        });
        ui.add_space(8.0);
        treemap(ui, &p, &a.spaces);
        ui.add_space(6.0);
        ui.horizontal_wrapped(|ui| {
            for part in Part::ALL {
                let (r, _) = ui.allocate_exact_size(Vec2::splat(10.0), Sense::hover());
                ui.painter().rect_filled(r, CornerRadius::same(3), part_colour(&p, part));
                ui.label(RichText::new(format!("{} {}", part.name().to_lowercase(), gb(a.spaces.iter().map(|x| x.get(part)).sum()))).font(body(12.0)).color(p.muted));
                ui.add_space(8.0);
            }
        });
    });
    ui.add_space(14.0);
    caches(ui, a, s);
    if reclaim {
        let plan = crux_core::health::reclaim_plan(&parent, &a.spaces, s.cache_dir.as_deref());
        a.request = Some(Request::Apply { title: format!("Move {} ({} files) to {}?", gb(plan.total_bytes()), plan.moves.len(), crux_core::health::RECLAIM_DIR), plan });
    }
    ui.add_space(14.0);
    // Findings.
    let lib = records(a, s);
    let mut findings: Vec<(crate::session::Sev, &str, String, String, Option<(&str, FindAct)>)> = Vec::new();
    use crate::session::Sev;
    let t = &s.triage;
    let g = &t.grouping;
    // Masters from a retired set.
    let src: Vec<String> = t.cal.sets.iter().filter(|x| x.flags.iter().any(|f| f.check == crux_core::grade::Check::Source)).map(|x| g.groups[x.group].label.clone()).collect();
    if !src.is_empty() {
        findings.push((Sev::Rej, "✕", format!("{} master flat{} built from a retired set", src.len(), if src.len() == 1 { "" } else { "s" }), format!("{}. crux won't match them; masters are never moved, so set them aside or rebuild them yourself.", src.join("; ")), Some(("Show in Calibrate", FindAct::Calibrate))));
    }
    // Lights with no usable flats.
    let none: Vec<String> = t.cal.choices.iter().filter(|c| c.chosen.is_none()).map(|c| format!("{} {}", c.filter, d_short(c.night))).collect();
    if !none.is_empty() {
        findings.push((Sev::Warn, "!", format!("{} light night{} with no usable flats", none.len(), if none.len() == 1 { "" } else { "s" }), none.join(", "), None));
    }
    // Loose flat folders (only flats).
    let mut roots: BTreeMap<PathBuf, (usize, usize)> = BTreeMap::new();
    for f in &lib {
        let e = roots.entry(f.root.clone()).or_default();
        e.0 += 1;
        if f.kind == FrameType::Flat && f.role == Role::Raw {
            e.1 += 1;
        }
    }
    let loose: Vec<(&PathBuf, usize)> = roots.iter().filter(|(_, (n, fl))| *n > 0 && n == fl).map(|(r, (_, fl))| (r, *fl)).collect();
    if !loose.is_empty() {
        let names: Vec<String> = loose.iter().map(|(r, _)| r.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default()).collect();
        let n: usize = loose.iter().map(|x| x.1).sum();
        findings.push((Sev::Info, "≡", format!("{} loose flat folders, {n} frames", loose.len()), format!("{}. Add them to the flat library so every target can find them: Refresh calibration library on Home or Calibrate.", names.join(", ")), Some(("Show them", FindAct::Query("type:flat".into())))));
    }
    // Panels short of their neighbours.
    let mut per: BTreeMap<u32, f64> = BTreeMap::new();
    for (i, l) in s.lights.iter().enumerate() {
        if let Some(pn) = l.panel {
            let e = per.entry(pn).or_default();
            if s.grade(i) != Grade::Reject {
                *e += l.exp;
            }
        }
    }
    if per.len() >= 4 {
        let mut v: Vec<f64> = per.values().copied().collect();
        v.sort_by(f64::total_cmp);
        let med = v[v.len() / 2];
        let thin: Vec<String> = per.iter().filter(|(_, sec)| **sec < med * 0.5).map(|(pn, _)| pn.to_string()).collect();
        if !thin.is_empty() {
            findings.push((Sev::Info, "◔", format!("{} panels well short of their neighbours", thin.len()), format!("Panels {} have under half the median kept integration ({:.1} h). Consider adding them to the next clear night's plan.", thin.join(", "), med / 3600.0), None));
        }
    }
    // Renamed rejects.
    let renamed = lib.iter().filter(|f| f.path.file_name().is_some_and(|n| n.to_string_lossy().to_uppercase().starts_with("BAD_"))).count();
    if renamed > 0 {
        findings.push((Sev::Info, "✎", format!("{renamed} renamed rejects"), "Files prefixed BAD_ (by N.I.N.A. or by hand). crux treats them like frames in a BAD folder: left out unless crux grades them good.".into(), Some(("Show them", FindAct::Query("bad_".into())))));
    }
    // Names and headers that disagree.
    let disagree: Vec<String> = t.scan.frames.iter().filter(|f| !f.class.conflicts().is_empty()).map(|f| f.meta.file_name().to_string()).collect();
    if !disagree.is_empty() {
        findings.push((Sev::Warn, "≠", format!("{} files whose name, folder or header disagree on the frame type", disagree.len()), disagree.iter().take(4).cloned().collect::<Vec<_>>().join(", "), None));
    }
    // Empty folders.
    if !a.empty_dirs.is_empty() {
        let names: Vec<String> = a.empty_dirs.iter().take(4).map(|d| d.display().to_string()).collect();
        findings.push((Sev::Info, "∅", format!("{} empty folder{}", a.empty_dirs.len(), if a.empty_dirs.len() == 1 { "" } else { "s" }), format!("{}. Delete them yourself if you don't need them.", names.join(", ")), None));
    }
    // Duplicates.
    let cands = crux_core::health::duplicate_candidates(&lib);
    match (&a.org.dups, a.org.dup_progress) {
        (Some(d), _) if d.is_empty() => findings.push((Sev::Info, "✓", "No duplicates".into(), format!("{} same-name, same-size files checked by content hash: none are identical.", cands.iter().map(|c| c.len()).sum::<usize>()), None)),
        (Some(d), _) => {
            let bytes: u64 = d.iter().map(|x| x.size * (x.paths.len() as u64 - 1)).sum();
            let ex: Vec<String> = d.iter().take(3).map(|x| x.paths.iter().map(|q| rel_to(q, &parent)).collect::<Vec<_>>().join(" = ")).collect();
            findings.push((Sev::Warn, "⧉", format!("{} sets of identical files ({} of extra copies)", d.len(), gb(bytes)), ex.join("; "), None));
        }
        (None, Some((dn, tt))) => findings.push((Sev::Info, "⧉", "Checking duplicates…".into(), format!("hashing {dn} of {tt} files"), None)),
        (None, None) if cands.is_empty() => findings.push((Sev::Good, "✓", "No duplicate subs".into(), format!("Checked {} image files by name and size.", lib.len()), None)),
        (None, None) => {
            let n: usize = cands.iter().map(|c| c.len()).sum();
            findings.push((Sev::Warn, "⧉", format!("{} possible duplicates", cands.len()), format!("{n} files share a name and size with another. Hashing confirms which are identical (reads them once)."), Some(("Check with hashes", FindAct::Hash))));
        }
    }
    let mut act = None;
    for (sev, icon, title, text, action) in &findings {
        let c = match sev {
            Sev::Rej => p.rej,
            Sev::Warn => p.warn,
            Sev::Info => p.info,
            Sev::Good => p.good,
        };
        egui::Frame::new().fill(p.panel).stroke(Stroke::new(1.0, p.line)).corner_radius(10).inner_margin(12).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal_top(|ui| {
                let (r, _) = ui.allocate_exact_size(Vec2::splat(28.0), Sense::hover());
                ui.painter().rect_filled(r, CornerRadius::same(8), alpha(c, 0.18));
                ui.painter().text(r.center(), Align2::CENTER_CENTER, *icon, semibold(13.0), c);
                ui.vertical(|ui| {
                    ui.label(RichText::new(title).font(semibold(13.0)));
                    ui.add(egui::Label::new(RichText::new(text).font(body(13.0)).color(p.muted)).wrap());
                    if let Some((label, fa)) = action {
                        ui.add_space(4.0);
                        if w::button(ui, &p, label, Kind::Normal, None).clicked() {
                            act = Some(fa.clone());
                        }
                    }
                });
            });
        });
        ui.add_space(8.0);
    }
    match act {
        Some(FindAct::Calibrate) => a.view = crate::app::View::Calibrate,
        Some(FindAct::Query(q)) => {
            a.org.query = q;
            a.org.node = None;
            a.org.tab = Tab::Browse;
        }
        Some(FindAct::Hash) => {
            a.org.dup_progress = Some((0, cands.iter().map(|c| c.len()).sum()));
            a.request = Some(Request::HashDuplicates(cands.iter().map(|c| c.iter().map(|f| (*f).clone()).collect()).collect()));
        }
        None => {}
    }
}

/// crux's caches: what each folder's costs, emptied with a button. Emptying
/// removes only what a scan rebuilds; forgetting (folders that are gone)
/// and removing an old in-folder `.crux` take a second click.
fn caches(ui: &mut egui::Ui, a: &mut UiState, s: &mut Session) {
    use crux_core::cachedir::{self, human};
    let p = a.pal;
    if a.org.caches.is_none() {
        a.org.caches = Some(cachedir::list());
    }
    let list = a.org.caches.clone().unwrap_or_default();
    let here = s.cache_dir.clone();
    let managed = here.as_ref().is_some_and(|h| list.iter().any(|c| crux_core::paths::key(&c.dir) == crux_core::paths::key(h)));
    let rb: u64 = list.iter().map(|c| c.rebuildable).sum();
    let kept: u64 = list.iter().map(|c| c.kept).sum();
    let old: u64 = list.iter().filter_map(|c| c.legacy.as_ref().map(|l| l.1)).sum();
    #[derive(Clone)]
    enum Act {
        Empty(PathBuf),
        EmptyAll,
        Forget(PathBuf),
        RemoveOld(PathBuf),
        Arm(PathBuf),
    }
    let mut act = None;
    w::card(ui, &p, |ui| {
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                w::h2(ui, "crux's cache");
                let wher = cachedir::root().map(|r| r.display().to_string()).unwrap_or_default();
                let text = if managed || here.is_none() {
                    format!("{} folder{} · {} of measurements, maps and thumbnails (rebuilt by the next scan) · {} of your decisions and move journals (kept) · in {wher}", list.len(), if list.len() == 1 { "" } else { "s" }, human(rb), human(kept))
                } else {
                    format!("This folder's cache is {}. Managed caches: {} folders, {} rebuildable, in {wher}", here.as_ref().map(|h| h.display().to_string()).unwrap_or_default(), list.len(), human(rb))
                };
                ui.add(egui::Label::new(RichText::new(text).font(body(13.0)).color(p.muted)).wrap());
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if w::button_enabled(ui, &p, &format!("Empty all ({})", human(rb)), Kind::Normal, None, rb > 0).clicked() {
                    act = Some(Act::EmptyAll);
                }
            });
        });
        if let Some(n) = &a.org.cache_note {
            ui.label(RichText::new(n).font(body(12.0)).color(p.good));
        }
        if list.is_empty() {
            return;
        }
        ui.add_space(6.0);
        egui::Grid::new("caches").num_columns(5).spacing([14.0, 6.0]).striped(false).show(ui, |ui| {
            for h in ["Folder", "Rebuildable", "Kept", "Opened", ""] {
                ui.label(RichText::new(h).font(semibold(12.0)).color(p.muted));
            }
            ui.end_row();
            for c in &list {
                let name = c.folder.as_ref().map(|f| f.display().to_string()).unwrap_or_else(|| c.dir.display().to_string());
                let open_now = here.as_ref().is_some_and(|h| crux_core::paths::key(h) == crux_core::paths::key(&c.dir));
                let mut t = RichText::new(clip(&name, 70)).font(body(13.0));
                if !c.folder_exists {
                    t = t.color(p.muted).strikethrough();
                }
                ui.label(t).on_hover_text(format!("{}\ncache: {}", name, c.dir.display()));
                ui.label(RichText::new(human(c.rebuildable)).font(mono(12.5)));
                ui.label(RichText::new(human(c.kept)).font(mono(12.5)).color(p.muted));
                let opened = c.last_used.map(|t| chrono::DateTime::<chrono::Local>::from(t).format("%-d %b %Y").to_string()).unwrap_or_else(|| "?".into());
                ui.label(RichText::new(if open_now { "open now".to_string() } else { opened }).font(body(12.5)).color(p.muted));
                ui.horizontal(|ui| {
                    if w::button_enabled(ui, &p, "Empty", Kind::Normal, None, c.rebuildable > 0).on_hover_text("Remove measurements, maps and thumbnails; the next scan measures again. Decisions and journals stay.").clicked() {
                        act = Some(Act::Empty(c.dir.clone()));
                    }
                    if !c.folder_exists {
                        let armed = a.org.armed.as_ref() == Some(&c.dir);
                        if w::button(ui, &p, if armed { "Really forget?" } else { "Forget" }, if armed { Kind::Danger } else { Kind::Normal }, None).on_hover_text("The folder is gone: remove its cache completely, decisions and journals included.").clicked() {
                            act = Some(if armed { Act::Forget(c.dir.clone()) } else { Act::Arm(c.dir.clone()) });
                        }
                    }
                });
                ui.end_row();
                if let Some((l, n)) = &c.legacy {
                    ui.label(RichText::new(format!("  ↳ old cache still in the data folder: {}", l.display())).font(body(12.5)).color(p.warn));
                    ui.label(RichText::new(human(*n)).font(mono(12.5)));
                    ui.label("");
                    ui.label(RichText::new("copied across").font(body(12.5)).color(p.muted));
                    let armed = a.org.armed.as_ref() == Some(l);
                    if w::button(ui, &p, if armed { "Really remove?" } else { "Remove" }, if armed { Kind::Danger } else { Kind::Normal }, None).on_hover_text("crux copied it here; this deletes the old .crux folder inside your data folder.").clicked() {
                        act = Some(if armed { Act::RemoveOld(l.clone()) } else { Act::Arm(l.clone()) });
                    }
                    ui.end_row();
                }
            }
        });
        if old > 0 {
            ui.add_space(4.0);
            w::muted(ui, &p, &format!("{} of old caches are still inside data folders. crux no longer writes there.", human(old)));
        }
    });
    let freed = |r: anyhow::Result<u64>, what: &str| match r {
        Ok(b) => format!("{what}: freed {}", human(b)),
        Err(e) => format!("{what} failed: {e:#}"),
    };
    let note = match act {
        None => return,
        Some(Act::Arm(d)) => {
            a.org.armed = Some(d);
            return;
        }
        Some(Act::Empty(d)) => freed(cachedir::empty(&d), "Emptied"),
        Some(Act::EmptyAll) => freed(list.iter().filter(|c| c.rebuildable > 0).map(|c| cachedir::empty(&c.dir)).sum::<anyhow::Result<u64>>(), "Emptied every cache"),
        Some(Act::Forget(d)) => freed(cachedir::forget(&d), "Forgot the cache"),
        Some(Act::RemoveOld(d)) => freed(cachedir::remove_legacy(&d), "Removed the old cache"),
    };
    a.org.cache_note = Some(note);
    a.org.armed = None;
    a.org.caches = None;
}

#[derive(Clone)]
enum FindAct {
    Calibrate,
    Query(String),
    Hash,
}

fn treemap(ui: &mut egui::Ui, p: &Pal, spaces: &[FolderSpace]) {
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 340.0), Sense::hover());
    let pa = ui.painter_at(rect);
    pa.rect_filled(rect, CornerRadius::same(10), p.panel2);
    let vals: Vec<f64> = spaces.iter().map(|x| x.total() as f64).collect();
    let mut tip = None;
    for (i, r) in squarify(&vals, rect.shrink(2.0)) {
        let sp = &spaces[i];
        let head = Rect::from_min_size(r.min, Vec2::new(r.width(), 18.0));
        let name = crate::home::pretty_name(&sp.name);
        if r.width() > 40.0 {
            pa.text(head.left_center() + Vec2::new(5.0, 1.0), Align2::LEFT_CENTER, clip(&format!("{name}  {}", gb(sp.total())), (r.width() / 7.0) as usize), semibold(11.5), p.fg);
        }
        let inner = Rect::from_min_max(Pos2::new(r.left() + 2.0, r.top() + 18.0), r.max - Vec2::splat(2.0));
        let pv: Vec<f64> = Part::ALL.iter().map(|pt| sp.get(*pt) as f64).collect();
        for (k, c) in squarify(&pv, inner) {
            let part = Part::ALL[k];
            let c = c.shrink(1.0);
            pa.rect_filled(c, CornerRadius::same(5), part_colour(p, part));
            if part == Part::Rebuildable {
                // Stripes mark rebuildable space.
                let mut x = c.left() - c.height();
                while x < c.right() {
                    let a0 = Pos2::new(x.max(c.left()), c.bottom() - (x.max(c.left()) - x));
                    let x1 = (x + c.height()).min(c.right());
                    let a1 = Pos2::new(x1, c.bottom() - (x1 - x));
                    pa.line_segment([a0, a1], Stroke::new(3.0, crate::theme::hex(0xe3ae35)));
                    x += 12.0;
                }
            }
            if c.width() > 70.0 && c.height() > 36.0 {
                pa.text(c.left_top() + Vec2::new(7.0, 6.0), Align2::LEFT_TOP, part.name(), semibold(12.0), Color32::from_rgb(11, 14, 20));
                pa.text(c.left_top() + Vec2::new(7.0, 21.0), Align2::LEFT_TOP, gb(sp.get(part)), mono(11.0), Color32::from_rgb(11, 14, 20));
            }
            if resp.hover_pos().is_some_and(|h| c.contains(h)) {
                tip = Some(format!("{name} · {} · {}", part.name(), gb(sp.get(part))));
            }
        }
    }
    if let Some(t) = tip {
        resp.on_hover_text(t);
    }
}

// ---------------------------------------------------------------- History

fn history(ui: &mut egui::Ui, a: &mut UiState, s: &mut Session) {
    let p = a.pal;
    ui.set_max_width(860.0);
    if a.org.history.is_empty() {
        egui::Frame::new().stroke(Stroke::new(1.5, p.line)).corner_radius(12).inner_margin(28).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.vertical_centered(|ui| w::muted(ui, &p, "No moves yet. Each apply is listed here, newest first, and you can undo any of them."));
        });
        return;
    }
    let mut undo = None;
    for e in a.org.history.clone() {
        egui::Frame::new().fill(p.panel).stroke(Stroke::new(1.0, p.line)).corner_radius(10).inner_margin(12).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                let c = if e.undone.is_some() { p.faint } else if e.finished { p.good } else { p.warn };
                let (r, _) = ui.allocate_exact_size(Vec2::splat(12.0), Sense::hover());
                ui.painter().circle_filled(r.center(), 4.5, c);
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(format!("Moved {} files", e.moves)).font(semibold(13.5)));
                        ui.label(RichText::new(format!("· {} · {}", gb(e.bytes), crate::home::pretty_name(&e.root.file_name().unwrap_or_default().to_string_lossy()))).color(p.muted));
                    });
                    let status = if e.undone.is_some() { " · undone" } else if !e.finished { " · stopped part-way" } else { "" };
                    ui.label(RichText::new(format!("{}{status}", e.started)).font(mono(12.0)).color(p.muted));
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if e.undone.is_none() && w::button(ui, &p, "Undo", Kind::Normal, None).clicked() {
                        undo = Some(e.clone());
                    }
                });
            });
        });
        ui.add_space(8.0);
    }
    if let Some(e) = undo {
        a.request = Some(Request::Undo(e));
    }
    let _ = s;
}
