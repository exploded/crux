//! One opened folder: crux's triage results reshaped for the screens, live
//! regrading as reject limits move, and your overrides (pinned, shared with
//! the CLI through the decisions sidecar).

use chrono::{NaiveDate, NaiveDateTime, Timelike};
use crux_core::cache::Thumb;
use crux_core::config::Config;
use crux_core::decisions::{self, Decisions};
use crux_core::grade::{grade_lights_with, Check, Flag, FrameReport, Grade};
use crux_core::pipeline::Triage;
use crux_core::transparency::{Affine, FrameTransparency};
use crux_core::{FrameType, Role};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

/// The four live reject limits, in slider order. Every slider runs from
/// forgiving (left) to selective (right): `strict_low` limits reject more
/// as the value falls, so their sliders run high to low.
pub const LIMITS: [Limit; 4] = [
    Limit { key: "transparency.reject", label: "Transparency below", min: 0.2, max: 0.95, step: 0.01, unit: "", decimals: 2, strict_low: false },
    Limit { key: "trails.reject_sigma", label: "Trail brighter than", min: 0.2, max: 6.0, step: 0.1, unit: "×", decimals: 1, strict_low: true },
    Limit { key: "focus.reject_ratio", label: "FWHM over night median", min: 1.1, max: 2.5, step: 0.05, unit: "×", decimals: 2, strict_low: true },
    Limit { key: "sky.reject_ratio", label: "Sky over panel's best", min: 2.0, max: 8.0, step: 0.1, unit: "×", decimals: 1, strict_low: true },
];

#[derive(Clone, Copy)]
pub struct Limit {
    pub key: &'static str,
    pub label: &'static str,
    pub min: f64,
    pub max: f64,
    pub step: f64,
    pub unit: &'static str,
    pub decimals: usize,
    /// Lower values reject more (the slider runs high to low).
    pub strict_low: bool,
}

/// A raw light as the screens see it.
#[derive(Clone)]
pub struct Light {
    /// Index into the scan's frames.
    pub fi: usize,
    pub path: PathBuf,
    pub rel: String,
    pub name: String,
    pub obj: String,
    pub target: String,
    pub panel: Option<u32>,
    pub filter: String,
    pub night: NaiveDate,
    pub time: NaiveDateTime,
    pub exp: f64,
    /// In a reject folder or named BAD_ (you rejected it before crux).
    /// Still stacked when crux grades it good, unless you pin Reject.
    pub bad: bool,
    /// Rejected by its name (BAD_…), not by its folder.
    pub bad_name: bool,
    pub width: u32,
    pub height: u32,
    pub t: Option<f64>,
    pub fwhm: Option<f64>,
    pub ecc: Option<f64>,
    pub stars: Option<u32>,
    pub sky: Option<f64>,
    pub skyr: Option<f64>,
    pub fr: Option<f64>,
    /// Worst trail, × pixel noise.
    pub tx: Option<f64>,
    pub grad: Option<f64>,
    pub talt: Option<f64>,
    pub malt: Option<f64>,
    pub millum: Option<f64>,
    pub pier: Option<String>,
    pub cells: Option<Vec<Option<f64>>>,
    /// Light index of the field's reference frame, and frame → reference pixels.
    pub reference: Option<usize>,
    pub to_ref: Option<Affine>,
    pub auto: Grade,
    pub flags: Vec<Flag>,
    pub notes: Vec<String>,
}

impl Light {
    pub fn hhmm(&self) -> String {
        format!("{:02}:{:02}", self.time.hour(), self.time.minute())
    }
}

/// One undoable change.
pub enum Undo {
    Grades(Vec<(PathBuf, Option<Grade>)>),
    Panels(Vec<(PathBuf, Option<u32>)>),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Sev {
    Info,
    Warn,
    Rej,
    Good,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Act {
    Dismiss,
    ReviewDisagree,
    ReviewBorder,
    OpenCalibration,
}

pub struct InboxItem {
    pub id: String,
    pub sev: Sev,
    pub title: String,
    pub body: String,
    pub evidence: Vec<String>,
    /// Light indices to show as thumbnails.
    pub thumbs: Vec<usize>,
    pub acts: Vec<(String, bool, Act)>,
}

/// The opened folder or one beside it, from headers only (Home's target
/// card and calendar, the Library card's folder count).
#[derive(Clone, Default)]
pub struct FolderSummary {
    pub path: PathBuf,
    pub objects: Vec<String>,
    /// Light seconds per filter.
    pub light_secs: BTreeMap<String, f64>,
    pub light_nights: BTreeMap<NaiveDate, f64>,
    pub flat_nights: Vec<NaiveDate>,
    pub flats: usize,
}

pub struct Session {
    pub root: PathBuf,
    pub cache_dir: Option<PathBuf>,
    pub base_cfg: Config,
    pub cfg: Config,
    pub triage: Triage,
    pub transp: HashMap<usize, FrameTransparency>,
    pub lights: Vec<Light>,
    pub by_fi: HashMap<usize, usize>,
    pub nights: Vec<NaiveDate>,
    pub decisions: Decisions,
    pub decisions_error: Option<String>,
    /// Current slider values, and crux's (config) values.
    pub limits: [f64; 4],
    pub defaults: [f64; 4],
    /// crux's grade for each light at the default limits.
    pub auto0: Vec<Grade>,
    pub undo: Vec<Undo>,
    /// Bumped whenever grades, pins or corrections change (views cache on it).
    pub generation: u64,
    pub thumbs: HashMap<usize, Thumb>,
    pub regrade_ms: f64,
    pub timings: Timings,
    pub plan_moves: usize,
    pub plan_kinds: BTreeMap<String, usize>,
    /// crux's own flat choices, before yours are laid over them.
    pub crux_choices: Vec<crux_core::calcheck::FlatChoice>,
}

#[derive(Clone, Default, Debug)]
pub struct Timings {
    pub header_files: usize,
    pub header_s: f64,
    pub cal_frames: usize,
    pub cal_measured: usize,
    pub cal_s: f64,
    pub lights: usize,
    pub lights_measured: usize,
    pub lights_s: f64,
    pub transparency_s: f64,
}

pub fn target_of(obj: &str) -> String {
    let low = obj.to_lowercase();
    let t = match low.find(" panel") {
        Some(i) => obj[..i].trim(),
        None => obj.trim(),
    };
    if t.is_empty() { "Unknown".into() } else { t.to_string() }
}

fn panel_num(obj: &str) -> Option<u32> {
    let low = obj.to_lowercase();
    let i = low.find("panel")?;
    obj[i + 5..].trim_start_matches([' ', '_', '-']).chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().ok()
}

impl Session {
    pub fn new(
        root: PathBuf,
        cache_dir: Option<PathBuf>,
        base_cfg: Config,
        triage: Triage,
        transp: HashMap<usize, FrameTransparency>,
        thumbs_by_fi: HashMap<usize, Thumb>,
        timings: Timings,
    ) -> Session {
        let (decisions, decisions_error) = match cache_dir.as_deref().map(Decisions::load) {
            Some(Ok(d)) => (d, None),
            Some(Err(e)) => (Decisions::default(), Some(format!("{e:#}"))),
            None => (Decisions::default(), None),
        };
        let defaults = LIMITS.map(|l| decisions::threshold(&base_cfg, l.key).unwrap_or(0.0));
        let limits = std::array::from_fn(|i| decisions.thresholds.get(LIMITS[i].key).copied().unwrap_or(defaults[i]));
        let frames = &triage.scan.frames;
        let mut lights = Vec::new();
        let mut ids: Vec<usize> = triage.reports.iter().map(|r| r.frame).collect();
        ids.retain(|&i| frames[i].class.frame_type == FrameType::Light && frames[i].class.role == Role::Raw);
        for fi in ids {
            let f = &frames[fi];
            let m = &f.meta;
            let Some(night) = f.class.night else { continue };
            let Some(time) = f.class.local_time else { continue };
            let obj = m.object.clone().unwrap_or_else(|| "Unknown".into());
            let lm = triage.metrics.get(&fi);
            let tx = lm.and_then(|lm| {
                let noise = (lm.noise_adu as f64).max(1e-6);
                lm.trails.iter().map(|t| t.brightness_adu as f64 / noise).fold(None, |a: Option<f64>, v| Some(a.map_or(v, |a| a.max(v))))
            });
            lights.push(Light {
                fi,
                path: m.path.clone(),
                rel: m.path.strip_prefix(&root).unwrap_or(&m.path).display().to_string(),
                name: m.file_name().to_string(),
                target: target_of(&obj),
                panel: panel_num(&obj),
                obj,
                filter: m.filter.clone().unwrap_or_default().to_uppercase(),
                night,
                time,
                exp: m.exptime.unwrap_or(0.0),
                bad: f.class.in_reject_dir,
                bad_name: f.class.reject_name_only,
                width: m.width,
                height: m.height,
                t: None,
                fwhm: lm.and_then(|x| x.fwhm).map(f64::from),
                ecc: lm.and_then(|x| x.ecc).map(f64::from),
                stars: lm.map(|x| x.n_stars),
                sky: None,
                skyr: None,
                fr: None,
                tx,
                grad: None,
                talt: None,
                malt: None,
                millum: None,
                pier: m.pierside.clone().map(|p| {
                    let p = p.to_lowercase();
                    if p.contains("west") { "West".into() } else if p.contains("east") { "East".into() } else { p }
                }),
                cells: None,
                reference: None,
                to_ref: None,
                auto: Grade::Good,
                flags: Vec::new(),
                notes: Vec::new(),
            });
        }
        lights.sort_by(|a, b| a.time.cmp(&b.time).then(a.name.cmp(&b.name)));
        for l in lights.iter_mut() {
            if let Some(n) = decisions.panel(&l.path) {
                l.panel = Some(n);
            }
        }
        let by_fi: HashMap<usize, usize> = lights.iter().enumerate().map(|(i, l)| (l.fi, i)).collect();
        for l in lights.iter_mut() {
            if let Some(t) = transp.get(&l.fi) {
                l.t = Some(t.value);
                l.cells = Some(t.cells.clone());
                l.reference = by_fi.get(&t.reference).copied();
                l.to_ref = t.to_ref;
            }
        }
        let mut nights: Vec<NaiveDate> = lights.iter().map(|l| l.night).collect();
        nights.sort();
        nights.dedup();
        let thumbs = thumbs_by_fi.into_iter().filter_map(|(fi, t)| by_fi.get(&fi).map(|&i| (i, t))).collect();
        let crux_choices = triage.cal.choices.clone();
        let mut s = Session {
            root,
            cache_dir,
            cfg: base_cfg.clone(),
            base_cfg,
            triage,
            transp,
            lights,
            by_fi,
            nights,
            decisions,
            decisions_error,
            limits: defaults,
            defaults,
            auto0: Vec::new(),
            undo: Vec::new(),
            generation: 0,
            thumbs,
            regrade_ms: 0.0,
            timings,
            plan_moves: 0,
            plan_kinds: BTreeMap::new(),
            crux_choices,
        };
        s.apply_flats();
        s.regrade();
        s.auto0 = s.lights.iter().map(|l| l.auto).collect();
        if s.limits != limits {
            s.limits = limits;
            s.regrade();
        }
        s.update_plan();
        s
    }

    /// Grade again from cached metrics with the current limits.
    pub fn regrade(&mut self) {
        let t0 = std::time::Instant::now();
        let mut cfg = self.base_cfg.clone();
        let vals: BTreeMap<String, f64> = LIMITS.iter().zip(self.limits).map(|(l, v)| (l.key.to_string(), v)).collect();
        decisions::apply_thresholds(&mut cfg, &vals);
        let t = &self.triage;
        let ped = |i: usize| t.pedestal.get(&i).copied().unwrap_or(cfg.camera.bias_level_adu);
        let reports = grade_lights_with(&t.scan.frames, &t.metrics, &t.errors, &ped, &cfg, &self.transp);
        let by: HashMap<usize, &FrameReport> = reports.iter().map(|r| (r.frame, r)).collect();
        for l in self.lights.iter_mut() {
            let Some(r) = by.get(&l.fi) else { continue };
            l.auto = r.grade;
            l.flags = r.flags.clone();
            l.flags.sort_by_key(|f| std::cmp::Reverse(f.grade));
            l.notes = r.notes.clone();
            l.sky = r.sky_e_s;
            l.skyr = r.sky_ratio;
            l.fr = r.fwhm_ratio;
            l.grad = r.gradient_pct;
            if let Some(s) = &r.sky {
                l.talt = s.target_alt;
                l.malt = Some(s.moon_alt);
                l.millum = Some(s.moon_illum);
            }
        }
        self.cfg = cfg;
        self.regrade_ms = t0.elapsed().as_secs_f64() * 1000.0;
        self.generation += 1;
    }

    /// Store limit changes in the decisions sidecar (values that differ from crux's).
    pub fn save_limits(&mut self) {
        for (i, l) in LIMITS.iter().enumerate() {
            if (self.limits[i] - self.defaults[i]).abs() < 1e-9 {
                self.decisions.thresholds.remove(l.key);
            } else {
                self.decisions.thresholds.insert(l.key.to_string(), self.limits[i]);
            }
        }
        self.save_decisions();
        self.update_plan();
    }

    /// Lay your flat choices over crux's.
    pub fn apply_flats(&mut self) {
        self.triage.cal.choices = self.crux_choices.clone();
        self.decisions.apply_flat_choices(&self.root, &mut self.triage.cal, &self.triage.grouping, self.triage.library.as_ref());
    }

    /// Choose the flats for one night and filter (`None`: back to crux's choice).
    pub fn pick_flat(&mut self, filter: &str, night: NaiveDate, pick: Option<(Option<String>, Option<String>, NaiveDate)>) {
        self.decisions.set_flat(&self.root, filter, night, pick);
        self.save_decisions();
        self.apply_flats();
        self.generation += 1;
    }

    /// Choose the dark or bias for a family of lights (`None`: crux's choice).
    pub fn pick_offset(&mut self, key: String, file: Option<&std::path::Path>) {
        match file {
            Some(f) => {
                self.decisions.offsets.insert(key, crux_core::paths::key(f));
            }
            None => {
                self.decisions.offsets.remove(&key);
            }
        }
        self.save_decisions();
        self.generation += 1;
    }

    pub fn save_decisions(&mut self) {
        if let Err(e) = self.decisions.save() {
            self.decisions_error = Some(format!("{e:#}"));
        }
    }

    pub fn pinned(&self, i: usize) -> Option<Grade> {
        self.decisions.get(&self.lights[i].path)
    }

    pub fn grade(&self, i: usize) -> Grade {
        self.pinned(i).unwrap_or(self.lights[i].auto)
    }

    /// Pin grades (None = back to crux's). Pinning crux's own grade clears the pin.
    pub fn set_override(&mut self, ids: &[usize], g: Option<Grade>) {
        let paths: Vec<PathBuf> = ids.iter().map(|&i| self.lights[i].path.clone()).collect();
        self.set_override_paths(&paths, g);
    }

    /// Pin grades by path (files outside this folder keep the pin as given).
    pub fn set_override_paths(&mut self, paths: &[PathBuf], g: Option<Grade>) {
        let prev: Vec<(PathBuf, Option<Grade>)> = paths.iter().map(|p| (p.clone(), self.decisions.get(p))).collect();
        for p in paths {
            let auto = self.by_path(p).map(|i| self.lights[i].auto);
            let keep = match (g, auto) {
                (Some(Grade::Reject), Some(a)) => a != Grade::Reject,
                (Some(_), Some(a)) => a == Grade::Reject || a == Grade::Warn,
                (Some(_), None) => true,
                (None, _) => false,
            };
            self.decisions.set(p, if keep { g } else { None });
        }
        self.undo.push(Undo::Grades(prev));
        self.changed();
    }

    /// "These are really panel N" (None clears the correction).
    pub fn set_panel(&mut self, paths: &[PathBuf], panel: Option<u32>) {
        let prev: Vec<(PathBuf, Option<u32>)> = paths.iter().map(|p| (p.clone(), self.decisions.panel(p))).collect();
        for p in paths {
            let k = crux_core::cache::key(p);
            let header = self.triage.scan.frames.iter().find(|f| crux_core::cache::key(&f.meta.path) == k).and_then(|f| f.meta.object.as_deref().and_then(crux_core::index::panel_of));
            match panel {
                Some(n) if Some(n) != header => {
                    self.decisions.panels.insert(k, n);
                }
                _ => {
                    self.decisions.panels.remove(&k);
                }
            }
        }
        for l in self.lights.iter_mut() {
            let k = crux_core::cache::key(&l.path);
            if let Some(&n) = self.decisions.panels.get(&k) {
                l.panel = Some(n);
            } else if paths.iter().any(|p| crux_core::cache::key(p) == k) {
                l.panel = panel_num(&l.obj);
            }
        }
        self.undo.push(Undo::Panels(prev));
        self.changed();
    }

    fn changed(&mut self) {
        self.save_decisions();
        self.update_plan();
        self.generation += 1;
    }

    pub fn by_path(&self, p: &std::path::Path) -> Option<usize> {
        let k = crux_core::cache::key(p);
        self.lights.iter().position(|l| crux_core::cache::key(&l.path) == k)
    }

    pub fn undo_last(&mut self) -> bool {
        let Some(u) = self.undo.pop() else { return false };
        match u {
            Undo::Grades(prev) => {
                for (p, g) in prev {
                    self.decisions.set(&p, g);
                }
            }
            Undo::Panels(prev) => {
                for (p, n) in prev {
                    let k = crux_core::cache::key(&p);
                    match n {
                        Some(n) => {
                            self.decisions.panels.insert(k.clone(), n);
                        }
                        None => {
                            self.decisions.panels.remove(&k);
                        }
                    }
                    if let Some(i) = self.by_path(&p) {
                        self.lights[i].panel = n.or_else(|| panel_num(&self.lights[i].obj));
                    }
                }
            }
        }
        self.changed();
        true
    }

    /// Moves `crux plan` would propose now (honouring your decisions).
    pub fn update_plan(&mut self) {
        let p = self.plan_with(None, false);
        self.plan_moves = p.moves.len();
        self.plan_kinds.clear();
        for m in &p.moves {
            let dir = m.to.parent().and_then(|d| d.strip_prefix(&self.root).ok()).map(|d| d.display().to_string()).unwrap_or_default();
            *self.plan_kinds.entry(dir).or_default() += 1;
        }
    }

    /// Plan options carrying your corrections and pins.
    pub fn plan_options(&self, organise: bool) -> crux_core::plan::PlanOptions {
        crux_core::plan::PlanOptions {
            organise,
            panels: self.decisions.panels.iter().map(|(k, v)| (k.clone(), *v)).collect(),
            grades: self.decisions.overrides.iter().map(|(k, o)| (k.clone(), o.grade)).collect(),
            ..Default::default()
        }
    }

    /// A move plan with the current limits and your decisions; `layout`
    /// replaces the config's (and your saved) layout.
    pub fn plan_with(&mut self, layout: Option<&crux_core::config::Layout>, organise: bool) -> crux_core::plan::Plan {
        let mut cfg = self.cfg.clone();
        self.decisions.apply_layout(&mut cfg);
        if let Some(l) = layout {
            cfg.layout = l.clone();
        }
        let reports = self.live_triage_reports();
        let saved = std::mem::replace(&mut self.triage.reports, reports);
        let p = crux_core::plan::build(&self.triage, &cfg, &self.plan_options(organise), self.cache_dir.as_deref());
        self.triage.reports = saved;
        p
    }

    /// Reports with the current limits and your pins applied.
    pub fn live_reports(&self) -> Vec<FrameReport> {
        self.live_triage_reports()
    }

    fn live_triage_reports(&self) -> Vec<FrameReport> {
        let mut reports = self.triage.reports.clone();
        // Reports with the current limits, then your pins.
        let by: HashMap<usize, usize> = self.by_fi.clone();
        for r in reports.iter_mut() {
            if let Some(&i) = by.get(&r.frame) {
                let l = &self.lights[i];
                r.grade = l.auto;
                r.flags = l.flags.clone();
            }
        }
        self.decisions.apply_overrides(&self.triage.scan.frames, &mut reports);
        reports
    }

    pub fn counts(&self, ids: impl Iterator<Item = usize>) -> [usize; 3] {
        let mut c = [0; 3];
        for i in ids {
            c[self.grade(i) as usize] += 1;
        }
        c
    }

    /// Frames within a small margin inside a reject limit.
    pub fn borderline(&self, i: usize) -> bool {
        let l = &self.lights[i];
        let [t, tr, fo, sk] = self.limits;
        l.t.is_some_and(|v| v >= t && v < t + 0.06)
            || l.tx.is_some_and(|v| v < tr && v >= tr - 0.2)
            || l.fr.is_some_and(|v| v < fo && v >= fo - 0.12)
            || l.skyr.is_some_and(|v| v < sk && v >= sk - 0.5)
    }

    /// Frames where crux and your folders disagree.
    pub fn disagrees(&self, i: usize) -> bool {
        let l = &self.lights[i];
        (!l.bad && l.auto == Grade::Reject) || (l.bad && l.auto == Grade::Good)
    }

    /// The field's best frame to blink against (not this frame).
    pub fn reference_of(&self, i: usize) -> Option<usize> {
        self.lights[i].reference.filter(|&r| r != i)
    }

    pub fn inbox(&self) -> Vec<InboxItem> {
        let mut items = Vec::new();
        let t = &self.triage;
        let g = &t.grouping;
        // Masters built from a retired (bad) set.
        let src: Vec<&crux_core::calcheck::SetReport> = t.cal.sets.iter().filter(|s| s.flags.iter().any(|f| f.check == Check::Source)).collect();
        if !src.is_empty() {
            let reason = src[0].flags.iter().find(|f| f.check == Check::Source).map(|f| f.reason.clone()).unwrap_or_default();
            let vsl = src.iter().filter_map(|s| s.vs_lights.as_ref()).map(|v| v.error.spread).fold(None, |a: Option<f64>, v| Some(a.map_or(v, |a| a.max(v))));
            items.push(InboxItem {
                id: "masters".into(),
                sev: Sev::Rej,
                title: format!("{} master flat{} {} built from bad flats", src.len(), if src.len() == 1 { "" } else { "s" }, if src.len() == 1 { "was" } else { "were" }),
                body: format!(
                    "{reason}.{} crux won't use them; masters are never moved, so rebuild them from good raw flats.",
                    vsl.map(|v| format!(" Against the lights they leave a {v:.1}% error.")).unwrap_or_default()
                ),
                evidence: src.iter().map(|s| g.groups[s.group].label.clone()).collect(),
                thumbs: Vec::new(),
                acts: vec![("Acknowledge".into(), true, Act::Dismiss), ("Show calibration".into(), false, Act::OpenCalibration)],
            });
        }
        // Flat sets crux rejects.
        let bad_sets: Vec<String> = t
            .cal
            .sets
            .iter()
            .filter(|s| s.grade == Grade::Reject && !g.groups[s.group].master && g.groups[s.group].kind == FrameType::Flat && !g.groups[s.group].retired)
            .map(|s| format!("{}: {}", g.groups[s.group].label, s.summary()))
            .collect();
        if !bad_sets.is_empty() {
            items.push(InboxItem {
                id: "badflats".into(),
                sev: Sev::Warn,
                title: format!("{} flat set{} would be retired to FLAT.bad", bad_sets.len(), if bad_sets.len() == 1 { "" } else { "s" }),
                body: "These sets failed crux's flat checks. The move plan retires them; nothing moves until you apply it.".into(),
                evidence: bad_sets,
                thumbs: Vec::new(),
                acts: vec![("Acknowledge".into(), true, Act::Dismiss), ("Show calibration".into(), false, Act::OpenCalibration)],
            });
        }
        // Flats far from their lights.
        let far: Vec<&crux_core::calcheck::FlatChoice> =
            t.cal.choices.iter().filter(|c| !c.pinned && (c.chosen.as_ref().is_some_and(|o| o.days.abs() > 7) || c.grade != Grade::Good)).collect();
        if !far.is_empty() {
            let n: usize = far.iter().map(|c| c.n_lights).sum();
            items.push(InboxItem {
                id: "flatchoice".into(),
                sev: Sev::Warn,
                title: format!("Flats for {} night{} need checking", far.len(), if far.len() == 1 { "" } else { "s" }),
                body: format!("{n} lights use flats that are far away in time or have warnings. Add same-night flat folders to the library and crux will check and use them."),
                evidence: far.iter().take(6).map(|c| format!("{} {} · {}", c.filter, crate::widgets::d_short(c.night), c.reason)).collect(),
                thumbs: Vec::new(),
                acts: vec![("Keep these flats".into(), true, Act::Dismiss)],
            });
        }
        // Disagreements with your folders.
        // Frames you've pinned are settled and drop out of the count; the
        // Frames list (`disagrees`) still shows them so a pin can be undone.
        let open = |i: usize| self.pinned(i).is_none() && self.disagrees(i);
        let kept_rej: Vec<usize> = (0..self.lights.len()).filter(|&i| open(i) && !self.lights[i].bad).collect();
        let bad_good: Vec<usize> = (0..self.lights.len()).filter(|&i| open(i) && self.lights[i].bad).collect();
        if !kept_rej.is_empty() || !bad_good.is_empty() {
            let frames = |n: usize| if n == 1 { "1 frame".to_string() } else { format!("{n} frames") };
            let mut body = Vec::new();
            if !kept_rej.is_empty() {
                body.push(format!("{} you kept look rejectable, so crux leaves them out of the stack.", frames(kept_rej.len())));
            }
            if !bad_good.is_empty() {
                body.push(format!(
                    "{} you rejected (in a reject folder or named BAD_) look fine to crux, so crux stacks them.",
                    frames(bad_good.len())
                ));
            }
            body.push("To overrule crux, review them and pin a frame with Keep (A) or Reject (R).".into());
            items.push(InboxItem {
                id: "disagree".into(),
                sev: Sev::Warn,
                title: format!("crux disagrees with you on {}", frames(kept_rej.len() + bad_good.len())),
                body: body.join(" "),
                evidence: Vec::new(),
                thumbs: kept_rej.iter().chain(&bad_good).copied().collect(),
                acts: vec![("Review them".into(), true, Act::ReviewDisagree)],
            });
        }
        let border: Vec<usize> = (0..self.lights.len()).filter(|&i| self.pinned(i).is_none() && self.borderline(i)).collect();
        if !border.is_empty() {
            items.push(InboxItem {
                id: "border".into(),
                sev: Sev::Info,
                title: format!("{} frames are close to a reject limit", border.len()),
                body: "Their transparency, trail brightness, focus or sky is close to a reject limit. They're kept for now: check each one and keep or reject it.".into(),
                evidence: Vec::new(),
                thumbs: border,
                acts: vec![("Review borderline".into(), true, Act::ReviewBorder)],
            });
        }
        // Raw darks and bias.
        let raw_offsets = g.groups.iter().any(|x| matches!(x.kind, FrameType::Dark | FrameType::Bias) && !x.master);
        let masters = g.groups.iter().filter(|x| matches!(x.kind, FrameType::Dark | FrameType::Bias) && x.master).count();
        if !raw_offsets && masters > 0 {
            items.push(InboxItem {
                id: "nodarks".into(),
                sev: Sev::Info,
                title: "No raw darks or bias in this folder: using masters".into(),
                body: format!("{masters} master dark/bias file{} carry no GAIN, OFFSET or CCD-TEMP, so they were matched on exposure and frame size only.", if masters == 1 { "" } else { "s" }),
                evidence: Vec::new(),
                thumbs: Vec::new(),
                acts: vec![("Acknowledge".into(), true, Act::Dismiss)],
            });
        }
        items.retain(|i| !self.decisions.dismissed.contains(&i.id));
        items
    }
}
