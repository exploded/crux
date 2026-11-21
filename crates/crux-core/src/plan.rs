//! Move plans: what to move, where, and why. Building a plan reads nothing
//! but file sizes and times; `moves` applies it.
//!
//! What a plan can hold (each move carries its reason):
//! - **reject**: rejected lights, bias and darks to `BAD\<reason>\`
//!   (`layout.reject`); rejected flats from a kept set to `FLAT.bad`.
//! - **retire**: every frame of a flat set crux rejects (or you name) to
//!   `FLAT.bad` (`layout.flat_retired`).
//! - **organise**: raw frames that aren't where the layout template puts them.
//! - **library**: good flat sets to their library folder.
//! - **combine**: library sets that agree, into one folder (the index is
//!   merged when the plan is applied).

use crate::classify::{FrameType, Role};
use crate::config::{wildcard_match, Config};
use crate::grade::{Check, FrameReport, Grade};
use crate::library::Library;
use crate::pipeline::Triage;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

pub const PLAN_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MoveKind {
    Reject,
    Retire,
    Organise,
    Library,
    Combine,
    /// Rebuildable intermediates to `_Reclaimable`.
    Reclaim,
}

impl MoveKind {
    pub fn name(self) -> &'static str {
        match self {
            MoveKind::Reject => "reject",
            MoveKind::Retire => "retire",
            MoveKind::Organise => "organise",
            MoveKind::Library => "library",
            MoveKind::Combine => "combine",
            MoveKind::Reclaim => "reclaim",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Move {
    pub from: PathBuf,
    pub to: PathBuf,
    /// Size and modification time when planned: apply refuses a changed file.
    pub size: u64,
    pub mtime_ns: i64,
    pub kind: MoveKind,
    pub reason: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Plan {
    pub version: u32,
    pub created: String,
    /// The scanned folder.
    pub root: PathBuf,
    /// The scan cache, so apply can follow the files (no re-measuring).
    pub cache_dir: Option<PathBuf>,
    pub library_root: Option<PathBuf>,
    /// Where the library's database is (`[library] dir`).
    #[serde(default)]
    pub library_dir: Option<PathBuf>,
    pub moves: Vec<Move>,
    /// Library set ids to merge into the first of each group.
    pub combine: Vec<Vec<String>>,
    /// Problems that stop the plan being applied (name clashes).
    pub conflicts: Vec<String>,
    /// Things worth knowing that don't block it.
    pub notes: Vec<String>,
    /// Combining a target's folders into one (`targets::combine_plan`):
    /// apply then brings the folders' caches and decisions across.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
}

impl Plan {
    pub fn total_bytes(&self) -> u64 {
        self.moves.iter().map(|m| m.size).sum()
    }

    pub fn count(&self, kind: MoveKind) -> usize {
        self.moves.iter().filter(|m| m.kind == kind).count()
    }
}

#[derive(Debug, Clone)]
pub struct PlanOptions {
    /// Rejected frames to BAD (or FLAT.bad for flats).
    pub rejects: bool,
    /// Flat sets crux rejects to FLAT.bad.
    pub retire: bool,
    /// Raw frames into the layout template.
    pub organise: bool,
    /// Good flat sets into the library.
    pub to_library: bool,
    /// Library sets that agree, into one folder.
    pub combine: bool,
    /// File-name patterns never moved as rejects (grade override: keep).
    pub keep: Vec<String>,
    /// File-name patterns rejected regardless of grade (override: reject).
    pub reject: Vec<String>,
    /// Flat sets to retire / never retire, as `FILTER:NIGHT` (`H:2026-10-29`).
    pub retire_sets: Vec<String>,
    pub keep_sets: Vec<String>,
    /// Panel corrections (cache key → panel), from your decisions.
    pub panels: HashMap<String, u32>,
    /// crux's grade overridden by yours (cache key → grade), for `{grade}`.
    pub grades: HashMap<String, Grade>,
}

impl Default for PlanOptions {
    fn default() -> Self {
        PlanOptions {
            rejects: true,
            retire: true,
            organise: false,
            to_library: false,
            combine: false,
            keep: Vec::new(),
            reject: Vec::new(),
            retire_sets: Vec::new(),
            keep_sets: Vec::new(),
            panels: HashMap::new(),
            grades: HashMap::new(),
        }
    }
}

/// Folder name for a reject reason.
pub fn reason_slug(check: Check) -> &'static str {
    match check {
        Check::Read => "unreadable",
        Check::Type => "wrong-type",
        Check::Stars => "few-stars",
        Check::Focus => "focus",
        Check::Tracking => "tracking",
        Check::Sky => "bright-sky",
        Check::Gradient => "gradient",
        Check::Transparency => "cloud",
        Check::Patchy => "patchy-cloud",
        Check::Trail => "trail",
        Check::Level => "level",
        Check::Noise => "noise",
        Check::Structure => "light-leak",
        Check::HotPixels => "hot-pixels",
        Check::Temperature => "temperature",
        Check::Exposure => "exposure",
        Check::Saturated => "saturated",
        Check::FlatChange => "flat-change",
        Check::FlatMatch => "flat-error",
        Check::Source => "bad-source",
        Check::Frames => "set",
        Check::Banding => "banding",
        Check::Library => "library",
        Check::Manual => "manual",
    }
}

/// The flag a rejected frame is filed under: its most decisive reject.
/// The reject reason a frame is filed under (most important first).
pub fn main_reject(r: &FrameReport) -> Option<Check> {
    const ORDER: [Check; 24] = [
        Check::Manual,
        Check::Read,
        Check::Type,
        Check::Saturated,
        Check::Structure,
        Check::Transparency,
        Check::Patchy,
        Check::Trail,
        Check::Focus,
        Check::Tracking,
        Check::Stars,
        Check::Sky,
        Check::Gradient,
        Check::Level,
        Check::Noise,
        Check::Temperature,
        Check::Banding,
        Check::HotPixels,
        Check::Exposure,
        Check::FlatChange,
        Check::FlatMatch,
        Check::Library,
        Check::Source,
        Check::Frames,
    ];
    let rejects: Vec<Check> = r.flags.iter().filter(|f| f.grade == Grade::Reject).map(|f| f.check).collect();
    ORDER.iter().copied().find(|c| rejects.contains(c))
}

/// "PANEL_26" from an object like "LMC Panel 26", else "".
pub fn panel_of(object: &str) -> String {
    let low = object.to_lowercase();
    let Some(i) = low.find("panel") else { return String::new() };
    let rest = &object[i + 5..];
    let digits: String = rest.trim_start_matches([' ', '_', '-']).chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        String::new()
    } else {
        format!("PANEL_{digits}")
    }
}

fn clean(v: &str) -> String {
    v.trim().chars().map(|c| if "<>:\"/\\|?*".contains(c) || c.is_control() { '_' } else { c }).collect()
}

/// Expand a layout template under `root`. Empty components drop out, so
/// `LIGHT/{panel}` is `LIGHT` for a target that isn't a mosaic. `..` steps
/// out of the folder (e.g. `../Flat library/{filter}/{night}`).
pub fn expand(root: &Path, template: &str, vars: &[(&str, String)]) -> PathBuf {
    let mut p = root.to_path_buf();
    for part in template.split(['/', '\\']) {
        let mut s = part.to_string();
        for (k, v) in vars {
            s = s.replace(&format!("{{{k}}}"), &clean(v));
        }
        let s = s.trim().to_string();
        if s == ".." {
            p.pop();
        } else if !s.is_empty() && s != "." {
            p.push(s);
        }
    }
    p
}

/// Tokens a layout template can use, for the rule builder.
pub const TOKENS: [&str; 9] = ["{target}", "{panel}", "{night}", "{filter}", "{exposure}", "{grade}", "{reason}", "{pier}", "{object}"];

fn norm(p: &Path) -> String {
    let k = crate::paths::key(p);
    let t = k.trim_end_matches(['\\', '/']);
    if t.is_empty() { k } else { t.to_string() }
}

fn matches_any(patterns: &[String], name: &str) -> bool {
    patterns.iter().any(|p| wildcard_match(&p.to_lowercase(), &name.to_lowercase()))
}

/// `H:2026-10-29` → ("H", night).
fn parse_spec(s: &str) -> Option<(String, chrono::NaiveDate)> {
    let (f, n) = s.split_once(':')?;
    Some((f.trim().to_uppercase(), n.trim().parse().ok()?))
}

struct Planned {
    to_dir: PathBuf,
    kind: MoveKind,
    reason: String,
}

/// Build a plan from a graded triage. `cache_dir` is stored so apply can
/// keep the cache in step with the files.
pub fn build(t: &Triage, cfg: &Config, opts: &PlanOptions, cache_dir: Option<&Path>) -> Plan {
    let root = &t.scan.root;
    let g = &t.grouping;
    let lay = &cfg.layout;
    let mut planned: BTreeMap<usize, Planned> = BTreeMap::new();
    let mut notes = Vec::new();
    let reports: HashMap<usize, &FrameReport> = t.reports.iter().chain(&t.cal.reports).map(|r| (r.frame, r)).collect();
    let mut specs = |v: &[String]| -> Vec<(String, chrono::NaiveDate)> {
        v.iter()
            .filter_map(|s| {
                let p = parse_spec(s);
                if p.is_none() {
                    notes.push(format!("ignored set {s:?}: write it as FILTER:NIGHT, e.g. H:2026-10-29"));
                }
                p
            })
            .collect()
    };
    let (retire_sets, keep_sets) = (specs(&opts.retire_sets), specs(&opts.keep_sets));
    let file_name = |i: usize| t.scan.frames[i].meta.file_name().to_string();
    let kept = |i: usize| matches_any(&opts.keep, &file_name(i));
    let forced = |i: usize| matches_any(&opts.reject, &file_name(i));
    let vars = |i: usize, night: Option<chrono::NaiveDate>, reason: &str| template_vars(t, opts, &reports, i, night, reason);

    // Flat sets: retire, or file their rejected frames.
    let lib = t.library.as_ref();
    let mut retiring: HashSet<usize> = HashSet::new();
    for gr in g.groups.iter().filter(|g| g.kind == FrameType::Flat && !g.master) {
        let filter = gr.filter.clone().unwrap_or_default().to_uppercase();
        let named = |v: &[(String, chrono::NaiveDate)]| v.iter().any(|(f, n)| *f == filter && Some(*n) == gr.night);
        let set = t.cal.set(gr.id);
        let auto = opts.retire && set.is_some_and(|s| s.grade == Grade::Reject) && !named(&keep_sets);
        if !gr.retired && (auto || named(&retire_sets)) {
            retiring.insert(gr.id);
            let why = if named(&retire_sets) {
                "you asked to retire this set".to_string()
            } else {
                set.map(|s| s.summary()).unwrap_or_default()
            };
            for &i in gr.members.iter().filter(|&&i| !t.scan.frames[i].class.in_reject_dir) {
                let dir = expand(root, &lay.flat_retired, &vars(i, gr.night, "retired"));
                planned.insert(i, Planned { to_dir: dir, kind: MoveKind::Retire, reason: format!("retire {}: {why}", gr.label) });
            }
            continue;
        }
        if opts.rejects {
            for &i in &gr.members {
                let f = &t.scan.frames[i];
                if f.class.in_reject_dir || kept(i) {
                    continue;
                }
                let r = reports.get(&i);
                let rejected = r.is_some_and(|r| r.grade == Grade::Reject) && r.and_then(|r| main_reject(r)).is_some_and(|c| !matches!(c, Check::FlatMatch | Check::Library | Check::Frames));
                if rejected || forced(i) {
                    let reason = if forced(i) { "rejected by you".to_string() } else { r.map(|r| r.summary()).unwrap_or_default() };
                    planned.insert(i, Planned { to_dir: expand(root, &lay.flat_retired, &vars(i, gr.night, "")), kind: MoveKind::Reject, reason });
                }
            }
        }
    }

    // Masters are never moved; say which ones crux rejects.
    for gr in g.groups.iter().filter(|g| g.master) {
        if let Some(s) = t.cal.set(gr.id).filter(|s| s.grade == Grade::Reject) {
            notes.push(format!("{} is rejected ({}); masters aren't moved — rebuild or set it aside yourself", gr.label, s.summary()));
        }
    }

    // Lights, bias and darks: rejects to BAD\<reason>.
    if opts.rejects {
        for (i, f) in t.scan.frames.iter().enumerate() {
            let c = &f.class;
            // Frames in a reject folder are already filed; ones only named
            // BAD_ still move to BAD\<reason> when rejected.
            if c.role != Role::Raw || (c.in_reject_dir && !c.reject_name_only) || planned.contains_key(&i) || kept(i) {
                continue;
            }
            if !matches!(c.frame_type, FrameType::Light | FrameType::Bias | FrameType::Dark | FrameType::DarkFlat) {
                continue;
            }
            let r = reports.get(&i);
            let check = r.filter(|r| r.grade == Grade::Reject).and_then(|r| main_reject(r));
            let (slug, reason) = match (check, forced(i)) {
                (_, true) => ("manual", "rejected by you".to_string()),
                (Some(ch), _) => (reason_slug(ch), r.map(|r| r.summary()).unwrap_or_default()),
                (None, false) => continue,
            };
            planned.insert(i, Planned { to_dir: expand(root, &lay.reject, &vars(i, None, slug)), kind: MoveKind::Reject, reason });
        }
    }

    // Good flat sets into the library.
    if opts.to_library {
        match lib {
            None => notes.push(format!("no flat library at {}: refresh the calibration library first", cfg.library.root.display())),
            Some(lib) => {
                for gr in g.groups.iter().filter(|g| g.kind == FrameType::Flat && !g.master && !g.retired && !retiring.contains(&g.id)) {
                    let Some(night) = gr.night else { continue };
                    let filter = gr.filter.clone().unwrap_or_else(|| "none".into()).to_uppercase();
                    let id = crate::library::set_id(&filter, night, gr.binning.unwrap_or(1), gr.width, gr.height, None, false);
                    let Some(s) = lib.sets.iter().find(|s| s.id == id) else {
                        notes.push(format!("{} isn't in the flat library yet: refresh the calibration library first", gr.label));
                        continue;
                    };
                    if !s.usable() {
                        continue;
                    }
                    let todo: Vec<usize> = gr.members.iter().copied().filter(|i| !planned.contains_key(i) && !t.scan.frames[*i].class.in_reject_dir).collect();
                    for i in todo {
                        planned.insert(i, Planned { to_dir: s.home.clone(), kind: MoveKind::Library, reason: format!("{} into the flat library", gr.label) });
                    }
                }
            }
        }
    }

    // Organise: raw frames not where the layout puts them.
    if opts.organise {
        for (i, f) in t.scan.frames.iter().enumerate() {
            let c = &f.class;
            if c.role != Role::Raw || planned.contains_key(&i) {
                continue;
            }
            // Frames already rejected (by you or crux): the reject layout.
            if c.in_reject_dir && matches!(c.frame_type, FrameType::Light | FrameType::Bias | FrameType::Dark | FrameType::DarkFlat) {
                // Kept by you, or a light crux grades good (it's processed):
                // left where it is.
                let good_light = c.frame_type == FrameType::Light && reports.get(&i).is_some_and(|r| r.grade == Grade::Good);
                if kept(i) || good_light {
                    continue;
                }
                let slug = reports.get(&i).filter(|r| r.grade == Grade::Reject).and_then(|r| main_reject(r)).map(reason_slug).unwrap_or("manual");
                let dir = expand(root, &lay.reject, &vars(i, None, slug));
                if f.path().parent().map(norm) != Some(norm(&dir)) {
                    planned.insert(i, Planned { to_dir: dir, kind: MoveKind::Organise, reason: format!("layout: {}", lay.reject) });
                }
                continue;
            }
            if c.in_reject_dir {
                // Retired flats: the retired-flat layout.
                if c.frame_type == FrameType::Flat {
                    let night = g.groups.iter().find(|gr| gr.kind == FrameType::Flat && gr.members.contains(&i)).and_then(|gr| gr.night);
                    let dir = expand(root, &lay.flat_retired, &vars(i, night, "retired"));
                    if f.path().parent().map(norm) != Some(norm(&dir)) {
                        planned.insert(i, Planned { to_dir: dir, kind: MoveKind::Organise, reason: format!("layout: {}", lay.flat_retired) });
                    }
                }
                continue;
            }
            let template = match c.frame_type {
                FrameType::Light => &lay.light,
                FrameType::Flat => &lay.flat,
                FrameType::Dark | FrameType::DarkFlat => &lay.dark,
                FrameType::Bias => &lay.bias,
                FrameType::Unknown => continue,
            };
            let night = (c.frame_type == FrameType::Flat)
                .then(|| g.groups.iter().find(|gr| gr.kind == FrameType::Flat && gr.members.contains(&i)).and_then(|gr| gr.night))
                .flatten();
            let dir = expand(root, template, &vars(i, night, ""));
            if f.path().parent().map(norm) != Some(norm(&dir)) {
                planned.insert(i, Planned { to_dir: dir, kind: MoveKind::Organise, reason: format!("layout: {}", template) });
            }
        }
    }

    let mut moves: Vec<Move> = planned
        .into_iter()
        .map(|(i, p)| {
            let m = &t.scan.frames[i].meta;
            Move { from: m.path.clone(), to: p.to_dir.join(m.file_name()), size: m.file_size, mtime_ns: m.mtime_ns, kind: p.kind, reason: p.reason }
        })
        .collect();

    // Combine library sets that agree: every file into the first set's folder.
    let mut combine = Vec::new();
    if opts.combine {
        match lib {
            None => notes.push("--combine: no flat library".into()),
            Some(lib) => {
                let pairs = lib.pairs(cfg);
                for group in lib.combinable_groups(&pairs) {
                    let sets: Vec<&crate::library::LibrarySet> = group.iter().filter_map(|id| lib.sets.iter().find(|s| &s.id == id)).collect();
                    let home = sets[0].home.clone();
                    let names: Vec<String> = sets.iter().map(|s| s.label()).collect();
                    let reason = format!("combine {} (agree within {:.1}%, same dust)", names.join(", "), cfg.thresholds.calibration.combine_pct);
                    for s in &sets {
                        for f in &s.files {
                            // A planned library move for this file now goes to the combined folder.
                            if let Some(m) = moves.iter_mut().find(|m| norm(&m.from) == norm(f)) {
                                if m.kind == MoveKind::Library {
                                    m.to = home.join(f.file_name().unwrap_or_default());
                                    m.kind = MoveKind::Combine;
                                    m.reason = reason.clone();
                                }
                                continue;
                            }
                            if f.parent().map(norm) == Some(norm(&home)) {
                                continue;
                            }
                            let Ok(md) = std::fs::metadata(f) else {
                                notes.push(format!("combine: {} is missing; skipped", f.display()));
                                continue;
                            };
                            let Ok((size, mtime_ns)) = crate::meta::file_stamp(f) else { continue };
                            let _ = md;
                            moves.push(Move {
                                from: f.clone(),
                                to: home.join(f.file_name().unwrap_or_default()),
                                size,
                                mtime_ns,
                                kind: MoveKind::Combine,
                                reason: reason.clone(),
                            });
                        }
                    }
                    combine.push(group);
                }
            }
        }
    }

    let conflicts = find_conflicts(&moves);
    Plan {
        version: PLAN_VERSION,
        created: chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
        root: root.clone(),
        cache_dir: cache_dir.map(Path::to_path_buf),
        library_root: lib.map(|l: &Library| l.root.clone()),
        library_dir: lib.map(|l: &Library| l.dir.clone()),
        moves,
        combine,
        conflicts,
        notes,
        target: None,
    }
}

/// Values for a frame's layout tokens.
pub fn template_vars(
    t: &Triage,
    opts: &PlanOptions,
    reports: &HashMap<usize, &FrameReport>,
    i: usize,
    night: Option<chrono::NaiveDate>,
    reason: &str,
) -> Vec<(&'static str, String)> {
    let m = &t.scan.frames[i].meta;
    let object = m.object.clone().unwrap_or_default();
    let k = crate::cache::key(&m.path);
    let panel = match opts.panels.get(&k) {
        Some(p) => format!("PANEL_{p}"),
        None => panel_of(&object),
    };
    let grade = opts.grades.get(&k).copied().or(reports.get(&i).map(|r| r.grade)).map(|g| g.name().to_string()).unwrap_or_default();
    let pier = m.pierside.as_deref().map(|p| if p.to_lowercase().contains("west") { "West" } else if p.to_lowercase().contains("east") { "East" } else { "" }).unwrap_or("");
    vec![
        ("target", crate::index::target_of(&object)),
        ("grade", grade),
        ("pier", pier.to_string()),
        ("object", object.clone()),
        ("panel", panel),
        ("filter", m.filter.clone().unwrap_or_default().to_uppercase()),
        ("night", night.or(t.scan.frames[i].class.night).map(|n| n.to_string()).unwrap_or_default()),
        ("exposure", m.exptime.map(|e| format!("{e:.0}s")).unwrap_or_default()),
        ("reason", reason.to_string()),
    ]
}

/// Where a layout template puts frame `i` (for previews).
pub fn dest_for(t: &Triage, opts: &PlanOptions, template: &str, i: usize, reason: &str) -> PathBuf {
    let reports: HashMap<usize, &FrameReport> = t.reports.iter().chain(&t.cal.reports).map(|r| (r.frame, r)).collect();
    let night = (t.scan.frames[i].class.frame_type == FrameType::Flat)
        .then(|| t.grouping.groups.iter().find(|gr| gr.kind == FrameType::Flat && gr.members.contains(&i)).and_then(|gr| gr.night))
        .flatten();
    expand(&t.scan.root, template, &template_vars(t, opts, &reports, i, night, reason))
}

/// Destinations that already exist, or that two moves share.
pub fn find_conflicts(moves: &[Move]) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen: HashMap<String, &Path> = HashMap::new();
    for m in moves {
        let k = norm(&m.to);
        if let Some(other) = seen.insert(k, &m.from) {
            out.push(format!("{} and {} would both move to {}", other.display(), m.from.display(), m.to.display()));
        }
        if m.to.exists() {
            out.push(format!("{} already exists (moving {} there would overwrite it)", m.to.display(), m.from.display()));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn templates_and_panels() {
        assert_eq!(panel_of("LMC Panel 26"), "PANEL_26");
        assert_eq!(panel_of("M42 panel_3"), "PANEL_3");
        assert_eq!(panel_of("NGC 7293"), "");
        let r = Path::new("t");
        let v = vec![("panel", String::new()), ("reason", "cloud".to_string()), ("night", "2026-10-18".into())];
        assert_eq!(expand(r, "LIGHT/{panel}", &v), r.join("LIGHT"));
        assert_eq!(expand(r, "BAD/{reason}", &v), r.join("BAD").join("cloud"));
        assert_eq!(expand(r, r"FLAT\{night}", &v), r.join("FLAT").join("2026-10-18"));
        assert_eq!(expand(r, "x/{object}", &[("object", "a:b/c".into())]), r.join("x").join("a_b_c"));
    }
}
